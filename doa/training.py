"""Training, checkpoint selection and GPU window access."""
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .data import samples
from .metrics import scores
from .models import build

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(time.strftime("[%m-%d %H:%M:%S]"), msg, flush=True)


class WindowStore:
    """Per-second arrays on the device; windows are gathered and standardized per batch."""

    def __init__(self, d, stats):
        self.num = torch.from_numpy(np.array(d["NUM"])).to(DEV)
        self.eeg = torch.from_numpy(np.array(d["EEG"])).to(DEV)
        self.off = torch.arange(C.W, device=DEV)
        self.set_stats(stats)

    def set_stats(self, stats):
        self.s = {k: torch.from_numpy(v).to(DEV) for k, v in stats.items()}

    def batch(self, start, last=None):
        """Standardized (eeg, num). `last` maps an input group to the last observed row of every window;
        later rows repeat that row (value carried forward)."""
        rows = start[:, None] + self.off
        num, eeg = self.num[rows], self.eeg[rows]
        if last:
            held = lambda v: start[:, None] + torch.minimum(self.off[None], v[:, None].clamp(min=0))
            for g, ch in C.GROUPS.items():
                if g in last:
                    num[:, :, ch] = self.num[held(last[g])][:, :, ch]
            if "eeg" in last:
                eeg = self.eeg[held(last["eeg"])]
        return (eeg - self.s["emu"]) / self.s["esd"], (num - self.s["nmu"]) / self.s["nsd"]


def random_gaps(n, gen):
    """Simulated sensor loss: 30% of the windows lose BIS, EEG, both or the drugs for their last 10-300 s."""
    on = torch.rand(n, device=DEV, generator=gen) < 0.3
    gap = torch.exp(torch.rand(n, device=DEV, generator=gen) * math.log(30.0)) * 10.0
    which = torch.randint(0, 4, (n,), device=DEV, generator=gen)
    last = C.W - 1 - gap.long().clamp(max=C.W)
    full = torch.full((n,), C.W - 1, device=DEV)
    pick = lambda cond: torch.where(on & cond, last, full)
    return {"bis": pick((which == 0) | (which == 2)), "eeg": pick((which == 1) | (which == 2)), "drug": pick(which == 3)}


def fixed_gap(n, groups, gap):
    return {g: torch.full((n,), C.W - 1 - gap, device=DEV) for g in groups} if gap else None


def predict(net, store, starts, scale, groups=(), gap=0, batch=1024):
    """Forecasts and quantiles in BIS units. `scale` is (mean, SD) of the training targets."""
    net.eval()
    P, Q = [], []
    with torch.no_grad():
        for b in torch.split(starts, batch):
            y, _, q = net(*store.batch(b, fixed_gap(len(b), groups, gap)))
            P.append(y)
            if q is not None:
                Q.append(q)
    out = lambda x: torch.cat(x).float().cpu().numpy() * scale[1] + scale[0]
    return out(P), out(Q) if Q else None


def train(store, d, split, cfg, tag, epochs=12, bs=384, warmup=3, average_last=4):
    """Train one network and save checkpoint, test predictions and summary under runs/<tag>.

    cfg: kind, inputs, H, seed, width, lr, dropout, gaps (bool), and depth/mult for AnesNet."""
    out = C.RUNS / tag
    done = out.with_name(tag + ".json")
    if done.exists():
        return json.loads(done.read_text())
    t0, H, seed = time.time(), cfg["H"], cfg["seed"]
    S = samples(d, split, H)
    X = {p: torch.from_numpy(s[1]).to(DEV) for p, s in S.items()}
    Y = {p: torch.from_numpy(s[2]).to(DEV) for p, s in S.items()}
    mean, sd = float(S["train"][2].mean()), float(S["train"][2].std() + 1e-6)
    bands = torch.bucketize(Y["train"], torch.tensor(C.BANDS, dtype=torch.float32, device=DEV), right=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = build(cfg).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=2e-4)
    huber, ce = nn.HuberLoss(delta=1.0), nn.CrossEntropyLoss()
    order = torch.Generator(device=DEV).manual_seed(seed + 10_000)
    gap_gen = torch.Generator(device=DEV).manual_seed(seed + 20_000)
    taus = torch.tensor([0.025, 0.975], device=DEV)
    scale = (mean, sd)

    def lr_factor(e):
        return (e + 1) / warmup if e < warmup else 0.5 * (1 + math.cos(math.pi * (e - warmup) / max(1, epochs - warmup)))

    def val_mae(**kw):
        return float(np.mean(np.abs(predict(net, store, X["val"], scale, **kw)[0] - S["val"][2])))

    best, best_state, mean_state, n_mean, history, n = 1e9, None, None, 0, [], len(Y["train"])
    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * lr_factor(ep)
        net.train()
        total = 0.0
        for b in torch.split(torch.randperm(n, device=DEV, generator=order), bs):
            y = Y["train"][b]
            eeg, num = store.batch(X["train"][b], random_gaps(len(b), gap_gen) if cfg.get("gaps") else None)
            yr, yc, q = net(eeg, num)
            loss = huber(yr * sd + mean, y) + 0.2 * ce(yc, bands[b])
            if q is not None:
                e = y[:, None] - (q * sd + mean)
                loss = loss + torch.maximum(taus * e, (taus - 1) * e).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.detach() * len(b)
        v = val_mae()
        history.append({"epoch": ep + 1, "train_loss": float(total) / n, "val_MAE": v})
        if v < best - 1e-3:
            best, best_state = v, {k: x.clone() for k, x in net.state_dict().items()}
        if ep >= epochs - average_last:
            state = net.state_dict()
            mean_state = {k: x.clone().float() for k, x in state.items()} if mean_state is None else \
                         {k: mean_state[k] + state[k].float() for k in mean_state}
            n_mean += 1
    candidates = {"best": best_state, "averaged": {k: x / n_mean for k, x in mean_state.items()}}
    val = {}
    for k, state in candidates.items():
        net.load_state_dict(state)
        val[k] = val_mae()
    chosen = min(val, key=val.get)
    net.load_state_dict(candidates[chosen])
    p, q = predict(net, store, X["test"], scale)
    idx, start, y = S["test"]
    np.savez(out.with_name(tag + ".npz"), y=y, yp=p.astype(np.float32), idx=idx, start=start,
             case=d["win_case"][idx], **({} if q is None else {"q": q.astype(np.float32)}))
    torch.save({"state": {k: x.cpu() for k, x in net.state_dict().items()}, "scale": scale, "cfg": cfg}, out.with_name(tag + ".pt"))
    r = {"tag": tag, "cfg": cfg, "val_MAE": val[chosen], "chosen": chosen, "test": scores(y, p), "history": history,
         "params": sum(x.numel() for x in net.parameters()), "seconds": time.time() - t0}
    if cfg.get("inputs") and "eeg" in cfg["inputs"]:
        r["val_MAE_bis_module_lost_60s"] = val_mae(groups=("bis", "eeg"), gap=60)
    done.write_text(json.dumps(r, indent=1))
    log(f"{tag}: val {val[chosen]:.3f} test MAE {r['test']['MAE']:.3f} RMSE {r['test']['RMSE']:.3f} ({r['seconds']:.0f} s)")
    return r


def load_network(tag):
    ck = torch.load(C.RUNS / f"{tag}.pt", map_location="cpu", weights_only=True)
    net = build(ck["cfg"])
    net.load_state_dict(ck["state"])
    return net.to(DEV).eval(), ck
