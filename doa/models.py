"""Forecasting networks. Every model returns (forecast, band logits, quantiles or None)."""
import torch
import torch.nn as nn

from . import config as C


class Conv1d(nn.Conv1d):
    """1-D convolution computed as an (L x 1) 2-D one, which needs far less GPU memory at these batch sizes."""

    def _conv_forward(self, x, weight, bias):
        return nn.functional.conv2d(x.unsqueeze(-1), weight.unsqueeze(-1), bias, (self.stride[0], 1),
                                    (self.padding[0], 0)).squeeze(-1)


class ConvEncoder(nn.Module):
    """(B, T, channels) -> (B, width, tokens)."""

    def __init__(self, c_in, width=48, tokens=24, dropout=0.15):
        super().__init__()
        norm = lambda: nn.GroupNorm(min(8, width), width)
        self.net = nn.Sequential(
            Conv1d(c_in, width, 7, stride=2, padding=3), norm(), nn.GELU(), nn.Dropout(dropout),
            Conv1d(width, width, 5, padding=2), norm(), nn.GELU(), nn.Dropout(dropout),
            Conv1d(width, width, 3, padding=1), norm(), nn.GELU(),
            nn.AdaptiveAvgPool1d(tokens))

    def forward(self, x):
        return self.net(x.transpose(1, 2))


def head(width, out, dropout):
    return nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, out))


class DoAForecaster(nn.Module):
    """Proposed model: modality encoders, time-aligned fusion, two-layer GRU, forecast, band and quantile heads."""

    def __init__(self, inputs, width=48, tokens=24, layers=2, dropout=0.15):
        super().__init__()
        self.sel = [c for g in ("bis", "hemo", "drug") if g in inputs for c in C.GROUPS[g]]
        self.eeg = ConvEncoder(C.N_EEG, width, tokens, dropout) if "eeg" in inputs else None
        self.num = ConvEncoder(len(self.sel), width, tokens, dropout)
        fused = width * (1 + (self.eeg is not None))
        self.gru = nn.GRU(fused, fused, num_layers=layers, batch_first=True, dropout=dropout)
        self.norm, self.drop = nn.LayerNorm(fused), nn.Dropout(dropout)
        self.reg, self.cls, self.q = head(fused, 1, dropout), head(fused, 4, dropout), head(fused, 2, dropout)

    def forward(self, eeg, num):
        z = self.num(num[:, :, self.sel]).transpose(1, 2)
        if self.eeg is not None:
            z = torch.cat([self.eeg(eeg).transpose(1, 2), z], -1)
        h = self.drop(self.norm(self.gru(z)[0][:, -1]))
        return self.reg(h).squeeze(-1), self.cls(h), self.q(h)


class TransformerForecaster(nn.Module):
    """Transformer encoder over the fused tokens of all channels."""

    def __init__(self, width=48, tokens=24, heads=6, layers=2, dropout=0.15):
        super().__init__()
        self.eeg_enc = ConvEncoder(C.N_EEG, width, tokens, dropout)
        self.num_enc = ConvEncoder(C.N_NUM, width, tokens, dropout)
        f = 2 * width
        self.pos = nn.Parameter(torch.zeros(1, tokens + 1, f))
        self.cls = nn.Parameter(torch.zeros(1, 1, f))
        layer = nn.TransformerEncoderLayer(f, heads, 4 * f, dropout, batch_first=True, activation="gelu", norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, layers)
        self.norm, self.drop = nn.LayerNorm(f), nn.Dropout(dropout)
        self.reg_head, self.cls_head = head(f, 1, dropout), head(f, 4, dropout)

    def forward(self, eeg, num):
        z = torch.cat([self.eeg_enc(eeg).transpose(1, 2), self.num_enc(num).transpose(1, 2)], -1)
        z = self.fusion(torch.cat([self.cls.expand(len(z), -1, -1), z], 1) + self.pos)
        h = self.drop(self.norm(z[:, 0]))
        return self.reg_head(h).squeeze(-1), self.cls_head(h), None


class PoolingMLP(nn.Module):
    """Same encoders, mean pooling over time instead of sequence modelling."""

    def __init__(self, width=48, tokens=24, dropout=0.15):
        super().__init__()
        f = 2 * width
        self.eeg_enc = ConvEncoder(C.N_EEG, width, tokens, dropout)
        self.num_enc = ConvEncoder(C.N_NUM, width, tokens, dropout)
        self.mix = nn.Sequential(nn.Linear(f, f), nn.GELU(), nn.Dropout(dropout))
        self.norm, self.drop = nn.LayerNorm(f), nn.Dropout(dropout)
        self.reg, self.cls_head = head(f, 1, dropout), head(f, 4, dropout)

    def forward(self, eeg, num):
        z = torch.cat([self.eeg_enc(eeg).transpose(1, 2), self.num_enc(num).transpose(1, 2)], -1)
        h = self.drop(self.norm(self.mix(z.mean(1))))
        return self.reg(h).squeeze(-1), self.cls_head(h), None


class AdaptedAnesNet(nn.Module):
    """1-D CNN with batch normalization and max pooling after AnesNet, on the 17 channels used here."""
    WIDTHS = {4: (8, 16, 32, 64), 7: (8, 16, 32, 32, 16, 16, 8)}

    def __init__(self, depth=7, mult=4):
        super().__init__()
        layers, c = [], C.N_NUM + C.N_EEG
        for i, w in enumerate(self.WIDTHS[depth]):
            k = 10 if i == 0 else 16
            layers += [nn.ConstantPad1d(((k - 1) // 2, k // 2), 0.0), Conv1d(c, w * mult, k),
                       nn.BatchNorm1d(w * mult), nn.ReLU(), nn.MaxPool1d(2)]
            c = w * mult
        self.net, self.fc = nn.Sequential(*layers), nn.Linear(c, 1)

    def forward(self, eeg, num):
        h = self.net(torch.cat([num, eeg], -1).transpose(1, 2)).mean(-1)
        return self.fc(h).squeeze(-1), torch.zeros(len(h), 4, device=h.device), None


def build(cfg):
    """Network of a configuration dict: kind, inputs, width, dropout (depth and mult for AnesNet)."""
    kind = cfg["kind"]
    if kind == "gru":
        return DoAForecaster(cfg["inputs"], cfg["width"], dropout=cfg["dropout"])
    if kind == "transformer":
        return TransformerForecaster(cfg["width"], dropout=cfg["dropout"])
    if kind == "mlp":
        return PoolingMLP(cfg["width"], dropout=cfg["dropout"])
    if kind == "anesnet":
        return AdaptedAnesNet(cfg["depth"], cfg["mult"])
    raise ValueError(kind)
