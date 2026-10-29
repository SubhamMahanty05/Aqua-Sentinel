from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, n_features)
        self.hidden = hidden
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, F
        _, (h, c) = self.encoder(x)
        b, t, _ = x.shape
        z = h[-1].unsqueeze(1).repeat(1, t, 1)
        decoded, _ = self.decoder(z)
        return self.out(decoded)


def reconstruction_error(model: LSTMAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE (B,) and per-feature last-step error (B, F)."""
    with torch.no_grad():
        y = model(x)
        per_tf = (y - x) ** 2
        sample = per_tf.mean(dim=(1, 2))
        last = per_tf[:, -1, :]
    return sample, last
