from __future__ import annotations

import torch
from torch import nn


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # x: B, N, F  a_hat: N, N
        h = torch.matmul(a_hat, x)
        return torch.relu(self.lin(h))


class STGNN(nn.Module):
    """Vanilla PyTorch spatio-temporal GNN: GCN per timestep, then GRU over time."""

    def __init__(self, n_nodes: int, in_dim: int = 1, hidden: int = 32):
        super().__init__()
        self.n_nodes = n_nodes
        self.g1 = GraphConv(in_dim, hidden)
        self.g2 = GraphConv(hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, in_dim)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # x: B, T, N, F
        b, t, n, f = x.shape
        spatial = []
        for ti in range(t):
            h = self.g1(x[:, ti], a_hat)
            h = self.g2(h, a_hat)
            spatial.append(h)
        seq = torch.stack(spatial, dim=1)  # B, T, N, H
        seq = seq.transpose(1, 2).reshape(b * n, t, -1)
        out, _ = self.gru(seq)
        rec = self.head(out).reshape(b, n, t, -1).transpose(1, 2)
        return rec


def node_errors(model: STGNN, x: torch.Tensor, a_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        y = model(x, a_hat)
        se = (y - x) ** 2
        sample = se.mean(dim=(1, 2, 3))
        per_node = se[:, -1].mean(dim=-1)  # B, N
    return sample, per_node
