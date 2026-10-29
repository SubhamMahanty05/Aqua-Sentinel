from __future__ import annotations

import networkx as nx
import numpy as np


def networkx_from_json(network: dict) -> nx.Graph:
    g = nx.Graph()
    for n in network["nodes"]:
        g.add_node(n["id"], **n)
    for e in network["edges"]:
        g.add_edge(e["start"], e["end"], id=e["id"], **{k: v for k, v in e.items() if k not in {"start", "end"}})
    return g


def hops(g: nx.Graph, a: str, b: str) -> int | None:
    if a not in g or b not in g:
        return None
    try:
        return int(nx.shortest_path_length(g, a, b))
    except nx.NetworkXNoPath:
        return None


def adjacency_from_network(network: dict, node_order: list[str]) -> np.ndarray:
    g = networkx_from_json(network)
    n = len(node_order)
    idx = {name: i for i, name in enumerate(node_order)}
    a = np.eye(n, dtype=np.float32)
    for u, v in g.edges():
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            a[i, j] = 1.0
            a[j, i] = 1.0
    deg = a.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    a_hat = a / deg
    return a_hat


def sensor_to_node_features(
    pressure_row: np.ndarray,
    sensor_ids: list[str],
    node_order: list[str],
    a_hat: np.ndarray,
    n_diffuse: int = 4,
) -> np.ndarray:
    """Map sparse sensor readings onto all nodes via graph diffusion."""
    idx = {n: i for i, n in enumerate(node_order)}
    x = np.zeros(len(node_order), dtype=np.float32)
    known = np.zeros(len(node_order), dtype=np.float32)
    for sid, val in zip(sensor_ids, pressure_row):
        if sid in idx:
            x[idx[sid]] = val
            known[idx[sid]] = 1.0
    h = x.copy()
    for _ in range(n_diffuse):
        h = a_hat @ h
        h = known * x + (1.0 - known) * h
    return h
