"""Shared utility functions for graph centrality experiments.

Provides centrality wrappers, graph I/O, obstacle generation,
and score loading from pre-computed Excel files.
"""

import itertools
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_graph(size, obstacles):
    """Create a grid graph and remove obstacle nodes.

    Args:
        size: Side length of the square grid.
        obstacles: Iterable of (row, col) positions to remove.

    Returns:
        A NetworkX grid graph with the given obstacles removed.
    """
    G = nx.grid_2d_graph(size, size)
    for obstacle in obstacles:
        G.remove_node(obstacle)
    return G


def generate_obstacles(n, n_obstacles, target_vertex, constrained=True):
    """Randomly place obstacles on an n×n grid.

    Args:
        n: Side length of the grid.
        n_obstacles: Number of obstacles to place.
        target_vertex: (row, col) vertex that must remain obstacle-free.
        constrained: If True, no two obstacles may be adjacent.

    Returns:
        A list of (row, col) obstacle positions.
    """
    grid_map = np.zeros((n, n))
    is_valid = np.full((n, n), True)
    is_valid[target_vertex[0]][target_vertex[1]] = False
    positions = [
        (i, j)
        for i, j in itertools.product(range(n), repeat=2)
        if is_valid[i][j]
    ]

    if constrained:
        obstacles = []
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for _ in range(n_obstacles):
            obstacle = random.choice(positions)
            obstacles.append(obstacle)
            is_valid[obstacle[0]][obstacle[1]] = False
            grid_map[obstacle[0]][obstacle[1]] = 1
            for dx, dy in neighbors:
                nx_ = obstacle[0] + dx
                ny_ = obstacle[1] + dy
                if 0 <= nx_ < n and 0 <= ny_ < n:
                    is_valid[nx_][ny_] = False
            positions = [
                (i, j)
                for i, j in itertools.product(range(n), repeat=2)
                if is_valid[i][j]
            ]
    else:
        obstacles = random.sample(positions, k=n_obstacles)

    return obstacles


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def read_graph(index, n_obstacles, target_vertex):
    """Load a pickled graph from disk.

    Args:
        index: Instance index.
        n_obstacles: Number of obstacles (used in the file path).
        target_vertex: (row, col) target vertex (used in the file path).

    Returns:
        The deserialized NetworkX graph.
    """
    path = (
        f"{target_vertex[0]}_{target_vertex[1]}"
        f"/graphs/{n_obstacles}/{index}.pickle"
    )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Centrality wrappers
# ---------------------------------------------------------------------------

def closeness_centrality(G):
    """Compute closeness centrality for all nodes in *G*."""
    return nx.closeness_centrality(G)


def harmonic_centrality(G):
    """Compute harmonic centrality for all nodes in *G*."""
    return nx.harmonic_centrality(G)


def betweenness_centrality(G):
    """Compute betweenness centrality for all nodes in *G*."""
    return nx.betweenness_centrality(G)


def katz_centrality(G):
    """Compute Katz centrality for all nodes in *G*."""
    return nx.katz_centrality(G)


def current_flow_closeness_centrality(G):
    """Compute current-flow closeness centrality for all nodes in *G*."""
    return nx.current_flow_closeness_centrality(G)


def eigenvector_centrality(G):
    """Compute eigenvector centrality for all nodes in *G*."""
    return nx.eigenvector_centrality(G, max_iter=5000)


# ---------------------------------------------------------------------------
# Score loading
# ---------------------------------------------------------------------------

CENTRALITY_COLUMNS = {
    "harmonic": "c_harmonic",
    "betweenness": "c_between",
    "current": "c_current",
    "closeness": "c_close",
    "katz": "c_katz",
    "eigenvector": "c_eigen",
}


def read_excel(target_vertex, centrality_type):
    """Load pre-computed centrality scores from an Excel file.

    Scores are negated so that *lower is better* (suitable for minimisation).

    Args:
        target_vertex: (row, col) target vertex (used in the file path).
        centrality_type: One of ``"harmonic"``, ``"betweenness"``,
            ``"current"``, ``"closeness"``, ``"katz"``, ``"eigenvector"``.

    Returns:
        A list of negated centrality scores, one per graph instance.

    Raises:
        KeyError: If *centrality_type* is not recognised.
    """
    column = CENTRALITY_COLUMNS[centrality_type]
    path = (
        f"{target_vertex[0]}_{target_vertex[1]}"
        f"/centrality_{target_vertex[0]}_{target_vertex[1]}.xlsx"
    )
    df = pd.read_excel(path)
    return [-score for score in df[column].tolist()]
