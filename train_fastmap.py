"""Train a FastMap embedding for graph instances.

Loads pre-generated grid-world graphs, computes a distance
representation (e.g. shortest-path based), and uses FastMap to
embed the instances into K-dimensional Euclidean space.
"""

import random

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

from fastmap import FastMap
from utils import read_graph


def get_graph_dataset(target_vertex, n_obstacles=2000, n_instances=2000):
    """Load all graph instances from disk.

    Args:
        target_vertex: (row, col) target vertex (used in file paths).
        n_obstacles: Number of obstacles per instance.
        n_instances: Total number of instances to load.

    Returns:
        A list of NetworkX graphs.
    """
    return [
        read_graph(i, n_obstacles, target_vertex)
        for i in range(n_instances)
    ]


# ---------------------------------------------------------------------------
# Distance functions for FastMap
# ---------------------------------------------------------------------------

def distance_bipartite_matching(g1, g2):
    """Bipartite matching distance between two graphs based on node positions.

    Computes the optimal assignment between node sets using the
    Manhattan distance, then returns the total cost.

    Args:
        g1: First NetworkX graph.
        g2: Second NetworkX graph.

    Returns:
        Total Manhattan distance under the optimal assignment.
    """
    vertices1 = np.array(list(g1.nodes))
    vertices2 = np.array(list(g2.nodes))
    cost = np.abs(vertices1[:, None, :] - vertices2[None, :, :]).sum(axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].sum()


def distance_histogram_js(p1, p2):
    """Jensen–Shannon distance between shortest-path-length histograms.

    Args:
        p1: Dict mapping node → shortest-path length from the target.
        p2: Dict mapping node → shortest-path length from the target.

    Returns:
        Jensen–Shannon distance between the two histograms.
    """
    scores1 = np.array(list(p1.values()))
    scores2 = np.array(list(p2.values()))
    h1, _ = np.histogram(scores1, bins=range(130), density=True)
    h2, _ = np.histogram(scores2, bins=range(130), density=True)
    return distance.jensenshannon(h1, h2)


def distance_exponential_sum(p1, p2, alpha=0.3):
    """Absolute difference of exponentially-weighted path-length sums.

    Computes ``|Σ α^d_i(p1) − Σ α^d_i(p2)|`` where ``d_i`` are the
    shortest-path lengths from the target to each node.

    Args:
        p1: Dict mapping node → shortest-path length from the target.
        p2: Dict mapping node → shortest-path length from the target.
        alpha: Decay factor for the exponential weighting.

    Returns:
        Absolute difference of the two sums.
    """
    s1 = sum(alpha ** d for d in p1.values())
    s2 = sum(alpha ** d for d in p2.values())
    return abs(s1 - s2)


def distance_degree_weighted(p1, p2, alpha=0.99, k=3):
    """Degree-weighted exponential distance between path-length dicts.

    Each entry is weighted by the node degree raised to the k-th power,
    then exponentially decayed by shortest-path length.

    Args:
        p1: Tuple ``(path_lengths, degree_view)`` for graph 1.
        p2: Tuple ``(path_lengths, degree_view)`` for graph 2.
        alpha: Decay factor.
        k: Degree exponent.

    Returns:
        Absolute difference of the two weighted sums.
    """
    d1, deg1 = p1
    d2, deg2 = p2
    s1 = sum((deg1[node] ** k) * (alpha ** dist) for node, dist in d1.items())
    s2 = sum((deg2[node] ** k) * (alpha ** dist) for node, dist in d2.items())
    return abs(s1 - s2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Train a FastMap embedding and save the result."""
    random.seed(1)
    target_vertex = (10, 10)
    graphs = get_graph_dataset(target_vertex)

    # Compute shortest-path lengths from the target for each graph.
    paths = [
        nx.single_source_shortest_path_length(g, target_vertex)
        for g in graphs
    ]

    # FastMap → embed objects into Euclidean space.
    K = 3
    fastmap = FastMap(paths, [1] * 2000, distance_exponential_sum)
    fastmap.fit(
        K,
        model_name=f"{target_vertex[0]}_{target_vertex[1]}/models/model1_K8_6.m",
    )
    points = fastmap.P
    print(f"Embedding shape: {points.shape}")
    np.save(
        f"{target_vertex[0]}_{target_vertex[1]}/points_2000_K8_6.npy",
        points,
    )


if __name__ == "__main__":
    main()