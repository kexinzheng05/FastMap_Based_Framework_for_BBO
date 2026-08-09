"""Brute-force baseline: evaluate all 2000 graph instances.

Iterates over every instance, computes the specified centrality for the
target vertex, and reports the best score and total evaluation time.
"""

import time

from utils import current_flow_closeness_centrality, read_graph


def main():
    """Evaluate centrality for all instances via brute force."""
    start = time.time()
    size = 64
    n_obstacles = 2000
    target_vertex = [10, 10]

    scores = []
    total_centrality_time = 0

    for i in range(2000):
        t0 = time.time()
        G = read_graph(i, n_obstacles, target_vertex)

        t1 = time.time()
        centrality = current_flow_closeness_centrality(G)
        target_score = centrality[tuple(target_vertex)]
        scores.append(target_score)

        elapsed = time.time() - t1
        total_centrality_time += elapsed

    best = max(scores)
    print(f"Best score: {best}")
    print(f"Total time: {time.time() - start:.2f}s")
    print(f"Avg centrality time: {total_centrality_time / 2000:.4f}s")


if __name__ == "__main__":
    main()