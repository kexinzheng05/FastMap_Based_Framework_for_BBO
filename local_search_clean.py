"""Local search optimisation over FastMap-embedded graph instances.

Given a set of graph instances embedded in Euclidean space (via FastMap),
performs multi-start local search using nearest-neighbour queries to
find the instance with the best (lowest) centrality score.
"""

import json
import math
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from utils import (
    closeness_centrality,
    harmonic_centrality,
    betweenness_centrality,
    katz_centrality,
    current_flow_closeness_centrality,
    eigenvector_centrality,
    read_excel,
    read_graph,
)

CENTRALITY_FUNCTIONS = {
    "closeness": closeness_centrality,
    "harmonic": harmonic_centrality,
    "betweenness": betweenness_centrality,
    "katz": katz_centrality,
    "current": current_flow_closeness_centrality,
    "eigenvector": eigenvector_centrality,
}


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _wait_for_enter(fig):
    """Block until Enter/Return is pressed on the matplotlib figure."""
    last_key = {"key": None}

    def on_key(event):
        last_key["key"] = event.key

    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    try:
        while True:
            pressed = plt.waitforbuttonpress(timeout=-1)
            if pressed is True:
                k = last_key["key"]
                if k in ("enter", "return", "\r", "\n"):
                    break
    finally:
        fig.canvas.mpl_disconnect(cid)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def local_search(
    points,
    scores,
    maxIter,
    method,
    is_visual=False,
    K=2,
    initial_radius=0.1,
    k=16,
    n_attempts=3,
    maxVisited=None,
):
    """Multi-start local search over embedded points.

    At each restart a random starting point is chosen, and the algorithm
    greedily moves to better-scoring neighbours until no improvement is
    found (first-improvement strategy).  Neighbourhood is defined via
    k-nearest-neighbours or radius queries, with adaptive expansion.

    Args:
        points: (n_points, D) array of embedded coordinates.
        scores: List of scores (lower is better).
        maxIter: Number of random restarts.
        method: ``"k_neighbors"`` or ``"radius_neighbors"``.
        is_visual: If True, show an interactive scatter-plot.
        K: Number of embedding dimensions to use.
        initial_radius: Starting radius for ``"radius_neighbors"``.
        k: Number of neighbours for ``"k_neighbors"``.
        n_attempts: Neighbourhood expansion attempts before giving up.
        maxVisited: Stop after visiting this many unique instances.

    Returns:
        A tuple ``(best_score, best_index, total_flips, n_visited,
        best_scores_per_flip)``.
    """
    points = points[:, :K]

    # --- optional visualisation setup ---
    if is_visual:
        plt.ion()
        fig = plt.figure(figsize=(6, 6))
        proj = "3d" if K == 3 else None
        ax = fig.add_subplot(111, projection=proj)

        np_scores = np.array(scores)
        sc = ax.scatter(*points.T, c=np_scores, cmap="coolwarm", s=50)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label("Score")

        neigh_scat = None
        curr_scat = None
        lines = []

    def clear_step_artists():
        nonlocal neigh_scat, curr_scat, lines
        if neigh_scat is not None:
            neigh_scat.remove()
            neigh_scat = None
        if curr_scat is not None:
            curr_scat.remove()
            curr_scat = None
        for ln in lines:
            ln.remove()
        lines = []

    def draw_step(curr_idx, neigh_idx, title_extra=""):
        nonlocal neigh_scat, curr_scat, lines
        clear_step_artists()
        neigh_coords = points[neigh_idx, :K].T
        neigh_scat = ax.scatter(
            *neigh_coords,
            s=90, marker="o", edgecolors="k",
            facecolors="none", linewidths=1.5,
        )
        for j in neigh_idx:
            line_coords = [
                [points[curr_idx, dim], points[j, dim]] for dim in range(K)
            ]
            (ln,) = ax.plot(*line_coords, linewidth=1.0, alpha=0.65)
            lines.append(ln)
        curr_coords = points[curr_idx : curr_idx + 1, :K].T
        curr_scat = ax.scatter(
            *curr_coords,
            s=200, marker="o", edgecolors="black",
            facecolors="red", linewidths=1.5, alpha=0.95,
        )
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        if K == 3:
            ax.set_zlabel("Z-axis")
        ax.set_title(
            f"Local search — idx={curr_idx}, "
            f"score={scores[curr_idx]:.6f} {title_extra}"
        )
        plt.draw()
        _wait_for_enter(fig)

    # --- main search loop ---
    random_starts = random.sample(range(len(points)), maxIter)
    neigh = NearestNeighbors(n_neighbors=k, radius=initial_radius).fit(points)

    curr_best_score = float("inf")
    curr_best_map_idx = None
    total_flips = 0
    visited = set()
    best_scores_iteration = []

    for i in range(maxIter):
        curr_map_idx = random_starts[i]
        curr_score = scores[curr_map_idx]

        visited.add(curr_map_idx)
        if maxVisited is not None and len(visited) >= maxVisited:
            return (
                curr_best_score, curr_best_map_idx,
                total_flips, len(visited), best_scores_iteration,
            )

        if curr_score < curr_best_score:
            curr_best_score = curr_score
            curr_best_map_idx = curr_map_idx

        n_flip = 0
        tabu_list = deque(maxlen=1)

        while n_flip < 100:
            found_better = False

            for attempt in range(n_attempts):
                if method == "radius_neighbors":
                    indices = neigh.radius_neighbors(
                        [points[curr_map_idx]],
                        radius=initial_radius * (1.5 ** attempt),
                        return_distance=False,
                    )[0]
                elif method == "k_neighbors":
                    indices = neigh.kneighbors(
                        [points[curr_map_idx]],
                        n_neighbors=k * (2 ** attempt),
                        return_distance=False,
                    )[0]

                neighbor_indices = [
                    j for j in indices[1:] if j not in tabu_list
                ]
                if not neighbor_indices:
                    continue
                if len(neighbor_indices) > 150:
                    neighbor_indices = random.sample(neighbor_indices, k=150)

                if is_visual:
                    draw_step(curr_map_idx, neighbor_indices)

                for index in neighbor_indices:
                    visited.add(index)
                    if maxVisited is not None and len(visited) >= maxVisited:
                        return (
                            curr_best_score, curr_best_map_idx,
                            total_flips + n_flip, len(visited),
                            best_scores_iteration,
                        )
                    if scores[index] < curr_score:
                        n_flip += 1
                        curr_score = scores[index]
                        curr_map_idx = index
                        tabu_list.append(curr_map_idx)
                        found_better = True
                        break

                if found_better:
                    break

            if not found_better:
                break

            if curr_score < curr_best_score:
                curr_best_score = curr_score
                curr_best_map_idx = curr_map_idx
            best_scores_iteration.append(curr_score)

        total_flips += n_flip

    return (
        curr_best_score, curr_best_map_idx,
        total_flips, len(visited), best_scores_iteration,
    )


def local_search_real(
    points,
    n_obstacles,
    target_vertex,
    centrality_type,
    maxIter,
    method,
    K=2,
    initial_radius=0.1,
    k=16,
    n_attempts=3,
    maxVisited=None,
):
    """Multi-start local search with on-the-fly centrality evaluation.

    Unlike ``local_search``, this function does **not** use pre-computed
    scores.  Each time a point is visited, its graph is loaded from disk
    and the requested centrality is computed from scratch.  Computed
    scores are cached so that revisits (if any) are free.

    Args:
        points: (n_points, D) array of embedded coordinates.
        n_obstacles: Number of obstacles (for graph file lookup).
        target_vertex: (row, col) target vertex.
        centrality_type: One of ``"closeness"``, ``"harmonic"``,
            ``"betweenness"``, ``"katz"``, ``"current"``,
            ``"eigenvector"``.
        maxIter: Number of random restarts.
        method: ``"k_neighbors"`` or ``"radius_neighbors"``.
        K: Number of embedding dimensions to use.
        initial_radius: Starting radius for ``"radius_neighbors"``.
        k: Number of neighbours for ``"k_neighbors"``.
        n_attempts: Neighbourhood expansion attempts before giving up.
        maxVisited: Stop after visiting this many unique instances.

    Returns:
        A tuple ``(best_score, best_index, total_flips, n_visited,
        n_evaluations, best_scores_per_flip)``.
    """
    centrality_func = CENTRALITY_FUNCTIONS[centrality_type]
    score_cache = {}  # index -> negated centrality score
    n_evaluations = 0  # count of actual centrality computations

    def evaluate(index):
        """Return the (negated) centrality score for instance *index*."""
        nonlocal n_evaluations
        if index not in score_cache:
            G = read_graph(index, n_obstacles, target_vertex)
            centrality = centrality_func(G)
            score = -centrality[tuple(target_vertex)]
            score_cache[index] = score
            n_evaluations += 1
        return score_cache[index]

    points = points[:, :K]

    # --- main search loop ---
    random_starts = random.sample(range(len(points)), maxIter)
    neigh = NearestNeighbors(n_neighbors=k, radius=initial_radius).fit(points)

    curr_best_score = float("inf")
    curr_best_map_idx = None
    total_flips = 0
    visited = set()
    best_scores_iteration = []

    for i in range(maxIter):
        curr_map_idx = random_starts[i]
        curr_score = evaluate(curr_map_idx)

        visited.add(curr_map_idx)
        if maxVisited is not None and len(visited) >= maxVisited:
            return (
                curr_best_score, curr_best_map_idx,
                total_flips, len(visited), n_evaluations,
                best_scores_iteration,
            )

        if curr_score < curr_best_score:
            curr_best_score = curr_score
            curr_best_map_idx = curr_map_idx

        n_flip = 0
        tabu_list = deque(maxlen=1)

        while n_flip < 100:
            found_better = False

            for attempt in range(n_attempts):
                if method == "radius_neighbors":
                    indices = neigh.radius_neighbors(
                        [points[curr_map_idx]],
                        radius=initial_radius * (1.5 ** attempt),
                        return_distance=False,
                    )[0]
                elif method == "k_neighbors":
                    indices = neigh.kneighbors(
                        [points[curr_map_idx]],
                        n_neighbors=k * (2 ** attempt),
                        return_distance=False,
                    )[0]

                neighbor_indices = [
                    j for j in indices[1:] if j not in tabu_list
                ]
                if not neighbor_indices:
                    continue
                if len(neighbor_indices) > 150:
                    neighbor_indices = random.sample(neighbor_indices, k=150)

                for index in neighbor_indices:
                    visited.add(index)
                    score = evaluate(index)
                    if maxVisited is not None and len(visited) >= maxVisited:
                        return (
                            curr_best_score, curr_best_map_idx,
                            total_flips + n_flip, len(visited),
                            n_evaluations, best_scores_iteration,
                        )
                    if score < curr_score:
                        n_flip += 1
                        curr_score = score
                        curr_map_idx = index
                        tabu_list.append(curr_map_idx)
                        found_better = True
                        break

                if found_better:
                    break

            if not found_better:
                break

            if curr_score < curr_best_score:
                curr_best_score = curr_score
                curr_best_map_idx = curr_map_idx
            best_scores_iteration.append(curr_score)

        total_flips += n_flip

    return (
        curr_best_score, curr_best_map_idx,
        total_flips, len(visited), n_evaluations,
        best_scores_iteration,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run both local_search (pre-computed) and local_search_real (on-the-fly)."""
    start = time.time()

    seeds = [
        0, 1, 2, 3, 4, 5, 42, 43, 44, 45,
        6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]
    target_vertex = (32, 32)
    n_obstacles = 2000
    centrality_type = "katz"

    # Load FastMap embeddings and normalise.
    points = np.load(
        f"{target_vertex[0]}_{target_vertex[1]}/points_2000_K8_1.npy"
    )[:, :3]
    scaler = MinMaxScaler()
    points = scaler.fit_transform(points)
    step1_time = time.time() - start

    # Ground-truth scores (for local_search and ranking).
    scores = read_excel(target_vertex, centrality_type=centrality_type)
    ranked_scores = sorted(scores)

    params = {
        "maxIter": 10,
        "method": "radius_neighbors",
        "K": 3,
        "initial_radius": 0.09,
        "k": 51,
        "n_attempts": 3,
        "maxVisited": 1200,
    }

    # ---- Run local_search (pre-computed scores) ----
    print("="*55)
    print("  local_search (pre-computed scores)")
    print("="*55)
    results_table = [
        ["seed", "# flip", "# visited", "rank"]
    ]
    total_seed_time = 0

    for seed in seeds:
        t0 = time.time()
        random.seed(seed)
        min_score, min_map_idx, total_flips, unique_visited, _ = local_search(
            points, scores, **params,
        )
        rank = ranked_scores.index(min_score) + 1
        results_table.append([seed, total_flips, unique_visited, rank])
        print(f"Seed {seed:2d} — flips: {total_flips}, visited: {unique_visited}, "
              f"rank: {rank}, time: {time.time() - t0:.2f}s")
        total_seed_time += time.time() - t0

    total_visit = sum(row[2] for row in results_table[1:])
    total_rank = sum(row[3] for row in results_table[1:])
    n_success = sum(1 for row in results_table[1:] if row[3] == 1)
    print(f"\n[Pre-computed] Step1+Step2 time: {step1_time + total_seed_time / len(seeds):.2f}s")
    print(f"[Pre-computed] Avg visited: {total_visit / len(seeds):.1f}")
    print(f"[Pre-computed] Success rate: {n_success / len(seeds):.2%}")
    print(f"[Pre-computed] Avg rank: {total_rank / len(seeds):.2f}")

    # ---- Run local_search_real (on-the-fly centrality) ----
    print()
    print("="*55)
    print("  local_search_real (on-the-fly centrality evaluation)")
    print("="*55)
    results_table_real = [
        ["seed", "# flip", "# visited", "# evals", "rank"]
    ]
    total_seed_time_real = 0

    for seed in seeds:
        t0 = time.time()
        random.seed(seed)
        min_score, min_map_idx, total_flips, unique_visited, n_evals, _ = (
            local_search_real(
                points,
                n_obstacles=n_obstacles,
                target_vertex=target_vertex,
                centrality_type=centrality_type,
                **params,
            )
        )
        rank = ranked_scores.index(min_score) + 1
        results_table_real.append([seed, total_flips, unique_visited, n_evals, rank])
        print(f"Seed {seed:2d} — flips: {total_flips}, visited: {unique_visited}, "
              f"evals: {n_evals}, rank: {rank}, time: {time.time() - t0:.2f}s")
        total_seed_time_real += time.time() - t0

    total_visit = sum(row[2] for row in results_table_real[1:])
    total_rank = sum(row[4] for row in results_table_real[1:])
    total_evals = sum(row[3] for row in results_table_real[1:])
    n_success = sum(1 for row in results_table_real[1:] if row[4] == 1)
    print(f"\n[Real] Step1+Step2 time: {step1_time + total_seed_time_real / len(seeds):.2f}s")
    print(f"[Real] Avg visited: {total_visit / len(seeds):.1f}")
    print(f"[Real] Avg evals: {total_evals / len(seeds):.1f}")
    print(f"[Real] Success rate: {n_success / len(seeds):.2%}")
    print(f"[Real] Avg rank: {total_rank / len(seeds):.2f}")


if __name__ == "__main__":
    main()