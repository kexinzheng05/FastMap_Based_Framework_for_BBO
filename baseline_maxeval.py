"""Random-sampling baseline with a fixed evaluation budget (maxeval).

Randomly samples *maxeval* instances from the full set of 2000 and
reports the rank of the best-sampled instance among all 2000.
Repeats across multiple random seeds to compute average rank and
success rate.
"""

import random
import time

import numpy as np

from utils import read_excel


def main():
    """Run the random-sampling baseline across multiple seeds."""
    start = time.time()
    target_vertex = (32, 32)
    scores = read_excel(target_vertex, centrality_type="katz")

    seeds = [
        0, 1, 2, 3, 4, 5, 42, 43, 44, 45,
        6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ]

    n_instances = 2000
    maxeval = 1200
    total_ranks = 0
    n_success = 0

    sorted_indices = sorted(range(n_instances), key=lambda i: scores[i])

    for seed in seeds:
        random.seed(seed)
        samples = random.sample(range(n_instances), k=maxeval)
        sample_scores = [scores[i] for i in samples]
        best_idx = samples[np.argmin(sample_scores)]

        best_rank = sorted_indices.index(best_idx) + 1
        print(f"Seed {seed:2d} — best rank: {best_rank}")
        total_ranks += best_rank
        if best_rank == 1:
            n_success += 1

    print(f"Time: {time.time() - start:.2f}s")
    print(f"Avg rank: {total_ranks / len(seeds):.2f}")
    print(f"Success rate: {n_success / len(seeds):.2%}")


if __name__ == "__main__":
    main()