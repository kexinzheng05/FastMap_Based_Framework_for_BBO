"""Generate random grid-world graphs with obstacles.

Creates connected grid graphs of a given size with randomly placed
obstacles, ensuring the resulting graph forms a single connected
component by merging smaller components back into the largest one.
"""

import itertools
import pickle
import random
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils import create_graph


def generate_obstacles(n, n_obstacles, target_vertex, constrained=True):
    """Randomly place obstacles on an n×n grid.

    Args:
        n: Side length of the grid.
        n_obstacles: Number of obstacles to place.
        target_vertex: (row, col) vertex that must remain obstacle-free.
        constrained: If True, no two obstacles may be adjacent.

    Returns:
        A tuple ``(obstacles, grid_map)`` where *obstacles* is a list of
        (row, col) positions and *grid_map* is an n×n binary array.
    """
    grid_map = np.zeros((n, n))
    is_valid = np.full((n, n), True)
    is_valid[target_vertex[0]][target_vertex[1]] = False
    positions = [
        (i, j)
        for i, j in itertools.product(range(n), repeat=2)
        if is_valid[i][j]
    ]
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    if constrained:
        obstacles = []
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
        target_neighbors = [
            (target_vertex[0] + dx, target_vertex[1] + dy)
            for dx, dy in neighbors
        ]
        while _is_isolated(target_neighbors, obstacles):
            obstacles = random.sample(positions, k=n_obstacles)
        for i, j in obstacles:
            grid_map[i][j] = 1

    return obstacles, grid_map


def _is_isolated(target_neighbors, obstacles):
    """Return True if every neighbour of the target is an obstacle."""
    for neighbour in target_neighbors:
        if neighbour not in obstacles:
            return False
    return True


def save_graph(G, index, n_obstacles, target_vertex):
    """Persist a graph to disk as a pickle file.

    Args:
        G: The NetworkX graph.
        index: Instance index (used in the filename).
        n_obstacles: Number of obstacles (used in the directory path).
        target_vertex: (row, col) target vertex (used in the directory path).
    """
    path = (
        f"{target_vertex[0]}_{target_vertex[1]}"
        f"/graphs/{n_obstacles}/{index}.pickle"
    )
    with open(path, "wb") as f:
        pickle.dump(G, f)


def _add_node(G, grid_map, added_node):
    """Add a node back into the graph and connect it to free neighbours.

    Args:
        G: The NetworkX graph (modified in place).
        grid_map: The n×n binary obstacle map.
        added_node: (row, col) position to add.

    Returns:
        The modified graph *G*.
    """
    n = grid_map.shape[0]
    G.add_node(added_node)
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx_, ny_ = added_node[0] + dx, added_node[1] + dy
        if 0 <= nx_ < n and 0 <= ny_ < n and grid_map[nx_][ny_] == 0:
            G.add_edge(added_node, (nx_, ny_))
    return G


def _bfs_min_removal(start, targets, grid_map):
    """Find the shortest path from *start* to any node in *targets*.

    Uses 0-1 BFS: moving through a free cell costs 0 and moving through
    an obstacle costs 1.

    Args:
        start: (row, col) source position.
        targets: Set of (row, col) target positions.
        grid_map: The n×n binary obstacle map.

    Returns:
        A tuple ``(path, cost)`` where *path* is the list of cells from
        *start* to the reached target and *cost* is the number of
        obstacles traversed.  Returns ``(None, inf)`` if no path exists.
    """
    n = grid_map.shape[0]
    dq = deque([start])
    dist = {start: 0}
    parent = {start: None}

    while dq:
        x, y = dq.popleft()
        if (x, y) in targets:
            path = []
            cur = (x, y)
            while cur:
                path.append(cur)
                cur = parent[cur]
            return path[::-1], dist[(x, y)]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < n and 0 <= ny_ < n:
                cost = 0 if grid_map[nx_][ny_] == 0 else 1
                new_dist = dist[(x, y)] + cost
                if (nx_, ny_) not in dist or new_dist < dist[(nx_, ny_)]:
                    dist[(nx_, ny_)] = new_dist
                    parent[(nx_, ny_)] = (x, y)
                    if cost == 0:
                        dq.appendleft((nx_, ny_))
                    else:
                        dq.append((nx_, ny_))

    return None, float("inf")


def largest_connected_component(G, grid_map, target):
    """Merge all connected components into one by removing obstacles.

    Iteratively connects the second-largest component to the largest
    by finding a minimum-obstacle path and swapping obstacles with
    non-articulation-point nodes.

    Args:
        G: The NetworkX graph (modified in place).
        grid_map: The n×n binary obstacle map (modified in place).
        target: (row, col) target vertex that must not be removed.

    Returns:
        A tuple ``(G, grid_map)`` after merging.
    """
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    while len(components) > 1:
        largest = components[0]
        main_set = set(largest)
        next_component = components[1]

        start = random.choice(list(next_component))
        path, n_removals = _bfs_min_removal(start, main_set, grid_map)

        for i, j in path:
            if grid_map[i][j] == 1:
                grid_map[i][j] = 0
                _add_node(G, grid_map, (i, j))

        # Replace removed obstacles with non-articulation-point nodes.
        art_points = set(nx.articulation_points(G))
        candidates = [
            node for node in G.nodes
            if node not in art_points and node != tuple(target)
        ]
        new_obs = random.sample(candidates, n_removals)
        for i, j in new_obs:
            grid_map[i][j] = 1
            G.remove_node((i, j))

        components = sorted(nx.connected_components(G), key=len, reverse=True)

    return G, grid_map


def save_map(grid_map, index, output_folder):
    """Save a visualisation of the obstacle map as a PNG image.

    Args:
        grid_map: The n×n binary obstacle map.
        index: Instance index (used in the filename).
        output_folder: Directory to write the image to.
    """
    plt.imshow(grid_map, cmap="binary", vmin=0, vmax=1)
    ax = plt.gca()
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    plt.savefig(f"{output_folder}/{index}.png", bbox_inches="tight")


def main():
    """Generate grid-world instances and save them to disk."""
    size = 64
    n_obstacles = 2000
    random.seed(0)
    target_vertex = [32, 32]

    for i in range(1000):
        # constrained=False  → no constraints for obstacles' locations
        # constrained=True   → no two adjacent obstacles
        obstacles, grid_map = generate_obstacles(
            size, n_obstacles, target_vertex, constrained=False
        )
        G = create_graph(size, obstacles)
        G, grid_map = largest_connected_component(G, grid_map, target_vertex)
        save_graph(G, i, n_obstacles, target_vertex)


if __name__ == "__main__":
    main()