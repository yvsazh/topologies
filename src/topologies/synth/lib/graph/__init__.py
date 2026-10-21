"""Random graphs, exact structural labels, and generator-tree drawing.

    g = synth.graph.sbm(sizes=[8, 8], probabilities=[[.8, .1], [.1, .8]])
    synth.graph(synth.record(graph=g), html="tree.html")
"""
from __future__ import annotations

import heapq
import html as html_lib
import math
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from topologies import synth


def _size(n, name="n"):
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(n)


def _probability(p):
    if not math.isfinite(float(p)) or not 0 <= p <= 1:
        raise ValueError("Probability must be finite and between zero and one")
    return float(p)


def _graph(n, edges, *, directed=False, x=None, y=None, edge_attr=None):
    pairs = sorted({tuple(e) for e in edges}) if edge_attr is None else list(edges)
    return synth.Graph({"num_nodes": n, "edges": [list(e) for e in pairs],
                        "directed": directed, "x": x, "y": y, "edge_attr": edge_attr},
                       directed=directed)


@synth.node(type=synth.Graph, tags=["graph", "random"], example={"n": 16, "p": .2})
def erdos_renyi(r, *, n: int = 32, p: float = .1, directed: bool = False):
    """Independent edges; no self loops, one edge per undirected pair."""
    n, p = _size(n), _probability(p)
    edges = [(u, v) for u in range(n) for v in range(n) if u != v and
             (directed or u < v) and r.random() < p]
    return _graph(n, edges, directed=directed)


@synth.node(type=synth.Graph, tags=["graph", "random"], example={"n": 24, "m": 2})
def barabasi_albert(r, *, n: int = 32, m: int = 2):
    """Preferential attachment from a clique of m+1 nodes, without duplicate edges."""
    n, m = _size(n), _size(m, "m")
    if not 1 <= m < n:
        raise ValueError("Preferential attachment requires 1 <= m < n")
    edges = [(u, v) for u in range(m + 1) for v in range(u + 1, m + 1)]
    degree = [m] * (m + 1) + [0] * (n - m - 1)
    for v in range(m + 1, n):
        candidates = list(range(v))
        for _ in range(m):
            u = r.weighted(candidates, [degree[k] for k in candidates])
            candidates.remove(u)
            edges.append((u, v))
            degree[u] += 1
            degree[v] += 1
    return _graph(n, edges)


@synth.node(type=synth.Graph, tags=["graph", "random"], example={"n": 24, "k": 4, "p": .1})
def watts_strogatz(r, *, n: int = 32, k: int = 4, p: float = .1):
    """An even-degree ring lattice with independently rewired clockwise edges."""
    n, k, p = _size(n), _size(k, "k"), _probability(p)
    if n < 1 or k >= n or k % 2:
        raise ValueError("Ring lattice requires positive n and even 0 <= k < n")
    original = [(u, (u + j) % n) for u in range(n) for j in range(1, k // 2 + 1)]
    edges = {tuple(sorted(e)) for e in original}
    for u, v in original:
        if r.random() >= p:
            continue
        candidates = [w for w in range(n) if w not in (u, v) and
                      tuple(sorted((u, w))) not in edges]
        if candidates:
            edges.remove(tuple(sorted((u, v))))
            edges.add(tuple(sorted((u, r.choice(candidates)))))
    return _graph(n, edges)


@synth.node(type=synth.Graph, tags=["graph", "communities"], example={
    "sizes": [8, 8], "probabilities": [[.8, .1], [.1, .8]]})
def sbm(ctx, *, sizes: Sequence[int] = (16, 16),
        probabilities: Sequence[Sequence[float]] = ((.8, .1), (.1, .8))):
    """Undirected stochastic block model with exact community node labels y."""
    sizes = [_size(n, "community size") for n in sizes]
    if not sizes or len(probabilities) != len(sizes) or any(
            len(row) != len(sizes) for row in probabilities):
        raise ValueError("Block probabilities must match community sizes")
    probs = [[_probability(p) for p in row] for row in probabilities]
    if any(probs[i][j] != probs[j][i] for i in range(len(sizes)) for j in range(len(sizes))):
        raise ValueError("Undirected block probabilities must be symmetric")
    labels = [i for i, n in enumerate(sizes) for _ in range(n)]
    edges = [(u, v) for u in range(len(labels)) for v in range(u + 1, len(labels))
             if ctx.rng.random() < probs[labels[u]][labels[v]]]
    ctx.latent(communities=labels)
    return _graph(len(labels), edges, y=labels)


@synth.node(type=synth.Graph, tags=["graph", "tree"], example={"n": 16})
def tree(r, *, n: int = 32, root: int = 0, directed: bool = False):
    """Uniform labeled tree from a Prüfer sequence; directed trees point away from root."""
    n = _size(n)
    if n < 1 or not 0 <= root < n or int(root) != root:
        raise ValueError("Tree requires positive n and a valid root")
    code = [r.randrange(n) for _ in range(max(0, n - 2))]
    degree = [1] * n
    for v in code:
        degree[v] += 1
    leaves = [v for v in range(n) if degree[v] == 1]
    heapq.heapify(leaves)
    edges = []
    for v in code:
        u = heapq.heappop(leaves)
        edges.append(tuple(sorted((u, v))))
        degree[v] -= 1
        if degree[v] == 1:
            heapq.heappush(leaves, v)
    if n > 1:
        edges.append(tuple(sorted((heapq.heappop(leaves), heapq.heappop(leaves)))))
    if directed:
        neighbours = [[] for _ in range(n)]
        for u, v in edges:
            neighbours[u].append(v)
            neighbours[v].append(u)
        edges, seen, queue = [], {root}, deque([root])
        while queue:
            u = queue.popleft()
            for v in neighbours[u]:
                if v not in seen:
                    edges.append((u, v))
                    seen.add(v)
                    queue.append(v)
    return _graph(n, edges, directed=directed)


@synth.node(type=synth.Graph, tags=["graph", "dag"], example={"n": 16, "p": .2})
def dag(ctx, *, n: int = 32, p: float = .1):
    """Random directed acyclic graph; y gives each node's topological rank."""
    n, p = _size(n), _probability(p)
    order = list(range(n))
    ctx.rng.shuffle(order)
    ranks = [0] * n
    for rank, v in enumerate(order):
        ranks[v] = rank
    edges = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n)
             if ctx.rng.random() < p]
    ctx.latent(topological_order=order)
    return _graph(n, edges, directed=True, y=ranks)


@synth.node(type=synth.Graph, tags=["graph", "grid"], example={"rows": 4, "cols": 4})
def grid(*, rows: int = 8, cols: int = 8, wrap: bool = False):
    """Rectangular nearest-neighbour grid with coordinates as node features."""
    rows, cols = _size(rows, "rows"), _size(cols, "cols")
    if not rows or not cols:
        raise ValueError("Grid dimensions must be positive")
    edges = set()
    for i in range(rows):
        for j in range(cols):
            for di, dj in ((1, 0), (0, 1)):
                a, b = i + di, j + dj
                if wrap:
                    a, b = a % rows, b % cols
                if a < rows and b < cols and (a, b) != (i, j):
                    edges.add(tuple(sorted((i * cols + j, a * cols + b))))
    return _graph(rows * cols, edges, x=[[i, j] for i in range(rows) for j in range(cols)])


@synth.node(type=synth.Graph, tags=["graph", "geometric"],
            example={"n": 24, "dim": 2, "radius": .3})
def geometric(r, *, n: int = 32, dim: int = 2, radius: float = .25):
    """Uniform points in the unit cube, connected by Euclidean distance <= radius."""
    n, dim = _size(n), _size(dim, "dim")
    if dim < 1 or not math.isfinite(radius) or radius < 0:
        raise ValueError("dim must be positive and radius must be finite and nonnegative")
    x = torch.rand(n, dim, generator=r.torch)
    distances = torch.cdist(x, x)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if distances[u, v] <= radius]
    return _graph(n, edges, x=x)


def _adjacency(x):
    p = synth.Graph.parts(x)
    directed = p.get("directed", getattr(x, "directed", False))
    neighbours = [set() for _ in range(p["num_nodes"])]
    for u, v in p["edges"]:
        neighbours[u].add(v)
        if not directed:
            neighbours[v].add(u)
    return p, neighbours


@synth.node(type=synth.Series, tags=["graph", "label"], example={
    "x": {"num_nodes": 3, "edges": [[0, 1], [1, 2]]}, "source": 0})
def shortest_paths(x, *, source: int = 0):
    """Unweighted shortest distances; -1 marks unreachable nodes."""
    p, neighbours = _adjacency(x)
    if not 0 <= source < p["num_nodes"] or int(source) != source:
        raise ValueError("source must be a valid node id")
    distance, queue = [-1] * p["num_nodes"], deque([source])
    distance[source] = 0
    while queue:
        u = queue.popleft()
        for v in sorted(neighbours[u]):
            if distance[v] < 0:
                distance[v] = distance[u] + 1
                queue.append(v)
    return torch.tensor(distance, dtype=torch.float32)


@synth.node(type=synth.Label(["False", "True"]), tags=["graph", "label"],
            example={"x": {"num_nodes": 3, "edges": [[0, 1], [1, 2]]}})
def is_connected(x, *, strong: bool = False):
    """Weak connectivity by default; strong=True checks reachability in both directions."""
    p, neighbours = _adjacency(x)
    if not neighbours:
        return False
    reverse = [set() for _ in neighbours]
    for u, edges in enumerate(neighbours):
        for v in edges:
            reverse[v].add(u)
    if not strong:
        neighbours = [a | b for a, b in zip(neighbours, reverse, strict=True)]

    def reached(adjacency):
        seen, queue = {0}, deque([0])
        while queue:
            for v in adjacency[queue.popleft()]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        return len(seen) == p["num_nodes"]

    return reached(neighbours) and (not strong or reached(reverse))


@synth.node(type=synth.Label(["False", "True"]), tags=["graph", "label"],
            example={"x": {"num_nodes": 3, "edges": [[0, 1], [1, 2], [0, 2]]}})
def has_cycle(x):
    """Exact cycle existence, respecting graph direction."""
    p, neighbours = _adjacency(x)
    if p.get("directed", getattr(x, "directed", False)):
        degree = [0] * len(neighbours)
        for edges in neighbours:
            for v in edges:
                degree[v] += 1
        queue, visited = deque(i for i, d in enumerate(degree) if d == 0), 0
        while queue:
            visited += 1
            for v in neighbours[queue.popleft()]:
                degree[v] -= 1
                if degree[v] == 0:
                    queue.append(v)
        return visited != len(neighbours)
    parent = list(range(p["num_nodes"]))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for u, v in p["edges"]:
        a, b = find(u), find(v)
        if a == b:
            return True
        parent[a] = b
    return False


@synth.node(type=synth.Graph, tags=["graph", "features"], example={
    "x": {"num_nodes": 2, "edges": [[0, 1]]}, "node_features": [[1.], [2.]]})
def with_features(x, *, node_features: Any = None, edge_features: Any = None):
    """Attach resolved nodes or containers as node/edge features, preserving labels."""
    p = synth.Graph.parts(x)
    if node_features is not None and len(node_features) != p["num_nodes"]:
        raise ValueError("Node feature rows must match num_nodes")
    if edge_features is not None and len(edge_features) != len(p["edges"]):
        raise ValueError("Edge feature rows must match edges")
    return _graph(p["num_nodes"], p["edges"], directed=p.get("directed", False),
                  x=p["x"] if node_features is None else node_features, y=p["y"],
                  edge_attr=p["edge_attr"] if edge_features is None else edge_features)


def draw(node, *, html: str | Path | None = None):
    """Draw the node dependency graph as self-contained SVG; optionally save HTML."""
    root = node.generator if hasattr(node, "generator") else node
    root = synth.as_generator(root) if isinstance(root, Mapping) else root
    if not isinstance(root, synth.Node):
        raise TypeError("Generator graph drawing requires a Node, generator mapping, or Spec")
    nodes, edges, seen = [], [], {}

    def visit(value, parent=None, label="root"):
        if isinstance(value, synth.Node):
            key = id(value)
            if key not in seen:
                seen[key] = len(nodes)
                nodes.append(f"{label}: {value.node_kind()}")
                for name, child in value.params().items():
                    visit(child, seen[key], name)
            if parent is not None:
                edges.append((parent, seen[key]))
        elif isinstance(value, Mapping):
            for name, child in value.items():
                visit(child, parent, str(name))
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                visit(child, parent, f"{label}[{i}]")

    visit(root)
    height = max(90, len(nodes) * 65 + 20)
    lines = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="{height}" '
              'role="img" aria-label="Generator dependencies">')]
    for a, b in edges:
        lines.append(f'<path d="M 350 {a * 65 + 43} C 690 {a * 65 + 43}, '
                     f'690 {b * 65 + 43}, 350 {b * 65 + 43}" fill="none" stroke="#888"/>')
    for i, label in enumerate(nodes):
        lines.append(f'<rect x="20" y="{i * 65 + 20}" width="330" height="46" '
                     'rx="6" fill="#f4f4f4" stroke="#777"/>')
        lines.append(f'<text x="30" y="{i * 65 + 48}" font-family="monospace" '
                     f'font-size="13">{html_lib.escape(label)}</text>')
    lines.append('</svg>')
    svg = "".join(lines)
    if html is not None:
        Path(html).write_text('<!doctype html><meta charset="utf-8"><title>Generator graph</title>'
                              + svg, encoding="utf-8")
    return svg


__all__ = ["barabasi_albert", "dag", "draw", "erdos_renyi", "geometric", "grid", "has_cycle",
           "is_connected", "sbm", "shortest_paths", "tree", "watts_strogatz", "with_features"]

synth.callable_module(__name__, "draw")
