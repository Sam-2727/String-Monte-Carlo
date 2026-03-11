"""
Ribbon graph generator for Kontsevich's ribbon graph formulation.

Generates non-isomorphic cubic ribbon graphs by:
1. Enumerating rotation systems on base cubic graphs
2. Filtering by face count
3. Removing isomorphic duplicates via graph automorphisms

Outputs disc boundary representations showing sewing patterns.
"""

from itertools import permutations, product as iproduct
from collections import defaultdict
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os

try:
    import networkx as nx
    from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
except ImportError:  # pragma: no cover
    nx = None
    weisfeiler_lehman_graph_hash = None


# ============================================================
# Base cubic graphs
# ============================================================

_CUBIC_GRAPH_CACHE = {}


def _generate_connected_cubic_graphs(
    nv,
    target_count=None,
    max_seeds=None,
    strict_isomorphism=False,
):
    """Generate connected non-isomorphic simple cubic graphs on nv vertices.

    If target_count is provided, generation stops once that many graphs are found.
    Otherwise a plateau heuristic is used, so completeness is not guaranteed.
    With strict_isomorphism=False, WL hash dedup is used for speed.
    """
    if nv in _CUBIC_GRAPH_CACHE:
        return _CUBIC_GRAPH_CACHE[nv]

    if nx is None or weisfeiler_lehman_graph_hash is None:
        raise RuntimeError(
            "networkx is required for dynamic cubic base graph generation"
        )

    reps_by_hash = {}
    reps = []
    seed = 0
    stagnation = 0
    # For unknown sizes, stop after many seeds with no new isomorphism class.
    stagnation_limit = max(2000, 200 * nv)

    if target_count is not None and max_seeds is None:
        max_seeds = max(50000, 5000 * nv)

    while True:
        if target_count is not None and len(reps) >= target_count:
            break
        if target_count is None and stagnation >= stagnation_limit:
            break
        if max_seeds is not None and seed >= max_seeds:
            break

        g = nx.random_regular_graph(3, nv, seed=seed)
        seed += 1
        h = weisfeiler_lehman_graph_hash(g, iterations=5)
        bucket = reps_by_hash.setdefault(h, [])
        if strict_isomorphism:
            if any(nx.is_isomorphic(g, r) for r in bucket):
                stagnation += 1
                continue
            bucket.append(g)
            reps.append(g)
            stagnation = 0
        else:
            # Fast path: WL hash key is treated as canonical in practice.
            if bucket:
                stagnation += 1
                continue
            bucket.append(g)
            reps.append(g)
            stagnation = 0

    base_graphs = []
    verts = list(range(1, nv + 1))
    for g in reps:
        edges = []
        for a, b in sorted(g.edges()):
            edges.append((a + 1, b + 1))
        base_graphs.append((edges, verts))

    # If fast hash-only mode failed to reach the requested target, retry strictly.
    if target_count is not None and len(base_graphs) < target_count and not strict_isomorphism:
        return _generate_connected_cubic_graphs(
            nv,
            target_count=target_count,
            max_seeds=max_seeds,
            strict_isomorphism=True,
        )

    _CUBIC_GRAPH_CACHE[nv] = base_graphs
    return _CUBIC_GRAPH_CACHE[nv]

def _get_base_graphs(nv, ne):
    """Known cubic graphs for small vertex counts."""
    if nv == 2 and ne == 3:
        # Theta graph: 2 vertices connected by 3 parallel edges
        return [([(1, 2), (1, 2), (1, 2)], [1, 2])]
    elif nv == 4 and ne == 6:
        # K4
        return [([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], [1, 2, 3, 4])]
    elif nv == 6 and ne == 9:
        # K_{3,3}
        k33 = ([(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6)],
               [1, 2, 3, 4, 5, 6])
        # Triangular prism
        prism = ([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (1, 4), (2, 5), (3, 6)],
                 [1, 2, 3, 4, 5, 6])
        return [k33, prism]
    elif nv >= 8 and nv % 2 == 0 and ne == (3 * nv) // 2:
        target = 19 if nv == 10 else None
        return _generate_connected_cubic_graphs(
            nv,
            target_count=target,
            strict_isomorphism=False,
        )
    return []


# ============================================================
# Core ribbon graph operations
# ============================================================

def _incident_edges(edges, verts):
    """Map each vertex to its list of incident edge indices."""
    inc = {v: [] for v in verts}
    for i, (a, b) in enumerate(edges):
        inc[a].append(i)
        inc[b].append(i)
    return inc


def _next_half_edge(frm, to, eidx, edges, rotation):
    """Navigate to next half-edge in face traversal.

    Arriving at vertex 'to' via edge eidx, find the next edge in the
    cyclic rotation at 'to', then traverse it away from 'to'.
    """
    rot = rotation[to]
    pos = rot.index(eidx)
    nxt = rot[(pos + 1) % len(rot)]
    a, b = edges[nxt]
    if a == to:
        return (to, b, nxt)
    else:
        return (to, a, nxt)


def _count_faces(edges, verts, rotation):
    """Count faces by tracing half-edge orbits."""
    visited = set()
    faces = 0
    for i, (a, b) in enumerate(edges):
        for he in [(a, b, i), (b, a, i)]:
            if he in visited:
                continue
            faces += 1
            cur = he
            while cur not in visited:
                visited.add(cur)
                cur = _next_half_edge(*cur, edges, rotation)
    return faces


def _get_all_face_boundaries(edges, verts, rotation):
    """Get all face boundaries as lists of (from, to, edge_idx) triples."""
    visited = set()
    faces = []
    for i, (a, b) in enumerate(edges):
        for he in [(a, b, i), (b, a, i)]:
            if he in visited:
                continue
            face = []
            cur = he
            while cur not in visited:
                visited.add(cur)
                face.append(cur)
                cur = _next_half_edge(*cur, edges, rotation)
            faces.append(face)
    return faces


def _trace_boundary(edges, verts, rotation):
    """Trace the first face boundary. For F=1 graphs, this is the full boundary."""
    start = (edges[0][0], edges[0][1], 0)
    boundary = []
    cur = start
    visited = set()
    while cur not in visited:
        visited.add(cur)
        boundary.append(cur)
        cur = _next_half_edge(*cur, edges, rotation)
    return boundary


# ============================================================
# Rotation system enumeration
# ============================================================

def _find_valid_rotations(edges, verts, target_faces):
    """Find all rotation systems yielding the target face count."""
    inc = _incident_edges(edges, verts)
    vert_perms = []
    for v in verts:
        # Cyclic rotations define the same local ribbon structure.
        # Keep one representative per cyclic class by fixing first edge.
        fixed_first = min(inc[v])
        reps = [list(p) for p in permutations(inc[v]) if p[0] == fixed_first]
        vert_perms.append(reps)

    results = []
    for combo in iproduct(*vert_perms):
        rotation = {verts[i]: list(combo[i]) for i in range(len(verts))}
        if _count_faces(edges, verts, rotation) == target_faces:
            results.append((edges, verts, rotation))
    return results


def _unique_valid_rotations_for_base(edges, verts, target_faces, verbose=False):
    """Enumerate and deduplicate valid rotations for one base graph."""
    autos = _compute_automorphisms(edges, verts)
    valid_count = 0
    seen = set()
    unique = []

    inc = _incident_edges(edges, verts)
    vert_perms = []
    for v in verts:
        fixed_first = min(inc[v])
        reps = [list(p) for p in permutations(inc[v]) if p[0] == fixed_first]
        vert_perms.append(reps)

    for combo in iproduct(*vert_perms):
        rotation = {verts[i]: list(combo[i]) for i in range(len(verts))}
        if _count_faces(edges, verts, rotation) != target_faces:
            continue
        valid_count += 1
        rg = (edges, verts, rotation)
        key_rg = _canonical_ribbon_key(rg, autos)
        if key_rg in seen:
            continue
        seen.add(key_rg)
        unique.append(rg)

    if verbose:
        print(
            f"  Base graph ({len(verts)}V, {len(edges)}E): "
            f"{len(autos)} automorphisms, {valid_count} rotation systems, {len(unique)} unique"
        )
    return unique, valid_count


def _unique_valid_rotations_for_base_task(args):
    """Worker wrapper for multiprocessing."""
    edges, verts, target_faces = args
    unique, valid_count = _unique_valid_rotations_for_base(
        edges, verts, target_faces, verbose=False
    )
    return unique, valid_count, len(verts), len(edges)


# ============================================================
# Isomorphism detection
# ============================================================

def _compute_automorphisms(edges, verts):
    """Compute graph automorphisms by brute force (fine for small graphs)."""
    n = len(verts)
    if nx is not None and n >= 8:
        g = nx.Graph()
        g.add_nodes_from(verts)
        g.add_edges_from(edges)
        gm = nx.algorithms.isomorphism.GraphMatcher(g, g)
        return [dict(m) for m in gm.isomorphisms_iter()]

    edge_ms = sorted(tuple(sorted(e)) for e in edges)
    autos = []
    for perm in permutations(range(n)):
        vmap = {verts[i]: verts[perm[i]] for i in range(n)}
        mapped = sorted(tuple(sorted([vmap[a], vmap[b]])) for a, b in edges)
        if mapped == edge_ms:
            autos.append(vmap)
    return autos


def _cyclic_eq(l1, l2):
    """Check if l1 is a cyclic rotation of l2."""
    if len(l1) != len(l2):
        return False
    n = len(l1)
    return any(l1[i:] + l1[:i] == l2 for i in range(n))


def _min_cyclic_tuple(seq):
    """Canonical representative of a cyclic list."""
    n = len(seq)
    return min(tuple(seq[i:] + seq[:i]) for i in range(n))


def _build_pair_groups(edges, vmap):
    """Build source/target edge groups compatible with a vertex map."""
    edge_pairs = [tuple(sorted(e)) for e in edges]
    mapped_pairs = [tuple(sorted([vmap[a], vmap[b]])) for a, b in edges]

    src_by_pair = defaultdict(list)
    for i, mp in enumerate(mapped_pairs):
        src_by_pair[mp].append(i)

    tgt_by_pair = defaultdict(list)
    for j, ep in enumerate(edge_pairs):
        tgt_by_pair[ep].append(j)

    pair_groups = []
    for pair, srcs in src_by_pair.items():
        tgts = tgt_by_pair.get(pair, [])
        if len(srcs) != len(tgts):
            return None
        pair_groups.append((srcs, tgts))
    return pair_groups


def _iter_edge_maps(pair_groups, idx=0, emap=None):
    """Yield all edge maps compatible with pair groups."""
    if emap is None:
        emap = {}
    if idx == len(pair_groups):
        yield emap
        return
    srcs, tgts = pair_groups[idx]
    for tp in permutations(tgts):
        new_emap = dict(emap)
        for s, t in zip(srcs, tp):
            new_emap[s] = t
        yield from _iter_edge_maps(pair_groups, idx + 1, new_emap)


def _is_simple_graph(edges):
    """Return True iff there are no parallel edges or loops."""
    seen = set()
    for a, b in edges:
        if a == b:
            return False
        p = (a, b) if a < b else (b, a)
        if p in seen:
            return False
        seen.add(p)
    return True


def _rotation_signature(rg, vmap, emap):
    """Canonical signature of a mapped ribbon graph rotation."""
    _, verts, rot = rg
    by_vertex = []
    for v in sorted(verts):
        mapped_v = vmap[v]
        mapped_rot = [emap[e] for e in rot[v]]
        by_vertex.append((mapped_v, _min_cyclic_tuple(mapped_rot)))
    return tuple(sorted(by_vertex))


def _canonical_ribbon_key(rg, autos):
    """Compute canonical key under graph automorphisms."""
    edges, _, _ = rg
    simple = _is_simple_graph(edges)
    pair_to_idx = None
    if simple:
        pair_to_idx = {}
        for i, (a, b) in enumerate(edges):
            p = (a, b) if a < b else (b, a)
            pair_to_idx[p] = i

    best = None
    for vmap in autos:
        if simple:
            # For simple graphs, vertex map determines edge map uniquely.
            emap = {}
            for i, (a, b) in enumerate(edges):
                ma, mb = vmap[a], vmap[b]
                p = (ma, mb) if ma < mb else (mb, ma)
                emap[i] = pair_to_idx[p]
            sig = _rotation_signature(rg, vmap, emap)
            if best is None or sig < best:
                best = sig
            continue

        pair_groups = _build_pair_groups(edges, vmap)
        if pair_groups is None:
            continue
        for emap in _iter_edge_maps(pair_groups):
            sig = _rotation_signature(rg, vmap, emap)
            if best is None or sig < best:
                best = sig
    return best


def _check_isomorphism(rg1, rg2, vmap, edges):
    """Check if vertex map 'vmap' induces a ribbon graph isomorphism rg1 -> rg2."""
    rot1, rot2 = rg1[2], rg2[2]
    verts = rg1[1]

    edge_pairs = [tuple(sorted(e)) for e in edges]
    mapped_pairs = [tuple(sorted([vmap[a], vmap[b]])) for a, b in edges]

    # Group source edges by their mapped pair
    src_by_pair = defaultdict(list)
    for i, mp in enumerate(mapped_pairs):
        src_by_pair[mp].append(i)

    tgt_by_pair = defaultdict(list)
    for j, ep in enumerate(edge_pairs):
        tgt_by_pair[ep].append(j)

    # Build pair groups for edge map candidates
    pair_groups = []
    for pair in src_by_pair:
        srcs = src_by_pair[pair]
        tgts = tgt_by_pair.get(pair, [])
        if len(srcs) != len(tgts):
            return False
        pair_groups.append((srcs, tgts))

    def try_maps(idx, emap):
        if idx == len(pair_groups):
            # Verify rotation match
            for v in verts:
                m_rot = [emap[e] for e in rot1[v]]
                if not _cyclic_eq(m_rot, rot2[vmap[v]]):
                    return False
            return True
        srcs, tgts = pair_groups[idx]
        for tp in permutations(tgts):
            new_emap = dict(emap)
            for s, t in zip(srcs, tp):
                new_emap[s] = t
            if try_maps(idx + 1, new_emap):
                return True
        return False

    return try_maps(0, {})


def _remove_isomorphisms(ribbon_graphs, verbose=False):
    """Remove isomorphic duplicates, grouping by base graph."""
    by_graph = defaultdict(list)
    for rg in ribbon_graphs:
        key = tuple(rg[0])
        by_graph[key].append(rg)

    unique_all = []
    for key, group in by_graph.items():
        edges = group[0][0]
        verts = group[0][1]
        autos = _compute_automorphisms(edges, verts)
        if verbose:
            print(
                f"  Base graph ({len(verts)}V, {len(edges)}E): "
                f"{len(autos)} automorphisms, {len(group)} rotation systems"
            )

        seen = set()
        unique = []
        for rg in group:
            key_rg = _canonical_ribbon_key(rg, autos)
            if key_rg in seen:
                continue
            seen.add(key_rg)
            unique.append(rg)
        unique_all.extend(unique)

    return unique_all


# ============================================================
# Main generation function
# ============================================================

def generate_ribbon_graphs(n_faces, n_edges, verbose=False, workers=None):
    """Generate non-isomorphic cubic ribbon graphs.

    Parameters:
        n_faces: number of faces
        n_edges: number of edges

    Returns:
        List of (edges, vertices, rotation) tuples.
    """
    n_vertices = 2 * n_edges // 3
    genus = (2 - n_vertices + n_edges - n_faces) // 2
    if verbose:
        print(f"V={n_vertices}, E={n_edges}, F={n_faces}, g={genus}")

    base_graphs = _get_base_graphs(n_vertices, n_edges)
    if not base_graphs:
        if verbose:
            print("No base cubic graphs available")
        return []
    if verbose:
        print(f"Using {len(base_graphs)} base cubic graph(s)")

    unique_all = []
    total_valid_rotation_systems = 0

    if workers is None:
        # Safe default across restricted runtimes. Use >1 explicitly to parallelize.
        workers = 1
    workers = max(1, int(workers))

    if workers == 1 or len(base_graphs) <= 1:
        for edges, verts in base_graphs:
            unique_base, valid_count = _unique_valid_rotations_for_base(
                edges, verts, n_faces, verbose=verbose
            )
            total_valid_rotation_systems += valid_count
            unique_all.extend(unique_base)
    else:
        tasks = [(edges, verts, n_faces) for edges, verts in base_graphs]
        executor = None
        try:
            executor = ProcessPoolExecutor(max_workers=workers)
        except (PermissionError, OSError):
            # Sandboxed environments may disallow process semaphores.
            executor = ThreadPoolExecutor(max_workers=workers)

        with executor as ex:
            futures = [ex.submit(_unique_valid_rotations_for_base_task, t) for t in tasks]
            for fut in as_completed(futures):
                unique_base, valid_count, nv, ne = fut.result()
                total_valid_rotation_systems += valid_count
                unique_all.extend(unique_base)
                if verbose:
                    print(
                        f"  Base graph ({nv}V, {ne}E): "
                        f"{valid_count} rotation systems, {len(unique_base)} unique"
                    )
    if verbose:
        print(f"Total valid rotation systems: {total_valid_rotation_systems}")
        print(f"Non-isomorphic ribbon graphs: {len(unique_all)}")

    return unique_all


def genus_face_to_edges(genus, n_faces):
    """For connected cubic graphs, compute edge count from genus and face count."""
    # Euler + cubic condition:
    # 2 - 2g = V - E + F and 3V = 2E -> E = 3(F - 2 + 2g)
    if genus < 0 or n_faces <= 0:
        raise ValueError("genus must be >= 0 and n_faces must be >= 1")
    n_edges = 3 * (n_faces - 2 + 2 * genus)
    if n_edges <= 0:
        raise ValueError("Invalid (genus, faces) for connected cubic ribbon graphs")
    # Cubic graph requires V = 2E/3 to be an integer.
    if (2 * n_edges) % 3 != 0:
        raise ValueError("Invalid (genus, faces): cubic vertex count is non-integer")
    return n_edges


def generate_ribbon_graphs_fixed_genus(genus, n_faces=1, verbose=False, workers=None):
    """Optimized fixed-genus entry point.

    Returns non-isomorphic cubic ribbon graphs for the given genus and face count.
    """
    n_edges = genus_face_to_edges(genus, n_faces)
    return generate_ribbon_graphs(n_faces, n_edges, verbose=verbose, workers=workers)


# ============================================================
# Disc boundary output
# ============================================================

def get_disc_boundary(rg):
    """Get disc boundary data for an F=1 ribbon graph.

    Returns dict with:
        'boundary': list of (from_vertex, to_vertex, edge_idx) triples
        'edge_sequence': list of 1-indexed edge labels around disc
        'vertex_sequence': list of vertex labels around disc
        'sewing': dict mapping edge_label -> (pos1, pos2) (1-indexed positions)
    """
    edges, verts, rotation = rg
    boundary = _trace_boundary(edges, verts, rotation)

    edge_seq = [e + 1 for _, _, e in boundary]
    vert_seq = [f for f, _, _ in boundary]

    edge_pos = defaultdict(list)
    for i, (_, _, e) in enumerate(boundary):
        edge_pos[e].append(i + 1)  # 1-indexed

    sewing = {e + 1: tuple(positions) for e, positions in edge_pos.items()}

    return {
        'boundary': boundary,
        'edge_sequence': edge_seq,
        'vertex_sequence': vert_seq,
        'sewing': sewing,
    }


def get_graph_and_sewing_data(rg):
    """Return essential output data: labeled edges and sewing pairs (for F=1)."""
    edges, verts, rotation = rg
    nf = _count_faces(edges, verts, rotation)
    data = {
        "edges": [(i + 1, a, b) for i, (a, b) in enumerate(edges)],
        "sewing_pairs": None,
    }
    if nf == 1:
        sewing = get_disc_boundary(rg)["sewing"]
        data["sewing_pairs"] = [(e, p1, p2) for e, (p1, p2) in sorted(sewing.items())]
    return data


def generate_essential_data_fixed_genus(genus, n_faces=1, verbose=False, workers=None):
    """Generate essential output data only (edges + sewing pairs)."""
    rgs = generate_ribbon_graphs_fixed_genus(
        genus, n_faces=n_faces, verbose=verbose, workers=workers
    )
    return [get_graph_and_sewing_data(rg) for rg in rgs]


def get_ribbon_graph_report(rg):
    """Build a detailed report for a ribbon graph.

    Returns dict with:
        'edges': list of (edge_label, v1, v2)
        'rotation_system': dict vertex -> cyclic list of edge labels
        'faces': list of face boundaries as (from, to, edge_label) triples
        'boundary': same shape as get_disc_boundary for F=1, else None
        'sewn_half_edge_pairs': list of explicit sewn half-edge pair records
    """
    edges, verts, rotation = rg

    edges_labeled = [(i + 1, a, b) for i, (a, b) in enumerate(edges)]
    rotation_system = {v: [e + 1 for e in rotation[v]] for v in sorted(verts)}

    faces = []
    for face in _get_all_face_boundaries(edges, verts, rotation):
        faces.append([(frm, to, e + 1) for frm, to, e in face])

    report = {
        "edges": edges_labeled,
        "rotation_system": rotation_system,
        "faces": faces,
        "boundary": None,
        "sewn_half_edge_pairs": [],
    }

    if len(faces) != 1:
        return report

    boundary_data = get_disc_boundary(rg)
    boundary = boundary_data["boundary"]
    sewing = boundary_data["sewing"]

    sewn_pairs = []
    for edge_label in sorted(sewing):
        p1, p2 = sewing[edge_label]
        he1 = boundary[p1 - 1]
        he2 = boundary[p2 - 1]
        sewn_pairs.append(
            {
                "edge": edge_label,
                "segments": (p1, p2),
                "half_edges": (
                    (he1[0], he1[1], he1[2] + 1),
                    (he2[0], he2[1], he2[2] + 1),
                ),
            }
        )

    report["boundary"] = boundary_data
    report["sewn_half_edge_pairs"] = sewn_pairs
    return report


def print_disc(rg, idx=None):
    """Print disc boundary representation for an F=1 ribbon graph."""
    edges, verts, rotation = rg
    ne = len(edges)
    nv = len(verts)
    genus = (2 - nv + ne - 1) // 2
    data = get_disc_boundary(rg)

    label = f"Graph {idx}" if idx is not None else "Graph"
    print(f"\n{'=' * 60}")
    print(f"{label}: V={nv}, E={ne}, g={genus}, boundary={len(data['boundary'])}")
    print(f"{'=' * 60}")

    print(f"\nCounterclockwise boundary:")
    print(f"{'Seg':>4} {'Vertex':>7} {'Edge':>5} {'Sewed to':>10}")
    print(f"{'-' * 32}")

    sewing_inv = {}
    for edge_label, (p1, p2) in data['sewing'].items():
        sewing_inv[p1] = p2
        sewing_inv[p2] = p1

    for i, (frm, to, eidx) in enumerate(data['boundary']):
        pos = i + 1
        partner = sewing_inv[pos]
        print(f"{pos:4d} {frm:7d} {eidx + 1:5d} {partner:10d}")

    print(f"\nEdge sequence:   {data['edge_sequence']}")
    print(f"Vertex sequence: {data['vertex_sequence']}")

    print(f"\nSewing pairs:")
    for edge_label in sorted(data['sewing']):
        p1, p2 = data['sewing'][edge_label]
        print(f"  Edge {edge_label}: segment {p1} <-> segment {p2}")


def print_graph_and_sewing(rg, idx=None):
    """Print a compact representation: graph edges + sewing pairs."""
    edges, verts, rotation = rg
    ne = len(edges)
    nv = len(verts)
    nf = _count_faces(edges, verts, rotation)
    genus = (2 - nv + ne - nf) // 2
    label = f"Graph {idx}" if idx is not None else "Graph"
    print(f"\n{label}: V={nv}, E={ne}, F={nf}, g={genus}")
    print(f"  Edges: {[ (i + 1, a, b) for i, (a, b) in enumerate(edges) ]}")
    if nf == 1:
        sewing = get_disc_boundary(rg)["sewing"]
        pairs = [(e, sewing[e][0], sewing[e][1]) for e in sorted(sewing)]
        print(f"  Sewing pairs (edge, seg1, seg2): {pairs}")


def print_ribbon_graph_report(rg, idx=None):
    """Print a full structural report for a ribbon graph."""
    edges, verts, rotation = rg
    ne = len(edges)
    nv = len(verts)
    nf = _count_faces(edges, verts, rotation)
    genus = (2 - nv + ne - nf) // 2
    report = get_ribbon_graph_report(rg)

    label = f"Graph {idx}" if idx is not None else "Graph"
    print(f"\n{'=' * 60}")
    print(f"{label}: V={nv}, E={ne}, F={nf}, g={genus}")
    print(f"{'=' * 60}")

    print("\nEdges:")
    for edge_label, a, b in report["edges"]:
        print(f"  e{edge_label}: {a} -- {b}")

    print("\nRotation system (counterclockwise at each vertex):")
    for v in sorted(report["rotation_system"]):
        print(f"  v{v}: {report['rotation_system'][v]}")

    print("\nFace boundaries:")
    for i, face in enumerate(report["faces"], start=1):
        print(f"  Face {i}: {face}")

    if report["boundary"] is None:
        return

    print("\nSewn half-edge pairs (F=1 boundary gluing):")
    for pair in report["sewn_half_edge_pairs"]:
        p1, p2 = pair["segments"]
        he1, he2 = pair["half_edges"]
        print(
            f"  e{pair['edge']}: seg {p1} {he1} <-> seg {p2} {he2}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    cases = [
        ("Genus 1", 1, 3, 1),
        ("Genus 0", 3, 3, 1),
        ("Genus 0", 4, 6, 1),
        ("Genus 2", 1, 9, 4),
    ]
    for label, n_faces, n_edges, expected in cases:
        t0 = perf_counter()
        rgs = generate_ribbon_graphs(n_faces, n_edges, verbose=False)
        dt = perf_counter() - t0
        assert len(rgs) == expected, f"Expected {expected}, got {len(rgs)}"
        g = (2 - (2 * n_edges // 3) + n_edges - n_faces) // 2
        print(f"\n{label} case: F={n_faces}, E={n_edges}, g={g}, runtime={dt:.4f}s")
        for i, rg in enumerate(rgs, start=1):
            print_graph_and_sewing(rg, i)
