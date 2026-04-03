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


# ============================================================
# Base cubic graphs
# ============================================================

def _get_base_graphs(nv, ne):
    """Known cubic graphs for small vertex counts."""
    if nv == 2 and ne == 3:
        # Theta graph: 2 vertices connected by 3 parallel edges
        return [([(1, 2), (1, 2), (1, 2)], [1, 2])]
    elif nv == 4 and ne == 6:
        # K4
        return [([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], [1, 2, 3, 4])]
    elif nv == 6 and ne == 9:
        # All connected cubic multigraphs on 6 vertices, up to graph isomorphism.
        return [
            ([
                (1, 2), (1, 2), (1, 3),
                (2, 4), (3, 4), (3, 5),
                (4, 6), (5, 6), (5, 6),
            ], [1, 2, 3, 4, 5, 6]),
            ([
                (1, 2), (1, 2), (1, 3),
                (2, 4), (3, 5), (3, 5),
                (4, 6), (4, 6), (5, 6),
            ], [1, 2, 3, 4, 5, 6]),
            ([
                (1, 2), (1, 2), (1, 3),
                (2, 4), (3, 5), (3, 6),
                (4, 5), (4, 6), (5, 6),
            ], [1, 2, 3, 4, 5, 6]),
            ([
                (1, 2), (1, 2), (1, 3),
                (2, 3), (3, 4), (4, 5),
                (4, 6), (5, 6), (5, 6),
            ], [1, 2, 3, 4, 5, 6]),
            ([
                (1, 4), (1, 5), (1, 6),
                (2, 4), (2, 5), (2, 6),
                (3, 4), (3, 5), (3, 6),
            ], [1, 2, 3, 4, 5, 6]),
            ([
                (1, 2), (2, 3), (3, 1),
                (4, 5), (5, 6), (6, 4),
                (1, 4), (2, 5), (3, 6),
            ], [1, 2, 3, 4, 5, 6]),
        ]
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
    vert_perms = [list(permutations(inc[v])) for v in verts]

    results = []
    for combo in iproduct(*vert_perms):
        rotation = {verts[i]: list(combo[i]) for i in range(len(verts))}
        if _count_faces(edges, verts, rotation) == target_faces:
            results.append((edges, verts, rotation))
    return results


# ============================================================
# Isomorphism detection
# ============================================================

def _compute_automorphisms(edges, verts):
    """Compute graph automorphisms by brute force (fine for small graphs)."""
    n = len(verts)
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


def _remove_isomorphisms(ribbon_graphs):
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
        print(f"  Base graph ({len(verts)}V, {len(edges)}E): "
              f"{len(autos)} automorphisms, {len(group)} rotation systems")

        unique = [group[0]]
        for rg in group[1:]:
            is_dup = False
            for u in unique:
                for auto in autos:
                    if _check_isomorphism(rg, u, auto, edges):
                        is_dup = True
                        break
                if is_dup:
                    break
            if not is_dup:
                unique.append(rg)
        unique_all.extend(unique)

    return unique_all


# ============================================================
# Main generation function
# ============================================================

def generate_ribbon_graphs(n_faces, n_edges):
    """Generate non-isomorphic cubic ribbon graphs.

    Parameters:
        n_faces: number of faces
        n_edges: number of edges

    Returns:
        List of (edges, vertices, rotation) tuples.
    """
    n_vertices = 2 * n_edges // 3
    genus = (2 - n_vertices + n_edges - n_faces) // 2
    print(f"V={n_vertices}, E={n_edges}, F={n_faces}, g={genus}")

    base_graphs = _get_base_graphs(n_vertices, n_edges)
    if not base_graphs:
        print("No base cubic graphs available")
        return []
    print(f"Using {len(base_graphs)} base cubic graph(s)")

    all_rgs = []
    for edges, verts in base_graphs:
        rgs = _find_valid_rotations(edges, verts, n_faces)
        all_rgs.extend(rgs)
    print(f"Total valid rotation systems: {len(all_rgs)}")

    if not all_rgs:
        return []

    unique = _remove_isomorphisms(all_rgs)
    print(f"Non-isomorphic ribbon graphs: {len(unique)}")
    return unique


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


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Genus 1, F=1 (theta graph)")
    print("=" * 60)
    rgs = generate_ribbon_graphs(1, 3)
    assert len(rgs) == 1, f"Expected 1, got {len(rgs)}"
    for i, rg in enumerate(rgs):
        print_disc(rg, i + 1)

    print("\n")
    print("=" * 60)
    print("TEST 2: Genus 0, F=3 (theta graph, planar)")
    print("=" * 60)
    rgs = generate_ribbon_graphs(3, 3)
    assert len(rgs) == 1, f"Expected 1, got {len(rgs)}"

    print("\n")
    print("=" * 60)
    print("TEST 3: Genus 0, F=4 (K4, planar)")
    print("=" * 60)
    rgs = generate_ribbon_graphs(4, 6)
    assert len(rgs) == 1, f"Expected 1, got {len(rgs)}"

    print("\n")
    print("=" * 60)
    print("TEST 4: Genus 2, F=1")
    print("=" * 60)
    rgs = generate_ribbon_graphs(1, 9)
    assert len(rgs) == 4, f"Expected 4, got {len(rgs)}"
    for i, rg in enumerate(rgs):
        print_disc(rg, i + 1)
