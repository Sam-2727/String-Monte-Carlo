"""
degeneration_analysis.py
========================
Diagnostic script for the period_matrix separating degeneration test.

Goals:
  1. Print boundary words and intersection matrices for all 4 genus-2 graphs.
  2. Identify the prism (dumbbell) graph and its left/right edge blocks.
  3. Find the intersection matrix restricted to each block to determine
     the natural homology cycles for the separating degeneration.
  4. Run a systematic degeneration scan and check:
       (a) |Omega_12| -> 0 as neck grows
       (b) Omega_11 and Omega_22 converge to some values
       (c) Left-right symmetry: if ell_left == ell_right, Omega_11 == Omega_22
  5. Independently compute the expected tau for each component torus
     using the restriction of the genus-2 boundary data to each block.

Usage:
  Run from the notebooks directory:
    python degeneration_analysis.py
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "python")))

import numpy as np
import importlib
import ell_to_tau as et
import partition_function as pf
from ribbon_graph_generator import (
    generate_ribbon_graphs, _trace_boundary, _get_all_face_boundaries,
    _incident_edges
)
importlib.reload(et)

print("=" * 70)
print("SECTION 1: ALL GENUS-2 RIBBON GRAPHS")
print("=" * 70)

rgs = generate_ribbon_graphs(n_faces=1, n_edges=9)
print(f"Number of non-isomorphic F=1 genus-2 graphs: {len(rgs)}")
print()

for idx, rg in enumerate(rgs):
    edges, verts, rotation = rg
    bdy = _trace_boundary(edges, verts, rotation)
    bword = tuple(step[2] for step in bdy)

    print(f"--- rgs[{idx}] ---")
    print(f"  edges:    {edges}")
    print(f"  vertices: {verts}")
    print(f"  rotation: {rotation}")
    print(f"  boundary word: {bword}")

    # edge positions
    ep = {}
    for pos, eidx in enumerate(bword):
        ep.setdefault(eidx, []).append(pos)
    print(f"  edge positions: {ep}")
    print()

print("=" * 70)
print("SECTION 2: FOCUS ON rgs[1] -- intersection matrix and edge blocks")
print("=" * 70)

rg2 = rgs[1]
edges2, verts2, rot2 = rg2
bdy2 = _trace_boundary(edges2, verts2, rot2)
bword2 = tuple(step[2] for step in bdy2)
print(f"rgs[1] boundary word: {bword2}")
print()

chord_data2 = et.edge_chord_intersection_matrix(rg2)
J = chord_data2["intersection_matrix"]
ep2 = chord_data2["edge_positions"]
print("Edge positions in boundary word:")
for e, positions in sorted(ep2.items()):
    print(f"  edge {e}: positions {positions}")
print()

print("Intersection matrix J (9x9):")
print(J)
print()

# Identify which edges from the prism base are in each block
# For the prism: edges 0,1,2 = left triangle; 3,4,5 = right triangle; 6,7,8 = neck
# (based on prism graph definition in ribbon_graph_generator.py)
left_block  = [0, 1, 2]
right_block = [3, 4, 5]
neck_block  = [6, 7, 8]

print("Intersection sub-matrix (left triangle edges 0,1,2):")
J_left = J[np.ix_(left_block, left_block)]
print(J_left)
print()

print("Intersection sub-matrix (right triangle edges 3,4,5):")
J_right = J[np.ix_(right_block, right_block)]
print(J_right)
print()

print("Intersection sub-matrix (neck edges 6,7,8):")
J_neck = J[np.ix_(neck_block, neck_block)]
print(J_neck)
print()

print("Cross-block: left x right =")
print(J[np.ix_(left_block, right_block)])
print()

print("Cross-block: left x neck =")
print(J[np.ix_(left_block, neck_block)])
print()

print("Cross-block: right x neck =")
print(J[np.ix_(right_block, neck_block)])
print()

# Find default homology basis
default_basis = et._find_edge_homology_basis_from_chord_data(chord_data2)["basis_pairs"]
print("Default basis (from automated search):")
for i, pair in enumerate(default_basis, 1):
    print(f"  alpha_{i} = {pair['alpha']}, beta_{i} = {pair['beta']}")
print()

print("=" * 70)
print("SECTION 3: FIND BLOCK-DIAGONAL (DEGENERATION-ADAPTED) BASIS")
print("=" * 70)

# For the separating degeneration, we want:
#   alpha_1, beta_1 supported entirely on edges within left_block union neck_block
#   alpha_2, beta_2 supported entirely on edges within right_block union neck_block
#
# OR better: cycles that each lie entirely within one of:
#   left_block = {0,1,2}, right_block = {3,4,5}
#   (neck edges connect the two sides, so a "separating" basis has
#    alpha_1/beta_1 with nonzero only in left_block, and
#    alpha_2/beta_2 with nonzero only in right_block)
#
# For this to be symplectic, we need J[alpha_i, beta_i] = 1 with
# alpha_i in left_block and beta_i in left_block (for i=1).
#
# Check if left block has a symplectic pair:
print("Looking for symplectic pairs within left_block (edges 0,1,2):")
found_left = []
for ea in left_block:
    for eb in left_block:
        if ea != eb and J[ea, eb] != 0:
            print(f"  J[{ea},{eb}] = {J[ea,eb]}")
            if abs(J[ea, eb]) == 1:
                found_left.append((ea, eb, int(J[ea,eb])))

if found_left:
    print(f"  Found symplectic pairs in left block: {found_left}")
else:
    print("  No symplectic pair purely within left block.")
print()

print("Looking for symplectic pairs within right_block (edges 3,4,5):")
found_right = []
for ea in right_block:
    for eb in right_block:
        if ea != eb and J[ea, eb] != 0:
            print(f"  J[{ea},{eb}] = {J[ea,eb]}")
            if abs(J[ea, eb]) == 1:
                found_right.append((ea, eb, int(J[ea,eb])))

if found_right:
    print(f"  Found symplectic pairs in right block: {found_right}")
else:
    print("  No symplectic pair purely within right block.")
print()

# Try all single-edge bases from left_block x right_block
print("Testing all candidate single-edge basis pairs (left alpha, left beta, right alpha, right beta):")
all_block_bases = []
for la in left_block:
    for lb in left_block:
        if la == lb: continue
        if abs(J[la, lb]) != 1: continue
        for ra in right_block:
            for rb in right_block:
                if ra == rb: continue
                if abs(J[ra, rb]) != 1: continue
                # Check symplecticity and mutual isotropy
                basis_try = [
                    {"alpha": [(la, 1)], "beta": [(lb, int(J[la,lb]))]},
                    {"alpha": [(ra, 1)], "beta": [(rb, int(J[ra,rb]))]},
                ]
                try:
                    validated = et.validate_edge_homology_basis(
                        chord_data2, basis_try, expected_genus=2)
                    print(f"  VALID: alpha_1=edge{la}, beta_1={J[la,lb]}*edge{lb}; "
                          f"alpha_2=edge{ra}, beta_2={J[ra,rb]}*edge{rb}")
                    all_block_bases.append(validated)
                except ValueError:
                    pass

if not all_block_bases:
    print("  No fully block-diagonal single-edge basis found.")
    print("  Will look for 2-edge combinations within each block...")
    # Try combinations (a±b) within each block
    from itertools import combinations
    for la, lb_left, lc in [left_block]:
        pass  # handled separately below
    print()

print()

print("=" * 70)
print("SECTION 4: GENUS-1 COMPARISON -- component torus tau")
print("=" * 70)

# The component torus of the left triangle (edges 0,1,2 with lengths l0,l1,l2)
# in the prism degeneration limit.
#
# KEY QUESTION: what ribbon graph represents each component torus?
#
# The prism's left component K_3 (triangle, vertices 1,2,3, edges 0,1,2)
# does NOT form a theta graph by itself (it's K_3, not K_{1,1} with 3 parallel edges).
#
# However, in the Strebel picture, the period tau of the left torus depends
# on which FACE the left triangle edges trace out, which is determined by
# the full prism rotation system.
#
# Strategy: compute tau by looking at which edges of rgs[1] form a theta-like
# subgraph with an F=1 face boundary restricted to those edges.
#
# For the theta graph comparison:
# If edges {a,b,c} appear in the boundary word as (...a...b...c...a...b...c...)
# (i.e., interleaved twice in succession), then the F=1 boundary restricted to
# {a,b,c} is exactly the theta graph boundary, and periods_improved applies.

print("Boundary word of rgs[1]:", bword2)
print()
print("Appearance of left-block edges in boundary word:")
left_positions = {e: ep2[e] for e in left_block}
print("  left edges positions:", left_positions)
right_positions = {e: ep2[e] for e in right_block}
print("  right edges positions:", right_positions)
neck_positions = {e: ep2[e] for e in neck_block}
print("  neck edges positions:", neck_positions)
print()

# For each block, find the sub-boundary-word ordering
left_order = sorted(left_block, key=lambda e: ep2[e][0])
right_order = sorted(right_block, key=lambda e: ep2[e][0])

print("Left block edges sorted by first boundary occurrence:", left_order)
print("Right block edges sorted by first boundary occurrence:", right_order)
print()

# Check if left block forms a theta-graph-like sub-boundary
# (i.e., do all left-block edges appear consecutively before any neck edges?)
all_left_pos = sorted(sum([list(ep2[e]) for e in left_block], []))
all_right_pos = sorted(sum([list(ep2[e]) for e in right_block], []))
all_neck_pos = sorted(sum([list(ep2[e]) for e in neck_block], []))

print("All left positions (sorted):", all_left_pos)
print("All right positions (sorted):", all_right_pos)
print("All neck positions (sorted):", all_neck_pos)
print()

# The boundary word has 18 positions (9 edges x 2 occurrences = 18 segments)
# Check if left positions and right positions are separated by neck positions
print("Full boundary word with block labels:")
block_labels = {e: 'L' for e in left_block}
block_labels.update({e: 'R' for e in right_block})
block_labels.update({e: 'N' for e in neck_block})
labeled = [f"{block_labels[e]}({e})" for e in bword2]
print("  " + " ".join(labeled))
print()

print("=" * 70)
print("SECTION 5: DEGENERATION SCAN")
print("=" * 70)

# Use the default basis (auto-detected) and the block basis (if found)
# Run for several x values and track the period matrix

ell_torus = [10, 7, 5]   # left = right torus edge lengths (small numbers for speed)

# Reference: genus-1 tau for theta graph with these parameters
L1 = 2 * sum(ell_torus)
l1_ref, l2_ref = ell_torus[0], ell_torus[1]
P_ref = et.periods_improved(L1, l1_ref, l2_ref)
tau_ref = P_ref[1] / P_ref[0]
print(f"Genus-1 reference (theta graph): L={L1}, l1={l1_ref}, l2={l2_ref}")
print(f"  tau_ref (theta graph) = {tau_ref:.6f}")
print()

print("Neck scan: compute Omega for rgs[1] with ell=[10,7,5, 10,7,5, x,x,x]")
print()

x_values = [1, 2, 5, 10, 20, 50]

# First try the default basis
print("Using default auto-detected basis:")
basis_default = et._find_edge_homology_basis_from_chord_data(chord_data2)["basis_pairs"]
print("  basis:", basis_default)
print()

for x in x_values:
    ell = ell_torus + ell_torus + [x, x, x]
    try:
        forms = et.make_cyl_eqn_improved_higher_genus(rg2, ell)
        Omega = et.period_matrix(
            forms=forms,
            ribbon_graph=rg2,
            ell_list=ell,
            custom_cycles=basis_default,
        )
        print(f"  x={x:3d}: Omega_11={Omega[0,0]:.4f}, Omega_22={Omega[1,1]:.4f}, "
              f"Omega_12={Omega[0,1]:.4f}, |Omega_12|={abs(Omega[0,1]):.4f}")
    except Exception as err:
        print(f"  x={x:3d}: ERROR: {err}")

print()

# If we found block-diagonal single-edge bases, test those too
if all_block_bases:
    print("Using block-diagonal basis (first found):")
    block_basis = all_block_bases[0]
    print("  basis:", block_basis)
    print()
    for x in x_values:
        ell = ell_torus + ell_torus + [x, x, x]
        try:
            forms = et.make_cyl_eqn_improved_higher_genus(rg2, ell)
            Omega = et.period_matrix(
                forms=forms,
                ribbon_graph=rg2,
                ell_list=ell,
                custom_cycles=block_basis,
            )
            print(f"  x={x:3d}: Omega_11={Omega[0,0]:.4f}, Omega_22={Omega[1,1]:.4f}, "
                  f"Omega_12={Omega[0,1]:.4f}, |Omega_12|={abs(Omega[0,1]):.4f}")
        except Exception as err:
            print(f"  x={x:3d}: ERROR: {err}")
    print()

print("=" * 70)
print("SECTION 6: CHECK SYMMETRY Omega_11 == Omega_22 FOR SYMMETRIC ELL")
print("=" * 70)

# With ell_left = ell_right, the surface has a left-right Z_2 symmetry.
# Under this symmetry, the period matrix should satisfy Omega_11 = Omega_22
# IF the homology basis is chosen symmetrically.
#
# Test: compare Omega_11 and Omega_22 for a symmetric ell list.

x_sym = 5
ell_sym = ell_torus + ell_torus + [x_sym]*3
print(f"Symmetric test: ell={ell_sym}")
print()

forms_sym = et.make_cyl_eqn_improved_higher_genus(rg2, ell_sym)

# Default basis
Omega_def = et.period_matrix(
    forms=forms_sym, ribbon_graph=rg2, ell_list=ell_sym,
    custom_cycles=basis_default)
print("Default basis result:")
print(f"  Omega = {Omega_def}")
print(f"  |Omega_11 - Omega_22| = {abs(Omega_def[0,0] - Omega_def[1,1]):.6f}")
print(f"  Im(Omega_11) > 0: {Omega_def[0,0].imag > 0}")
print(f"  Im(Omega_22) > 0: {Omega_def[1,1].imag > 0}")
eigenvals = np.linalg.eigvalsh(Omega_def.imag)
print(f"  Eigenvalues of Im(Omega): {eigenvals}")
print(f"  Im(Omega) positive definite: {np.all(eigenvals > 0)}")
print()

if all_block_bases:
    Omega_blk = et.period_matrix(
        forms=forms_sym, ribbon_graph=rg2, ell_list=ell_sym,
        custom_cycles=all_block_bases[0])
    print("Block-diagonal basis result:")
    print(f"  Omega = {Omega_blk}")
    print(f"  |Omega_11 - Omega_22| = {abs(Omega_blk[0,0] - Omega_blk[1,1]):.6f}")
    eigenvals_blk = np.linalg.eigvalsh(Omega_blk.imag)
    print(f"  Eigenvalues of Im(Omega): {eigenvals_blk}")
    print(f"  Im(Omega) positive definite: {np.all(eigenvals_blk > 0)}")
    print()

print("=" * 70)
print("SECTION 7: WHAT TAU SHOULD THE DIAGONAL CONVERGE TO?")
print("=" * 70)

# The component torus from the prism degeneration is NOT the theta graph.
# It has a different boundary word structure.
#
# Strategy: read off the boundary sub-word for left-block edges only,
# and use that to infer the effective F=1 theta-graph description if possible.
#
# For the left triangle (edges 0,1,2) with the specific rotation from rgs[1]:
# When neck -> infinity, the "effective" boundary word restricted to the
# left triangle is determined by how the left edges appear in the full word.
#
# Key: in the boundary word, the left triangle edges appear at certain positions.
# Between two consecutive appearances of left-triangle edges (with no neck
# edges in between), we get the "effective" left-torus sub-word.

print("Full boundary word of rgs[1] with block labels:")
print(" ".join(f"{block_labels[e]}{e}" for e in bword2))
print()

# Extract the boundary word restricted to left+right blocks (ignoring neck)
# This shows the relative ordering of left and right edges
non_neck = [e for e in bword2 if e not in neck_block]
print("Boundary word with neck edges removed:", non_neck)
print()

# The left-sub-word (only left edges) in order of appearance:
left_subword = [e for e in bword2 if e in left_block]
right_subword = [e for e in bword2 if e in right_block]
print("Left sub-word (positions within full boundary word):", left_subword)
print("Right sub-word:", right_subword)
print()

# Check: do left and right sub-words match the theta graph pattern (e0,e1,e2,e0,e1,e2)?
# The theta graph boundary is (edge_a, edge_b, edge_c, edge_a, edge_b, edge_c).
print("Is left sub-word a theta-graph pattern?")
if len(left_subword) == 6:
    first_half = left_subword[:3]
    second_half = left_subword[3:]
    is_theta = (sorted(first_half) == list(range(0,3))) and (first_half == second_half)
    print(f"  first half: {first_half}, second half: {second_half}")
    print(f"  is theta pattern: {is_theta}")
    if is_theta:
        # The left sub-word IS a theta graph pattern
        # The effective theta graph for the component torus uses edges 0,1,2
        # with lengths ell_torus[0], ell_torus[1], ell_torus[2]
        # and the edge labeling from the sub-word order
        sub_e0, sub_e1, sub_e2 = first_half
        print(f"  Effective theta: edges {sub_e0},{sub_e1},{sub_e2} with lengths "
              f"{ell_torus[sub_e0]}, {ell_torus[sub_e1]}, {ell_torus[sub_e2]}")
        L_eff = 2 * sum(ell_torus)
        P_eff = et.periods_improved(L_eff, ell_torus[sub_e0], ell_torus[sub_e1])
        tau_eff = P_eff[1] / P_eff[0]
        print(f"  tau_eff (component torus) = {tau_eff:.6f}")
else:
    print(f"  Left sub-word has {len(left_subword)} entries, expected 6.")
print()

print("Is right sub-word a theta-graph pattern?")
if len(right_subword) == 6:
    first_half_r = right_subword[:3]
    second_half_r = right_subword[3:]
    is_theta_r = (first_half_r == second_half_r)
    print(f"  first half: {first_half_r}, second half: {second_half_r}")
    print(f"  is theta pattern: {is_theta_r}")
    if is_theta_r:
        sub_e3, sub_e4, sub_e5 = first_half_r
        rel_idx = [sub_e3 - 3, sub_e4 - 3, sub_e5 - 3]
        print(f"  Effective theta: edges {sub_e3},{sub_e4},{sub_e5} with lengths "
              f"{ell_torus[rel_idx[0]]}, {ell_torus[rel_idx[1]]}, {ell_torus[rel_idx[2]]}")
print()

print("=" * 70)
print("DONE")
print("=" * 70)
