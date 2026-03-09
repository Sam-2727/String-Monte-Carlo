# Light-Cone String Theory: Discretized Mandelstam Diagram Amplitudes

## Overview

This package implements a computational framework for calculating string theory amplitudes at arbitrary genus using discretized Mandelstam diagrams in light-cone gauge. The core idea is:

1. The circular cross-sections of cylinders in Mandelstam diagrams are discretized into L equally spaced points.
2. The string field wavefunctional (as defined on the disk, cf. Polchinski Ch 2.8) is approximated by its values at these L boundary points.
3. Amplitudes are computed by composing elementary building blocks — vertex identifications and propagator quadratic forms — and performing finite-dimensional Gaussian integrals.

The package is written in Mathematica (.wl/.m) and should support both symbolic and numerical evaluation, with a toggle to keep everything numerical from the start for performance.

---

## Physical Setup

### Light-Cone Gauge and Discretization

In light-cone gauge, each string carries a longitudinal momentum p⁺ > 0, and the spatial extent of the string is proportional to p⁺.

**Discretization scale convention.** L denotes the total circumference (sum of L_i) at the leftmost external vertex slice, and L₁ denotes the smallest L_i among the leftmost external vertices. We normalize p₁⁺ = 1. Then the discretized circumferences are:

- L_i = p_i⁺ · L₁ must be a positive integer for every external string i.
- L_e = k_e⁺ · L₁ must be a positive integer for every internal propagator edge e.

Once L (equivalently L₁) is chosen, this fixes which ratios of external p_i⁺ are allowed (they must be rational with denominator dividing L₁), and similarly for internal momenta k_e⁺.

The boundary of each disk is discretized into L_i equally spaced points:

- z_k = e^{2πi k / L_i}, for k = 1, ..., L_i

The field X(z) on the boundary has L_i independent degrees of freedom: either the L_i position-space values X(z_k), or equivalently L_i discrete Fourier modes X_n. These are related by the standard discrete Fourier transform.

### Wavefunctional on the Disk

For the free boson, the vacuum wavefunctional in mode space is:

Ψ₀[X] ∝ exp(−∑_{n=1}^{L/2} (ω(n) / 2α') |X_n|²)

where ω(n) is the lattice dispersion relation:

ω(n) = (L/π) sin(π n / L)

This is the eigenvalue of the nearest-neighbor finite-difference operator √(−Δ_lattice) on an L-point periodic lattice. It has the correct continuum limit ω(n) → n as L → ∞ (for n ≪ L), preserves conjugate symmetry ω(n) = ω(L−n), and does not introduce a spurious zero at the Nyquist frequency (ω(L/2) = L/π ≠ 0).

In position space, this becomes a quadratic form:

Ψ₀[X] ∝ exp(−(1/2) X^T · M · X)

where M is an L×L circulant matrix obtained by inverse-DFT of the diagonal entries (ω(n)/α').

**Zero mode (n=0):** The center-of-mass mode x₀ = (1/√L)∑ X(z_k) is not damped by the wavefunctional — it is flat (delta-function normalizable).

**Zero mode gauge fixing convention:** For each connected diagram, the total quadratic form M (after assembly and sewing) has exactly one zero eigenvalue with eigenvector u = (1, 1, …, 1)/√N — the uniform translation mode. We fix this gauge freedom by **projecting to the subspace ∑_k X_k = 0**, i.e., setting the n = 0 Fourier mode to zero. This is the most symmetric choice: it removes the translational degree of freedom without singling out any boundary point.

Concretely:
- M already satisfies M u = 0 (the zero mode is in its null space), so no modification of M is needed.
- The source vector J must satisfy ∑_k J_k = 0 (equivalently, J ⊥ u). This is the **momentum conservation condition**: if external momenta sum to zero, the source has no zero-mode component. We project J → J − (J · u) u.
- The Gaussian integral is performed over the (N−1)-dimensional subspace via eigendecomposition: det′(M) = ∏ (nonzero eigenvalues), M⁺ = pseudoinverse.
- The resulting gauge-fixed amplitude is missing the overall factor δ(∑ pᵢ), which is understood.

### Vertex Operator Wavefunctionals

For an external state |k⟩ = e^{ik·X(0)} |0⟩ (vertex operator inserted at the origin of the disk), the wavefunctional picks up a linear term in the exponent:

Ψ_k[X] ∝ exp(−(1/2) X^T · M · X + J^T · X + c)

where J encodes the momentum insertion and c is a constant. The general form of any Gaussian wavefunctional is therefore a triple (M, J, c): a quadratic form matrix, a linear source vector, and a scalar.

### Transverse Directions

Each transverse direction μ = 1, ..., d (where d = D−2 = 24 for the bosonic string) contributes independently to the amplitude. The quadratic form M is the **same** for all μ — it depends only on the geometry (circumferences, propagator lengths, twists). Only the source vector J^μ differs per direction (encoding the transverse momenta p^μ of external states).

For vacuum external states (J = 0), the contribution of d transverse directions is:

log Z_total = d × [(N−1)/2 log(2π) − (1/2) log det'(M)] + c_total

The determinant contribution is simply raised to the d-th power. For momentum eigenstates, each μ contributes its own J^μ · M⁺ · J^μ term. The implementation uses the global variable `$LCTransverseDimension` (default 1 for backward compatibility).

---

## Architecture: Computational Graph

### Mandelstam Diagram vs. Computational Graph

The **Mandelstam diagram** is the physical/geometric object — a flat Riemann surface with cylindrical ends, interaction points, specific moduli (lengths, twists), and a light-cone time function x⁺ flowing left to right. It lives in the physics.

The **computational graph** is a data structure that extracts from the Mandelstam diagram exactly the information our algorithm requires, in a way that is explicit and efficient. The relationship is: every Mandelstam diagram determines a unique computational graph, and the computational graph plus moduli (T_e, m_e) plus external data is sufficient to compute the amplitude. The computational graph discards the continuous geometry of the surface and retains only the combinatorial and discrete data that feeds into the assembly of the global quadratic form.

### Graph Structure

The computational graph is a directed graph (directed left-to-right, consistent with light-cone time flow) with:

- **Nodes** = interaction vertices. Every node has a circular boundary with L_v boundary points.
- **Edges** = propagators. Each edge is directed (from earlier to later in light-cone time). This direction determines which end of the propagator is "in" and which is "out," which is needed to define the sign of the twist unambiguously.

**Self-loops are excluded.** Since every edge flows strictly forward in light-cone time, an edge cannot start and end at the same node.

The cyclic ordering of incident edges around each node matters — it determines which contiguous arc of the boundary circle maps to which propagator. Two different cyclic orderings can give different Mandelstam diagram topologies.

For the current implementation, **all vertices are cubic** (trivalent): each interaction node has exactly three incident edges. General n-to-m vertices are deferred but should not be precluded by the architecture.

#### Node Data

Each node v has:
- A unique label.
- **Boundary size L_v**: the number of discretized boundary points.
- **Arc partition**: the circular boundary is partitioned into contiguous arcs, one per incident edge. Each arc specifies which contiguous subset of the L_v boundary points connects to a given edge. The cyclic ordering of arcs around the boundary is stored explicitly.
- **External flag** (boolean): whether this node has an external state attached. External nodes are ordinary interaction vertices that happen to have one arc designated as "external" rather than connected to a propagator edge. See "External States" below.

For a cubic interaction vertex where edges with circumferences L_{e1}, L_{e2}, L_{e3} meet, with one edge on one side (say e1) and two on the other (e2, e3):
- L_v = L_{e1} = L_{e2} + L_{e3}
- The arc partition has the L_{e2} contiguous boundary points mapping to e2 and the L_{e3} contiguous boundary points mapping to e3 (and all L_v points mapping to e1).

#### Edge Data

Each directed edge e from node u to node v has:
- A unique label.
- **Source and target nodes**: u (earlier in light-cone time) and v (later).
- **Circumference L_e**: number of discretized boundary points on each end.
- **Length T_e** (continuous, not discretized): Euclidean light-cone time elapsed. Controls suppression of high modes.
- **Twist m_e** (integer, naturally discretized): twist offset in units of 2π/L_e. In position space, this is a cyclic permutation of boundary point labels. The sign convention for the twist is fixed by the edge direction.

### External States

External states are **not** separate nodes. Instead, certain interaction nodes carry an **external flag**, indicating that one of their arcs is an external boundary rather than a connection to a propagator.

- A **leftmost external node** (an "incoming" external state) has an external arc corresponding to its "in" variables. The external wavefunctional will be sewn onto these specific variables.
- A **rightmost external node** (an "outgoing" external state) has an external arc corresponding to its "out" variables. The external wavefunctional will be sewn onto these specific variables.

This avoids introducing separate external nodes connected by zero-length propagators, which would double the variable count at external boundaries and require handling the numerically degenerate limit T → 0 of the propagator (where coth(0) and 1/sinh(0) diverge).

Each external node stores:
- Which arc is external.
- Whether it is "in" (leftmost) or "out" (rightmost).
- The vertex operator data needed to construct the wavefunctional (M_ext, J_ext, c_ext).

---

## Two Types of Contributions (Fundamental Distinction)

There are exactly **two types of contributions** to the amplitude, and they are fundamentally different:

### 1. Identifications (at Nodes)

Vertices are **purely combinatorial** — they contribute no quadratic form. They are delta-function constraints that **identify** (equate) boundary degrees of freedom. The arc partition at a node specifies how the L_v boundary points are shared among incident edges.

**Implementation:** Identifications are encoded in the index bookkeeping — which entries of the global matrix correspond to which arc of which node. No matrix entries are generated by vertices themselves.

(An interaction-point operator insertion at the junction may eventually be needed, related to the conformal anomaly / ρ-mapping prefactor, but this is deferred for now and would be a local factor at the junction point, not a quadratic form on all L variables.)

### 2. Quadratic Forms (from Propagators and External Wavefunctionals)

**Propagators** and **external wavefunctionals** are the only sources of nontrivial matrix entries:

- A **propagator** on edge e contributes a 2L_e × 2L_e block to the global quadratic form, coupling the arc variables at its two endpoint nodes.
- An **external wavefunctional** at a node v contributes an L_ext × L_ext block (plus linear and constant terms) to the global quadratic form, where L_ext is the size of the external arc.

---

## Propagator: Detailed Form

### Continuum (Mode Space)

For each Fourier mode n with lattice dispersion ω(n) = (L/π) sin(πn/L), the propagator kernel coupling X_n^{in} and X_n^{out} is:

K_n(X_n^{out}, X_n^{in}; T, θ) ∝ exp[−(ω/2) coth(ωT) (|X_n^{out}|² + |X_n^{in}|²) + (ω/sinh(ωT)) Re(X_n^{out*} · e^{inθ} · X_n^{in})]

where ω = ω(n). This ensures the propagator is consistent with the wavefunctional (both use the same dispersion relation), so the T→0 limit correctly reproduces the identity kernel on the lattice.

Key behavior:
- Large T: coth(ωT) → 1, 1/sinh(ωT) → 2e^{−ωT}. High modes are exponentially suppressed — the propagator projects onto low-lying states.
- Zero mode (n=0, ω=0): free-particle (diffusion) kernel ∝ exp(−(x₀^{out} − x₀^{in})² / 2T). Requires separate treatment.

### Discretized (Position Space)

With L boundary points per end, the propagator is a 2L × 2L matrix acting on the vector (X_k^{in}, X_k^{out}), k = 1, ..., L. It has the block structure:

```
    ⎛  A      B·Pₘ  ⎞
Q = ⎜                ⎟
    ⎝ (B·Pₘ)ᵀ   A   ⎠
```

where:
- **A** is the L×L circulant matrix built from the diagonal entries (ω(n)/2)coth(ω(n)T) via inverse DFT over modes n = 0, 1, ..., L−1, with ω(n) = (L/π)sin(πn_phys/L) the lattice dispersion.
- **B** is the L×L circulant matrix built from −ω(n)/(2 sinh(ω(n)T)) via inverse DFT. (The sign: the off-diagonal blocks have the opposite sign to the diagonal in the exponent −(1/2) x^T Q x.)
- **Pₘ** is the L×L cyclic permutation matrix implementing shift by m (encoding the twist θ = 2πm/L).

Since A and B are circulants, they are diagonalized by the DFT matrix, so construction and manipulation are efficient: O(L log L) via FFT.

**Length T remains continuous** — there is no discretization of it. It simply parameterizes the matrix entries.

**Twist θ is naturally discretized** to multiples of 2π/L, i.e., θ = 2πm/L for integer m. In position space, this is just a cyclic offset in the identification of boundary points — a permutation of indices, computationally free.

---

## Computation Pipeline

### Step 1: Pre-Computation — Variable Count and Index Map

Once the computational graph (topology, all L_e, all arc partitions) is specified:

1. **Total variable count** is determined: N_total = ∑_v L_v, summed over all nodes.
2. **Global index map** is constructed: each boundary point at each node is assigned a unique index in {1, ..., N_total}. The arc partitions at each node specify which global indices correspond to which edge's "in" or "out" boundary points.
3. **Global quadratic form** is pre-allocated: an N_total × N_total matrix M, an N_total-vector J, and a scalar c. In numerical mode, these are initialized as sparse arrays / packed arrays for performance.

### Step 2: Assembly

For each edge (propagator) e from node u to node v:
1. Construct the 2L_e × 2L_e propagator matrix Q_e(T_e, m_e).
2. Using the global index map, identify which L_e global indices at node u correspond to the "in" end of edge e, and which L_e global indices at node v correspond to the "out" end.
3. Add Q_e into the appropriate entries of the global matrix M.

(External wavefunctionals are **not** added during assembly. They are applied in a separate final step.)

### Step 3: Schur Complement over Internal Variables

After assembly, partition the N_total variables into:
- **External variables**: the boundary points at external arcs of external nodes. These are the "in" variables at leftmost external nodes and "out" variables at rightmost external nodes.
- **Internal variables**: all other boundary points.

Integrate out internal variables via **Schur complement**:

If the quadratic form on (x_ext, x_int) has block structure:

```
    ⎛ M_ee   M_ei ⎞       ⎛ J_e ⎞
M = ⎜              ⎟,  J = ⎜     ⎟
    ⎝ M_ie   M_ii ⎠       ⎝ J_i ⎠
```

then the result of integrating out x_int is:

- M_eff = M_ee − M_ei · M_ii⁻¹ · M_ie
- J_eff = J_e − M_ei · M_ii⁻¹ · J_i
- c_eff = c + (1/2) J_i^T · M_ii⁻¹ · J_i − (1/2) log det M_ii

The result is a quadratic form (M_eff, J_eff, c_eff) on the external variables only, as a function of the moduli {T_e, m_e}.

### Step 4: Sewing External Wavefunctionals

For each external node v with wavefunctional (M_ext, J_ext, c_ext):

1. Add (M_ext, J_ext, c_ext) to the effective quadratic form on the external variables of node v (using the appropriate "in" or "out" variable set, depending on whether the node is leftmost or rightmost).

### Step 5: Zero Mode Projection

The sewn quadratic form has exactly one zero eigenvalue (the overall translational mode) with eigenvector u = (1, …, 1)/√N. Project to the gauge-fixed subspace ∑ X_k = 0:

- Project J: J → J − (J · u) u. (If momentum is conserved, J · u = 0 already.)
- M is unchanged (it already annihilates u).

### Step 6: Gaussian Integral

Perform the Gaussian integral over the (N−1)-dimensional gauge-fixed subspace using eigendecomposition of M:

- Compute eigenvalues {λ_i} and eigenvectors {v_i} of M.
- Exclude the zero eigenvalue (the projected-out translation mode).
- log Z = ((N−1)/2) log(2π) − (1/2) ∑ log λ_i + (1/2) J^T M⁺ J + c

where the sums run over nonzero eigenvalues only, and M⁺ = ∑ (1/λ_i) |v_i⟩⟨v_i| is the pseudoinverse.

The result is the **gauge-fixed amplitude** (missing the overall δ(∑ pᵢ) factor) as a function of the moduli {T_e, m_e} and external momenta.

---

## Performance Considerations

- **Numerical mode**: Use `SparseArray` for the global matrix M during assembly (it is block-sparse). For the Schur complement / determinant, convert to dense or use iterative methods as appropriate.
- **Packed arrays**: In numerical mode, all dense sub-blocks (propagator matrices, circulants) should be packed arrays.
- **Circulant construction**: Build propagator blocks in Fourier space (diagonal), then inverse-DFT to position space. This is O(L log L) per propagator.
- **Determinant**: For the final log det M_ii, exploit sparsity or block structure. For moderate N_total, dense Cholesky is fine. For large N_total, block elimination following the graph structure may be advantageous.
- **Symbolic/Numerical toggle**: The package should support a mode flag. Symbolic mode: all matrices and operations are exact. Numerical mode: all matrices are machine-precision floats from the start, with moduli and momenta as fixed numerical values. Essential for performance at large L and high genus.

---

## Diagram Specification Format

### Graph-Based Specification

```mathematica
(* --- Discretization parameters --- *)
(* L1 = smallest external circumference (sets the scale) *)
(* External momenta: p_i^+ = L_i / L1, normalized so min is 1 *)
(* Internal momenta: k_e^+ = L_e / L1 *)

(* --- Nodes --- *)
(* Each node has a unique label, an arc partition, and optional external flag *)
(* Arc partition: ordered list of {edgeLabel, L_e} going cyclically around the boundary *)
(* For external nodes, one arc is {"ext", L_ext, "in"/"out", vertexOpData} *)

LCNode["split",
    arcPartition -> {
        {"ext", 5, "in", V1},     (* external arc, 5 points, incoming *)
        {"e2", 3},                 (* internal arc to edge e2, 3 points *)
        {"e3", 2}                  (* internal arc to edge e3, 2 points *)
    }
]

LCNode["join",
    arcPartition -> {
        {"e2", 3},                 (* internal arc from edge e2 *)
        {"e3", 2},                 (* internal arc from edge e3 *)
        {"ext", 5, "out", V1star}  (* external arc, 5 points, outgoing *)
    }
]

(* --- Edges (directed, left to right in light-cone time) --- *)
LCEdge["e2", "split" -> "join", circumference -> 3, length -> T2, twist -> m2]
LCEdge["e3", "split" -> "join", circumference -> 3, length -> T3, twist -> m3]

(* --- Diagram --- *)
diagram = LCDiagram[
    nodes -> {split, join},
    edges -> {e2, e3},
    discretizationScale -> L1
]
```

### Example: Genus 1 Two-Point Function

```mathematica
L1 = 5; L2 = 3; L3 = 2;  (* L1 = L2 + L3, so p2/p1 = 3/5, p3/p1 = 2/5 *)

diagram = LCDiagram[
    nodes -> {
        LCNode["split", arcPartition -> {
            {"ext", L1, "in", V1},
            {"e2", L2},
            {"e3", L3}
        }],
        LCNode["join", arcPartition -> {
            {"e2", L2},
            {"e3", L3},
            {"ext", L1, "out", V1star}
        }]
    },
    edges -> {
        LCEdge["e2", "split" -> "join", circumference -> L2, length -> T2, twist -> m2],
        LCEdge["e3", "split" -> "join", circumference -> L3, length -> T3, twist -> m3]
    },
    discretizationScale -> L1
]

amplitude = EvaluateDiagram[diagram]
(* Returns function of T2, T3, m2, m3, and external momenta *)
```

**Variable count:** N_total = L_split + L_join = 5 + 5 = 10.
After Schur complement over internal variables (all 10 minus the 5+5 external), the effective quadratic form lives on the 5 incoming + 5 outgoing external variables.

Actually: note that here every variable at each node participates in at least one propagator or external arc, so the split between "internal" and "external" is: the 5 external-arc points at "split" (incoming) and the 5 external-arc points at "join" (outgoing) are external; the same points also participate in propagator couplings via the internal arcs, since a single boundary point at "split" belongs to both the external arc (as part of the L1=5 boundary) and one of the internal arcs (e2 or e3). The identification at the vertex means these are the same variable — the propagator quadratic form acts on the same variables that the external wavefunctional will later be sewn onto. The "internal" integration in the Schur complement is over any variables that are purely internal (not part of any external arc); in this example, there are none — all variables are external. The Schur complement is trivial, and the full quadratic form from the two propagators directly gives M_eff on the 10 external variables, to which the wavefunctionals are then applied.

---

## Continuum Limit and Validation

The discretized 3-point vertex (at tree level, no propagators) should reproduce the known **Neumann matrices** N^{rs}_{mn} as L → ∞. At finite L, the discrete Neumann matrices serve as approximations, and the rate of convergence is a key diagnostic.

Additional checks:
- The circulant matrices (boundary Green's function) should approximate −α' ln|2 sin(π(j−k)/L)| and their condition number grows with L — monitor this.
- The interaction-point prefactor (from the conformal anomaly / ρ-mapping ρ = ∑ αᵣ ln(z − zᵣ)) is regulated by the discretization. Whether the correct continuum prefactor emerges as L → ∞ or must be inserted by hand needs to be checked.
- For external states with momentum k, the Gaussian integral at sewing has both quadratic and linear pieces; the result should be det × exp(rational quadratic in kᵢ), reproducing Koba-Nielsen-like kinematic factors.

---

## Summary of Key Design Decisions

1. **Computational graph representation**, not time-slices. The computational graph is a directed graph (edges flow left-to-right in light-cone time) that is a minimal, faithful encoding of the Mandelstam diagram for our algorithm. Nodes = interaction vertices, edges = propagators. Self-loops are excluded by the time-ordering.
2. **Two types of contributions**: Identifications at nodes (combinatorial, no matrix) vs. quadratic forms from propagators and external wavefunctionals. This is the fundamental architectural distinction.
3. **External states are flags on nodes**, not separate nodes. An external node is an ordinary interaction vertex with one arc designated as external ("in" or "out"). External wavefunctionals are sewn on in a final step, using the specific "in" or "out" variables at that node. This avoids doubling variable count and the numerical degeneracy of zero-length propagators.
4. **Pre-allocated global matrix**: Total variable count N_total = ∑_v L_v is determined upfront from the graph. The global quadratic form (M, J, c) is allocated once with fixed size, then assembled from propagator blocks. Use sparse arrays / packed arrays for performance.
5. **Discretization scale**: L₁ (smallest external circumference) sets the scale. All circumferences L_i = p_i⁺ · L₁ and L_e = k_e⁺ · L₁ must be positive integers. Scaling with L₁ is the primary convergence parameter.
6. **Twist is discrete**: θ = 2πm/L, implemented as a cyclic permutation of boundary-point labels. Computationally free.
7. **Length is continuous**: T parameterizes matrix entries via coth(nT) and 1/sinh(nT). No discretization.
8. **Cubic vertices only** for the current implementation. General n-to-m vertices are deferred but not precluded by the architecture.
9. **Cyclic ordering** of arcs around each node is stored explicitly and matters for the topology.
10. **Symbolic/Numerical toggle**: Essential for scaling to high genus. Numerical mode uses machine-precision floats throughout.
11. **Computation pipeline**: (a) Build global index map, (b) assemble propagator blocks into global matrix, (c) Schur complement over internal variables, (d) sew external wavefunctionals onto external variables.
