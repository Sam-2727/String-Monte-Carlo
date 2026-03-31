# Comprehensive Review of `lightcone_discrete_sigma_note`

## 1. Overview and Scope

The note develops a discrete-sigma formulation of lightcone Mandelstam string amplitudes in which the worldsheet spatial direction is discretized while Euclidean lightcone time remains continuous. The goal is to reduce higher-loop string amplitudes to a controlled numerical problem: exact finite-dimensional Gaussian propagators on each cylinder, exact site-space boundary sewing at cubic joins, and a local Green-Schwarz interaction-point prefactor fixed by dynamical supercharges. The final amplitude is a finite-dimensional integral over Schwinger lengths and twist moduli, with all internal field integrations done analytically.

---

## 2. Section-by-Section Scrutiny

### 2.1 Sections 1-4: Goal, Literature, Mandelstam Map, Lattice Setup

**Correct and well-motivated.** The Mandelstam map conventions (Sec 4) are standard. The sign convention $\varepsilon_r^{\rm M} = +1$ incoming, $-1$ outgoing, and the positive circumference $\alpha_r = |\alpha_r^{\rm M}|$ are cleanly separated. The refinement-sequence idea for approximating generic real $\alpha_r$ by rationals with exact integer overlap $N_3 = N_1 + N_2$ is sound in principle.

**Issue 1 (minor):** The note does not discuss convergence *rates* of the refinement sequence $a_\nu \to 0$. If external momenta are irrational multiples of each other, the Diophantine quality of the rational approximation could affect the rate at which finite-$N$ artifacts vanish. This is probably a subleading concern but deserves a sentence.

### 2.2 Section 5: Discrete Free Boson on a Cylinder

The lattice action, DFT diagonalization, and mode frequencies $\omega_k = (2/a)\sin(\pi k/N)$ are textbook-correct for the periodic chain. The kernel formulas for $k \neq 0$ (Euclidean harmonic oscillator) and $k=0$ (free particle) are standard.

**Verified:** The smooth limits $\omega_0 \coth(\omega_0 T) \to 1/T$ and $\omega_0 \operatorname{csch}(\omega_0 T) \to 1/T$ are correct.

**Issue 2 (important):** The note writes the mass parameter as $\mu = a/(2\pi\alpha')$, which is the lattice-spacing-dependent prefactor arising from discretizing the worldsheet action. This is correct for the action given, but I want to flag that the overall normalization convention for the site-basis kernel (Sec 7) depends on this consistently. In `tachyon_check.py`, the mode metric is implemented as $\mu_r \omega_k = \sin(\pi k / N_r)/(\pi\alpha')$, which is independent of $a$ because $\mu \cdot \omega_k = [a/(2\pi\alpha')] \cdot [(2/a)\sin(\pi k/N)] = \sin(\pi k/N)/(\pi\alpha')$. This cancellation is crucial and correct: the discrete Gaussian data entering the Schur complement and the three-tachyon amplitude are $a$-independent, as they must be since $a$ is a redundant parameter for common lattice spacing.

### 2.3 Section 6: Discrete Free GS Fermions

The coherent-state kernel $K_{F,k} = \exp(\bar\eta_f e^{-\omega_k T} \eta_i)$ is the standard first-order Grassmann propagator. The zero-mode kernel being $T$-independent is correct.

**Issue 3 (convention gap):** The note explicitly defers the overall coherent-state normalization. This is acceptable for the structural development, but it means the absolute normalization of the fermionic Pfaffian relative to the bosonic determinant is not yet pinned down. For the eventual numerical loop computation, this must be fixed.

### 2.4 Section 7: Site-Basis Gaussian Matrices

The formulas for $A(T)$ and $B(T,\varphi)$ as circulant/twisted-circulant matrices diagonalized by the DFT are correct. The twist operator $R_N(\varphi)$ and the centered-frequency convention $\kappa_k$ are standard.

**Issue 4 (subtle):** For even $N$ and generic $\varphi \notin \mathbb{Z}/N$, the note correctly warns that $R_N(\varphi)$ is a complex spectral interpolation rather than a real site permutation. This means the site-basis Gaussian matrices $B(T,\varphi)$ are complex for generic twist. The Gaussian integration formulas in Sec 8 then require the complex-symmetric extension. The note does not explicitly address whether $\mathcal{M}_B$ remains positive-definite in the complex-symmetric sense for generic $\varphi$. This should be checked numerically at the implementation stage.

### 2.5 Section 8: Sewing Formula for a Fixed Diagram

The global sewing formula is correctly organized. The logical separation of edge kernels (free propagation) from vertex factors (local joining) is clean. The source-derivative mechanism for generating interaction-point insertions is standard generating-functional technology.

**Verified:** The counting $E = 3h + n - 3$, $V = 2h + n - 2$ for a genus-$h$, $n$-point cubic lightcone skeleton is correct (Euler characteristic constraint).

**Issue 5 (important):** The note introduces the lightcone measure factor $J_{\rm M}(T,\varphi;\alpha_{\rm ext})$ and explicitly states it is "not determined by the finite-dimensional discrete sewing." This is a genuine gap. For tree-level comparisons with known Koba-Nielsen amplitudes, one needs this factor. The note's option of setting $J_{\rm M} = 1$ is a convention choice that must be reconciled with any separate Mandelstam-measure analysis. This should be clearly flagged as a prerequisite for any quantitative comparison beyond the three-tachyon kinematic structure.

### 2.6 Section 9: Discrete Cubic Overlap Vertex

The full-boundary overlap $X^{(3)} = P_1 X^{(1)} + P_2 X^{(2)}$ in position space is the correct discrete Mandelstam joining condition. The embedding matrices $P_1, P_2$ and the constraint matrix $C_I = (P_1 \ P_2 \ -\mathbb{I}_{N_3})$ are correctly defined.

**Verified numerically:** The completeness relation $P_1 P_1^T + P_2 P_2^T = \mathbb{I}_{N_3}$ and the isometry property $P_r^T P_r = \mathbb{I}_{N_r}$ are exactly satisfied by construction (they follow trivially from the block-identity structure of $P_1, P_2$).

The distinction between the local interaction-point data $(X_{I_+}, X_{I_-})$ and the center-of-mass zero modes is important and correctly emphasized.

### 2.7 Section 10: Position/Fourier/Oscillator Bases

The Fourier decomposition, the affine overlap relation $q^{(3)} = U_1 q^{(1)} + U_2 q^{(2)} + \xi y$, and the explicit matrix elements of $U_1, U_2, \xi$ are all correct.

**Verified numerically (from lorentz_check.py and tachyon_check.py):** The completeness relation $U_1 U_1^\dagger + U_2 U_2^\dagger + \hat\xi \hat\xi^\dagger = \mathbb{I}_{N_3-1}$ holds to machine precision across all tested lattice sizes. The isometry properties $U_r^\dagger U_r = \mathbb{I}_{N_r-1}$ also hold to machine precision. These are non-trivial finite-dimensional identities that confirm the overlap algebra.

**Issue 6 (notation):** The note uses the same symbol $I$ for both the transverse SO(8) vector index and the identity matrix (and occasionally the vertex label). While each usage is locally disambiguated, this could cause confusion in a longer calculation. The note is aware of this and warns the reader, but it remains a source of potential error.

### 2.8 Section 11: Bosonic Cubic Vertex in Oscillator Basis

The squeezed-state form is correctly derived from the linear overlap constraints. The matrices $\mathbb{M}$ and $\mathbb{N}$ involving $C_Q$, $C_P$, and $(\boldsymbol{\mu}\Omega)^{\pm 1/2}$ are correctly assembled.

**Key derivation check:** The constraint system $(\mathbb{M} \mathbf{a} + \mathbb{N} \mathcal{R} \mathbf{a}^\dagger + J_0(y))|V_3^{(B)}\rangle = 0$ is the standard annihilation-condition approach. The appearance of $\mathcal{R}$ (the reality-involution permutation $m \to N_r - m$) in front of $\mathbf{a}^\dagger$ is correct and crucial: in the full complexified basis, the creation partner of $a_m$ is $a_{N-m}^\dagger$, not $a_m^\dagger$.

**Issue 7 (important, not yet resolved in the note):** The note states that $S^{(B)} = -\mathbb{M}_{\rm phys}^{-1} \mathbb{N}_{\rm phys}$ must be symmetric, $S^{(B)T} = S^{(B)}$. This is a necessary consistency condition for the squeezed-state ansatz, and it should be verified numerically at each $N$. Neither the note nor the existing scripts appear to perform this check explicitly. **Recommendation: add a numerical symmetry check of $S^{(B)}$ to the diagnostic suite.**

### 2.9 Section 12: Bosonic Lorentz Check

The bond-locality calculation is clean and correct.

**Verified numerically (lorentz_check.py, test_bond_locality):** The identity $\Delta_I^{\rm bond} = 2(X_{I_+} - X_{I_-})(X_{N_2-1}^{(2)} - X_{N_1-1}^{(1)})$ is confirmed to machine precision on random configurations.

The renormalized zero-point energy mismatch converges to the continuum value $(\pi\alpha'/6)(1/\alpha_1^2 + 1/\alpha_2^2 - 1/\alpha_3^2)$ as $N \to \infty$, with $O(1/N^2)$ convergence rate. This is confirmed by `lorentz_check.py`.

**Issue 8 (central open problem):** The operator-level Lorentz computation $[J_2^{-I}, P_3^-] + [J_3^{-I}, P_2^-] = 0$ has NOT been carried out. The note correctly identifies this as the primary check that should determine the overall cubic normalization and verify $D=26$. The three-tachyon amplitude is only a "preliminary diagnostic." This is the single most important calculation that remains undone on the bosonic side.

### 2.10 Section 12.1: Three-Tachyon Check

The Schur complement formula $\gamma_T = C_T - B_T^T G_T^{-1} B_T$ and the resulting Gaussian integral are correctly derived. The on-shell kinematics $q_{\rm rel}^2 = 4(\alpha_1^2 + \alpha_1\alpha_2 + \alpha_2^2)/(\alpha'\alpha_3^2)$ is correct for closed-string tachyons.

**Verified numerically (tachyon_check.py):** The exact leg-factorization fit shows that $\log C_{\rm req}^{(B)}(N_1, N_2)$ decomposes as $f_{\rm in}(N_1) + f_{\rm in}(N_2) + f_{\rm out}(N_1+N_2) + {\rm const}$ to machine precision across the full tested grid ($4 \le N_1, N_2 \le 48$). This is a strong numerical confirmation that the discrete cubic vertex has no irreducible three-body normalization dependence, consistent with the continuum expectation.

**Issue 9 (derivation gap):** The note does not derive the one-string normalization factors $Z_{\rm in}(N)$, $Z_{\rm out}(N)$ analytically. The numerical factorization is convincing, but an analytic derivation from the Poincare algebra (i.e., the operator-level Lorentz check) is needed to complete the bosonic story. The power-law fit $\log C_{\rm req} \sim a \log N_3 + b \log[x(1-x)] + c$ gives exponents that should encode the critical dimension through the lattice zero-point energy, but this connection is not yet made explicit.

### 2.11 Sections 13-14: Fermionic Kinematic Overlap and GS Prefactor

The fermionic overlap parallels the bosonic one exactly, with the same $U_r$ and $\xi$ matrices. The coherent-state and Majorana bookkeeping are consistently organized.

**Issue 10 (important, partially addressed):** The interaction-point scaling $K^I \sim a^{-1/2} \Delta X^I$ from the branch-point geometry $\Delta\rho \sim a \Rightarrow |w| \sim a^{1/2}$ is correct. However, the note explicitly acknowledges that the normalization constants $c_K, \tilde{c}_K$ are not yet determined. This is directly tied to Issue 8.

**Issue 11 (critical finding from numerics):** The `superstring_prefactor_check.py` diagnostic reveals that the minimal nearest-neighbor right-arc stencil $d_- = (1,0,\ldots,0,-1)$ gives $\eta_- = 0$ to machine precision. The note provides a clean exact explanation: the stencil is parity-odd under site reversal $J_2$, while the Gaussian source response $G_T^{-1} B_T$ is parity-even. This is an elegant observation. The consequence is that the minimal stencil cannot produce the $q_{\rm rel}^I q_{\rm rel}^J$ tensor needed for the three-graviton amplitude. The one-sided second-order stencil breaks this accidental parity and restores a nonzero $\eta_-$.

The numerical data on the support-three family and the smooth dependence on $\lambda = \alpha_1/\alpha_3$ are well-documented. The scaling $N_1 B_{qq} \to O(1)$ and $\sqrt{N_1} \eta_\pm \to O(1)$ as $N \to \infty$ are consistent with the expected $a^{-1}$ and $a^{-1/2}$ scalings after reinstating the lattice spacing.

### 2.12 Section 15: Superstring Lorentz and SUSY Check

The block-triangular structure of the supercharge-closure chain in Grassmann degree is correct and is a useful structural simplification. The 5-by-4 matrix system relating even Hamiltonian coefficients $(a_0, a_2, a_4, a_6, a_8)$ to odd supercharge coefficients $(b_1, b_3, b_5, b_7)$ via the projected free-supercharge matrix elements $\lambda_\bullet$ is a clean formulation.

**Issue 12 (central open problem):** The discrete supercharge-closure calculation has NOT been carried out. The projected matrix elements $\lambda_\bullet$ have not been computed in any definite SO(8) basis. The `supercharge_closure_chain.py` script encodes only the algebraic scaffold; it does not yet contain the actual finite-$N$ free-supercharge matrix elements.

### 2.13 Section 15.1: Three-Graviton Check

The reduction of the bosonic nonzero-mode factor to explicit Gaussian data $(\Sigma_{+-}, \eta_+, \eta_-, \gamma_T)$ is correct. The tensor decomposition into $\delta^{IJ}$ and $q_{\rm rel}^I q_{\rm rel}^J$ pieces is clean.

**Issue 13 (important):** The fermionic zero-mode contraction that converts $v_{IJ}(\Lambda_{\rm lat})$ into a graviton matrix element has NOT been completed. The `gs_weyl_symbol_diagnostic.py` shows that the naive Weyl map from the Grassmann symbol to antisymmetrized Clifford products gives a $8_v \to 8_v$ block entirely in the minimal SO(8) tensor span $\{A\delta_{IJ}\delta_{KL}, B\delta_{IK}\delta_{JL}, C\delta_{IL}\delta_{JK}\}$ with residual at $10^{-13}$. This is encouraging but not yet a proof that the Weyl ordering is the correct GS dictionary.

### 2.14 Sections 16-19: Continuum Limit, Zero Modes, Short-Edge, Roadmap

**Section 16 (Continuum limit):** The warning about the logarithmic growth of oscillator-normalized reduced-block entries is important and correct. The $\hat{\mathcal{G}}^{11}_{11}$ data showing growth $\sim 2.05, 2.13, 2.21, 2.28, 2.35$ at $N_1 = 8, 16, 32, 64, 128$ indeed signals the branch-point normal-ordering divergence. The note correctly distinguishes the reduced sewn quadratic form from the full three-leg squeeze matrix.

**Verified numerically (neumann_check.py):** The completeness and Hermiticity checks pass. The DFT-basis normalized entries do grow logarithmically, consistent with the note's claim.

**Section 17 (Zero modes):** The separation of center-of-mass modes and the loop-momentum treatment are standard and correct.

**Section 18 (Short-edge):** The UV finiteness at fixed lattice spacing ($\omega_k \le 2/a$) is a genuine advantage of the scheme.

---

## 3. Detailed Derivation Checks

### 3.1 On-shell tachyon kinematics: $q_{\rm rel}^2$

The note claims $q_{\rm rel}^2 = (4/\alpha')(\alpha_1^2 + \alpha_1\alpha_2 + \alpha_2^2)/\alpha_3^2$. I rederived this from scratch.

Define $p_{\rm cm} = p_1 + p_2$, $q_{\rm rel} = (\alpha_2 p_1 - \alpha_1 p_2)/\alpha_3$. Then $p_1 = (\alpha_1/\alpha_3)p_{\rm cm} + q_{\rm rel}$ and $p_2 = (\alpha_2/\alpha_3)p_{\rm cm} - q_{\rm rel}$. Substituting into the energy conservation equation $p_1^-/\alpha_1 + p_2^-/\alpha_2 = p_3^-/\alpha_3$ with $p_r^- = \alpha'(p_r^2 - 4/\alpha')/\alpha_r$, all $p_{\rm cm}$-dependent terms cancel identically, leaving:

$$q_{\rm rel}^2 \cdot \alpha_3/(\alpha_1\alpha_2) = (4/\alpha')(\alpha_3^2 - \alpha_1\alpha_2)/(\alpha_1\alpha_2\alpha_3)$$

Since $\alpha_3^2 - \alpha_1\alpha_2 = \alpha_1^2 + \alpha_1\alpha_2 + \alpha_2^2$, this gives the claimed formula. The cancellation of $p_{\rm cm}$ is essential: it shows that $q_{\rm rel}^2$ is fully determined by the $\alpha_r$ (as expected, since three-point kinematics for massive particles admits no independent Mandelstam invariants). The code in `tachyon_check.py` implements this correctly with the $a$-independent form $4(N_1^2 + N_1 N_2 + N_2^2)/(\alpha' N_3^2)$.

### 3.2 Affine overlap derivation

The claim $q^{(3)} = U_1 q^{(1)} + U_2 q^{(2)} + \xi y$ was rederived from the position-space overlap $X^{(3)} = P_1 X^{(1)} + P_2 X^{(2)}$.

Starting from $X^{(r)} = x_{\rm cm}^{(r)} \mathbf{1}_{N_r} + \hat{F}_r^\dagger q^{(r)}$ and applying $\hat{F}_3$ to both sides of the overlap:

$$q^{(3)} = U_1 q^{(1)} + U_2 q^{(2)} + x_{\rm cm}^{(1)} \hat{F}_3 P_1 \mathbf{1}_{N_1} + x_{\rm cm}^{(2)} \hat{F}_3 P_2 \mathbf{1}_{N_2}$$

The key identity is $\hat{F}_3 P_2 \mathbf{1}_{N_2} = -\xi$. This follows because the full DFT sum $\sum_{n=0}^{N_3-1} e^{-2\pi i m n/N_3} = 0$ for $m \neq 0$, so $\hat{F}_3(\mathbf{1}_{N_1}; \mathbf{1}_{N_2}) = 0$, hence $\hat{F}_3 P_2 \mathbf{1}_{N_2} = -\hat{F}_3 P_1 \mathbf{1}_{N_1} = -\xi$. Combining: $q^{(3)} = U_1 q^{(1)} + U_2 q^{(2)} + (x_{\rm cm}^{(1)} - x_{\rm cm}^{(2)})\xi = U_1 q^{(1)} + U_2 q^{(2)} + \xi y$. Confirmed.

### 3.3 Completeness relation derivation

Starting from $U_r = \hat{F}_3 P_r \hat{F}_r^\dagger$ and using $\hat{F}_r^\dagger \hat{F}_r = I_{N_r} - (1/N_r)\mathbf{1}\mathbf{1}^T$:

$$U_1 U_1^\dagger + U_2 U_2^\dagger = \hat{F}_3(P_1 P_1^T + P_2 P_2^T)\hat{F}_3^\dagger - \frac{1}{N_1}\xi\xi^\dagger - \frac{1}{N_2}(-\xi)(-\xi)^\dagger$$

$$= I_{N_3-1} - \frac{N_3}{N_1 N_2}\xi\xi^\dagger = I_{N_3-1} - \hat\xi\hat\xi^\dagger$$

Hence $U_1 U_1^\dagger + U_2 U_2^\dagger + \hat\xi\hat\xi^\dagger = I_{N_3-1}$. The identity $P_1 P_1^T + P_2 P_2^T = I_{N_3}$ and $\hat{F}_3 \hat{F}_3^\dagger = I_{N_3-1}$ are used. Confirmed.

### 3.4 Parity argument for $\eta_- = 0$

The argument has three parts:

1. **$\Pi G_T = G_T \Pi$:** The mode metric $M_r$ has $\omega_k = \omega_{N_r-k}$, so it commutes with the mode-reversal permutation. The overlap matrices $U_r$ satisfy a covariance property under simultaneous site reversal on all legs, inherited from the fact that the overlap $X^{(3)} = (X^{(1)}, X^{(2)})$ is a concatenation that commutes with simultaneous reversal within each block. Thus both diagonal blocks $M_r$ and off-diagonal blocks $U_r^T M_3 U_s$ of $G_T$ commute with $\Pi = \text{diag}(\Pi_1, \Pi_2)$.

2. **$\Pi B_T = B_T$:** The affine vector $\xi$ in the real nonzero-mode basis, under the mode-reversal parity, transforms as an even vector. This is because $\xi_m = (1/\sqrt{N_3})\sum_{n=0}^{N_1-1} e^{-2\pi i m n/N_3}$ satisfies $\xi_{N_3-m} = \bar\xi_m$, and in the real cosine/sine basis the parity acts on the pair $(k, N-k)$ by exchanging the real and imaginary parts with appropriate signs, under which $\xi$ is even. The matrix $U_r^T M_3$ is parity-covariant, so $B_T = (U_1^T M_3 \xi, U_2^T M_3 \xi)$ is parity-even.

3. **$d_- = (1, 0, \ldots, 0, -1)$ is parity-odd under $J_2$:** Under site reversal $n \to N_2-1-n$, the row $(1, 0, \ldots, 0, -1)$ maps to $(-1, 0, \ldots, 0, 1) = -d_-$. So $D_- \Pi = -D_-$.

Combining: $\eta_- = -D_- G_T^{-1} B_T = -D_- \Pi \cdot \Pi G_T^{-1} B_T = +D_- G_T^{-1} B_T = -\eta_-$, hence $\eta_- = 0$. Confirmed.

The left-arc stencil $d_+ = (-1, 1, 0, \ldots, 0)$ does NOT have definite parity under $J_1$: $J_1 d_+^T = (0, \ldots, 0, 1, -1)^T \neq \pm d_+^T$. So $\eta_+$ is generically nonzero. Confirmed.

### 3.5 Bond-locality formula

The note claims $\Delta_I^{\rm bond} = 2(X_{I_+} - X_{I_-})(X_{N_2-1}^{(2)} - X_{N_1-1}^{(1)})$. I verified this by direct expansion.

On the overlap, the bond sums are $\sum_{n=0}^{N_r-1}(X_{n+1}^{(r)} - X_n^{(r)})^2$ with periodic identification. The outgoing leg-3 bond sum uses $X^{(3)} = (X_0^{(1)}, X_1^{(1)}, \ldots, X_{N_1-1}^{(1)}, X_0^{(2)}, X_1^{(2)}, \ldots, X_{N_2-1}^{(2)})$. Every bulk bond (away from the join) cancels pairwise between the incoming and outgoing sums. The four surviving bonds are:

- Leg 1 wrap-around: $(X_0^{(1)} - X_{N_1-1}^{(1)})^2$
- Leg 2 wrap-around: $(X_0^{(2)} - X_{N_2-1}^{(2)})^2$
- Leg 3 cross-bonds at the join: $-(X_0^{(2)} - X_{N_1-1}^{(1)})^2 - (X_0^{(1)} - X_{N_2-1}^{(2)})^2$

Setting $X_{I_+} = X_0^{(1)}$, $X_{I_-} = X_0^{(2)}$:
$(X_{I_+} - X_{N_1-1})^2 + (X_{I_-} - X_{N_2-1})^2 - (X_{I_-} - X_{N_1-1})^2 - (X_{I_+} - X_{N_2-1})^2$

Expanding: $X_{I_+}^2 - 2X_{I_+}X_{N_1-1} + X_{I_-}^2 - 2X_{I_-}X_{N_2-1} - X_{I_-}^2 + 2X_{I_-}X_{N_1-1} - X_{I_+}^2 + 2X_{I_+}X_{N_2-1}$

$= 2(X_{I_+} - X_{I_-})(X_{N_2-1} - X_{N_1-1})$

Wait -- this gives $(X_{N_2-1}^{(2)} - X_{N_1-1}^{(1)})$, which has the OPPOSITE sign from the note's $X_{N_2-1}^{(2)} - X_{N_1-1}^{(1)}$. Let me re-check...

Actually, re-reading the note: it writes $\Delta_I^{\rm bond} = B_1 + B_2 - B_3$, and the formula gives $2(X_{I_+} - X_{I_-})(X_{N_2-1}^{(2)} - X_{N_1-1}^{(1)})$. My expansion above gives the same thing. The note's displayed intermediate formula has the same result. **Confirmed.**

### 3.6 Critical dimension signal at $D_\perp = 24$

The numerical claim is that the leg-factorization residual for $\log C_{\rm req}^{(B)} = q_{\rm rel}^2/(2\gamma_T) - D_\perp \log\kappa_{\rm 1d}$ has a sharp minimum at $D_\perp = 24$:

| $D_\perp$ | RMS residual |
|---|---|
| 22 | $1.08 \times 10^{-3}$ |
| 23 | $5.42 \times 10^{-4}$ |
| **24** | **$1.48 \times 10^{-9}$** |
| 25 | $5.42 \times 10^{-4}$ |
| 26 | $1.08 \times 10^{-3}$ |

This is tested over 3249 independent joins on a $4 \le N_1, N_2 \le 60$ grid, making it an extremely high-dimensional consistency check. The probability of accidental cancellation at a specific integer is negligible.

The fact that the two individual pieces ($q_{\rm rel}^2/(2\gamma_T)$ and $\log\kappa_{\rm 1d}$) are separately NOT leg-factorizable (with residuals $\sim 10^{-2}$ and $\sim 10^{-4}$ respectively) but their combination at $D_\perp = 24$ becomes factorizable to $10^{-9}$ is a non-trivial cancellation. Physically, this reflects the same zero-point-energy/normal-ordering balance that gives $D = 26$ from the Lorentz algebra, but detected here at the amplitude level.

**Potential concern:** Could a systematic normalization error in either ingredient (say, a missing factor of 2 somewhere) shift the critical dimension to a different value? No: the factorization test is a high-dimensional over-determined system over thousands of $(N_1, N_2)$ pairs, and the $D_\perp$-scan is symmetric around 24 with $D_\perp = 24$ being a factor of $\sim 10^6$ better than its neighbors. A normalization error would shift the minimum to a non-integer value, which would be immediately visible.

### 3.7 Bose-Fermi cancellation at higher loops (not yet in note)

In the superstring, the amplitude involves $\det'(\mathcal{M}_B)^{-D_\perp/2} \cdot \text{Pf}(\mathbb{A}_F)$. In the continuum, the Jacobi abstruse identity ensures exact cancellation of nonzero-mode determinants on the torus (one-loop cosmological constant vanishes). On the lattice, this cancellation holds only approximately, with the error vanishing as $a \to 0$. The rate of this Bose-Fermi cancellation at finite $N$ could be an important numerical stability issue for the higher-loop program. **Recommendation: test the Bose-Fermi determinant ratio on a single cylinder at one loop to characterize the finite-$N$ violation of spacetime supersymmetry.**

---

### 3.8 Large-$N$ asymptotics of the one-string factors

The note reports the fitted asymptotic form:

$$f_{\rm in}(N) = -\lambda_{\rm gauge} N + 7\log N + \pi/N + (\pi^2/72)/N^2 + c_{\rm in} + O(N^{-3})$$
$$f_{\rm out}(N) = \lambda_{\rm gauge} N - 5\log N - \pi/N + (\pi^2/72)/N^2 + c_{\rm out} + O(N^{-3})$$

The gauge-invariant combination for the full three-point function gives:

$$\log C_{\rm req} = C_{\rm tail} + 7\log N_1 + 7\log N_2 - 5\log N_3 + \pi(1/N_1 + 1/N_2 - 1/N_3) + \ldots$$

**The $\log N$ coefficients:** At $D_\perp = 24$, the incoming coefficient 7 and outgoing coefficient $-5$ should encode the critical-dimension data. In the continuum, the vertex normalization involves $\prod_r |\alpha_r|^{c_r}$ with $c_r$ related to the central charge and the Mandelstam map Jacobian. The combination $7 + 7 - 5 = 9$ is the total power of the lattice spacing $a$ (since $\log C_{\rm req}$ includes $-9\log a$ in the physical-$\alpha$ version). This 9 should decompose into contributions from:
- The $(D_\perp/2)\log\det M$ term in $\kappa_{\rm 1d}$ (zero-point scaling)
- The $-\frac{1}{2}\log\det G_T$ term (sewing determinant)
- The $\frac{1}{2}\log(2\pi/\gamma_T)$ term (Schur complement)

An analytic derivation of these integer coefficients from the finite-$N$ Gaussian structure would provide a strong independent check of the numerical fits. **Recommendation: derive the leading $\log N$ asymptotics of $\gamma_T$, $\det G_T$, and $\det M_r$ using Euler-Maclaurin or Szego-type asymptotic theorems for Toeplitz-like determinants.**

**The $1/N$ coefficient $\pm\pi$:** This is the finite-size correction to the zero-point energy, related to the Casimir energy on the cylinder. The fact that it enters with opposite signs for incoming and outgoing legs (but cancels in the gauge-invariant combination as $\pi(1/N_1 + 1/N_2 - 1/N_3)$) is consistent with the renormalized ZPE mismatch formula already checked in `lorentz_check.py`.

**The $1/N^2$ coefficient $\pi^2/72$:** This appears with the SAME sign on both incoming and outgoing legs (and in the gauge-invariant combination as $(\pi^2/72)(1/N_1^2 + 1/N_2^2 + 1/N_3^2)$). This is a more delicate finite-size correction. The value $\pi^2/72$ is suggestive: $72 = 12 \times 6$, and the Bernoulli-number expansion of $\cot(x)$ involves $B_{2k}/(2k)!$ coefficients. A systematic asymptotic expansion of the discrete formulas would pin this down.

---

## 4. Cross-Checks Between Note and Code

| Claim in note | Script | Status |
|---|---|---|
| Completeness $U_1 U_1^\dagger + U_2 U_2^\dagger + \hat\xi\hat\xi^\dagger = I$ | `lorentz_check.py`, `tachyon_check.py`, `neumann_check.py` | **Verified** to $10^{-14}$ |
| Bond locality $\Delta_I^{\rm bond}$ formula | `lorentz_check.py` | **Verified** to machine precision |
| ZPE mismatch convergence to continuum | `lorentz_check.py` | **Verified**, $O(1/N^2)$ rate |
| Leg factorization of $C_{\rm req}^{(B)}$ | `tachyon_check.py` | **Verified** to $10^{-12}$ across grid |
| $\eta_- = 0$ for minimal right-arc stencil | `superstring_prefactor_check.py` | **Verified**, $|\eta_-| < 6 \times 10^{-15}$ for $4 \le N_1, N_2 \le 40$ |
| Parity explanation for $\eta_- = 0$ | `superstring_prefactor_check.py` | **Verified**: $[P, G_T] = 0$ and $P B_T = B_T$ to machine precision |
| Second-order stencil restores $\eta_- \neq 0$ | `superstring_prefactor_check.py` | **Verified** |
| SO(8) Weyl map tensor structure | `gs_weyl_symbol_diagnostic.py` | **Verified**: residual $\sim 10^{-13}$ |

---

## 5. Identified Errors and Issues

### 4.1 No outright errors found in the derivations
All formulas that I have been able to check against the numerical implementations are consistent. The Gaussian integration, overlap algebra, Schur complement, and parity arguments are correct.

### 4.2 Potential sign/convention issues to watch

1. **Wick rotation sign.** The note uses $\tau_{\rm M} = -i\tau$. This gives $e^{iS_{\rm M}} \to e^{-S_{\rm E}}$ with positive bosonic quadratic form. Consistent throughout, but care is needed when comparing with references that use $\tau_{\rm M} = +i\tau$.

2. **Orientation signs at cubic vertices.** The canonical one-form invariance with $(\epsilon_1, \epsilon_2, \epsilon_3) = (+1, +1, -1)$ must be consistently maintained in the momentum overlap $\pi^{(r)} = U_r^\dagger \pi^{(3)}$ (not $-U_r^\dagger \pi^{(3)}$). The note handles this correctly by separating the variational derivation from the overall vertex orientation.

3. **Fermionic coherent-state conventions.** The sign in $K_F = \exp(\bar\eta_f U \eta_i)$ vs $\exp(-\bar\eta_f U \eta_i)$ and the corresponding Berezin measure convention must be tracked carefully through the sewing. The note is explicit about its convention but defers the absolute normalization.

### 4.3 Items explicitly identified as unresolved in the note

These are not errors but honest gaps that the note itself flags:

- Continuum lightcone measure $J_{\rm M}$ (Sec 8)
- Operator-level bosonic Lorentz check and $D=26$ derivation (Sec 12)
- Higher-genus spin-structure/GSO bookkeeping (Sec 6, 17)
- Discrete supercharge-closure calculation (Sec 15)
- Full component convention for $v_{IJ}$ and the GS prefactor dictionary (Sec 14-15)
- Quartic and higher contact terms at order $g^2$ (Sec 3, 15)

---

## 6. Recommendations for Next Steps

### Priority 1: Operator-level bosonic Lorentz check (Sec 12)

This is the single most important calculation that should be done next. The ingredients are:
- The discrete free Hamiltonian $\mathcal{H}_r^{\rm lat}$ (already explicit)
- The discrete Lorentz generator $J_2^{-I}$ in site/oscillator variables
- The cubic overlap state $|V_3^{(B)}\rangle$ (already explicit)

The calculation reduces to evaluating the commutator $[J_2^{-I}, P_3^-]|V_3^{(B)}\rangle$ at finite $N$ and extracting the join-local remainder. This should:
1. Fix the overall cubic normalization $\mathcal{C}_3^{(B)}$ as a function of $N$
2. Verify that the join-local anomaly cancels at $D=26$ in the continuum limit
3. Derive the one-string normalization factors $Z_{\rm in}(N)$, $Z_{\rm out}(N)$ analytically, confirming the numerical factorization already observed

**Concrete deliverable:** A script that computes the commutator numerically at finite $N$ and plots the residual vs $D_\perp$ to identify the critical dimension.

### Priority 2: Discrete supercharge-closure calculation (Sec 15)

Once the bosonic Lorentz check is done, the next step is to:
1. Write out the free lightcone GS supercharge $Q_2^-$ in the finite-$N$ site/oscillator basis, in one fixed SO(8) gamma-matrix convention (the one already chosen in `so8_gamma.py`)
2. Compute the projected matrix elements $\lambda_\bullet$ that enter the block-triangular closure chain
3. Solve the resulting algebraic system for $(b_1, b_3, b_5, b_7)$ and check the top residual
4. Determine whether the symmetric second-order stencil $(t_+, t_-) = (-1/2, +1/2)$ is sufficient or whether additional local data are needed

**This is the step that either validates or falsifies the discrete-sigma program for the superstring.** If the closure chain is solvable with smooth large-$N$ behavior, the construction works. If not, one needs to enlarge the local ansatz.

### Priority 3: Three-graviton amplitude completion

With the supercharge coefficients in hand:
1. Complete the fermionic zero-mode contraction using the explicit Clifford module
2. Combine with the already-computed bosonic prefactor tensor
3. Verify that the result tends to $V^{IJK} V^{I'J'K'}$ in the continuum limit

### Priority 4: First loop integrand

With the cubic vertex fully fixed:
1. Assemble the one-loop (genus-1) two-point or vacuum diagram
2. Include the twist modulus integration
3. Compute the explicit finite-$N$ integrand as a function of $(T, \varphi)$
4. Check UV (short-edge) and IR (long-cylinder) behavior

### Priority 5: Full squeeze matrix extraction and continuum Neumann comparison

The current `neumann_check.py` computes only the reduced sewn quadratic form (after eliminating leg 3). For a proper comparison with continuum lightcone Neumann coefficients, one should:
1. Extract $S^{(B)} = -\mathbb{M}_{\rm phys}^{-1} \mathbb{N}_{\rm phys}$ from the full three-leg overlap
2. Verify $S^{(B)T} = S^{(B)}$ numerically
3. Compare the entries against continuum $\bar{N}^{rs}_{mn}$ in a matched Fourier convention

### Secondary items

- **Refined continuum-limit analysis of $\gamma_T$ and the tachyon exponent:** The current data show clean convergence but the analytic large-$N$ asymptotics of $\gamma_T(N, \alpha)$ have not been derived. A saddle-point or Euler-Maclaurin analysis of the Schur complement would give the leading and subleading corrections.

- **Implementation of the $\sigma$-twist for loop diagrams:** The formulas are in the note but no code yet implements the twisted propagator $K_B(T, \varphi)$ or the associated twisted sewing.

- **Spin-structure/GSO bookkeeping:** Deferred but will be needed for any physical superstring loop computation. The framework for this is standard but the discrete-$\sigma$ form of the fermion periodicity conditions needs to be spelled out.

---

## 7. New Result: Neumann Coefficient Extraction and a Structural Finding

### 7.1 The symplectic obstruction

Attempting to construct the full three-leg squeeze matrix $S^{(B)} = -\mathbb{M}_{\rm phys}^{-1}\mathbb{N}_{\rm phys}$ from the annihilation constraints revealed a structural issue: the symplectic compatibility condition $C_Q C_P^T = 0$ (required for $S$ to be symmetric) fails:

$$C_Q C_P^T = 2(U_1, U_2) \neq 0$$

This is verified to machine precision across all tested lattice sizes. Consequently, the naive $\mathbb{M}^{-1}\mathbb{N}$ gives a NON-symmetric matrix ($|S - S^T| \sim 10^{15}$), which is inconsistent with the squeezed-state ansatz.

**Root cause:** The cubic overlap state is a boundary state (position-space delta function), not a normalizable squeezed state. The squeeze matrix has $|S| = 1$ eigenvalues, making the matrix inversion inherently ill-conditioned. The position and momentum constraints are not "orthogonal" in the symplectic sense needed for the naive inversion to work.

**Note for the tex note:** The formula $S^{(B)} = -\mathbb{M}_{\rm phys}^{-1}\mathbb{N}_{\rm phys}$ (around line 1152) should be used with care. It is formally correct as a distributional identity for the boundary state, but its direct numerical evaluation via matrix inversion is ill-conditioned. The correct numerical extraction uses position-space Gaussian matrix elements.

### 7.2 Correct Neumann extraction via Gaussian moments

The Neumann coefficients can be extracted directly from position-space Gaussian matrix elements:

$$\bar{N}^{rs}_{mn} = -L_m^{(r)} G_T^{-1} (L_n^{(s)})^T$$

where $L_m^{(r)}$ is the effective linear functional of the two-leg Gaussian obtained by acting with $a_m^{(r)}$ on the combined bra-ket product and using the overlap to eliminate leg 3.

For incoming legs: $L_m^{(r)} = \sqrt{M_r/2} \cdot (\text{embedding}) - (1/\sqrt{2M_r}) \cdot G_{T,m:}$

For the outgoing leg: $L_k^{(3)} = \sqrt{2M_3} \cdot (U_1, U_2)_{k,:}$

(The leg-3 formula uses $a_k^{(3)\dagger}\psi_0^{(3)} = \sqrt{2\mu_3\omega_k} \cdot q_k^{(3)} \cdot \psi_0^{(3)}$.)

**Symmetry is guaranteed** by the symmetry of $G_T^{-1}$, and verified numerically to $10^{-16}$.

**Convergence behavior:** The diagonal Neumann entries $\bar{N}^{11}_{00}$ grow logarithmically with $N$ (branch-point normal-ordering divergence), while $\bar{N}^{33}_{00} \to -0.5$. The Hamiltonian matrix elements $H2_{mn} = -\bar{N}^{12}_{mn}(\omega_m/(2p_1^+) + \omega_n/(2p_2^+))$ vanish as $O(1/N)$.

This extraction is implemented in `squeeze_matrix_check.py` and provides the first step toward the operator-level Lorentz check.

### 7.3 What remains for the full operator-level Lorentz check

The Neumann extraction above gives the building blocks, but the full commutator $[J_2^{-I}, P_3^-] + [J_3^{-I}, P_2^-] = 0$ additionally requires:

1. The Virasoro-like rotation part of $J_2^{-I}$, which involves products of oscillators at different mode numbers
2. The normal-ordering anomaly at the interaction point, which comes from the contraction of the Virasoro generators with the Neumann coefficients
3. The construction of $J_3^{-I}$ as the local interaction-point completion

The Neumann extraction provides ingredient (2) in principle, but assembling the full anomaly coefficient requires the Virasoro generator in the finite-$N$ oscillator basis, which is a further step.

---

---

# End-to-End Review of the Updated Note (2026-03-30)

## Issues Found

### Issue A (moderate): Fermionic squeeze matrix needs the same boundary-state caveat

Section 13 (Fermionic kinematic overlap, lines 2200-2215) still writes:

$$S^{(F)} = -\mathbb{M}_{F,\rm phys}^{-1}\mathbb{N}_{F,\rm phys}$$

and claims "$S^{(F)}$ must be antisymmetric, $S^{(F)T} = -S^{(F)}$." However, the fermionic overlap matrices $C_\Theta$ and $C_{\Pi_\Theta}$ have the **same structure** as the bosonic $C_Q$ and $C_P$:

$$C_\Theta = (U_1, U_2, -I), \quad C_{\Pi_\Theta} = \begin{pmatrix}I & 0 & -U_1^T\\0 & I & -U_2^T\end{pmatrix}$$

Since these have $C_\Theta C_{\Pi_\Theta}^T = 2(U_1, U_2) \neq 0$ (identical to the bosonic obstruction), the antisymmetry condition $\mathbb{M}_F\mathbb{N}_F^T + \mathbb{N}_F\mathbb{M}_F^T = 0$ also fails. The required condition expands to:

$$\mathbb{M}_F\mathbb{N}_F^T + \mathbb{N}_F\mathbb{M}_F^T = \begin{pmatrix}2C_\Theta C_\Theta^T & 0 \\ 0 & -2C_{\Pi_\Theta}C_{\Pi_\Theta}^T\end{pmatrix} \neq 0$$

**Mitigating factor:** For Grassmann variables, the overlap state is automatically normalizable (unlike the bosonic boundary state), because Grassmann integrals are bounded. So the formal squeezed-state representation $\exp(\frac{1}{2}B^\dagger S^{(F)} B^\dagger)|0\rangle$ is well-defined as a finite Grassmann polynomial regardless of the eigenvalue magnitude of $S^{(F)}$.

**Suggested fix:** Add a caveat analogous to the bosonic one. The safest formulation would note that the direct inversion may not give an antisymmetric matrix numerically, and that the Grassmann-moment extraction (the fermionic analogue of the Gaussian-moment Neumann definition) is the preferred finite-$N$ definition.

### Issue B (minor): Wording "not a stable numerical definition" understates the obstruction

Line 1237: "the naive direct inversion $-\mathbb{M}_{\rm phys}^{-1}\mathbb{N}_{\rm phys}$ is not a stable numerical definition of a symmetric squeeze matrix."

The obstruction is not merely numerical instability — it is algebraic: $C_Q C_P^T \neq 0$ means $\mathbb{M}\mathbb{N}^T \neq \mathbb{N}\mathbb{M}^T$ exactly, so $S = -\mathbb{M}^{-1}\mathbb{N}$ is provably non-symmetric even in exact arithmetic. The current wording could mislead a reader into thinking higher precision would help.

**Suggested fix:** Replace "is not a stable numerical definition" with "does not yield a symmetric matrix, because $C_Q C_P^T \neq 0$ implies $\mathbb{M}\mathbb{N}^T \neq \mathbb{N}\mathbb{M}^T$ algebraically."

### Issue C (minor): Notation collision between $\mu_\pm$ in TTM and $\eta_\pm$ in prefactor check

The TTM subsection (Section 12.2) defines $\mu_\pm \equiv \zeta_\pm - L_\pm G_T^{-1}B_T$ as the completed one-point sources. The superstring prefactor section (Section 15.1) defines $\eta_\pm \equiv -D_\pm G_T^{-1}B_T$ for the analogous quantities in the interaction-point stencil analysis. These are structurally similar objects with different names. Not a conflict (they live in different sections), but worth noting for readers who work through both.

### Issue D (cosmetic): Equation label scheme inconsistency

Most equations have systematic section-based labels like `\label{eq:s14-003}`, but some key equations use descriptive labels: `\label{eq:V3B-squeezed}`, `\label{eq:bosonic-symplectic-obstruction}`, `\label{eq:gammaT}`, `\label{eq:massless-normalized-operator}`, `\label{eq:ttm-bosonic-coefficients}`, `\label{eq:gaussian-moment-neumann}`. Both schemes are fine individually, but mixing them may cause confusion when searching for a specific equation. No fix needed, just awareness.

### Issue E (physics): The Dijkgraaf-Motl paragraph should clarify the $\alpha$ convention

The new paragraph (after line 2476) unpacks the matrix-string interpretation and writes $v^{ij}(\Lambda) \leftrightarrow 16\,\Sigma^j\widetilde\Sigma^i$. The numerical coefficient 16 (from Dijkgraaf-Motl) depends on the normalization convention for the Clifford algebra and the spin fields. The note should specify that this is the Dijkgraaf-Motl normalization convention, which may differ from the Pankiewicz-Stefanski convention used in the `gs_zero_mode_prefactor.py` helper by an overall constant depending on the $\alpha$ ratio.

## Items Verified as Correct

1. **$C_Q C_P^T = 2(U_1, U_2)$** — confirmed analytically and numerically.
2. **Gaussian-moment Neumann definition** — symmetry guaranteed by $G_T^{-1}$ symmetry, verified numerically to $10^{-16}$.
3. **TTM kinematics $q_{\rm rel}^2 = 4/\alpha'$** — independently rederived.
4. **TTM covariance $C_{1,N} = \pi\alpha'/(2N\sin(\pi/N))$** — correct from $\mu\omega_1$ with the bra-ket factor of 1/2.
5. **TTM tensor decomposition** — Gaussian moment computation is correct; the trace part $A_{\rm tr}^{(M)} = O(1/N)$ and the physical coefficient $B_{\rm rel}^{(M)}$ converges to a smooth continuum profile.
6. **All equation labels** — no duplicates, all cross-references resolve correctly.
7. **LaTeX compilation** — clean, no errors or warnings.
8. **Section numbering** — "Section 15.1" in the TTM subsection correctly refers to the three-graviton check.
9. **Updated summary** — correctly describes both the TTT and TTM diagnostics.
10. **Scattered "squeeze matrix" → "Gaussian-moment Neumann" updates** — consistent throughout Sections 16, 19, 20, 21.

## Suggested Next Steps (revised focus)

**Key reorientation:** The point of this project is to develop a practical numerical method for computing string amplitudes in lightcone gauge by discretizing the sigma circle. Lorentz invariance is already guaranteed by standard LC gauge quantization — it does not need to be re-derived. The goal is to show the discrete-sigma method *works numerically*, for both bosonic and superstring cases.

### Priority 1: Complete the bosonic three-tachyon amplitude to a numerical value

The kinematic structure ($\gamma_T$, leg factorization, critical-dimension signal) is established. What's needed to turn this into a complete numerical amplitude:
- Fix the overall cubic normalization $\mathcal{C}_3^{(B)}$ by matching to the known continuum three-tachyon coupling constant
- Verify convergence of the full amplitude (not just the kinematic factor) as $N \to \infty$
- Benchmark against the known analytic result

### Priority 2: Complete the superstring three-graviton amplitude

This is the key test that the method works for the superstring. The bosonic nonzero-mode part is already reduced explicitly. What remains:
- Fix the GS prefactor by importing the continuum $v_{IJ}(\Lambda)$ coefficients (already in `gs_zero_mode_prefactor.py`)
- Complete the fermionic zero-mode contraction using the Clifford module (already in `gs_zero_mode_module.py`)
- Evaluate the full matrix element and verify it matches the Einstein-Hilbert/Yang-Mills-squared tensor
- The stencil choice (second-order vs minimal) should be determined by matching to the known answer

### Priority 3: First loop integrand

With the cubic vertex validated at tree level, assemble the one-loop diagram:
- Implement the twisted propagator $K_B(T, \varphi)$ with the sigma-shift operator
- Build the one-loop two-point or vacuum integrand as an explicit function of $(T, \varphi)$
- Verify UV (short-edge) and IR (long-cylinder) behavior
- Compare the integrated result against known one-loop amplitudes

### Priority 4: Continuum extrapolation and error control

Develop the practical numerical pipeline:
- Systematic $N \to \infty$ extrapolation using Richardson or similar methods
- Characterize the convergence rate ($O(1/N)$, $O(1/N^2)$, etc.) for different quantities
- Uncertainty estimates for finite-$N$ computations

### Lower priority items

- **Add fermionic boundary-state caveat to Section 13** (the same $C_\Theta C_{\Pi_\Theta}^T \neq 0$ obstruction applies, but this is a formal issue, not a numerical blocker)
- **Verify Neumann convergence against continuum** (useful for validation but not on the critical path)
- **Spin-structure/GSO bookkeeping** (needed for physical superstring loops)

---

---

# Test Suite Results (2026-03-30)

## Summary: 20/20 tests pass

Three test scripts validate the discrete-sigma numerical method across all major components.

## test_tachyon_amplitude.py (9/9 pass)

| Test | Key result | Status |
|---|---|---|
| Overlap identities (8,12) | completeness $1.3 \times 10^{-15}$ | PASS |
| Overlap identities (16,24) | completeness $3.1 \times 10^{-15}$ | PASS |
| Overlap identities (32,48) | completeness $6.5 \times 10^{-15}$ | PASS |
| $\gamma_T$ convergence (2:3) | Richardson extrapolation $\to 0.35662$ | PASS |
| $\gamma_T$ convergence (1:1) | Richardson extrapolation $\to 0.36077$ | PASS |
| $\gamma_T$ convergence (1:2) | Richardson extrapolation $\to 0.34917$ | PASS |
| **Critical dimension** | **Best $D_\perp = 24$, separation $10^5$** | **PASS** |
| Large-$N$ asymptotics | $C_{\rm tail} = -22.496$, rms $= 2.5 \times 10^{-7}$ | PASS |
| Ratio independence | Informational (spread at finite $N$) | PASS |

### Critical dimension detail:

| $D_\perp$ | RMS factorization residual |
|---|---|
| 22 | $5.44 \times 10^{-4}$ |
| 23 | $2.72 \times 10^{-4}$ |
| **24** | **$2.62 \times 10^{-9}$** |
| 25 | $2.72 \times 10^{-4}$ |
| 26 | $5.44 \times 10^{-4}$ |

The factorization residual at $D_\perp = 24$ is $10^5 \times$ smaller than its neighbors.

### $\gamma_T$ convergence (ratio 2:3):

| $N_1$ | $\gamma_T$ |
|---|---|
| 8 | 0.279060 |
| 16 | 0.336817 |
| 32 | 0.346463 |
| 64 | 0.351470 |
| 128 | 0.354022 |
| 256 | 0.355317 |
| $\infty$ (Richardson) | 0.356623 |

## test_graviton_prefactor.py (6/6 pass)

| Test | Key result | Status |
|---|---|---|
| Parity obstruction | $\eta_- = 3.3 \times 10^{-16}$ (zero to machine precision) | PASS |
| Parity scan (4..30) | max$|\eta_-| = 3.5 \times 10^{-15}$ across 729 pairs | PASS |
| Second-order stencil | Restores $\eta_- = -0.0557$, $B_{qq} = 0.063$ | PASS |
| Prefactor convergence | $N_1 B_{qq} \to 1.147$ (converging) | PASS |
| Ratio scan | Smooth $\lambda$-dependence | PASS |
| **Weyl tensor structure** | **Residual $6.4 \times 10^{-14}$** | **PASS** |

### Stencil comparison at $(N_1, N_2) = (16, 24)$:

| Stencil | $\eta_+$ | $\eta_-$ | $B_{qq}$ |
|---|---|---|---|
| minimal/minimal | 0.1096 | 0.0000 | 0.0000 |
| minimal/second | 0.1096 | $-0.0557$ | 0.0538 |
| second/second | 0.1278 | $-0.0557$ | 0.0628 |

### Prefactor convergence (second/second, ratio 2:3):

| $N_1$ | $N_1 B_{qq}$ | $A_\delta$ |
|---|---|---|
| 16 | 1.005 | 0.355 |
| 32 | 1.079 | 0.330 |
| 64 | 1.117 | 0.318 |
| 128 | 1.137 | 0.312 |
| 256 | 1.147 | 0.309 |

### Weyl-map tensor structure:
The $8_v \to 8_v$ block of $v_{IJ}$ under the candidate Weyl map fits exactly onto $\{A\delta_{IJ}\delta_{KL}, B\delta_{IK}\delta_{JL}, C\delta_{IL}\delta_{JK}\}$ with residual $6.4 \times 10^{-14}$. No exotic tensor structures appear.

## test_neumann_extraction.py (5/5 pass)

| Test | Key result | Status |
|---|---|---|
| Symplectic obstruction | $C_Q C_P^T = 2(U_1, U_2)$ exact | PASS |
| Neumann symmetry | $\bar{N}^{rs}_{mn} = \bar{N}^{sr}_{nm}$ to $6 \times 10^{-16}$ | PASS |
| Neumann vs reduced Gaussian | $\gamma_T$ agreement exact | PASS |
| Massless covariance $C_{1,N}$ | Formula matches to $10^{-16}$ | PASS |
| TTM trace suppression | $A_{\rm tr} = O(1/N)$, $B_{\rm rel}$ converges | PASS |

### TTM convergence (ratio 2:3):

| $N_1$ | $A_{\rm tr}$ | $B_{\rm rel}$ |
|---|---|---|
| 16 | 0.0362 | 2.077 |
| 32 | 0.0181 | 1.998 |
| 64 | 0.0091 | 1.959 |
| 128 | 0.0045 | 1.940 |
| 256 | 0.0023 | 1.931 |

$A_{\rm tr} \approx 0.58/N_1$ (lattice artifact, vanishes in continuum). $B_{\rm rel} \to 1.92$ (physical).

---

## Production-Ready Components (for Codex extraction)

### Core infrastructure (production quality):
- **`tachyon_check.py`**: Real-basis overlap data, mode metrics, Schur complement, full Gaussian assembly. All functions tested, all identities verified to machine precision.
- **`bosonic_massless_check.py`**: TTM amplitude with exact external-state normalization. Clean tensor decomposition.
- **`superstring_prefactor_check.py`**: Bosonic interaction-point prefactor with stencil family parameterization and parity analysis.
- **`squeeze_matrix_check.py`**: Gaussian-moment Neumann coefficient extraction with guaranteed symmetry.

### Supporting diagnostics (reference quality):
- **`lorentz_check.py`**: ZPE mismatch, bond locality, overlap completeness.
- **`neumann_check.py`**: Reduced quadratic-form convergence analysis.
- **`so8_gamma.py`**, **`gs_zero_mode_prefactor.py`**, **`gs_zero_mode_module.py`**: SO(8) convention and Clifford module.
- **`gs_weyl_symbol_diagnostic.py`**: Weyl-map tensor structure test.
- **`supercharge_closure_chain.py`**: Algebraic scaffold for the SUSY closure problem.

### What's ready for production use:
1. The cubic overlap algebra (`overlap_data`, `mode_metric`, `real_zero_sum_basis`)
2. The three-tachyon Gaussian assembly ($G_T$, $B_T$, $\gamma_T$, $\kappa_{1d}$)
3. The TTM tensor decomposition ($\Sigma_1$, $\mu_\pm$, $A_{\rm tr}$, $B_{\rm rel}$)
4. The bosonic prefactor stencils (minimal and second-order families)
5. The Gaussian-moment Neumann extraction
6. The leg-factorization test and critical-dimension scan

## test_graviton_assembly.py (3/3 pass)

| Test | Key result | Status |
|---|---|---|
| Fermionic zero-modes | Clifford error $= 0$, Weyl fits $< 4 \times 10^{-13}$ | PASS |
| $v_{IJ}$ matrix elements | $A=25$, $B=-32+12i$, $C=-32-12i$, residual $6.4 \times 10^{-14}$ | PASS |
| Ratio dependence | Smooth $\alpha$-dependence, all Weyl residuals $< 3 \times 10^{-12}$ | PASS |

### Graviton $v_{IJ}$ matrix elements (Weyl quantization, $\alpha = 1$):

The $8_v \to 8_v$ block of the quantized $v_{IJ}(\Lambda)$ operator decomposes as:
$$\langle K|v_{IJ}|L\rangle = A\,\delta_{IJ}\delta_{KL} + B\,\delta_{IK}\delta_{JL} + C\,\delta_{IL}\delta_{JK}$$

with $A = 25$, $B = -32+12i$, $C = -32-12i$. The imaginary parts come from the odd-Grassmann-degree pieces ($y_2$ and $y_6$ terms in the prefactor). This is the first explicit computation of these matrix elements in the discrete-sigma framework.

### Bosonic prefactor convergence (second/second stencil, ratio 2:3):

| $N_1$ | $A_\delta$ | $B_{qq}$ |
|---|---|---|
| 16 | 0.3548 | 0.0628 |
| 32 | 0.3301 | 0.0337 |
| 64 | 0.3177 | 0.0175 |
| 128 | 0.3116 | 0.0089 |
| 256 | 0.3086 | 0.0045 |

$A_\delta \to 0.305$ (finite limit), $B_{qq} \to 0$ as $O(1/N)$.

### What needs completion before production:
1. **Cubic normalization matching**: $C_{\rm tail} = -22.496$ identified but not yet matched to the continuum $g_c$.
2. **Fermionic zero-mode contraction**: Weyl map validated but full graviton matrix element not yet assembled.
3. **Twisted propagator**: Formulas in the note but no code yet for the $\varphi$-dependent sewing.
4. **Loop integrand assembly**: Requires twisted propagator + multi-vertex sewing.

---

---

# Review of Codex Updates (latest round)

## New files

| File | Tests | Status |
|---|---|---|
| `fermionic_graviton_contraction.py` | 3/3 | PASS |
| `superstring_decisive_test.py` | 2/2 | PASS |
| `superstring_normalization_factorization.py` | 2/2 | PASS |

Total suite: 37/37 passing.

## Key new result: explicit fermionic Grassmann contraction

The `fermionic_graviton_contraction.py` implements the 16-Grassmann-variable integral for the tree-level three-graviton fermionic factor in eq. (s15-1-009b) of the main note. The benchmark at $(N_1, N_2) = (128, 192)$ with the trace-dropped second-order stencil gives:
- Graviton channels: nonzero with correct transverse-rotation ratios ($1:1/2$ for perp23:perp24, verified to $10^{-15}$)
- Dilaton channels: zero exactly
- B-field channels: zero exactly

## Open question: comparison to known analytic answer

**The fermionic contraction has NOT yet been compared to the known continuum three-graviton coupling.** The current tests verify:
1. Internal consistency (dilaton/B-field vanish, transverse ratios correct)
2. Channel selection rules match SO(8) symmetry expectations

But they do NOT yet verify:
- That the overall normalization matches the Einstein-Hilbert coupling $\kappa$
- That the ratio between different graviton channels (e.g., parallel vs perpendicular) matches the Yang-Mills-squared tensor $V^{IJK}V^{I'J'K'}$
- That the $\lambda$-dependence matches the known Mandelstam-map Jacobian

**This is the critical next step.** The known continuum answer for the three-graviton lightcone amplitude involves the tensor $V^{IJK}(p_1, p_2, p_3)V^{I'J'K'}(p_1, p_2, p_3)$ contracted with the polarizations. A concrete test would be: for specific numerical polarizations and momenta, compute the continuum answer analytically and compare to the discrete result at large $N$.

The Dijkgraaf-Motl convention used in the code ($v^{ij} \leftrightarrow 16\Sigma^j\widetilde\Sigma^i$) should reproduce this, but the matching has not been verified quantitatively.

---

# Review of Codex Updates (second round)

## Changes to main note (365 lines added)

### Reorientation (confirmed from previous round)
- Section 12 renamed "Bosonic continuum target and numerical checks"
- Scope paragraph now explicitly states LC Lorentz invariance is taken as known background
- Roadmap/deliverables prioritize numerical amplitude validation over operator proofs

### New superstring content (major addition)

**Explicit fermionic contraction scan** (eqs s15-1-038e through s15-1-038j):
- Full scan over stencil family $t \in \{0, 1/8, ..., 3/4\}$ and $\lambda \in \{1/4, 1/3, 3/8, 2/5, 1/2\}$
- Channel relations verified to $10^{-14}$: $\mathcal{A}_F(\epsilon_{23},\epsilon_{24},\epsilon_\parallel)/\mathcal{A}_F(\epsilon_{23},\epsilon_{23},\epsilon_\parallel) = 1/2$ and $\lambda^2\mathcal{A}_F(\epsilon_\parallel,\epsilon_{23},\epsilon_{23})/\mathcal{A}_F(\epsilon_{23},\epsilon_{23},\epsilon_\parallel) = 1$
- Dilaton and B-field channels zero to machine precision
- Uses actual Grassmann contraction, not Weyl proxy

**Fermionic rank-1 factorization** (eqs s15-1-038h through s15-1-038j):
- $\tilde{\mathcal{A}}_{\rm diag}(\lambda,t) = \lim N_1\mathcal{A}_F$ is rank-1 with $\sigma_2/\sigma_1 = 1.11 \times 10^{-4}$
- Normalized profile deviation $< 7 \times 10^{-4}$ across the positive branch

**Weyl-block auxiliary (retained as cross-check)**:
- $A=25$, $B=-32+12i$, $C=-32-12i$ at $\alpha=1$ (same as before)
- Trace-dropped on-shell: $8A_{\rm on}+B_{\rm on}+C_{\rm on}=0$ exactly
- $M_{qq}/M_\delta = -8$, giving $\mathcal{A}_{\rm vec,\parallel}/\mathcal{A}_{\rm vec,\perp} = -7$ and $\mathcal{A}_{\rm dilaton}=0$

### Issues found

**None.** All new formulas checked:
- $8 \cdot 8/\lambda^2 + (-64/\lambda^2) = 0$ ✓
- $(8-64)/8 = -7$ ✓
- $1/2$ channel ratio from transverse rotational symmetry ✓
- $\lambda^2$ scaling relation is a nontrivial check — consistent with the continuum expectation that the three-graviton vertex carries a factor $\lambda^{-2}$ from the interaction-point geometry

### Open question (reiterated)
The channel relations~(s15-1-038f) and their $\lambda^2$ scaling have not yet been matched quantitatively against the known continuum $V^{IJK}V^{I'J'K'}$ tensor. This is the next critical comparison.

### Companion note updated
- Added fermionic scan results, channel relations~\eqref{eq:channel-relations}, and rank-1 factorization~\eqref{eq:ferm-rank1}
- Total: 11 pages, 37/37 tests referenced

---

# Detailed Scrutiny of the Fermionic Contraction (2026-03-30)

## Step-by-step verification of `fermionic_graviton_contraction.py`

### 1. Interaction-point zero-mode argument

The code substitutes $\Lambda^a \to -(1-\lambda)\lambda_1^a + \lambda\lambda_2^a$ where $\lambda = \alpha_1/\alpha_3$. This matches the note's eq (s15-1-009b): $v_{IJ}(\lambda\lambda_2 - (1-\lambda)\lambda_1)$. **Correct.**

### 2. External state wavefunction

The code builds $\Psi_\epsilon(\lambda) = (1/16)\sum_{ij}\epsilon_{ij}v_{ji}(\lambda)$, calling `v_prefactor_polynomial(alpha, j, i)` with transposed indices. For symmetric graviton polarizations $\epsilon_{ij} = \epsilon_{ji}$, this gives $\text{tr}(\epsilon\cdot v)/16$. **Correct.**

### 3. State 3 after delta-function integration

The delta function $\delta^8(\lambda_1+\lambda_2+\lambda_3)$ sets $\lambda_3 = -\lambda_1-\lambda_2$. The code substitutes $\Lambda \to \lambda_1 + \lambda_2$ into $\Psi_3$. Since $v_{IJ}$ is an even polynomial in $\Lambda$ (degrees 0,2,4,6,8), $v_{IJ}(-\Lambda) = v_{IJ}(\Lambda)$, so $\Psi_3(-\lambda_1-\lambda_2) = \Psi_3(\lambda_1+\lambda_2)$. **Correct.**

### 4. Polarization tensors

| Name | Definition | Trace | Frobenius norm |
|---|---|---|---|
| parallel | $\sqrt{8/7}(\hat{q}\hat{q} - \delta/8)$ | 0 | 1 |
| perp23 | $(e_2e_2 - e_3e_3)/\sqrt{2}$ | 0 | 1 |
| perp24 | $(e_2e_2 - e_4e_4)/\sqrt{2}$ | 0 | 1 |
| dilaton | $\delta/\sqrt{8}$ | $\sqrt{8}$ | 1 |
| b23 | $(e_2e_3 - e_3e_2)/\sqrt{2}$ | 0 (antisymmetric) | 1 |

All normalized, traceless graviton polarizations, trace dilaton, antisymmetric B-field. **Correct.**

**Note on $\hat{q}$ direction:** $\hat{q} = e_1$ (direction 0). This is consistent with $B_{qq}$ being added to the (0,0) entry of the bosonic tensor (line 330-331). **Correct.**

### 5. Decomposition into $R_\delta$ and $R_{qq}$ responses

$T_{\rm bos}^{IJ} = A_\delta\delta^{IJ} + B_{qq}\hat{q}^I\hat{q}^J$, so:

$T_{\rm bos}^{IJ}v_{IJ} = A_\delta\sum_i v_{ii} + B_{qq}v_{00}$

The `delta_piece` sums $v_{ii}$ over $i=0,...,7$ (line 378-384) and `qq_piece` is $v_{00}$ alone (line 386-390). **Correct.**

### 6. Closed-form formula: $\mathcal{R}_{qq}^{(23,23,\parallel)} = 4\sqrt{14}(1-\lambda)^2$

This is a numerically observed identity, verified to $2 \times 10^{-13}$ on the standard $\lambda$-grid and off-grid at $\lambda = 1/5, 3/5$. The factor $4\sqrt{14}$ encodes the specific SO(8) gamma-matrix trace in the 16-Grassmann integral. The $(1-\lambda)^2$ factor arises from the quadratic dependence on the coefficient $-(1-\lambda)$ in the substitution $\Lambda \to -(1-\lambda)\lambda_1 + \lambda\lambda_2$.

The assembled amplitude formula (eq s15-1-038d3d in the main note):

$$\mathcal{A}_F(\epsilon_{23},\epsilon_{23},\epsilon_\parallel; N,\lambda,t) = 4\sqrt{14}(1-\lambda)^2 B_{qq}(N;\lambda,t)$$

is verified to $1.6 \times 10^{-14}$ across the full $(t,\lambda,s)$ scan. **Correct.**

### 7. Channel ratios

From the closed forms:
- $\mathcal{A}_F(23,24,\parallel)/\mathcal{A}_F(23,23,\parallel) = 1/2$ — follows because $R_{qq}^{(23,24,\parallel)} = (1/2)R_{qq}^{(23,23,\parallel)}$
- $\lambda^2\mathcal{A}_F(\parallel,23,23)/\mathcal{A}_F(23,23,\parallel) = 1$ — follows because $R_{qq}^{(\parallel,23,23)} = R_{qq}^{(23,23,\parallel)}/\lambda^2$

The first ratio reflects transverse rotational symmetry (perp24 vs perp23). The second is a nontrivial $\lambda$-scaling that should be derivable from the SO(8) algebra. **Both verified to $10^{-14}$.**

### 8. Outstanding questions

1. **The coefficient $4\sqrt{14}$**: This specific algebraic number has not been derived analytically from the SO(8) gamma-matrix structure. It is purely a numerical observation. A derivation would provide an independent check.

2. **Comparison to continuum $V^{IJK}V^{I'J'K'}$**: The closed-form channel amplitudes have not been compared to the known Einstein-Hilbert cubic vertex. The continuum answer for $\langle h_1 h_2 h_3 | H_3 \rangle$ involves $V^{IJK}(p_1,p_2,p_3)V^{I'J'K'}(p_1,p_2,p_3)$ contracted with polarizations, and this should give specific numerical values for each channel that can be compared to the discrete results.

3. **Overall normalization**: The amplitude $\mathcal{A}_F \propto B_{qq}(N;\lambda,t)$ with the $1/N$ scaling still present. The continuum limit $N_1 \mathcal{A}_F \to 4\sqrt{14}(1-\lambda)^2 \cdot (N_1 B_{qq})_\infty$ gives a finite number, but its value has not been matched to $g_c\kappa$ (the string coupling times the gravitational coupling).

### 9. Verdict on the fermionic contraction code

**The implementation is correct.** Every step — the Grassmann algebra (merge signs, sparse multiplication), the interaction-point substitution, the external-state wavefunctions, the polarization tensors, the $A_\delta/B_{qq}$ decomposition, and the top-form extraction — has been verified. The closed-form formulas are robust numerical observations, verified across multiple $\lambda$ values including off-grid spot checks.

The missing piece is not computational but physical: matching the computed channel amplitudes to the known continuum answer.

---

## 8. Overall Assessment

The note is technically solid. Every formula I have been able to check against the numerical implementations is correct. The conceptual organization---exact kinematic overlap in position space, followed by a local dynamical prefactor---is clean and physically motivated. The identification of the parity obstruction for the minimal right-arc stencil is an important finding that narrows the interaction-point ambiguity.

The main risk is that the program stalls at the supercharge-closure step (Priority 2). If the projected closure chain turns out to have no smooth solution in the current local ansatz, one would need to enlarge the operator basis, and the question is whether that enlargement remains tractable. The note is honest about this risk.

The secondary risk is that the continuum lightcone measure factor $J_{\rm M}$, which the note does not determine, turns out to interact nontrivially with the discrete sewing in a way that complicates the higher-loop program. At tree level this is a normalization issue; at loop level it could affect the integrand structure.

Overall: a well-constructed framework with the right internal consistency checks in place, awaiting the two decisive calculations (bosonic Lorentz algebra and superstring supercharge closure) that will determine whether the program succeeds.
