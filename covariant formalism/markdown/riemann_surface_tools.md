# `riemann_surface_tools.py`: contents and cross-checks

Companion notes for [`python/riemann_surface_tools.py`](../python/riemann_surface_tools.py).
This module sits on top of `ell_to_tau.py` and packages the analytic pieces
needed by the bc-correlator work: A-normalized holomorphic one-forms,
radial antiderivatives / Abel-Jacobi maps on the disc frame, Riemann theta
functions with characteristics, the prime form built from an odd
characteristic, the Riemann class vector `Delta`, and the normalized
`sigma`-function ratios that appear in the Verlinde-Verlinde formula, as well
as the general bc-system correlator itself. The sign/normalization
conventions follow the Strebel note, with the concrete implementation choices
recorded explicitly below. The file now also contains the large-`L`
renormalization machinery for the scheme-dependent determinant factor
$\left(\det A'\right)^{-1/2}$, together with the canonical convention

$$
\lvert Z_1\rvert^2
= \bigl[\det \mathrm{Im}(\Omega)\bigr]^{1/2} \left(\det A'\right)^{-1/2}.
$$

after renormalization, a naive chiral choice
$Z_1 = +\sqrt{\lvert Z_1\rvert^2}$, and the higher-genus
sigma-normalization machinery induced by the special
$\lambda = 1$, $(n,m) = (g,1)$ equation.

---

## 1. What is in `riemann_surface_tools.py`

### Data container

`RiemannSurfaceData` (a frozen dataclass) bundles, for a single choice of
ribbon graph + edge lengths:

| field | meaning |
| --- | --- |
| `genus` | $g$. |
| `Omega` | normalized period matrix ($g \times g$); scalar $\tau$ on genus 1 via the `.tau` property. |
| `normalized_forms` | the A-normalized holomorphic one-forms in the disc frame (tuple of callables). |
| `antiderivatives` | the radial antiderivatives $F_I(z) = \int_0^z \omega_I$ used for the Abel-Jacobi map. |
| `A_periods`, `B_periods` | the raw A- and B-periods of the forms passed to `normalize_holomorphic_forms`. **Note:** these are the pre-normalization periods returned by `ell_to_tau.period_matrix`, **not** $I$ and $\Omega$. The `Omega` field and everything derived from `antiderivatives` / `normalized_forms` is the A-normalized object. |
| `basis_pairs` | the chosen symplectic $\{\alpha,\beta\}$ cycles, each stored as edge-chord decompositions `[(edge_idx, coeff), ...]`. |
| `edge_midpoints` | the disc-frame representatives `(z0, z1)` for each edge-chord, used to integrate forms along cycles. |

Two additional frozen dataclasses organize the new $\lvert Z_1\rvert^2$
pipeline:

| dataclass | fields | meaning |
| --- | --- | --- |
| `LargeLFitResult` | `c, gamma, alpha, r2, max_abs_log_residual, total_lengths, log_values, edge_length_sets` | output of the large-$L$ fit $\log Z = c + \gamma L + \alpha \log L$; its property `.finite_part` is $\exp(c)$ |
| `RenormalizedZ1Data` | `abs_z1_sq, normalization_factor, renormalized_det_factor, fit, surface` | end-to-end output of a $\lvert Z_1\rvert^2$ estimator; in the canonical convention `normalization_factor = 1` |

### Constructor

`build_surface_data(...)`:
- Genus 1: pass `L, l1, l2` (the theta graph). Builds the F=1 ribbon graph,
  `ell_list = [l1, l2, L//2 - l1 - l2]`, the improved one-form via
  `elt.make_cyl_eqn_improved`, and calls `elt.normalize_holomorphic_forms`
  to package the rest.
- Higher genus: pass `forms=[omega_1, ..., omega_g]` together with the
  `ribbon_graph` and `ell_list` used to build them (the forms from
  `elt.make_cyl_eqn_improved_higher_genus` already carry the right metadata
  so these extra keywords can be omitted). Optional `basis_pairs` or
  `custom_cycles` let you override the auto-detected symplectic basis.

### Abel-Jacobi map

- `abel_map(z, surface, basepoint=0.0j)`: returns the vector
  $\zeta(z) - \zeta(\text{basepoint})$, where each component is the straight-line
  radial integral of the A-normalized form from `basepoint` to `z`.
- `abel_difference(z, w, surface)`: returns $F(z) - F(w)$ component-wise.
  This is the quantity that appears inside theta factors in the
  Verlinde-Verlinde formula.

Implementation choice:
- In the notation of `Strebel.tex`, the unspecified basepoint $z_0$ is taken
  to be $0$ by default.
- The path from $z_0$ to $z$ is the straight radial segment in the disc
  frame. More generally, `abel_map(z, surface, basepoint=z_0)` returns
  $F(z) - F(z_0)$ using the same radial primitive.
- This is a concrete disc-frame convention rather than an extra mathematical
  statement. For the prime form and the bc-correlator formula, the basic
  building block is $\zeta(z) - \zeta(w)$, so the arbitrary basepoint cancels.

### Theta characteristics

- `theta_characteristics(g, parity=None, as_half_integers=False)`:
  enumerate all $4^g$ characteristics (or just even/odd, which partition
  them into $2^{g-1}(2^g \pm 1)$ subsets). Default output is binary bits
  `((a1,...,ag), (b1,...,bg))` in $\{0,1\}$, interpreted as
  $\varepsilon = a/2$ and $\delta = b/2$.
- `characteristic_parity(char)`: returns the Arf invariant
  $4\,\varepsilon \cdot \delta \pmod 2$; $1$ means odd.
- `odd_characteristic(g)`: returns the first odd characteristic of genus g
  (used as a convenient default for the prime form).
- `_coerce_characteristic`: internal normalization; accepts either `{0,1}`
  binary bits or $\{0, 1/2\}$ half-integer entries.

### Riemann theta

Convention:

$$
\theta[\varepsilon, \delta](y \mid \Omega)
= \sum_n \exp\!\Bigl(
  i\pi (n+\varepsilon)^T \Omega (n+\varepsilon)
  + 2 i\pi (n+\varepsilon)^T (y+\delta)
\Bigr).
$$

- `riemann_theta(y, Omega, characteristic=None, nmax=None, tol=1e-12)`:
  plain value. Uses a centered box truncation $n_i \in [-n_{\max}, n_{\max}]$.
- `riemann_theta_gradient`: y-gradient
  $\mathrm{grad}[I] = \sum_n (2 i\pi)(n+\varepsilon)_I \, (\mathrm{summand})_n$.
- `riemann_theta_with_gradient`: returns both.
- `theta_truncation(Omega, tol)`: chooses a default `nmax` from the
  smallest eigenvalue of $\mathrm{Im}(\Omega)$ so that the truncation error on the
  Gaussian tail is bounded by `tol`.

### Prime form

`prime_form(z, w, surface, characteristic=None, nmax=None, tol=1e-12)`:

$$
E(z,w)
= \frac{\theta[\delta]\!\bigl(\zeta(z) - \zeta(w) \mid \Omega\bigr)}
{\sqrt{\omega[\delta](z)\,\omega[\delta](w)}}.
$$

with $\omega[\delta](z) = \omega_I(z)\,\partial_{y_I}\theta[\delta](0 \mid \Omega)$ for an
odd characteristic `delta`. The square root uses the principal complex
branch, so the absolute value and local limits are unambiguous while the
overall sign depends on branch choices; compare ratios or local limits.

Implementation choices:
- If `characteristic=None`, the code uses `odd_characteristic(g)`, i.e. the
  first odd characteristic in the enumeration order returned by
  `theta_characteristics(g, parity="odd")`.
- The denominator uses the principal complex square root for both
  $\sqrt{\omega[\delta](z)}$ and $\sqrt{\omega[\delta](w)}$.
- In genus 1, with the current branch choice and characteristic `((1),(1))`,
  the implemented local limit is

  $$
  E(z,w) \sim z - w
  $$

  in the disc coordinate. Equivalently,

  $$
  \frac{E(z, z+\varepsilon)}{\varepsilon} \to -1,
  \qquad
  \frac{E(z, z+\varepsilon)}{z-(z+\varepsilon)} \to 1.
  $$

- For the genus-1 checks below, the corresponding flat-coordinate formula is

  $$
  E_u(z,w) = - \frac{\theta_1(\pi u \mid \tau)}{\pi \theta_1'(0 \mid \tau)},
  \qquad
  u = F(z) - F(w).
  $$

### Riemann Class Vector

`riemann_constant_vector(surface, quad_limit=200)` computes the vector
`Delta` appearing in `Strebel.tex`:

$$
\Delta_I
= \frac{1}{2}\bigl(1 - \Omega_{II}\bigr)
+ \sum_{J \ne I} \int_{\alpha_J} \omega_J(z)\,\zeta_I(z)\,dz.
$$

The implementation is literal:

1. Start from the diagonal term $\frac{1}{2}(1 - \Omega_{II})$.
2. For each $I$ and each $J \ne I$, read the chosen $\alpha_J$ cycle from
   `surface.basis_pairs[J]["alpha"]`.
3. Each cycle is stored as an edge-chord decomposition
   `[(edge_idx, coeff), ...]`.
4. For every edge chord:
   - read its disc-frame endpoints `(z0, z1)` from `surface.edge_midpoints`
   - define the integrand $\omega_J(z)\,\zeta_I(z)$, with $\omega_J(z)$
     evaluated by `_evaluate_one_form(...)` and $\zeta_I(z)$ supplied by
     `surface.antiderivatives[I](z)`

   - integrate that function along the straight segment from `z0` to `z1`
     using `_segment_integral`
   - multiply by the cycle coefficient `coeff`
5. Sum over the segments and then over $J \ne I$.

Implementation choices:
- The same Abel-map convention is used here as everywhere else in the file:
  $\zeta_I(z)$ means the radial primitive from basepoint $0$.
- The cycle integral is done in the disc frame along the same straight
  boundary-chord representatives that were already used to compute the
  period matrix.
- `_segment_integral` uses `scipy.integrate.quad` separately on the real and
  imaginary parts of $f(z(t))\,z'(t)$ for the affine path

  $$
  z(t) = z_0 + t(z_1 - z_0), \qquad t \in [0,1].
  $$

Genus-1 simplification:
- When $g = 1$, there is no $J \ne I$ term, so the formula reduces to

  $$
  \Delta = \frac{1}{2}(1 - \tau),
  $$

  which is one of the unit tests below.

> **Warning — `riemann_constant_vector` is buggy at $g \ge 2$ and must not
> be used as-is.** Until this formula is fixed, use
> `riemann_constant_vector_canonical(...)` (below) for any $g \ge 2$ work.
> The legacy helper is kept only for genus-1 diagnostics and historical
> comparison. The downstream public helpers in `riemann_surface_tools.py`
> now default to the canonical algorithm at $g > 1$ while preserving the
> historical genus-1 half-period convention.

Two sharp diagnostics that `riemann_constant_vector(...)` is wrong at
$g = 2$:

- **Riemann vanishing fails.** The Riemann theorem requires
  $\theta(\zeta(p) - \Delta \mid \Omega) = 0$ at every point $p \in X$.
  On stored genus-2 topology 1 with `ell_list = [100] * 9`, the returned
  $\Delta = (0.0740 - 0.5158 i,\; 0.0740 - 0.5158 i)$ gives

  | point $p$        | $\lvert \theta(\zeta(p) - \Delta)\rvert$ |
  | ---              | ---                                       |
  | $0.08 + 0.12 i$  | $2.08 \times 10^{1}$                      |
  | $-0.14 + 0.16 i$ | $1.69 \times 10^{1}$                      |
  | $0.19 - 0.09 i$  | $4.08 \times 10^{1}$                      |
  | $0.21 + 0.09 i$  | $2.54 \times 10^{1}$                      |
  | $-0.16 + 0.12 i$ | $1.82 \times 10^{1}$                      |

  Generic $\lvert\theta(y)\rvert$ on this $\Omega$ is of order $\sim 3$,
  so these values are strictly larger than the generic scale, not merely
  not small.

- **`sigma_ratio` is divisor-dependent.** On the same topology with
  `ell_list = [300] * 9`, taking $z = 0.08 + 0.12 i$,
  $w = -0.19 + 0.13 i$:

  | divisor                               | $\sigma(z) / \sigma(w)$    | $\lvert\sigma(z)/\sigma(w)\rvert$ |
  | ---                                   | ---                        | --- |
  | $[0.21 + 0.09 i,\; -0.16 + 0.12 i]$   | $-0.2334 + 0.2413 i$       | $0.336$ |
  | $[0.17 + 0.05 i,\; -0.12 + 0.11 i]$   | $-0.8793 + 0.4752 i$       | $0.999$ |
  | $[0.22 + 0.08 i,\; -0.09 + 0.16 i]$   | $-1.4393 - 0.1098 i$       | $1.443$ |

  The sigma-ratio is a well-defined invariant of the surface and should
  not depend on the auxiliary divisor choice; here it varies by a factor
  of ~4.3 across three divisors.

Under the replacement $\Delta \to$
`riemann_constant_vector_canonical(surface, nmax=6)` both diagnostics
recover. On the same `[300] * 9` surface the three divisors now give

    sigma_ratio = 0.76678220 - 0.14752560 i
                = 0.76678467 - 0.14752500 i
                = 0.76678450 - 0.14752343 i

agreeing to ~5 decimals (limited by `scipy.quad` on the singular-weighted
form); and Riemann vanishing holds at $\lvert\theta\rvert \sim 10^{-6}$.

Root cause (full writeup in [known_genus2_ghost_issues.md](known_genus2_ghost_issues.md) §6):

- The implemented cycle integral is a disc-chord integral, not the
  surface-loop integral Fay's formula requires. On the simply-connected
  disc interior $d(\omega_J\,\zeta_I\,dz) = \omega_I \wedge \omega_J = 0$,
  so the Fay cycle integral's disc-frame image vanishes by Stokes, while
  the code instead returns only the middle chord leg.
- The sign convention in the `Strebel.tex` formula corresponds to $-K$
  rather than $K$, so even with the correct surface integral it would
  produce a sign-flipped $\Delta$.

At $g = 1$ the buggy cycle-integral branch is empty
($J \ne I$ is empty), so $\Delta = (1 - \tau)/2$ is exact modulo the
period lattice. The function is therefore harmless at $g = 1$ and is
retained for the genus-1 unit tests and historical diagnostics only.

### Canonical Riemann Class Vector via Deconinck Algorithm 1

`riemann_constant_vector_canonical(surface, *, form_idx=0, zero_radius=0.99,
coeff_tol=1e-10, nmax=None, tol=1e-12, filter_points=None,
sign_convention="strebel")` computes $\Delta$ without any cycle integral,
following Deconinck, Patterson, Swierczewski (2015) Algorithm 1
([*Computing the Riemann Constant Vector*][deconinck2015]). This is the
recommended computation at $g \ge 2$ and gives the same result (mod the
period lattice) as the old formula at $g = 1$.

[deconinck2015]: https://depts.washington.edu/bdecon/papers/pdfs/rcv.pdf

#### Mathematical input: two classical characterisations of $\Delta$

- **Canonical divisor characterisation** (Deconinck Theorem 11). For any
  canonical divisor $C$ of degree $2g - 2$ on $X$,

  $$
  2\,\Delta \equiv -A(P_0, C) \pmod{\Lambda},
  $$

  where $A(P_0, C) = \sum_i \zeta(c_i)$ is the Abel map of $C$ with basepoint
  $P_0$ and $\Lambda = \mathbb{Z}^g + \Omega\mathbb{Z}^g$.

- **Riemann vanishing** (Riemann theorem, as stated in Deconinck Thm 13). For
  any effective divisor $D$ of degree $g - 1$,

  $$
  \theta\!\bigl(\Delta + A(P_0, D) \mid \Omega\bigr) = 0.
  $$

The first characterisation determines $\Delta$ modulo $\tfrac{1}{2}\Lambda$
(halving a mod-$\Lambda$ relation leaves a $\tfrac{1}{2}\Lambda$ ambiguity).
The second then pins down the unique half-lattice representative.

#### Algorithm

For a surface with genus $g$:

1. **Canonical divisor from the zeros of a holomorphic one-form.** Read
   the polynomial coefficients of the A-normalized improved form
   `surface.normalized_forms[form_idx]`. Its zeros inside the disc form a
   canonical divisor $C$:

       _canonical_divisor_zeros_from_form(form, zero_radius, coeff_tol)

   returns `np.roots(polynomial)` filtered to $|z| < \text{zero\_radius}$,
   after dropping trailing coefficients with $|c_n| < \text{coeff\_tol}$.
   A correctly built A-normalized form at genus $g$ has exactly $2g - 2$
   such interior zeros; any other count raises `ValueError`.

2. **Abel-map of the canonical divisor.**

   $$
   A(P_0, C) = \sum_{c \in C} \zeta(c).
   $$

   Each $\zeta(c) = F(c)$ is the module's radial Abel primitive with
   basepoint $P_0 = 0$, i.e. `abel_map(c, surface)`.

3. **Base half-value.**

   $$
   \Delta_{\mathrm{base}} = -\tfrac{1}{2}\,A(P_0, C).
   $$

4. **Half-lattice disambiguation.** Enumerate the $2^{2g}$ half-lattice
   candidates

   $$
   h_{a,b} = a + \Omega\,b,
   \qquad
   a, b \in \{0, \tfrac{1}{2}\}^g.
   $$

   For each candidate, form $\Delta_{\mathrm{cand}} = \Delta_{\mathrm{base}} + h$ and
   compute the multi-divisor filter score

   $$
   s(h) = \sum_{k} \bigl|\theta\!\bigl(\Delta_{\mathrm{cand}} +
           (g - 1)\,\zeta(p_k) \mid \Omega\bigr)\bigr|^2,
   $$

   where the $p_k$ are the `filter_points`. At $g = 2$ the argument
   $(g - 1)\zeta(p_k) = \zeta(p_k)$ is the Abel image of the single-point
   effective divisor $D = p_k$. For $g > 2$ it is the image of
   $D = (g - 1)\,p_k$, a valid effective divisor of degree $g - 1$.

   Return the candidate with the smallest $s(h)$. In practice exactly one
   candidate gives $s(h) \sim 10^{-12}$ while the rest are $\gtrsim 1$, so
   the filter is effectively a sharp selection.

5. **Sign convention.** The returned value is $-K$ by default
   (`sign_convention="strebel"`), which is the value $\Delta$ such that
   $\theta(\zeta(p) - \Delta)$ vanishes for every $p$. This matches the
   $-\Delta$ that appears inside `sigma_ratio`, `bc_correlator_geometric_factor`,
   and the other downstream helpers in this module. Passing
   `sign_convention="deconinck"` instead returns the unnegated $K$ in the
   Mumford/Fay/Deconinck convention, which satisfies $\theta(\zeta(p) + K) = 0$.

#### Default `filter_points`

The default 5 disc-interior points are

    (0.11 + 0.09 i,
     -0.08 + 0.17 i,
      0.22 - 0.05 i,
      0.19 + 0.23 i,
      0.03 - 0.24 i).

These are deliberately away from the boundary prevertices at $|z| = 1$, and
cover roughly symmetric phases so the filter is not degenerate for any
common surface symmetry. The caller may override by passing `filter_points`
explicitly; any $\ge 1$ distinct disc-interior points suffice to pin down
$\Delta$ generically.

#### What this avoids

- **No cycle integrals.** The disc-chord-integral issue of
  `riemann_constant_vector` is bypassed entirely. Stokes on
  $\omega_J \zeta_I\,dz$ inside the disc is therefore irrelevant.
- **No `Strebel.tex` sign assumption.** The algorithm reads $\Delta$ off
  the canonical divisor and the Riemann vanishing theorem directly, so no
  sign convention in a quoted formula is trusted.
- **No fundamental-polygon assumption.** The derivation does not depend on
  the fundamental domain being a $4g$-gon; the ribbon-graph 18-gon is
  fine.

#### Genus-1 behaviour

At $g = 1$ the canonical divisor has degree $2g - 2 = 0$, so
`_canonical_divisor_zeros_from_form` returns an empty array and
$A(P_0, C) = 0$. The algorithm reduces to

$$
\Delta = -\bigl(a + \tau\,b\bigr)
\quad\text{with } a, b \in \{0, \tfrac{1}{2}\}
\text{ chosen to minimise }\lvert\theta(\Delta \mid \tau)\rvert,
$$

which reproduces $(1 - \tau)/2$ modulo the period lattice. On the
`(L, l1, l2) = (20, 3, 4)` surface this gives numerically
$\Delta_{\mathrm{canonical}} - \Delta_{\mathrm{old}} = (-1, 0)$, an integer
lattice shift, so the genus-1 $\sigma$-value on
`(z, w_0) = (0.31 - 0.12i, -0.17 + 0.14i)` is the same
`0.431008760111272 - 0.224918148464132 i` under either convention.

### Sigma Function

The sigma machinery now has two layers.

The first layer computes relative sigma data:

- `sigma_ratio(z, w, surface, divisor_points=..., ...) = sigma(z) / sigma(w)`
- `sigma_value(z, ..., normalization_point=y0, normalization_value=c)`
  by imposing $\sigma(y_0) = c$

This is the raw ratio-based layer that is always available.

#### Input Data

`sigma_ratio` requires:
- `surface`: the A-normalized `RiemannSurfaceData`
- `divisor_points=[z_1, ..., z_g]`: exactly $g$ generic points
- optionally a precomputed `Delta`

The divisor points must be generic in the usual sense:
- they should not coincide with one another
- they should avoid the evaluation points `z`, `w`
- they should avoid choices that make the theta factor vanish

If these genericity conditions fail, the code raises `ZeroDivisionError`.

#### Formula Used

From the `lambda = 1`, `(n,m)=(g,1)` formula in `Strebel.tex`,

$$
Z_1 \det\!\bigl(\omega_I(z_i)\bigr)
=
Z_1^{-1/2}\,
\theta\!\bigl(\sum_i \zeta(z_i) - \zeta(w) - \Delta \mid \Omega\bigr)\,
\frac{\prod_{i<i'} E(z_i,z_{i'}) \prod_i \sigma(z_i)}
{\prod_i E(z_i,w)\,\sigma(w)}.
$$

Now fix a generic divisor $z_1,\ldots,z_g$ once and for all and compare the same
formula written for $w = z$ and for $w = w_{\mathrm{ref}}$. All factors depending only on the
fixed divisor cancel. This gives

$$
\frac{\sigma(z)}{\sigma(w_{\mathrm{ref}})}
=
\frac{\theta\!\bigl(S - \zeta(z) \mid \Omega\bigr)}
{\theta\!\bigl(S - \zeta(w_{\mathrm{ref}}) \mid \Omega\bigr)}
\frac{\prod_i E(z_i, w_{\mathrm{ref}})}
{\prod_i E(z_i, z)},
$$

where

$$
S = \sum_i \zeta(z_i) - \Delta.
$$

That is exactly what `sigma_ratio` implements.

#### Algorithm Implemented By `sigma_ratio`

Given `z`, `w`, `surface`, and `divisor_points=[z_1,...,z_g]`:

1. Check that the number of divisor points is exactly the surface genus $g$.
2. If `Delta` is not supplied, compute it with `riemann_constant_vector`.
3. Compute

   $$
   \zeta_{\mathrm{sum}} = \sum_i \zeta(z_i).
   $$

4. Form the theta arguments

   $$
   a_z = \zeta_{\mathrm{sum}} - \zeta(z) - \Delta,
   \qquad
   a_w = \zeta_{\mathrm{sum}} - \zeta(w) - \Delta.
   $$

5. Evaluate the theta factors

   $$
   \theta_z = \theta(a_z \mid \Omega),
   \qquad
   \theta_w = \theta(a_w \mid \Omega).
   $$

6. Build the prime-form products

   $$
   P_z = \prod_i E(z_i, z),
   \qquad
   P_w = \prod_i E(z_i, w).
   $$

7. Return

   $$
   \frac{\theta_z}{\theta_w}\frac{P_w}{P_z}.
   $$

The implementation computes exactly this expression, with the code variables
`arg_z`, `arg_w`, `theta_z`, `theta_w`, `prime_prod_z`, and `prime_prod_w`
corresponding to the mathematical quantities $a_z$, $a_w$, $\theta_z$,
$\theta_w$, $P_z$, and $P_w$.

#### Algorithm Implemented By `sigma_value`

`sigma_value(z, ..., normalization_point=y0, normalization_value=c)` is just

$$
\sigma(z) = c \,\sigma_{\mathrm{ratio}}(z, y_0, \ldots).
$$

So the entire normalization freedom of `sigma` is encoded in the pair
`(y0, c)`.

Implementation choices:
- The normalization is external and user-chosen; the module does not try to
  infer a preferred absolute normalization from geometry.
- The same divisor is used in numerator and denominator, which is why the
  divisor dependence cancels from the resulting normalized sigma.
- The verified invariant statement is divisor-independence of the normalized
  sigma ratio, not any stronger closed-form identification such as
  $\sigma \sim \sqrt{f}$ in genus 1.

#### Canonical Higher-Genus Sigma From Chiral `Z_1`

Once a chiral $Z_1$ has been chosen, the special $\lambda = 1$,
$(n,m) = (g,1)$ equation can be used to fix the overall sigma constant for
genus $g > 1$.

The code now provides:

- `canonical_chiral_z1(abs_z1_sq)`:

  $$
  Z_1 = +\sqrt{\lvert Z_1\rvert^2}
  $$

  i.e. the naive positive-real square root of the canonically normalized
  $\lvert Z_1\rvert^2$
- `sigma_scale_from_z1(anchor_b_points, anchor_c_point, surface, ..., z1=...)`
- `canonical_sigma_value(z, surface, ..., z1=...)`

Here the idea is:

1. Start from the current user-normalized sigma

   $$
   \tilde{\sigma}(z)
   $$

   obtained from `sigma_value(z, ...)`

2. Assume the canonically normalized sigma is

   $$
   \sigma(z) = C\,\tilde{\sigma}(z)
   $$

3. For fixed anchor data `z_i`, `w`, compute the special geometric factor
   `A_tilde` from `lambda_one_geometric_z1_factor(...)`, which satisfies

   $$
   \widetilde{A} = Z_1^{3/2} / C^{g-1}
   $$

   because the special equation contains `g` factors of `sigma(z_i)` and one
   factor of `sigma(w)` in the denominator
4. Therefore

   $$
   C^{g-1} = Z_1^{3/2} / \widetilde{A}
   $$

5. The code chooses the principal `(g-1)`-st root of that complex number

This is exactly what `sigma_scale_from_z1(...)` returns, and
`canonical_sigma_value(...)` multiplies the old `sigma_value(...)` by that
scale factor.

Genus-1 caveat:
- for $g = 1$, the overall sigma constant cancels out of the special equation
- so this procedure cannot fix sigma normalization on the torus
- `sigma_scale_from_z1(...)` therefore raises `ValueError` at genus 1

So the current state of the code is:
- genus 1: sigma ratios plus an externally imposed normalization
- genus $g > 1$: sigma ratios plus a canonical overall normalization derived
  from the chosen chiral $Z_1$

### Renormalized $\left(\det A'\right)^{-1/2}$ and $\lvert Z_1\rvert^2$

The last section of `Strebel.tex` expresses the Weyl-frame dependent chiral
boson quantity through

$$
\lvert Z_1\rvert^2
= \mathcal N_1 \bigl[\det \mathrm{Im}(\Omega)\bigr]^{1/2}
\left(\det A'\right)^{-1/2}.
$$

The determinant factor is scheme-dependent, so the code does **not** use the
raw finite-lattice value directly. Instead it follows the same large-`L`
renormalization prescription used earlier in the notes for the bosonic
partition function.

Preferred convention in the current file:

$$
\mathcal N_1 = 1,
$$

so after renormalization the determinant formula itself defines $\lvert Z_1\rvert^2$.

#### Fit Object

`_fit_large_l_behavior(total_lengths, log_values, edge_length_sets=None)`
fits

$$
\log Z(L) = c + \gamma L + \alpha \log L
$$

by linear least squares on the design matrix

$$
[1, L, \log L].
$$

It returns:
- `c`: the finite part
- `gamma`: coefficient of the linear divergence
- `alpha`: coefficient of the logarithmic divergence
- `r2`, `max_abs_log_residual`: fit-quality diagnostics
- the sorted `L` samples and the corresponding logged values

The renormalized quantity is always taken to be

$$
\exp(c).
$$

#### Determinant Renormalization

`fit_renormalized_aprime_factor(ribbon_graph, base_edge_lengths, scales=..., min_edge_length=200)`
implements the determinant fit for fixed moduli.

Input convention:
- `base_edge_lengths = (ell_1^(0), ..., ell_E^(0))` fixes the moduli point
  through the edge-length ratios
- `scales = (s_1, ..., s_n)` generates the actual fitting data by

  $$
  \ell_a = s\,\ell_a^{(0)}
  $$

- the total lattice size used in the fit is

  $$
  L = 2 \sum_a \ell_a
  $$

- every sampled edge length must satisfy

  $$
  \ell_a \ge \ell_{\min}
  $$

  where $\ell_{\min} = 200$ by default, exactly to avoid the small-`l`
  artifacts seen elsewhere in the project

Algorithm:

1. For each `scale`:
   - build `edge_lengths = scale * base_edge_lengths`
   - construct `A'` using `partition_function.traced_matter_matrix_f1`
   - symmetrize it as `0.5 * (A' + A'^T)`
   - compute

     $$
     \log Z_{\det} = -\frac{1}{2}\log \det(A')
     $$

     via `partition_function.logdet_cholesky`
   - record the pair `(L, log Z_det)`
2. Fit all samples to

   $$
   \log Z_{\det}(L) = c(\Omega) + \gamma L + \alpha \log L
   $$

3. Return $\exp(c(\Omega))$ as the renormalized determinant factor feeding into
   $\lvert Z_1\rvert^2$

So in this module the phrase "renormalized determinant factor" always means

$$
\exp(c(\Omega)),
$$

not the raw finite-lattice value $\left(\det A'\right)^{-1/2}$.

#### Surface Builder for the Large-`L` Regime

`build_surface_from_ribbon_graph(ribbon_graph, edge_lengths, ...)` is a small
convenience wrapper around `elt.make_cyl_eqn_improved_higher_genus(...)`
followed by `build_surface_data(...)`, so the same fixed-moduli large surface
can be used both for the period-matrix
data and for the $\lvert Z_1\rvert^2$ extraction.

#### Canonical `|Z_1|^2` and Chiral `Z_1`

The preferred path now is:

1. Fit the renormalized determinant finite part `exp(c(Omega))`
2. Define

   $$
   \lvert Z_1\rvert^2
   = \bigl[\det \mathrm{Im}(\Omega)\bigr]^{1/2} \exp(c(\Omega))
   $$

3. Choose the naive chiral representative

   $$
   Z_1 = +\sqrt{\lvert Z_1\rvert^2}
   $$

The corresponding helpers are:

- `abs_z1_sq_from_renormalized_det(surface, renormalized_det_factor, normalization_factor=1)`
- `canonical_abs_z1_sq(surface, renormalized_det_factor=...)`
- `estimate_canonical_abs_z1_sq(...)`
- `canonical_chiral_z1(abs_z1_sq)`

So the canonical public convention in the current file is the one with
$\mathcal N_1 = 1$, not the older extracted-$\mathcal N_1$ scheme.

#### `|Z_1|^2` From the Last Strebel Equation as a Diagnostic

For `lambda = 1` and `(n,m) = (g,1)`, the last equation of `Strebel.tex` is

$$
Z_1 \det\!\bigl(\omega_I(z_i)\bigr)
=
Z_1^{-1/2}\,
\theta\!\bigl(\sum_i \zeta(z_i) - \zeta(w) - \Delta \mid \Omega\bigr)\,
\frac{\prod_{i<i'} E(z_i, z_{i'}) \prod_i \sigma(z_i)}
{\prod_i E(z_i, w)\,\sigma(w)}.
$$

Move the determinant to the other side and define

$$
A
=
\frac{
\theta\!\bigl(\sum_i \zeta(z_i) - \zeta(w) - \Delta \mid \Omega\bigr)
\prod_{i<i'} E(z_i, z_{i'}) \prod_i \sigma(z_i)
}{
\det\!\bigl(\omega_I(z_i)\bigr)\,\prod_i E(z_i, w)\,\sigma(w)
}.
$$

Then

$$
A = Z_1^{3/2}.
$$

The helper `lambda_one_geometric_z1_factor(...)` computes exactly this complex
number `A`, using:
- `b_points = [z_1, ..., z_g]`
- `c_point = w`
- the already-implemented `abel_map`, `riemann_theta`, `prime_form`, and
  `sigma_value`

Because the phase of `Z_1` is convention-dependent and not needed for that
older diagnostic, the module then extracts the physically relevant modulus by

$$
\lvert Z_1\rvert^2 = \lvert A\rvert^{4/3}.
$$

This is implemented by `abs_z1_sq_from_lambda_one(...)`.

#### Final Assembly

`abs_z1_sq_from_renormalized_det(surface, normalization_factor, renormalized_det_factor)`
computes

$$
\lvert Z_1\rvert^2
= \mathcal N_1 \bigl[\det \mathrm{Im}(\Omega)\bigr]^{1/2} \exp(c(\Omega)).
$$

The canonical high-level helper is now

`estimate_canonical_abs_z1_sq(...)`

which does:

1. fit the large-`L` determinant data
2. build a large reference surface
3. evaluate the canonical convention
   $\lvert Z_1\rvert^2 = [\det \mathrm{Im}(\Omega)]^{1/2} \exp(c)$

The older helper

`estimate_abs_z1_sq(...)`

is still present as a diagnostic path that compares the determinant-based
definition against the special `lambda=1` equation.

### General Verlinde-Verlinde bc Correlator

The module now also exposes the full holomorphic correlator formula quoted in
the last section of `Strebel.tex`.

#### Ghost-Number Selection Rule

Before evaluating the correlator, the code enforces the standard bc
selection rule

$$
n_c - n_b = (1 - 2\lambda)(g - 1),
$$

equivalently

$$
n_b - n_c = (2\lambda - 1)(g - 1).
$$

This is implemented by the internal helper

`_ghost_number_selection_rule(lambda_weight, genus, n_b=..., n_c=...)`

which raises `ValueError` if the requested insertion numbers are inconsistent
with the anomaly.

#### Geometric Factor

`bc_correlator_geometric_factor(b_points, c_points, surface, lambda_weight=..., ...)`
returns exactly the geometric part of the Verlinde-Verlinde formula:

$$
\frac{
\theta\!\Bigl(\sum_i \zeta(z_i) - \sum_j \zeta(w_j) - (2\lambda - 1)\Delta \,\Big|\, \Omega\Bigr)
\Bigl[\prod_{i<i'} E(z_i, z_{i'})\Bigr]
\Bigl[\prod_{j<j'} E(w_j, w_{j'})\Bigr]
\Bigl[\prod_i \sigma(z_i)^{2\lambda - 1}\Bigr]
}{
\Bigl[\prod_{i,j} E(z_i, w_j)\Bigr]
\Bigl[\prod_j \sigma(w_j)^{2\lambda - 1}\Bigr]
}.
$$

In other words, this function returns

$$
Z_1^{1/2} \left\langle \prod_i b_\lambda(z_i)\prod_j c_{1-\lambda}(w_j) \right\rangle,
$$

namely the full holomorphic correlator with the chiral prefactor
`Z_1^{-1/2}` stripped off.

Algorithm:

1. Check the ghost-number selection rule.
2. Compute `Delta` if it was not supplied.
3. Form

   $$
   \theta_{\mathrm{arg}}
   = \sum_i \zeta(z_i) - \sum_j \zeta(w_j) - (2\lambda - 1)\Delta.
   $$

4. Evaluate the theta factor at `theta_arg`.
5. Build the three prime-form products:

   $$
   P_{bb} = \prod_{i<i'} E(z_i, z_{i'}),
   \qquad
   P_{cc} = \prod_{j<j'} E(w_j, w_{j'}),
   \qquad
   P_{bc} = \prod_{i,j} E(z_i, w_j).
   $$

6. Build the sigma products:

   $$
   \sigma_b = \prod_i \sigma(z_i)^{2\lambda - 1},
   \qquad
   \sigma_c = \prod_j \sigma(w_j)^{2\lambda - 1},
   $$

   using the same normalized `sigma_value(...)` convention as the rest of the
   module.
7. Return

   $$
   \theta \, P_{bb} P_{cc}\,\sigma_b \big/ \left(P_{bc}\,\sigma_c\right).
   $$

So this function is completely determined by the already-implemented Abel map,
theta function, prime form, and sigma machinery.

#### Full Holomorphic Correlator

`bc_correlator(...)` is a thin wrapper around the geometric-factor function.

- If `z1` is **not** supplied, it returns the same geometric factor described
  above.
- If `z1` **is** supplied, it returns

  $Z_1^{-1/2}$ times the geometric factor

  using the principal complex square root for `sqrt(z1)`.

This split is intentional:
- the geometric factor is robustly defined from the surface data alone
- the chiral quantity `Z_1` itself is convention-dependent, and at present the
  project only controls `|Z_1|^2` more reliably than a fully fixed holomorphic
  branch of `Z_1`

### Choice Summary

These are the main implementation choices that are fixed in code whenever
`Strebel.tex` leaves them implicit:

| quantity | implementation choice |
| --- | --- |
| Abel-map basepoint `z_0` | default `z_0 = 0` |
| Abel-map path | straight radial segment in the disc frame |
| holomorphic one-forms | A-normalized using `ell_to_tau.normalize_holomorphic_forms` |
| default odd characteristic | first odd characteristic in enumeration order |
| prime-form square-root branch | principal complex branch |
| Riemann class `Delta` (recommended, $g \ge 1$) | `riemann_constant_vector_canonical(...)`: Deconinck Algorithm 1 (canonical divisor from zeros of `normalized_forms[0]`, $2^{2g}$ half-lattice search, filter points are 5 fixed disc-interior points) |
| Riemann class `Delta` (legacy, $g = 1$ only) | `riemann_constant_vector(...)`: direct $\frac{1}{2}(1 - \Omega_{II})$ + disc-chord cycle integrals; fails Riemann vanishing at $g \ge 2$ |
| canonical $\Delta$ sign convention | `sign_convention="strebel"` (default): returns $\Delta$ with $\theta(\zeta(p) - \Delta) = 0$, matching the `-Delta` used in `sigma_ratio` and `bc_correlator_geometric_factor` |
| raw `sigma` normalization | user-specified by fixing $\sigma(z_0) = c$ at a chosen normalization point |
| canonical higher-genus `sigma` normalization | derived from the chosen chiral $Z_1 = +\sqrt{\lvert Z_1\rvert^2}$ via the $\lambda = 1$, $(n,m) = (g,1)$ equation |
| `sigma` divisor | any generic divisor of length $g$; normalized result is divisor-independent |
| bc selection rule | enforced as $n_c - n_b = (1 - 2\lambda)(g - 1)$ |
| default correlator output | the Verlinde-Verlinde geometric factor with `Z_1^{-1/2}` stripped off unless `z1` is explicitly supplied |
| determinant renormalization | fit $-\frac{1}{2}\log \det A' = c + \gamma L + \alpha \log L$ and keep $\exp(c)$ |
| canonical `|Z_1|^2` convention | $\mathcal N_1 = 1$, so $\lvert Z_1\rvert^2 = [\det \mathrm{Im}(\Omega)]^{1/2} \exp(c)$ |
| chiral `Z_1` convention | naive positive square root $Z_1 = +\sqrt{\lvert Z_1\rvert^2}$ |
| genus-1 local limit | $E(z,w) \sim z-w$ in the disc coordinate |
| genus-1 flat-coordinate formula used in checks | $E_u = -\theta_1(\pi u\mid\tau)/(\pi \theta_1'(0\mid\tau))$ |

### Internal helpers

- `_evaluate_one_form(f, z)`: collapses the `(singular, polynomial)` tuple
  returned by the improved forms back into a single value
  `singular * polynomial`.
- `_make_radial_antiderivative(f)`: for `ell_to_tau`-style forms that carry
  `.coeffs`, uses the exact polynomial antiderivative; otherwise falls
  back to `scipy.integrate.quad` along the ray `t*p, t in [0,1]`, with a
  per-endpoint cache.
- `_segment_integral(func, z0, z1)`: numerically integrates `func(z) dz`
  along the straight segment from `z0` to `z1`, again by splitting into real
  and imaginary parts and using `quad`.

---

## 2. Cross-checks performed

All checks below avoid the small-`l` regime. For the dedicated large-`L`
renormalization fit we use the weaker but explicit cutoff

$$
\ell_a \ge 200,
$$

while the heavier genus-2 geometry diagnostics use larger values such as
`50, 100, 150, 500` per edge as recorded below. Reproducibility: unless noted otherwise, run
from `covariant formalism/python` with Python 3 and NumPy / SciPy / mpmath.
For genus 2 the calculations use ~50 s wall clock on a modern laptop
because of the `9000 x 4500` complex SVD inside
`make_cyl_eqn_improved_higher_genus`.

### 2.1 Pre-existing unit tests (`test_riemann_surface_tools.py`)

Run:

```bash
python3 -m unittest test_riemann_surface_tools -v
```

These sixteen tests pass in the current workspace:

1. **Synthetic large-`L` fit recovers exact coefficients**:
   feeding exact data of the form `c + gamma L + alpha log L` into
   `_fit_large_l_behavior` reproduces `c`, `gamma`, `alpha`, and `exp(c)`.

2. **Abel-Jacobi periods reproduce (1, tau)** at `(L, l1, l2) = (20, 3, 4)`:
   the alpha-period integral is `1.0` and the beta-period integral is
   `surface.tau` to 9 decimals.
3. **Genus-1 Riemann class** at `(L, l1, l2) = (20, 3, 4)`:
   `riemann_constant_vector(surface) = (1 - tau) / 2` to 10 decimals.
4. **Genus-1 characteristics vs. Jacobi `mp.jtheta`** at
   `tau = 0.37 + 0.91i`, `y = 0.23 + 0.07i`, `nmax = 8`:

   - `((0),(0))` -> `jtheta(3, pi y, q)`
   - `((1),(0))` -> `jtheta(2, pi y, q)`
   - `((0),(1))` -> `jtheta(4, pi y, q)`
   - `((1),(1))` -> `-jtheta(1, pi y, q)`

   to 12 decimals.
5. **Prime form local limit** `E(z, z+eps) / eps -> -1` for
   `z = 0.21 + 0.17i`, `eps = 1e-6 (1 + 0.4i)` on `(L, l1, l2) = (20, 3, 4)`
   with the odd characteristic `((1),(1))`. Equivalently,
   `E(z, z+eps) / (z-(z+eps)) -> 1`.
6. **Genus-1 normalized sigma ratio is divisor-independent** at
   `(L, l1, l2) = (20, 3, 4)`: with normalization point
   `w0 = -0.17 + 0.14i`, the values produced from divisor
   `[0.23 + 0.11i]` and divisor `[-0.09 + 0.27i]` agree to 9 decimals at
   `z = 0.31 - 0.12i`.
7. **Genus-1 sigma normalization is imposed exactly**:
   `sigma_value(w0, ..., normalization_point=w0)` returns `1` to 12 decimals
   with the default normalization value.
8. **Identified genus-1 boundary points differ by a period-lattice vector**
   at `(l1, l2, l3) = (500, 600, 700)`, `L = 3600`.
9. **Identified genus-2 boundary points differ by a period-lattice vector**
   on stored genus-2 topology `1` with `ell_list = [100] * 9`.
10. **Genus-2 theta constant cross-check** between `rst.riemann_theta` and
   `elt.riemann_theta_constant_genus2` on the handmade symmetric
   `Omega = [[0.9i, 0.11+0.07i],[0.11+0.07i, 1.2i]]` at
   characteristic `((1,0),(0,1))`, `nmax = 8`.
11. **Genus-1 `|Z_1|^2` pipeline is self-consistent**:
    `estimate_abs_z1_sq(...)`, `abs_z1_sq_from_lambda_one(...)`, and
    `abs_z1_sq_from_renormalized_det(...)` agree numerically on the same large
    torus surface.
12. **Canonical `|Z_1|^2` uses the `\mathcal N_1 = 1` convention**:
    `estimate_canonical_abs_z1_sq(...)` returns `normalization_factor = 1`
    and agrees with `abs_z1_sq_from_renormalized_det(...)`.
13. **Higher-genus sigma normalization cannot be fixed at genus 1**:
    `sigma_scale_from_z1(...)` raises `ValueError` on the torus, as it should.
14. **Genus-2 sigma scale from chiral `Z_1` satisfies the special equation**:
    the scale returned by `sigma_scale_from_z1(...)` reproduces
    `Z_1^{3/2}` in the `lambda=1`, `(n,m)=(g,1)` identity.
15. **bc selection rule is enforced**:
    asking for an inconsistent genus-1 correlator at `lambda = 2` raises
    `ValueError`.
16. **Genus-1 `lambda = 1` correlator matches the special last-equation
    helper**:
    `bc_correlator_geometric_factor([z],[w],...) / omega(z)` agrees with
    `lambda_one_geometric_z1_factor([z], w, ...)` to 10 decimals.

### 2.2 Extended low-genus diagnostics (ad-hoc, edge length >= 500)

These checks were run once in a throwaway diagnostic script; the code is
not committed. Each item includes the exact parameters so a future agent
can rebuild the same script and rerun.

#### (A) Genus-1 A-normalization via the antiderivative

Build `surface = rst.build_surface_data(L, l1, l2)` for each
`(l1, l2, l3)` below (`L = 2(l1+l2+l3)`). Compute the alpha- and beta-
period sums by summing `F(z1) - F(z0)` over the edge-chord decomposition
stored in `surface.basis_pairs[0]`.

| `(l1, l2, l3)` | `L` | alpha-period | tau | tol |
| --- | --- | --- | --- | --- |
| (500, 600, 700) | 3600 | `1.0000000+0.0000000i` | `0.5327928+0.8047088i` | < 1e-6 |
| (800, 700, 500) | 4000 | `1.0000000-0.0000000i` | `0.4271731+0.9413014i` | < 1e-6 |
| (600, 600, 600) | 3600 | `1.0000000-0.0000000i` | `0.5000000+0.8660254i` | < 1e-6 |

The equilateral case pins `tau = 1/2 + i sqrt(3)/2` to 7 decimals, which
is the analytic answer for the symmetric theta graph.

#### (B) Genus-1 prime form is a (-1/2, -1/2)-differential

In the A-normalized flat-u coordinate the analytic prime form is

$$
E_u(z,w) = - \frac{\theta_1(\pi u \mid \tau)}{\pi \theta_1'(0 \mid \tau)},
\qquad
u = F(z) - F(w).
$$

Pulled back to the disc frame the prime form picks up `1/sqrt(f(z) f(w))`.
Check:

$$
E_{\mathrm{disc}}(z,w)\sqrt{f(z)f(w)} = E_u(z,w)
$$

up to one overall sign (square-root branch). For each `(l1, l2, l3)`
listed below and each of the three test pairs

$$
(z,w) \in \{
(0.18+0.22i,\ 0.31-0.13i),\,
(0.05+0.4i,\ -0.22+0.09i),\,
(-0.35+0.05i,\ 0.12+0.28i)
\}
$$

(all strictly inside the unit disc, well away from the Strebel
prevertices) compute the ratio `(E_disc * sqrt(fz fw)) / E_u`. All three
points should give the same ratio, with modulus 1.

| `(l1, l2, l3)` | `L` | `|ratio|` | spread over 3 pairs | tol |
| --- | --- | --- | --- | --- |
| (500, 600, 700) | 3600 | `1.000000` | `6.03e-16` | < 1e-4 |
| (700, 600, 500) | 3600 | `1.000000` | `5.03e-16` | < 1e-4 |

The spread is effectively machine precision, far below the >= 500
Strebel-grid tolerance.

Characteristic used: `((1),(1))` (the unique odd genus-1 one). Theta
truncation `nmax=12`. The mpmath helpers used are

    q = exp(i pi tau)
    _jtheta1_pi(u, tau)          = mp.jtheta(1, pi*u, q)
    _jtheta1_pi_prime_at_0(tau)  = mp.jtheta(1, 0, q, derivative=1) * pi.

so the checked flat-coordinate formula is implemented numerically as
`E_u(z,w) = -_jtheta1_pi(u, tau) / _jtheta1_pi_prime_at_0(tau)`.

#### (C) Genus-1 prime form antisymmetry

On `surface = rst.build_surface_data(L=2*(500+600+700), l1=500, l2=600)`,
check `E(z,w) + E(w,z) == 0` for

$$
(z,w) = (0.21+0.17i, -0.13+0.28i),
\qquad
(z,w) = (-0.3+0.05i, 0.09-0.31i).
$$

Both pairs: `|E(z,w) + E(w,z)| / max(|E|) < 1e-9`. Characteristic
`((1),(1))`, `nmax=10`.

#### (D) Theta-characteristic counts

For `g = 1, 2, 3`, `rst.theta_characteristics(g, parity=...)` returns
`2^{g-1} (2^g + 1)` even, `2^{g-1} (2^g - 1)` odd, and `4^g` total:

| g | even | odd | total |
| --- | --- | --- | --- |
| 1 | 3 | 1 | 4 |
| 2 | 10 | 6 | 16 |
| 3 | 36 | 28 | 64 |

All match.

#### (E) Genus-1 sigma diagnostics

On `surface = rst.build_surface_data(L=20, l1=3, l2=4)`, take

$$
z = 0.31 - 0.12i,
\qquad
w_0 = -0.17 + 0.14i.
$$

Then:

1. Using

       sigma_value(z, surface,
                   divisor_points=[0.23 + 0.11i],
                   normalization_point=w0)

   and

       sigma_value(z, surface,
                   divisor_points=[-0.09 + 0.27i],
                   normalization_point=w0)

   gives the same value to 9 decimal places:

       0.431008760111272 - 0.224918148464132 i

   up to the displayed rounding.

2. Evaluating

       sigma_value(w0, surface,
                   divisor_points=[0.23 + 0.11i],
                   normalization_point=w0)

   returns `1.0 + 0.0i` to 12 decimal places, confirming that the imposed
   normalization is respected.

These are the right invariant checks for the current `sigma` algorithm.
During development we also checked that a stronger genus-1 guess like
`sigma(z) ~ sqrt(f(z))` is **not** correct as a raw identity in the present
disc-frame convention, so the code and tests intentionally avoid asserting
that.

#### (F) Genus-2 theta on a handmade symmetric Omega

Use `Omega = [[0.9i, 0.11+0.07i], [0.11+0.07i, 1.2i]]` (no discretization
noise). For every one of the 16 characteristics, compare

`rst.riemann_theta(zeros(2), Omega, characteristic=char, nmax=12)`

against `elt.riemann_theta_constant_genus2(Omega, char, nmax=12)`. Max
error across all 16 characteristics: `2.22e-16`.

#### (G) Genus-2 prime form is independent of odd spin structure

This is the most direct numerical check of the statement in `Strebel.tex`
that the prime form

$$
E(z,w)
= \frac{\theta[\delta]\!\bigl(\zeta(z)-\zeta(w)\mid\Omega\bigr)}
{\sqrt{\omega[\delta](z)\,\omega[\delta](w)}}
$$

is independent of the choice of odd characteristic `delta`.

Important counting note:
- Genus 2 has **6** odd spin structures, not 3.
- The code therefore checks all 6 odd characteristics returned by
  `rst.theta_characteristics(2, parity="odd")`:

  `((0,1),(0,1)), ((0,1),(1,1)), ((1,0),(1,0)), ((1,0),(1,1)), ((1,1),(0,1)), ((1,1),(1,0))`

Surface used:
- stored genus-2 topology `1` from `compact_partition.get_stored_genus2_graph(1)`
- ribbon graph rebuilt with the same `boundary -> succ -> rotation` helper used
  in `genus2_one_point.py`
- edge lengths `ell_list = [50] * 9`

Evaluation points:

$$
z = 0.12 + 0.08i,
\qquad
w = -0.07 + 0.18i.
$$

The resulting period matrix was

    Omega =
    [[ 0.5185934 +1.03169483i  -0.25929798-0.51584996i ]
     [-0.25929542-0.51584487i   0.5185934 +1.03169483i ]]

Using `nmax = 8`, the six prime-form values were:

| odd characteristic | `E(z,w)` | abs diff to first | rel diff to first |
| --- | --- | --- | --- |
| `((0,1),(0,1))` | `0.18926603325445587 - 0.09850407331554953i` | `0` | `0` |
| `((0,1),(1,1))` | `0.18926601801462850 - 0.09850401901288648i` | `5.64e-08` | `2.64e-07` |
| `((1,0),(1,0))` | `0.18926592183387750 - 0.09850410135546928i` | `1.15e-07` | `5.38e-07` |
| `((1,0),(1,1))` | `0.18926605187577059 - 0.09850407720448598i` | `1.90e-08` | `8.92e-08` |
| `((1,1),(0,1))` | `0.18926606502897625 - 0.09850409893785575i` | `4.08e-08` | `1.91e-07` |
| `((1,1),(1,0))` | `0.18926604219892180 - 0.09850405781066393i` | `1.79e-08` | `8.39e-08` |

Summary:

    max abs diff = 1.1489465777309607e-07
    max rel diff = 5.384884181063142e-07

So the odd-spin-structure dependence is absent to better than `10^{-6}`
relative accuracy on this genus-2 discretized surface.

Convergence check:
- Repeating the same calculation on the smaller surface `ell_list = [20] * 9`
  gave a worse max relative spread of about `2.85e-06`.
- The improvement from `2.85e-06` down to `5.38e-07` as the edge lengths grow
  is strong evidence that the residual mismatch is a discretization artifact
  of the approximate genus-2 surface construction, not a true spin-structure
  dependence of the prime-form formula.

#### (H) Abel-map jumps for identified boundary points lie on the period lattice

Another important consistency check is that if two boundary points are
identified in the disc construction, then their Abel-map difference should be
an integral period-lattice vector:

$$
\zeta(z_2) - \zeta(z_1) = m + \Omega n,
$$

for some integer vectors `m, n in Z^g`.

The code tests this edge by edge using the raw boundary chords.

Algorithm:

1. For each edge `e`, let `c_e` be the raw oriented boundary chord joining its
   two boundary occurrences.
2. Let the symplectic basis chosen by `period_matrix` be
   `{alpha_j, beta_j}`.
3. Express the expected period-lattice jump of `c_e` by its intersections with
   the basis cycles:

   $$
   m_j = c_e \cdot \beta_j,
   \qquad
   n_j = -\,c_e \cdot \alpha_j.
   $$

   Then the predicted Abel jump is

   $$
   \Delta \zeta_e(\mathrm{pred}) = m + \Omega n.
   $$

4. Compute the actual Abel jump from the two stored midpoint representatives
   `(z_1, z_2)` of edge `e`:

   $$
   \Delta \zeta_e(\mathrm{act}) = \zeta(z_2) - \zeta(z_1).
   $$

5. Compare `Delta zeta_e(act)` and `Delta zeta_e(pred)` for every edge and
   record the worst componentwise absolute error.

Genus-1 large-length check:
- Surface: theta graph with `(l1, l2, l3) = (500, 600, 700)`, `L = 3600`
- Result:

      worst edge error = 1.964965873654926e-08

Genus-2 check:
- Surface: stored genus-2 topology `1`
- Ribbon graph rebuilt with the same helper used in `genus2_one_point.py`
- Edge lengths tested:

| `ell_list` | worst edge | worst error |
| --- | --- | --- |
| `[50] * 9`  | `2` | `6.135144459993016e-06` |
| `[100] * 9` | `5` | `1.63306833103678e-06` |
| `[150] * 9` | `5` | `7.419309695682251e-07` |

Interpretation:
- The error decreases steadily with edge length, which is exactly what one
  expects if the residual mismatch comes from discretization / numerical
  integration rather than a conceptual error in the Abel-map construction.
- Following the project convention of avoiding tiny edge lengths, the
  regression test in `test_riemann_surface_tools.py` now uses
  `ell_list = [100] * 9` and checks

  $$
  \max(\text{edge error}) < 10^{-5}.
  $$

This is a direct test that the Abel map already knows about the quotient by the
period lattice encoded by the sewn boundary identifications.

#### (I) Genus-2 period matrix symmetry at large edge lengths

Ribbon graph: the stored F=1 topology-1 genus-2 graph returned by
`compact_partition.get_stored_genus2_graph(1)`, converted to the
`(edges, verts, rotation)` ribbon-graph format with the standard succ /
rotation construction used by `genus2_one_point.py`.

Set `ell_list = [500] * 9`, `L = 2 * sum(ell_list) = 9000`. Build the
forms with `elt.make_cyl_eqn_improved_higher_genus` and the period matrix
with `elt.period_matrix(forms=forms, ribbon_graph=..., ell_list=...,
return_data=True)`.

Result:

    Omega =
    [[ 0.51859357+1.03169474i  -0.2592968 -0.5158474i ]
     [-0.25929677-0.51584734i   0.51859357+1.03169474i ]]

    max|Omega - Omega.T| = 6.490e-08
    Im(Omega) eigenvalues = [0.5158474, 1.54754209]

Riemann bilinear says `Omega` must be symmetric and `Im Omega` must be
positive definite. Both hold: the symmetry error is a discretization
artifact (the only source of asymmetry is the scipy.quad integration of
`(1 - z^2)^{-1/3}`-style singular forms along radial rays), and the
imaginary eigenvalues are both strictly positive. Compute time ~50 s.

Tighter symmetry is possible with larger `ell_list`, at the cost of the
`m^3 = (sum ell)^3` SVD inside `make_cyl_eqn_improved_higher_genus`.

#### (J) Large-`L` renormalization of `(\det A')^{-1/2}` and canonical torus `|Z_1|^2`

This is the new check corresponding to the code path added for the bc ghost
partition-function prefactor.

Setup:
- ribbon graph: genus-1 theta graph
- fixed moduli represented by `base_edge_lengths = (1, 1, 1)`
- fitting scales:

  `scales = (200, 240, 300)`

- so the actual edge lengths and total lattice lengths are

| edge lengths | `L = 2 * sum ell_a` |
| --- | --- |
| `(200, 200, 200)` | `1200` |
| `(240, 240, 240)` | `1440` |
| `(300, 300, 300)` | `1800` |

For each sample:
1. build `A'` with `partition_function.traced_matter_matrix_f1`
2. compute

   $$
   \log Z_{\det} = -\frac{1}{2}\log \det(A')
   $$

3. fit

   $$
   \log Z_{\det} = c + \gamma L + \alpha \log L
   $$

Observed determinant data:

| `L` | `log Z_det` |
| --- | --- |
| `1200` | `-7.060727605920757` |
| `1440` | `-7.26835771524325` |
| `1800` | `-7.522480458763146` |

Fit result:

    c      =  1.0130478313885292
    gamma  = -6.077664037671385e-08
    alpha  = -1.1387327235441789
    exp(c) =  2.753981909259428
    R^2    =  1.0
    max residual = 4.973799150320701e-14

The fitted logarithmic coefficient is very close to the genus-1 value
`alpha(1) = -1.13889` quoted earlier in `Strebel.tex`, while the fitted
linear term is numerically negligible on this three-point torus sample.

Now build the large reference surface at the largest sample,
`edge_lengths = (300, 300, 300)`. The resulting modulus is

$$
\tau = 0.5000000000000002 + 0.8660254037844386\,i,
$$

as expected for the equilateral theta graph.

In the current preferred convention we define

$$
\lvert Z_1\rvert^2
= \bigl[\det \mathrm{Im}(\Omega)\bigr]^{1/2}\exp\!\bigl(c(\Omega)\bigr)
$$

with no extra moduli-independent factor. On the largest torus sample this gives

    |Z_1|^2 = 2.5628689466361014

using

    det Im(Omega)          = Im(tau)           = 0.8660254037844386
    [det Im(Omega)]^{1/2}  = sqrt(Im(tau))     = 0.9306048591020996
    exp(c(Omega))                              = 2.753981909259428.

This is exactly what `canonical_abs_z1_sq(...)` and
`estimate_canonical_abs_z1_sq(...)` implement.

Historical note:
- the older helper `estimate_abs_z1_sq(...)` also compares this determinant
  definition against the special `lambda=1` formula and reports the implied
  `\mathcal N_1`
- that path is still useful diagnostically, but it is no longer the preferred
  normalization scheme recorded in this note

#### (K) General bc correlator code path

The new general correlator helper is checked in two simple but sharp ways.

1. Selection rule:

   On the genus-1 surface `surface = rst.build_surface_data(L=20, l1=3, l2=4)`,
   the call

   `bc_correlator_geometric_factor([0.21+0.17i], [], surface, lambda_weight=2, divisor_points=[0.23+0.11i], normalization_point=0, nmax=8)`

   raises `ValueError`, because genus 1 requires

   $$
   n_c - n_b = (1 - 2\lambda)(g - 1) = 0
   $$

   for any `lambda`.

2. Compatibility with the special `lambda = 1`, `(n,m) = (g,1)` helper:

   On the same genus-1 surface, with

   `z = 0.21 + 0.17i`, `w = -0.17 + 0.14i`, `divisor_points = [0.23 + 0.11i]`,
   `normalization_point = w`, `nmax = 8`,

   the general geometric factor satisfies

   `bc_correlator_geometric_factor([z], [w], surface, lambda_weight=1, ...) / omega(z)`

   =

   `lambda_one_geometric_z1_factor([z], w, surface, ...)`

   to 10 decimal places.

   This is the right compatibility relation because for `g=1`, `lambda=1`
   the special last-equation helper differs from the raw correlator geometric
   factor by exactly the determinant of the one-by-one holomorphic-form matrix,
   namely `omega(z)`.

#### (K.1) Canonical higher-genus sigma normalization from chiral `Z_1`

The new higher-genus sigma-normalization helpers are checked in two steps.

1. Torus obstruction:

   On genus 1, `sigma_scale_from_z1(...)` raises `ValueError`, because the
   `lambda=1`, `(n,m)=(g,1)` special equation is insensitive to

   $$
   \sigma(z) \to C\,\sigma(z)
   $$

   when `g=1`.

2. Genus-2 normalization check:

   On stored genus-2 topology `1` with `ell_list = [100] * 9`, choose

   `anchor_b_points = [0.12 + 0.08i, -0.07 + 0.18i]`,
   `anchor_c_point = 0.04 - 0.19i`,
   `divisor_points = [0.23 + 0.11i, -0.17 + 0.14i]`,
   `Z_1 = 1.7`

   Then:

   - compute the scale `C` from `sigma_scale_from_z1(...)`
   - compute the old special helper `A_tilde` from
     `lambda_one_geometric_z1_factor(...)`

   Since genus 2 has `g-1 = 1`, the normalization law reduces to

   $$
   C\,\widetilde{A} = Z_1^{3/2}.
   $$

   The unit test verifies this identity directly to 9 decimal places. This is
   the sharp check that the higher-genus sigma normalization is being fixed by
   the chosen chiral `Z_1` in exactly the intended way.

#### (L) Genus-1 `<b(z)c(0)>` is flat after the correct `f(z)` rescaling

For the genus-1 bc ghosts (`lambda = 2`) with one `b` insertion at `z` and one
`c` insertion at `0`, the geometric factor returned by

`G(z,0) = bc_correlator_geometric_factor([z], [0], surface, lambda_weight=2, ...)`

contains all of the nontrivial `z`-dependence, since `Z_1` is independent of
the insertion point.

The correct genus-1 flat-frame invariant is

$$
\frac{\lvert G(z,0)\rvert^2}{\lvert f(z)\rvert^4},
$$

not `|f(z)|^4 |G(z,0)|^2`.

Reason:
- the torus identities for the genus-1 prime form and the validated sigma
  formula imply

  $$
  G(z,0) \propto \frac{f(z)^2}{f(0)}
  $$

  up to a `z`-independent factor
- therefore dividing by `|f(z)|^4` removes the entire insertion-point
  dependence

This was checked numerically by scanning 20 interior points `z` on several
large genus-1 surfaces, all with the same correlator inputs

`c-point = 0`, `divisor_points = [0.23 + 0.11i]`, `normalization_point = 0`,
`normalization_value = 1`, `nmax = 10`

and evaluating

$$
Q(z) = \frac{\lvert G(z,0)\rvert^2}{\lvert f(z)\rvert^4}.
$$

The 20 sample points were:

    0.06+0.10i,  0.11+0.15i,  0.18+0.07i,  0.26+0.14i,
   -0.08+0.16i, -0.15+0.09i, -0.22+0.18i, -0.28+0.05i,
   -0.11-0.12i, -0.19-0.18i, -0.05-0.25i,  0.09-0.20i,
    0.17-0.11i,  0.27-0.06i,  0.03+0.28i, -0.02+0.31i,
    0.14+0.24i, -0.24-0.08i,  0.31+0.02i,  0.07-0.31i.

Results:

| edge lengths `(l1,l2,l3)` | `tau` | max relative variation of `Q(z)` |
| --- | --- | --- |
| `(600,600,600)` | `0.5000000000000001 + 0.8660254037844385i` | `8.83e-15` |
| `(500,500,1500)` | `0.713614581452909 + 0.7005385279467516i` | `4.69e-14` |
| `(500,1000,1000)` | `0.4999999855560717 + 0.7234053551631896i` | `1.70e-14` |
| `(500,1500,3000)` | `0.6344028298988311 + 0.607951342480161i` | `8.70e-15` |
| `(500,750,1250)` | `0.6048020915467756 + 0.7178340473766368i` | `1.05e-14` |
| `(600,600,800)` | `0.5612805323342086 + 0.8276256182735199i` | `1.50e-14` |
| `(500,600,700)` | `0.532792754596249 + 0.8047088482514668i` | `1.32e-14` |
| `(800,700,500)` | `0.42717313370051724 + 0.9413013690189171i` | `1.56e-14` |

So the corrected invariant is constant to essentially machine precision across
both insertion point and moduli, exactly as expected.

#### (M) Canonical $\Delta$ via Deconinck Algorithm 1: consistency across all $\Delta$-dependent diagnostics

Having introduced `riemann_constant_vector_canonical(...)`, we re-ran every
$\Delta$-dependent diagnostic in sections (E)-(L) with the new $\Delta$ and
verified that nothing regresses.

##### Genus-1 diagnostics are numerically unchanged

At $g = 1$ on `(L, l1, l2) = (20, 3, 4)`, the canonical algorithm returns a
$\Delta$ that differs from the old $(1 - \tau)/2$ by the exact integer
lattice shift

$$
\Delta_{\mathrm{canonical}} - \Delta_{\mathrm{old}} = (-1,\; 0)
= -(1,\; 0) \in \Lambda.
$$

This is a pure integer shift with no $\tau$-period component, so the
quasi-periodic factor $\exp(2\pi i n(\zeta(w_0) - \zeta(z)))$ that would
otherwise distinguish `sigma_value` under a $\tau$-shift is absent, and
every $\Delta$-dependent genus-1 quantity agrees **numerically** with the
recorded old values:

| diagnostic | quantity | with canonical $\Delta$ |
| --- | --- | --- |
| (E) sigma divisor-independence | $\lvert \sigma(z)_{\text{div1}} - \sigma(z)_{\text{div2}} \rvert$ | $2.48 \times 10^{-16}$ |
| (E) sigma normalization | $\lvert \sigma(w_0) - 1 \rvert$ | $1.21 \times 10^{-18}$ |
| (E) recorded $\sigma(z)$ value | `0.431008760111272 - 0.224918148464132 i` | matches exactly |
| (K.2) $G/\omega(z)$ vs `lambda_one_geometric_z1_factor` | $\lvert \mathrm{diff} \rvert$ | $8.9 \times 10^{-16}$ |
| (L) $Q(z) = \lvert G\rvert^2 / \lvert f\rvert^4$ flatness | max rel var across 20 sample points | see below |

(L) results with canonical $\Delta$ on 4 moduli (same `divisor_points`,
`normalization_point`, `nmax = 10` as the original test):

| edge lengths `(l1,l2,l3)` | `tau` | max relative variation of $Q(z)$ |
| --- | --- | --- |
| `(600,600,600)` | `0.5000000000000001 + 0.8660254037844385 i` | `4.49e-14` |
| `(500,600,700)` | `0.5327927545962490 + 0.8047088482514668 i` | `1.79e-14` |
| `(800,700,500)` | `0.42717313370051724 + 0.9413013690189171 i` | `5.00e-14` |
| `(500,1500,3000)` | `0.6344028298988311 + 0.6079513424801610 i` | `2.33e-14` |

All numbers match the original (L) run to the same order of magnitude,
confirming that at $g = 1$ the new algorithm is strictly a drop-in
replacement.

##### Genus-2 Riemann vanishing and sigma-ratio on three moduli

At $g = 2$ the new algorithm actually changes behaviour, because the old
one is wrong there (see `known_genus2_ghost_issues.md` §6). Using
`riemann_constant_vector_canonical(surface, nmax=6)` on stored topology 1:

| `ell_list` | description | $\max_p \lvert\theta(\zeta(p)-\Delta)\rvert$ over 6 test points | sigma-ratio spread $\lvert\max/\min - 1\rvert$ over 3 divisors |
| --- | --- | --- | --- |
| `[100] * 9` | $Z_3$-symmetric | `1.20e-06` | `7.02e-06` |
| `[250,250,250,250,250,250,250,250,700]` | mildly asymmetric | `4.39e-07` | `1.76e-06` |
| `[210, 230, 250, 270, 290, 310, 330, 350, 370]` | fully asymmetric | `1.19e-07` | `5.71e-07` |

Compare to the old `riemann_constant_vector`, which gives
$\max_p \lvert \theta \rvert \sim 30\text{-}45$ and sigma-ratio spread
$\sim 4$ on all three moduli.

The test points for Riemann vanishing were

    0.08+0.12i, -0.14+0.16i, 0.19-0.09i, 0.21+0.09i, -0.16+0.12i, 0.30+0.15i.

The sigma-ratio divisor sets were

    (z, w) = (0.08+0.12i, -0.19+0.13i)
    div_1 = [(0.21+0.09i), (-0.16+0.12i)]
    div_2 = [(0.17+0.05i), (-0.12+0.11i)]
    div_3 = [(0.22+0.08i), (-0.09+0.16i)].

##### Canonical-divisor representative independence

Running the algorithm with `form_idx=0` and `form_idx=1` on the same
surface uses the zeros of two linearly independent A-normalized forms as
distinct canonical divisors. The resulting $\Delta$ values must differ by
a period-lattice vector (since both are valid Riemann constants mod
$\Lambda$). Numerically, on all three moduli above,

$$
\Delta_{\mathrm{form\_idx}=0} - \Delta_{\mathrm{form\_idx}=1}
= (0, -1) + \Omega\,(1, 0) \in \Lambda,
$$

with residual $\lesssim 10^{-6}$ per component — exactly as expected.
Either choice is a valid $\Delta$; each is internally consistent with
itself in every downstream identity.

##### Genus-2 higher-genus sigma normalization (K.1.2) with canonical $\Delta$

With canonical $\Delta$ supplied, the sharp identity

$$
C \cdot \widetilde{A} = Z_1^{3/2}
$$

from (K.1.2) (stored topology 1, `ell_list = [100] * 9`, `Z_1 = 1.7`,
anchor data exactly as in (K.1.2)) holds to

    |C * A_tilde - Z_1^(3/2)| = 4.5e-16

which is the same precision as the original check.

##### Unit-test addendum

The unit-test suite in `test_riemann_surface_tools.py` gained four new
tests (all passing):

1. `test_canonical_riemann_constant_matches_genus1_half_period_modulo_lattice`
2. `test_canonical_riemann_constant_satisfies_vanishing_genus2_symmetric`
3. `test_canonical_riemann_constant_satisfies_vanishing_genus2_asymmetric`
   (this is the regression test against the old "Z_3-symmetry artifact"
   failure mode)
4. `test_canonical_riemann_constant_makes_sigma_ratio_divisor_independent_genus2`

Total suite runtime is now ~65 s (25 tests pass).

### 2.3 Known non-bug to be aware of

`RiemannSurfaceData.A_periods` and `.B_periods` are the A/B periods of
the **unnormalized** forms that were passed into
`normalize_holomorphic_forms` (`ell_to_tau.py` returns them straight
through). After A-normalization the true periods are `I` and `Omega`,
which is what `surface.Omega`, `surface.normalized_forms`, and
`surface.antiderivatives` are consistent with. This mismatch only affects
downstream code that reads `surface.A_periods` / `.B_periods` directly
(nothing in the bc-correlator flow does), but it is worth a docstring
clarification or a flip to the normalized convention to avoid a
foot-gun.

Likewise, `sigma_value` is the **raw normalized** sigma rather than the
canonically `Z_1`-normalized one. Its overall multiplicative constant is left
as user input through `(normalization_point, normalization_value)`. This is
not a bug: it is exactly the ratio-based layer described above.

For genus `g>1`, the canonical overall sigma constant is now fixed instead by
`sigma_scale_from_z1(...)` / `canonical_sigma_value(...)` once a chiral `Z_1`
has been chosen. On the torus this remains impossible, and the code raises
explicitly there.

The older quantity `normalization_factor_from_lambda_one(...)` is therefore
best understood as a diagnostic comparison between two normalization schemes,
not as the preferred definition of `\mathcal N_1`.

Likewise, the full holomorphic correlator `bc_correlator(...)` depends on a
choice of chiral `Z_1` branch whenever `z1` is supplied. The default public
entry point therefore treats the geometric factor as the primary output and
only inserts `Z_1^{-1/2}` when the caller explicitly passes a nonzero `z1`.

Not a `riemann_surface_tools.py` issue, but noted during the review:
`ell_to_tau.make_cyl_eqn` (line 121) has a self-flagged
`coeffs[-1] = 0.0` with comment "FIGURE THIS OUT LATER -- COULD BE
WRONG". Only the *improved* solvers feed the bc-correlator building
blocks, so this line does not affect the checks above, but it is a
landmine worth cleaning up.

---

## 3. Reproducing the genus-2 check in one command

The dominant cost is the higher-genus form solve. Minimal runnable
reference (edge length 500 per edge, ~50 s on M-series CPU):

```bash
cd "covariant formalism/python"
python3 - <<'PY'
import numpy as np, ell_to_tau as elt, compact_partition as cp
gd = cp.get_stored_genus2_graph(1)
edges = [(a, b) for _, a, b in gd["edges"]]
verts = sorted({v for _, a, b in gd["edges"] for v in (a, b)})
boundary = tuple(gd["boundary"])
succ = {v: {} for v in verts}
for i, (_, to_v, e) in enumerate(boundary):
    nf, _, ne = boundary[(i + 1) % len(boundary)]
    succ[to_v][e - 1] = ne - 1
rotation = {}
for v in verts:
    incident = [idx for idx, (a, b) in enumerate(edges) if a == v or b == v]
    order = [incident[0]]
    while True:
        nxt = succ[v][order[-1]]
        if nxt == order[0]:
            break
        order.append(nxt)
    rotation[v] = order
rg = (edges, verts, rotation)
ell = [500] * 9
forms = elt.make_cyl_eqn_improved_higher_genus(rg, ell)
data = elt.period_matrix(forms=forms, ribbon_graph=rg,
                         ell_list=ell, return_data=True)
Om = np.asarray(data["Omega"], dtype=np.complex128)
print("max|Omega - Omega.T| =", float(np.max(np.abs(Om - Om.T))))
print("Im(Omega) eigvals    =", np.linalg.eigvalsh(Om.imag))
PY
```
