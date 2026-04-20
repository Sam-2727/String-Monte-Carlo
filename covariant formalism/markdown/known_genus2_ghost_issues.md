# Known Genus-2 Ghost Issues

This note records the concrete numerical failures currently known in the
genus-2 ghost pipeline. It only includes diagnostics that directly exhibit a
problem in the code or in the currently implemented formulae.

## 1. `sigma_ratio(...)` Is Divisor-Dependent In Genus 2

This is the clearest failure in the current code.

### Setup

- Surface: stored genus-2 topology `1`
- Edge lengths: `[300] * 9`
- Evaluation points:
  - `z = 0.08 + 0.12 i`
  - `w = -0.19 + 0.13 i`
- Divisor choices:
  - set 1: `[(0.21+0.09i), (-0.16+0.12i)]`
  - set 2: `[(0.17+0.05i), (-0.12+0.11i)]`
  - set 3: `[(0.22+0.08i), (-0.09+0.16i)]`

### Measured Values

Using the current implementation of `sigma_ratio(z, w, surface, divisor_points=...)`:

- set 1:
  - `sigma_ratio = -0.2334175052082666 + 0.2412845498429773 i`
  - `|sigma_ratio| = 0.33571113435598077`
- set 2:
  - `sigma_ratio = -0.8792750379668643 + 0.47517120869104756 i`
  - `|sigma_ratio| = 0.9994559870052017`
- set 3:
  - `sigma_ratio = -1.4393148244005336 - 0.10981953212564267 i`
  - `|sigma_ratio| = 1.443498352397894`

This ratio should be independent of the auxiliary divisor. It is not.

## 2. Decomposition Of The Broken Sigma-Ratio Formula

For the same surface, same `z,w`, and the same three divisor sets, it is useful
to record separately:

- the raw theta ratio
- the prime-form correction
- the full combined sigma-ratio expression

This decomposition is only a diagnostic record. It does not by itself identify
which ingredient is wrong.

### Setup

Same setup as in section 1.

For each divisor set, define:

- `theta ratio`
  \[
  \frac{\theta(\sum_i \zeta(z_i)-\zeta(z)-\Delta)}
       {\theta(\sum_i \zeta(z_i)-\zeta(w)-\Delta)}
  \]
- `prime correction`
  \[
  \frac{\prod_i E(z_i,w)}{\prod_i E(z_i,z)}
  \]
- `combined = theta ratio * prime correction`

### Measured Values

- set 1:
  - `theta ratio = 0.7046845527770178 - 0.48798627275364537 i`
  - `|theta ratio| = 0.8571527992829171`
  - `prime correction = -0.38413646813964686 + 0.0763905300813303 i`
  - `|prime correction| = 0.39165844717165055`
  - `combined = -0.2334175052082666 + 0.2412845498429773 i`
  - `|combined| = 0.33571113435598077`

- set 2:
  - `theta ratio = 0.7068423519925493 - 0.4911195006519857 i`
  - `|theta ratio| = 0.8607116093622851`
  - `prime correction = -1.1539503783385774 - 0.12952863483303217 i`
  - `|prime correction| = 1.161197288538625`
  - `combined = -0.8792750379668643 + 0.47517120869104756 i`
  - `|combined| = 0.9994559870052017`

- set 3:
  - `theta ratio = 0.7154333898877595 - 0.482982558146463 i`
  - `|theta ratio| = 0.8632016490021277`
  - `prime correction = -1.310791260361776 - 1.0384039363437463 i`
  - `|prime correction| = 1.6722608837304662`
  - `combined = -1.4393148244005336 - 0.10981953212564267 i`
  - `|combined| = 1.443498352397894`

## 3. The Old Anchor-Based Genus-2 `bbb` Correlator Is Anchor-Dependent

The original genus-2 pipeline based on:

- `sigma_ratio(...)`
- `sigma_value(...)`
- `sigma_scale_from_z1(...)`
- `canonical_sigma_value(...)`

does not produce an auxiliary-data-independent `bbb` correlator.

### Setup

- Surface: stored genus-2 topology `1`
- Edge lengths: `[300] * 9`
- Total length: `L = 5400`
- `b` insertion points:
  - `z_1 = 0.08 + 0.12 i`
  - `z_2 = -0.14 + 0.16 i`
  - `z_3 = 0.19 - 0.09 i`
- Canonical `|Z_1|^2` from the renormalized determinant:
  - `|Z_1|^2 = 5.235326888706135`

Anchor data sets tested:

- set 1:
  - `anchor_b_points = [0.12+0.08i, -0.11+0.17i]`
  - `anchor_c_point = -0.19+0.13i`
  - `divisor_points = [0.21+0.09i, -0.16+0.12i]`
- set 2:
  - `anchor_b_points = [0.05+0.18i, -0.07+0.14i]`
  - `anchor_c_point = -0.21+0.07i`
  - `divisor_points = [0.17+0.05i, -0.12+0.11i]`
- set 3:
  - `anchor_b_points = [0.09+0.04i, -0.18+0.19i]`
  - `anchor_c_point = -0.13+0.09i`
  - `divisor_points = [0.22+0.08i, -0.09+0.16i]`

### Measured Values

Using the old anchor-based construction:

- set 1:
  - `|sigma_scale| = 0.007178407651466103`
  - `|<bbb>| = 1.5668532396342248e-14`
- set 2:
  - `|sigma_scale| = 0.000935707492984164`
  - `|<bbb>| = 4.6913489890187284e-17`
- set 3:
  - `|sigma_scale| = 0.0023007386567942497`
  - `|<bbb>| = 3.452791924329849e-15`

The correlator should not depend on the auxiliary anchor/divisor data. It does.

## 4. Equal-`L` Genus-2 Igusa Ratio Check Fails With The Old Anchor-Based `bbb` Correlator

### Setup

- Surface topology: stored genus-2 topology `1`
- Same total length for both moduli: `L = 5400`
- Modulus 1 edge lengths: `[300, 300, 300, 300, 300, 300, 300, 300, 300]`
- Modulus 2 edge lengths: `[250, 250, 250, 250, 250, 250, 250, 250, 700]`
- `b` insertion points:
  - `z_1 = 0.08 + 0.12 i`
  - `z_2 = -0.14 + 0.16 i`
  - `z_3 = 0.19 - 0.09 i`

The checked ratio was:

\[
\frac{|\langle bbb\rangle_{\Omega_1}|^2}{|\langle bbb\rangle_{\Omega_2}|^2}
\left(\frac{|Z_1(\Omega_1)|^2}{|Z_1(\Omega_2)|^2}\right)^{26}

\quad\text{vs}\quad

\frac{|\det S(\Omega_1)|^2}{|\det S(\Omega_2)|^2}
\frac{|\chi_{10}(\Omega_2)|^2}{|\chi_{10}(\Omega_1)|^2}.
\]

### Measured Values

Using the old anchor-based `bbb` correlator:

- `LHS = 189.2705776411201`
- `RHS = 7.139757771156295`
- `LHS / RHS = 26.509383610428486`
- relative difference `= 25.509383610428486`

## 5. Equal-`L` Genus-2 Igusa Ratio Check Still Fails With The New Direct `lambda=1` Solver

The old anchor-based sigma pipeline was replaced by a direct genus-2
construction that solves the three needed sigma values from the
`lambda = 1`, `(n,m) = (2,1)` equations on the same three `b` insertion
points.

This removes the anchor/divisor dependence of the old method, but the final
Igusa ratio check still does not pass.

### Setup

Same setup as in section 4.

### Measured Values

Using the direct genus-2 `lambda=1` solver:

- modulus 1:
  - `direct <bbb> = -8.053452310102549e-12 + 5.375190984076192e-12 i`
  - `|direct <bbb>| = 9.682498243035732e-12`
- modulus 2:
  - `direct <bbb> = 1.247294419141055e-12 + 1.3676849899307532e-13 i`
  - `|direct <bbb>| = 1.2547704930931558e-12`

Ratio comparison:

- `LHS = 159.06147440669713`
- `RHS = 7.139757771156295`
- `LHS / RHS = 22.278273227879673`
- relative difference `= 21.278273227879673`

This means the old sigma-normalization bug was real, but fixing it did not by
itself make the final genus-2 Igusa ratio test pass.

## 6. Root Cause: `riemann_constant_vector` Is Wrong At Genus 2

All of the failures in sections 1–5 trace back to a single bug: the Riemann
class vector `Delta` computed by `riemann_constant_vector` in
`riemann_surface_tools.py` does not satisfy the Riemann vanishing theorem at
genus 2.

### Direct diagnostic: Riemann vanishing fails

For any point `p` on the surface, the Riemann theorem requires

    theta(zeta(p) - Delta | Omega) = 0.

On stored genus-2 topology `1` with `ell_list = [100] * 9`, the implementation
returns

    Delta_current = (0.07403653 - 0.51584739 i,
                     0.07403653 - 0.51584739 i)

and `theta(zeta(p) - Delta_current)` is large at every tested point:

| point `p`        | `|theta(zeta(p) - Delta_current)|` |
| ---              | ---                                |
| `0.08 + 0.12 i`  | `2.08e+01`                         |
| `-0.14 + 0.16 i` | `1.69e+01`                         |
| `0.19 - 0.09 i`  | `4.08e+01`                         |
| `0.21 + 0.09 i`  | `2.54e+01`                         |
| `-0.16 + 0.12 i` | `1.82e+01`                         |

For comparison, `|theta(y)|` at a generic argument on this `Omega` is of order
`~3`, so these values are not merely "not small" — they are larger than the
generic scale.

### Why genus 1 passes: the buggy code path is empty there

The formula implemented at `riemann_surface_tools.py:500-548` is

    Delta_I = 1/2 (1 - Omega_{II})
              + sum_{J != I} int_{alpha_J} omega_J(z) zeta_I(z) dz.

For `g = 1` the `J != I` loop is empty and the cycle-integral correction never
runs. `Delta = (1 - tau)/2` is then exact, which is why
`test_genus1_riemann_constant_matches_half_period_formula` passes and genus-1
sigma tests are divisor-independent. The bug lives entirely in the
cycle-integral correction that only activates for `g > 1`.

### Other genus-2 ingredients are correct

The following were checked individually on genus-2 surfaces and all pass, so
the Riemann-vanishing failure cannot be blamed on them:

- Prime-form local limit `E(z, z + eps) / (z - (z + eps)) -> 1` for all six
  odd characteristics (error `~1e-10`).
- Prime-form antisymmetry `E(z, w) + E(w, z) = 0` (error `~1e-16`).
- Prime-form independence of the chosen odd spin structure (spread
  `~1e-7`, see §2.2 (G) of `riemann_surface_tools.md`).
- A-normalization `int_{alpha_J} omega_I = delta_{IJ}` (error `~1e-16`).
- B-period integrals reproduce `Omega` (error `~1e-10`).
- Abel-map jumps on identified boundary points match the period lattice
  (§2.2 (H) of `riemann_surface_tools.md`).
- `riemann_theta` matches `elt.riemann_theta_constant_genus2` at handmade
  `Omega` (error `~1e-16`).

### Numerical fit for the correct Delta

Fitting `Delta` to minimize `sum_p |theta(zeta(p) - Delta)|^2` over several
test points for the symmetric `ell_list = [100] * 9` modulus converges (fit
loss `1.7e-13`) to

    Delta_fit = (0.07403658 - 0.51584733 i,
                 0.33333327 - 2.4e-08    i).

The first component matches `Delta_current` at the fit tolerance. The second
component differs by

    Delta_fit - Delta_current
      = (~0, Omega_{11} / 2 + ... )
      = (1, 0) + Omega (1/3, 2/3)   (mod the period lattice).

This is **not** a lattice vector. Note the `(1/3, 2/3)` — the appearance of
thirds is a hint, not a coincidence: for the topology-1 graph with all edge
lengths equal, the surface has a `Z_3` symmetry, and the `(1/3, 2/3)`
fractional shift ends up being equivalent to a half-lattice shift under that
symmetry (see sign sweep below).

### The `(1/3, 2/3)` shift is NOT universal: asymmetric-modulus test

To confirm that `Omega (1/3, 2/3)` is a Z_3-symmetry artifact rather than a
universal correction, test on a fully asymmetric modulus where every edge
length is distinct and every edge is `>= 200`:

    ell = [210, 230, 250, 270, 290, 310, 330, 350, 370],   L = 5220.

The resulting period matrix

    Omega = [[ 0.5624+1.0476 i  -0.2887-0.5495 i]
             [-0.2887-0.5495 i   0.5291+1.0133 i]]

has no Z_2 or Z_3 symmetry (the off-diagonals are no longer `~ -Omega_00 / 2`
and the diagonals are not equal).

Riemann vanishing `|theta(zeta(p) - Delta)|` at 9 disc-interior points:

| point `p`        | `Delta_current` | `Delta_current + Omega (1/3, 2/3)` |
| ---              | ---             | ---                                |
| `0.08 + 0.12 i`  | `2.32e+01`      | `4.97e-01`                         |
| `-0.14 + 0.16 i` | `1.93e+01`      | `4.80e-01`                         |
| `0.19 - 0.09 i`  | `4.38e+01`      | `7.80e-01`                         |
| `0.21 + 0.09 i`  | `2.80e+01`      | `5.17e-01`                         |
| `-0.16 + 0.12 i` | `2.07e+01`      | `5.03e-01`                         |
| `0.30 + 0.15 i`  | `2.54e+01`      | `4.26e-01`                         |
| `-0.20 + 0.30 i` | `1.46e+01`      | `4.09e-01`                         |
| `0.02 - 0.15 i`  | `3.85e+01`      | `7.70e-01`                         |
| `0.35 + 0.05 i`  | `3.71e+01`      | `5.88e-01`                         |

The shift knocks the Riemann-vanishing failure down by a factor of ~50 but
leaves `|theta| ~ 0.4 - 0.8` — nowhere near vanishing. On a `Z_3`-symmetric
surface this shift accidentally lands in the correct half-lattice coset;
once the symmetry is broken the accident disappears.

`sigma_ratio` divisor-independence (same `z = 0.08 + 0.12 i`,
`w = -0.19 + 0.13 i` as section 1), with
`Delta = Delta_current + Omega (1/3, 2/3)`:

| divisor                               | `sigma_ratio`              | `|sigma_ratio|` |
| ---                                   | ---                        | ---             |
| `[0.21 + 0.09 i, -0.16 + 0.12 i]`     | `-0.538 + 0.164 i`         | `0.562`         |
| `[0.17 + 0.05 i, -0.12 + 0.11 i]`     | `-1.967 + 0.040 i`         | `1.968`         |
| `[0.22 + 0.08 i, -0.09 + 0.16 i]`     | `-1.201 - 2.092 i`         | `2.412`         |

Still divisor-dependent — `|sigma_ratio|` swings by a factor of ~`4.3` across
the three divisor choices, essentially the same spread as §1 with the raw
`Delta_current`.

So the `(1/3, 2/3)` fix is specific to the `Z_3`-symmetric
`ell = [100] * 9` surface (and by extension any `[N] * 9` surface) and
gives no correction at all for a generic modulus. The correct `Delta` must
be determined per-modulus — either by the numerical fit described below or
by Deconinck's canonical-divisor algorithm.

### Confirmation that this fixes `sigma_ratio`

Plugging `Delta_corrected = Delta_current + Omega (1/3, 2/3)` into
`sigma_ratio` for the `ell_list = [300] * 9` case (i.e. the same setup as §1
of this note) gives divisor-independent values:

| divisor                              | `sigma_ratio`                     |
| ---                                  | ---                               |
| `[(0.21 + 0.09 i), (-0.16 + 0.12 i)]`| `0.76678220 - 0.14752560 i`       |
| `[(0.17 + 0.05 i), (-0.12 + 0.11 i)]`| `0.76678467 - 0.14752500 i`       |
| `[(0.22 + 0.08 i), (-0.09 + 0.16 i)]`| `0.76678450 - 0.14752343 i`       |

All three agree to roughly five decimals — the residual spread is at the
scale of the scipy.quad integration of the singular-weighted form, not a
real divisor dependence. Compare to §1, where `|sigma_ratio|` varied between
`0.336` and `1.443` across the same three divisors.

### Why the cycle-integral term as implemented is off

The formula quoted in `Strebel.tex` is the Fay/Mumford formula, derived by
integrating around a standard `4g`-gon fundamental polygon with a specific
corner as the basepoint for `zeta_I`. The current implementation:

- takes the basepoint of `zeta_I` at the disc origin `z = 0` (an interior
  point of the disc), rather than at a polygon corner;
- represents each `alpha_J` cycle by the straight disc-frame chord between
  the two midpoints `(z0, z1)` of the edge's boundary occurrences, rather
  than as a loop based at the polygon corner.

For a non-closed integrand like `omega_J(z) zeta_I(z) dz`, the choice of
representative path matters — it is not a homological invariant. The
disc-chord representation differs from the canonical polygon-edge
representation by a quantity that is lattice-valued in genus 1 (harmless)
but non-lattice in genus 2.

### Comparison against the standard (Deconinck, 2015) formula

The standard reference is Deconinck, Patterson, Swierczewski,
*Computing the Riemann Constant Vector* (2015). Their definition,
eq. (16) p. 9, is

    K_j(P) = (1 + Omega_{jj})/2
             - sum_{k != j} oint_{a_k} omega_k(Q) A_j(P, Q) dQ,

where `A_j(P, Q) = int_P^Q omega_j` along any path on the surface. Under the
identification `zeta_I(z) = A_I(P, z)` the integrands in Strebel's formula
and Deconinck's formula are the **same**, but the two disagree in two
places:

| piece                  | Strebel         | Deconinck       |
| ---                    | ---             | ---             |
| constant term          | `(1 - Omega)/2` | `(1 + Omega)/2` |
| sign on cycle integral | `+`             | `-`             |

Modulo the period lattice these differences do **not** cancel — `Omega_{II}`
placed in a single component is not a lattice vector (a lattice vector is
`m + Omega n` for integer vectors, always placing `Omega n` along a full
column, never a single diagonal entry). So the two formulas give genuinely
different `Delta`.

Deconinck also note explicitly (p. 10, around eq. 24) that even their
formula (16) is only correct **modulo `(1/2) Lambda`**:

> In general there are `2^(2g)` half-lattice vectors `h in (1/2) Lambda`
> such that `K(P_0) = h - (1/2) A(P_0, C) (mod Lambda)`. Therefore a
> second objective is to find an appropriate half-lattice vector.

Their Algorithm 1 pins `K` down by (a) computing `-(1/2) A(P_0, C)` for any
canonical divisor `C`, then (b) testing all `2^(2g) = 16` half-lattice
candidates and keeping the one for which `|theta(K, Omega)|` is smallest
(Riemann vanishing).

### Sign-convention sweep: no formula works universally

I tested all four sign variants

    (1 ± Omega_{II})/2  ±  int_{alpha_J} omega_J zeta_I dz

plus "no-integral" and "integral-only" reductions, combined with every
half-lattice shift (`16` candidates at `g = 2`) and with finer fractional
shifts down to `1/12`-lattice granularity. Two moduli tested: the symmetric
`ell = [100] * 9` and the asymmetric
`ell = [250, 250, 250, 250, 250, 250, 250, 250, 700]`.

Symmetric modulus `ell = [100] * 9`, `max |theta(zeta(p) - Delta)|` over 6
disc-interior points:

| formula                                          | best fractional shift            | `max |theta|`  |
| ---                                              | ---                              | ---            |
| pure half-lattice (constant term only)           | `(1/2, 0) + Omega (1/2, 0)`      | `5.2e-01`      |
| `(1 - Omega)/2 + int`   (Strebel)                | `Omega (1/2, 1/2)`               | `1.48`         |
| `(1 + Omega)/2 - int`   (Deconinck)              | `(0, 1/2)`                       | `3.7e+01`      |
| `(1 - Omega)/2 - int`                            | `Omega (1/2, 1/2)`               | `9.1e-01`      |
| `(1 + Omega)/2 + int`                            | `(1/2, 1/2)`                     | `3.7e+01`      |
| integral only                                    | `(1/2, 1/2) + Omega (1/2, 0)`    | `9.6e-01`      |
| `(1 - Omega)/2` no integral                      | `(1/2, 0) + Omega (1/2, 1/2)`    | `7.9e-01`      |
| **Strebel + `Omega (1/3, 2/3)`** (3-lattice)     | —                                | **`4.9e-06`**  |

Asymmetric modulus
`ell = [250, 250, 250, 250, 250, 250, 250, 250, 700]`:

| formula                                | best fractional shift                        | `max |theta|`  |
| ---                                    | ---                                          | ---            |
| pure half-lattice                      | `(1/2, 0) + Omega (1/2, 0)`                  | `4.3e-01`      |
| Strebel + half-lattice (`1/2`)         | `Omega (1/2, 1/2)`                           | `6.9e-01`      |
| Strebel + `1/3`-lattice                | `Omega (1/3, 2/3)`                           | `8.0e-01`      |
| Strebel + `1/4`-lattice                | `Omega (2/4, 3/4)`                           | `5.2e-01`      |
| Strebel + `1/6`-lattice                | `(1/6, 4/6) + Omega (3/6, 5/6)`              | `4.9e-01`      |
| Strebel + `1/12`-lattice               | `(0, 1/12) + Omega (4/12, 8/12)`             | `2.6e-01`      |

### What the sweep tells us

Three observations, in decreasing order of importance:

1. **No sign choice plus a rational-lattice shift (down to `1/12`)
   universally reproduces the Riemann constant.** For the asymmetric
   modulus every candidate tested leaves a non-lattice residual of order
   `0.25 - 1`. The residual is **constant in `ell`** (I verified across
   `ell = 100, 200, 300` for the symmetric case) — so this is not
   scipy-quad integration error, it is a genuine mismatch between the
   disc-frame cycle integral and the Fay/Deconinck cycle integral.

2. **The clean `Omega (1/3, 2/3)` result was a symmetry artifact.** For the
   `[100] * 9` surface (topology 1 with all edges equal, a `Z_3`-symmetric
   hyperelliptic surface) this shift coincidentally equals a half-lattice
   shift after modding out by the surface `Z_3` symmetry, and produces
   `|theta| ~ 5e-6`. The moment the symmetry is broken (asymmetric edge
   lengths) the `1/3, 2/3` shift is no better than any other rational
   shift, and the best rational-lattice candidate is still two orders of
   magnitude away from vanishing.

3. **The Strebel sign convention is very likely wrong.** Strebel uses
   `+(1 - Omega)/2 + int` while Deconinck (matching Mumford/Fay) uses
   `+(1 + Omega)/2 - int`. But **fixing the sign alone is not enough**;
   even with Deconinck's sign convention, no half-lattice shift gets
   `|theta|` below ~`0.4`, so the implementation is also missing the
   half-lattice determination step.

### Downstream impact

Every failure in §§1–5 traces to this one `Delta`:

- §1, §2: `sigma_ratio` uses `Delta` at
  `riemann_surface_tools.py:598-599`; wrong `Delta` -> divisor-dependent
  ratio.
- §3: the old anchor-based `bbb` calls `sigma_value`, which calls
  `sigma_ratio`, which uses `Delta`.
- §4, §5: both the old anchor pipeline and the newer direct `lambda = 1`
  solver (`genus2_sigma_values_from_lambda_one`) compute a theta factor at
  argument `zeta_b - 3 Delta` using the same wrong `Delta`, so the final
  `bbb` correlator is wrong even with the sigma-normalization fix.

### Fix options

Given that the sign sweep rules out any "fix the sign + add a half-lattice
shift" recipe, the preferred path is to bypass the cycle-integral formula
entirely and use the classical characterization of the Riemann constant
through the canonical divisor.

1. **Preferred: Deconinck Algorithm 1 adapted to the disc frame.** By
   Theorem 11 in Deconinck (2015), `2 K(P_0) = -A(P_0, C) (mod Lambda)`
   for any canonical divisor `C`. At `g = 2`, `C` has degree `2g - 2 = 2`,
   so it is the two-point divisor of zeros of any holomorphic 1-form
   `omega_I`. The two zeros of `omega_I` can be computed directly from the
   `.coeffs` / `.singular_points` data already carried by the improved
   higher-genus forms — no cycle-integral needed. Then iterate over the
   `2^(2g) = 16` half-lattice candidates `h` and keep the one for which
   `|theta(h - (1/2) A(P_0, C), Omega)|` is smallest. This is exactly
   Deconinck Algorithm 1 and is convention-free.

2. **Fallback: direct numerical fit of the Riemann vanishing theorem.**
   Pick several disc-interior points `p_k` and minimize
   `sum_k |theta(zeta(p_k) - Delta)|^2` over `Delta`. Slower (an
   optimization per surface build) and requires careful global-minimum
   handling, but is zero-theory.

3. **Not recommended: patch the cycle-integral formula.** The sweep above
   shows no clean counterterm makes the disc-frame cycle integral equal
   the Fay/Deconinck cycle integral; deriving such a counterterm for the
   ribbon-graph 18-gon fundamental domain is substantial analytic work
   and more fragile than options 1 and 2.

Any fix must be guarded by a regression test that checks Riemann
vanishing at a handful of disc-interior points at genus 2 on at least two
different moduli (including one that breaks the `Z_3` symmetry of
topology-1 equal-edge-length surfaces), so that this specific failure
mode is never re-introduced.
