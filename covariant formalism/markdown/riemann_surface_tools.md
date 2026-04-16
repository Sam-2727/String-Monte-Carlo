# `riemann_surface_tools.py`: contents and cross-checks

Companion notes for [`python/riemann_surface_tools.py`](../python/riemann_surface_tools.py).
This module sits on top of `ell_to_tau.py` and packages the analytic pieces
needed by the bc-correlator work: A-normalized holomorphic one-forms,
radial antiderivatives / Abel-Jacobi maps on the disc frame, Riemann theta
functions with characteristics, the prime form built from an odd
characteristic, the Riemann class vector `Delta`, and the normalized
`sigma`-function ratios that appear in the Verlinde-Verlinde formula. The
sign/normalization conventions follow the Strebel note, with the concrete
implementation choices recorded explicitly below.

---

## 1. What is in `riemann_surface_tools.py`

### Data container

`RiemannSurfaceData` (a frozen dataclass) bundles, for a single choice of
ribbon graph + edge lengths:

| field | meaning |
| --- | --- |
| `genus` | `g`. |
| `Omega` | normalized period matrix (`g x g`); scalar `tau` on genus 1 via the `.tau` property. |
| `normalized_forms` | the A-normalized holomorphic one-forms in the disc frame (tuple of callables). |
| `antiderivatives` | the radial antiderivatives `F_I(z) = \int_0^z \omega_I` used for the Abel-Jacobi map. |
| `A_periods`, `B_periods` | the raw A- and B-periods of the forms passed to `normalize_holomorphic_forms`. **Note:** these are the pre-normalization periods returned by `ell_to_tau.period_matrix`, **not** `I` and `Omega`. The `Omega` field and everything derived from `antiderivatives` / `normalized_forms` is the A-normalized object. |
| `basis_pairs` | the chosen symplectic {alpha, beta} cycles, each stored as edge-chord decompositions `[(edge_idx, coeff), ...]`. |
| `edge_midpoints` | the disc-frame representatives `(z0, z1)` for each edge-chord, used to integrate forms along cycles. |

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
  `zeta(z) - zeta(basepoint)`, where each component is the straight-line
  radial integral of the A-normalized form from `basepoint` to `z`.
- `abel_difference(z, w, surface)`: returns `F(z) - F(w)` component-wise.
  This is the quantity that appears inside theta factors in the
  Verlinde-Verlinde formula.

Implementation choice:
- In the notation of `Strebel.tex`, the unspecified basepoint `z_0` is taken
  to be `0` by default.
- The path from `z_0` to `z` is the straight radial segment in the disc
  frame. More generally, `abel_map(z, surface, basepoint=z_0)` returns
  `F(z) - F(z_0)` using the same radial primitive.
- This is a concrete disc-frame convention rather than an extra mathematical
  statement. For the prime form and the bc-correlator formula, the basic
  building block is `zeta(z) - zeta(w)`, so the arbitrary basepoint cancels.

### Theta characteristics

- `theta_characteristics(g, parity=None, as_half_integers=False)`:
  enumerate all `4**g` characteristics (or just even/odd, which partition
  them into `2^{g-1}(2^g +/- 1)` subsets). Default output is binary bits
  `((a1,...,ag), (b1,...,bg))` in `{0,1}`, interpreted as eps = a/2 and
  delta = b/2.
- `characteristic_parity(char)`: returns the Arf invariant
  `4 * eps . delta  (mod 2)`; 1 means odd.
- `odd_characteristic(g)`: returns the first odd characteristic of genus g
  (used as a convenient default for the prime form).
- `_coerce_characteristic`: internal normalization; accepts either `{0,1}`
  binary bits or `{0, 1/2}` half-integer entries.

### Riemann theta

Convention:

    theta[eps, delta](y | Omega) = sum_n exp( i pi (n+eps)^T Omega (n+eps)
                                             + 2 i pi (n+eps)^T (y+delta) ).

- `riemann_theta(y, Omega, characteristic=None, nmax=None, tol=1e-12)`:
  plain value. Uses a centered box truncation `n_i in [-nmax, nmax]`.
- `riemann_theta_gradient`: y-gradient
  `grad[I] = sum_n (2 i pi)(n+eps)_I * (theta summand)_n`.
- `riemann_theta_with_gradient`: returns both.
- `theta_truncation(Omega, tol)`: chooses a default `nmax` from the
  smallest eigenvalue of `Im(Omega)` so that the truncation error on the
  Gaussian tail is bounded by `tol`.

### Prime form

`prime_form(z, w, surface, characteristic=None, nmax=None, tol=1e-12)`:

    E(z,w) = theta[delta]( zeta(z) - zeta(w) | Omega )
             / sqrt( omega[delta](z) * omega[delta](w) )

with `omega[delta](z) = omega_I(z) * d_{y_I} theta[delta](0|Omega)` for an
odd characteristic `delta`. The square root uses the principal complex
branch, so the absolute value and local limits are unambiguous while the
overall sign depends on branch choices; compare ratios or local limits.

Implementation choices:
- If `characteristic=None`, the code uses `odd_characteristic(g)`, i.e. the
  first odd characteristic in the enumeration order returned by
  `theta_characteristics(g, parity="odd")`.
- The denominator uses the principal complex square root for both
  `sqrt(omega[delta](z))` and `sqrt(omega[delta](w))`.
- In genus 1, with the current branch choice and characteristic `((1),(1))`,
  the implemented local limit is

      E(z,w) ~ z - w

  in the disc coordinate. Equivalently,

      E(z, z+eps) / eps -> -1,
      E(z, z+eps) / (z-(z+eps)) -> 1.

- For the genus-1 checks below, the corresponding flat-coordinate formula is

      E_u(z,w) = - theta_1(pi u | tau) / (pi theta_1'(0 | tau)),
      u = F(z) - F(w).

### Riemann Class Vector

`riemann_constant_vector(surface, quad_limit=200)` computes the vector
`Delta` appearing in `Strebel.tex`:

    Delta_I = 1/2 (1 - Omega_{II})
              + sum_{J != I} int_{alpha_J} omega_J(z) zeta_I(z) dz.

The implementation is literal:

1. Start from the diagonal term `1/2 (1 - Omega_{II})`.
2. For each `I` and each `J != I`, read the chosen `alpha_J` cycle from
   `surface.basis_pairs[J]["alpha"]`.
3. Each cycle is stored as an edge-chord decomposition
   `[(edge_idx, coeff), ...]`.
4. For every edge chord:
   - read its disc-frame endpoints `(z0, z1)` from `surface.edge_midpoints`
   - define the integrand

         omega_J(z) * zeta_I(z)
         = _evaluate_one_form(surface.normalized_forms[J], z)
           * surface.antiderivatives[I](z)

   - integrate that function along the straight segment from `z0` to `z1`
     using `_segment_integral`
   - multiply by the cycle coefficient `coeff`
5. Sum over the segments and then over `J != I`.

Implementation choices:
- The same Abel-map convention is used here as everywhere else in the file:
  `zeta_I(z)` means the radial primitive from basepoint `0`.
- The cycle integral is done in the disc frame along the same straight
  boundary-chord representatives that were already used to compute the
  period matrix.
- `_segment_integral` uses `scipy.integrate.quad` separately on the real and
  imaginary parts of `func(z(t)) z'(t)` for the affine path

      z(t) = z0 + t (z1 - z0),   t in [0,1].

Genus-1 simplification:
- When `g=1`, there is no `J != I` term, so the formula reduces to

      Delta = 1/2 (1 - tau),

  which is one of the unit tests below.

### Sigma Function

The module does **not** attempt to construct an absolute canonical
`sigma(z)`. Instead it computes:

- `sigma_ratio(z, w, surface, divisor_points=..., ...) = sigma(z) / sigma(w)`
- `sigma_value(z, ..., normalization_point=y0, normalization_value=c)`
  by imposing `sigma(y0) = c`

This is the right level of generality because the Verlinde-Verlinde formula
only determines `sigma` up to an overall multiplicative constant.

#### Input Data

`sigma_ratio` requires:
- `surface`: the A-normalized `RiemannSurfaceData`
- `divisor_points=[z_1, ..., z_g]`: exactly `g` generic points
- optionally a precomputed `Delta`

The divisor points must be generic in the usual sense:
- they should not coincide with one another
- they should avoid the evaluation points `z`, `w`
- they should avoid choices that make the theta factor vanish

If these genericity conditions fail, the code raises `ZeroDivisionError`.

#### Formula Used

From the `lambda = 1`, `(n,m)=(g,1)` formula in `Strebel.tex`,

    Z_1 det(omega_I(z_i))
    =
    Z_1^{-1/2}
    theta(sum_i zeta(z_i) - zeta(w) - Delta | Omega)
    * [ prod_{i<i'} E(z_i,z_i') prod_i sigma(z_i) ]
      / [ prod_i E(z_i,w) sigma(w) ].

Now fix a generic divisor `z_1,...,z_g` once and for all and compare the same
formula written for `w=z` and for `w=w_ref`. All factors depending only on the
fixed divisor cancel. This gives

    sigma(z) / sigma(w_ref)
    =
    theta(S - zeta(z) | Omega) / theta(S - zeta(w_ref) | Omega)
    * prod_i E(z_i, w_ref) / prod_i E(z_i, z),

where

    S = sum_i zeta(z_i) - Delta.

That is exactly what `sigma_ratio` implements.

#### Algorithm Implemented By `sigma_ratio`

Given `z`, `w`, `surface`, and `divisor_points=[z_1,...,z_g]`:

1. Check that the number of divisor points is exactly `g = surface.genus`.
2. If `Delta` is not supplied, compute it with `riemann_constant_vector`.
3. Compute

       zeta_sum = sum_i abel_map(z_i, surface).

4. Form the theta arguments

       arg_z = zeta_sum - abel_map(z, surface) - Delta,
       arg_w = zeta_sum - abel_map(w, surface) - Delta.

5. Evaluate the theta factors

       theta_z = riemann_theta(arg_z, Omega),
       theta_w = riemann_theta(arg_w, Omega).

6. Build the prime-form products

       prime_prod_z = prod_i prime_form(z_i, z, surface),
       prime_prod_w = prod_i prime_form(z_i, w, surface).

7. Return

       (theta_z / theta_w) * (prime_prod_w / prime_prod_z).

The implementation computes exactly this expression, with the same theta and
prime-form conventions already described above.

#### Algorithm Implemented By `sigma_value`

`sigma_value(z, ..., normalization_point=y0, normalization_value=c)` is just

    sigma(z) = c * sigma_ratio(z, y0, ...).

So the entire normalization freedom of `sigma` is encoded in the pair
`(y0, c)`.

Implementation choices:
- The normalization is external and user-chosen; the module does not try to
  infer a preferred absolute normalization from geometry.
- The same divisor is used in numerator and denominator, which is why the
  divisor dependence cancels from the resulting normalized sigma.
- The verified invariant statement is divisor-independence of the normalized
  sigma ratio, not any stronger closed-form identification such as
  `sigma ~ sqrt(f)` in genus 1.

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
| Riemann class `Delta` | computed from the `Strebel.tex` formula using the stored alpha cycles |
| `sigma` normalization | user-specified by fixing `sigma(normalization_point)=normalization_value` |
| `sigma` divisor | any generic divisor of length `g`; normalized result is divisor-independent |
| genus-1 local limit | `E(z,w) ~ z-w` in the disc coordinate |
| genus-1 flat-coordinate formula used in checks | `E_u = -theta_1(pi u|tau)/(pi theta_1'(0|tau))` |

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

All tests below use **every edge length >= 500** to stay away from the
small-l Strebel artifacts. Reproducibility: unless noted otherwise, run
from `covariant formalism/python` with Python 3 and NumPy / SciPy / mpmath.
For genus 2 the calculations use ~50 s wall clock on a modern laptop
because of the `9000 x 4500` complex SVD inside
`make_cyl_eqn_improved_higher_genus`.

### 2.1 Pre-existing unit tests (`test_riemann_surface_tools.py`)

Run:

    python3 -m unittest test_riemann_surface_tools -v

These seven tests pass in the current workspace:

1. **Abel-Jacobi periods reproduce (1, tau)** at `(L, l1, l2) = (20, 3, 4)`:
   the alpha-period integral is `1.0` and the beta-period integral is
   `surface.tau` to 9 decimals.
2. **Genus-1 Riemann class** at `(L, l1, l2) = (20, 3, 4)`:
   `riemann_constant_vector(surface) = (1 - tau) / 2` to 10 decimals.
3. **Genus-1 characteristics vs. Jacobi `mp.jtheta`** at
   `tau = 0.37 + 0.91i`, `y = 0.23 + 0.07i`, `nmax = 8`:

   - `((0),(0))` -> `jtheta(3, pi y, q)`
   - `((1),(0))` -> `jtheta(2, pi y, q)`
   - `((0),(1))` -> `jtheta(4, pi y, q)`
   - `((1),(1))` -> `-jtheta(1, pi y, q)`

   to 12 decimals.
4. **Prime form local limit** `E(z, z+eps) / eps -> -1` for
   `z = 0.21 + 0.17i`, `eps = 1e-6 (1 + 0.4i)` on `(L, l1, l2) = (20, 3, 4)`
   with the odd characteristic `((1),(1))`. Equivalently,
   `E(z, z+eps) / (z-(z+eps)) -> 1`.
5. **Genus-1 normalized sigma ratio is divisor-independent** at
   `(L, l1, l2) = (20, 3, 4)`: with normalization point
   `w0 = -0.17 + 0.14i`, the values produced from divisor
   `[0.23 + 0.11i]` and divisor `[-0.09 + 0.27i]` agree to 9 decimals at
   `z = 0.31 - 0.12i`.
6. **Genus-1 sigma normalization is imposed exactly**:
   `sigma_value(w0, ..., normalization_point=w0)` returns `1` to 12 decimals
   with the default normalization value.
7. **Genus-2 theta constant cross-check** between `rst.riemann_theta` and
   `elt.riemann_theta_constant_genus2` on the handmade symmetric
   `Omega = [[0.9i, 0.11+0.07i],[0.11+0.07i, 1.2i]]` at
   characteristic `((1,0),(0,1))`, `nmax = 8`.

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

    E_u(z,w) = - theta_1(pi u | tau) / (pi theta_1'(0 | tau)),   u = F(z) - F(w).

Pulled back to the disc frame the prime form picks up `1/sqrt(f(z) f(w))`.
Check:

    E_disc(z,w) * sqrt(f(z) f(w))  ==  E_u(z,w)

up to one overall sign (square-root branch). For each `(l1, l2, l3)`
listed below and each of the three test pairs

    (z,w) in { (0.18+0.22i, 0.31-0.13i),
               (0.05+0.4i,  -0.22+0.09i),
               (-0.35+0.05i, 0.12+0.28i) }

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

so the checked flat-coordinate formula is

    E_u(z,w) = - _jtheta1_pi(u, tau) / _jtheta1_pi_prime_at_0(tau).

#### (C) Genus-1 prime form antisymmetry

On `surface = rst.build_surface_data(L=2*(500+600+700), l1=500, l2=600)`,
check `E(z,w) + E(w,z) == 0` for

    (z,w) = (0.21+0.17i, -0.13+0.28i),
    (z,w) = (-0.3+0.05i, 0.09-0.31i).

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

    z  = 0.31 - 0.12i,
    w0 = -0.17 + 0.14i.

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

    rst.riemann_theta(zeros(2), Omega, characteristic=char, nmax=12)

against `elt.riemann_theta_constant_genus2(Omega, char, nmax=12)`. Max
error across all 16 characteristics: `2.22e-16`.

#### (G) Genus-2 prime form is independent of odd spin structure

This is the most direct numerical check of the statement in `Strebel.tex`
that the prime form

    E(z,w) = theta[delta](zeta(z)-zeta(w)|Omega)
             / sqrt(omega[delta](z) omega[delta](w))

is independent of the choice of odd characteristic `delta`.

Important counting note:
- Genus 2 has **6** odd spin structures, not 3.
- The code therefore checks all 6 odd characteristics returned by
  `rst.theta_characteristics(2, parity="odd")`:

      ((0,1),(0,1)), ((0,1),(1,1)),
      ((1,0),(1,0)), ((1,0),(1,1)),
      ((1,1),(0,1)), ((1,1),(1,0)).

Surface used:
- stored genus-2 topology `1` from `compact_partition.get_stored_genus2_graph(1)`
- ribbon graph rebuilt with the same `boundary -> succ -> rotation` helper used
  in `genus2_one_point.py`
- edge lengths `ell_list = [50] * 9`

Evaluation points:

    z = 0.12 + 0.08i,
    w = -0.07 + 0.18i.

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

#### (H) Genus-2 period matrix symmetry at large edge lengths

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

Likewise, `sigma_value` is a **normalized** sigma rather than an absolute
canonical sigma. The overall multiplicative constant is intentionally left as
user input through `(normalization_point, normalization_value)`. This is not a
bug: it is the natural ambiguity left by the defining formula.

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
