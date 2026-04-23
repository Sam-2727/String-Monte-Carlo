# `genus3_bc_chi18_ratio_check.py`

Companion note for
[`python/genus3_bc_chi18_ratio_check.py`](../python/genus3_bc_chi18_ratio_check.py).

This script is the genus-3 analogue of the existing genus-2 Igusa check in
[`python/genus2_bc_igusa_ratio_check.py`](../python/genus2_bc_igusa_ratio_check.py).

## Goal

For two genus-3 moduli \(\Omega_1,\Omega_2\), the script compares the ratio

$$
\frac{
\bigl|\langle b(z_1)\cdots b(z_6)\rangle_{\Sigma(\Omega_1)}\bigr|^2
\;|Z_{\mathrm{chiral}}(\Omega_1)|^{52}
}{
\bigl|\langle b(z_1)\cdots b(z_6)\rangle_{\Sigma(\Omega_2)}\bigr|^2
\;|Z_{\mathrm{chiral}}(\Omega_2)|^{52}
}
$$

against the genus-3 modular expression

$$
\frac{
|\det S(\Omega_1)|^2 / |\chi_{18}(\Omega_1)|
}{
|\det S(\Omega_2)|^2 / |\chi_{18}(\Omega_2)|
}
=
\frac{|\det S(\Omega_1)|^2}{|\det S(\Omega_2)|^2}
\frac{|\chi_{18}(\Omega_2)|}{|\chi_{18}(\Omega_1)|}.
$$

Equivalently, the script prints

$$
\text{lhs}
=
\frac{|\langle b^6\rangle_1|^2}{|\langle b^6\rangle_2|^2}
\left(\frac{|Z_{\mathrm{chiral},1}|^2}{|Z_{\mathrm{chiral},2}|^2}\right)^{26}
$$

and

$$
\text{rhs}
=
\frac{|\det S_1|^2}{|\det S_2|^2}
\frac{|\chi_{18,2}|}{|\chi_{18,1}|}.
$$

## What the script does

1. Load a stored genus-3 one-face ribbon graph from
   [`python/genus3_t_duality.py`](../python/genus3_t_duality.py).
2. Reconstruct the rotation-system ribbon graph from the stored boundary data.
3. Fit shared genus-3 large-\(L\) coefficients \((\gamma,\alpha)\) for the
   renormalized determinant factor using
   `riemann_surface_tools.fit_genus_universal_aprime_coefficients(...)`.
4. For each of the two chosen edge-length lists:
   - compute the renormalized determinant finite part,
   - build the genus-3 surface data,
   - compute canonical `|Z_chiral|^2`, `|Z_1|^2`, and the naive chiral branch
     `Z_1 = +sqrt(|Z_1|^2)`.
5. Evaluate the genus-3 six-`b` correlator using the generic higher-genus
   `bc_correlator(...)` path in
   [`python/riemann_surface_tools.py`](../python/riemann_surface_tools.py),
   then multiply by the canonical sigma scale from
   `sigma_scale_from_z1(...)` so the result uses the canonically normalized
   higher-genus sigma.
6. Build the genus-3 \(6\times 6\) matrix \(S_I(z_j)\) from the six quadratic
   monomials

   $$
   \omega_1^2,\ \omega_1\omega_2,\ \omega_1\omega_3,\ \omega_2^2,\ \omega_2\omega_3,\ \omega_3^2
   $$

   and take its determinant.
7. Compute \(\chi_{18}\) as the product of the 36 even genus-3 theta constants.
8. Print the two sides of the squared ratio check and the resulting relative
   error.

## Important genericity condition

The auxiliary `divisor_points` must be disjoint from

- the six `b` insertion points,
- the anchor `b` points used in `sigma_scale_from_z1(...)`,
- the anchor `c` point used as the sigma normalization point.

During the first draft of this script, the default divisor accidentally
contained two of the actual `b` insertion points. That made individual
`sigma(z_i)` values blow up and completely spoiled the correlator check.
The Python helper now rejects such overlaps explicitly, and the defaults
have been changed to a safe divisor.

## Notes on conventions

- The script checks the **squared / absolute-value** relation, so the modular
  factor is \(1/|\chi_{18}|\), not the holomorphic square root \(1/\chi_9\).
- The \(S\)-matrix uses the generic nonhyperelliptic genus-3 basis coming from
  quadratic products of the three A-normalized holomorphic one-forms.
- The higher-genus sigma normalization is fixed canonically from the chosen
  chiral `Z_1` using the existing `lambda = 1`, `(n,m) = (g,1)` machinery.

## Usage

Default run:

```bash
./.venv/bin/python 'covariant formalism/python/genus3_bc_chi18_ratio_check.py' --nmax 4
```

The script now prints coarse progress messages because genus-3 surface
construction can take a while before the first numerical output appears.

## Debugging note and current status

During the first end-to-end run, the genus-3 check appeared to fail at the
10% level. The issue was **not** the genus-3 period matrix / Abel map /
Riemann-constant data.

The actual bug was that the original default `divisor_points`

```python
(0.21 + 0.09j, -0.16 + 0.12j, 0.07 - 0.15j)
```

accidentally included two of the actual `b` insertion points used in the check.
That made some intermediate `sigma(z_i)` values singular or numerically huge,
which in turn destroyed the canonical pure-`b^6` correlator.

This is now fixed in two ways:

1. `riemann_surface_tools.sigma_ratio(...)` explicitly rejects divisor points
   that coincide with either of the two evaluation points `z` or `w`.
2. `genus3_bc_chi18_ratio_check.py` now checks that the chosen
   `divisor_points` are disjoint from
   - the six `b` insertion points,
   - the anchor `b` points used in `sigma_scale_from_z1(...)`,
   - the anchor `c` point used as the sigma normalization point.

The safe default divisor is now

```python
(0.17 + 0.05j, -0.12 + 0.11j, 0.11 - 0.09j)
```

which is disjoint from all insertion and anchor data in the default check.

## Riemann-vanishing sanity check

Before fixing the correlator layer, the genus-3 surface data was tested
directly against the genus-3 Riemann-vanishing condition. On topology `1`,
for both moduli below, the values of

$$
\left|\theta\bigl(\zeta(p)+\zeta(q)-\Delta \mid \Omega\bigr)\right|
$$

and

$$
\left|\theta\bigl(2\zeta(p)-\Delta \mid \Omega\bigr)\right|
$$

were of order `10^-6`, while generic theta values on the same surface are
order `1`. So the genus-3 surface data itself appears healthy.

## Single pair result after the fix

Running

```bash
./.venv/bin/python 'covariant formalism/python/genus3_bc_chi18_ratio_check.py' --nmax 4
```

on topology `1` with

- modulus 1: `edge_lengths = [300] * 15`
- modulus 2: `edge_lengths = [250] * 14 + [700]`

now gives

- `lhs = 30.433832616810374`
- `rhs = 30.448796239717865`
- `relative error = 0.0004914356150464821`

So after removing the bad divisor choice, the squared genus-3 relation agrees
at the `5e-4` level rather than failing at the `1e-1` level.

## Additional modulus checks

Using the same topology `1`, the same six `b` insertion points, the same safe
divisor / anchor data, and the same theta cutoff `nmax = 4`, we compared the
base symmetric modulus

```python
M1 = (300,) * 15
```

against three additional moduli:

```python
M2 = (250,250,250,250,250,250,250,250,250,250,250,250,250,250,700)
M3 = (210,230,250,270,290,310,330,350,370,390,410,430,450,470,490)
M4 = (240,240,260,260,280,280,300,300,320,320,340,340,360,360,380)
```

The resulting pairwise ratio checks against `M1` were:

| pair | lhs | rhs | relative error |
| --- | ---: | ---: | ---: |
| `M1` vs `M2` | `30.433832616810374` | `30.448796239717865` | `4.91e-4` |
| `M1` vs `M3` | `0.5459654859398498` | `0.5456399523190048` | `5.96e-4` |
| `M1` vs `M4` | `0.5873311688742242` | `0.5870819094651516` | `4.24e-4` |

So on the tested topology and modulus range, the repaired genus-3 squared
relation is consistently matching at the `10^-4` to `10^-3` level.
