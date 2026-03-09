"""
Diagnostic script for free fermion determinant mismatch.
All comparisons done as ratios of two moduli points in log space.
"""
import numpy as np
import mpmath as mp
import partition_function as pf
import ell_to_tau as et
from itertools import product

mp.mp.dps = 50

def get_tau(L, l1, l2):
    P1, P2, P3 = et.periods_improved(L, l1, l2)
    return P2 / P1

def theta_over_eta(tau, n=3):
    tau_mp = mp.mpc(tau)
    nome = mp.exp(mp.pi * 1j * tau_mp)
    theta_n = mp.jtheta(n, 0, nome)
    eta = mp.mpc(et.dedekind_eta(tau))
    return complex(theta_n / eta)

sign_combos = list(product([False, True], repeat=3))
sign_labels = {s: "(" + ",".join("+i" if b else "-i" for b in s) + ")" for s in sign_combos}

# Two moduli points (numerator and denominator) for the ratio test
# Using the same convention as in the notebook: (3k, k, 9k) / (5k, 7k, k)
print("="*80)
print("FERMION RATIO TEST: ddet_log(num) - ddet_log(den) vs analytic ratio")
print("All ratios are |det D|^2(num) / |det D|^2(den), with 64^L cancelling")
print("="*80)

k_vals = [3, 5, 7, 9, 11, 15, 19, 25, 31]

# Collect data for convergence analysis
data_by_sign = {s: [] for s in sign_combos}
data_by_sign_guess = {s: [] for s in sign_combos}  # ±1 signs
analytic_data = {n: [] for n in [2, 3, 4]}

for kk in k_vals:
    kp1 = kk + 1
    L = 13 * 2 * kp1

    l1_num, l2_num = 3 * kp1, 1 * kp1
    l1_den, l2_den = 5 * kp1, 7 * kp1

    tau_num = get_tau(L, l1_num, l2_num)
    tau_den = get_tau(L, l1_den, l2_den)

    # Weyl correction (same as notebook)
    f_num = et.make_cyl_eqn_improved(L, l1_num, l2_num)
    f_den = et.make_cyl_eqn_improved(L, l1_den, l2_den)
    b_num = et.pole_intercept_average(f_num, L, l1_num, l2_num)
    b_den = et.pole_intercept_average(f_den, L, l1_den, l2_den)

    # Numerical: ±i signs
    for s in sign_combos:
        log_num = pf.ddet_log(L, l1_num, l2_num, signs=s)
        log_den = pf.ddet_log(L, l1_den, l2_den, signs=s)
        ratio = float(mp.exp(log_num - log_den))
        data_by_sign[s].append(ratio)

    # Numerical: ±1 signs (ddet_log_guess)
    for s in sign_combos:
        log_num = pf.ddet_log_guess(L, l1_num, l2_num, signs=s)
        log_den = pf.ddet_log_guess(L, l1_den, l2_den, signs=s)
        ratio = float(mp.exp(log_num - log_den))
        data_by_sign_guess[s].append(ratio)

    # Analytic: theta_n/eta ratios (skip n=1 since theta_1=0)
    for n in [2, 3, 4]:
        te_num = abs(theta_over_eta(tau_num, n))
        te_den = abs(theta_over_eta(tau_den, n))
        # Try different Weyl exponents
        analytic_data[n].append({
            'raw': te_num / te_den,
            'weyl_1_24': te_num / te_den * float(mp.exp((mp.mpf(b_num) - mp.mpf(b_den)) / 24)),
            'weyl_neg_1_24': te_num / te_den * float(mp.exp(-(mp.mpf(b_num) - mp.mpf(b_den)) / 24)),
            'weyl_1_12': te_num / te_den * float(mp.exp((mp.mpf(b_num) - mp.mpf(b_den)) / 12)),
            'b_num': b_num,
            'b_den': b_den,
        })

# Print convergence table
print(f"\n{'k':>4s}", end="")
for s in sign_combos:
    print(f"  {sign_labels[s]:>14s}", end="")
print()
print("-" * (4 + 16 * 8))
for i, kk in enumerate(k_vals):
    print(f"{kk:4d}", end="")
    for s in sign_combos:
        print(f"  {data_by_sign[s][i]:14.8f}", end="")
    print()

print(f"\nAnalytic targets (no Weyl correction):")
print(f"  theta_2 (R-NS):  ", [f"{analytic_data[2][i]['raw']:.8f}" for i in range(len(k_vals))])
print(f"  theta_3 (NS-NS): ", [f"{analytic_data[3][i]['raw']:.8f}" for i in range(len(k_vals))])
print(f"  theta_4 (NS-R):  ", [f"{analytic_data[4][i]['raw']:.8f}" for i in range(len(k_vals))])

print(f"\nAnalytic targets (Weyl +1/24):")
print(f"  theta_2 (R-NS):  ", [f"{analytic_data[2][i]['weyl_1_24']:.8f}" for i in range(len(k_vals))])
print(f"  theta_3 (NS-NS): ", [f"{analytic_data[3][i]['weyl_1_24']:.8f}" for i in range(len(k_vals))])
print(f"  theta_4 (NS-R):  ", [f"{analytic_data[4][i]['weyl_1_24']:.8f}" for i in range(len(k_vals))])

# Now check: which numerical curves are converging to which analytic values?
print("\n" + "="*80)
print("CONVERGENCE CHECK: largest k value comparison")
print("="*80)
i_last = -1  # last k value
print(f"k = {k_vals[i_last]}")
print(f"\nNumerical (±i signs):")
for s in sign_combos:
    print(f"  {sign_labels[s]:20s}: {data_by_sign[s][i_last]:.8f}")

print(f"\nNumerical (±1 signs, ddet_log_guess):")
for s in sign_combos:
    print(f"  {sign_labels[s]:20s}: {data_by_sign_guess[s][i_last]:.8f}")

print(f"\nAnalytic (raw, no Weyl):")
for n in [2, 3, 4]:
    labels = {2: "theta_2 R-NS", 3: "theta_3 NS-NS", 4: "theta_4 NS-R"}
    print(f"  {labels[n]:20s}: {analytic_data[n][i_last]['raw']:.8f}")

print(f"\nAnalytic (Weyl +1/24):")
for n in [2, 3, 4]:
    labels = {2: "theta_2 R-NS", 3: "theta_3 NS-NS", 4: "theta_4 NS-R"}
    print(f"  {labels[n]:20s}: {analytic_data[n][i_last]['weyl_1_24']:.8f}")

print(f"\nAnalytic (Weyl -1/24):")
for n in [2, 3, 4]:
    labels = {2: "theta_2 R-NS", 3: "theta_3 NS-NS", 4: "theta_4 NS-R"}
    print(f"  {labels[n]:20s}: {analytic_data[n][i_last]['weyl_neg_1_24']:.8f}")

print(f"\nAnalytic (Weyl +1/12):")
for n in [2, 3, 4]:
    labels = {2: "theta_2 R-NS", 3: "theta_3 NS-NS", 4: "theta_4 NS-R"}
    print(f"  {labels[n]:20s}: {analytic_data[n][i_last]['weyl_1_12']:.8f}")

# Check symmetry: pairs that should give the same spin structure
print("\n" + "="*80)
print("SYMMETRY CHECK: sign combos that should give same spin structure")
print("="*80)
pairs = [
    ("FFF", "TTT", (False,False,False), (True,True,True)),
    ("FFT", "TTF", (False,False,True), (True,True,False)),
    ("FTF", "TFT", (False,True,False), (True,False,True)),
    ("FTT", "TFF", (False,True,True), (True,False,False)),
]
for label1, label2, s1, s2 in pairs:
    ratios_equal = [abs(data_by_sign[s1][i] - data_by_sign[s2][i]) < 1e-6
                    for i in range(len(k_vals))]
    print(f"  {sign_labels[s1]} vs {sign_labels[s2]}: "
          f"ratios equal? {all(ratios_equal)} "
          f"(vals: {data_by_sign[s1][-1]:.8f} vs {data_by_sign[s2][-1]:.8f})")

# Check: how many distinct numerical values are there?
print("\n" + "="*80)
print("DISTINCT NUMERICAL VALUES at largest k")
print("="*80)
vals = [(s, data_by_sign[s][-1]) for s in sign_combos]
vals.sort(key=lambda x: x[1])
for s, v in vals:
    print(f"  {sign_labels[s]:20s}: {v:.10f}")
