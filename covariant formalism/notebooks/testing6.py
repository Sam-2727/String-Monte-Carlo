import sys, os
import partition_function as pf
import ell_to_tau as et
import importlib
importlib.reload(pf)
importlib.reload(et)
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from itertools import product


### Free fermion section ###

#running this function doesn't work for free fermion
def ddet_log_plot(k_start=1, k_end=20, k_step=2):
    k_vals = list(range(k_start, k_end + 1, k_step))
    sign_combos = list(product([False, True], repeat=3))

    # Numerical ratios: original (±i)
    orig_ratios = {s: [] for s in sign_combos}
    for k in k_vals:
        kp1 = k + 1
        for s in sign_combos:
            log_num = (pf.ddet_log(13 * 2 * kp1, 3 * kp1, 1 * kp1, signs=s))
            log_den = (pf.ddet_log(13 * 2 * kp1, 5 * kp1, 7 * kp1, signs=s))
            orig_ratios[s].append(float(mp.e ** ((log_num - log_den))))

    # Analytic: |θ_n/η|² ratio × bcorr
    spin_labels = {1: "R/R", 2: "R/NS", 3: "NS/NS", 4: "NS/R"}
    spin_colors = {1: "green", 2: "purple", 3: "red", 4: "orange"}
    analytic = {n: [] for n in [1, 2, 3, 4]}
    for k in k_vals:
        kp1 = k + 1
        f_num = et.make_cyl_eqn_improved(13 * 2 * kp1, 3 * kp1, 1 * kp1)
        f_den = et.make_cyl_eqn_improved(13 * 2 * kp1, 5 * kp1, 7 * kp1)
        b_num = et.pole_intercept_average(f_num, 13 * 2 * kp1, 3 * kp1, 1 * kp1)
        b_den = et.pole_intercept_average(f_den, 13 * 2 * kp1, 5 * kp1, 7 * kp1)
        bcorr = float(mp.e ** ( (mp.mpf(b_num) - mp.mpf(b_den))/ mp.mpf("12") ))
        for n in [1, 2, 3, 4]:
            a_num = et.theta3_eta_sqrt(13 * 2 * kp1, 3 * kp1, 1 * kp1, n=n)
            a_den = et.theta3_eta_sqrt(13 * 2 * kp1, 5 * kp1, 7 * kp1, n=n)
            analytic[n].append(float(abs(a_num / a_den)**2) * bcorr)

    # Plot: original ddet (±i) + analytic
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.tab10
    for idx, s in enumerate(sign_combos):
        label = "(±i) " + "(" + ",".join("+i" if b else "-i" for b in s) + ")"
        plt.plot(k_vals, orig_ratios[s], marker="o", linestyle="-", color=cmap(idx), label=label, markersize=4)
    for n in [1, 2, 3, 4]:
        plt.plot(k_vals, analytic[n], marker="x", linestyle="--", color=spin_colors[n], label=f"{spin_labels[n]} |θ/η|²", linewidth=2)
    plt.xlabel("k")
    plt.ylabel("ddet ratio")
    plt.title("Original: D with ±i signs, analytic = |θ_n/η|² ratio")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True)
    plt.ylim([0,5])
    plt.show()

    return np.array(k_vals), orig_ratios, analytic

kvals_d, orig, ana_orig = ddet_log_plot()


### Free boson and bc determinant section ###
# the following does work for the free boson:
def data1_log(k_start: int = 1, k_end: int = 40, k_step: int = 2):
    """
    Python analogue of:

    data1 = Table[
      Module[{numDet, denDet, rhsNum, rhsDen},
        numDet = combinedDet2[3(k+1), (k+1), 9(k+1)];
        denDet = combinedDet2[5(k+1), 7(k+1), (k+1)];
        rhsNum = computeRHS[3k, k, 9k, Round[3k/2], Round[k/2], Round[9k/2]];
        rhsDen = computeRHS[5k,7k,k, Round[5k/2],Round[7k/2],Round[k/2]];
        {
          numDet[[1]]/denDet[[1]],
          numDet[[2]]/denDet[[2]],
          rhsNum[[1]]/rhsDen[[1]],
          rhsNum[[2]]/rhsDen[[2]],
          rhsNum[[3]]/rhsDen[[3]],
          Exp[3/2*rhsNum[[4]]]/Exp[3/2*rhsDen[[4]]],
          rhsNum[[5]]/rhsDen[[5]],
          rhsNum[[6]],
          rhsNum[[7]]
        }
      ],
      {k,1,30,2}
    ];

    Here combined_det2_log returns (log_bdet, log_pdet).
    """
    rows = []
    k_vals = list(range(k_start, k_end + 1, k_step))

    for k in k_vals:
        kp1 = k + 1

        # ----- determinant side in log-space -----
        # numDet = combinedDet2[3(k+1), (k+1), 9(k+1)]
        log_num_b, log_num_p = pf.combined_det2_log(13 * 2*kp1, 3 * kp1, 1 * kp1)

        # denDet = combinedDet2[5(k+1), 7(k+1), (k+1)]
        log_den_b, log_den_p = pf.combined_det2_log(13 * 2*kp1, 5 * kp1, 7* kp1)

        # ratios
        bdet_ratio = mp.e ** (log_num_b - log_den_b)
        pdet_ratio = mp.e ** (log_num_p - log_den_p)

        # ----- RHS side -----
        # rhsNum = computeRHS[3k,k,9k, Round[3k/2], Round[k/2], Round[9k/2]]
        L_num = 2 * (3*k + 1*k + 9*k)  # = 26k
        rhs_num = et.compute_rhs(
            L=L_num, l1=3*k, l2=1*k)

        # rhsDen = computeRHS[5k,7k,k, Round[5k/2], Round[7k/2], Round[k/2]]
        L_den = 2 * (5*k + 7*k + 1*k)  # = 26k
        rhs_den = et.compute_rhs(
            L=L_den, l1=5*k, l2=7*k )

        # unpack (Python indices)
        f0_num, eta_num, im_num, b_num, jac_num, tau_num, per1_num = rhs_num
        f0_den, eta_den, im_den, b_den, jac_den, tau_den, per1_den = rhs_den
        # Mathematica requested outputs:
        # rhsNum[[1]]/rhsDen[[1]] = fZero^{-2} ratio
        f0_ratio  = f0_num / f0_den

        # rhsNum[[2]]/rhsDen[[2]] = eta ratio
        eta_ratio = eta_num / eta_den

        # rhsNum[[3]]/rhsDen[[3]] = Im(tau) ratio
        im_ratio  = im_num / im_den
        # Exp[3/2*rhsNum[[4]]]/Exp[3/2*rhsDen[[4]]] = exp(3/2*(b_num-b_den))
        bexp_ratio = mp.e ** (mp.mpf("1.5") * (mp.mpf(b_num) - mp.mpf(b_den)))

        # rhsNum[[5]]/rhsDen[[5]] = jacobian ratio
        jac_ratio = jac_num / jac_den

        # rhsNum[[6]] = tau_num
        # rhsNum[[7]] = per1_num

        row = [
            complex(bdet_ratio),     # numDet[[1]]/denDet[[1]]
            complex(pdet_ratio),     # numDet[[2]]/denDet[[2]]
            complex(f0_ratio),       # rhsNum[[1]]/rhsDen[[1]]
            complex(eta_ratio),      # rhsNum[[2]]/rhsDen[[2]]
            complex(im_ratio),       # rhsNum[[3]]/rhsDen[[3]]
            complex(bexp_ratio),     # exp(3/2*(b_num-b_den))
            complex(jac_ratio),      # rhsNum[[5]]/rhsDen[[5]]
            complex(tau_num),        # rhsNum[[6]]
            complex(per1_num),       # rhsNum[[7]]
        ]
        rows.append(row)

    return np.array(k_vals, dtype=int), np.array(rows, dtype=np.complex128)

# This is a guess for the matter determinant
kvals,data1=data1_log()
eq79MatterPart = (np.abs(data1[:, 4]) ** (-13.0)) * \
             (np.abs(data1[:, 3]) ** (-52.0)) * \
             (data1[:, 5] ** (26.0/18.0))

y = np.abs(eq79MatterPart)  
matterDet = data1[:,1]
y2 = np.abs(matterDet)

plt.figure()
plt.plot(kvals, y, marker="o", linestyle="-")
plt.plot(kvals, y2, marker="o", linestyle="-",color="orange")
plt.xlabel("k")
plt.ylabel("eq79Guess1 (real part)")
plt.title("eq79Guess1 vs k")
plt.grid(True)
plt.ylim({2.8,3.7})
plt.show()

kVals = np.arange(1, 40, 2)

## Now test combined bc ghost and matter determinant

def test_formula5(l1_scaling_num,l2_scaling_num,l1_scaling_den,l2_scaling_den,L,kvals_range,bi_exponent):
        L_formula_mult = L
        l3_scaling_num = L // 2 - l1_scaling_num - l2_scaling_num
        l3_scaling_den = L // 2 - l1_scaling_den - l2_scaling_den
        kvals = list(kvals_range)
        formulas = []
        dets = []
    
        for k in kvals:
                kp1 = k + 1                
                L_formula = L_formula_mult * k
                # Denominator partition (reference)
                l1_den_f = l1_scaling_den * k
                l2_den_f = l2_scaling_den * k
                l3_den_f = l3_scaling_den * k                
                P1_ref, P2_ref, P3_ref = et.periods_improved(L_formula, l1_den_f, l2_den_f)
                tau_ref = P2_ref / P1_ref
                eta_ref = et.dedekind_eta(tau_ref)
                bs_ref = et.calculate_b(L_formula, l1_den_f, l2_den_f)
                bs_arr_ref = np.array(et.average_b(L_formula, l1_den_f, l2_den_f, bs_ref))
                theta_ref = [2 * np.pi * li / L_formula for li in [l1_den_f, l2_den_f, l3_den_f]]
                trig_ref = 2 + 2 * sum(np.cos(np.pi - 2 * th) for th in theta_ref)
                # I take the 1/abs(period)^2 here because it has the same scaling as f as a function of \tau, which is normalized by the period such that f(0)/period=1/period
                ref_terms = (
                        1 / abs(P1_ref)**2,
                        abs(np.mean(bs_arr_ref**2))**2,
                        np.mean(np.abs(bs_arr_ref))**bi_exponent,
                        tau_ref.imag**(-13),
                        abs(eta_ref)**(-48)
                        )
                
                l1_num_f = l1_scaling_num * k
                l2_num_f = l2_scaling_num * k
                l3_num_f = l3_scaling_num * k
                P1_num, P2_num, P3_num = et.periods_improved(L_formula, l1_num_f, l2_num_f)
                tau_num = P2_num / P1_num
                eta_num = et.dedekind_eta(tau_num)
                # this is a better version of the calculation of b that converges quicker in the L\to\infty limit
                # You should use this version when computing the b_i/nu_i as it is more careful with the phases
                # don't use the pole_intercept_average in data1_log as it only calculates absolute values.
                bs_num = et.calculate_b(L_formula, l1_num_f, l2_num_f)
                bs_arr_num = np.array(et.average_b(L_formula, l1_num_f, l2_num_f, bs_num))
                theta_num = [2 * np.pi * li / L_formula for li in [l1_num_f, l2_num_f, l3_num_f]]
                trig_num = 2 + 2 * sum(np.cos(2 * np.pi / 3 - 2 * th) for th in theta_num)

                num_terms = (
                        1 / abs(P1_num)**2,
                        abs(np.mean(bs_arr_num**2))**2,
                        np.mean(np.abs(bs_arr_num))**bi_exponent,
                        tau_num.imag**(-13),
                        abs(eta_num)**(-48)
                        )
                formula_ratio = np.prod([num_terms[i] / ref_terms[i] for i in range(5)])
                L_det = L * kp1
                
                l1_den_d = l1_scaling_den * kp1
                l2_den_d = l2_scaling_den * kp1
                log_bdet_ref, log_prime_det_ref = pf.combined_det2_log(L_det, l1_den_d, l2_den_d)
                
                l1_num_d = l1_scaling_num * kp1
                l2_num_d = l2_scaling_num * kp1
                log_bdet_num, log_prime_det_num = pf.combined_det2_log(L_det, l1_num_d, l2_num_d)
                
                log_det_ratio = (log_bdet_num - log_bdet_ref) + (log_prime_det_num - log_prime_det_ref)
                det_product = float(mp.e ** log_det_ratio)
                formulas.append(formula_ratio)
                dets.append(det_product)
    
        return {'kvals': kvals, 'formula': formulas, 'det': dets}    
results5 = test_formula5(3, 1, 5, 7, 26, range(1, 21, 2), bi_exponent=9/2)

print(f"{'k':>4}  {'formula':>14}  {'det':>14}  {'formula/det':>14}")
print("-" * 52)
for k, f, d in zip(results5['kvals'], results5['formula'], results5['det']):
    print(f"{k:4d}  {f:14.8f}  {d:14.8f}  {f/d:14.10f}")