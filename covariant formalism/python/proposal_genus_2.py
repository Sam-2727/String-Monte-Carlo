import contextlib
import io

import mpmath as mp
import numpy as np

import partition_function as pf
import ribbon_graph_generator as rgg
import ell_to_tau as et
import genus2_pair_candidate as g2pc


def analytic_factor(graph, ell_list, pair=((0, 3, 5), (1, 2, 4)), theta_tol=1e-12):
    forms = et.make_cyl_eqn_improved_higher_genus(graph, ell_list)
    data = g2pc.genus2_pair_candidate(
        forms,
        ribbon_graph=graph,
        ell_list=ell_list,
        pair=pair,
        theta_tol=theta_tol,
    )
    return data


def numeric_log_amplitude(graph, ell_list):
    # Corrected matter exponent for matter + bc.
    return pf.traced_numeric_amplitude_log_psi1_f1(
        graph,
        ell_list,
        matter_power=-13,
    )


def main():
    # The benchmark in genus2_benchmark_results.txt uses ribbon_graphs[1].
    with contextlib.redirect_stdout(io.StringIO()):
        ribbon_graphs = rgg.generate_ribbon_graphs(1, 9)
    graph = ribbon_graphs[1]

    ell_ref = [48, 40, 44, 44, 48, 40, 44, 44, 44]
    ell_cases = [
        [52, 36, 44, 44, 52, 36, 44, 44, 44],
        [56, 32, 44, 44, 56, 32, 44, 44, 44],
        [60, 28, 44, 44, 60, 28, 44, 44, 44],
    ]

    pair = ((0, 3, 5), (1, 2, 4))

    print("Using graph index 1 and pair", pair)
    print("ref ell =", ell_ref)

    ref_analytic = analytic_factor(graph, ell_ref, pair=pair)
    ref_numeric_log = numeric_log_amplitude(graph, ell_ref)

    print("reference candidate_factor =", ref_analytic["candidate_factor"])
    print("reference selected_pair   =", ref_analytic["selected_pair"])
    print("reference nu_factor       =", ref_analytic["nu_factor"])
    print("reference Omega =")
    print(ref_analytic["Omega"])
    print()

    for i, ell_num in enumerate(ell_cases, start=1):
        num_analytic = analytic_factor(graph, ell_num, pair=pair)
        num_numeric_log = numeric_log_amplitude(graph, ell_num)

        analytic_ratio = num_analytic["candidate_factor"] / ref_analytic["candidate_factor"]
        numeric_ratio = complex(mp.e ** (num_numeric_log - ref_numeric_log))
        rel_diff = abs(numeric_ratio - analytic_ratio) / abs(analytic_ratio)

        print(f"case {i}")
        print("  num ell        =", ell_num)
        print("  analytic ratio =", analytic_ratio)
        print("  numeric ratio  =", numeric_ratio)
        print("  rel diff       =", rel_diff)
        print("  candidate_factor(num) =", num_analytic["candidate_factor"])
        print("  nu_factor(num)        =", num_analytic["nu_factor"])
        print()


if __name__ == "__main__":
    main()