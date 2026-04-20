import argparse
import contextlib
import io
import itertools
import json
import os
from typing import List, Tuple

# Keep linear algebra deterministic and avoid thread oversubscription.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import mpmath as mp
import numpy as np

import ell_to_tau as et
import genus2_pair_candidate as g2pc
import partition_function as pf
import ribbon_graph_generator as rgg


Triple = Tuple[int, int, int]
Pair = Tuple[Triple, Triple]


def complementary_pairs_of_six() -> List[Pair]:
    verts = tuple(range(6))
    out = []
    # choose triples containing 0 to avoid double counting A|Ac and Ac|A
    for a, b in itertools.combinations(range(1, 6), 2):
        tri1 = tuple(sorted((0, a, b)))
        tri2 = tuple(sorted(tuple(v for v in verts if v not in tri1)))
        out.append((tri1, tri2))
    return out


def parse_ell_list(text: str) -> List[int]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError('Expected a JSON list of 9 integers.')
    return [int(x) for x in data]


def log_ratio_to_complex(log_num, log_ref):
    return complex(mp.e ** (log_num - log_ref))


def analyze_pair_family(graph, ell_ref, ell_num, theta_tol=1e-12):
    forms_ref = et.make_cyl_eqn_improved_higher_genus(graph, ell_ref)
    forms_num = et.make_cyl_eqn_improved_higher_genus(graph, ell_num)

    holo_ref = et.genus2_matter_bc_candidate(
        forms_ref,
        ribbon_graph=graph,
        ell_list=ell_ref,
        theta_tol=theta_tol,
    )
    holo_num = et.genus2_matter_bc_candidate(
        forms_num,
        ribbon_graph=graph,
        ell_list=ell_num,
        theta_tol=theta_tol,
    )
    holo_ratio = holo_num['candidate_factor'] / holo_ref['candidate_factor']

    pair_rows = []
    for pair in complementary_pairs_of_six():
        pref = g2pc.genus2_pair_candidate(
            forms_ref,
            ribbon_graph=graph,
            ell_list=ell_ref,
            pair=pair,
            theta_tol=theta_tol,
        )
        pnum = g2pc.genus2_pair_candidate(
            forms_num,
            ribbon_graph=graph,
            ell_list=ell_num,
            pair=pair,
            theta_tol=theta_tol,
        )
        ratio = pnum['candidate_factor'] / pref['candidate_factor']
        pair_rows.append({
            'pair': pair,
            'ratio': float(ratio),
            'ref_nu_factor': pref['nu_factor'],
            'num_nu_factor': pnum['nu_factor'],
        })

    return {
        'holo_only_ref': {
            'candidate_factor': holo_ref['candidate_factor'],
            'nu_factor': holo_ref['nu_factor'],
            'group_errors': holo_ref['group_errors'],
            'Omega': [[complex(x) for x in row] for row in holo_ref['Omega']],
        },
        'holo_only_num': {
            'candidate_factor': holo_num['candidate_factor'],
            'nu_factor': holo_num['nu_factor'],
            'group_errors': holo_num['group_errors'],
            'Omega': [[complex(x) for x in row] for row in holo_num['Omega']],
        },
        'holo_only_ratio': float(holo_ratio),
        'pair_candidates': pair_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph-index', type=int, default=1)
    ap.add_argument('--ref-ell', required=True, help='JSON list of 9 ints')
    ap.add_argument('--num-ell', required=True, help='JSON list of 9 ints')
    ap.add_argument('--theta-tol', type=float, default=1e-12)
    ap.add_argument('--json-out', default='')
    args = ap.parse_args()

    ell_ref = parse_ell_list(args.ref_ell)
    ell_num = parse_ell_list(args.num_ell)

    with contextlib.redirect_stdout(io.StringIO()):
        ribbon_graphs = rgg.generate_ribbon_graphs(1, 9)
    graph = ribbon_graphs[args.graph_index]

    psi1_ref = pf.traced_numeric_amplitude_log_psi1_f1(graph, ell_ref, matter_power=-13)
    psi1_num = pf.traced_numeric_amplitude_log_psi1_f1(graph, ell_num, matter_power=-13)
    psi1_ratio = log_ratio_to_complex(psi1_num, psi1_ref)

    square_ref = pf.traced_matter_bc_log_f1(graph, ell_ref, matter_power=-13)
    square_num = pf.traced_matter_bc_log_f1(graph, ell_num, matter_power=-13)
    square_ratio = log_ratio_to_complex(square_num, square_ref)

    analytic = analyze_pair_family(graph, ell_ref, ell_num, theta_tol=args.theta_tol)

    rows = []
    for row in analytic['pair_candidates']:
        ratio = row['ratio']
        err_psi1 = abs(psi1_ratio - ratio) / abs(ratio)
        err_square = abs(square_ratio - ratio) / abs(ratio)
        rows.append({
            **row,
            'rel_err_vs_psi1': float(err_psi1),
            'rel_err_vs_squareB': float(err_square),
        })
    rows_sorted_psi1 = sorted(rows, key=lambda r: r['rel_err_vs_psi1'])
    rows_sorted_square = sorted(rows, key=lambda r: r['rel_err_vs_squareB'])

    out = {
        'graph_index': args.graph_index,
        'ref_ell': ell_ref,
        'num_ell': ell_num,
        'numeric': {
            'psi1_ratio': psi1_ratio,
            'squareB_ratio': square_ratio,
        },
        'analytic_holomorphic_only': {
            'ratio': analytic['holo_only_ratio'],
            'ref': analytic['holo_only_ref'],
            'num': analytic['holo_only_num'],
            'rel_err_vs_psi1': float(abs(psi1_ratio - analytic['holo_only_ratio']) / abs(analytic['holo_only_ratio'])),
            'rel_err_vs_squareB': float(abs(square_ratio - analytic['holo_only_ratio']) / abs(analytic['holo_only_ratio'])),
        },
        'best_pairs_vs_psi1': rows_sorted_psi1[:10],
        'best_pairs_vs_squareB': rows_sorted_square[:10],
    }

    def cfmt(z):
        if isinstance(z, complex):
            return {'re': z.real, 'im': z.imag}
        return z

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return [sanitize(v) for v in obj]
        if isinstance(obj, complex):
            return cfmt(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    sout = sanitize(out)
    print(json.dumps(sout, indent=2))
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(sout, f, indent=2)


if __name__ == '__main__':
    main()
