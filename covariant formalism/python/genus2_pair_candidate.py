import numpy as np
import ell_to_tau as et


def genus2_pair_candidate(forms, *, ribbon_graph, ell_list, pair=((0, 3, 5), (1, 2, 4)), averaging_method='overlap', chi10_normalization='product', theta_nmax=None, theta_tol=1e-12):
    """Improved genus-2 analytic candidate found empirically for the vacuum amplitude.

    pair is a complementary pair of 0-based triples of the six averaged surface vertices.
    The default pair corresponds to surface vertices {1,4,6} and {2,3,5} in 1-based labels.
    """
    norm_data = et.normalize_holomorphic_forms(
        forms,
        ribbon_graph=ribbon_graph,
        ell_list=ell_list,
        return_data=True,
    )
    norm_forms = norm_data['normalized_forms']
    Omega = np.asarray(norm_data['Omega'], dtype=np.complex128)

    raw = et.calculate_nu(forms=norm_forms, ribbon_graph=ribbon_graph, ell_list=ell_list, return_data=True)
    avg = et.average_nu(nus=raw, method=averaging_method, return_data=True)
    averaged_nu = np.asarray(avg['averaged_nu'], dtype=np.complex128)
    triple_dets = et.triple_determinants_from_nu(averaged_nu)

    tri1, tri2 = tuple(pair[0]), tuple(pair[1])
    if tri1 not in triple_dets or tri2 not in triple_dets:
        raise ValueError(f"Requested pair {pair} is not available. Available triples are {sorted(triple_dets)}")
    nu_factor = 0.5 * (abs(triple_dets[tri1]) ** 2 + abs(triple_dets[tri2]) ** 2)

    im_omega = np.asarray(np.imag(Omega), dtype=np.float64)
    chi10 = et.igusa_chi10_genus2(Omega, nmax=theta_nmax, tol=theta_tol, normalization=chi10_normalization)
    modular_factor = float(np.linalg.det(im_omega)) ** (-13.0) * abs(chi10) ** (-2.0)

    out = dict(norm_data)
    out.update({
        'raw_nu': raw['nu_matrix'],
        'averaged_nu': averaged_nu,
        'triple_determinants': triple_dets,
        'nu_factor': float(nu_factor),
        'selected_pair': (tri1, tri2),
        'candidate_factor': float(modular_factor * nu_factor),
    })
    return out
