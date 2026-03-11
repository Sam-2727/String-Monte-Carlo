import unittest
from itertools import permutations, product

import Fast_Ribbon_Generator as rgg


class TestRibbonGraphGenerator(unittest.TestCase):
    @staticmethod
    def _label(kind, text):
        print(f"[{kind}] {text}")

    @staticmethod
    def _independent_next_half_edge(frm, to, eidx, edges, rotation):
        rot = rotation[to]
        nxt = rot[(rot.index(eidx) + 1) % len(rot)]
        a, b = edges[nxt]
        return (to, b, nxt) if a == to else (to, a, nxt)

    @classmethod
    def _independent_count_faces(cls, edges, rotation):
        visited = set()
        faces = 0
        for i, (a, b) in enumerate(edges):
            for he in ((a, b, i), (b, a, i)):
                if he in visited:
                    continue
                faces += 1
                cur = he
                while cur not in visited:
                    visited.add(cur)
                    cur = cls._independent_next_half_edge(*cur, edges, rotation)
        return faces

    @classmethod
    def _independent_valid_rotations(cls, edges, verts, target_faces):
        incident = {v: [] for v in verts}
        for i, (a, b) in enumerate(edges):
            incident[a].append(i)
            incident[b].append(i)

        choices = [list(permutations(incident[v])) for v in verts]
        valid = []
        for combo in product(*choices):
            rotation = {verts[i]: list(combo[i]) for i in range(len(verts))}
            if cls._independent_count_faces(edges, rotation) == target_faces:
                valid.append((edges, verts, rotation))
        return valid

    def test_expected_counts_for_known_cases(self):
        # Cases documented in ribbon_graph_generator.py main block.
        self._label("PRODUCTION", "Running generate_ribbon_graphs on known benchmark cases.")
        self.assertEqual(len(rgg.generate_ribbon_graphs(1, 3)), 1)
        self.assertEqual(len(rgg.generate_ribbon_graphs(3, 3)), 1)
        self.assertEqual(len(rgg.generate_ribbon_graphs(4, 6)), 1)
        self.assertEqual(len(rgg.generate_ribbon_graphs(1, 9)), 4)

    def test_face_count_matches_request(self):
        cases = [(1, 3), (3, 3), (4, 6), (1, 9)]
        for n_faces, n_edges in cases:
            with self.subTest(n_faces=n_faces, n_edges=n_edges):
                self._label(
                    "PRODUCTION",
                    f"Generating ribbon graphs for (F={n_faces}, E={n_edges}) and validating face count.",
                )
                rgs = rgg.generate_ribbon_graphs(n_faces, n_edges)
                for edges, verts, rotation in rgs:
                    self.assertEqual(
                        rgg._count_faces(edges, verts, rotation),  # noqa: SLF001
                        n_faces,
                    )

    def test_disc_boundary_invariants_for_f1_graphs(self):
        # F=1 cases where disc boundary representation applies.
        cases = [(1, 3), (1, 9)]
        for n_faces, n_edges in cases:
            with self.subTest(n_faces=n_faces, n_edges=n_edges):
                self._label(
                    "PRODUCTION",
                    f"Checking get_disc_boundary invariants for (F={n_faces}, E={n_edges}).",
                )
                rgs = rgg.generate_ribbon_graphs(n_faces, n_edges)
                for rg in rgs:
                    data = rgg.get_disc_boundary(rg)
                    boundary = data["boundary"]
                    sewing = data["sewing"]
                    ne = len(rg[0])

                    # Boundary should contain exactly one oriented half-edge per edge
                    # in an F=1 cubic ribbon graph.
                    self.assertEqual(len(boundary), 2 * ne)
                    self.assertEqual(len(data["edge_sequence"]), 2 * ne)
                    self.assertEqual(len(data["vertex_sequence"]), 2 * ne)

                    # Each edge appears exactly twice along the boundary.
                    self.assertEqual(len(sewing), ne)
                    for edge_label, positions in sewing.items():
                        self.assertTrue(1 <= edge_label <= ne)
                        self.assertEqual(len(positions), 2)
                        p1, p2 = positions
                        self.assertNotEqual(p1, p2)
                        self.assertTrue(1 <= p1 <= 2 * ne)
                        self.assertTrue(1 <= p2 <= 2 * ne)

    def test_no_base_graphs_for_unsupported_size(self):
        self._label("PRODUCTION", "Checking unsupported size returns empty list.")
        self.assertEqual(rgg.generate_ribbon_graphs(2, 5), [])

    def test_independent_enumeration_matches_generator_theta(self):
        # Independent rotation search on theta graph (V=2,E=3), then same
        # dedup routine as production code.
        self._label("BRUTE_FORCE", "Enumerating all local rotations on theta graph.")
        edges, verts = rgg._get_base_graphs(2, 3)[0]  # noqa: SLF001
        independent_all = self._independent_valid_rotations(edges, verts, 1)

        # Non-trivial check: exactly half of 3! * 3! = 36 rotations give F=1.
        self._label("BRUTE_FORCE", f"Found {len(independent_all)} valid rotations with F=1.")
        self.assertEqual(len(independent_all), 18)

        independent_unique = rgg._remove_isomorphisms(independent_all)  # noqa: SLF001
        self._label("PRODUCTION", "Running generate_ribbon_graphs(1, 3) for comparison.")
        generated = rgg.generate_ribbon_graphs(1, 3)
        self.assertEqual(len(independent_unique), len(generated))

    def test_independent_enumeration_matches_generator_k4_planar(self):
        # Independent search on K4 (V=4,E=6) for planar F=4 case.
        self._label("BRUTE_FORCE", "Enumerating all local rotations on K4 for F=4.")
        edges, verts = rgg._get_base_graphs(4, 6)[0]  # noqa: SLF001
        independent_all = self._independent_valid_rotations(edges, verts, 4)
        independent_unique = rgg._remove_isomorphisms(independent_all)  # noqa: SLF001
        self._label("PRODUCTION", "Running generate_ribbon_graphs(4, 6) for comparison.")
        generated = rgg.generate_ribbon_graphs(4, 6)
        self.assertEqual(len(independent_unique), len(generated))


if __name__ == "__main__":
    unittest.main()
