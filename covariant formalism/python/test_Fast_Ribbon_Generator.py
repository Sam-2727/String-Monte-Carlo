import unittest
from itertools import permutations, product

import Fast_Ribbon_Generator as rgg


class TestRibbonGraphGenerator(unittest.TestCase):
    @staticmethod
    def _label(kind, text):
        print(f"[{kind}] {text}")

    def _start_test(self, name, content, expected):
        print(f"\n[TEST] {name}")
        print(f"[CONTENT] {content}")
        print(f"[EXPECTED/CROSS-CHECK] {expected}")

    def _pass_test(self, name):
        print(f"[RESULT] PASS - {name}")

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

    @staticmethod
    def _print_illustration_from_generator_data(rg, idx=1):
        edges, verts, rotation = rg
        n_faces = rgg._count_faces(edges, verts, rotation)  # noqa: SLF001
        print(f"\nGraph {idx}: V={len(verts)}, E={len(edges)}, F={n_faces}")
        print(f"Edges: {[(i + 1, a, b) for i, (a, b) in enumerate(edges)]}")

        if n_faces != 1:
            return

        data = rgg.get_disc_boundary(rg)
        boundary = data["boundary"]
        sewing = data["sewing"]
        segments = [
            (i + 1, frm, to, eidx + 1) for i, (frm, to, eidx) in enumerate(boundary)
        ]
        sewing_pairs = [(e, p1, p2) for e, (p1, p2) in sorted(sewing.items())]

        print(f"Segments (seg, from, to, edge): {segments}")
        print(f"Sewing pairs (edge, seg1, seg2): {sewing_pairs}")

    def test_expected_counts_for_known_cases(self):
        # Content: benchmark cardinalities for known small cases.
        # Expected/cross-check: counts match documented production values.
        # Pass condition: all count equalities hold.
        self._start_test(
            "test_expected_counts_for_known_cases",
            "Check known non-isomorphic graph counts for (F,E) in {(1,3),(3,3),(4,6),(1,9)}.",
            "Expected counts are {1,1,1,4}; cross-check against current generator output.",
        )
        # Cases documented in ribbon_graph_generator.py main block.
        self._label("PRODUCTION", "Running generate_ribbon_graphs on known benchmark cases.")
        self.assertEqual(len(rgg.generate_ribbon_graphs(1, 3)), 1)
        self.assertEqual(len(rgg.generate_ribbon_graphs(3, 3)), 1)
        self.assertEqual(len(rgg.generate_ribbon_graphs(4, 6)), 1)
        self.assertEqual(len(rgg.generate_ribbon_graphs(1, 9)), 4)
        self._pass_test("test_expected_counts_for_known_cases")

    def test_face_count_matches_request(self):
        # Content: every generated ribbon graph should realize requested face count.
        # Expected/cross-check: recomputed faces from traversal equals input F.
        # Pass condition: equality holds for all generated graphs in each case.
        self._start_test(
            "test_face_count_matches_request",
            "Recount faces on each generated ribbon graph for several (F,E) inputs.",
            "Independent recount via _count_faces matches requested F exactly.",
        )
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
        self._pass_test("test_face_count_matches_request")

    def test_disc_boundary_invariants_for_f1_graphs(self):
        # Content: F=1 boundary object has coherent combinatorics.
        # Expected/cross-check: boundary length, per-edge multiplicity, and positions are valid.
        # Pass condition: all invariants hold for all generated F=1 graphs.
        self._start_test(
            "test_disc_boundary_invariants_for_f1_graphs",
            "Validate boundary/sewing invariants for one-face ribbon graphs.",
            "Boundary has 2E segments and each edge appears exactly twice with valid positions.",
        )
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
        self._pass_test("test_disc_boundary_invariants_for_f1_graphs")

    def test_face_boundary_invariants_for_multi_face_graphs(self):
        # Content: generalized face-boundary payload for F>1 graphs.
        # Expected/cross-check: total boundary length is 2E and each edge appears twice globally.
        # Pass condition: all generalized face/gluing invariants hold.
        self._start_test(
            "test_face_boundary_invariants_for_multi_face_graphs",
            "Validate generalized face-boundary/gluing payload for multi-face ribbon graphs.",
            "Across all faces, the total segment count is 2E and every edge appears exactly twice.",
        )
        cases = [(1, 2), (1, 3), (2, 2)]
        for genus, n_faces in cases:
            with self.subTest(genus=genus, n_faces=n_faces):
                self._label(
                    "PRODUCTION",
                    f"Checking generalized face-boundary invariants for (g={genus}, F={n_faces}).",
                )
                rgs = rgg.generate_ribbon_graphs_fixed_genus(genus, n_faces=n_faces)
                for rg in rgs[:5]:
                    data = rgg.get_face_boundary_data(rg)
                    ne = len(rg[0])
                    self.assertEqual(len(data["face_boundaries"]), n_faces)
                    self.assertEqual(sum(len(face) for face in data["face_boundaries"]), 2 * ne)
                    self.assertEqual(len(data["face_edge_sequences"]), n_faces)
                    self.assertEqual(len(data["face_vertex_sequences"]), n_faces)
                    self.assertEqual(len(data["gluing"]), ne)
                    self.assertEqual(len(data["gluing_pairs"]), ne)

                    for edge_label, positions in data["gluing"].items():
                        self.assertTrue(1 <= edge_label <= ne)
                        self.assertEqual(len(positions), 2)
                        for face_idx, pos_idx in positions:
                            self.assertTrue(1 <= face_idx <= n_faces)
                            self.assertTrue(1 <= pos_idx <= len(data["face_boundaries"][face_idx - 1]))

                    for face, edge_seq, vert_seq in zip(
                        data["face_boundaries"],
                        data["face_edge_sequences"],
                        data["face_vertex_sequences"],
                    ):
                        self.assertEqual(len(face), len(edge_seq))
                        self.assertEqual(len(face), len(vert_seq))
                        self.assertEqual(
                            tuple(edge for _, _, edge in face),
                            edge_seq,
                        )
                        self.assertEqual(
                            tuple(frm for frm, _, _ in face),
                            vert_seq,
                        )
        self._pass_test("test_face_boundary_invariants_for_multi_face_graphs")

    def test_get_disc_boundary_rejects_non_f1_graphs(self):
        # Content: disc-boundary helper should reject multi-face inputs.
        # Expected/cross-check: ValueError is raised for F>1 ribbon graphs.
        # Pass condition: exception is raised.
        self._start_test(
            "test_get_disc_boundary_rejects_non_f1_graphs",
            "Call get_disc_boundary on a multi-face ribbon graph.",
            "Cross-check the helper raises ValueError instead of silently returning one face.",
        )
        self._label("PRODUCTION", "Checking get_disc_boundary rejects non-F=1 inputs.")
        rg = rgg.generate_ribbon_graphs_fixed_genus(1, n_faces=2)[0]
        with self.assertRaises(ValueError):
            rgg.get_disc_boundary(rg)
        self._pass_test("test_get_disc_boundary_rejects_non_f1_graphs")

    def test_no_base_graphs_for_unsupported_size(self):
        # Content: unsupported (F,E) should return no graphs.
        # Expected/cross-check: generation returns empty list.
        # Pass condition: exact empty list.
        self._start_test(
            "test_no_base_graphs_for_unsupported_size",
            "Ask generator for unsupported cubic size.",
            "Cross-check output is [].",
        )
        self._label("PRODUCTION", "Checking unsupported size returns empty list.")
        self.assertEqual(rgg.generate_ribbon_graphs(2, 5), [])
        self._pass_test("test_no_base_graphs_for_unsupported_size")

    def test_ribbon_graph_report_contains_expected_pieces_for_f1(self):
        # Content: structured report includes required fields for F=1.
        # Expected/cross-check: report contains edges/rotation/faces/boundary and coherent sewing half-edges.
        # Pass condition: all field-level and consistency assertions pass.
        self._start_test(
            "test_ribbon_graph_report_contains_expected_pieces_for_f1",
            "Build detailed report for theta F=1 case and validate structural fields.",
            "Cross-check edge labels, boundary section existence, and opposite-orientation sewing pairs.",
        )
        self._label("PRODUCTION", "Validating full ribbon graph report for an F=1 case.")
        rg = rgg.generate_ribbon_graphs(1, 3)[0]
        report = rgg.get_ribbon_graph_report(rg)

        # Core ribbon graph structure is present.
        self.assertEqual(len(report["edges"]), 3)
        self.assertEqual(set(report["rotation_system"]), {1, 2})
        self.assertEqual(len(report["faces"]), 1)

        # F=1-specific boundary and sewing data is present.
        self.assertIsNotNone(report["boundary"])
        self.assertEqual(len(report["sewn_half_edge_pairs"]), 3)

        # Each sewn pair references the same edge label on both half-edges.
        seen_edges = set()
        for pair in report["sewn_half_edge_pairs"]:
            edge = pair["edge"]
            seen_edges.add(edge)
            he1, he2 = pair["half_edges"]
            self.assertEqual(he1[2], edge)
            self.assertEqual(he2[2], edge)
            self.assertNotEqual(pair["segments"][0], pair["segments"][1])
            # Sewing must pair opposite orientations of the same edge.
            self.assertEqual((he1[0], he1[1]), (he2[1], he2[0]))

        self.assertEqual(seen_edges, {1, 2, 3})
        self._pass_test("test_ribbon_graph_report_contains_expected_pieces_for_f1")

    def test_ribbon_graph_report_boundary_only_for_f1(self):
        # Content: non-F=1 report should not include F=1 boundary sewing payload.
        # Expected/cross-check: boundary is None and sewn_half_edge_pairs is empty.
        # Pass condition: exact values match.
        self._start_test(
            "test_ribbon_graph_report_boundary_only_for_f1",
            "Generate non-F=1 case and inspect report boundary section behavior.",
            "Cross-check report['boundary'] is None and no sewn_half_edge_pairs are emitted.",
        )
        self._label("PRODUCTION", "Checking non-F=1 report omits boundary sewing section.")
        rg = rgg.generate_ribbon_graphs(4, 6)[0]
        report = rgg.get_ribbon_graph_report(rg)
        self.assertIsNone(report["boundary"])
        self.assertEqual(report["sewn_half_edge_pairs"], [])
        self._pass_test("test_ribbon_graph_report_boundary_only_for_f1")

    def test_illustration_output_lives_in_tests(self):
        # Content: print essential data examples from test context only.
        # Expected/cross-check: each printed object has edges and sewing_pairs keys.
        # Pass condition: generation and essential extraction succeed for selected cases.
        self._start_test(
            "test_illustration_output_lives_in_tests",
            "Emit sample essential output rows (edges + sewing_pairs) for small F=1 cases.",
            "Cross-check get_graph_and_sewing_data returns expected keys without runtime errors.",
        )
        self._label("ILLUSTRATION", "Printing essential edges/sewing data from generator.")
        cases = [(1, 3), (1, 9)]
        for n_faces, n_edges in cases:
            with self.subTest(n_faces=n_faces, n_edges=n_edges):
                rgs = rgg.generate_ribbon_graphs(n_faces, n_edges)
                for i, rg in enumerate(rgs, start=1):
                    data = rgg.get_graph_and_sewing_data(rg)
                    print(f"\nGraph {i}: edges={data['edges']}")
                    print(f"Sewing pairs={data['sewing_pairs']}")
        self._pass_test("test_illustration_output_lives_in_tests")

    def test_essential_data_only_for_genus_3(self):
        # Content: genus-3 essential output path.
        # Expected/cross-check: each sample has 15 edges and 15 sewing pairs for (g=3,F=1).
        # Pass condition: all sampled payloads satisfy essential schema/size checks.
        self._start_test(
            "test_essential_data_only_for_genus_3",
            "Generate essential payloads for fixed genus 3 and one face.",
            "Cross-check edge and sewing counts are 15 for sampled outputs.",
        )
        self._label(
            "PRODUCTION",
            "Genus-3 check: generate all F=1 ribbon graphs and keep only essential "
            "combinatorial data (edge list + boundary sewing pairs).",
        )
        # For connected cubic ribbon graphs:
        #   E = 3(F - 2 + 2g)
        # so at genus g=3 with one face F=1, we have E=15 and V=10.
        #
        # This test intentionally validates only the essential output payload
        # used by downstream workflows:
        #   1) edges: labeled abstract graph edges
        #   2) sewing_pairs: which boundary segments are glued for each edge label
        #
        # We avoid geometry/diagram assertions here on purpose.
        data_all = rgg.generate_essential_data_fixed_genus(3, n_faces=1)
        self.assertGreater(len(data_all), 0)
        for data in data_all[:5]:
            self.assertIn("edges", data)
            self.assertIn("sewing_pairs", data)
            self.assertEqual(len(data["edges"]), 15)
            self.assertEqual(len(data["sewing_pairs"]), 15)
        self._pass_test("test_essential_data_only_for_genus_3")

    def test_fixed_genus_api_matches_direct_parameters(self):
        # Content: fixed-genus API consistency with direct (F,E) call.
        # Expected/cross-check: both paths produce same number of non-isomorphic graphs.
        # Pass condition: counts are equal.
        self._start_test(
            "test_fixed_genus_api_matches_direct_parameters",
            "Compare fixed-genus generation against direct (F=1,E=15) generation.",
            "Cross-check output cardinalities are identical.",
        )
        self._label(
            "PRODUCTION",
            "Genus-3 consistency: fixed-genus API should match direct generation with (F=1, E=15).",
        )
        # This guards the genus->edge conversion path used in production calls.
        by_genus = rgg.generate_ribbon_graphs_fixed_genus(3, n_faces=1)
        by_direct = rgg.generate_ribbon_graphs(1, 15)
        self.assertEqual(len(by_genus), len(by_direct))
        self._pass_test("test_fixed_genus_api_matches_direct_parameters")

    def test_independent_enumeration_matches_generator_theta(self):
        # Content: independent brute-force enumeration on theta graph.
        # Expected/cross-check: brute-force valid-rotation count and dedup cardinality agree with generator.
        # Pass condition: expected 18 raw valid rotations and same final unique count as production.
        self._start_test(
            "test_independent_enumeration_matches_generator_theta",
            "Brute-force theta rotations and compare with production generator.",
            "Cross-check raw count=18 and deduplicated count matches generate_ribbon_graphs(1,3).",
        )
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
        self._pass_test("test_independent_enumeration_matches_generator_theta")

    def test_independent_enumeration_matches_generator_k4_planar(self):
        # Content: independent brute-force enumeration on K4 planar case.
        # Expected/cross-check: deduplicated brute-force cardinality matches production.
        # Pass condition: equal counts.
        self._start_test(
            "test_independent_enumeration_matches_generator_k4_planar",
            "Brute-force K4 planar rotations and compare with production generator.",
            "Cross-check deduplicated brute-force count equals generate_ribbon_graphs(4,6).",
        )
        # Independent search on K4 (V=4,E=6) for planar F=4 case.
        self._label("BRUTE_FORCE", "Enumerating all local rotations on K4 for F=4.")
        edges, verts = rgg._get_base_graphs(4, 6)[0]  # noqa: SLF001
        independent_all = self._independent_valid_rotations(edges, verts, 4)
        independent_unique = rgg._remove_isomorphisms(independent_all)  # noqa: SLF001
        self._label("PRODUCTION", "Running generate_ribbon_graphs(4, 6) for comparison.")
        generated = rgg.generate_ribbon_graphs(4, 6)
        self.assertEqual(len(independent_unique), len(generated))
        self._pass_test("test_independent_enumeration_matches_generator_k4_planar")

    def test_generalized_essential_data_for_multi_face_fixed_genus(self):
        # Content: essential fixed-genus payload should stay useful for F>1.
        # Expected/cross-check: generalized face-boundary keys are populated and sewing_pairs stays None.
        # Pass condition: schema and small invariants hold on sampled outputs.
        self._start_test(
            "test_generalized_essential_data_for_multi_face_fixed_genus",
            "Generate essential payloads for fixed genus with multiple faces.",
            "Cross-check generalized face-boundary keys are present and internally consistent.",
        )
        self._label(
            "PRODUCTION",
            "Checking generalized essential payloads for (g=1, F=2).",
        )
        data_all = rgg.generate_essential_data_fixed_genus(1, n_faces=2)
        self.assertGreater(len(data_all), 0)
        for data in data_all[:5]:
            self.assertIn("edges", data)
            self.assertIn("n_faces", data)
            self.assertIn("face_boundaries", data)
            self.assertIn("face_edge_sequences", data)
            self.assertIn("face_vertex_sequences", data)
            self.assertIn("gluing_pairs", data)
            self.assertIn("sewing_pairs", data)
            self.assertEqual(data["n_faces"], 2)
            self.assertIsNone(data["sewing_pairs"])
            self.assertEqual(len(data["gluing_pairs"]), len(data["edges"]))
            self.assertEqual(
                sum(len(face) for face in data["face_boundaries"]),
                2 * len(data["edges"]),
            )
        self._pass_test("test_generalized_essential_data_for_multi_face_fixed_genus")


if __name__ == "__main__":
    unittest.main()
