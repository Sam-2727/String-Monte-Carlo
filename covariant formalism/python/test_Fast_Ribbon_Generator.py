import unittest

import Fast_Ribbon_Generator as rgg


class TestRibbonGraphGenerator(unittest.TestCase):
    @staticmethod
    def _has_parallel_edge(rg):
        edges, _, _ = rg
        edge_pairs = {tuple(sorted(edge)) for edge in edges}
        return len(edge_pairs) < len(edges)

    @staticmethod
    def _print_illustration_from_generator_data(rg, idx=1):
        edges, verts, rotation = rg
        n_faces = rgg._count_faces(edges, verts, rotation)  # noqa: SLF001
        print(f"\nGraph {idx}: V={len(verts)}, E={len(edges)}, F={n_faces}")
        print(f"Edges: {[(i + 1, a, b) for i, (a, b) in enumerate(edges)]}")

        data = rgg.get_disc_boundary(rg)
        boundary = data["boundary"]
        sewing = data["sewing"]
        segments = [
            (i + 1, frm, to, eidx + 1) for i, (frm, to, eidx) in enumerate(boundary)
        ]
        sewing_pairs = [(e, p1, p2) for e, (p1, p2) in sorted(sewing.items())]

        print(f"Segments (seg, from, to, edge): {segments}")
        print(f"Sewing pairs (edge, seg1, seg2): {sewing_pairs}")

    def test_enumerate_genus2_one_face_graphs(self):
        print("\n[TEST] Enumerate genus-2 ribbon graphs with one face")
        print("[CONTENT] Run the generator for (g=2, F=1) and print the output data.")
        print("[EXPECTED/CROSS-CHECK] Enumeration completes without explicit assertions.")

        ribbon_graphs = rgg.generate_ribbon_graphs_fixed_genus(
            2,
            n_faces=1,
            verbose=True,
            workers=1,
        )

        for idx, rg in enumerate(ribbon_graphs, start=1):
            self._print_illustration_from_generator_data(rg, idx=idx)

    def test_genus2_two_face_graphs_include_multigraphs(self):
        exact_rgs = rgg.generate_ribbon_graphs_fixed_genus(
            2,
            n_faces=2,
            workers=1,
        )
        fallback_rgs = rgg.generate_ribbon_graphs_fixed_genus(
            2,
            n_faces=2,
            workers=1,
            max_exact_multigraph_vertices=6,
        )

        self.assertEqual(len(exact_rgs), 263)
        self.assertTrue(any(self._has_parallel_edge(rg) for rg in exact_rgs))
        self.assertTrue(all(rgg._count_faces(*rg) == 2 for rg in exact_rgs))  # noqa: SLF001
        self.assertTrue(all(not self._has_parallel_edge(rg) for rg in fallback_rgs))
        self.assertLess(len(fallback_rgs), len(exact_rgs))


if __name__ == "__main__":
    unittest.main()
