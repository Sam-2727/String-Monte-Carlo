import unittest

import Fast_Ribbon_Generator as rgg


class TestRibbonGraphGenerator(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
