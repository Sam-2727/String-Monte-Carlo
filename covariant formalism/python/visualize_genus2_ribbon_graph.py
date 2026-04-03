"""Render stored one-face ribbon graph data as SVG diagrams.

Usage examples:
    PYTHONPATH='covariant formalism/python' ./.venv/bin/python \
        'covariant formalism/python/visualize_genus2_ribbon_graph.py' \
        --genus 2 --topology 1 --output '/tmp/genus2_topology_1_boundary.svg'

    PYTHONPATH='covariant formalism/python' ./.venv/bin/python \
        'covariant formalism/python/visualize_genus2_ribbon_graph.py' \
        --genus 3 --topology 1683 --output '/tmp/genus3_topology_1683_boundary.svg'

    PYTHONPATH='covariant formalism/python' ./.venv/bin/python \
        'covariant formalism/python/visualize_genus2_ribbon_graph.py' \
        --genus 3 --topology 1683 --show-basis \
        --output '/tmp/genus3_topology_1683_with_basis.svg'

    PYTHONPATH='covariant formalism/python' ./.venv/bin/python \
        'covariant formalism/python/visualize_genus2_ribbon_graph.py' \
        --genus 2 --all --output-dir '/tmp/genus2_svgs'

The picture shows the traced disc boundary as colored segments. Segments with
the same edge label belong to the same sewn edge, and the optional chords show
the sewing pairs inside the disc.
"""

from __future__ import annotations

import argparse
import colorsys
import math
from pathlib import Path

import compact_partition as cp
import ell_to_tau as et
import genus3_t_duality as g3

SEGMENT_FILL_OPACITY = 0.82
SEWING_CHORD_OPACITY = 0.88
BASIS_CHORD_OPACITY = 1.00


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    step = (stop - start) / float(count - 1)
    return [start + step * i for i in range(count)]


def _point(cx: float, cy: float, radius: float, theta: float) -> tuple[float, float]:
    return (
        cx + radius * math.cos(theta),
        cy - radius * math.sin(theta),
    )


def _hex_color(edge_label: int, n_edges: int) -> str:
    hue = ((edge_label - 1) % max(n_edges, 1)) / float(max(n_edges, 1))
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.55, 0.95)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(255 * red)),
        int(round(255 * green)),
        int(round(255 * blue)),
    )


def _darken_hex(color: str, factor: float = 0.55) -> str:
    color = color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(red * factor)),
        int(round(green * factor)),
        int(round(blue * factor)),
    )


def _segment_angles(segment_idx: int, n_segments: int) -> tuple[float, float]:
    theta0 = (math.pi / 2.0) + (2.0 * math.pi * segment_idx / n_segments)
    theta1 = (math.pi / 2.0) + (2.0 * math.pi * (segment_idx + 1) / n_segments)
    return theta0, theta1


def _ring_sector_polygon(
    cx: float,
    cy: float,
    inner_radius: float,
    outer_radius: float,
    theta0: float,
    theta1: float,
    *,
    samples: int = 18,
) -> str:
    outer_pts = [
        _point(cx, cy, outer_radius, theta)
        for theta in _linspace(theta0, theta1, samples)
    ]
    inner_pts = [
        _point(cx, cy, inner_radius, theta)
        for theta in _linspace(theta1, theta0, samples)
    ]
    pts = outer_pts + inner_pts
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def _build_legend_entries(graph_data: dict) -> list[tuple[int, int, int]]:
    return [
        (int(edge), int(seg1), int(seg2))
        for edge, seg1, seg2 in graph_data["sewing_pairs"]
    ]


def _basis_cycle_name(kind: str, pair_idx: int) -> str:
    prefix = "a" if kind == "alpha" else "b"
    return f"{prefix}{pair_idx}"


def _basis_cycle_color(kind: str, pair_idx: int) -> str:
    alpha_palette = ["#d73027", "#f46d43", "#fdae61", "#f46d43"]
    beta_palette = ["#4575b4", "#3288bd", "#66c2a5", "#5e4fa2"]
    palette = alpha_palette if kind == "alpha" else beta_palette
    return palette[(pair_idx - 1) % len(palette)]


def _format_basis_terms(cycle: dict) -> str:
    parts: list[str] = []
    for term_idx, term in enumerate(cycle["terms"]):
        sign = "-" if term["coeff"] < 0 else "+"
        atom = f"e{term['edge_label']}"
        if term_idx == 0:
            parts.append(atom if sign == "+" else f"-{atom}")
        else:
            parts.append(f" {sign} {atom}")
    return "".join(parts)


def _basis_overlay_data(genus: int, graph_data: dict) -> dict:
    boundary_zero = tuple(
        (int(frm), int(to), int(edge_label) - 1)
        for frm, to, edge_label in graph_data["boundary"]
    )
    chord_data = et._edge_chord_data_from_boundary(boundary_zero, genus=genus)
    basis_data = et._find_edge_homology_basis_from_chord_data(chord_data)

    cycles: list[dict] = []
    for pair_idx, pair in enumerate(basis_data["basis_pairs"], start=1):
        for kind in ("alpha", "beta"):
            terms: list[dict] = []
            for edge_idx, coeff in pair[kind]:
                pos0, pos1 = chord_data["edge_positions"][edge_idx]
                seg_start = int(pos0 + 1) if coeff > 0 else int(pos1 + 1)
                seg_end = int(pos1 + 1) if coeff > 0 else int(pos0 + 1)
                terms.append(
                    {
                        "edge_label": int(edge_idx + 1),
                        "coeff": int(coeff),
                        "seg_start": seg_start,
                        "seg_end": seg_end,
                    }
                )
            cycles.append(
                {
                    "name": _basis_cycle_name(kind, pair_idx),
                    "kind": kind,
                    "pair_idx": pair_idx,
                    "color": _basis_cycle_color(kind, pair_idx),
                    "terms": terms,
                }
            )

    return {
        "basis_algorithm": basis_data["basis_algorithm"],
        "cycles": cycles,
    }


def _topology_count(genus: int) -> int:
    if genus == 2:
        return len(cp.GENUS2_F1_GRAPH_DATA)
    if genus == 3:
        return int(g3.GENUS3_GRAPH_COUNT)
    raise ValueError(f"Unsupported genus {genus}; expected 2 or 3.")


def _get_graph_data(genus: int, topology: int) -> dict:
    if genus == 2:
        return cp.get_stored_genus2_graph(topology)
    if genus == 3:
        return g3.get_stored_genus3_graph(topology)
    raise ValueError(f"Unsupported genus {genus}; expected 2 or 3.")


def _append_topology_sheet_panel(
    lines: list[str],
    *,
    topology: int,
    graph_data: dict,
    left: float,
    top: float,
    panel_width: float,
    panel_height: float,
    with_chords: bool,
) -> None:
    boundary = tuple(graph_data["boundary"])
    n_segments = len(boundary)
    n_edges = len(graph_data["edges"])

    cx = left + 0.5 * panel_width
    cy = top + 0.56 * panel_height
    outer_radius = 140.0
    inner_radius = 104.0
    chord_radius = 92.0
    segment_label_radius = 122.0
    segment_number_radius = 166.0

    lines.append(
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{panel_width:.2f}" height="{panel_height:.2f}" '
        'rx="18" ry="18" fill="#fcfcfe" stroke="#d8d8e0" stroke-width="1.4"/>'
    )
    lines.append(
        f'<text class="sheet-title" x="{cx:.2f}" y="{top + 34.0:.2f}" text-anchor="middle">'
        f'Topology {topology}</text>'
    )
    chord_lines: list[str] = []
    if with_chords:
        for edge_label, seg1, seg2 in graph_data["sewing_pairs"]:
            theta1 = sum(_segment_angles(seg1 - 1, n_segments)) / 2.0
            theta2 = sum(_segment_angles(seg2 - 1, n_segments)) / 2.0
            x1, y1 = _point(cx, cy, chord_radius, theta1)
            x2, y2 = _point(cx, cy, chord_radius, theta2)
            color = _darken_hex(_hex_color(edge_label, n_edges), 0.52)
            chord_lines.append(
                f'<path d="M {x1:.2f} {y1:.2f} Q {cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}" '
                f'fill="none" stroke="{color}" stroke-width="2.8" stroke-opacity="{SEWING_CHORD_OPACITY:.2f}"/>'
            )

    for segment_idx, (_, _, edge_label) in enumerate(boundary):
        theta0, theta1 = _segment_angles(segment_idx, n_segments)
        theta_mid = 0.5 * (theta0 + theta1)
        color = _hex_color(edge_label, n_edges)
        polygon = _ring_sector_polygon(
            cx,
            cy,
            inner_radius,
            outer_radius,
            theta0,
            theta1,
            samples=14,
        )
        lines.append(
            f'<polygon points="{polygon}" fill="{color}" fill-opacity="{SEGMENT_FILL_OPACITY:.2f}" '
            'stroke="#ffffff" stroke-width="1.5"/>'
        )

        label_x, label_y = _point(cx, cy, segment_label_radius, theta_mid)
        lines.append(
            f'<text class="sheet-edge-label" x="{label_x:.2f}" y="{label_y:.2f}" '
            'text-anchor="middle" dominant-baseline="middle">'
            f'e{edge_label}</text>'
        )

        seg_x, seg_y = _point(cx, cy, segment_number_radius, theta_mid)
        lines.append(
            f'<text class="sheet-segment-number" x="{seg_x:.2f}" y="{seg_y:.2f}" '
            'text-anchor="middle" dominant-baseline="middle">'
            f's{segment_idx + 1}</text>'
        )

    lines.append(
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{inner_radius - 16.0:.2f}" fill="#ffffff" fill-opacity="0.95"/>'
    )
    lines.extend(chord_lines)


def _write_svg_for_topology(
    *,
    genus: int,
    topology: int,
    graph_data: dict,
    output_path: Path,
    with_chords: bool,
    show_basis: bool,
) -> Path:
    # Draw one stored topology as a segmented boundary circle plus a legend that
    # spells out the sewing data used elsewhere in the codebase.
    boundary = tuple(graph_data["boundary"])
    n_segments = len(boundary)
    n_edges = len(graph_data["edges"])
    basis_overlay = _basis_overlay_data(genus, graph_data) if show_basis else None
    basis_rows = 0 if basis_overlay is None else len(basis_overlay["cycles"])

    width = 1200
    legend_left = 810
    legend_top = 155
    row_gap = 36
    basis_top = legend_top + n_edges * row_gap + 44
    basis_row_gap = 30
    info_top = basis_top + (34 + basis_rows * basis_row_gap if show_basis else 0)
    height = max(980, int(info_top + 28 + n_segments * 21 + 60))
    cx = 410.0
    cy = 450.0
    outer_radius = 300.0
    inner_radius = 225.0
    chord_radius = 205.0
    segment_label_radius = 262.0
    segment_number_radius = 345.0

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "  .title { font: 700 28px serif; fill: #222; }",
        "  .subtitle { font: 500 16px serif; fill: #555; }",
        "  .edge-label { font: 700 18px monospace; fill: #1f1f1f; }",
        "  .segment-number { font: 500 13px monospace; fill: #444; }",
        "  .legend-title { font: 700 18px serif; fill: #222; }",
        "  .legend-text { font: 500 15px monospace; fill: #2a2a2a; }",
        "  .frame { fill: #fbfbfd; stroke: #d8d8e0; stroke-width: 1.5; }",
        "</style>",
        f'<rect class="frame" x="18" y="18" width="{width - 36}" height="{height - 36}" rx="18" ry="18"/>',
        f'<text class="title" x="56" y="68">Stored genus-{genus} topology {topology}</text>',
        f'<text class="subtitle" x="56" y="96">Boundary segments are ordered counterclockwise. Matching edge labels share a color.</text>',
    ]
    if show_basis and basis_overlay is not None:
        lines.append(
            f'<text class="subtitle" x="56" y="120">Highlighted a_i / b_i chords show one symplectic basis of 1-cycles ({_svg_escape(basis_overlay["basis_algorithm"])}).</text>'
        )

    sewing_chord_lines: list[str] = []
    if with_chords:
        for edge_label, seg1, seg2 in graph_data["sewing_pairs"]:
            theta1 = sum(_segment_angles(seg1 - 1, n_segments)) / 2.0
            theta2 = sum(_segment_angles(seg2 - 1, n_segments)) / 2.0
            x1, y1 = _point(cx, cy, chord_radius, theta1)
            x2, y2 = _point(cx, cy, chord_radius, theta2)
            color = _darken_hex(_hex_color(edge_label, n_edges), 0.52)
            sewing_chord_lines.append(
                f'<path d="M {x1:.2f} {y1:.2f} Q {cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}" '
                f'fill="none" stroke="{color}" stroke-width="3.8" stroke-opacity="{SEWING_CHORD_OPACITY:.2f}"/>'
            )

    basis_chord_lines: list[str] = []
    if show_basis and basis_overlay is not None:
        for cycle in basis_overlay["cycles"]:
            dash = "" if cycle["kind"] == "alpha" else ' stroke-dasharray="11 7"'
            for term in cycle["terms"]:
                theta1 = sum(_segment_angles(term["seg_start"] - 1, n_segments)) / 2.0
                theta2 = sum(_segment_angles(term["seg_end"] - 1, n_segments)) / 2.0
                x1, y1 = _point(cx, cy, chord_radius - 18.0, theta1)
                x2, y2 = _point(cx, cy, chord_radius - 18.0, theta2)
                basis_chord_lines.append(
                    f'<path d="M {x1:.2f} {y1:.2f} Q {cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}" '
                    f'fill="none" stroke="{cycle["color"]}" stroke-width="6.0" stroke-opacity="{BASIS_CHORD_OPACITY:.2f}"{dash}/>'
                )

    for segment_idx, (frm, to, edge_label) in enumerate(boundary):
        theta0, theta1 = _segment_angles(segment_idx, n_segments)
        theta_mid = 0.5 * (theta0 + theta1)
        color = _hex_color(edge_label, n_edges)
        polygon = _ring_sector_polygon(
            cx,
            cy,
            inner_radius,
            outer_radius,
            theta0,
            theta1,
        )
        lines.append(
            f'<polygon points="{polygon}" fill="{color}" fill-opacity="{SEGMENT_FILL_OPACITY:.2f}" '
            'stroke="#ffffff" stroke-width="2.0"/>'
        )

        label_x, label_y = _point(cx, cy, segment_label_radius, theta_mid)
        lines.append(
            f'<text class="edge-label" x="{label_x:.2f}" y="{label_y:.2f}" '
            'text-anchor="middle" dominant-baseline="middle">'
            f'e{edge_label}</text>'
        )

        seg_x, seg_y = _point(cx, cy, segment_number_radius, theta_mid)
        lines.append(
            f'<text class="segment-number" x="{seg_x:.2f}" y="{seg_y:.2f}" '
            'text-anchor="middle" dominant-baseline="middle">'
            f's{segment_idx + 1}</text>'
        )

        tip_x, tip_y = _point(cx, cy, outer_radius + 10.0, theta_mid)
        lines.append(
            f'<circle cx="{tip_x:.2f}" cy="{tip_y:.2f}" r="2.7" fill="{color}" stroke="#ffffff" stroke-width="0.8"/>'
        )

    lines.append(
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{inner_radius - 20.0:.2f}" fill="#ffffff" fill-opacity="0.94"/>'
    )
    lines.extend(sewing_chord_lines)
    lines.extend(basis_chord_lines)
    lines.append(
        f'<text class="subtitle" x="{cx:.2f}" y="{cy - 8.0:.2f}" text-anchor="middle">{n_segments} boundary segments</text>'
    )
    lines.append(
        f'<text class="subtitle" x="{cx:.2f}" y="{cy + 18.0:.2f}" text-anchor="middle">{n_edges} sewn edges</text>'
    )

    lines.append(f'<text class="legend-title" x="{legend_left}" y="{legend_top - 24}">Edge Legend</text>')
    for idx, (edge_label, seg1, seg2) in enumerate(_build_legend_entries(graph_data)):
        y = legend_top + idx * row_gap
        color = _hex_color(edge_label, n_edges)
        lines.append(
            f'<rect x="{legend_left}" y="{y - 14}" width="22" height="22" '
            f'rx="4" ry="4" fill="{color}" stroke="#ffffff" stroke-width="1.0"/>'
        )
        lines.append(
            f'<text class="legend-text" x="{legend_left + 34}" y="{y + 1}" dominant-baseline="middle">'
            f'e{edge_label}: s{seg1} ↔ s{seg2}</text>'
        )

    if show_basis and basis_overlay is not None:
        lines.append(f'<text class="legend-title" x="{legend_left}" y="{basis_top}">Symplectic Basis</text>')
        for idx, cycle in enumerate(basis_overlay["cycles"], start=1):
            y = basis_top + 28 + (idx - 1) * basis_row_gap
            dash = "" if cycle["kind"] == "alpha" else ' stroke-dasharray="11 7"'
            lines.append(
                f'<line x1="{legend_left}" y1="{y:.2f}" x2="{legend_left + 24}" y2="{y:.2f}" '
                f'stroke="{cycle["color"]}" stroke-width="5.0"{dash}/>'
            )
            lines.append(
                f'<text class="legend-text" x="{legend_left + 34}" y="{y + 1:.2f}" dominant-baseline="middle">'
                f'{_svg_escape(cycle["name"])} = {_svg_escape(_format_basis_terms(cycle))}</text>'
            )

    lines.append(f'<text class="legend-title" x="{legend_left}" y="{info_top}">Boundary Data</text>')
    for idx, (frm, to, edge_label) in enumerate(boundary, start=1):
        y = info_top + 28 + (idx - 1) * 21
        lines.append(
            f'<text class="legend-text" x="{legend_left}" y="{y}" dominant-baseline="middle">'
            f's{idx}: ({frm} → {to}, e{edge_label})</text>'
        )

    lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_svg_sheet(
    *,
    genus: int,
    output_path: Path,
    with_chords: bool,
) -> Path:
    total_topologies = _topology_count(genus)
    topologies = range(1, total_topologies + 1)
    cols = 3
    rows = math.ceil(total_topologies / cols)
    panel_width = 500.0
    panel_height = 500.0
    gap_x = 28.0
    gap_y = 28.0
    margin = 28.0

    width = int((2 * margin) + cols * panel_width + (cols - 1) * gap_x)
    height = int((2 * margin) + rows * panel_height + (rows - 1) * gap_y)

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "  .sheet-title { font: 700 24px serif; fill: #222; }",
        "  .sheet-edge-label { font: 700 14px monospace; fill: #1f1f1f; }",
        "  .sheet-segment-number { font: 500 10px monospace; fill: #444; }",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f6f7fb"/>',
    ]

    for idx, topology in enumerate(topologies):
        row = idx // cols
        col = idx % cols
        left = margin + col * (panel_width + gap_x)
        top = margin + row * (panel_height + gap_y)
        _append_topology_sheet_panel(
            lines,
            topology=topology,
            graph_data=_get_graph_data(genus, topology),
            left=left,
            top=top,
            panel_width=panel_width,
            panel_height=panel_height,
            with_chords=with_chords,
        )

    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _default_output(genus: int, topology: int, output_dir: Path) -> Path:
    return output_dir / f"genus{genus}_topology_{topology}_boundary.svg"


def main() -> None:
    # By default the script renders one stored topology. Passing --all switches
    # to a batch mode that writes one SVG per topology into --output-dir, while
    # --sheet combines all stored topologies into a single summary SVG.
    parser = argparse.ArgumentParser(
        description=(
            "Visualize stored one-face ribbon graph data as a segmented disc boundary."
        )
    )
    parser.add_argument(
        "--genus",
        type=int,
        choices=(2, 3),
        default=2,
        help="Stored topology family to render.",
    )
    parser.add_argument(
        "--topology",
        type=int,
        default=1,
        help="Stored topology index within the chosen genus.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render all stored one-face topologies for the chosen genus.",
    )
    parser.add_argument(
        "--sheet",
        action="store_true",
        help="Render all stored topologies into one summary sheet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path for a single topology render.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for generated SVG files. Used by --all and as the default single-file location.",
    )
    parser.add_argument(
        "--no-chords",
        action="store_true",
        help="Do not draw interior sewing chords.",
    )
    parser.add_argument(
        "--show-basis",
        action="store_true",
        help="Overlay one symplectic a_i/b_i basis of 1-cycles on top of the sewing chords.",
    )
    args = parser.parse_args()

    if (not args.sheet) and args.all and args.output is not None:
        raise ValueError("Use either --all with --output-dir, or --output for a single topology.")
    if args.sheet and args.show_basis:
        raise ValueError("--show-basis is only supported for single-topology or --all renders, not --sheet.")

    if args.sheet:
        output_path = (
            args.output
            if args.output is not None
            else args.output_dir / f"genus{args.genus}_topology_sheet.svg"
        )
        written = _write_svg_sheet(
            genus=args.genus,
            output_path=output_path,
            with_chords=not args.no_chords,
        )
        print(f"Wrote {written}")
        return

    topologies = (
        range(1, _topology_count(args.genus) + 1)
        if args.all
        else [int(args.topology)]
    )

    for topology in topologies:
        graph_data = _get_graph_data(args.genus, topology)
        # In single-topology mode, --output lets you choose an exact filename.
        # Otherwise we generate a descriptive default name in the output dir.
        output_path = (
            args.output
            if (not args.all and args.output is not None)
            else _default_output(args.genus, topology, args.output_dir)
        )
        written = _write_svg_for_topology(
            genus=args.genus,
            topology=topology,
            graph_data=graph_data,
            output_path=output_path,
            with_chords=not args.no_chords,
            show_basis=args.show_basis,
        )
        print(f"Wrote {written}")


if __name__ == "__main__":
    main()
