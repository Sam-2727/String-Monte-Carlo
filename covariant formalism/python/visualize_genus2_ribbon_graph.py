"""Render stored genus-2 one-face ribbon graph data as SVG diagrams.

Usage examples:
    PYTHONPATH='covariant formalism/python' ./.venv/bin/python \
        'covariant formalism/python/visualize_genus2_ribbon_graph.py' \
        --topology 1 --output '/tmp/genus2_topology_1_boundary.svg'

    PYTHONPATH='covariant formalism/python' ./.venv/bin/python \
        'covariant formalism/python/visualize_genus2_ribbon_graph.py' \
        --all --output-dir '/tmp/genus2_svgs'

The picture shows the traced disc boundary as 18 colored segments. Segments
with the same edge label belong to the same sewn edge, and the optional chords
show the sewing pairs inside the disc.
"""

from __future__ import annotations

import argparse
import colorsys
import math
from pathlib import Path

import compact_partition as cp


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

    if with_chords:
        for edge_label, seg1, seg2 in graph_data["sewing_pairs"]:
            theta1 = sum(_segment_angles(seg1 - 1, n_segments)) / 2.0
            theta2 = sum(_segment_angles(seg2 - 1, n_segments)) / 2.0
            x1, y1 = _point(cx, cy, chord_radius, theta1)
            x2, y2 = _point(cx, cy, chord_radius, theta2)
            color = _hex_color(edge_label, n_edges)
            lines.append(
                f'<path d="M {x1:.2f} {y1:.2f} Q {cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}" '
                f'fill="none" stroke="{color}" stroke-width="2.2" stroke-opacity="0.72"/>'
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
            f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.88" '
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


def _write_svg_for_topology(
    *,
    topology: int,
    graph_data: dict,
    output_path: Path,
    with_chords: bool,
) -> Path:
    # Draw one stored topology as a segmented boundary circle plus a legend that
    # spells out the sewing data used elsewhere in the codebase.
    boundary = tuple(graph_data["boundary"])
    n_segments = len(boundary)
    n_edges = len(graph_data["edges"])

    width = 1200
    height = 980
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
        f'<text class="title" x="56" y="68">Stored genus-2 topology {topology}</text>',
        f'<text class="subtitle" x="56" y="96">Boundary segments are ordered counterclockwise. Matching edge labels share a color.</text>',
    ]

    if with_chords:
        for edge_label, seg1, seg2 in graph_data["sewing_pairs"]:
            theta1 = sum(_segment_angles(seg1 - 1, n_segments)) / 2.0
            theta2 = sum(_segment_angles(seg2 - 1, n_segments)) / 2.0
            x1, y1 = _point(cx, cy, chord_radius, theta1)
            x2, y2 = _point(cx, cy, chord_radius, theta2)
            color = _hex_color(edge_label, n_edges)
            lines.append(
                f'<path d="M {x1:.2f} {y1:.2f} Q {cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}" '
                f'fill="none" stroke="{color}" stroke-width="3.0" stroke-opacity="0.70"/>'
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
            f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.88" '
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
    lines.append(
        f'<text class="subtitle" x="{cx:.2f}" y="{cy - 8.0:.2f}" text-anchor="middle">18 boundary segments</text>'
    )
    lines.append(
        f'<text class="subtitle" x="{cx:.2f}" y="{cy + 18.0:.2f}" text-anchor="middle">9 sewn edges</text>'
    )

    legend_left = 810
    legend_top = 155
    row_gap = 36
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

    info_top = legend_top + n_edges * row_gap + 44
    lines.append(f'<text class="legend-title" x="{legend_left}" y="{info_top}">Boundary Data</text>')
    for idx, (frm, to, edge_label) in enumerate(boundary, start=1):
        y = info_top + 28 + (idx - 1) * 21
        if y > height - 40:
            break
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
    output_path: Path,
    with_chords: bool,
) -> Path:
    topologies = range(1, len(cp.GENUS2_F1_GRAPH_DATA) + 1)
    cols = 3
    rows = math.ceil(len(cp.GENUS2_F1_GRAPH_DATA) / cols)
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
            graph_data=cp.get_stored_genus2_graph(topology),
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


def _default_output(topology: int, output_dir: Path) -> Path:
    return output_dir / f"genus2_topology_{topology}_boundary.svg"


def main() -> None:
    # By default the script renders one stored topology. Passing --all switches
    # to a batch mode that writes one SVG per topology into --output-dir, while
    # --sheet combines all stored topologies into a single summary SVG.
    parser = argparse.ArgumentParser(
        description=(
            "Visualize stored genus-2 one-face ribbon graph data as a segmented disc boundary."
        )
    )
    parser.add_argument(
        "--topology",
        type=int,
        default=1,
        help="Stored genus-2 topology index from compact_partition.py.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render all stored genus-2 one-face topologies.",
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
    args = parser.parse_args()

    if (not args.sheet) and args.all and args.output is not None:
        raise ValueError("Use either --all with --output-dir, or --output for a single topology.")

    if args.sheet:
        output_path = (
            args.output
            if args.output is not None
            else args.output_dir / "genus2_topology_sheet.svg"
        )
        written = _write_svg_sheet(
            output_path=output_path,
            with_chords=not args.no_chords,
        )
        print(f"Wrote {written}")
        return

    topologies = (
        range(1, len(cp.GENUS2_F1_GRAPH_DATA) + 1)
        if args.all
        else [int(args.topology)]
    )

    for topology in topologies:
        graph_data = cp.get_stored_genus2_graph(topology)
        # In single-topology mode, --output lets you choose an exact filename.
        # Otherwise we generate a descriptive default name in the output dir.
        output_path = (
            args.output
            if (not args.all and args.output is not None)
            else _default_output(topology, args.output_dir)
        )
        written = _write_svg_for_topology(
            topology=topology,
            graph_data=graph_data,
            output_path=output_path,
            with_chords=not args.no_chords,
        )
        print(f"Wrote {written}")


if __name__ == "__main__":
    main()
