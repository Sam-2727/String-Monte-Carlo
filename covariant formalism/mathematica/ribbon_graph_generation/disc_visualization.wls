#!/usr/bin/env wolframscript
(* Disc visualization for F=1 ribbon graphs *)

Clear["Global`*"];

(* Copy necessary functions from export script *)
<< "/Users/xiyin/LocalWork/StringMC_local/ribbon graphs/ribbon_graph_generator_v1_fixed.wls"

(* Trace face boundary to get vertex sequence *)
TraceFaceBoundary[labeledEdges_, rotation_, vertices_] := Module[
  {halfEdges, visited, startHE, current, boundary},

  (* Create half-edges *)
  halfEdges = Flatten[Table[
    {{le[[1, 1]], le[[1, 2]], le[[2]]}, {le[[1, 2]], le[[1, 1]], le[[2]]}},
    {le, labeledEdges}
  ], 1];

  visited = Association[Table[he -> False, {he, halfEdges}]];

  (* Start from first half-edge *)
  startHE = halfEdges[[1]];
  current = startHE;
  boundary = {};

  (* Trace the boundary *)
  While[!visited[current],
    visited[current] = True;
    AppendTo[boundary, {current[[1]], current[[2]], current[[3]]}]; (* {from, to, edge} *)

    (* Move to next half-edge *)
    Module[{to, label, vertexIdx, rotAtVertex, pos, nextLabel, nextEdge},
      {from, to, label} = current;
      vertexIdx = FirstPosition[vertices, to][[1]];
      rotAtVertex = rotation[[vertexIdx]];
      pos = FirstPosition[rotAtVertex, label][[1]];
      nextLabel = rotAtVertex[[Mod[pos, Length[rotAtVertex]] + 1]];
      nextEdge = Select[labeledEdges, #[[2]] == nextLabel &][[1, 1]];

      If[nextEdge[[1]] == to,
        current = {to, nextEdge[[2]], nextLabel},
        current = {to, nextEdge[[1]], nextLabel}
      ]
    ]
  ];

  boundary
];

(* Draw disc visualization *)
DrawDiscVisualization[rg_] := Module[
  {graph, rotation, vertices, labeledEdges, boundary, n,
   points, arcs, labels, discRadius, v, e, f, g},

  {graph, rotation} = rg;
  vertices = VertexList[graph];
  v = Length[vertices];
  e = EdgeCount[graph];
  labeledEdges = MapIndexed[{#1, First[#2]} &, EdgeList[graph]];

  (* Trace face boundary *)
  boundary = TraceFaceBoundary[labeledEdges, rotation, vertices];
  n = Length[boundary];

  f = 1;  (* Single face *)
  g = (2 - v + e - f) / 2;

  discRadius = 1.0;

  (* Points on circle for each boundary position *)
  points = Table[
    discRadius * {Cos[2 * Pi * (i - 1) / n], Sin[2 * Pi * (i - 1) / n]},
    {i, n}
  ];

  (* Draw arcs for each half-edge along the boundary *)
  arcs = Table[
    Module[{from, to, edge, angle1, angle2, midAngle, labelPos},
      {from, to, edge} = boundary[[i]];
      angle1 = 2 * Pi * (i - 1) / n;
      angle2 = 2 * Pi * i / n;

      (* Label position slightly inside the disc *)
      midAngle = (angle1 + angle2) / 2;
      labelPos = 0.85 * discRadius * {Cos[midAngle], Sin[midAngle]};

      {
        Thickness[0.008],
        Black,
        Circle[{0, 0}, discRadius, {angle1, angle2}],
        (* Edge label *)
        Text[Style[edge, Bold, 10], labelPos]
      }
    ],
    {i, n}
  ];

  (* Vertex labels on boundary *)
  labels = Table[
    Module[{from, to, edge, p, angle, offset},
      {from, to, edge} = boundary[[i]];
      p = points[[i]];
      angle = 2 * Pi * (i - 1) / n;
      offset = 0.15 * {Cos[angle], Sin[angle]};

      {
        PointSize[0.015],
        Red,
        Point[p],
        Text[Style[from, Bold, 10, Red], p + offset]
      }
    ],
    {i, n}
  ];

  (* Circle boundary *)
  boundary = {
    Thickness[0.003],
    Gray,
    Circle[{0, 0}, discRadius]
  };

  Panel[Column[{
    Style["Disc Representation (F=1)", Bold, 16],
    Spacer[5],
    Grid[{
      {Style["Vertices:", Bold], Style[v, Black],
       Style["Edges:", Bold], Style[e, Black]},
      {Style["Genus:", Bold], Style[g, Black],
       Style["Boundary points:", Bold], Style[n, Black]}
    }, Frame -> All, Spacings -> {2, 1}],
    Spacer[10],
    Graphics[{boundary, arcs, labels},
      Background -> White,
      ImageSize -> 500,
      PlotRange -> {{-1.3, 1.3}, {-1.3, 1.3}},
      AspectRatio -> Automatic]
  }, Alignment -> Center],
  Background -> RGBColor[0.95, 0.95, 0.95],
  FrameMargins -> 15]
];

(* Export disc visualizations for F=1 graphs *)
outputDir = "/Users/xiyin/LocalWork/StringMC_local/ribbon graphs/disc_visualizations/";

(* Create output directory *)
If[!DirectoryQ[outputDir], CreateDirectory[outputDir]];
Print["Output directory: ", outputDir];

(* Theta graph (g=1, F=1) *)
Print["\nGenerating disc visualization for theta graph..."];
graphs = GenerateCubicRibbonGraphs[1, 3];
If[Length[graphs] > 0,
  vis = DrawDiscVisualization[graphs[[1]]];
  Export[outputDir <> "disc_theta_g1_F1.png", vis];
  Print["Exported: disc_theta_g1_F1.png"];
];

(* Genus 2 graphs (F=1) *)
Print["\nGenerating disc visualizations for genus 2 graphs..."];
graphs = GenerateCubicRibbonGraphs[1, 9];
Do[
  vis = DrawDiscVisualization[graphs[[i]]];
  Export[outputDir <> "disc_genus2_F1_graph_" <> ToString[i] <> ".png", vis];
  Print["Exported: disc_genus2_F1_graph_" <> ToString[i] <> ".png"];
  ,
  {i, Length[graphs]}
];

Print["\nDone! Exported ", Length[graphs] + 1, " disc visualizations to ", outputDir];
