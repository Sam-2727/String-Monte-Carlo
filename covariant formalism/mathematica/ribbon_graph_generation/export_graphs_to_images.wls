#!/usr/bin/env wolframscript
(* Export Ribbon Graphs to PNG Images - FIXED *)
(* Fixed critical bug in automorphism handling *)

Clear["Global`*"];

(* ========== COPY ALL FUNCTIONS FROM V1 ========== *)

GenerateCubicRibbonGraphs[nFaces_, nEdges_] := Module[
  {nVertices, genus, cubicGraphs, ribbonGraphs},
  nVertices = 2 * nEdges / 3;
  genus = (2 - nVertices + nEdges - nFaces) / 2;
  Print["V=", nVertices, ", E=", nEdges, ", F=", nFaces, ", g=", genus];
  cubicGraphs = GenerateBaseCubicGraphs[nVertices, nEdges];
  If[Length[cubicGraphs] == 0, Return[{}]];
  ribbonGraphs = Flatten[GenerateRotationSystems[#, nFaces] & /@ cubicGraphs, 1];
  Print["Found ", Length[ribbonGraphs], " ribbon graphs total"];
  If[Length[ribbonGraphs] == 0, Return[{}]];
  If[nVertices == 2 && nEdges == 3,
    ribbonGraphs = RemoveIsomorphismsTheta[ribbonGraphs],
    ribbonGraphs = RemoveIsomorphismsGeneral[ribbonGraphs]
  ];
  Print["Found ", Length[ribbonGraphs], " non-isomorphic ribbon graphs"];
  ribbonGraphs
];

MakePrismGraph3[] := Graph[Range[6], {
  1 <-> 2, 2 <-> 3, 3 <-> 1, 4 <-> 5, 5 <-> 6, 6 <-> 4,
  1 <-> 4, 2 <-> 5, 3 <-> 6
}];

GenerateBaseCubicGraphs[n_, e_] := Module[{graphs = {}},
  Which[
    n == 2 && e == 3, graphs = {Graph[{1, 2}, {1 <-> 2, 1 <-> 2, 1 <-> 2}]},
    n == 4 && e == 6, graphs = {CompleteGraph[4]},
    n == 6 && e == 9, graphs = {CompleteGraph[{3, 3}], MakePrismGraph3[]},
    n == 8 && e == 12, graphs = {CubeGraph[3]},
    True, graphs = {}
  ];
  Print["Using ", Length[graphs], " base cubic graph(s)"];
  graphs
];

GenerateRotationSystems[graph_, targetFaces_] := Module[
  {vertices, labeledEdges, incidentEdges, allRotations, validRibbons},
  vertices = VertexList[graph];
  labeledEdges = MapIndexed[{#1, First[#2]} &, EdgeList[graph]];
  incidentEdges = Table[
    Select[labeledEdges, MemberQ[{#[[1, 1]], #[[1, 2]]}, v] &][[All, 2]],
    {v, vertices}
  ];
  allRotations = Tuples[Permutations /@ incidentEdges];
  validRibbons = Select[allRotations, CountFaces[labeledEdges, #, vertices] == targetFaces &];
  Print["  Found ", Length[validRibbons], " rotation systems with F=", targetFaces];
  Table[{graph, rot}, {rot, validRibbons}]
];

CountFaces[labeledEdges_, rotation_, vertices_] := Module[
  {halfEdges, visited, faces, current},
  halfEdges = Flatten[Table[
    {{le[[1, 1]], le[[1, 2]], le[[2]]}, {le[[1, 2]], le[[1, 1]], le[[2]]}},
    {le, labeledEdges}
  ], 1];
  visited = Association[Table[he -> False, {he, halfEdges}]];
  faces = 0;
  Do[
    If[!visited[he],
      faces++;
      current = he;
      While[!visited[current],
        visited[current] = True;
        current = NextHalfEdge[current, labeledEdges, rotation, vertices];
        If[visited[current], Break[]]
      ]
    ],
    {he, halfEdges}
  ];
  faces
];

GetFaceBoundaries[labeledEdges_, rotation_, vertices_] := Module[
  {halfEdges, visited, faces, current, face},
  halfEdges = Flatten[Table[
    {{le[[1, 1]], le[[1, 2]], le[[2]]}, {le[[1, 2]], le[[1, 1]], le[[2]]}},
    {le, labeledEdges}
  ], 1];
  visited = Association[Table[he -> False, {he, halfEdges}]];
  faces = {};
  Do[
    If[!visited[he],
      face = {};
      current = he;
      While[!visited[current],
        visited[current] = True;
        AppendTo[face, current[[3]]];
        current = NextHalfEdge[current, labeledEdges, rotation, vertices];
        If[visited[current], Break[]]
      ];
      AppendTo[faces, face]
    ],
    {he, halfEdges}
  ];
  faces
];

NextHalfEdge[he_, labeledEdges_, rotation_, vertices_] := Module[
  {from, to, label, vertexIdx, rotAtVertex, pos, nextLabel, nextEdge},
  {from, to, label} = he;
  vertexIdx = FirstPosition[vertices, to][[1]];
  rotAtVertex = rotation[[vertexIdx]];
  pos = FirstPosition[rotAtVertex, label][[1]];
  nextLabel = rotAtVertex[[Mod[pos, Length[rotAtVertex]] + 1]];
  nextEdge = Select[labeledEdges, #[[2]] == nextLabel &][[1, 1]];
  If[nextEdge[[1]] == to,
    {to, nextEdge[[2]], nextLabel},
    {to, nextEdge[[1]], nextLabel}
  ]
];

RemoveIsomorphismsTheta[ribbonGraphs_] := Module[{unique, removed},
  If[Length[ribbonGraphs] == 0, Return[{}]];
  Print["Using specialized theta graph isomorphism check..."];
  unique = {ribbonGraphs[[1]]};
  removed = 0;
  Do[
    If[!AnyTrue[unique, ThetaRibbonIsomorphic[ribbonGraphs[[i]], #] &],
      AppendTo[unique, ribbonGraphs[[i]]], removed++
    ],
    {i, 2, Length[ribbonGraphs]}
  ];
  Print["Removed ", removed, " isomorphic duplicates"];
  unique
];

ThetaRibbonIsomorphic[rg1_, rg2_] := Module[{rot1, rot2},
  rot1 = rg1[[2]]; rot2 = rg2[[2]];
  AnyTrue[{False, True}, Function[vertexSwap,
    AnyTrue[Permutations[{1, 2, 3}], Function[edgePerm,
      ThetaRotationsMatch[rot1, rot2, edgePerm, vertexSwap]
    ]]
  ]]
];

ThetaRotationsMatch[rot1_, rot2_, edgePerm_, vertexSwap_] := Module[
  {r1v1, r1v2, r2v1, r2v2},
  r1v1 = Table[edgePerm[[rot1[[1, i]]]], {i, 3}];
  r1v2 = Table[edgePerm[[rot1[[2, i]]]], {i, 3}];
  r2v1 = rot2[[1]]; r2v2 = rot2[[2]];
  If[vertexSwap,
    CyclicMatch[r1v2, r2v1] && CyclicMatch[r1v1, r2v2],
    CyclicMatch[r1v1, r2v1] && CyclicMatch[r1v2, r2v2]
  ]
];

RemoveIsomorphismsGeneral[ribbonGraphs_] := Module[
  {graphGroups, unique, totalRemoved},
  If[Length[ribbonGraphs] == 0, Return[{}]];
  Print["Grouping by underlying graph..."];
  graphGroups = GatherBy[ribbonGraphs, Function[rg,
    Sort[EdgeList[rg[[1]]] /. UndirectedEdge -> List]
  ]];
  Print["Found ", Length[graphGroups], " distinct graph structure(s)"];
  unique = {}; totalRemoved = 0;
  Do[
    Print["Processing group ", i, "/", Length[graphGroups], " (", Length[graphGroups[[i]]], " ribbon graphs)"];
    Module[{groupUnique, removed},
      {groupUnique, removed} = RemoveIsomorphismsInGroup[graphGroups[[i]]];
      unique = Join[unique, groupUnique];
      totalRemoved += removed;
    ],
    {i, Length[graphGroups]}
  ];
  Print["Removed ", totalRemoved, " isomorphic duplicates"];
  unique
];

RemoveIsomorphismsInGroup[ribbonGraphs_] := Module[
  {graph, automorphisms, unique, removed},
  graph = ribbonGraphs[[1, 1]];
  Print["  Computing automorphisms..."];
  automorphisms = ComputeGraphAutomorphisms[graph];
  Print["  Found ", Length[automorphisms], " automorphisms"];
  unique = {ribbonGraphs[[1]]}; removed = 0;
  Print["  Checking isomorphisms..."];
  Do[
    If[Mod[i, 1000] == 0,
      Print["    Processed ", i, "/", Length[ribbonGraphs], " (", Length[unique], " unique)"]
    ];
    If[!AnyTrue[unique, RibbonIsomorphicQ[ribbonGraphs[[i]], #, automorphisms, graph] &],
      AppendTo[unique, ribbonGraphs[[i]]], removed++
    ],
    {i, 2, Length[ribbonGraphs]}
  ];
  {unique, removed}
];

(* FIXED: Compute graph automorphisms with proper handling of large groups *)
ComputeGraphAutomorphisms[graph_] := Module[{ag, elements, n, groupOrder},
  n = VertexCount[graph];
  ag = Quiet[GraphAutomorphismGroup[graph]];

  If[Head[ag] === PermutationGroup,
    groupOrder = GroupOrder[ag];

    (* FIXED: Increased threshold and added warning for large groups *)
    If[groupOrder <= 100000,
      (* Use full group for manageable sizes *)
      elements = GroupElements[ag];
      ,
      (* For very large groups, warn user and use generators *)
      Print["    WARNING: Automorphism group is very large (order ", groupOrder, ")"];
      Print["    Using generators only - may miss some isomorphisms"];
      elements = Prepend[GroupGenerators[ag], Cycles[{}]];
    ];

    Return[PermutationList[#, n] & /@ elements]
  ];

  (* Fallback: identity only *)
  {Range[n]}
];

RibbonIsomorphicQ[rg1_, rg2_, automorphisms_, graph_] := Module[
  {rot1, rot2, edges, edgePairs},
  rot1 = rg1[[2]]; rot2 = rg2[[2]];
  edges = EdgeList[graph];
  edgePairs = Table[Sort[{edges[[i, 1]], edges[[i, 2]]}], {i, Length[edges]}];
  AnyTrue[automorphisms, CheckAutomorphism[#, rot1, rot2, edgePairs] &]
];

CheckAutomorphism[vertexPerm_, rot1_, rot2_, edgePairs_] := Module[
  {n, m, edgeMap, i, oldPair, newPair, found, v, mappedRot, targetRot},
  n = Length[rot1]; m = Length[edgePairs];
  edgeMap = Table[0, {m}];
  For[i = 1, i <= m, i++,
    oldPair = edgePairs[[i]];
    newPair = Sort[{vertexPerm[[oldPair[[1]]]], vertexPerm[[oldPair[[2]]]]}];
    found = FirstPosition[edgePairs, newPair];
    If[found === Missing["NotFound"], Return[False], edgeMap[[i]] = found[[1]]]
  ];
  For[v = 1, v <= n, v++,
    mappedRot = Table[edgeMap[[rot1[[v, j]]]], {j, Length[rot1[[v]]]}];
    targetRot = rot2[[vertexPerm[[v]]]];
    If[!CyclicMatch[mappedRot, targetRot], Return[False]]
  ];
  True
];

CyclicMatch[list1_, list2_] := Module[{n},
  n = Length[list1];
  If[n != Length[list2], Return[False]];
  MemberQ[Table[RotateLeft[list1, i], {i, 0, n - 1}], list2]
];

(* ========== VISUALIZATION FUNCTION ========== *)

ShowRibbonGraph[rg_] := Module[
  {graph, rotation, v, e, f, g, labeledEdges, vertices, faceBoundaries,
   faceColors, darkBg, lightText, accentColor},
  {graph, rotation} = rg;
  vertices = VertexList[graph];
  v = Length[vertices];
  e = EdgeCount[graph];
  labeledEdges = MapIndexed[{#1, First[#2]} &, EdgeList[graph]];
  faceBoundaries = GetFaceBoundaries[labeledEdges, rotation, vertices];
  f = Length[faceBoundaries];
  g = (2 - v + e - f) / 2;

  darkBg = RGBColor[0.15, 0.15, 0.18];
  lightText = RGBColor[0.9, 0.9, 0.92];
  accentColor = RGBColor[0.3, 0.6, 0.9];
  faceColors = Table[ColorData["Rainbow"][i / (Max[f, 1] + 1)], {i, f}];

  Panel[Column[{
    Style["Ribbon Graph", Bold, 20, lightText],
    Spacer[5],
    Grid[{
      {Style["Vertices:", Bold, lightText], Style[v, lightText],
       Style["Edges:", Bold, lightText], Style[e, lightText]},
      {Style["Faces:", Bold, lightText], Style[f, lightText],
       Style["Genus:", Bold, lightText], Style[g, lightText]},
      {Style["Surface:", Bold, lightText],
       Style[If[g == 0, "Sphere", If[g == 1, "Torus", "Genus " <> ToString[g] <> " surface"]], accentColor],
       SpanFromLeft, SpanFromLeft}
    }, Frame -> All, Background -> RGBColor[0.2, 0.2, 0.23],
       FrameStyle -> RGBColor[0.4, 0.4, 0.45], Spacings -> {2, 1}],
    Spacer[10],
    Style["Rotation System (cyclic order at each vertex):", Bold, 14, accentColor],
    Spacer[5],
    Grid[Table[
      {Style["V" <> ToString[i] <> ":", Bold, lightText],
       Style[Row[Riffle[rotation[[i]], " → "]], lightText]},
      {i, Min[Length[rotation], 8]}
    ], Alignment -> Left, Frame -> All, Background -> RGBColor[0.2, 0.25, 0.2],
       FrameStyle -> RGBColor[0.3, 0.4, 0.3], Spacings -> {2, 1}],
    Spacer[10],
    Style["Face Boundaries (edges around each face):", Bold, 14, accentColor],
    Spacer[5],
    Grid[Table[
      {Graphics[{faceColors[[i]], Disk[]}, ImageSize -> 20, Background -> darkBg],
       Style["Face " <> ToString[i] <> ":", Bold, lightText],
       Style["[ " <> StringRiffle[ToString /@ faceBoundaries[[i]], ", "] <> " ]",
             FontFamily -> "Courier", lightText],
       Style["(" <> ToString[Length[faceBoundaries[[i]]]] <> " edges)",
             Italic, RGBColor[0.6, 0.6, 0.65]]},
      {i, Min[Length[faceBoundaries], 10]}
    ], Alignment -> {Left, Left, Left, Left}, Frame -> True,
       Background -> RGBColor[0.18, 0.22, 0.28], FrameStyle -> RGBColor[0.3, 0.4, 0.5],
       Spacings -> {2, 1}],
    If[Length[faceBoundaries] > 10,
      Style["(showing first 10 faces)", Italic, 10, RGBColor[0.6, 0.6, 0.65]],
      Nothing],
    Spacer[10],
    Style["Note: Each face is bounded by the listed edges traversed in cyclic order.",
          Italic, 11, RGBColor[0.7, 0.7, 0.75]],
    Spacer[10],
    Graph[graph,
      VertexLabels -> Placed["Name", Center],
      VertexLabelStyle -> Directive[Bold, 10, White],
      VertexStyle -> Directive[RGBColor[0.4, 0.7, 1.0], EdgeForm[{White}]],
      VertexSize -> 0.06,
      EdgeStyle -> Directive[Thick, RGBColor[0.6, 0.8, 1.0]],
      EdgeLabels -> Table[i -> Placed[Style[ToString[i], Bold, 11, Background -> RGBColor[0.9, 0.5, 0.2]],
                                       {0.5, {0, 0.3}}], {i, e}],
      GraphLayout -> "SpringElectricalEmbedding",
      ImageSize -> 450,
      Background -> RGBColor[0.12, 0.12, 0.15]]
  }, Alignment -> Center, Spacings -> 2],
  Background -> darkBg, FrameMargins -> 20, ImageSize -> 700]
];

(* ========== EXPORT SCRIPT ========== *)

Print["════════════════════════════════════════════════════════"];
Print["EXPORTING RIBBON GRAPHS TO PNG IMAGES (FIXED)"];
Print["════════════════════════════════════════════════════════\n"];

(* Create output directory *)
outputDir = FileNameJoin[{Directory[], "ribbon_graph_images"}];
If[!DirectoryQ[outputDir], CreateDirectory[outputDir]];
Print["Output directory: ", outputDir, "\n"];

(* Generate and export graphs *)
exportGraphs[name_, nFaces_, nEdges_] := Module[{graphs, filename, vis},
  Print["Generating: ", name];
  graphs = GenerateCubicRibbonGraphs[nFaces, nEdges];

  Do[
    filename = FileNameJoin[{outputDir, name <> "_graph_" <> ToString[i] <> ".png"}];
    vis = ShowRibbonGraph[graphs[[i]]];
    Export[filename, vis, "PNG", ImageResolution -> 150];
    Print["  Exported: ", filename];
    ,
    {i, Length[graphs]}
  ];
  Print[""];
];

(* Export all test cases *)
exportGraphs["theta_torus_g1_F1", 1, 3];
exportGraphs["theta_sphere_g0_F3", 3, 3];
exportGraphs["K4_sphere_g0_F4", 4, 6];
exportGraphs["genus2_F1", 1, 9];

Print["════════════════════════════════════════════════════════"];
Print["EXPORT COMPLETE!"];
Print["All images saved to: ", outputDir];
Print["════════════════════════════════════════════════════════"];
