import Lake
open Lake DSL System

package «LeanRace» where
  moreLinkArgs := #["/usr/lib/liblapack.so", "/usr/lib/libblas.so", "-lm"]

input_file covariantBdetSrc where
  path := "native/covariant_bdet.c"

target covariantBdetO pkg : FilePath := do
  let srcJob ← covariantBdetSrc.fetch
  let lean ← getLeanInstall
  let oFile := pkg.buildDir / "native" / "covariant_bdet.o"
  buildO oFile srcJob #[] #["-I", lean.includeDir.toString, "-O3"]

input_file covariantImprovedSrc where
  path := "native/covariant_improved.c"

target covariantImprovedO pkg : FilePath := do
  let srcJob ← covariantImprovedSrc.fetch
  let lean ← getLeanInstall
  let oFile := pkg.buildDir / "native" / "covariant_improved.o"
  buildO oFile srcJob #[] #["-I", lean.includeDir.toString, "-O3"]

lean_lib «LeanRace» where
  moreLinkObjs := #[covariantBdetO, covariantImprovedO]

@[default_target]
lean_exe leanrace where
  root := `Main

lean_exe «covariant-bench» where
  root := `CovariantBench

lean_exe «covariant-improved-bench» where
  root := `CovariantImprovedBench
  supportInterpreter := true

lean_exe «covariant-det-bench» where
  root := `CovariantDetBench
  supportInterpreter := true

lean_exe «covariant-period-det-bench» where
  root := `CovariantPeriodDetBench
  supportInterpreter := true
