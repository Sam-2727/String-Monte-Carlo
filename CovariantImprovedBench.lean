import LeanRace

open LeanRace.Covariant
open Lean

def defaultImprovedCases : List (Nat × Nat × Nat) :=
  [(160, 40, 30), (320, 80, 60), (640, 160, 120)]

structure ImprovedBenchConfig where
  cases : List CylCase := []
  repeats : Nat := 5
  number : Nat := 1
  warmup : Nat := 1
  targetMs : Nat := 50
  json : Bool := false

def parseNatArg (s : String) : Except String Nat :=
  match s.trimAscii.toString.toNat? with
  | some n => .ok n
  | none => .error s!"Invalid natural number: {s}"

def parseCaseArg (s : String) : Except String CylCase := do
  match s.splitOn "," with
  | [a, b, c] =>
      let L ← parseNatArg a
      let l1 ← parseNatArg b
      let l2 ← parseNatArg c
      match CylCase.mk? L l1 l2 with
      | .ok out => .ok out
      | .error err => .error err
  | _ => .error s!"Invalid case format: {s}"

partial def parseArgs : ImprovedBenchConfig → List String → Except String ImprovedBenchConfig
  | cfg, [] =>
      if cfg.cases.isEmpty then do
        let cases ← defaultImprovedCases.mapM (fun (L, l1, l2) =>
          match CylCase.mk? L l1 l2 with
          | .ok c => .ok c
          | .error e => .error e)
        return { cfg with cases := cases }
      else
        .ok cfg
  | cfg, "--case" :: value :: rest => do
      let c ← parseCaseArg value
      parseArgs { cfg with cases := cfg.cases ++ [c] } rest
  | cfg, "--repeat" :: value :: rest => do
      let n ← parseNatArg value
      parseArgs { cfg with repeats := n } rest
  | cfg, "--number" :: value :: rest => do
      let n ← parseNatArg value
      parseArgs { cfg with number := n } rest
  | cfg, "--warmup" :: value :: rest => do
      let n ← parseNatArg value
      parseArgs { cfg with warmup := n } rest
  | cfg, "--target-ms" :: value :: rest => do
      let n ← parseNatArg value
      parseArgs { cfg with targetMs := n } rest
  | cfg, "--json" :: rest =>
      parseArgs { cfg with json := true } rest
  | _, flag :: _ =>
      .error s!"Unknown or incomplete argument: {flag}"

def runConfig (cfg : ImprovedBenchConfig) : IO UInt32 := do
  if cfg.repeats == 0 || cfg.number == 0 then
    IO.eprintln "repeat and number must be positive."
    return 1
  let mut results : Array ImprovedCaseBenchmark := #[]
  for c in cfg.cases do
    let r ← benchmarkImprovedCase c cfg.repeats cfg.number cfg.warmup cfg.targetMs
    results := results.push r
    if !cfg.json then
      printImprovedCaseBenchmark r
  if cfg.json then
    IO.println <| Json.compress <| Json.mkObj [("cases", toJson results)]
  return 0

def main (args : List String) : IO UInt32 := do
  match parseArgs {} args with
  | .ok cfg => runConfig cfg
  | .error err =>
      IO.eprintln err
      return 1
