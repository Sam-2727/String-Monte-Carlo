import Lean
import Std

open Lean

namespace LeanRace
namespace Covariant

def pi : Float := 3.14159265358979323846

structure NumericError where
  message : String
deriving Repr, Inhabited

instance : ToString NumericError where
  toString e := e.message

structure C64 where
  re : Float
  im : Float
deriving Repr, Inhabited, BEq

namespace C64

def zero : C64 := ⟨0.0, 0.0⟩
def one : C64 := ⟨1.0, 0.0⟩

instance : Add C64 where
  add a b := ⟨a.re + b.re, a.im + b.im⟩

instance : Sub C64 where
  sub a b := ⟨a.re - b.re, a.im - b.im⟩

instance : Neg C64 where
  neg a := ⟨-a.re, -a.im⟩

instance : Mul C64 where
  mul a b := ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩

instance : Div C64 where
  div a b :=
    let denom := b.re * b.re + b.im * b.im
    ⟨(a.re * b.re + a.im * b.im) / denom, (a.im * b.re - a.re * b.im) / denom⟩

def scale (x : Float) (z : C64) : C64 := ⟨x * z.re, x * z.im⟩

def conj (z : C64) : C64 := ⟨z.re, -z.im⟩

def normSq (z : C64) : Float := z.re * z.re + z.im * z.im

def abs (z : C64) : Float := Float.sqrt z.normSq

def arg (z : C64) : Float := Float.atan2 z.im z.re

def cis (theta : Float) : C64 := ⟨Float.cos theta, Float.sin theta⟩

def exp (z : C64) : C64 :=
  let mag := Float.exp z.re
  ⟨mag * Float.cos z.im, mag * Float.sin z.im⟩

def log (z : C64) : C64 := ⟨Float.log z.abs, z.arg⟩

def powReal (z : C64) (p : Float) : C64 :=
  if z.re == 0.0 && z.im == 0.0 then
    zero
  else
    exp (scale p (log z))

def chop (tol : Float) (z : C64) : C64 :=
  let re := if Float.abs z.re < tol then 0.0 else z.re
  let im := if Float.abs z.im < tol then 0.0 else z.im
  ⟨re, im⟩

def asJson (z : C64) : Json :=
  Json.mkObj [("re", Lean.toJson z.re), ("im", Lean.toJson z.im)]

instance : ToJson C64 where
  toJson := asJson

end C64

structure Vec (α : Type) (n : Nat) where
  data : Array α
  size_eq : data.size = n

namespace Vec

def ofFn (f : Fin n → α) : Vec α n :=
  ⟨Array.ofFn f, Array.size_ofFn⟩

def get (v : Vec α n) (i : Fin n) : α :=
  v.data[i.1]'(by simp [v.size_eq] at i ⊢)

def getNat! [Inhabited α] (v : Vec α n) (i : Nat) : α :=
  v.data[i]!

def asJson [ToJson α] (v : Vec α n) : Json :=
  Lean.toJson v.data.toList

instance [ToJson α] : ToJson (Vec α n) where
  toJson := asJson

end Vec

structure Mat (α : Type) (m n : Nat) where
  data : Array α
  size_eq : data.size = m * n

namespace Mat

def index (n row col : Nat) : Nat := row * n + col

def ofFn (f : Fin m → Fin n → α) : Mat α m n :=
  let data := Array.ofFn fun ij : Fin (m * n) =>
    let row := ij.1 / n
    let col := ij.1 % n
    have hrow : row < m := by
      have hij : ij.1 < n * m := by simp [Nat.mul_comm] at ij ⊢
      exact Nat.div_lt_of_lt_mul hij
    have hn : n ≠ 0 := by
      intro h0
      simpa [h0] using ij.2
    have hcol : col < n := Nat.mod_lt _ (Nat.pos_of_ne_zero hn)
    f ⟨row, hrow⟩ ⟨col, hcol⟩
  ⟨data, Array.size_ofFn⟩

def get (M : Mat α m n) (row : Fin m) (col : Fin n) : α :=
  let idx := index n row.1 col.1
  have hidx : idx < M.data.size := by
    have h1 : row.1 * n + col.1 < row.1 * n + n := Nat.add_lt_add_left col.2 _
    have h2 : row.1 * n + n ≤ m * n := by
      calc
        row.1 * n + n = (row.1 + 1) * n := by rw [Nat.succ_mul]
        _ ≤ m * n := Nat.mul_le_mul_right n (Nat.succ_le_of_lt row.2)
    have h : row.1 * n + col.1 < m * n := Nat.lt_of_lt_of_le h1 h2
    simpa [M.size_eq] using h
  M.data[idx]'hidx

def getNat! [Inhabited α] (M : Mat α m n) (row col : Nat) : α :=
  M.data[index n row col]!

def asJson [ToJson α] [Inhabited α] (M : Mat α m n) : Json :=
  let rows := (Array.range m).map fun row =>
    Json.arr <| (Array.range n).map fun col => Lean.toJson (M.getNat! row col)
  Json.arr rows

instance [ToJson α] [Inhabited α] : ToJson (Mat α m n) where
  toJson := asJson

end Mat

structure CylCase where
  half : Nat
  l1 : Nat
  l2 : Nat
  hsum : l1 + l2 ≤ half
deriving Repr

namespace CylCase

def mk? (L l1 l2 : Nat) : Except String CylCase :=
  if _ : L % 2 = 0 then
    let half := L / 2
    if hsum : l1 + l2 ≤ half then
      .ok ⟨half, l1, l2, hsum⟩
    else
      .error "Need l1 + l2 <= L/2."
  else
    .error "L must be even."

def L (c : CylCase) : Nat := 2 * c.half

def l3 (c : CylCase) : Nat := c.half - (c.l1 + c.l2)

def allEqual (c : CylCase) : Bool := c.l1 == c.l2 && c.l2 == c.l3

def allOdd (c : CylCase) : Bool :=
  c.l1 % 2 == 1 && c.l2 % 2 == 1 && c.l3 % 2 == 1

def allEven (c : CylCase) : Bool :=
  c.l1 % 2 == 0 && c.l2 % 2 == 0 && c.l3 % 2 == 0

def coeffCount (c : CylCase) : Nat :=
  if c.allEqual then
    1
  else if c.allOdd then
    c.half - 1
  else if c.allEven then
    c.half + 1
  else
    c.half

def asJson (c : CylCase) : Json :=
  Json.mkObj
    [ ("L", Lean.toJson c.L)
    , ("l1", Lean.toJson c.l1)
    , ("l2", Lean.toJson c.l2)
    ]

instance : ToJson CylCase where
  toJson := asJson

end CylCase

structure Poly (n : Nat) where
  coeffs : Vec C64 n

structure ImprovedPoly (c : CylCase) where
  coeffs : Vec C64 c.coeffCount
  packedCoeffs : FloatArray
  w1 : C64
  w2 : C64
  useTwoFactor : Bool

structure PeriodTriple where
  p1 : C64
  p2 : C64
  p3 : C64
deriving Repr

instance : ToJson PeriodTriple where
  toJson p := Json.mkObj
    [("p1", Lean.toJson p.p1), ("p2", Lean.toJson p.p2), ("p3", Lean.toJson p.p3)]

def natToFloat (n : Nat) : Float := Float.ofNat n

def roundHalfEvenNat (n : Nat) : Nat :=
  let q := n / 2
  if n % 2 == 0 then q else if q % 2 == 0 then q else q + 1

def polyEval {n : Nat} (coeffs : Vec C64 n) (z : C64) : C64 :=
  Id.run do
    let mut acc := C64.zero
    for idx in [0:n] do
      let j := n - 1 - idx
      acc := acc * z + coeffs.getNat! j
    return acc

def antiCoeffs {n : Nat} (coeffs : Vec C64 n) : Vec C64 n :=
  Vec.ofFn fun i =>
    coeffs.get i / ⟨natToFloat (i.1 + 1), 0.0⟩

def polyAntiderivativeEval {n : Nat} (coeffs : Vec C64 n) (z : C64) : C64 :=
  let anti := antiCoeffs coeffs
  z * polyEval anti z

@[extern "lean_covariant_complex_gram"]
opaque complexGramNative
  (rows cols mode : USize) (entries : @& FloatArray) : FloatArray

@[extern "lean_covariant_complex_solve"]
opaque complexSolveNative
  (n : USize) (entries rhsEntries : @& FloatArray) : FloatArray

@[extern "lean_covariant_complex_log_abs_det"]
opaque complexLogAbsDetNative (n : USize) (entries : @& FloatArray) : Float

@[extern "lean_covariant_real_spd_log_det"]
opaque realSpdLogDetNative (n : USize) (entries : @& FloatArray) : Float

def complexGramConjMode : USize := 0
def complexGramTransposeMode : USize := 1

@[inline] def packedComplexScalarCount (n : Nat) : Nat := 2 * n

@[inline] def packedComplexMatrixScalarCount (rows cols : Nat) : Nat :=
  2 * rows * cols

@[inline] def packedComplexGet (xs : FloatArray) (i : Nat) : C64 :=
  ⟨xs[2 * i]!, xs[2 * i + 1]!⟩

@[inline] def packedComplexSet! (xs : FloatArray) (i : Nat) (z : C64) : FloatArray :=
  let ys := xs.set! (2 * i) z.re
  ys.set! (2 * i + 1) z.im

@[inline] def packedComplexPush (xs : FloatArray) (z : C64) : FloatArray :=
  (xs.push z.re).push z.im

@[inline] def packedComplexReplicate (n : Nat) (z : C64) : FloatArray :=
  Id.run do
    let mut out := FloatArray.emptyWithCapacity (packedComplexScalarCount n)
    for _ in [0:n] do
      out := packedComplexPush out z
    return out

@[inline] def colMajorIndex (rows row col : Nat) : Nat :=
  row + col * rows

@[inline] def rowMajorIndex (cols row col : Nat) : Nat :=
  row * cols + col

@[inline] def mod1Nat (x L : Nat) : Nat :=
  ((x - 1) % L) + 1

def packedComplexE1 (n : Nat) : FloatArray :=
  Id.run do
    let mut out := FloatArray.emptyWithCapacity (packedComplexScalarCount n)
    for i in [0:n] do
      if i == 0 then
        out := out.push 1.0
        out := out.push 0.0
      else
        out := out.push 0.0
        out := out.push 0.0
    return out

def unpackComplexVec (n : Nat) (coeffs : FloatArray) : Except NumericError (Vec C64 n) := do
  if coeffs.size != packedComplexScalarCount n then
    throw ⟨s!"Packed complex vector size mismatch: expected {packedComplexScalarCount n}, got {coeffs.size}."⟩
  let data := Array.ofFn fun i : Fin n =>
    let re := coeffs[2 * i.1]!
    let im := coeffs[2 * i.1 + 1]!
    ⟨re, im⟩
  return ⟨data, Array.size_ofFn⟩

@[inline] unsafe def packedComplexGetReNat (xs : FloatArray) (i : Nat) : Float :=
  xs[i]!

@[inline] unsafe def packedComplexGetImNat (xs : FloatArray) (i : Nat) : Float :=
  xs[i + 1]!

@[inline] unsafe def packedComplexSetPairNat
    (xs : FloatArray) (i : Nat) (re im : Float) : FloatArray :=
  let ys := xs.set! i re
  ys.set! (i + 1) im

def mappedIndex1 (c : CylCase) (k1 : Nat) : Nat :=
  if k1 ≤ c.l1 then
    c.half + c.l1 + 1 - k1
  else if k1 ≤ c.l1 + c.l2 then
    c.half + 2 * c.l1 + c.l2 + 1 - k1
  else
    c.L + c.l1 + c.l2 + 1 - k1

def buildBaseAngleTables (c : CylCase) : FloatArray × FloatArray :=
  Id.run do
    let mut left := FloatArray.emptyWithCapacity c.half
    let mut right := FloatArray.emptyWithCapacity c.half
    let twopiOverL := 2.0 * pi / natToFloat c.L
    for row in [0:c.half] do
      let k1 := row + 1
      let mapped := mappedIndex1 c k1
      left := left.push (twopiOverL * natToFloat mapped)
      right := right.push (twopiOverL * natToFloat k1)
    return (left, right)

unsafe def buildPowerPairMatrixLeanUnsafe
    (rows cols : Nat)
    (leftBase rightBase leftScale rightScale : FloatArray) : FloatArray :=
  Id.run do
    let rowsU := rows.toUSize
    let colsU := cols.toUSize
    let mut leftPow := leftBase
    let mut rightPow := rightBase
    let mut out := FloatArray.emptyWithCapacity (packedComplexMatrixScalarCount rows cols)
    let mut col : USize := 0
    while _hcol : col < colsU do
      let mut row : USize := 0
      while _hrow : row < rowsU do
        let i := 2 * row.toNat
        let lpRe := packedComplexGetReNat leftPow i
        let lpIm := packedComplexGetImNat leftPow i
        let rpRe := packedComplexGetReNat rightPow i
        let rpIm := packedComplexGetImNat rightPow i
        let lsRe := packedComplexGetReNat leftScale i
        let lsIm := packedComplexGetImNat leftScale i
        let rsRe := packedComplexGetReNat rightScale i
        let rsIm := packedComplexGetImNat rightScale i
        let zRe := lsRe * lpRe - lsIm * lpIm + (rsRe * rpRe - rsIm * rpIm)
        let zIm := lsRe * lpIm + lsIm * lpRe + (rsRe * rpIm + rsIm * rpRe)
        out := out.push zRe
        out := out.push zIm
        let lbRe := packedComplexGetReNat leftBase i
        let lbIm := packedComplexGetImNat leftBase i
        let rbRe := packedComplexGetReNat rightBase i
        let rbIm := packedComplexGetImNat rightBase i
        let nextLpRe := lpRe * lbRe - lpIm * lbIm
        let nextLpIm := lpRe * lbIm + lpIm * lbRe
        let nextRpRe := rpRe * rbRe - rpIm * rbIm
        let nextRpIm := rpRe * rbIm + rpIm * rbRe
        leftPow := packedComplexSetPairNat leftPow i nextLpRe nextLpIm
        rightPow := packedComplexSetPairNat rightPow i nextRpRe nextRpIm
        row := row + 1
      col := col + 1
    return out

def buildPowerPairMatrixLean
    (rows cols : Nat)
    (leftBase rightBase leftScale rightScale : FloatArray) : FloatArray :=
  unsafe buildPowerPairMatrixLeanUnsafe rows cols leftBase rightBase leftScale rightScale

unsafe def buildBaseMatrixUnsafe (c : CylCase) (leftAngles rightAngles : FloatArray) : FloatArray :=
  Id.run do
    let rowsU := c.half.toUSize
    let colsU := c.half.toUSize
    let mut out := FloatArray.emptyWithCapacity (packedComplexMatrixScalarCount c.half c.half)
    let mut col : USize := 0
    while _hcol : col < colsU do
      let n1 := natToFloat (col.toNat + 1)
      let mut row : USize := 0
      while _hrow : row < rowsU do
        let i := row.toNat
        let leftTheta := leftAngles[i]! * n1
        let rightTheta := rightAngles[i]! * n1
        out := out.push (Float.cos leftTheta + Float.cos rightTheta)
        out := out.push (Float.sin leftTheta + Float.sin rightTheta)
        row := row + 1
      col := col + 1
    return out

def buildBaseMatrix (c : CylCase) : FloatArray :=
  let (leftAngles, rightAngles) := buildBaseAngleTables c
  unsafe buildBaseMatrixUnsafe c leftAngles rightAngles

def improvementFactor (angle phase1 phase2 : Float) (twoFactor : Bool) : C64 :=
  let z0 := C64.one - C64.cis (2.0 * angle)
  let z1 := C64.one - C64.cis (2.0 * (angle - phase1))
  if twoFactor then
    z0.powReal (-0.5) * z1.powReal (-0.5)
  else
    let z2 := C64.one - C64.cis (2.0 * (angle - phase2))
    let exponent := -1.0 / 3.0
    z0.powReal exponent * z1.powReal exponent * z2.powReal exponent

def singularFactor (z w1 w2 : C64) (twoFactor : Bool) : C64 :=
  let z2 := z * z
  if twoFactor then
    (C64.one - z2).powReal (-0.5) * (C64.one - z2 * w1).powReal (-0.5)
  else
    let exponent := -1.0 / 3.0
    (C64.one - z2).powReal exponent *
      (C64.one - z2 * w1).powReal exponent *
      (C64.one - z2 * w2).powReal exponent

def improvedTwoFactor (c : CylCase) : Bool :=
  c.l3 == 0 || c.l1 == 0 || c.l2 == 0

def buildPowerPairMatrix
    (rows cols : Nat)
    (leftBase rightBase leftScale rightScale : FloatArray) : FloatArray :=
  buildPowerPairMatrixLean rows cols leftBase rightBase leftScale rightScale

def buildImprovedRowData (c : CylCase) : FloatArray × FloatArray × FloatArray × FloatArray :=
  Id.run do
    let twopiOverL := 2.0 * pi / natToFloat c.L
    let halfStep := pi / natToFloat c.L
    let phase1 := twopiOverL * natToFloat c.l1
    let phase2 := twopiOverL * natToFloat (c.l1 + c.l2)
    let twoFactor := improvedTwoFactor c
    let mut leftBase := FloatArray.emptyWithCapacity (packedComplexScalarCount c.half)
    let mut rightBase := FloatArray.emptyWithCapacity (packedComplexScalarCount c.half)
    let mut leftScale := FloatArray.emptyWithCapacity (packedComplexScalarCount c.half)
    let mut rightScale := FloatArray.emptyWithCapacity (packedComplexScalarCount c.half)
    for row in [0:c.half] do
      let k1 := row + 1
      let mapped := mappedIndex1 c k1
      let angleL := twopiOverL * natToFloat mapped - halfStep
      let angleR := twopiOverL * natToFloat k1 - halfStep
      leftBase := packedComplexPush leftBase (C64.cis angleL)
      rightBase := packedComplexPush rightBase (C64.cis angleR)
      leftScale := packedComplexPush leftScale (improvementFactor angleL phase1 phase2 twoFactor)
      rightScale := packedComplexPush rightScale (improvementFactor angleR phase1 phase2 twoFactor)
    return (leftBase, rightBase, leftScale, rightScale)

def buildImprovedMatrix (c : CylCase) : FloatArray :=
  let (leftBase, rightBase, leftScale, rightScale) := buildImprovedRowData c
  buildPowerPairMatrix c.half c.coeffCount leftBase rightBase leftScale rightScale

def buildBdetRowBases (c : CylCase) : FloatArray × FloatArray :=
  Id.run do
    let factor := 2.0 * pi / natToFloat c.L
    let mut rightBase := FloatArray.emptyWithCapacity (packedComplexScalarCount c.half)
    let mut leftBase := FloatArray.emptyWithCapacity (packedComplexScalarCount c.half)
    for row in [0:c.half] do
      let k1 := row + 1
      let mapped := mappedIndex1 c k1
      rightBase := packedComplexPush rightBase (C64.cis (factor * natToFloat k1))
      leftBase := packedComplexPush leftBase (C64.cis (factor * natToFloat mapped))
    return (rightBase, leftBase)

def buildBdetMatrix (c : CylCase) : FloatArray :=
  let (rightBase, leftBase) := buildBdetRowBases c
  buildPowerPairMatrix
    c.half c.half
    rightBase leftBase
    (packedComplexReplicate c.half C64.one)
    (packedComplexReplicate c.half (-C64.one))

def kernelValue (L d : Nat) : Float :=
  let c0 := Float.cos (pi / natToFloat L)
  let s0 := Float.sin (pi / natToFloat L)
  let denom := c0 - Float.cos (2.0 * pi * natToFloat d / natToFloat L)
  (-s0) / (natToFloat L * denom)

def buildKernelArray (L : Nat) : FloatArray :=
  Id.run do
    let mut out := FloatArray.emptyWithCapacity L
    for d in [0:L] do
      out := out.push (kernelValue L d)
    return out

def matFromKernel (kernel : FloatArray) (L row1 col1 : Nat) : Float :=
  kernel[(col1 + L - row1) % L]!

def tempGet (temp : FloatArray) (half row1 col1 : Nat) : Float :=
  temp[rowMajorIndex half (row1 - 1) (col1 - 1)]!

def tempSet! (temp : FloatArray) (half row1 col1 : Nat) (value : Float) : FloatArray :=
  temp.set! (rowMajorIndex half (row1 - 1) (col1 - 1)) value

def segmentMap (c : CylCase) (which x : Nat) : Nat :=
  if which == 1 then
    c.half + c.l1 + 1 - x
  else if which == 2 then
    c.half + 2 * c.l1 + c.l2 + 1 - x
  else
    mod1Nat (c.L + c.l1 + c.l2 + 1 - x) c.L

def primeSegments (c : CylCase) : Nat × Array (Nat × Nat) :=
  if c.l1 + c.l2 == c.half then
    (2, #[(1, c.l1), (c.l1 + 1, c.l1 + c.l2 - 1), (c.l1 + c.l2 + 1, c.half)])
  else
    (3, #[(1, c.l1), (c.l1 + 1, c.l1 + c.l2), (c.l1 + c.l2 + 1, c.half - 1)])

def buildPrimeDetReducedMatrix (c : CylCase) : FloatArray :=
  Id.run do
    let kernel := buildKernelArray c.L
    let mut temp := FloatArray.emptyWithCapacity (c.half * c.half)
    for row in [0:c.half] do
      for col in [0:c.half] do
        temp := temp.push (matFromKernel kernel c.L (row + 1) (col + 1))
    let (segmentFlag, segments) := primeSegments c
    let n := c.half
    let mappedN := segmentMap c segmentFlag n
    for idx in [0:segmentFlag] do
      let (a, b) := segments[idx]!
      if a ≤ b then
        for k in [a:b + 1] do
          let mappedK := segmentMap c (idx + 1) k
          let value :=
            tempGet temp c.half k n +
              matFromKernel kernel c.L mappedK n +
              matFromKernel kernel c.L mappedK mappedN +
              matFromKernel kernel c.L k mappedN
          temp := tempSet! temp c.half k n value
    for idx in [0:segmentFlag] do
      let (a, b) := segments[idx]!
      if a ≤ b then
        for l in [a:b + 1] do
          let mappedL := segmentMap c (idx + 1) l
          let value :=
            tempGet temp c.half n l +
              matFromKernel kernel c.L mappedN l +
              matFromKernel kernel c.L mappedN mappedL +
              matFromKernel kernel c.L n mappedL
          temp := tempSet! temp c.half n l value
    let tempNN :=
      tempGet temp c.half n n +
        matFromKernel kernel c.L mappedN n +
        matFromKernel kernel c.L mappedN mappedN +
        matFromKernel kernel c.L n mappedN
    temp := tempSet! temp c.half n n tempNN
    let fixedNN := tempGet temp c.half n n
    for iIdx in [0:segmentFlag] do
      let (ai, bi) := segments[iIdx]!
      if ai ≤ bi then
        for jIdx in [0:segmentFlag] do
          let (aj, bj) := segments[jIdx]!
          if aj ≤ bj then
            for k in [ai:bi + 1] do
              let mappedK := segmentMap c (iIdx + 1) k
              let tempKN := tempGet temp c.half k n
              for l in [aj:bj + 1] do
                let mappedL := segmentMap c (jIdx + 1) l
                let value :=
                  tempGet temp c.half k l +
                    matFromKernel kernel c.L mappedK l +
                    matFromKernel kernel c.L mappedK mappedL +
                    matFromKernel kernel c.L k mappedL -
                    tempKN -
                    tempGet temp c.half n l +
                    fixedNN
                temp := tempSet! temp c.half k l value
    let reducedN := c.half - 1
    let mut reduced := FloatArray.emptyWithCapacity (reducedN * reducedN)
    for col in [0:reducedN] do
      for row in [0:reducedN] do
        reduced := reduced.push (tempGet temp c.half (row + 1) (col + 1))
    return reduced

def makeCylEqn (c : CylCase) (chopTol : Float := 1e-12) :
    Except NumericError (Poly c.half) := do
  let mat := buildBaseMatrix c
  let gram := complexGramNative c.half.toUSize c.half.toUSize complexGramConjMode mat
  if gram.size != packedComplexMatrixScalarCount c.half c.half then
    throw ⟨"Native complex Gram build failed for make_cyl_eqn."⟩
  let gram' := packedComplexSet! gram 0 (packedComplexGet gram 0 + C64.one)
  let raw := complexSolveNative c.half.toUSize gram' (packedComplexE1 c.half)
  let coeffs ← unpackComplexVec c.half raw
  let coeffs' := Vec.ofFn fun i =>
    let z := (coeffs.get i).chop chopTol
    if i.1 + 1 == c.half then C64.zero else z
  return ⟨coeffs'⟩

def makeCylEqnImproved (c : CylCase) (chopTol : Float := 1e-12) :
    Except NumericError (ImprovedPoly c) := do
  if hEq : c.allEqual then
    have hcount : c.coeffCount = 1 := by
      simp [CylCase.coeffCount, hEq]
    let coeffs : Vec C64 c.coeffCount := hcount ▸ (⟨#[C64.one], by simp⟩ : Vec C64 1)
    let twopiOverL := 2.0 * pi / natToFloat c.L
    let phase1 := twopiOverL * natToFloat c.l1
    let phase2 := twopiOverL * natToFloat (c.l1 + c.l2)
    let packed := packedComplexE1 1
    return ⟨coeffs, packed, C64.cis (-2.0 * phase1), C64.cis (-2.0 * phase2), improvedTwoFactor c⟩
  let twopiOverL := 2.0 * pi / natToFloat c.L
  let phase1 := twopiOverL * natToFloat c.l1
  let phase2 := twopiOverL * natToFloat (c.l1 + c.l2)
  let w1 := C64.cis (-2.0 * phase1)
  let w2 := C64.cis (-2.0 * phase2)
  let twoFactor := improvedTwoFactor c
  let mat := buildImprovedMatrix c
  let gram :=
    complexGramNative c.half.toUSize c.coeffCount.toUSize complexGramTransposeMode mat
  if gram.size != packedComplexMatrixScalarCount c.coeffCount c.coeffCount then
    throw ⟨"Native complex Gram build failed for make_cyl_eqn_improved."⟩
  let gram' := packedComplexSet! gram 0 (packedComplexGet gram 0 + C64.one)
  let raw := complexSolveNative c.coeffCount.toUSize gram' (packedComplexE1 c.coeffCount)
  let coeffs ← unpackComplexVec c.coeffCount raw
  let coeffs' := Vec.ofFn fun i => (coeffs.get i).chop chopTol
  let packed :=
    Id.run do
      let mut out := FloatArray.emptyWithCapacity raw.size
      for i in [0:c.coeffCount] do
        out := packedComplexPush out ((coeffs.getNat! i).chop chopTol)
      return out
  return ⟨coeffs', packed, w1, w2, twoFactor⟩

def periodsWithPoly (c : CylCase) (p : Poly c.half) : PeriodTriple :=
  let m1 := roundHalfEvenNat c.l1
  let m2 := roundHalfEvenNat c.l2
  let m3 := roundHalfEvenNat c.l3
  let anti := antiCoeffs p.coeffs
  let F := fun z => z * polyEval anti z
  let expAngle := fun s : Nat => C64.cis (2.0 * pi * natToFloat s / natToFloat c.L)
  let k1 := m1
  let k2 := c.l1 + m2
  let k3 := c.l1 + c.l2 + m3
  let w1L := expAngle (c.half + c.l1 + 1 - k1)
  let w1R := expAngle k1
  let w2L := expAngle (c.half + 2 * c.l1 + c.l2 + 1 - k2)
  let w2R := expAngle k2
  let w3L := expAngle (c.L + c.l1 + c.l2 + 1 - k3)
  let w3R := expAngle k3
  { p1 := F w1L - F w1R
  , p2 := F w2L - F w2R
  , p3 := F w3L - F w3R
  }

def improvedPanelCount {c : CylCase} (p : ImprovedPoly c) : Nat :=
  if p.useTwoFactor then
    4096
  else
    1024

unsafe def polyEvalThreeChoppedPackedLeanUnsafe
    (coeffs : FloatArray) (coeffCount : Nat) (z1 z2 z3 : C64) (tol : Float) :
    C64 × C64 × C64 :=
  Id.run do
    let mut a1Re := 0.0
    let mut a1Im := 0.0
    let mut a2Re := 0.0
    let mut a2Im := 0.0
    let mut a3Re := 0.0
    let mut a3Im := 0.0
    let mut remaining : USize := coeffCount.toUSize
    while _h : 0 < remaining do
      let j := remaining.toNat - 1
      let cRe := coeffs[2 * j]!
      let cIm := coeffs[2 * j + 1]!
      let next1Re := a1Re * z1.re - a1Im * z1.im + cRe
      let next1Im := a1Re * z1.im + a1Im * z1.re + cIm
      let next2Re := a2Re * z2.re - a2Im * z2.im + cRe
      let next2Im := a2Re * z2.im + a2Im * z2.re + cIm
      let next3Re := a3Re * z3.re - a3Im * z3.im + cRe
      let next3Im := a3Re * z3.im + a3Im * z3.re + cIm
      a1Re := next1Re
      a1Im := next1Im
      a2Re := next2Re
      a2Im := next2Im
      a3Re := next3Re
      a3Im := next3Im
      remaining := remaining - 1
    let chopComponent := fun x : Float => if Float.abs x < tol then 0.0 else x
    (⟨chopComponent a1Re, chopComponent a1Im⟩,
     ⟨chopComponent a2Re, chopComponent a2Im⟩,
     ⟨chopComponent a3Re, chopComponent a3Im⟩)

def polyEvalThreeChoppedPackedLean
    (coeffs : FloatArray) (coeffCount : Nat) (z1 z2 z3 : C64) (tol : Float) :
    C64 × C64 × C64 :=
  unsafe polyEvalThreeChoppedPackedLeanUnsafe coeffs coeffCount z1 z2 z3 tol

def polyEvalThreeChoppedPacked
    (coeffs : FloatArray) (coeffCount : Nat) (z1 z2 z3 : C64) (tol : Float) :
    C64 × C64 × C64 :=
  polyEvalThreeChoppedPackedLean coeffs coeffCount z1 z2 z3 tol

def periodsImprovedWithPanelCount (c : CylCase) (p : ImprovedPoly c) (panels : Nat) :
    Except NumericError PeriodTriple := do
  if panels == 0 then
    throw ⟨"periods_improved requires a positive panel count."⟩
  let twopiOverL := 2.0 * pi / natToFloat c.L
  let theta1 := twopiOverL * natToFloat c.l1
  let theta2 := twopiOverL * natToFloat c.l2
  let invPanels := 1.0 / natToFloat panels
  let cornerA := C64.one
  let cornerB := C64.cis (theta1 + theta2)
  let cornerD := C64.cis (pi + theta1)
  let mut accA := C64.zero
  let mut accB := C64.zero
  let mut accD := C64.zero
  for i in [0:panels] do
    let u := (natToFloat i + 0.5) * invPanels
    let oneMinus := 1.0 - u
    let t := 1.0 - oneMinus * oneMinus * oneMinus
    let jac := 3.0 * oneMinus * oneMinus
    let zA := C64.scale t cornerA
    let zB := C64.scale t cornerB
    let zD := C64.scale t cornerD
    let (polyA, polyB, polyD) := polyEvalThreeChoppedPacked p.packedCoeffs c.coeffCount zA zB zD 1e-12
    let sA := singularFactor zA p.w1 p.w2 p.useTwoFactor
    let sB := singularFactor zB p.w1 p.w2 p.useTwoFactor
    let sD := singularFactor zD p.w1 p.w2 p.useTwoFactor
    accA := accA + C64.scale jac (sA * polyA * cornerA)
    accB := accB + C64.scale jac (sB * polyB * cornerB)
    accD := accD + C64.scale jac (sD * polyD * cornerD)
  let a := C64.scale invPanels accA
  let b := C64.scale invPanels accB
  let d := C64.scale invPanels accD
  return { p1 := a - d, p2 := b - d, p3 := a - b }

def periodsImprovedWith (c : CylCase) (p : ImprovedPoly c) :
    Except NumericError PeriodTriple :=
  periodsImprovedWithPanelCount c p (improvedPanelCount p)

def periods (c : CylCase) : Except NumericError PeriodTriple := do
  let p ← makeCylEqn c
  return periodsWithPoly c p

def periodsImproved (c : CylCase) : Except NumericError PeriodTriple := do
  let p ← makeCylEqnImproved c
  periodsImprovedWith c p

def bdetLogCore (c : CylCase) (normalizationBase : Float) : Except NumericError Float := do
  let raw := complexLogAbsDetNative c.half.toUSize (buildBdetMatrix c)
  if raw.isNaN || !raw.isFinite then
    throw ⟨"Native bdet_log failed."⟩
  return 2.0 * raw - natToFloat c.L * Float.log normalizationBase

def bdetLog (c : CylCase) : Except NumericError Float :=
  bdetLogCore c 64.0

def primeDetLogCore (c : CylCase) (normalizationDivisor : Float) : Except NumericError Float := do
  if c.half == 1 then
    return 0.0 - Float.log normalizationDivisor
  let raw := realSpdLogDetNative (c.half - 1).toUSize (buildPrimeDetReducedMatrix c)
  if raw.isNaN || !raw.isFinite then
    throw ⟨"Native prime_det_log failed."⟩
  return raw - Float.log normalizationDivisor

def primeDetLog (c : CylCase) : Except NumericError Float :=
  primeDetLogCore c (natToFloat c.half)

@[noinline] def benchMakeCyl (c : CylCase) (token : Nat) : Except NumericError (Poly c.half) :=
  makeCylEqn c (if token == 0 then 5e-13 else 1e-12)

@[noinline] def benchMakeCylImproved (c : CylCase) (token : Nat) :
    Except NumericError (ImprovedPoly c) :=
  makeCylEqnImproved c (if token == 0 then 5e-13 else 1e-12)

@[noinline] def benchPeriods (c : CylCase) (token : Nat) : Except NumericError PeriodTriple := do
  let p ← benchMakeCyl c token
  return periodsWithPoly c p

@[noinline] def benchPeriodsImprovedTotal (c : CylCase) (token : Nat) :
    Except NumericError PeriodTriple := do
  let p ← benchMakeCylImproved c token
  let panels := if token == 0 then improvedPanelCount p + 1 else improvedPanelCount p
  periodsImprovedWithPanelCount c p panels

@[noinline] def benchPeriodsImprovedPrebuilt (c : CylCase) (p : ImprovedPoly c) (token : Nat) :
    Except NumericError PeriodTriple :=
  let panels := if token == 0 then improvedPanelCount p + 1 else improvedPanelCount p
  periodsImprovedWithPanelCount c p panels

private def improvedPrebuiltBenchAction (c : CylCase) (improved : Except NumericError (ImprovedPoly c)) :
    Nat → Except NumericError PeriodTriple :=
  fun token =>
    match improved with
    | .ok p => benchPeriodsImprovedPrebuilt c p token
    | .error e => .error e

@[noinline] def benchBdetLog (c : CylCase) (token : Nat) : Except NumericError Float :=
  bdetLogCore c (if token == 0 then 65.0 else 64.0)

@[noinline] def benchPrimeDetLog (c : CylCase) (token : Nat) : Except NumericError Float :=
  let divisor := if token == 0 then natToFloat c.half + 1.0 else natToFloat c.half
  primeDetLogCore c divisor

structure TimingSummary where
  meanMs : Float
  medianMs : Float
  minMs : Float
  maxMs : Float
  stdevMs : Float
deriving Repr

def TimingSummary.asJson (t : TimingSummary) : Json :=
  Json.mkObj
    [ ("mean_ms", Lean.toJson t.meanMs)
    , ("median_ms", Lean.toJson t.medianMs)
    , ("min_ms", Lean.toJson t.minMs)
    , ("max_ms", Lean.toJson t.maxMs)
    , ("stdev_ms", Lean.toJson t.stdevMs)
    ]

instance : ToJson TimingSummary where
  toJson := TimingSummary.asJson

def arrayMean (xs : Array Float) : Float :=
  xs.foldl (fun acc x => acc + x) 0.0 / natToFloat xs.size

def arrayMin (xs : Array Float) : Float :=
  xs.foldl (fun acc x => if x < acc then x else acc) (xs[0]!)

def arrayMax (xs : Array Float) : Float :=
  xs.foldl (fun acc x => if acc < x then x else acc) (xs[0]!)

def arrayMedian (xs : Array Float) : Float :=
  let ys := xs.qsort (fun a b => a < b)
  if ys.size % 2 == 1 then
    ys[ys.size / 2]!
  else
    let i := ys.size / 2
    (ys[i - 1]! + ys[i]!) / 2.0

def arrayStdDev (xs : Array Float) : Float :=
  let mean := arrayMean xs
  let var := xs.foldl (fun acc x =>
    let d := x - mean
    acc + d * d) 0.0 / natToFloat xs.size
  Float.sqrt var

def summarizeTimes (xs : Array Float) : TimingSummary :=
  { meanMs := arrayMean xs
  , medianMs := arrayMedian xs
  , minMs := arrayMin xs
  , maxMs := arrayMax xs
  , stdevMs := arrayStdDev xs
  }

structure TimedResult (α : Type) where
  name : String
  ok : Bool
  timing? : Option TimingSummary
  value? : Option α
  error? : Option String

def TimedResult.asJson [ToJson α] (r : TimedResult α) : Json :=
  Json.mkObj
    [ ("name", Lean.toJson r.name)
    , ("ok", Lean.toJson r.ok)
    , ("timing", match r.timing? with | some t => Lean.toJson t | none => Json.null)
    , ("value", match r.value? with | some v => Lean.toJson v | none => Json.null)
    , ("error", match r.error? with | some e => Lean.toJson e | none => Json.null)
    ]

instance [ToJson α] : ToJson (TimedResult α) where
  toJson := TimedResult.asJson

partial def calibratePureAction (action : Nat → Except NumericError α) (minCalls targetMs : Nat) :
    IO (Except String Nat) := do
  if minCalls == 0 then
    return .error "repeat and number must be positive."
  let guardMessage := "Unreachable benchmark token."
  let target := natToFloat targetMs
  let maxCalls := 1048576
  let runBatch : Nat → Nat → IO (Except String (Nat × Float)) := fun serial0 n => do
    let mut serial := serial0
    let start ← IO.monoNanosNow
    for _ in [0:n] do
      let token := serial
      serial := serial + 1
      match action token with
      | Except.ok _ => pure ()
      | Except.error e =>
          if e.message == guardMessage then
            return Except.error guardMessage
          else
            return Except.error e.message
    let stop ← IO.monoNanosNow
    return Except.ok (serial, natToFloat (stop - start) / 1000000.0)
  if targetMs == 0 then
    return .ok minCalls
  let rec loop (serial calls : Nat) : IO (Except String Nat) := do
    match ← runBatch serial calls with
    | Except.error err => return Except.error err
    | Except.ok (serial', elapsedMs) =>
        if elapsedMs >= target || calls >= maxCalls then
          return Except.ok calls
        loop serial' (calls * 2)
  loop 1 minCalls

def measurePureAction (action : Nat → Except NumericError α) (repeats number warmup targetMs : Nat) :
    IO (Except String (TimingSummary × α)) := do
  let guardMessage := "Unreachable benchmark token."
  let callsPerSample ← match ← calibratePureAction action number targetMs with
    | .ok n => pure n
    | .error err => return .error err
  let mut serial := 1
  for _ in [0:warmup] do
    for _ in [0:callsPerSample] do
      let token := serial
      serial := serial + 1
      match action token with
      | .ok _ => pure ()
      | .error e => return .error e.message
  let mut samples := #[]
  let mut last? : Option α := none
  for _ in [0:repeats] do
    let start ← IO.monoNanosNow
    for _ in [0:callsPerSample] do
      let token := serial
      serial := serial + 1
      match action token with
      | .ok v => last? := some v
      | .error e =>
          if e.message == guardMessage then
            return .error guardMessage
          else
            return .error e.message
    let stop ← IO.monoNanosNow
    let elapsed := natToFloat (stop - start) / 1000000.0 / natToFloat callsPerSample
    samples := samples.push elapsed
  match last? with
  | some value => return .ok (summarizeTimes samples, value)
  | none => return .error "No benchmark samples were collected."

def timedPureResult (name : String) (action : Nat → Except NumericError α)
    (repeats number warmup targetMs : Nat) : IO (TimedResult α) := do
  match ← measurePureAction action repeats number warmup targetMs with
  | .ok (timing, value) =>
      return { name := name, ok := true, timing? := some timing, value? := some value, error? := none }
  | .error err =>
      return { name := name, ok := false, timing? := none, value? := none, error? := some err }

instance : ToJson (Poly n) where
  toJson p := Json.mkObj [("coeffs", Lean.toJson p.coeffs)]

instance : ToJson (ImprovedPoly c) where
  toJson p := Json.mkObj [("coeffs", Lean.toJson p.coeffs)]

structure CaseBenchmark where
  caseInfo : CylCase
  makeCyl : TimedResult (Poly caseInfo.half)
  makeCylImproved : TimedResult (ImprovedPoly caseInfo)
  periodsBase : TimedResult PeriodTriple
  periodsImprovedTotal : TimedResult PeriodTriple
  periodsImprovedPrebuilt : TimedResult PeriodTriple
  bdet : TimedResult Float
  primeDet : TimedResult Float

instance : ToJson CaseBenchmark where
  toJson r := Json.mkObj
    [ ("case", Lean.toJson r.caseInfo)
    , ("make_cyl_eqn", Lean.toJson r.makeCyl)
    , ("make_cyl_eqn_improved", Lean.toJson r.makeCylImproved)
    , ("periods", Lean.toJson r.periodsBase)
    , ("periods_improved(total)", Lean.toJson r.periodsImprovedTotal)
    , ("periods_improved(f prebuilt)", Lean.toJson r.periodsImprovedPrebuilt)
    , ("bdet_log", Lean.toJson r.bdet)
    , ("prime_det_log", Lean.toJson r.primeDet)
    ]

def benchmarkCase (c : CylCase) (repeats number warmup targetMs : Nat) : IO CaseBenchmark := do
  let improved := makeCylEqnImproved c
  let makeCyl ← timedPureResult "make_cyl_eqn" (benchMakeCyl c) repeats number warmup targetMs
  let makeCylImproved ←
    timedPureResult "make_cyl_eqn_improved" (benchMakeCylImproved c)
      repeats number warmup targetMs
  let periodsBase ← timedPureResult "periods" (benchPeriods c) repeats number warmup targetMs
  let periodsImprovedTotal ←
    timedPureResult "periods_improved(total)" (benchPeriodsImprovedTotal c)
      repeats number warmup targetMs
  let periodsImprovedPrebuilt ←
    timedPureResult "periods_improved(f prebuilt)"
      (improvedPrebuiltBenchAction c improved)
      repeats number warmup targetMs
  let bdet ← timedPureResult "bdet_log" (benchBdetLog c) repeats number warmup targetMs
  let primeDet ← timedPureResult "prime_det_log" (benchPrimeDetLog c) repeats number warmup targetMs
  return {
    caseInfo := c
    , makeCyl := makeCyl
    , makeCylImproved := makeCylImproved
    , periodsBase := periodsBase
    , periodsImprovedTotal := periodsImprovedTotal
    , periodsImprovedPrebuilt := periodsImprovedPrebuilt
    , bdet := bdet
    , primeDet := primeDet
    }

structure ImprovedCaseBenchmark where
  caseInfo : CylCase
  makeCylImproved : TimedResult (ImprovedPoly caseInfo)
  periodsImprovedTotal : TimedResult PeriodTriple
  periodsImprovedPrebuilt : TimedResult PeriodTriple

instance : ToJson ImprovedCaseBenchmark where
  toJson r := Json.mkObj
    [ ("case", Lean.toJson r.caseInfo)
    , ("make_cyl_eqn_improved", Lean.toJson r.makeCylImproved)
    , ("periods_improved(total)", Lean.toJson r.periodsImprovedTotal)
    , ("periods_improved(f prebuilt)", Lean.toJson r.periodsImprovedPrebuilt)
    ]

def benchmarkImprovedCase (c : CylCase) (repeats number warmup targetMs : Nat) :
    IO ImprovedCaseBenchmark := do
  let improved := makeCylEqnImproved c
  let makeCylImproved ←
    timedPureResult "make_cyl_eqn_improved" (benchMakeCylImproved c)
      repeats number warmup targetMs
  let periodsImprovedTotal ←
    timedPureResult "periods_improved(total)" (benchPeriodsImprovedTotal c)
      repeats number warmup targetMs
  let periodsImprovedPrebuilt ←
    timedPureResult "periods_improved(f prebuilt)"
      (improvedPrebuiltBenchAction c improved)
      repeats number warmup targetMs
  return {
    caseInfo := c
    , makeCylImproved := makeCylImproved
    , periodsImprovedTotal := periodsImprovedTotal
    , periodsImprovedPrebuilt := periodsImprovedPrebuilt
    }

structure DetCaseBenchmark where
  caseInfo : CylCase
  bdet : TimedResult Float
  primeDet : TimedResult Float

instance : ToJson DetCaseBenchmark where
  toJson r := Json.mkObj
    [ ("case", Lean.toJson r.caseInfo)
    , ("bdet_log", Lean.toJson r.bdet)
    , ("prime_det_log", Lean.toJson r.primeDet)
    ]

def benchmarkDetCase (c : CylCase) (repeats number warmup targetMs : Nat) :
    IO DetCaseBenchmark := do
  let bdet ← timedPureResult "bdet_log" (benchBdetLog c) repeats number warmup targetMs
  let primeDet ←
    timedPureResult "prime_det_log" (benchPrimeDetLog c) repeats number warmup targetMs
  return {
    caseInfo := c
    , bdet := bdet
    , primeDet := primeDet
    }

structure PeriodDetCaseBenchmark where
  caseInfo : CylCase
  periodsBase : TimedResult PeriodTriple
  periodsImprovedTotal : TimedResult PeriodTriple
  periodsImprovedPrebuilt : TimedResult PeriodTriple
  bdet : TimedResult Float
  primeDet : TimedResult Float

instance : ToJson PeriodDetCaseBenchmark where
  toJson r := Json.mkObj
    [ ("case", Lean.toJson r.caseInfo)
    , ("periods", Lean.toJson r.periodsBase)
    , ("periods_improved(total)", Lean.toJson r.periodsImprovedTotal)
    , ("periods_improved(f prebuilt)", Lean.toJson r.periodsImprovedPrebuilt)
    , ("bdet_log", Lean.toJson r.bdet)
    , ("prime_det_log", Lean.toJson r.primeDet)
    ]

def benchmarkPeriodDetCase (c : CylCase) (repeats number warmup targetMs : Nat) :
    IO PeriodDetCaseBenchmark := do
  let improved := makeCylEqnImproved c
  let periodsBase ← timedPureResult "periods" (benchPeriods c) repeats number warmup targetMs
  let periodsImprovedTotal ←
    timedPureResult "periods_improved(total)" (benchPeriodsImprovedTotal c)
      repeats number warmup targetMs
  let periodsImprovedPrebuilt ←
    timedPureResult "periods_improved(f prebuilt)"
      (improvedPrebuiltBenchAction c improved)
      repeats number warmup targetMs
  let bdet ← timedPureResult "bdet_log" (benchBdetLog c) repeats number warmup targetMs
  let primeDet ←
    timedPureResult "prime_det_log" (benchPrimeDetLog c) repeats number warmup targetMs
  return {
    caseInfo := c
    , periodsBase := periodsBase
    , periodsImprovedTotal := periodsImprovedTotal
    , periodsImprovedPrebuilt := periodsImprovedPrebuilt
    , bdet := bdet
    , primeDet := primeDet
    }

def padRightStr (s : String) (n : Nat) : String :=
  if s.length < n then
    s ++ String.ofList (List.replicate (n - s.length) ' ')
  else
    s

def printTimedRow {α} (name : String) (row : TimedResult α) : IO Unit := do
  match row.timing? with
  | some t =>
      IO.println s!"{padRightStr name 32}  yes  {t.medianMs}  {t.meanMs}  {t.minMs}  {t.maxMs}  {t.stdevMs}"
  | none =>
      IO.println s!"{padRightStr name 32}  no   -"
      match row.error? with
      | some err => IO.println s!"  error: {err}"
      | none => pure ()

def printCaseBenchmark (r : CaseBenchmark) : IO Unit := do
  IO.println ""
  IO.println s!"Case (L={r.caseInfo.L}, l1={r.caseInfo.l1}, l2={r.caseInfo.l2})"
  IO.println "function                         ok   median_ms   mean_ms   min_ms   max_ms   stdev_ms"
  IO.println "-------------------------------------------------------------------------------------------"
  printTimedRow "make_cyl_eqn" r.makeCyl
  printTimedRow "make_cyl_eqn_improved" r.makeCylImproved
  printTimedRow "periods" r.periodsBase
  printTimedRow "periods_improved(total)" r.periodsImprovedTotal
  printTimedRow "periods_improved(f prebuilt)" r.periodsImprovedPrebuilt
  printTimedRow "bdet_log" r.bdet
  printTimedRow "prime_det_log" r.primeDet

def printImprovedCaseBenchmark (r : ImprovedCaseBenchmark) : IO Unit := do
  IO.println ""
  IO.println s!"Case (L={r.caseInfo.L}, l1={r.caseInfo.l1}, l2={r.caseInfo.l2})"
  IO.println "function                         ok   median_ms   mean_ms   min_ms   max_ms   stdev_ms"
  IO.println "-------------------------------------------------------------------------------------------"
  printTimedRow "make_cyl_eqn_improved" r.makeCylImproved
  printTimedRow "periods_improved(total)" r.periodsImprovedTotal
  printTimedRow "periods_improved(f prebuilt)" r.periodsImprovedPrebuilt

def printDetCaseBenchmark (r : DetCaseBenchmark) : IO Unit := do
  IO.println ""
  IO.println s!"Case (L={r.caseInfo.L}, l1={r.caseInfo.l1}, l2={r.caseInfo.l2})"
  IO.println "function                         ok   median_ms   mean_ms   min_ms   max_ms   stdev_ms"
  IO.println "-------------------------------------------------------------------------------------------"
  printTimedRow "bdet_log" r.bdet
  printTimedRow "prime_det_log" r.primeDet

def printPeriodDetCaseBenchmark (r : PeriodDetCaseBenchmark) : IO Unit := do
  IO.println ""
  IO.println s!"Case (L={r.caseInfo.L}, l1={r.caseInfo.l1}, l2={r.caseInfo.l2})"
  IO.println "function                         ok   median_ms   mean_ms   min_ms   max_ms   stdev_ms"
  IO.println "-------------------------------------------------------------------------------------------"
  printTimedRow "periods" r.periodsBase
  printTimedRow "periods_improved(total)" r.periodsImprovedTotal
  printTimedRow "periods_improved(f prebuilt)" r.periodsImprovedPrebuilt
  printTimedRow "bdet_log" r.bdet
  printTimedRow "prime_det_log" r.primeDet

end Covariant
end LeanRace
