// A tour of the "standard" low-level dialects: one small function per dialect
// (or dialect pair), all lowered together by build.sh and called from C by
// driver.c. Every function is plain scalar code -- no memory yet (Chapter 4).

// ---- arith: integer and float arithmetic, comparisons, conversions --------
// The op name carries the type family (addi vs addf) and, where it matters,
// the signedness (fptosi = float -> *signed* int). i32 itself has no sign.
func.func @arithmetic_example(%a: i32, %b: f32) -> i32 {
  %1 = arith.constant 42 : i32       // an integer constant
  %2 = arith.addi %a, %1 : i32       // integer addition
  %3 = arith.fptosi %b : f32 to i32  // float -> signed integer (truncates)
  %4 = arith.addi %2, %3 : i32
  return %4 : i32
}

// ---- math: functions beyond basic arithmetic --------------------------------
// math.sin / math.sqrt lower to LLVM intrinsics (llvm.intr.sin, ...), which the
// back end turns into an instruction (fsqrt) or a libm call (sinf).
func.func @math_example(%x: f32) -> f32 {
  %1 = math.sin %x : f32        // sin(x) * sqrt(x)
  %2 = math.sqrt %x : f32
  %3 = arith.mulf %1, %2 : f32
  return %3 : f32
}

// ---- index: platform-sized integer arithmetic ------------------------------
// Row-major flattening: offset = i * stride + j.
func.func @compute_offset(%i: index, %j: index, %stride: index) -> index {
  %1 = index.mul %i, %stride
  %2 = index.add %1, %j
  return %2 : index
}

// ---- cf: unstructured control flow -----------------------------------------
// Block arguments select between values, the way an LLVM phi node does.
func.func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
  cf.cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)
^bb1(%x: i32):
  return %x : i32
}

// cf.switch: a multi-way jump selected by an integer.
func.func @classify(%v: i32) -> i32 {
  cf.switch %v : i32, [
    default: ^other,
    0: ^zero,
    1: ^one
  ]
^zero:
  %c10 = arith.constant 10 : i32
  return %c10 : i32
^one:
  %c20 = arith.constant 20 : i32
  return %c20 : i32
^other:
  %cm1 = arith.constant -1 : i32
  return %cm1 : i32
}

// ---- scf: structured control flow ------------------------------------------
// scf.if with results: both branches yield a value; the if *is* an expression.
func.func @abs_diff(%a: i32, %b: i32) -> i32 {
  %lt = arith.cmpi slt, %a, %b : i32
  %r = scf.if %lt -> (i32) {
    %d = arith.subi %b, %a : i32
    scf.yield %d : i32
  } else {
    %d = arith.subi %a, %b : i32
    scf.yield %d : i32
  }
  return %r : i32
}

// scf.while: Collatz steps until n reaches 1. The "before" region computes the
// condition and forwards values; the "after" region is the loop body.
func.func @collatz_steps(%n0: i64) -> i64 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64
  %res:2 = scf.while (%n = %n0, %steps = %c0) : (i64, i64) -> (i64, i64) {
    %more = arith.cmpi ne, %n, %c1 : i64
    scf.condition(%more) %n, %steps : i64, i64
  } do {
  ^bb0(%n: i64, %steps: i64):
    %rem = arith.remui %n, %c2 : i64
    %even = arith.cmpi eq, %rem, %c0 : i64
    %next = scf.if %even -> (i64) {
      %h = arith.divui %n, %c2 : i64
      scf.yield %h : i64
    } else {
      %t = arith.muli %n, %c3 : i64
      %t1 = arith.addi %t, %c1 : i64
      scf.yield %t1 : i64
    }
    %s1 = arith.addi %steps, %c1 : i64
    scf.yield %next, %s1 : i64, i64
  }
  return %res#1 : i64
}
