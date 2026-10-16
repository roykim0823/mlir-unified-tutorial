// Low-level LLVM dialect: structs and arrays.
//
// Demonstrates the LLVM dialect's C-like struct and fixed-size array
// types, plus the insertvalue / extractvalue ops used to build and
// read them.
//
// Already entirely in the llvm dialect, so it translates directly:
//   mlir-translate structs_arrays.mlir --mlir-to-llvmir
// build.sh compiles it and calls it from driver.c.
module {
  // !llvm.struct<(i32, f32)> mirrors the C struct:
  //     struct Pair { int a; float b; };
  llvm.func @make_pair() -> !llvm.struct<(i32, f32)> {
    // Zero-initialized struct, then fill each field.
    %z  = llvm.mlir.zero     : !llvm.struct<(i32, f32)>
    %a  = llvm.mlir.constant(42 : i32) : i32
    %b  = llvm.mlir.constant(3.140000e+00 : f32) : f32
    %s0 = llvm.insertvalue %a, %z [0] : !llvm.struct<(i32, f32)>
    %s1 = llvm.insertvalue %b, %s0[1] : !llvm.struct<(i32, f32)>
    llvm.return %s1 : !llvm.struct<(i32, f32)>
  }

  // Read field 0 (the i32) back out of a struct value.
  llvm.func @pair_first(%p: !llvm.struct<(i32, f32)>) -> i32 {
    %0 = llvm.extractvalue %p[0] : !llvm.struct<(i32, f32)>
    llvm.return %0 : i32
  }

  // Passing structs BY VALUE is not the C ABI. LLVM returns the aggregate
  // { i32, float } in two registers (w0 and s0 on ARM64), but a C function
  // returning `struct Pair` packs it into one 64-bit register (x0) -- clang
  // performs that "ABI lowering" when it compiles C; LLVM IR does not. So C
  // cannot safely call @make_pair / @pair_first directly. Going through
  // memory is unambiguous: the struct's in-memory LAYOUT does match C's.
  // This is exactly why MLIR's C interface passes memref descriptors by
  // pointer (see c_interface.mlir).
  llvm.func @make_pair_into(%out: !llvm.ptr) {
    %s = llvm.call @make_pair() : () -> !llvm.struct<(i32, f32)>
    llvm.store %s, %out : !llvm.struct<(i32, f32)>, !llvm.ptr
    llvm.return
  }

  llvm.func @pair_second(%p: !llvm.ptr) -> f32 {
    %s = llvm.load %p : !llvm.ptr -> !llvm.struct<(i32, f32)>
    %b = llvm.extractvalue %s[1] : !llvm.struct<(i32, f32)>
    llvm.return %b : f32
  }

  // !llvm.array<4 x i32> is a compile-time-fixed array, just like
  // `int x[4]` in C. No shape metadata, no strides.
  llvm.func @array_demo() -> i32 {
    // Build a constant array {1, 2, 3, 4}.
    %arr = llvm.mlir.constant(dense<[1, 2, 3, 4]> : tensor<4xi32>)
         : !llvm.array<4 x i32>

    // Read element index 2 (== 3).
    %v   = llvm.extractvalue %arr[2] : !llvm.array<4 x i32>

    // Write that value back into slot 0; produces a NEW array
    // value (LLVM dialect is SSA — arrays are value-typed here).
    %arr2 = llvm.insertvalue %v, %arr[0] : !llvm.array<4 x i32>
    %head = llvm.extractvalue %arr2[0] : !llvm.array<4 x i32>
    llvm.return %head : i32
  }

  // Array of structs: !llvm.array<2 x !llvm.struct<(i32, f32)>>.
  // Shows that struct types compose into arrays the same way C does.
  llvm.func @pair_array_first_a() -> i32 {
    %z  = llvm.mlir.zero
        : !llvm.array<2 x !llvm.struct<(i32, f32)>>
    %k  = llvm.mlir.constant(7 : i32) : i32
    // Nested index path [0, 0] = element 0, field 0.
    %a1 = llvm.insertvalue %k, %z[0, 0]
        : !llvm.array<2 x !llvm.struct<(i32, f32)>>
    %r  = llvm.extractvalue %a1[0, 0]
        : !llvm.array<2 x !llvm.struct<(i32, f32)>>
    llvm.return %r : i32
  }
}
