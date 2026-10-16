// Identity-matrix kernel, destination-passing style: the caller owns the
// output buffer %out and the kernel writes the matrix into it.
//
// The compute is pure tensor (`tensor.generate`), but the signature takes a
// memref<?x?xi32>. That memref is NOT incidental: a pure tensor has no memory,
// so to return data to a caller we need a real buffer.
// `bufferization.materialize_in_destination` says "the value %t must end up
// in %out".
//
// Catch: tensor.generate has no destination operand, so one-shot-bufferize
// allocates a fresh buffer for it and then memref.copy's that into %out. See
// identity_fill_inplace.mlir for the variant that writes into %out directly,
// and ../03-low-level-llvm/bufferization.mlir for the bufferization deep dive.
//
// Exported to C/Python as `_mlir_ciface_identity` (aot_main.py, run_fill).
module {
  func.func @identity(%out: memref<?x?xi32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %out, %c0 : memref<?x?xi32>
    %n = memref.dim %out, %c1 : memref<?x?xi32>

    %t = tensor.generate %m, %n {
      ^bb0(%i: index, %j: index):
        %ni = arith.index_cast %i : index to i32
        %nj = arith.index_cast %j : index to i32
        %eq = arith.cmpi eq, %ni, %nj : i32
        %v  = arith.extui %eq : i1 to i32
        tensor.yield %v : i32
    } : tensor<?x?xi32>

    bufferization.materialize_in_destination %t in writable %out
      : (tensor<?x?xi32>, memref<?x?xi32>) -> ()
    return
  }
}
