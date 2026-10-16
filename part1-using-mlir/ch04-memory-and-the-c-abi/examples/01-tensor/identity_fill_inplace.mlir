// Identity matrix, destination-passing style, written so that bufferization
// can prove it may write STRAIGHT INTO the caller's buffer.
//
// identity_fill.mlir builds its result with tensor.generate, which has no
// destination operand: one-shot-bufferize must allocate a fresh buffer for it
// and then memref.copy that into %out. Here the computation is a linalg.map
// whose `outs` operand is the caller's buffer viewed as a tensor:
//
//   * bufferization.to_tensor ... restrict writable  -- "this tensor IS %out,
//     nothing else aliases it, and you may write into it";
//   * linalg.map outs(%dest) -- destination-passing: the op says where its
//     result should live;
//   * materialize_in_destination -- the result must end up in %out (it already
//     is, so the copy becomes a self-copy that canonicalize deletes).
//
// After one-shot-bufferize there is no memref.alloc at all.
func.func @identity(%out: memref<?x?xi32>) attributes {llvm.emit_c_interface} {
  %dest = bufferization.to_tensor %out restrict writable
      : memref<?x?xi32> to tensor<?x?xi32>
  %t = linalg.map outs(%dest : tensor<?x?xi32>) (%init: i32) {
    %i = linalg.index 0 : index                  // current row
    %j = linalg.index 1 : index                  // current column
    %eq = arith.cmpi eq, %i, %j : index          // i == j ?
    %v = arith.extui %eq : i1 to i32             // 1 on the diagonal, else 0
    linalg.yield %v : i32
  }
  bufferization.materialize_in_destination %t in restrict writable %out
      : (tensor<?x?xi32>, memref<?x?xi32>) -> ()
  return
}
