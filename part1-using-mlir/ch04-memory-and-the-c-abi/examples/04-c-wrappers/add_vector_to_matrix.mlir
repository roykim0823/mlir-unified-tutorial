// C-compatible wrappers, on a kernel whose ABI is worth looking at: adding a
// vector to every row of a matrix (NumPy's `M + v` broadcast),
//
//     out[i, j] = M[i, j] + v[j]        M: m x n,  v: n,  out: m x n
//
// Two calling conventions for the same math:
//
//   @add_vector_to_matrix       returns a NEW matrix (callee allocates).
//                               Written on tensors; bufferization turns the
//                               result into a malloc'd memref. Carries
//                               `llvm.emit_c_interface`, so lowering also
//                               emits `_mlir_ciface_add_vector_to_matrix`.
//
//   @add_vector_to_matrix_into  destination-passing (caller allocates %out).
//                               Written on memrefs. NO attribute: by default
//                               it is only callable with the memref
//                               descriptors expanded into scalars -- unless
//                               the pipeline runs --llvm-request-c-wrappers,
//                               which adds the attribute to every function.

#row = affine_map<(i, j) -> (i, j)>   // M and out: element (i, j)
#bcast = affine_map<(i, j) -> (j)>    // v: the same v[j] for every row i

func.func @add_vector_to_matrix(%M: tensor<?x?xf32>, %v: tensor<?xf32>)
    -> tensor<?x?xf32> attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %M, %c0 : tensor<?x?xf32>
  %n = tensor.dim %M, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%m, %n) : tensor<?x?xf32>
  %r = linalg.generic {indexing_maps = [#row, #bcast, #row],
                       iterator_types = ["parallel", "parallel"]}
      ins(%M, %v : tensor<?x?xf32>, tensor<?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %unused: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

func.func @add_vector_to_matrix_into(%M: memref<?x?xf32>, %v: memref<?xf32>,
                                     %out: memref<?x?xf32>) {
  linalg.generic {indexing_maps = [#row, #bcast, #row],
                  iterator_types = ["parallel", "parallel"]}
      ins(%M, %v : memref<?x?xf32>, memref<?xf32>)
      outs(%out : memref<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %unused: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  }
  return
}
