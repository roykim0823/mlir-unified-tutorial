// 04-broadcast's two named ops, on tensors this time (INSPECT-ONLY).
//
// Run through the same generalize + fuse passes as separate_ops.mlir, the
// broadcast disappears into the add's indexing maps: the result is exactly the
// hand-written add_vec_to_mat_generic.mlir from 04-broadcast, with the
// (d0, d1) -> (d1) map on the vector operand. build.sh writes it to
// build/broadcast_add_fused.mlir.
func.func @add_vec_to_mat(%matrix: tensor<3x4xf32>, %vector: tensor<4xf32>) -> tensor<3x4xf32> {
  %e0 = tensor.empty() : tensor<3x4xf32>
  %bcast = linalg.broadcast ins(%vector : tensor<4xf32>)
                            outs(%e0 : tensor<3x4xf32>)
                            dimensions = [0]
  %e1 = tensor.empty() : tensor<3x4xf32>
  %sum = linalg.add ins(%matrix, %bcast : tensor<3x4xf32>, tensor<3x4xf32>)
                    outs(%e1 : tensor<3x4xf32>) -> tensor<3x4xf32>
  return %sum : tensor<3x4xf32>
}
