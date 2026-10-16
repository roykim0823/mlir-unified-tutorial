// Broadcasting as ONE linalg.generic — the broadcast lives in an indexing map.
//
// add_vec_to_mat.mlir does the job with two named ops (linalg.broadcast, then
// linalg.add). A broadcast is really just an indexing map that ignores a loop
// dimension: the vector is read at (d1) no matter which row d0 we are on.
//
//   matrix: (d0, d1) -> (d0, d1)   element [i, j]
//   vector: (d0, d1) -> (d1)       element [j], for every row i  <- broadcast
//   out:    (d0, d1) -> (d0, d1)   element [i, j]
//
// Both dimensions are "parallel": nothing is summed away (compare the
// reduction in 02-reduce, where it was the OUTPUT map that dropped d1).
// Exported as `_mlir_ciface_add_vec_to_mat_generic` (see run.py).
#mat = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d1)>

func.func @add_vec_to_mat_generic(%matrix: memref<3x4xf32>, %vector: memref<4xf32>,
                                  %out: memref<3x4xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.generic {
      indexing_maps  = [#mat, #row, #mat],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%matrix, %vector : memref<3x4xf32>, memref<4xf32>)
    outs(%out : memref<3x4xf32>) {
  ^bb0(%m: f32, %v: f32, %o: f32):
    %s = arith.addf %m, %v : f32
    linalg.yield %s : f32
  }
  return
}
