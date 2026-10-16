// affine.apply in a real loop nest: tiled traversal of an array.
//
// The outer loop steps by the tile size (32); the inner loop walks the
// elements of one tile. affine.apply turns the (tile, within-tile) pair into a
// flat array index `%i + %j`. build.sh lowers this with -lower-affine so you
// can watch the affine ops expand into scf + arith.

// Static size: 256 is a multiple of 32, so every tile is full.
func.func @tiled_loop(%arg0: memref<256xf32>) {
  affine.for %i = 0 to 256 step 32 {       // outer: one iteration per tile
    affine.for %j = 0 to 32 {              // inner: elements within the tile
      %idx = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %j)
      %val = memref.load %arg0[%idx] : memref<256xf32>
      %dbl = arith.addf %val, %val : f32
      memref.store %dbl, %arg0[%idx] : memref<256xf32>
    }
  }
  return
}

// Dynamic size: %n is only known at run time and need not be a multiple of 32,
// so the last tile is partial. The inner loop's bounds are themselves affine
// maps of the outer index (a dim) and %n (a symbol), and the upper bound takes
// the `min` of two results: min(i + 32, n).
#tile_lb = affine_map<(d0) -> (d0)>
#tile_ub = affine_map<(d0)[s0] -> (d0 + 32, s0)>
func.func @tiled_loop_dynamic(%arg0: memref<?xf32>, %n: index) {
  affine.for %i = 0 to %n step 32 {
    affine.for %ii = #tile_lb(%i) to min #tile_ub(%i)[%n] {
      %val = affine.load %arg0[%ii] : memref<?xf32>
      %dbl = arith.addf %val, %val : f32
      affine.store %dbl, %arg0[%ii] : memref<?xf32>
    }
  }
  return
}
