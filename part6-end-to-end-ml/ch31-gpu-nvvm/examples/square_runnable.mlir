// square.mlir plus a @main, so the result can actually be EXECUTED on a CUDA box
// with mlir-runner (see run_on_gpu.sh). Added in this port: the 8_gpu
// run_on_gpu.sh pointed mlir-runner at square.mlir directly, but @square takes
// a memref argument, so it cannot be an entry point - something has to
// allocate the array and print the result.
//
// gpu.host_register pins the host buffer and maps it into the GPU's address
// space (cuMemHostRegister via mgpuMemHostRegisterMemRef), so the kernel can
// read and write the very same memory @main filled. This is the pattern
// upstream's CUDA integration tests use. Expected output on a GPU:
//   [0,  1,  4,  9,  16,  25,  36,  49]

func.func @square(%a: memref<1024xf32>) {
  %c0    = arith.constant 0    : index
  %c1    = arith.constant 1    : index
  %c1024 = arith.constant 1024 : index
  scf.parallel (%i) = (%c0) to (%c1024) step (%c1) {
    %v = memref.load %a[%i] : memref<1024xf32>
    %t = arith.mulf %v, %v : f32
    memref.store %t, %a[%i] : memref<1024xf32>
    scf.reduce
  }
  return
}

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %c0    = arith.constant 0    : index
  %c1    = arith.constant 1    : index
  %c1024 = arith.constant 1024 : index
  %a = memref.alloc() : memref<1024xf32>
  scf.for %i = %c0 to %c1024 step %c1 {
    %ii = arith.index_cast %i : index to i32
    %f  = arith.sitofp %ii : i32 to f32
    memref.store %f, %a[%i] : memref<1024xf32>
  }
  %u = memref.cast %a : memref<1024xf32> to memref<*xf32>
  gpu.host_register %u : memref<*xf32>
  call @square(%a) : (memref<1024xf32>) -> ()
  %head = memref.subview %a[0] [8] [1] : memref<1024xf32> to memref<8xf32, strided<[1]>>
  %hu = memref.cast %head : memref<8xf32, strided<[1]>> to memref<*xf32>
  call @printMemrefF32(%hu) : (memref<*xf32>) -> ()
  memref.dealloc %a : memref<1024xf32>
  return
}
