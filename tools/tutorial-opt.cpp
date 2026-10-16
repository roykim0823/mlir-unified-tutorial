#include "lib/Conversion/MatmulToBlas/MatmulToBlas.h"
#include "lib/Conversion/PolyToStandard/PolyToStandard.h"
#include "lib/Dialect/Tiny/TinyDialect.h"
#include "lib/Dialect/Tiny/TinyPasses.h"
#include "lib/Dialect/TinyLoop/TinyLoopDialect.h"
#include "lib/Dialect/TinyLoop/TinyLoopPasses.h"
#include "lib/Dialect/TinyTile/TinyTileDialect.h"
#include "lib/Dialect/TinyTile/TinyTilePasses.h"
#include "lib/Transform/TinyTransform/TinyTransformOps.h"
#include "lib/Dialect/Noisy/NoisyDialect.h"
#ifdef ENABLE_SOLVER
#include "lib/Transform/Noisy/Passes.h"
#endif
#include "lib/Dialect/Poly/PolyDialect.h"
#include "lib/Transform/Affine/Passes.h"
#include "lib/Transform/Arith/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

void polyToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Poly
  manager.addPass(mlir::tutorial::poly::createPolyToStandard());
  manager.addPass(mlir::createCanonicalizerPass());

  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize, from
  // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);

  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // Needed to lower memref.subview
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());

  manager.addPass(mlir::createSCFToControlFlowPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

// Part 6, stage one: tensors to buffers. Kept separate from the LLVM
// descent so the bufferized module can be inspected (and so the BLAS pass,
// which needs memrefs, has a clean place to run after it).
void linalgToBufferizationPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize, from
  // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);
}

// Part 6, stage two: bufferized linalg down to the LLVM dialect.
void bufferizationToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Replace matmuls with BLAS calls AFTER bufferization (the pass needs
  // memrefs) but BEFORE the loop lowering (which would dissolve the matmul).
  manager.addPass(mlir::tutorial::createMatmulToBlas());

  // Convert remaining linalg ops to loops
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::affine::createLoopFusionPass());
  manager.addPass(mlir::createSCFToControlFlowPass());

  // Convert to LLVM - order matters here
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  // Math ops with no LLVM intrinsic (e.g. bert's math.erf) become libm calls
  manager.addPass(mlir::createConvertMathToLibmPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tutorial::poly::PolyDialect>();
  registry.insert<mlir::tiny::TinyDialect>();
  registry.insert<mlir::tiny_loop::TinyLoopDialect>();
  registry.insert<mlir::tiny_tile::TinyTileDialect>();
  registry.insert<mlir::tutorial::noisy::NoisyDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();

  mlir::tutorial::registerAffinePasses();
  mlir::tiny::registerPasses();
  mlir::tiny_loop::registerPasses();
  mlir::tiny_tile::registerPasses();
  mlir::tiny::registerTinyTransformExtension(registry);
#ifdef ENABLE_SOLVER
  mlir::tutorial::noisy::registerNoisyPasses();
#endif
  mlir::tutorial::registerArithPasses();

  // Dialect conversion passes
  mlir::tutorial::poly::registerPolyToStandardPasses();
  mlir::tutorial::registerMatmulToBlasPasses();

  mlir::PassPipelineRegistration<>(
      "poly-to-llvm", "Run passes to lower the poly dialect to LLVM",
      polyToLLVMPipelineBuilder);

  mlir::PassPipelineRegistration<>(
      "linalg-to-bufferization",
      "Run passes to lower the linalg dialect to bufferized memrefs",
      linalgToBufferizationPipelineBuilder);

  mlir::PassPipelineRegistration<>(
      "bufferization-to-llvm", "Run passes to lower bufferized code to LLVM",
      bufferizationToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR unified tutorial pass driver", registry));
}
