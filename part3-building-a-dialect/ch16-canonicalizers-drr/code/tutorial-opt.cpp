// Driver for Chapter 9's preserved C++ DifferenceOfSquares pattern
// (scaffolding around the frozen pattern). See README.md in this directory.
#include "lib/Dialect/Poly/PolyDialect.h"
#include "part3-building-a-dialect/ch16-canonicalizers-drr/code/DifferenceOfSquares.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tutorial::poly::PolyDialect>();
  mlir::registerAllDialects(registry);

  mlir::PassRegistration<mlir::tutorial::poly::DifferenceOfSquaresPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
