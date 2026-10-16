// Chapter 3's driver, pre-tablegen (the original hand-written form shown in
// chapter 3 section 3): explicit PassRegistration lines, no generated
// register*Passes() functions. See README.md in this directory.
#include "part2-writing-passes/ch10-first-pass-cpp/code/AffineFullUnroll.h"
#include "part2-writing-passes/ch10-first-pass-cpp/code/AffineFullUnrollPatternRewrite.h"
#include "part2-writing-passes/ch10-first-pass-cpp/code/MulToAdd.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();
  mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPassAsPatternRewrite>();
  mlir::PassRegistration<mlir::tutorial::MulToAddPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
