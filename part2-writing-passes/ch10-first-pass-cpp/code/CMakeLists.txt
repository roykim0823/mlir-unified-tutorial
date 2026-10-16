add_mlir_library(TutorialCh05Passes
    PARTIAL_SOURCES_INTENDED
    AffineFullUnroll.cpp
    AffineFullUnrollPatternRewrite.cpp
    MulToAdd.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/part2-writing-passes/ch10-first-pass-cpp/code/

    LINK_LIBS PUBLIC
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(tutorial-opt-ch10 tutorial-opt.cpp)

llvm_update_compile_flags(tutorial-opt-ch10)
target_link_libraries(tutorial-opt-ch10 PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    TutorialCh05Passes
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions
    MLIRPass
)

mlir_check_all_link_libraries(tutorial-opt-ch10)
