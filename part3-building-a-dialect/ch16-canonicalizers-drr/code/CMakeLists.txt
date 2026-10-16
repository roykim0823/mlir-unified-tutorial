add_mlir_library(TutorialCh11DifferenceOfSquares
    PARTIAL_SOURCES_INTENDED
    DifferenceOfSquares.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/part3-building-a-dialect/ch16-canonicalizers-drr/code/

    DEPENDS
    MLIRPolyOpsIncGen

    LINK_LIBS PUBLIC
    MLIRPoly
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(tutorial-opt-ch16 tutorial-opt.cpp)

llvm_update_compile_flags(tutorial-opt-ch16)
target_link_libraries(tutorial-opt-ch16 PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    TutorialCh11DifferenceOfSquares
    MLIRPoly
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions
    MLIRPass
)

mlir_check_all_link_libraries(tutorial-opt-ch16)
