configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_SOURCE_DIR}/lit.cfg.py
)

set(UNIFIED_TEST_DEPENDS
  tutorial-opt
  FileCheck
)

add_lit_testsuite(check-all "Running all unified tutorial tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS -v
  DEPENDS ${UNIFIED_TEST_DEPENDS}
)

foreach(part part1 part2 part3 part4 part5 part6)
  add_custom_target(check-${part}
    COMMAND ${LLVM_EXTERNAL_LIT} ${CMAKE_CURRENT_BINARY_DIR} --filter=${part} -v
    DEPENDS ${UNIFIED_TEST_DEPENDS}
    COMMENT "Running ${part} tests"
  )
endforeach()
