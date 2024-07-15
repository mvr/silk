
file(GLOB_RECURSE gpu_unit_test_sources CONFIGURE_DEPENDS "unit/test_*.cu")
add_executable(gpu_unit_test ${gpu_unit_test_sources} )
target_link_libraries(gpu_unit_test gtest_main gtest cadical f2reduce)

add_executable(expose_clang_bug "test_clang_bug.cu")
target_link_libraries(expose_clang_bug gtest_main gtest)
