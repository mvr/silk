
file(GLOB_RECURSE gpu_unit_test_sources CONFIGURE_DEPENDS "unit/test_*.cu")
add_executable(gpu_unit_test ${gpu_unit_test_sources} )
target_link_libraries(gpu_unit_test gtest_main gtest cadical f2reduce)
set_property(TARGET gpu_unit_test PROPERTY CUDA_ARCHITECTURES native)

