
add_executable(testnet testnet.cu)

add_executable(silk main_kernel.cu heap_manager.cu entry_point.cu parse_options.cu)
target_link_libraries(silk cadical f2reduce)

# uncomment if you want to test register spillages on different architectures:
# set_property(TARGET silk PROPERTY CUDA_ARCHITECTURES 52 70 75 80 86 89)
