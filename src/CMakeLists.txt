
add_executable(testnet testnet.cu)

add_executable(silk main_kernel.cu heap_manager.cu entry_point.cu parse_options.cu)
target_link_libraries(silk cadical f2reduce)
