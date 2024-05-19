
add_executable(silk main_kernel.cu entry_point.cpp)
target_link_libraries(silk cadical f2reduce)
