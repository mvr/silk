cmake_minimum_required(VERSION 3.10)
project(silk)

include(${CMAKE_CURRENT_LIST_DIR}/cpads/include/cpads/CommonConfig.cmake)

if(DEFINED NUM_PROCESSORS)
add_definitions(-DNUM_PROCESSORS=${NUM_PROCESSORS})
else()
message(WARNING "NUM_PROCESSORS not defined, using default value 8")
add_definitions(-DNUM_PROCESSORS=8)
endif()

add_subdirectory(cedilla)

include_directories("cpads/include")
include_directories("cedilla/include")
include_directories("include")

add_subdirectory(test)

add_subdirectory(src)
