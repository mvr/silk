cmake_minimum_required(VERSION 3.5)
project(silk)
set(CMAKE_CXX_STANDARD 17)

if(DEFINED NUM_PROCESSORS)
add_definitions(-DNUM_PROCESSORS=${NUM_PROCESSORS})
else()
message(WARNING "NUM_PROCESSORS not defined, using default value 8")
add_definitions(-DNUM_PROCESSORS=8)
endif()

add_compile_options( -O3 -g )
add_link_options( -lpthread )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-compare -pthread -std=c++17")

include(CheckLanguage)
check_language(CUDA)
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

if(CMAKE_VERSION VERSION_LESS "3.24.0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native")
else()
set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# hack for using Clang as a CUDA compiler:
if("${CMAKE_CUDA_COMPILER}" MATCHES "clang*")
set(_CMAKE_CUDA_FLAGS_BEFORE ${CMAKE_CUDA_FLAGS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++98")
enable_language(CUDA)
set(CMAKE_CUDA_FLAGS ${_CMAKE_CUDA_FLAGS_BEFORE})
unset(_CMAKE_CUDA_FLAGS_BEFORE)
else()
enable_language(CUDA)
endif()

# set actual CUDA flags:
if("${CMAKE_CUDA_COMPILER}" MATCHES "clang*")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__STRICT_ANSI__ -Xcuda-ptxas -v")
else()
# whereas NVIDIA does
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread --ptxas-options=-v")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -std=c++17")

add_subdirectory(cedilla)

include_directories("cpads/include")
include_directories("cedilla/include")
include_directories("include")

add_subdirectory(test)

add_subdirectory(src)
