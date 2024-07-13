#!/bin/bash

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

echo "Determining processor count..."

if [ -z "$NPROCS" ]; then
    NPROCS="$( getconf _NPROCESSORS_ONLN )"
fi
if [ -z "$NPROCS" ]; then
    NPROCS="$( getconf NPROCESSORS_ONLN )"
fi
if [ -z "$NPROCS" ]; then
    NPROCS=8
    echo "Could not determine processor count; defaulting to $NPROCS"
else
    echo "Processor count: $NPROCS"
fi

echo "Checking dependencies..."

if [ -z "$CUDACXX" ]; then
    CUDACXX="nvcc"
fi

if "$CUDACXX" --version; then
    printf "\033[32;1m CUDA compiler detected \033[0m\n"
else
    printf "\033[31;1mError:\033[0m you need to install nvcc in order to compile Silk\n"
    exit 1
fi

if cmake --version; then
    printf "\033[32;1m cmake detected \033[0m\n"
else
    printf "\033[31;1mError:\033[0m you need to install cmake in order to compile Silk\n"
    exit 1
fi

if git --version; then
    printf "\033[32;1m git detected \033[0m\n"
else
    printf "\033[31;1mError:\033[0m you need to install git in order to compile Silk\n"
    exit 1
fi

mkdir build

set -ex

git submodule update --init --recursive

cd build

if "$CUDACXX" --version | grep spectral; then
    EXTRA_CMAKE_ARGS="-DUSE_SCALE=true -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON"
else
    EXTRA_CMAKE_ARGS=" "
fi

cmake $EXTRA_CMAKE_ARGS -DCMAKE_CUDA_COMPILER="$CUDACXX" -DNUM_PROCESSORS=$NPROCS -DCMAKE_BUILD_TYPE=Release ..

make -j $NPROCS

cd ..

cp "nnue/nnue_20928M.dat" "build/src/silk_nnue.dat"

# build/src/silk
# build/test/gpu/gpu_unit_test
