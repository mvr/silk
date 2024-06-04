#!/bin/bash

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

build/test/gpu/gpu_unit_test
