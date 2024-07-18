#!/bin/bash

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

# We run the unit tests followed by Silk itself, early-exiting
# if either program fails for some reason:
set -e
echo "Running unit tests..."
build/test/gpu/gpu_unit_test
echo "Running integration test..."
build/src/silk examples/2c3.rle 8 8 8 > build/test.out

# Now check that the program found the correct number of fizzles:
set +e
num_fizzles="$( grep 'Found fizzle' build/test.out | wc -l )"
expected_count=33
if [ "$num_fizzles" -ne "$expected_count" ]; then
    echo "Error: found $num_fizzles fizzles but expected $expected_count"
    exit 1
else
    echo "Success: found exactly $expected_count fizzles"
fi
