#!/bin/bash

# Create a minimal bare-bones text-only (no binaries) source
# distribution that can be compiled without Internet access.

TARGET_DIR="$1"

if [ -z "$TARGET_DIR" ]; then
    echo "Usage: ./minisilk.sh TARGET_DIR"
    exit 1
fi

TARGET_DIR="$( readlink -f "$TARGET_DIR" )"

if [ -f "$TARGET_DIR/recompile.sh" ]; then
    rm -r "$TARGET_DIR"
fi

mkdir -p "$TARGET_DIR"

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

git submodule update --init --recursive

cp -r "include" "$TARGET_DIR"
cp -r "src" "$TARGET_DIR"
cp -r "examples" "$TARGET_DIR"
cat "recompile.sh" | grep -vE '^(git|cp)' > "$TARGET_DIR/recompile.sh"
cat "CMakeLists.txt" | grep -v "test" > "$TARGET_DIR/CMakeLists.txt"
cat "run_unit_tests.sh" | grep -v "gpu_unit_test" > "$TARGET_DIR/run_unit_tests.sh"

mkdir "$TARGET_DIR/cedilla"
cp -r "cedilla/f2reduce" "$TARGET_DIR/cedilla"
cp -r "cedilla/cadical" "$TARGET_DIR/cedilla"
cp -r "cedilla/include" "$TARGET_DIR/cedilla"
cat "cedilla/CMakeLists.txt" | grep -vE "test|programs" > "$TARGET_DIR/cedilla/CMakeLists.txt"
rm -r "$TARGET_DIR/cedilla/cadical/scripts"
rm -r "$TARGET_DIR/cedilla/cadical/test"

mkdir "$TARGET_DIR/concurrentqueue"

cp concurrentqueue/*.h "$TARGET_DIR/concurrentqueue"
cp concurrentqueue/*.md "$TARGET_DIR/concurrentqueue"

mkdir -p "$TARGET_DIR/cpads/include"
cp -r "cpads/include/cpads" "$TARGET_DIR/cpads/include"

mkdir -p "$TARGET_DIR/cxxopts/include"
cp cxxopts/LICEN* "$TARGET_DIR/cxxopts"
cp cxxopts/include/cxxopts.hpp "$TARGET_DIR/cxxopts/include"

grep -A 100 -B 1 "c[o]pied into" minisilk.sh > "$TARGET_DIR/imsilk.sh"

exit 0

#!/bin/bash
# this gets copied into imsilk.sh

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

mkdir -p ~/.msilk
rm -r ~/.msilk/minisilk
rm ~/.msilk/minisilk.tar.gz

tar -czvf ~/.msilk/minisilk.tar.gz .

set -e

cp -r . ~/.msilk/minisilk
cd ~/.msilk/minisilk

bash recompile.sh
bash run_unit_tests.sh
