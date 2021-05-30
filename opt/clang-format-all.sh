#!/bin/bash

# exit immediatly on error ( no need to keep checking )
set -e

CLANG_FMT_SRCS=$(find ../src/ \( -name '*.c' -o -name '*.cc'-o -name '*.cpp' -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' \))
CLANG_FMT_TESTS=$(find ../tests/ \( -name '*.c' -o -name '*.cc'-o -name '*.cpp' -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' \))

for filename in $CLANG_FMT_SRCS; do
    clang-format --verbose -style=file -i $filename
done

for filename in $CLANG_FMT_TESTS; do
    clang-format --verbose -style=file -i $filename
done
