#!/usr/bin/env bash
set -euo pipefail

samples=(
    "samples/sample1.c"
    "samples/sample2.c"
    "samples/sample3.c"
    "samples/sample4.c"
    "samples/sample5.c"
)

mkdir -p build

for src in "${samples[@]}"; do
    base="$(basename "$src" .c)"
    echo "Translating $src"
    python translator.py "$src"
    mv assembly.out "build/${base}.s"
    gcc -m32 -c "build/${base}.s" -o "build/${base}.o"
    echo "Built build/${base}.o"
    echo "-------------------------"

done

echo "All sample programs translated and assembled successfully."
