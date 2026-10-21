#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p build
if [[ -n "${TIMES_FONT_DIR:-}" ]]; then
    # Pass an optional font-directory macro without rewriting the TeX source.
    font_dir="$(realpath -- "$TIMES_FONT_DIR")/"
    tex_input="\\def\\TimesFontPath{$font_dir}\\input{main.tex}"
else
    tex_input=main.tex
fi
for pass in 1 2 3; do
    xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
        -jobname=Md_Zunayedul_Islam_Model_Pruning -output-directory=build "$tex_input" \
        > "build/compile-pass-$pass.log" 2>&1
done
cp build/Md_Zunayedul_Islam_Model_Pruning.pdf Md_Zunayedul_Islam_Model_Pruning.pdf
printf 'Report built: %s/Md_Zunayedul_Islam_Model_Pruning.pdf\n' "$PWD"
