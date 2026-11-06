#!/usr/bin/env bash
# Builds the project, generates a dataset if missing, runs the GPU filter on
# every image and stores a log under artifacts/ as proof of execution.
#
# Usage: ./run.sh [COUNT] [WIDTH] [HEIGHT] [FILTER]
set -euo pipefail

COUNT="${1:-200}"
WIDTH="${2:-512}"
HEIGHT="${3:-512}"
FILTER="${4:-edges}"

cd "$(dirname "$0")"

make
if [ ! -d data/input ] || [ -z "$(ls -A data/input 2>/dev/null)" ]; then
  python3 scripts/generate_dataset.py --count "$COUNT" --width "$WIDTH" \
      --height "$HEIGHT" --output_dir data/input
fi

mkdir -p artifacts data/output
./bin/batch_image_filter --input_dir data/input --output_dir data/output \
    --filter "$FILTER" | tee artifacts/execution_log.txt

# Convert a few before/after pairs to PNG for easy viewing.
python3 scripts/make_previews.py --input_dir data/input \
    --output_dir data/output --preview_dir artifacts/previews \
    --filter "$FILTER" --num 5 || echo "Preview step skipped."
