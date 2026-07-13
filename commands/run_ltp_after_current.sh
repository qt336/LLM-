#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/public/code/qintian/AC_RoPE
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

BASE_RUN_DIR="workspace/OLMo-180M-ce-40-eyepe-olmo-c4-noremap-origc4-first40-allpunct-punctgap10to20-40k"
RUN_NAME="OLMo-180M-ce-40-eyepe-olmo-c4-noremap-origc4-first40-unfiltered-ltp-80k"
SAVE_FOLDER="workspace/${RUN_NAME}"
BASE_CONFIG="${PROJECT_ROOT}/${BASE_RUN_DIR}/latest-unsharded/config.yaml"
LOAD_PATH="${PROJECT_ROOT}/${BASE_RUN_DIR}/latest-unsharded"
TRAIN_DATA="${PROJECT_ROOT}/dataset/olmo_c4/first40_unfiltered/train_first40_unfiltered.npy"

python scripts/prepare_first40_unfiltered_olmo_c4.py \
  --source-path "${PROJECT_ROOT}/dataset/olmo_c4/part-000-00000.npy" \
  --output-path "${TRAIN_DATA}"

python scripts/watch_attention_offset_distributions.py \
  --run-dir "${SAVE_FOLDER}" \
  --poll-seconds 300 \
  --layer-idx 1 \
  --head-idx 1 \
  --target-token-id 15 \
  --max-offset 80 \
  --num-samples 1024 \
  --batch-size 4 \
  --device cuda &
WATCHER_PID=$!

cleanup() {
  kill "${WATCHER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

torchrun --nproc_per_node=8 \
  scripts/train.py "${BASE_CONFIG}" \
  --load_path="${LOAD_PATH}" \
  --run_name="${RUN_NAME}" \
  --save_folder="${SAVE_FOLDER}" \
  --max_duration=80000 \
  --data.paths="[${TRAIN_DATA}]" \
  --data.sample_range.start=0 \
  --data.sample_range.stop=1675111