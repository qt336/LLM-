#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/public/code/qintian/AC_RoPE
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 scripts/train.py configs/c4/length-512/ce-eyepe/OLMo-180M-ce-eyepe1-local-noremap-punctgapgt4-40k.yaml
