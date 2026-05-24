#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ROPE_CKPT=/path/to/rope_checkpoint \
#   FOPE_YARN_CKPT=/path/to/fope_yarn_checkpoint \
#   bash commands/eval.sh
#
# Optional:
#   NPROC_PER_NODE=4 bash commands/eval.sh
swanlab login -k JiajEBS83WEnPgZvcX4Bv
NPROC_PER_NODE=8

ROPE_CKPT="/mnt/h_public/hlk/pxlin/CRoPE/Fourier-Position-Embedding-main/workspace/OLMo-60M-ce-512-rope-c4/step78019-unsharded"
FOPE_YARN_CKPT=/mnt/h_public/hlk/pxlin/CRoPE/Fourier-Position-Embedding-main/workspace/OLMo-60M-ce-512-fourier-c4/step78019-unsharded
ATTN_SSM_CKPT=workspace/OLMo-60M-ce-512-attn-ssm1/step78019-unsharded

if [[ -z "${ROPE_CKPT}" ]]; then
  echo "ROPE_CKPT is empty. Set it to your checkpoint path." >&2
  exit 1
fi

if [[ -z "${FOPE_YARN_CKPT}" ]]; then
  echo "FOPE_YARN_CKPT is empty. Set it to your checkpoint path." >&2
  exit 1
fi

if [[ -z "${ATTN_SSM_CKPT}" ]]; then
  echo "ATTN_SSM_CKPT is empty. Set it to your checkpoint path." >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# echo "==== Extrapolation Eval: mcsrope @ 1024 ===="
# torchrun --nproc_per_node="${NPROC_PER_NODE}" scripts/train.py \
#   configs/c4/length-1024/ce-extra/plain/OLMo-60M-mcsrope.yaml \
#   --load_path=/mnt/h_public/hlk/pxlin/CRoPE/Fourier-Position-Embedding-main/workspace/OLMo-60M-ce-512-mcsrope-c4/step78019-unsharded \
#   --run_name="eval-mcsrope-1024-${RUN_TAG}" \
#   --save_folder="workspace/eval-mcsrope-1024-${RUN_TAG}" \
#   --eval_on_load=true \
#   --max_duration=0 \
#   --save_interval_unsharded=null \
#   --save_num_unsharded_checkpoints_to_keep=0 \
#   --save_num_checkpoints_to_keep=0 \
#   --swanlab --swanlab.entity=MiracleLpX --swanlab.project=RoPE --swanlab.name=eval-mcsrope-1024-${RUN_TAG}


# echo "==== Extrapolation Eval: RoPE @ 1024 ===="
# torchrun --nproc_per_node="${NPROC_PER_NODE}" scripts/train.py \
#   configs/c4/length-1024/ce-extra/plain/OLMo-60M-ce-yarn.yaml \
#   --load_path="${ROPE_CKPT}" \
#   --run_name="eval-rope-1024-${RUN_TAG}" \
#   --save_folder="workspace/eval-rope-1024-${RUN_TAG}" \
#   --eval_on_load=true \
#   --max_duration=0 \
#   --save_interval_unsharded=null \
#   --save_num_unsharded_checkpoints_to_keep=0 \
#   --save_num_checkpoints_to_keep=0 \
#   --swanlab --swanlab.entity=MiracleLpX --swanlab.project=RoPE --swanlab.name=eval-rope-1024-${RUN_TAG}

# echo "==== Extrapolation Eval: FoPE + YARN @ 1024 ===="
# torchrun --nproc_per_node="${NPROC_PER_NODE}" scripts/train.py \
#   configs/c4/length-1024/ce-extra/plain/OLMo-60M-ce-fourier-eye_xavier_norm_0_3-sep_basis_head-ignore_clamp_zero-yarn_after.yaml \
#   --load_path="${FOPE_YARN_CKPT}" \
#   --run_name="eval-fope-yarn-1024-${RUN_TAG}" \
#   --save_folder="workspace/eval-fope-yarn-1024-${RUN_TAG}" \
#   --eval_on_load=true \
#   --max_duration=0 \
#   --save_interval_unsharded=null \
#   --save_num_unsharded_checkpoints_to_keep=0 \
#   --save_num_checkpoints_to_keep=0 \
#   --swanlab --swanlab.entity=MiracleLpX \
#   --swanlab.project=RoPE \
#   --swanlab.name=eval-fope-yarn-1024-${RUN_TAG}


echo "==== Extrapolation Eval: Attn-SSM @ 1024 ===="
torchrun --nproc_per_node="${NPROC_PER_NODE}" scripts/train.py \
  configs/c4/length-1024/ce-extra/plain/OLMo-60M-ce-attn-ssm-yarn.yaml \
  --load_path="${ATTN_SSM_CKPT}" \
  --run_name="eval-attn-ssm-yarn-1024-${RUN_TAG}" \
  --save_folder="workspace/eval-attn-ssm-yarn-1024-${RUN_TAG}" \
  --eval_on_load=true \
  --max_duration=0 \
  --save_interval_unsharded=null \
  --save_num_unsharded_checkpoints_to_keep=0 \
  --save_num_checkpoints_to_keep=0 \
  --swanlab --swanlab.entity=MiracleLpX \
  --swanlab.project=RoPE \
  --swanlab.name=eval-attn-ssm-1024-${RUN_TAG}
