# ##### Books #####
# #### length-512 ####
# torchrun --nproc_per_node=4 scripts/train.py configs/books/length-512/ce-fourier/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero.yaml

# #### length-1024 ####
# torchrun --nproc_per_node=4 scripts/train.py configs/books/length-1024/ce-fourier/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero.yaml

# ##### C4 #####
# #### length-512 ####
# torchrun --nproc_per_node=8 scripts/train.py configs/c4/length-512/ce-fourier/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero.yaml

# #### length-1024 ####
# torchrun --nproc_per_node=4 scripts/train.py configs/c4/length-1024/ce-fourier/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero.yaml


#!/bin/bash
set -o pipefail
export PATH=/opt/conda/bin:/opt/conda/condabin:$PATH
export TMPDIR=/mnt/h_public/hlk/pxlin/hf_cache
ln -s /opt/maca/tools/cu-bridge/bin/cucc /opt/maca/tools/cu-bridge/bin/nvcc
cd /mnt/h_public/hlk/pxlin/CRoPE/Fourier-Position-Embedding-main
export HF_DATASETS_OFFLINE=1
export HF_HOME='/mnt/public/code/hlk/hf_cache'
export SSL_CERT_FILE='/mnt/public/code/hlk/open-r1-main/cacert.pem'
export PYTHONPATH=/mnt/h_public/hlk/pxlin/CRoPE/Fourier-Position-Embedding-main:$PYTHONPATH


pip install --upgrade transformers
pip install swanlab warmup_scheduler scikit-learn evaluate datasets matplotlib seaborn
pip install protobuf==3.20.*
pip install latex2sympy2_extended math_verify trl==0.16.1 antlr4-python3-runtime==4.13.2
pip install .

swanlab login -k JiajEBS83WEnPgZvcX4Bv

torchrun --nproc_per_node=8 scripts/train.py   configs/c4/length-512/ce-eyepe/OLMo-60M-ce-eyepe1.yaml   --run_name=OLMo-60M-ce-512-eyepe3  --save_folder=workspace/OLMo-60M-ce-512-eyepe3 --swanlab --swanlab.entity=MiracleLpX --swanlab.project=RoPE --swanlab.name=OLMo-60M-ce-512-eyepe3

torchrun --nproc_per_node=8 scripts/train.py   configs/c4/length-512/ce-eyepe/OLMo-60M-ce-eyepe1.yaml   --run_name=OLMo-60M-ce-512-eyepe-1  --save_folder=workspace/OLMo-60M-ce-512-eyepe-1--swanlab --swanlab.entity=MiracleLpX --swanlab.project=RoPE --swanlab.name=OLMo-60M-ce-512-eyepe-1


torchrun --nproc_per_node=8 scripts/train.py   configs/c4/length-512/ce-rope/OLMo-60M-ce.yaml   --run_name=OLMo-60M-ce-512-rope-c4   --save_folder=workspace/OLMo-60M-ce-512-rope-c4 --swanlab --swanlab.entity=MiracleLpX --swanlab.project=RoPE --swanlab.name=OLMo-60M-ce-512-rope-c4 

torchrun --nproc_per_node=8 scripts/train.py   configs/c4/length-512/ce-mcsrope/OLMo-60M-ce-mcsrope.yaml   --run_name=OLMo-60M-ce-512-mcsrope-c4   --save_folder=workspace/OLMo-60M-ce-512-mcsrope-c4 --swanlab --swanlab.entity=MiracleLpX --swanlab.project=RoPE --swanlab.name=OLMo-60M-ce-512-mcsrope-c4 

torchrun --nproc_per_node=8 scripts/train.py   configs/c4/length-512/ce-mixedpe/OLMo-60M-ce-mixedpe.yaml   --run_name=OLMo-60M-ce-512-mixedpe-c4   --save_folder=workspace/OLMo-60M-ce-512-mixedpe-c4 --swanlab --swanlab.entity=MiracleLpX --swanlab.project=RoPE --swanlab.name=OLMo-60M-ce-512-mixedpe-c4 

torchrun --nproc_per_node=8 scripts/train.py   configs/c4/length-512/ce-eyepe/OLMo-60M-ce-eyepe.yaml   --run_name=OLMo-60M-ce-512-eyepe-c4   --save_folder=workspace/OLMo-60M-ce-512-eyepe-c4 --swanlab --swanlab.entity=MiracleLpX --swanlab.project=RoPE --swanlab.name=OLMo-60M-ce-512-eyepe-c4 

torchrun --nproc_per_node=8 scripts/train.py configs/c4/length-512/ce-fourier/OLMo-180M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero.yaml   --run_name=OLMo-60M-ce-512-fourier-c4 --save_folder=workspace/OLMo-60M-ce-512-fourier-c4 --swanlab --swanlab.entity=MiracleLpX --swanlab.project=RoPE --swanlab.name=OLMo-60M-ce-512-fourier-c4 

torchrun --nproc_per_node=4 scripts/train.py   configs/c4/length-1024/ce-eyepe/OLMo-60M-ce-eyepe.yaml   --run_name=OLMo-60M-ce-1024-eyepe-c4   --save_folder=workspace/OLMo-60M-ce-1024-eyepe-c4 --swanlab --swanlab.entity=MiracleLpX --swanlab.project=RoPE --swanlab.name=OLMo-60M-ce-1024-eyepe-c4 
