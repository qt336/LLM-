# old='/home/linpengxiao/data/OLMo/preprocessed/c4/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix/gpt-neox-olmo-dolma-v1_5'
# new='/mnt/h_public/hlk/pxlin/CRoPE/Fourier-Position-Embedding-main/data'

# old='/home/linpengxiao/data/OLMo/eval-data/perplexity/v3_small_gptneox20b/c4_en/val'
# new='/mnt/h_public/hlk/pxlin/CRoPE/Fourier-Position-Embedding-main/data/valid'

# find /mnt/h_public/hlk/pxlin/CRoPE/Fourier-Position-Embedding-main/configs -type f \( -name "*.yaml" -o -name "*.yml" \) \
#   -exec sed -i "s|$old|$new|g" {} +

old='wandb'
new='swanlab'

find /mnt/h_public/hlk/pxlin/CRoPE/Fourier-Position-Embedding-main -type f \( -name "*.py" -o -name "*.yaml" \) \
  -exec sed -i "s|$old|$new|g" {} +

old='WANDB'
new='SWANLAB'

find /mnt/h_public/hlk/pxlin/CRoPE/Fourier-Position-Embedding-main -type f \( -name "*.py" -o -name "*.yaml" \) \
  -exec sed -i "s|$old|$new|g" {} +