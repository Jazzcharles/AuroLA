#!/bin/bash
set -euo pipefail

### AuroLA-7B, AudioCaps, adjust alpha to get the best fusion score ###
RERANK_RESULT=${RERANK_RESULT:-/home/jilan_xu/AuroLA/hf_export/AuroLA-7B/downstream/audiocaps/AuroLA-rerank-7B_rerank_result_50.pth}
TOPK=${TOPK:-50}
ALPHA_A2T=${ALPHA_A2T:-2.0}
ALPHA_T2A=${ALPHA_T2A:-0.9}
python get_rerank_score.py \
  --rerank_result "${RERANK_RESULT}" \
  --topk "${TOPK}" \
  --alpha_a2t "${ALPHA_A2T}" \
  --alpha_t2a "${ALPHA_T2A}"

### AuroLA-7B, Clotho, adjust alpha to get the best fusion score ###
# RERANK_RESULT=${RERANK_RESULT:-/home/jilan_xu/AuroLA/hf_export/AuroLA-7B/downstream/clotho/AuroLA-rerank-7B_rerank_result_50.pth}
# TOPK=${TOPK:-50}
# ALPHA_A2T=${ALPHA_A2T:-2.0}
# ALPHA_T2A=${ALPHA_T2A:-0.1}
# python get_rerank_score.py \
#   --rerank_result "${RERANK_RESULT}" \
#   --topk "${TOPK}" \
#   --alpha_a2t "${ALPHA_A2T}" \
#   --alpha_t2a "${ALPHA_T2A}"


### AuroLA-3B, AudioCaps, adjust alpha to get the best fusion score ###
# RERANK_RESULT=${RERANK_RESULT:-/home/jilan_xu/AuroLA/hf_export/AuroLA-3B/downstream/audiocaps/AuroLA-rerank-3B_rerank_result_50.pth}
# TOPK=${TOPK:-50}
# ALPHA_A2T=${ALPHA_A2T:-0.8}
# ALPHA_T2A=${ALPHA_T2A:-0.8}
# python get_rerank_score.py \
#   --rerank_result "${RERANK_RESULT}" \
#   --topk "${TOPK}" \
#   --alpha_a2t "${ALPHA_A2T}" \
#   --alpha_t2a "${ALPHA_T2A}"

### AuroLA-3B, Clotho, adjust alpha to get the best fusion score ###
# RERANK_RESULT=${RERANK_RESULT:-/home/jilan_xu/AuroLA/hf_export/AuroLA-3B/downstream/clotho/AuroLA-rerank-3B_rerank_result_50.pth}
# TOPK=${TOPK:-50}
# ALPHA_A2T=${ALPHA_A2T:-3.0}
# ALPHA_T2A=${ALPHA_T2A:-0.4}
# python get_rerank_score.py \
#   --rerank_result "${RERANK_RESULT}" \
#   --topk "${TOPK}" \
#   --alpha_a2t "${ALPHA_A2T}" \
#   --alpha_t2a "${ALPHA_T2A}"

