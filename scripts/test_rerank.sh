#!/bin/bash
set -euo pipefail

export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}

BASE_PORT=${BASE_PORT:-9805}
RANDOM_PORT=$((RANDOM % 1000))
MASTER_PORT=${MASTER_PORT:-$((BASE_PORT + RANDOM_PORT))}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

DATASETS=${DATASETS:-audiocaps,clotho}
TOPK=${TOPK:-50}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}

# Retrieval model output root (must already contain each dataset's rerank_metadata.json and similarity_matrix.pth)
RETRIEVAL_ROOT=${RETRIEVAL_ROOT:-/home/jilan_xu/AuroLA/checkpoints/qwen2_5-omni-7b_AuroLA/downstream}

# Rerank model checkpoint
RERANK_MODEL_NAME_OR_PATH=${RERANK_MODEL_NAME_OR_PATH:-/home/jilan_xu/AuroLA/checkpoints/qwen2_5-omni-7b_AuroLA_Rerank}

# Base model path for processor
ORIGINAL_MODEL_NAME_OR_PATH=${ORIGINAL_MODEL_NAME_OR_PATH:-/home/jilan_xu/checkpoints/Qwen2.5-Omni-7B}

# Optional, defaults to RETRIEVAL_ROOT when empty
OUTPUT_ROOT=${OUTPUT_ROOT:-${RETRIEVAL_ROOT}}

# Dataset paths are passed via env vars (same style as test_retrieval.sh)
export AUDIOCAPS_METADATA=${AUDIOCAPS_METADATA:-datasets/annotations/audiocaps/test.json}
export AUDIOCAPS_AUDIO_DIR=${AUDIOCAPS_AUDIO_DIR:-/home/jilan_xu/data/audiocaps/}

export CLOTHO_METADATA=${CLOTHO_METADATA:-datasets/annotations/clothov2/descs_cap_test.json}
export CLOTHO_AUDIO_DIR=${CLOTHO_AUDIO_DIR:-/home/jilan_xu/data/clothov2}

export AUTO_ACD_METADATA=${AUTO_ACD_METADATA:-datasets/annotations/autoacd/test.json}
export AUTO_ACD_AUDIO_DIR=${AUTO_ACD_AUDIO_DIR:-/mnt/data/}

torchrun \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  ./test_rerank.py \
  --rerank_model_name_or_path "${RERANK_MODEL_NAME_OR_PATH}" \
  --original_model_name_or_path "${ORIGINAL_MODEL_NAME_OR_PATH}" \
  --retrieval_root "${RETRIEVAL_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --datasets "${DATASETS}" \
  --topk "${TOPK}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

echo "Completed rerank testing"
