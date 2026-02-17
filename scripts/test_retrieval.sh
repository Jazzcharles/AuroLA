#!/bin/bash
set -euo pipefail

export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}

BASE_PORT=${BASE_PORT:-9801}
RANDOM_PORT=$((RANDOM % 1000))
MASTER_PORT=${MASTER_PORT:-$((BASE_PORT + RANDOM_PORT))}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

DATASETS=${DATASETS:-audiocaps,clotho,auto-acd}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}

# Model paths and output paths
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/home/jilan_xu/AuroLA/checkpoints/qwen2_5-omni-3b_AuroLA_3b}
ORIGINAL_MODEL_NAME_OR_PATH=${ORIGINAL_MODEL_NAME_OR_PATH:-/home/jilan_xu/checkpoints/Qwen2.5-Omni-3B}
OUTPUT_DIR=${OUTPUT_DIR:-${MODEL_NAME_OR_PATH}/downstream}

# Dataset paths are passed directly to test_retrieval.py via env vars.
export AUDIOCAPS_METADATA=${AUDIOCAPS_METADATA:-datasets/annotations/audiocaps/test.json}
export AUDIOCAPS_AUDIO_DIR=${AUDIOCAPS_AUDIO_DIR:-/home/jilan_xu/data/audiocaps/}

export CLOTHO_METADATA=${CLOTHO_METADATA:-datasets/annotations/clothov2/descs_cap_test.json}
export CLOTHO_AUDIO_DIR=${CLOTHO_AUDIO_DIR:-/home/jilan_xu/data/clothov2}

export AUTO_ACD_METADATA=${AUTO_ACD_METADATA:-datasets/annotations/autoacd/test.json}
export AUTO_ACD_AUDIO_DIR=${AUTO_ACD_AUDIO_DIR:-/mnt/data/}

# export AUDIOCAPS_METADATA=${AUDIOCAPS_METADATA:-/mnt/vision_user/xjl/metadata/Audio-Text-Data/downstream/AudioCaps/train.json}
# export AUDIOCAPS_AUDIO_DIR=${AUDIOCAPS_AUDIO_DIR:-/mnt/vision_user/xjl/data/AudioCaps/audio}

# export CLOTHO_METADATA=${CLOTHO_METADATA:-/mnt/vision_user/xjl/metadata/Audio-Text-Data/downstream/ClothoV2/descs_cap_test.json}
# export CLOTHO_AUDIO_DIR=${CLOTHO_AUDIO_DIR:-/mnt/vision_user/xjl/data/clotho/audio}

# export AUTO_ACD_METADATA=${AUTO_ACD_METADATA:-/mnt/vision_user/xjl/metadata/Audio-Text-Data/downstream/AutoACD/test.json}
# export AUTO_ACD_AUDIO_DIR=${AUTO_ACD_AUDIO_DIR:-/mnt/vision_user/xjl/data/Audio-Text-Data}

torchrun \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  ./test_retrieval.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --original_model_name_or_path "${ORIGINAL_MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --datasets "${DATASETS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

echo "Completed retrieval testing"
