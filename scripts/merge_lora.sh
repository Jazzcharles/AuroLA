# ORIGINAL_MODEL_ID=/home/jilan_xu/checkpoints/Qwen2.5-Omni-7B
# MODEL_ID=checkpoints/qwen2_5-omni-7b_AuroLA_TextPretrain
# SAVE_PATH=./checkpoints/AuroLA-TextPretrained-merged

ORIGINAL_MODEL_ID=/home/jilan_xu/checkpoints/Qwen2.5-Omni-3B
MODEL_ID=checkpoints/qwen2_5-omni-3b_AuroLA_TextPretrain_3b
SAVE_PATH=./checkpoints/AuroLA-TextPretrained-merged_3b


CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 utils/merge_lora.py \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --save_path $SAVE_PATH
