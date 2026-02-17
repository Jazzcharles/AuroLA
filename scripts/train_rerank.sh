NUM_GPUS=8
NPROC_PER_NODE=8
NNODES=1
NODE_RANK=0 # 1
MASTER_ADDR=127.0.0.1
MASTER_PORT=29507

DISTRIBUTED_ARGS="
    --nnodes=${NNODES} \
    --nproc_per_node ${NUM_GPUS} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

# arguments that are very likely to be changed
# according to your own case
MODEL_ID=qwen2_5-omni-7b                                  

# Dataset configuration
# For single dataset, use array with one element
# For multiple datasets, add more elements (must match length of METADATA_PATHS)
AUDIO_FOLDERS=(
    /mnt/data/
)
METADATA_PATHS=(
    /home/jilan_xu/metadata/wavcaps/train_subset_WavCaps_AudioSet_LAION-Audio_VGGSound_ACCL_EKSound_stage1_hardnegative_thresh0.4.json
)

TRAIN_VISION_ENCODER=False                              
USE_VISION_LORA=False                                  
TRAIN_VISION_PROJECTOR=False   
# TRAIN_AUDIO_ENCODER=False

USE_LORA=True                                           
Q_LORA=False                                           
LORA_R=128                                                
LORA_ALPHA=256                                           
RUN_ID=${MODEL_ID}_AuroLA_Rerank
# USE_AUDIO_LORA=False

DS_STAGE=zero2                                          
PER_DEVICE_BATCH_SIZE=4                               
GRAD_ACCUM=4                                            
NUM_EPOCHS=2                                         

LR=2e-5                                               
MODEL_MAX_LEN=1024

torchrun $DISTRIBUTED_ARGS train/train_rerank.py \
    --model_id $MODEL_ID \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to tensorboard \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 20 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    $(for i in "${!AUDIO_FOLDERS[@]}"; do echo "--audio_folder ${AUDIO_FOLDERS[$i]}"; echo "--metadata_path ${METADATA_PATHS[$i]}"; done)