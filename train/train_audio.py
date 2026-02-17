
import os
import sys
import warnings
import logging
from dataclasses import asdict
from pathlib import Path
import yaml

current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.insert(0, module_path)

import librosa
from utils.audio_utils import safe_librosa_load
# Use a guarded audio loader globally to prevent oversized/invalid reads.
librosa.load = safe_librosa_load

from accelerate.utils import DistributedType, is_deepspeed_available
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers
from transformers import Trainer

from utils.arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from data.dataset_audioverse import AudioVerse
from torch.utils.data import ConcatDataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
from utils.qwen_utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
)


def train():
    # Reduce noisy warnings from tokenizer/processor internals.
    warnings.filterwarnings("ignore", message=".*System prompt modified.*")
    warnings.filterwarnings("ignore", message=".*audio output may not work.*")
    root_logger = logging.getLogger("root")
    root_logger.setLevel(logging.ERROR)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    
    # Parse repeated arguments explicitly for multi-dataset training.
    audio_folders_manual = []
    metadata_paths_manual = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--audio_folder' and i + 1 < len(sys.argv):
            audio_folders_manual.append(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--metadata_path' and i + 1 < len(sys.argv):
            metadata_paths_manual.append(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    # Override parser output when repeated CLI flags are provided.
    if len(audio_folders_manual) > 0:
        data_args.audio_folder = audio_folders_manual
    if len(metadata_paths_manual) > 0:
        data_args.metadata_path = metadata_paths_manual

    # Persist resolved arguments for reproducibility.
    output_dir = getattr(training_args, 'output_dir', None)
    assert output_dir is not None, "output_dir is required"
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))

    compute_dtype = (torch.float16 if training_args.fp16 else
                 (torch.bfloat16 if training_args.bf16 else torch.float32))

    # QLoRA is incompatible with DeepSpeed ZeRO/FSDP in this setup.
    if lora_args.q_lora:
        rank0_print(
            "QLoRA detected: forcing DistributedType.NO (disable DeepSpeed / ZeRO)."
        )
        training_args.distributed_state.distributed_type = DistributedType.NO
        training_args.deepspeed = None

    device_map = None
    if lora_args.q_lora:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            device_map = None
        else:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        is_zero3 = False
        if is_deepspeed_available():
            is_zero3 = training_args.deepspeed_plugin and training_args.deepspeed_plugin.zero_stage == 3

        if len(training_args.fsdp) > 0 or is_zero3:
            raise ValueError("FSDP or ZeRO3 are not compatible with QLoRA.")

    # Optional 4-bit quantization for QLoRA.
    bnb_config = None
    if lora_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig
        rank0_print("Quantization for LLM enabled...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4", 
        )
    
    # Build model/tokenizer/processor from registry.
    rank0_print("Loading model, tokenizer, processor...")
    loader = LOADERS[model_args.model_family_id](
        model_hf_path=model_args.model_hf_path,
        model_local_path=model_args.model_local_path,
        compute_dtype=compute_dtype,
        bnb_config=bnb_config,
        use_flash_attn=training_args.use_flash_attn,
        device_map=device_map,
    )
    model, tokenizer, processor = loader.load(pretrain=False)
    tokenizer.model_max_length = training_args.model_max_length
    
    # Sync contrastive hyper-parameters to model config.
    if hasattr(training_args, 'contrastive_alpha'):
        model.config.contrastive_alpha = training_args.contrastive_alpha
        rank0_print(f"Setting contrastive_alpha to {training_args.contrastive_alpha}")
    if hasattr(training_args, 'contrastive_lambda'):
        model.config.contrastive_lambda = training_args.contrastive_lambda
        rank0_print(f"Setting contrastive_lambda to {training_args.contrastive_lambda}")
    if hasattr(training_args, 'contrastive_beta'):
        model.config.contrastive_beta = training_args.contrastive_beta
        rank0_print(f"Setting contrastive_beta to {training_args.contrastive_beta}")

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Freeze selected modules according to training flags.
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
    if not training_args.train_vision_encoder:
        rank0_print(f"Vision encoder is freezed... including:")
        for module in vision_encoder_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    if not training_args.train_vision_projector:
        rank0_print(f"Vision projector is freezed... including:")
        for module in vision_projector_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)
    
    audio_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["audio_encoder"]
    if not training_args.train_audio_encoder:
        rank0_print(f"Audio encoder is freezed... including:")
        for module in audio_encoder_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    # Freeze additional multimodal components when present.
    if "others" in MODULE_KEYWORDS[model_args.model_family_id]:
        rank0_print(f"Other multimodal component is freezed... including:")
        for other_key in MODULE_KEYWORDS[model_args.model_family_id]["others"]:
            rank0_print(f"\t{other_key}")
            eval(f"model.{other_key}").requires_grad_(False)

    # Configure LoRA target modules.
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    if not (lora_args.use_lora or (training_args.train_vision_encoder and lora_args.use_vision_lora) or (training_args.train_audio_encoder and lora_args.use_audio_lora)):
        rank0_print("No LoRA enabled...")        
    else:
        named_modules = {n: m for n, m in model.named_modules()}
        lora_modules = []
        full_modules = []

        if training_args.train_vision_encoder and lora_args.use_vision_lora:
            rank0_print("LoRA for vision encoder enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, vision_encoder_keys))
        elif training_args.train_vision_encoder:
            rank0_print("Vision encoder will be fully trained...")
            full_modules.extend(vision_encoder_keys)

        # Configure audio encoder training mode.
        if training_args.train_audio_encoder and lora_args.use_audio_lora:
            rank0_print("LoRA for audio encoder enabled and will be trained...")
            lora_modules.extend(find_all_linear_names(named_modules, audio_encoder_keys))
        elif training_args.train_audio_encoder:
            rank0_print("Audio encoder will be fully trained...")
            full_modules.extend(audio_encoder_keys)
        
        if lora_args.use_lora and training_args.train_llm:
            rank0_print("LoRA for LLM enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, llm_keys))

        elif training_args.train_llm is False:
            rank0_print("LLM will be freezed...")
            pass
        else:
            rank0_print("LLM will be fully trained...")
            full_modules.extend(llm_keys)
        
        if training_args.train_vision_projector:
            rank0_print("Vision projector will be fully trained...")
            full_modules.extend(vision_projector_keys)
        
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_modules,
            modules_to_save=full_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            use_dora=lora_args.use_dora,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        # Initialize LoRA from an existing adapter when provided; otherwise create a new adapter.
        if lora_args.lora_weight_path and Path(lora_args.lora_weight_path).exists():
            rank0_print(f"Loading pretrained LoRA adapter from: {lora_args.lora_weight_path}")
            rank0_print("Using adapter config from checkpoint for initialization.")
            model = PeftModel.from_pretrained(
                model,
                lora_args.lora_weight_path,
                is_trainable=True,
            )
        else:
            if lora_args.lora_weight_path:
                rank0_print(
                    f"Warning: lora_weight_path not found: {lora_args.lora_weight_path}. "
                    "Falling back to creating a new LoRA adapter."
                )
            model = get_peft_model(model, lora_config)

    # Print trainable parameters for quick verification.
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")
    
    # Summarize model parameter statistics.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_M = trainable_params / 1e6
    total_params_M = total_params / 1e6
    rank0_print("=" * 80)
    rank0_print(f"Model Parameter Statistics:")
    rank0_print(f"  Total parameters: {total_params_M:.2f} M")
    rank0_print(f"  Trainable parameters: {trainable_params_M:.2f} M")
    rank0_print(f"  Non-trainable parameters: {total_params_M - trainable_params_M:.2f} M")
    rank0_print(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    rank0_print("=" * 80)
    

    # Build training dataset(s).
    rank0_print("Loading data...")
    
    audio_folders = data_args.audio_folder if data_args.audio_folder is not None else []
    metadata_paths = data_args.metadata_path if data_args.metadata_path is not None else []
    
    if isinstance(audio_folders, str):
        audio_folders = [audio_folders]
    if isinstance(metadata_paths, str):
        metadata_paths = [metadata_paths]
    
    rank0_print(f"Found {len(audio_folders)} dataset(s) to load")
    
    assert len(audio_folders) == len(metadata_paths), \
        f"Number of audio_folder ({len(audio_folders)}) must match number of metadata_path ({len(metadata_paths)})"
    assert len(audio_folders) > 0, "At least one audio_folder and metadata_path must be provided"
    
    datasets = []
    for i, (audio_folder, metadata_path) in enumerate(zip(audio_folders, metadata_paths)):
        rank0_print(f"Loading dataset {i+1}/{len(audio_folders)}: audio_folder={audio_folder}, metadata_path={metadata_path}")
        dataset = AudioVerse(
            audio_folder=audio_folder,
            metadata_path=metadata_path,
            tokenizer=tokenizer,
            use_tag_clustering=data_args.use_tag_clustering,
            tag_cluster_path=data_args.tag_cluster_path,
            use_positive_mask=data_args.use_positive_mask,
        )
        datasets.append(dataset)
        rank0_print(f"  Dataset {i+1} loaded with {len(dataset)} samples")
    
    if len(datasets) > 1:
        train_dataset = ConcatDataset(datasets)
        rank0_print(f"Concatenated {len(datasets)} datasets, total samples: {len(train_dataset)}")
    else:
        train_dataset = datasets[0]
        rank0_print(f"Using single dataset with {len(train_dataset)} samples")
    
    eval_dataset = None
    training_args.eval_strategy = "no"

    # Build collator.
    data_collator = COLLATORS[model_args.model_family_id](
        tokenizer=tokenizer,
        processor=processor,
    )

    # Disable reentrant checkpointing to avoid known issues with this model path.
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset, 
    )
    
    # Resume training if a valid checkpoint path is provided.
    resume_from_checkpoint = None
    if hasattr(training_args, 'resume_from_checkpoint') and training_args.resume_from_checkpoint:
        resume_path = training_args.resume_from_checkpoint
        if resume_path and resume_path.lower() not in ['false', ''] and Path(resume_path).exists():
            resume_from_checkpoint = resume_path
            rank0_print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        elif resume_path and resume_path.lower() not in ['false', '']:
            rank0_print(f"Warning: resume_from_checkpoint specified but path does not exist: {resume_path}")
            rank0_print("Starting training from scratch instead.")
    
    print('Start training...')
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)
    

if __name__ == "__main__":
    train()
