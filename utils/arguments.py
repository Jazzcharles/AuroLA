from typing import Dict, Optional, List
from dataclasses import dataclass, field

import transformers

from supported_models import MODEL_HF_PATH, MODEL_FAMILIES


@dataclass
class ModelArguments:
    model_id: str = field(default="llava-1.5-7b")
    model_local_path: Optional[str] = field(default=None)
    stage1_model_local_path: Optional[str] = field(default=None)

    def __post_init__(self):
        assert self.model_id in MODEL_HF_PATH, f"Unknown model_id: {self.model_id}"
        self.model_hf_path: str = MODEL_HF_PATH[self.model_id]
        assert self.model_id in MODEL_FAMILIES, f"Unknown model_id: {self.model_id}"
        self.model_family_id: str = MODEL_FAMILIES[self.model_id]

        if not self.model_local_path:
            self.model_local_path = self.model_hf_path


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data json file."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data json file."}
    )
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    audio_folder: Optional[List[str]] = field(default=None, metadata={"help": "Audio folder path(s). Can be specified multiple times for multiple datasets."})
    metadata_path: Optional[List[str]] = field(default=None, metadata={"help": "Metadata path(s). Can be specified multiple times for multiple datasets."})
    num_frames: Optional[int] = field(default=8)
    user_key: Optional[str] = field(default="human")
    assistant_key: Optional[str] = field(default="gpt")
    image_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the image data json file."}
    )
    text_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the text data json file."}
    )
    query_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the query data json file."}
    )
    cand_pool_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the cand pool data json file."}
    )
    instructions_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the instructions data json file."}
    )
    rerank_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the rerank data json file."}
    )
    image_path_prefix: Optional[str] = field(
        default=None, metadata={"help": "Path to the image files."}
    )
    ### added ###
    num_hard_negative: int = field(default=0, metadata={"help": "Number of hard negative samples."})
    num_positives: int = field(default=0, metadata={"help": "Number of positive samples."})
    use_tag_clustering: bool = field(default=False, metadata={"help": "Use tag clustering in the data."})
    tag_cluster_path: Optional[str] = field(default=None, metadata={"help": "Path to the tag cluster data json file."}) 
    use_positive_mask: bool = field(default=False, metadata={"help": "Use positive mask in the data."})
    use_text_masking: bool = field(default=False, metadata={"help": "Use text masking in the data."})
    text_masking_ratio: float = field(default=0.0, metadata={"help": "Ratio of text masking in the data."})
    rerank_option: str = field(default="retrieval", metadata={"help": "Rerank option: 'retrieval', 'generative', or 'retrieval_generative'."})
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_flash_attn: bool = False
    train_vision_encoder: bool = False
    train_vision_projector: bool = False
    vision_projector_lr: float = None 
    train_audio_encoder: bool = False
    train_llm: bool = True
    contrastive_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Alpha parameter for HN-NCE loss. Lower values (e.g., 0.3-0.5) help when false negatives exist in the data."
        }
    )
    contrastive_lambda: float = field(
        default=1.0,
        metadata={
            "help": "Lambda parameter for Hybrid NCE loss. Global scaling factor for extra positive pairs. "
                   "Diagonal weight=1, extra positive weight=lambda*IoU(cluster_i, cluster_j). "
                   "Lower values (e.g., 0.3-0.7) help when false positives exist. Lambda=0 gives standard InfoNCE."
        }
    )
    contrastive_beta: float = field(
        default=0.1,
        metadata={
            "help": "Beta parameter for HN-NCE loss. Controls the hardness of negative samples. "
                   "Higher values (e.g., 0.2-0.5) emphasize harder negatives."
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False


@dataclass
class LoraArguments:
    use_lora: bool = True
    use_vision_lora: bool = True
    use_audio_lora: bool = False
    q_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    use_dora: bool = False
    vision_lora_r: int = 16
    vision_lora_alpha: int = 16
    audio_lora_r: int = 16
    audio_lora_alpha: int = 16