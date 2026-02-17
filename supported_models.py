from typing import Dict, List
from collections import OrderedDict

from collators import COLLATORS
from loaders import LOADERS


MODULE_KEYWORDS: Dict[str, Dict[str, List]] = {
    # "qwen2-vl-7b": {
    #     "vision_encoder": ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"],
    #     "vision_projector": ["visual.merger"],
    #     "llm": ["model"]
    # },
    # "qwen2-vl-2b": {
    #     "vision_encoder": ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"],
    #     "vision_projector": ["visual.merger"],
    #     "llm": ["model"]
    # },
    "qwen2_5-omni-7b": {
        "vision_encoder": ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"],
        "vision_projector": ["visual.merger"],
        "audio_encoder": ["audio_tower.conv1", "audio_tower.conv2", "audio_tower.positional_embedding", "audio_tower.audio_bos_eos_token", "audio_tower.layers", "audio_tower.ln_post", "audio_tower.avg_pooler", "audio_tower.proj"],
        "llm": ["model"]
    },
    "qwen2_5-omni-3b": {
        "vision_encoder": ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"],
        "vision_projector": ["visual.merger"],
        "audio_encoder": ["audio_tower.conv1", "audio_tower.conv2", "audio_tower.positional_embedding", "audio_tower.audio_bos_eos_token", "audio_tower.layers", "audio_tower.ln_post", "audio_tower.avg_pooler", "audio_tower.proj"],
        "llm": ["model"]
    },
}


MODEL_HF_PATH = OrderedDict()
MODEL_FAMILIES = OrderedDict()


def register_model(model_id: str, model_family_id: str, model_hf_path: str) -> None:
    if model_id in MODEL_HF_PATH or model_id in MODEL_FAMILIES:
        raise ValueError(f"Duplicate model_id: {model_id}")
    MODEL_HF_PATH[model_id] = model_hf_path
    MODEL_FAMILIES[model_id] = model_family_id


#=============================================================
# register_model(
#     model_id="qwen2-vl-7b",
#     model_family_id="qwen2-vl-7b",
#     model_hf_path="/home/jilan_xu/checkpoints/Qwen2-VL-7B-Instruct"
# )

# register_model(
#     model_id="qwen2-vl-2b",
#     model_family_id="qwen2-vl-2b",
#     model_hf_path="/home/jilan_xu/checkpoints/Qwen2-VL-2B-Instruct"
# )

register_model(
    model_id="qwen2_5-omni-7b",
    model_family_id="qwen2_5-omni-7b",
    model_hf_path="/home/jilan_xu/checkpoints/Qwen2.5-Omni-7B"
    # model_hf_path="/mnt/vision_user/xjl/checkpoints/Qwen2.5-Omni-7B"
)

register_model(
    model_id="qwen2_5-omni-3b",
    model_family_id="qwen2_5-omni-3b",
    model_hf_path="/home/jilan_xu/checkpoints/Qwen2.5-Omni-3B"
    # model_hf_path="/mnt/vision_user/huggingface/Qwen2.5-Omni-3B"
)


# sanity check
for model_family_id in MODEL_FAMILIES.values():
    assert model_family_id in COLLATORS, f"Collator not found for model family: {model_family_id}"
    assert model_family_id in LOADERS, f"Loader not found for model family: {model_family_id}"
    assert model_family_id in MODULE_KEYWORDS, f"Module keywords not found for model family: {model_family_id}"


if __name__ == "__main__":
    temp = "Model ID"
    ljust = 30
    print("Supported models:")
    print(f"  {temp.ljust(ljust)}: HuggingFace Path")
    print("  ------------------------------------------------")
    for model_id, model_hf_path in MODEL_HF_PATH.items():
        print(f"  {model_id.ljust(ljust)}: {model_hf_path}")
