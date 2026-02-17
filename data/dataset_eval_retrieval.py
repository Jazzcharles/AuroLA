import json
import os
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset

from .base.base_collactor import BaseDataCollator
from .dataset_audioverse import construct_messages
from qwen_omni_utils import process_mm_info


def _resolve_audio_path(audio_dir: str, audio_id: str) -> str:
    base = os.path.join(audio_dir, audio_id)
    if os.path.exists(base):
        return base

    known_extensions = [".wav", ".flac", ".mp3", ".mkv"]
    has_ext = any(base.lower().endswith(ext) for ext in known_extensions)

    if has_ext:
        stem = base
        for ext in known_extensions:
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break
    else:
        stem = base

    for ext in ["wav", "flac", "mp3", "mkv"]:
        candidate = f"{stem}.{ext}"
        if os.path.exists(candidate):
            return candidate

    return base


def _extract_id(anno: Dict) -> str:
    for key in ["video_id", "image_id", "image", "id"]:
        if key in anno:
            return anno[key]
    raise KeyError(f"Cannot find id field in annotation keys: {list(anno.keys())}")


class EvalDatasetRetrieval(Dataset):
    def __init__(self, audio_dir, metadata, tokenizer, processor):
        assert audio_dir, "audio_dir is required"
        assert metadata, "metadata is required"

        self.audio_dir = audio_dir
        self.annos = json.load(open(metadata))

        self.worker_init_fn = None
        self.use_sampler = True

        self.tokenizer = tokenizer
        self.processor = processor
        self.collate_fn = EvalDataCollator(self.tokenizer, self.processor)

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, i):
        anno = self.annos[i]
        sample_id = i

        raw_captions = anno.get("desc", anno.get("caption"))
        if isinstance(raw_captions, str):
            raw_captions = [raw_captions]

        audio_path = _resolve_audio_path(self.audio_dir, _extract_id(anno))
        audio_message = construct_messages(audio=audio_path)
        text_messages = [construct_messages(text=cap) for cap in raw_captions]

        return audio_message, text_messages, sample_id


class EvalDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def process_messages(self, messages, ids, type="audio"):
        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        inputs = self.processor(
            text=texts,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else None
        pixel_values = inputs["pixel_values"] if "pixel_values" in inputs else None
        image_grid_thw = inputs["image_grid_thw"] if "image_grid_thw" in inputs else None
        pixel_values_videos = (
            inputs["pixel_values_videos"] if "pixel_values_videos" in inputs else None
        )
        video_grid_thw = inputs["video_grid_thw"] if "video_grid_thw" in inputs else None
        input_features = inputs["input_features"] if "input_features" in inputs else None
        feature_attention_mask = (
            inputs["feature_attention_mask"]
            if "feature_attention_mask" in inputs
            else None
        )

        return {
            f"{type}_input_ids": input_ids,
            f"{type}_attention_mask": attention_mask,
            f"{type}_input_features": input_features,
            f"{type}_feature_attention_mask": feature_attention_mask,
            f"{type}_pixel_values": pixel_values,
            f"{type}_image_grid_thw": image_grid_thw,
            f"{type}_pixel_values_videos": pixel_values_videos,
            f"{type}_video_grid_thw": video_grid_thw,
            f"{type}_labels": labels,
            f"{type}_has_hard_negative": False,
        }

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        audio_messages = []
        text_messages = []
        audio_ids = []
        text_ids = []

        for item in messages:
            audio_msg, text_msg_list, sample_id = item[0], item[1], item[2]
            audio_messages.append(audio_msg)
            audio_ids.append(sample_id)

            for text_message in text_msg_list:
                text_messages.append(text_message)
                text_ids.append(sample_id)

        return_dict = {}
        return_dict.update(self.process_messages(audio_messages, audio_ids, type="audio"))
        return_dict.update(self.process_messages(text_messages, text_ids, type="text"))
        return_dict["ids"] = audio_ids
        return_dict["ids_txt"] = text_ids
        return return_dict
