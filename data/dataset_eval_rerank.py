import json
import os
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset

from qwen_omni_utils import process_mm_info

from .base.base_collactor import BaseDataCollator
from .dataset_eval_retrieval import _resolve_audio_path, _extract_id

def construct_rerank_messages_single_candidate_inference(query_dict, cand_dict):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I will provide you with a query and a candidate. "
                        "Please evaluate whether the candidate matches the query. "
                        "If it does, respond with 'Yes'; if it doesn't, respond with 'No'."
                    ),
                }
            ],
        }
    ]
    query = [{"type": "text", "text": "Query:"}]
    cand = [{"type": "text", "text": "Candidate:"}]

    if "audio" in query_dict:
        query.append({"type": "audio", "audio": query_dict["audio"]})
    if "txt" in query_dict:
        query.append({"type": "text", "text": query_dict["txt"]})
    if "audio" in cand_dict:
        cand.append({"type": "audio", "audio": cand_dict["audio"]})
    if "txt" in cand_dict:
        cand.append({"type": "text", "text": cand_dict["txt"]})

    message[0]["content"].extend(query)
    message[0]["content"].extend(cand)
    return message


class _BaseRerankDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        audio_dir,
        tokenizer,
        processor,
        rerank_metadata_path,
        topk=50,
        dataset_name="unknown",
        type="audio",
    ):
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        if not os.path.exists(rerank_metadata_path):
            raise FileNotFoundError(f"Rerank metadata not found: {rerank_metadata_path}")

        self.audio_dir = audio_dir
        self.annos = json.load(open(metadata_path))
        self.idx = list(range(len(self.annos)))
        self.dataset_name = dataset_name
        self.type = type
        self.TOPK = int(topk)

        self.worker_init_fn = None
        self.use_sampler = True

        self.tokenizer = tokenizer
        self.processor = processor
        self.collate_fn = EvalDataCollatorRerank(self.tokenizer, self.processor)

        self.rerank_metadata = json.load(open(rerank_metadata_path))

    def get_text_by_index(self, idx):
        if isinstance(idx, str) and ":" in idx:
            idx, caption_idx = idx.split(":")
            idx = int(idx)
            caption_idx = int(caption_idx)
        else:
            idx = int(idx)
            caption_idx = 0

        if idx < 0 or idx >= len(self.annos):
            return None

        anno = self.annos[idx]
        raw_captions = anno.get("desc", anno.get("caption", None))
        if isinstance(raw_captions, list):
            if len(raw_captions) == 0:
                return None
            if caption_idx >= len(raw_captions):
                caption_idx = 0
            raw_captions = raw_captions[caption_idx]
        return raw_captions

    def get_audio_path_by_index(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= len(self.annos):
            return None

        anno = self.annos[idx]
        item_id = _extract_id(anno)
        if item_id is None:
            return None

        audio_path = _resolve_audio_path(self.audio_dir, item_id)
        return audio_path if os.path.exists(audio_path) else None


class EvalDatasetRerank_Audio2Text(_BaseRerankDataset):
    def __init__(
        self,
        metadata_path,
        audio_dir,
        tokenizer,
        processor,
        rerank_metadata_path,
        topk=50,
        dataset_name="unknown",
        type="audio",
    ):
        super().__init__(
            metadata_path=metadata_path,
            audio_dir=audio_dir,
            tokenizer=tokenizer,
            processor=processor,
            rerank_metadata_path=rerank_metadata_path,
            topk=topk,
            dataset_name=dataset_name,
            type=type,
        )

        self.rerank_metadata_dict_audio2text = {}
        for entry in self.rerank_metadata["audio2text"]:
            metadata_idx = entry.get("audio_id")
            self.rerank_metadata_dict_audio2text[metadata_idx] = entry

    def __len__(self):
        return len(self.rerank_metadata_dict_audio2text)

    def __getitem__(self, i):
        audio_path = self.get_audio_path_by_index(i)
        assert i in self.rerank_metadata_dict_audio2text, f"Index {i} not found in rerank metadata for audio2text"

        metadata_entry_audio2text = self.rerank_metadata_dict_audio2text[i]
        audio2text_indices = metadata_entry_audio2text.get("top_text_ids", [])

        query_dict_audio = {
            "audio": audio_path,
            "txt": "Find a caption describing the sound events in the given audio.",
        }

        audio2text_messages = []
        total_text_ids = []

        for text_idx in audio2text_indices[: self.TOPK]:
            cand_text = self.get_text_by_index(text_idx)
            if cand_text is None:
                continue

            cand_dict = {"txt": cand_text}
            message = construct_rerank_messages_single_candidate_inference(query_dict_audio, cand_dict)
            audio2text_messages.append(message)
            total_text_ids.append(text_idx)

        return audio2text_messages, i, total_text_ids


class EvalDatasetRerank_Text2Audio(_BaseRerankDataset):
    def __init__(
        self,
        metadata_path,
        audio_dir,
        tokenizer,
        processor,
        rerank_metadata_path,
        topk=50,
        dataset_name="unknown",
        type="audio",
    ):
        super().__init__(
            metadata_path=metadata_path,
            audio_dir=audio_dir,
            tokenizer=tokenizer,
            processor=processor,
            rerank_metadata_path=rerank_metadata_path,
            topk=topk,
            dataset_name=dataset_name,
            type=type,
        )

        self.rerank_metadata_list_text2audio = []

        captions_per_audio = 5 if ("audiocaps" in self.dataset_name or "clotho" in self.dataset_name) else 1
        for i in range(0, len(self.rerank_metadata["text2audio"]), captions_per_audio):
            for j in range(captions_per_audio):
                entry = self.rerank_metadata["text2audio"][i + j]
                metadata_idx = entry.get("text_id")
                unique_text_id = f"{metadata_idx}:{j}"
                self.rerank_metadata_list_text2audio.append((unique_text_id, entry))

    def __len__(self):
        return len(self.rerank_metadata_list_text2audio)

    def __getitem__(self, i):
        unique_text_id, metadata_entry_text2audio = self.rerank_metadata_list_text2audio[i]
        current_text = self.get_text_by_index(unique_text_id)

        text2audio_indices = metadata_entry_text2audio.get("top_audio_ids", [])
        query_dict_text = {
            "txt": f"Find an audio containing the sound events in the following caption: {current_text}"
        }

        text2audio_messages = []
        total_audio_ids = []

        for audio_idx in text2audio_indices[: self.TOPK]:
            cand_audio_path = self.get_audio_path_by_index(audio_idx)
            if cand_audio_path is None:
                continue

            cand_dict = {"audio": cand_audio_path}
            message = construct_rerank_messages_single_candidate_inference(query_dict_text, cand_dict)
            text2audio_messages.append(message)
            total_audio_ids.append(audio_idx)

        return text2audio_messages, unique_text_id, total_audio_ids


class EvalDataCollatorRerank(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def process_messages(self, messages):
        texts = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        inputs = self.processor(
            text=texts,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID

        attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else None
        pixel_values = inputs["pixel_values"] if "pixel_values" in inputs else None
        image_grid_thw = inputs["image_grid_thw"] if "image_grid_thw" in inputs else None
        pixel_values_videos = inputs["pixel_values_videos"] if "pixel_values_videos" in inputs else None
        video_grid_thw = inputs["video_grid_thw"] if "video_grid_thw" in inputs else None
        input_features = inputs["input_features"] if "input_features" in inputs else None
        feature_attention_mask = inputs["feature_attention_mask"] if "feature_attention_mask" in inputs else None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "labels": labels,
        }

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        all_messages = []
        index_list = []
        candidate_ids_list = []

        for item in messages:
            if len(item) != 3:
                raise ValueError(f"Unexpected item format with {len(item)} elements. Expected 3 elements.")

            msg_list, index, candidate_ids = item[0], item[1], item[2]
            index_list.append(index)
            candidate_ids_list.append(candidate_ids)
            all_messages.extend(msg_list)

        return_dict = {}
        if len(all_messages) > 0:
            return_dict.update(self.process_messages(all_messages))

        return_dict["index"] = index_list
        return_dict["candidate_ids"] = candidate_ids_list
        return return_dict
