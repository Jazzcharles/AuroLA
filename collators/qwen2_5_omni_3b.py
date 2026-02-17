from typing import Dict, Sequence
import numpy as np

import torch

from . import register_collator
from .base import BaseDataCollator
from qwen_omni_utils import process_mm_info


@register_collator("qwen2_5-omni-3b")
class Qwen2_5Omni3BDataCollator(BaseDataCollator):
    MAX_AUDIO_SAMPLES = 480000  # ≈30s @ 16kHz
    MIN_AUDIO_SAMPLES = 32000

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # supports both (audio, text) and (audio, text, positive_mask)
        message_without_positive_mask = []
        all_positive_masks = []

        if len(messages[0]) == 3:  # (audio, text, positive_mask)
            for item in messages:
                for idx in range(len(item) // 3):
                    message_without_positive_mask.append(item[idx * 3: idx * 3 + 2])
                    all_positive_masks.append(item[idx * 3 + 2])
            all_positive_masks = torch.stack(all_positive_masks)
        elif len(messages[0]) == 2:  # (audio, text)
            message_without_positive_mask = messages
            all_positive_masks = None
        else:
            raise ValueError(f"Invalid message length: {len(messages[0])}")

        category_size = len(message_without_positive_mask[0])
        new_messages = []
        for category in range(category_size):
            for item in message_without_positive_mask:
                new_messages.append(item[category])

        texts = self.processor.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=False)
        audios, images, videos = process_mm_info(new_messages, use_audio_in_video=True)

        if audios is not None:
            safe_audios = []
            for idx, a in enumerate(audios):
                try:
                    if a is None:
                        a = np.zeros(self.MIN_AUDIO_SAMPLES, dtype=np.float32)
                    if isinstance(a, torch.Tensor):
                        a = a.cpu().numpy()
                    if a.ndim > 1:
                        a = a.squeeze()
                    if len(a) < self.MIN_AUDIO_SAMPLES:
                        padded = np.zeros(self.MIN_AUDIO_SAMPLES, dtype=np.float32)
                        padded[:len(a)] = a
                        a = padded
                    if len(a) > self.MAX_AUDIO_SAMPLES:
                        a = a[:self.MAX_AUDIO_SAMPLES]
                    safe_audios.append(a)
                except Exception as e:
                    print(f"[Warning] fixing audio idx {idx}: {e}")
                    safe_audios.append(np.zeros(self.MIN_AUDIO_SAMPLES, dtype=np.float32))
            audios = safe_audios

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

        return dict(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            input_features=inputs.get("input_features"),
            feature_attention_mask=inputs.get("feature_attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_grid_thw=inputs.get("video_grid_thw"),
            labels=labels,
            positive_mask=all_positive_masks,
        )
