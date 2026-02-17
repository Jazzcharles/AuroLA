from typing import Dict, Sequence
import numpy as np

import torch

from . import register_collator
from .base import BaseDataCollator
from qwen_omni_utils import process_mm_info

def extract_inputs(inputs, PAD_TOKEN_ID, IGNORE_TOKEN_ID):
    if 'attention_mask' in inputs:
        attention_mask = inputs['attention_mask']
    else:
        attention_mask = None 
    if 'pixel_values' in inputs:
        pixel_values = inputs['pixel_values']
    else: 
        pixel_values = None 
    if 'image_grid_thw' in inputs:
        image_grid_thw = inputs['image_grid_thw']
    else:
        image_grid_thw = None 
    if 'input_features' in inputs:
        input_features = inputs['input_features']
    else:
        input_features = None 
    if 'feature_attention_mask' in inputs:
        feature_attention_mask = inputs['feature_attention_mask']
    else:
        feature_attention_mask = None 

    input_ids = inputs['input_ids']
    labels = input_ids.clone()
    labels[labels == PAD_TOKEN_ID] = IGNORE_TOKEN_ID

    return input_ids, labels, attention_mask, input_features, feature_attention_mask 

def debug_print_tokens(
    tokenizer,
    input_ids,
    sample_idx=0,
    max_tokens=300,
):
    ids = input_ids[sample_idx].tolist()

    print("\n" + "=" * 100)
    print(f"[DEBUG] Token inspection for sample {sample_idx}")
    print("=" * 100)

    for i, tid in enumerate(ids[:max_tokens]):
        token = tokenizer.convert_ids_to_tokens(tid)
        decoded = tokenizer.decode([tid], skip_special_tokens=False)

        flags = []
        if tid == tokenizer.pad_token_id:
            flags.append("PAD")
        if tid == tokenizer.eos_token_id:
            flags.append("EOS")
        if tid == 151644:
            flags.append("<<151644>>")

        flag_str = " | ".join(flags)

        print(
            f"{i:04d} | id={tid:6d} | {token:>15} | {repr(decoded):>20} | {flag_str}"
        )

    print("=" * 100)

       


class Qwen2_5Omni7BDataCollatorRerank(BaseDataCollator):
    MAX_AUDIO_SAMPLES = 480000  # ≈30s @ 16kHz
    MIN_AUDIO_SAMPLES = 32000

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id

    def __call__(self, messages: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # training stage, set the tokenizer padding side to right
        self.processor.tokenizer.padding_side = 'right'

        category_size = len(messages[0])
        batch_size = len(messages)

        # rerank_messages = messages        
        rerank_messages = []
        for category in range(category_size):
            for item in messages:
                rerank_messages.append(item[category])

        rerank_texts = self.processor.apply_chat_template(rerank_messages, tokenize=False, add_generation_prompt=False)
        
        # ====== 音频安全加载部分 ======
        # try:
        audios, images, videos = process_mm_info(rerank_messages, use_audio_in_video=False)
        

        ### for safety, ensure no empty audio ###
        if audios is not None:
            safe_audios = []
            none_count = 0
            for idx, a in enumerate(audios):
                try:
                    if a is None:
                        none_count += 1
                        a = np.zeros(self.MIN_AUDIO_SAMPLES, dtype=np.float32)
                    # to numpy
                    if isinstance(a, torch.Tensor):
                        a = a.cpu().numpy()
                    # ensure 1D
                    if a.ndim > 1:
                        a = a.squeeze()
                    
                    # pad too-short audio
                    if len(a) < self.MIN_AUDIO_SAMPLES:
                        padded = np.zeros(self.MIN_AUDIO_SAMPLES, dtype=np.float32)
                        padded[:len(a)] = a
                        a = padded
                    
                    # crop too long audio (optional)
                    if len(a) > self.MAX_AUDIO_SAMPLES:
                        a = a[:self.MAX_AUDIO_SAMPLES]

                    safe_audios.append(a)
                except Exception as e:
                    print(f"[Warning] fixing audio idx {idx}: {e}")
                    safe_audios.append(np.zeros(self.MIN_AUDIO_SAMPLES, dtype=np.float32))

            audios = safe_audios
            
        inputs = self.processor(
            text=rerank_texts,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        # print('Continue processing...')
        
        rerank_input_ids, rerank_labels, rerank_attention_mask, rerank_input_features, rerank_feature_attention_mask = extract_inputs(inputs, self.PAD_TOKEN_ID, self.IGNORE_TOKEN_ID)
        # debug_print_tokens(
        #     tokenizer=self.processor.tokenizer,
        #     input_ids=rerank_input_ids,
        #     sample_idx=0,      # 先看第一个 rerank sample
        #     max_tokens=200
        # )


        IM_START = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        IM_END = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ASSISTANT_TOKEN_ID = self.processor.tokenizer.convert_tokens_to_ids("assistant")
        YES_TOKEN_ID = self.processor.tokenizer.convert_tokens_to_ids("Yes")    
        NO_TOKEN_ID = self.processor.tokenizer.convert_tokens_to_ids("No")

        rerank_labels[:] = self.IGNORE_TOKEN_ID

        for i in range(len(rerank_input_ids)):
            ids = rerank_input_ids[i]

            for j in range(len(ids) - 3):
                if ids[j] == IM_START and ids[j + 1] == ASSISTANT_TOKEN_ID:
                    start = j + 3  # 跳过 '\n'

                    # 找对应的 im_end
                    for k in range(start, len(ids)):
                        if ids[k] == IM_END:
                            end = k
                            break
                    else:
                        end = len(ids)

                    rerank_labels[i, start:end] = ids[start:end]
                    break

        # print(YES_TOKEN_ID, NO_TOKEN_ID, f"[DEBUG] Rerank labels: {rerank_labels}")
        
        result = dict(
            input_ids=rerank_input_ids,
            attention_mask=rerank_attention_mask,
            input_features=rerank_input_features,
            feature_attention_mask=rerank_feature_attention_mask,
            labels=rerank_labels,
        )
        return result
    
