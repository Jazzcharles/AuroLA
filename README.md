# AuroLA

This repo contains the official implementation of the following paper:

<div align="center">

### **Scaling Audio-Text Retrieval with Multimodal Large Language Model**

[Jilan Xu](https://jazzcharles.github.io/)<sup>1</sup> · 
[Carl Thome](https://carlthome.github.io/)<sup>2</sup> · 
[Danijela Horak](https://scholar.google.com/citations?user=ZqQT2FwAAAAJ&hl=en)<sup>2</sup> · 
[Weidi Xie](https://weidixie.github.io/)<sup>3</sup> · 
[Andrew Zisserman](https://scholar.google.com/citations?user=UZ5wscMAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup>Visual Geometry Group, University of Oxford  
<sup>2</sup>Epidemic Sound  
<sup>3</sup>Shanghai Jiao Tong University

</div>

[![Paper](https://img.shields.io/badge/cs.SD-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2602.18010)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Jazzcharles/AuroLA-7B)
<!-- [![Project Page](https://img.shields.io/badge/Homepage-Website-green)]() -->


## News <a name="news"></a>

- `[2026/02]` Initial release of the code and checkpoints [AuroLA-3B](https://huggingface.co/Jazzcharles/AuroLA-3B),[AuroLA-rerank-3B](https://huggingface.co/Jazzcharles/AuroLA-rerank-3B), [AuroLA-7B](https://huggingface.co/Jazzcharles/AuroLA-7B), [AuroLA-rerank-7B](https://huggingface.co/Jazzcharles/AuroLA-rerank-7B)


## Overview
AuroLA is a a novel contrastive
language-audio pre-training framework that re-purposes
Multimodal Large Language Models (MLLMs) as a unified
backbone for retrieval. 
In this repo, we provide:
- **AuroLA**: a unified MLLM-based embedding model for audio & text feature extraction and fast retrieval.
- **AuroLA-rerank**: an MLLM-based generative model for precise re-ranking.
- **AudioVerse (TODO)**: a large-scale and
diverse audio dataset paired with rich and accurate multi-granular captions.


## Quick Start 
Clone the repository and install the dependicies
```python
git clone https://github.com/Jazzcharles/AuroLA
cd AuroLA
pip install -r requirements.txt
```

Try the model to extract audio and text features. For detail, please refer to [demo_retrieval](demo/demo_retrieval.py) and [demo_rerank](demo/demo_rerank.py).

```python
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info

def add_embed_token(tokenizer, model, emb_token="<emb>"):
    emb_tokens = [emb_token]
    num_new_tokens = tokenizer.add_tokens(emb_tokens)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
    model.config.emb_token_ids = emb_token_ids
    return emb_token_ids[0]

# Tokenize / process (audio side)
def process_input(message, device):
    texts = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    audios, images, videos = process_mm_info(message, use_audio_in_video=False)
    inputs = processor(
        text=texts,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(device)
    return inputs

# Extract features at the position before <emb> token
def get_embed_feature(hidden_states, input_ids, embed_index):
    embed_indices = torch.argmax((input_ids == embed_index).int(), dim=1)
    embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]
    return embed_features

# 1) Load model + processor (same style as Qwen2.5-Omni)
model_path = "Jazzcharles/AuroLA-7B"  # or your HF repo id

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
tokenizer = processor.tokenizer

emb_token_ids = add_embed_token(tokenizer, model)


# 2) Prepare retrieval inputs
# audio paths and text queries can be any same-batch lists
audio_files = [
    "/mnt/data/AudioCaps/audio/--0w1YA1Hm4_30.wav", 
    "/mnt/data/AudioCaps/audio/-AheI8Epim4_30.wav", 
    "/mnt/data/AudioCaps/audio/-BUWGM7qeUM_10.wav",
]
text_queries = [
    "A vehicle driving as a man and woman are talking and laughing",
    "Muffled sounds followed by metal being hit",
    "Wind is blowing and heavy rain is falling and splashing",
]

# Build audio-side messages
audio_messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": a},
                {"type": "text", "text": "Summarize above audio in one word:"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<emb>."}],
        },
    ]
    for a in audio_files
]

# Build text-side messages
text_messages = [
    [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"{t}\nSummarize above sentence in one word:"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<emb>."}],
        },
    ]
    for t in text_queries
]

# 3) Tokenize / process (audio side & text side)
audio_inputs = process_input(audio_messages, device)
text_inputs = process_input(text_messages, device)

# 4) Forward and extract features
with torch.inference_mode():
    audio_out = model(**audio_inputs, output_hidden_states=True, return_dict=True, use_audio_in_video=False)
    audio_feat = get_embed_feature(audio_out.hidden_states[-1], audio_inputs['input_ids'], emb_token_ids)
    
    text_out = model(**text_inputs, output_hidden_states=True, return_dict=True, use_audio_in_video=False)
    text_feat = get_embed_feature(text_out.hidden_states[-1], text_inputs['input_ids'], emb_token_ids)

# 5) Similarity + top-k retrieval
audio_feat = F.normalize(audio_feat, dim=-1)
text_feat = F.normalize(text_feat, dim=-1)
score = text_feat @ audio_feat.T  # [N_text, N_audio]
print(score.shape, score)
```

## Pre-trained checkpoints
We provide pre-trained checkpoints under two different settings: 
- PT-setting: The model is pre-trained on AudioVerse, but excluding AudioCaps, Clotho, VGGSound, and EPIC-Sounds. This is for fair evaluation comparison with exisiting works. 
- Full-setting: The model is pre-trained on full AudioVerse dataset. This model achieves the SOTA performance.

| Checkpoint | Setting | AudioCaps T2A | AudioCaps A2T | Clotho T2A | Clotho A2T |
| --- | --- | --- | --- | --- | --- |
| [AuroLA-3B](https://huggingface.co/Jazzcharles/AuroLA-3B-PT) | PT | 42.5 | 52.5 | 25.0 | 30.8 |
| [AuroLA-3B](https://huggingface.co/Jazzcharles/AuroLA-3B) | Full | 44.9 | 60.5 | 25.6 | 34.9 |
| [AuroLA-rerank-3B](https://huggingface.co/Jazzcharles/AuroLA-rerank-3B) | Full | 49.7 | 64.4 | 27.3 | 38.9 |
| [AuroLA-7B](https://huggingface.co/Jazzcharles/AuroLA-7B-PT) | PT | 43.3 | 53.0 | 26.0 | 34.4 |
| [AuroLA-7B](https://huggingface.co/Jazzcharles/AuroLA-7B) | Full | 46.8 | 64.0 | 26.7 | 36.5 |
| [AuroLA-rerank-7B](https://huggingface.co/Jazzcharles/AuroLA-rerank-7B) | Full | 51.0 | 65.6 | 28.2 | 38.6 |
 
## Evaluate HF AuroLA model
See [eval.md](docs/eval.md)

## Fine-tuning scripts
See [finetune.md](docs/finetune.md)

## Acknowledgement
This codebase is based on [VAST](https://github.com/CASIA-IVA-Lab/VAST), [LamRA](https://github.com/Code-kunkun/LamRA), [WavCaps](https://github.com/XinhaoMei/WavCaps). Thanks for their great work.

## Citation
If you find our work helps, please cite our paper.

```bibtex
@misc{xu2026scalingaudiotextretrievalmultimodal,
      title={Scaling Audio-Text Retrieval with Multimodal Large Language Models}, 
      author={Jilan Xu and Carl Thomé and Danijela Horak and Weidi Xie and Andrew Zisserman},
      year={2026},
      eprint={2602.18010},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2602.18010}, 
}
```
