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