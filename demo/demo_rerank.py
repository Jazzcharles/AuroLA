import torch
from qwen_omni_utils import process_mm_info
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

def construct_rerank_message(audio_path, text):
    return [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "I will provide you with a query and a candidate. "
                    "Please evaluate whether the candidate matches the query. "
                    "If it does, respond with 'Yes'; if it doesn't, respond with 'No'."
                ),
            },
            {"type": "text", "text": "Query:"},
            {"type": "audio", "audio": audio_path},
            {
                "type": "text",
                "text": "Find a caption describing the sound events in the given audio.",
            },
            {"type": "text", "text": "Candidate:"},
            {"type": "text", "text": text},
        ],
    }]


def process_input(messages, device):
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(
        text=texts,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(device)
    return inputs

def rerank_yes_prob(messages):
    inputs = process_input(messages, device)
    yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("No", add_special_tokens=False).input_ids[0]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
        )
        logits = outputs.scores[0]
        yes_probs = torch.softmax(logits[:, [yes_id, no_id]], dim=-1)[:, 0]
    return yes_probs


# 1) Load model + processor
model_path = "hf_export/AuroLA-rerank-3B"  # or your HF repo id
# model_path = "hf_export/AuroLA-rerank-7B"  # or your HF repo id

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
tokenizer = processor.tokenizer

# 2) Prepare one audio-text pair
audio_files = [
    "/mnt/data/AudioCaps/audio/--0w1YA1Hm4_30.wav",
    "/mnt/data/AudioCaps/audio/--0w1YA1Hm4_30.wav",
]
text_queries = [
    "A vehicle driving as a man and woman are talking and laughing",
    "Wind is blowing and heavy rain is falling and splashing",
]
# 3) Build rerank prompt and score
messages = [construct_rerank_message(audio_path, text) for (audio_path, text) in zip(audio_files, text_queries)]
yes_prob = rerank_yes_prob(messages)
yes_prob = [prob.detach().cpu().item() for prob in yes_prob]

print("The probability of the audio matching the text is:", yes_prob)
