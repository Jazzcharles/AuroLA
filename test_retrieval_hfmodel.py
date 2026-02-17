import json
import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor

from utils.distributed import all_gather_list, ddp_allgather
from utils.misc import (
    save_hparams,
    tensors_to_device,
)
from utils.logger import SystemPromptWarningFilter
from test_retrieval import (
    _init_distributed, 
    add_embed_token, 
    _compute_metric_ret,
    build_val_loaders,
    parse_cli,
    save_top50_retrieval_results,
)

def _split_text_audio_inputs(batch):
    text_input = {}
    audio_input = {}
    for k, v in batch.items():
        if k.startswith("text_"):
            text_input[k.replace("text_", "")] = v
        elif k.startswith("audio_"):
            audio_input[k.replace("audio_", "")] = v
    return text_input, audio_input


def get_embed_feature(hidden_states, input_ids, embed_index):
    embed_indices = torch.argmax((input_ids == embed_index).int(), dim=1)
    embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]
    return embed_features


def main():
    logging.getLogger("root").addFilter(SystemPromptWarningFilter())

    args = parse_cli()
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    assert datasets, "datasets cannot be empty"
    allowed = {"audiocaps", "clotho", "auto-acd"}
    invalid = [d for d in datasets if d not in allowed]
    assert not invalid, f"Unsupported datasets: {invalid}. Allowed: {sorted(allowed)}"

    args.output_dir = args.output_dir or os.path.join(args.model_name_or_path, "downstream")
    os.makedirs(args.output_dir, exist_ok=True)
    _init_distributed(args)
    save_hparams(args.output_dir, args, filename="hps.json")
    device = torch.device("cuda", args.local_rank)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer = processor.tokenizer

    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    add_embed_token(tokenizer, model)

    val_loaders = build_val_loaders(
        datasets,
        tokenizer,
        processor,
        args.batch_size,
        args.num_workers,
    )

    eval_log = {}
    model.eval()
    for dataset, val_loader in val_loaders.items():
        ids = []
        ids_txt = []
        text_features = []
        audio_features = []

        for batch in tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"[eval] {dataset}",
            disable=dist.get_rank() != 0,
        ):
            batch = tensors_to_device(
                batch,
                device,
                model.dtype,
                float_tensor_keys={"pixel_values"},
                float_tensor_key_substrings=("input_features",),
            )
            with torch.inference_mode():
                text_inputs, audio_inputs = _split_text_audio_inputs(batch)
                text_embeds = model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
                text_embeds = get_embed_feature(text_embeds, text_inputs['input_ids'], model.config.emb_token_ids[0])
                audio_embeds = model(**audio_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
                audio_embeds = get_embed_feature(audio_embeds, audio_inputs['input_ids'], model.config.emb_token_ids[0])
        
            text_features.append(text_embeds)
            audio_features.append(audio_embeds)

            ids += batch['ids']
            if "ids_txt" in batch:
                ids_txt += batch['ids_txt']
            else:
                ids_txt += batch['ids']

        ids = [j for i in all_gather_list(ids) for j in i]
        ids_txt = [j for i in all_gather_list(ids_txt) for j in i]
        text_features = ddp_allgather(torch.cat(text_features, dim=0))
        audio_features = ddp_allgather(torch.cat(audio_features, dim=0))
        
        dataset_output_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        text_features_norm = F.normalize(text_features, dim=-1)
        audio_features_norm = F.normalize(audio_features, dim=-1)
        score_matrix = torch.matmul(text_features_norm, audio_features_norm.permute(1, 0))
        
        log = _compute_metric_ret(score_matrix, ids, ids_txt, direction="text2audio")
        log2 = _compute_metric_ret(score_matrix, ids, ids_txt, direction="audio2text")
        log.update(log2)

        if dist.get_rank() == 0:
            save_top50_retrieval_results(score_matrix, ids, ids_txt, dataset_output_dir, top_k=100)
            pth_output_path = os.path.join(dataset_output_dir, "similarity_matrix.pth")
            torch.save(
                {
                    "similarity_matrix": score_matrix.cpu(),
                    "ids": ids,
                    "ids_txt": ids_txt,
                },
                pth_output_path,
            )
            print(f"[INFO] Saved similarity matrix, ids, and ids_txt to {pth_output_path}")
            print(f"[test_retrieval] {dataset} log: {log}")

        eval_log[dataset] = log

    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        eval_log_path = os.path.join(args.output_dir, "testing_allinone_metrics.json")
        with open(eval_log_path, "w") as f:
            json.dump(eval_log, f, indent=4, ensure_ascii=False)
        print(f"[test_retrieval] Eval log saved to {eval_log_path}")

if __name__ == "__main__":
    main()
