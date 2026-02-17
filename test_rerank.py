import argparse
import logging
import os
from typing import List, Tuple

import librosa
from utils.audio_utils import safe_librosa_load
librosa.load = safe_librosa_load

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from data.dataset_eval_rerank import (
    EvalDatasetRerank_Audio2Text,
    EvalDatasetRerank_Text2Audio,
)
from utils.distributed import DistributedSampler_wopadding
from utils.misc import get_eval_env_paths, save_hparams, tensors_to_device
from utils.logger import SystemPromptWarningFilter


def _init_distributed() -> Tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return rank, world_size, device


def _build_loader(dataset, batch_size, num_workers, world_size):
    per_rank_batch_size = max(1, batch_size // max(1, world_size))
    if world_size > 1:
        sampler = DistributedSampler_wopadding(dataset, shuffle=False, drop_last=False)
    else:
        sampler = SequentialSampler(dataset)

    loader_kwargs = {
        "dataset": dataset,
        "sampler": sampler,
        "batch_size": per_rank_batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": dataset.collate_fn,
        "drop_last": False,
        "worker_init_fn": dataset.worker_init_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


def _flatten_candidate_ids(candidate_ids_batch: List[List]) -> List:
    flat = []
    for one_query_ids in candidate_ids_batch:
        flat.extend(one_query_ids)
    return flat


def _score_one_direction(model, tokenizer, loader, device):
    yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("No", add_special_tokens=False).input_ids[0]

    local_scores = []
    local_index = []
    local_candidate_ids = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            loader,
            total=len(loader),
            desc="[eval] rerank",
            disable=dist.is_initialized() and dist.get_rank() != 0,
        ):
            indexs = batch.pop("index")
            candidate_ids = batch.pop("candidate_ids")
            model_inputs = tensors_to_device(
                batch,
                device,
                model.dtype,
                float_tensor_keys={"pixel_values", "pixel_values_videos", "input_features"},
            )

            outputs = model.generate(
                **model_inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            logits = outputs.scores[0]
            probs = torch.softmax(logits[:, [yes_id, no_id]], dim=-1)[:, 0]

            local_scores.extend(probs.detach().cpu().tolist())
            local_index.extend(indexs)
            local_candidate_ids.extend(_flatten_candidate_ids(candidate_ids))
            # break

    return local_scores, local_index, local_candidate_ids


def _gather_list(local_list: List, world_size: int) -> List:
    if world_size == 1:
        return local_list

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_list)

    merged = []
    for part in gathered:
        merged.extend(part)
    return merged


def _run_direction(model, tokenizer, loader, device, world_size):
    local_scores, local_index, local_candidate_ids = _score_one_direction(model, tokenizer, loader, device)

    all_scores = _gather_list(local_scores, world_size)
    all_index = _gather_list(local_index, world_size)
    all_candidate_ids = _gather_list(local_candidate_ids, world_size)

    return {
        "rerank_scores": torch.tensor(all_scores, dtype=torch.float32).numpy(),
        "rerank_all_index": all_index,
        "rerank_all_candidate_ids": all_candidate_ids,
    }


def parse_cli():
    parser = argparse.ArgumentParser(description="Rerank testing with DDP datasets")
    parser.add_argument("--rerank_model_name_or_path", type=str, required=True)
    parser.add_argument("--original_model_name_or_path", type=str, required=True)
    parser.add_argument("--retrieval_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--datasets", type=str, default="audiocaps")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    logging.getLogger("root").addFilter(SystemPromptWarningFilter())

    args = parse_cli()
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    assert datasets, "datasets cannot be empty"

    if not args.output_root:
        args.output_root = args.retrieval_root
    os.makedirs(args.output_root, exist_ok=True)

    rank, world_size, device = _init_distributed()
    if rank == 0:
        save_hparams(args.output_root, args, filename="rerank_hps.json")

    processor = AutoProcessor.from_pretrained(args.original_model_name_or_path)
    tokenizer = processor.tokenizer
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.rerank_model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    for dataset in datasets:
        dataset_dir = os.path.join(args.retrieval_root, dataset)
        rerank_meta_path = os.path.join(dataset_dir, "rerank_metadata.json")
        retrieval_mat_path = os.path.join(dataset_dir, "similarity_matrix.pth")
        assert os.path.exists(rerank_meta_path), f"Rerank metadata not found: {rerank_meta_path}"
        assert os.path.exists(retrieval_mat_path), f"Retrieval matrix not found: {retrieval_mat_path}"
        
        retrieval_obj = torch.load(retrieval_mat_path, map_location="cpu")
        metadata_path, audio_dir = get_eval_env_paths(dataset)
        retrieval_ids = retrieval_obj["ids"]
        retrieval_ids_txt = retrieval_obj["ids_txt"]
        effective_topk = min(args.topk, len(retrieval_ids), len(retrieval_ids_txt))

        print('Begin loading datasets...')
        ds_audio2text = EvalDatasetRerank_Audio2Text(
            metadata_path=metadata_path,
            audio_dir=audio_dir,
            tokenizer=tokenizer,
            processor=processor,
            rerank_metadata_path=rerank_meta_path,
            topk=effective_topk,
            dataset_name=dataset,
        )
        ds_text2audio = EvalDatasetRerank_Text2Audio(
            metadata_path=metadata_path,
            audio_dir=audio_dir,
            tokenizer=tokenizer,
            processor=processor,
            rerank_metadata_path=rerank_meta_path,
            topk=effective_topk,
            dataset_name=dataset,
        )

        loader_audio2text = _build_loader(ds_audio2text, args.batch_size, args.num_workers, world_size)
        loader_text2audio = _build_loader(ds_text2audio, args.batch_size, args.num_workers, world_size)

        print('Begin reranking...')
        audio2text_result = _run_direction(model, tokenizer, loader_audio2text, device, world_size)
        text2audio_result = _run_direction(model, tokenizer, loader_text2audio, device, world_size)

        if rank == 0:
            out_dir = os.path.join(args.output_root, dataset)
            os.makedirs(out_dir, exist_ok=True)
            rerank_name = os.path.basename(args.rerank_model_name_or_path.rstrip("/"))
            save_path = os.path.join(out_dir, f"{rerank_name}_rerank_result_{effective_topk}.pth")
            torch.save(
                {
                    "retrieval_similarity_matrix": retrieval_obj["similarity_matrix"],
                    "retrieval_ids": retrieval_obj["ids"],
                    "retrieval_ids_txt": retrieval_obj["ids_txt"],
                    "audio2text_rerank_scores": audio2text_result["rerank_scores"],
                    "audio2text_rerank_index_list": audio2text_result["rerank_all_index"],
                    "audio2text_rerank_total_candidate_ids_list": audio2text_result["rerank_all_candidate_ids"],
                    "text2audio_rerank_scores": text2audio_result["rerank_scores"],
                    "text2audio_rerank_index_list": text2audio_result["rerank_all_index"],
                    "text2audio_rerank_total_candidate_ids_list": text2audio_result["rerank_all_candidate_ids"],
                    "topk": effective_topk,
                },
                save_path,
            )
            print(f"[test_rerank] Saved: {save_path}")
            print(
                "[test_rerank] "
                f"audio2text_queries={len(audio2text_result['rerank_all_index'])}, "
                f"audio2text_scores={len(audio2text_result['rerank_scores'])}, "
                f"text2audio_queries={len(text2audio_result['rerank_all_index'])}, "
                f"text2audio_scores={len(text2audio_result['rerank_scores'])}"
            )

        if world_size > 1:
            dist.barrier()

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
