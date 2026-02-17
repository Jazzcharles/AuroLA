import argparse
import json
import logging
import os
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoProcessor

from data.dataset_eval_retrieval import EvalDatasetRetrieval
from data.loader import PrefetchLoader
from model.qwen_omni import Qwen2_50OmniRetForConditionalGeneration
from utils.distributed import DistributedSampler_wopadding, all_gather_list, ddp_allgather
from utils.misc import (
    get_eval_env_paths,
    save_hparams,
    tensors_to_device,
    visualize_retrieval_results,
)
from utils.logger import SystemPromptWarningFilter


def _init_distributed(args):
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")


def add_embed_token(tokenizer, model, emb_token="<emb>"):
    emb_tokens = [emb_token]
    num_new_tokens = tokenizer.add_tokens(emb_tokens)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
    model.config.emb_token_ids = emb_token_ids
    if dist.get_rank() == 0:
        print("ADD emb_token_ids:", emb_token_ids)


def build_test_model(args):
    model = Qwen2_50OmniRetForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=None,
    )

    device = torch.device("cuda", args.local_rank)
    model.to(device)
    return model


def build_val_loaders(datasets, tokenizer, processor, batch_size, num_workers):
    val_loaders = {}
    per_rank_batch_size = max(1, batch_size // dist.get_world_size())
    for dataset in datasets:
        metadata, audio_dir = get_eval_env_paths(dataset)
        dataset_obj = EvalDatasetRetrieval(
            audio_dir=audio_dir,
            metadata=metadata,
            tokenizer=tokenizer,
            processor=processor,
        )
        if dataset_obj.use_sampler:
            sampler = DistributedSampler_wopadding(dataset_obj)
        else:
            sampler = SequentialSampler(dataset_obj)

        loader = DataLoader(
            dataset_obj,
            sampler=sampler,
            batch_size=per_rank_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=dataset_obj.collate_fn,
            drop_last=False,
            worker_init_fn=dataset_obj.worker_init_fn,
        )
        val_loaders[dataset] = PrefetchLoader(loader)
    return val_loaders

def _compute_metric_ret(score_matrix, ids, ids_txt, direction="text2audio"):
    assert score_matrix.shape == (len(ids_txt), len(ids))
    if direction == "text2audio":
        indice_matrix = score_matrix.sort(dim=-1, descending=True)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            gt_indice = ids.index(ids_txt[i])
            rank.append(indice_matrix[i].index(gt_indice))
        rank = torch.tensor(rank).to(score_matrix)
        r1 = (rank < 1).sum().item() / len(ids_txt)
        r5 = (rank < 5).sum().item() / len(ids_txt)
        r10 = (rank < 10).sum().item() / len(ids_txt)
        return {
            "text2audio_r1": round(r1 * 100, 1),
            "text2audio_r5": round(r5 * 100, 1),
            "text2audio_r10": round(r10 * 100, 1),
            "text2audio_ravg": round((r1 + r5 + r10) / 3 * 100, 1),
        }

    indice_matrix = score_matrix.sort(dim=0, descending=True)[1].permute(1, 0).tolist()
    rank = []
    for i in range(len(ids)):
        gt_indices = []
        for idx, id_ in enumerate(ids_txt):
            if id_ == ids[i]:
                gt_indices.append(idx)
        rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))
    rank = torch.tensor(rank).to(score_matrix)
    r1 = (rank < 1).sum().item() / len(ids)
    r5 = (rank < 5).sum().item() / len(ids)
    r10 = (rank < 10).sum().item() / len(ids)
    return {
        "audio2text_r1": round(r1 * 100, 1),
        "audio2text_r5": round(r5 * 100, 1),
        "audio2text_r10": round(r10 * 100, 1),
        "audio2text_ravg": round((r1 + r5 + r10) / 3 * 100, 1),
    }


def save_top50_retrieval_results(score_matrix, ids, ids_txt, output_dir, top_k=50):
    score_matrix = score_matrix.detach().cpu().float()
    n_text, n_audio = score_matrix.shape

    assert len(ids) == n_audio
    assert len(ids_txt) == n_text

    top_k = min(top_k, n_text, n_audio)

    text_uid = []
    local_counter = defaultdict(int)
    for audio_id in ids_txt:
        idx = local_counter[audio_id]
        text_uid.append(f"{audio_id}:{idx}")
        local_counter[audio_id] += 1

    a2t_scores, a2t_indices = torch.topk(score_matrix, k=top_k, dim=0)
    audio2text_results = []
    for audio_col in range(n_audio):
        indices = a2t_indices[:, audio_col].tolist()
        audio2text_results.append(
            {
                "audio_id": ids[audio_col],
                "top_text_ids": [text_uid[i] for i in indices],
            }
        )

    t2a_scores, t2a_indices = torch.topk(score_matrix, k=top_k, dim=1)
    text2audio_results = []
    for i in range(len(t2a_indices)):
        indices = t2a_indices[i].tolist()
        text2audio_results.append(
            {
                "text_id": ids_txt[i],
                "top_audio_ids": [ids[j] for j in indices],
            }
        )

    output = {"audio2text": audio2text_results, "text2audio": text2audio_results}
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rerank_metadata.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Saved retrieval results to {output_path}")


def parse_cli():
    parser = argparse.ArgumentParser(description="Unified retrieval testing")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--original_model_name_or_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--datasets", type=str, default="audiocaps")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--visualize_top_k", type=int, default=5)
    parser.add_argument("--visualize_max_samples", type=int, default=30)
    return parser.parse_args()


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

    original_model_path = args.original_model_name_or_path or args.model_name_or_path
    processor = AutoProcessor.from_pretrained(original_model_path)
    tokenizer = processor.tokenizer

    model = build_test_model(args)
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
            evaluation_dict = model.forward_inference(batch, "ret%ta", compute_loss=False)
            text_features.append(evaluation_dict["text_feature"])
            audio_features.append(evaluation_dict["audio_feature"])
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
            if args.visualise:
                visualize_retrieval_results(
                    score_matrix=score_matrix,
                    ids=ids,
                    ids_txt=ids_txt,
                    dataset=val_loader.loader.dataset,
                    output_dir=os.path.join(dataset_output_dir, "visualisation"),
                    top_k=args.visualize_top_k,
                    max_samples=args.visualize_max_samples,
                )
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
