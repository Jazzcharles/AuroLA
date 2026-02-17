import os
import json
import re
from typing import Tuple

import torch
import numpy as np

from utils.audio_utils import safe_librosa_load

def get_eval_env_paths(dataset: str) -> Tuple[str, str]:
    env_key = dataset.replace("-", "_").upper()
    metadata = os.environ.get(f"{env_key}_METADATA", "")
    audio_dir = os.environ.get(f"{env_key}_AUDIO_DIR", "")
    assert metadata, f"Missing env: {env_key}_METADATA"
    assert audio_dir, f"Missing env: {env_key}_AUDIO_DIR"
    return metadata, audio_dir


def save_hparams(output_dir: str, args, filename: str = "hps.json") -> None:
    os.makedirs(os.path.join(output_dir, "log"), exist_ok=True)
    payload = vars(args).copy() if not isinstance(args, dict) else args.copy()
    with open(os.path.join(output_dir, "log", filename), "w") as f:
        json.dump(payload, f, indent=4)


def tensors_to_device(
    data: dict,
    device: torch.device,
    dtype: torch.dtype,
    float_tensor_keys=None,
    float_tensor_key_substrings=None,
):
    float_tensor_keys = set(float_tensor_keys or [])
    float_tensor_key_substrings = tuple(float_tensor_key_substrings or [])

    for key in data.keys():
        value = data[key]
        if not isinstance(value, torch.Tensor):
            continue
        if key in float_tensor_keys or any(substr in key for substr in float_tensor_key_substrings):
            data[key] = value.to(device).to(dtype)
        else:
            data[key] = value.to(device)
    return data


def _extract_id_from_anno(anno: dict):
    for key in ["video_id", "image_id", "image", "id"]:
        if key in anno:
            return anno[key]
    return None


def _resolve_audio_path(audio_dir: str, audio_id) -> str:
    base = os.path.join(audio_dir, str(audio_id))
    if os.path.exists(base):
        return base

    known_extensions = [".wav", ".flac", ".mp3", ".mkv"]
    has_ext = any(base.lower().endswith(ext) for ext in known_extensions)
    stem = base
    if has_ext:
        for ext in known_extensions:
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break

    for ext in ["wav", "flac", "mp3", "mkv"]:
        candidate = f"{stem}.{ext}"
        if os.path.exists(candidate):
            return candidate
    return base


def _sanitize_filename(name: str, max_len: int = 80) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name))
    return safe[:max_len] if len(safe) > max_len else safe


def _caption_list(anno: dict):
    raw = anno.get("desc", anno.get("caption", ""))
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    return [str(raw)]


def _resolve_anno_index(index_or_id, annos):
    try:
        idx = int(index_or_id)
        if 0 <= idx < len(annos):
            return idx
    except Exception:
        pass

    target = str(index_or_id)
    for i, anno in enumerate(annos):
        sample_id = _extract_id_from_anno(anno)
        if sample_id is not None and str(sample_id) == target:
            return i
    return None


def visualize_retrieval_results(
    score_matrix,
    ids,
    ids_txt,
    dataset,
    output_dir,
    top_k: int = 5,
    max_samples: int = 30,
):
    import matplotlib.pyplot as plt

    score_matrix = score_matrix.detach().cpu().float()
    ids = list(ids)
    ids_txt = list(ids_txt)
    top_k = max(1, min(top_k, score_matrix.shape[0], score_matrix.shape[1]))

    os.makedirs(output_dir, exist_ok=True)
    a2t_dir = os.path.join(output_dir, "a2t")
    t2a_dir = os.path.join(output_dir, "t2a")
    os.makedirs(a2t_dir, exist_ok=True)
    os.makedirs(t2a_dir, exist_ok=True)

    annos = dataset.annos
    audio_dir = dataset.audio_dir

    # Build row-wise text strings aligned with ids_txt order (ids_txt is anno index).
    text_counters = {}
    text_rows = []
    for row_id in ids_txt:
        row_idx = _resolve_anno_index(row_id, annos)
        row_key = str(row_idx) if row_idx is not None else str(row_id)
        idx = text_counters.get(row_key, 0)
        text_counters[row_key] = idx + 1
        if row_idx is None:
            cands = [""]
        else:
            cands = _caption_list(annos[row_idx])
        caption = cands[idx] if idx < len(cands) else cands[0]
        text_rows.append(caption)

    max_audio_queries = min(len(ids), max_samples)
    max_text_queries = min(len(ids_txt), max_samples)

    # Audio-to-Text: waveform + top-k text.
    for col, audio_index in enumerate(ids[:max_audio_queries]):
        anno_idx = _resolve_anno_index(audio_index, annos)
        if anno_idx is not None:
            anno = annos[anno_idx]
            real_audio_id = _extract_id_from_anno(anno)
            audio_path = _resolve_audio_path(audio_dir, real_audio_id)
        else:
            real_audio_id = audio_index
            audio_path = _resolve_audio_path(audio_dir, audio_index)
        y, sr = safe_librosa_load(audio_path, sr=16000)
        y = np.asarray(y, dtype=np.float32)
        t = np.arange(len(y)) / float(sr)

        top_scores, top_rows = torch.topk(score_matrix[:, col], k=top_k, dim=0)
        top_rows = top_rows.tolist()
        top_scores = top_scores.tolist()

        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        axes[0].plot(t, y, linewidth=0.8, color="#1f77b4")
        axes[0].set_title(f"Audio idx={audio_index}, id={real_audio_id}")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")

        lines = []
        for rank, (r, s) in enumerate(zip(top_rows, top_scores), start=1):
            lines.append(f"{rank}. ({s:.4f}) {text_rows[r]}")
        axes[1].axis("off")
        axes[1].text(0.0, 1.0, "\n\n".join(lines), va="top", fontsize=10, wrap=True)
        axes[1].set_title(f"Top-{top_k} Text")

        fig.tight_layout()
        save_name = _sanitize_filename(f"{col:06d}_{audio_index}.png")
        fig.savefig(os.path.join(a2t_dir, save_name), dpi=120)
        plt.close(fig)

    # Text-to-Audio: text + top-k audios with paired/non-paired color.
    for row, text_index in enumerate(ids_txt[:max_text_queries]):
        text_query = text_rows[row]
        gt_audio_index = str(text_index)

        top_scores, top_cols = torch.topk(score_matrix[row, :], k=top_k, dim=0)
        top_cols = top_cols.tolist()
        top_scores = top_scores.tolist()

        fig, axes = plt.subplots(top_k + 1, 1, figsize=(14, 2.5 * (top_k + 1)))
        axes[0].axis("off")
        axes[0].text(
            0.0,
            0.85,
            f"Text Query idx={text_index}:\n{text_query}",
            va="top",
            fontsize=12,
            wrap=True,
        )

        for i, (col_idx, score) in enumerate(zip(top_cols, top_scores), start=1):
            cand_audio_index = ids[col_idx]
            cand_anno_idx = _resolve_anno_index(cand_audio_index, annos)
            if cand_anno_idx is not None:
                cand_real_audio_id = _extract_id_from_anno(annos[cand_anno_idx])
                cand_path = _resolve_audio_path(audio_dir, cand_real_audio_id)
            else:
                cand_real_audio_id = cand_audio_index
                cand_path = _resolve_audio_path(audio_dir, cand_audio_index)
            y, sr = safe_librosa_load(cand_path, sr=16000)
            y = np.asarray(y, dtype=np.float32)
            t = np.arange(len(y)) / float(sr)

            ax = axes[i]
            ax.plot(t, y, linewidth=0.8, color="#1f77b4")
            ax.set_title(
                f"Top-{i} Audio idx={cand_audio_index}, id={cand_real_audio_id} | score={score:.4f}",
                fontsize=10,
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amp")

            color = "green" if str(cand_audio_index) == gt_audio_index else "red"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

        fig.tight_layout()
        save_name = _sanitize_filename(f"{row:06d}_{text_index}.png")
        fig.savefig(os.path.join(t2a_dir, save_name), dpi=120)
        plt.close(fig)
