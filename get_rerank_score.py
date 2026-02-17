import argparse
import torch
from test_retrieval import _compute_metric_ret

def merge_rerank_result_to_retrieval_simlarity(
    retrieval_similarity_matrix,  # (N_text, N_audio)
    retrieval_ids,                 # (N_audio,)
    retrieval_ids_txt,             # (N_text,)
    rerank_scores,                 # (N_queries * TOPK,)
    rerank_index_list,             # audio2text: audio_col_idx
                                   # text2audio: "text_id:text_sub_id"
    rerank_total_text_ids_list,    # audio2text: "text_id:text_sub_id"
                                   # text2audio: audio_id
    topk=50,
    alpha=0.5,
    direction="audio2text"
):
    device = retrieval_similarity_matrix.device
    dtype = retrieval_similarity_matrix.dtype

    # -------- Step 1: audio_id -> col index --------
    audio_id_to_col = {
        int(audio_id): idx for idx, audio_id in enumerate(retrieval_ids)
    }

    # -------- Step 2: (audio_id, sub_id) -> text row index --------
    text_row_map = {}
    counter = {}
    for row_idx, audio_id in enumerate(retrieval_ids_txt):
        audio_id = int(audio_id)
        sub_id = counter.get(audio_id, 0)
        text_row_map[(audio_id, sub_id)] = row_idx
        counter[audio_id] = sub_id + 1

    # -------- Step 3: clone similarity --------
    new_similarity = retrieval_similarity_matrix.clone()
    rerank_scores = torch.as_tensor(rerank_scores, device=device, dtype=dtype)

    # -------- Step 4: merge --------
    ptr = 0

    for q_idx, q_index in enumerate(rerank_index_list):

        if direction == "audio2text":
            # query 是 audio
            audio_col_idx = int(q_index)
            audio_id = int(retrieval_ids[audio_col_idx])

            for _ in range(topk):
                score_rank = rerank_scores[ptr]
                text_id_str = rerank_total_text_ids_list[ptr]
                ptr += 1

                text_audio_id, text_sub_id = map(int, text_id_str.split(":"))

                if text_audio_id != audio_id:
                    continue

                key = (text_audio_id, text_sub_id)
                if key not in text_row_map:
                    continue

                text_row_idx = text_row_map[key]
                score_ret = new_similarity[text_row_idx, audio_col_idx]

                new_similarity[text_row_idx, audio_col_idx] = (
                    score_ret + alpha * score_rank
                )

        elif direction == "text2audio":
            # query 是 text
            text_audio_id, text_sub_id = map(int, q_index.split(":"))
            key = (text_audio_id, text_sub_id)
            if key not in text_row_map:
                ptr += topk
                continue

            text_row_idx = text_row_map[key]

            for _ in range(topk):
                score_rank = rerank_scores[ptr]
                audio_id = int(rerank_total_text_ids_list[ptr])
                ptr += 1

                if audio_id not in audio_id_to_col:
                    continue

                audio_col_idx = audio_id_to_col[audio_id]
                score_ret = new_similarity[text_row_idx, audio_col_idx]

                new_similarity[text_row_idx, audio_col_idx] = (
                    score_ret + alpha * score_rank
                )

        else:
            raise ValueError(f"Unknown direction: {direction}")

    return new_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search alpha for rerank score merging.")
    parser.add_argument("--rerank_result", type=str, required=True, help="Path to *_rerank_result_*.pth")
    parser.add_argument(
        "--topk",
        type=int,
        default=-1,
        help="Top-k used in rerank result; if -1, use value in file or fallback to 50.",
    )
    parser.add_argument("--alpha_a2t", type=float, default=1.0, help="Alpha for audio2text merging.")
    parser.add_argument("--alpha_t2a", type=float, default=1.0, help="Alpha for text2audio merging.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src_dir = args.rerank_result
    t = torch.load(src_dir, weights_only=False)
    print(t.keys())

    topk = args.topk if args.topk > 0 else int(t.get("topk", 50))
    retrieval_similarity_matrix = t['retrieval_similarity_matrix']
    retrieval_ids = t['retrieval_ids']
    retrieval_ids_txt = t['retrieval_ids_txt']
    audio2text_rerank_scores = t['audio2text_rerank_scores']
    audio2text_rerank_index_list = t['audio2text_rerank_index_list']
    audio2text_rerank_total_candidate_ids_list = t['audio2text_rerank_total_candidate_ids_list']

    text2audio_rerank_scores = t['text2audio_rerank_scores']
    text2audio_rerank_index_list = t['text2audio_rerank_index_list']
    text2audio_rerank_total_candidate_ids_list = t['text2audio_rerank_total_candidate_ids_list']

    print('retrieval_similarity_matrix.shape:', retrieval_similarity_matrix.shape)
    print('len(retrieval_ids):', len(retrieval_ids))
    print('len(retrieval_ids_txt):', len(retrieval_ids_txt))
    print()
    print('audio2text_rerank_scores.shape:', audio2text_rerank_scores.shape)
    print('len(audio2text_rerank_index_list):', len(audio2text_rerank_index_list))
    print('len(audio2text_rerank_total_candidate_ids_list):', len(audio2text_rerank_total_candidate_ids_list))
    print()
    print('text2audio_rerank_scores.shape:', text2audio_rerank_scores.shape)
    print('len(text2audio_rerank_index_list):', len(text2audio_rerank_index_list))
    print('len(text2audio_rerank_total_candidate_ids_list):', len(text2audio_rerank_total_candidate_ids_list))

    new_similarity = merge_rerank_result_to_retrieval_simlarity(
        retrieval_similarity_matrix, retrieval_ids, retrieval_ids_txt, audio2text_rerank_scores, audio2text_rerank_index_list, audio2text_rerank_total_candidate_ids_list,
        alpha=args.alpha_a2t, direction='audio2text', topk=topk,
    )
    log = _compute_metric_ret(new_similarity, retrieval_ids, retrieval_ids_txt, direction='text2audio')
    log2 = _compute_metric_ret(new_similarity, retrieval_ids, retrieval_ids_txt, direction='audio2text')
    log.update(log2)
    # print(f"alpha: {args.alpha_a2t}, log: {log}")


    print('-'*100)
    print('text2audio, based on best audio2text similarity matrix')
    new_similarity = merge_rerank_result_to_retrieval_simlarity(
        new_similarity, retrieval_ids, retrieval_ids_txt, text2audio_rerank_scores, text2audio_rerank_index_list, text2audio_rerank_total_candidate_ids_list,
        alpha=args.alpha_t2a, direction='text2audio'
    )
    log = _compute_metric_ret(new_similarity, retrieval_ids, retrieval_ids_txt, direction='text2audio')
    log2 = _compute_metric_ret(new_similarity, retrieval_ids, retrieval_ids_txt, direction='audio2text')
    log.update(log2)
    print("Final rerank score merging result:")
    print(f"alpha_a2t: {args.alpha_a2t}, alpha_t2a: {args.alpha_t2a}, log: {log}")



### best ones: (47.1, 60.5), (27.7, 37.6) ###
# src_dir = '/mnt/vision_user/xjl/avbench/qwen/checkpoints/qwen2_5-omni-7b_HybridNCE_BEST_KMeans/downstream/retrieval-audiocaps/rerank_result.pth'
# src_dir = '/mnt/vision_user/xjl/avbench/qwen/checkpoints/qwen2_5-omni-7b_HybridNCE_BEST_KMeans/downstream/retrieval-clothov2/rerank_result.pth'
### 也还不错: (46.9, 59.5), (27.6, 37.1) ###
# src_dir = '/mnt/vision_user/xjl/avbench/qwen/checkpoints/qwen2_5-omni-7b_HybridNCE_BEST_KMeans/downstream/retrieval-audiocaps/qwen2_5-omni-7b_Rerank_Debug_Pointwise_ReverseDirection_rerank_result.pth' 
# src_dir = '/mnt/vision_user/xjl/avbench/qwen/checkpoints/qwen2_5-omni-7b_HybridNCE_BEST_KMeans/downstream/retrieval-clothov2/qwen2_5-omni-7b_Rerank_Debug_Pointwise_ReverseDirection_rerank_result.pth'
### failed ones ###
# src_dir = '/mnt/vision_user/xjl/avbench/qwen/checkpoints/qwen2_5-omni-7b_HybridNCE_BEST_KMeans/downstream/retrieval-audiocaps/qwen2_5-omni-7b_Rerank_Pointwise_HybridNCE_RetrievalGenerativeBased_rerank_result.pth' 
