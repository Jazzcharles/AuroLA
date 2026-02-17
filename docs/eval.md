## Evaluate HF AuroLA model
Please follow the steps to test AuroLA on AudioCaps/Clotho/Auto-ACD:
### 1. Prepare data for evaluation

Download [AudioCaps](https://audiocaps.github.io/) and [Clotho](https://zenodo.org/records/4783391) datasets and their metadata.
All audio should be placed under AUDIOCAPS_AUDIO_DIR and CLOTHO_AUDIO_DIR.
We have also prepared the metadata [here](datasets/annotations/). An example is here:
```bash
{
    "video_id": "--0w1YA1Hm4_30",
    "caption": [
        "A vehicle driving as a man and woman are talking and laughing",
        "Men speak and laugh with humming of an engine",
        "High pitched speaking and laughing",
        "Humming of an engine with a woman and men speaking",
        "People talking with the dull roar of a vehicle on the road"
    ]
}
```
### 2. Evaluate AuroLA 
Change the MODEL_NAME_OR_PATH, AUDIOCAPS_METADATA, AUDIOCAPS_AUDIO_DIR and OUTPUT_DIR in test_retrieval_hfmodel.sh, and run:
```bash
bash scripts/test_retrieval_hfmodel.sh
```
This will save the prediction results and (generated metadata, similarity matrix) for re-ranking to OUTPUT_DIR.

### 3. (Optional) Evaluate AuroLA-rerank
Change the RETRIEVAL_ROOT (the location of generated metadata, similarity matrix), RERANK_MODEL_NAME_OR_PATH, AUDIOCAPS_METADATA, AUDIOCAPS_AUDIO_DIR and OUTPUT_DIR in test_rerank_hfmodel.sh, and run:
```bash
bash scripts/test_rerank_hfmodel.sh
```
Change the RERANK_RESULT, ALPHA_A2T and ALPHA_T2A. Then, get the final reranking score:
```bash
bash scripts/get_rerank_score.sh
```
