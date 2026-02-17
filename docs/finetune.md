## Fine-tuning scripts
### 1. Prepare data for fine-tuning
To fine-tune AuroLA with LoRA, first you need to prepare the audio files and metadata:
Download [AudioCaps](https://audiocaps.github.io/) and the metadata, or use the metadata provided [here](datasets/annotations/audiocaps/train.json). You can also prepare your own dataset and metadata. An example is here:
```bash
{
    "video_id": "---1_cCGK4M_0",
    "caption": [
        "Idling car, train blows horn and passes"
    ]
}
```

### 2. Fine-tune AuroLA with LoRA
Download the pre-trained checkpoint and change MODEL_LOCAL_PATH, AUDIO_FOLDERS, METADATA_PATHS in [finetune_retrieval.sh](scripts/finetune_retrieval.sh), and run
```bash
bash scripts/finetune_retrieval.sh
```

### 3. Evaluate fine-tuned model 
Change ORIGINAL_MODEL_NAME_OR_PATH (/path/to/AuroLA) and MODEL_NAME_OR_PATH (/path/to/LoRA) in [test_retrieval.sh](scripts/test_retrieval.sh), and run
```bash
bash scripts/test_retrieval.sh
```
