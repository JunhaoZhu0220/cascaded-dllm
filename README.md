# Introduction [TBD]

This repository implements a multi-stage discrete diffusion model for text generation with support for fine-tuning from pretrained models.

## Environment Setup
For preprocessing the data, use the conda environment `keybert`:
```bash
conda env create -f environment_keybert.yml
conda activate keybert
python -m spacy download en_core_web_md
```
For training the cascaded dllm, use the conda environment `dllm`:
```bash
conda env create -f environment_dllm.yml
conda activate dllm
```

## Data Preprocessing
The `./preprocess` folder, contains three distinct preprocessing methods, identified by the suffixes `constituency`, `keybert_no_stopword`, and `keybert`. Each method employs a different strategy to generate the prefix and target sequences.

### HPC Users (Slurm)
For HPC Users that support Slurm, you can refer to the provided scripts in `./scripts/preprocess` to launch the preprocessing jobs. The scripts take the dataset name and mode as command-line arguments.

e.g. Using constituency method to preprocess the training set of openwebtext & validation set of wikitext103:
```bash
sbatch scripts/preprocess/constituency.slurm openwebtext train
sbatch scripts/preprocess/constituency.slurm wikitext103 validation
```

### Non-HPC Users
For users without Slurm access, you can process the shards sequentially using a bash loop:

e.g. Using constituency method to preprocess the training set of openwebtext:

```bash
for i in {0..31}; do
  python preprocess/data_preprocess_constituency.py \
    --shard_idx $i \
    --dataset_mode train \
    --dataset_name openwebtext \
    --n_process 8 \
    --num_shards 32
done
```


### Implementation Note
Due to potential deadlock issues with [`spaCy`](https://github.com/explosion/spaCy)'s multithreading, this repository adopts a brute-force shard-based approach. The dataset is split into multiple shards, and each shard is preprocessed independently to ensure stability.

## Training Command Structure

```bash
python train.py stage@_global_=<stage> model=<model_type> [pretrained.use_pretrained=<bool>]
```


## Model Configuration (`model=`)

### Standard Model (`model=small`)
- Uses standard discrete diffusion training
- No prefix shuffling during training

### Shuffle Model (`model=small_shuffle`)
- Enables prefix position shuffling during training
- Randomly shuffles the positions of prefix tokens using `torch.randperm()`

## Fine-tuning Configuration (`pretrained.use_pretrained`)

### From Scratch Training (`pretrained.use_pretrained=false`)
- Default behavior
- Initializes model with random weights
- Full training from beginning
- WandB experiment name: `{dataset}-{model}-stage{N}`

### Fine-tuning Mode (`pretrained.use_pretrained=true`)
- Loads pretrained weights from specified model path
- Default model: `"louaaron/sedd-small"`
- Continues training from pretrained checkpoint
- WandB experiment name: `{dataset}-{model}-stage{N}-finetuning`

## Example Commands

### Training from Scratch

### Stage 1
```bash
python train.py stage@_global_=stage1 model=small_shuffle
```

### Stage 2
```bash
python train.py stage@_global_=stage2 model=small_shuffle
```

### Stage 3
```bash
python train.py stage@_global_=stage3 model=small_shuffle
```

### Finetuning with Pre-trained Weight
### Stage 1
```bash
python train.py stage@_global_=stage1 model=small_shuffle pretrained.use_pretrained=true
```

### Stage 2
```bash
python train.py stage@_global_=stage2 model=small_shuffle pretrained.use_pretrained=true
```

### Stage 3
```bash
python train.py stage@_global_=stage3 model=small_shuffle pretrained.use_pretrained=true
```