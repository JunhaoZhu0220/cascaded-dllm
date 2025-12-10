# Cascaded Discrete Diffusion Language Model (Cascaded-DLLM)

This repository implements a multi-stage discrete diffusion model for text generation with support for fine-tuning from pretrained models. The model generates text progressively through three stages, refining the output at each level.

## Environment Setup

For preprocessing the data, use the conda environment `keybert`:
```bash
conda env create -f environment_keybert.yml
conda activate keybert
python -m spacy download en_core_web_md
```

For training the cascaded DLLM, use the conda environment `dllm`:
```bash
conda env create -f environment_dllm.yml
conda activate dllm
```

## Data Preprocessing

The `./preprocess` folder contains two distinct preprocessing methods, identified by the suffixes `constituency` and `keybert_no_stopword`. Each method employs a different strategy to generate the prefix and target sequences.

### HPC Users (Slurm)

For HPC users that support Slurm, you can refer to the provided scripts in `./scripts/preprocess` to launch the preprocessing jobs. The scripts take the dataset name and mode as command-line arguments.

Example: Using constituency method to preprocess the training set of openwebtext & validation set of wikitext103:
```bash
sbatch scripts/preprocess/constituency.slurm openwebtext train
sbatch scripts/preprocess/constituency.slurm wikitext103 validation
```

### Non-HPC Users

For users without Slurm access, you can process the shards sequentially using a bash loop:

Example: Using constituency method to preprocess the training set of openwebtext:
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

## Training

### Training Command Structure

The configuration name should correspond to one of the stage config files in `configs/stage/`:

```bash
python train.py stage@_global_=<stage_config_name> model=<model_type> [pretrained.use_pretrained=<bool>]
```

**Parameters:**
- `<stage_config_name>`: Name of the stage configuration file (without `.yaml` extension), e.g., `stage1-lr3e-4-32_64_128`, `stage2-lr5e-4-32_64_128`, etc.
- `<model_type>`: Model architecture, options: `small`, `medium`
- `[pretrained.use_pretrained=<bool>]`: Optional flag to load pretrained weights (default: `false`)

### Available Stage Configurations

Available stage configurations can be found in `configs/stage/`:
- Stage 1: `stage1-lr*-*_*_*` (various learning rates and batch configurations)
- Stage 2: `stage2-lr*-*_*_*`
- Stage 3: `stage3-lr*-*_*_*`

Examples: `stage1-lr3e-4-32_64_128`, `stage1-lr5e-4-32_64_128`, `stage2-lr3e-4-32_64_128`, etc.

### Fine-tuning Configuration

#### From Scratch Training (`pretrained.use_pretrained=false`)
- Default behavior
- Initializes model with random weights
- Full training from beginning
- WandB experiment name: `{dataset}-{model}-stage{N}`

#### Fine-tuning Mode (`pretrained.use_pretrained=true`)
- Loads pretrained weights from specified model path
- Default model: `"louaaron/sedd-small"`
- Continues training from pretrained checkpoint
- WandB experiment name: `{dataset}-{model}-stage{N}-finetuning`

### Training Examples

#### Training from Scratch - Stage 1
```bash
python train.py stage@_global_=stage1-lr3e-4-32_64_128 model=small
```

#### Training from Scratch - Stage 2
```bash
python train.py stage@_global_=stage2-lr3e-4-32_64_128 model=small
```

#### Training from Scratch - Stage 3
```bash
python train.py stage@_global_=stage3-lr3e-4-32_64_128 model=small
```

#### Fine-tuning with Pre-trained Weights
```bash
# Stage 1 with pretrained weights
python train.py stage@_global_=stage1-lr3e-4-32_64_128 model=small pretrained.use_pretrained=true

# Stage 2 with pretrained weights
python train.py stage@_global_=stage2-lr3e-4-32_64_128 model=small pretrained.use_pretrained=true

# Stage 3 with pretrained weights
python train.py stage@_global_=stage3-lr3e-4-32_64_128 model=small pretrained.use_pretrained=true
```

#### Using Different Stage Configurations
```bash
# Train with different learning rate
python train.py stage@_global_=stage1-lr5e-4-32_64_128 model=small

# Train with medium model size
python train.py stage@_global_=stage2-lr3e-4-32_64_128 model=medium
```

#### Resume Training from Checkpoint
```bash
python train.py stage@_global_=stage2-lr3e-4-32_64_128 model=small load_dir=<path_to_checkpoint>
```

## Text Generation

To generate text samples using a trained cascaded model, use the sampling script:

```bash
python run_sample_cascade.py \
  --stage1_model_path <path_to_stage1_checkpoint> \
  --stage2_model_path <path_to_stage2_checkpoint> \
  --stage3_model_path <path_to_stage3_checkpoint> \
  --batch_size 32 \
  --steps 1024 \
  --predictor analytic \
  --output_dir ./sampling_outputs \
  [--eval_perplexity]
```

### Sampling Configuration Options

- **`--stage1_model_path`**: Path to the trained Stage 1 model checkpoint
- **`--stage2_model_path`**: Path to the trained Stage 2 model checkpoint
- **`--stage3_model_path`**: Path to the trained Stage 3 model checkpoint
- **`--batch_size`**: Number of samples to generate in parallel (default: 32)
- **`--steps`**: Number of diffusion steps for generation (default: 1024). Higher values may produce better quality samples but take longer.
- **`--predictor`**: Prediction method - 'analytic' or other sampling strategies
- **`--output_dir`**: Directory to save generated samples (default: `./sampling_outputs`)
- **`--eval_perplexity`**: Optional flag to evaluate perplexity of generated samples using a language model

### Output Format

Generated samples are saved as separate text files for each stage:
- `sample_stage1.txt`: Output from Stage 1 (default 32 tokens)
- `sample_stage2.txt`: Output from Stage 2 (default 64 tokens)
- `sample_stage3.txt`: Final output from Stage 3 (default 128 tokens)

Each sample is separated by a line of "=" characters (80 characters wide).

## Configuration

### Main Configuration File

The `configs/cascade_config.yaml` file controls the overall training setup:

```yaml
defaults:
  - stage@_global_: stage1
  - model: small
  - _self_  

data:
  dataloader: keybert_no_stopword  # Options: 'constituency', 'keybert'

pretrained:
  use_pretrained: false
  model_path: "louaaron/sedd-small"
```

### Model Configurations

Available model types in `configs/model/`:

- **`small.yaml`**: Small model configuration
  - Hidden size: 768
  - Number of attention heads: 12
  - Number of transformer blocks: 12
  - Conditional embedding dimension: 128

- **`medium.yaml`**: Medium model configuration
  - Hidden size: 1024
  - Number of attention heads: 16
  - Number of transformer blocks: 24
  - Conditional embedding dimension: 128