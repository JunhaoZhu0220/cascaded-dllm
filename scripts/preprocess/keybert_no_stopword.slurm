#!/bin/bash

#SBATCH --job-name=data_preprocess
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --output=sbatch_log/keybert_no_stopword_proc/%x_%A_%a.out
#SBATCH --account=pr_133_tandon_advanced
#SBATCH --array=0-31

# Singularity path
ext3_path=/scratch/jz5770/discrete-diffusion-env/overlay-50G-10M.ext3
sif_path=/scratch/jz5770/discrete-diffusion-env/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

dataset_name=$1
dataset_mode=$2

# start running
singularity exec \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
    source /ext3/env.sh
    conda activate keybert
    cd /scratch/jz5770/cascaded-dllm
    python preprocess/data_preprocess_keybert_no_stopword.py --shard_idx ${SLURM_ARRAY_TASK_ID} --dataset_mode ${dataset_mode} --dataset_name ${dataset_name}
"