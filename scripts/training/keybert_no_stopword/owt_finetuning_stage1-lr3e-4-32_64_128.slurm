#!/bin/bash

#SBATCH --job-name=dd_s1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --output=sbatch_log/keybert_no_stopword_owt_stage1_finetuning-%j.out
#SBATCH --error=sbatch_log/keybert_no_stopword_owt_stage1_finetuning-%j.err
#SBATCH --account=pr_133_tandon_advanced
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jz5770@nyu.edu

# Singularity path
ext3_path=/scratch/jz5770/discrete-diffusion-env/overlay-50G-10M.ext3
sif_path=/scratch/jz5770/discrete-diffusion-env/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
``
# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
    source /ext3/env.sh
    conda activate dllm
    cd /scratch/jz5770/cascaded-dllm
    python train.py stage@_global_=stage1-lr3e-4-32_64_128 model=small pretrained.use_pretrained=true data.dataloader=keybert_no_stopword
"