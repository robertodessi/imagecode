#!/bin/bash

#SBATCH --job-name=tok2_rdmemb

#SBATCH --output=/private/home/rdessi/imagecode/baselines/clip/output/new_output/tokens2_randomemb_-%j.out
#SBATCH --error=/private/home/rdessi/imagecode/baselines/clip/output/new_output/tokens2_randomemb_-%j.out

#SBATCH --cpus-per-task=10

#SBATCH --partition=devlab

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --gpus-per-node=1

#SBATCH --time=72:00:00

conda activate egg


srun --label /private/home/rdessi/imagecode/baselines/clip/launchers/srun_tokens2_init_randomemb.sh
