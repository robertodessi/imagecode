#!/bin/bash

#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/private/home/rdessi/imagecode/baselines/clip/output/new_output/pretr_tokens1_randomemb_%j.out
#SBATCH --error=/private/home/rdessi/imagecode/baselines/clip/output/new_output/pretr_tokens1_randomemb_%j.err
#SBATCH --job-name=pret_tok1_rdmemb
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=devlab
#SBATCH --time=72:00:00

srun /private/home/rdessi/imagecode/baselines/clip/launchers/srun_tokens1_init_randomemb.sh
