#!/bin/bash

#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --error=/private/home/rdessi/imagecode/baselines/clip/output/new_output/tokens1_random_%j.err
#SBATCH --output=/private/home/rdessi/imagecode/baselines/clip/output/new_output/tokens1_random_%j.out
#SBATCH --job-name=tok1_rdm
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=devlab
#SBATCH --time=72:00:00

srun /private/home/rdessi/imagecode/baselines/clip/launchers/srun_tokens1_init_random.sh
