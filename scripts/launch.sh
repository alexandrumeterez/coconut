#!/bin/bash
#SBATCH --job-name=coconut
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --output=/n/netscratch/kempner_sham_lab/Lab/ameterez/logs/%A_%a.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mem=150GB
#SBATCH --array=0-2

source ~/.bashrc
conda deactivate
conda activate coconut

CONFIG=args/gsm_coconut.yaml
python slurm_run.py sweep_config=$CONFIG