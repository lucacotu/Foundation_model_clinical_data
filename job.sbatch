#!/bin/bash
#SBATCH --job-name=surv
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=luca.cotugno@studio.unibo.it 
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=31G
#SBATCH --partition=rtx2080
#SBATCH --output=log_slurm/out_%j.log
#SBATCH --chdir=/scratch.hpc/luca.cotugno/Foundation_model_clinical_data 
#SBATCH --gres=gpu:1

.venv/bin/python3 "$@"
