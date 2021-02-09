#!/bin/bash -l

#SBATCH -A brain_lab
#SBATCH --ntasks=4
#SBATCH -t 48:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --output=log/slurm/metro-%A.log
#SBATCH -J metro-ipg
#SBATCH --exclude=udc-ba25-[23,27,28],udc-ba26-[23,24,25],udc-ba27-[23,24]

module purge
module load singularity

singularity exec --nv /project/Driver_in_the_loop/pl-1.0.8_latest.sif python -m wandb.cli agent uvrhzn2n