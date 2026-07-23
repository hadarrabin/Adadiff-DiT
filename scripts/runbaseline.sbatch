#!/bin/bash
#SBATCH --job-name=mri_something
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
#SBATCH --account=acct-ugproj

#source /path/to/venv/bin/activate
source /ugproj/sipl-prj10848/ugproj/sipl-prj10848/etc/profile.d/conda.sh  # Ensure correct path to conda.sh
source ~/.bashrc
conda init
conda activate mri_env


# Run the training script

cd /ugproj/sipl-prj10848/Idan/fastMRI_adadiff
python -u run_literature_baseline.py --modality T1

# Deactivate virtual environment
conda deactivate


