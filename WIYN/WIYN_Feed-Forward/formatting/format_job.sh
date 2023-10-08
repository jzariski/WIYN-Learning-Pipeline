#!/bin/bash
#SBATCH --job-name=FORMAT
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=72:00:00
#SBATCH --account=kkratter
#SBATCH --partition=standard
#SBATCH --output=format.out
#SBATCH --error=format.err
 
# SLURM Inherits your environment. cd $SLURM_SUBMIT_DIR not needed
python /home/u5/jzariski/TelescopeNet-main/WIYN/WIYN_Feed-Forward/formatting/format_data.py

