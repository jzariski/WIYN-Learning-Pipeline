#!/bin/bash
#SBATCH --job-name=TRAIN
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=72:00:00
#SBATCH --account=kkratter
#SBATCH --partition=standard
#SBATCH --output=train_LSTM.out
#SBATCH --error=train_LSTM.err
 
# SLURM Inherits your environment. cd $SLURM_SUBMIT_DIR not needed
python /home/u5/jzariski/TelescopeNet-main/WIYN/WIYN_Recurrent/train_wiyn_recurrent/Architectures/RecurrentNN.py


