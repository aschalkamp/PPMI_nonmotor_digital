#!/bin/bash --login
#SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=SchalkampA@cardiff.ac.uk # Your email address
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48gb
#SBATCH --time=05:20:00 # Time limit hh:mm:ss
#SBATCH -o log/tsfresh/%x-%A-%a.out
#SBATCH -e log/tsfresh/%x-%A-%a.err
#SBATCH --job-name=ppmi_tsfresh # Descriptive job name
#SBATCH --partition=c_compute_dri1 # Use a serial partition 24 cores/7days

module load anaconda
source activate timeseries

modalities=( 'ambulatory' 'sleepmetrics2' 'pulserate' 'prv' )
mod=${modalities[${SLURM_ARRAY_TASK_ID}]}
python PPMI_DataPreparation/studywatch/scripts/extract_features.py $mod ''