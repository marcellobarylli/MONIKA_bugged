#!/bin/bash

#SBATCH --job-name=RUNPIG
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.err
#SBATCH -t 01:00:00
#SBATCH -N 5
#SBATCH --tasks-per-node 32 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user.example@gmail.com
#SBATCH --array=1-2

module purge
module load 2022
module load R/4.2.1-foss-2022a
source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate monika # replace with your own filepath to your conda env

cd /home/mbarylli/MONIKA # replace with your own filepath to MONIKA

# Define the parameters for each task
case $SLURM_ARRAY_TASK_ID in
    1) CMS="cms123"; DATA_TYPE="proteomics"; DATA_FILE="data/proteomics_for_pig_cms123.csv";;
    2) CMS="cmsALL"; DATA_TYPE="proteomics"; DATA_FILE="data/proteomics_for_pig_cmsALL.csv";;
    3) CMS="cms123"; DATA_TYPE="transcriptomics"; DATA_FILE="data/transcriptomics_for_pig_cms123.csv";;
    4) CMS="cmsALL"; DATA_TYPE="transcriptomics"; DATA_FILE="data/transcriptomics_for_pig_cmsALL.csv";;
esac

# Run the command with the specified parameters
mpirun python piglasso.py --p 154 --n 1337 --Q 1000 --b_perc 0.65 --llo 0.01 --lhi 1.5 --lamlen 500 --run_type $DATA_TYPE --cms $CMS --data_file $DATA_FILE
