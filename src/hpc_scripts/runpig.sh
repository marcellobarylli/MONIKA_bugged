#!/bin/bash
#SBATCH --job-name=RUNPIG
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.err
#SBATCH -t 00:10:00
#SBATCH -N 5
#SBATCH --tasks-per-node 32 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user.example@gmail.com
#SBATCH --array=1-4

module purge
module load 2022
module load R/4.2.1-foss-2022a
source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate monika # replace with your own filepath to your conda env
cd /home/mbarylli/MONIKA/src # replace with your own filepath to MONIKA

# Define the parameters for each task
case $SLURM_ARRAY_TASK_ID in
    1) RUN_TYPE="proteomics"; CMS="cms123";;
    2) RUN_TYPE="proteomics"; CMS="cmsALL";;
    3) RUN_TYPE="transcriptomics"; CMS="cms123";;
    4) RUN_TYPE="transcriptomics"; CMS="cmsALL";;
esac

# Set common parameters
P=154
N=1337
Q=1000
B_PERC=0.65
LLO=0.01
LHI=1.5
LAMLEN=500
PRIOR_CONF=90

# Run the command with the specified parameters
mpirun python piglasso.py \
    --p $P \
    --n $N \
    --Q $Q \
    --b_perc $B_PERC \
    --llo $LLO \
    --lhi $LHI \
    --lamlen $LAMLEN \
    --run_type $RUN_TYPE \
    --cms $CMS \
    --prior_conf $PRIOR_CONF
