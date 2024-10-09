#!/bin/bash

#SBATCH --job-name=ARRAY_job
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.err
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH --tasks-per-node 64 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marcelloibz@gmail.com
#SBATCH --array=1-420%5

module purge
module load 2022
module load R/4.2.1-foss-2022a
source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate monika

cd /home/user/MONIKA/src/

# Define arrays for p, n, and b_perc values
p_values=(150)
n_values=(50 100 300 500 700 900 1100)
b_perc_values=(0.6 0.65 0.7)

# Fixed values
FP_FN_VAL=0.0
Q_VAL=1000

# Recalculate the total number of combinations
total_combinations=420

# Calculate the index for p, n, b_perc, and seed based on SLURM_ARRAY_TASK_ID
index=$((SLURM_ARRAY_TASK_ID - 1))
seed_index=$((index / 21))
p_index=0  # Only one p_value
n_index=$(((index % 21) / 3))
b_perc_index=$((index % 3))

# Get the values for p, n, b_perc, and seed
P_VAL=${p_values[$p_index]}
N_VAL=${n_values[$n_index]}
B_PERC_VAL=${b_perc_values[$b_perc_index]}
SEED_VAL=$((seed_index + 11))

# Run the Python script with the mapped values of --p, --n, --b_perc, --Q, and --seed
mpirun python piglasso.py --p $P_VAL --n $N_VAL --fp_fn $FP_FN_VAL --Q $Q_VAL --b_perc $B_PERC_VAL --llo 0.01 --lhi 0.5 --lamlen 100 --seed $SEED_VAL --dens 0.05

