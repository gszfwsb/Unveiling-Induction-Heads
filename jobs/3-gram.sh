#!/bin/bash
#SBATCH --job-name=2-gram_experiment # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00 
#SBATCH --output=/gpfs/radev/project/zhuoran_yang/sc3226/Unveiling-Induction-Heads/jobs/slurm_output/%j.out
#SBATCH --error=/gpfs/radev/project/zhuoran_yang/sc3226/Unveiling-Induction-Heads/jobs/slurm_output/%j.err
#SBATCH --requeue 

echo '-------------------------------'
cd ${SLURM_SUBMIT_DIR}
echo ${SLURM_SUBMIT_DIR}
echo Running on host $(hostname)
echo Time is $(date)
echo SLURM_NODES are $(echo ${SLURM_NODELIST})
echo '-------------------------------'
echo -e '\n\n'

export PROCS=${SLURM_CPUS_ON_NODE}

# Set the working directory
cd /gpfs/radev/project/zhuoran_yang/sc3226/Unveiling-Induction-Heads/RPE

module load miniconda
conda activate scgpt1

python3 /gpfs/radev/project/zhuoran_yang/sc3226/Unveiling-Induction-Heads/RPE/train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 3 --c-alpha 0.01 --device cuda --n-epochs 50000   --lr 1 --train-cmd CWa --low-degree 2
