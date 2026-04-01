#!/bin/bash
#SBATCH -J GANO        # Job name
#SBATCH -o GANO.o%j    # Name of stdout output file
#SBATCH -p gh         # Queue (partition) name
#SBATCH -N 1        # Total # of nodes
#SBATCH -n 1        # Total # of mpi tasks
#SBATCH -t 00:30:00        # Run time (hh:mm:ss)
#SBATCH -A EAR24026        # Allocation name (req'd if you have more than 1)

module load gcc cuda
module load python3

source /work/10845/kaichengz/vista/.bashrc
conda activate /work/10845/kaichengz/gmprocess
python run_GANO.py
