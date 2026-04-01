#!/bin/bash
#SBATCH -J GANO        # Job name
#SBATCH -o GANO.o%j    # Name of stdout output file
#SBATCH -p gh         # Queue (partition) name
#SBATCH -N 16        # Total # of nodes
#SBATCH -n 16        # Total # of mpi tasks
#SBATCH -t 07:00:00        # Run time (hh:mm:ss)
#SBATCH -A EAR24026        # Allocation name (req'd if you have more than 1)

module load gcc cuda
module load python3

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

source /work/10845/kaichengz/vista/.bashrc
conda activate /work/10845/kaichengz/gmprocess
ibrun -np 16 python train_GANO_ddp.py --master ${head_node} --job-id $SLURM_JOB_ID
