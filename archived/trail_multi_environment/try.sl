#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:00:59
#SBATCH --output=rl.out
#SBATCH -Aanakano_429
module load gcc/8.3.0 
export LD_PRELOAD=/spack/apps/gcc/8.3.0/lib64/libstdc++.so.6
mpirun -bind-to none -n 4 ./cart
