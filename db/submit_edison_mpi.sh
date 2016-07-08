#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J mpi24
#SBATCH -o mpi24.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH
cd $SLURM_SUBMIT_DIR

#srun -n 4 python-mpi ./mpi_test.py

date
srun -n 24 python-mpi ./schema_generator.py --mpi --list_of_cats cats_24.txt --schema dr3 --load_db
date

echo DONE

