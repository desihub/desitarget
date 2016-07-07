#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J test
#SBATCH -o test.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH,project
export OMP_NUM_THREADS=24
cd $SLURM_SUBMIT_DIR
date
srun -n 1 -c ${OMP_NUM_THREADS} python schema_generator_parallel.py -cores 24 -file_tractor_cats 24_tractor_cats.txt
date
echo DONE
