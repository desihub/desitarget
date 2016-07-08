#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J serial1
#SBATCH -o serial1.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH
cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=1

echo SERIAL run
date
srun -n 1 -c ${OMP_NUM_THREADS} python schema_generator.py --serial --list_of_cats cats_1.txt --schema dr3 --load_db
date

#echo MULTI cores 1
#date
#srun -n 1 -c ${OMP_NUM_THREADS} python schema_generator.py --cores 1 --list_of_cats cats_1.txt --schema dr3 --load_db
#date
#
#echo SERIAL run 24 cats
#date
#srun -n 1 -c ${OMP_NUM_THREADS} python schema_generator.py --serial --list_of_cats cats_24.txt --schema dr3 --load_db
#date


echo DONE
