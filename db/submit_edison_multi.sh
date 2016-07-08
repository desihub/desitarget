#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J multi24
#SBATCH -o multi24.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH
cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=24
echo cores=${OMP_NUM_THREADS}
date
srun -n 1 -c ${OMP_NUM_THREADS} python schema_generator.py --cores ${OMP_NUM_THREADS} --list_of_cats cats_24.txt --schema dr3 --load_db
date

echo DONE
