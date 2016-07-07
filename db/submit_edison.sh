#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J test
#SBATCH -o test.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH,project
cd $SLURM_SUBMIT_DIR
export fits_file=/global/cscratch1/sd/desiproc/dr3/tractor/000/tractor-0001m002.fits

export OMP_NUM_THREADS=1
echo cores=${OMP_NUM_THREADS} (serial)
date
srun -n 1 -c ${OMP_NUM_THREADS} python schema_generator.py --fits_file ${fits_file} --schema dr3 --load_db 1
date

export OMP_NUM_THREADS=1
echo cores=${OMP_NUM_THREADS}
date
srun -n 1 -c ${OMP_NUM_THREADS} python schema_generator.py --cores ${OMP_NUM_THREADS} --fits_file ${fits_file} --schema dr3 --load_db 1
date

export OMP_NUM_THREADS=24
echo cores=${OMP_NUM_THREADS}
date
srun -n 1 -c ${OMP_NUM_THREADS} python schema_generator.py --cores ${OMP_NUM_THREADS} --fits_file ${fits_file} --schema dr3 --load_db 1
date

echo DONE
