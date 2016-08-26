#!/bin/bash -l

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J loaddb
#SBATCH -o loaddb.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH
cd $SLURM_SUBMIT_DIR
sanity="jobids_loaddb.txt"
echo "SLURM_JOBID="$SLURM_JOBID >> ${sanity}

##mpi
#date
#srun -n 24 python-mpi ./schema_generator.py --mpi --list_of_cats cats_24.txt --schema dr3 --load_db
#date

##multi
export OMP_NUM_THREADS=24
echo cores=${OMP_NUM_THREADS}
date
srun -n 1 -c ${OMP_NUM_THREADS} python tractor_load.py --cores ${OMP_NUM_THREADS} --list_of_cats remaining_dr3_cats.out --schema dr3 --load_db
date

##serial
#date
#srun -n 1 -c ${OMP_NUM_THREADS} python schema_generator.py --serial --list_of_cats cats_1.txt --schema dr3 --load_db
#date

echo DONE

