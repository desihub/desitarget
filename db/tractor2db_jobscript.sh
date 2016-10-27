#!/bin/bash -l

#SBATCH -p regular 
#SBATCH -N 1
#SBATCH -t 25:00:00
#SBATCH -J prodload
#SBATCH -o prodload.o%j
#SBATCH -L SCRATCH
cd $SLURM_SUBMIT_DIR
sanity="prodload_jobids.txt"
echo "SLURM_JOBID="$SLURM_JOBID >> ${sanity}

export NERSC_HOST=`/usr/common/usg/bin/nersc_host`
if [ "$NERSC_HOST" == "cori" ] ; then
    export CORES_ON_NODE=32
elif [ "$NERSC_HOST" == "edison" ] ; then
    export CORES_ON_NODE=24
fi

##mpi
#date
#srun -n ${CORES_ON_NODE} python-mpi ./tractor_load.py --mpi --list_of_cats dr3_cats_qso.txt --schema dr3 --load_db
#date

##multi
export OMP_NUM_THREADS=${CORES_ON_NODE}
echo cores=${OMP_NUM_THREADS}
date
srun -n 1 -c ${OMP_NUM_THREADS} python tractor_load.py --cores ${OMP_NUM_THREADS} --list_of_cats dr3_cats_qso.txt --schema dr3 --load_db
date

##serial
#date
#srun -n 1 -c ${OMP_NUM_THREADS} python tractor_load.py --serial --list_of_cats dr3_tractor_cats_qso.txt --schema dr3 --load_db
#date

echo DONE

