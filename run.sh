
#module unload desimodules
#source /project/projectdirs/desi/software/desi_environment.sh 18.7
#module swap desitarget/0.22.0

#module unload desimodules
#source /project/projectdirs/desi/software/desi_environment.sh 18.7
#export PATH=$PATH:/global/homes/q/qmxp55/DESI/desitarget/bin
#export PYTHONPATH=$PYTHONPATH:/global/homes/q/qmxp55/DESI/desitarget/py

export LSDIR='/global/homes/q/qmxp55/DESI/desitarget_data/mini_sweep'
export TARGDIR='/global/homes/q/qmxp55/DESI/desitarget_data/targetdir'
#export WEBDIR='/project/projectdirs/desi/www/users/adamyers'
export DR='7.1'
export VERSION='0.24.0'

#mkdir $TARGDIR
#(if it doesn't exist!)

python bin/select_targets $LSDIR/ $TARGDIR/targets-dr$DR-$VERSION.fits
