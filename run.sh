
#module unload desimodules
#source /project/projectdirs/desi/software/desi_environment.sh 18.7
#module swap desitarget/0.22.0

#module unload desimodules
#source /project/projectdirs/desi/software/desi_environment.sh 18.7
#export PATH=$PATH:/global/homes/q/qmxp55/DESI/desitarget/bin
#export PYTHONPATH=$PYTHONPATH:/global/homes/q/qmxp55/DESI/desitarget/py

export LSDIR='/global/homes/q/qmxp55/DESI/desitarget_data/mini_sweep'
export TARGDIR='/global/homes/q/qmxp55/DESI/desitarget_data/targetdir'
export WEBDIR='/global/homes/q/qmxp55/DESI/desitarget_data/WEB'
export DR='7.1'
export VERSION='0.25.0'
export RANDOMDIR='/project/projectdirs/desi/target/catalogs/dr7.1/0.22.0/'
#mkdir $TARGDIR
#(if it doesn't exist!)

#python bin/select_targets $LSDIR/ $TARGDIR/targets-dr$DR-$VERSION.fits
python bin/make_imaging_weight_map $RANDOMDIR/randoms-dr7.1-0.22.0.fits $TARGDIR/targets-dr$DR-$VERSION.fits $TARGDIR/pixweight-dr$DR-$VERSION.fits
python bin/run_target_qa $TARGDIR/targets-dr$DR-$VERSION.fits $WEBDIR/desitargetQA-dr$DR-$VERSION -w $TARGDIR/pixweight-dr$DR-$VERSION.fits 
#python bin/run_target_qa $TARGDIR/targets-dr$DR-$VERSION.fits $WEBDIR/desitargetQA-dr$DR-$VERSION --nosystematics
