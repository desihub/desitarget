"""
desitarget.train.train_mva_decals
=================================

- Training code developed by E. Burtin
- Update to run with DR3 by Ch. Yeche

This example can be run with the module desitarget/bin/qso_training

Three actions controlled by "step" flag: train - test - extract_myRF

Two examples of random forest (with and without r_mag)
Two examples of adaboost (with and without r_mag)

Inputs
------

The training samples and the test sample are available on nesrc at:
/global/project/projectdirs/desi/target/qso_training/

The qso  training sample qso_dr3_nora36-42.fits is obtained
with QSOs from the fat stripe 82 and bright QSOs (sigma(r)<0.02) of the
rest of the footprint. Note that the 36<ra<42 region was exluded of
the training sample to allow independant test over this 36<ra<42 region

The star  training sample qso_dr3_nora36-42_normalized.fits is obtained
with PSF objects of stripe 82, which are not variable (NNVariability<0.3)
and not known QSOs. "normalized" means that the r_mag distribution of the
stars is exactly the same as that of qsos.

The test sample Stripe82_dr3_decals is the stripe 82. Note this file
contains the results of the four algorithm for seed 0. The new probabilities
should be _strictly_ identical

Outputs
-------

train
    Four compressed files are produced. Each file corresponds to one algorithm
    (*i.e.* adaboost/random forest, with/withour rmag).

test
    Produce the probabilities for the four algorithms, the results are stored in
    Stripe82_dr3_decals_newTraining.fits

extract_myRF
    Use the file rf_model_dr3.pkl.gz produced in step "train" and convert it
    in a numpy array that can be read by desitarget.myRF class.
    The results is the compressed numpy array rf_model_dr3.npz
"""

import astropy.io.fits as pyfits
import numpy as np
import sys
import os

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from desitarget.myRF import myRF

#------------------------------------------------------------
def magsExtFromFlux(dataArray):

    gflux  = dataArray.decam_flux[:,1]/dataArray.decam_mw_transmission[:,1]
    rflux  = dataArray.decam_flux[:,2]/dataArray.decam_mw_transmission[:,2]
    zflux  = dataArray.decam_flux[:,4]/dataArray.decam_mw_transmission[:,4]
    W1flux = dataArray.wise_flux[:,0]/dataArray.wise_mw_transmission[:,0]
    W2flux = dataArray.wise_flux[:,1]/dataArray.wise_mw_transmission[:,1]

    W1flux[np.isnan(W1flux)]=0.
    W2flux[np.isnan(W2flux)]=0.
    gflux[np.isnan(gflux)]=0.
    rflux[np.isnan(rflux)]=0.
    zflux[np.isnan(zflux)]=0.
    W1flux[np.isinf(W1flux)]=0.
    W2flux[np.isinf(W2flux)]=0.
    gflux[np.isinf(gflux)]=0.
    rflux[np.isinf(rflux)]=0.
    zflux[np.isinf(zflux)]=0.

    g=np.where( gflux>0,22.5-2.5*np.log10(gflux), 0.)
    r=np.where( rflux>0,22.5-2.5*np.log10(rflux), 0.)
    z=np.where( zflux>0,22.5-2.5*np.log10(zflux), 0.)
    W1=np.where( W1flux>0, 22.5-2.5*np.log10(W1flux), 0.)
    W2=np.where( W2flux>0, 22.5-2.5*np.log10(W2flux), 0.)

    g[np.isnan(g)]=0.
    g[np.isinf(g)]=0.
    r[np.isnan(r)]=0.
    r[np.isinf(r)]=0.
    z[np.isnan(z)]=0.
    z[np.isinf(z)]=0.
    W1[np.isnan(W1)]=0.
    W1[np.isinf(W1)]=0.
    W2[np.isnan(W2)]=0.
    W2[np.isinf(W2)]=0.

    return g,r,z,W1,W2

#------------------------------------------------------------
def colors(nbEntries,nfeatures,g,r,z,W1,W2):

    colors  = np.zeros((nbEntries,nfeatures))

    colors[:,0]=g-r
    colors[:,1]=r-z
    colors[:,2]=g-z
    colors[:,3]=g-W1
    colors[:,4]=r-W1
    colors[:,5]=z-W1
    colors[:,6]=g-W2
    colors[:,7]=r-W2
    colors[:,8]=z-W2
    colors[:,9]=W1-W2
    colors[:,10]=r

    return colors

def train_mva_decals(Step,debug=False):

# number of variables
    nfeatures = 11

#----------------------------------------------------------
#   files to be used for training and for tests
#----------------------------------------------------------

# files available on nersc
    modelDir='./'
#    dataDir='/global/project/projectdirs/desi/target/qso_training/'
    dataDir='./'

# region of control   36<ra<42 is removed
    starTraining = dataDir+'star_dr3_nora36-42_normalized.fits' #dr3
    qsoTraining = dataDir+'qso_dr3_nora36-42.fits' #dr3

# Test over stripe 82
    fileName = 'Stripe82_dr3_decals' #dr3
    objectTesting  = dataDir+fileName+'.fits'
    outputFile = './'+fileName+'_newTraining.fits'

    if Step=='train' :
        star0 = pyfits.open(starTraining,memmap=True)[1].data
        star0_g,star0_r,star0_z,star0_W1,star0_W2 = magsExtFromFlux(star0)
        star = star0[(star0_g>0)&(star0_r<22.7)]

        qso0 = pyfits.open(qsoTraining,memmap=True)[1].data
        qso0_g,qso0_r,qso0_z,qso0_W1,qso0_W2 = magsExtFromFlux(qso0)
        qso = qso0[(qso0_r>0)&(qso0_r<22.7)]

    elif ( Step=='test' or  Step=='extract_myRF' ) :
        object = pyfits.open(objectTesting,memmap=True)[1].data
        object_g,object_r,object_z,object_W1,object_W2 = magsExtFromFlux(object)
        nobjecttot = len(object)
        object_colors = colors(nobjecttot,nfeatures,object_g,object_r,object_z,object_W1,object_W2)
        
    else :
        print('Unknown option')
        sys.exit()



#------------------------------------------------------------
    if Step== 'train' :
#------------------------------------------------------------


#----------------------------------------------------------
#   prepare arrays for Machine Learning
#----------------------------------------------------------

        print('qsos in file:',len(qso))
        print('star in file:',len(star))
        nqsotot = len(qso)
        nqso   = len(qso)
        nstartot = len(star)
        nstar = len(star)

        if nqsotot*nstartot==0 : sys.exit()

        data = np.zeros((nqso+nstar,nfeatures))
        target = np.zeros(nqso+nstar)

        qso_g,qso_r,qso_z,qso_W1,qso_W2 = magsExtFromFlux(qso)
        qso_colors = colors(nqsotot,nfeatures,qso_g,qso_r,qso_z,qso_W1,qso_W2)

        if debug :
            debug_qso_cols = pyfits.ColDefs([
                    pyfits.Column(name='r',format='E' ,array=qso_r[:]),
                    pyfits.Column(name='g',format='E' ,array=qso_g[:]),
                    pyfits.Column(name='z',format='E' ,array=qso_z[:]),
                    pyfits.Column(name='W1',format='E' ,array=qso_W1[:]),
                    pyfits.Column(name='W2',format='E' ,array=qso_W2[:]),
                    pyfits.Column(name='colors',format='11E' ,array=qso_colors[:,:]),
                    ])
            hduQso = pyfits.BinTableHDU.from_columns(debug_qso_cols)
            hduQso.writeto('debug_qso.fits',clobber=True)

            print(' Debug qsos')
            print(qso_colors)


        star_g,star_r,star_z,star_W1,star_W2 = magsExtFromFlux(star)
        star_colors = colors(nstartot,nfeatures,star_g,star_r,star_z,star_W1,star_W2)

        if debug :
            debug_star_cols = pyfits.ColDefs([
                    pyfits.Column(name='r',format='E' ,array=star_r[:]),
                    pyfits.Column(name='g',format='E' ,array=star_g[:]),
                    pyfits.Column(name='z',format='E' ,array=star_z[:]),
                    pyfits.Column(name='W1',format='E' ,array=star_W1[:]),
                    pyfits.Column(name='W2',format='E' ,array=star_W2[:]),
                    pyfits.Column(name='colors',format='11E' ,array=star_colors[:,:]),
                    ])
            hduStar = pyfits.BinTableHDU.from_columns(debug_star_cols)
            hduStar.writeto('debug_star.fits',clobber=True)
            print(' Debug stars')
            print(star_colors)

    # final arrays
        data[0:nqso,:]=qso_colors[0:nqso,:]
        data[nqso:nqso+nstar,:]=star_colors[0:nstar,:]
        target[0:nqso]=1
        target[nqso:nqso+nstar]=0

    #----------------------------------------------------------
    #   Start the training
    #----------------------------------------------------------

        print('training over ',nqso,' qsos and ',nstar,' stars')

        print('with random Forest')
        np.random.seed(0)
        rf = RandomForestClassifier(200)
        rf.fit(data, target)
        joblib.dump(rf, modelDir+'rf_model_dr3.pkl.gz',compress=9)
        np.random.seed(0)
        rf.fit(data[:,0:9], target)
        joblib.dump(rf, modelDir+'rf_model_normag_dr3.pkl.gz',compress=9)

        print('with adaBoost')
        ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),
                         algorithm="SAMME.R",
                         n_estimators=200)
        np.random.seed(0)
        ada.fit(data, target)
        joblib.dump(ada, modelDir+'adaboost_model_dr3.pkl.gz',compress=9)
        np.random.seed(0)
        ada.fit(data[:,0:9], target)
        joblib.dump(ada, modelDir+'adaboost_model_normag_dr3.pkl.gz',compress=9)

        sys.exit()

#------------------------------------------------------------
    if Step== 'test' :
#------------------------------------------------------------
        print('Check over a test sample')

#-----------------------
        print('random Forest over ', len(object_colors),' objects ')

        rf = joblib.load(modelDir+'rf_model_dr3.pkl.gz')
        pobject_rf = rf.predict_proba(object_colors)

        rf = joblib.load(modelDir+'rf_model_normag_dr3.pkl.gz')
        pobject_rf_ns = rf.predict_proba(object_colors[:,0:9])

#-----------------------
        print('adaBoost over ', len(object_colors),' objects ')

        ada = joblib.load(modelDir+'adaboost_model_dr3.pkl.gz')
        pobject_ada = ada.predict_proba(object_colors)

        ada = joblib.load(modelDir+'adaboost_model_normag_dr3.pkl.gz')
        pobject_ada_ns = ada.predict_proba(object_colors[:,0:9])

#-----------------------
        print('updating fits file')


        hdusel = pyfits.BinTableHDU(data=object)
        print('create fit file with',len(object),' objects')
        orig_cols = object.columns
        new_cols = pyfits.ColDefs([
                pyfits.Column(name='PADA_new'       ,format='E' ,array=pobject_ada[:,1]),
                pyfits.Column(name='PADAnomagr_new'  ,format='E' ,array=pobject_ada_ns[:,1]),
                pyfits.Column(name='PRANDF_new'     ,format='E' ,array=pobject_rf[:,1]),
                pyfits.Column(name='PRANDFnomagr_new',format='E' ,array=pobject_rf_ns[:,1]),
                ])
        hduNew = pyfits.BinTableHDU.from_columns(orig_cols + new_cols)
        hduNew.writeto(outputFile,clobber=True)

        sys.exit()

#------------------------------------------------------------
    if Step== 'extract_myRF' :
#------------------------------------------------------------
        print('Produce the random forest with our own persistency')

        rf = joblib.load(modelDir+'rf_model_dr3.pkl.gz')
#        rf = joblib.load(modelDir+'rf_model_elg_ref.pkl.gz')

        newDir= modelDir+'RF/'
        print ('dump all files in ',newDir)
        if not os.path.isdir(newDir) :
            os.makedirs(newDir)
        joblib.dump(rf, newDir+'bdt.pkl')

        nTrees=200
#        nTrees=500
        myrf =  myRF(object_colors,newDir,numberOfTrees=nTrees,version=2)
        myrf.saveForest(modelDir+'rf_model_dr3.npz')
#        myrf.saveForest(modelDir+'rf_model_new.npz')

        sys.exit()
