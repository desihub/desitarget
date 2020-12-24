#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import joblib  # VERSION 0.9.4 < 0.10.0 to get separate files
from sklearn.externals import joblib as skljoblib
from myRF import myRF

print(" Il faut utiliser l'environement export_RF : conda activate export_RF")

# CONFIGURATION
VERSION = 2
nTrees = 500

fpn_SCIKIT_RFmodel = "./WorkingDir/DR9s/RFmodel/DR9s_LOW/model_DR9s_LOW_z[0.0, 6.0]_MDepth30_MLNodes1000_nTrees500.pkl.gz"
fpn_DESI_RFmodel = "./Res/rf_model_dr9s_test.npz"

# fpn_SCIKIT_RFmodel = "./WorkingDir/DR9s/RFmodel/DR9s_HighZ/model_DR9s_HighZ_z[3.2, 6.0]_MDepth28_MLNodes1000_nTrees500.pkl.gz"
# fpn_DESI_RFmodel = "./Res/rf_model_dr9s_HighZ.npz"

dpn_DESI_RFmodel = "./tmpdir/"

# CHARGEMENT DU RF MODEL SOUS FORMAT SCIKIT
RF = skljoblib.load(fpn_SCIKIT_RFmodel)
print("\nLoading RF :", fpn_SCIKIT_RFmodel, "\n")

# Sauvegarde du RF fragmenté en plusieurs fichiers accessibles par myRF
s = joblib.dump(list(RF), dpn_DESI_RFmodel + "bdt.pkl", compress=False)

# Conversion et sauvegarde du RF au format DESI
data = np.array([]).astype(float)
myrf = myRF(data, dpn_DESI_RFmodel, numberOfTrees=nTrees, version=VERSION)
myrf.saveForest(fpn_DESI_RFmodel)
