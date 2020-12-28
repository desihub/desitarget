#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ****************************************
# RF TRAINING
# ****************************************

import argparse
import time
import pprint
import multiprocessing as mp
import copy

import numpy as np
import astropy.io.fits as pyfits
import joblib
from sklearn.ensemble import RandomForestClassifier

from desitarget.train.train_test_RF.util.funcs import Time2StrFunc, GetColorsFunc, Flat2TreeHyParamConvFunc


def train_RF(fpn_config, MODEL, dpn_RFmodel):
    print("\n///**********TRAIN RF**********///")
    infoRootStr = "INFO::TRAIN RF: "
    errRootStr = "ERR::TRAIN RF: "

    # ***CONFIG DATA LOADING***
    # CONFIG
    configDict = dict(np.load(fpn_config, allow_pickle=True)['CONFIG'][()])
    RELEASE = str(configDict['RELEASE'])
    fpn_STARS_TrainingSample = str(configDict['fpn_STARS_TrainingSample'])
    fpn_QSO_TrainingSample = str(configDict['fpn_QSO_TrainingSample'])
    random_state_seed = int(configDict['random_state_seed'])
    feature_names = list(configDict['feature_names'])
    n_jobs = int(configDict['n_jobs'])
    fpn_model_template = str(configDict['fpn_model_template'])

    # MODEL
    hyParamDict = dict(configDict['MODEL'][MODEL])
    BANDS = str(hyParamDict['BANDS'][0])

    flatHyParamSpaceDict = hyParamDict['ALGO-hyParamSpace']['flatHyParamSpaceDict']
    hyParamSpaceTags = hyParamDict['ALGO-hyParamSpace']['hyParamSpaceTags']
    hyParamSpaceItems = hyParamDict['ALGO-hyParamSpace']['hyParamSpaceItems']
    hyParamSpaceShape = hyParamDict['ALGO-hyParamSpace']['hyParamSpaceShape']
    hyParamSpaceSize = hyParamDict['ALGO-hyParamSpace']['hyParamSpaceSize']

    hyParamDictTemplate = copy.deepcopy(hyParamDict['ALGO'])

    # ***INITIALIZATION***
    n_jobs = min(mp.cpu_count(), n_jobs)

    # print init infos
    infoStr = "RELEASE : ('{:s}')".format(RELEASE)
    print(infoRootStr + infoStr)
    infoStr = "BANDS : ('{:s}')".format(BANDS)
    print(infoRootStr + infoStr)
    infoStr = "MODEL : ('{:s}')".format(MODEL)
    print(infoRootStr + infoStr)
    print(infoRootStr + "HYPERPARAMETERS SPACE :")
    pprint.pprint(flatHyParamSpaceDict, width=1)
    print(infoRootStr + "hyParamSpaceSize : (", hyParamSpaceSize, ")")
    print(infoRootStr + "hyParamSpaceShape : (", hyParamSpaceShape, ")")
    print()
    infoStr = "feature_names : ({:s})".format(str(feature_names))
    print(infoRootStr + infoStr)
    infoStr = "n_features : ({:d})".format(len(feature_names))
    print(infoRootStr + infoStr)
    infoStr = "n_jobs : ({:d})"
    print(infoRootStr + infoStr.format(n_jobs))
    print()

    # ***TRAINING DATA LOADING***
    # STARS
    infoStr = "STARS Training Sample : ('{:s}')".format(fpn_STARS_TrainingSample)
    print(infoRootStr + infoStr)
    STARS_data = pyfits.open(fpn_STARS_TrainingSample)[1].data
    STARS_colors = GetColorsFunc(STARS_data, feature_names)
    n_STARS = len(STARS_data)
    infoStr = "n_STARS : ({:d})".format(n_STARS)
    print(infoRootStr + infoStr)

    # QSO
    infoStr = "QSO Training Sample : ('{:s}')".format(fpn_QSO_TrainingSample)
    print(infoRootStr + infoStr)
    QSO_data = pyfits.open(fpn_QSO_TrainingSample)[1].data
    QSO_colors = GetColorsFunc(QSO_data, feature_names)
    n_QSO = len(QSO_data)
    QSO_zred = QSO_data['zred']
    infoStr = "n_QSO ({:d})".format(n_QSO)
    print(infoRootStr + infoStr)
    print()

    # ***TRAINING DES RF POUR DIFFÉRENTS JEUX DE PARAMÈTRES***
    infoStr = "TRAINING ..."
    print(infoRootStr + infoStr)
    glob_startTime = time.time()
    for it in range(hyParamSpaceSize):
        print("******************************************************************************")
        startTime = time.time()

        # Definition du jeu d'hyparamètre courant
        coords = np.unravel_index(it, hyParamSpaceShape)
        currentHyParams = Flat2TreeHyParamConvFunc(coords, hyParamDictTemplate,
                                                   hyParamSpaceTags, hyParamSpaceItems)
        nTrees = int(currentHyParams['RF']['nTrees'])
        maxDepth = currentHyParams['RF']['maxDepth']
        maxLNodes = currentHyParams['RF']['maxLNodes']
        min_zred = currentHyParams['RF']['min_zred']
        print(infoRootStr + "nTrees : (", nTrees, ")")
        print(infoRootStr + "maxDepth : (", maxDepth, ")")
        print(infoRootStr + "maxLNodes : (", maxLNodes, ")")
        print(infoRootStr + "min_zred : (", min_zred, ")")

        # QSO data reduction
        if len(min_zred) == 2:
            keep_zred = (QSO_zred >= min_zred[0]) & (QSO_zred <= min_zred[1])
        else:
            keep_zred = QSO_zred > min_zred[0]
        tmpQSO_colors = QSO_colors[keep_zred]
        tmp_n_QSO = tmpQSO_colors.shape[0]

        assert(tmp_n_QSO > 0), 'No QSO'
        infoStr = "({:d}) QSO selected over ({:d}) in total for zred > {:s}".format(tmp_n_QSO, n_QSO, str(min_zred))
        print(infoRootStr + infoStr)

        # Training data
        train_colors = np.vstack([STARS_colors, tmpQSO_colors])
        train_labels = np.hstack([np.zeros(n_STARS), np.ones(tmp_n_QSO)]).astype(np.int8)

        # RF initialization
        np.random.seed(random_state_seed)
        RF = RandomForestClassifier(nTrees, max_depth=maxDepth, max_leaf_nodes=maxLNodes, n_jobs=n_jobs)
        # RF training
        RF.fit(train_colors, train_labels)

        # RF storing
        fpn_RFmodel = dpn_RFmodel + fpn_model_template.format(
            MODEL, str(min_zred), str(maxDepth), str(maxLNodes), str(nTrees)) + '.pkl.gz'
        joblib.dump(RF, fpn_RFmodel, compress=9)
        infoStr = "Save RF model : ('{:s}')".format(fpn_RFmodel)
        print(infoRootStr + infoStr)

        tm = time.time() - startTime
        infoStr = "Training time per RF : ({:s})".format(Time2StrFunc(tm))
        print(infoRootStr + infoStr)

    print("******************************************************************************")
    tm = time.time() - glob_startTime
    infoStr = "Total training time : ({:s})".format(Time2StrFunc(tm))
    print(infoRootStr + infoStr)
