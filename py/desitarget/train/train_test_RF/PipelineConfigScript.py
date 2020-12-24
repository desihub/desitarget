#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

from collections import OrderedDict
# import pprint

import numpy as np

from desitarget.train.train_test_RF.util.funcs import RecHyParamDictExplFunc


def PipelineConfigScript(fpn_QSO_TrainingSample, fpn_STARS_TrainingSample,
                         fpn_TestSample, fpn_QLF, fpn_config):
    # ***CONFIGURATION***

    RELEASE = 'DR9s'  # seulement à titre informatif, aucun impact dans le pipeline
    random_state_seed = 666  # pour l'entraînement => reproductibilité
    n_jobs = 20  # pour l'entraînement et l'application des RF

    # Selection criteria
    OBJ_extraKeys = ['TYPE']
    OBJ_extraSelCMD = "OBJ_selection_OK & (OBJ_TYPE == 'PSF')"

    # rmag bins
    min_rmag = 17.5
    max_rmag = 23.0
    rmag_binStep = 0.2

    # zred bins
    min_zred = 0.
    max_zred = 6.
    zred_binStep = 0.2

    # HYPERPARAMETERS SPACE (TOUJOURS LISTE DE LISTES !!!)
    # '[[MODELE-DR9s_LOW], [MODEL-DR9s_HighZ]]'

    # 1ere étape
    # =============================================================================
    # nTreesVect = [[500], [500]] # [int]
    # maxDepthVect = [[None], [None]] # "None" ou [int]
    # maxLNodesVect = [[None], [None]] # "None" ou [int]
    # min_zredVect = [[[0., 6.]], [[3.2, 6.]]] # [float] "[0., 6.]"
    # =============================================================================

    nTreesVect = [[500], [500]]  # [int]
    maxDepthVect = [[25], [25]]  # "None" ou [int]
    maxLNodesVect = [[850], [850]]  # "None" ou [int]
    min_zredVect = [[[0., 6.]], [[3.2, 6.]]]  # [float] "[0., 6.]"

    # ***CONFIGURATION PAR DÉFAUT***

    # NE PAS MODIFIER
    # MODEL file path name template
    # dirpn_model = "./WorkingDir/" + str(RELEASE) + "/RFmodel/{}/"
    # rootpn_model = "model_{}_z{}_MDepth{}_MLNodes{}_nTrees{}"
    fpn_model_template = "/model_{}_z{}_MDepth{}_MLNodes{}_nTrees{}"

    # NE PAS MODIFIER
    # TEST file path name template
    # dirpn_test = "./WorkingDir/" + str(RELEASE) + "/test/{}/"
    # rootpn_test = "test_{}_z{}_MDepth{}_MLNodes{}_nTrees{}"
    fpn_test_template = "/test_{}_z{}_MDepth{}_MLNodes{}_nTrees{}"

    # NE PAS MODIFIER.
    feature_names = ['g_r', 'r_z', 'g_z', 'g_W1', 'r_W1', 'z_W1', 'g_W2', 'r_W2',
                     'z_W2', 'W1_W2', 'r']

    print("\n///**********TS CONFIG SCRIPT**********///")

    LIST_MODEL_NAME = []

    # ***FUNCTION***

    from functools import reduce
    from operator import mul

    def HyParamDictFunc(hyParamDict):

        flatHyParamSpaceDict = RecHyParamDictExplFunc(hyParamDict['ALGO'])

        hyParamSpaceTags = list(flatHyParamSpaceDict.keys())
        hyParamSpaceItems = list(flatHyParamSpaceDict.values())
        hyParamSpaceShape = [len(el) for el in hyParamSpaceItems]
        hyParamSpaceSize = reduce(mul, hyParamSpaceShape)

        hyParamDict['ALGO-hyParamSpace'] = dict()
        hyParamDict['ALGO-hyParamSpace']['flatHyParamSpaceDict'] = flatHyParamSpaceDict
        hyParamDict['ALGO-hyParamSpace']['hyParamSpaceTags'] = hyParamSpaceTags
        hyParamDict['ALGO-hyParamSpace']['hyParamSpaceItems'] = hyParamSpaceItems
        hyParamDict['ALGO-hyParamSpace']['hyParamSpaceShape'] = hyParamSpaceShape
        hyParamDict['ALGO-hyParamSpace']['hyParamSpaceSize'] = hyParamSpaceSize

    # ***MODELE DR9s_LOW***

    LIST_MODEL_NAME.append(['DR9s_LOW'])

    hyParamDict_DR9s_LOW = dict()
    hyParamDict_DR9s_LOW['MODEL'] = ['DR9s_LOW']
    hyParamDict_DR9s_LOW['BANDS'] = ['grzW']
    hyParamDict_DR9s_LOW['RELEASE'] = [RELEASE]
    hyParamDict_DR9s_LOW['ALGO'] = dict()
    tmpDict = OrderedDict()
    tmpDict['maxDepth'] = maxDepthVect[0]
    tmpDict['maxLNodes'] = maxLNodesVect[0]
    tmpDict['min_zred'] = min_zredVect[0]
    tmpDict['nTrees'] = nTreesVect[0]
    hyParamDict_DR9s_LOW['ALGO']['RF'] = tmpDict

    HyParamDictFunc(hyParamDict_DR9s_LOW)

    # ***MODELE DR9s_HighZ***

    LIST_MODEL_NAME.append(['DR9s_HighZ'])

    hyParamDict_DR9s_HighZ = dict()
    hyParamDict_DR9s_HighZ['MODEL'] = ['DR9s_HighZ']
    hyParamDict_DR9s_HighZ['BANDS'] = ['grzWHighz']
    hyParamDict_DR9s_HighZ['RELEASE'] = [RELEASE]
    hyParamDict_DR9s_HighZ['ALGO'] = dict()
    tmpDict = OrderedDict()
    tmpDict['maxDepth'] = maxDepthVect[1]
    tmpDict['maxLNodes'] = maxLNodesVect[1]
    tmpDict['min_zred'] = min_zredVect[1]
    tmpDict['nTrees'] = nTreesVect[1]
    hyParamDict_DR9s_HighZ['ALGO']['RF'] = tmpDict

    HyParamDictFunc(hyParamDict_DR9s_HighZ)

    # ***MODÈLE À TESTER***

    testConfigDict = dict()
    nTEST = len(LIST_MODEL_NAME)

    for numTEST in range(nTEST):
        TEST_NAME = 'TEST_' + LIST_MODEL_NAME[numTEST][0]
        testConfigDict[TEST_NAME] = {}
        testConfigDict[TEST_NAME]['MODEL_NAME'] = LIST_MODEL_NAME[numTEST]

    # ***SCENARIO = COMBINAISON DES MODÈLES TESTÉS***

    # ATTENTION:
    # - "proba_thold" toujours entre [0., 1.] ou "None"
    # Si différent de "None", "target_density" non considéré.
    # "None" possible que si "rmag_thold" (et par conséquent aussi "slope") = 0.
    # - "rmag_thold" toujours entre [min_rmag, max_rmag]
    # - "slope" toujours positives ou nulles
    # - couples ("rmag_thold", "slope") ordonnés par ordre croissant !!!
    # "rmag_thold" et "slope" doivent contenir un nombre identique d'éléments

    scenarioConfigDict = dict()

    # 2nd étape
    # =============================================================================
    # scenarioConfigDict['SCEN_DR9s_FULL'] = \
    #     {'TEST_NAME': ['TEST_DR9s_LOW', 'TEST_DR9s_HighZ'],
    #       'target_density': [245, 15],
    #       'proba_thold': [None, None],
    #       'rmag_thold': [[20.], [20.]],
    #       'slope': [[0.], [0.]]}
    # =============================================================================

    # DR9
    scenarioConfigDict['SCEN_DR9s_FULL'] = \
        {'TEST_NAME': ['TEST_DR9s_LOW', 'TEST_DR9s_HighZ'],
         'target_density': [245, 15],
         'proba_thold': [0.93, 0.55],  # [0.83, 0.55]
         'rmag_thold': [[21.5], [20.5]],
         'slope': [[0.025], [0.025]]}

    # DR7
    # =============================================================================
    # scenarioConfigDict['SCEN_DR7s_FULL'] = \
    #     {'TEST_NAME': ['TEST_DR7s_LOW', 'TEST_DR7s_HighZ'],
    #       'target_density': [245, 15],
    #       'proba_thold': [0.83, 0.55], # [0.83, 0.55]
    #       'rmag_thold': [[20.8, 21.5, 22.3], [20.5]],
    #       'slope': [[0.025, 0.15, 0.70], [0.025]]}
    # =============================================================================

    # ***CONSTRUCTION DU FICHIER DE CONFIG***

    configDict = dict()
    configDict['RELEASE'] = RELEASE
    configDict['fpn_STARS_TrainingSample'] = fpn_STARS_TrainingSample
    configDict['fpn_QSO_TrainingSample'] = fpn_QSO_TrainingSample
    configDict['fpn_TestSample'] = fpn_TestSample
    configDict['fpn_QLF'] = fpn_QLF
    configDict['feature_names'] = feature_names
    configDict['random_state_seed'] = random_state_seed
    configDict['n_jobs'] = n_jobs
    configDict['fpn_model_template'] = fpn_model_template
    configDict['fpn_test_template'] = fpn_test_template
    configDict['OBJ_extraKeys'] = OBJ_extraKeys
    configDict['OBJ_extraSelCMD'] = OBJ_extraSelCMD

    configDict['min_rmag'] = min_rmag
    configDict['max_rmag'] = max_rmag
    configDict['rmag_binStep'] = rmag_binStep

    configDict['min_zred'] = min_zred
    configDict['max_zred'] = max_zred
    configDict['zred_binStep'] = zred_binStep

    configDict['MODEL'] = dict()
    configDict['MODEL'][hyParamDict_DR9s_LOW['MODEL'][0]] = hyParamDict_DR9s_LOW
    configDict['MODEL'][hyParamDict_DR9s_HighZ['MODEL'][0]] = hyParamDict_DR9s_HighZ
    configDict['TEST'] = testConfigDict
    configDict['SCENARIO'] = scenarioConfigDict

    # ***STORING DU FICHIER DE CONFIG***

    if os.path.isfile(fpn_config):
        os.remove(fpn_config)
    np.savez(fpn_config, **{'CONFIG': configDict})
    print("Save :", fpn_config)

    # pprint.pprint(configDict, width = 1)
