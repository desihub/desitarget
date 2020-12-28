"""
    Permet de convertir un RandomForestClassifier de sk-learn sous la forme requise par desi_target dans la classe myRF.

    Contient également un petit test pour vérifier si le script fonctionne correctement !

    Ce Code n'a plus de soucis avec la version de joblib et fonctionne donc avec les versions récentes de scikit-learn

    Fonctionne (testé) avec : scikit_learn : 0.22.1 et joblib : 0.14.1
"""

import os
import subprocess
import sys

import fitsio
import numpy as np
import time

from desitarget.myRF import myRF
from pkg_resources import resource_filename

import joblib
from sklearn.ensemble import RandomForestClassifier


def build_desi_tree(tree):

    n_nodes = tree.tree_.node_count                               # nombre de noeud dans l'arbre
    children_left = tree.tree_.children_left                      # liste des noeuds fils gauche de chaque noeud
    children_right = tree.tree_.children_right                    # liste des noeuds fils droit de chaque noeud
    feature = tree.tree_.feature                                  # liste des features sélectionnées à chaque noeud
    threshold = tree.tree_.threshold                              # liste des probabilités de coupures à chaque noeud
    value = tree.tree_.value[:, 0, :]                             # liste du nombre d'objet dans la classe 0 et la classe 1 à chaque noeud
    probability = value[:, 1] / (value[:, 0] + value[:, 1])       # Probabilité d'être dans la classe 1 à chaque noeud

    desi_tree = np.zeros(n_nodes, dtype='int16, int16, int8, float32, float32')

    for i in range(n_nodes):
        desi_tree[i] = (children_left[i], children_right[i], feature[i], threshold[i], probability[i])

    return desi_tree


def build_desi_forest(rf_input):
    print("File load : ", rf_input)
    rf = joblib.load(rf_input)                                    # RandomForestClassifier

    print(rf, "\n")

    forest = rf.estimators_                                       # List of DecisionTreeClassifiers
    nTrees = len(forest)                                          # number of Trees
    desi_forest = []                                              # List of Tree in Desi format

    for i in range(nTrees):
        tree = forest[i]                                          # DecisionTreeClassifier
        desi_forest.append(build_desi_tree(tree))

    return desi_forest


def convert_and_save_to_desi(rf_input, filename_output):
    print("Starting convertion...")
    desi_forest = build_desi_forest(rf_input)
    np.savez_compressed(filename_output, desi_forest)
    print("Desi format is saved at : ", filename_output, "\n")


# number of variables
nfeatures = 11


def read_file(inputFile):

    sample = fitsio.read(inputFile, columns=['RA', 'DEC', 'TYPE', 'zred', 'g_r',
                                             'r_z', 'g_z', 'g_W1', 'r_W1',
                                             'z_W1', 'g_W2', 'r_W2', 'z_W2',
                                             'W1_W2', 'r'], ext=1)

    # We keep only Correct Candidates.
    reduce_sample = sample[(((sample['TYPE'][:] == 'PSF ') |
                             (sample['TYPE'][:] == 'PSF')) &
                            (sample['r'][:] < 23.0) &
                            (sample['r'][:] > 0.0))]

    print("\n############################################")
    print('Input file = ', inputFile)
    print('Original size: ', len(sample))
    print('Reduce size: ', len(reduce_sample))
    print("############################################\n")

    return reduce_sample


def build_attributes(nbEntries, nfeatures, sample):

    colors = np.zeros((nbEntries, nfeatures))

    colors[:, 0] = sample['g_r'][:]
    colors[:, 1] = sample['r_z'][:]
    colors[:, 2] = sample['g_z'][:]
    colors[:, 3] = sample['g_W1'][:]
    colors[:, 4] = sample['r_W1'][:]
    colors[:, 5] = sample['z_W1'][:]
    colors[:, 6] = sample['g_W2'][:]
    colors[:, 7] = sample['r_W2'][:]
    colors[:, 8] = sample['z_W2'][:]
    colors[:, 9] = sample['W1_W2'][:]
    colors[:, 10] = sample['r'][:]

    return colors


def compute_proba_desi(sample, rf_fileName):

    attributes = build_attributes(len(sample), nfeatures, sample)

    pathToRF = '.'
    print('Load Old Random Forest : ')
    print('    * ' + rf_fileName)
    print('Random Forest over: ', len(attributes), ' objects\n')

    myrf = myRF(attributes, pathToRF, numberOfTrees=500, version=2)
    myrf.loadForest(rf_fileName)
    proba_rf = myrf.predict_proba()

    return proba_rf


def compute_proba_sk_learn(sample, RF_file):

    attributes = build_attributes(len(sample), nfeatures, sample)

    print('Load New Random Forest : ')
    print('    * ' + RF_file)
    print('Random Forest over: ', len(attributes), ' objects\n')

    RF = joblib.load(RF_file)
    proba_rf = RF.predict_proba(attributes)[:, 1]

    return proba_rf


def compare_sklearn_desi(RF_filename_sklearn, RF_filename_desi, test_sample):
    zred = test_sample['zred'][:]
    r = test_sample['r'][:]
    sel_qso = zred > 0

    proba_rf_desi = compute_proba_desi(test_sample, RF_filename_desi)
    proba_rf_sk_learn = compute_proba_sk_learn(test_sample, RF_filename_sklearn)
    diff = proba_rf_desi - proba_rf_sk_learn

    print('\n############################################')
    print('difference entre les deux manieres de calculer')
    print('min = ', np.min(diff))
    print('max = ', np.max(diff))
    print('mean = ', np.mean(diff))
    print('std = ', np.std(diff))
    print('############################################\n')

    cut_dr8 = 0.88 - 0.03*np.tanh(r - 20.5)

    sel_desi = proba_rf_desi > cut_dr8
    sel_sk_learn = proba_rf_sk_learn > cut_dr8

    effi_desi = float(len(r[sel_desi & sel_qso]))/float(len(r[sel_qso]))
    effi_sk_learn = float(len(r[sel_sk_learn & sel_qso]))/float(len(r[sel_qso]))

    print('\n############################################')
    print('efficacite desi = ', effi_desi)
    print('efficacite sk_learn= ', effi_sk_learn)
    print('############################################\n')


def test_convertion(test_file, RF_filename_input, RF_filename_output):
    print("\n***************************************")
    print("Test of the convertion :")
    print("***************************************\n")
    test_sample = read_file(testfile)
    compare_sklearn_desi(RF_filename_sklearn, RF_filename_desi, test_sample)
