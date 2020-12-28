#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ****************************************
# MAKE TEST SAMPLE
# ****************************************

import argparse
import numpy as np
import astropy.io.fits as pyfits

from desitarget.train.data_preparation.funcs import shift_photo_north, Flux2MagFunc, ColorsFunc, AreaFunc


def make_test_sample(fpn_input, fpn_output, RELEASE='DR9', is_north=False):
    # ***CONFIGURATION***
    min_rmag = 17.5
    max_rmag = 23.0  # 23.5 pour SV avec DR8/9 ?

    # Selection criteria
    OBJ_extraSelCMD = ""
    if RELEASE == 'DR7':
        OBJ_extraKeys = ['BRIGHTSTARINBLOB']
        OBJ_extraSelCMD += "OBJ_selection_OK &= True"
        OBJ_extraSelCMD += " & (OBJ_rmag > {:f})".format(min_rmag)
        OBJ_extraSelCMD += " & (OBJ_W1mag > 0.)"
        OBJ_extraSelCMD += " & (OBJ_W2mag > 0.)"
        OBJ_extraSelCMD += " & (~OBJ_BRIGHTSTARINBLOB)"
    elif RELEASE == 'DR8':
        OBJ_extraKeys = ['MASKBITS']
        OBJ_extraSelCMD += "OBJ_selection_OK &= True"
        OBJ_extraSelCMD += " & (OBJ_rmag > {:f})".format(min_rmag)
        OBJ_extraSelCMD += " & (OBJ_W1mag > 0.)"
        OBJ_extraSelCMD += " & (OBJ_W2mag > 0.)"
        print("MASKBIT USED : 1, 11, 12, 13 ")
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 1))   == 0)"  # 'BRIGHT'
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 11))  == 0)"  # 'MEDIUM'
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 12))  == 0)"  # 'GALAXY'
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 13))  == 0)"  # 'CLUSTER'
    elif RELEASE == 'DR9':
        OBJ_extraKeys = ['MASKBITS']
        OBJ_extraSelCMD += "OBJ_selection_OK &= True"
        OBJ_extraSelCMD += " & (OBJ_rmag > {:f})".format(min_rmag)
        OBJ_extraSelCMD += " & (OBJ_W1mag > 0.)"
        OBJ_extraSelCMD += " & (OBJ_W2mag > 0.)"
        print("MASKBIT USED : 1, 5, 6, 7, 10, 12, 13 ")
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 1))   == 0)"  # 'BRIGHT'
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 5))  == 0)"  # 'MEDIUM'
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 6))  == 0)"  # 'GALAXY'
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 7))  == 0)"  # 'CLUSTER'
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 10))  == 0)"  # 'MEDIUM'
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 12))  == 0)"  # 'GALAXY'
        OBJ_extraSelCMD += " & ((OBJ_MASKBITS & np.power(2, 13))  == 0)"  # 'CLUSTER'
    else:
        assert(False), "'RELEASE' unvailable !"

    # ***OBJ DATA LOADING & SELECTION***

    # Data loading
    OBJ_data = pyfits.open(fpn_input, memmap=True)[1].data
    print("n_OBJ avant selection :", len(OBJ_data))
    print("n_QSO avant selection :", np.sum(OBJ_data['zred'] > 0.))

    # "shift_photo_north"
    if is_north:
        backup_FLUX_GRZ = [OBJ_data.FLUX_G.copy(), OBJ_data.FLUX_R.copy(), OBJ_data.FLUX_Z.copy()]
        OBJ_data.FLUX_G, OBJ_data.FLUX_R, OBJ_data.FLUX_Z = shift_photo_north(OBJ_data.FLUX_G, OBJ_data.FLUX_R, OBJ_data.FLUX_Z)

    # Flux to magnitudes
    OBJ_gmag, OBJ_rmag, OBJ_zmag, OBJ_W1mag, OBJ_W2mag = Flux2MagFunc(OBJ_data)

    # Data selection
    OBJ_selection_OK = (OBJ_gmag > 0.) & (OBJ_rmag > 0.) & (OBJ_zmag > 0.)
    OBJ_selection_OK &= (OBJ_rmag < max_rmag)

    # Extra data selection
    for key in OBJ_extraKeys:
        exec('OBJ_' + key + " = OBJ_data['" + key + "']")
    exec(OBJ_extraSelCMD)

    n_OBJ = np.sum(OBJ_selection_OK)
    assert(n_OBJ > 0), 'No OBJ'

    # Reduction !
    OBJ_data = OBJ_data[OBJ_selection_OK]

    # Display some infos
    OBJ_ra = OBJ_data['RA']
    OBJ_dec = OBJ_data['DEC']
    n_QSO = np.sum(OBJ_data['zred'] > 0.)

    min_ra = OBJ_ra.min()
    max_ra = OBJ_ra.max()
    min_dec = OBJ_dec.min()
    max_dec = OBJ_dec.max()

    skyArea = AreaFunc(min_dec*np.pi/180., max_dec*np.pi/180., min_ra*np.pi/180., max_ra*np.pi/180.)

    print("Selection finie")
    print("min_ra : {:d} deg, max_ra : {:d} deg".format(int(min_ra.round()), int(max_ra.round())))
    print("min_dec : {:d} deg, max_dec : {:d} deg".format(int(min_dec.round()), int(max_dec.round())))
    print("skyArea : {:d} deg2".format(int(skyArea.round())))
    print("n_OBJ :", n_OBJ)
    print("n_QSO :", n_QSO)

    # ***COMPUTE AND ADD COLORS***

    color_names = ['g_r', 'r_z', 'g_z', 'g_W1', 'r_W1', 'z_W1', 'g_W2', 'r_W2', 'z_W2', 'W1_W2', 'r']
    n_colors = len(color_names)

    list_cols = []

    OBJ_gmag, OBJ_rmag, OBJ_zmag, OBJ_W1mag, OBJ_W2mag = Flux2MagFunc(OBJ_data)

    OBJ_Colors = ColorsFunc(n_OBJ, n_colors, OBJ_gmag, OBJ_rmag, OBJ_zmag, OBJ_W1mag, OBJ_W2mag)

    for i, col_name in enumerate(color_names):
        col = pyfits.Column(name=col_name, format='D', array=OBJ_Colors[:, i])
        list_cols.append(col)

    OBJ_hdu = pyfits.BinTableHDU(data=OBJ_data)
    if is_north:
        for i, col_name in enumerate(['FLUX_G', 'FLUX_R', 'FLUX_Z']):
            col = pyfits.Column(name=col_name + '_s',  format='D', array=OBJ_data[col_name].copy())
            list_cols.append(col)
            OBJ_data[col_name] = backup_FLUX_GRZ[i][OBJ_selection_OK]

    OBJ_hdu = pyfits.BinTableHDU.from_columns(list(OBJ_hdu.columns) + list_cols)

    # ***TEST SAMPLE STORING***

    OBJ_hdu.writeto(fpn_output, overwrite=True)
    print("Save :", fpn_output)
