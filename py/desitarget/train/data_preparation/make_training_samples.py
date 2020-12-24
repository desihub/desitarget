#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ****************************************
# MAKE STARS & QSO TRAINING SAMPLES
# ****************************************
from terminaltables import DoubleTable
from desitarget.train.data_preparation.funcs import Flux2MagFunc, ColorsFunc
from desitarget.train.data_preparation.PredCountsFromQLF_ClassModule import PredCountsFromQLF_Class

import numpy as np
import astropy.io.fits as pyfits
import warnings
warnings.simplefilter("ignore")

# ***QLF DATA FILE PATH NAME***

fpn_QLF_data = '../../py/desitarget/train/data_preparation/ROSS4_tabR.txt'

# ***STARS & QSO INPUT/OUTPUT FILE PATH NAMES***


def make_training_samples(fpn_QSO_input, fpn_STARS_input, fpn_QSO_output, fpn_STARS_output):
    # for test : remove_test_region --> False ect...
    remove_test_region = True

    # ***CONFIGURATION***

    NORM_MODE = "NORMALIZE_PER_RMAG_BINS"  # "NORMALIZE_PER_RMAG_BINS" by default
    initRndm = 666

    # zred bins
    min_zred = 0.
    max_zred = 6.
    zred_binStep = 0.2

    # rmag bins
    min_rmag = 17.5
    max_rmag = 23.0
    rmag_binStep = 0.2

    # Selection parameters
    STARS_NOBS_MIN = 2
    QSO_NOBS_MIN = 2
    # Need to find a balance between acceptable errors in the measured data and
    # good representativeness of the photometric scattering inherent to QSO.
    # (ML model has to be trained over data which match real photo. data)
    QSO_MAX_MAG_ERR_LEVEL = 0.02  # 0.02 by default

    # ***FUNCTION***
    def MAG_ERR_Func(FLUX, FLUX_IVAR):
        res = 2.5/np.log(10.)
        res /= (np.abs(FLUX)*np.sqrt(FLUX_IVAR))
        res[np.isinf(res)] = 0.
        res[np.isnan(res)] = 0.
        res[res < 0.] = 0.
        return res

    zred_binVect = np.arange(min_zred, max_zred + zred_binStep/2., zred_binStep)

    rmag_binVect = np.trunc(np.arange(min_rmag, max_rmag + rmag_binStep/2., rmag_binStep)*10.)/10.
    rmag_binVect = np.minimum(rmag_binVect, max_rmag)
    n_rmag_bin = rmag_binVect.size - 1

    # ***CHECK "NORM_MODE" VALIDITY***
    if (NORM_MODE != "NORMALIZE_PER_RMAG_BINS"):
        assert(False), "NORM_MODE {:s} non valid".format(NORM_MODE)

    # ***COMPUTE QLF4Compl_dNdrmag***
    predCountsObj = PredCountsFromQLF_Class()
    predCountsObj.LoadQLF_Data(fpn_QLF_data)

    # Fictive efficiency required by "predCountsObj"
    tmp_eff = np.ones((n_rmag_bin, zred_binVect.size - 1))

    predCountsObj.LoadEffData(tmp_eff, zred_binVect, rmag_binVect)
    predCountsObj.PrelOpFunc()
    predCountsObj.R_ZREDComplEvalFunc(zred_binVect, rmag_binVect)
    QLF4Compl_dNdzdrmag = predCountsObj.QLF4Compl_dNdzdmag[1:, 1:]

    QLF4Compl_dNdrmag = np.sum(QLF4Compl_dNdzdrmag, axis=1)

    # ***STARS SELECTION***

    STARS_data = pyfits.open(fpn_STARS_input, memmap=True)[1].data
    n_STARS = len(STARS_data)
    print("n_STARS initial :", n_STARS)

    STARS_gmag, STARS_rmag, STARS_zmag, \
        STARS_W1mag, STARS_W2mag = Flux2MagFunc(STARS_data)

    STARS_rmag_OK = (STARS_rmag >= min_rmag) & (STARS_rmag <= max_rmag)

    STARS_g_z_W1_W2_mag_OK = (STARS_gmag > 0) & (STARS_zmag > 0)
    STARS_g_z_W1_W2_mag_OK &= (STARS_W1mag > 0) & (STARS_W2mag > 0)

    # STARS_noBSinBLOB_OK = ~STARS_data['BRIGHTSTARINBLOB']
    # "http://legacysurvey.org/dr9/bitmasks/"
    print("[WARNING] REMOVE FROM THE STARS TRAINING maskbits 1, 5, 6, 7, 10, 12, 13")
    STARS_maskbits_OK = (STARS_data['MASKBITS'] & np.power(2, 1)) == 0  # 'BRIGHT'
    STARS_maskbits_OK = (STARS_data['MASKBITS'] & np.power(2, 5)) == 0
    STARS_maskbits_OK = (STARS_data['MASKBITS'] & np.power(2, 6)) == 0
    STARS_maskbits_OK = (STARS_data['MASKBITS'] & np.power(2, 7)) == 0
    STARS_maskbits_OK &= (STARS_data['MASKBITS'] & np.power(2, 10)) == 0  # 'BLOP'
    STARS_maskbits_OK &= (STARS_data['MASKBITS'] & np.power(2, 12)) == 0  # 'GALAXY'
    STARS_maskbits_OK &= (STARS_data['MASKBITS'] & np.power(2, 13)) == 0  # 'CLUSTER'

    STARS_NOBS_OK = ((STARS_data.NOBS_G >= STARS_NOBS_MIN) &
                     (STARS_data.NOBS_R >= STARS_NOBS_MIN) &
                     (STARS_data.NOBS_Z >= STARS_NOBS_MIN) &
                     (STARS_data.NOBS_W1 >= STARS_NOBS_MIN) &
                     (STARS_data.NOBS_W2 >= STARS_NOBS_MIN))

    # Already pre-sel PSF during data collection on NERSC
    STARS_PSF_OK = STARS_data['TYPE'] == "PSF"

    # --> we allready check that
    noTestRegion_OK = ~ ((STARS_data.RA <= 45.) & (STARS_data.RA >= 30.) & (np.abs(STARS_data.DEC) <= 5.))

    # STARS_OK =  STARS_rmag_OK & STARS_g_z_W1_W2_mag_OK & STARS_noBSinBLOB_OK
    STARS_OK = STARS_rmag_OK & STARS_g_z_W1_W2_mag_OK & STARS_maskbits_OK
    STARS_OK &= STARS_NOBS_OK & STARS_PSF_OK

    if remove_test_region:
        STARS_OK &= noTestRegion_OK

    STARS_data = STARS_data[STARS_OK]
    STARS_rmag = STARS_rmag[STARS_OK]

    n_STARS = len(STARS_data)
    print("n_STARS after selection/before normalization :", n_STARS)
    print()

    # ***QSO SELECTION***

    QSO_data = pyfits.open(fpn_QSO_input, memmap=True)[1].data
    n_QSO = len(QSO_data)
    print("n_QSO initial :", n_QSO)

    QSO_gmag, QSO_rmag, QSO_zmag, \
        QSO_W1mag, QSO_W2mag = Flux2MagFunc(QSO_data)

    QSO_rmag_OK = (QSO_rmag >= min_rmag) & (QSO_rmag <= max_rmag)

    QSO_g_z_W1_W2_mag_OK = (QSO_gmag > 0) & (QSO_zmag > 0)
    QSO_g_z_W1_W2_mag_OK &= (QSO_W1mag > 0) & (QSO_W2mag > 0)

    # QSO_noBSinBLOB_OK = ~QSO_data['BRIGHTSTARINBLOB']
    # "http://legacysurvey.org/dr8/bitmasks/"
    print("[WARNING] REMOVE FROM THE QSO TRAINING maskbits 1, 5, 6, 7, 10, 12, 13")
    QSO_maskbits_OK = (QSO_data['MASKBITS'] & np.power(2, 1)) == 0  # 'BRIGHT'
    QSO_maskbits_OK = (QSO_data['MASKBITS'] & np.power(2, 5)) == 0
    QSO_maskbits_OK = (QSO_data['MASKBITS'] & np.power(2, 6)) == 0
    QSO_maskbits_OK = (QSO_data['MASKBITS'] & np.power(2, 7)) == 0
    QSO_maskbits_OK &= (QSO_data['MASKBITS'] & np.power(2, 10)) == 0  # 'BLOP'
    QSO_maskbits_OK &= (QSO_data['MASKBITS'] & np.power(2, 12)) == 0  # 'GALAXY'
    QSO_maskbits_OK &= (QSO_data['MASKBITS'] & np.power(2, 13)) == 0  # 'CLUSTER'

    # FatStripe82
    FatStripe82_OK = (QSO_data.RA <= 45.) & (QSO_data.RA >= 0.) & (np.abs(QSO_data.DEC) <= 5.)
    FatStripe82_OK |= (QSO_data.RA <= 360.) & (QSO_data.RA >= 317.) & (np.abs(QSO_data.DEC) <= 2.)

    # OutOfFatStripe82
    brightQSOutOfFatStripe82_OK = ~ FatStripe82_OK

    brightQSOutOfFatS82_exSel = (
        (MAG_ERR_Func(QSO_data.FLUX_G, QSO_data.FLUX_IVAR_G) < QSO_MAX_MAG_ERR_LEVEL) &
        (MAG_ERR_Func(QSO_data.FLUX_R, QSO_data.FLUX_IVAR_R) < QSO_MAX_MAG_ERR_LEVEL) &
        (MAG_ERR_Func(QSO_data.FLUX_Z, QSO_data.FLUX_IVAR_Z) < QSO_MAX_MAG_ERR_LEVEL))

    # =============================================================================
    # brightQSOutOfFatS82_exSel &= (
    #     (MAG_ERR_Func(QSO_data.FLUX_W1, QSO_data.FLUX_IVAR_W1) < QSO_MAX_MAG_ERR_LEVEL) &
    #     (MAG_ERR_Func(QSO_data.FLUX_W1, QSO_data.FLUX_IVAR_W1) < QSO_MAX_MAG_ERR_LEVEL))
    # =============================================================================

    brightQSOutOfFatS82_exSel &= (
        (MAG_ERR_Func(QSO_data.FLUX_G, QSO_data.FLUX_IVAR_G) > 0) &
        (MAG_ERR_Func(QSO_data.FLUX_R, QSO_data.FLUX_IVAR_R) > 0) &
        (MAG_ERR_Func(QSO_data.FLUX_Z, QSO_data.FLUX_IVAR_Z) > 0) &
        (MAG_ERR_Func(QSO_data.FLUX_W1, QSO_data.FLUX_IVAR_W1) > 0) &
        (MAG_ERR_Func(QSO_data.FLUX_W1, QSO_data.FLUX_IVAR_W1) > 0))

    brightQSOutOfFatS82_exSel &= ((QSO_data.NOBS_G >= QSO_NOBS_MIN) &
                                  (QSO_data.NOBS_R >= QSO_NOBS_MIN) &
                                  (QSO_data.NOBS_Z >= QSO_NOBS_MIN) &
                                  (QSO_data.NOBS_W1 >= QSO_NOBS_MIN) &
                                  (QSO_data.NOBS_W2 >= QSO_NOBS_MIN))

    brightQSOutOfFatStripe82_OK &= brightQSOutOfFatS82_exSel

    # PSF
    QSO_PSF_OK = (QSO_data['TYPE'] == "PSF")

    # Remove TestRegion --> we allready check that
    noTestRegion_OK = ~ ((QSO_data.RA <= 45.) & (QSO_data.RA >= 30.) &
                         (np.abs(QSO_data.DEC) <= 5.))

    # QSO_OK
    # QSO_OK = QSO_rmag_OK & QSO_g_z_W1_W2_mag_OK & QSO_noBSinBLOB_OK
    QSO_OK = QSO_rmag_OK & QSO_g_z_W1_W2_mag_OK & QSO_maskbits_OK
    QSO_OK &= (FatStripe82_OK | brightQSOutOfFatStripe82_OK) & QSO_PSF_OK

    if remove_test_region:
        QSO_OK &= noTestRegion_OK

    QSO_data = QSO_data[QSO_OK]
    QSO_rmag = QSO_rmag[QSO_OK]
    n_QSO = len(QSO_data)

    print("n_QSO selected in FatStripe82 :", np.sum(QSO_OK & FatStripe82_OK))
    print("n_QSO selected outside FatStripe82 :", np.sum(QSO_OK & brightQSOutOfFatStripe82_OK))
    print("n_QSO after selection :", n_QSO)
    print()

    # QSO rmag histogram
    QSO_dNdrmag = np.histogram(QSO_rmag, bins=rmag_binVect)[0]

    # Given only for informative purpose to assess sample completeness.

    max_QSO_dNdrmag_ind = np.argmax(QSO_dNdrmag)
    scale_factor = QLF4Compl_dNdrmag[max_QSO_dNdrmag_ind]/QSO_dNdrmag[max_QSO_dNdrmag_ind]

    QSO_W_drmag = QLF4Compl_dNdrmag/(scale_factor*QSO_dNdrmag)
    QSO_W_drmag[QLF4Compl_dNdrmag == 0.] = np.nan
    QSO_W_drmag[np.isnan(QSO_W_drmag)] = np.nan
    QSO_W_drmag[np.isinf(QSO_W_drmag)] = np.nan

    if False:
        ref_QSO_dNdrmag_ind = np.nanargmin(QSO_W_drmag)
        scale_factor = QLF4Compl_dNdrmag[ref_QSO_dNdrmag_ind]/QSO_dNdrmag[ref_QSO_dNdrmag_ind]
        QSO_W_drmag = QLF4Compl_dNdrmag/(scale_factor*QSO_dNdrmag)
    else:
        nw_QSO_target = n_QSO
        QSO_W_drmag *= nw_QSO_target/np.sum(QSO_dNdrmag*QSO_W_drmag)

    QSO_W_drmag[QLF4Compl_dNdrmag == 0.] = 1.
    QSO_W_drmag[np.isnan(QSO_W_drmag)] = 1.
    QSO_W_drmag[np.isinf(QSO_W_drmag)] = 1.

    w_QSO_dNdrmag = QSO_dNdrmag*QSO_W_drmag
    nw_QSO = np.sum(w_QSO_dNdrmag)

    # ***STARS rmag. DISTRIBUTION NORMALIZATION***
    # Random generator seeding for reproducibility
    rs = np.random.RandomState(int(initRndm))

    list_STARS_sel_ind = []
    tab2print = []
    tab2print.append(["m_rmag", "M_rmag", "n_STARS_drmag", "n_sel_STARS_drmag", "n_QSO_drmag", "nw_QSO_drmag"])
    n_STARS_QSO_ratio = n_STARS/n_QSO

    for rmag_biNum in range(n_rmag_bin):
        m_rmag = rmag_binVect[rmag_biNum]
        M_rmag = rmag_binVect[rmag_biNum + 1]

        STARS_drmag_OK = (STARS_rmag >= m_rmag) & (STARS_rmag < M_rmag)
        n_STARS_drmag = np.sum(STARS_drmag_OK)
        n_QSO_drmag = int(QSO_dNdrmag[rmag_biNum])

        if NORM_MODE == "NORMALIZE_PER_RMAG_BINS":
            n_STARS_QSO_ratio = n_STARS_drmag/n_QSO_drmag

        if (n_STARS_QSO_ratio >= 1.):
            n_sel_STARS_drmag = int(n_STARS_drmag/n_STARS_QSO_ratio)
            STARS_rnd_sel_ind = rs.choice(n_STARS_drmag, n_sel_STARS_drmag, replace=False)
        else:
            # STARS_rnd_sel_ind = np.array([])
            STARS_rnd_sel_ind = slice(None)

        STARS_sel_ind = np.arange(n_STARS)[STARS_drmag_OK][STARS_rnd_sel_ind]
        list_STARS_sel_ind.extend(list(STARS_sel_ind.astype(np.int)))

        tab2print.append(["{:.1f}".format(m_rmag), "{:.1f}".format(M_rmag),
                          str(n_STARS_drmag), str(n_sel_STARS_drmag),
                          str(n_QSO_drmag),
                          str(int(w_QSO_dNdrmag[rmag_biNum]))])

    tab_title = "STARS/QSO TABLE"
    tab = DoubleTable(tab2print, tab_title)
    tab.inner_row_border = True
    tab.justify_columns = {0: 'center', 1: 'center', 2: 'center', 3: 'center',
                           4: 'center', 5: 'center'}
    print(tab.table)
    print()

    STARS_data = STARS_data[list_STARS_sel_ind]
    n_STARS = len(STARS_data)

    print("n_QSO after selection :", n_QSO)
    print("n_STARS after selection & normalization :", n_STARS)
    print()

    # ***COMPUTE AND ADD COLORS***

    color_names = ['g_r', 'r_z', 'g_z', 'g_W1', 'r_W1', 'z_W1', 'g_W2', 'r_W2', 'z_W2', 'W1_W2', 'r']
    n_colors = len(color_names)

    # STARS
    list_cols = []

    STARS_gmag, STARS_rmag, STARS_zmag, STARS_W1mag, STARS_W2mag = Flux2MagFunc(STARS_data)

    STARS_Colors = ColorsFunc(n_STARS, n_colors, STARS_gmag, STARS_rmag, STARS_zmag, STARS_W1mag, STARS_W2mag)

    for i, col_name in enumerate(color_names):
        col = pyfits.Column(name=col_name,  format='D', array=STARS_Colors[:, i])
        list_cols.append(col)

    STARS_hdu = pyfits.BinTableHDU(data=STARS_data)
    STARS_hdu = pyfits.BinTableHDU.from_columns(list(STARS_hdu.columns) + list_cols)

    # QSO
    list_cols = []

    QSO_gmag, QSO_rmag, QSO_zmag, QSO_W1mag, QSO_W2mag = Flux2MagFunc(QSO_data)

    QSO_Colors = ColorsFunc(n_QSO, n_colors, QSO_gmag, QSO_rmag, QSO_zmag, QSO_W1mag, QSO_W2mag)

    for i, col_name in enumerate(color_names):
        col = pyfits.Column(name=col_name,  format='D', array=QSO_Colors[:, i])
        list_cols.append(col)

    QSO_hdu = pyfits.BinTableHDU(data=QSO_data)
    QSO_hdu = pyfits.BinTableHDU.from_columns(list(QSO_hdu.columns) + list_cols)

    # ***STARS & QSO TRAINING FITS CATALOG STORING***

    STARS_hdu.writeto(fpn_STARS_output, overwrite=True)
    print("Save :", fpn_STARS_output)

    QSO_hdu.writeto(fpn_QSO_output, overwrite=True)
    print("Save :", fpn_QSO_output)
