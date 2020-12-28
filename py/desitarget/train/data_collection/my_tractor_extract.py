#!/usr/bin/env python

import subprocess
import datetime
from argparse import ArgumentParser
import numpy as np
import fitsio
from data_collection.RA_DEC_MatchingClassModule import RA_DEC_MatchingClass

# settings.
fpn_QSO_cat = "/global/cfs/cdirs/desi/target/analysis/RF/Catalogs/DR16Q_red.fits"
fpn_var_cat = "/global/cfs/cdirs/desi/target/analysis/RF/Catalogs/Str82_variability_wise_bdt_qso_star_DR7_BOSS_-50+60.fits"
radius4matching = 1.4/3600.  # [deg]
NNVar_th = 0.5

# reading arguments.
parser = ArgumentParser()
parser.add_argument('-i', '--infits', type=str, default=None, metavar='INFITS', help='input fits')
parser.add_argument('-o', '--outfits', type=str, default=None, metavar='OUTFITS', help='output fits')
parser.add_argument('-r', '--release', type=str, default=None, metavar='RELEASE', help='release ("dr7","dr8s", "dr8n")')
parser.add_argument('-rd', '--radec', type=str, default='0,360,-90,90', metavar='RADEC', help='ramin,ramax,decmin,decmax')
parser.add_argument('-s', '--selcrit', type=str, default=None, metavar='SELCRIT', help='selection criterion ("qso", "stars", "test")')
parser.add_argument('-l', '--logfile', type=str, default='none', metavar='LOGFILE', help='log file')

arg = parser.parse_args()
INFITS, OUTFITS, RELEASE, RADEC, SELCRIT, LOGFILE = arg.infits, arg.outfis, arg.release, arg.radec, arg.selcrit, arg.logfile

# RADEC.
RAMIN, RAMAX, DECMIN, DECMAX = np.array(RADEC.split(',')).astype('float')

# print()
print('[start: '+datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")+']')
# print()

# reading.
hdu = fitsio.FITS(INFITS)[1]
ra = hdu['RA'][:]
dec = hdu['DEC'][:]

if (RAMAX < RAMIN):
    keep_radec = ((ra > RAMIN) | (ra < RAMAX)) & (dec > DECMIN) & (dec < DECMAX)
else:
    keep_radec = (ra > RAMIN) & (ra < RAMAX) & (dec > DECMIN) & (dec < DECMAX)

ra = ra[keep_radec]
dec = dec[keep_radec]

ramin = np.min(ra)
ramax = np.max(ra)
decmax = np.max(dec)
decmin = np.min(dec)

# QSO.
if (SELCRIT == 'qso'):
    QSO_hdu = fitsio.FITS(fpn_QSO_cat)[1]
    QSO_ra = QSO_hdu['RA'][:]
    QSO_dec = QSO_hdu['DEC'][:]
    if (ramax < ramin):
        QSO_keep_radec = ((QSO_ra > ramin) | (QSO_ra < ramax)) & (QSO_dec > decmin) & (QSO_dec < decmax)
    else:
        QSO_keep_radec = (QSO_ra > ramin) & (QSO_ra < ramax) & (QSO_dec > decmin) & (QSO_dec < decmax)
    if np.any(QSO_keep_radec):
        QSO_ra = QSO_ra[QSO_keep_radec]
        QSO_dec = QSO_dec[QSO_keep_radec]
        RA_DEC_MatchingObj = RA_DEC_MatchingClass()
        RA_DEC_MatchingObj.LoadRA_DEC_CatalogData(ra, dec)
        RA_DEC_MatchingObj.LoadRA_DEC4MatchingData(QSO_ra, QSO_dec)
        RA_DEC_MatchingObj(radius4matching, 1)  # "1" seul voisin le plus proche.
        res = RA_DEC_MatchingObj.nNeighResInd
        valid_res = res[0] > -1
        if np.any(valid_res):  # facultatif.
            hdu_temp = hdu[:][keep_radec][res[0][valid_res]]
            QSO_hdu_temp = QSO_hdu[:][QSO_keep_radec][res[1][valid_res]]
        else:
            hdu_temp = np.zeros(0, dtype=hdu[:].dtype)
            QSO_hdu_temp = np.zeros(0, dtype=QSO_hdu[:].dtype)
    else:
        hdu_temp = np.zeros(0, dtype=hdu[:].dtype)
        QSO_hdu_temp = np.zeros(0, dtype=QSO_hdu[:].dtype)

    newhdu = fitsio.FITS(OUTFITS, 'rw')
    newhdu.write(hdu_temp)
    newhdu[1].insert_column('ra_SDSS', QSO_hdu_temp['RA'])
    newhdu[1].insert_column('dec_SDSS', QSO_hdu_temp['DEC'])
    newhdu[1].insert_column('zred', QSO_hdu_temp['Z'])
    newhdu.close()

# STARS
elif (SELCRIT == 'stars'):
    # ATTENTION EN FONCTION DES RELEASES ILS NE SONT PAS CAPABLES DE
    # GARDER LE MEME NOM DE VARIABLE pour dr8 : 'PSF '.
    keep_PSF = (hdu['TYPE'][:][keep_radec] == 'PSF')
    hdu_temp = hdu[:][keep_radec][keep_PSF]

    # Virer les objets *connus* ET variables.
    if np.any(keep_PSF):
        var_hdu = fitsio.FITS(fpn_var_cat)[1]
        var_ra = var_hdu['RA'][:]
        var_dec = var_hdu['DEC'][:]
        if (ramax < ramin):
            var_keep_radec = ((var_ra > ramin) | (var_ra < ramax)) & (var_dec > decmin) & (var_dec < decmax)
        else:
            var_keep_radec = (var_ra > ramin) & (var_ra < ramax) & (var_dec > decmin) & (var_dec < decmax)
        if np.any(var_keep_radec):
            ra = hdu_temp['RA']
            dec = hdu_temp['DEC']
            var_ra = var_ra[var_keep_radec]
            var_dec = var_dec[var_keep_radec]
            RA_DEC_MatchingObj = RA_DEC_MatchingClass()
            RA_DEC_MatchingObj.LoadRA_DEC_CatalogData(ra, dec)
            RA_DEC_MatchingObj.LoadRA_DEC4MatchingData(var_ra, var_dec)
            RA_DEC_MatchingObj(radius4matching, 1)   # "1" seul voisin le plus proche.
            res = RA_DEC_MatchingObj.nNeighResInd
            valid_res = res[0] > -1
            if np.any(valid_res):
                rej_data_ind = res[0][valid_res]
                var_hdu_temp = var_hdu[:][var_keep_radec][res[1][valid_res]]
                rej_var = var_hdu_temp['NNVariability'] > NNVar_th
                hdu_temp = np.delete(hdu_temp, rej_data_ind[rej_var])

    # Virer les QSO connus.
    if len(hdu_temp) > 0:
        QSO_hdu = fitsio.FITS(fpn_QSO_cat)[1]
        QSO_ra = QSO_hdu['RA'][:]
        QSO_dec = QSO_hdu['DEC'][:]
        if (ramax < ramin):
            QSO_keep_radec = ((QSO_ra > ramin) | (QSO_ra < ramax)) & (QSO_dec > decmin) & (QSO_dec < decmax)
        else:
            QSO_keep_radec = (QSO_ra > ramin) & (QSO_ra < ramax) & (QSO_dec > decmin) & (QSO_dec < decmax)
        if np.any(QSO_keep_radec):
            ra = hdu_temp['RA']
            dec = hdu_temp['DEC']
            QSO_ra = QSO_ra[QSO_keep_radec]
            QSO_dec = QSO_dec[QSO_keep_radec]
            RA_DEC_MatchingObj = RA_DEC_MatchingClass()
            RA_DEC_MatchingObj.LoadRA_DEC_CatalogData(ra, dec)
            RA_DEC_MatchingObj.LoadRA_DEC4MatchingData(QSO_ra, QSO_dec)
            RA_DEC_MatchingObj(radius4matching, 1)  # "1" seul voisin le plus proche.
            res = RA_DEC_MatchingObj.nNeighResInd
            valid_res = res[0] > -1
            rej_data_ind = res[0][valid_res]
            hdu_temp = np.delete(hdu_temp, rej_data_ind)

    newhdu = fitsio.FITS(OUTFITS, 'rw')
    newhdu.write(hdu_temp)
    newhdu.close()

# TEST SAMPLE.
elif (SELCRIT == 'test'):
    hdu_temp = hdu[:][keep_radec]
    # Identifier les QSOs connus.
    QSO_hdu = fitsio.FITS(fpn_QSO_cat)[1]
    QSO_ra = QSO_hdu['RA'][:]
    QSO_dec = QSO_hdu['DEC'][:]
    if (ramax < ramin):
        QSO_keep_radec = ((QSO_ra > ramin) | (QSO_ra < ramax)) & (QSO_dec > decmin) & (QSO_dec < decmax)
    else:
        QSO_keep_radec = (QSO_ra > ramin) & (QSO_ra < ramax) & (QSO_dec > decmin) & (QSO_dec < decmax)
    QSO_hdu_temp = QSO_hdu[:][QSO_keep_radec]
    if np.any(QSO_keep_radec):
        QSO_ra = QSO_ra[QSO_keep_radec]
        QSO_dec = QSO_dec[QSO_keep_radec]
        RA_DEC_MatchingObj = RA_DEC_MatchingClass()
        RA_DEC_MatchingObj.LoadRA_DEC_CatalogData(ra, dec)
        RA_DEC_MatchingObj.LoadRA_DEC4MatchingData(QSO_ra, QSO_dec)
        RA_DEC_MatchingObj(radius4matching, 1)  # "1" seul voisin le plus proche
        res = RA_DEC_MatchingObj.nNeighResInd
        valid_res = res[0] > -1
        sel_data_ind = res[0][valid_res]
        QSO_hdu_temp = QSO_hdu_temp[res[1][valid_res]]
    else:
        sel_data_ind = np.array([]).astype(int)

    zred, ra_SDSS, dec_SDSS = np.zeros(len(hdu_temp))*np.nan, np.zeros(len(hdu_temp))*np.nan, np.zeros(len(hdu_temp))*np.nan
    zred[sel_data_ind], ra_SDSS[sel_data_ind], dec_SDSS[sel_data_ind] = QSO_hdu_temp['Z'], QSO_hdu_temp['RA'], QSO_hdu_temp['DEC']

    newhdu = fitsio.FITS(OUTFITS, 'rw')
    newhdu.write(hdu_temp)
    newhdu[-1].insert_column('ra_SDSS', ra_SDSS)
    newhdu[-1].insert_column('dec_SDSS', dec_SDSS)
    newhdu[-1].insert_column('zred', zred)
    newhdu.close()

# PAR DÉFAUT
else:
    print("[WARNING] WE DO NOTHING BECAUSE SELCRIT is not in the list : [qso, stars, test]")

# print()
print('[end: ' + datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + ']')
# print()

if (LOGFILE != 'none'):
    subprocess.call('echo ' + OUTFITS + ' >> ' + LOGFILE, shell=True)
