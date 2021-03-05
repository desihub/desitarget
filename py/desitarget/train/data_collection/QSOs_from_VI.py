#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import fitsio

import os

from astropy.coordinates import SkyCoord
from astropy import units as u

SWEEP_south = '/global/homes/e/edmondc/Legacy_survey_sweep/south/sweep/9.0/'
SWEEP_north = '/global/homes/e/edmondc/Legacy_survey_sweep/north/sweep/9.0/'

sweepname_south = SWEEP_south + 'sweep-{}{}{}-{}{}{}.fits'
sweepname_north = SWEEP_north + 'sweep-{}{}{}-{}{}{}.fits'


def build_catalog(list_tiles):
    print(f"[INFO] Read {list_tiles[0]} ...")
    cat = fitsio.FITS(list_tiles[0])[1][:]
    if len(list_tiles) > 1:
        for name in list_tiles[1:]:
            print(f"[INFO] Read {name} ...")
            cat = np.concatenate((cat, np.array(fitsio.FITS(name)[1][:], dtype=cat.dtype)))
    print("[INFO] WE CONSERVE ONLY QSO WITH z <= 5.0")
    sel = (cat['redshift'] <= 5.0)
    return cat[sel]


def plot_tiles(coord_cat):
    ra, dec = coord_cat.ra.degree, coord_cat.dec.degree

    plt.figure(figsize=(11, 7))
    plt.plot(ra[dec<32], dec[dec<32], ls='', marker='.', color='blue', label='sweep south')
    plt.plot(ra[dec>=32], dec[dec>=32], ls='', marker='.', color='red', label='sweep north')

    plt.plot([30, 30, 45, 45, 30], [-5, 5, 5, -5, -5], ls='--', color='black', label='test zone')

    plt.grid()
    plt.legend()
    plt.xticks([i*10 for i in range(3, 25)])
    plt.yticks([i*5 for i in range(-5, 20)])
    plt.xlabel('RA')
    plt.ylabel('Dec')

    plt.show()


def build_ra_dec_list():
    #add sweep containg tiles observed to build the vi_catalog
    # Tile in North
    ra_list_north, dec_list_north = [], []

    ra_list_north += [[90, 100], [90, 100]]
    dec_list_north += [[40, 45], [45, 50]]

    ra_list_north += [[100, 110], [100, 110], [100, 110], [100, 110], [100, 110], [100, 110]]
    dec_list_north += [[30, 35], [35, 40], [40, 45], [45, 50], [50, 55], [55, 60]]

    ra_list_north += [[110, 120], [110, 120], [110, 120], [110, 120]]
    dec_list_north += [[30, 35], [35, 40], [40, 45], [45, 50]]

    ra_list_north += [[120, 130], [120, 130], [120, 130]]
    dec_list_north += [[30, 35], [35, 40], [80, 85]]

    ra_list_north += [[130, 140], [130, 140]]
    dec_list_north += [[30, 35], [80, 85]]

    ra_list_north += [[140, 150], [140, 150], [140, 150], [140, 150]]
    dec_list_north += [[30, 35], [60, 65], [65, 70], [80, 85]]

    ra_list_north += [[150, 160], [150, 160]]
    dec_list_north += [[30, 35], [80, 85]]

    ra_list_north += [[160, 170], [160, 170]]
    dec_list_north += [[30, 35], [80, 85]]

    ra_list_north += [[170, 180]]
    dec_list_north += [[80, 85]]

    ra_list_north += [[210, 220], [210, 220], [210, 220]]
    dec_list_north += [[45, 50], [50, 55], [55, 60]]

    #Tile in South
    ra_list_south, dec_list_south = [], []

    ra_list_south += [[30, 40], [30, 40], [30, 40]]
    dec_list_south += [[-10, -5], [-5, 0], [0, 5]]

    ra_list_south += [[80, 90], [80, 90]]
    dec_list_south += [[-25, -20], [-20, -15]]

    ra_list_south += [[110, 120], [110, 120], [110, 120]]
    dec_list_south += [[10, 15], [10, 15], [30, 35]]

    ra_list_south += [[120, 130], [120, 130]]
    dec_list_south += [[30, 35], [20, 25]]

    ra_list_south += [[130, 140], [130, 140]]
    dec_list_south += [[20, 25], [30, 35]]

    ra_list_south += [[140, 150], [140, 150]]
    dec_list_south += [[0, 5], [30, 35]]

    ra_list_south += [[150, 160], [150, 160]]
    dec_list_south += [[0, 5], [30, 35]]

    ra_list_south += [[160, 170]]
    dec_list_south += [[30, 35]]

    ra_list_south += [[190, 200]]
    dec_list_south += [[25, 30]]

    return ra_list_south, dec_list_south, ra_list_north, dec_list_north


def reorganise(lst):
    for elt in lst:
        elt[1], elt[2], elt[3] = elt[2], elt[3], elt[1]
    return lst


def correct_number(lst):
    for j in range(len(lst)):
        for i in [0, 2, 3, 5]:
            if len(lst[j][i]) == 1:
                lst[j][i] = f'00{lst[j][i]}'
            elif len(lst[j][i]) == 2:
                lst[j][i] = f'0{lst[j][i]}'
    return lst


def build_list_name(ra, dec):
    lst = []
    for i in range(len(ra)):
        ra1, ra2 = str(ra[i][0]), str(ra[i][1])
        dec1, dec2 = dec[i][0], dec[i][1]

        if dec1<0:
            dec1 = str(dec1)[1:]
            sgn1 = 'm'
        else:
            dec1 = str(dec1)
            sgn1 = 'p'
        if dec2<0:
            dec2 = str(dec2)[1:]
            sgn2 = 'm'
        else:
            dec2 = str(dec2)
            sgn2 = 'p'
        lst += [[ra1, ra2, sgn1, dec1, sgn2, dec2]]
    return correct_number(reorganise(lst))


def match_cat_to_dr9(coord_cat, list_name, sweepname):
    #build_structure with juste one element --> remove it  after the loop :D
    qso_dr9 = np.array(fitsio.FITS(sweepname.format('100', 'p', '030', '110', 'p', '035'), 'r')[1][0:2])
    print("\n", qso_dr9.shape)

    for name in list_name:
        sel_in_cat = (coord_cat.ra.degree < float(name[3])) & (coord_cat.ra.degree  > float(name[0]))
        if name[1] == 'm':
            sel_in_cat &=  (coord_cat.dec.degree  > - float(name[2]))
        else:
            sel_in_cat &=  (coord_cat.dec.degree  > float(name[2]))
        if name[4] == 'm':
            sel_in_cat &= (coord_cat.dec.degree  < - float(name[5]))
        else:
            sel_in_cat &= (coord_cat.dec.degree  < float(name[5]))

        if sel_in_cat.sum() != 0:
            print("\n[SWEEP] : ", name)
            print("    * Number of objetcs in this sweep in catalog : ", sel_in_cat.sum())
            sweep = fitsio.FITS(sweepname.format(*name), 'r')['SWEEP']
            coord_sweep = SkyCoord(ra=sweep['RA'][:]*u.degree, dec=sweep['DEC'][:]*u.degree)

            #pas dans l'autre sens ca n'a pas de sens sinon ...
            idx, d2d, d3d = coord_cat[sel_in_cat].match_to_catalog_sky(coord_sweep)

            sel = (d2d.arcsec < 1)
            print("    * Number of objetcs selected in the sweep file : ", sel.sum())

            qso_dr9 = np.concatenate((qso_dr9, sweep[idx[sel]]))
            print(qso_dr9.shape)

    qso_dr9 = qso_dr9[2:]
    print(qso_dr9.shape, "\n")

    return qso_dr9


def find_south_in_sdss(qso_south):
    coord_qso_south = SkyCoord(ra=qso_south['RA'][:]*u.degree, dec=qso_south['DEC'][:]*u.degree)
    sdss_cat = fitsio.FITS('/global/cfs/cdirs/desi/target/analysis/RF/Catalogs/DR16Q_red.fits')[1]
    coord_qso_sdss = SkyCoord(ra=sdss_cat['RA'][:]*u.degree, dec=sdss_cat['DEC'][:]*u.degree)
    idx, d2d, d3d = coord_qso_south.match_to_catalog_sky(coord_qso_sdss)
    sel = (d2d.arcsec >= 1)

    print(f"    * Number of VI QSOs which are overlapping SDSS : {sel.size - sel.sum()}")
    return sel


def find_north_in_south(qso_south, qso_north):
    coord_qso_south = SkyCoord(ra=qso_south['RA'][:]*u.degree, dec=qso_south['DEC'][:]*u.degree)
    coord_qso_north= SkyCoord(ra=qso_north['RA'][:]*u.degree, dec=qso_north['DEC'][:]*u.degree)
    idx, d2d, d3d = coord_qso_north.match_to_catalog_sky(coord_qso_south)
    sel_north = (d2d.arcsec >= 1)
    print("    * Nombre de targets dans North qui sont dans l'overlap du South : ", sel_north.size - sel_north.sum())
    return sel_north


def extract_qsos_from_vi(vi_tiles, fits_save_name):
    cat = build_catalog(vi_tiles)
    coord_cat = SkyCoord(ra=cat['TARGET_RA'][:]*u.degree, dec=cat['TARGET_DEC'][:]*u.degree)
    plot_tiles(coord_cat)

    ra_list_south, dec_list_south, ra_list_north, dec_list_north = build_ra_dec_list()
    list_name_south = build_list_name(ra_list_south, dec_list_south)
    list_name_north = build_list_name(ra_list_north, dec_list_north)
    print(f"[INFO] Number of HXPIXEL in South : {len(list_name_south)}")
    print(f"[INFO] Number of HXPIXEL in North : {len(list_name_north)}")

    qso_dr9_south = match_cat_to_dr9(coord_cat, list_name_south, sweepname_south)
    qso_dr9_north = match_cat_to_dr9(coord_cat, list_name_north, sweepname_north)

    print("Remove targets in South which are already in SDSS (avoid overlap and double check)")
    sel_not_in_sdss = find_south_in_sdss(qso_dr9_south)
    qso_dr9_south = qso_dr9_south[sel_not_in_sdss]

    print("Remove targets in North which are also in South")
    sel_north = find_north_in_south(qso_dr9_south, qso_dr9_north)
    qso_dr9_north = qso_dr9_north[sel_north]

    #new column ISNORTH for build_colors
    is_north = np.concatenate((np.zeros(qso_dr9_south.size, dtype=bool), np.ones(qso_dr9_north.size, dtype=bool)))

    if os.path.isfile(fits_save_name):
        os.remove(fits_save_name)
    fits = fitsio.FITS(fits_save_name, 'rw')
    fits.write(qso_dr9_south)
    fits[-1].append(qso_dr9_north)
    fits[-1].insert_column('IS_NORTH', is_north)

    ## add redshitf :
    coord_qso_vi_in_dr9 = SkyCoord(ra=fits[-1]['RA'][:]*u.degree, dec=fits[-1]['DEC'][:]*u.degree)
    idx, d2d, d3d = coord_cat.match_to_catalog_sky(coord_qso_vi_in_dr9)
    zred = cat['redshift'][:][idx]
    zred = np.array(zred, dtype='f8') #need this to add this qso to the Training Catalog.
    fits[-1].insert_column('zred', zred)

    fits.close()


def sdss_vi_merger(fits_file_1, fits_file_2, fits_file_output):
    if os.path.isfile(fits_file_output):
        os.remove(fits_file_output)
    fits = fitsio.FITS(fits_file_output, 'rw')

    fits_1 = fitsio.FITS(fits_file_1, 'r')[1]
    fits_2 = fitsio.FITS(fits_file_2, 'r')[1]

    print('Nbr QSOs from SDSS :', fits_1[:].size)
    print('Nbr QSOs from VI :', fits_2[:].size)

    fits.write(fits_1[:])
    fits[-1].append(fits_2[:])
    print('Nbr QSOs for training :', fits[1][:].size)
    fits.close()


def add_qso_vi_test(fits_file_1, fits_file_2, fits_file_output, zone_test):
    fits_1 = fitsio.FITS(fits_file_1, 'r')[1]
    fits_2 = fitsio.FITS(fits_file_2, 'r')[1]

    sel = (fits_2['RA'][:] >= zone_test[0]) & (fits_2['RA'][:] <= zone_test[1]) & (fits_2['DEC'][:] >= zone_test[2]) & (fits_2['DEC'][:] <= zone_test[3])

    print('Nbr new quasars in Test zone : ', sel.sum())

    if os.path.isfile(fits_file_output):
        os.remove(fits_file_output)
    fits = fitsio.FITS(fits_file_output, 'rw')

    fits.write(fits_1[:])
    fits[-1].append(fits_2[:][sel])
    fits.close()
