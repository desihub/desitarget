#!/usr/bin/env python

import sys
import subprocess
import numpy as np
import astropy.io.fits as fits


def sweep_meta(release, outfits):
    if (release == 'dr3'):
        sweepdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr3/sweep/3.1'
    if (release == 'dr4'):
        sweepdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr4/sweep/4.0'
    if (release == 'dr5'):
        sweepdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr5/sweep/5.0'
    if (release == 'dr6'):
        sweepdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr6/sweep/6.0'
    if (release == 'dr7'):
        sweepdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr7/sweep/7.1'
    if (release == 'dr8n'):  # BASS/MzLS
        sweepdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/north/sweep/8.0'
    if (release == 'dr8s'):  # DECaLS
        sweepdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/south/sweep/8.0'
    if (release == 'dr9n'):
        sweepdir = '/global/cscratch1/sd/adamyers/dr9m/north/sweep/'
    if (release == 'dr9s'):
        sweepdir = '/global/cscratch1/sd/adamyers/dr9m/south/sweep/'

    # listing the sweep files
    tmpstr = "ls " + sweepdir + "/sweep-???[pm]???-???[pm]???.fits | awk -F \"/\" \"{print $NF}\""
    p1 = subprocess.Popen(tmpstr, stdout=subprocess.PIPE, shell=True)
    sweeplist = np.array(p1.communicate()[0].decode('ascii').split('\n'))[:-1]
    nsweep = len(sweeplist)

    ramin, ramax, decmin, decmax = np.zeros(nsweep), np.zeros(nsweep), np.zeros(nsweep), np.zeros(nsweep)

    for i in range(nsweep):
        sweeplist[i] = sweeplist[i][-26:]
        sweep = sweeplist[i]
        ramin[i] = float(sweep[6:9])
        ramax[i] = float(sweep[14:17])
        if (sweep[9] == 'm'):
            decmin[i] = -1. * float(sweep[10:13])
        else:
            decmin[i] = float(sweep[10:13])
        if (sweep[17] == 'm'):
            decmax[i] = -1. * float(sweep[18:21])
        else:
            decmax[i] = float(sweep[18:21])

    collist = []
    collist.append(fits.Column(name='sweepname', format='26A', array=sweeplist))
    collist.append(fits.Column(name='ramin', format='E', array=ramin))
    collist.append(fits.Column(name='ramax', format='E', array=ramax))
    collist.append(fits.Column(name='decmin', format='E', array=decmin))
    collist.append(fits.Column(name='decmax', format='E', array=decmax))
    cols = fits.ColDefs(collist)
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(outfits, overwrite=True)
