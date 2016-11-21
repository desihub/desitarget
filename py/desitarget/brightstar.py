# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=====================
desitarget.brightstar
=====================

Module for studying and masking bright stars in the sweeps
"""
from __future__ import (absolute_import, division)
#
from time import time
import numpy as np
import fitsio
from glob import glob
from astropy.coordinates import SkyCoord
from astropy import units as u

from . import __version__ as desitarget_version
from . import gitversion

from desiutil import depend
from desitarget import io
from desitarget.internal import sharedmem

def collect_bright_stars(band,maglim,numproc=4,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1',outfilename=False,verbose=True):
    """Extract a structure from the sweeps containing only bright stars in a given band to a given magnitude limit

    Parameters
    ----------
    band : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z"
    maglim : :class:`float`
        The upper limit in that magnitude band for which to assemble a list of bright stars
    numproc : :class:`int`, optional
        Number of processes over which to parallelize
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory containing either sweeps or tractor files...e.g. for dr3 this might be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3/sweeps/dr3.1
    outdirname : :class:`str`, optional, defaults to not writing anything to file
        File name to which to write the output structure of bright stars
    verbose : :class:`bool`, optional
        Send to write progress to screen

    Returns
    -------
    :class:`recarray`
        The structure of bright stars from the sweeps limited in the passed band to the
        passed maglim
    """

    #ADM use io.py to retrieve list of sweeps or tractor files
    infiles = io.list_sweepfiles(rootdirname)
    if len(infiles) == 0:
        infiles = io.list_tractorfiles(rootdirname)
    if len(infiles) == 0:
        print('FATAL: no sweep or tractor files found in {}'.format(rootdirname))
        sys.exit(1)

    #ADM set band to uppercase if passed as lower case
    band = band.upper()
    #ADM the band as an integer location
    bandint = "UGRIZY".find(band)

    #ADM change input magnitude to a flux to test against
    fluxlim = 10.**((22.5-maglim)/2.5)

    #ADM parallel formalism from this step forward is stolen cuts.select_targets

    #ADM function to grab the bright stars from a given file
    def _get_bright_stars(filename):
        '''Retrieves bright stars from a sweeps/Tractor file'''
        objs = io.read_tractor(filename)
        w = np.where(objs["DECAM_FLUX"][...,bandint] > fluxlim)
        if len(w[0]) > 0:
            return objs[w]

    #ADM counter for how many files have been processed
    #ADM critical to use np.ones because a numpy scalar allows in place modifications
    # c.f https://www.python.org/dev/peps/pep-3104/
    totfiles = np.ones((),dtype='i8')*len(infiles)
    nfiles = np.ones((), dtype='i8')
    t0 = time()
    def _update_status(result):
        '''wrapper function for the critical reduction operation,
        that occurs on the main parallel process'''
        if verbose and nfiles%25 == 0:
            elapsed = time() - t0
            rate = nfiles / elapsed
            print('{}/{} files; {:.1f} files/sec; {:.1f} total mins elapsed'.format(nfiles, totfiles, rate, elapsed/60.))
        nfiles[...] += 1  #this is an in-place modification
        return result

    #ADM did we ask to parallelize, or not?
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            starstruc = pool.map(_get_bright_stars, infiles, reduce=_update_status)
    else:
        starstruc = []
        for file in infiles:
            starstruc.append(_update_status(_get_bright_stars(file)))

    #ADM note that if there were no bright stars in a file then
    #ADM the _get_bright_stars function will have returned NoneTypes
    #ADM so we need to filter those out
    starstruc = [x for x in starstruc if x != None]
    #ADM concatenate all of the output recarrays
    starstruc = np.hstack(starstruc)

    #ADM if the name of a file for output is passed, then write to it
    if outfilename:
        fitsio.write(outfilename, starstruc, clobber=True)

    return starstruc


def model_bright_stars(band,instarfile,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/'):

    """Build a dictionary of the fraction of bricks containing a star of a given 
    magnitude in a given band as function of Galactic l and b

    Parameters
    ----------
    band : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z"
    instarfile : :class:`str`
        File of bright objects in (e.g.) sweeps, created by collect_bright_stars
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory for a data release...e.g. for dr3 this would be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/

    Returns
    -------
    :class:`recarray`
        dictionary of the fraction of bricks containing a star of a given
        magnitude in a given band as function of Galactic l Keys are mag
        bin CENTERS, values are arrays running from 0->1 to 359->360
    :class:`recarray`
        dictionary of the fraction of bricks containing a star of a given
        magnitude in a given band as function of Galactic b. Keys are mag
        bin CENTERS, values are arrays running from -90->-89 to 89->90

    Notes
    -----
    converts using coordinates of the brick center, so is an approximation

    """
    #ADM histogram bin edges in Galactic coordinates at resolution of 1 degree
    lbinedges = np.arange(361)
    bbinedges = np.arange(-90,91)

    #ADM set band to uppercase if passed as lower case
    band = band.upper()
    #ADM the band as an integer location
    bandint = "UGRIZY".find(band)

    #ADM read in the bright object file
    fx = fitsio.FITS(instarfile)
    objs = fx[1].read()
    #ADM convert fluxes in band of interest for each object to magnitudes
    mags = 22.5-2.5*np.log10(objs["DECAM_FLUX"][...,bandint])
    #ADM Galactic l and b for each object of interest
    c = SkyCoord(objs["RA"]*u.degree, objs["DEC"]*u.degree, frame='icrs')
    lobjs = c.galactic.l.degree
    bobjs = c.galactic.b.degree

    #ADM construct histogram bin edges in magnitude in passed band
    magstep = 0.1
    magmin = -1.5 #ADM magnitude of Sirius to 1 d.p.
    magmax = np.max(mags)
    magbinedges = np.arange(np.rint((magmax-magmin)/magstep))*magstep+magmin

    #ADM read in the data-release specific brick information file
    fx = fitsio.FITS(glob(rootdirname+'/survey-bricks-dr*.fits.gz')[0], upper=True)
    bricks = fx[1].read(columns=['RA','DEC'])

    #ADM convert RA/Dec of the brick center to Galatic coordinates and
    #ADM build a histogram of the number of bins at each coordinate...
    #ADM using the center is imperfect, so this is approximate at best
    c = SkyCoord(bricks["RA"]*u.degree, bricks["DEC"]*u.degree, frame='icrs')
    lhistobrick = (np.histogram(c.galactic.l.degree,bins=lbinedges))[0]
    bhistobrick = (np.histogram(c.galactic.b.degree,bins=bbinedges))[0]

    #ADM loop through the magnitude bins and populate a dictionary
    #ADM of the number of stars in this magnitude range per brick
    ldict = {}
    bdict = {}
    for mag in magbinedges:
        key = "{:.2f}".format(mag+(0.5*magstep))
        #ADM range in magnitude
        w = np.where( (mags >= mag) & (mags < mag+magstep) )
        if len(w[0]):
            #ADM histograms of numbers of objects in l, b
            lhisto = (np.histogram(lobjs[w],bins=lbinedges))[0]
            bhisto = (np.histogram(bobjs[w],bins=bbinedges))[0]
            #ADM fractions of objects in l, b per brick
            #ADM use a where so that 0/0 results in 0
            lfrac = np.where(lhistobrick > 0, lhisto/lhistobrick, 0)
            bfrac = np.where(bhistobrick > 0, bhisto/bhistobrick, 0)
            #ADM populate the dictionaries
            ldict[key], bdict[key] = lfrac, bfrac

    return ldict, bdict


