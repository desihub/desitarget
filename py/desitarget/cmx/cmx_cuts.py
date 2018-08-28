"""
desitarget.cmx.cmx_cuts
========================

Target Selection for DESI commissioning (cmx)

https://desi.lbl.gov/trac/wiki/TargetSelectionWG/CommissioningTargets

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (*e.g.* STD_DITHER).
"""
import warnings
from time import time
import os.path

import numbers
import sys

import numpy as np
from pkg_resources import resource_filename

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Row

from desitarget import io
from desitarget.internal import sharedmem
from desitarget.cmx.cmx_targetmask import cmx_mask

from desitarget.gaiamatch import match_gaia_to_primary, pop_gaia_coords

#ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

#ADM start the clock
start = time()


def isSTD_dither(gflux=None):
    """Placeholder for Gaia dithering targets
       
    Args:
        gflux
            The flux in nano-maggies of g
    
    Returns:
        mask : array_like. True if and only if the object is Gaia
            dithering target.
    """

    isdither &= (gflux > 0)

    return isdither


def apply_cuts(objects, gaiamatch=False,
               gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Perform commissioning (cmx) target selection on objects, return target mask arrays

    Args:
        objects: numpy structured array with UPPERCASE columns needed for
            target selection, OR a string tractor/sweep filename

    Options:
        gaiamatch : defaults to ``False``
            if ``True``, match to Gaia DR2 chunks files and populate 
            Gaia columns to facilitate the MWS selection
        gaiadir : defaults to the the Gaia DR2 path at NERSC
             Root directory of a Gaia Data Release as used by the Legacy Surveys. 

    Returns:
        desi_target, where each element is
        an ndarray of target selection bitmask flags for each object

    See desitarget.cmx.cmx_targetmask for the definition of each bit
    """
    #- Check if objects is a filename instead of the actual data
    if isinstance(objects, str):
        objects = io.read_tractor(objects)

    #ADM add Gaia information, if requested, and if we're going to actually
    #ADM process the target classes that need Gaia columns
    if gaiamatch:
        log.info('Matching Gaia to {} primary objects...t = {:.1f}s'
                 .format(len(objects),time()-start))
        gaiainfo = match_gaia_to_primary(objects, gaiadir=gaiadir)
        log.info('Done with Gaia match for {} primary objects...t = {:.1f}s'
                 .format(len(objects),time()-start))
        #ADM remove the GAIA_RA, GAIA_DEC columns as they aren't
        #ADM in the imaging surveys data model
        gaiainfo = pop_gaia_coords(gaiainfo)
        #ADM add the Gaia column information to the primary array
        for col in gaiainfo.dtype.names:
            objects[col] = gaiainfo[col]

    #- ensure uppercase column names if astropy Table
    if isinstance(objects, (Table, Row)):
        for col in list(objects.columns.values()):
            if not col.name.isupper():
                col.name = col.name.upper()

    #ADM flag whether we're using northen (BASS/MZLS) or
    #ADM southern (DECaLS) photometry
    photsys_north = _isonnorthphotsys(objects["PHOTSYS"])
    photsys_south = ~_isonnorthphotsys(objects["PHOTSYS"])

    #ADM the observed r-band flux
    #ADM make copies of values that we may reassign due to NaNs
    obs_rflux = objects['FLUX_R']

    #- undo Milky Way extinction
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']
    release = objects['RELEASE']

    gfluxivar = objects['FLUX_IVAR_G']
    rfluxivar = objects['FLUX_IVAR_R']
    zfluxivar = objects['FLUX_IVAR_Z']

    gnobs = objects['NOBS_G']
    rnobs = objects['NOBS_R']
    znobs = objects['NOBS_Z']

    gfracflux = objects['FRACFLUX_G']
    rfracflux = objects['FRACFLUX_R']
    zfracflux = objects['FRACFLUX_Z']

    gfracmasked = objects['FRACMASKED_G']
    rfracmasked = objects['FRACMASKED_R']
    zfracmasked = objects['FRACMASKED_Z']

    gallmask = objects['ALLMASK_G']
    rallmask = objects['ALLMASK_R']
    zallmask = objects['ALLMASK_Z']

    gsnr = objects['FLUX_G'] * np.sqrt(objects['FLUX_IVAR_G'])
    rsnr = objects['FLUX_R'] * np.sqrt(objects['FLUX_IVAR_R'])
    zsnr = objects['FLUX_Z'] * np.sqrt(objects['FLUX_IVAR_Z'])
    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    w2snr = objects['FLUX_W2'] * np.sqrt(objects['FLUX_IVAR_W2'])

    #ADM issue a warning if gaiamatch was not sent but there's no Gaia information
    if np.max(objects['PARALLAX']) == 0. and ~gaiamatch:
        log.warning("Zero objects have a parallax. Did you mean to send gaiamatch?")

    #ADM add the Gaia columns
    gaia = objects['REF_ID'] != -1
    pmra = objects['PMRA']
    pmdec = objects['PMDEC']
    parallax = objects['PARALLAX']
    parallaxivar = objects['PARALLAX_IVAR']
    #ADM derive the parallax/parallax_error, but set to 0 where the error is bad
    parallaxovererror = np.where(parallaxivar > 0., parallax*np.sqrt(parallaxivar), 0.)
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    gaiabmag = objects['GAIA_PHOT_BP_MEAN_MAG']
    gaiarmag = objects['GAIA_PHOT_RP_MEAN_MAG']
    gaiaaen = objects['GAIA_ASTROMETRIC_EXCESS_NOISE']
    gaiadupsource = objects['GAIA_DUPLICATED_SOURCE']

    #ADM if the RA proper motion is not NaN, then 31 parameters were solved for
    #ADM in Gaia astrometry. Use this to set gaiaparamssolved (value is 3 for NaNs)
    gaiaparamssolved = np.zeros_like(gaia)+31
    w = np.where(np.isnan(pmra))[0]
    if len(w) > 0:
        gaiaparamssolved[w] = 3

    #ADM test if these columns exist, as they aren't in the Tractor files as of DR7
    gaiabprpfactor = None
    gaiasigma5dmax = None
    try:
        gaiabprpfactor = objects['GAIA_PHOT_BP_RP_EXCESS_FACTOR']
        gaiasig5dmax = objects['GAIA_ASTROMETRIC_SIGMA5D_MAX']
    except:
        pass

    #ADM Mily Way Selection requires Galactic b
    _, galb = _gal_coords(objects["RA"],objects["DEC"])

    dither = isSTD_dither(gflux=gflux)

    # Construct the targetflag bits for DECaLS (i.e. South)
    # This should really be refactored into a dedicated function.
    cmx_target  = dither * cmx_mask.STD_GAIA

    return cmx_target


def select_targets(infiles, numproc=4, gaiamatch=False,
            gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Process input files in parallel to select commissioning (cmx) targets

    Parameters
    ----------
    infiles : :class:`list` or `str` 
        A list of input filenames (tractor or sweep files) OR a single filename
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use
    gaiamatch : :class:`boolean`, optional, defaults to ``False``
        If ``True``, match to Gaia DR2 chunks files and populate Gaia columns
    gaiadir : :class:`str`, optional, defaults to Gaia DR2 path at NERSC
        Root directory of a Gaia Data Release as used by the Legacy Surveys.

    Returns
    -------   
    :class:`~numpy.ndarray`
        The subset of input targets which pass the cuts, including extra
        columns for `DESI_TARGET`

    Notes
    -----
        - if numproc==1, use serial code instead of parallel
    """
    from desiutil.log import get_logger
    log = get_logger()

    #- Convert single file to list of files
    if isinstance(infiles,str):
        infiles = [infiles,]

    #- Sanity check that files exist before going further
    for filename in infiles:
        if not os.path.exists(filename):
            raise ValueError("{} doesn't exist".format(filename))

    def _finalize_targets(objects, desi_target):
        #- desi_target includes BGS_ANY and MWS_ANY, so we can filter just
        #- on desi_target != 0
        keep = (desi_target != 0)
        objects = objects[keep]
        desi_target = desi_target[keep]

        #- Add *_target mask columns
        #ADM note that only desi_target is defined for commissioning
        #ADM so just pass that around
        targets = desitarget.targets.finalize(
            objects, desi_target, desi_target, desi_target)

        return io.fix_tractor_dr1_dtype(targets)

    #- functions to run on every brick/sweep file
    def _select_targets_file(filename):
        '''Returns targets in filename that pass the cuts'''
        objects = io.read_tractor(filename)
        desi_target, bgs_target, mws_target = apply_cuts(objects,gaiamatch,gaiadir)

        return _finalize_targets(objects, desi_target, desi_target, desi_target)

    # Counter for number of bricks processed;
    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    t0 = time()
    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if nbrick%50 == 0 and nbrick>0:
            rate = nbrick / (time() - t0)
            log.info('{} files; {:.1f} files/sec'.format(nbrick, rate))

        nbrick[...] += 1    # this is an in-place modification
        return result

    #- Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            targets = pool.map(_select_targets_file, infiles, reduce=_update_status)
    else:
        targets = list()
        for x in infiles:
            targets.append(_update_status(_select_targets_file(x)))

    targets = np.concatenate(targets)

    return targets
