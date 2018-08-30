"""
desitarget.cmx.cmx_cuts
========================

`Target Selection for DESI commissioning (cmx) derived from `the wiki`_.

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (*e.g.* STD_DITHER).

.. _`the Gaia data model`: https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
.. _`the wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/CommissioningTargets
"""

from time import time
import numpy as np

import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Row

from desitarget import io
from desitarget.cuts import _psflike
from desitarget.internal import sharedmem
from desitarget.cmx.cmx_targetmask import cmx_mask

#ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

#ADM start the clock
start = time()

def passesSTD_logic(gfracflux=None, rfracflux=None, zfracflux=None,
                    objtype=None, gaia=None, pmra=None, pmdec=None,
                    aen=None, dupsource=None, paramssolved=None,
                    primary=None):
    """The default logic/mask cuts for commissioning stars

    Parameters
    ----------
    gfracflux, rfracflux, zfracflux : :class:`array_like` or :class:`None` 
        Profile-weighted fraction of the flux from other sources divided
        by the total flux in g, r and z bands.
    objtype : :class:`array_like` or :class:`None`
        The Legacy Surveys TYPE to restrict to point sources.
    gaia : :class:`boolean array_like` or :class:`None`
       ``True`` if there is a match between this object in the Legacy
       Surveys and in Gaia.
    pmra, pmdec : :class:`array_like` or :class:`None`
        Gaia-based proper motion in RA and Dec and parallax
        (same units as the Gaia data model).
    aen : :class:`array_like` or :class:`None`
        Gaia Astrometric Excess Noise (as in the Gaia Data Model).
    dupsource : :class:`array_like` or :class:`None`
        Whether the source is a duplicate in Gaia (as in the Gaia Data model).
    paramssolved : :class:`array_like` or :class:`None`
        How many parameters were solved for in Gaia (as in the Gaia Data model).
    primary : :class:`array_like` or :class:`None`
        If given, the ``BRICK_PRIMARY`` column of the catalogue.

    Returns
    -------
    :class:`array_like` 
        True if and only if the object passes the logic cuts for cmx stars.

    Notes
    -----
    - Current version (08/30/18) is version 4 on `the wiki`_.
    - See also `the Gaia data model`_.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
        
    std = primary.copy()

    # ADM A point source with a Gaia match.
    std &= _psflike(objtype)
    std &= gaia

    # ADM An Isolated source.
    fracflux = [gfracflux, rfracflux, zfracflux]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # fracflux can be Inf/NaN.
        for bandint in (0, 1, 2):  # g, r, z.
            std &= fracflux[bandint] < 0.01

    # ADM No obvious issues with the astrometry.
    std &= (aen < 1) & (paramssolved == 31)

    # ADM Finite proper motions.
    std & = np.isfinite(pmra) & np.isfinite(pmdec)

    # ADM Unique source (not a duplicated source).
    std &= ~dupsource

    return std


def isSTD_dither(obs_gflux=None, obs_rflux=None, obs_zflux=None,
                 gfracflux=None, rfracflux=None, zfracflux=None,
                 objtype=None, gaia=None, aen=None, pmra=None, pmdec=None,
                 objtype=None, primary=None):
    """Gaia stars for dithering tests during commissioning

    Parameters
    ----------
    obs_gflux, obs_rflux, obs_zflux : :class:`array_like` or :class:`None`
        The flux in nano-maggies of g, r, z bands WITHOUT any
        Galactic extinction correction.
    gfracflux, rfracflux, zfracflux : :class:`array_like` or :class:`None` 
        Profile-weighted fraction of the flux from other sources divided
        by the total flux in g, r and z bands.
    objtype : :class:`array_like` or :class:`None`
        The Legacy Surveys TYPE to restrict to point sources.
    gaia : :class:`boolean array_like` or :class:`None`
       ``True`` if there is a match between this object in the Legacy
       Surveys and in Gaia.
    pmra, pmdec : :class:`array_like` or :class:`None`
        Gaia-based proper motion in RA and Dec and parallax
        (same units as the Gaia data model).
    aen : :class:`array_like` or :class:`None`
        Gaia Astrometric Excess Noise (as in the Gaia Data Model).
    dupsource : :class:`array_like` or :class:`None`
        Whether the source is a duplicate in Gaia (as in the Gaia Data model).
    paramssolved : :class:`array_like` or :class:`None`
        How many parameters were solved for in Gaia (as in the Gaia Data model).
    primary : :class:`array_like` or :class:`None`
        If given, the ``BRICK_PRIMARY`` column of the catalogue.

    Returns
    -------
    :class:`array_like` 
        True if and only if the object is Gaia dithering target.

    Notes
    -----
    - Current version (08/30/18) is version 4 on `the wiki`_.
    - See also `the Gaia data model`_.       
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
        
    std = primary.copy()
    std &= _psflike(objtype)
    std &= gaia

    bgs &= rflux > 10**((22.5-20.0)/2.5)
    bgs &= rflux <= 10**((22.5-19.5)/2.5)


    return bgs

    isdither &= (gflux > 0)

    return isdither


def apply_cuts(objects):
    """Perform commissioning (cmx) target selection on objects, return target mask arrays

    Parameters
    ----------
    objects: numpy structured array with UPPERCASE columns needed for
        target selection, OR a string tractor/sweep filename

    Returns
    -------
    :class:`~numpy.ndarray`
        commissioning target selection bitmask flags for each object

    See desitarget.cmx.cmx_targetmask.cmx_mask for the definition of each bit
    """
    #- Check if objects is a filename instead of the actual data
    if isinstance(objects, str):
        objects = io.read_tractor(objects)

    #- Ensure uppercase column names if astropy Table
    if isinstance(objects, (Table, Row)):
        for col in list(objects.columns.values()):
            if not col.name.isupper():
                col.name = col.name.upper()

    # ADM Currently only coded for objects with Gaia matches
    # ADM (e.g. DR7 or above). Fail for earlier Data Releases.
    release = objects['RELEASE']
    if release < 7000:
        log.critical('Commissioning cuts only coded for DR7 or above')
        raise ValueError

    # ADM The observed g/r/z fluxes.
    obs_rflux = objects['FLUX_G']
    obs_rflux = objects['FLUX_R']
    obs_rflux = objects['FLUX_Z']

    # ADM The de-extincted g/r/z/ fluxes.
    from desitarget.cuts import unextinct_fluxes
    flux = unextinct_fluxes(objects)
    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']

    # ADM The Legacy Surveys object type and fracflux flags.
    objtype = objects['TYPE']
    gfracflux = objects['FRACFLUX_G']
    rfracflux = objects['FRACFLUX_R']
    zfracflux = objects['FRACFLUX_Z']

    # ADM Add the Gaia columns...
    # ADM if we don't have REF_CAT in the sweeps use the
    # ADM minimum value of REF_ID to identify Gaia sources. This will
    # ADM introduce a small number (< 0.001%) of Tycho-only sources.
    gaia = objects['REF_ID'] > 0
    if "REF_CAT" in objects.dtype.names:
        gaia = (objects['REF_CAT'] == b'G2') | (objects['REF_CAT'] == 'G2')
    pmra = objects['PMRA']
    pmdec = objects['PMDEC']
    pmraivar = objects['PMRA_IVAR']
    gaiaaen = objects['GAIA_ASTROMETRIC_EXCESS_NOISE']
    gaiadupsource = objects['GAIA_DUPLICATED_SOURCE']

    # ADM If proper motion is not NaN, 31 parameters were solved for
    # ADM in Gaia astrometry. Or, gaiaparamssolved should be 3 for NaNs).
    # ADM In the sweeps, NaN has not been preserved...but PMRA_IVAR == 0
    # ADM in the sweeps is equivalent to PMRA of NaN in Gaia.
    if 'GAIA_ASTROMETRIC_PARAMS_SOLVED' in objects.dtype.names:
        gaiaparamssolved = objects['GAIA_ASTROMETRIC_PARAMS_SOLVED']
    else:
        gaiaparamssolved = np.zeros_like(gaia)+31
        w = np.where( np.isnan(pmra) | (pmraivar == 0) )[0]
        if len(w) > 0:
            gaiaparamssolved[w] = 3

    dither = isSTD_dither(gflux=gflux)

    # ADM Construct the targetflag bits
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
