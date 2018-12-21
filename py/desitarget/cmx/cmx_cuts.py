"""
desitarget.cmx.cmx_cuts
========================

Target Selection for DESI commissioning (cmx) derived from `the cmx wiki`_.

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (*e.g.* STD_TEST).

.. _`the Gaia data model`: https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
.. _`the cmx wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/CommissioningTargets
"""

from time import time
import numpy as np
import os
import fitsio
import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Row

from desitarget import io
from desitarget.cuts import _psflike, _is_row, _get_colnames
from desitarget.cuts import _prepare_optical_wise, _prepare_gaia

from desitarget.internal import sharedmem
from desitarget.targets import finalize
from desitarget.cmx.cmx_targetmask import cmx_mask

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()


def _get_cmxdir(cmxdir=None):
    """Retrieve the base cmx directory with appropriate error checking.

    cmxdir : :class:`str`, optional, defaults to :envvar:`CMX_DIR`
        Directory in which to find commmissioning files to which to match, such as the
        CALSPEC stars. If not specified, the cmx directory is taken to be the value of
        the :envvar:`CMX_DIR` environment variable.
    """
    # ADM if cmxdir was not passed, default to environment variable
    if cmxdir is None:
        cmxdir = os.environ.get('CMX_DIR')
    # ADM fail if the cmx directory is not set or passed.
    if not os.path.exists(cmxdir):
        log.info('pass cmxdir or correctly set the $CMX_DIR environment variable...')
        msg = 'Commissioning files not found in {}'.format(cmxdir)
        log.critical(msg)
        raise ValueError(msg)

    return cmxdir


def passesSTD_logic(gfracflux=None, rfracflux=None, zfracflux=None,
                    objtype=None, gaia=None, pmra=None, pmdec=None,
                    aen=None, dupsource=None, paramssolved=None,
                    primary=None):
    """The default logic/mask cuts for commissioning stars.

    Parameters
    ----------
    gfracflux, rfracflux, zfracflux : :class:`array_like` or :class:`None`
        Profile-weighted fraction of the flux from other sources divided
        by the total flux in g, r and z bands.
    objtype : :class:`array_like` or :class:`None`
        The Legacy Surveys `TYPE` to restrict to point sources.
    gaia : :class:`boolean array_like` or :class:`None`
        ``True`` if there is a match between this object in the Legacy
        Surveys and in Gaia.
    pmra, pmdec : :class:`array_like` or :class:`None`
        Gaia-based proper motion in RA and Dec.
    aen : :class:`array_like` or :class:`None`
        Gaia Astrometric Excess Noise.
    dupsource : :class:`array_like` or :class:`None`
        Whether the source is a duplicate in Gaia.
    paramssolved : :class:`array_like` or :class:`None`
        How many parameters were solved for in Gaia.
    primary : :class:`array_like` or :class:`None`
        ``True`` for objects that should be passed through the selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object passes the logic cuts for cmx stars.

    Notes
    -----
    - Current version (08/30/18) is version 4 on `the cmx wiki`_.
    - All Gaia quantities are as in `the Gaia data model`_.
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
        warnings.simplefilter('ignore')  # fracflux can be Inf/NaN.
        for bandint in (0, 1, 2):  # g, r, z.
            std &= fracflux[bandint] < 0.01

    # ADM No obvious issues with the astrometry.
    std &= (aen < 1) & (paramssolved == 31)

    # ADM Finite proper motions.
    std &= np.isfinite(pmra) & np.isfinite(pmdec)

    # ADM Unique source (not a duplicated source).
    std &= ~dupsource

    return std


def isSV0_STD_bright(gflux=None, rflux=None, zflux=None,
                     pmra=None, pmdec=None, parallax=None,
                     gaiagmag=None, isgood=None, primary=None):
    """A selection that resembles bright STD stars for initial SV.

    Parameters
    ----------
    gflux, rflux, zflux : :class:`array_like` or :class:`None`
        Galactic-extinction-corrected flux in nano-maggies in g, r, z bands.
    pmra, pmdec, parallax : :class:`array_like` or :class:`None`
        Gaia-based proper motion in RA and Dec, and parallax
        (same units as `the Gaia data model`_).
    gaiagmag : :class:`array_like` or :class:`None`
        Gaia-based g MAGNITUDE (not Galactic-extinction-corrected).
    isgood : :class:`array_like` or :class:`None`
        ``True`` for objects that pass the logic cuts in
        :func:`~desitarget.cmx.cmx_cuts.passesSTD_logic`.
    primary : :class:`array_like` or :class:`None`
        ``True`` for objects that should be passed through the selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a cmx "bright standard" target.

    Notes
    -----
    - See also `the Gaia data model`_.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    isbright = primary.copy()
    # ADM passes all of the default logic cuts.
    isbright &= isgood

    # ADM the STD color cuts from the main survey.
    # Clip to avoid warnings from negative numbers.
    # ADM we're pretty bright for the STDs, so this should be safe.
    gflux = gflux.clip(1e-16)
    rflux = rflux.clip(1e-16)
    zflux = zflux.clip(1e-16)

    # ADM optical colors for halo TO or bluer.
    grcolor = 2.5 * np.log10(rflux / gflux)
    rzcolor = 2.5 * np.log10(zflux / rflux)
    isbright &= rzcolor < 0.2
    isbright &= grcolor > 0.
    isbright &= grcolor < 0.35

    # ADM Gaia magnitudes in the "bright" range (15 < G < 18).
    isbright &= gaiagmag >= 15.
    isbright &= gaiagmag < 18.

    # ADM a parallax smaller than 1 mas.
    isbright &= parallax < 1.

    # ADM a proper motion larger than 2 mas/yr.
    pm = np.sqrt(pmra**2. + pmdec**2.)
    isbright &= pm > 2.

    return isbright


def isSV0_BGS(rflux=None, objtype=None, primary=None):
    """Initial SV-like Bright Galaxy Survey selection (for MzLS/BASS imaging).

    Parameters
    ----------
    rflux : :class:`array_like` or :class:`None`
        Galactic-extinction-corrected flux in nano-maggies in r-band.
    objtype : :class:`array_like` or :class:`None`
        The Legacy Surveys `TYPE`.
    primary : :class:`array_like` or :class:`None`
        ``True`` for objects that should be passed through the selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is an initial BGS target for SV.

    Notes
    -----
    - Returns the equivalent of ALL BGS classes (for the northern imaging).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    isbgs = primary.copy()

    # ADM simple selection is objects brighter than r of 20...
    isbgs &= rflux > 10**((22.5-20.0)/2.5)
    # ADM ...that are not point-like.
    isbgs &= ~_psflike(objtype)

    return isbgs


def isSV0_MWS(rflux=None, obs_rflux=None, objtype=None,
              gaiagmag=None, gaiabmag=None, gaiarmag=None,
              pmra=None, pmdec=None, parallax=None, parallaxovererror=None,
              photbprpexcessfactor=None, astrometricsigma5dmax=None,
              galb=None, gaia=None, primary=None):
    """Initial SV-like Milky Way Survey selection (for MzLS/BASS imaging).

    Parameters
    ----------
    rflux, obs_rflux : :class:`array_like` or :class:`None`
        Flux in nano-maggies in r-band, with (`rflux`) and
        without (`obs_rflux`) Galactic extinction correction.
    objtype : :class:`array_like` or :class:`None`
        The Legacy Surveys `TYPE` to restrict to point sources.
    gaiagmag, gaiabmag, gaiarmag : :class:`array_like` or :class:`None`
        Gaia-based g-, b- and r-band MAGNITUDES.
    pmra, pmdec, parallax : :class:`array_like` or :class:`None`
        Gaia-based proper motion in RA and Dec, and parallax.
    parallaxovererror : :class:`array_like` or :class:`None`
        Gaia-based parallax/error.
    photbprpexcessfactor : :class:`array_like` or :class:`None`
        Gaia_based BP/RP excess factor.
    astrometricsigma5dmax : :class:`array_like` or :class:`None`
        Longest semi-major axis of 5-d error ellipsoid.
    galb: : :class:`array_like` or :class:`None`
        Galactic latitude (degrees).
    gaia : :class:`boolean array_like` or :class:`None`
        ``True`` if there is a match between this object in the Legacy
        Surveys and in Gaia.
    primary : :class:`array_like` or :class:`None`
        ``True`` for objects that should be passed through the selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is an initial MWS target for SV.

    Notes
    -----
    - Returns the equivalent of ALL MWS classes (for the northern imaging).
    - All Gaia quantities are as in `the Gaia data model`_.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    ismws = primary.copy()
    isnear = primary.copy()
    iswd = primary.copy()

    # ADM apply the selection for all MWS-MAIN targets.
    # ADM main targets match to a Gaia source.
    ismws &= gaia
    # ADM main targets are point-like.
    ismws &= _psflike(objtype)
    # ADM main targets are 16 <= r < 19.
    ismws &= rflux > 10**((22.5-19.0)/2.5)
    ismws &= rflux <= 10**((22.5-16.0)/2.5)
    # ADM main targets are robs < 20.
    ismws &= obs_rflux > 10**((22.5-20.0)/2.5)

    # ADM apply the selection for MWS-NEARBY targets.
    # ADM must be a Legacy Surveys object that matches a Gaia source.
    isnear &= gaia
    # ADM Gaia G mag of less than 20.
    isnear &= gaiagmag < 20.
    # ADM parallax cut corresponding to 100pc.
    isnear &= parallax > 10.
    # ADM NOTE TO THE MWS GROUP: There is no bright cut on G. IS THAT THE REQUIRED BEHAVIOR?

    # ADM apply the selection for MWS-WD targets.
    # ADM must be a Legacy Surveys object that matches a Gaia source.
    iswd &= gaia
    # ADM Gaia G mag of less than 20.
    iswd &= gaiagmag < 20.
    # ADM Galactic b at least 20o from the plane.
    iswd &= np.abs(galb) > 20.
    # ADM gentle cut on parallax significance.
    iswd &= parallaxovererror > 1.
    # ADM Color/absolute magnitude cuts of (defining the WD cooling sequence):
    # ADM Gabs > 5.
    # ADM Gabs > 5.93 + 5.047(Bp-Rp).
    # ADM Gabs > 6(Bp-Rp)3 - 21.77(Bp-Rp)2 + 27.91(Bp-Rp) + 0.897
    # ADM Bp-Rp < 1.7
    Gabs = gaiagmag+5.*np.log10(parallax.clip(1e-16))-10.
    br = gaiabmag - gaiarmag
    iswd &= Gabs > 5.
    iswd &= Gabs > 5.93 + 5.047*br
    iswd &= Gabs > 6*br*br*br - 21.77*br*br + 27.91*br + 0.897
    iswd &= br < 1.7
    # ADM Finite proper motion to reject quasars.
    pm = np.sqrt(pmra**2. + pmdec**2.)
    iswd &= pm > 2.

    # ADM As of DR7, photbprpexcessfactor and astrometricsigma5dmax are not in the
    # ADM imaging catalogs. Until they are, ignore these cuts.
    if photbprpexcessfactor is not None:
        # ADM remove problem objects, which often have bad astrometry.
        iswd &= photbprpexcessfactor < 1.7 + 0.06*br*br

    if astrometricsigma5dmax is not None:
        # ADM Reject white dwarfs that have really poor astrometry while.
        # ADM retaining white dwarfs that only have relatively poor astrometry.
        iswd &= ((astrometricsigma5dmax < 1.5) |
                 ((astrometricexcessnoise < 1.) & (parallaxovererror > 4.) & (pm > 10.)))

    # ADM return any object that passes any of the MWS cuts.
    return ismws | isnear | iswd


def isSTD_dither(obs_gflux=None, obs_rflux=None, obs_zflux=None,
                 isgood=None, primary=None):
    """Gaia stars for dithering tests during commissioning.

    Parameters
    ----------
    obs_gflux, obs_rflux, obs_zflux : :class:`array_like` or :class:`None`
        The flux in nano-maggies of g, r, z bands WITHOUT any
        Galactic extinction correction.
    isgood : :class:`array_like` or :class:`None`
        ``True`` for objects that pass the logic cuts in
        :func:`~desitarget.cmx.cmx_cuts.passesSTD_logic`.
    primary : :class:`array_like` or :class:`None`
        ``True`` for objects that should be passed through the selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a Gaia "dither" target.
    :class:`array_like`
        A priority shift of 10*(25-rmag) based on r-band magnitude.

    Notes
    -----
    - Current version (08/30/18) is version 4 on `the cmx wiki`_.
    """
    if primary is None:
        primary = np.ones_like(obs_rflux, dtype='?')

    isdither = primary.copy()
    # ADM passes all of the default logic cuts.
    isdither &= isgood

    # ADM not too bright in g, r, z (> 15 mags).
    isdither &= obs_gflux < 10**((22.5-15.0)/2.5)
    isdither &= obs_rflux < 10**((22.5-15.0)/2.5)
    isdither &= obs_zflux < 10**((22.5-15.0)/2.5)

    # ADM prioritize based on magnitude.
    # ADM OK to clip, as these are all Gaia matches.
    rmag = 22.5-2.5*np.log10(obs_rflux.clip(1e-16))
    prio = np.array((10*(25-rmag)).astype(int))

    return isdither, prio


def isSTD_test(obs_gflux=None, obs_rflux=None, obs_zflux=None,
               isgood=None, primary=None):
    """Very bright Gaia stars for early commissioning tests.

    Parameters
    ----------
    obs_gflux, obs_rflux, obs_zflux : :class:`array_like` or :class:`None`
        The flux in nano-maggies of g, r, z bands WITHOUT any
        Galactic extinction correction.
    isgood : :class:`array_like` or :class:`None`
        ``True`` for objects that pass the logic cuts in
        :func:`~desitarget.cmx.cmx_cuts.passesSTD_logic`.
    primary : :class:`array_like` or :class:`None`
        ``True`` for objects that should be passed through the selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a Gaia "test" target.

    Notes
    -----
    - Current version (08/30/18) is version 4 on `the cmx wiki`_.
    - See also `the Gaia data model`_.
    """
    if primary is None:
        primary = np.ones_like(obs_rflux, dtype='?')

    istest = primary.copy()
    # ADM passes all of the default logic cuts.
    istest &= isgood

    # ADM not too bright in g, r, z (> 13 mags)
    istest &= obs_gflux < 10**((22.5-13.0)/2.5)
    istest &= obs_rflux < 10**((22.5-13.0)/2.5)
    istest &= obs_zflux < 10**((22.5-13.0)/2.5)
    # ADM but brighter than dither targets in g (g < 15).
    istest &= obs_gflux > 10**((22.5-15.0)/2.5)

    return istest


def isSTD_calspec(ra=None, dec=None, cmxdir=None, matchrad=1.,
                  primary=None):
    """Match to CALSPEC stars for commissioning tests.

    Parameters
    ----------
    ra, dec : :class:`array_like` or :class:`None`
        Right Ascension and Declination in degrees.
    cmxdir : :class:`str`, optional, defaults to :envvar:`CMX_DIR`
        Directory in which to find commmissioning files to which to match, such as the
        CALSPEC stars. If not specified, the cmx directory is taken to be the value of
        the :envvar:`CMX_DIR` environment variable.
    matchrad : :class:`float`, optional, defaults to 1 arcsec
        The matching radius in arcseconds.
    primary : :class:`array_like` or :class:`None`
        ``True`` for objects that should be passed through the selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a "CALSPEC" target.
    """
    if primary is None:
        primary = np.ones_like(ra, dtype='?')

    iscalspec = primary.copy()

    # ADM retrieve/check the cmxdir.
    cmxdir = _get_cmxdir(cmxdir)
    # ADM get the CALSPEC objects.
    cmxfile = os.path.join(cmxdir, 'calspec.fits')
    cals = io.read_external_file(cmxfile)

    # ADM match the calspec and sweeps objects.
    calmatch = np.zeros_like(primary, dtype='?')
    cobjs = SkyCoord(ra, dec, unit='degree')
    ccals = SkyCoord(cals['RA'], cals["DEC"], unit='degree')

    # ADM make sure to catch the case of a single sweeps object being passed.
    if cobjs.size == 1:
        sep = cobjs.separation(ccals)
        # ADM set matching objects to True.
        calmatch = np.any(sep < matchrad*u.arcsec)
    else:
        # ADM This triggers a (non-malicious) Cython RuntimeWarning on search_around_sky:
        # RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
        # ADM Caused by importing a scipy compiled against an older numpy than is installed?
        # e.g. stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idobjs, idcals, _, _ = ccals.search_around_sky(cobjs, matchrad*u.arcsec)
        # ADM set matching objects to True.
        calmatch[idobjs] = True

    # ADM something has to both match and been passed through as True.
    iscalspec &= calmatch

    return iscalspec


def apply_cuts(objects, cmxdir=None):
    """Perform commissioning (cmx) target selection on objects, return target mask arrays

    Parameters
    ----------
    objects: numpy structured array with UPPERCASE columns needed for
        target selection, OR a string tractor/sweep filename
    cmxdir : :class:`str`, optional, defaults to :envvar:`CMX_DIR`
        Directory in which to find commmissioning files to which to match, such as the
        CALSPEC stars. If not specified, the cmx directory is taken to be the value of
        the :envvar:`CMX_DIR` environment variable.

    Returns
    -------
    :class:`~numpy.ndarray`
        commissioning target selection bitmask flags for each object

    See desitarget.cmx.cmx_targetmask.cmx_mask for the definition of each bit
    """
    # -Check if objects is a filename instead of the actual data
    if isinstance(objects, str):
        objects = io.read_tractor(objects)
    # -Ensure uppercase column names if astropy Table
    if isinstance(objects, (Table, Row)):
        for col in list(objects.columns.values()):
            if not col.name.isupper():
                col.name = col.name.upper()

    # ADM retrieve/check the cmxdir.
    cmxdir = _get_cmxdir(cmxdir)

    # ADM As we need the column names.
    colnames = _get_colnames(objects)

    photsys_north, photsys_south, obs_rflux, gflux, rflux, zflux,              \
        w1flux, w2flux, objtype, release, gfluxivar, rfluxivar, zfluxivar,     \
        gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,                  \
        gfracmasked, rfracmasked, zfracmasked,                                 \
        gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,               \
        gsnr, rsnr, zsnr, w1snr, w2snr, dchisq, deltaChi2, brightstarinblob =  \
        _prepare_optical_wise(objects, colnames=colnames)

    # ADM in addition, cmx needs ra and dec.
    ra, dec = objects["RA"], objects["DEC"]

    # ADM Currently only coded for objects with Gaia matches
    # ADM (e.g. DR6 or above). Fail for earlier Data Releases.
    if np.any(release < 6000):
        log.critical('Commissioning cuts only coded for DR6 or above')
        raise ValueError
    if (np.max(objects['PMRA']) == 0.) & np.any(release < 7000):
        d = "/project/projectdirs/desi/target/gaia_dr2_match_dr6"
        log.info("Zero objects have a proper motion.")
        log.critical(
            "Did you mean to send the Gaia-matched sweeps in, e.g., {}?"
            .format(d)
        )
        raise IOError

    # Process the Gaia inputs for target selection.
    gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr, gaiagmag, gaiabmag,   \
        gaiarmag, gaiaaen, gaiadupsource, Grr, gaiaparamssolved, gaiabprpfactor, \
        gaiasigma5dmax, galb = _prepare_gaia(objects, colnames=colnames)

    # ADM a couple of extra columns; the observed g/z fluxes.
    obs_gflux, obs_zflux = objects['FLUX_G'], objects['FLUX_Z']

    # ADM initially, every object passes the cuts (is True).
    # ADM need to guard against the case of a single row being passed.
    # ADM initially every class has a priority shift of zero.
    if _is_row(objects):
        primary = np.bool_(True)
        priority_shift = np.array(0)
    else:
        primary = np.ones_like(objects, dtype=bool)
        priority_shift = np.zeros_like(objects, dtype=int)

    # ADM determine if an object passes the default logic for cmx stars.
    isgood = passesSTD_logic(
        gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
        objtype=objtype, gaia=gaia, pmra=pmra, pmdec=pmdec,
        aen=gaiaaen, dupsource=gaiadupsource, paramssolved=gaiaparamssolved,
        primary=primary
    )

    # ADM determine if an object is a "dither" star.
    std_dither, shift_dither = isSTD_dither(
        obs_gflux=obs_gflux, obs_rflux=obs_rflux, obs_zflux=obs_zflux,
        isgood=isgood, primary=primary
    )
    # ADM set up an initial priority shift.

    # ADM determine if an object is a bright test star.
    std_test = isSTD_test(
        obs_gflux=obs_gflux, obs_rflux=obs_rflux, obs_zflux=obs_zflux,
        isgood=isgood, primary=primary
    )

    # ADM determine if an object matched a CALSPEC standard.
    std_calspec = isSTD_calspec(
        ra=ra, dec=dec, cmxdir=cmxdir, primary=primary
    )

    # ADM determine if an object is SV0_STD_BRIGHT. Resembles first
    # ADM iteration of SV, but locked in cmx_cuts (and could be altered).
    sv0_std_bright = isSV0_STD_bright(
        gflux=gflux, rflux=rflux, zflux=zflux,
        pmra=pmra, pmdec=pmdec, parallax=parallax,
        gaiagmag=gaiagmag, isgood=isgood, primary=primary
    )

    # ADM determine if an object is SV0_BGS
    sv0_bgs = isSV0_BGS(
        rflux=rflux, objtype=objtype, primary=primary
    )

    # ADM determine if an object is SV0_MWS
    sv0_mws = isSV0_MWS(
        rflux=rflux, obs_rflux=obs_rflux, objtype=objtype,
        gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag,
        pmra=pmra, pmdec=pmdec, parallax=parallax,
        parallaxovererror=parallaxovererror,
        photbprpexcessfactor=gaiabprpfactor,
        astrometricsigma5dmax=gaiasigma5dmax,
        galb=galb, gaia=gaia, primary=primary
    )

    # ADM Construct the targetflag bits.
    cmx_target = std_dither * cmx_mask.STD_GAIA
    cmx_target |= std_test * cmx_mask.STD_TEST
    cmx_target |= std_calspec * cmx_mask.STD_CALSPEC
    cmx_target |= sv0_std_bright * cmx_mask.SV0_STD_BRIGHT
    cmx_target |= sv0_bgs * cmx_mask.SV0_BGS
    cmx_target |= sv0_mws * cmx_mask.SV0_MWS

    # ADM update the priority with any shifts.
    # ADM we may need to update this logic if there are other shifts.
    priority_shift[std_dither] = shift_dither[std_dither]

    return cmx_target, priority_shift


def select_targets(infiles, numproc=4, cmxdir=None):
    """Process input files in parallel to select commissioning (cmx) targets

    Parameters
    ----------
    infiles : :class:`list` or `str`
        A list of input filenames (tractor or sweep files) OR a single filename.
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    cmxdir : :class:`str`, optional, defaults to :envvar:`CMX_DIR`
        Directory in which to find commmissioning files to which to match, such as the
        CALSPEC stars. If not specified, the cmx directory is taken to be the value of
        the :envvar:`CMX_DIR` environment variable.

    Returns
    -------
    :class:`~numpy.ndarray`
        The subset of input targets which pass the cmx cuts, including an extra
        column for `CMX_TARGET`.

    Notes
    -----
        - if numproc==1, use serial code instead of parallel.
    """
    from desiutil.log import get_logger
    log = get_logger()

    # -Convert single file to list of files.
    if isinstance(infiles, str):
        infiles = [infiles, ]

    # -Sanity check that files exist before going further.
    for filename in infiles:
        if not os.path.exists(filename):
            raise ValueError("{} doesn't exist".format(filename))

    # ADM retrieve/check the cmxdir.
    cmxdir = _get_cmxdir(cmxdir)

    def _finalize_targets(objects, cmx_target, priority_shift):
        # -desi_target includes BGS_ANY and MWS_ANY, so we can filter just
        # -on desi_target != 0
        keep = (cmx_target != 0)
        objects = objects[keep]
        cmx_target = cmx_target[keep]
        priority_shift = priority_shift[keep]

        # -Add *_target mask columns
        # ADM note that only cmx_target is defined for commissioning
        # ADM so just pass that around
        targets = finalize(objects, cmx_target, cmx_target, cmx_target,
                           survey='cmx')
        # ADM shift the priorities of targets with functional priorities.
        targets["PRIORITY_INIT"] += priority_shift

        return targets

    # -functions to run on every brick/sweep file
    def _select_targets_file(filename):
        '''Returns targets in filename that pass the cuts'''
        objects = io.read_tractor(filename)
        cmx_target, priority_shift = apply_cuts(objects, cmxdir=cmxdir)

        return _finalize_targets(objects, cmx_target, priority_shift)

    # Counter for number of bricks processed;
    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    t0 = time()

    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if nbrick % 50 == 0 and nbrick > 0:
            rate = nbrick / (time() - t0)
            log.info('{} files; {:.1f} files/sec'.format(nbrick, rate))

        nbrick[...] += 1    # this is an in-place modification
        return result

    # -Parallel process input files
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
