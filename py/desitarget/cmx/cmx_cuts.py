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
import numpy.lib.recfunctions as rfn
import healpy as hp
import os
import fitsio
import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Row

from pkg_resources import resource_filename

from desitarget import io
from desitarget.cuts import _psflike, _is_row, _get_colnames, _prepare_gaia
from desitarget.cuts import _prepare_optical_wise, _check_BGS_targtype_sv
from desitarget.cuts import shift_photo_north
from desitarget.internal import sharedmem
from desitarget.targets import finalize, resolve
from desitarget.cmx.cmx_targetmask import cmx_mask
from desitarget.geomask import sweep_files_touch_hp, bundle_bricks
from desitarget.geomask import is_in_hp, is_in_gal_box
from desitarget.gaiamatch import gaia_dr_from_ref_cat, is_in_Galaxy
from desitarget.gaiamatch import find_gaia_files_hp

# ADM Main Survey functions, used for mini-SV.
from desitarget.cuts import isLRG as isLRG_MS
from desitarget.cuts import isELG as isELG_MS
from desitarget.cuts import isQSO_randomforest as isQSO_MS
from desitarget.cuts import isBGS as isBGS_MS

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
    if (cmxdir is None) or (not os.path.exists(cmxdir)):
        log.info('pass cmxdir or correctly set the $CMX_DIR environment variable...')
        msg = 'Commissioning files not found in {}'.format(cmxdir)
        log.critical(msg)
        raise ValueError(msg)

    return cmxdir


def _getColors(nbEntries, nfeatures, gflux, rflux, zflux, w1flux, w2flux):

    limitInf = 1.e-04
    gflux = gflux.clip(limitInf)
    rflux = rflux.clip(limitInf)
    zflux = zflux.clip(limitInf)
    w1flux = w1flux.clip(limitInf)
    w2flux = w2flux.clip(limitInf)

    g = np.where(gflux > limitInf, 22.5-2.5*np.log10(gflux), 0.)
    r = np.where(rflux > limitInf, 22.5-2.5*np.log10(rflux), 0.)
    z = np.where(zflux > limitInf, 22.5-2.5*np.log10(zflux), 0.)
    W1 = np.where(w1flux > limitInf, 22.5-2.5*np.log10(w1flux), 0.)
    W2 = np.where(w2flux > limitInf, 22.5-2.5*np.log10(w2flux), 0.)

    photOK = (g > 0.) & (r > 0.) & (z > 0.) & (W1 > 0.) & (W2 > 0.)

    colors = np.zeros((nbEntries, nfeatures))
    colors[:, 0] = g-r
    colors[:, 1] = r-z
    colors[:, 2] = g-z
    colors[:, 3] = g-W1
    colors[:, 4] = r-W1
    colors[:, 5] = z-W1
    colors[:, 6] = g-W2
    colors[:, 7] = r-W2
    colors[:, 8] = z-W2
    colors[:, 9] = W1-W2
    colors[:, 10] = r

    return colors, r, photOK


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
        ``True`` if and only if the object passes the logic cuts for
         cmx stars with fracflux_X < 0.01.
    :class:`array_like`
        ``True`` if and only if the object passes the logic cuts for
         cmx stars with fracflux_X < 0.002.


    Notes
    -----
    - This version (08/30/18) is version 4 on `the cmx wiki`_.
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

    # ADM tighter isolation cuts.
    tight = std.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # fracflux can be Inf/NaN.
        for bandint in (0, 1, 2):  # g, r, z.
            tight &= fracflux[bandint] < 0.002

    return std, tight


def isSV0_BGS(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
              rfiberflux=None,
              gnobs=None, rnobs=None, znobs=None,
              gfracmasked=None, rfracmasked=None, zfracmasked=None,
              gfracflux=None, rfracflux=None, zfracflux=None,
              gfracin=None, rfracin=None, zfracin=None,
              gfluxivar=None, rfluxivar=None, zfluxivar=None,
              maskbits=None, Grr=None, w1snr=None, gaiagmag=None,
              objtype=None, primary=None):
    """Definition of an SV0-like BGS target. Returns a boolean array.

    Parameters
    ----------
    See :func:`~desitarget.cuts.set_target_bits`.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is an SV-like BGS target.

    Notes
    -----
    - Current version (02/19/20) is version 55 on `the cmx wiki`_.
    - Hardcoded for south=False.
    - Combines bright/faint/faint_ext/fibmag BGS-like SV classes into
      one bit.
    - `desitarget.cmx.cmx_cuts.apply_cuts()` also additionally removes
      objects from this class that either have Gaia provenance and Gaia
      G < 16 OR that have Legacy Surveys g < 16.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    sv0_bgs = np.zeros_like(rflux, dtype='?')

    for targtype in ["bright", "faint", "faint_ext", "fibmag"]:
        bgs = isBGS(
            gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
            rfiberflux=rfiberflux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
            gfracmasked=gfracmasked, rfracmasked=rfracmasked,
            zfracmasked=zfracmasked,
            gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
            gfracin=gfracin, rfracin=rfracin, zfracin=zfracin,
            gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
            maskbits=maskbits, Grr=Grr, w1snr=w1snr, gaiagmag=gaiagmag,
            objtype=objtype, primary=primary, south=False, targtype=targtype
            )
        sv0_bgs |= bgs

    return sv0_bgs


def isBGS(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          rfiberflux=None,
          gnobs=None, rnobs=None, znobs=None,
          gfracmasked=None, rfracmasked=None, zfracmasked=None,
          gfracflux=None, rfracflux=None, zfracflux=None,
          gfracin=None, rfracin=None, zfracin=None,
          gfluxivar=None, rfluxivar=None, zfluxivar=None,
          maskbits=None, Grr=None, w1snr=None, gaiagmag=None,
          objtype=None, primary=None, south=True, targtype=None):
    """Definition of BGS target classes for SV. Returns a boolean array.

    Parameters
    ----------
    south : :class:`boolean`, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS)
        if ``south=False``, otherwise use cuts appropriate to the
        Southern imaging survey (DECaLS).
    targtype : :class:`str`, optional, defaults to ``faint``
        Pass ``bright`` for the ``BGS_BRIGHT`` selection
        or ``faint`` for the ``BGS_FAINT`` selection
        or ``faint_ext`` for the ``BGS_FAINT_EXTENDED`` selection
        or ``lowq`` for the ``BGS_LOW_QUALITY`` selection
        or ``fibmag`` for the ``BGS_FIBER_MAGNITUDE`` selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a BGS target
        of type ``targtype``.

    Notes
    -----
    - Current version (10/14/19) is version 105 on `the SV wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    _check_BGS_targtype_sv(targtype)

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()

    bgs &= notinBGS_mask(gflux=gflux, rflux=rflux, zflux=zflux,
                         gnobs=gnobs, rnobs=rnobs, znobs=znobs, primary=primary,
                         gfracmasked=gfracmasked, rfracmasked=rfracmasked,
                         zfracmasked=zfracmasked, zfracflux=zfracflux,
                         gfracflux=gfracflux, rfracflux=rfracflux,
                         gfracin=gfracin, rfracin=rfracin, zfracin=zfracin,
                         w1snr=w1snr, gfluxivar=gfluxivar, rfluxivar=rfluxivar,
                         zfluxivar=zfluxivar, Grr=Grr, gaiagmag=gaiagmag,
                         maskbits=maskbits, objtype=objtype, targtype=targtype)

    bgs &= isBGS_colors(rflux=rflux, rfiberflux=rfiberflux, south=south,
                        targtype=targtype, primary=primary)

    return bgs


def notinBGS_mask(gflux=None, rflux=None, zflux=None, gnobs=None, rnobs=None, znobs=None, primary=None,
                  gfracmasked=None, rfracmasked=None, zfracmasked=None,
                  gfracflux=None, rfracflux=None, zfracflux=None,
                  gfracin=None, rfracin=None, zfracin=None, w1snr=None,
                  gfluxivar=None, rfluxivar=None, zfluxivar=None, Grr=None,
                  gaiagmag=None, maskbits=None, objtype=None, targtype=None):
    """Standard set of masking cuts used by all BGS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isBGS_faint` for parameters).
    """
    _check_BGS_targtype_sv(targtype)

    if primary is None:
        primary = np.ones_like(gnobs, dtype='?')
    bgs_qcs = primary.copy()
    bgs = primary.copy()

    # quality cuts definitions
    bgs_qcs &= (gnobs >= 1) & (rnobs >= 1) & (znobs >= 1)
    bgs_qcs &= (gfracmasked < 0.4) & (rfracmasked < 0.4) & (zfracmasked < 0.4)
    bgs_qcs &= (gfracflux < 5.0) & (rfracflux < 5.0) & (zfracflux < 5.0)
    bgs_qcs &= (gfracin > 0.3) & (rfracin > 0.3) & (zfracin > 0.3)
    bgs_qcs &= (gfluxivar > 0) & (rfluxivar > 0) & (zfluxivar > 0)
    bgs_qcs &= (maskbits & 2**1) == 0
    # color box
    bgs_qcs &= rflux > gflux * 10**(-1.0/2.5)
    bgs_qcs &= rflux < gflux * 10**(4.0/2.5)
    bgs_qcs &= zflux > rflux * 10**(-1.0/2.5)
    bgs_qcs &= zflux < rflux * 10**(4.0/2.5)

    if targtype == 'lowq':
        bgs &= Grr > 0.6
        bgs |= gaiagmag == 0
        bgs |= (Grr < 0.6) & (~_psflike(objtype)) & (gaiagmag != 0)
        bgs &= ~bgs_qcs
    else:
        bgs &= Grr > 0.6
        bgs |= gaiagmag == 0
        bgs |= (Grr < 0.6) & (~_psflike(objtype)) & (gaiagmag != 0)
        bgs &= bgs_qcs

    return bgs


def isBGS_colors(rflux=None, rfiberflux=None, south=True, targtype=None, primary=None):
    """Standard set of masking cuts used by all BGS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isBGS` for parameters).
    """

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()

    if targtype == 'lowq':
        bgs &= rflux > 10**((22.5-20.1)/2.5)
    elif targtype == 'bright':
        bgs &= rflux > 10**((22.5-19.5)/2.5)
    elif targtype == 'faint':
        bgs &= rflux > 10**((22.5-20.1)/2.5)
        bgs &= rflux <= 10**((22.5-19.5)/2.5)
    elif targtype == 'faint_ext':
        bgs &= rflux > 10**((22.5-20.5)/2.5)
        bgs &= rflux <= 10**((22.5-20.1)/2.5)
        bgs &= ~np.logical_and(rflux <= 10**((22.5-20.1)/2.5), rfiberflux > 10**((22.5-21.0511)/2.5))
    elif targtype == 'fibmag':
        bgs &= rflux <= 10**((22.5-20.1)/2.5)
        bgs &= rfiberflux > 10**((22.5-21.0511)/2.5)
    else:
        _check_BGS_targtype_sv(targtype)

    return bgs


def isSV0_MWS(rflux=None, obs_rflux=None, objtype=None, paramssolved=None,
              gaiagmag=None, gaiabmag=None, gaiarmag=None, parallaxerr=None,
              pmra=None, pmdec=None, parallax=None, parallaxovererror=None,
              photbprpexcessfactor=None, astrometricsigma5dmax=None,
              gaiaaen=None, galb=None, gaia=None, primary=None):
    """Initial SV-like Milky Way Survey selections (MzLS/BASS imaging).

    Parameters
    ----------
    - See :func:`~desitarget.cuts.set_target_bits` for parameters.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a MWS_MAIN or MWS_NEARBY
        target from early SV/main survey classes.
    :class:`array_like`
        ``True`` if and only if the object is an early-SV/main survey
        MWS_WD target.
    :class:`array_like`
        ``True`` if and only if the object is an early-SV/main survey
        SV0_MWS_FAINT target.

    Notes
    -----
    - All Gaia quantities are as in `the Gaia data model`_.
    - Returns the equivalent of PRIMARY target classes from version 55
      (02/19/20) of `the cmx wiki`_. Ignores target classes that "smell"
      like secondary targets (as they are outside of the footprint or are
      based on catalog-matching). Simplifies flag cuts, and simplifies
      the MWS_MAIN class to not include sub-classes.
    - `desitarget.cmx.cmx_cuts.apply_cuts()` also additionally removes
      objects from this class that either have Gaia provenance and Gaia
      G < 16 OR that have Legacy Surveys g < 16.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    ismws = primary.copy()
    ismws_faint = primary.copy()
    isnear = primary.copy()
    iswd = primary.copy()

    # ADM apply the selection for all MWS-MAIN targets.
    # ADM main targets match to a Gaia source.
    ismws &= gaia
    # ADM main targets are point-like.
    ismws &= _psflike(objtype)

    # APC faint MWS filler for minsv3+ tiles
    # APC no constraint on obs_rflux
    ismws_faint &= ismws
    ismws_faint &= rflux > 10**((22.5-21.0)/2.5)
    ismws_faint &= rflux <= 10**((22.5-19.0)/2.5)

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
    isnear &= (parallax + parallaxerr) > 10.
    # ADM all astrometric parameters were measured.
    isnear &= paramssolved == 31

    # ADM do not target any WDs for which entries are NaN
    # ADM and turn off the NaNs for those entries.
    if photbprpexcessfactor is not None:
        nans = (np.isnan(gaiagmag) | np.isnan(gaiabmag) | np.isnan(gaiarmag) |
                np.isnan(parallax) | np.isnan(photbprpexcessfactor))
    else:
        nans = (np.isnan(gaiagmag) | np.isnan(gaiabmag) | np.isnan(gaiarmag) |
                np.isnan(parallax))

    if np.isscalar(nans):
        if nans:
            parallax = gaiagmag = gaiabmag = gaiarmag = 0.0
            if photbprpexcessfactor is not None:
                photbprpexcessfactor = 0.0
    else:
        w = np.where(nans)[0]
        if len(w) > 0:
            parallax, gaiagmag = parallax.copy(), gaiagmag.copy()
            gaiabmag, gaiarmag = gaiabmag.copy(), gaiarmag.copy()
            if photbprpexcessfactor is not None:
                photbprpexcessfactor = photbprpexcessfactor.copy()
            # ADM safe to make these zero regardless of cuts as...
                photbprpexcessfactor[w] = 0.
            parallax[w] = 0.
            gaiagmag[w], gaiabmag[w], gaiarmag[w] = 0., 0., 0.

    # ADM ...we'll turn off all bits here.
    iswd &= ~nans

    # ADM apply the selection for MWS-WD targets.
    # ADM must be a Legacy Surveys object that matches a Gaia source.
    iswd &= gaia
    # ADM all astrometric parameters were measured.
    iswd &= paramssolved == 31
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
    # ADM Inexplicably I'm getting a Runtimewarning here for
    # ADM a few values in the sqrt, so I'm catching it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
                 ((gaiaaen < 1.) & (parallaxovererror > 4.) & (pm > 10.)))

    # ADM return any object that passes the MWS cuts.
    return ismws | isnear, iswd, ismws_faint


def isSV0_LRG(gflux=None, rflux=None, zflux=None, w1flux=None,
              rfiberflux=None, zfiberflux=None,
              gflux_snr=None, rflux_snr=None, zflux_snr=None, w1flux_snr=None,
              gnobs=None, rnobs=None, znobs=None, maskbits=None,
              primary=None):
    """Target Definition of an SV0-like LRG. Returns a boolean array.

    Parameters
    ----------
    See :func:`~desitarget.cuts.set_target_bits`.

    Returns
    -------
    :class:`array_like` or :class:`float`
        ``True`` if and only if the object is an LRG color-selected
        target. If `floats` are passed, a `float` is returned.

    Notes
    -----
    - Current version (02/19/20) is version 50 on `the cmx wiki`_.
    - Hardcoded for south=False.
    - Combines all LRG-like SV classes into one bit.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    lrg &= notinLRG_mask(
        primary=primary, rflux=rflux, zflux=zflux, w1flux=w1flux,
        zfiberflux=zfiberflux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
        rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr,
        maskbits=maskbits
    )

    # ADM pass the lrg that pass cuts as primary, to restrict to the
    # ADM sources that weren't in a mask/logic cut.
    lrg, _, _, _, _ = isLRG_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
        zfiberflux=zfiberflux, south=False, primary=lrg
    )

    # ADM isLRG_colors() forces arrays, so catch the single-object case.
    if _is_row(rflux):
        return lrg[0]

    return lrg


def notinLRG_mask(primary=None, rflux=None, zflux=None, w1flux=None,
                  zfiberflux=None, gnobs=None, rnobs=None, znobs=None,
                  rflux_snr=None, zflux_snr=None, w1flux_snr=None,
                  maskbits=None):
    """See :func:`~desitarget.sv1.sv1_cuts.isLRG` for details.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is NOT masked for poor quality.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    lrg &= (rflux_snr > 0) & (rflux > 0)   # ADM quality in r.
    lrg &= (zflux_snr > 0) & (zflux > 0) & (zfiberflux > 0)   # ADM quality in z.
    lrg &= (w1flux_snr > 4) & (w1flux > 0)  # ADM quality in W1.

    # ADM observed in every band.
    lrg &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    # ADM ALLMASK (5, 6, 7), BRIGHT OBJECT (1, 11, 12, 13) bits not set.
    for bit in [1, 5, 6, 7, 11, 12, 13]:
        lrg &= ((maskbits & 2**bit) == 0)

    return lrg


def isLRG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 zfiberflux=None, south=True, primary=None):
    """See :func:`~desitarget.sv1.sv1_cuts.isLRG` for details.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()
    lrginit, lrgsuper = np.tile(primary, [2, 1])

    gmag = 22.5 - 2.5 * np.log10(gflux.clip(1e-7))
    # ADM safe as these fluxes are set to > 0 in notinLRG_mask.
    rmag = 22.5 - 2.5 * np.log10(rflux.clip(1e-7))
    zmag = 22.5 - 2.5 * np.log10(zflux.clip(1e-7))
    w1mag = 22.5 - 2.5 * np.log10(w1flux.clip(1e-7))
    zfibermag = 22.5 - 2.5 * np.log10(zfiberflux.clip(1e-7))

    if south:

        # LRG_INIT: Nominal optical + Nominal IR:
        lrginit &= zmag - w1mag > 0.8 * (rmag-zmag) - 0.6     # non-stellar cut
        lrginit &= zfibermag < 21.5                           # faint limit
        lrginit &= rmag - zmag > 0.7                          # remove outliers
        lrg_opt = (gmag - w1mag > 2.6) & (gmag - rmag > 1.4)  # low-z cut
        lrg_opt |= rmag - w1mag > 1.8                         # ignore low-z cut for faint objects
        lrg_opt &= rmag - zmag > (zmag - 16.83) * 0.45        # sliding optical cut
        lrg_opt &= rmag - zmag > (zmag - 13.80) * 0.19        # low-z sliding optical cut
        lrg_ir = rmag - w1mag > 1.1                           # Low-z cut
        lrg_ir &= rmag - w1mag > (w1mag - 17.23) * 1.8        # sliding IR cut
        lrg_ir &= rmag - w1mag > w1mag - 16.38                # low-z sliding IR cut
        lrginit &= lrg_opt | lrg_ir

        # LRG_SUPER: SV superset:
        lrgsuper &= zmag - w1mag > 0.8 * (rmag-zmag) - 0.8    # non-stellar cut
        lrgsuper &= (zmag < 20.5) | (zfibermag < 21.9)        # faint limit
        lrgsuper &= rmag - zmag > 0.6                         # remove outliers
        lrg_opt = (gmag - w1mag > 2.5) & (gmag - rmag > 1.3)  # low-z cut
        lrg_opt |= rmag - w1mag > 1.7                         # ignore low-z cut for faint objects
        # straight cut for low-z:
        lrg_opt_lowz = zmag < 20.2
        lrg_opt_lowz &= rmag - zmag > (zmag - 17.15) * 0.45
        lrg_opt_lowz &= rmag - zmag > (zmag - 14.12) * 0.19
        # curved sliding cut for high-z:
        lrg_opt_highz = zmag >= 20.2
        lrg_opt_highz &= ((zmag - 23.15) / 1.3)**2 + (rmag - zmag + 2.5)**2 > 4.485**2
        lrg_opt &= lrg_opt_lowz | lrg_opt_highz
        lrg_ir = rmag - w1mag > 1.0                           # Low-z cut
        # low-z sliding cut:
        lrg_ir_lowz = (w1mag < 18.96) & (rmag - w1mag > (w1mag - 17.46) * 1.8)
        # high-z sliding cut:
        lrg_ir_highz = (w1mag >= 18.96) & ((w1mag - 21.65)**2 + ((rmag - w1mag + 0.66) / 1.5)**2 > 3.5**2)
        lrg_ir_highz |= (w1mag >= 18.96) & (rmag - w1mag > 3.1)
        lrg_ir &= lrg_ir_lowz | lrg_ir_highz
        lrgsuper &= lrg_opt | lrg_ir

    else:

        # LRG_INIT: Nominal optical + Nominal IR:
        lrginit &= zmag - w1mag > 0.8 * (rmag-zmag) - 0.65      # non-stellar cut
        lrginit &= zfibermag < 21.5                             # faint limit
        lrginit &= rmag - zmag > 0.7                            # remove outliers
        lrg_opt = (gmag - w1mag > 2.67) & (gmag - rmag > 1.45)  # low-z cut
        lrg_opt |= rmag - w1mag > 1.85                          # ignore low-z cut for faint objects
        lrg_opt &= rmag - zmag > (zmag - 16.69) * 0.45          # sliding optical cut
        lrg_opt &= rmag - zmag > (zmag - 13.68) * 0.19          # low-z sliding optical cut
        lrg_ir = rmag - w1mag > 1.15                            # Low-z cut
        lrg_ir &= rmag - w1mag > (w1mag - 17.193) * 1.8         # sliding IR cut
        lrg_ir &= rmag - w1mag > w1mag - 16.343                 # low-z sliding IR cut
        lrginit &= lrg_opt | lrg_ir

        # LRG_SUPER: SV superset:
        lrgsuper &= zmag - w1mag > 0.8 * (rmag-zmag) - 0.85     # non-stellar cut
        lrgsuper &= (zmag < 20.5) | (zfibermag < 21.9)          # faint limit
        lrgsuper &= rmag - zmag > 0.6                           # remove outliers
        lrg_opt = (gmag - w1mag > 2.57) & (gmag - rmag > 1.35)  # low-z cut
        lrg_opt |= rmag - w1mag > 1.75                          # ignore low-z cut for faint objects
        # straight cut for low-z:
        lrg_opt_lowz = zmag < 20.2
        lrg_opt_lowz &= rmag - zmag > (zmag - 17.025) * 0.45    # sliding optical cut
        lrg_opt_lowz &= rmag - zmag > (zmag - 14.015) * 0.19    # low-z sliding optical cut
        # curved sliding cut for high-z:
        lrg_opt_highz = zmag >= 20.2
        lrg_opt_highz &= ((zmag - 23.175) / 1.3)**2 + (rmag - zmag + 2.43)**2 > 4.485**2
        lrg_opt &= lrg_opt_lowz | lrg_opt_highz
        lrg_ir = rmag - w1mag > 1.05                            # Low-z cut
        # low-z sliding cut:
        lrg_ir_lowz = (w1mag < 18.94) & (rmag - w1mag > (w1mag - 17.43) * 1.8)
        # high-z sliding cut:
        lrg_ir_highz = (w1mag >= 18.94) & ((w1mag - 21.63)**2 + ((rmag - w1mag + 0.65) / 1.5)**2 > 3.5**2)
        lrg_ir_highz |= (w1mag >= 18.94) & (rmag - w1mag > 3.1)
        lrg_ir &= lrg_ir_lowz | lrg_ir_highz
        lrgsuper &= lrg_opt | lrg_ir

    lrg &= lrginit | lrgsuper

    lrginit4, lrginit8 = lrginit.copy(), lrginit.copy()
    lrgsuper4, lrgsuper8 = lrgsuper.copy(), lrgsuper.copy()

    # ADM 4-pass LRGs are z < 20
    lrginit4 &= zmag < 20.
    lrgsuper4 &= zmag < 20.
    # ADM 8-pass LRGs are z >= 20
    lrginit8 &= zmag >= 20.
    lrgsuper8 &= zmag >= 20.

    return lrg, lrginit4, lrgsuper4, lrginit8, lrgsuper8


def isSV0_QSO(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
              gsnr=None, rsnr=None, zsnr=None, w1snr=None, w2snr=None,
              gnobs=None, rnobs=None, znobs=None, maskbits=None,
              dchisq=None, objtype=None, primary=None):
    """Target Definition of an SV0-like QSO. Returns a boolean array.

    Parameters
    ----------
    See :func:`~desitarget.cuts.set_target_bits`.

    Returns
    -------
    :class:`array_like` or :class:`float`
        ``True`` if and only if the object is an SV-like QSO target.
         If `floats` are passed, a `float` is returned.
    :class:`array_like` or :class:`float`
        ``True`` if and only if the object is an SV-like QSO target that
        passes something like the QSO_Z5 (high-z) selection from SV.

    Notes
    -----
    - Current version (02/19/20) is version 51 on `the cmx wiki`_.
    - Current version (03/10/20) for the high-z (QSO_Z5 selection)
      is version 59 on `the cmx wiki`_.
    - Hardcoded for south=False.
    - Combines all QSO-like SV classes into one bit.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    qsocolor_north = isQSO_cuts(
        primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
        w1flux=w1flux, w2flux=w2flux,
        gnobs=gnobs, rnobs=rnobs, znobs=znobs,
        dchisq=dchisq, maskbits=maskbits,
        objtype=objtype, w1snr=w1snr, w2snr=w2snr,
        south=False
        )

    qsorf_north = isQSO_randomforest(
        primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
        w1flux=w1flux, w2flux=w2flux,
        gnobs=gnobs, rnobs=rnobs, znobs=znobs,
        dchisq=dchisq, maskbits=maskbits,
        objtype=objtype, south=False
        )

    qsohizf_north = isQSO_highz_faint(
        primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
        w1flux=w1flux, w2flux=w2flux,
        gnobs=gnobs, rnobs=rnobs, znobs=znobs,
        dchisq=dchisq, maskbits=maskbits,
        objtype=objtype, south=False
        )

    qsocolor_high_z_north = isQSO_color_high_z(
        gflux=gflux, rflux=rflux, zflux=zflux,
        w1flux=w1flux, w2flux=w2flux, south=False
        )

    qsoz5_north = isQSOz5_cuts(
        primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
        gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
        gnobs=gnobs, rnobs=rnobs, znobs=znobs,
        w1flux=w1flux, w2flux=w2flux, w1snr=w1snr, w2snr=w2snr,
        dchisq=dchisq, maskbits=maskbits, objtype=objtype,
        south=False
        )

    qsocolor_highz_north = (qsocolor_north & qsocolor_high_z_north)
    qsorf_highz_north = (qsorf_north & qsocolor_high_z_north)
    qsocolor_lowz_north = (qsocolor_north & ~qsocolor_high_z_north)
    qsorf_lowz_north = (qsorf_north & ~qsocolor_high_z_north)
    qso_north = (qsocolor_lowz_north | qsorf_lowz_north | qsocolor_highz_north
                 | qsorf_highz_north | qsohizf_north | qsoz5_north)

    # ADM The individual routines return arrays, so we need
    # ADM a check to preserve the single-object case.
    if _is_row(rflux):
        return qso_north[0], qsoz5_north[0]

    return qso_north, qsoz5_north


def isQSO_cuts(gflux=None, rflux=None, zflux=None,
               w1flux=None, w2flux=None, w1snr=None, w2snr=None,
               dchisq=None, maskbits=None, objtype=None,
               gnobs=None, rnobs=None, znobs=None, primary=None, south=True):
    """Definition of QSO target classes from color cuts. Returns a boolean array.

    Parameters
    ----------
    south : :class:`boolean`, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
        otherwise use cuts appropriate to the Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` for objects that pass the quasar color/morphology/logic cuts.

    Notes
    -----
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    qso = primary.copy()

    # ADM Reject objects in masks.
    # ADM BRIGHT BAILOUT GALAXY CLUSTER (1, 10, 12, 13) bits not set.
    if maskbits is not None:
        for bit in [1, 10, 12, 13]:
            qso &= ((maskbits & 2**bit) == 0)

    # ADM observed in every band.
    qso &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    # ADM relaxed morphology cut for SV.
    # ADM we never target sources with dchisq[..., 0] = 0, so force
    # ADM those to have large values of morph2 to avoid divide-by-zero.
    # ADM the atleast_1d's are to catch the single-object case.
    d1, d0 = np.atleast_1d(dchisq[..., 1]), np.atleast_1d(dchisq[..., 0])
    bigmorph = np.zeros_like(d0)+1e9
    dcs = np.divide(d1 - d0, d0, out=bigmorph, where=d0 != 0)
    if south:
        morph2 = dcs < 0.01
    else:
        morph2 = dcs < 0.005
    qso &= _psflike(objtype) | morph2

    # ADM SV cuts are different for WISE SNR.
    if south:
        qso &= w1snr > 2.5
        qso &= w2snr > 1.5
    else:
        qso &= w1snr > 3
        qso &= w2snr > 2

    # ADM perform the color cuts to finish the selection.
    qso &= isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                        w1flux=w1flux, w2flux=w2flux,
                        primary=primary, south=south)

    return qso


def isQSO_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 primary=None, south=True):
    """Test if sources have quasar-like colors in a color box.
    (see, e.g., :func:`~desitarget.sv1.sv1_cuts.isQSO_cuts`).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    qso = primary.copy()

    # ADM Create some composite fluxes.
    wflux = 0.75*w1flux + 0.25*w2flux
    grzflux = (gflux + 0.8*rflux + 0.5*zflux) / 2.3

    # ADM perform the magnitude cuts.
    if south:
        qso &= rflux > 10**((22.5-23.)/2.5)    # r<23.0 (different for SV)
    else:
        qso &= rflux > 10**((22.5-22.8)/2.5)    # r<22.7
    qso &= grzflux < 10**((22.5-17.)/2.5)    # grz>17

    # ADM the optical color cuts.
    qso &= rflux < gflux * 10**(1.3/2.5)    # (g-r)<1.3
    qso &= zflux > rflux * 10**(-0.4/2.5)   # (r-z)>-0.4
    qso &= zflux < rflux * 10**(3.0/2.5)    # (r-z)<3.0 (different for SV)

    # ADM the WISE-optical color cut.
    if south:
        qso &= wflux * gflux > zflux * grzflux * 10**(-1.3/2.5)  # (grz-W) > (g-z)-1.3 (different for SV)
    else:
        qso &= wflux * gflux > zflux * grzflux * 10**(-1.1/2.5)  # (grz-W) > (g-z)-1.1 (different for SV)

    # ADM the WISE color cut.
    qso &= w2flux > w1flux * 10**(-0.4/2.5)  # (W1-W2) > -0.4

    # ADM Stricter WISE cuts on stellar contamination for objects on Main Sequence.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    mainseq = rflux > gflux * 10**(0.2/2.5)  # g-r > 0.2
    if south:
        mainseq &= rflux**(1+1.53) > gflux * zflux**1.53 * 10**((-0.075+0.20)/2.5)
        mainseq &= rflux**(1+1.53) < gflux * zflux**1.53 * 10**((+0.075+0.20)/2.5)
    else:
        mainseq &= rflux**(1+1.53) > gflux * zflux**1.53 * 10**((-0.075+0.20)/2.5)
        mainseq &= rflux**(1+1.53) < gflux * zflux**1.53 * 10**((+0.075+0.20)/2.5)

    mainseq &= w2flux < w1flux * 10**(0.3/2.5)  # ADM W1 - W2 !(NOT) > 0.3
    qso &= ~mainseq

    return qso


def isQSO_color_high_z(gflux=None, rflux=None, zflux=None,
                       w1flux=None, w2flux=None, south=True):
    """
    Color cut to select Highz QSO (z>~2.)
    """
    # ADM the np.atleast_1d's are to catch the single-object case.
    gflux = np.atleast_1d(gflux)
    rflux = np.atleast_1d(rflux)
    zflux = np.atleast_1d(zflux)
    w1flux = np.atleast_1d(w1flux)
    w2flux = np.atleast_1d(w2flux)

    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    wflux = 0.75*w1flux + 0.25*w2flux
    grzflux = (gflux + 0.8*rflux + 0.5*zflux) / 2.3

    # ADM we raise -ve fluxes to fractional powers, here, which produces NaN as
    # ADM e.g. -ve**0.4 is only defined for complex numbers! After testing, I find
    # ADM when gflux, rflux or zflux are -ve qso_hz is always False
    # ADM when wflux is -ve qso_hz is always True.
    # ADM So, I've hardcoded that logic to prevent NaN.
    qso_hz = (wflux < 0) & (gflux >= 0) & (rflux >= 0) & (zflux >= 0)
    ii = (wflux >= 0) & (gflux >= 0) & (rflux >= 0) & (zflux >= 0)
    qso_hz[ii] = ((wflux[ii] < gflux[ii]*10**(2.0/2.5)) |
                  (rflux[ii]*(gflux[ii]**0.4) >
                   gflux[ii]*(wflux[ii]**0.4)*10**(0.3/2.5)))  # (g-w<2.0 or g-r>O.4*(g-w)+0.3)
    if south:
        qso_hz[ii] &= (wflux[ii] * (rflux[ii]**1.2) <
                       (zflux[ii]**1.2) * grzflux[ii] * 10**(+0.8/2.5))  # (grz-W) < (r-z)*1.2+0.8
    else:
        qso_hz[ii] &= (wflux[ii] * (rflux[ii]**1.2) <
                       (zflux[ii]**1.2) * grzflux[ii] * 10**(+0.7/2.5))  # (grz-W) < (r-z)*1.2+0.7

    return qso_hz


def isQSO_randomforest(gflux=None, rflux=None, zflux=None, w1flux=None,
                       w2flux=None, objtype=None, release=None, dchisq=None,
                       maskbits=None, gnobs=None, rnobs=None, znobs=None,
                       primary=None, south=True):
    """Definition of QSO target class using random forest. Returns a boolean array.

    Parameters
    ----------
    south : :class:`boolean`, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
        otherwise use cuts appropriate to the Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` for objects that pass the quasar color/morphology/logic cuts.

    Notes
    -----
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    # BRICK_PRIMARY
    if primary is None:
        primary = np.ones_like(gflux, dtype=bool)

    # Build variables for random forest.
    nFeatures = 11  # Number of attributes describing each object to be classified by the rf.
    nbEntries = rflux.size
    # ADM shift the northern photometry to the southern system.
    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    # ADM photOK here should ensure (g > 0.) & (r > 0.) & (z > 0.) & (W1 > 0.) & (W2 > 0.)
    colors, r, photOK = _getColors(nbEntries, nFeatures, gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d(r)

    # ADM Preselection to speed up the process
    rMax = 23.0  # r < 23.0 (different for SV)
    rMin = 17.5  # r > 17.5
    preSelection = (r < rMax) & (r > rMin) & photOK & primary

    # ADM observed in every band.
    preSelection &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    # ADM relaxed morphology cut for SV.
    # ADM we never target sources with dchisq[..., 0] = 0, so force
    # ADM those to have large values of morph2 to avoid divide-by-zero.
    # ADM the np.atleast_1d's are to catch the single-object case.
    d1, d0 = np.atleast_1d(dchisq[..., 1]), np.atleast_1d(dchisq[..., 0])
    bigmorph = np.zeros_like(d0)+1e9
    dcs = np.divide(d1 - d0, d0, out=bigmorph, where=d0 != 0)
    if south:
        morph2 = dcs < 0.015
    else:
        morph2 = dcs < 0.02
    preSelection &= _psflike(objtype) | morph2

    # ADM Reject objects in masks.
    # ADM BRIGHT BAILOUT GALAXY CLUSTER (1, 10, 12, 13) bits not set.
    if maskbits is not None:
        for bit in [1, 10, 12, 13]:
            preSelection &= ((maskbits & 2**bit) == 0)

    # "qso" mask initialized to "preSelection" mask
    qso = np.copy(preSelection)

    if np.any(preSelection):

        from desitarget.myRF import myRF

        # Data reduction to preselected objects
        colorsReduced = colors[preSelection]
        r_Reduced = r[preSelection]
        colorsIndex = np.arange(0, nbEntries, dtype=np.int64)
        colorsReducedIndex = colorsIndex[preSelection]

        # Path to random forest files
        pathToRF = resource_filename('desitarget', 'data')
        # ADM Use RF trained over DR7
        rf_fileName = pathToRF + '/rf_model_dr7.npz'
        rf_HighZ_fileName = pathToRF + '/rf_model_dr7_HighZ.npz'

        # rf initialization - colors data duplicated within "myRF"
        rf = myRF(colorsReduced, pathToRF, numberOfTrees=500, version=2)
        rf_HighZ = myRF(colorsReduced, pathToRF, numberOfTrees=500, version=2)
        # rf loading
        rf.loadForest(rf_fileName)
        rf_HighZ.loadForest(rf_HighZ_fileName)
        # Compute rf probabilities
        tmp_rf_proba = rf.predict_proba()
        tmp_rf_HighZ_proba = rf_HighZ.predict_proba()
        # Compute optimized proba cut (all different for SV)
        # ADM the probabilities are different for the north and the south.
        if south:
            pcut = np.where(r_Reduced > 20.0,
                            0.60 - (r_Reduced - 20.0) * 0.10, 0.60)
            pcut[r_Reduced > 22.0] = 0.40 - 0.25 * (r_Reduced[r_Reduced > 22.0] - 22.0)
            pcut_HighZ = 0.40
        else:
            pcut = np.where(r_Reduced > 20.0,
                            0.65 - (r_Reduced - 20.0) * 0.075, 0.65)
            pcut[r_Reduced > 22.0] = 0.50 - 0.25 * (r_Reduced[r_Reduced > 22.0] - 22.0)
            pcut_HighZ = np.where(r_Reduced > 20.5,
                                  0.5 - (r_Reduced - 20.5) * 0.025, 0.5)

        # Add rf proba test result to "qso" mask
        qso[colorsReducedIndex] = \
            (tmp_rf_proba >= pcut) | (tmp_rf_HighZ_proba >= pcut_HighZ)

    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "numpy.ndarray"
    if nbEntries == 1:
        qso = qso[0]

    return qso


def isQSO_highz_faint(gflux=None, rflux=None, zflux=None, w1flux=None,
                      w2flux=None, objtype=None, release=None, dchisq=None,
                      gnobs=None, rnobs=None, znobs=None,
                      maskbits=None, primary=None, south=True):
    """Definition of QSO target for highz (z>2.0) faint QSOs. Returns a boolean array.

    Parameters
    ----------
    south : :class:`boolean`, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
        otherwise use cuts appropriate to the Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` for objects that pass the quasar color/morphology/logic cuts.

    Notes
    -----
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    # BRICK_PRIMARY
    if primary is None:
        primary = np.ones_like(gflux, dtype=bool)

    # Build variables for random forest.
    nFeatures = 11  # Number of attributes describing each object to be classified by the rf.
    nbEntries = rflux.size
    # ADM shift the northern photometry to the southern system.
    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    # ADM photOK here should ensure (g > 0.) & (r > 0.) & (z > 0.) & (W1 > 0.) & (W2 > 0.).
    colors, r, photOK = _getColors(nbEntries, nFeatures, gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d(r)

    # ADM Preselection to speed up the process.
    # Selection of faint objects.
    rMax = 23.5  # r < 23.5
    rMin = 22.7  # r > 22.7
    preSelection = (r < rMax) & (r > rMin) & photOK & primary

    # ADM observed in every band.
    preSelection &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    # Color Selection of QSO with z>2.0.
    wflux = 0.75*w1flux + 0.25*w2flux
    grzflux = (gflux + 0.8*rflux + 0.5*zflux) / 2.3
    # ADM "color_cut" isn't used. If it WAS to be used, we'd need to guard against raising
    # ADM negative fluxes to fractional powers, e.g. (-0.11)**0.3 is a complex number!
    # color_cut = ((wflux < gflux*10**(2.7/2.5)) |
    #              (rflux*(gflux**0.3) > gflux*(wflux**0.3)*10**(0.3/2.5)))  # (g-w<2.7 or g-r>O.3*(g-w)+0.3)
    # color_cut &= (wflux * (rflux**1.5) < (zflux**1.5) * grzflux * 10**(+1.6/2.5))  # (grz-W) < (r-z)*1.5+1.6
    # preSelection &= color_cut

    # Standard morphology cut.
    preSelection &= _psflike(objtype)

    # ADM Reject objects in masks.
    # ADM BRIGHT BAILOUT GALAXY CLUSTER (1, 10, 12, 13) bits not set.
    if maskbits is not None:
        for bit in [1, 10, 12, 13]:
            preSelection &= ((maskbits & 2**bit) == 0)

    # "qso" mask initialized to "preSelection" mask.
    qso = np.copy(preSelection)

    if np.any(preSelection):

        from desitarget.myRF import myRF

        # Data reduction to preselected objects.
        colorsReduced = colors[preSelection]
        colorsReduced[:, 10] = 22.8
        r_Reduced = r[preSelection]
        colorsIndex = np.arange(0, nbEntries, dtype=np.int64)
        colorsReducedIndex = colorsIndex[preSelection]

        # Path to random forest files.
        pathToRF = resource_filename('desitarget', 'data')
        # Use RF trained over DR7.
        rf_fileName = pathToRF + '/rf_model_dr7.npz'

        # rf initialization - colors data duplicated within "myRF".
        rf = myRF(colorsReduced, pathToRF, numberOfTrees=500, version=2)

        # rf loading.
        rf.loadForest(rf_fileName)

        # Compute rf probabilities.
        tmp_rf_proba = rf.predict_proba()

        # Compute optimized proba cut (all different for SV).
        # The probabilities may be different for the north and the south.
        if south:
            pcut = np.where(r_Reduced < 23.2,  0.40 + (r_Reduced-22.8)*.9, .76 + (r_Reduced-23.2)*.4)
        else:
            pcut = np.where(r_Reduced < 23.2,  0.40 + (r_Reduced-22.8)*.9, .76 + (r_Reduced-23.2)*.4)

        # Add rf proba test result to "qso" mask
        qso[colorsReducedIndex] = (tmp_rf_proba >= pcut)

    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "numpy.ndarray"
    if nbEntries == 1:
        qso = qso[0]

    return qso


def isQSOz5_cuts(gflux=None, rflux=None, zflux=None,
                 gsnr=None, rsnr=None, zsnr=None,
                 gnobs=None, rnobs=None, znobs=None,
                 w1flux=None, w2flux=None, w1snr=None, w2snr=None,
                 dchisq=None, maskbits=None, objtype=None, primary=None,
                 south=True):
    """Definition of z~5 QSO target classes from color cuts. Returns a boolean array.

    Parameters
    ----------
    south : :class:`boolean`, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
        otherwise use cuts appropriate to the Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` for objects that pass the quasar color/morphology/logic cuts.

    Notes
    -----
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    qso = primary.copy()

    # ADM Reject objects in masks.
    # ADM BRIGHT BAILOUT GALAXY CLUSTER (1, 10, 12, 13) bits not set.
    if maskbits is not None:
        # for bit in [10, 12, 13]:
        for bit in [1, 10, 12, 13]:
            qso &= ((maskbits & 2**bit) == 0)

    # ADM observed in every band.
    qso &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    # ADM relaxed morphology cut for SV.
    # ADM we never target sources with dchisq[..., 0] = 0, so force
    # ADM those to have large values of morph2 to avoid divide-by-zero.
    # ADM the atleast_1d's are to catch the single-object case.
    d1, d0 = np.atleast_1d(dchisq[..., 1]), np.atleast_1d(dchisq[..., 0])
    bigmorph = np.zeros_like(d0)+1e9
    dcs = np.divide(d1 - d0, d0, out=bigmorph, where=d0 != 0)
    if south:
        morph2 = dcs < 0.01
    else:
        # ADM currently identical, but leave as a placeholder for now.
        morph2 = dcs < 0.01
    qso &= _psflike(objtype) | morph2

    # ADM SV cuts are different for WISE SNR.
    if south:
        qso &= w1snr > 3
        qso &= w2snr > 2
    else:
        qso &= w1snr > 3
        qso &= w2snr > 2

    # ADM perform the color cuts to finish the selection.
    qso &= isQSOz5_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                          gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                          w1flux=w1flux, w2flux=w2flux,
                          primary=primary, south=south)

    return qso


def isQSOz5_colors(gflux=None, rflux=None, zflux=None,
                   gsnr=None, rsnr=None, zsnr=None,
                   w1flux=None, w2flux=None, primary=None, south=True):
    """Color cut to select z~5 quasar targets.
    (See :func:`~desitarget.sv1.sv1_cuts.isQSOz5_cuts`).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    qso = primary.copy()

    # ADM never target sources with negative W1 or z fluxes.
    qso &= (w1flux >= 0.) & (zflux >= 0.)

    # ADM now safe to update w1flux and zflux to avoid warnings.
    # ADM the np.atleast_1d's are to catch the single-object case.
    w1flux = np.atleast_1d(w1flux)
    zflux = np.atleast_1d(zflux)
    w1flux[~qso] = 0.
    zflux[~qso] = 0.

    # flux limit, z < 21.4.
    # ADM may switch to a zfiberflux cut later.
    qso &= zflux > 10**((22.5-21.4)/2.5)

    # gr cut, SNg < 3 | g > 24.5 | g-r > 1.8.
    SNRg = gsnr < 3
    gcut = gflux < 10**((22.5-24.5)/2.5)
    grcut = rflux > 10**(1.8/2.5) * gflux
    qso &= SNRg | gcut | grcut

    # zw1w2 cuts: SNz > 5
    # & w1-w2 > 0.5 & z- w1 < 4.5 & z-w1 > 2.0  (W1, W2 in Vega).
    qso &= zsnr > 5

    qsoz5 = qso & (w2flux > 10**(-0.14/2.5) * w1flux)  # w1-w2 > -0.14 in AB magnitude.
    qsoz5 &= (w1flux < 10**((4.5-2.699)/2.5) * zflux) & (w1flux > 10**((2.0-2.699)/2.5) * zflux)

    # rzW1 cuts: (SNr < 3 |
    # (r-z < 3.2*(z-w1) - 6.5 & r-z > 1.0 & r-z < 3.9) | r-z > 4.4).
    SNRr = rsnr < 3
    # ADM N/S currently identical, but leave as a placeholder for now.
    if south:
        rzw1cut = (
            (w1flux**3.2 * rflux > 10**((6.5-3.2*2.699)/2.5) * (zflux**(3.2+1)))
            & (zflux > 10**(1.0/2.5) * rflux) & (zflux < 10**(3.9/2.5) * rflux)
        )
        rzcut = zflux > 10**(4.4/2.5) * rflux  # for z~6 quasar
    else:
        rzw1cut = (
            (w1flux**3.2 * rflux > 10**((6.5-3.2*2.699)/2.5) * (zflux**(3.2+1)))
            & (zflux > 10**(1.0/2.5) * rflux) & (zflux < 10**(3.9/2.5) * rflux)
        )
        rzcut = zflux > 10**(4.4/2.5) * rflux

    qsoz5 &= SNRr | rzw1cut | rzcut

    # additional cuts for z~ 4.3-4.8 quasar
    # & w1-w2 > 0.3 & z-w1 < 4.5 & z-w1 > 2.5 & SNr > 3 & r-z > -1.0 & r-z < 1.5, W1,W2 in Vega
    qsoz45 = qso & (w2flux > 10**(-0.34/2.5) * w1flux)  # W1,W2 in AB.
    qsoz45 &= (w1flux < 10**((4.5-2.699)/2.5) * zflux) & (w1flux > 10**((2.5-2.699)/2.5) * zflux)
    qsoz45 &= rsnr > 3
    qsoz45 &= (zflux > 10**(-1.0/2.5) * rflux) & (zflux < 10**(1.5/2.5) * rflux)

    qso &= qsoz5 | qsoz45

    return qso


def isSV0_ELG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
              gsnr=None, rsnr=None, zsnr=None, gfiberflux=None,
              gnobs=None, rnobs=None, znobs=None,
              maskbits=None, primary=None):
    """Definition of an SV0-like ELG target. Returns a boolean array.

    Parameters
    ----------
    See :func:`~desitarget.cuts.set_target_bits`.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is an SV-like ELG target.

    Notes
    -----
    - Current version (10/14/19) is version 107 on `the SV wiki`_.
    - Hardcoded for south=False.
    - Combines all ELG-like SV classes into one bit.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    elg &= notinELG_mask(maskbits=maskbits, gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                         gnobs=gnobs, rnobs=rnobs, znobs=znobs, primary=primary)

    svgtot, svgfib, fdrgtot, fdrgfib = isELG_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
        gfiberflux=gfiberflux, south=False, primary=elg
    )

    return svgtot | svgfib | fdrgtot | fdrgfib


def notinELG_mask(maskbits=None, gsnr=None, rsnr=None, zsnr=None,
                  gnobs=None, rnobs=None, znobs=None, primary=None):
    """Standard set of masking cuts used by all ELG target selection classes.
    (see :func:`~desitarget.cuts.set_target_bits` for parameters).
    """
    if primary is None:
        primary = np.ones_like(maskbits, dtype='?')
    elg = primary.copy()

    # ADM good signal-to-noise in all bands.
    elg &= (gsnr > 0) & (rsnr > 0) & (zsnr > 0)

    # ADM observed in every band.
    elg &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    # ADM ALLMASK (5, 6, 7), BRIGHT OBJECT (1, 11, 12, 13) bits not set.
    for bit in [1, 5, 6, 7, 11, 12, 13]:
        elg &= ((maskbits & 2**bit) == 0)

    return elg


def isELG_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 gfiberflux=None, primary=None, south=True):
    """Color cuts for ELG target selection classes
    (see, e.g., :func:`desitarget.cuts.set_target_bits` for parameters).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    # ADM some cuts specific to north or south
    if south:
        gtotfaint_fdr = 23.5
        gfibfaint_fdr = 24.1
        lowzcut_zp = -0.15
    else:
        gtotfaint_fdr = 23.6
        gfibfaint_fdr = 24.2
        lowzcut_zp = -0.35

    # ADM work in magnitudes not fluxes. THIS IS ONLY OK AS the snr cuts
    # ADM in notinELG_mask ENSURE positive fluxes in all of g, r and z.
    g = 22.5 - 2.5*np.log10(gflux.clip(1e-16))
    r = 22.5 - 2.5*np.log10(rflux.clip(1e-16))
    z = 22.5 - 2.5*np.log10(zflux.clip(1e-16))

    # ADM gfiberflux can be zero but is never negative. So this is safe.
    gfib = 22.5 - 2.5*np.log10(gfiberflux.clip(1e-16))

    # ADM these are safe as the snr cuts in notinELG_mask ENSURE positive
    # ADM fluxes in all of g, r and z...so things near colors of zero but
    # ADM that actually have negative fluxes will never be targeted.
    rz = r - z
    gr = g - r

    # ADM all classes have g > 20.
    elg &= g >= 20

    # ADM parent classes for SV (relaxed) and FDR cuts.
    sv, fdr = elg.copy(), elg.copy()

    # ADM create the SV classes.
    sv &= rz > -1.           # blue cut.
    sv &= gr < -1.2*rz+2.5   # OII cut.
    sv &= (gr < 0.2) | (gr < 1.15*rz + lowzcut_zp)   # star/lowz cut.

    # ADM gfib/g split for SV-like classes.
    svgtot, svgfib = sv.copy(), sv.copy()
    coii = gr + 1.2*rz  # color defined perpendicularly to the -ve slope cut.
    svgtot &= coii < 1.6 - 7.2*(g-gtotfaint_fdr)     # sliding cut.
    svgfib &= coii < 1.6 - 7.2*(gfib-gfibfaint_fdr)  # sliding cut.

    # ADM create the FDR classes.
    fdr &= (rz > 0.3)                 # rz cut.
    fdr &= (rz < 1.6)                 # rz cut.
    fdr &= gr < -1.20*rz + 1.6        # OII cut.
    fdr &= gr < 1.15*rz + lowzcut_zp  # star/lowz cut.

    # ADM gfib/g split for FDR-like classes.
    fdrgtot, fdrgfib = fdr.copy(), fdr.copy()
    fdrgtot &= g < gtotfaint_fdr      # faint cut.
    fdrgfib &= gfib < gfibfaint_fdr   # faint cut.

    return svgtot, svgfib, fdrgtot, fdrgfib


def isSV0_STD(gflux=None, rflux=None, zflux=None, primary=None,
              gfracflux=None, rfracflux=None, zfracflux=None,
              gfracmasked=None, rfracmasked=None, zfracmasked=None,
              gnobs=None, rnobs=None, znobs=None,
              gfluxivar=None, rfluxivar=None, zfluxivar=None, objtype=None,
              gaia=None, astrometricexcessnoise=None, paramssolved=None,
              pmra=None, pmdec=None, parallax=None, dupsource=None,
              gaiagmag=None, gaiabmag=None, gaiarmag=None, bright=False):
    """Select STD targets using color cuts and photometric quality cuts.

    Parameters
    ----------
    bright : :class:`boolean`, defaults to ``False``
        if ``True`` apply magnitude cuts for "bright" conditions; otherwise,
        choose "normal" brightness standards. Cut is performed on `gaiagmag`.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a STD star.

    Notes
    -----
    - This version (11/05/18) is version 24 on `the SV wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    std = primary.copy()

    # ADM apply type=PSF cut.
    std &= _psflike(objtype)

    # ADM apply fracflux, S/N cuts and number of observations cuts.
    fracflux = [gfracflux, rfracflux, zfracflux]
    fluxivar = [gfluxivar, rfluxivar, zfluxivar]
    nobs = [gnobs, rnobs, znobs]
    fracmasked = [gfracmasked, rfracmasked, zfracmasked]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # fracflux can be Inf/NaN
        for bandint in (0, 1, 2):  # g, r, z
            std &= fracflux[bandint] < 0.01
            std &= fluxivar[bandint] > 0
            std &= nobs[bandint] > 0
            std &= fracmasked[bandint] < 0.6

    # ADM apply the Legacy Surveys (optical) magnitude and color cuts.
    std &= isSTD_colors(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux)

    # ADM apply the Gaia quality cuts.
    std &= isSTD_gaia(primary=primary, gaia=gaia,
                      astrometricexcessnoise=astrometricexcessnoise,
                      pmra=pmra, pmdec=pmdec, parallax=parallax,
                      dupsource=dupsource, paramssolved=paramssolved,
                      gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag)

    # ADM brightness cuts in Gaia G-band.
    if bright:
        gbright, gfaint = 15., 18.
    else:
        gbright, gfaint = 16., 19.

    std &= gaiagmag >= gbright
    std &= gaiagmag < gfaint

    return std


def isSTD_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 primary=None):
    """Select STD stars based on Legacy Surveys color cuts. Returns a boolean array.
    see :func:`~desitarget.sv1.sv1_cuts.isSTD` for other details.
    """
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    std = primary.copy()

    # Clip to avoid warnings from negative numbers.
    # ADM we're pretty bright for the STDs, so this should be safe.
    gflux = gflux.clip(1e-16)
    rflux = rflux.clip(1e-16)
    zflux = zflux.clip(1e-16)

    # ADM optical colors for halo TO or bluer.
    grcolor = 2.5 * np.log10(rflux / gflux)
    rzcolor = 2.5 * np.log10(zflux / rflux)
    std &= rzcolor < 0.2
    std &= grcolor > 0.
    std &= grcolor < 0.35

    return std


def isSTD_gaia(primary=None, gaia=None, astrometricexcessnoise=None,
               pmra=None, pmdec=None, parallax=None,
               dupsource=None, paramssolved=None,
               gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Gaia quality cuts used to define STD star targets
    see :func:`~desitarget.sv1.sv1_cuts.isSTD` for other details.
    """
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    std = primary.copy()

    # ADM Bp and Rp are both measured.
    std &= ~np.isnan(gaiabmag - gaiarmag)

    # ADM no obvious issues with the astrometry solution.
    std &= astrometricexcessnoise < 1
    std &= paramssolved == 31

    # ADM finite proper motions.
    std &= np.isfinite(pmra)
    std &= np.isfinite(pmdec)

    # ADM a parallax smaller than 1 mas.
    std &= parallax < 1.

    # ADM calculate the overall proper motion magnitude
    # ADM inexplicably I'm getting a Runtimewarning here for
    # ADM a few values in the sqrt, so I'm catching it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = np.sqrt(pmra**2. + pmdec**2.)

    # ADM a proper motion larger than 2 mas/yr.
    std &= pm > 2.

    # ADM fail if dupsource is not Boolean, as was the case for the 7.0 sweeps.
    # ADM otherwise logic checks on dupsource will be misleading.
    if not (dupsource.dtype.type == np.bool_):
        log.error('GAIA_DUPLICATED_SOURCE (dupsource) should be boolean!')
        raise IOError

    # ADM a unique Gaia source.
    std &= ~dupsource

    return std


def isSTD_dither(obs_gflux=None, obs_rflux=None, obs_zflux=None,
                 isgood=None, primary=None):
    """Gaia stars for dithering-and-other tests during commissioning.

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
        ``True`` if and only if the object is a Gaia "STD_GAIA" target.
    :class:`array_like`
        A priority shift of 10*(25-rmag) based on r-band magnitude.

    Notes
    -----
    - This version (08/30/18) is version 4 on `the cmx wiki`_.
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


def isSTD_dither_gaia(ra=None, dec=None, gmag=None, rmag=None, aen=None,
                      paramssolved=None, dupsource=None, pmra=None, pmdec=None,
                      nside=2, primary=None, test=False):
    """Gaia stars for dithering tests outside of the Legacy Surveys area.

    Parameters
    ----------
    ra, dec : :class:`array_like` or :class:`None`
        Right Ascension and Declination in degrees.
    gmag, rmag : :class:`array_like` or :class:`None`
        GAIA_PHOT_G_MEAN_MAG, GAIA_PHOT_R_MEAN_MAG.
    aen : :class:`array_like` or :class:`None`
        Gaia Astrometric Excess Noise.
    paramssolved : :class:`array_like` or :class:`None`
        How many parameters were solved for in Gaia.
    dupsource : :class:`array_like` or :class:`None`
        Whether the source is a duplicate in Gaia.
    pmra, pmdec : :class:`array_like` or :class:`None`
        Gaia-based proper motion in RA and Dec.
    nside : :class:`int`, optional, defaults to 2
        (NESTED) HEALPix nside, if targets are being parallelized.
        The default of 2 should be benign for serial processing.
    primary : :class:`array_like` or :class:`None`
        ``True`` for objects that should be passed through the selection.
    test : :class:`bool`, optional, defaults to ``False``
        If ``True``, then we're running unit tests and don't have to
        find and read every possible Gaia file.

    Returns
    -------
    :class:`array_like`
        ``True`` if the object is a Gaia "STD_DITHER_GAIA" target.
    :class:`array_like`
        A priority shift of 10*(25-rmag) based on `rmag`.

    Notes
    -----
    - This version (11/17/20) is version 70 on `the cmx wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gmag, dtype='?')

    issdg = primary.copy()

    # ADM not too bright in Gaia G, R.
    issdg &= gmag > 11.5
    issdg &= rmag > 11.5

    # ADM No obvious issues with the astrometry.
    issdg &= (aen < 1) & (paramssolved == 31)

    # ADM Finite proper motions.
    issdg &= np.isfinite(pmra) & np.isfinite(pmdec)

    # ADM Unique Gaia source (not a duplicated source).
    issdg &= ~dupsource

    # ADM CUT TO G < 19 where |b| < 20.
    blt20 = is_in_gal_box([ra, dec], [0, 360, -20, 20], radec=True)
    issdg &= (gmag < 19) | ~blt20

    # ADM remove any sources that have neighbors within 7"...
    # ADM for speed, run only sources for which issdg is still True.
    ii_true = np.where(issdg)[0]
    if len(ii_true) > 0:
        # ADM determine the pixels of interest.
        theta, phi = np.radians(90-dec), np.radians(ra)
        pixlist = list(set(hp.ang2pix(nside, theta, phi, nest=True)))
        # ADM read in the necessary Gaia files.
        fns = find_gaia_files_hp(nside, pixlist, neighbors=True)
        gaiaobjs = []
        gaiacols = ["RA", "DEC", "PHOT_G_MEAN_MAG", "PHOT_RP_MEAN_MAG"]
        for i, fn in enumerate(fns):
            if i % 25 == 0:
                log.info("Read {}/{} files for STD_DITHER_GAIA...t={:.1f}s"
                         .format(i, len(fns), time()-start))
            try:
                gaiaobjs.append(fitsio.read(fn, columns=gaiacols))
            except OSError:
                if test:
                    pass
                else:
                    msg = "failed to find or open the following file: (ffopen) "
                    msg += fn
                    log.critical(msg)
                    raise OSError

        gaiaobjs = np.concatenate(gaiaobjs)
        # ADM match the dither sources to the broader Gaia sources at 7".
        csdg = SkyCoord(ra[ii_true]*u.degree, dec[ii_true]*u.degree)
        cgaia = SkyCoord(gaiaobjs["RA"]*u.degree, gaiaobjs["DEC"]*u.degree)
        idsdg, idgaia, d2d, _ = cgaia.search_around_sky(csdg, 7*u.arcsec)
        # ADM remove source matches with d2d=0 (i.e. the source itself!).
        idgaia, idsdg = idgaia[d2d > 0], idsdg[d2d > 0]
        # ADM remove matches within 5 mags of a Gaia source.
        badmag = (
            (gmag[ii_true][idsdg] + 5 > gaiaobjs["PHOT_G_MEAN_MAG"][idgaia]) |
            (rmag[ii_true][idsdg] + 5 > gaiaobjs["PHOT_RP_MEAN_MAG"][idgaia]))
        issdg[ii_true[idsdg][badmag]] = False

    # ADM prioritize based on magnitude.
    prio = np.array((10*(25-rmag)).astype(int))

    return issdg, prio


def isSTD_dither_spec(gaiagmag=None, gaiarmag=None, obs_rflux=None,
                      isgood=None, primary=None):
    """Gaia stars for dithering-only tests during commissioning.

    Parameters
    ----------
    gaiagmag, gaiarmag : :class:`array_like` or :class:`None`
        The Gaia G-band and R-band mean magnitudes.
    obs_rflux : :class:`array_like` or :class:`None`
        The flux in nano-maggies in Legacy Surveys r-band WITHOUT any
        Galactic extinction correction. Used for prioritizing.
    isgood : :class:`array_like` or :class:`None`
        ``True`` for objects that pass the logic cuts in
        :func:`~desitarget.cmx.cmx_cuts.passesSTD_logic`.
    primary : :class:`array_like` or :class:`None`
        ``True`` for objects that should be passed through the selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a Gaia "STD_DITHER" target.
    :class:`array_like`
        A priority shift of 10*(25-rmag) based on r-band magnitude.

    Notes
    -----
    - This version (11/02/19) is version 48 on `the cmx wiki`_.
    """
    if primary is None:
        primary = np.ones_like(obs_rflux, dtype='?')

    isdither = primary.copy()
    # ADM passes all of the default logic cuts.
    isdither &= isgood

    # ADM don't target Gaia objects that have NaN magnitudes.
    # ADM remember to catch the single-object (non-array) case.
    if not _is_row(gaiagmag):
        gaiagmag[np.isnan(gaiagmag)] = 0.
        gaiarmag[np.isnan(gaiarmag)] = 0.

    # ADM not too bright in Gaia G, R (> 11.5 mags).
    isdither &= gaiagmag >= 11.5
    isdither &= gaiarmag >= 11.5

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
    - This version (08/30/18) is version 4 on `the cmx wiki`_.
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
    # ADM but brighter than STD_GAIA targets in g (g < 15).
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
        Directory to find commmissioning files to match, such as for the
        CALSPEC stars. If not specified, taken to be the value of the
        :envvar:`CMX_DIR` environment variable.
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


def isBACKUP(ra=None, dec=None, gaiagmag=None, primary=None):
    """BACKUP targets based on Gaia magnitudes.

    Parameters
    ----------
    ra, dec: :class:`array_like` or :class:`None`
        Right Ascension and Declination in degrees.
    gaiagmag: :class:`array_like` or :class:`None`
        Gaia-based g MAGNITUDE (not Galactic-extinction-corrected).
        (same units as `the Gaia data model`_).
    primary : :class:`array_like` or :class:`None`
        ``True`` for objects that should be passed through the selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a bright "BACKUP" target.
    :class:`array_like`
        ``True`` if and only if the object is a faint "BACKUP" target.
    """
    if primary is None:
        primary = np.ones_like(gaiagmag, dtype='?')

    isbackupbright = primary.copy()
    isbackupfaint = primary.copy()

    # ADM determine which sources are close to the Galaxy.
    in_gal = is_in_Galaxy([ra, dec], radec=True)

    # ADM bright targets are 13 < G < 16.
    isbackupbright &= gaiagmag >= 13
    isbackupbright &= gaiagmag < 16

    # ADM faint targets are 16 < G < 19.
    isbackupfaint &= gaiagmag >= 16
    isbackupfaint &= gaiagmag < 19
    # ADM and are "far from" the Galaxy.
    isbackupfaint &= ~in_gal

    return isbackupbright, isbackupfaint


def isFIRSTLIGHT(gaiadtype, cmxdir=None, nside=None, pixlist=None):
    """First light/Mini-SV targets via reading files from Arjun Dey.

    Parameters
    ----------
    gaiadtype: :class:`dtype`
        Data type (dtype) for Gaia-only CMX targets.
    cmxdir : :class:`str`, optional, defaults to :envvar:`CMX_DIR`
        Directory to find commmissioning files. If not specified,
        taken from the :envvar:`CMX_DIR` environment variable.
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPix nside used with `pixlist` and `bundlefiles`.
    pixlist : :class:`list` or `int`, optional, defaults to `None`
        Only return targets in a set of (NESTED) HEALpixels at `nside`.
        Useful for parallelizing, as input files will only be processed
        if they touch a pixel in the passed list.

    Returns
    -------
    :class:`array_like`
        bit values for each of the first light targets.
    :class:`array_like`
        Array of the first light targets munged into Gaia-only format.
    """
    # ADM retrieve/check the cmxdir.
    cmxdir = _get_cmxdir(cmxdir)
    # ADM get the M31 objects.
    cmx_target = []
    flout = []
    progs = ["M31", "ORI", "ROS", "M33",
             "SV0_MWS_CLUSTER", "SV0_MWS_CLUSTER_VERYBRIGHT"]
    for filenum, prog in enumerate(progs):
        # ADM flag whether this is not a "true" first light program.
        isfl = prog[:3] != 'SV0'

        cmxfile = os.path.join(cmxdir, "{}-targets.fits".format(prog))
        flobjsin = fitsio.read(cmxfile)

        # ADM create the gaia-only-like array.
        flobjsout = np.zeros(len(flobjsin), dtype=gaiadtype)

        # ADM set the Gaia Source ID and DR where possible.
        if isfl:
            gaiaid = []
            for flobjs in flobjsin["DESIGNATION"]:
                try:
                    # ADM the if/else is to maintain compatibility with
                    # ADM both fitsio 0.9.11 and 1.0+.
                    if isinstance(flobjs, np.bytes_):
                        gid = int(flobjs.decode().split("DR2")[-1])
                    else:
                        gid = int(flobjs.split("DR2")[-1])
                    gaiaid.append(gid)
                except ValueError:
                    gaiaid.append(-1)
            flobjsout['REF_ID'] = gaiaid
            flobjsout['REF_CAT'] = 'F1'
        else:
            flobjsout['REF_ID'] = flobjsin['REF_ID']
            flobjsout['REF_CAT'] = 'F1'

        # ADM transfer columns from Arjun's files to standard data model.
        for col in ["RA", "DEC"]:
            flobjsout[col] = flobjsin[col]
        for col in ["PMRA", "PMDEC"]:
            flobjsout[col] = flobjsin[col]
            if isfl:
                ii = flobjsin[col+"_ERROR"] != 0
                flobjsout[col+"_IVAR"][ii] = 1./(flobjsin[col+"_ERROR"][ii]**2.)
                flobjsout["REF_EPOCH"] = flobjsin["EPOCH"]
                flobjsout["GAIA_PHOT_G_MEAN_MAG"] = flobjsin["GAIA_G"]
            else:
                flobjsout["REF_EPOCH"] = flobjsin["REF_EPOCH"]
                flobjsout["GAIA_PHOT_G_MEAN_MAG"] = flobjsin["PHOT_G_MEAN_MAG"]
        # ADM add unique identifiers based on the file and row-in-file.
        flobjsout["GAIA_BRICKID"] = filenum
        flobjsout["GAIA_OBJID"] = np.arange(len(flobjsin))

        # ADM record the bit values for each class name. The if/else is
        # ADM to maintain compatibility with both fitsio 0.9.11 and 1.0+.
        if isfl:
            if isinstance(flobjsin["CLASS"][0], np.bytes_):
                cmx_target.append(
                    [cmx_mask[prog+"_"+c.decode().rstrip()]
                     for c in flobjsin["CLASS"]]
                )
            else:
                cmx_target.append(
                    [cmx_mask[prog+"_"+c.rstrip()] for c in flobjsin["CLASS"]]
                )
        else:
            cmx_target.append([cmx_mask[prog] for c in flobjsin])

        flout.append(flobjsout)

    cmx_target = np.concatenate(cmx_target)
    flout = np.concatenate(flout)

    # ADM restrict to only targets in a set of HEALPixels, if requested.
    if pixlist is not None:
        ii = is_in_hp(flout, nside, pixlist)
        cmx_target = cmx_target[ii]
        flout = flout[ii]

    return cmx_target, flout


def apply_cuts_gaia(numproc=4, cmxdir=None, nside=None, pixlist=None,
                    test=False):
    """Gaia-only-based CMX target selection, return target mask arrays.

    Parameters
    ----------
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    cmxdir : :class:`str`, optional, defaults to :envvar:`CMX_DIR`
        Directory to find commmissioning files to match, such as for the
        CALSPEC stars. If not specified, taken to be the value of the
        :envvar:`CMX_DIR` environment variable.
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPix nside used with `pixlist` and `bundlefiles`.
    pixlist : :class:`list` or `int`, optional, defaults to `None`
        Only return targets in a set of (NESTED) HEALpixels at `nside`.
        Useful for parallelizing, as input files will only be processed
        if they touch a pixel in the passed list.
    test : :class:`bool`, optional, defaults to ``False``
        If ``True``, then we're running unit tests and don't have to
        find and read every possible Gaia file.

    Returns
    -------
    :class:`~numpy.ndarray`
        Commissioning target selection bitmask flags for each object.
    :class:`~numpy.ndarray`
        numpy structured array of Gaia sources that were read in from
        file for the passed pixel constraints (or no pixel constraints).
    :class:`array_like`
        a priority shift of 10*(25-rmag) based on GAIA_PHOT_RP_MEAN_MAG.
        (for STD_DITHER_GAIA sources).

    Notes
    -----
        - May take a long time if no pixel constraints are passed.
        - Only run on Gaia-only target selections.
        - The environment variable $GAIA_DIR must be set.

    See desitarget.cmx.cmx_targetmask.cmx_mask for bit definitions.
    """
    from desitarget.gfa import all_gaia_in_tiles
    # ADM No Gaia-only CMX target classes are fainter than G=20.
    gaiaobjs = all_gaia_in_tiles(maglim=20, numproc=numproc, allsky=True,
                                 mindec=-90, mingalb=0, addobjid=True,
                                 nside=nside, pixlist=pixlist, addparams=True)
    # ADM the convenience function we use adds an empty TARGETID
    # ADM field which we need to remove before finalizing.
    gaiaobjs = rfn.drop_fields(gaiaobjs, "TARGETID")

    primary = np.ones_like(gaiaobjs, dtype=bool)
    priority_shift = np.zeros_like(gaiaobjs, dtype=int)

    # ADM the relevant input quantities.
    ra, dec = gaiaobjs["RA"], gaiaobjs["DEC"]
    pmra, pmdec = gaiaobjs["PMRA"], gaiaobjs["PMDEC"]
    gaiagmag = gaiaobjs["GAIA_PHOT_G_MEAN_MAG"]
    gaiarmag = gaiaobjs["GAIA_PHOT_RP_MEAN_MAG"]
    aen = gaiaobjs["GAIA_ASTROMETRIC_EXCESS_NOISE"]
    dupsource = gaiaobjs["GAIA_DUPLICATED_SOURCE"]
    paramssolved = gaiaobjs["GAIA_ASTROMETRIC_PARAMS_SOLVED"]

    # ADM determine if an object matched a CALSPEC standard.
    std_calspec = isSTD_calspec(
        ra=ra, dec=dec, cmxdir=cmxdir, primary=primary
    )

    # ADM determine if an object is a BACKUP target.
    backup_bright, backup_faint = isBACKUP(
        ra=ra, dec=dec, gaiagmag=gaiagmag, primary=primary
    )

    # ADM grab the information on the FIRST LIGHT targets.
    fl_target, flobjs = isFIRSTLIGHT(gaiaobjs.dtype, cmxdir=cmxdir,
                                     nside=nside, pixlist=pixlist)

    sdg, prio = isSTD_dither_gaia(
        ra=ra, dec=dec, gmag=gaiagmag, rmag=gaiarmag, aen=aen,
        paramssolved=paramssolved, dupsource=dupsource, pmra=pmra, pmdec=pmdec,
        nside=nside, primary=primary, test=test
    )

    # ADM the priority shift for Gaia-only cmx sources.
    priority_shift[sdg] = prio[sdg]

    # ADM Construct the target flag bits.
    cmx_target = std_calspec * cmx_mask.STD_CALSPEC
    cmx_target |= backup_bright * cmx_mask.BACKUP_BRIGHT
    cmx_target |= backup_faint * cmx_mask.BACKUP_FAINT
    cmx_target |= sdg * cmx_mask.STD_DITHER_GAIA

    # ADM add in the first light program targets.
    cmx_target = np.concatenate([cmx_target, fl_target])
    gaiaobjs = np.concatenate([gaiaobjs, flobjs])
    priority_shift = np.concatenate([priority_shift, np.zeros_like(fl_target)])

    return cmx_target, gaiaobjs, priority_shift


def apply_cuts(objects, cmxdir=None, noqso=False):
    """Commissioning (cmx) target selection, return target mask arrays.

    Parameters
    ----------
    objects : :class:`~numpy.ndarray`
        numpy structured array with UPPERCASE columns needed for
        target selection, OR a string tractor/sweep filename
    cmxdir : :class:`str`, optional, defaults to :envvar:`CMX_DIR`
        Directory to find commmissioning files to which to match, such
        as the CALSPEC stars. If not specified, the cmx directory is
        taken to be the value of 2:envvar:`CMX_DIR`.
    noqso : :class:`boolean`, optional, defaults to ``False``
        If passed, do not run the quasar selection. All QSO bits will be
        set to zero. Intended use is to speed unit tests.

    Returns
    -------
    :class:`~numpy.ndarray`
        commissioning target selection bitmask flags for each object.
    :class:`array_like`
        a priority shift of 10*(25-rmag) based on r-band magnitude.
        (for `STD_DITHER`, `STD_GAIA` sources).

    See desitarget.cmx.cmx_targetmask.cmx_mask for bit definitions.
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

    photsys_north, photsys_south, obs_rflux, gflux, rflux, zflux,                     \
        w1flux, w2flux, gfiberflux, rfiberflux, zfiberflux,                           \
        objtype, release, ra, dec, gfluxivar, rfluxivar, zfluxivar, w1fluxivar,       \
        gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,                         \
        gfracmasked, rfracmasked, zfracmasked,                                        \
        gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,                      \
        gsnr, rsnr, zsnr, w1snr, w2snr, dchisq, deltaChi2, maskbits, refcat =         \
        _prepare_optical_wise(objects)

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

    # ADM default for target classes we WON'T process is all False.
    tcfalse = primary & False

    # ADM determine if an object passes the default logic for cmx stars.
    isgood, istight = passesSTD_logic(
        gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
        objtype=objtype, gaia=gaia, pmra=pmra, pmdec=pmdec,
        aen=gaiaaen, dupsource=gaiadupsource, paramssolved=gaiaparamssolved,
        primary=primary
    )

    # ADM determine if an object is a "STD_GAIA" star.
    # ADM and priority shift.
    std_dither, shift_dither = isSTD_dither(
        obs_gflux=obs_gflux, obs_rflux=obs_rflux, obs_zflux=obs_zflux,
        isgood=isgood, primary=primary
    )

    # ADM determine if an object is a "STD_DITHER" star.
    # ADM and priority shift. Note the tighter isgood cuts.
    std_dither_spec, shift_dither_spec = isSTD_dither_spec(
        gaiagmag=gaiagmag, gaiarmag=gaiarmag, obs_rflux=obs_rflux,
        isgood=istight, primary=primary
    )

    # ADM determine if an object is a bright test star.
    std_test = isSTD_test(
        obs_gflux=obs_gflux, obs_rflux=obs_rflux, obs_zflux=obs_zflux,
        isgood=isgood, primary=primary
    )

    # ADM determine if an object matched a CALSPEC standard.
    std_calspec = isSTD_calspec(
        ra=ra, dec=dec, cmxdir=cmxdir, primary=primary
    )

    # ADM determine if an object is SV0_BGS.
    sv0_bgs = isSV0_BGS(
        gflux=gflux, rflux=rflux, zflux=zflux,
        w1flux=w1flux, w2flux=w2flux, rfiberflux=rfiberflux,
        gnobs=gnobs, rnobs=rnobs, znobs=znobs, gfracmasked=gfracmasked,
        rfracmasked=rfracmasked, zfracmasked=zfracmasked,
        gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
        gfracin=gfracin, rfracin=rfracin, zfracin=zfracin,
        gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
        maskbits=maskbits, Grr=Grr, w1snr=w1snr, gaiagmag=gaiagmag,
        objtype=objtype, primary=primary
    )

    # ADM determine if an object is SV0_MWS, WD or SV0_MWS_FAINT.
    sv0_mws, sv0_wd, sv0_mws_faint = isSV0_MWS(
        rflux=rflux, obs_rflux=obs_rflux, objtype=objtype,
        gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag,
        pmra=pmra, pmdec=pmdec, parallax=parallax,
        parallaxerr=parallaxerr, parallaxovererror=parallaxovererror,
        photbprpexcessfactor=gaiabprpfactor,
        astrometricsigma5dmax=gaiasigma5dmax,
        gaiaaen=gaiaaen, paramssolved=gaiaparamssolved,
        galb=galb, gaia=gaia, primary=primary
    )

    # ADM determine if an object is SV0_LRG.
    sv0_lrg = isSV0_LRG(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
        rfiberflux=rfiberflux, zfiberflux=zfiberflux,
        gflux_snr=gsnr, rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr,
        gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits,
        primary=primary
    )

    # ADM determine if an object is SV0_ELG.
    sv0_elg = isSV0_ELG(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
        gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, gfiberflux=gfiberflux,
        gnobs=gnobs, rnobs=rnobs, znobs=znobs,
        maskbits=maskbits, primary=primary
    )

    # ADM determine if an object is SV0_QSO.
    if noqso:
        # ADM don't run quasar cuts if requested, for speed.
        sv0_qso, sv0_qso_z5 = tcfalse, tcfalse
    else:
        sv0_qso, sv0_qso_z5 = isSV0_QSO(
            primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
            w1flux=w1flux, w2flux=w2flux,
            gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, w1snr=w1snr, w2snr=w2snr,
            gnobs=gnobs, rnobs=rnobs, znobs=znobs,
            objtype=objtype, dchisq=dchisq, maskbits=maskbits
        )

    # ADM run the SV0 STD target types for both faint and bright.
    # ADM Make sure to pass all of the needed columns! At one point we stopped
    # ADM passing objtype, which meant no standards were being returned.
    sv0_std_classes = []
    for bright in [False, True]:
        sv0_std_classes.append(
            isSV0_STD(
                primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                gfracmasked=gfracmasked, rfracmasked=rfracmasked, objtype=objtype,
                zfracmasked=zfracmasked, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
                gaia=gaia, astrometricexcessnoise=gaiaaen, paramssolved=gaiaparamssolved,
                pmra=pmra, pmdec=pmdec, parallax=parallax, dupsource=gaiadupsource,
                gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag, bright=bright
            )
        )
    sv0_std_faint, sv0_std_bright = sv0_std_classes

    # ADM the nominal main survey cuts for standard stars. These are currently
    # ADM identical to the SV0 cuts, so treat accordingly:
    std_faint, std_bright = sv0_std_classes

    # ADM incorporate target classes from the Main Survey for Mini-SV.
    # ADM this should be the combination of all of the northerna and all
    # ADM of the southern cuts.
    south_cuts = [False, True]

    # ADM Main Survey LRGs.
    # ADM initially set everything to arrays of False for the LRGs
    # ADM the zeroth element stores northern targets bits (south=False).
    lrg_classes = [tcfalse, tcfalse]
    for south in south_cuts:
        lrg_classes[int(south)] = isLRG_MS(
            primary=primary,
            gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
            zfiberflux=zfiberflux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
            rfluxivar=rfluxivar, zfluxivar=zfluxivar, w1fluxivar=w1fluxivar,
            maskbits=maskbits, south=south
        )
    lrg_north, lrg_south = lrg_classes
    # ADM combine LRG target bits for an LRG target based on any imaging.
    mini_sv_lrg = (lrg_north & photsys_north) | (lrg_south & photsys_south)

    # ADM Main Survey ELGs.
    # ADM initially set everything to arrays of False for the ELGs
    # ADM the zeroth element stores northern targets bits (south=False).
    elg_classes = [tcfalse, tcfalse]
    for south in south_cuts:
        elg_classes[int(south)] = isELG_MS(
            primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
            gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
            gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits,
            south=south
        )
    elg_north, elg_south = elg_classes
    # ADM combine ELG target bits for an ELG target based on any imaging.
    mini_sv_elg = (elg_north & photsys_north) | (elg_south & photsys_south)

    # ADM Main Survey QSOs.
    # ADM initially set everything to arrays of False for the QSOs
    # ADM the zeroth element stores northern targets bits (south=False).
    qso_classes = [[tcfalse, tcfalse], [tcfalse, tcfalse]]
    # ADM don't run quasar cuts if requested, for speed.
    if not noqso:
        for south in south_cuts:
            qso_classes[int(south)] = isQSO_MS(
                primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2,
                maskbits=maskbits, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                objtype=objtype, release=release, ra=ra, dec=dec, south=south
            )
    qso_north, qso_hiz_north = qso_classes[0]
    qso_south, qso_hiz_south = qso_classes[1]

    # ADM combine QSO target bits for a QSO target based on any imaging.
    mini_sv_qso = (qso_north & photsys_north) | (qso_south & photsys_south)

    # ADM Main Survey BGS (Bright).
    # ADM initially set everything to arrays of False for the BGS
    # ADM the zeroth element stores northern targets bits (south=False).
    bgs_classes = [tcfalse, tcfalse]
    for south in south_cuts:
        bgs_classes[int(south)] = isBGS_MS(
            rfiberflux=rfiberflux, gflux=gflux, rflux=rflux, zflux=zflux,
            w1flux=w1flux, w2flux=w2flux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
            gfracmasked=gfracmasked, rfracmasked=rfracmasked, zfracmasked=zfracmasked,
            gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
            gfracin=gfracin, rfracin=rfracin, zfracin=zfracin,
            gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
            maskbits=maskbits, Grr=Grr, refcat=refcat, w1snr=w1snr, gaiagmag=gaiagmag,
            objtype=objtype, primary=primary, south=south, targtype="bright"
        )
    bgs_north, bgs_south = bgs_classes

    # ADM combine BGS targeting bits for a BGS selected in any imaging.
    mini_sv_bgs_bright = (
        bgs_north & photsys_north) | (bgs_south & photsys_south)

    # ADM explicitly restrict "bright" target classes to G/g >= 16.
    # ADM clip to avoid NaN on np.log10 of -ve numbers.
    obs_gmag = 22.5-2.5*np.log10(np.clip(obs_gflux, 1e-16, 1e16))
    too_bright = (obs_gmag < 16) | (gaia & (gaiagmag < 16))
    sv0_bgs &= ~too_bright
    sv0_mws &= ~too_bright
    sv0_wd &= ~too_bright
    mini_sv_bgs_bright &= ~too_bright

    # ADM Construct the target flag bits.
    cmx_target = std_dither * cmx_mask.STD_GAIA
    cmx_target |= std_dither_spec * cmx_mask.STD_DITHER
    cmx_target |= std_test * cmx_mask.STD_TEST
    cmx_target |= std_calspec * cmx_mask.STD_CALSPEC
    cmx_target |= sv0_std_faint * cmx_mask.SV0_STD_FAINT
    cmx_target |= sv0_std_bright * cmx_mask.SV0_STD_BRIGHT
    cmx_target |= sv0_bgs * cmx_mask.SV0_BGS
    cmx_target |= sv0_mws * cmx_mask.SV0_MWS
    cmx_target |= sv0_lrg * cmx_mask.SV0_LRG
    cmx_target |= sv0_elg * cmx_mask.SV0_ELG
    cmx_target |= sv0_qso * cmx_mask.SV0_QSO
    cmx_target |= sv0_qso_z5 * cmx_mask.SV0_QSO_Z5
    cmx_target |= sv0_wd * cmx_mask.SV0_WD
    cmx_target |= std_faint * cmx_mask.STD_FAINT
    cmx_target |= std_bright * cmx_mask.STD_BRIGHT
    cmx_target |= mini_sv_lrg * cmx_mask.MINI_SV_LRG
    cmx_target |= mini_sv_elg * cmx_mask.MINI_SV_ELG
    cmx_target |= mini_sv_qso * cmx_mask.MINI_SV_QSO
    cmx_target |= mini_sv_bgs_bright * cmx_mask.MINI_SV_BGS_BRIGHT
    cmx_target |= sv0_mws_faint * cmx_mask.SV0_MWS_FAINT

    # ADM update the priority with any shifts.
    # ADM we may need to update this logic if there are other shifts.
    priority_shift[std_dither] = shift_dither[std_dither]
    priority_shift[std_dither_spec] = shift_dither_spec[std_dither_spec]

    return cmx_target, priority_shift


def select_targets(infiles, numproc=4, cmxdir=None, noqso=False,
                   nside=None, pixlist=None, bundlefiles=None, extra=None,
                   resolvetargs=True, backup=True, test=False):
    """Process input files in parallel to select commissioning (cmx) targets

    Parameters
    ----------
    infiles : :class:`list` or `str`
        List of input filenames (tractor/sweep files) OR one filename.
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    cmxdir : :class:`str`, optional, defaults to :envvar:`CMX_DIR`
        Directory to find commmissioning files to which to match, such
        as the CALSPEC stars. If not specified, the cmx directory is
        taken to be the value of :envvar:`CMX_DIR`.
    noqso : :class:`boolean`, optional, defaults to ``False``
        If passed, do not run the quasar selection. All QSO bits will be
        set to zero. Intended use is to speed unit tests.
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPix nside used with `pixlist` and `bundlefiles`.
    pixlist : :class:`list` or `int`, optional, defaults to `None`
        Only return targets in a set of (NESTED) HEALpixels at `nside`.
        Useful for parallelizing, as input files will only be processed
        if they touch a pixel in the passed list.
    bundlefiles : :class:`int`, defaults to `None`
        If not `None`, then, instead of selecting gfas, print the slurm
        script to run in pixels at `nside`. Is an integer rather than
        a boolean for historical reasons.
    extra : :class:`str`, optional
        Extra command line flags to be passed to the executable lines in
        the output slurm script. Used in conjunction with `bundlefiles`.
    resolvetargs : :class:`boolean`, optional, defaults to ``True``
        If ``True``, resolve overlapping north/south Legacy Surveys
        targets into a set of unique sources based on location.
    backup : :class:`boolean`, optional, defaults to ``True``
        If ``True``, also run the Gaia-only BACKUP_BRIGHT/FAINT targets.
    test : :class:`bool`, optional, defaults to ``False``
        If ``True``, then we're running unit tests and don't have to
        find and read every possible Gaia file.

    Returns
    -------
    :class:`~numpy.ndarray`
        The subset of input targets which pass the cmx cuts, including an extra
        column for `CMX_TARGET`.

    Notes
    -----
        - if numproc==1, use serial code instead of parallel.
    """
    # ADM the code can have memory issues for nside=2 with large numproc.
    if nside is not None and nside < 4 and numproc > 8:
        msg = 'Memory may be an issue near Plane for nside < 4 and numproc > 8'
        log.warning(msg)

    # -Convert single file to list of files.
    if isinstance(infiles, str):
        infiles = [infiles, ]

    # -Sanity check that files exist before going further.
    for filename in infiles:
        if not os.path.exists(filename):
            raise ValueError("{} doesn't exist".format(filename))

    # ADM retrieve/check the cmxdir.
    cmxdir = _get_cmxdir(cmxdir)

    # ADM if the pixlist option was sent, we'll need to
    # ADM know which HEALPixels touch each file.
    if pixlist is not None:
        filesperpixel, _, _ = sweep_files_touch_hp(
            nside, pixlist, infiles)

    # ADM if the bundlefiles option was sent, call the packing code.
    if bundlefiles is not None:
        # ADM determine if one or two input directories were passed.
        surveydirs = list(set([os.path.dirname(fn) for fn in infiles]))
        bundle_bricks([0], bundlefiles, nside, gather=False, extra=extra,
                      prefix='cmx_targets', surveydirs=surveydirs)
        return

    # ADM restrict to only input files in a set of HEALPixels, if requested.
    if pixlist is not None:
        # ADM a hack to ensure we have the correct targeting data model.
        # ADM outside of the Legacy Surveys footprint.
        dummy = infiles[0]
        infiles = list(set(np.hstack([filesperpixel[pix] for pix in pixlist])))
        if len(infiles) == 0:
            log.info('ZERO sweep files in passed pixel list!!!')
            log.info('Run with dummy sweep file to write Gaia-only objects...')
            infiles = [dummy]
        log.info("Processing files in (nside={}, pixel numbers={}) HEALPixels"
                 .format(nside, pixlist))

    # ADM a little more information if we're slurming across nodes.
    if os.getenv('SLURMD_NODENAME') is not None:
        log.info('Running on Node {}'.format(os.getenv('SLURMD_NODENAME')))

    def _finalize_targets(objects, cmx_target, priority_shift=None, gaiadr=None):
        keep = (cmx_target != 0)
        objects = objects[keep]
        cmx_target = cmx_target[keep]
        if priority_shift is not None:
            priority_shift = priority_shift[keep]
        if gaiadr is not None:
            gaiadr = gaiadr[keep]

        # -Add *_target mask columns
        # ADM note that only cmx_target is defined for commissioning
        # ADM so just pass that around
        targets = finalize(objects, cmx_target, cmx_target, cmx_target,
                           survey='cmx', gaiadr=gaiadr)
        # ADM shift the priorities of targets with functional priorities.
        if priority_shift is not None:
            targets["PRIORITY_INIT"] += priority_shift

        # ADM resolve any duplicates between imaging data releases.
        if resolvetargs and gaiadr is None:
            targets = resolve(targets)

        return targets

    # -functions to run on every brick/sweep file
    def _select_targets_file(filename):
        '''Returns targets in filename that pass the cuts'''
        objects = io.read_tractor(filename)
        cmx_target, priority_shift = apply_cuts(objects,
                                                cmxdir=cmxdir, noqso=noqso)
        return _finalize_targets(objects, cmx_target,
                                 priority_shift=priority_shift)

    # Counter for number of bricks processed;
    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    t0 = time()

    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if nbrick % 20 == 0 and nbrick > 0:
            elapsed = time() - t0
            rate = elapsed / nbrick
            log.info('{} files; {:.1f} secs/file; {:.1f} total mins elapsed'
                     .format(nbrick, rate, elapsed/60.))
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

    if backup:
        # ADM also process Gaia-only targets.
        log.info('Retrieve additional Gaia-only (backup) objects...t = {:.1f} mins'
                 .format((time()-t0)/60))

        # ADM force to no more than numproc=4 for I/O limited (Gaia) processes.
        numproc4 = numproc
        if numproc4 > 4:
            log.info('Forcing numproc to 4 for I/O limited parts of code')
            numproc4 = 4

        # ADM set the target bits that are based only on Gaia.
        cmx_target, gaiaobjs, priority_shift = apply_cuts_gaia(
            numproc=numproc4, cmxdir=cmxdir, nside=nside, pixlist=pixlist,
            test=test)

        # ADM determine the Gaia Data Release.
        gaiadr = gaia_dr_from_ref_cat(gaiaobjs["REF_CAT"])

        # ADM add the relevant bits and IDs to the Gaia targets.
        gaiatargs = _finalize_targets(gaiaobjs, cmx_target,
                                      priority_shift=priority_shift,
                                      gaiadr=gaiadr)

        # ADM make the Gaia-only data structure resemble the main targets.
        gaiatargets = np.zeros(len(gaiatargs), dtype=targets.dtype)
        for col in set(gaiatargs.dtype.names).intersection(set(targets.dtype.names)):
            gaiatargets[col] = gaiatargs[col]

        # ADM remove any duplicates. Order is important here, as np.unique
        # ADM keeps the first occurence, and we want to retain sweeps
        # ADM information as much as possible.
        if len(infiles) > 0:
            alltargs = np.concatenate([targets, gaiatargets])
            # ADM Retain First Light objects as a special program.
            ii = ((alltargs["REF_CAT"] != b'F1') & (alltargs["REF_CAT"] != 'F1'))
            # ADM Retain all non-Gaia sources, which have REF_ID of -1 or 0
            # ADM and so are all duplicates on REF_ID.
            ii &= alltargs["REF_ID"] > 0
            # ADM Always retain the STD_DITHER_GAIA targets, even if they're
            # ADM duplicated in the Legacy Surveys footprint.
            ii &= (alltargs["CMX_TARGET"] & cmx_mask.STD_DITHER_GAIA) == 0
            targs = alltargs[ii]
            _, ind = np.unique(targs["REF_ID"], return_index=True)
            targs = targs[ind]
            targets = np.concatenate([targs, alltargs[~ii]])
        else:
            targets = gaiatargets

    # ADM restrict to only targets in a set of HEALPixels, if requested.
    if pixlist is not None:
        ii = is_in_hp(targets, nside, pixlist)
        targets = targets[ii]

    return targets
