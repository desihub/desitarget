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

from pkg_resources import resource_filename

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
        ``True`` if and only if the object passes the logic cuts for cmx stars.

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

    # ADM calculate the overall proper motion magnitude.
    # ADM inexplicably I'm getting a Runtimewarning here for
    # ADM a few values in the sqrt, so I'm catching it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = np.sqrt(pmra**2. + pmdec**2.)

    # ADM a proper motion larger than 2 mas/yr.
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
    - Returns the equivalent of ALL BGS classes.
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
              gaiaaen=None, galb=None, gaia=None, primary=None):
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
    gaiaaen : :class:`array_like` or :class:`None`
        Gaia-based measure of Astrometric Excess Noise.
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
    - Returns the equivalent of ALL MWS classes.
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

    # ADM do not target any WDs for which entries are NaN
    # ADM and turn off the NaNs for those entries.
    if photbprpexcessfactor is not None:
        nans = (np.isnan(gaiagmag) | np.isnan(gaiabmag) | np.isnan(gaiarmag) |
                np.isnan(parallax) | np.isnan(photbprpexcessfactor))
    else:
        nans = (np.isnan(gaiagmag) | np.isnan(gaiabmag) | np.isnan(gaiarmag) |
                np.isnan(parallax))
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

    # ADM return any object that passes any of the MWS cuts.
    return ismws | isnear | iswd


def isSV0_LRG(gflux=None, rflux=None, zflux=None, w1flux=None,
              rflux_snr=None, zflux_snr=None, w1flux_snr=None,
              gflux_ivar=None, primary=None):
    """Target Definition of LRG. Returns a boolean array.

    Parameters
    ----------
    - See :func:`~desitarget.cuts.set_target_bits` for parameters.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is an LRG target.

    Notes
    -----
    - This version (03/19/19) is version 56 on `the SV wiki`_.
    """
    # ADM LRG SV0 targets.
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    lrg &= notinLRG_mask(primary=primary, rflux=rflux, zflux=zflux, w1flux=w1flux,
                         rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr)

    # ADM pass the lrg that pass cuts as primary, to restrict to the
    # ADM sources that weren't in a mask/logic cut.
    lrg_all = isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                           gflux_ivar=gflux_ivar, primary=lrg)

    return lrg_all


def notinLRG_mask(primary=None, rflux=None, zflux=None, w1flux=None,
                  rflux_snr=None, zflux_snr=None, w1flux_snr=None):
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
    lrg &= (zflux_snr > 0) & (zflux > 0)   # ADM quality in z.
    lrg &= (w1flux_snr > 4) & (w1flux > 0)  # ADM quality in W1.

    return lrg


def isLRG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 gflux_ivar=None, primary=None):
    """See :func:`~desitarget.sv1.sv1_cuts.isLRG` for details.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    # ADM take care to explicitly copy these individually to guard
    # ADM against accidentally changing the type of the bitmasks.
    lrg, lrg1, lrg2 = primary.copy(), primary.copy(), primary.copy()
    lrg3, lrg4, lrg5 = primary.copy(), primary.copy(), primary.copy()

    # ADM safe as these fluxes are set to > 0 in notinLRG_mask.
    gmag = 22.5 - 2.5 * np.log10(gflux.clip(1e-7))
    rmag = 22.5 - 2.5 * np.log10(rflux.clip(1e-7))
    zmag = 22.5 - 2.5 * np.log10(zflux.clip(1e-7))
    w1mag = 22.5 - 2.5 * np.log10(w1flux.clip(1e-7))

    # subsample 1: nominal optical + nominal IR:
    lrg1 &= zmag - w1mag > 0.8*(rmag-zmag) - 0.735    # non-stellar cut
    lrg1 &= zmag < 20.365                             # faint limit
    lrg1 &= rmag - zmag > 0.85                        # broad color box
    lrg1 &= (rmag - zmag > 1.25) | ((gmag - rmag > 1.655) & (gflux_ivar > 0))  # Low-z cut
    lrg_opt = rmag - zmag > (zmag - 17.105) / 1.8     # sliding optical cut
    lrg_ir = rmag - w1mag > (w1mag - 17.723) / 0.385  # sliding IR cut
    lrg1 &= lrg_opt | lrg_ir

    # subsample 2: optical + IR + low-z extension:
    lrg2 &= zmag - w1mag > 0.8*(rmag-zmag) - 0.735  # non-stellar cut
    lrg2 &= zmag < 20.365                           # faint limit
    lrg2 &= rmag - zmag > 0.85                      # broad color box
    lrg2 &= (rmag - zmag > 1.25) | ((gmag - rmag > 1.655) & (gflux_ivar > 0))  # Low-z cut
    lrg_opt = ((rmag - zmag > (zmag - 17.105) / 1.8) |
               (zmag < 19.655))                             # sliding optical cut with low-z extension
    lrg_ir = ((rmag - w1mag > (w1mag - 17.723) / 0.385) |
              ((w1mag < 19.139) & (rmag - w1mag < 1.833)))  # sliding IR cut with low-z extension
    lrg2 &= lrg_opt | lrg_ir

    # subsample 3: optical + IR + high-z extension:
    lrg3 &= zmag - w1mag > 0.8*(rmag-zmag) - 0.735  # non-stellar cut
    lrg3 &= zmag < 20.755                           # extended faint limit
    lrg3 &= rmag - zmag > 0.85                      # broad color box
    lrg3 &= (rmag - zmag > 1.25) | ((gmag - rmag > 1.655) & (gflux_ivar > 0))  # Low-z cut
    lrg_opt = rmag - zmag > (zmag - 17.105) / 1.8     # sliding optical cut
    lrg_ir = rmag - w1mag > (w1mag - 17.723) / 0.385  # sliding IR cut
    lrg3 &= lrg_opt | lrg_ir

    # subsample 4: optical + IR + low-z + high-z + relaxed cuts
    # (relaxed stellar rejection + relaxed sliding cuts + relaxed g-r cut):
    lrg4 &= zmag - w1mag > 0.8*(rmag-zmag) - 0.935  # relaxed non-stellar cut
    lrg4 &= zmag < 20.755                           # extended faint limit
    lrg4 &= rmag - zmag > 0.85                      # broad color box
    lrg4 &= (rmag - zmag > 1.25) | ((gmag - rmag > 1.455) & (gflux_ivar > 0))  # relaxed low-z cut
    lrg_opt = ((((zmag - 22.1) / 1.3)**2 + (rmag - zmag + 1.04)**2 > 3.0**2) |
               (zmag < 19.655))                             # curved sliding optical cut with low-z extension
    lrg_ir = (((w1mag - 21.)**2 + ((rmag - w1mag - 0.47) / 1.5)**2 > 2.5**2) |
              ((w1mag < 19.139) & (rmag - w1mag < 1.833)))  # curved sliding IR cut with low-z extension
    lrg4 &= lrg_opt | lrg_ir

    # subsample 5 (this is a superset of all other subsamples):
    # optical + IR + low-z + high-z + more relaxed cuts
    # (relaxed stellar rejection + relaxed sliding cuts + no g-r cut + relaxed broad r-z cut):
    lrg5 &= zmag - w1mag > 0.8*(rmag-zmag) - 0.935  # relaxed non-stellar cut
    lrg5 &= zmag < 20.755                           # extended faint limit
    lrg5 &= rmag - zmag > 0.75                      # relaxed broad color box
    lrg_opt = ((((zmag - 22.1) / 1.3)**2 + (rmag - zmag + 1.04)**2 > 3.0**2) |
               (zmag < 19.655))                             # curved sliding optical cut with low-z extension
    lrg_ir = (((w1mag - 21.)**2 + ((rmag - w1mag - 0.47) / 1.5)**2 > 2.5**2) |
              ((w1mag < 19.139) & (rmag - w1mag < 1.833)))  # curved sliding IR cut with low-z extension
    lrg5 &= lrg_opt | lrg_ir

    lrg = lrg1 | lrg2 | lrg3 | lrg4 | lrg5

    return lrg


def isSV0_QSO(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
              objtype=None, release=None, dchisq=None, brightstarinblob=None,
              primary=None):
    """Early SV QSO target class using random forest. Returns a boolean array.

    Parameters
    ----------
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.

    Returns
    -------
    :class:`array_like`
        ``True`` for objects that pass the quasar color/morphology/logic cuts.

    Notes
    -----
    - This version (06/05/19) is version 68 on `the SV wiki`_.
    """
    # BRICK_PRIMARY
    if primary is None:
        primary = np.ones_like(gflux, dtype=bool)

    # Build variables for random forest.
    nFeatures = 11  # Number of attributes describing each object to be classified by the rf.
    nbEntries = rflux.size
    # ADM shift the northern photometry to the southern system.
    # ADM we don't need to exactly correspond to SV for SV0.
    # gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    # ADM photOK here should ensure (g > 0.) & (r > 0.) & (z > 0.) & (W1 > 0.) & (W2 > 0.)
    colors, r, photOK = _getColors(nbEntries, nFeatures, gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d(r)

    # ADM Preselection to speed up the process
    rMax = 23.0  # r < 23.0 (different for SV)
    rMin = 17.5  # r > 17.5
    preSelection = (r < rMax) & (r > rMin) & photOK & primary

    # ADM relaxed morphology cut for SV.
    # ADM we never target sources with dchisq[..., 0] = 0, so force
    # ADM those to have large values of morph2 to avoid divide-by-zero.
    d1, d0 = dchisq[..., 1], dchisq[..., 0]
    bigmorph = np.array(np.zeros_like(d0) + 1e9)
    dcs = np.divide(d1 - d0, d0, out=bigmorph, where=d0 != 0)
    morph2 = dcs < 0.02
    preSelection &= _psflike(objtype) | morph2

    # CAC Reject objects flagged inside a blob.
    preSelection &= ~brightstarinblob

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
        # Compute optimized proba cut (all different for SV/main).
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


def isSV0_ELG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
              gsnr=None, rsnr=None, zsnr=None, maskbits=None, primary=None):
    """Definition of ELG target classes. Returns a boolean array.

    Parameters
    ----------
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is an SV0 ELG target.

    Notes
    -----
    - This version (03/19/19) is version 76 on `the SV wiki`_.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    elg &= notinELG_mask(maskbits=maskbits, gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                         primary=primary)

    elg &= isELG_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                        w2flux=w2flux, primary=primary)

    return elg


def notinELG_mask(maskbits=None, gsnr=None, rsnr=None, zsnr=None, primary=None):
    """Standard set of masking cuts used by all ELG target selection classes.
    (see :func:`~desitarget.cuts.set_target_bits` for parameters).
    """
    if primary is None:
        primary = np.ones_like(maskbits, dtype='?')
    elg = primary.copy()

    # ADM good signal-to-noise in all bands.
    elg &= (gsnr > 0) & (rsnr > 0) & (zsnr > 0)
    # ADM ALLMASK (5, 6, 7), BRIGHT OBJECT (1, 11, 12, 13) bits not set.
    for bit in [1, 5, 6, 7, 11, 12, 13]:
        elg &= ((maskbits & 2**bit) == 0)

    return elg


def isELG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 w2flux=None, primary=None):
    """Color cuts for ELG target selection classes
    (see, e.g., :func:`desitarget.cuts.set_target_bits` for parameters).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    # ADM work in magnitudes instead of fluxes. NOTE THIS IS ONLY OK AS
    # ADM the snr masking in ALL OF g, r AND z ENSURES positive fluxes.
    g = 22.5 - 2.5*np.log10(gflux.clip(1e-16))
    r = 22.5 - 2.5*np.log10(rflux.clip(1e-16))
    z = 22.5 - 2.5*np.log10(zflux.clip(1e-16))
    # ADM this is a color defined perpendicularly to the negative slope
    # ADM cut; coii thus follows the OII flux gradient.
    coii = (g - r) + 1.2*(r - z)

    elg &= g > 20                       # bright cut.
    elg &= r - z > -1.0                 # blue cut.
    elg &= g - r < -1.2*(r - z) + 2.5   # OII flux cut.

    elg &= (g - r < 0.2) | (g - r < 1.15*(r - z) - 0.35)  # remove stars and low-z galaxies.
    elg &= coii < 1.6 - 7.2*(g - 23.6)  # sliding cut.

    return elg


def isSTD(gflux=None, rflux=None, zflux=None, primary=None,
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

    photsys_north, photsys_south, obs_rflux, gflux, rflux, zflux,                     \
        w1flux, w2flux, rfiberflux, objtype, release,                                 \
        gfluxivar, rfluxivar, zfluxivar,                                              \
        gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,                         \
        gfracmasked, rfracmasked, zfracmasked,                                        \
        gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,                      \
        gsnr, rsnr, zsnr, w1snr, w2snr,                                               \
        dchisq, deltaChi2, brightstarinblob, maskbits =                               \
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
    # ADM and priority shift.
    std_dither, shift_dither = isSTD_dither(
        obs_gflux=obs_gflux, obs_rflux=obs_rflux, obs_zflux=obs_zflux,
        isgood=isgood, primary=primary
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

    # ADM determine if an object is SV0_STD_BRIGHT. Resembles first
    # ADM iteration of SV, but locked in cmx_cuts (and could be altered).
    sv0_std_bright = isSV0_STD_bright(
        gflux=gflux, rflux=rflux, zflux=zflux,
        pmra=pmra, pmdec=pmdec, parallax=parallax,
        gaiagmag=gaiagmag, isgood=isgood, primary=primary
    )

    # ADM determine if an object is SV0_BGS.
    sv0_bgs = isSV0_BGS(
        rflux=rflux, objtype=objtype, primary=primary
    )

    # ADM determine if an object is SV0_MWS.
    sv0_mws = isSV0_MWS(
        rflux=rflux, obs_rflux=obs_rflux, objtype=objtype,
        gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag,
        pmra=pmra, pmdec=pmdec, parallax=parallax,
        parallaxovererror=parallaxovererror,
        photbprpexcessfactor=gaiabprpfactor,
        astrometricsigma5dmax=gaiasigma5dmax,
        gaiaaen=gaiaaen,
        galb=galb, gaia=gaia, primary=primary
    )

    # ADM determine if an object is SV0_LRG.
    sv0_lrg = isSV0_LRG(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
        rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr,
        gflux_ivar=gfluxivar, primary=primary
    )

    # ADM determine if an object is SV0_ELG.
    sv0_elg = isSV0_ELG(
        primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
        gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, maskbits=maskbits
    )

    # ADM determine if an object is SV0_QSO.
    sv0_qso = isSV0_QSO(
        primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
        w1flux=w1flux, w2flux=w2flux, objtype=objtype,
        dchisq=dchisq, brightstarinblob=brightstarinblob
    )

    # ADM run the STD target types for both faint and bright.
    # ADM Make sure to pass all of the needed columns! At one point we stopped
    # ADM passing objtype, which meant no standards were being returned.
    std_classes = []
    for bright in [False, True]:
        std_classes.append(
            isSTD(
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
    std_faint, std_bright = std_classes

    # ADM Construct the target flag bits.
    cmx_target = std_dither * cmx_mask.STD_GAIA
    cmx_target |= std_test * cmx_mask.STD_TEST
    cmx_target |= std_calspec * cmx_mask.STD_CALSPEC
    cmx_target |= sv0_std_bright * cmx_mask.SV0_STD_BRIGHT
    cmx_target |= sv0_bgs * cmx_mask.SV0_BGS
    cmx_target |= sv0_mws * cmx_mask.SV0_MWS
    cmx_target |= sv0_lrg * cmx_mask.SV0_LRG
    cmx_target |= sv0_elg * cmx_mask.SV0_ELG
    cmx_target |= sv0_qso * cmx_mask.SV0_QSO
    cmx_target |= std_faint * cmx_mask.STD_FAINT
    cmx_target |= std_bright * cmx_mask.STD_BRIGHT

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

    return targets
