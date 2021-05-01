"""
desitarget.cuts
===============

An old copy of the Main Survey cuts (../cuts.py) that were used for commissioning (cmx).

.. _`the Gaia data model`: https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
.. _`the Legacy Surveys`: http://www.legacysurvey.org/
.. _`the wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection
.. _`Legacy Surveys mask`: http://www.legacysurvey.org/dr8/bitmasks/
"""
import warnings
from time import time
import os.path

import numbers
import sys

import fitsio
import numpy as np
import healpy as hp
from pkg_resources import resource_filename
import numpy.lib.recfunctions as rfn
from importlib import import_module

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Row

from desitarget import io
from desitarget.internal import sharedmem
from desitarget.gaiamatch import match_gaia_to_primary, find_gaia_files_hp
from desitarget.gaiamatch import pop_gaia_coords, pop_gaia_columns, unextinct_gaia_mags
from desitarget.gaiamatch import gaia_dr_from_ref_cat, is_in_Galaxy, gaia_psflike
from desitarget.targets import finalize, resolve
from desitarget.geomask import bundle_bricks, pixarea2nside, sweep_files_touch_hp
from desitarget.geomask import box_area, hp_in_box, is_in_box, is_in_hp
from desitarget.geomask import cap_area, hp_in_cap, is_in_cap, imaging_mask

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()


def shift_photo_north(gflux=None, rflux=None, zflux=None):
    """Convert fluxes in the northern (BASS/MzLS) to the southern (DECaLS) system.

    Parameters
    ----------
    gflux, rflux, zflux : :class:`array_like` or `float`
        The flux in nano-maggies of g, r, z bands.

    Returns
    -------
    The equivalent fluxes shifted to the southern system.

    Notes
    -----
    - see also https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=3390;filename=Raichoor_DESI_05Dec2017.pdf;version=1
    - Update for DR9 https://desi.lbl.gov/trac/attachment/wiki/TargetSelectionWG/TargetSelection/North_vs_South_dr9.png
    """
    # ADM if floats were sent, treat them like arrays.
    flt = False
    if _is_row(gflux):
        flt = True
        gflux = np.atleast_1d(gflux)
        rflux = np.atleast_1d(rflux)
        zflux = np.atleast_1d(zflux)

    # ADM only use the g-band color shift when r and g are non-zero
    gshift = gflux * 10**(-0.4*0.004)
    w = np.where((gflux != 0) & (rflux != 0))
    gshift[w] = (gflux[w] * 10**(-0.4*0.004) * (gflux[w]/rflux[w])**complex(-0.059)).real

    # ADM only use the r-band color shift when r and z are non-zero
    # ADM and only use the z-band color shift when r and z are non-zero
    w = np.where((rflux != 0) & (zflux != 0))
    rshift = rflux * 10**(0.4*0.003)
    zshift = zflux * 10**(0.4*0.013)

    rshift[w] = (rflux[w] * 10**(0.4*0.003) * (rflux[w]/zflux[w])**complex(-0.024)).real
    zshift[w] = (zflux[w] * 10**(0.4*0.013) * (rflux[w]/zflux[w])**complex(+0.015)).real

    if flt:
        return gshift[0], rshift[0], zshift[0]

    return gshift, rshift, zshift


def isLRG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          zfiberflux=None, rfluxivar=None, zfluxivar=None, w1fluxivar=None,
          gnobs=None, rnobs=None, znobs=None, maskbits=None, primary=None,
          south=True):
    """
    Parameters
    ----------
    south: boolean, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS)
        if ``south=False``, otherwise use cuts appropriate to the
        Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is an LRG target.

    Notes
    -----
    - Current version (12/07/2020) is version 232 on `the wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    # ADM LRG targets.
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    # ADM basic quality cuts.
    lrg &= notinLRG_mask(
        primary=primary, rflux=rflux, zflux=zflux, w1flux=w1flux,
        zfiberflux=zfiberflux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
        rfluxivar=rfluxivar, zfluxivar=zfluxivar, w1fluxivar=w1fluxivar,
        maskbits=maskbits
    )

    # ADM color-based selection of LRGs.
    lrg &= isLRG_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
        zfiberflux=zfiberflux, south=south, primary=primary
    )

    return lrg


def notinLRG_mask(primary=None, rflux=None, zflux=None, w1flux=None,
                  zfiberflux=None, gnobs=None, rnobs=None, znobs=None,
                  rfluxivar=None, zfluxivar=None, w1fluxivar=None,
                  maskbits=None):
    """See :func:`~desitarget.cuts.isLRG` for details.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is NOT masked for poor quality.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    # ADM to maintain backwards-compatibility with mocks.
    if zfiberflux is None:
        log.warning('Setting zfiberflux to zflux!!!')
        zfiberflux = zflux.copy()

    lrg &= (rfluxivar > 0) & (rflux > 0)   # ADM quality in r.
    lrg &= (zfluxivar > 0) & (zflux > 0) & (zfiberflux > 0)   # ADM quality in z.
    lrg &= (w1fluxivar > 0) & (w1flux > 0)  # ADM quality in W1.

    # ADM observed in every band.
    lrg &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    # ADM default mask bits from the Legacy Surveys not set.
    lrg &= imaging_mask(maskbits)

    return lrg


def isLRG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 zfiberflux=None, ggood=None,
                 w2flux=None, primary=None, south=True):
    """(see, e.g., :func:`~desitarget.cuts.isLRG`).

    Notes:
        - the `ggood` and `w2flux` inputs are an attempt to maintain
          backwards-compatibility with the mocks.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    # ADM to maintain backwards-compatibility with mocks.
    if zfiberflux is None:
        log.warning('Setting zfiberflux to zflux!!!')
        zfiberflux = zflux.copy()

    gmag = 22.5 - 2.5 * np.log10(gflux.clip(1e-7))
    # ADM safe as these fluxes are set to > 0 in notinLRG_mask.
    rmag = 22.5 - 2.5 * np.log10(rflux.clip(1e-7))
    zmag = 22.5 - 2.5 * np.log10(zflux.clip(1e-7))
    w1mag = 22.5 - 2.5 * np.log10(w1flux.clip(1e-7))
    zfibermag = 22.5 - 2.5 * np.log10(zfiberflux.clip(1e-7))

    if south:
        lrg &= zmag - w1mag > 0.8 * (rmag-zmag) - 0.6    # non-stellar cut.
        lrg &= (
            ((gmag - w1mag > 2.6) & (gmag - rmag > 1.4))
            | (rmag - w1mag > 1.8)                       # low-z cut.
        )
        lrg &= rmag - zmag > (zmag - 16.83) * 0.45       # double sliding cut 1.
        lrg &= rmag - zmag > (zmag - 13.80) * 0.19       # double sliding cut 2.
    else:
        lrg &= zmag - w1mag > 0.8 * (rmag-zmag) - 0.6   # non-stellar cut.
        lrg &= (
            ((gmag - w1mag > 2.67) & (gmag - rmag > 1.45))
            | (rmag - w1mag > 1.85)                      # low-z cut.
        )
        lrg &= rmag - zmag > (zmag - 16.79) * 0.45       # double sliding cut 1.
        lrg &= rmag - zmag > (zmag - 13.76) * 0.19       # double sliding cut 2.

    lrg &= zfibermag < 21.5    # faint limit.

    return lrg


def isELG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          gsnr=None, rsnr=None, zsnr=None, gnobs=None, rnobs=None, znobs=None,
          maskbits=None, south=True, primary=None):
    """Definition of ELG target classes. Returns a boolean array.
    (see :func:`~desitarget.cuts.set_target_bits` for parameters).

    Notes:
    - Current version (12/09/20) is version 233 on `the wiki`_.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    elg &= notinELG_mask(maskbits=maskbits, gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                         gnobs=gnobs, rnobs=rnobs, znobs=znobs, primary=primary)

    elg &= isELG_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                        w2flux=w2flux, south=south, primary=primary)

    return elg


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

    # ADM default mask bits from the Legacy Surveys not set.
    elg &= imaging_mask(maskbits)

    return elg


def isELG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 w2flux=None, south=True, primary=None):
    """Color cuts for ELG target selection classes
    (see, e.g., :func:`~desitarget.cuts.set_target_bits` for parameters).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    # ADM work in magnitudes instead of fluxes. NOTE THIS IS ONLY OK AS
    # ADM the snr masking in ALL OF g, r AND z ENSURES positive fluxes.
    g = 22.5 - 2.5*np.log10(gflux.clip(1e-16))
    r = 22.5 - 2.5*np.log10(rflux.clip(1e-16))
    z = 22.5 - 2.5*np.log10(zflux.clip(1e-16))

    # ADM cuts shared by the northern and southern selections.
    elg &= g > 20                       # bright cut.
    elg &= r - z > 0.3                  # blue cut.
    elg &= r - z < 1.6                  # red cut.
    elg &= g - r < -1.2*(r - z) + 1.6   # OII flux cut.

    # ADM cuts that are unique to the north or south.
    if south:
        elg &= g < 23.4  # faint cut.
        # ADM south has the FDR cut to remove stars and low-z galaxies.
        elg &= g - r < 1.15*(r - z) - 0.15
    else:
        elg &= g < 23.5  # faint cut.
        elg &= g - r < 1.15*(r - z) - 0.20  # remove stars and low-z galaxies.

    return elg


def _check_BGS_targtype(targtype):
    """Fail if `targtype` is not one of the strings 'bright', 'faint' or 'wise'.
    """
    targposs = ['faint', 'bright', 'wise']

    if targtype not in targposs:
        msg = 'targtype must be one of {} not {}'.format(targposs, targtype)
        log.critical(msg)
        raise ValueError(msg)


def isBGS(rfiberflux=None, gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          gnobs=None, rnobs=None, znobs=None, gfracmasked=None, rfracmasked=None, zfracmasked=None,
          gfracflux=None, rfracflux=None, zfracflux=None, gfracin=None, rfracin=None, zfracin=None,
          gfluxivar=None, rfluxivar=None, zfluxivar=None, maskbits=None, Grr=None, refcat=None,
          w1snr=None, gaiagmag=None, objtype=None, primary=None, south=True, targtype=None):
    """Definition of BGS target classes. Returns a boolean array.

    Args
    ----
    targtype: str, optional, defaults to ``faint``
        Pass ``bright`` to use colors appropriate to the ``BGS_BRIGHT`` selection
        or ``faint`` to use colors appropriate to the ``BGS_FAINT`` selection
        or ``wise`` to use colors appropriate to the ``BGS_WISE`` selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a BGS target of type ``targtype``.

    Notes
    -----
    - Current version (10/24/18) is version 143 on `the wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    _check_BGS_targtype(targtype)

    # ------ Bright Galaxy Survey
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()

    bgs &= notinBGS_mask(gnobs=gnobs, rnobs=rnobs, znobs=znobs, primary=primary,
                         gfracmasked=gfracmasked, rfracmasked=rfracmasked, zfracmasked=zfracmasked,
                         gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                         gfracin=gfracin, rfracin=rfracin, zfracin=zfracin, w1snr=w1snr,
                         gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar, Grr=Grr,
                         gaiagmag=gaiagmag, maskbits=maskbits, targtype=targtype)

    bgs &= isBGS_colors(rfiberflux=rfiberflux, gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                        w2flux=w2flux, south=south, targtype=targtype, primary=primary)

    bgs |= isBGS_lslga(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, refcat=refcat,
                       maskbits=maskbits, south=south, targtype=targtype)

    return bgs


def notinBGS_mask(gnobs=None, rnobs=None, znobs=None, primary=None,
                  gfracmasked=None, rfracmasked=None, zfracmasked=None,
                  gfracflux=None, rfracflux=None, zfracflux=None,
                  gfracin=None, rfracin=None, zfracin=None, w1snr=None,
                  gfluxivar=None, rfluxivar=None, zfluxivar=None, Grr=None,
                  gaiagmag=None, maskbits=None, targtype=None):
    """Standard set of masking cuts used by all BGS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isBGS` for parameters).
    """
    _check_BGS_targtype(targtype)

    if primary is None:
        primary = np.ones_like(gnobs, dtype='?')
    bgs = primary.copy()

    bgs &= (gnobs >= 1) & (rnobs >= 1) & (znobs >= 1)
    bgs &= (gfracmasked < 0.4) & (rfracmasked < 0.4) & (zfracmasked < 0.4)
    bgs &= (gfracflux < 5.0) & (rfracflux < 5.0) & (zfracflux < 5.0)
    bgs &= (gfracin > 0.3) & (rfracin > 0.3) & (zfracin > 0.3)
    bgs &= (gfluxivar > 0) & (rfluxivar > 0) & (zfluxivar > 0)

    # ADM geometric masking cuts from the Legacy Surveys.
    bgs &= imaging_mask(maskbits)

    if targtype == 'bright':
        bgs &= ((Grr > 0.6) | (gaiagmag == 0))
    elif targtype == 'faint':
        bgs &= ((Grr > 0.6) | (gaiagmag == 0))
    elif targtype == 'wise':
        bgs &= Grr < 0.4
        bgs &= Grr > -1
        bgs &= w1snr > 5

    return bgs


def isBGS_colors(rfiberflux=None, gflux=None, rflux=None, zflux=None, w1flux=None,
                 w2flux=None, south=True, targtype=None, primary=None):
    """Standard set of color-based cuts used by all BGS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isBGS` for parameters).
    """
    _check_BGS_targtype(targtype)

    # ADM to maintain backwards-compatibility with mocks.
    if rfiberflux is None:
        log.warning('Setting rfiberflux to rflux!!!')
        rfiberflux = rflux.copy()

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()
    fmc = np.zeros_like(rflux, dtype='?')

    if south:
        bgs &= rflux > gflux * 10**(-1.0/2.5)
        bgs &= rflux < gflux * 10**(4.0/2.5)
        bgs &= zflux > rflux * 10**(-1.0/2.5)
        bgs &= zflux < rflux * 10**(4.0/2.5)
    else:
        bgs &= rflux > gflux * 10**(-1.0/2.5)
        bgs &= rflux < gflux * 10**(4.0/2.5)
        bgs &= zflux > rflux * 10**(-1.0/2.5)
        bgs &= zflux < rflux * 10**(4.0/2.5)

    g = 22.5 - 2.5*np.log10(gflux.clip(1e-16))
    r = 22.5 - 2.5*np.log10(rflux.clip(1e-16))
    z = 22.5 - 2.5*np.log10(zflux.clip(1e-16))
    rfib = 22.5 - 2.5*np.log10(rfiberflux.clip(1e-16))

    # Fibre Magnitude Cut (FMC) -- This is a low surface brightness cut
    # with the aim of increase the redshift success rate.
    fmc |= ((rfib < (2.9 + 1.2 + 1.0) + r) & (r < 17.8))
    fmc |= ((rfib < 22.9) & (r < 20.0) & (r > 17.8))
    fmc |= ((rfib < 2.9 + r) & (r > 20))

    bgs &= fmc

    if targtype == 'bright':
        bgs &= rflux > 10**((22.5-19.5)/2.5)
    elif targtype == 'faint':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        bgs &= rflux <= 10**((22.5-19.5)/2.5)
    elif targtype == 'wise':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        bgs &= w1flux*gflux > (zflux*rflux)*10**(-0.2)

    return bgs


def isBGS_lslga(gflux=None, rflux=None, zflux=None, w1flux=None, refcat=None,
                maskbits=None, south=True, targtype=None):
    """Module to recover the LSLGA objects in all BGS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isBGS` for parameters).
    """
    _check_BGS_targtype(targtype)

    bgs = np.zeros_like(rflux, dtype='?')

    # the LSLGA galaxies.
    LX = bgs.copy()
    # ADM Could check on "L2" for DR8, need to check on "LX" post-DR8.
    if refcat is not None:
        rc1d = np.atleast_1d(refcat)
        if isinstance(rc1d[0], str):
            LX = [(rc[0] == "L") if len(rc) > 0 else False for rc in rc1d]
        else:
            LX = [(rc.decode()[0] == "L") if len(rc) > 0 else False for rc in rc1d]
        if np.ndim(refcat) == 0:
            LX = np.array(LX[0], dtype=bool)
        else:
            LX = np.array(LX, dtype=bool)

    bgs |= LX
    # ADM geometric masking cuts from the Legacy Surveys.
    bgs &= imaging_mask(maskbits, bgsmask=True)

    if targtype == 'bright':
        bgs &= rflux > 10**((22.5-19.5)/2.5)
    elif targtype == 'faint':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        bgs &= rflux <= 10**((22.5-19.5)/2.5)
    elif targtype == 'wise':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        bgs &= w1flux*gflux > (zflux*rflux)*10**(-0.2)

    return bgs


def isQSO_randomforest(gflux=None, rflux=None, zflux=None, maskbits=None,
                       w1flux=None, w2flux=None, objtype=None, release=None,
                       gnobs=None, rnobs=None, znobs=None, deltaChi2=None,
                       primary=None, ra=None, dec=None, south=True, return_probs=False):
    """Define QSO targets from a Random Forest. Returns a boolean array.

    Parameters
    ----------
    south : :class:`boolean`, defaults to ``True``
        If ``False``, shift photometry to the Northern (BASS/MzLS)
        imaging system.
    return_probs : :class:`boolean`, defaults to ``False``
        If ``True``, return the QSO/high-z QSO probabilities in addition
        to the QSO target booleans. Only coded up for DR8 or later of the
        Legacy Surveys. Will return arrays of zeros for earlier DRs.

    Returns
    -------
    :class:`array_like`
        ``True`` for objects that are Random Forest quasar targets.
    :class:`array_like`
        ``True`` for objects that are high-z RF quasar targets.
    :class:`array_like`
        The (float) probability that a target is a quasar. Only returned
        if `return_probs` is ``True``.
    :class:`array_like`
        The (float) probability that a target is a high-z quasar. Only
        returned if `return_probs` is ``True``.

    Notes
    -----
    - Current version (20/11/20) is version 173 on `the wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    # ADM Primary (True for anything to initially consider as a possible target).
    if primary is None:
        primary = np.ones_like(gflux, dtype=bool)

    # RELEASE
    # ADM default to RELEASE of 5000 if nothing is passed.
    if release is None:
        release = np.zeros_like(gflux, dtype='?') + 5000
    release = np.atleast_1d(release)

    # Build variables for random forest
    nFeatures = 11   # Number of attributes describing each object to be classified by the rf
    nbEntries = rflux.size
    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    colors, r, photOK = _getColors(nbEntries, nFeatures, gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d(r)

    # Preselection to speed up the process
    rMax = 22.7   # r < 22.7
    rMin = 17.5   # r > 17.5
    preSelection = (r < rMax) & (r > rMin) & photOK & primary

    # ADM targets have to be observed in every band.
    preSelection &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    if objtype is not None:
        preSelection &= _psflike(objtype)
    if deltaChi2 is not None:
        deltaChi2 = np.atleast_1d(deltaChi2)
        preSelection[release < 5000] &= deltaChi2[release < 5000] > 30.
    # ADM Reject objects in masks.
    # ADM BRIGHT BAILOUT GALAXY CLUSTER (1, 10, 12, 13) bits not set.
    # ALLMASK_G	| ALLMASK_R | ALLMASK_Z (5, 6, 7) bits not set.
    # Now only 1, 12, 13
    if maskbits is not None:
        # ADM default mask bits from the Legacy Surveys not set.
        preSelection &= imaging_mask(maskbits)

    # "qso" mask initialized to "preSelection" mask.
    qso = np.copy(preSelection)
    # ADM to specifically store the selection from the "HighZ" RF.
    qsohiz = np.copy(preSelection)

    # ADM these store the probabilities, should they need returned.
    pqso = np.zeros_like(qso, dtype='>f4')
    pqsohiz = np.zeros_like(qso, dtype='>f4')

    if np.any(preSelection):

        from desitarget.myRF import myRF

        # Data reduction to preselected objects
        colorsReduced = colors[preSelection]
        releaseReduced = release[preSelection]
        r_Reduced = r[preSelection]
        colorsIndex = np.arange(0, nbEntries, dtype=np.int64)
        colorsReducedIndex = colorsIndex[preSelection]

        # Path to random forest files
        pathToRF = resource_filename('desitarget', 'data')
        # rf filenames
        rf_DR3_fileName = pathToRF + '/rf_model_dr3.npz'
        rf_DR7_fileName = pathToRF + '/rf_model_dr7.npz'
        rf_DR7_HighZ_fileName = pathToRF + '/rf_model_dr7_HighZ.npz'
        rf_DR8_fileName = pathToRF + '/rf_model_dr8.npz'
        rf_DR8_HighZ_fileName = pathToRF + '/rf_model_dr8_HighZ.npz'
        rf_DR9_fileName = pathToRF + '/rf_model_dr9.npz'
        rf_DR9_HighZ_fileName = pathToRF + '/rf_model_dr9_HighZ.npz'

        tmpReleaseOK = releaseReduced < 5000
        if np.any(tmpReleaseOK):
            # rf initialization - colors data duplicated within "myRF"
            rf_DR3 = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                          numberOfTrees=200, version=1)
            # rf loading
            rf_DR3.loadForest(rf_DR3_fileName)
            # Compute rf probabilities
            tmp_rf_proba = rf_DR3.predict_proba()
            tmp_r_Reduced = r_Reduced[tmpReleaseOK]
            # Compute optimized proba cut
            pcut = np.where(tmp_r_Reduced > 20.0,
                            0.95 - (tmp_r_Reduced - 20.0) * 0.08, 0.95)
            # Add rf proba test result to "qso" mask
            qso[colorsReducedIndex[tmpReleaseOK]] = tmp_rf_proba >= pcut
            # ADM no high-z selection for DR3.
            qsohiz &= False

        tmpReleaseOK = (releaseReduced >= 5000) & (releaseReduced < 8000)
        if np.any(tmpReleaseOK):
            # rf initialization - colors data duplicated within "myRF"
            rf = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                      numberOfTrees=500, version=2)
            rf_HighZ = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                            numberOfTrees=500, version=2)
            # rf loading
            rf.loadForest(rf_DR7_fileName)
            rf_HighZ.loadForest(rf_DR7_HighZ_fileName)
            # Compute rf probabilities
            tmp_rf_proba = rf.predict_proba()
            tmp_rf_HighZ_proba = rf_HighZ.predict_proba()
            # Compute optimized proba cut
            tmp_r_Reduced = r_Reduced[tmpReleaseOK]
            pcut = np.where(tmp_r_Reduced > 20.8,
                            0.83 - (tmp_r_Reduced - 20.8) * 0.025, 0.83)
            pcut[tmp_r_Reduced > 21.5] = 0.8125 - 0.15 * (tmp_r_Reduced[tmp_r_Reduced > 21.5] - 21.5)
            pcut[tmp_r_Reduced > 22.3] = 0.6925 - 0.70 * (tmp_r_Reduced[tmp_r_Reduced > 22.3] - 22.3)
            pcut_HighZ = np.where(tmp_r_Reduced > 20.5,
                                  0.55 - (tmp_r_Reduced - 20.5) * 0.025, 0.55)

            # Add rf proba test result to "qso" mask
            qso[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_proba >= pcut) | (tmp_rf_HighZ_proba >= pcut_HighZ)
            # ADM populate a mask specific to the "HighZ" selection.
            qsohiz[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_HighZ_proba >= pcut_HighZ)

        tmpReleaseOK = (releaseReduced >= 8000) & (releaseReduced < 9000)
        if np.any(tmpReleaseOK):
            # rf initialization - colors data duplicated within "myRF"
            rf = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                      numberOfTrees=500, version=2)
            rf_HighZ = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                            numberOfTrees=500, version=2)
            # rf loading
            rf.loadForest(rf_DR8_fileName)
            rf_HighZ.loadForest(rf_DR8_HighZ_fileName)
            # Compute rf probabilities
            tmp_rf_proba = rf.predict_proba()
            tmp_rf_HighZ_proba = rf_HighZ.predict_proba()
            # Compute optimized proba cut
            tmp_r_Reduced = r_Reduced[tmpReleaseOK]
            pcut = 0.88 - 0.03*np.tanh(tmp_r_Reduced - 20.5)
            pcut_HighZ = 0.55

            # Add rf proba test result to "qso" mask
            qso[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_proba >= pcut) | (tmp_rf_HighZ_proba >= pcut_HighZ)
            # ADM populate a mask specific to the "HighZ" selection.
            qsohiz[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_HighZ_proba >= pcut_HighZ)
            # ADM store the probabilities in case they need returned.
            pqso[colorsReducedIndex[tmpReleaseOK]] = tmp_rf_proba
            # ADM populate a mask specific to the "HighZ" selection.
            pqsohiz[colorsReducedIndex[tmpReleaseOK]] = tmp_rf_HighZ_proba

        tmpReleaseOK = releaseReduced >= 9000
        if np.any(tmpReleaseOK):
            # rf initialization - colors data duplicated within "myRF"
            rf = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                      numberOfTrees=500, version=2)
            rf_HighZ = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                            numberOfTrees=500, version=2)
            # rf loading
            rf.loadForest(rf_DR9_fileName)
            rf_HighZ.loadForest(rf_DR9_HighZ_fileName)
            # Compute rf probabilities
            tmp_rf_proba = rf.predict_proba()
            tmp_rf_HighZ_proba = rf_HighZ.predict_proba()
            # Compute optimized proba cut
            tmp_r_Reduced = r_Reduced[tmpReleaseOK]
            if not south:
                # threshold selection for North footprint.
                pcut = 0.857 - 0.03*np.tanh(tmp_r_Reduced - 20.5)
                pcut_HighZ = 0.7
            else:
                pcut = np.ones(tmp_rf_proba.size)
                pcut_HighZ = np.ones(tmp_rf_HighZ_proba.size)
                is_des = (gnobs[preSelection][tmpReleaseOK] > 4) &\
                         (rnobs[preSelection][tmpReleaseOK] > 4) &\
                         (znobs[preSelection][tmpReleaseOK] > 4) &\
                         ((ra[preSelection][tmpReleaseOK] >= 320) | (ra[preSelection][tmpReleaseOK] <= 100)) &\
                         (dec[preSelection][tmpReleaseOK] <= 10)
                # threshold selection for DES footprint.
                pcut[is_des] = 0.75 - 0.05*np.tanh(tmp_r_Reduced[is_des] - 20.5)
                pcut_HighZ[is_des] = 0.50
                # threshold selection for South footprint.
                pcut[~is_des] = 0.85 - 0.04*np.tanh(tmp_r_Reduced[~is_des] - 20.5)
                pcut_HighZ[~is_des] = 0.65

            # Add rf proba test result to "qso" mask
            qso[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_proba >= pcut) | (tmp_rf_HighZ_proba >= pcut_HighZ)
            # ADM populate a mask specific to the "HighZ" selection.
            qsohiz[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_HighZ_proba >= pcut_HighZ)
            # ADM store the probabilities in case they need returned.
            pqso[colorsReducedIndex[tmpReleaseOK]] = tmp_rf_proba
            # ADM populate a mask specific to the "HighZ" selection.
            pqsohiz[colorsReducedIndex[tmpReleaseOK]] = tmp_rf_HighZ_proba

    # In case of call for a single object passed to the function with
    # scalar arguments. Return "numpy.bool_" instead of "~numpy.ndarray".
    if nbEntries == 1:
        qso = qso[0]
        qsohiz = qsohiz[0]
        pqso = pqso[0]
        pqsohiz = pqsohiz[0]

    # ADM if requested, return the probabilities as well.
    if return_probs:
        return qso, qsohiz, pqso, pqsohiz
    return qso, qsohiz


def _psflike(psftype):
    """ If the object is PSF """
    # ADM explicitly checking for NoneType. I can't see why we'd ever want to
    # ADM run this test on empty information. In the past we have had bugs where
    # ADM we forgot to pass objtype=objtype in, e.g., isSTD
    if psftype is None:
        raise ValueError("NoneType submitted to _psfflike function")

    psftype = np.asarray(psftype)
    # ADM in Python3 these string literals become byte-like
    # ADM so to retain Python2 compatibility we need to check
    # ADM against both bytes and unicode
    # ADM, also 'PSF' for astropy.io.fits; 'PSF ' for fitsio (sigh)
    psflike = ((psftype == 'PSF') | (psftype == b'PSF') |
               (psftype == 'PSF ') | (psftype == b'PSF '))
    return psflike


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


def _is_row(table):
    """Return True/False if this is a row of a table instead of a full table.

    supports numpy.ndarray, astropy.io.fits.FITS_rec, and astropy.table.Table
    """
    import astropy.io.fits.fitsrec
    import astropy.table.row
    if isinstance(table, (astropy.io.fits.fitsrec.FITS_record, astropy.table.row.Row)) or \
       np.isscalar(table):
        return True
    else:
        return False
