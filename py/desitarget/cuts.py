"""
desitarget.cuts
===============

Target Selection for DECALS catalogue data derived from `the wiki`_.

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (*e.g.* LRG, ELG or QSO).

.. _`the Gaia data model`: https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
.. _`the Legacy Surveys`: http://www.legacysurvey.org/
.. _`the wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection
"""
import warnings
from time import time
import os.path

import numbers
import sys

import numpy as np
import healpy as hp
from pkg_resources import resource_filename

from astropy.table import Table, Row

from desitarget import io
from desitarget.internal import sharedmem
from desitarget.gaiamatch import match_gaia_to_primary
from desitarget.gaiamatch import pop_gaia_coords, pop_gaia_columns
from desitarget.targets import finalize, resolve
from desitarget.geomask import bundle_bricks, pixarea2nside, check_nside
from desitarget.geomask import box_area, hp_in_box, is_in_box, is_in_hp
from desitarget.geomask import cap_area, hp_in_cap, is_in_cap

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()


def _gal_coords(ra, dec):
    """Shift RA, Dec to Galactic coordinates.

    Parameters
    ----------
    ra, dec : :class:`array_like` or `float`
        RA, Dec coordinates (degrees)

    Returns
    -------
    The Galactic longitude and latitude (l, b)
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    if hasattr(ra, 'unit') and hasattr(dec, 'unit') and ra.unit is not None and dec.unit is not None:
        c = SkyCoord(ra.to(u.deg), dec.to(u.deg))
    else:
        c = SkyCoord(ra*u.deg, dec*u.deg)
    gc = c.transform_to('galactic')

    return gc.l.value, gc.b.value


def shift_photo_north_pure(gflux=None, rflux=None, zflux=None):
    """Same as :func:`~desitarget.cuts.shift_photo_north_pure` accounting for zero fluxes.

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
    """

    gshift = gflux * 10**(-0.4*0.029) * (gflux/rflux)**(-0.068)
    rshift = rflux * 10**(+0.4*0.012) * (rflux/zflux)**(-0.029)
    zshift = zflux * 10**(-0.4*0.000) * (rflux/zflux)**(+0.009)

    return gshift, rshift, zshift


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
    """

    # ADM only use the g-band color shift when r and g are non-zero
    gshift = gflux * 10**(-0.4*0.029)
    w = np.where((gflux != 0) & (rflux != 0))
    gshift[w] = (gflux[w] * 10**(-0.4*0.029) * (gflux[w]/rflux[w])**complex(-0.068)).real

    # ADM only use the r-band color shift when r and z are non-zero
    # ADM and only use the z-band color shift when r and z are non-zero
    w = np.where((rflux != 0) & (zflux != 0))
    rshift = rflux * 10**(+0.4*0.012)
    zshift = zflux * 10**(-0.4*0.000)

    rshift[w] = (rflux[w] * 10**(+0.4*0.012) * (rflux[w]/zflux[w])**complex(-0.029)).real
    zshift[w] = (zflux[w] * 10**(-0.4*0.000) * (rflux[w]/zflux[w])**complex(+0.009)).real

    return gshift, rshift, zshift


def isLRG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 w2flux=None, ggood=None, primary=None, south=True):
    """(see, e.g., :func:`~desitarget.cuts.isLRGpass`).
    """

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    if ggood is None:
        ggood = np.ones_like(gflux, dtype='?')

    lrg = primary.copy()

    if south:
        # ADM intercept -ve, e.g. -0.6 on the wiki for
        # (z-W1) > 0.8*(r-z) - 0.6
        nsc_rzmult, nsc_inter = 0.8, 0.6  # non-stellar cut
        b_lim, f_lim = 18.01, 20.41       # bright/faint limits
        cbox_lo, cbox_hi = 0.75, 2.45     # broad color box
        # ADM cut limits are -ve, e.g. -17.18, -15.11 on the wiki for
        # (z-17.18)/2 < r-z < (z-15.11)/2
        osc_lo, osc_hi = 17.18, 15.11     # optical sliding cut
        osc_div = 2.                      # denominator in optical sliding cut
        elbow_rz, elbow_gr = 1.15, 1.65   # cut redshifts < 0.4, keep elbow at 0.4-0.5
    else:
        nsc_rzmult, nsc_inter = 0.8, 0.735
        b_lim, f_lim = 17.965, 20.365
        cbox_lo, cbox_hi = 0.85, 2.55
        osc_lo, osc_hi = 17.105, 14.885
        osc_div = 1.8
        elbow_rz, elbow_gr = 1.25, 1.655

    # ADM Basic flux and color box cuts.
    lrg &= (zflux > 10**(0.4*(22.5-f_lim)))   # z < 20.41  (south)
    lrg &= (zflux < 10**(0.4*(22.5-b_lim)))   # z > 18.01  (south)
    lrg &= (zflux < 10**(0.4*cbox_hi)*rflux)  # r-z < 2.45 (south)
    lrg &= (zflux > 10**(0.4*cbox_lo)*rflux)  # r-z > 0.75 (south)

    # ADM code can overflow, since float32 arrays have a max of 3e38.
    with np.errstate(over='ignore'):
        # ADM non-stellar cut. e.g., in the south:
        # (z-W1) > 0.8*(r-z) - 0.6  ->  0.8r + W1 < 1.8z + 0.6
        lrg &= ((w1flux*rflux**complex(nsc_rzmult)).real >
                ((zflux**complex(1+nsc_rzmult))*10**(-0.4*nsc_inter)).real)
        # ADM complex/real allows -ve fluxes to be raised to a fractional power.

        # ADM optical sliding cut, e.g. in the south:
        # (z-17.18)/2 < r-z  ->  3z < 17.18 + 2r
        # (z-15.11)/2 > r-z  ->  3z > 15.11 + 2r
        lrg &= (zflux**(1.+osc_div) > 10**(0.4*(22.5-osc_lo))*rflux**osc_div)
        lrg &= (zflux**(1.+osc_div) < 10**(0.4*(22.5-osc_hi))*rflux**osc_div)

        # ADM redshift cut with elbow, e.g. in the south:
        # (r-z > 1.15) OR (g-r > 1.65 and FLUX_IVAR_G > 0)
        lrg &= np.logical_or((zflux > 10**(0.4*elbow_rz)*rflux),
                             (ggood & (rflux > 10**(0.4*elbow_gr)*gflux)))

    return lrg


def isLRG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          rflux_snr=None, zflux_snr=None, w1flux_snr=None,
          gflux_ivar=None, primary=None, south=True):
    """(see, e.g., :func:`~desitarget.cuts.isLRGpass`).
    """
    # ----- Luminous Red Galaxies
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    # Some basic quality in r, z, and W1.  Note by @moustakas: no allmask cuts
    # used!).  Also note: We do not require gflux>0!  Objects can be very red.
    lrg = primary.copy()
    lrg &= (rflux_snr > 0) & (rflux > 0)    # and rallmask == 0
    lrg &= (zflux_snr > 0) & (zflux > 0)    # and zallmask == 0
    lrg &= (w1flux_snr > 4) & (w1flux > 0)

    ggood = (gflux_ivar > 0)  # and gallmask == 0

    # Apply color, flux, and star-galaxy separation cuts
    lrg &= isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                        w2flux=w2flux, ggood=ggood, primary=primary, south=south)

    return lrg


def isLRGpass(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
              rflux_snr=None, zflux_snr=None, w1flux_snr=None,
              gflux_ivar=None, primary=None, south=True):
    """LRGs in different passes (one pass, two pass etc.).

    Args:
        south: boolean, defaults to ``True``
            Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
            otherwise use cuts appropriate to the Southern imaging survey (DECaLS).

    Returns:
        mask : array_like. True if and only if the object is an LRG
            target.

    Notes:
    - As of 11/2/18, based on version 158 on `the wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    # ----- Luminous Red Galaxies
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    lrg = primary.copy()

    # ADM apply the color and flag selection for all LRGs
    lrg &= isLRG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                 rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr,
                 gflux_ivar=gflux_ivar, primary=primary, south=south)

    lrg1pass = lrg.copy()
    lrg2pass = lrg.copy()

    # ADM CRITICALLY, the bright and faint limits are set in isLRG_colors()
    # ADM so we only need to impose a central cut at >/< 20 mags
    if south:
        midbreak = 20.
    else:
        midbreak = 20.   # ADM placeholders for different future 1/2 pass splits.

    lrg1pass &= zflux > 10**((22.5-midbreak)/2.5)
    lrg2pass &= zflux <= 10**((22.5-midbreak)/2.5)

    return lrg, lrg1pass, lrg2pass


def isELG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          gallmask=None, rallmask=None, zallmask=None, brightstarinblob=None,
          south=True, primary=None):
    """Definition of ELG target classes. Returns a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        gallmask, rallmask, zallmask: array_like
            Bitwise mask set if the central pixel from all images
            satisfy each condition in g, r, z.
        brightstarinblob: boolean array_like or None
            ``True`` if the object shares a blob with a "bright" (Tycho-2) star.
        south: boolean, defaults to ``True``
            Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
            otherwise use cuts appropriate to the Southern imaging survey (DECaLS).
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only if the object is an ELG
            target.

    Notes:
    - Current version (08/01/18) is version 144 on `the wiki`_.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    elg &= notinELG_mask(gallmask=gallmask, rallmask=rallmask, zallmask=zallmask,
                         brightstarinblob=brightstarinblob, primary=primary)

    elg &= isELG_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                        south=south, primary=primary)

    return elg


def notinELG_mask(gallmask=None, rallmask=None, zallmask=None,
                  brightstarinblob=None, primary=None):
    """Standard set of masking cuts used by all ELG target selection classes
    (see, e.g., :func:`~desitarget.cuts.isELG` for parameters).
    """
    if primary is None:
        primary = np.ones_like(gallmask, dtype='?')
    elg = primary.copy()

    elg &= (gallmask == 0) & (rallmask == 0) & (zallmask == 0)
    elg &= ~brightstarinblob

    return elg


def isELG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 w2flux=None, south=True, primary=None):
    """Color cuts for ELG target selection classes
    (see, e.g., :func:`~desitarget.cuts.isELG` for parameters).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    # ADM cuts shared by the northern and southern selections.
    elg &= gflux < 10**((22.5-21.0)/2.5)          # g>21
    elg &= zflux > rflux * 10**(0.3/2.5)          # (r-z)>0.3
    elg &= zflux < rflux * 10**(1.6/2.5)          # (r-z)<1.6

    # ADM clip to avoid warnings from negative numbers raised to fractional powers.
    # ADM make sure to do this after the (r-z) cuts to prevent the recovery of
    # ADM very bright objects with strange colors.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    elg &= zflux**1.2 < gflux * rflux**0.2 * 10**(1.6/2.5)           # (g-r)<1.6-1.2(r-z)

    # ADM cuts that are unique to the north or south.
    if south:
        elg &= gflux > 10**((22.5-23.45)/2.5)                        # g<23.45
        # ADM the south has the original FDR cut to remove stars and low-z galaxies.
        elg &= rflux**2.15 < gflux * zflux**1.15 * 10**(-0.15/2.5)   # (g-r)<1.15(r-z)-0.15
    else:
        elg &= gflux > 10**((22.5-23.7)/2.5)      # g<23.7
        elg &= rflux > 10**((22.5-23.3)/2.5)      # r<23.3
        # ADM the north has a modified FDR cut to remove stars and low-z galaxies.
        elg &= rflux**2.40 < gflux * zflux**1.40 * 10**(-0.35/2.5)   # (g-r)<1.40(r-z)-0.35

    return elg


def isSTD_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 primary=None, south=True):
    """Select STD stars based on Legacy Surveys color cuts. Returns a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to ``True``
            Use color-cuts based on photometry from the "south" (DECaLS) as
            opposed to the "north" (MzLS+BASS).

    Returns:
        mask : boolean array, True if the object has colors like a STD star target

    Notes:
        - Current version (08/01/18) is version 121 on `the wiki`_.
    """

    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    std = primary.copy()

    # Clip to avoid warnings from negative numbers.
    # ADM we're pretty bright for the STDs, so this should be safe
    gflux = gflux.clip(1e-16)
    rflux = rflux.clip(1e-16)
    zflux = zflux.clip(1e-16)

    # ADM optical colors for halo TO or bluer
    grcolor = 2.5 * np.log10(rflux / gflux)
    rzcolor = 2.5 * np.log10(zflux / rflux)
    # Currently no difference in north vs south color-cuts.
    if south:
        std &= rzcolor < 0.2
        std &= grcolor > 0.
        std &= grcolor < 0.35
    else:
        std &= rzcolor < 0.2
        std &= grcolor > 0.
        std &= grcolor < 0.35

    return std


def isSTD_gaia(primary=None, gaia=None, astrometricexcessnoise=None,
               pmra=None, pmdec=None, parallax=None,
               dupsource=None, paramssolved=None,
               gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Gaia quality cuts used to define STD star targets.

    Args:
        primary: array_like or None
          If given, the BRICK_PRIMARY column of the catalogue.
        gaia: boolean array_like or None
            True if there is a match between this object in
            `the Legacy Surveys`_ and in Gaia.
        astrometricexcessnoise: array_like or None
            Excess noise of the source in Gaia (as in `the Gaia data model`_).
        pmra, pmdec, parallax: array_like or None
            Gaia-based proper motion in RA and Dec and parallax
            (same units as the Gaia data model).
        dupsource: array_like or None
            Whether the source is a duplicate in Gaia (as in `the Gaia data model`_).
        paramssolved: array_like or None
            How many parameters were solved for in Gaia (as in `the Gaia data model`_).
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            Gaia-based g, b and  r MAGNITUDES (not Galactic-extinction-corrected).
            (same units as `the Gaia data model`_).

    Returns:
        mask : boolean array, True if the object passes Gaia quality cuts.

    Notes:
    - Current version (08/01/18) is version 121 on `the wiki`_.
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

    # ADM fail if dupsource is not Boolean, as was the case for the 7.0 sweeps
    # ADM otherwise logic checks on dupsource will be misleading.
    if not (dupsource.dtype.type == np.bool_):
        log.error('GAIA_DUPLICATED_SOURCE (dupsource) should be boolean!')
        raise IOError

    # ADM a unique Gaia source.
    std &= ~dupsource

    return std


def isSTD(gflux=None, rflux=None, zflux=None, primary=None,
          gfracflux=None, rfracflux=None, zfracflux=None,
          gfracmasked=None, rfracmasked=None, zfracmasked=None,
          gnobs=None, rnobs=None, znobs=None,
          gfluxivar=None, rfluxivar=None, zfluxivar=None, objtype=None,
          gaia=None, astrometricexcessnoise=None, paramssolved=None,
          pmra=None, pmdec=None, parallax=None, dupsource=None,
          gaiagmag=None, gaiabmag=None, gaiarmag=None, bright=False,
          usegaia=True, south=True):
    """Select STD targets using color cuts and photometric quality cuts (PSF-like
    and fracflux).  See isSTD_colors() for additional info.

    Args:
        gflux, rflux, zflux: array_like
            The flux in nano-maggies of g, r, z bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        gfracflux, rfracflux, zfracflux: array_like
            Profile-weighted fraction of the flux from other sources divided
            by the total flux in g, r and z bands.
        gfracmasked, rfracmasked, zfracmasked: array_like
            Fraction of masked pixels in the g, r and z bands.
        gnobs, rnobs, znobs: array_like
            The number of observations (in the central pixel) in g, r and z.
        gfluxivar, rfluxivar, zfluxivar: array_like
            The flux inverse variances in g, r, and z bands.
        objtype: array_like or None
            The TYPE column of the catalogue to restrict to point sources.
        gaia: boolean array_like or None
            True if there is a match between this object in
            `the Legacy Surveys`_ and in Gaia.
        astrometricexcessnoise: array_like or None
            Excess noise of the source in Gaia.
        paramssolved: array_like or None
            How many parameters were solved for in Gaia.
        pmra, pmdec, parallax: array_like or None
            Gaia-based proper motion in RA and Dec and parallax
        dupsource: array_like or None
            Whether the source is a duplicate in Gaia.
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            Gaia-based g-, b- and r-band MAGNITUDES.
        bright: boolean, defaults to ``False``
           if ``True`` apply magnitude cuts for "bright" conditions; otherwise,
           choose "normal" brightness standards. Cut is performed on `gaiagmag`.
        usegaia: boolean, defaults to ``True``
           if ``True`` then call :func:`~desitarget.cuts.isSTD_gaia` to set the
           logic cuts. If Gaia is not available (perhaps if you're using mocks)
           then send ``False``, in which case we use the LS r-band magnitude as
           a proxy for the Gaia G-band magnitude (ignoring---incorrectly---that
           we have already corrected for Galactic extinction.)
        south: boolean, defaults to ``True``
            Use color-cuts based on photometry from the "south" (DECaLS) as
            opposed to the "north" (MzLS+BASS).

    Returns:
        mask : boolean array, True if the object has colors like a STD star.

    Notes:
        - Gaia-based quantities are as in `the Gaia data model`_.
        - Current version (08/01/18) is version 127 on `the wiki`_.

    """
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    std = primary.copy()

    # ADM apply the Legacy Surveys (optical) magnitude and color cuts.
    std &= isSTD_colors(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux, south=south)

    # ADM apply the Gaia quality cuts.
    if usegaia:
        std &= isSTD_gaia(primary=primary, gaia=gaia, astrometricexcessnoise=astrometricexcessnoise,
                          pmra=pmra, pmdec=pmdec, parallax=parallax,
                          dupsource=dupsource, paramssolved=paramssolved,
                          gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag)

    # ADM apply type=PSF cut
    std &= _psflike(objtype)

    # ADM apply fracflux, S/N cuts and number of observations cuts.
    fracflux = [gfracflux, rfracflux, zfracflux]
    fluxivar = [gfluxivar, rfluxivar, zfluxivar]
    nobs = [gnobs, rnobs, znobs]
    fracmasked = [gfracmasked, rfracmasked, zfracmasked]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # fracflux can be Inf/NaN
        for bandint in (0, 1, 2):         # g, r, z
            std &= fracflux[bandint] < 0.01
            std &= fluxivar[bandint] > 0
            std &= nobs[bandint] > 0
            std &= fracmasked[bandint] < 0.6

    # ADM brightness cuts in Gaia G-band
    if bright:
        gbright = 15.
        gfaint = 18.
    else:
        gbright = 16.
        gfaint = 19.

    if usegaia:
        std &= gaiagmag >= gbright
        std &= gaiagmag < gfaint
    else:
        # Use LS r-band as a Gaia G-band proxy.
        gaiamag_proxy = 22.5 - 2.5 * np.log10(rflux.clip(1e-16))
        std &= gaiamag_ >= gbright
        std &= gaiamag_ < gfaint

    return std


def isMWS_main(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
               gnobs=None, rnobs=None, gfracmasked=None, rfracmasked=None,
               pmra=None, pmdec=None, parallax=None, obs_rflux=None, objtype=None,
               gaia=None, gaiagmag=None, gaiabmag=None, gaiarmag=None,
               gaiaaen=None, gaiadupsource=None, primary=None, south=True):
    """Set bits for main ``MWS`` targets.

    Args:
        see :func:`~desitarget.cuts.set_target_bits` for parameters.

    Returns:
        mask1 : array_like.
            ``True`` if and only if the object is a ``MWS_BROAD`` target.
        mask2 : array_like.
            ``True`` if and only if the object is a ``MWS_MAIN_RED`` target.
        mask3 : array_like.
            ``True`` if and only if the object is a ``MWS_MAIN_BLUE`` target.

    Notes:
        - as of 11/2/18, based on version 158 on `the wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM currently no difference between N/S for MWS, so easiest
    # ADM just to use one selection
    # if south:

    # ADM do not target any objects for which entries are NaN
    # ADM and turn off the NaNs for those entries
    nans = (np.isnan(rflux) | np.isnan(gflux) |
            np.isnan(parallax) | np.isnan(pmra) | np.isnan(pmdec))
    w = np.where(nans)[0]
    if len(w) > 0:
        # ADM make copies as we are reassigning values
        rflux, gflux, obs_rflux = rflux.copy(), gflux.copy(), obs_rflux.copy()
        parallax, pmra, pmdec = parallax.copy(), pmra.copy(), pmdec.copy()
        rflux[w], gflux[w], obs_rflux[w] = 0., 0., 0.
        parallax[w], pmra[w], pmdec[w] = 0., 0., 0.
        mws &= ~nans
        log.info('{}/{} NaNs in file...t = {:.1f}s'
                 .format(len(w), len(mws), time()-start))

    mws &= notinMWS_main_mask(gaia=gaia, gfracmasked=gfracmasked, gnobs=gnobs,
                              gflux=gflux, rfracmasked=rfracmasked, rnobs=rnobs,
                              rflux=rflux, gaiadupsource=gaiadupsource, primary=primary)

    # ADM pass the mws that pass cuts as primary, to restrict to the
    # ADM sources that weren't in a mask/logic cut.
    mws, red, blue = isMWS_main_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
        pmra=pmra, pmdec=pmdec, parallax=parallax, obs_rflux=obs_rflux, objtype=objtype,
        gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag, gaiaaen=gaiaaen,
        primary=mws, south=south
    )

    return mws, red, blue


def notinMWS_main_mask(gaia=None, gfracmasked=None, gnobs=None, gflux=None,
                       rfracmasked=None, rnobs=None, rflux=None,
                       gaiadupsource=None, primary=None):
    """Standard set of masking-based cuts used by MWS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isMWS_main` for parameters).
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM apply the mask/logic selection for all MWS-MAIN targets
    # ADM main targets match to a Gaia source
    mws &= gaia
    mws &= (gfracmasked < 0.5) & (gflux > 0) & (gnobs > 0)
    mws &= (rfracmasked < 0.5) & (rflux > 0) & (rnobs > 0)

    mws &= ~gaiadupsource

    return mws


def isMWS_main_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                      pmra=None, pmdec=None, parallax=None, obs_rflux=None, objtype=None,
                      gaiagmag=None, gaiabmag=None, gaiarmag=None, gaiaaen=None,
                      primary=None, south=True):
    """Set of color-based cuts used by MWS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isMWS_main` for parameters).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    mws = primary.copy()

    # ADM main targets are point-like based on DECaLS morphology
    # ADM and GAIA_ASTROMETRIC_NOISE.
    mws &= _psflike(objtype)
    mws &= gaiaaen < 3.0

    # ADM main targets are 16 <= r < 19
    mws &= rflux > 10**((22.5-19.0)/2.5)
    mws &= rflux <= 10**((22.5-16.0)/2.5)

    # ADM main targets are robs < 20
    mws &= obs_rflux > 10**((22.5-20.0)/2.5)

    # ADM calculate the overall proper motion magnitude
    # ADM inexplicably I'm getting a Runtimewarning here for
    # ADM a few values in the sqrt, so I'm catching it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = np.sqrt(pmra**2. + pmdec**2.)

    # ADM make a copy of the main bits for a red/blue split
    red = mws.copy()
    blue = mws.copy()

    # ADM MWS-BLUE is g-r < 0.7
    blue &= rflux < gflux * 10**(0.7/2.5)                      # (g-r)<0.7

    # ADM MWS-RED and MWS-BROAD have g-r >= 0.7
    red &= rflux >= gflux * 10**(0.7/2.5)                      # (g-r)>=0.7
    broad = red.copy()

    # ADM MWS-RED also has parallax < 1mas and proper motion < 7.
    red &= pm < 7.
    red &= parallax < 1.

    # ADM MWS-BROAD has parallax > 1mas OR proper motion > 7.
    broad &= (parallax >= 1.) | (pm >= 7.)

    return broad, red, blue


def isMWS_nearby(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 objtype=None, gaia=None, primary=None,
                 pmra=None, pmdec=None, parallax=None, parallaxerr=None,
                 obs_rflux=None, gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for NEARBY Milky Way Survey targets.

    Args:
        see :func:`~desitarget.cuts.set_target_bits` for parameters.

    Returns:
        mask : array_like.
            True if and only if the object is a MWS-NEARBY target.

    Notes:
    - Current version (09/20/18) is version 129 on `the wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM do not target any objects for which entries are NaN
    # ADM and turn off the NaNs for those entries
    nans = np.isnan(gaiagmag) | np.isnan(parallax)
    w = np.where(nans)[0]
    if len(w) > 0:
        # ADM make copies as we are reassigning values
        parallax, gaiagmag = parallax.copy(), gaiagmag.copy()
        parallax[w], gaiagmag[w] = 0., 0.
        mws &= ~nans
        log.info('{}/{} NaNs in file...t = {:.1f}s'
                 .format(len(w), len(mws), time()-start))

    # ADM apply the selection for all MWS-NEARBY targets
    # ADM must be a Legacy Surveys object that matches a Gaia source
    mws &= gaia
    # ADM Gaia G mag of less than 20
    mws &= gaiagmag < 20.
    # ADM parallax cut corresponding to 100pc
    mws &= (parallax + parallaxerr) > 10.   # NB: "+" is correct
    # ADM NOTE TO THE MWS GROUP: There is no bright cut on G. IS THAT THE REQUIRED BEHAVIOR?

    return mws


def isMWS_WD(primary=None, gaia=None, galb=None, astrometricexcessnoise=None,
             pmra=None, pmdec=None, parallax=None, parallaxovererror=None,
             photbprpexcessfactor=None, astrometricsigma5dmax=None,
             gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for WHITE DWARF Milky Way Survey targets.

    Args:
        see :func:`~desitarget.cuts.set_target_bits` for parameters.

    Returns:
        mask : array_like.
            True if and only if the object is a MWS-WD target.

    Notes:
    - Current version (08/01/18) is version 121 on `the wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM do not target any objects for which entries are NaN
    # ADM and turn off the NaNs for those entries
    nans = (np.isnan(gaiagmag) | np.isnan(gaiabmag) | np.isnan(gaiarmag) |
            np.isnan(parallax))
    w = np.where(nans)[0]
    if len(w) > 0:
        parallax, gaiagmag = parallax.copy(), gaiagmag.copy()
        gaiabmag, gaiarmag = gaiabmag.copy(), gaiarmag.copy()
        parallax[w] = 0.
        gaiagmag[w], gaiabmag[w], gaiarmag[w] = 0., 0., 0.
        mws &= ~nans
        log.info('{}/{} NaNs in file...t = {:.1f}s'
                 .format(len(w), len(mws), time()-start))

    # ADM apply the selection for all MWS-WD targets
    # ADM must be a Legacy Surveys object that matches a Gaia source
    mws &= gaia
    # ADM Gaia G mag of less than 20
    mws &= gaiagmag < 20.

    # ADM Galactic b at least 20o from the plane
    mws &= np.abs(galb) > 20.

    # ADM gentle cut on parallax significance
    mws &= parallaxovererror > 1.

    # ADM Color/absolute magnitude cuts of (defining the WD cooling sequence):
    # ADM Gabs > 5
    # ADM Gabs > 5.93 + 5.047(Bp-Rp)
    # ADM Gabs > 6(Bp-Rp)3 - 21.77(Bp-Rp)2 + 27.91(Bp-Rp) + 0.897
    # ADM Bp-Rp < 1.7
    Gabs = gaiagmag+5.*np.log10(parallax.clip(1e-16))-10.
    br = gaiabmag - gaiarmag
    mws &= Gabs > 5.
    mws &= Gabs > 5.93 + 5.047*br
    mws &= Gabs > 6*br*br*br - 21.77*br*br + 27.91*br + 0.897
    mws &= br < 1.7

    # ADM Finite proper motion to reject quasars
    # ADM Inexplicably I'm getting a Runtimewarning here for
    # ADM a few values in the sqrt, so I'm catching it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = np.sqrt(pmra**2. + pmdec**2.)
    mws &= pm > 2.

    # ADM As of DR7, photbprpexcessfactor and astrometricsigma5dmax are not in the
    # ADM imaging catalogs. Until they are, ignore these cuts
    if photbprpexcessfactor is not None:
        # ADM remove problem objects, which often have bad astrometry
        mws &= photbprpexcessfactor < 1.7 + 0.06*br*br

    if astrometricsigma5dmax is not None:
        # ADM Reject white dwarfs that have really poor astrometry while
        # ADM retaining white dwarfs that only have relatively poor astrometry
        mws &= ((astrometricsigma5dmax < 1.5) |
                ((astrometricexcessnoise < 1.) & (parallaxovererror > 4.) & (pm > 10.)))

    return mws


def isMWSSTAR_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None, south=True):
    """Select a reasonable range of g-r colors for MWS targets. Returns a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to ``True``
            Use color-cuts based on photometry from the "south" (DECaLS) as
            opposed to the "north" (MzLS+BASS).

    Returns:
        mask : boolean array, True if the object has colors like an old stellar population,
        which is what we expect for the main MWS sample

    Notes:
        The full MWS target selection also includes PSF-like and fracflux
        cuts and will include Gaia information; this function is only to enforce
        a reasonable range of color/TEFF when simulating data.

    """
    # ----- Old stars, g-r > 0
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    mwsstar = primary.copy()

    # - colors g-r > 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grcolor = 2.5 * np.log10(rflux / gflux)
        # Assume no difference in north vs south color-cuts.
        if south:
            mwsstar &= (grcolor > 0.0)
        else:
            mwsstar &= (grcolor > 0.0)

    return mwsstar


def _check_BGS_targtype(targtype):
    """Fail if `targtype` is not one of the strings 'bright', 'faint' or 'wise'.
    """
    targposs = ['faint', 'bright', 'wise']

    if targtype not in targposs:
        msg = 'targtype must be one of {} not {}'.format(targposs, targtype)
        log.critical(msg)
        raise ValueError(msg)


def _check_BGS_targtype_sv(targtype):
    """Fail if `targtype` is not one of the strings 'bright', 'faint', 'faint_ext', 'lowq' or 'fibmag'.
    """
    targposs = ['faint', 'bright', 'faint_ext', 'lowq', 'fibmag']

    if targtype not in targposs:
        msg = 'targtype must be one of {} not {}'.format(targposs, targtype)
        log.critical(msg)
        raise ValueError(msg)


def isBGS(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          gnobs=None, rnobs=None, znobs=None, gfracmasked=None, rfracmasked=None, zfracmasked=None,
          gfracflux=None, rfracflux=None, zfracflux=None, gfracin=None, rfracin=None, zfracin=None,
          gfluxivar=None, rfluxivar=None, zfluxivar=None, brightstarinblob=None, Grr=None,
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
                         gaiagmag=gaiagmag, brightstarinblob=brightstarinblob, targtype=targtype)

    bgs &= isBGS_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                        south=south, targtype=targtype, primary=primary)

    return bgs


def notinBGS_mask(gnobs=None, rnobs=None, znobs=None, primary=None,
                  gfracmasked=None, rfracmasked=None, zfracmasked=None,
                  gfracflux=None, rfracflux=None, zfracflux=None,
                  gfracin=None, rfracin=None, zfracin=None, w1snr=None,
                  gfluxivar=None, rfluxivar=None, zfluxivar=None, Grr=None,
                  gaiagmag=None, brightstarinblob=None, targtype=None):
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

    bgs &= ~brightstarinblob

    if targtype == 'bright':
        bgs &= ((Grr > 0.6) | (gaiagmag == 0))
    elif targtype == 'faint':
        bgs &= ((Grr > 0.6) | (gaiagmag == 0))
    elif targtype == 'wise':
        bgs &= Grr < 0.4
        bgs &= Grr > -1
        bgs &= w1snr > 5

    return bgs


def isBGS_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 south=True, targtype=None, primary=None):
    """Standard set of color-based cuts used by all BGS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isBGS` for parameters).
    """
    _check_BGS_targtype(targtype)

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()

    if targtype == 'bright':
        bgs &= rflux > 10**((22.5-19.5)/2.5)
    elif targtype == 'faint':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        bgs &= rflux <= 10**((22.5-19.5)/2.5)
    elif targtype == 'wise':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        bgs &= w1flux*gflux > (zflux*rflux)*10**(-0.2)

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

    return bgs


def isQSO_cuts(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
               w1snr=None, w2snr=None, deltaChi2=None, brightstarinblob=None,
               release=None, objtype=None, primary=None, optical=False, south=True):
    """Definition of QSO target classes from color cuts. Returns a boolean array.

    Parameters
    ----------
    optical : :class:`boolean`, defaults to ``False``
        Apply just optical color-cuts.
    south : :class:`boolean`, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
        otherwise use cuts appropriate to the Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` for objects that pass the quasar color/morphology/logic cuts.

    Notes
    -----
    - Current version (12/07/18) is version 159 on `the wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """

    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                       w1flux=w1flux, w2flux=w2flux,
                       optical=optical, south=south)

    qso &= w1snr > 4
    qso &= w2snr > 2

    # ADM default to RELEASE of 6000 if nothing is passed.
    if release is None:
        release = np.zeros_like(gflux, dtype='?')+6000

    qso &= ((deltaChi2 > 40.) | (release >= 5000))

    if primary is not None:
        qso &= primary

    if objtype is not None:
        qso &= _psflike(objtype)

    # CAC Reject objects flagged inside a blob.
    if brightstarinblob is not None:
        qso &= ~brightstarinblob

    return qso


def isQSO_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 optical=False, south=True):
    """Tests if sources have quasar-like colors in a color box.
    (see, e.g., :func:`~desitarget.cuts.isQSO_cuts`).
    """
    # ----- Quasars
    # Create some composite fluxes.
    wflux = 0.75*w1flux + 0.25*w2flux
    grzflux = (gflux + 0.8*rflux + 0.5*zflux) / 2.3

    qso = np.ones_like(gflux, dtype='?')
    qso &= rflux < 10**((22.5-17.5)/2.5)    # r>17.5
    qso &= rflux > 10**((22.5-22.7)/2.5)    # r<22.7
    qso &= grzflux < 10**((22.5-17)/2.5)    # grz>17
    qso &= rflux < gflux * 10**(1.3/2.5)    # (g-r)<1.3
    qso &= zflux > rflux * 10**(-0.3/2.5)   # (r-z)>-0.3
    qso &= zflux < rflux * 10**(1.1/2.5)    # (r-z)<1.1

    if not optical:
        qso &= w2flux > w1flux * 10**(-0.4/2.5)                   # (W1-W2)>-0.4
        qso &= wflux * gflux > zflux * grzflux * 10**(-1.0/2.5)   # (grz-W)>(g-z)-1.0

    # Harder cut on stellar contamination
    mainseq = rflux > gflux * 10**(0.20/2.5)

    # Clip to avoid warnings from negative numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    mainseq &= rflux**(1+1.5) > gflux * zflux**1.5 * 10**((-0.100+0.175)/2.5)
    mainseq &= rflux**(1+1.5) < gflux * zflux**1.5 * 10**((+0.100+0.175)/2.5)
    if not optical:
        mainseq &= w2flux < w1flux * 10**(0.3/2.5)
    qso &= ~mainseq

    return qso


def isQSO_randomforest(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                       objtype=None, release=None, deltaChi2=None, brightstarinblob=None,
                       primary=None, south=True):
    """Convenience function for backwards-compatability prior to north/south split.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        objtype: array_like or None
            If given, the TYPE column of the Tractor catalogue.
        release: array_like[ntargets]
            `The Legacy Surveys`_ imaging RELEASE.
        deltaChi2: array_like or None
             If given, difference in chi2 bteween PSF and SIMP morphology
        brightstarinblob: boolean array_like or None
            ``True`` if the object shares a blob with a "bright" (Tycho-2) star.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to ``True``
            Call isQSO_randomforest_north if ``south=False``,
            otherwise call isQSO_randomforest_south.

    Returns:
        mask : array_like. True if and only if the object is a QSO
            target.

    Notes:
        as of 10/16/18, based on version 143 on `the wiki`_.
    """

    if south is False:
        return isQSO_randomforest_north(gflux=gflux, rflux=rflux, zflux=zflux,
                                        w1flux=w1flux, w2flux=w2flux, objtype=objtype,
                                        release=release, deltaChi2=deltaChi2,
                                        brightstarinblob=brightstarinblob, primary=primary)
    else:
        return isQSO_randomforest_south(gflux=gflux, rflux=rflux, zflux=zflux,
                                        w1flux=w1flux, w2flux=w2flux, objtype=objtype,
                                        release=release, deltaChi2=deltaChi2,
                                        brightstarinblob=brightstarinblob, primary=primary)


def isQSO_randomforest_north(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                             objtype=None, release=None, deltaChi2=None, brightstarinblob=None,
                             primary=None):
    """
    Target definition of QSO using a random forest for the BASS/MzLS photometric system.
    (see :func:`~desitarget.cuts.isQSO_randomforest`).
    """
    # BRICK_PRIMARY
    if primary is None:
        primary = np.ones_like(gflux, dtype=bool)

    # RELEASE
    # ADM default to RELEASE of 6000 if nothing is passed.
    if release is None:
        release = np.zeros_like(gflux, dtype='?') + 6000
    release = np.atleast_1d(release)

    # Build variables for random forest
    nFeatures = 11   # Number of attributes describing each object to be classified by the rf
    nbEntries = rflux.size
    gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)
    colors, r, photOK = _getColors(nbEntries, nFeatures, gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d(r)

    # Preselection to speed up the process
    rMax = 22.7   # r < 22.7
    rMin = 17.5   # r > 17.5
    preSelection = (r < rMax) & (r > rMin) & photOK & primary

    if objtype is not None:
        preSelection &= _psflike(objtype)
    if deltaChi2 is not None:
        deltaChi2 = np.atleast_1d(deltaChi2)
        preSelection[release < 5000] &= deltaChi2[release < 5000] > 30.
    # CAC Reject objects flagged inside a blob.
    if brightstarinblob is not None:
        preSelection &= ~brightstarinblob

    # "qso" mask initialized to "preSelection" mask
    qso = np.copy(preSelection)

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
        rf_DR5_fileName = pathToRF + '/rf_model_dr7.npz'
        rf_DR5_HighZ_fileName = pathToRF + '/rf_model_dr7_HighZ.npz'

        tmpReleaseOK = releaseReduced < 6000
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

        tmpReleaseOK = releaseReduced >= 6000
        if np.any(tmpReleaseOK):
            # rf initialization - colors data duplicated within "myRF"
            rf_DR5 = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                          numberOfTrees=500, version=2)
            rf_DR5_HighZ = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                                numberOfTrees=500, version=2)
            # rf loading
            rf_DR5.loadForest(rf_DR5_fileName)
            rf_DR5_HighZ.loadForest(rf_DR5_HighZ_fileName)
            # Compute rf probabilities
            tmp_rf_proba = rf_DR5.predict_proba()
            tmp_rf_HighZ_proba = rf_DR5_HighZ.predict_proba()
            # Compute optimized proba cut
            tmp_r_Reduced = r_Reduced[tmpReleaseOK]
            pcut = np.where(tmp_r_Reduced > 20.,
                            0.60 - (tmp_r_Reduced - 20.) * 0.08, 0.60)
            pcut_HighZ = 0.42
            # Add rf proba test result to "qso" mask
            qso[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_proba >= pcut) | (tmp_rf_HighZ_proba >= pcut_HighZ)

    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "~numpy.ndarray"
    if nbEntries == 1:
        qso = qso[0]

    return qso


def isQSO_randomforest_south(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                             objtype=None, release=None, deltaChi2=None, brightstarinblob=None,
                             primary=None):
    """
    Target definition of QSO using a random forest for the DECaLS photometric system.
    (see :func:`~desitarget.cuts.isQSO_randomforest`).
    """
    # BRICK_PRIMARY
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
    colors, r, photOK = _getColors(nbEntries, nFeatures, gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d(r)

    # Preselection to speed up the process
    rMax = 22.7   # r < 22.7
    rMin = 17.5   # r > 17.5
    preSelection = (r < rMax) & (r > rMin) & photOK & primary

    if objtype is not None:
        preSelection &= _psflike(objtype)
    if deltaChi2 is not None:
        deltaChi2 = np.atleast_1d(deltaChi2)
        preSelection[release < 5000] &= deltaChi2[release < 5000] > 30.
    # CAC Reject objects flagged inside a blob.
    if brightstarinblob is not None:
        preSelection &= ~brightstarinblob

    # "qso" mask initialized to "preSelection" mask
    qso = np.copy(preSelection)

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
        rf_DR5_fileName = pathToRF + '/rf_model_dr7.npz'
        rf_DR5_HighZ_fileName = pathToRF + '/rf_model_dr7_HighZ.npz'

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

        tmpReleaseOK = releaseReduced >= 5000
        if np.any(tmpReleaseOK):
            # rf initialization - colors data duplicated within "myRF"
            rf_DR5 = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                          numberOfTrees=500, version=2)
            rf_DR5_HighZ = myRF(colorsReduced[tmpReleaseOK], pathToRF,
                                numberOfTrees=500, version=2)
            # rf loading
            rf_DR5.loadForest(rf_DR5_fileName)
            rf_DR5_HighZ.loadForest(rf_DR5_HighZ_fileName)
            # Compute rf probabilities
            tmp_rf_proba = rf_DR5.predict_proba()
            tmp_rf_HighZ_proba = rf_DR5_HighZ.predict_proba()
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

    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "~numpy.ndarray"
    if nbEntries == 1:
        qso = qso[0]

    return qso


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


def _isonnorthphotsys(photsys):
    """ If the object is from the northen photometric system """
    # ADM explicitly checking for NoneType. I can't see why we'd ever want to
    # ADM run this test on empty information. In the past we have had bugs where
    # ADM we forgot to populate variables before passing them
    if photsys is None:
        raise ValueError("NoneType submitted to _isonnorthphotsys function")

    psftype = np.asarray(photsys)
    # ADM in Python3 these string literals become byte-like
    # ADM so to retain Python2 compatibility we need to check
    # ADM against both bytes and unicode
    northern = ((photsys == 'N') | (photsys == b'N'))
    return northern


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


def _get_colnames(objects):
    """Simple wrapper to get the column names."""

    # ADM capture the case that a single FITS_REC is passed
    import astropy.io.fits.fitsrec
    if isinstance(objects, astropy.io.fits.fitsrec.FITS_record):
        colnames = objects.__dict__['array'].dtype.names
    else:
        colnames = objects.dtype.names

    return colnames


def _prepare_optical_wise(objects, colnames=None):
    """Process the Legacy Surveys inputs for target selection."""

    if colnames is None:
        colnames = _get_colnames(objects)

    # ADM flag whether we're using northen (BASS/MZLS) or
    # ADM southern (DECaLS) photometry
    photsys_north = _isonnorthphotsys(objects["PHOTSYS"])
    photsys_south = ~_isonnorthphotsys(objects["PHOTSYS"])

    # ADM rewrite the fluxes to shift anything on the northern Legacy Surveys
    # ADM system to approximate the southern system
    # ADM turn off shifting the northern photometry to match the southern
    # ADM photometry. The consensus at the May, 2018 DESI collaboration meeting
    # ADM in Tucson was not to do this.
#    wnorth = np.where(photsys_north)
#    if len(wnorth[0]) > 0:
#        gshift, rshift, zshift = shift_photo_north(objects["FLUX_G"][wnorth],
#                                                   objects["FLUX_R"][wnorth],
#                                                   objects["FLUX_Z"][wnorth])
#        objects["FLUX_G"][wnorth] = gshift
#        objects["FLUX_R"][wnorth] = rshift
#        objects["FLUX_Z"][wnorth] = zshift

    # ADM the observed r-band flux (used for F standards and MWS, below)
    # ADM make copies of values that we may reassign due to NaNs
    obs_rflux = objects['FLUX_R']

    # - undo Milky Way extinction
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    rfiberflux = flux['RFIBERFLUX']
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

    gfracin = objects['FRACIN_G']
    rfracin = objects['FRACIN_R']
    zfracin = objects['FRACIN_Z']

    gallmask = objects['ALLMASK_G']
    rallmask = objects['ALLMASK_R']
    zallmask = objects['ALLMASK_Z']

    gsnr = objects['FLUX_G'] * np.sqrt(objects['FLUX_IVAR_G'])
    rsnr = objects['FLUX_R'] * np.sqrt(objects['FLUX_IVAR_R'])
    zsnr = objects['FLUX_Z'] * np.sqrt(objects['FLUX_IVAR_Z'])
    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    w2snr = objects['FLUX_W2'] * np.sqrt(objects['FLUX_IVAR_W2'])

    # For BGS target selection.
    brightstarinblob = (objects['BRIGHTBLOB'] & 2**0) != 0

    # Delta chi2 between PSF and SIMP morphologies; note the sign....
    dchisq = objects['DCHISQ']
    deltaChi2 = dchisq[..., 0] - dchisq[..., 1]

    # ADM remove handful of NaN values from DCHISQ values and make them unselectable.
    w = np.where(deltaChi2 != deltaChi2)
    # ADM this is to catch the single-object case for unit tests.
    if len(w[0]) > 0:
        deltaChi2[w] = -1e6

    return (photsys_north, photsys_south, obs_rflux, gflux, rflux, zflux,
            w1flux, w2flux, rfiberflux, objtype, release, gfluxivar, rfluxivar, zfluxivar,
            gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,
            gfracmasked, rfracmasked, zfracmasked,
            gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,
            gsnr, rsnr, zsnr, w1snr, w2snr, dchisq, deltaChi2, brightstarinblob)


def _prepare_gaia(objects, colnames=None):
    """Process the various Gaia inputs for target selection."""

    if colnames is None:
        colnames = _get_colnames(objects)

    # ADM Add the Gaia columns...
    # ADM if we don't have REF_CAT in the sweeps use the
    # ADM minimum value of REF_ID to identify Gaia sources. This will
    # ADM introduce a small number (< 0.001%) of Tycho-only sources.
    gaia = objects['REF_ID'] > 0
    if "REF_CAT" in colnames:
        gaia = (objects['REF_CAT'] == b'G2') | (objects['REF_CAT'] == 'G2')
    pmra = objects['PMRA']
    pmdec = objects['PMDEC']
    pmraivar = objects['PMRA_IVAR']
    parallax = objects['PARALLAX']
    parallaxivar = objects['PARALLAX_IVAR']
    # ADM derive the parallax/parallax_error, but set to 0 where the error is bad
    parallaxovererror = np.where(parallaxivar > 0., parallax*np.sqrt(parallaxivar), 0.)

    # We also need the parallax uncertainty, to select MWS_NEARBY targets.
    parallaxerr = np.zeros_like(parallax) - 1e8  # make large and negative
    notzero = parallaxivar > 0
    if np.sum(notzero) > 0:
        parallaxerr[notzero] = 1 / np.sqrt(parallaxivar[notzero])
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    gaiabmag = objects['GAIA_PHOT_BP_MEAN_MAG']
    gaiarmag = objects['GAIA_PHOT_RP_MEAN_MAG']
    gaiaaen = objects['GAIA_ASTROMETRIC_EXCESS_NOISE']
    # ADM a mild hack, as GAIA_DUPLICATED_SOURCE was a 0/1 integer at some point.
    gaiadupsource = objects['GAIA_DUPLICATED_SOURCE']
    if issubclass(gaiadupsource.dtype.type, np.integer):
        if len(set(np.atleast_1d(gaiadupsource)) - set([0, 1])) == 0:
            gaiadupsource = objects['GAIA_DUPLICATED_SOURCE'].astype(bool)

    # For BGS target selection
    Grr = gaiagmag - 22.5 + 2.5*np.log10(objects['FLUX_R'])

    # ADM If proper motion is not NaN, 31 parameters were solved for
    # ADM in Gaia astrometry. Or, gaiaparamssolved should be 3 for NaNs).
    # ADM In the sweeps, NaN has not been preserved...but PMRA_IVAR == 0
    # ADM in the sweeps is equivalent to PMRA of NaN in Gaia.
    if 'GAIA_ASTROMETRIC_PARAMS_SOLVED' in colnames:
        gaiaparamssolved = objects['GAIA_ASTROMETRIC_PARAMS_SOLVED']
    else:
        gaiaparamssolved = np.zeros_like(gaia) + 31
        w = np.where(np.isnan(pmra) | (pmraivar == 0))[0]
        if len(w) > 0:
            # ADM we need to check the case of a single row being passed
            if _is_row(gaiaparamssolved):
                gaiaparamsolved = 3
            else:
                gaiaparamssolved[w] = 3

    # ADM Add these columns if they exist, or set them to none.
    # ADM They aren't in the Tractor files as of DR7.
    gaiabprpfactor = None
    gaiasigma5dmax = None
    if 'GAIA_PHOT_BP_RP_EXCESS_FACTOR' in colnames:
        gaiabprpfactor = objects['GAIA_PHOT_BP_RP_EXCESS_FACTOR']
    if 'GAIA_ASTROMETRIC_SIGMA5D_MAX' in colnames:
        gaiasig5dmax = objects['GAIA_ASTROMETRIC_SIGMA5D_MAX']

    # ADM Mily Way Selection requires Galactic b
    _, galb = _gal_coords(objects["RA"], objects["DEC"])

    return (gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr, gaiagmag,
            gaiabmag, gaiarmag, gaiaaen, gaiadupsource, Grr, gaiaparamssolved,
            gaiabprpfactor, gaiasigma5dmax, galb)


def unextinct_fluxes(objects):
    """Calculate unextincted DECam and WISE fluxes.

    Args:
        objects: array or Table with columns FLUX_G, FLUX_R, FLUX_Z,
            MW_TRANSMISSION_G, MW_TRANSMISSION_R, MW_TRANSMISSION_Z,
            FLUX_W1, FLUX_W2, MW_TRANSMISSION_W1, MW_TRANSMISSION_W2

    Returns:
        array or Table with columns GFLUX, RFLUX, ZFLUX, W1FLUX, W2FLUX

    Output type is Table if input is Table, otherwise numpy structured array
    """
    dtype = [('GFLUX', 'f4'), ('RFLUX', 'f4'), ('ZFLUX', 'f4'),
             ('W1FLUX', 'f4'), ('W2FLUX', 'f4'), ('RFIBERFLUX', 'f4')]
    if _is_row(objects):
        result = np.zeros(1, dtype=dtype)[0]
    else:
        result = np.zeros(len(objects), dtype=dtype)

    result['GFLUX'] = objects['FLUX_G'] / objects['MW_TRANSMISSION_G']
    result['RFLUX'] = objects['FLUX_R'] / objects['MW_TRANSMISSION_R']
    result['ZFLUX'] = objects['FLUX_Z'] / objects['MW_TRANSMISSION_Z']
    result['W1FLUX'] = objects['FLUX_W1'] / objects['MW_TRANSMISSION_W1']
    result['W2FLUX'] = objects['FLUX_W2'] / objects['MW_TRANSMISSION_W2']
    result['RFIBERFLUX'] = objects['FIBERFLUX_R'] / objects['MW_TRANSMISSION_R']

    if isinstance(objects, Table):
        return Table(result)
    else:
        return result


def set_target_bits(photsys_north, photsys_south, obs_rflux,
                    gflux, rflux, zflux, w1flux, w2flux, rfiberflux,
                    objtype, release, gfluxivar, rfluxivar, zfluxivar,
                    gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,
                    gfracmasked, rfracmasked, zfracmasked,
                    gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,
                    gsnr, rsnr, zsnr, w1snr, w2snr, deltaChi2, dchisq,
                    gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr,
                    gaiagmag, gaiabmag, gaiarmag, gaiaaen, gaiadupsource,
                    gaiaparamssolved, gaiabprpfactor, gaiasigma5dmax, galb,
                    tcnames, qso_optical_cuts, qso_selection, brightstarinblob,
                    Grr, primary):
    """Perform target selection on parameters, returning target mask arrays.

    Parameters
    ----------
    photsys_north, photsys_south : :class:`~numpy.ndarray`
        ``True`` for objects that were drawn from northern (MzLS/BASS) or
        southern (DECaLS) imaging, respectively.
    obs_rflux : :class:`~numpy.ndarray`
        `rflux` but WITHOUT any Galactic extinction correction.
    gflux, rflux, zflux, w1flux, w2flux : :class:`~numpy.ndarray`
        The flux in nano-maggies of g, r, z, W1 and W2 bands.
        Corrected for Galactic extinction.
    rfiberflux : :class:`~numpy.ndarray`
        Predicted fiber flux in 1 arcsecond seeing in r-band.
        Corrected for Galactic extinction.
    objtype, release : :class:`~numpy.ndarray`
        `The Legacy Surveys`_ imaging ``TYPE`` and ``RELEASE`` columns.
    gfluxivar, rfluxivar, zfluxivar: :class:`~numpy.ndarray`
        The flux inverse variances in g, r, and z bands.
    gnobs, rnobs, znobs: :class:`~numpy.ndarray`
        The number of observations (in the central pixel) in g, r and z.
    gfracflux, rfracflux, zfracflux: :class:`~numpy.ndarray`
        Profile-weighted fraction of the flux from other sources divided
        by the total flux in g, r and z bands.
    gfracmasked, rfracmasked, zfracmasked: :class:`~numpy.ndarray`
        Fraction of masked pixels in the g, r and z bands.
    gallmask, rallmask, zallmask: :class:`~numpy.ndarray`
        Bitwise mask set if the central pixel from all images
        satisfy each condition in g, r, z.
    gsnr, rsnr, zsnr, w1snr, w2snr: :class:`~numpy.ndarray`
        Signal-to-noise in g, r, z, W1 and W2 defined as the flux per
        band divided by sigma (flux x sqrt of the inverse variance).
    deltaChi2: :class:`~numpy.ndarray`
        chi2 difference between PSF and SIMP, dchisq_PSF - dchisq_SIMP.
    dchisq: :class:`~numpy.ndarray`
        Difference in chi2  between successively more-complex model fits.
        Columns are model fits, in order, of PSF, REX, EXP, DEV, COMP.
    gaia: :class:`~numpy.ndarray`
        ``True`` if there is a match between this object in
        `the Legacy Surveys`_ and in Gaia.
    pmra, pmdec, parallax, parallaxovererror: :class:`~numpy.ndarray`
        Gaia-based proper motion in RA and Dec, and parallax and error.
    gaiagmag, gaiabmag, gaiarmag: :class:`~numpy.ndarray`
            Gaia-based g-, b- and r-band MAGNITUDES.
    gaiaaen, gaiadupsource, gaiaparamssolved: :class:`~numpy.ndarray`
        Gaia-based measures of Astrometric Excess Noise, whether the source
        is a duplicate, and how many parameters were solved for.
    gaiabprpfactor, gaiasigma5dmax: :class:`~numpy.ndarray`
        Gaia_based BP/RP excess factor and longest semi-major axis
        of 5-d error ellipsoid.
    galb: :class:`~numpy.ndarray`
        Galactic latitude (degrees).
    tcnames : :class:`list`, defaults to running all target classes
        A list of strings, e.g. ['QSO','LRG']. If passed, process targeting only
        for those specific target classes. A useful speed-up when testing.
        Options include ["ELG", "QSO", "LRG", "MWS", "BGS", "STD"].
    qso_optical_cuts : :class:`boolean` defaults to ``False``
        Apply just optical color-cuts when selecting QSOs with
        ``qso_selection="colorcuts"``.
    qso_selection : :class:`str`, optional, defaults to ``'randomforest'``
        The algorithm to use for QSO selection; valid options are
        ``'colorcuts'`` and ``'randomforest'``
    brightstarinblob: boolean array_like or None
        ``True`` if the object shares a blob with a "bright" (Tycho-2) star.
    Grr: array_like or None
        Gaia G band magnitude minus observational r magnitude.
    primary : :class:`~numpy.ndarray`
        ``True`` for objects that should be considered when setting bits.
    survey : :class:`str`, defaults to ``'main'``
        Specifies which target masks yaml file and target selection cuts
        to use. Options are ``'main'`` and ``'svX``' (where X is 1, 2, 3 etc.)
        for the main survey and different iterations of SV, respectively.

    Returns
    -------
    :class:`~numpy.ndarray`
        (desi_target, bgs_target, mws_target) where each element is
        an ndarray of target selection bitmask flags for each object.

    Notes
    -----
    - Gaia quantities have units that are the same as `the Gaia data model`_.
    """

    from desitarget.targetmask import desi_mask, bgs_mask, mws_mask

    if "LRG" in tcnames:
        lrg_north, lrg1pass_north, lrg2pass_north = isLRGpass(
            primary=primary,
            gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, gflux_ivar=gfluxivar,
            rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr, south=False
        )

        lrg_south, lrg1pass_south, lrg2pass_south = isLRGpass(
            primary=primary,
            gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, gflux_ivar=gfluxivar,
            rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr, south=True
        )
    else:
        # ADM if not running the LRG selection, set everything to arrays of False
        lrg_north, lrg1pass_north, lrg2pass_north = ~primary, ~primary, ~primary
        lrg_south, lrg1pass_south, lrg2pass_south = ~primary, ~primary, ~primary

    # ADM combine LRG target bits for an LRG target based on any imaging
    lrg = (lrg_north & photsys_north) | (lrg_south & photsys_south)
    lrg1pass = (lrg1pass_north & photsys_north) | (lrg1pass_south & photsys_south)
    lrg2pass = (lrg2pass_north & photsys_north) | (lrg2pass_south & photsys_south)

    if "ELG" in tcnames:
        elg_classes = []
        for south in [False, True]:
            elg_classes.append(
                isELG(
                    primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
                    gallmask=gallmask, rallmask=rallmask, zallmask=zallmask,
                    brightstarinblob=brightstarinblob, south=south)
            )
        elg_north, elg_south = elg_classes
    else:
        # ADM if not running the ELG selection, set everything to arrays of False.
        elg_north, elg_south = ~primary, ~primary

    # ADM combine ELG target bits for an ELG target based on any imaging
    elg = (elg_north & photsys_north) | (elg_south & photsys_south)

    if "QSO" in tcnames:
        if qso_selection == 'colorcuts':
            # ADM determine quasar targets in the north and the south separately
            qso_north = isQSO_cuts(
                primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                w1flux=w1flux, w2flux=w2flux,
                deltaChi2=deltaChi2, brightstarinblob=brightstarinblob,
                objtype=objtype, w1snr=w1snr, w2snr=w2snr, release=release,
                optical=qso_optical_cuts, south=False
            )
            qso_south = isQSO_cuts(
                primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                w1flux=w1flux, w2flux=w2flux,
                deltaChi2=deltaChi2, brightstarinblob=brightstarinblob,
                objtype=objtype, w1snr=w1snr, w2snr=w2snr, release=release,
                optical=qso_optical_cuts, south=True
            )
        elif qso_selection == 'randomforest':
            # ADM determine quasar targets in the north and the south separately
            qso_north = isQSO_randomforest(
                primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                w1flux=w1flux, w2flux=w2flux,
                deltaChi2=deltaChi2, brightstarinblob=brightstarinblob,
                objtype=objtype, release=release, south=False
            )
            qso_south = isQSO_randomforest(
                primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                w1flux=w1flux, w2flux=w2flux,
                deltaChi2=deltaChi2, brightstarinblob=brightstarinblob,
                objtype=objtype, release=release, south=True
            )
        else:
            raise ValueError('Unknown qso_selection {}; valid options are {}'.format(
                qso_selection, qso_selection_options))
    else:
        # ADM if not running the QSO selection, set everything to arrays of False
        qso_north, qso_south = ~primary, ~primary

    # ADM combine quasar target bits for a quasar target based on any imaging
    qso = (qso_north & photsys_north) | (qso_south & photsys_south)

    # ADM set the BGS bits
    if "BGS" in tcnames:
        bgs_classes = []
        for targtype in ["bright", "faint", "wise"]:
            for south in [False, True]:
                bgs_classes.append(
                    isBGS(
                        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                        gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                        gfracmasked=gfracmasked, rfracmasked=rfracmasked, zfracmasked=zfracmasked,
                        gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                        gfracin=gfracin, rfracin=rfracin, zfracin=zfracin,
                        gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
                        brightstarinblob=brightstarinblob, Grr=Grr, w1snr=w1snr, gaiagmag=gaiagmag,
                        objtype=objtype, primary=primary, south=south, targtype=targtype
                    )
                )

        bgs_bright_north, bgs_bright_south,      \
            bgs_faint_north, bgs_faint_south,    \
            bgs_wise_north, bgs_wise_south =     \
            bgs_classes
    else:
        # ADM if not running the BGS selection, set everything to arrays of False
        bgs_bright_north, bgs_bright_south = ~primary, ~primary
        bgs_faint_north, bgs_faint_south = ~primary, ~primary
        bgs_wise_north, bgs_wise_south = ~primary, ~primary

    # ADM combine BGS targeting bits for a BGS selected in any imaging
    bgs_bright = (bgs_bright_north & photsys_north) | (bgs_bright_south & photsys_south)
    bgs_faint = (bgs_faint_north & photsys_north) | (bgs_faint_south & photsys_south)
    bgs_wise = (bgs_wise_north & photsys_north) | (bgs_wise_south & photsys_south)

    if "MWS" in tcnames:
        mws_classes = []
        # ADM run the MWS target types for both north and south
        for south in [False, True]:
            mws_classes.append(
                isMWS_main(
                    gaia=gaia, gaiaaen=gaiaaen, gaiadupsource=gaiadupsource,
                    gflux=gflux, rflux=rflux, obs_rflux=obs_rflux, objtype=objtype,
                    gnobs=gnobs, rnobs=rnobs,
                    gfracmasked=gfracmasked, rfracmasked=rfracmasked,
                    pmra=pmra, pmdec=pmdec, parallax=parallax,
                    primary=primary, south=south
                )
            )

        mws_broad_n, mws_red_n, mws_blue_n,       \
            mws_broad_s, mws_red_s, mws_blue_s =  \
            np.vstack(mws_classes)

        mws_nearby = isMWS_nearby(
            gaia=gaia, gaiagmag=gaiagmag, parallax=parallax,
            parallaxerr=parallaxerr
        )
        mws_wd = isMWS_WD(
            gaia=gaia, galb=galb, astrometricexcessnoise=gaiaaen,
            pmra=pmra, pmdec=pmdec, parallax=parallax, parallaxovererror=parallaxovererror,
            photbprpexcessfactor=gaiabprpfactor, astrometricsigma5dmax=gaiasigma5dmax,
            gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag
        )
    else:
        # ADM if not running the MWS selection, set everything to arrays of False
        mws_broad_n, mws_red_n, mws_blue_n = ~primary, ~primary, ~primary
        mws_broad_s, mws_red_s, mws_blue_s = ~primary, ~primary, ~primary
        mws_nearby, mws_wd = ~primary, ~primary

    if "STD" in tcnames:
        # ADM Make sure to pass all of the needed columns! At one point we stopped
        # ADM passing objtype, which meant no standards were being returned.
        std_faint = isSTD(
            primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
            gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
            gfracmasked=gfracmasked, rfracmasked=rfracmasked, objtype=objtype,
            zfracmasked=zfracmasked, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
            gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
            gaia=gaia, astrometricexcessnoise=gaiaaen, paramssolved=gaiaparamssolved,
            pmra=pmra, pmdec=pmdec, parallax=parallax, dupsource=gaiadupsource,
            gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag, bright=False
        )
        std_bright = isSTD(
            primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
            gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
            gfracmasked=gfracmasked, rfracmasked=rfracmasked, objtype=objtype,
            zfracmasked=zfracmasked, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
            gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
            gaia=gaia, astrometricexcessnoise=gaiaaen, paramssolved=gaiaparamssolved,
            pmra=pmra, pmdec=pmdec, parallax=parallax, dupsource=gaiadupsource,
            gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag, bright=True
        )
        # ADM the standard WDs are currently identical to the MWS WDs
        std_wd = mws_wd
    else:
        # ADM if not running the standards selection, set everything to arrays of False
        std_faint, std_bright, std_wd = ~primary, ~primary, ~primary

    # ADM combine the north/south MWS bits.
    mws_broad = (mws_broad_n & photsys_north) | (mws_broad_s & photsys_south)
    mws_blue = (mws_blue_n & photsys_north) | (mws_blue_s & photsys_south)
    mws_red = (mws_red_n & photsys_north) | (mws_red_s & photsys_south)

    # Construct the targetflag bits for DECaLS (i.e. South).
    desi_target = lrg_south * desi_mask.LRG_SOUTH
    desi_target |= elg_south * desi_mask.ELG_SOUTH
    desi_target |= qso_south * desi_mask.QSO_SOUTH

    # Construct the targetflag bits for MzLS and BASS (i.e. North).
    desi_target |= lrg_north * desi_mask.LRG_NORTH
    desi_target |= elg_north * desi_mask.ELG_NORTH
    desi_target |= qso_north * desi_mask.QSO_NORTH

    # Construct the targetflag bits combining north and south.
    desi_target |= lrg * desi_mask.LRG
    desi_target |= elg * desi_mask.ELG
    desi_target |= qso * desi_mask.QSO

    # ADM add the per-pass information in the south...
    desi_target |= lrg1pass_south * desi_mask.LRG_1PASS_SOUTH
    desi_target |= lrg2pass_south * desi_mask.LRG_2PASS_SOUTH
    # ADM ...the north...
    desi_target |= lrg1pass_north * desi_mask.LRG_1PASS_NORTH
    desi_target |= lrg2pass_north * desi_mask.LRG_2PASS_NORTH
    # ADM ...and combined.
    desi_target |= lrg1pass * desi_mask.LRG_1PASS
    desi_target |= lrg2pass * desi_mask.LRG_2PASS

    # ADM Standards.
    desi_target |= std_faint * desi_mask.STD_FAINT
    desi_target |= std_bright * desi_mask.STD_BRIGHT
    desi_target |= std_wd * desi_mask.STD_WD

    # BGS bright and faint, south.
    bgs_target = bgs_bright_south * bgs_mask.BGS_BRIGHT_SOUTH
    bgs_target |= bgs_faint_south * bgs_mask.BGS_FAINT_SOUTH
    bgs_target |= bgs_wise_south * bgs_mask.BGS_WISE_SOUTH

    # BGS bright and faint, north.
    bgs_target |= bgs_bright_north * bgs_mask.BGS_BRIGHT_NORTH
    bgs_target |= bgs_faint_north * bgs_mask.BGS_FAINT_NORTH
    bgs_target |= bgs_wise_north * bgs_mask.BGS_WISE_NORTH

    # BGS combined, bright and faint
    bgs_target |= bgs_bright * bgs_mask.BGS_BRIGHT
    bgs_target |= bgs_faint * bgs_mask.BGS_FAINT
    bgs_target |= bgs_wise * bgs_mask.BGS_WISE

    # ADM MWS main, nearby, and WD.
    mws_target = mws_broad * mws_mask.MWS_BROAD
    mws_target |= mws_wd * mws_mask.MWS_WD
    mws_target |= mws_nearby * mws_mask.MWS_NEARBY

    # ADM MWS main north/south split.
    mws_target |= mws_broad_n * mws_mask.MWS_BROAD_NORTH
    mws_target |= mws_broad_s * mws_mask.MWS_BROAD_SOUTH

    # ADM MWS main blue/red split.
    mws_target |= mws_blue * mws_mask.MWS_MAIN_BLUE
    mws_target |= mws_blue_n * mws_mask.MWS_MAIN_BLUE_NORTH
    mws_target |= mws_blue_s * mws_mask.MWS_MAIN_BLUE_SOUTH
    mws_target |= mws_red * mws_mask.MWS_MAIN_RED
    mws_target |= mws_red_n * mws_mask.MWS_MAIN_RED_NORTH
    mws_target |= mws_red_s * mws_mask.MWS_MAIN_RED_SOUTH

    # Are any BGS or MWS bit set?  Tell desi_target too.
    desi_target |= (bgs_target != 0) * desi_mask.BGS_ANY
    desi_target |= (mws_target != 0) * desi_mask.MWS_ANY

    return desi_target, bgs_target, mws_target


def apply_cuts(objects, qso_selection='randomforest', gaiamatch=False,
               tcnames=["ELG", "QSO", "LRG", "MWS", "BGS", "STD"],
               qso_optical_cuts=False, survey='main'):
    """Perform target selection on objects, returning target mask arrays.

    Parameters
    ----------
    objects : :class:`~numpy.ndarray` or `str`
        numpy structured array with UPPERCASE columns needed for
        target selection, OR a string tractor/sweep filename.
    qso_selection : :class:`str`, optional, defaults to ``'randomforest'``
        The algorithm to use for QSO selection; valid options are
        ``'colorcuts'`` and ``'randomforest'``
    gaiamatch : :class:`boolean`, optional, defaults to ``False``
        If ``True``, match to Gaia DR2 chunks files and populate Gaia columns
        to facilitate the MWS and STD selections.
    tcnames : :class:`list`, defaults to running all target classes
        A list of strings, e.g. ['QSO','LRG']. If passed, process targeting only
        for those specific target classes. A useful speed-up when testing.
        Options include ["ELG", "QSO", "LRG", "MWS", "BGS", "STD"].
    qso_optical_cuts : :class:`boolean` defaults to ``False``
        Apply just optical color-cuts when selecting QSOs with
        ``qso_selection="colorcuts"``.
    survey : :class:`str`, defaults to ``'main'``
        Specifies which target masks yaml file and target selection cuts
        to use. Options are ``'main'`` and ``'svX``' (where X is 1, 2, 3 etc.)
        for the main survey and different iterations of SV, respectively.

    Returns
    -------
    :class:`~numpy.ndarray`
        (desi_target, bgs_target, mws_target) where each element is
        an ndarray of target selection bitmask flags for each object.

    Notes
    -----
    - If ``objects`` is an astropy Table with lowercase column names, this
      converts them to UPPERCASE in-place, thus modifying the input table.
      To avoid this, pass in ``objects.copy()`` instead.
    - See :mod:`desitarget.targetmask` for the definition of each bit.

    """
    # - Check if objects is a filename instead of the actual data
    if isinstance(objects, str):
        objects = io.read_tractor(objects)

    # ADM add Gaia information, if requested, and if we're going to actually
    # ADM process the target classes that need Gaia columns
    if gaiamatch and ("MWS" in tcnames or "STD" in tcnames):
        log.info('Matching Gaia to {} primary objects...t = {:.1f}s'
                 .format(len(objects), time()-start))
        gaiainfo = match_gaia_to_primary(objects)
        log.info('Done with Gaia match for {} primary objects...t = {:.1f}s'
                 .format(len(objects), time()-start))
        # ADM remove the GAIA_RA, GAIA_DEC columns as they aren't
        # ADM in the imaging surveys data model.
        gaiainfo = pop_gaia_coords(gaiainfo)
        # ADM if we need to match to Gaia, stick to the first Gaia data model
        # ADM that we adopted for DR7.
        gaiainfo = pop_gaia_columns(
            gaiainfo,
            ['REF_CAT', 'GAIA_PHOT_BP_RP_EXCESS_FACTOR',
             'GAIA_ASTROMETRIC_SIGMA5D_MAX', 'GAIA_ASTROMETRIC_PARAMS_SOLVED']
        )
        # ADM add the Gaia column information to the primary array.
        for col in gaiainfo.dtype.names:
            objects[col] = gaiainfo[col]

    # - ensure uppercase column names if astropy Table.
    if isinstance(objects, (Table, Row)):
        for col in list(objects.columns.values()):
            if not col.name.isupper():
                col.name = col.name.upper()

    # ADM As we need the column names
    colnames = _get_colnames(objects)

    # ADM process the Legacy Surveys columns for Target Selection.
    photsys_north, photsys_south, obs_rflux, gflux, rflux, zflux,                      \
        w1flux, w2flux, rfiberflux, objtype, release, gfluxivar, rfluxivar, zfluxivar, \
        gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,                          \
        gfracmasked, rfracmasked, zfracmasked,                                         \
        gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,                       \
        gsnr, rsnr, zsnr, w1snr, w2snr, dchisq, deltaChi2, brightstarinblob =          \
        _prepare_optical_wise(objects, colnames=colnames)

    # Process the Gaia inputs for target selection.
    gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr, gaiagmag, gaiabmag,  \
        gaiarmag, gaiaaen, gaiadupsource, Grr, gaiaparamssolved, gaiabprpfactor,      \
        gaiasigma5dmax, galb = _prepare_gaia(objects, colnames=colnames)

    # ADM initially, every object passes the cuts (is True).
    # ADM need to guard against the case of a single row being passed.
    if _is_row(objects):
        primary = np.bool_(True)
    else:
        primary = np.ones_like(objects, dtype=bool)

    # ADM set different bits based on whether we're using the main survey
    # code or an iteration of SV.
    if survey == 'main':
        import desitarget.cuts as targcuts
    elif survey == 'sv1':
        import desitarget.sv1.sv1_cuts as targcuts
    else:
        msg = "survey must be either 'main'or 'sv1', not {}!!!".format(survey)
        log.critical(msg)
        raise ValueError(msg)

    desi_target, bgs_target, mws_target = targcuts.set_target_bits(
        photsys_north, photsys_south, obs_rflux,
        gflux, rflux, zflux, w1flux, w2flux, rfiberflux,
        objtype, release, gfluxivar, rfluxivar, zfluxivar,
        gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,
        gfracmasked, rfracmasked, zfracmasked,
        gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,
        gsnr, rsnr, zsnr, w1snr, w2snr, deltaChi2, dchisq,
        gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr,
        gaiagmag, gaiabmag, gaiarmag, gaiaaen, gaiadupsource,
        gaiaparamssolved, gaiabprpfactor, gaiasigma5dmax, galb,
        tcnames, qso_optical_cuts, qso_selection, brightstarinblob,
        Grr, primary
    )

    return desi_target, bgs_target, mws_target


def check_input_files(infiles, numproc=4):
    """
    Process input files in parallel to check whether they have
    any bugs that will prevent select_targets from completing,
    or whether files are corrupted.
    Useful to run before a full run of select_targets.

    Args:
        infiles: list of input filenames (tractor or sweep files),
            OR a single filename

    Optional:
        numproc: number of parallel processes

    Returns:
        Nothing, but prints any problematic files to screen
        together with information on the problem

    Notes:
        if numproc==1, use serial code instead of parallel
    """
    # ADM set up default logging
    from desiutil.log import get_logger
    log = get_logger()

    # - Convert single file to list of files
    if isinstance(infiles, str):
        infiles = [infiles, ]

    # - Sanity check that files exist before going further
    for filename in infiles:
        if not os.path.exists(filename):
            raise ValueError("{} doesn't exist".format(filename))

    # - function to run on every brick/sweep file
    def _check_input_files(filename):
        '''Check for corrupted values in a file'''
        from functools import partial
        from os.path import getsize

        # ADM read in Tractor or sweeps files
        objects = io.read_tractor(filename)
        # ADM if everything is OK the default meassage will be "OK"
        filemessageroot = 'OK'
        filemessageend = ''
        # ADM columns that shouldn't have zero values
        cols = [
            'BRICKID',
            # 'RA_IVAR', 'DEC_IVAR',
            'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
            #  'WISE_FLUX',
            #  'WISE_MW_TRANSMISSION','DCHISQ'
            ]
        # ADM for each of these columnes that shouldn't have zero values,
        # ADM loop through and look for zero values
        for colname in cols:
            if np.min(objects[colname]) == 0:
                filemessageroot = "WARNING...some values are zero for"
                filemessageend += " "+colname

        # ADM now, loop through entries in the file and search for 4096-byte
        # ADM blocks that are all zeros (a sign of corruption in file-writing)
        # ADM Note that fits files are padded by 2880 bytes, so we only want to
        # ADM process the file length (in bytes) - 2880
        bytestop = getsize(filename) - 2880

        with open(filename, 'rb') as f:
            for block_number, data in enumerate(iter(partial(f.read, 4096), b'')):
                if not any(data):
                    if block_number*4096 < bytestop:
                        filemessageroot = "WARNING...some values are zero for"
                        filemessageend += ' 4096-byte-block-#{0}'.format(block_number)

        return [filename, filemessageroot+filemessageend]

    # Counter for number of bricks processed;
    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    t0 = time()

    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if nbrick % 25 == 0 and nbrick > 0:
            elapsed = time() - t0
            rate = nbrick / elapsed
            log.info('{} files; {:.1f} files/sec; {:.1f} total mins elapsed'.format(nbrick, rate, elapsed/60.))
        nbrick[...] += 1    # this is an in-place modification
        return result

    # - Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            fileinfo = pool.map(_check_input_files, infiles, reduce=_update_status)
    else:
        fileinfo = list()
        for fil in infiles:
            fileinfo.append(_update_status(_check_input_files(fil)))

    fileinfo = np.array(fileinfo)
    w = np.where(fileinfo[..., 1] != 'OK')

    if len(w[0]) == 0:
        log.info('ALL FILES ARE OK')
    else:
        for fil in fileinfo[w]:
            log.info(fil[0], fil[1])

    return len(w[0])


qso_selection_options = ['colorcuts', 'randomforest']
Method_sandbox_options = ['XD', 'RF_photo', 'RF_spectro']


def select_targets(infiles, numproc=4, qso_selection='randomforest',
                   gaiamatch=False, sandbox=False, FoMthresh=None, Method=None,
                   nside=None, pixlist=None, bundlefiles=None, filespersec=0.12,
                   radecbox=None, radecrad=None,
                   tcnames=["ELG", "QSO", "LRG", "MWS", "BGS", "STD"],
                   survey='main'):
    """Process input files in parallel to select targets.

    Parameters
    ----------
    infiles : :class:`list` or `str`
        A list of input filenames (tractor or sweep files) OR a single filename.
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    qso_selection : :class:`str`, optional, defaults to ``'randomforest'``
        The algorithm to use for QSO selection; valid options are
        ``'colorcuts'`` and ``'randomforest'``.
    gaiamatch : :class:`boolean`, optional, defaults to ``False``
        If ``True``, match to Gaia DR2 chunks files and populate Gaia columns
        to facilitate the MWS and STD selections.
    sandbox : :class:`boolean`, optional, defaults to ``False``
        If ``True``, use the sample selection cuts in :mod:`desitarget.sandbox.cuts`.
    FoMthresh : :class:`float`, optional, defaults to `None`
        If a value is passed then run `apply_XD_globalerror` for ELGs in
        the sandbox. This will write out an "FoM.fits" file for every ELG target
        in the sandbox directory.
    Method : :class:`str`, optional, defaults to `None`
        Method used in the sandbox.
    nside : :class:`int`, optional, defaults to `None`
        The (NESTED) HEALPixel nside to be used with the `pixlist` and `bundlefiles` inputs.
    pixlist : :class:`list` or `int`, optional, defaults to `None`
        Only return targets in a set of (NESTED) HEALpixels at the supplied `nside`.
        Also useful for parallelizing as input files will only be processed if they
        touch a pixel in the passed list.
    bundlefiles : :class:`int`, defaults to `None`
        If not `None`, then instead of selecting the skies, print, to screen, the slurm
        script that will approximately balance the input file distribution at `bundlefiles`
        files per node. So, for instance, if `bundlefiles` is 100 then commands would be
        returned with the correct `pixlist` values set to pass to the code to pack at
        about 100 files per node across all of the passed `infiles`.
    filespersec : :class:`float`, optional, defaults to 1
        The rough number of files processed per second by the code (parallelized across
        a chosen number of nodes). Used in conjunction with `bundlefiles` for the code
        to estimate time to completion when parallelizing across pixels.
    radecbox : :class:`list`, defaults to `None`
        4-entry list of coordinates [ramin, ramax, decmin, decmax] forming the edges
        of a box in RA/Dec (degrees). Only targets in this box region will be processed.
    radecrad : :class:`list`, defaults to `None`
        3-entry list of coordinates [ra, dec, radius] forming a "circle" on the sky. For
        RA/Dec/radius in degrees. Only targets in this circle region will be processed.
    tcnames : :class:`list`, defaults to running all target classes
        A list of strings, e.g. ['QSO','LRG']. If passed, process targeting only
        for those specific target classes. A useful speed-up when testing.
        Options include ["ELG", "QSO", "LRG", "MWS", "BGS", "STD"].
    survey : :class:`str`, defaults to ``'main'``
        Specifies which target masks yaml file and target selection cuts
        to use. Options are ``'main'`` and ``'svX``' (where X is 1, 2, 3 etc.)
        for the main survey and different iterations of SV, respectively.

    Returns
    -------
    :class:`~numpy.ndarray`
        The subset of input targets which pass the cuts, including extra
        columns for ``DESI_TARGET``, ``BGS_TARGET``, and ``MWS_TARGET`` target
        selection bitmasks.

    Notes
    -----
        - if numproc==1, use serial code instead of parallel.
        - only one of pixlist, radecbox, radecrad should be passed. They are all
          intended to denote regions on the sky, using different formalisms.
    """
    from desiutil.log import get_logger
    log = get_logger()

    log.info("Running on the {} survey".format(survey))

    # - Convert single file to list of files.
    if isinstance(infiles, str):
        infiles = [infiles, ]

    # - Sanity check that files exist before going further.
    for filename in infiles:
        if not os.path.exists(filename):
            msg = "{} doesn't exist".format(filename)
            log.critical(msg)
            raise ValueError(msg)

    # ADM check that only one of pixlist, radecrad, radecbox was sent.
    inputs = [ins for ins in (pixlist, radecbox, radecrad) if ins is not None]
    if len(inputs) > 1:
        msg = "Only one of pixist, radecbox or radecrad can be passed"
        log.critical(msg)
        raise ValueError(msg)

    # ADM if radecbox was sent, determine which pixels touch the box.
    if radecbox is not None:
        nside = pixarea2nside(box_area(radecbox))
        pixlist = hp_in_box(nside, radecbox)
        log.info("Run targets in box bounded by [RAmin, RAmax, Decmin, Decmax]={}"
                 .format(radecbox))

    # ADM if radecrad was sent, determine which pixels touch the box.
    if radecrad is not None:
        nside = pixarea2nside(cap_area(np.array(radecrad[2])))
        pixlist = hp_in_cap(nside, radecrad)
        log.info("Run targets in cap bounded by [centerRA, centerDec, radius]={}"
                 .format(radecrad))

    # ADM if the pixlist or bundlefiles option was sent, we'll need to know
    # ADM which HEALPixels touch each file.
    if pixlist is not None or bundlefiles is not None:
        # ADM work with pixlist as an array.
        pixlist = np.atleast_1d(pixlist)
        # ADM sanity check that nside is OK.
        check_nside(nside)
        # ADM a list of HEALPixels that touch each file.
        # ADM this will break for Tractor files!!!
        pixelsperfile = [io.decode_sweep_name(file, nside=nside) for file in infiles]
        # ADM a flattened array of all HEALPixels touched by the input
        # ADM files. Each HEALPixel can appear multiple times if it's
        # ADM touched by multiple input sweep files.
        pixnum = np.hstack(pixelsperfile)
        # ADM restrict input pixels to only those that touch an input file.
        ii = [pix in pixnum for pix in pixlist]
        pixlist = pixlist[ii]
        # ADM create a list of files that touch each HEALPixel.
        filesperpixel = [[] for pix in range(np.max(pixnum)+1)]
        for ifile, pixels in enumerate(pixelsperfile):
            for pix in pixels:
                filesperpixel[pix].append(infiles[ifile])

    # ADM if the bundlefiles option was sent, call the packing code.
    if bundlefiles is not None:
        prefix = "targets"
        if survey != "main":
            prefix = "{}_targets".format(survey)
        bundle_bricks(pixnum, bundlefiles, nside,
                      brickspersec=filespersec, gather=False,
                      prefix=prefix, surveydir=os.path.dirname(infiles[0]))
        return

    # ADM restrict to only input files in a set of HEALPixels, if requested.
    if pixlist is not None:
        infiles = list(set(np.hstack([filesperpixel[pix] for pix in pixlist])))
        if len(infiles) == 0:
            log.warning('ZERO files in passed pixel list!!!')
        log.info("Processing files in (nside={}, pixel numbers={}) HEALPixels"
                 .format(nside, pixlist))

    # ADM a little more information if we're slurming across nodes.
    if os.getenv('SLURMD_NODENAME') is not None:
        log.info('Running on Node {}'.format(os.getenv('SLURMD_NODENAME')))

    def _finalize_targets(objects, desi_target, bgs_target, mws_target):
        # - desi_target includes BGS_ANY and MWS_ANY, so we can filter just
        # - on desi_target != 0
        keep = (desi_target != 0)
        objects = objects[keep]
        desi_target = desi_target[keep]
        bgs_target = bgs_target[keep]
        mws_target = mws_target[keep]

        # - Add *_target mask columns
        targets = finalize(objects, desi_target, bgs_target, mws_target,
                           survey=survey)
        # ADM resolve any duplicates between imaging data releases.
        targets = resolve(targets)

        return targets

    # - functions to run on every brick/sweep file
    def _select_targets_file(filename):
        '''Returns targets in filename that pass the cuts'''
        objects = io.read_tractor(filename)
        desi_target, bgs_target, mws_target = apply_cuts(
            objects, qso_selection=qso_selection, gaiamatch=gaiamatch,
            tcnames=tcnames, survey=survey
        )

        return _finalize_targets(objects, desi_target, bgs_target, mws_target)

    def _select_sandbox_targets_file(filename):
        '''Returns targets in filename that pass the sandbox cuts'''
        from desitarget.sandbox.cuts import apply_sandbox_cuts
        objects = io.read_tractor(filename)
        desi_target, bgs_target, mws_target = apply_sandbox_cuts(objects, FoMthresh, Method)

        return _finalize_targets(objects, desi_target, bgs_target, mws_target)

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

    # - Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            if sandbox:
                log.info("You're in the sandbox...")
                targets = pool.map(_select_sandbox_targets_file, infiles, reduce=_update_status)
            else:
                targets = pool.map(_select_targets_file, infiles, reduce=_update_status)
    else:
        targets = list()
        if sandbox:
            log.info("You're in the sandbox...")
            for x in infiles:
                targets.append(_update_status(_select_sandbox_targets_file(x)))
        else:
            for x in infiles:
                targets.append(_update_status(_select_targets_file(x)))

    # ADM it's possible that somebody could pass an arangment of HEALPixels
    # ADM that contain no targets, in which case exit (somewhat) gracefully.
    if targets == []:
        log.warning('ZERO targets for passed file list or region!!!')
        return targets

    targets = np.concatenate(targets)

    # ADM restrict to only targets in a set of HEALPixels, if requested.
    if pixlist is not None:
        ii = is_in_hp(targets, nside, pixlist)
        targets = targets[ii]

    # ADM restrict to only targets in an RA, Dec box, if requested.
    if radecbox is not None:
        ii = is_in_box(targets, radecbox)
        targets = targets[ii]

    # ADM restrict to only targets in an RA, Dec, radius cap, if requested.
    if radecrad is not None:
        ii = is_in_cap(targets, radecrad)
        targets = targets[ii]

    return targets
