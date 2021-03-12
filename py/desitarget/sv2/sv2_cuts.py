"""
desitarget.sv2.sv2_cuts
=======================

Target Selection for DESI Survey Validation derived from `the SV2 wiki`_.

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (*e.g.* LRG, ELG or QSO).

.. _`the Gaia data model`: https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
.. _`the SV2 wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/SV2
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


def isGAIA_STD(ra=None, dec=None, galb=None, gaiaaen=None, pmra=None, pmdec=None,
               parallax=None, parallaxovererror=None, ebv=None, gaiabprpfactor=None,
               gaiasigma5dmax=None, gaiagmag=None, gaiabmag=None, gaiarmag=None,
               gaiadupsource=None, gaiaparamssolved=None,
               primary=None, test=False, nside=2):
    """Standards based solely on Gaia data.

    Parameters
    ----------
    ebv : :class:`array_like`
        E(B-V) values from the SFD dust maps.
    test : :class:`bool`, optional, defaults to ``False``
        If ``True``, then we're running unit tests and don't have to
        find and read every possible Gaia file.
    nside : :class:`int`, optional, defaults to 2
        (NESTED) HEALPix nside, if targets are being parallelized.
        The default of 2 should be benign for serial processing.

    Returns
    -------
    :class:`array_like`
        ``True`` if the object is a bright "GAIA_STD_FAINT" target.
    :class:`array_like`
        ``True`` if the object is a faint "GAIA_STD_BRIGHT" target.
    :class:`array_like`
        ``True`` if the object is a white dwarf "GAIA_STD_WD" target.

    Notes
    -----
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    if primary is None:
        primary = np.ones_like(gaiagmag, dtype='?')

    # ADM restrict all classes to dec >= -30.
    primary &= dec >= -30.
    std = primary.copy()

    # ADM the regular "standards" codes need to know whether something has
    # ADM a Gaia match. Here, everything is a Gaia match.
    gaia = np.ones_like(gaiagmag, dtype='?')

    # ADM determine the Gaia-based white dwarf standards.
    std_wd = isMWS_WD(
        primary=primary, gaia=gaia, galb=galb, astrometricexcessnoise=gaiaaen,
        pmra=pmra, pmdec=pmdec, parallax=parallax,
        parallaxovererror=parallaxovererror, photbprpexcessfactor=gaiabprpfactor,
        astrometricsigma5dmax=gaiasigma5dmax, gaiagmag=gaiagmag,
        gaiabmag=gaiabmag, gaiarmag=gaiarmag
        )

    # ADM apply the Gaia quality cuts for standards.
    std &= isSTD_gaia(primary=primary, gaia=gaia, astrometricexcessnoise=gaiaaen,
                      pmra=pmra, pmdec=pmdec, parallax=parallax,
                      dupsource=gaiadupsource, paramssolved=gaiaparamssolved,
                      gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag)

    # ADM restrict to point sources.
    ispsf = gaia_psflike(gaiaaen, gaiagmag)
    std &= ispsf

    # ADM de-extinct the magnitudes before applying color cuts.
    gd, bd, rd = unextinct_gaia_mags(gaiagmag, gaiabmag, gaiarmag, ebv)

    # ADM apply the Gaia color cuts for standards.
    bprp = bd - rd
    gbp = gd - bd
    std &= bprp > 0.2
    std &= bprp < 0.9
    std &= gbp > -1.*bprp/2.0
    std &= gbp < 0.3-bprp/2.0

    # ADM remove any sources that have neighbors in Gaia within 3.5"...
    # ADM for speed, run only sources for which std is still True.
    log.info("Isolating Gaia-only standards...t={:.1f}s".format(time()-start))
    ii_true = np.where(std)[0]
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
                log.info("Read {}/{} files for Gaia-only standards...t={:.1f}s"
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
        # ADM match the standards to the broader Gaia sources at 3.5".
        matchrad = 3.5*u.arcsec
        cstd = SkyCoord(ra[ii_true]*u.degree, dec[ii_true]*u.degree)
        cgaia = SkyCoord(gaiaobjs["RA"]*u.degree, gaiaobjs["DEC"]*u.degree)
        idstd, idgaia, d2d, _ = cgaia.search_around_sky(cstd, matchrad)
        # ADM remove source matches with d2d=0 (i.e. the source itself!).
        idgaia, idstd = idgaia[d2d > 0], idstd[d2d > 0]
        # ADM remove matches within 5 mags of a Gaia source.
        badmag = (
            (gaiagmag[ii_true][idstd] + 5 > gaiaobjs["PHOT_G_MEAN_MAG"][idgaia]) |
            (gaiarmag[ii_true][idstd] + 5 > gaiaobjs["PHOT_RP_MEAN_MAG"][idgaia]))
        std[ii_true[idstd][badmag]] = False

    # ADM add the brightness cuts in Gaia G-band.
    std_bright = std.copy()
    std_bright &= gaiagmag >= 16
    std_bright &= gaiagmag < 18

    std_faint = std.copy()
    std_faint &= gaiagmag >= 16
    std_faint &= gaiagmag < 19

    return std_faint, std_bright, std_wd


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
    :class:`array_like`
        ``True`` if and only if the object is a very faint "BACKUP"
        target.

    Notes
    -----
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gaiagmag, dtype='?')

    # ADM restrict all classes to dec >= -30.
    primary &= dec >= -30.

    isbackupbright = primary.copy()
    isbackupfaint = primary.copy()
    isbackupveryfaint = primary.copy()

    # ADM determine which sources are close to the Galaxy.
    in_gal = is_in_Galaxy([ra, dec], radec=True)

    # ADM bright targets are 10 < G < 16.
    isbackupbright &= gaiagmag >= 10
    isbackupbright &= gaiagmag < 16

    # ADM faint targets are 16 < G < 18.
    isbackupfaint &= gaiagmag >= 16
    isbackupfaint &= gaiagmag < 18.
    # ADM and are "far from" the Galaxy.
    isbackupfaint &= ~in_gal

    # ADM very faint targets are 18. < G < 19.
    isbackupveryfaint &= gaiagmag >= 18.
    isbackupveryfaint &= gaiagmag < 19
    # ADM and are "far from" the Galaxy.
    isbackupveryfaint &= ~in_gal

    return isbackupbright, isbackupfaint, isbackupveryfaint


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
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
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
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
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


def isSTD_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 primary=None, south=True):
    """Select STD stars based on Legacy Surveys color cuts. Returns a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            Set to ``True`` for objects to initially consider as possible targets.
            Defaults to everything being ``True``.
        south: boolean, defaults to ``True``
            Use color-cuts based on photometry from the "south" (DECaLS) as
            opposed to the "north" (MzLS+BASS).

    Returns:
        mask : boolean array, True if the object has colors like a STD star target

    Notes:
        - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
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
            Set to ``True`` for objects to initially consider as possible targets.
            Defaults to everything being ``True``.
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
        - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
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
          usegaia=True, maskbits=None, south=True):
    """Select STD targets using color cuts and photometric quality cuts (PSF-like
    and fracflux).  See isSTD_colors() for additional info.

    Args:
        gflux, rflux, zflux: array_like
            The flux in nano-maggies of g, r, z bands.
        primary: array_like or None
            Set to ``True`` for objects to initially consider as possible targets.
            Defaults to everything being ``True``.
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
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
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

    # ADM don't target standards in Legacy Surveys mask regions.
    std &= imaging_mask(maskbits, mwsmask=True)

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
        gbright = 16.
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
               pmra=None, pmdec=None, parallax=None, parallaxerr=None,
               obs_rflux=None, objtype=None, gaia=None,
               gaiagmag=None, gaiabmag=None, gaiarmag=None,
               gaiaaen=None, gaiadupsource=None, paramssolved=None,
               primary=None, south=True, maskbits=None):
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
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM currently no difference between N/S for MWS, so easiest
    # ADM just to use one selection
    # if south:
    mws &= notinMWS_main_mask(gaia=gaia, gfracmasked=gfracmasked, gnobs=gnobs,
                              gflux=gflux, rfracmasked=rfracmasked, rnobs=rnobs,
                              rflux=rflux, gaiadupsource=gaiadupsource,
                              primary=primary, maskbits=maskbits)

    # ADM pass the mws that pass cuts as primary, to restrict to the
    # ADM sources that weren't in a mask/logic cut.
    mws, red, blue = isMWS_main_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
        pmra=pmra, pmdec=pmdec, parallax=parallax, parallaxerr=parallaxerr,
        obs_rflux=obs_rflux, objtype=objtype, gaiagmag=gaiagmag,
        gaiabmag=gaiabmag, gaiarmag=gaiarmag, gaiaaen=gaiaaen,
        paramssolved=paramssolved, primary=mws, south=south
    )

    return mws, red, blue


def notinMWS_main_mask(gaia=None, gfracmasked=None, gnobs=None, gflux=None,
                       rfracmasked=None, rnobs=None, rflux=None, maskbits=None,
                       gaiadupsource=None, primary=None):
    """Standard set of masking-based cuts used by MWS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isMWS_main` for parameters).
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM don't target MWS-like targets in Legacy Surveys mask regions.
    mws &= imaging_mask(maskbits, mwsmask=True)

    # ADM apply the mask/logic selection for all MWS-MAIN targets
    # ADM main targets match to a Gaia source
    mws &= gaia
    mws &= (gfracmasked < 0.5) & (gflux > 0) & (gnobs > 0)
    mws &= (rfracmasked < 0.5) & (rflux > 0) & (rnobs > 0)

    mws &= ~gaiadupsource

    return mws


def isMWS_main_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                      pmra=None, pmdec=None, parallax=None, parallaxerr=None,
                      obs_rflux=None, objtype=None, paramssolved=None,
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

    # ADM Turn off any NaNs for astrometric quantities to suppress
    # ADM warnings. Won't target these, using cuts on paramssolved
    # ADM (or will explicitly target them based on paramsssolved).
    ii = paramssolved != 31
    parallax = parallax.copy()
    parallax[ii], pm[ii] = 0., 0.

    # ADM MWS-RED and MWS-BROAD have g-r >= 0.7
    red &= rflux >= gflux * 10**(0.7/2.5)                      # (g-r)>=0.7
    broad = red.copy()

    # ADM MWS-RED also has parallax < max(3parallax_err,1)mas
    # ADM and proper motion < 7
    # ADM and all astrometric parameters are measured.
    red &= parallax < np.maximum(3*parallaxerr, 1)
    red &= pm < 7.
    red &= paramssolved == 31

    # ADM MWS-BROAD has parallax > max(3parallax_err,1)mas
    # ADM OR proper motion > 7.
    # ADM OR astrometric parameters not measured.
    broad &= ((parallax >= np.maximum(3*parallaxerr, 1)) |
              (pm >= 7.)
              | (paramssolved != 31))

    return broad, red, blue


def isMWS_nearby(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 objtype=None, gaia=None, primary=None, paramssolved=None,
                 pmra=None, pmdec=None, parallax=None, parallaxerr=None,
                 obs_rflux=None, gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for NEARBY Milky Way Survey targets.

    Args:
        see :func:`~desitarget.cuts.set_target_bits` for parameters.

    Returns:
        mask : array_like.
            True if and only if the object is a MWS-NEARBY target.

    Notes:
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
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
    # APC Gaia G mag of more than 16
    mws &= gaiagmag > 16.
    # ADM all astrometric parameters are measured.
    mws &= paramssolved == 31
    # ADM parallax cut corresponding to 100pc
    mws &= (parallax + parallaxerr) > 10.   # NB: "+" is correct

    return mws


def isMWS_bhb(primary=None, objtype=None,
              gaia=None, gaiaaen=None, gaiadupsource=None, gaiagmag=None,
              gflux=None, rflux=None, zflux=None,
              w1flux=None, w1snr=None, maskbits=None,
              gnobs=None, rnobs=None, znobs=None,
              gfracmasked=None, rfracmasked=None, zfracmasked=None,
              parallax=None, parallaxerr=None):
    """Set bits for BHB Milky Way Survey targets

    Parameters
    ----------
    see :func:`~desitarget.cuts.set_target_bits` for other parameters.

    Returns
    -------
    mask : array_like.
        True if and only if the object is a MWS-BHB target.

    Notes
    -----
    - Criteria supplied by Sergey Koposov
    - gflux, rflux, zflux, w1flux have been corrected for extinction
      (unlike other MWS selections, which use obs_flux).
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM do not target any objects for which entries are NaN
    # ADM and turn off the NaNs for those entries
    nans = np.isnan(gflux) | np.isnan(rflux) | np.isnan(zflux) | np.isnan(w1flux) | np.isnan(parallax) | np.isnan(gaiagmag)
    w = np.where(nans)[0]
    if len(w) > 0:
        # ADM make copies as we are reassigning values
        rflux, gflux, zflux, w1flux = rflux.copy(), gflux.copy(), zflux.copy(), w1flux.copy()
        parallax = parallax.copy()
        gaigmag = gaiagmag.copy()
        rflux[w], gflux[w], zflux[w], w1flux[w] = 0., 0., 0., 0.
        parallax[w] = 0.
        gaiagmag[w] = 0.
        mws &= ~nans
        log.info('{}/{} NaNs in file...t = {:.1f}s'
                 .format(len(w), len(mws), time()-start))

    gmag = 22.5 - 2.5 * np.log10(gflux.clip(1e-7))
    rmag = 22.5 - 2.5 * np.log10(rflux.clip(1e-7))
    zmag = 22.5 - 2.5 * np.log10(zflux.clip(1e-7))

    gmr = gmag-rmag
    rmz = rmag-zmag

    # ADM don't target MWS-like targets in Legacy Surveys mask regions.
    mws &= imaging_mask(maskbits, mwsmask=True)

    # APC must be a Legacy Surveys object that matches a Gaia source
    mws &= gaia
    # APC type must be PSF
    mws &= _psflike(objtype)
    # APC no sources brighter than Gaia G = 10
    mws &= gaiagmag > 10.
    # APC exclude nearby sources by parallax
    mws &= parallax <= 0.1 + 3*parallaxerr

    mws &= (gfracmasked < 0.5) & (gflux > 0) & (gnobs > 0)
    mws &= (rfracmasked < 0.5) & (rflux > 0) & (rnobs > 0)
    mws &= (zfracmasked < 0.5) & (zflux > 0) & (znobs > 0)

    # APC no gaia duplicated sources
    mws &= ~gaiadupsource
    # APC gaia astrometric excess noise < 3
    mws &= gaiaaen < 3.0

    # APC BHB extinction-corrected color range -0.35 <= gmr <= -0.02
    mws &= (gmr >= -0.35) & (gmr <= -0.02)

    # Coefficients from Sergey Koposov
    bhb_sel = rmz - (1.07163*gmr**5 - 1.42272*gmr**4 + 0.69476*gmr**3 - 0.12911*gmr**2 + 0.66993*gmr - 0.11368)
    mws &= (bhb_sel >= -0.05) & (bhb_sel <= 0.05)

    # APC back out the WISE error = 1/sqrt(ivar) from the SNR = flux*sqrt(ivar)
    w1fluxerr = w1flux/(w1snr.clip(1e-7))
    w1mag_faint = 22.5 - 2.5 * np.log10((w1flux-3*w1fluxerr).clip(1e-7))

    # APC WISE cut (Sergey Koposov)
    mws &= rmag - 2.3*gmr - w1mag_faint < -1.5

    # APC Legacy magnitude limits
    mws &= (rmag >= 16.) & (rmag <= 20.)
    return mws


def isMWS_WD(primary=None, gaia=None, galb=None, astrometricexcessnoise=None,
             pmra=None, pmdec=None, parallax=None, parallaxovererror=None,
             photbprpexcessfactor=None, astrometricsigma5dmax=None,
             gaiagmag=None, gaiabmag=None, gaiarmag=None, paramssolved=None):
    """Set bits for WHITE DWARF Milky Way Survey targets.

    Args:
        see :func:`~desitarget.cuts.set_target_bits` for parameters.

    Returns:
        mask : array_like.
            True if and only if the object is a MWS-WD target.

    Notes:
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM do not target any objects for which entries are NaN
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
        mws &= ~nans
#        log.info('{}/{} NaNs in file...t = {:.1f}s'
#                 .format(len(w), len(mws), time()-start))

    # ADM apply the selection for all MWS-WD targets
    # ADM must be a Legacy Surveys object that matches a Gaia source
    mws &= gaia

    # ADM and all astrometric parameters are measured.
    mws &= paramssolved == 31

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


def isMWSSTAR_colors(gflux=None, rflux=None, zflux=None,
                     w1flux=None, w2flux=None, primary=None, south=True):
    """A reasonable range of g-r for MWS targets. Returns a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            Set to ``True`` for objects to initially consider as possible targets.
            Defaults to everything being ``True``.
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
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
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


def isQSO_cuts(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
               w1snr=None, w2snr=None, deltaChi2=None, maskbits=None,
               gnobs=None, rnobs=None, znobs=None,
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
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """

    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                       w1flux=w1flux, w2flux=w2flux,
                       optical=optical, south=south)

    if south:
        qso &= w1snr > 4
        qso &= w2snr > 2
    else:
        qso &= w1snr > 4
        qso &= w2snr > 3

    # ADM observed in every band.
    qso &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    # ADM default to RELEASE of 6000 if nothing is passed.
    if release is None:
        release = np.zeros_like(gflux, dtype='?')+6000

    qso &= ((deltaChi2 > 40.) | (release >= 5000))

    if primary is not None:
        qso &= primary

    if objtype is not None:
        qso &= _psflike(objtype)

    # ADM default mask bits from the Legacy Surveys not set.
    qso &= imaging_mask(maskbits)

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
    qso &= zflux > rflux * 10**(-0.4/2.5)   # (r-z)>-0.4
    qso &= zflux < rflux * 10**(1.1/2.5)    # (r-z)<1.1

    if not optical:
        if south:
            qso &= w2flux > w1flux * 10**(-0.4/2.5)                   # (W1-W2)>-0.4
        else:
            qso &= w2flux > w1flux * 10**(-0.3/2.5)                   # (W1-W2)>-0.3
        qso &= wflux * gflux > zflux * grzflux * 10**(-1.0/2.5)   # (grz-W)>(g-z)-1.0

    # Harder cut on stellar contamination
    mainseq = rflux > gflux * 10**(0.20/2.5)  # g-r>0.2

    # Clip to avoid warnings from negative numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    mainseq &= rflux**(1+1.53) > gflux * zflux**1.53 * 10**((-0.100+0.20)/2.5)
    mainseq &= rflux**(1+1.53) < gflux * zflux**1.53 * 10**((+0.100+0.20)/2.5)
    if not optical:
        mainseq &= w2flux < w1flux * 10**(0.3/2.5)
    qso &= ~mainseq

    return qso


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
    - Current version (03/09/21) is version 1 on `the SV2 wiki`_.
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

    # For the last version of QSO target selection we add a cut on the Band W1 and W2!
    limitInf = 1.e-04
    w1flux, w2flux = w1flux.clip(limitInf), w2flux.clip(limitInf)
    W1 = np.atleast_1d(np.where(w1flux > limitInf, 22.5-2.5*np.log10(w1flux), 0.))
    W2 = np.atleast_1d(np.where(w2flux > limitInf, 22.5-2.5*np.log10(w2flux), 0.))
    W1_cut, W2_cut = 22.3, 22.3

    # Preselection to speed up the process
    rMax = 23.0   # r < 23.0
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
        W1_Reduced, W2_Reduced = W1[preSelection], W2[preSelection]
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
        rf_DR9_fileName = pathToRF + '/rf_model_dr9_final.npz'

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
            # rf loading
            rf.loadForest(rf_DR9_fileName)
            # Compute rf probabilities
            tmp_rf_proba = rf.predict_proba()
            # Compute optimized proba cut
            tmp_r_Reduced = r_Reduced[tmpReleaseOK]
            tmp_W1_Reduced, tmp_W2_Reduced = W1_Reduced[tmpReleaseOK], W2_Reduced[tmpReleaseOK]

            # NO SECOND RF ! ==> keep the structure but no impact on the selection
            tmp_rf_HighZ_proba, pcut_HighZ = np.zeros(tmp_r_Reduced.size), 1.1

            if not south:
                # threshold selection for North footprint.
                pcut = 0.88 - 0.04*np.tanh(tmp_r_Reduced - 20.5)
            else:
                pcut = np.ones(tmp_rf_proba.size)
                is_des = (gnobs[preSelection][tmpReleaseOK] > 4) &\
                         (rnobs[preSelection][tmpReleaseOK] > 4) &\
                         (znobs[preSelection][tmpReleaseOK] > 4) &\
                         ((ra[preSelection][tmpReleaseOK] >= 320) | (ra[preSelection][tmpReleaseOK] <= 100)) &\
                         (dec[preSelection][tmpReleaseOK] <= 10)
                # threshold selection for DES footprint.
                pcut[is_des] = 0.7 - 0.05*np.tanh(tmp_r_Reduced[is_des] - 20.5)
                # threshold selection for South footprint.
                pcut[~is_des] = 0.84 - 0.04*np.tanh(tmp_r_Reduced[~is_des] - 20.5)

            # Add rf proba test result to "qso" mask
            qso[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_proba >= pcut) & (tmp_W1_Reduced <= W1_cut) & (tmp_W2_Reduced <= W2_cut)
            # (NO IMPACT) ADM populate a mask specific to the "HighZ" selection.
            qsohiz[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_HighZ_proba >= pcut_HighZ)
            # ADM store the probabilities in case they need returned.
            pqso[colorsReducedIndex[tmpReleaseOK]] = tmp_rf_proba
            # (NO IMPACT) ADM populate a mask specific to the "HighZ" selection.
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


def set_target_bits(photsys_north, photsys_south, obs_rflux,
                    gflux, rflux, zflux, w1flux, w2flux,
                    gfiberflux, rfiberflux, zfiberflux, objtype, release,
                    ra, dec, gfluxivar, rfluxivar, zfluxivar, w1fluxivar,
                    gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,
                    gfracmasked, rfracmasked, zfracmasked,
                    gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,
                    gsnr, rsnr, zsnr, w1snr, w2snr, deltaChi2, dchisq,
                    gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr,
                    gaiagmag, gaiabmag, gaiarmag, gaiaaen, gaiadupsource,
                    gaiaparamssolved, gaiabprpfactor, gaiasigma5dmax, galb,
                    tcnames, qso_optical_cuts, qso_selection,
                    maskbits, Grr, refcat, primary, resolvetargs=True):
    """Perform target selection on parameters, return target mask arrays.

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
    gfiberflux, rfiberflux, zfiberflux : :class:`~numpy.ndarray`
        Predicted fiber flux in 1 arcsecond seeing in g/r/z-band.
        Corrected for Galactic extinction.
    objtype, release : :class:`~numpy.ndarray`
        `The Legacy Surveys`_ imaging ``TYPE`` and ``RELEASE`` columns.
    gfluxivar, rfluxivar, zfluxivar, w1fluxivar: :class:`~numpy.ndarray`
        The flux inverse variances in g, r, z and W1 bands.
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
        Gaia-based measures of Astrometric Excess Noise, whether the
        source is a duplicate, and how many parameters were solved for.
    gaiabprpfactor, gaiasigma5dmax: :class:`~numpy.ndarray`
        Gaia_based BP/RP excess factor and longest semi-major axis
        of 5-d error ellipsoid.
    galb: :class:`~numpy.ndarray`
        Galactic latitude (degrees).
    tcnames : :class:`list`, defaults to running all target classes
        A list of strings, e.g. ['QSO','LRG']. If passed, process only
        only those specific target classes. A useful speed-up for tests.
        Options include ["ELG", "QSO", "LRG", "MWS", "BGS", "STD"].
    qso_optical_cuts : :class:`boolean` defaults to ``False``
        Apply just optical color-cuts when selecting QSOs with
        ``qso_selection="colorcuts"``.
    qso_selection : :class:`str`, optional, defaults to `'randomforest'`
        The algorithm to use for QSO selection; valid options are
        `'colorcuts'` and `'randomforest'`
    maskbits: boolean array_like or None
        General `Legacy Surveys mask`_ bits.
    Grr: array_like or None
        Gaia G band magnitude minus observational r magnitude.
    primary : :class:`~numpy.ndarray`
        ``True`` for objects that should be considered when setting bits.
    survey : :class:`str`, defaults to ``'main'``
        Specifies which target masks yaml file and target selection cuts
        to use. Options are ``'main'`` and ``'svX``' (X is 1, 2, etc.)
        for the main survey and different iterations of SV, respectively.
    resolvetargs : :class:`boolean`, optional, defaults to ``True``
        If ``True``, if only northern (southern) sources are passed then
        only apply the northern (southern) cuts to those sources.
    ra, dec : :class:`~numpy.ndarray`
        The Ra, Dec position of objects

    Returns
    -------
    :class:`~numpy.ndarray`
        (desi_target, bgs_target, mws_target) where each element is
        an ndarray of target selection bitmask flags for each object.

    Notes
    -----
    - Gaia quantities have units as for `the Gaia data model`_.
    """

    from desitarget.targetmask import desi_mask, bgs_mask, mws_mask

    # ADM if resolvetargs is set, limit to only sending north/south objects
    # ADM through north/south cuts.
    south_cuts = [False, True]
    if resolvetargs:
        # ADM if only southern objects were sent this will be [True], if
        # ADM only northern it will be [False], else it wil be both.
        south_cuts = list(set(np.atleast_1d(photsys_south)))

    # ADM default for target classes we WON'T process is all False.
    tcfalse = primary & False

    # ADM initially set everything to arrays of False for the LRG selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    lrg_classes = [tcfalse, tcfalse]
    if "LRG" in tcnames:
        for south in south_cuts:
            lrg_classes[int(south)] = isLRG(
                primary=primary,
                gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                zfiberflux=zfiberflux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                rfluxivar=rfluxivar, zfluxivar=zfluxivar, w1fluxivar=w1fluxivar,
                maskbits=maskbits, south=south
            )
    lrg_north, lrg_south = lrg_classes

    # ADM combine LRG target bits for an LRG target based on any imaging.
    lrg = (lrg_north & photsys_north) | (lrg_south & photsys_south)

    # ADM initially set everything to arrays of False for the ELG selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    elg_classes = [tcfalse, tcfalse]
    if "ELG" in tcnames:
        for south in south_cuts:
            elg_classes[int(south)] = isELG(
                primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
                gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits,
                south=south
            )
    elg_north, elg_south = elg_classes

    # ADM combine ELG target bits for an ELG target based on any imaging.
    elg = (elg_north & photsys_north) | (elg_south & photsys_south)

    # ADM initially set everything to arrays of False for the QSO selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    qso_classes = [[tcfalse, tcfalse], [tcfalse, tcfalse]]
    if "QSO" in tcnames:
        for south in south_cuts:
            if qso_selection == 'colorcuts':
                # ADM determine quasar targets in the north and the south separately
                # ADM the [0] here is critical as isQSO_cuts only returns one bit
                # ADM and the other bit (which is the "high-z" bit from the Random
                # ADM Forest needs to be set to all "False".
                qso_classes[int(south)][0] = isQSO_cuts(
                    primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                    w1flux=w1flux, w2flux=w2flux,
                    deltaChi2=deltaChi2, maskbits=maskbits,
                    gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                    objtype=objtype, w1snr=w1snr, w2snr=w2snr, release=release,
                    optical=qso_optical_cuts, south=south
                )
            elif qso_selection == 'randomforest':
                # ADM determine quasar targets in the north and the south separately
                qso_classes[int(south)] = isQSO_randomforest(
                    primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                    w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2,
                    maskbits=maskbits, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                    objtype=objtype, release=release, ra=ra, dec=dec, south=south
                )
            else:
                raise ValueError('Unknown qso_selection {}; valid options are {}'.format(
                    qso_selection, qso_selection_options))
    qso_north, qso_hiz_north = qso_classes[0]
    qso_south, qso_hiz_south = qso_classes[1]

    # ADM combine QSO targeting bits for a QSO selected in any imaging.
    qso = (qso_north & photsys_north) | (qso_south & photsys_south)
    qsohiz = (qso_hiz_north & photsys_north) | (qso_hiz_south & photsys_south)

    # ADM initially set everything to arrays of False for the BGS selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    bgs_classes = [[tcfalse, tcfalse, tcfalse], [tcfalse, tcfalse, tcfalse]]
    # ADM set the BGS bits
    if "BGS" in tcnames:
        for south in south_cuts:
            bgs_store = []
            for targtype in ["bright", "faint", "wise"]:
                bgs_store.append(
                    isBGS(
                        rfiberflux=rfiberflux, gflux=gflux, rflux=rflux, zflux=zflux,
                        w1flux=w1flux, w2flux=w2flux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                        gfracmasked=gfracmasked, rfracmasked=rfracmasked, zfracmasked=zfracmasked,
                        gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                        gfracin=gfracin, rfracin=rfracin, zfracin=zfracin,
                        gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
                        maskbits=maskbits, Grr=Grr, refcat=refcat, w1snr=w1snr, gaiagmag=gaiagmag,
                        objtype=objtype, primary=primary, south=south, targtype=targtype
                    )
                )
            bgs_classes[int(south)] = bgs_store
    bgs_bright_north, bgs_faint_north, bgs_wise_north = bgs_classes[0]
    bgs_bright_south, bgs_faint_south, bgs_wise_south = bgs_classes[1]

    # ADM combine BGS targeting bits for a BGS selected in any imaging.
    bgs_bright = (bgs_bright_north & photsys_north) | (bgs_bright_south & photsys_south)
    bgs_faint = (bgs_faint_north & photsys_north) | (bgs_faint_south & photsys_south)
    bgs_wise = (bgs_wise_north & photsys_north) | (bgs_wise_south & photsys_south)

    # ADM 10% of the BGS_FAINT sources need the BGS_FAINT_HIP bit set.
    # ADM form a seed using RA/Dec in case we parallelized by HEALPixel.
    # SJB seeds must be within 0 - 2**32-1
    # SJB np1.18 scalar vs. vector support, but note that HIP won't be
    #     set identically for vector vs. calling scalar N times.
    uniqseed = int(np.mean(zflux)*1e5) % (2**32 - 1)
    np.random.seed(uniqseed)
    hip = None
    if np.isscalar(bgs_faint):
        if bgs_faint:
            nbgsf = 1
            hip = np.random.uniform(0, 1) < 0.1
    else:
        w = np.where(bgs_faint)[0]
        nbgsf = len(w)
        if nbgsf > 0:
            hip = np.random.choice(w, nbgsf//10, replace=False)

    # ADM initially set everything to arrays of False for the MWS selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    mws_classes = [[tcfalse, tcfalse, tcfalse], [tcfalse, tcfalse, tcfalse]]
    mws_nearby = tcfalse
    mws_bhb = tcfalse
    if "MWS" in tcnames:
        mws_nearby = isMWS_nearby(
            gaia=gaia, gaiagmag=gaiagmag, parallax=parallax,
            parallaxerr=parallaxerr, paramssolved=gaiaparamssolved
        )

        mws_bhb = isMWS_bhb(
                    primary=primary,
                    objtype=objtype,
                    gaia=gaia, gaiaaen=gaiaaen, gaiadupsource=gaiadupsource, gaiagmag=gaiagmag,
                    gflux=gflux, rflux=rflux, zflux=zflux,
                    w1flux=w1flux, w1snr=w1snr,
                    gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                    gfracmasked=gfracmasked, rfracmasked=rfracmasked, zfracmasked=zfracmasked,
                    parallax=parallax, parallaxerr=parallaxerr, maskbits=maskbits
             )

        # ADM run the MWS target types for (potentially) both north and south.
        for south in south_cuts:
            mws_classes[int(south)] = isMWS_main(
                    gaia=gaia, gaiaaen=gaiaaen, gaiadupsource=gaiadupsource,
                    gflux=gflux, rflux=rflux, obs_rflux=obs_rflux, objtype=objtype,
                    gnobs=gnobs, rnobs=rnobs, gfracmasked=gfracmasked,
                    rfracmasked=rfracmasked, pmra=pmra, pmdec=pmdec,
                    parallax=parallax, parallaxerr=parallaxerr, maskbits=maskbits,
                    paramssolved=gaiaparamssolved, primary=primary, south=south
            )
    mws_broad_n, mws_red_n, mws_blue_n = mws_classes[0]
    mws_broad_s, mws_red_s, mws_blue_s = mws_classes[1]

    # ADM treat the MWS WD selection specially, as we have to run the
    # ADM white dwarfs for standards and MWS science targets.
    mws_wd = tcfalse
    if "MWS" in tcnames or "STD" in tcnames:
        mws_wd = isMWS_WD(
            gaia=gaia, galb=galb, astrometricexcessnoise=gaiaaen,
            pmra=pmra, pmdec=pmdec, parallax=parallax,
            parallaxovererror=parallaxovererror, paramssolved=gaiaparamssolved,
            photbprpexcessfactor=gaiabprpfactor, astrometricsigma5dmax=gaiasigma5dmax,
            gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag
        )

    # ADM initially set everything to False for the standards.
    std_faint, std_bright, std_wd = tcfalse, tcfalse, tcfalse
    if "STD" in tcnames:
        # ADM run the MWS_MAIN target types for both faint and bright.
        # ADM Make sure to pass all of the needed columns! At one point we stopped
        # ADM passing objtype, which meant no standards were being returned.
        std_classes = []
        for bright in [False, True]:
            std_classes.append(
                isSTD(
                    primary=primary, zflux=zflux, rflux=rflux, gflux=gflux, maskbits=maskbits,
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
        # ADM the standard WDs are currently identical to the MWS WDs.
        std_wd = mws_wd

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
    desi_target |= qsohiz * desi_mask.QSO_HIZ

    # ADM Standards.
    desi_target |= std_faint * desi_mask.STD_FAINT
    desi_target |= std_bright * desi_mask.STD_BRIGHT
    desi_target |= std_wd * desi_mask.STD_WD

    # BGS targets, south.
    bgs_target = bgs_bright_south * bgs_mask.BGS_BRIGHT_SOUTH
    bgs_target |= bgs_faint_south * bgs_mask.BGS_FAINT_SOUTH
    # ADM turn off BGS_WISE until we're sure we'll use it.
    # bgs_target |= bgs_wise_south * bgs_mask.BGS_WISE_SOUTH

    # BGS targets, north.
    bgs_target |= bgs_bright_north * bgs_mask.BGS_BRIGHT_NORTH
    bgs_target |= bgs_faint_north * bgs_mask.BGS_FAINT_NORTH
    # ADM turn off BGS_WISE until we're sure we'll use it.
    # bgs_target |= bgs_wise_north * bgs_mask.BGS_WISE_NORTH

    # BGS targets, combined.
    bgs_target |= bgs_bright * bgs_mask.BGS_BRIGHT
    bgs_target |= bgs_faint * bgs_mask.BGS_FAINT
    # ADM turn off BGS_WISE until we're sure we'll use it.
    # bgs_target |= bgs_wise * bgs_mask.BGS_WISE
    # ADM set 10% of the BGS_FAINT targets to BGS_FAINT_HIP.
    if hip is not None:
        if hip is True:
            bgs_target |= bgs_mask.BGS_FAINT_HIP
        else:
            bgs_target[hip] |= bgs_mask.BGS_FAINT_HIP

    # ADM MWS main, nearby, and WD.
    mws_target = mws_broad * mws_mask.MWS_BROAD
    mws_target |= mws_wd * mws_mask.MWS_WD
    mws_target |= mws_nearby * mws_mask.MWS_NEARBY
    mws_target |= mws_bhb * mws_mask.MWS_BHB

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
