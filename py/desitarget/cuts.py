"""
desitarget.cuts
===============

Target Selection for Legacy Surveys catalogue data derived from `the wiki`_.

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (*e.g.* LRG, ELG or QSO).

.. _`the Gaia data model`: https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
.. _`the Legacy Surveys`: http://www.legacysurvey.org/
.. _`the wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection
.. _`the SV3 wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/SV3
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


def MWS_too_bright(gaiagmag=None, zfibertotflux=None):
    """Whether a target is too bright to include for MWS observations.

    Parameters
    ----------
    gaiagmag : :class:`~numpy.ndarray`
            Gaia-based g-band MAGNITUDE.
    zfibertotflux : :class:`~numpy.ndarray`
        Predicted fiber flux from ALL sources at object's location in 1
        arcsecond seeing in z. NOT corrected for Galactic extinction.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is FAINTER than the MWS
        bright-cut limits.

    Notes
    -----
    - Current version (04/02/21) is version 17 on `the SV3 wiki`_.
    """
    # ADM set up an array to store objects that is all True.
    too_bright = np.ones_like(gaiagmag, dtype='?')

    # ADM True if Gaia G is too bright.
    # ADM remember that gaiagmag of 0 corresponds to missing sources.
    too_bright &= (gaiagmag < 15) & (gaiagmag != 0)

    # ADM or True if the Legacy Surveys zfibertot is too bright.
    # ADM remember that zflux of 0 corresponds to missing sources.
    zmag = 22.5-2.5*np.log10(zfibertotflux.clip(1e-7))
    too_bright |= (zmag < 15) & (zfibertotflux != 0)

    return too_bright


def random_fraction_of_trues(fraction, bool_array):
    """Return True for a random subset of array entries that are True.

    Parameters
    ----------
    fraction : :class:`float`
        The fraction of the True entries to retain as True. Should be
        between 0 and 1.
    bool_array : :class:`array_like` or `bool`
        A boolean array, or scalar.

    Returns
    -------
    :class:`array_like`
        The original `bool_array`, with a random `fraction` of ``True``
        entries retained as ``True`` and the others set to ``False``.
        If a scalar is passed, then a scalar is returned.
    """
    # ADM check that the input fraction was between 0 and 1.
    if not 0 <= fraction <= 1:
        msg = "fraction must be between 0 and 1, not {}".format(fraction)
        log.critical(msg)
        raise ValueError(msg)

    if np.isscalar(bool_array):
        # ADM catch the corner case that a scalar was passed.
        chosen = np.random.uniform(0, 1) < fraction
    else:
        # ADM create a random array with the correct fraction of Trues.
        chosen = np.random.random(len(bool_array)) < fraction

    # ADM return True for the subset in bool_array that was chosen.
    return bool_array & chosen


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
    - Update for DR9 https://desi.lbl.gov/trac/attachment/wiki/TargetSelectionWG/TargetSelection/North_vs_South_dr9.png
    """
    gshift = gflux * 10**(-0.4*0.004) * (gflux/rflux)**(-0.059)
    rshift = rflux * 10**(0.4*0.003) * (rflux/zflux)**(-0.024)
    zshift = zflux * 10**(0.4*0.013) * (rflux/zflux)**(+0.015)

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
    - Current version (01/28/21) is version 244 on `the wiki`_.
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
        gaiabmag=gaiabmag, gaiarmag=gaiarmag, paramssolved=gaiaparamssolved
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


def isBACKUP(ra=None, dec=None,
             gaiagmag=None, gaiabmag=None, gaiarmag=None,
             parallax=None, parallaxerr=None,
             primary=None):
    """BACKUP targets based on Gaia magnitudes.

    Parameters
    ----------
    ra, dec: :class:`array_like` or :class:`None`
        Right Ascension and Declination in degrees.
    gaiagmag, gaiabmag, gaiarmag: :class:`array_like` or :class:`None`
            Gaia-based g, b and  r MAGNITUDES (not corrected for Galactic
            extinction (same units as `the Gaia data model`_).
    parallax, parallaxerr: :class:`array_like` or :class:`None`
            Gaia parallax and error (same units as `the Gaia data model`_)
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
    :class:`array_like`
        ``True`` if and only if the object is a gaia-selected giant
        "BACKUP" target.

    Notes
    -----
    - Current version (10/22/21) is version 273 on `the wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gaiagmag, dtype='?')

    # APC In this case the BP-RP relations use the fluxes without
    # APC any correction for extinction, by design.
    bprp = gaiabmag - gaiarmag

    # ADM restrict all classes to dec >= -30.
    primary &= dec >= -30.
    # APC require measured  gaia color.
    primary &= ~np.isnan(bprp)
    # APC hard bright limits
    # See https://github.com/desihub/desitarget/pull/766
    primary &= gaiagmag >= 11.2
    primary &= gaiabmag >= 11.2
    primary &= gaiarmag >= 11.2

    isbackupbright = primary.copy()
    isbackupfaint = primary.copy()
    isbackupveryfaint = primary.copy()
    isbackupgiant = primary.copy()

    # ADM determine which sources are close to the Galaxy.
    in_gal = is_in_Galaxy([ra, dec], radec=True)

    # APC bright targets are 11.2 + 0.6(BP-RP) < G < 16.
    isbackupbright &= gaiagmag >= 11.2 + 0.6*bprp
    isbackupbright &= gaiagmag < 16.0

    # APC Giant candidates have low parallax
    is_gaiagiant = parallax < (3 * parallaxerr + 0.1)

    # APC giant targets are min(17.5 + 0.6(BP-RP), 19) < G < 16
    isbackupgiant &= gaiagmag >= 16.0
    isbackupgiant &= gaiagmag < np.minimum(17.5 + 0.6*bprp, 19)
    # APC and are likely giants
    isbackupgiant &= is_gaiagiant

    # APC faint targets are 16 < G < 18
    isbackupfaint &= gaiagmag >= 16.0
    isbackupfaint &= gaiagmag < 18.0
    # APC and are not halo giant candidates
    isbackupfaint &= ~is_gaiagiant
    # ADM and are "far from" the Galaxy.
    isbackupfaint &= ~in_gal

    # ADM very faint targets are 18. < G < 19.
    isbackupveryfaint &= gaiagmag >= 18.
    isbackupveryfaint &= gaiagmag < 19
    # APC and are not halo giant candidates
    isbackupveryfaint &= ~is_gaiagiant
    # ADM and are "far from" the Galaxy.
    isbackupveryfaint &= ~in_gal

    return isbackupbright, isbackupfaint, isbackupveryfaint, isbackupgiant


def isLRG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          zfiberflux=None, rfluxivar=None, zfluxivar=None, w1fluxivar=None,
          gaiagmag=None, gnobs=None, rnobs=None, znobs=None, maskbits=None,
          zfibertotflux=None, primary=None, south=True):
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
    - Current version (05/07/21) is version 260 on `the wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    # ADM LRG targets.
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg_quality = primary.copy()

    # ADM basic quality cuts.
    lrg_quality &= notinLRG_mask(
        primary=primary, rflux=rflux, zflux=zflux, w1flux=w1flux,
        zfiberflux=zfiberflux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
        rfluxivar=rfluxivar, zfluxivar=zfluxivar, w1fluxivar=w1fluxivar,
        gaiagmag=gaiagmag, maskbits=maskbits, zfibertotflux=zfibertotflux
    )

    # ADM color-based selection of LRGs.
    lrg = isLRG_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
        zfiberflux=zfiberflux, south=south, primary=primary
    )

    lrg &= lrg_quality

    return lrg


def notinLRG_mask(primary=None, rflux=None, zflux=None, w1flux=None,
                  zfiberflux=None, gnobs=None, rnobs=None, znobs=None,
                  rfluxivar=None, zfluxivar=None, w1fluxivar=None,
                  gaiagmag=None, maskbits=None, zfibertotflux=None):
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

    lrg &= (gaiagmag == 0) | (gaiagmag > 18)  # remove bright GAIA sources

    # ADM remove stars with zfibertot < 17.5 that are missing from GAIA.
    lrg &= zfibertotflux < 10**(-0.4*(17.5-22.5))

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
        lrg &= zmag - w1mag > 0.8 * (rmag - zmag) - 0.6  # non-stellar cut
        lrg &= zfibermag < 21.6                   # faint limit
        lrg &= (gmag - w1mag > 2.9) | (rmag - w1mag > 1.8)  # low-z cuts
        lrg &= (
            ((rmag - w1mag > (w1mag - 17.14) * 1.8)
             & (rmag - w1mag > (w1mag - 16.33) * 1.))
            | (rmag - w1mag > 3.3)
        )  # double sliding cuts and high-z extension
    else:
        lrg &= zmag - w1mag > 0.8 * (rmag - zmag) - 0.6  # non-stellar cut
        lrg &= zfibermag < 21.61                   # faint limit
        lrg &= (gmag - w1mag > 2.97) | (rmag - w1mag > 1.8)  # low-z cuts
        lrg &= (
            ((rmag - w1mag > (w1mag - 17.13) * 1.83)
             & (rmag - w1mag > (w1mag - 16.31) * 1.))
            | (rmag - w1mag > 3.4)
        )  # double sliding cuts and high-z extension

    return lrg


def isELG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          gfiberflux=None, gsnr=None, rsnr=None, zsnr=None,
          gnobs=None, rnobs=None, znobs=None,
          maskbits=None, south=True, primary=None):
    """Definition of ELG target classes. Returns a boolean array.
    (see :func:`~desitarget.cuts.set_target_bits` for parameters).

    Notes:
    - Current version (03/27/21) is version 8 on `the SV3 wiki`_.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    nomask = notinELG_mask(
        maskbits=maskbits, gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
        gnobs=gnobs, rnobs=rnobs, znobs=znobs, primary=primary)

    elgvlo, elg = isELG_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                               w1flux=w1flux, w2flux=w2flux,
                               gfiberflux=gfiberflux, south=south,
                               primary=primary)

    return elgvlo & nomask, elg & nomask


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
                 w2flux=None, gfiberflux=None, south=True, primary=None):
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
    gfib = 22.5 - 2.5*np.log10(gfiberflux.clip(1e-16))

    # ADM cuts shared by the northern and southern selections.
    elg &= g > 20                       # bright cut.
    elg &= r - z > 0.15                  # blue cut.
#    elg &= r - z < 1.6                  # red cut.

    # ADM cuts that are unique to the north or south. Identical for sv3
    # ADM but keep the north/south formalism in case we use it later.
    if south:
        elg &= gfib < 24.1  # faint cut.
        elg &= g - r < 0.5*(r - z) + 0.1  # remove stars, low-z galaxies.
    else:
        elg &= gfib < 24.1  # faint cut.
        elg &= g - r < 0.5*(r - z) + 0.1  # remove stars, low-z galaxies.

    # ADM separate a low-priority and a regular sample.
    elgvlo = elg.copy()

    # ADM low-priority OII flux cut.
    elgvlo &= g - r < -1.2*(r - z) + 1.6
    elgvlo &= g - r >= -1.2*(r - z) + 1.3

    # ADM high-priority OII flux cut.
    elg &= g - r < -1.2*(r - z) + 1.3

    return elgvlo, elg


def isSTD_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 primary=None, south=True):
    """Select STD stars based on Legacy Surveys. Returns a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            Set to ``True`` for objects to consider as possible STDs.
            Defaults to everything being ``True``.
        south: boolean, defaults to ``True``
            Use color-cuts based on photometry from the "south" (DECaLS)
            as opposed to the "north" (MzLS+BASS).

    Returns:
        mask : boolean array, True if the object has colors like a STD
        star target.

    Notes:
        - Current version (08/01/18) is version 121 on `the wiki`_.
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
            Set to ``True`` for objects to consider as possible STDs.
            Defaults to everything being ``True``.
        gaia: boolean array_like or None
            True if there is a match between this object in
            `the Legacy Surveys`_ and in Gaia.
        astrometricexcessnoise: array_like or None
            Excess noise of the source in Gaia
        pmra, pmdec, parallax: array_like or None
            Gaia-based proper motion in RA and Dec and parallax
            (same units as the Gaia data model).
        dupsource: array_like or None
            Whether the source is a duplicate in Gaia.
        paramssolved: array_like or None
            How many parameters were solved for in Gaia.
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            Gaia-based g, b and  r MAGNITUDES (not corrected for Galactic
            extinction (same units as `the Gaia data model`_).

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
    std &= paramssolved >= 31

    # ADM finite proper motions.
    std &= np.isfinite(pmra)
    std &= np.isfinite(pmdec)

    # ADM a parallax smaller than 1 mas.
    std &= parallax < 1.

    # ADM calculate the overall proper motion magnitude
    pm = np.sqrt(pmra**2. + pmdec**2.)

    # ADM a proper motion larger than 2 mas/yr.
    std &= pm > 2.

    # ADM fail if dupsource is not Boolean, as was the case for the 7.0
    # ADM sweeps, otherwise logic checks on dupsource will be misleading.
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
    """Select STD targets using color, photometric quality cuts (PSF-like
    and fracflux).  See isSTD_colors() for additional info.

    Args:
        gflux, rflux, zflux: array_like
            The flux in nano-maggies of g, r, z bands.
        primary: array_like or None
            ``True`` for possible targets. Defaults to ``True`` for all.
        gfracflux, rfracflux, zfracflux: array_like
            Profile-weighted fraction of the flux from other sources
            divided by the total flux in g, r and z bands.
        gfracmasked, rfracmasked, zfracmasked: array_like
            Fraction of masked pixels in the g, r and z bands.
        gnobs, rnobs, znobs: array_like
            Number of observations (in the central pixel) in g, r and z.
        gfluxivar, rfluxivar, zfluxivar: array_like
            The flux inverse variances in g, r, and z bands.
        objtype: array_like or None
            TYPE column of the catalogue to restrict to point sources.
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
           if ``True`` apply mag cuts for "bright" conditions; otherwise,
           for "dark" conditions. Cut is performed on `gaiagmag`.
        usegaia: boolean, defaults to ``True``
           if ``True`` then call :func:`~desitarget.cuts.isSTD_gaia` to
           set the logic cuts. If Gaia is not available (perhaps if
           you're using mocks) then send ``False``, in which case we use
           the LS r-band as a proxy for Gaia G. ``False`` ignores that we
           have already corrected for Galactic extinction (incorrectly).
        south: boolean, defaults to ``True``
            Use color-cuts based on photometry from the "south" (DECaLS)
            as opposed to the "north" (MzLS+BASS).

    Returns:
        mask : boolean array, True if the object has colors like a STD.

    Notes:
    - Gaia-based quantities are as in `the Gaia data model`_.
    - Current version (02/18/21) is version 246 on `the wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    std = primary.copy()

    # ADM apply the Legacy Surveys (optical) magnitude and color cuts.
    std &= isSTD_colors(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                        south=south)

    # ADM apply the Gaia quality cuts.
    if usegaia:
        std &= isSTD_gaia(primary=primary, gaia=gaia,
                          astrometricexcessnoise=astrometricexcessnoise,
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
            ``True`` if the object is a ``MWS_BROAD`` target.
        mask2 : array_like.
            ``True`` if the object is a ``MWS_MAIN_RED`` target.
        mask3 : array_like.
            ``True`` if the object is a ``MWS_MAIN_BLUE`` target.
        mask4 : array_like.
            ``True`` if the object is a ``MWS_FAINT_RED`` target.
        mask5 : array_like.
            ``True`` if the object is a ``MWS_FAINT_BLUE`` target.

    Notes:
    - Current version (05/04/21) is version 256 on `the wiki`_.
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

    # APC Copy cuts up to this point for later use in faint selection
    mws_faint = mws.copy()

    # ADM pass the mws that pass cuts as primary, to restrict to the
    # ADM sources that weren't in a mask/logic cut.
    mws, red, blue = isMWS_main_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
        pmra=pmra, pmdec=pmdec, parallax=parallax, parallaxerr=parallaxerr,
        obs_rflux=obs_rflux, objtype=objtype, gaiagmag=gaiagmag,
        gaiabmag=gaiabmag, gaiarmag=gaiarmag, gaiaaen=gaiaaen,
        paramssolved=paramssolved, primary=mws, south=south
    )

    faint_red, faint_blue = isMWS_faint_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
        pmra=pmra, pmdec=pmdec, parallax=parallax, parallaxerr=parallaxerr,
        obs_rflux=obs_rflux, objtype=objtype, gaiagmag=gaiagmag,
        gaiabmag=gaiabmag, gaiarmag=gaiarmag, gaiaaen=gaiaaen,
        paramssolved=paramssolved, primary=mws_faint, south=south
    )

    return mws, red, blue, faint_red, faint_blue


def notinMWS_main_mask(gaia=None, gfracmasked=None, gnobs=None, gflux=None,
                       rfracmasked=None, rnobs=None, rflux=None, maskbits=None,
                       gaiadupsource=None, primary=None):
    """Standard set of masking-based cuts used by MWS target classes
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


def isMWS_faint_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                       pmra=None, pmdec=None, parallax=None, parallaxerr=None,
                       obs_rflux=None, objtype=None, paramssolved=None,
                       gaiagmag=None, gaiabmag=None, gaiarmag=None, gaiaaen=None,
                       primary=None, south=True):
    """Set of cuts to define a fainter extension to the MWS main sample.
    (see, e.g., :func:`~desitarget.cuts.isMWS_main` for parameters).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    faint = primary.copy()

    # APC main targets are point-like based on DECaLS morphology
    # APC and GAIA_ASTROMETRIC_NOISE.
    faint &= _psflike(objtype)
    faint &= gaiaaen < 2.0

    # APC faint targets are 19 <= r < 20
    faint &= rflux > 10**((22.5-20.0)/2.5)
    faint &= rflux <= 10**((22.5-19.0)/2.5)

    # APC faint targets are robs < 20.1
    faint &= obs_rflux > 10**((22.5-20.1)/2.5)

    # ADM calculate the overall proper motion magnitude
    pm = np.sqrt(pmra**2. + pmdec**2.)

    # ADM make a copy of the main bits for a red/blue split
    faint_red = faint.copy()
    faint_blue = faint.copy()

    # APC MWS-FAINT-BLUE is g-r < 0.7
    faint_blue &= rflux < gflux * 10**(0.7/2.5)  # (g-r)<0.7

    # ADM Turn off any NaNs for astrometric quantities to suppress
    # ADM warnings. Won't target these, using cuts on paramssolved
    # ADM (or will explicitly target them based on paramsssolved).
    ii = paramssolved < 31
    parallax = parallax.copy()
    parallax[ii] = 0.

    # ADM MWS-FAINT-RED has g-r >= 0.7
    faint_red &= rflux >= gflux * 10**(0.7/2.5)  # (g-r)>=0.7

    # APC MWS-FAINT-RED has samle plx cut as MAIN-RED
    # APC and proper motion < 3 (lower than MAIN-RED)
    # ACP and all astrometric parameters are measured.
    faint_red &= parallax < (3 * parallaxerr + 0.3)
    faint_red &= pm < 3.
    faint_red &= paramssolved >= 31

    # APC There is no equivalent of MWS-MAIN-BROAD for the faint sample.
    # APC stars failing the astrometry cuts or without the Gaia parameters
    # APC are discarded.

    return faint_red, faint_blue


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
    mws &= gaiaaen < 2.0

    # ADM main targets are 16 <= r < 19
    mws &= rflux > 10**((22.5-19.0)/2.5)
    mws &= rflux <= 10**((22.5-16.0)/2.5)

    # ADM main targets are robs < 20
    mws &= obs_rflux > 10**((22.5-20.0)/2.5)

    # ADM calculate the overall proper motion magnitude
    pm = np.sqrt(pmra**2. + pmdec**2.)

    # ADM make a copy of the main bits for a red/blue split
    red = mws.copy()
    blue = mws.copy()

    # ADM MWS-BLUE is g-r < 0.7
    blue &= rflux < gflux * 10**(0.7/2.5)  # (g-r)<0.7

    # ADM Turn off any NaNs for astrometric quantities to suppress
    # ADM warnings. Won't target these, using cuts on paramssolved
    # ADM (or will explicitly target them based on paramsssolved).
    ii = paramssolved < 31
    parallax = parallax.copy()
    parallax[ii], pm[ii] = 0., 0.

    # ADM MWS-RED and MWS-BROAD have g-r >= 0.7
    red &= rflux >= gflux * 10**(0.7/2.5)  # (g-r)>=0.7
    broad = red.copy()

    # ADM MWS-RED also has parallax < (3parallax_err+0.3)mas
    # ADM and proper motion < 5 * sqrt(rflux/r_19)
    # ADM and all astrometric parameters are measured.
    pmmax = 5 * np.sqrt(rflux/(10**((22.5-19)/2.5)))
    plxmax = 3 * parallaxerr + 0.3
    red &= parallax < plxmax
    red &= pm < pmmax
    red &= paramssolved >= 31

    # ADM MWS-BROAD has parallax > (3parallax_err+0.3)mas
    # ADM OR proper motion > max value
    # ADM OR astrometric parameters not measured.
    broad &= ((parallax >= plxmax) |
              (pm >= pmmax)
              | (paramssolved < 31))

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
    - Current version (09/20/18) is version 129 on `the wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM do not target any objects for which entries are NaN
    # ADM and turn off the NaNs for those entries.
    nans = np.isnan(gaiagmag) | np.isnan(parallax)
    w = np.where(nans)[0]
    if len(w) > 0:
        # ADM make copies as we are reassigning values
        parallax, gaiagmag = parallax.copy(), gaiagmag.copy()
        parallax[w], gaiagmag[w] = 0., 0.
        mws &= ~nans
        log.info('{}/{} NaNs in file...t = {:.1f}s'
                 .format(len(w), len(mws), time()-start))

    # ADM apply the selection for all MWS-NEARBY targets.
    # ADM must be a Legacy Surveys object that matches a Gaia source.
    mws &= gaia
    # ADM Gaia G mag of less than 20.
    mws &= gaiagmag < 20.
    # APC Gaia G mag of more than 16.
    mws &= gaiagmag > 16.
    # ADM all astrometric parameters are measured.
    mws &= paramssolved >= 31
    # ADM parallax cut corresponding to 100pc.
    mws &= (parallax + parallaxerr) > 10.   # NB: "+" is correct.

    return mws


def isMWS_bhb(primary=None, objtype=None,
              gaia=None, gaiaaen=None, gaiadupsource=None, gaiagmag=None,
              gflux=None, rflux=None, zflux=None,
              w1flux=None, w1snr=None, maskbits=None,
              gnobs=None, rnobs=None, znobs=None,
              gfracmasked=None, rfracmasked=None, zfracmasked=None,
              parallax=None, parallaxerr=None):
    """Set bits for BHB Milky Way Survey targets.

    Parameters
    ----------
    see :func:`~desitarget.cuts.set_target_bits` for other parameters.

    Returns
    -------
    mask : array_like.
        True if and only if the object is a MWS-BHB target.

    Notes
    -----
    - Criteria supplied by Sergey Koposov.
    - gflux, rflux, zflux, w1flux have been corrected for extinction
      (unlike other MWS selections, which use obs_flux).
    - Current version (02/18/21) is version 246 on `the wiki`_.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM do not target any objects for which entries are NaN
    # ADM and turn off the NaNs for those entries
    nans = (np.isnan(gflux) | np.isnan(rflux) | np.isnan(zflux) |
            np.isnan(w1flux) | np.isnan(parallax) | np.isnan(gaiagmag))
    w = np.where(nans)[0]
    if len(w) > 0:
        # ADM make copies as we are reassigning values
        rflux, gflux, zflux = rflux.copy(), gflux.copy(), zflux.copy()
        w1flux = w1flux.copy()
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

    # APC must be a Legacy Surveys object that matches a Gaia source.
    mws &= gaia
    # APC type must be PSF.
    mws &= _psflike(objtype)
    # APC no sources brighter than Gaia G = 10.
    mws &= gaiagmag > 10.
    # APC exclude nearby sources by parallax.
    mws &= parallax <= 0.1 + 3*parallaxerr

    mws &= (gfracmasked < 0.5) & (gflux > 0) & (gnobs > 0)
    mws &= (rfracmasked < 0.5) & (rflux > 0) & (rnobs > 0)
    mws &= (zfracmasked < 0.5) & (zflux > 0) & (znobs > 0)

    # APC no gaia duplicated sources.
    mws &= ~gaiadupsource
    # APC gaia astrometric excess noise < 2.
    mws &= gaiaaen < 2.0

    # APC BHB extinction-corrected color range -0.35 <= gmr <= -0.02.
    mws &= (gmr >= -0.35) & (gmr <= -0.02)

    # Coefficients from Sergey Koposov.
    bhb_sel = rmz - (1.07163*gmr**5 - 1.42272*gmr**4 + 0.69476*gmr**3
                     - 0.12911*gmr**2 + 0.66993*gmr - 0.11368)
    mws &= (bhb_sel >= -0.05) & (bhb_sel <= 0.05)

    # APC back out WISE error = 1/sqrt(ivar) from SNR = flux*sqrt(ivar).
    w1fluxerr = w1flux/(w1snr.clip(1e-7))
    w1mag_faint = 22.5 - 2.5 * np.log10((w1flux-3*w1fluxerr).clip(1e-7))

    # APC WISE cut (Sergey Koposov).
    mws &= rmag - 2.3*gmr - w1mag_faint < -1.5

    # APC Legacy magnitude limits.
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
    - Current version (08/01/18) is version 121 on `the wiki`_.
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
    mws &= paramssolved >= 31

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
    """Fail if `targtype` is not one 'bright', 'faint' or 'wise'.
    """
    targposs = ['faint', 'bright', 'wise']

    if targtype not in targposs:
        msg = 'targtype must be one of {} not {}'.format(targposs, targtype)
        log.critical(msg)
        raise ValueError(msg)


def _check_BGS_targtype_sv(targtype):
    """check `targtype` ('bright', 'faint', 'faint_ext', 'lowq', 'fibmag').
    """
    targposs = ['faint', 'bright', 'faint_ext', 'lowq', 'fibmag']

    if targtype not in targposs:
        msg = 'targtype must be one of {} not {}'.format(targposs, targtype)
        log.critical(msg)
        raise ValueError(msg)


def isBGS(rfiberflux=None, gflux=None, rflux=None, zflux=None, rfibertotflux=None,
          w1flux=None, w2flux=None, gnobs=None, rnobs=None, znobs=None,
          gfluxivar=None, rfluxivar=None, zfluxivar=None, maskbits=None,
          Grr=None, refcat=None, w1snr=None, w2snr=None, gaiagmag=None,
          objtype=None, primary=None, south=True, targtype=None):
    """Definition of BGS target classes. Returns a boolean array.

    Args
    ----
    targtype: str, optional, defaults to ``faint``
        Pass ``bright`` for the ``BGS_BRIGHT`` selection.
        or ``faint`` for the ``BGS_FAINT`` selection
        or ``wise`` for the ``BGS_WISE`` (AGN-like) selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if the object is a BGS target of type ``targtype``.

    Notes
    -----
    - Current version (05/11/21) is version 264 on `the wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    _check_BGS_targtype(targtype)

    # ADM to maintain backwards-compatibility with mocks.
    if rfiberflux is None:
        log.warning('Setting rfiberflux to rflux!!!')
        rfiberflux = rflux.copy()

    # ------ Bright Galaxy Survey
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()

    if targtype == 'wise':
        bgs &= isBGS_wise(
            rfiberflux=rfiberflux, gflux=gflux, rflux=rflux, zflux=zflux,
            rfibertotflux=rfibertotflux, w1flux=w1flux, w2flux=w2flux,
            refcat=refcat, maskbits=maskbits,
            w1snr=w1snr, w2snr=w2snr, Grr=Grr, gaiagmag=gaiagmag,
            gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
            gnobs=gnobs, rnobs=rnobs, znobs=znobs, south=south,
            targtype=targtype, objtype=objtype, primary=primary
        )
    else:
        bgs &= notinBGS_mask(
            gnobs=gnobs, rnobs=rnobs, znobs=znobs, primary=primary,
            gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
            Grr=Grr, gaiagmag=gaiagmag, maskbits=maskbits, targtype=targtype
        )
        bgs &= isBGS_colors(
            rfiberflux=rfiberflux, gflux=gflux, rflux=rflux, zflux=zflux,
            rfibertotflux=rfibertotflux, w1flux=w1flux, maskbits=maskbits,
            south=south, targtype=targtype, primary=primary
        )
        bgs |= isBGS_sga(
            rfiberflux=rfiberflux, gflux=gflux, rflux=rflux, zflux=zflux,
            rfibertotflux=rfibertotflux, w1flux=w1flux, refcat=refcat,
            maskbits=maskbits, south=south, targtype=targtype
        )

    return bgs


def notinBGS_mask(gnobs=None, rnobs=None, znobs=None, primary=None,
                  gfluxivar=None, rfluxivar=None, zfluxivar=None, Grr=None,
                  gaiagmag=None, maskbits=None, targtype=None):
    """Standard masking cuts used by all BGS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isBGS` for parameters).
    """
    _check_BGS_targtype(targtype)

    if primary is None:
        primary = np.ones_like(gnobs, dtype='?')
    bgs = primary.copy()

    bgs &= (gnobs >= 1) & (rnobs >= 1) & (znobs >= 1)
    bgs &= (gfluxivar > 0) & (rfluxivar > 0) & (zfluxivar > 0)

    # ADM geometric masking cuts from the Legacy Surveys.
    # BRIGHT & CLUSTER for BGS
    bgs &= imaging_mask(maskbits, bgsmask=True)

    if targtype == 'bright':
        bgs &= ((Grr > 0.6) | (gaiagmag == 0))
    elif targtype == 'faint':
        bgs &= ((Grr > 0.6) | (gaiagmag == 0))

    return bgs


def isBGS_colors(rfiberflux=None, gflux=None, rflux=None, zflux=None,
                 w1flux=None, rfibertotflux=None, maskbits=None,
                 south=True, targtype=None, primary=None):
    """Standard color-based cuts used by all BGS target selection classes
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
    w1 = 22.5 - 2.5*np.log10(w1flux.clip(1e-16))
    rfib = 22.5 - 2.5*np.log10(rfiberflux.clip(1e-16))

    # Fibre Magnitude Cut (FMC) -- This is a low surface brightness cut
    # with the aim of increase the redshift success rate.
    fmc |= ((rfib < (2.9 + 1.2 + 1.0) + r) & (r < 17.8))
    fmc |= ((rfib < 22.9) & (r < 20.0) & (r > 17.8))
    # ?????????????????????????????????????????????????????
    # Remove next line for the Main Survey?????????????????
    # ?????????????????????????????????????????????????????
    fmc |= ((rfib < 2.9 + r) & (r > 20))

    bgs &= fmc

    # BASS r-mag offset with DECaLS.
    offset = 0.04

    # D. Schlegel - ChangHoon H. color selection to get a high redshift
    # success rate.
    if south:
        schlegel_color = (z - w1) - 3/2.5 * (g - r) + 1.2
        rfibcol = (rfib < 20.75) | ((rfib < 21.5) & (schlegel_color > 0.))
    else:
        schlegel_color = (z - w1) - 3/2.5 * (g - (r-offset)) + 1.2
        rfibcol = (rfib < 20.75+offset) | ((rfib < 21.5+offset) &
                                           (schlegel_color > 0.))

    if targtype == 'bright':
        if south:
            bgs &= rflux > 10**((22.5-19.5)/2.5)
            bgs &= rflux <= 10**((22.5-12.0)/2.5)
            bgs &= rfibertotflux <= 10**((22.5-15.0)/2.5)
        else:
            bgs &= rflux > 10**((22.5-(19.5+offset))/2.5)
            bgs &= rflux <= 10**((22.5-12.0)/2.5)
            bgs &= rfibertotflux <= 10**((22.5-15.0)/2.5)
    elif targtype == 'faint':
        if south:
            bgs &= rflux > 10**((22.5-20.175)/2.5)
            bgs &= rflux <= 10**((22.5-19.5)/2.5)
            bgs &= (rfibcol)
        else:
            bgs &= rflux > 10**((22.5-(20.220))/2.5)
            bgs &= rflux <= 10**((22.5-(19.5+offset))/2.5)
            bgs &= (rfibcol)

    return bgs


def isBGS_wise(rfiberflux=None, gflux=None, rflux=None, zflux=None, w1flux=None,
               rfibertotflux=None, w2flux=None, refcat=None, maskbits=None,
               w1snr=None, w2snr=None, Grr=None, gaiagmag=None,
               gfluxivar=None, rfluxivar=None, zfluxivar=None,
               gnobs=None, rnobs=None, znobs=None, south=True, targtype=None,
               objtype=None, primary=None):
    """Standard color-based cuts used by all BGS target selection classes
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

    g = 22.5 - 2.5*np.log10(gflux.clip(1e-16))
    r = 22.5 - 2.5*np.log10(rflux.clip(1e-16))
    z = 22.5 - 2.5*np.log10(zflux.clip(1e-16))
    w1 = 22.5 - 2.5*np.log10(w1flux.clip(1e-16))
    w2 = 22.5 - 2.5*np.log10(w2flux.clip(1e-16))
    rfib = 22.5 - 2.5*np.log10(rfiberflux.clip(1e-16))
    rfibtot = 22.5 - 2.5*np.log10(rfibertotflux.clip(1e-16))

    agnw2 = primary.copy()
    agnw2 = (z - w2 - (g - r) > -0.5) &   \
            (w2snr > 10) &\
            ((maskbits & 2**(9)) == 0) &  \
            (w2flux > 0)

    quality = primary.copy()
    quality = (rflux > 0) & (gflux > 0) & (zflux > 0) &                   \
              (gfluxivar > 0) & (rfluxivar > 0) & (zfluxivar > 0) &       \
              (gnobs >= 1) & (rnobs >= 1) & (znobs >= 1) &                \
              (imaging_mask(maskbits, bgsmask=True)) &                    \
              ((gaiagmag == 0) | ((gaiagmag != 0) & (gaiagmag > 16))) &   \
              (r < 20.3) & (r > 16) &                                     \
              (rfib < 22) & (rfibtot > 15) &                              \
              (((rfib < 21.5) & (r > 19.5)) | (r < 19.5)) &               \
              (((_psflike(objtype)) & (r < 17.5)) | (~_psflike(objtype)))

    stars = primary.copy()
    stars = ((gaiagmag != 0) & (Grr < 0.6)) | ((gaiagmag == 0) &
                                               (_psflike(objtype)))

    additional = primary.copy()
    additional = ((w1 - w2) > -0.2) &          \
                 (z - w1 - (g - r) > -0.7) &   \
                 (w1snr > 10)

    agn_ext = primary.copy()
    agn_ext = (agnw2) & (additional)

    AGN = primary.copy()
    AGN = (agn_ext) & (quality) & ~((stars) & (~agn_ext))
    AGN &= (Grr < 0.6) & (gaiagmag != 0)

    return AGN


def isBGS_sga(gflux=None, rflux=None, zflux=None, w1flux=None, refcat=None,
              rfibertotflux=None, rfiberflux=None,
              maskbits=None, south=True, targtype=None):
    """Module to recover the SGA objects in all BGS target classes
    (see, e.g., :func:`~desitarget.cuts.isBGS` for parameters).
    """

    _check_BGS_targtype(targtype)

    bgs = np.zeros_like(rflux, dtype='?')

    # the SGA galaxies.
    LX = np.zeros_like(rflux, dtype='?')
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

    # Make sure to include all the SGA galaxies.
    bgs |= LX
    # ADM geometric masking cuts from the Legacy Surveys.
    # Remove SGA in BRIGHT and CLUSTER.
    bgs &= imaging_mask(maskbits, bgsmask=True)

    g = 22.5 - 2.5*np.log10(gflux.clip(1e-16))
    r = 22.5 - 2.5*np.log10(rflux.clip(1e-16))
    z = 22.5 - 2.5*np.log10(zflux.clip(1e-16))
    w1 = 22.5 - 2.5*np.log10(w1flux.clip(1e-16))
    rfib = 22.5 - 2.5*np.log10(rfiberflux.clip(1e-16))

    # BASS r-mag offset with DECaLS.
    offset = 0.04

    # D. Schlegel - ChangHoon H. color selection to get a high redshift
    # success rate.
    if south:
        schlegel_color = (z - w1) - 3/2.5 * (g - r) + 1.2
        rfibcol = (rfib < 20.75) | ((rfib < 21.5) & (schlegel_color > 0.))
    else:
        schlegel_color = (z - w1) - 3/2.5 * (g - (r-offset)) + 1.2
        rfibcol = (rfib < 20.75+offset) | ((rfib < 21.5+offset)
                                           & (schlegel_color > 0.))

    if targtype == 'bright':
        if south:
            bgs &= rflux > 10**((22.5-19.5)/2.5)
            bgs &= rflux <= 10**((22.5-12.0)/2.5)
            bgs &= rfibertotflux <= 10**((22.5-15.0)/2.5)
        else:
            bgs &= rflux > 10**((22.5-(19.5+offset))/2.5)
            bgs &= rflux <= 10**((22.5-12.0)/2.5)
            bgs &= rfibertotflux <= 10**((22.5-15.0)/2.5)
    elif targtype == 'faint':
        if south:
            bgs &= rflux > 10**((22.5-20.175)/2.5)
            bgs &= rflux <= 10**((22.5-19.5)/2.5)
            bgs &= (rfibcol)
        else:
            bgs &= rflux > 10**((22.5-(20.220))/2.5)
            bgs &= rflux <= 10**((22.5-(19.5+offset))/2.5)
            bgs &= (rfibcol)

    return bgs


def isQSO_cuts(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
               w1snr=None, w2snr=None, maskbits=None,
               gnobs=None, rnobs=None, znobs=None,
               objtype=None, primary=None, optical=False, south=True):
    """QSO targets from color cuts. Returns a boolean array.

    Parameters
    ----------
    optical : :class:`boolean`, defaults to ``False``
        Apply just optical color-cuts.
    south : :class:`boolean`, defaults to ``True``
        Use cuts for Northern imaging (BASS/MzLS) if ``south=False``,
        otherwise use cuts for Southern imaging (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` for objects passing quasar color/morphology/logic cuts.

    Notes
    -----
    - Current version (06/05/19) is version 176 on `the wiki`_.
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
            qso &= w2flux > w1flux * 10**(-0.4/2.5)              # (W1-W2)>-0.4
        else:
            qso &= w2flux > w1flux * 10**(-0.3/2.5)              # (W1-W2)>-0.3
        # (grz-W)>(g-z)-1.0
        qso &= wflux * gflux > zflux * grzflux * 10**(-1.0/2.5)

    # Harder cut on stellar contamination
    mainseq = rflux > gflux * 10**(0.20/2.5)  # g-r>0.2

    # Clip to avoid warnings for -ve numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    mainseq &= rflux**(1+1.53) > gflux * zflux**1.53 * 10**((-0.100+0.20)/2.5)
    mainseq &= rflux**(1+1.53) < gflux * zflux**1.53 * 10**((+0.100+0.20)/2.5)
    if not optical:
        mainseq &= w2flux < w1flux * 10**(0.3/2.5)
    qso &= ~mainseq

    return qso


def isQSO_randomforest(gflux=None, rflux=None, zflux=None, maskbits=None,
                       w1flux=None, w2flux=None, objtype=None,
                       gnobs=None, rnobs=None, znobs=None,
                       primary=None, ra=None, dec=None, south=True,
                       return_probs=False):
    """Define QSO targets from a Random Forest. Returns a boolean array.

    Parameters
    ----------
    south : :class:`boolean`, defaults to ``True``
        If ``False``, shift photometry to the Northern (BASS/MzLS)
        imaging system.
    return_probs : :class:`boolean`, defaults to ``False``
        If ``True``, return QSO probabilities in addition to the target
        boolean. Only coded up for DR8 or later of the Legacy Surveys
        imaging. Will return arrays of zeros for earlier Data Releases.

    Returns
    -------
    :class:`array_like`
        ``True`` for objects that are Random Forest quasar targets.
    :class:`array_like`
        The (float) probability that a target is a quasar. Only returned
        if `return_probs` is ``True``.

    Notes
    -----
    - Current version (05/06/21) is version 259 on `the wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    # ADM Primary (True for any initially possible target).
    if primary is None:
        primary = np.ones_like(gflux, dtype=bool)

    # Build variables for random forest
    # Number of attributes describing each object to be classified.
    nFeatures = 11
    nbEntries = rflux.size
    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    colors, r, photOK = _getColors(nbEntries, nFeatures,
                                   gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d(r)

    # For the last version of QSO selection we add cuts on W1 and W2!
    limitInf = 1.e-04
    w1flux, w2flux = w1flux.clip(limitInf), w2flux.clip(limitInf)
    W1 = np.atleast_1d(np.where(w1flux > limitInf,
                                22.5-2.5*np.log10(w1flux), 0.))
    W2 = np.atleast_1d(np.where(w2flux > limitInf,
                                22.5-2.5*np.log10(w2flux), 0.))
    W1_cut, W2_cut = 22.3, 22.3

    # Preselection to speed up the process.
    rMax = 23.0   # r < 23.0
    rMin = 16.5   # r > 16.5
    preSelection = (r < rMax) & (r > rMin) & photOK & primary

    # ADM targets have to be observed in every band.
    preSelection &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

    if objtype is not None:
        preSelection &= _psflike(objtype)

    if maskbits is not None:
        # ADM default mask bits from the Legacy Surveys not set.
        preSelection &= imaging_mask(maskbits)

    # "qso" mask initialized to "preSelection" mask.
    qso = np.copy(preSelection)

    # ADM these store the probabilities, should they need returned.
    pqso = np.zeros_like(qso, dtype='>f4')

    if np.any(preSelection):

        from desitarget.myRF import myRF

        # Data reduction to preselected objects.
        colorsReduced = colors[preSelection]
        r_Reduced = r[preSelection]
        W1_Reduced, W2_Reduced = W1[preSelection], W2[preSelection]
        colorsIndex = np.arange(0, nbEntries, dtype=np.int64)
        colorsReducedIndex = colorsIndex[preSelection]

        # Path to random forest files.
        pathToRF = resource_filename('desitarget', 'data')
        # rf filename.
        rf_DR9_fileName = pathToRF + '/rf_model_dr9_final.npz'

        # rf initialization - colors data duplicated within "myRF".
        rf = myRF(colorsReduced, pathToRF, numberOfTrees=500, version=2)

        # rf loading.
        rf.loadForest(rf_DR9_fileName)
        # Compute rf probabilities.
        tmp_rf_proba = rf.predict_proba()
        # Compute optimized proba cut.
        tmp_r_Reduced = r_Reduced
        tmp_W1_Reduced = W1_Reduced
        tmp_W2_Reduced = W2_Reduced

        if not south:
            # threshold selection for North footprint.
            pcut = 0.88 - 0.04*np.tanh(tmp_r_Reduced - 20.5)
        else:
            pcut = np.ones(tmp_rf_proba.size)
            is_des = (gnobs[preSelection] > 4) &                                \
                     (rnobs[preSelection] > 4) &                                \
                     (znobs[preSelection] > 4) &                                \
                     ((ra[preSelection] >= 320) | (ra[preSelection] <= 100)) &  \
                     (dec[preSelection] <= 10)
            # threshold selection for DES footprint.
            pcut[is_des] = 0.7 - 0.05*np.tanh(tmp_r_Reduced[is_des] - 20.5)
            # threshold selection for South footprint.
            pcut[~is_des] = 0.84 - 0.04*np.tanh(tmp_r_Reduced[~is_des] - 20.5)

        # Add rf proba test result to "qso" mask
        qso[colorsReducedIndex] = (tmp_rf_proba >= pcut) &                      \
            (tmp_W1_Reduced <= W1_cut) & (tmp_W2_Reduced <= W2_cut)
        # ADM store the probabilities in case they need returned.
        pqso[colorsReducedIndex] = tmp_rf_proba

    # In case of call for a single object passed to the function with
    # scalar arguments. Return "numpy.bool_" instead of "~numpy.ndarray".
    if nbEntries == 1:
        qso = qso[0]
        pqso = pqso[0]

    # ADM if requested, return the probabilities as well.
    if return_probs:
        return qso, pqso
    return qso


def _psflike(psftype):
    """ If the object is PSF """
    # ADM explicitly checking for NoneType. In the past we have had bugs
    # ADM where we forgot to pass objtype=objtype in, e.g., isSTD.
    if psftype is None:
        msg = "NoneType submitted to _psfflike function"
        log.critical(msg)
        raise ValueError(msg)

    psftype = np.asarray(psftype)
    # ADM in Python3 these string literals become byte-like
    # ADM so to retain Python2 compatibility we need to check
    # ADM against both bytes and unicode.
    # ADM Also 'PSF' for astropy.io.fits; 'PSF ' for fitsio (sigh).
    psflike = ((psftype == 'PSF') | (psftype == b'PSF') |
               (psftype == 'PSF ') | (psftype == b'PSF '))

    return psflike


def _isonnorthphotsys(photsys):
    """ If the object is from the northen photometric system """
    # ADM explicitly checking for NoneType. In the past we have had bugs
    # ADM where we forgot to populate variables before passing them.
    if photsys is None:
        msg = "NoneType submitted to _isonnorthphotsys function"
        log.critical(msg)
        raise ValueError(msg)

    psftype = np.asarray(photsys)
    # ADM in Python3 these string literals become byte-like
    # ADM so to retain Python2 compatibility we need to check
    # ADM against both bytes and unicode.
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
    """Return True/False if this is a table row instead of a full table.

    supports numpy.ndarray, astropy.io.fits.FITS_rec, astropy.table.Table
    """
    import astropy.io.fits.fitsrec
    import astropy.table.row
    if isinstance(table, (astropy.io.fits.fitsrec.FITS_record,
                          astropy.table.row.Row)) or \
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


def _prepare_optical_wise(objects, mask=True):
    """Process the Legacy Surveys inputs for target selection.

    Parameters
    ----------
    mask : :class:`boolean`, optional, defaults to ``True``
        Send ``False`` to turn off any masking cuts based on the `MASKBITS` column. The
        default behavior is to always mask using `MASKBITS`.
    """
    # ADM flag whether we're using northen (BASS/MZLS) or
    # ADM southern (DECaLS) photometry.
    photsys_north = _isonnorthphotsys(objects["PHOTSYS"])
    photsys_south = ~photsys_north
    # ADM catch case where single object or row is passed.
    if isinstance(photsys_north, bool):
        photsys_south = not(photsys_north)

    # ADM the observed r-band flux (used for F standards and MWS, below).
    # ADM make copies of values that we may reassign due to NaNs.
    obs_rflux = objects['FLUX_R']
    # ADM don't de-extinct the FIBERTOTFLUXES as they're mainly used to
    # ADM limit bright-end observations to prevent fiber cross-talk.
    gfibertotflux = objects['FIBERTOTFLUX_G']
    rfibertotflux = objects['FIBERTOTFLUX_R']
    zfibertotflux = objects['FIBERTOTFLUX_Z']

    # - undo Milky Way extinction.
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    gfiberflux = flux['GFIBERFLUX']
    rfiberflux = flux['RFIBERFLUX']
    zfiberflux = flux['ZFIBERFLUX']
    objtype = objects['TYPE']
    release = objects['RELEASE']

    ra = objects['RA']
    dec = objects['DEC']

    gfluxivar = objects['FLUX_IVAR_G']
    rfluxivar = objects['FLUX_IVAR_R']
    zfluxivar = objects['FLUX_IVAR_Z']
    w1fluxivar = objects['FLUX_IVAR_W1']

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

    refcat = objects['REF_CAT']

    maskbits = objects['MASKBITS']
    # ADM if we asked to turn off masking behavior, turn it off.
    if not mask:
        maskbits = objects['MASKBITS'].copy()
        maskbits[...] = 0

    # Delta chi2 between PSF and SIMP morphologies; note the sign....
    dchisq = objects['DCHISQ']
    deltaChi2 = dchisq[..., 0] - dchisq[..., 1]

    # ADM remove any NaNs from DCHISQ values, make them unselectable.
    # SJB support py3.8 + np1.18 for both scalars and vectors.
    if np.isscalar(deltaChi2):
        if np.isnan(deltaChi2):
            deltaChi2 = -1e6
    else:
        w = np.isnan(deltaChi2)
        deltaChi2[w] = -1e6

    return (photsys_north, photsys_south, obs_rflux, gflux, rflux, zflux,
            w1flux, w2flux, gfiberflux, rfiberflux, zfiberflux,
            gfibertotflux, rfibertotflux, zfibertotflux, objtype, release,
            ra, dec, gfluxivar, rfluxivar, zfluxivar, w1fluxivar,
            gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,
            gfracmasked, rfracmasked, zfracmasked,
            gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,
            gsnr, rsnr, zsnr, w1snr, w2snr,
            dchisq, deltaChi2, maskbits, refcat)


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
    # ADM derive parallax/parallax_error, set to 0 where error is bad.
    parallaxovererror = np.where(parallaxivar > 0.,
                                 parallax*np.sqrt(parallaxivar), 0.)

    # We also need the parallax uncertainty, for MWS_NEARBY targets.
    parallaxerr = np.zeros_like(parallax) - 1e8  # make large and -ve.
    notzero = parallaxivar > 0
    if np.sum(notzero) > 0:
        parallaxerr[notzero] = 1 / np.sqrt(parallaxivar[notzero])
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    gaiabmag = objects['GAIA_PHOT_BP_MEAN_MAG']
    gaiarmag = objects['GAIA_PHOT_RP_MEAN_MAG']
    gaiaaen = objects['GAIA_ASTROMETRIC_EXCESS_NOISE']
    # ADM a mild hack, as GAIA_DUPLICATED_SOURCE was 0/1 at some point.
    gaiadupsource = objects['GAIA_DUPLICATED_SOURCE']
    if issubclass(gaiadupsource.dtype.type, np.integer):
        if len(set(np.atleast_1d(gaiadupsource)) - set([0, 1])) == 0:
            gaiadupsource = objects['GAIA_DUPLICATED_SOURCE'].astype(bool)

    # For BGS target selection.
    # ADM first guard against FLUX_R < 0 (I've checked this generates
    # ADM the same set of targets as Grr = NaN).
    Grr = gaiagmag - 22.5 + 2.5*np.log10(1e-16)
    ii = objects['FLUX_R'] > 0
    # ADM catch the case where Grr is a scalar.
    if isinstance(Grr, np.float):
        if ii:
            Grr = gaiagmag - 22.5 + 2.5*np.log10(objects['FLUX_R'])
    else:
        Grr[ii] = gaiagmag[ii] - 22.5 + 2.5*np.log10(objects['FLUX_R'][ii])

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
            # ADM we need to check the case of a single row being passed.
            if _is_row(gaiaparamssolved):
                gaiaparamsolved = 3
            else:
                gaiaparamssolved[w] = 3

    # ADM Add these columns if they exist, or set them to none.
    # ADM (They weren't in the Tractor files prior to DR8).
    gaiabprpfactor = None
    gaiasigma5dmax = None
    if 'GAIA_PHOT_BP_RP_EXCESS_FACTOR' in colnames:
        gaiabprpfactor = objects['GAIA_PHOT_BP_RP_EXCESS_FACTOR']
    if 'GAIA_ASTROMETRIC_SIGMA5D_MAX' in colnames:
        gaiasigma5dmax = objects['GAIA_ASTROMETRIC_SIGMA5D_MAX']

    # ADM Milky Way Selection requires Galactic b.
    _, galb = _gal_coords(objects["RA"], objects["DEC"])

    return (gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr,
            gaiagmag, gaiabmag, gaiarmag, gaiaaen, gaiadupsource, Grr,
            gaiaparamssolved, gaiabprpfactor, gaiasigma5dmax, galb)


def unextinct_fluxes(objects):
    """Calculate unextincted DECam and WISE fluxes.

    Args:
        objects: array or Table with columns FLUX_G, FLUX_R, FLUX_Z,
            MW_TRANSMISSION_G, MW_TRANSMISSION_R, MW_TRANSMISSION_Z,
            FLUX_W1, FLUX_W2, MW_TRANSMISSION_W1, MW_TRANSMISSION_W2.

    Returns:
        array or Table with columns GFLUX, RFLUX, ZFLUX, W1FLUX, W2FLUX.

    Output type is the same as the input type.
    """
    dtype = [('GFLUX', 'f4'), ('RFLUX', 'f4'), ('ZFLUX', 'f4'),
             ('W1FLUX', 'f4'), ('W2FLUX', 'f4'),
             ('GFIBERFLUX', 'f4'), ('RFIBERFLUX', 'f4'), ('ZFIBERFLUX', 'f4')]

    if _is_row(objects):
        result = np.zeros(1, dtype=dtype)[0]
    else:
        result = np.zeros(len(objects), dtype=dtype)

    result['GFLUX'] = objects['FLUX_G'] / objects['MW_TRANSMISSION_G']
    result['RFLUX'] = objects['FLUX_R'] / objects['MW_TRANSMISSION_R']
    result['ZFLUX'] = objects['FLUX_Z'] / objects['MW_TRANSMISSION_Z']
    result['W1FLUX'] = objects['FLUX_W1'] / objects['MW_TRANSMISSION_W1']
    result['W2FLUX'] = objects['FLUX_W2'] / objects['MW_TRANSMISSION_W2']
    result['GFIBERFLUX'] = objects['FIBERFLUX_G'] / objects['MW_TRANSMISSION_G']
    result['RFIBERFLUX'] = objects['FIBERFLUX_R'] / objects['MW_TRANSMISSION_R']
    result['ZFIBERFLUX'] = objects['FIBERFLUX_Z'] / objects['MW_TRANSMISSION_Z']

    if isinstance(objects, Table):
        return Table(result)
    else:
        return result


def set_target_bits(photsys_north, photsys_south, obs_rflux,
                    gflux, rflux, zflux, w1flux, w2flux,
                    gfiberflux, rfiberflux, zfiberflux, gfibertotflux,
                    rfibertotflux, zfibertotflux, objtype, release,
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
        Predicted fiber flux from object in 1 arcsecond seeing in g/r/z.
        Corrected for Galactic extinction.
    gfibertotflux, rfibertotflux, zfibertotflux : :class:`~numpy.ndarray`
        Predicted fiber flux from ALL sources at object's location in 1
        arcsecond seeing in g/r/z. NOT corrected for Galactic extinction.
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

    # ADM if resolvetargs is set, limit to only sending north/south
    # ADM objects through north/south cuts.
    south_cuts = [False, True]
    if resolvetargs:
        # ADM if only southern objects were sent this will be [True], if
        # ADM only northern it will be [False], else it wil be both.
        south_cuts = list(set(np.atleast_1d(photsys_south)))

    # ADM default for target classes we WON'T process is all False.
    tcfalse = primary & False

    # ADM initially set everything to arrays of False for LRG selection.
    # ADM zeroth element stores the northern targets bits (south=False).
    lrg_classes = [tcfalse, tcfalse]
    if "LRG" in tcnames:
        for south in south_cuts:
            lrg_classes[int(south)] = isLRG(
                primary=primary,
                gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                zfiberflux=zfiberflux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                rfluxivar=rfluxivar, zfluxivar=zfluxivar, w1fluxivar=w1fluxivar,
                gaiagmag=gaiagmag, zfibertotflux=zfibertotflux,
                maskbits=maskbits, south=south
            )
    lrg_north, lrg_south = lrg_classes

    # ADM combine LRG target bits for an LRG target based on any imaging.
    lrg = (lrg_north & photsys_north) | (lrg_south & photsys_south)

    # ADM initially set everything to arrays of False for ELG selection.
    # ADM zeroth element stores the northern targets bits (south=False).
    elg_classes = [[tcfalse, tcfalse], [tcfalse, tcfalse]]
    if "ELG" in tcnames:
        for south in south_cuts:
            elg_classes[int(south)] = isELG(
                primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
                gfiberflux=gfiberflux, gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits,
                south=south
            )
    elg_vlo_north, elg_lop_north = elg_classes[0]
    elg_vlo_south, elg_lop_south = elg_classes[1]

    # ADM combine ELG target bits for an ELG target based on any imaging.
    elg_vlo = (elg_vlo_north & photsys_north) | (elg_vlo_south & photsys_south)
    elg_lop = (elg_lop_north & photsys_north) | (elg_lop_south & photsys_south)
    elg_north = elg_vlo_north | elg_lop_north
    elg_south = elg_vlo_south | elg_lop_south
    elg = elg_vlo | elg_lop

    # ADM form a seed using zflux in case we parallelized by HEALPixel.
    # SJB seeds must be within 0 - 2**32-1
    uniqseed = int(np.mean(zflux)*1e5) % (2**32 - 1)
    np.random.seed(uniqseed)
    # ADM 10% of ELG_LOP and ELG_VLO targets need ELG_HIP set.
    elg_hip = random_fraction_of_trues(0.1, elg_lop)
    elg_hip |= random_fraction_of_trues(0.1, elg_vlo)

    # ADM initially set everything to arrays of False for QSO selection.
    # ADM zeroth element stores the northern targets bits (south=False).
    qso_classes = [tcfalse, tcfalse]
    if "QSO" in tcnames:
        for south in south_cuts:
            if qso_selection == 'colorcuts':
                # ADM find QSO targets in the north and south separately.
                qso_classes[int(south)] = isQSO_cuts(
                    primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                    w1flux=w1flux, w2flux=w2flux, maskbits=maskbits,
                    gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                    objtype=objtype, w1snr=w1snr, w2snr=w2snr,
                    optical=qso_optical_cuts, south=south
                )
            elif qso_selection == 'randomforest':
                # ADM find QSO targets in the north and south separately.
                qso_classes[int(south)] = isQSO_randomforest(
                    primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                    w1flux=w1flux, w2flux=w2flux, maskbits=maskbits,
                    gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                    objtype=objtype, ra=ra, dec=dec, south=south
                )
            else:
                msg = "Unknown qso_selection {}; valid options are {}".format(
                    qso_selection, qso_selection_options)
                log.critical(msg)
                raise ValueError(msg)
    qso_north, qso_south = qso_classes

    # ADM combine QSO targeting bits for a QSO selected in any imaging.
    qso = (qso_north & photsys_north) | (qso_south & photsys_south)

    # ADM initially set everything to arrays of False for BGS selection.
    # ADM zeroth element stores the northern targets bits (south=False).
    bgs_classes = [[tcfalse, tcfalse, tcfalse], [tcfalse, tcfalse, tcfalse]]
    # ADM set the BGS bits.
    if "BGS" in tcnames:
        for south in south_cuts:
            bgs_store = []
            for targtype in ["bright", "faint", "wise"]:
                bgs_store.append(
                    isBGS(
                        rfiberflux=rfiberflux, rfibertotflux=rfibertotflux, gflux=gflux,
                        rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux, gnobs=gnobs,
                        rnobs=rnobs, znobs=znobs, gfluxivar=gfluxivar, rfluxivar=rfluxivar,
                        zfluxivar=zfluxivar, maskbits=maskbits, Grr=Grr, refcat=refcat,
                        w1snr=w1snr, w2snr=w2snr, gaiagmag=gaiagmag, objtype=objtype,
                        primary=primary, south=south, targtype=targtype
                    )
                )
            bgs_classes[int(south)] = bgs_store
    bgs_bright_north, bgs_faint_north, bgs_wise_north = bgs_classes[0]
    bgs_bright_south, bgs_faint_south, bgs_wise_south = bgs_classes[1]

    # ADM combine BGS targeting bits for a BGS selected in any imaging.
    bgs_bright = (bgs_bright_north & photsys_north) | (bgs_bright_south & photsys_south)
    bgs_faint = (bgs_faint_north & photsys_north) | (bgs_faint_south & photsys_south)
    bgs_wise = (bgs_wise_north & photsys_north) | (bgs_wise_south & photsys_south)

    # ADM 20% of the BGS_FAINT sources need the BGS_FAINT_HIP bit set.
    # ADM note that the random seed is set after the ELG selection.
    bgs_faint_hip = random_fraction_of_trues(0.2, bgs_faint)

    # ADM initially set everything to arrays of False for the MWS selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    mws_classes = [[tcfalse, tcfalse, tcfalse, tcfalse, tcfalse],
                   [tcfalse, tcfalse, tcfalse, tcfalse, tcfalse]]
    mws_nearby = tcfalse
    mws_bhb = tcfalse
    # ADM this denotes a bright limit for all MWS sources.
    too_bright = MWS_too_bright(gaiagmag=gaiagmag, zfibertotflux=zfibertotflux)
    if "MWS" in tcnames:
        mws_nearby = isMWS_nearby(
            gaia=gaia, gaiagmag=gaiagmag, parallax=parallax,
            parallaxerr=parallaxerr, paramssolved=gaiaparamssolved
        )
        # ADM impose bright limits for all MWS_NEARBY targets.
        mws_nearby &= ~too_bright

        mws_bhb = isMWS_bhb(
            primary=primary, objtype=objtype, gaia=gaia, gaiaaen=gaiaaen,
            gaiadupsource=gaiadupsource, gaiagmag=gaiagmag,
            gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w1snr=w1snr,
            gnobs=gnobs, rnobs=rnobs, znobs=znobs, gfracmasked=gfracmasked,
            rfracmasked=rfracmasked, zfracmasked=zfracmasked,
            parallax=parallax, parallaxerr=parallaxerr, maskbits=maskbits
        )
        # ADM impose bright limits for all MWS_BHB targets.
        mws_bhb &= ~too_bright

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

            # ADM impose bright limits for all MWS_MAIN targets.
            mws_classes[int(south)] &= ~too_bright

    mws_broad_n, mws_red_n, mws_blue_n, mws_faint_red_n, mws_faint_blue_n = mws_classes[0]
    mws_broad_s, mws_red_s, mws_blue_s, mws_faint_red_s, mws_faint_blue_s = mws_classes[1]

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
        # ADM impose bright limits for all MWS_WD targets.
        mws_wd &= ~too_bright

    # ADM initially set everything to False for the standards.
    std_faint, std_bright, std_wd = tcfalse, tcfalse, tcfalse
    if "STD" in tcnames:
        # ADM run the MWS_MAIN target types for both faint and bright.
        # ADM Make sure to pass all needed columns! Once,t we stopped
        # ADM passing objtype, which meant no standards were returned!
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
    mws_faint_red = (mws_faint_red_n & photsys_north) | (mws_faint_red_s & photsys_south)
    mws_faint_blue = (mws_faint_blue_n & photsys_north) | (mws_faint_blue_s & photsys_south)

    # Construct the targetflag bits for DECaLS (i.e. South).
    desi_target = lrg_south * desi_mask.LRG_SOUTH
    desi_target |= elg_south * desi_mask.ELG_SOUTH
    desi_target |= elg_vlo_south * desi_mask.ELG_VLO_SOUTH
    desi_target |= elg_lop_south * desi_mask.ELG_LOP_SOUTH
    desi_target |= qso_south * desi_mask.QSO_SOUTH

    # Construct the targetflag bits for MzLS and BASS (i.e. North).
    desi_target |= lrg_north * desi_mask.LRG_NORTH
    desi_target |= elg_north * desi_mask.ELG_NORTH
    desi_target |= elg_vlo_north * desi_mask.ELG_VLO_NORTH
    desi_target |= elg_lop_north * desi_mask.ELG_LOP_NORTH
    desi_target |= qso_north * desi_mask.QSO_NORTH

    # Construct the targetflag bits combining north and south.
    desi_target |= lrg * desi_mask.LRG
    desi_target |= elg * desi_mask.ELG
    desi_target |= elg_vlo * desi_mask.ELG_VLO
    desi_target |= elg_lop * desi_mask.ELG_LOP
    desi_target |= qso * desi_mask.QSO

    # ADM set fraction of the ELG_LOP/ELG_VLO targets to ELG_HIP.
    desi_target |= elg_hip * desi_mask.ELG_HIP

    # ADM Standards.
    desi_target |= std_faint * desi_mask.STD_FAINT
    desi_target |= std_bright * desi_mask.STD_BRIGHT
    desi_target |= std_wd * desi_mask.STD_WD

    # BGS targets, south.
    bgs_target = bgs_bright_south * bgs_mask.BGS_BRIGHT_SOUTH
    bgs_target |= bgs_faint_south * bgs_mask.BGS_FAINT_SOUTH
    bgs_target |= bgs_wise_south * bgs_mask.BGS_WISE_SOUTH

    # BGS targets, north.
    bgs_target |= bgs_bright_north * bgs_mask.BGS_BRIGHT_NORTH
    bgs_target |= bgs_faint_north * bgs_mask.BGS_FAINT_NORTH
    bgs_target |= bgs_wise_north * bgs_mask.BGS_WISE_NORTH

    # BGS targets, combined.
    bgs_target |= bgs_bright * bgs_mask.BGS_BRIGHT
    bgs_target |= bgs_faint * bgs_mask.BGS_FAINT
    bgs_target |= bgs_wise * bgs_mask.BGS_WISE

    # ADM set fraction of the BGS_FAINT targets to BGS_FAINT_HIP.
    bgs_target |= bgs_faint_hip * bgs_mask.BGS_FAINT_HIP

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

    # Add MWS FAINT blue/red split. Only ever set in v1.1 of desitarget.
#    mws_target |= mws_faint_blue * mws_mask.MWS_FAINT_BLUE
#    mws_target |= mws_faint_blue_n * mws_mask.MWS_FAINT_BLUE_NORTH
#    mws_target |= mws_faint_blue_s * mws_mask.MWS_FAINT_BLUE_SOUTH
#    mws_target |= mws_faint_red * mws_mask.MWS_FAINT_RED
#    mws_target |= mws_faint_red_n * mws_mask.MWS_FAINT_RED_NORTH
#    mws_target |= mws_faint_red_s * mws_mask.MWS_FAINT_RED_SOUTH

    # Are any BGS or MWS bit set?  Tell desi_target too.
    desi_target |= (bgs_target != 0) * desi_mask.BGS_ANY
    desi_target |= (mws_target != 0) * desi_mask.MWS_ANY

    return desi_target, bgs_target, mws_target


def apply_cuts_gaia(numproc=4, survey='main', nside=None, pixlist=None,
                    test=False):
    """Gaia-only-based target selection, return target mask arrays.

    Parameters
    ----------
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    survey : :class:`str`, defaults to ``'main'``
        Which target masks yaml file and target selection cuts to use.
        Options are ``'main'`` and ``'svX``' (where X is 1, 2, 3 etc.)
        for the main survey and different iterations of SV, respectively.
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPix nside used with `pixlist` and `bundlefiles`.
    pixlist : :class:`list` or `int`, optional, defaults to `None`
        Only return targets in a set of (NESTED) HEALpixels at `nside`.
        Useful for parallelizing, as input files will only be processed
        if they touch a pixel in the passed list.
    test : :class:`bool`, optional, defaults to ``False``
        If ``True``, then we're running unit tests and don't have to find
        and read every possible Gaia file when calling
        :func:`~desitarget.cuts.apply_cuts_gaia`.

    Returns
    -------
    :class:`~numpy.ndarray`
        desi_target selection bitmask flags for each object.
    :class:`~numpy.ndarray`
        bgs_target selection bitmask flags for each object.
    :class:`~numpy.ndarray`
        mws_target selection bitmask flags for each object.
    :class:`~numpy.ndarray`
        numpy structured array of Gaia sources that were read in from
        file for the passed pixel constraints (or no pixel constraints).

    Notes
    -----
        - May take a long time if no pixel constraints are passed.
        - Only run on Gaia-only target selections.
        - The environment variable $GAIA_DIR must be set.

    See desitarget.svX.svX_targetmask.desi_mask or
    desitarget.targetmask.desi_mask for bit definitions.
    """
    # ADM set different bits based on whether we're using the main survey
    # code or an iteration of SV.
    if survey == 'main':
        import desitarget.cuts as targcuts
        from desitarget.targetmask import desi_mask, mws_mask
    elif survey[:2] == 'sv':
        try:
            targcuts = import_module("desitarget.{}.{}_cuts".format(survey,
                                                                    survey))
            targmask = import_module("desitarget.{}.{}_targetmask".format(
                survey, survey))
        except ModuleNotFoundError:
            msg = 'Bitmask yaml or cuts do not exist for survey type {}'.format(
                survey)
            log.critical(msg)
            raise ModuleNotFoundError(msg)
        desi_mask, mws_mask = targmask.desi_mask, targmask.mws_mask
    else:
        msg = "survey must be either 'main'or 'svX', not {}!!!".format(survey)
        log.critical(msg)
        raise ValueError(msg)

    from desitarget.gfa import all_gaia_in_tiles
    # ADM No Gaia-only target classes are fainter than G of 19.
    # ADM or are north of dec=-30.
    gaiaobjs = all_gaia_in_tiles(maglim=19, numproc=numproc, allsky=True,
                                 mindec=-30, mingalb=0, addobjid=True,
                                 nside=nside, pixlist=pixlist, addparams=True,
                                 test=test)
    # ADM the convenience function we use adds an empty TARGETID
    # ADM field which we need to remove before finalizing.
    gaiaobjs = rfn.drop_fields(gaiaobjs, "TARGETID")

    # ADM the relevant input quantities.
    ra = gaiaobjs["RA"]
    dec = gaiaobjs["DEC"]
    ebv = gaiaobjs["EBV"]
    gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr, gaiagmag, \
        gaiabmag, gaiarmag, gaiaaen, gaiadupsource, Grr, gaiaparamssolved, \
        gaiabprpfactor, gaiasigma5dmax, galb = _prepare_gaia(gaiaobjs)

    # ADM determine if an object is a BACKUP target.
    primary = np.ones_like(gaiaobjs, dtype=bool)
    if survey == 'main':
        backup_bright, backup_faint, backup_very_faint, backup_giant = targcuts.isBACKUP(
            ra=ra, dec=dec, gaiagmag=gaiagmag,
            gaiabmag=gaiabmag, gaiarmag=gaiarmag,
            parallax=parallax, parallaxerr=parallaxerr,
            primary=primary)
    else:
        backup_bright, backup_faint, backup_very_faint = targcuts.isBACKUP(
            ra=ra, dec=dec, gaiagmag=gaiagmag,
            primary=primary)

    # ADM determine if a target is a Gaia-only standard.
    primary = np.ones_like(gaiaobjs, dtype=bool)
    std_faint, std_bright, std_wd = targcuts.isGAIA_STD(
        ra=ra, dec=dec, galb=galb, gaiaaen=gaiaaen, pmra=pmra, pmdec=pmdec,
        parallax=parallax, parallaxovererror=parallaxovererror, ebv=ebv,
        gaiabprpfactor=gaiabprpfactor, gaiasigma5dmax=gaiasigma5dmax,
        gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag,
        gaiadupsource=gaiadupsource, gaiaparamssolved=gaiaparamssolved,
        primary=primary, nside=nside, test=test)

    # ADM Construct the target flag bits.
    mws_target = backup_bright * mws_mask.BACKUP_BRIGHT
    mws_target |= backup_faint * mws_mask.BACKUP_FAINT
    mws_target |= backup_very_faint * mws_mask.BACKUP_VERY_FAINT
    if survey == 'main':
        mws_target |= backup_giant * mws_mask.BACKUP_GIANT
    mws_target |= std_faint * mws_mask.GAIA_STD_FAINT
    mws_target |= std_bright * mws_mask.GAIA_STD_BRIGHT
    mws_target |= std_wd * mws_mask.GAIA_STD_WD

    bgs_target = np.zeros_like(mws_target)

    # ADM remember that desi_target must have MWS_ANY set as BACKUP
    # ADM targets fall under the auspices of the MWS program.
    desi_target = (mws_target != 0) * desi_mask.MWS_ANY

    return desi_target, bgs_target, mws_target, gaiaobjs


def apply_cuts(objects, qso_selection='randomforest',
               tcnames=["ELG", "QSO", "LRG", "MWS", "BGS", "STD"],
               qso_optical_cuts=False, survey='main', resolvetargs=True,
               mask=True):
    """Perform target selection on objects, returning target mask arrays.

    Parameters
    ----------
    objects : :class:`~numpy.ndarray` or `str`
        numpy structured array with UPPERCASE columns needed for
        target selection, OR a string tractor/sweep filename.
    qso_selection : :class:`str`, optional, default is ``'randomforest'``
        The algorithm to use for QSO selection; valid options are
        ``'colorcuts'`` and ``'randomforest'``
    tcnames : :class:`list`, defaults to running all target classes
        A list of strings, e.g. ['QSO','LRG']. If passed, process targets
        only for those specific classes. A useful speed-up when testing.
        Options include ["ELG", "QSO", "LRG", "MWS", "BGS", "STD"].
    qso_optical_cuts : :class:`boolean` defaults to ``False``
        Apply just optical color-cuts when selecting QSOs with
        ``qso_selection="colorcuts"``.
    survey : :class:`str`, defaults to ``'main'``
        Which target masks yaml file and target selection cuts to use.
        Options are ``'main'`` and ``'svX``' (where X is 1, 2, 3 etc.)
        for the main survey and different iterations of SV, respectively.
    resolvetargs : :class:`boolean`, optional, defaults to ``True``
        If ``True``, if `objects` consists of all northern (southern)
        sources then only apply the northern (southern) cuts.
    mask : :class:`boolean`, optional, defaults to ``True``
        Send ``False`` to turn off any masking cuts based on the
        `MASKBITS` column. Default is to always mask using `MASKBITS`.

    Returns
    -------
    :class:`~numpy.ndarray`
        (desi_target, bgs_target, mws_target) where each element is
        an ndarray of target selection bitmask flags for each object.

    Notes
    -----
    - If ``objects`` is an astropy Table with lowercase column names,
      they will be converted to UPPERCASE in-place, thus modifying the
      input table. To avoid this, pass in ``objects.copy()`` instead.
    - See :mod:`desitarget.targetmask` for the definition of each bit.
    """
    # - Check if objects is a filename instead of the actual data.
    if isinstance(objects, str):
        objects = io.read_tractor(objects)

    # - ensure uppercase column names if astropy Table.
    if isinstance(objects, (Table, Row)):
        for col in list(objects.columns.values()):
            if not col.name.isupper():
                col.name = col.name.upper()

    # ADM as we need the column names.
    colnames = _get_colnames(objects)

    # ADM process the Legacy Surveys columns for Target Selection.
    photsys_north, photsys_south, obs_rflux, gflux, rflux, zflux,               \
        w1flux, w2flux, gfiberflux, rfiberflux, zfiberflux,                     \
        gfibertotflux, rfibertotflux, zfibertotflux,                            \
        objtype, release, ra, dec, gfluxivar, rfluxivar, zfluxivar, w1fluxivar, \
        gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,                   \
        gfracmasked, rfracmasked, zfracmasked,                                  \
        gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,                \
        gsnr, rsnr, zsnr, w1snr, w2snr, dchisq, deltaChi2, maskbits, refcat =   \
        _prepare_optical_wise(objects, mask=mask)

    # Process the Gaia inputs for target selection.
    gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr, gaiagmag,      \
        gaiabmag, gaiarmag, gaiaaen, gaiadupsource, Grr, gaiaparamssolved,      \
        gaiabprpfactor, gaiasigma5dmax, galb =                                  \
        _prepare_gaia(objects, colnames=colnames)

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
    elif survey[:2] == 'sv':
        targcuts = import_module("desitarget.{}.{}_cuts".format(survey, survey))
    else:
        msg = "survey must be either 'main'or 'svX', not {}!!!".format(survey)
        log.critical(msg)
        raise ValueError(msg)

    desi_target, bgs_target, mws_target = targcuts.set_target_bits(
        photsys_north, photsys_south, obs_rflux,
        gflux, rflux, zflux, w1flux, w2flux, gfiberflux, rfiberflux, zfiberflux,
        gfibertotflux, rfibertotflux, zfibertotflux, objtype, release,
        ra, dec, gfluxivar, rfluxivar, zfluxivar, w1fluxivar,
        gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,
        gfracmasked, rfracmasked, zfracmasked,
        gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,
        gsnr, rsnr, zsnr, w1snr, w2snr, deltaChi2, dchisq,
        gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr,
        gaiagmag, gaiabmag, gaiarmag, gaiaaen, gaiadupsource,
        gaiaparamssolved, gaiabprpfactor, gaiasigma5dmax, galb,
        tcnames, qso_optical_cuts, qso_selection,
        maskbits, Grr, refcat, primary, resolvetargs=resolvetargs,
        )

    return desi_target, bgs_target, mws_target


qso_selection_options = ['colorcuts', 'randomforest']


def select_targets(infiles, numproc=4, qso_selection='randomforest',
                   gaiasub=False, nside=None, pixlist=None, bundlefiles=None,
                   extra=None, radecbox=None, radecrad=None, mask=True,
                   tcnames=["ELG", "QSO", "LRG", "MWS", "BGS", "STD"],
                   survey='main', resolvetargs=True, backup=True,
                   return_infiles=False, test=False):
    """Process input files in parallel to select targets.

    Parameters
    ----------
    infiles : :class:`list` or `str`
        List of input filenames (tractor or sweep) OR a single filename.
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    qso_selection : :class:`str`, optional, default is ``'randomforest'``
        The algorithm to use for QSO selection; valid options are
        ``'colorcuts'`` and ``'randomforest'``.
    gaiasub : :class:`boolean`, optional, defaults to ``False``
        If ``True``, substitute Gaia EDR3 proper motion and parallax
        columns over the sweeps values. If ``True``, then the GAIA_DIR
        environment variable must be set to find the Gaia sweeps files.
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPixel nside used with `pixlist` and `bundlefiles`.
    pixlist : :class:`list` or `int`, optional, defaults to `None`
        Only return targets in (NESTED) HEALpixels at `nside`. Also
        useful for parallelizing as input files will only be processed
        if they touch a pixel in the passed list.
    bundlefiles : :class:`int`, defaults to `None`
        If not `None`, then instead of selecting targets, print a slurm
        script that balances the input file distribution at `bundlefiles`
        files per node. So, e.g., if `bundlefiles` is 100 commands would
        be printed with `pixlist` values to pass to the code to pack at
        about 100 files per node across all of the passed `infiles`.
    extra : :class:`str`, optional
        Extra command line flags to be passed to the executable lines in
        the output slurm script. Used in conjunction with `bundlefiles`.
    radecbox : :class:`list`, defaults to `None`
        4-entry coordinate list [ramin, ramax, decmin, decmax].
        Only targets inside this "box" region will be processed.
    radecrad : :class:`list`, defaults to `None`
        3-entry coordinate list [ra, dec, radius] for RA/Dec/radius in
        degrees. Only targets in this "circle" region will be processed.
    mask : :class:`boolean`, optional, defaults to ``True``
        Send ``False`` to turn off any masking cuts based on `MASKBITS`.
        The default behavior is to always mask using `MASKBITS`.
    tcnames : :class:`list`, defaults to running all target classes
        List of strings, e.g. ['QSO','LRG']. If passed, process targets
        only for those specific classes. A useful speed-up when testing.
        Options include ["ELG", "QSO", "LRG", "MWS", "BGS", "STD"].
    survey : :class:`str`, defaults to ``'main'``
        Which target masks yaml file and target selection cuts to use.
        Options are ``'main'`` and ``'svX``' (where X is 1, 2, 3 etc.)
        for the main survey and different iterations of SV, respectively.
    resolvetargs : :class:`boolean`, optional, defaults to ``True``
        If ``True``, resolve targets into northern targets in northern
        regions and southern targets in southern regions.
    backup : :class:`boolean`, optional, defaults to ``True``
        If ``True``, also run the Gaia-only BACKUP_BRIGHT/FAINT targets.
    return_infiles : :class:`boolean`, optional, defaults to ``False``
        If ``True``, return the filenamess from `infiles` that were
        processed. Useful when running with `pixlist`, `radecbox` or
        `radecrad` to see which files were actually required.
    test : :class:`bool`, optional, defaults to ``False``
        If ``True``, then we're running unit tests and don't have to
        find and read every possible Gaia file when calling
        :func:`~desitarget.cuts.apply_cuts_gaia`.

    Returns
    -------
    :class:`~numpy.ndarray`
        The subset of input targets which pass the cuts, including extra
        columns for ``DESI_TARGET``, ``BGS_TARGET``, and ``MWS_TARGET``
        target selection bitmasks.
    :class:`list`, only returned if `return_infiles` is ``True``
        A list of the input files that actually needed to be processed.

    Notes
    -----
        - if numproc==1, use serial code instead of parallel.
        - only one of `pixlist`, `radecbox`, `radecrad` should be passed.
          They all denote regions on the sky, using different formalisms.
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
        log.info("Run targets in box inside [RAmin, RAmax, Decmin, Decmax]={}"
                 .format(radecbox))

    # ADM if radecrad was sent, determine which pixels touch the box.
    if radecrad is not None:
        nside = pixarea2nside(cap_area(np.array(radecrad[2])))
        pixlist = hp_in_cap(nside, radecrad)
        log.info("Run targets in cap bounded by [centerRA, centerDec, radius]={}"
                 .format(radecrad))

    # ADM if the pixlist or bundlefiles option was sent, we'll need to
    # ADM know which HEALPixels touch each file.
    if pixlist is not None:
        filesperpixel, _, _ = sweep_files_touch_hp(
            nside, pixlist, infiles)

    # ADM if the bundlefiles option was sent, call the packing code.
    if bundlefiles is not None:
        prefix = "targets"
        if survey != "main":
            prefix = "{}_targets".format(survey)
        # ADM determine if one or two input directories were passed.
        surveydirs = list(set([os.path.dirname(fn) for fn in infiles]))
        bundle_bricks([0], bundlefiles, nside, gather=False, extra=extra,
                      prefix=prefix, surveydirs=surveydirs)
        if return_infiles:
            return None, None
        return None

    # ADM restrict input files to a set of HEALPixels, if requested.
    if pixlist is not None:
        # ADM a hack to ensure we have the correct targeting data model
        # ADM outside of the Legacy Surveys footprint.
        dummy = infiles[0]
        infiles = list(np.unique(np.hstack([filesperpixel[pix] for pix in pixlist])))

        if len(infiles) == 0:
            log.info('ZERO sweep files in passed pixel list!!!')
            log.info('Run with dummy sweep file to write Gaia-only objects...')
            infiles = [dummy]
        log.info("Processing files in (nside={}, pixel numbers={}) HEALPixels"
                 .format(nside, pixlist))

    # ADM a little more information if we're slurming across nodes.
    if os.getenv('SLURMD_NODENAME') is not None:
        log.info('Running on Node {}'.format(os.getenv('SLURMD_NODENAME')))

    def _finalize_targets(objects, desi_target, bgs_target, mws_target,
                          gaiadr=None):
        # - desi_target includes BGS_ANY and MWS_ANY, so we can filter
        # - on desi_target != 0
        keep = (desi_target != 0)
        objects = objects[keep]
        desi_target = desi_target[keep]
        bgs_target = bgs_target[keep]
        mws_target = mws_target[keep]
        if gaiadr is not None:
            gaiadr = gaiadr[keep]

        # - Add *_target mask columns
        targets = finalize(objects, desi_target, bgs_target, mws_target,
                           survey=survey, darkbright=True, gaiadr=gaiadr)

        # ADM resolve any duplicates between imaging data releases.
        if resolvetargs and gaiadr is None:
            targets = resolve(targets)

        return targets

    # - functions to run on every brick/sweep file
    def _select_targets_file(filename):
        '''Returns targets in filename that pass the cuts'''
        objects = io.read_tractor(filename, gaiasub=gaiasub)
        desi_target, bgs_target, mws_target = apply_cuts(
            objects, qso_selection=qso_selection, tcnames=tcnames,
            survey=survey, resolvetargs=resolvetargs, mask=mask
        )

        return _finalize_targets(objects, desi_target, bgs_target, mws_target)

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
            log.info('{}/{} files; {:.1f} secs/file; {:.1f} total mins elapsed'
                     .format(nbrick, len(infiles), rate, elapsed/60.))

        nbrick[...] += 1    # this is an in-place modification.
        return result

    # - Parallel process input files.
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            targets = pool.map(_select_targets_file, infiles,
                               reduce=_update_status)
    else:
        targets = list()
        for x in infiles:
            targets.append(_update_status(_select_targets_file(x)))

    targets = np.concatenate(targets)

    if backup:
        # ADM also process Gaia-only targets.
        log.info('Retrieve extra Gaia-only (backup) objects...t = {:.1f} mins'
                 .format((time()-t0)/60))

        # ADM force to numproc<=4 for I/O limited (Gaia-only) processes.
        numproc4 = numproc
        if numproc4 > 4:
            log.info('Forcing numproc to 4 for I/O limited parts of code')
            numproc4 = 4

        # ADM set the target bits that are based only on Gaia.
        gaia_desi_target, gaia_bgs_target, gaia_mws_target, gaiaobjs = \
            apply_cuts_gaia(numproc=numproc4, survey=survey, nside=nside,
                            pixlist=pixlist, test=test)

        # ADM it's possible that somebody could pass HEALPixels that
        # ADM contain no additional targets.
        if len(gaiaobjs) > 0:
            # ADM determine the Gaia Data Release.
            gaiadr = gaia_dr_from_ref_cat(gaiaobjs["REF_CAT"])

            # ADM add the relevant bits and IDs to the Gaia targets.
            # ADM first set up empty DESI and BGS columns.
            gaiatargs = _finalize_targets(
                gaiaobjs, gaia_desi_target, gaia_bgs_target, gaia_mws_target,
                gaiadr=gaiadr)

            # ADM make the Gaia-only data structure resemble the targets.
            gaiatargets = np.zeros(len(gaiatargs), dtype=targets.dtype)
            sc = set(
                gaiatargs.dtype.names).intersection(set(targets.dtype.names))
            for col in sc:
                gaiatargets[col] = gaiatargs[col]
            # ADM Gaia-only target always have PHOTSYS="G".
            gaiatargets["PHOTSYS"] = "G"

            # ADM merge any Legacy Surveys targets with Gaia targets.
            if len(infiles) > 0:
                targets = np.concatenate([targets, gaiatargets])
            else:
                targets = gaiatargets

    # ADM it's possible that somebody could pass HEALPixels that
    # ADM contain no targets, in which case exit (somewhat) gracefully.
    if len(targets) == 0:
        log.warning('ZERO targets for passed file list or region!!!')
        if return_infiles:
            return targets, infiles
        return targets

    # ADM restrict to only targets in a set of HEALPixels, if requested.
    if pixlist is not None:
        ii = is_in_hp(targets, nside, pixlist)
        targets = targets[ii]

    # ADM restrict to only targets in an RA, Dec box, if requested.
    if radecbox is not None:
        ii = is_in_box(targets, radecbox)
        targets = targets[ii]

    # ADM restrict to only targets in an RA, Dec, radius cap, if needed.
    if radecrad is not None:
        ii = is_in_cap(targets, radecrad)
        targets = targets[ii]

    if return_infiles:
        return targets, infiles
    return targets
