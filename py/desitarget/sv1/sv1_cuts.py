"""
desitarget.sv1.sv1_cuts
=======================

Target Selection for DESI Survey Validation derived from `the SV wiki`_.

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (*e.g.* LRG, ELG or QSO).

.. _`the Gaia data model`: https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
.. _`the SV wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/SurveyValidation
"""

import sys
import numpy as np
import warnings

from time import time
from pkg_resources import resource_filename

from desitarget.cuts import _getColors, _psflike, _check_BGS_targtype_sv
from desitarget.cuts import shift_photo_north
from desitarget.gaiamatch import is_in_Galaxy

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()


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
    - Current version (10/24/19) is version 114 on `the SV wiki`_.
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

    # ADM faint targets are 16 < G < 18.5.
    isbackupfaint &= gaiagmag >= 16
    isbackupfaint &= gaiagmag < 18.5
    # ADM and are "far from" the Galaxy.
    isbackupfaint &= ~in_gal

    # ADM very faint targets are 18.5 < G < 19.
    isbackupveryfaint &= gaiagmag >= 18.5
    isbackupveryfaint &= gaiagmag < 19
    # ADM and are "far from" the Galaxy.
    isbackupveryfaint &= ~in_gal

    return isbackupbright, isbackupfaint, isbackupveryfaint


def isLRG(gflux=None, rflux=None, zflux=None, w1flux=None,
          zfiberflux=None, rflux_snr=None, zflux_snr=None, w1flux_snr=None,
          primary=None, south=True):
    """Target Definition of LRG. Returns a boolean array.

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
    :class:`array_like`
        ``True`` for a 4 PASS nominal optical + nominal IR LRG.
    :class:`array_like`
        ``True`` for a 4 PASS object in the LRG SV superset.
    :class:`array_like`
        ``True`` for an 8 PASS nominal optical + nominal IR LRG.
    :class:`array_like`
        ``True`` for an 8 PASS object in the LRG SV superset.

    Notes
    -----
    - Current version (10/14/19) is version 104 on `the SV wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    # ADM LRG SV targets.
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    lrg &= notinLRG_mask(
        primary=primary, rflux=rflux, zflux=zflux, w1flux=w1flux,
        zfiberflux=zfiberflux,
        rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr
    )

    # ADM pass the lrg that pass cuts as primary, to restrict to the
    # ADM sources that weren't in a mask/logic cut.
    lrg_all, lrginit4, lrgsuper4, lrginit8, lrgsuper8 = isLRG_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
        zfiberflux=zfiberflux, south=south, primary=lrg
    )

    return lrg_all, lrginit4, lrgsuper4, lrginit8, lrgsuper8


def notinLRG_mask(primary=None, rflux=None, zflux=None, w1flux=None,
                  zfiberflux=None,
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

    # ADM to maintain backwards-compatibility with mocks.
    if zfiberflux is None:
        log.warning('Setting zfiberflux to zflux!!!')
        zfiberflux = zflux.copy()

    lrg &= (rflux_snr > 0) & (rflux > 0)   # ADM quality in r.
    lrg &= (zflux_snr > 0) & (zflux > 0) & (zfiberflux > 0)   # ADM quality in z.
    lrg &= (w1flux_snr > 4) & (w1flux > 0)  # ADM quality in W1.

    return lrg


def isLRG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 zfiberflux=None, south=True, primary=None):
    """See :func:`~desitarget.sv1.sv1_cuts.isLRG` for details.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()
    lrginit, lrgsuper = np.tile(primary, [2, 1])

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


def isfiller(gflux=None, rflux=None, zflux=None, w1flux=None,
             gflux_snr=None, rflux_snr=None, zflux_snr=None, w1flux_snr=None,
             rfiberflux=None, south=True, primary=None):
    """Definition of LRG-like low-z filler sample selection.

    Parameters
    ----------
    south: boolean, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS)
        if ``south=False``, otherwise use cuts appropriate to the
        Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` if the object is an LRG-like low-z filler target.

    Notes
    -----
    - Current version (09/03/19) is version 90 on `the SV wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    filler = primary.copy()

    # ADM to maintain backwards-compatibility with mocks.
    if rfiberflux is None:
        log.warning('Setting rfiberflux to rflux!!!')
        rfiberflux = rflux.copy()

    filler &= (gflux_snr > 0) & (gflux > 0)    # ADM quality in g.
    filler &= (rflux_snr > 0) & (rflux > 0) & (rfiberflux > 0)   # ADM quality in r.
    filler &= (zflux_snr > 0) & (zflux > 0)    # ADM quality in z.
    filler &= (w1flux_snr > 4) & (w1flux > 0)  # ADM quality in W1.

    # ADM safe as these fluxes are set to > 0
    gmag = 22.5 - 2.5 * np.log10(gflux.clip(1e-7))
    rmag = 22.5 - 2.5 * np.log10(rflux.clip(1e-7))
    zmag = 22.5 - 2.5 * np.log10(zflux.clip(1e-7))
    w1mag = 22.5 - 2.5 * np.log10(w1flux.clip(1e-7))
    rfibermag = 22.5 - 2.5 * np.log10(rfiberflux.clip(1e-7))

    # North and South currently have the same cuts
    filler &= (rmag > 19.5) & (rmag < 21) & (rfibermag < 22)  # magnitude limits
    filler &= zmag - w1mag > 0.8 * (rmag-zmag) - 0.8          # non-stellar cut
    # high-z cuts
    filler &= (rmag - w1mag) < 0.5 * (gmag - rmag) + 1.4
    filler &= (rmag - w1mag) < 1.7 * ((gmag - rmag) - 0.8) + 1.2
    filler &= (gmag - rmag) > 0.7                             # low-z cut
    filler &= (rmag - zmag) > 0.25 * (rmag + 0.6) - 4.5       # sliding cut
    # Remove overlap with BGS SV selection, these cuts don't apply to the final selection
    filler &= (rmag > 20.5) & (rfibermag > 21.05)

    return filler


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
    - Current version (11/05/18) is version 24 on `the SV wiki`_.
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


def isQSO_cuts(gflux=None, rflux=None, zflux=None,
               w1flux=None, w2flux=None, w1snr=None, w2snr=None,
               dchisq=None, maskbits=None, objtype=None,
               primary=None, south=True):
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
    - Current version (09/25/19) is version 100 on `the SV wiki`_.
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

    # ADM relaxed morphology cut for SV.
    # ADM we never target sources with dchisq[..., 0] = 0, so force
    # ADM those to have large values of morph2 to avoid divide-by-zero.
    d1, d0 = dchisq[..., 1], dchisq[..., 0]
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
        qso &= w1snr > 3.5
        qso &= w2snr > 2.5

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

    # ADM never target sources that are far too bright (mag < 0).
    # ADM this guards against overflow warnings in powers.
    qso &= (gflux < 1e9) & (rflux < 1e9) & (zflux < 1e9)
    gflux[~qso] = 1e9
    rflux[~qso] = 1e9
    zflux[~qso] = 1e9

    # ADM Create some composite fluxes.
    wflux = 0.75*w1flux + 0.25*w2flux
    grzflux = (gflux + 0.8*rflux + 0.5*zflux) / 2.3

    # ADM perform the magnitude cuts.
    if south:
        qso &= rflux > 10**((22.5-23.)/2.5)    # r<23.0 (different for SV)
    else:
        qso &= rflux > 10**((22.5-22.7)/2.5)    # r<22.7
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
        mainseq &= rflux**(1+1.53) > gflux * zflux**1.53 * 10**((-0.100+0.20)/2.5) 
        mainseq &= rflux**(1+1.53) < gflux * zflux**1.53 * 10**((+0.100+0.20)/2.5) 

    mainseq &= w2flux < w1flux * 10**(0.3/2.5)  # ADM W1 - W2 !(NOT) > 0.3
    qso &= ~mainseq

    return qso


def isQSO_color_high_z(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, south=True):
    """
    Color cut to select Highz QSO (z>~2.)
    """
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


def isQSO_randomforest(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                       objtype=None, release=None, dchisq=None, maskbits=None,
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
    - Current version (09/25/19) is version 100 on `the SV wiki`_.
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

    # ADM relaxed morphology cut for SV.
    # ADM we never target sources with dchisq[..., 0] = 0, so force
    # ADM those to have large values of morph2 to avoid divide-by-zero.
    d1, d0 = dchisq[..., 1], dchisq[..., 0]
    bigmorph = np.zeros_like(d0)+1e9
    dcs = np.divide(d1 - d0, d0, out=bigmorph, where=d0 != 0)
    if south:
        morph2 = dcs < 0.015
    else:
        morph2 = dcs < 0.015
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
                            0.75 - (r_Reduced - 20.0) * 0.05, 0.75)
            pcut[r_Reduced > 22.0] = 0.65 - 0.34 * (r_Reduced[r_Reduced > 22.0] - 22.0)
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


def isQSO_highz_faint(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                      objtype=None, release=None, dchisq=None, maskbits=None,
                      primary=None, south=True):
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
    - Current version (09/25/19) is version 100 on `the SV wiki`_.
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
            pcut = np.where(r_Reduced < 23.2,  0.50 + (r_Reduced-22.8)*.75, .80 + (r_Reduced-23.2)*.5) 

        # Add rf proba test result to "qso" mask
        qso[colorsReducedIndex] = (tmp_rf_proba >= pcut)

    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "numpy.ndarray"
    if nbEntries == 1:
        qso = qso[0]

    return qso


def isQSOz5_cuts(gflux=None, rflux=None, zflux=None,
                 gsnr=None, rsnr=None, zsnr=None,
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
    - Current version (10/24/19) is version 112 on `the SV wiki`_.
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
        # for bit in [1, 10, 12, 13]:
        for bit in [10, 12, 13]:
            qso &= ((maskbits & 2**bit) == 0)

    # ADM relaxed morphology cut for SV.
    # ADM we never target sources with dchisq[..., 0] = 0, so force
    # ADM those to have large values of morph2 to avoid divide-by-zero.
    d1, d0 = dchisq[..., 1], dchisq[..., 0]
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

    # ADM never target sources that are far too bright (mag < 0).
    # ADM this guards against overflow warnings in powers.
    qso &= (gflux < 1e9) & (rflux < 1e9) & (zflux < 1e9)
    gflux[~qso] = 1e9
    rflux[~qso] = 1e9
    zflux[~qso] = 1e9

    # ADM never target sources with negative W1 or z fluxes.
    qso &= (w1flux >= 0.) & (zflux >= 0.)
    # ADM now safe to update w1flux and zflux to avoid warnings.
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


def isBGS(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, rfiberflux=None,
          gnobs=None, rnobs=None, znobs=None, gfracmasked=None, rfracmasked=None, zfracmasked=None,
          gfracflux=None, rfracflux=None, zfracflux=None, gfracin=None, rfracin=None, zfracin=None,
          gfluxivar=None, rfluxivar=None, zfluxivar=None, maskbits=None, Grr=None,
          w1snr=None, gaiagmag=None, objtype=None, primary=None, south=True, targtype=None):
    """Definition of BGS target classes. Returns a boolean array.

    Parameters
    ----------
    south : :class:`boolean`, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
        otherwise use cuts appropriate to the Southern imaging survey (DECaLS).
    targtype : :class:`str`, optional, defaults to ``faint``
        Pass ``bright`` to use colors appropriate to the ``BGS_BRIGHT`` selection
        or ``faint`` to use colors appropriate to the ``BGS_FAINT`` selection
        or ``faint_ext`` to use colors appropriate to the ``BGS_FAINT_EXTENDED`` selection
        or ``lowq`` to use colors appropriate to the ``BGS_LOW_QUALITY`` selection
        or ``fibmag`` to use colors appropriate to the ``BGS_FIBER_MAGNITUDE`` selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a BGS target of type ``targtype``.

    Notes
    -----
    - Current version (10/14/19) is version 105 on `the SV wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    _check_BGS_targtype_sv(targtype)

    # ------ Bright Galaxy Survey
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()

    bgs &= notinBGS_mask(gflux=gflux, rflux=rflux, zflux=zflux, gnobs=gnobs, rnobs=rnobs, znobs=znobs, primary=primary,
                         gfracmasked=gfracmasked, rfracmasked=rfracmasked, zfracmasked=zfracmasked,
                         gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                         gfracin=gfracin, rfracin=rfracin, zfracin=zfracin, w1snr=w1snr,
                         gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar, Grr=Grr,
                         gaiagmag=gaiagmag, maskbits=maskbits, objtype=objtype, targtype=targtype)

    bgs &= isBGS_colors(rflux=rflux, rfiberflux=rfiberflux, south=south, targtype=targtype, primary=primary)

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

    # ADM to maintain backwards-compatibility with mocks.
    if rfiberflux is None:
        log.warning('Setting rfiberflux to rflux!!!')
        rfiberflux = rflux.copy()

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


def isELG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          gsnr=None, rsnr=None, zsnr=None, gfiberflux=None,
          gnobs=None, rnobs=None, znobs=None,
          maskbits=None, south=True, primary=None):
    """Definition of ELG target classes. Returns a boolean array.

    Parameters
    ----------
    south : :class:`boolean`, defaults to ``True``
        If ``False``, use cuts for the Northern imaging (BASS/MzLS)
        otherwise use cuts for the Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is an ELG target.

    Notes
    -----
    - Current version (10/14/19) is version 107 on `the SV wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    elg &= notinELG_mask(maskbits=maskbits, gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                         gnobs=gnobs, rnobs=rnobs, znobs=znobs, primary=primary)

    svgtot, svgfib, fdrgtot, fdrgfib = isELG_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
        gfiberflux=gfiberflux, south=south, primary=elg
    )

    return svgtot, svgfib, fdrgtot, fdrgfib


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

    # ADM to maintain backwards-compatibility with mocks.
    if gfiberflux is None:
        log.warning('Setting gfiberflux to gflux!!!')
        gfiberflux = gflux.copy()

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


def isMWS_main_sv(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                  gnobs=None, rnobs=None, gfracmasked=None, rfracmasked=None,
                  pmra=None, pmdec=None, parallax=None, obs_rflux=None, objtype=None,
                  gaia=None, gaiagmag=None, gaiabmag=None, gaiarmag=None,
                  gaiaaen=None, gaiadupsource=None, primary=None, south=True):
    """Set bits for main ``MWS`` SV targets.

    Args:
        see :func:`~desitarget.cuts.set_target_bits` for parameters.

    Returns:
        mask1 : array_like.
            ``True`` if and only if the object is a ``MWS_MAIN_BROAD`` target.
        mask2 : array_like.
            ``True`` if and only if the object is a ``MWS_MAIN_FAINT`` target.

    Notes:
        - as of 26/7/19, based on version 79 on `the wiki`_.
        - for SV, no astrometric selection or colour separation
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

    mws &= notinMWS_main_sv_mask(gaia=gaia, gfracmasked=gfracmasked, gnobs=gnobs,
                                 gflux=gflux, rfracmasked=rfracmasked, rnobs=rnobs,
                                 rflux=rflux, gaiadupsource=gaiadupsource, primary=primary)

    # ADM main targets are point-like based on DECaLS morphology
    # ADM and GAIA_ASTROMETRIC_NOISE.
    mws &= _psflike(objtype)
    mws &= gaiaaen < 3.0

    # ADM main targets are robs < 20
    mws &= obs_rflux > 10**((22.5-20.0)/2.5)

    # APC Degine faint and bright samples
    mws_faint = mws.copy()

    # ADM main targets are 16 <= r < 19
    mws &= rflux > 10**((22.5-19.0)/2.5)
    mws &= rflux <= 10**((22.5-16.0)/2.5)

    mws_faint &= rflux > 10**((22.5-20.0)/2.5)
    mws_faint &= rflux <= 10**((22.5-19.0)/2.5)

    return mws, mws_faint


def notinMWS_main_sv_mask(gaia=None, gfracmasked=None, gnobs=None, gflux=None,
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


def isMWS_nearby(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 objtype=None, gaia=None, primary=None,
                 pmra=None, pmdec=None, parallax=None, parallaxerr=None,
                 obs_rflux=None, gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for NEARBY Milky Way Survey targets.

    Parameters
    ----------
    see :func:`~desitarget.sv1.sv1_cuts.set_target_bits` for parameters.

    Returns
    -------
    mask : array_like.
        True if and only if the object is a MWS-NEARBY target.

    Notes
    -----
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
    # APC Gaia G mag of more than 16
    mws &= gaiagmag > 16.
    # ADM parallax cut corresponding to 100pc
    mws &= (parallax + parallaxerr) > 10.  # NB: "+" is correct

    return mws


def isMWS_WD(primary=None, gaia=None, galb=None, astrometricexcessnoise=None,
             pmra=None, pmdec=None, parallax=None, parallaxovererror=None,
             photbprpexcessfactor=None, astrometricsigma5dmax=None,
             gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for WHITE DWARF Milky Way Survey targets.

    Parameters
    ----------
    see :func:`~desitarget.sv1.sv1_cuts.set_target_bits` for other parameters.

    Returns
    -------
    mask : array_like.
        True if and only if the object is a MWS-WD target.

    Notes
    -----
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

    # ADM Finite proper motion to reject quasars.
    # ADM Inexplicably I'm getting a Runtimewarning here for
    # ADM a few values in the sqrt, so I'm catching it.
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


def set_target_bits(photsys_north, photsys_south, obs_rflux,
                    gflux, rflux, zflux, w1flux, w2flux,
                    gfiberflux, rfiberflux, zfiberflux,
                    objtype, release, gfluxivar, rfluxivar, zfluxivar,
                    gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,
                    gfracmasked, rfracmasked, zfracmasked,
                    gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,
                    gsnr, rsnr, zsnr, w1snr, w2snr, deltaChi2, dchisq,
                    gaia, pmra, pmdec, parallax, parallaxovererror, parallaxerr,
                    gaiagmag, gaiabmag, gaiarmag, gaiaaen, gaiadupsource,
                    gaiaparamssolved, gaiabprpfactor, gaiasigma5dmax, galb,
                    tcnames, qso_optical_cuts, qso_selection,
                    maskbits, Grr, primary, resolvetargs=True):
    """Perform target selection on parameters, return target mask arrays.

    Returns
    -------
    :class:`~numpy.ndarray`
        (desi_target, bgs_target, mws_target) where each element is
        an ndarray of target selection bitmask flags for each object.

    Notes
    -----
    - Units for Gaia quantities are the same as `the Gaia data model`_.
    - See :func:`~desitarget.cuts.set_target_bits` for parameters.
    """

    from desitarget.sv1.sv1_targetmask import desi_mask, bgs_mask, mws_mask

    # ADM if resolvetargs is set, limit to only sending north/south objects
    # ADM through north/south cuts.
    south_cuts = [False, True]
    if resolvetargs:
        # ADM if only southern objects were sent this will be [True], if
        # ADM only northern it will be [False], else it wil be both.
        south_cuts = list(set(photsys_south))

    # ADM initially set everything to arrays of False for the LRG selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    lrg_classes = [[~primary, ~primary, ~primary, ~primary, ~primary],
                   [~primary, ~primary, ~primary, ~primary, ~primary]]
    if "LRG" in tcnames:
        # ADM run the LRG target types (potentially) for both north and south.
        for south in south_cuts:
            lrg_classes[int(south)] = isLRG(
                primary=primary, south=south,
                gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                zfiberflux=zfiberflux, rflux_snr=rsnr, zflux_snr=zsnr,
                w1flux_snr=w1snr
            )
    lrg_north, lrginit_n_4, lrgsup_n_4, lrginit_n_8, lrgsup_n_8 = lrg_classes[0]
    lrg_south, lrginit_s_4, lrgsup_s_4, lrginit_s_8, lrgsup_s_8 = lrg_classes[1]

    # ADM initialize arrays to False for the filler selection;
    # ADM the zeroth element stores the northern bits (south=False).
    filler_classes = [~primary, ~primary]
    if "LRG" in tcnames:
        for south in south_cuts:
            filler_classes[int(south)] = isfiller(
                primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
                w1flux=w1flux, gflux_snr=gsnr, rflux_snr=rsnr, zflux_snr=zsnr,
                w1flux_snr=w1snr, rfiberflux=rfiberflux, south=south
            )
    filler_n, filler_s = filler_classes

    # ADM combine LRG target bits for an LRG target based on any imaging
    lrg = (lrg_north & photsys_north) | (lrg_south & photsys_south)
    lrginit_4 = (lrginit_n_4 & photsys_north) | (lrginit_s_4 & photsys_south)
    lrgsup_4 = (lrgsup_n_4 & photsys_north) | (lrgsup_s_4 & photsys_south)
    lrginit_8 = (lrginit_n_8 & photsys_north) | (lrginit_s_8 & photsys_south)
    lrgsup_8 = (lrgsup_n_8 & photsys_north) | (lrgsup_s_8 & photsys_south)
    filler = (filler_n & photsys_north) | (filler_s & photsys_south)

    # ADM initially set everything to arrays of False for the ELG selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    elg_classes = [[~primary, ~primary, ~primary, ~primary],
                   [~primary, ~primary, ~primary, ~primary]]
    if "ELG" in tcnames:
        for south in south_cuts:
            elg_classes[int(south)] = isELG(
                primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
                gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, gfiberflux=gfiberflux,
                gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits,
                south=south
            )

    elgsvgtot_n, elgsvgfib_n, elgfdrgtot_n, elgfdrgfib_n = elg_classes[0]
    elgsvgtot_s, elgsvgfib_s, elgfdrgtot_s, elgfdrgfib_s = elg_classes[1]

    # ADM combine ELG target bits for an ELG target based on any imaging.
    elg_n = elgsvgtot_n | elgsvgfib_n | elgfdrgtot_n | elgfdrgfib_n
    elg_s = elgsvgtot_s | elgsvgfib_s | elgfdrgtot_s | elgfdrgfib_s
    elg = (elg_n & photsys_north) | (elg_s & photsys_south)
    elgsvgtot = (elgsvgtot_n & photsys_north) | (elgsvgtot_s & photsys_south)
    elgsvgfib = (elgsvgfib_n & photsys_north) | (elgsvgfib_s & photsys_south)
    elgfdrgtot = (elgfdrgtot_n & photsys_north) | (elgfdrgtot_s & photsys_south)
    elgfdrgfib = (elgfdrgfib_n & photsys_north) | (elgfdrgfib_s & photsys_south)

    # ADM initially set everything to arrays of False for the QSO selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    qso_classes = [[~primary, ~primary, ~primary, ~primary, ~primary],
                   [~primary, ~primary, ~primary, ~primary, ~primary]]
    if "QSO" in tcnames:
        for south in south_cuts:
            qso_store = []
            qso_store.append(
                isQSO_cuts(
                    primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                    w1flux=w1flux, w2flux=w2flux,
                    dchisq=dchisq, maskbits=maskbits,
                    objtype=objtype, w1snr=w1snr, w2snr=w2snr,
                    south=south
                )
            )
            # ADM SV mock selection needs to apply only the color cuts
            # ADM and ignore the Random Forest selections.
            if qso_selection == 'colorcuts':
                qso_store.append(~primary)
                qso_store.append(~primary)
            else:
                qso_store.append(
                    isQSO_randomforest(
                        primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                        w1flux=w1flux, w2flux=w2flux,
                        dchisq=dchisq, maskbits=maskbits,
                        objtype=objtype, south=south
                    )
                )
                qso_store.append(
                    isQSO_highz_faint(
                        primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                        w1flux=w1flux, w2flux=w2flux,
                        dchisq=dchisq, maskbits=maskbits,
                        objtype=objtype, south=south
                    )
                )
            qso_store.append(
                isQSO_color_high_z(
                    gflux=gflux, rflux=rflux, zflux=zflux,
                    w1flux=w1flux, w2flux=w2flux, south=south
                )
            )
            qso_store.append(
                isQSOz5_cuts(
                    primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
                    gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                    w1flux=w1flux, w2flux=w2flux, w1snr=w1snr, w2snr=w2snr,
                    dchisq=dchisq, maskbits=maskbits, objtype=objtype,
                    south=south
                )
            )
            qso_classes[int(south)] = qso_store
    qsocolor_north, qsorf_north, qsohizf_north, qsocolor_high_z_north, qsoz5_north = qso_classes[0]
    qsocolor_south, qsorf_south, qsohizf_south, qsocolor_high_z_south, qsoz5_south = qso_classes[1]

    # ADM combine quasar target bits for a quasar target based on any imaging.
    qsocolor_highz_north = (qsocolor_north & qsocolor_high_z_north)
    qsorf_highz_north = (qsorf_north & qsocolor_high_z_north)
    qsocolor_lowz_north = (qsocolor_north & ~qsocolor_high_z_north)
    qsorf_lowz_north = (qsorf_north & ~qsocolor_high_z_north)
    qso_north = (qsocolor_lowz_north | qsorf_lowz_north | qsocolor_highz_north
                 | qsorf_highz_north | qsohizf_north | qsoz5_north)

    qsocolor_highz_south = (qsocolor_south & qsocolor_high_z_south)
    qsorf_highz_south = (qsorf_south & qsocolor_high_z_south)
    qsocolor_lowz_south = (qsocolor_south & ~qsocolor_high_z_south)
    qsorf_lowz_south = (qsorf_south & ~qsocolor_high_z_south)
    qso_south = (qsocolor_lowz_south | qsorf_lowz_south | qsocolor_highz_south
                 | qsorf_highz_south | qsohizf_south | qsoz5_south)

    qso = (qso_north & photsys_north) | (qso_south & photsys_south)
    qsocolor_highz = (qsocolor_highz_north & photsys_north) | (qsocolor_highz_south & photsys_south)
    qsorf_highz = (qsorf_highz_north & photsys_north) | (qsorf_highz_south & photsys_south)
    qsocolor_lowz = (qsocolor_lowz_north & photsys_north) | (qsocolor_lowz_south & photsys_south)
    qsorf_lowz = (qsorf_lowz_north & photsys_north) | (qsorf_lowz_south & photsys_south)
    qsohizf = (qsohizf_north & photsys_north) | (qsohizf_south & photsys_south)
    qsoz5 = (qsoz5_north & photsys_north) | (qsoz5_south & photsys_south)

    # ADM initially set everything to arrays of False for the BGS selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    bgs_classes = [[~primary, ~primary, ~primary, ~primary, ~primary],
                   [~primary, ~primary, ~primary, ~primary, ~primary]]
    if "BGS" in tcnames:
        for south in south_cuts:
            bgs_store = []
            for targtype in ["bright", "faint", "faint_ext", "lowq", "fibmag"]:
                bgs_store.append(
                    isBGS(
                        gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                        rfiberflux=rfiberflux, gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                        gfracmasked=gfracmasked, rfracmasked=rfracmasked, zfracmasked=zfracmasked,
                        gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                        gfracin=gfracin, rfracin=rfracin, zfracin=zfracin,
                        gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
                        maskbits=maskbits, Grr=Grr, w1snr=w1snr, gaiagmag=gaiagmag,
                        objtype=objtype, primary=primary, south=south, targtype=targtype
                    )
                )
            bgs_classes[int(south)] = bgs_store
    bgs_bright_north, bgs_faint_north, bgs_faint_ext_north, bgs_lowq_north, bgs_fibmag_north = bgs_classes[0]
    bgs_bright_south, bgs_faint_south, bgs_faint_ext_south, bgs_lowq_south, bgs_fibmag_south = bgs_classes[1]

    # ADM combine BGS targeting bits for a BGS selected in any imaging
    bgs_bright = (bgs_bright_north & photsys_north) | (bgs_bright_south & photsys_south)
    bgs_faint = (bgs_faint_north & photsys_north) | (bgs_faint_south & photsys_south)
    bgs_faint_ext = (bgs_faint_ext_north & photsys_north) | (bgs_faint_ext_south & photsys_south)
    bgs_lowq = (bgs_lowq_north & photsys_north) | (bgs_lowq_south & photsys_south)
    bgs_fibmag = (bgs_fibmag_north & photsys_north) | (bgs_fibmag_south & photsys_south)

    # ADM initially set everything to arrays of False for the MWS selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    mws_classes = [[~primary, ~primary], [~primary, ~primary]]
    mws_nearby = ~primary
    if "MWS" in tcnames:
        mws_nearby = isMWS_nearby(
            gaia=gaia, gaiagmag=gaiagmag, parallax=parallax,
            parallaxerr=parallaxerr
        )
        # ADM run the MWS_MAIN target types for both north and south
        for south in south_cuts:
            mws_classes[int(south)] = isMWS_main_sv(
                    gaia=gaia, gaiaaen=gaiaaen, gaiadupsource=gaiadupsource,
                    gflux=gflux, rflux=rflux, obs_rflux=obs_rflux, objtype=objtype,
                    gnobs=gnobs, rnobs=rnobs,
                    gfracmasked=gfracmasked, rfracmasked=rfracmasked,
                    pmra=pmra, pmdec=pmdec, parallax=parallax,
                    primary=primary, south=south
            )
    mws_n, mws_faint_n = mws_classes[0]
    mws_s, mws_faint_s = mws_classes[1]

    # ADM treat the MWS WD selection specially, as we have to run the
    # ADM white dwarfs for standards
    # APC Science WDs now enter as secondary targets, so in principle the
    # APC assignment std_wd = mws_wd could be done here rather than below.
    mws_wd = ~primary
    if "MWS" in tcnames or "STD" in tcnames:
        mws_wd = isMWS_WD(
            gaia=gaia, galb=galb, astrometricexcessnoise=gaiaaen,
            pmra=pmra, pmdec=pmdec, parallax=parallax, parallaxovererror=parallaxovererror,
            photbprpexcessfactor=gaiabprpfactor, astrometricsigma5dmax=gaiasigma5dmax,
            gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag
        )
    else:
        mws_wd = ~primary

    # ADM initially set everything to False for the standards.
    std_faint, std_bright, std_wd = ~primary, ~primary, ~primary
    if "STD" in tcnames:
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
        # ADM the standard WDs are currently identical to the MWS WDs
        std_wd = mws_wd

    # ADM combine the north/south MWS bits.
    mws = (mws_n & photsys_north) | (mws_s & photsys_south)
    mws_faint = (mws_faint_n & photsys_north) | (mws_faint_s & photsys_south)

    # ADM the formal bit-setting using desi_mask/bgs_mask/mws_mask...
    # Construct the targetflag bits combining north and south.
    desi_target = lrg * desi_mask.LRG
    desi_target |= elg * desi_mask.ELG
    desi_target |= qso * desi_mask.QSO

    # ADM add the per-bit information in the south for LRGs...
    desi_target |= lrginit_s_4 * desi_mask.LRG_INIT_4PASS_SOUTH
    desi_target |= lrgsup_s_4 * desi_mask.LRG_SUPER_4PASS_SOUTH
    desi_target |= lrginit_s_8 * desi_mask.LRG_INIT_8PASS_SOUTH
    desi_target |= lrgsup_s_8 * desi_mask.LRG_SUPER_8PASS_SOUTH
    desi_target |= filler_s * desi_mask.LOWZ_FILLER_SOUTH
    # ADM ...and ELGs...
    desi_target |= elgsvgtot_s * desi_mask.ELG_SV_GTOT_SOUTH
    desi_target |= elgsvgfib_s * desi_mask.ELG_SV_GFIB_SOUTH
    desi_target |= elgfdrgtot_s * desi_mask.ELG_FDR_GTOT_SOUTH
    desi_target |= elgfdrgfib_s * desi_mask.ELG_FDR_GFIB_SOUTH
    # ADM ...and QSOs.
    desi_target |= qsocolor_lowz_south * desi_mask.QSO_COLOR_4PASS_SOUTH
    desi_target |= qsorf_lowz_south * desi_mask.QSO_RF_4PASS_SOUTH
    desi_target |= qsocolor_highz_south * desi_mask.QSO_COLOR_8PASS_SOUTH
    desi_target |= qsorf_highz_south * desi_mask.QSO_RF_8PASS_SOUTH
    desi_target |= qsohizf_south * desi_mask.QSO_HZ_F_SOUTH
    desi_target |= qsoz5_south * desi_mask.QSO_Z5_SOUTH

    # ADM add the per-bit information in the north for LRGs...
    desi_target |= lrginit_n_4 * desi_mask.LRG_INIT_4PASS_NORTH
    desi_target |= lrgsup_n_4 * desi_mask.LRG_SUPER_4PASS_NORTH
    desi_target |= lrginit_n_8 * desi_mask.LRG_INIT_8PASS_NORTH
    desi_target |= lrgsup_n_8 * desi_mask.LRG_SUPER_8PASS_NORTH
    desi_target |= filler_n * desi_mask.LOWZ_FILLER_NORTH
    # ADM ...and ELGs...
    desi_target |= elgsvgtot_n * desi_mask.ELG_SV_GTOT_NORTH
    desi_target |= elgsvgfib_n * desi_mask.ELG_SV_GFIB_NORTH
    desi_target |= elgfdrgtot_n * desi_mask.ELG_FDR_GTOT_NORTH
    desi_target |= elgfdrgfib_n * desi_mask.ELG_FDR_GFIB_NORTH
    # ADM ...and QSOs.
    desi_target |= qsocolor_lowz_north * desi_mask.QSO_COLOR_4PASS_NORTH
    desi_target |= qsorf_lowz_north * desi_mask.QSO_RF_4PASS_NORTH
    desi_target |= qsocolor_highz_north * desi_mask.QSO_COLOR_8PASS_NORTH
    desi_target |= qsorf_highz_north * desi_mask.QSO_RF_8PASS_NORTH
    desi_target |= qsohizf_north * desi_mask.QSO_HZ_F_NORTH
    desi_target |= qsoz5_north * desi_mask.QSO_Z5_NORTH

    # ADM combined per-bit information for the LRGs...
    desi_target |= lrginit_4 * desi_mask.LRG_INIT_4PASS
    desi_target |= lrgsup_4 * desi_mask.LRG_SUPER_4PASS
    desi_target |= lrginit_8 * desi_mask.LRG_INIT_8PASS
    desi_target |= lrgsup_8 * desi_mask.LRG_SUPER_8PASS
    desi_target |= filler * desi_mask.LOWZ_FILLER
    # ADM ...and ELGs...
    desi_target |= elgsvgtot * desi_mask.ELG_SV_GTOT
    desi_target |= elgsvgfib * desi_mask.ELG_SV_GFIB
    desi_target |= elgfdrgtot * desi_mask.ELG_FDR_GTOT
    desi_target |= elgfdrgfib * desi_mask.ELG_FDR_GFIB
    # ADM ...and QSOs.
    desi_target |= qsocolor_lowz * desi_mask.QSO_COLOR_4PASS
    desi_target |= qsorf_lowz * desi_mask.QSO_RF_4PASS
    desi_target |= qsocolor_highz * desi_mask.QSO_COLOR_8PASS
    desi_target |= qsorf_highz * desi_mask.QSO_RF_8PASS
    desi_target |= qsohizf * desi_mask.QSO_HZ_F
    desi_target |= qsoz5 * desi_mask.QSO_Z5

    # ADM Standards.
    desi_target |= std_faint * desi_mask.STD_FAINT
    desi_target |= std_bright * desi_mask.STD_BRIGHT
    desi_target |= std_wd * desi_mask.STD_WD

    # BGS bright and faint, south.
    bgs_target = bgs_bright_south * bgs_mask.BGS_BRIGHT_SOUTH
    bgs_target |= bgs_faint_south * bgs_mask.BGS_FAINT_SOUTH
    bgs_target |= bgs_faint_ext_south * bgs_mask.BGS_FAINT_EXT_SOUTH
    bgs_target |= bgs_lowq_south * bgs_mask.BGS_LOWQ_SOUTH
    bgs_target |= bgs_fibmag_south * bgs_mask.BGS_FIBMAG_SOUTH

    # BGS bright and faint, north.
    bgs_target |= bgs_bright_north * bgs_mask.BGS_BRIGHT_NORTH
    bgs_target |= bgs_faint_north * bgs_mask.BGS_FAINT_NORTH
    bgs_target |= bgs_faint_ext_north * bgs_mask.BGS_FAINT_EXT_NORTH
    bgs_target |= bgs_lowq_north * bgs_mask.BGS_LOWQ_NORTH
    bgs_target |= bgs_fibmag_north * bgs_mask.BGS_FIBMAG_NORTH

    # BGS combined, bright and faint
    bgs_target |= bgs_bright * bgs_mask.BGS_BRIGHT
    bgs_target |= bgs_faint * bgs_mask.BGS_FAINT
    bgs_target |= bgs_faint_ext * bgs_mask.BGS_FAINT_EXT
    bgs_target |= bgs_lowq * bgs_mask.BGS_LOWQ
    bgs_target |= bgs_fibmag * bgs_mask.BGS_FIBMAG

    # ADM MWS main, nearby, and WD.
    mws_target = mws * mws_mask.MWS_MAIN_BROAD
    mws_target |= mws_faint * mws_mask.MWS_MAIN_FAINT
    # APC Science WDs are secondary targets now
    # mws_target |= mws_wd * mws_mask.MWS_WD
    mws_target |= mws_nearby * mws_mask.MWS_NEARBY

    # ADM MWS main north/south split.
    mws_target |= mws_n * mws_mask.MWS_MAIN_BROAD_NORTH
    mws_target |= mws_s * mws_mask.MWS_MAIN_BROAD_SOUTH

    # ADM MWS main faint north/south split.
    mws_target |= mws_faint_n * mws_mask.MWS_MAIN_FAINT_NORTH
    mws_target |= mws_faint_s * mws_mask.MWS_MAIN_FAINT_SOUTH

    # Are any BGS or MWS bit set?  Tell desi_target too.
    desi_target |= (bgs_target != 0) * desi_mask.BGS_ANY
    desi_target |= (mws_target != 0) * desi_mask.MWS_ANY

    return desi_target, bgs_target, mws_target
