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

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()


def isLRG(gflux=None, rflux=None, zflux=None, w1flux=None,
          rflux_snr=None, zflux_snr=None, w1flux_snr=None,
          gflux_ivar=None, primary=None, south=True):
    """Target Definition of LRG. Returns a boolean array.

    Parameters
    ----------
    south: boolean, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
        otherwise use cuts appropriate to the Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is an LRG target.
    :class:`array_like`
        ``True`` if the object is a nominal optical + nominal IR LRG.
    :class:`array_like`
        ``True`` if the object is an optical + IR + low-z extension LRG.
    :class:`array_like`
        ``True`` if the object is an optical + IR + high-z extension LRG.
    :class:`array_like`
        ``True`` if the object is an optical + IR + low-z + high-z + relaxed cuts LRG.
    :class:`array_like`
        ``True`` if the object is a superset of all other subsamples LRG.

    Notes
    -----
    - Current version (03/19/19) is version 56 on `the SV wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    # ADM LRG SV targets, pass-based.
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    lrg &= notinLRG_mask(primary=primary, rflux=rflux, zflux=zflux, w1flux=w1flux,
                         rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr)

    # ADM pass the lrg that pass cuts as primary, to restrict to the
    # ADM sources that weren't in a mask/logic cut.
    lrg_all, lrg1, lrg2, lrg3, lrg4, lrg5 = isLRG_colors(gflux=gflux, rflux=rflux,
                                                         zflux=zflux, w1flux=w1flux,
                                                         gflux_ivar=gflux_ivar,
                                                         south=south, primary=lrg)

    return lrg_all, lrg1, lrg2, lrg3, lrg4, lrg5


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
                 gflux_ivar=None, south=True, primary=None):
    """See :func:`~desitarget.sv1.sv1_cuts.isLRG` for details.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()
    lrg1, lrg2, lrg3, lrg4, lrg5 = np.tile(primary, [5, 1])

    # ADM safe as these fluxes are set to > 0 in notinLRG_mask.
    gmag = 22.5 - 2.5 * np.log10(gflux.clip(1e-7))
    rmag = 22.5 - 2.5 * np.log10(rflux.clip(1e-7))
    zmag = 22.5 - 2.5 * np.log10(zflux.clip(1e-7))
    w1mag = 22.5 - 2.5 * np.log10(w1flux.clip(1e-7))

    if south:

        # subsample 1: Nominal optical + Nominal IR:
        lrg1 &= zmag - w1mag > 0.8*(rmag-zmag) - 0.6   # non-stellar cut
        lrg1 &= zmag < 20.41                           # faint limit
        lrg1 &= rmag - zmag > 0.75                     # broad color box
        lrg1 &= (rmag - zmag > 1.15) | ((gmag - rmag > 1.65) & (gflux_ivar > 0))  # Low-z cut
        lrg_opt = rmag - zmag > (zmag - 17.18) / 2     # sliding optical cut
        lrg_ir = rmag - w1mag > (w1mag - 17.74) / 0.4  # sliding IR cut
        lrg1 &= lrg_opt | lrg_ir

        # subsample 2: optical + IR + low-z extension:
        lrg2 &= zmag - w1mag > 0.8*(rmag-zmag) - 0.6  # non-stellar cut
        lrg2 &= zmag < 20.41                          # faint limit
        lrg2 &= rmag - zmag > 0.75                    # broad color box
        lrg2 &= (rmag - zmag > 1.15) | ((gmag - rmag > 1.65) & (gflux_ivar > 0))  # Low-z cut
        lrg_opt = ((rmag - zmag > (zmag - 17.18) / 2) |
                   (zmag < 19.7))                            # sliding optical cut with low-z extension
        lrg_ir = ((rmag - w1mag > (w1mag - 17.74) / 0.4) |
                  ((w1mag < 19.15) & (rmag - w1mag < 1.88)))  # sliding IR cut with low-z extension
        lrg2 &= lrg_opt | lrg_ir

        # subsample 3: optical + IR + high-z extension:
        lrg3 &= zmag - w1mag > 0.8*(rmag-zmag) - 0.6   # non-stellar cut
        lrg3 &= zmag < 20.80                           # extended faint limit
        lrg3 &= rmag - zmag > 0.75                     # broad color box
        lrg3 &= (rmag - zmag > 1.15) | ((gmag - rmag > 1.65) & (gflux_ivar > 0))  # Low-z cut
        lrg_opt = rmag - zmag > (zmag - 17.18) / 2     # sliding optical cut
        lrg_ir = rmag - w1mag > (w1mag - 17.74) / 0.4  # sliding IR cut
        lrg3 &= lrg_opt | lrg_ir

        # subsample 4: optical + IR + low-z + high-z + relaxed cuts
        # (relaxed stellar rejection + relaxed sliding cuts + relaxed g-r cut):
        lrg4 &= zmag - w1mag > 0.8*(rmag-zmag) - 0.8  # relaxed non-stellar cut
        lrg4 &= zmag < 20.80                          # extended faint limit
        lrg4 &= rmag - zmag > 0.75                    # broad color box
        lrg4 &= (rmag - zmag > 1.15) | ((gmag - rmag > 1.45) & (gflux_ivar > 0))  # relaxed low-z cut
        lrg_opt = ((((zmag - 22.) / 1.3)**2 + (rmag - zmag + 1.27)**2 > 3.0**2) |
                   (zmag < 19.7))                             # curved sliding optical cut with low-z extension
        lrg_ir = (((w1mag - 21.01)**2 + ((rmag - w1mag - 0.42) / 1.5)**2 > 2.5**2) |
                  ((w1mag < 19.15) & (rmag - w1mag < 1.88)))  # curved sliding IR cut with low-z extension
        lrg4 &= lrg_opt | lrg_ir

        # subsample 5 (this is a superset of all other subsamples):
        # optical + IR + low-z + high-z + more relaxed cuts
        # (relaxed stellar rejection + relaxed sliding cuts + no g-r cut + relaxed broad r-z cut):
        lrg5 &= zmag - w1mag > 0.8*(rmag-zmag) - 0.8   # relaxed non-stellar cut
        lrg5 &= zmag < 20.80                           # extended faint limit
        lrg5 &= rmag - zmag > 0.65                     # relaxed broad color box
        lrg_opt = ((((zmag - 22.) / 1.3)**2 + (rmag - zmag + 1.27)**2 > 3.0**2) |
                   (zmag < 19.7))                             # curved sliding optical cut with low-z extension
        lrg_ir = (((w1mag - 21.01)**2 + ((rmag - w1mag - 0.42) / 1.5)**2 > 2.5**2) |
                  ((w1mag < 19.15) & (rmag - w1mag < 1.88)))  # curved sliding IR cut with low-z extension
        lrg5 &= lrg_opt | lrg_ir

    else:

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

    # lrg1pass = lrg.copy()
    # lrg2pass = lrg.copy()

    # # ADM one-pass LRGs are (the BGS limit) <= z < 20
    # lrg1pass &= zmag < 20.
    # # ADM two-pass LRGs are 20 <= z < (the two pass limit)
    # lrg2pass &= zmag >= 20.

    return lrg, lrg1, lrg2, lrg3, lrg4, lrg5


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
               dchisq=None, brightstarinblob=None,
               objtype=None, primary=None, south=True):
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
    - Current version (06/05/19) is version 68 on `the SV wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    qso = primary.copy()

    # ADM Reject objects flagged inside a bright star blob.
    qso &= ~brightstarinblob

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
                       objtype=None, release=None, dchisq=None, brightstarinblob=None,
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
    - Current version (06/05/19) is version 68 on `the SV wiki`_.
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


def isQSO_highz_faint(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                      objtype=None, release=None, dchisq=None, brightstarinblob=None,
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
    - Current version (04/05/19) is version 64 on `the SV wiki`_.
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

    #  Reject objects flagged inside a blob.
    preSelection &= ~brightstarinblob

    # "qso" mask initialized to "preSelection" mask.
    qso = np.copy(preSelection)

    if np.any(preSelection):

        from desitarget.myRF import myRF

        # Data reduction to preselected objects.
        colorsReduced = colors[preSelection]
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
            pcut = 0.30
        else:
            # pcut = 0.35
            pcut = 0.30

        # Add rf proba test result to "qso" mask
        qso[colorsReducedIndex] = (tmp_rf_proba >= pcut)

    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "numpy.ndarray"
    if nbEntries == 1:
        qso = qso[0]

    return qso


def isBGS(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, rfiberflux=None,
          gnobs=None, rnobs=None, znobs=None, gfracmasked=None, rfracmasked=None, zfracmasked=None,
          gfracflux=None, rfracflux=None, zfracflux=None, gfracin=None, rfracin=None, zfracin=None,
          gfluxivar=None, rfluxivar=None, zfluxivar=None, brightstarinblob=None, Grr=None,
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
    - Current version (02/06/19) is version 36 on `the SV wiki`_.
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
                         gaiagmag=gaiagmag, brightstarinblob=brightstarinblob, objtype=objtype, targtype=targtype)

    bgs &= isBGS_colors(rflux=rflux, rfiberflux=rfiberflux, south=south, targtype=targtype, primary=primary)

    return bgs


def notinBGS_mask(gflux=None, rflux=None, zflux=None, gnobs=None, rnobs=None, znobs=None, primary=None,
                  gfracmasked=None, rfracmasked=None, zfracmasked=None,
                  gfracflux=None, rfracflux=None, zfracflux=None,
                  gfracin=None, rfracin=None, zfracin=None, w1snr=None,
                  gfluxivar=None, rfluxivar=None, zfluxivar=None, Grr=None,
                  gaiagmag=None, brightstarinblob=None, objtype=None, targtype=None):
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
    bgs_qcs &= ~brightstarinblob
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
    elif targtype == 'fibmag':
        bgs &= rflux <= 10**((22.5-20.1)/2.5)
        bgs &= rfiberflux > 10**((22.5-21.0511)/2.5)
    else:
        _check_BGS_targtype_sv(targtype)

    return bgs


def isELG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
          gsnr=None, rsnr=None, zsnr=None, maskbits=None, south=True,
          primary=None):
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
    - Current version (03/19/19) is version 76 on `the SV wiki`_.
    - See :func:`~desitarget.cuts.set_target_bits` for other parameters.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    elg &= notinELG_mask(maskbits=maskbits, gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                         primary=primary)

    elg &= isELG_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                        w2flux=w2flux, south=south, primary=primary)

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
                 w2flux=None, south=True, primary=None):
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

    # ADM cuts shared by the northern and southern selections.
    elg &= g > 20                       # bright cut.
    elg &= r - z > -1.0                 # blue cut.
    elg &= g - r < -1.2*(r - z) + 2.5   # OII flux cut.

    # ADM cuts that are unique to the north or south.
    if south:
        elg &= (g - r < 0.2) | (g - r < 1.15*(r - z) - 0.15)  # remove stars and low-z galaxies.
        elg &= coii < 1.6 - 7.2*(g - 23.5)  # sliding cut.
    else:
        elg &= (g - r < 0.2) | (g - r < 1.15*(r - z) - 0.35)  # remove stars and low-z galaxies.
        elg &= coii < 1.6 - 7.2*(g - 23.6)  # sliding cut.

    return elg


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
    # ADM parallax cut corresponding to 100pc
    mws &= (parallax + parallaxerr) > 10.  # NB: "+" is correct
    # ADM NOTE TO THE MWS GROUP: There is no bright cut on G. IS THAT THE REQUIRED BEHAVIOR?

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
    lrg_classes = [[~primary, ~primary, ~primary, ~primary, ~primary, ~primary],
                   [~primary, ~primary, ~primary, ~primary, ~primary, ~primary]]
    if "LRG" in tcnames:
        # ADM run the LRG target types (potentially) for both north and south.
        for south in south_cuts:
            lrg_classes[int(south)] = isLRG(
                    primary=primary, south=south,
                    gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                    gflux_ivar=gfluxivar, rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr
            )
    lrg_north, lrginit_n, lrglowz_n, lrghighz_n, lrgrelax_n, lrgsuper_n = lrg_classes[0]
    lrg_south, lrginit_s, lrglowz_s, lrghighz_s, lrgrelax_s, lrgsuper_s = lrg_classes[1]

    # ADM combine LRG target bits for an LRG target based on any imaging
    lrg = (lrg_north & photsys_north) | (lrg_south & photsys_south)
    lrginit = (lrginit_n & photsys_north) | (lrginit_s & photsys_south)
    lrglowz = (lrglowz_n & photsys_north) | (lrglowz_s & photsys_south)
    lrghighz = (lrghighz_n & photsys_north) | (lrghighz_s & photsys_south)
    lrgrelax = (lrgrelax_n & photsys_north) | (lrgrelax_s & photsys_south)
    lrgsuper = (lrgsuper_n & photsys_north) | (lrgsuper_s & photsys_south)

    # ADM initially set everything to arrays of False for the ELG selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    elg_classes = [~primary, ~primary]
    if "ELG" in tcnames:
        for south in south_cuts:
            elg_classes[int(south)] = isELG(
                primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
                gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, maskbits=maskbits, south=south
            )
    elg_north, elg_south = elg_classes

    # ADM combine ELG target bits for an ELG target based on any imaging.
    elg = (elg_north & photsys_north) | (elg_south & photsys_south)

    # ADM initially set everything to arrays of False for the QSO selection
    # ADM the zeroth element stores the northern targets bits (south=False).
    qso_classes = [[~primary, ~primary, ~primary, ~primary],
                   [~primary, ~primary, ~primary, ~primary]]
    if "QSO" in tcnames:
        for south in south_cuts:
            qso_store = []
            qso_store.append(
                isQSO_cuts(
                    primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                    w1flux=w1flux, w2flux=w2flux,
                    dchisq=dchisq, brightstarinblob=brightstarinblob,
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
                        dchisq=dchisq, brightstarinblob=brightstarinblob,
                        objtype=objtype, south=south
                    )
                )
                qso_store.append(
                    isQSO_highz_faint(
                        primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                        w1flux=w1flux, w2flux=w2flux,
                        dchisq=dchisq, brightstarinblob=brightstarinblob,
                        objtype=objtype, south=south
                    )
                )
            qso_store.append(
                isQSO_color_high_z(
                    gflux=gflux, rflux=rflux, zflux=zflux,
                    w1flux=w1flux, w2flux=w2flux, south=south
                )
            )
            qso_classes[int(south)] = qso_store
    qsocolor_north, qsorf_north, qsohizf_north, qsocolor_high_z_north = qso_classes[0]
    qsocolor_south, qsorf_south, qsohizf_south, qsocolor_high_z_south = qso_classes[1]

    # ADM combine quasar target bits for a quasar target based on any imaging.
    qsocolor_highz_north = (qsocolor_north & qsocolor_high_z_north)
    qsorf_highz_north = (qsorf_north & qsocolor_high_z_north)
    qsocolor_lowz_north = (qsocolor_north & ~qsocolor_high_z_north)
    qsorf_lowz_north = (qsorf_north & ~qsocolor_high_z_north)
    qso_north = (qsocolor_lowz_north | qsorf_lowz_north | qsocolor_highz_north
                 | qsorf_highz_north | qsohizf_north)

    qsocolor_highz_south = (qsocolor_south & qsocolor_high_z_south)
    qsorf_highz_south = (qsorf_south & qsocolor_high_z_south)
    qsocolor_lowz_south = (qsocolor_south & ~qsocolor_high_z_south)
    qsorf_lowz_south = (qsorf_south & ~qsocolor_high_z_south)
    qso_south = (qsocolor_lowz_south | qsorf_lowz_south | qsocolor_highz_south
                 | qsorf_highz_south | qsohizf_south)

    qso = (qso_north & photsys_north) | (qso_south & photsys_south)
    qsocolor_highz = (qsocolor_highz_north & photsys_north) | (qsocolor_highz_south & photsys_south)
    qsorf_highz = (qsorf_highz_north & photsys_north) | (qsorf_highz_south & photsys_south)
    qsocolor_lowz = (qsocolor_lowz_north & photsys_north) | (qsocolor_lowz_south & photsys_south)
    qsorf_lowz = (qsorf_lowz_north & photsys_north) | (qsorf_lowz_south & photsys_south)
    qsohizf = (qsohizf_north & photsys_north) | (qsohizf_south & photsys_south)

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
                        brightstarinblob=brightstarinblob, Grr=Grr, w1snr=w1snr, gaiagmag=gaiagmag,
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
    mws_classes = [[~primary, ~primary, ~primary], [~primary, ~primary, ~primary]]
    mws_nearby = ~primary
    if "MWS" in tcnames:
        mws_nearby = isMWS_nearby(
            gaia=gaia, gaiagmag=gaiagmag, parallax=parallax,
            parallaxerr=parallaxerr
        )
        # ADM run the MWS_MAIN target types for both north and south
        for south in south_cuts:
            mws_classes[int(south)] = isMWS_main(
                    gaia=gaia, gaiaaen=gaiaaen, gaiadupsource=gaiadupsource,
                    gflux=gflux, rflux=rflux, obs_rflux=obs_rflux, objtype=objtype,
                    gnobs=gnobs, rnobs=rnobs,
                    gfracmasked=gfracmasked, rfracmasked=rfracmasked,
                    pmra=pmra, pmdec=pmdec, parallax=parallax,
                    primary=primary, south=south
            )
    mws_n, mws_red_n, mws_blue_n = mws_classes[0]
    mws_s, mws_red_s, mws_blue_s = mws_classes[1]

    # ADM treat the MWS WD selection specially, as we have to run the
    # ADM white dwarfs for standards and MWS science targets.
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
    mws_blue = (mws_blue_n & photsys_north) | (mws_blue_s & photsys_south)
    mws_red = (mws_red_n & photsys_north) | (mws_red_s & photsys_south)

    # ADM the formal bit-setting using desi_mask/bgs_mask/mws_mask...
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

    # ADM add the per-bit information in the south for LRGs...
    desi_target |= lrginit_s * desi_mask.LRG_INIT_SOUTH
    desi_target |= lrglowz_s * desi_mask.LRG_LOWZ_SOUTH
    desi_target |= lrghighz_s * desi_mask.LRG_HIGHZ_SOUTH
    desi_target |= lrgrelax_s * desi_mask.LRG_RELAX_SOUTH
    desi_target |= lrgsuper_s * desi_mask.LRG_SUPER_SOUTH
    # ADM ...and QSOs.
    desi_target |= qsocolor_lowz_south * desi_mask.QSO_COLOR_4PASS_SOUTH
    desi_target |= qsorf_lowz_south * desi_mask.QSO_RF_4PASS_SOUTH
    desi_target |= qsocolor_highz_south * desi_mask.QSO_COLOR_8PASS_SOUTH
    desi_target |= qsorf_highz_south * desi_mask.QSO_RF_8PASS_SOUTH
    desi_target |= qsohizf_south * desi_mask.QSO_HZ_F_SOUTH

    # ADM add the per-bit information in the north for LRGs...
    desi_target |= lrginit_n * desi_mask.LRG_INIT_NORTH
    desi_target |= lrglowz_n * desi_mask.LRG_LOWZ_NORTH
    desi_target |= lrghighz_n * desi_mask.LRG_HIGHZ_NORTH
    desi_target |= lrgrelax_n * desi_mask.LRG_RELAX_NORTH
    desi_target |= lrgsuper_n * desi_mask.LRG_SUPER_NORTH
    # ADM ...and QSOs.
    desi_target |= qsocolor_lowz_north * desi_mask.QSO_COLOR_4PASS_NORTH
    desi_target |= qsorf_lowz_north * desi_mask.QSO_RF_4PASS_NORTH
    desi_target |= qsocolor_highz_north * desi_mask.QSO_COLOR_8PASS_NORTH
    desi_target |= qsorf_highz_north * desi_mask.QSO_RF_8PASS_NORTH
    desi_target |= qsohizf_north * desi_mask.QSO_HZ_F_NORTH

    # ADM combined per-bit information for the LRGs...
    desi_target |= lrginit * desi_mask.LRG_INIT
    desi_target |= lrglowz * desi_mask.LRG_LOWZ
    desi_target |= lrghighz * desi_mask.LRG_HIGHZ
    desi_target |= lrgrelax * desi_mask.LRG_RELAX
    desi_target |= lrgsuper * desi_mask.LRG_SUPER
    # ADM ...and QSOs.
    desi_target |= qsocolor_lowz * desi_mask.QSO_COLOR_4PASS
    desi_target |= qsorf_lowz * desi_mask.QSO_RF_4PASS
    desi_target |= qsocolor_highz * desi_mask.QSO_COLOR_8PASS
    desi_target |= qsorf_highz * desi_mask.QSO_RF_8PASS
    desi_target |= qsohizf * desi_mask.QSO_HZ_F

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
    mws_target = mws * mws_mask.MWS_MAIN
    mws_target |= mws_wd * mws_mask.MWS_WD
    mws_target |= mws_nearby * mws_mask.MWS_NEARBY

    # ADM MWS main north/south split.
    mws_target |= mws_n * mws_mask.MWS_MAIN_NORTH
    mws_target |= mws_s * mws_mask.MWS_MAIN_SOUTH

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
