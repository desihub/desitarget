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

from desitarget.cuts import _getColors, _psflike, _check_BGS_targtype
from desitarget.cuts import shift_photo_north

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()


def isLRG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
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
        ``True`` if the object is a ONE pass (bright) LRG target.
    :class:`array_like`
        ``True`` if the object is a TWO pass (fainter) LRG target.

    Notes
    -----
    - Current version (11/05/18) is version 24 on `the SV wiki`_.
    - See :func:`~desitarget.sv1.sv1_cuts.set_target_bits` for other parameters.
    """
    # ADM LRG SV targets, pass-based.
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    lrg &= notinLRG_mask(primary=primary, rflux=rflux, zflux=zflux, w1flux=w1flux,
                         rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr)

    # ADM pass the lrg that pass cuts as primary, to restrict to the
    # ADM sources that weren't in a mask/logic cut.
    lrg, lrg1pass, lrg2pass = isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                                           w1flux=w1flux, w2flux=w2flux,
                                           south=south, primary=lrg)

    return lrg, lrg1pass, lrg2pass


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

    lrg &= (rflux_snr > 0) & (rflux > 0)   # ADM quality in r
    lrg &= (zflux_snr > 0) & (zflux > 0)   # ADM quality in z
    lrg &= (w1flux_snr > 4) & (w1flux > 0)  # ADM quality in W1

    return lrg


def isLRG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 w2flux=None, south=True, primary=None):
    """See :func:`~desitarget.sv1.sv1_cuts.isLRG` for details.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    lrg = primary.copy()

    # ADM safe as these fluxes are set to > 0 in notinLRG_mask
    rmag = 22.5 - 2.5 * np.log10(rflux.clip(1e-7))
    zmag = 22.5 - 2.5 * np.log10(zflux.clip(1e-7))
    w1mag = 22.5 - 2.5 * np.log10(w1flux.clip(1e-7))

    if south:
        lrg &= zmag - w1mag > 0.8*(rmag-zmag) - 0.8  # Non-stellar cut
        lrg &= zmag < 20.8                           # z < 20.8 (two exposure limit)
        lrg &= zmag >= 18.01                         # z > 18.01 (reduce overlap with BGS)
        lrg &= rmag - zmag > 0.65                    # r-z > 0.65 (broad color box)
        lrg &= (
            (((zmag - 22.) / 1.3)**2 + (rmag - zmag + 1.27)**2 > 3.0**2) |
            (zmag < 19.7) |
            ((w1mag - 21.01)**2 + ((rmag - w1mag - 0.42) / 1.5)**2 > 2.5**2) |
            ((w1mag < 19.15) & (rmag-w1mag < 1.88))  # Curved sliding optical and IR cuts with low-z extension
        )
    else:
        lrg &= zmag - w1mag > 0.8*(rmag-zmag) - 0.935  # Non-stellar cut
        lrg &= zmag < 20.755                           # z < 20.755 (two exposure limit)
        lrg &= zmag >= 17.965                          # z > 17.965 (reduce overlap with BGS)
        lrg &= rmag - zmag > 0.75                      # r-z > 0.75 (broad color box)
        lrg &= (
            (((zmag - 22.1) / 1.3)**2 + (rmag - zmag + 1.04)**2 > 3.0**2) |
            (zmag < 19.655) |
            ((w1mag - 21.)**2 + ((rmag - w1mag - 0.47) / 1.5)**2 > 2.5**2) |
            ((w1mag < 19.139) & (rmag-w1mag < 1.833))  # Curved sliding optical and IR cuts with low-z extension
        )

    lrg1pass = lrg.copy()
    lrg2pass = lrg.copy()

    # ADM one-pass LRGs are (the BGS limit) <= z < 20
    lrg1pass &= zmag < 20.
    # ADM two-pass LRGs are 20 <= z < (the two pass limit)
    lrg2pass &= zmag >= 20.

    return lrg, lrg1pass, lrg2pass


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
    - See :func:`~desitarget.sv1.sv1_cuts.set_target_bits` for other parameters.
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
    - Current version (11/05/18) is version 33 on `the SV wiki`_.
    - See :func:`~desitarget.sv1.sv1_cuts.set_target_bits` for other parameters.
    """
    if not south:
        gflux, rflux, zflux = shift_photo_north(gflux, rflux, zflux)

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    qso = primary.copy()

    # ADM Reject objects flagged inside a bright star blob.
    qso &= ~brightstarinblob

    # ADM relaxed morphology cut for SV.
    morph2 = (dchisq[..., 1] - dchisq[..., 0])/dchisq[..., 0] < 0.01
    qso &= _psflike(objtype) | morph2

    # ADM SV cuts are different for WISE SNR.
    qso &= w1snr > 2.5
    qso &= w2snr > 1.5

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
    qso &= rflux > 10**((22.5-23.)/2.5)    # r < 23.0 (different for SV)
    qso &= grzflux < 10**((22.5-17.)/2.5)    # grz > 17

    # ADM the optical color cuts.
    qso &= rflux < gflux * 10**(1.3/2.5)    # (g-r) < 1.3
    qso &= zflux > rflux * 10**(-0.3/2.5)   # (r-z) > -0.3
    qso &= zflux < rflux * 10**(3.0/2.5)    # (r-z) < 3.0 (different for SV)

    # ADM the WISE-optical color cut.
    qso &= wflux * gflux > zflux * grzflux * 10**(-1.3/2.5)  # (grz-W) > (g-z)-1.3 (different for SV)

    # ADM the WISE color cut.
    qso &= w2flux > w1flux * 10**(-0.4/2.5)  # (W1-W2) > -0.4

    # ADM Stricter WISE cuts on stellar contamination for objects on Main Sequence.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    mainseq = rflux > gflux * 10**(0.2/2.5)  # ADM g-r > 0.2
    mainseq &= rflux**(1+1.5) > gflux * zflux**1.5 * 10**((-0.075+0.175)/2.5)
    mainseq &= rflux**(1+1.5) < gflux * zflux**1.5 * 10**((+0.075+0.175)/2.5)
    mainseq &= w2flux < w1flux * 10**(0.3/2.5)  # ADM W1 - W2 !(NOT) > 0.3
    qso &= ~mainseq

    return qso


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
    - Current version (11/05/18) is version 33 on `the SV wiki`_.
    - See :func:`~desitarget.sv1.sv1_cuts.set_target_bits` for other parameters.
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
    morph2 = (dchisq[..., 1] - dchisq[..., 0])/dchisq[..., 0] < 0.015
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
        else:
            pcut = np.where(r_Reduced > 20.0,
                            0.45 - (r_Reduced - 20.0) * 0.10, 0.45)
        pcut_HighZ = 0.40

        # Add rf proba test result to "qso" mask
        qso[colorsReducedIndex] = \
            (tmp_rf_proba >= pcut) | (tmp_rf_HighZ_proba >= pcut_HighZ)

    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "numpy.ndarray"
    if nbEntries == 1:
        qso = qso[0]

    return qso


def isBGS(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
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
        or ``wise`` to use colors appropriate to the ``BGS_WISE`` selection.

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is a BGS target of type ``targtype``.

    Notes
    -----
    - Current version (11/05/18) is version 24 on `the SV wiki`_.
    - See :func:`~desitarget.sv1.sv1_cuts.set_target_bits` for other parameters.
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
    (see, e.g., :func:`~desitarget.cuts.isBGS_faint` for parameters).
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
    else:
        _check_BGS_targtype(targtype)

    return bgs


def isBGS_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 south=True, targtype=None, primary=None):
    """Standard set of masking cuts used by all BGS target selection classes
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


def isELG(gflux=None, rflux=None, zflux=None,
          gallmask=None, rallmask=None, zallmask=None,
          gsnr=None, rsnr=None, zsnr=None, south=True, primary=None):
    """Definition of ELG target classes. Returns a boolean array.

    Parameters
    ----------
    south : :class:`boolean`, defaults to ``True``
        Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
        otherwise use cuts appropriate to the Southern imaging survey (DECaLS).

    Returns
    -------
    :class:`array_like`
        ``True`` if and only if the object is in the ELG FDR box.
    :class:`array_like`
        ``True`` if the object is in a faint extension to the ELG FDR box.
    :class:`array_like`
        ``True`` if the object passes a blue extension to the ELG box in (r-z).
    :class:`array_like`
        ``True`` if the object passes a red extension to the ELG box in (r-z).

    Notes
    -----
    - Current version (11/05/18) is version 24 on `the SV wiki`_.
    - See :func:`~desitarget.sv1.sv1_cuts.set_target_bits` for other parameters.
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elg = primary.copy()

    elg &= notinELG_mask(gallmask=gallmask, rallmask=rallmask, zallmask=zallmask,
                         gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, primary=primary)

    # ADM pass the elg that pass cuts as primary, to restrict to the
    # ADM sources that weren't in a mask/logic cut.
    elgfdr, elgfdrfaint, elgrzblue, elgrzred = isELG_colors(
        gflux=gflux, rflux=rflux, zflux=zflux, south=south, primary=elg
    )

    return elgfdr, elgfdrfaint, elgrzblue, elgrzred


def notinELG_mask(gallmask=None, rallmask=None, zallmask=None,
                  gsnr=None, rsnr=None, zsnr=None, primary=None):
    """Standard set of masking cuts used by all ELG target selection classes
    (see, e.g., :func:`~desitarget.sv1.sv1_cuts.isELG` for parameters).
    """
    if primary is None:
        primary = np.ones_like(gallmask, dtype='?')
    elg = primary.copy()

    elg &= (gallmask == 0) & (rallmask == 0) & (zallmask == 0)
    elg &= (gsnr > 0) & (rsnr > 0) & (zsnr > 0)

    return elg


def isELG_colors(gflux=None, rflux=None, zflux=None, primary=None,
                 south=True):
    """Color cuts for ELG target selection classes
    (see, e.g., :func:`~desitarget.sv1.sv1_cuts.isELG`).
    """
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    elgfdr, elgfdrfaint, elgrzblue, elgrzred = \
        primary.copy(), primary.copy(), primary.copy(), primary.copy()

    # ADM determine colors and magnitudes
    g = 22.5-2.5*np.log10(gflux.clip(1e-16))  # ADM clip is safe as we never target g < 20
    gr = -2.5*np.log10(gflux/rflux)
    rz = -2.5*np.log10(rflux/zflux)

    # ADM note that there is currently no north/south split
    # ADM FDR box
    elgfdr &= (g >= 20.00) & (g < 23.45) & (rz > 0.3) & (rz < 1.6) & \
              (gr < 1.15*rz-0.15) & (gr < 1.6-1.2*rz)
    # ADM FDR box faint
    elgfdrfaint &= (g >= 23.45) & (g < 23.65) & (rz > 0.3) & (rz < 1.6) & \
                   (gr < 1.15*rz-0.15) & (gr < 1.6-1.2*rz)
    # ADM blue rz box extension
    elgrzblue &= (g >= 20.00) & (g < 23.65) & \
                 (rz > 0.0) & (rz < 0.3) & (gr < 0.2)
    # ADM red rz box extension
    elgrzred &= (g >= 20.00) & (g < 23.65) & \
                (gr < 1.15*rz-0.15) & ((rz > 1.6) | (gr > 1.6-1.2*rz)) & (gr < 2.5-1.2*rz)

    return elgfdr, elgfdrfaint, elgrzblue, elgrzred


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


def set_target_bits(photsys_north, photsys_south, obs_rflux,
                    gflux, rflux, zflux, w1flux, w2flux,
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
        ``qso_selection="colorcuts"``. This has no effect in SV!!!
    qso_selection : :class:`str`, optional, defaults to ``'randomforest'``
        The algorithm to use for QSO selection; valid options are
        ``'colorcuts'`` and ``'randomforest'``. This has no effect in SV!!!
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

    from desitarget.sv1.sv1_targetmask import desi_mask, bgs_mask, mws_mask

    if "LRG" in tcnames:
        lrg_classes = []
        # ADM run the LRG target types for both north and south.
        for south in [False, True]:
            lrg_classes.append(
                isLRG(
                    primary=primary, south=south,
                    gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                    gflux_ivar=gfluxivar, rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr
                )
            )
        lrg_north, lrg1pass_north, lrg2pass_north,  \
            lrg_south, lrg1pass_south, lrg2pass_south = \
            np.vstack(lrg_classes)
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
                    gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, south=south
                )
            )
        elgfdr_north, elgfdrfaint_north, elgrzblue_north, elgrzred_north,  \
            elgfdr_south, elgfdrfaint_south, elgrzblue_south, elgrzred_south = \
            np.vstack(elg_classes)
    else:
        # ADM if not running the ELG selection, set everything to arrays of False.
        elgfdr_north, elgfdrfaint_north, elgrzblue_north, elgrzred_north = \
            ~primary, ~primary, ~primary, ~primary
        elgfdr_south, elgfdrfaint_south, elgrzblue_south, elgrzred_south = \
            ~primary, ~primary, ~primary, ~primary

    # ADM combine ELG target bits for an ELG target based on any imaging
    elg_north = elgfdr_north | elgfdrfaint_north | elgrzblue_north | elgrzred_north
    elg_south = elgfdr_south | elgfdrfaint_south | elgrzblue_south | elgrzred_south
    elg = (elg_north & photsys_north) | (elg_south & photsys_south)
    elgfdr = (elgfdr_north & photsys_north) | (elgfdr_south & photsys_south)
    elgfdrfaint = (elgfdrfaint_north & photsys_north) | (elgfdrfaint_south & photsys_south)
    elgrzblue = (elgrzblue_north & photsys_north) | (elgrzblue_south & photsys_south)
    elgrzred = (elgrzred_north & photsys_north) | (elgrzred_south & photsys_south)

    if "QSO" in tcnames:
        qso_classes = []
        for south in [False, True]:
            qso_classes.append(
                isQSO_cuts(
                    primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                    w1flux=w1flux, w2flux=w2flux,
                    dchisq=dchisq, brightstarinblob=brightstarinblob,
                    objtype=objtype, w1snr=w1snr, w2snr=w2snr,
                    south=south
                )
            )
            qso_classes.append(
                isQSO_randomforest(
                    primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                    w1flux=w1flux, w2flux=w2flux,
                    dchisq=dchisq, brightstarinblob=brightstarinblob,
                    objtype=objtype, south=south
                )
            )
        qsocolor_north, qsorf_north, qsocolor_south, qsorf_south = \
            qso_classes

    else:
        # ADM if not running the QSO selection, set everything to arrays of False
        qsocolor_north, qsorf_north, qsocolor_south, qsorf_south = \
                                    ~primary, ~primary, ~primary, ~primary

    # ADM combine quasar target bits for a quasar target based on any imaging
    qso_north = qsocolor_north | qsorf_north
    qso_south = qsocolor_south | qsorf_south
    qso = (qso_north & photsys_north) | (qso_south & photsys_south)
    qsocolor = (qsocolor_north & photsys_north) | (qsocolor_south & photsys_south)
    qsorf = (qsorf_north & photsys_north) | (qsorf_south & photsys_south)

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

        bgs_bright_north, bgs_bright_south, \
            bgs_faint_north, bgs_faint_south,   \
            bgs_wise_north, bgs_wise_south =    \
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
        # ADM run the MWS_MAIN target types for both north and south
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

        mws_n, mws_red_n, mws_blue_n,   \
            mws_s, mws_red_s, mws_blue_s =  \
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
        mws_n, mws_red_n, mws_blue_n = ~primary, ~primary, ~primary
        mws_s, mws_red_s, mws_blue_s = ~primary, ~primary, ~primary
        mws_nearby, mws_wd = ~primary, ~primary

    if "STD" in tcnames:
        std_classes = []
        # ADM run the MWS_MAIN target types for both faint and bright.
        # ADM Make sure to pass all of the needed columns! At one point we stopped
        # ADM passing objtype, which meant no standards were being returned.
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
    else:
        # ADM if not running the standards selection, set everything to arrays of False
        std_faint, std_bright, std_wd = ~primary, ~primary, ~primary

    # ADM combine the north/south MWS bits.
    mws = (mws_n & photsys_north) | (mws_s & photsys_south)
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

    # ADM add the per-bit information in the south for LRGs...
    desi_target |= lrg1pass_south * desi_mask.LRG_1PASS_SOUTH
    desi_target |= lrg2pass_south * desi_mask.LRG_2PASS_SOUTH
    # ADM ...and ELGs...
    desi_target |= elgfdr_south * desi_mask.ELG_FDR_SOUTH
    desi_target |= elgfdrfaint_south * desi_mask.ELG_FDR_FAINT_SOUTH
    desi_target |= elgrzblue_south * desi_mask.ELG_RZ_BLUE_SOUTH
    desi_target |= elgrzred_south * desi_mask.ELG_RZ_RED_SOUTH
    # ADM ...and QSOs.
    desi_target |= qsocolor_south * desi_mask.QSO_COLOR_SOUTH
    desi_target |= qsorf_south * desi_mask.QSO_RF_SOUTH

    # ADM add the per-bit information in the north for LRGs...
    desi_target |= lrg1pass_north * desi_mask.LRG_1PASS_NORTH
    desi_target |= lrg2pass_north * desi_mask.LRG_2PASS_NORTH
    # ADM ...and ELGs...
    desi_target |= elgfdr_north * desi_mask.ELG_FDR_NORTH
    desi_target |= elgfdrfaint_north * desi_mask.ELG_FDR_FAINT_NORTH
    desi_target |= elgrzblue_north * desi_mask.ELG_RZ_BLUE_NORTH
    desi_target |= elgrzred_north * desi_mask.ELG_RZ_RED_NORTH
    # ADM ...and QSOs.
    desi_target |= qsocolor_north * desi_mask.QSO_COLOR_NORTH
    desi_target |= qsorf_north * desi_mask.QSO_RF_NORTH

    # ADM combined per-bit information for the LRGs...
    desi_target |= lrg1pass * desi_mask.LRG_1PASS
    desi_target |= lrg2pass * desi_mask.LRG_2PASS
    # ADM ...and ELGs...
    desi_target |= elgfdr * desi_mask.ELG_FDR
    desi_target |= elgfdrfaint * desi_mask.ELG_FDR_FAINT
    desi_target |= elgrzblue * desi_mask.ELG_RZ_BLUE
    desi_target |= elgrzred * desi_mask.ELG_RZ_RED
    # ADM ...and QSOs.
    desi_target |= qsocolor * desi_mask.QSO_COLOR
    desi_target |= qsorf * desi_mask.QSO_RF

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
