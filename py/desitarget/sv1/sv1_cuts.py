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

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()


def isLRG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                 w2flux=None, ggood=None, primary=None, south=True):
    """Convenience function for backwards-compatability prior to north/south split.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1 and W2 bands (if needed).
        ggood: array_like
            Set to True for objects with good g-band photometry.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to True
            Call isLRG_colors_north if south=False, otherwise call isLRG_colors_south.
    
    Returns:
        mask : array_like. True if and only if the object is an LRG target.
    """
    if south==False:
        return isLRG_colors_north(gflux=gflux, rflux=rflux, zflux=zflux, 
                                  w1flux=w1flux, w2flux=w2flux,
                                  ggood=ggood, primary=primary)
    else:
        return isLRG_colors_south(gflux=gflux, rflux=rflux, zflux=zflux, 
                                  w1flux=w1flux, w2flux=w2flux,
                                  ggood=ggood, primary=primary)


def isLRG_colors_north(gflux=None, rflux=None, zflux=None, w1flux=None,
                        w2flux=None, ggood=None, primary=None):
    """This function applies just the flux and color cuts for the BASS/MzLS photometric system.
    (see :func:`~desitarget.sv1.sv1_cuts.isLRG_colors_south` for details).

    """
    # ADM currently no difference between N/S for LRG colors, so easiest
    # ADM just to use one function.
    return isLRG_colors_south(gflux=gflux, rflux=rflux, zflux=zflux, ggood=ggood,
                              w1flux=w1flux, w2flux=w2flux, primary=primary)


def isLRG_colors_south(gflux=None, rflux=None, zflux=None, w1flux=None,
                        w2flux=None, ggood=None, primary=None):
    """See :func:`~desitarget.cuts.isLRG_south` for details.
    This function applies just the flux and color cuts for the DECaLS photometric system.

    Notes:
        - Current version (09/21/18) is version 17 on `the SV wiki`_.
    """

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    if ggood is None:
        ggood = np.ones_like(gflux, dtype='?')

    # Basic flux and color cuts
    lrg = primary.copy()
    lrg &= (zflux > 10**(0.4*(22.5-20.4))) # z<20.4
    lrg &= (zflux < 10**(0.4*(22.5-18))) # z>18
    lrg &= (zflux < 10**(0.4*2.5)*rflux) # r-z<2.5
    lrg &= (zflux > 10**(0.4*0.8)*rflux) # r-z>0.8

    # The code below can overflow, since the fluxes are float32 arrays
    # which have a maximum value of 3e38. Therefore, if eg. zflux~1.0e10
    # this will overflow, and crash the code.
    with np.errstate(over='ignore'):
        # ADM updated Zhou/Newman cut:
        # Wlrg = -0.6 < (z-w1) - 0.7*(r-z) < 1.0 ->
        # 0.7r + W < 1.7z + 0.6 &&
        # 0.7r + W > 1.7z - 1.0
        lrg &= ( (w1flux*rflux**complex(0.7)).real > 
                 ((zflux**complex(1.7))*10**(-0.4*0.6)).real  )
        lrg &= ( (w1flux*rflux**complex(0.7)).real < 
                 ((zflux**complex(1.7))*10**(0.4*1.0)).real )
        # ADM note the trick of making the exponents complex and taking the real
        # ADM part to allow negative fluxes to be raised to a fractional power.

        # Now for the work-horse sliding flux-color cut:
        # ADM updated Zhou/Newman cut:
        # mlrg2 = z-2*(r-z-1.2) < 19.45 -> 3*z < 19.45-2.4-2*r
        lrg &= (zflux**3 > 10**(0.4*(22.5+2.4-19.45))*rflux**2)
        # Another guard against bright & red outliers
        # mlrg2 = z-2*(r-z-1.2) > 17.4 -> 3*z > 17.4-2.4-2*r
        lrg &= (zflux**3 < 10**(0.4*(22.5+2.4-17.4))*rflux**2)

        # Finally, a cut to exclude the z<0.4 objects while retaining the elbow at
        # z=0.4-0.5.  r-z>1.2 || (good_data_in_g and g-r>1.7).  Note that we do not
        # require gflux>0.
        lrg &= np.logical_or((zflux > 10**(0.4*1.2)*rflux), (ggood & (rflux>10**(0.4*1.7)*gflux)))

    return lrg


def isLRG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
          rflux_snr=None, zflux_snr=None, w1flux_snr=None,
          gflux_ivar=None, primary=None, south=True):
    """Convenience function for backwards-compatability prior to north/south split.
       
    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1 and W2 bands (if needed).
        rflux_snr, zflux_snr, w1flux_snr: array_like
            The signal-to-noise in the r, z and W1 bands defined as the flux
            per band divided by sigma (flux x the sqrt of the inverse variance).
        gflux_ivar: array_like
            The inverse variance of the flux in g-band.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to True
            Call isLRG_north if south=False, otherwise call isLRG_south.
    
    Returns:
        mask : array_like. True if and only if the object is an LRG
            target.
    """
    if south==False:
        return isLRG_north(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                           rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr,
                           gflux_ivar=gflux_ivar, primary=primary)
    else:
        return isLRG_south(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                           rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr,
                           gflux_ivar=gflux_ivar, primary=primary)


def isLRG_north(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                rflux_snr=None, zflux_snr=None, w1flux_snr=None,
                gflux_ivar=None, primary=None):
    """Target Definition of LRG for the BASS/MzLS photometric system. Returns a boolean array.
    (see :func:`~desitarget.sv1.sv1_cuts.isLRG_south`).
    """
    # ADM currently no difference between N/S for LRG masking, so easiest
    # ADM just to use one function.
    return isLRG_south(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                       w2flux=w2flux, rflux_snr=rflux_snr, zflux_snr=zflux_snr,
                       w1flux_snr=w1flux_snr, gflux_ivar=gflux_ivar,
                       primary=primary)


def isLRG_south(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
          rflux_snr=None, zflux_snr=None, w1flux_snr=None,
          gflux_ivar=None, primary=None):
    """Target Definition of LRG for the DECaLS photometric system.. Returns a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1 and W2 bands (if needed).
        rflux_snr, zflux_snr, w1flux_snr: array_like
            The signal-to-noise in the r, z and W1 bands defined as the flux
            per band divided by sigma (flux x the sqrt of the inverse variance).
        gflux_ivar: array_like
            The inverse variance of the flux in g-band
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only if the object is an LRG
            target.

    Notes:
        - Current version (09/21/18) is version 17 on `the SV wiki`_.
    """
    #----- Luminous Red Galaxies
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    # Some basic quality in r, z, and W1.  Note by @moustakas: no allmask cuts
    # used!).  Also note: We do not require gflux>0!  Objects can be very red.
    lrg = primary.copy()
    lrg &= (rflux_snr > 0) # and rallmask == 0
    lrg &= (zflux_snr > 0) # and zallmask == 0
    lrg &= (w1flux_snr > 4)
    lrg &= (rflux > 0)
    lrg &= (zflux > 0)
    ggood = (gflux_ivar > 0) # and gallmask == 0

    # Apply color, flux, and star-galaxy separation cuts
    lrg &= isLRG_colors_south(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                              w2flux=w2flux, ggood=ggood, primary=primary)

    return lrg


def isLRGpass(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
          rflux_snr=None, zflux_snr=None, w1flux_snr=None,
          gflux_ivar=None, primary=None, south=True):
    """Convenience function for backwards-compatability prior to north/south split.
       
    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1 and W2 bands (if needed).
        rflux_snr, zflux_snr, w1flux_snr: array_like
            The signal-to-noise in the r, z and W1 bands defined as the flux
            per band divided by sigma (flux x the sqrt of the inverse variance).
        gflux_ivar: array_like
            The inverse variance of the flux in g-band.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to True
            Call isLRG_north if south=False, otherwise call isLRG_south.
    
    Returns:
        mask0 : array_like. 
            True if and only if the object is an LRG target.
        mask1 : array_like. 
            True if the object is a ONE pass (bright) LRG target.
        mask2 : array_like. 
            True if the object is a TWO pass (fainter) LRG target.
    """
    if south==False:
        return isLRGpass_north(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                           rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr,
                           gflux_ivar=gflux_ivar, primary=primary)
    else:
        return isLRGpass_south(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                           rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr,
                           gflux_ivar=gflux_ivar, primary=primary)


def isLRGpass_north(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                    rflux_snr=None, zflux_snr=None, w1flux_snr=None,
                    gflux_ivar=None, primary=None):
    """LRGs in different passes (one pass, two pass etc.) for the MzLS/BASS system.
    (See :func:`~desitarget.sv1.sv1_cuts.isLRGpass_south` for details).
    """
    # ADM currently no difference between N/S for LRG pass selection, so easiest
    # ADM just to use one function.
    return isLRGpass_south(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                           w2flux=w2flux, rflux_snr=rflux_snr, zflux_snr=zflux_snr,
                           w1flux_snr=w1flux_snr, gflux_ivar=gflux_ivar,
                           primary=primary)


def isLRGpass_south(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
          rflux_snr=None, zflux_snr=None, w1flux_snr=None,
          gflux_ivar=None, primary=None):
    """LRGs in different passes (one pass, two pass etc.) for the DECaLS system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1 and W2 bands (if needed).
        rflux_snr, zflux_snr, w1flux_snr: array_like
            The signal-to-noise in the r, z and W1 bands defined as the flux
            per band divided by sigma (flux x the sqrt of the inverse variance).
        gflux_ivar: array_like
            The inverse variance of the flux in g-band.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask0 : array_like. 
            True if and only if the object is an LRG target.
        mask1 : array_like. 
            True if the object is a ONE pass (bright) LRG target.
        mask2 : array_like. 
            True if the object is a TWO pass (fainter) LRG target.

    Notes:
        - Current version (09/21/18) is version 17 on `the SV wiki`_.
    """
    # ----- Luminous Red Galaxies
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    lrg = primary.copy()

    # ADM apply the color and flag selection for all LRGs.
    lrg &= isLRG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                 rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr,
                 gflux_ivar=gflux_ivar, primary=primary)

    lrg1pass = lrg.copy()
    lrg2pass = lrg.copy()

    # ADM one-pass LRGs are 18 (the BGS limit) <= z < 20.
    lrg1pass &= zflux > 10**((22.5-20.0)/2.5)
    lrg1pass &= zflux <= 10**((22.5-18.0)/2.5)

    # ADM two-pass LRGs are 20 <= z < 20.4.
    lrg2pass &= zflux > 10**((22.5-20.4)/2.5)
    lrg2pass &= zflux <= 10**((22.5-20.0)/2.5)

    return lrg, lrg1pass, lrg2pass


def isELG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None,
                gallmask=None, rallmask=None, zallmask=None, south=True):
    """Convenience function for backwards-compatability prior to north/south split.
    
    Args:   
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        gallmask, rallmask, zallmask: array_like
            Bitwise mask set if the central pixel from all images
            satisfy each condition in g, r, z
        south: boolean, defaults to True
            Call isELG_north if south=False, otherwise call isELG_south.

    Returns:
        mask : array_like. True if and only if the object is an ELG
            target.
    """
    if south==False:
        return isELG_north(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                            gallmask=gallmask, rallmask=rallmask, zallmask=zallmask)
    else:
        return isELG_south(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux)


def isELG_north(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None,
                gallmask=None, rallmask=None, zallmask=None):
    """Target Definition of ELG for the BASS/MzLS photometric system. Returning a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        gallmask, rallmask, zallmask: array_like
            Bitwise mask set if the central pixel from all images 
            satisfy each condition in g, r, z 

    Returns:
        mask : array_like. True if and only if the object is an ELG
            target.

    """

    #----- Emission Line Galaxies
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    elg = primary.copy()

    elg &= (gallmask == 0)
    elg &= (rallmask == 0)
    elg &= (zallmask == 0)
    
    elg &= gflux < 10**((22.5-21.0)/2.5)                       # g>21
    elg &= gflux > 10**((22.5-23.7)/2.5)                       # g<23.7
    elg &= rflux > 10**((22.5-23.3)/2.5)                       # r<23.3
    elg &= zflux > rflux * 10**(0.3/2.5)                       # (r-z)>0.3
    elg &= zflux < rflux * 10**(1.6/2.5)                       # (r-z)<1.6

    # Clip to avoid warnings from negative numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    # ADM this is the original FDR cut to remove stars and low-z galaxies.
    #elg &= rflux**2.15 < gflux * zflux**1.15 * 10**(-0.15/2.5) # (g-r)<1.15(r-z)-0.15
    elg &= rflux**2.40 < gflux * zflux**1.40 * 10**(-0.35/2.5) # (g-r)<1.40(r-z)-0.35
    elg &= zflux**1.2 < gflux * rflux**0.2 * 10**(1.6/2.5)     # (g-r)<1.6-1.2(r-z)

    return elg


def isELG_south(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None):
    """Target Definition of ELG for the DECaLS photometric system. Returning a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only if the object is an ELG
            target.

    """
    #----- Emission Line Galaxies
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    elg = primary.copy()
    elg &= gflux < 10**((22.5-21.0)/2.5)                       # g>21
    elg &= rflux > 10**((22.5-23.4)/2.5)                       # r<23.4
    elg &= zflux > rflux * 10**(0.3/2.5)                       # (r-z)>0.3
    elg &= zflux < rflux * 10**(1.6/2.5)                       # (r-z)<1.6

    # Clip to avoid warnings from negative numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    elg &= rflux**2.15 < gflux * zflux**1.15 * 10**(-0.15/2.5) # (g-r)<1.15(r-z)-0.15
    elg &= zflux**1.2 < gflux * rflux**0.2 * 10**(1.6/2.5)     # (g-r)<1.6-1.2(r-z)

    return elg


def isSTD_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None):
    """Select STD stars based on Legacy Surveys color cuts. Returns a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : boolean array, True if the object has colors like a STD star target

    Notes:
        - Current version (08/01/18) is version 121 on the wiki:
            https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection?version=121#STD
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

    Args:
        primary: array_like or None
          If given, the BRICK_PRIMARY column of the catalogue.
        gaia: boolean array_like or None
            True if there is a match between this object in the Legacy
            Surveys and in Gaia.
        astrometricexcessnoise: array_like or None
            Excess noise of the source in Gaia (as in the Gaia Data Model).
        pmra, pmdec, parallax: array_like or None
            Gaia-based proper motion in RA and Dec and parallax
            (same units as the Gaia data model).
        dupsource: array_like or None
            Whether the source is a duplicate in Gaia (as in the Gaia Data model).
        paramssolved: array_like or None
            How many parameters were solved for in Gaia (as in the Gaia Data model).
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            (Extinction-corrected) Gaia-based g-, b- and r-band MAGNITUDES
            (same units as the Gaia data model).

    Returns:
        mask : boolean array, True if the object passes Gaia quality cuts.

        Notes:
        - Current version (08/01/18) is version 121 on the wiki:
        https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection?version=121#STD
        - Gaia data model is at:
        https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
    """
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    std = primary.copy()

    # ADM Bp and Rp are both measured.
    std &=  ~np.isnan(gaiabmag - gaiarmag)
    
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
          usegaia=True):
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
            True if there is a match between this object in the Legacy
            Surveys and in Gaia.
        astrometricexcessnoise: array_like or None
            Excess noise of the source in Gaia (as in the Gaia Data Model).
        paramssolved: array_like or None
            How many parameters were solved for in Gaia (as in the Gaia Data model).
        pmra, pmdec, parallax: array_like or None
            Gaia-based proper motion in RA and Dec and parallax
            (same units as the Gaia data model).
        dupsource: array_like or None
            Whether the source is a duplicate in Gaia (as in the Gaia Data model).
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            Gaia-based g-, b- and r-band MAGNITUDES (same units as Gaia data model).
        bright: boolean, defaults to ``False`` 
           if ``True`` apply magnitude cuts for "bright" conditions; otherwise, 
           choose "normal" brightness standards. Cut is performed on `gaiagmag`.
        usegaia: boolean, defaults to ``True``
           if ``True`` then  call :func:`~desitarget.cuts.isSTD_gaia` to set the 
           logic cuts. If Gaia is not available (perhaps if you're using mocks)
           then send ``False`` and pass `gaiagmag` as 22.5-2.5*np.log10(`robs`) 
           where `robs` is `rflux` without a correction.for Galactic extinction.

    Returns:
        mask : boolean array, True if the object has colors like a STD star

    Notes:
        - Gaia data model is at:
            https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
        - Current version (08/01/18) is version 121 on the wiki:
            https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection?version=121#STD
    """
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    std = primary.copy()

    # ADM apply the Legacy Surveys (optical) magnitude and color cuts.
    std &= isSTD_colors(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux)

    # ADM apply the Gaia quality cuts.
    if usegaia:
        std &= isSTD_gaia(primary=primary, gaia=gaia, astrometricexcessnoise=astrometricexcessnoise, 
                          pmra=pmra, pmdec=pmdec, parallax=parallax,
                          dupsource=dupsource, paramssolved=paramssolved,
                          gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag)

    # ADM apply type=PSF cut.
    std &= _psflike(objtype)

    # ADM apply fracflux, S/N cuts and number of observations cuts.
    fracflux = [gfracflux, rfracflux, zfracflux]
    fluxivar = [gfluxivar, rfluxivar, zfluxivar]
    nobs = [gnobs, rnobs, znobs]
    fracmasked = [gfracmasked, rfracmasked, zfracmasked]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # fracflux can be Inf/NaN
        for bandint in (0, 1, 2):  # g, r, z
            std &= fracflux[bandint] < 0.01
            std &= fluxivar[bandint] > 0
            std &= nobs[bandint] > 0
            std &= fracmasked[bandint] < 0.6

    # ADM brightness cuts in Gaia G-band.
    if bright:
        gbright = 15.
        gfaint = 18.
    else:
        gbright = 16.
        gfaint = 19.

    std &= gaiagmag >= gbright
    std &= gaiagmag < gfaint

    return std


def isMWS_main(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
               objtype=None, gaia=None, primary=None,
               pmra=None, pmdec=None, parallax=None, obs_rflux=None,
               gaiagmag=None, gaiabmag=None, gaiarmag=None, south=True):
    """Set bits for ``MWS_MAIN`` targets.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like or None
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            The ``TYPE`` column of `the Legacy Surveys`_ catalogue.
        gaia: boolean array_like or None
            True if there is a match between this object in
            `the Legacy Surveys`_ and in Gaia.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        pmra, pmdec, parallax: array_like or None
            Gaia-based proper motion in RA and Dec and parallax.
        obs_rflux: array_like or None
            ``rflux`` but WITHOUT any Galactic extinction correction.
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            (Extinction-corrected) Gaia-based g-, b- and r-band MAGNITUDES.
        south: boolean, defaults to ``True``
            Call :func:`~desitarget.cuts.isMWS_main_north` if ``south=False``,
            otherwise call :func:`~desitarget.cuts.isMWS_main_south`.

    Returns:
        mask : array_like. ``True`` if and only if the object is a ``MWS_MAIN`` target.

    Notes:
        Gaia quantities have the same units as `the Gaia data model`_.
    """
    if south==False:
        return isMWS_main_north(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
            objtype=objtype, gaia=gaia, primary=primary, pmra=pmra, pmdec=pmdec, parallax=parallax,
            obs_rflux=obs_rflux, gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag)
    else:
        return isMWS_main_south(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
            objtype=objtype, gaia=gaia, primary=primary, pmra=pmra, pmdec=pmdec, parallax=parallax,
            obs_rflux=obs_rflux, gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag)

def isMWS_main_north(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                     objtype=None, gaia=None, primary=None,
                     pmra=None, pmdec=None, parallax=None,
                     obs_rflux=None, gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for ``MWS_MAIN`` targets for the BASS/MzLS photometric system
    (see :func:`~desitarget.cuts.isMWS_main`).
    """
    # ADM currently no difference between N/S for MWS, so easiest
    # ADM just to use one function.
    return isMWS_main_south(gflux=gflux,rflux=rflux,zflux=zflux,w1flux=w1flux,w2flux=w2flux,
                            objtype=objtype,gaia=gaia,primary=primary,
                            pmra=pmra,pmdec=pmdec,parallax=parallax,obs_rflux=obs_rflux,
                            gaiagmag=gaiagmag,gaiabmag=gaiabmag,gaiarmag=gaiarmag)

def isMWS_main_south(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                     objtype=None, gaia=None, primary=None,
                     pmra=None, pmdec=None, parallax=None,
                     obs_rflux=None, gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for ``MWS_MAIN`` targets for the DECaLS photometric system
    (see :func:`~desitarget.cuts.isMWS_main`).
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM do not target any objects for which entries are NaN
    # ADM and turn off the NaNs for those entries.
    nans = (np.isnan(rflux) | np.isnan(gflux) |
               np.isnan(parallax) | np.isnan(pmra) | np.isnan(pmdec))
    w = np.where(nans)[0]
    if len(w) > 0:
        #A DM make copies as we are reassigning values.
        rflux, gflux, obs_rflux = rflux.copy(), gflux.copy(), obs_rflux.copy()
        parallax, pmra, pmdec = parallax.copy(), pmra.copy(), pmdec.copy()
        rflux[w], gflux[w], obs_rflux[w] = 0., 0., 0.
        parallax[w], pmra[w], pmdec[w] = 0., 0., 0.
        mws &= ~nans
        log.info('{}/{} NaNs in file...t = {:.1f}s'
                 .format(len(w),len(mws),time()-start))

    # ADM apply the selection for all MWS-MAIN targets.
    # ADM main targets match to a Gaia source.
    mws &= gaia
    # ADM main targets are point-like.
    mws &= _psflike(objtype)
    # ADM main targets are 16 <= r < 19.
    mws &= rflux > 10**((22.5-19.0)/2.5)
    mws &= rflux <= 10**((22.5-16.0)/2.5)
    # ADM main targets are robs < 20.
    mws &= obs_rflux > 10**((22.5-20.0)/2.5)

    # ADM calculate the overall proper motion magnitude.
    # ADM inexplicably I'm getting a Runtimewarning here for
    # ADM a few values in the sqrt, so I'm catching it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = np.sqrt(pmra**2. + pmdec**2.)

    # ADM make a copy of the main bits for a red/blue split.
    red = mws.copy()
    blue = mws.copy()

    # ADM MWS-BLUE is g-r < 0.7.
    blue &= rflux < gflux * 10**(0.7/2.5)                      # (g-r)<0.7

    # ADM MWS-RED is g-r >= 0.7 and parallax < 1mas...
    red &= parallax < 1.
    red &= rflux >= gflux * 10**(0.7/2.5)                      # (g-r)>=0.7
    # ADM ...and proper motion < 7.
    red &= pm < 7.

    return mws, red, blue


def isMWS_nearby(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                 objtype=None, gaia=None, primary=None,
                 pmra=None, pmdec=None, parallax=None, parallaxerr=None,
                 obs_rflux=None, gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for NEARBY Milky Way Survey targets.

    Notes:
    - Current version (09/20/18) is version 129 on `the wiki`_.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like or None
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            The TYPE column of the catalogue to restrict to point sources.
        gaia: boolean array_like or None
            True if there is a match between this object in the Legacy
            Surveys and in Gaia.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        pmra, pmdec, parallax, parallaxerr: array_like or None
            Gaia-based proper motion in RA and Dec and parallax (and
            uncertainty) (same units as `the Gaia data model`_).
        pmra, pmdec, parallax, parallaxerr: array_like or None
            Gaia-based proper motion in RA and Dec and parallax (and
            uncertainty) (same units as the Gaia data model, e.g.:
            https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html).
        obs_rflux: array_like or None
            `rflux` but WITHOUT any Galactic extinction correction
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            (Extinction-corrected) Gaia-based g-, b- and r-band MAGNITUDES
            (same units as the Gaia data model).

    Returns:
        mask : array_like. 
            True if and only if the object is a MWS-NEARBY target.

    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM do not target any objects for which entries are NaN
    # ADM and turn off the NaNs for those entries.
    nans = np.isnan(gaiagmag) | np.isnan(parallax)
    w = np.where(nans)[0]
    if len(w) > 0:
        # ADM make copies as we are reassigning values.
        parallax, gaiagmag = parallax.copy(), gaiagmag.copy()
        parallax[w], gaiagmag[w] = 0., 0.
        mws &= ~nans
        log.info('{}/{} NaNs in file...t = {:.1f}s'
                 .format(len(w),len(mws),time()-start))

    # ADM apply the selection for all MWS-NEARBY targets.
    # ADM must be a Legacy Surveys object that matches a Gaia source.
    mws &= gaia
    # ADM Gaia G mag of less than 20.
    mws &= gaiagmag < 20.
    # ADM parallax cut corresponding to 100pc.
    mws &= (parallax + parallaxerr) > 10. # NB: "+" is correct
    # ADM NOTE TO THE MWS GROUP: There is no bright cut on G. IS THAT THE REQUIRED BEHAVIOR?

    return mws


def isMWS_WD(primary=None, gaia=None, galb=None, astrometricexcessnoise=None, 
             pmra=None, pmdec=None, parallax=None, parallaxovererror=None,
             photbprpexcessfactor=None, astrometricsigma5dmax=None,
             gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for WHITE DWARF Milky Way Survey targets.

    Args:
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        gaia: boolean array_like or None
            True if there is a match between this object in the Legacy
            Surveys and in Gaia.
        galb: array_like or None
            Galactic latitude (degrees).
        astrometricexcessnoise: array_like or None
            Excess noise of the source in Gaia (as in the Gaia Data Model).
        pmra, pmdec, parallax, parallaxovererror: array_like or None
            Gaia-based proper motion in RA and Dec, and parallax and error
            (same units as the Gaia data model).
        photbprpexcessfactor: array_like or None
            Gaia_based BP/RP excess factor (as in the Gaia Data model).
        astrometricsigma5dmax: array_like or None
            Longest semi-major axis of 5-d error ellipsoid (as in Gaia Data model).
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            (Extinction-corrected) Gaia-based g-, b- and r-band MAGNITUDES
            (same units as the Gaia data model).

    Returns:
        mask : array_like. 
            True if and only if the object is a MWS-WD target.

    Notes:
        - Gaia data model is at:
            https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
        - Current version (08/01/18) is version 121 on the wiki:
            https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection?version=121#WhiteDwarfsMWS-WD

    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    # ADM do not target any objects for which entries are NaN
    # ADM and turn off the NaNs for those entries.
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
                 .format(len(w),len(mws),time()-start))

    # ADM apply the selection for all MWS-WD targets.
    # ADM must be a Legacy Surveys object that matches a Gaia source.
    mws &= gaia
    # ADM Gaia G mag of less than 20.
    mws &= gaiagmag < 20.

    # ADM Galactic b at least 20o from the plane.
    mws &= np.abs(galb) > 20.

    # ADM gentle cut on parallax significance.
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
    # ADM imaging catalogs. Until they are, ignore these cuts.
    if photbprpexcessfactor is not None:
        # ADM remove problem objects, which often have bad astrometry.
        mws &= photbprpexcessfactor < 1.7 + 0.06*br*br

    if astrometricsigma5dmax is not None:
        # ADM Reject white dwarfs that have really poor astrometry while
        # ADM retaining white dwarfs that only have relatively poor astrometry.
        mws &= ( (astrometricsigma5dmax < 1.5) | 
                 ((astrometricexcessnoise < 1.) & (parallaxovererror > 4.) & (pm > 10.)) )

    return mws


def isMWSSTAR_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None):
    """Select a reasonable range of g-r colors for MWS targets. Returns a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : boolean array, True if the object has colors like an old stellar population,
        which is what we expect for the main MWS sample

    Notes:
        The full MWS target selection also includes PSF-like and fracflux
        cuts and will include Gaia information; this function is only to enforce
        a reasonable range of color/TEFF when simulating data.

    """
    #----- Old stars, g-r > 0
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    mwsstar = primary.copy()

    #- colors g-r > 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grcolor = 2.5 * np.log10(rflux / gflux)
        mwsstar &= (grcolor > 0.0)

    return mwsstar


def isQSO_colors(gflux, rflux, zflux, w1flux, w2flux, optical=False, south=True):
    """Convenience function for backwards-compatability prior to north/south split.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        optical: boolean, defaults to False
            Just apply optical color-cuts
        south: boolean, defaults to ``True``
            Call isQSO_colors_north if ``south=False``, otherwise call isQSO_colors_south.

    Returns:
        mask : array_like. True if the object has QSO-like colors.
    """
    if south == False:
        return isQSO_colors_north(gflux, rflux, zflux, w1flux, w2flux, 
                                  optical=optical)
    else:
        return isQSO_colors_south(gflux, rflux, zflux, w1flux, w2flux, 
                                  optical=optical)


def isQSO_colors_north(gflux, rflux, zflux, w1flux, w2flux, optical=False):
    """Tests if sources have quasar-like colors for the BASS/MzLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        optical: boolean, defaults to False
            Just apply optical color-cuts

    Returns:
        mask : array_like. True if the object has QSO-like colors.
    """
    #----- Quasars
    # Create some composite fluxes.
    wflux = 0.75* w1flux + 0.25*w2flux
    grzflux = (gflux + 0.8*rflux + 0.5*zflux) / 2.3

    qso = np.ones(len(gflux), dtype='?')
    qso &= rflux > 10**((22.5-22.7)/2.5)    # r<22.7
    qso &= grzflux < 10**((22.5-17)/2.5)    # grz>17
    qso &= rflux < gflux * 10**(1.3/2.5)    # (g-r)<1.3
    qso &= zflux > rflux * 10**(-0.3/2.5)   # (r-z)>-0.3
    qso &= zflux < rflux * 10**(1.1/2.5)    # (r-z)<1.1

    if not optical:
        qso &= w2flux > w1flux * 10**(-0.4/2.5) # (W1-W2)>-0.4
        qso &= wflux * gflux > zflux * grzflux * 10**(-1.0/2.5) # (grz-W)>(g-z)-1.0

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


def isQSO_colors_south(gflux, rflux, zflux, w1flux, w2flux, optical=False):
    """Tests if sources have quasar-like colors for the DECaLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        optical: boolean, defaults to False
            Just apply optical color-cuts

    Returns:
        mask : array_like. True if the object has QSO-like colors.
    """
    #----- Quasars
    # Create some composite fluxes.
    wflux = 0.75* w1flux + 0.25*w2flux
    grzflux = (gflux + 0.8*rflux + 0.5*zflux) / 2.3

    qso = np.ones(len(gflux), dtype='?')
    qso &= rflux > 10**((22.5-22.7)/2.5)    # r<22.7
    qso &= grzflux < 10**((22.5-17)/2.5)    # grz>17
    qso &= rflux < gflux * 10**(1.3/2.5)    # (g-r)<1.3
    qso &= zflux > rflux * 10**(-0.3/2.5)   # (r-z)>-0.3
    qso &= zflux < rflux * 10**(1.1/2.5)    # (r-z)<1.1

    if not optical:
        qso &= w2flux > w1flux * 10**(-0.4/2.5) # (W1-W2)>-0.4
        qso &= wflux * gflux > zflux * grzflux * 10**(-1.0/2.5) # (grz-W)>(g-z)-1.0

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


def isQSO_cuts(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr,
               deltaChi2, brightstarinblob=None,
               release=None, objtype=None, primary=None, south=True, optical=False):
    """Convenience function for backwards-compatability prior to north/south split.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        w1snr: array_like[ntargets]
            S/N in the W1 band.
        w2snr: array_like[ntargets]
            S/N in the W2 band.
        deltaChi2: array_like[ntargets]
            chi2 difference between PSF and SIMP models,  dchisq_PSF - dchisq_SIMP.
        brightstarinblob: boolean array_like or None
            ``True`` if the object shares a blob with a "bright" (Tycho-2) star.
        release: array_like[ntargets]
            `The Legacy Surveys`_ imaging RELEASE.
        objtype (optional): array_like or None
            If given, the TYPE column of the Tractor catalogue.
        primary (optional): array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to ``True``
            Call isQSO_cuts_north if ``south=False``, otherwise call isQSO_cuts_south.
        optical: boolean, defaults to `False`
            Just apply optical color-cuts

    Returns:
        mask : array_like. True if and only if the object is a QSO
            target.
    """
    if south == False:
        return isQSO_cuts_north(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr,
                                deltaChi2, brightstarinblob=brightstarinblob,
                                release=release, objtype=objtype, primary=primary, optical=optical)
    else:
        return isQSO_cuts_south(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr,
                                deltaChi2, brightstarinblob=brightstarinblob,
                                release=release, objtype=objtype, primary=primary, optical=optical)


def isQSO_cuts_north(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr,
                     deltaChi2, brightstarinblob=None,
                     release=None, objtype=None, primary=None, optical=False):
    """Cuts based QSO target selection for the BASS/MzLS photometric system.
    (see :func:`~desitarget.cuts.isQSO_cuts`).

    Notes:
        Uses isQSO_colors() to make color cuts first, then applies
            w1snr, w2snr, deltaChi2, and optionally primary and objtype cuts
    """
    qso = isQSO_colors_north(gflux=gflux, rflux=rflux, zflux=zflux,
                             w1flux=w1flux, w2flux=w2flux, optical=optical)

    qso &= w1snr > 4
    qso &= w2snr > 2

    # ADM default to RELEASE of 6000 if nothing is passed.
    if release is None:
        release = np.zeros_like(gflux, dtype='?')+6000

    qso &= ((deltaChi2>40.) | (release>=5000) )

    if primary is not None:
        qso &= primary

    if objtype is not None:
        qso &= _psflike(objtype)

    # CAC Reject objects flagged inside a blob.
    if brightstarinblob is not None:
        qso &= ~brightstarinblob

    return qso


def isQSO_cuts_south(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr,
                     deltaChi2, brightstarinblob=None,
                     release=None, objtype=None, primary=None, optical=False):
    """Cuts based QSO target selection for the DECaLS photometric system.
    (see :func:`~desitarget.cuts.isQSO_cuts`).

    Notes:
        Uses isQSO_colors() to make color cuts first, then applies
            w1snr, w2snr, deltaChi2, and optionally primary and objtype cuts
    """
    qso = isQSO_colors_south(gflux=gflux, rflux=rflux, zflux=zflux,
                             w1flux=w1flux, w2flux=w2flux, optical=optical)

    qso &= w1snr > 4
    qso &= w2snr > 2

    # ADM default to RELEASE of 5000 if nothing is passed.                                                                                                                                           
    if release is None:
        release = np.zeros_like(gflux, dtype='?')+5000

    qso &= ((deltaChi2>40.) | (release>=5000) )

    if primary is not None:
        qso &= primary

    if objtype is not None:
        qso &= _psflike(objtype)

    # CAC Reject objects flagged inside a blob.
    if brightstarinblob is not None:
        qso &= ~brightstarinblob

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
                               
    if south == False:
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
    if primary is None :
        primary = np.ones_like(gflux, dtype=bool)

    # RELEASE
    # ADM default to RELEASE of 5000 if nothing is passed.
    if release is None :
        release = np.zeros_like(gflux, dtype='?') + 5000
    release = np.atleast_1d(release)

    # Build variables for random forest
    nFeatures = 11 # Number of attributes describing each object to be classified by the rf
    nbEntries = rflux.size
    colors, r, photOK = _getColors(nbEntries, nFeatures, gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d(r)

    # Preselection to speed up the process
    rMax = 22.7 # r < 22.7
    rMin = 17.5 # r > 17.5
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
        colorsIndex =  np.arange(0, nbEntries, dtype=np.int64)
        colorsReducedIndex =  colorsIndex[preSelection]
    
        # Path to random forest files
        pathToRF = resource_filename('desitarget', 'data')
        # rf filenames
        rf_DR3_fileName = pathToRF + '/rf_model_dr3.npz'
        rf_DR5_fileName = pathToRF + '/rf_model_dr5.npz'
        rf_DR5_HighZ_fileName = pathToRF + '/rf_model_dr5_HighZ.npz'
        
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
                             0.88 - (tmp_r_Reduced - 20.8) * 0.025, 0.88)
            pcut[tmp_r_Reduced > 21.5] = 0.8625 - 0.05 * (tmp_r_Reduced[tmp_r_Reduced > 21.5] - 21.5)
            pcut[tmp_r_Reduced > 22.3] = 0.8225 - 0.53 * (tmp_r_Reduced[tmp_r_Reduced > 22.3] - 22.3)
            pcut_HighZ = np.where(tmp_r_Reduced > 20.5,
                                  0.55 - (tmp_r_Reduced - 20.5) * 0.025, 0.55)
            # Add rf proba test result to "qso" mask
            qso[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_proba >= pcut) | (tmp_rf_HighZ_proba >= pcut_HighZ)
    
    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "numpy.ndarray"
    if nbEntries == 1 :
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
    if primary is None :
        primary = np.ones_like(gflux, dtype=bool)

    # RELEASE
    # ADM default to RELEASE of 5000 if nothing is passed.
    if release is None :
        release = np.zeros_like(gflux, dtype='?') + 5000
    release = np.atleast_1d(release)

    # Build variables for random forest
    nFeatures = 11 # Number of attributes describing each object to be classified by the rf
    nbEntries = rflux.size
    colors, r, photOK = _getColors(nbEntries, nFeatures, gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d(r)

    # Preselection to speed up the process
    rMax = 22.7 # r < 22.7
    rMin = 17.5 # r > 17.5
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
        colorsReducedIndex =  colorsIndex[preSelection]
    
        # Path to random forest files
        pathToRF = resource_filename('desitarget', 'data')
        # rf filenames
        rf_DR3_fileName = pathToRF + '/rf_model_dr3.npz'
        rf_DR5_fileName = pathToRF + '/rf_model_dr5.npz'
        rf_DR5_HighZ_fileName = pathToRF + '/rf_model_dr5_HighZ.npz'
        
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
                            0.88 - (tmp_r_Reduced - 20.8) * 0.025, 0.88)
            pcut[tmp_r_Reduced > 21.5] = 0.8625 - 0.05 * (tmp_r_Reduced[tmp_r_Reduced > 21.5] - 21.5)
            pcut[tmp_r_Reduced > 22.3] = 0.8225 - 0.53 * (tmp_r_Reduced[tmp_r_Reduced > 22.3] - 22.3)
            pcut_HighZ = np.where(tmp_r_Reduced > 20.5,
                                  0.55 - (tmp_r_Reduced - 20.5) * 0.025, 0.55)
            # Add rf proba test result to "qso" mask
            qso[colorsReducedIndex[tmpReleaseOK]] = \
                (tmp_rf_proba >= pcut) | (tmp_rf_HighZ_proba >= pcut_HighZ)
    
    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "numpy.ndarray"
    if nbEntries == 1 :
        qso = qso[0]
    
    return qso


def isBGS(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
          gnobs=None, rnobs=None, znobs=None, gfracmasked=None, rfracmasked=None, zfracmasked=None,
          gfracflux=None, rfracflux=None, zfracflux=None, gfracin=None, rfracin=None, zfracin=None,
          gfluxivar=None, rfluxivar=None, zfluxivar=None, brightstarinblob=None, Grr=None,
          w1snr=None, gaiagmag=None, objtype=None, primary=None, south=True, targtype=None):
    """Convenience function for backwards-compatability prior to north/south split.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        gnobs, rnobs, znobs: array_like or None
            Number of observations in g, r, z bands.
        gfracmasked, rfracmasked, zfracmasked: array_like or None
            Profile-weighted fraction of pixels masked from all observations of this object in g,r,z.
        fracflux, rfracflux, zfracflux: array_like or None
            Profile-weighted fraction of the flux from other sources divided by the total flux in g,r,z.
        gfracin, rfracin, zfracin: array_like or None
            Fraction of a source's flux within the blob in g,r,z.
        gfluxivar, rfluxivar, zfluxivar: array_like or None
            inverse variance of FLUX g,r,z.
        brightstarinblob: boolean array_like or None
            ``True`` if the object shares a blob with a "bright" (Tycho-2) star.
        Grr: array_like or None
            Gaia G band magnitude minus observational r magnitude.
        w1snr: array_like or None
            W1 band signal to noise.
        gaiagmag: array_like or None
            Gaia G band magnitude.
        objtype: array_like or None
            If given, The TYPE column of the catalogue.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to ``True``
            Use cuts appropriate to the Northern imaging surveys (BASS/MzLS) if ``south=False``,
            otherwise use cuts appropriate to the Southern imaging survey (DECaLS).
        targtype: str, optional, defaults to ``faint``
            Pass ``bright`` to use colors appropriate to the ``BGS_BRIGHT`` selection
            or ``faint`` to use colors appropriate to the ``BGS_BRIGHT`` selection
            or ``wise`` to use colors appropriate to the ``BGS_BRIGHT`` selection.

    Returns:
        mask : array_like. True if and only if the object is a BGS target.
    """
    _check_BGS_targtype(targtype)

    #------ Bright Galaxy Survey
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()

    bgs &= notin_BGS_mask(gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                          gfracmasked=gfracmasked, rfracmasked=rfracmasked, zfracmasked=zfracmasked,
                          gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                          gfracin=gfracin, rfracin=rfracin, zfracin=zfracin, w1snr=w1snr, 
                          gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar, Grr=Grr, 
                          gaiagmag=gaiagmag, brightstarinblob=brightstarinblob, targtype=targtype)

    bgs &= isBGS_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                        south=south, targtype=targtype)

    return bgs


def notin_BGS_mask(gnobs=None, rnobs=None, znobs=None,
                   gfracmasked=None, rfracmasked=None, zfracmasked=None,
                   gfracflux=None, rfracflux=None, zfracflux=None,
                   gfracin=None, rfracin=None, zfracin=None, w1snr=None,
                   gfluxivar=None, rfluxivar=None, zfluxivar=None, Grr=None,
                   gaiagmag=None, brightstarinblob=None, targtype=None):
    """Standard set of masking cuts used by all BGS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isBGS_faint` for parameters).
    """
    _check_BGS_targtype(targtype)
    bgs = np.ones(len(gnobs), dtype='?')

    bgs &= (gnobs >= 1) & (rnobs >= 1) & (znobs >= 1)
    bgs &= (gfracmasked < 0.4) & (rfracmasked < 0.4) & (zfracmasked < 0.4)
    bgs &= (gfracflux < 5.0) & (rfracflux < 5.0) & (zfracflux < 5.0)
    bgs &= (gfracin > 0.3) & (rfracin > 0.3) & (zfracin > 0.3)
    bgs &= (gfluxivar > 0) & (rfluxivar > 0) & (zfluxivar > 0)

    bgs &= ~brightstarinblob

    if targtype == 'bright':
        bgs &= ( (Grr > 0.6) | (gaiagmag == 0) )
    elif targtype == 'faint':
        bgs &= ( (Grr > 0.6) | (gaiagmag == 0) )
    elif targtype == 'wise':
        bgs &= Grr < 0.4
        bgs &= Grr > -1
        bgs &= w1snr > 5
    else:
        _check_BGS_targtype(targtype)

    return bgs


def isBGS_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                 south=True, targtype=None):
    """Standard set of masking cuts used by all BGS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isBGS` for parameters).
    """
    bgs = np.ones(len(gflux), dtype='?')

    if targtype == 'bright':
        bgs &= rflux > 10**((22.5-19.5)/2.5)
    elif targtype == 'faint':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        bgs &= rflux <= 10**((22.5-19.5)/2.5)
    elif targtype == 'wise':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        bgs &= w1flux*gflux > (zflux*rflux)*10**(-0.2)
    else:
        _check_BGS_targtype(targtype)

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

