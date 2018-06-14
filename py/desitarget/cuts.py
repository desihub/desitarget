"""
desitarget.cuts
===============

Target Selection for DECALS catalogue data

https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (*e.g.* LRG, ELG or QSO).
"""
import warnings
from time import time
import os.path

import numbers
import sys

import numpy as np
from astropy.table import Table, Row
from pkg_resources import resource_filename

from desitarget import io
from desitarget.internal import sharedmem
import desitarget.targets
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask

from desitarget.gaiamatch import match_gaia_to_primary

#ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

#ADM start the clock
start = time()

def shift_photo_north_pure(gflux=None, rflux=None, zflux=None):
    """Same as :func:`~desitarget.cuts.shift_photo_north_pure` accounting for zero fluxes

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
    """Convert fluxes in the northern (BASS/MzLS) to the southern (DECaLS) system

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
    
    #ADM only use the g-band color shift when r and g are non-zero
    gshift = gflux * 10**(-0.4*0.029)
    w = np.where( (gflux != 0) & (rflux != 0) )
    gshift[w] = (gflux[w] * 10**(-0.4*0.029) * (gflux[w]/rflux[w])**complex(-0.068)).real

    #ADM only use the r-band color shift when r and z are non-zero
    #ADM and only use the z-band color shift when r and z are non-zero
    w = np.where( (rflux != 0) & (zflux != 0) )
    rshift = rflux * 10**(+0.4*0.012)
    zshift = zflux * 10**(-0.4*0.000)

    rshift[w] = (rflux[w] * 10**(+0.4*0.012) * (rflux[w]/zflux[w])**complex(-0.029)).real
    zshift[w] = (zflux[w] * 10**(-0.4*0.000) * (rflux[w]/zflux[w])**complex(+0.009)).real

    return gshift, rshift, zshift


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
    """See :func:`~desitarget.cuts.isLRG_north` for details.
    This function applies just the flux and color cuts for the BASS/MzLS photometric system.
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
        # This is the star-galaxy separation cut:
        # ADM original Eisenstein/Dawson cut
        # Wlrg = (z-W)-(r-z)/3 + 0.3 >0 , which is equiv to r+3*W < 4*z+0.9
        # lrg &= (rflux*w1flux**3 > (zflux**4)*10**(-0.4*0.9))
        # ADM updated Zhou/Newman cut:
        # Wlrg = -0.6 < (z-w1) - 0.7*(r-z) < 1.0 ->
        # 0.7r + W < 1.7z + 0.6 &&
        # 0.7r + W > 1.7z - 1.0
        lrg &= ( (w1flux*rflux**complex(0.7)).real > 
                 ((zflux**complex(1.7))*10**(-0.4*0.6)).real  )
        lrg &= ( (w1flux*rflux**complex(0.7)).real < 
                 ((zflux**complex(1.7))*10**(0.4*1.0)).real )
        #ADM note the trick of making the exponents complex and taking the real
        #ADM part to allow negative fluxes to be raised to a fractional power

        # Now for the work-horse sliding flux-color cut:
        # ADM original Eisenstein/Dawson cut:
        # mlrg2 = z-2*(r-z-1.2) < 19.6 -> 3*z < 19.6-2.4-2*r
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


def isLRG_colors_south(gflux=None, rflux=None, zflux=None, w1flux=None,
                        w2flux=None, ggood=None, primary=None):
    """See :func:`~desitarget.cuts.isLRG_south` for details.
    This function applies just the flux and color cuts for the DECaLS photometric system.
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
        # This is the star-galaxy separation cut:
        # ADM original Eisenstein/Dawson cut
        # Wlrg = (z-W)-(r-z)/3 + 0.3 >0 , which is equiv to r+3*W < 4*z+0.9
        # lrg &= (rflux*w1flux**3 > (zflux**4)*10**(-0.4*0.9))
        # ADM updated Zhou/Newman cut:
        # Wlrg = -0.6 < (z-w1) - 0.7*(r-z) < 1.0 ->
        # 0.7r + W < 1.7z + 0.6 &&
        # 0.7r + W > 1.7z - 1.0
        lrg &= ( (w1flux*rflux**complex(0.7)).real > 
                 ((zflux**complex(1.7))*10**(-0.4*0.6)).real  )
        lrg &= ( (w1flux*rflux**complex(0.7)).real < 
                 ((zflux**complex(1.7))*10**(0.4*1.0)).real )
        #ADM note the trick of making the exponents complex and taking the real
        #ADM part to allow negative fluxes to be raised to a fractional power

        # Now for the work-horse sliding flux-color cut:
        # ADM original Eisenstein/Dawson cut:
        # mlrg2 = z-2*(r-z-1.2) < 19.6 -> 3*z < 19.6-2.4-2*r
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
        This is Rongpu Zhou's update to the LRG selection discussed at
        the December, 2017 SLAC collaboration meeting (see, e.g.:
        https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=3400)
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
    lrg &= isLRG_colors_north(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                              w2flux=w2flux, ggood=ggood, primary=primary)

    return lrg


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
        This is Rongpu Zhou's update to the LRG selection discussed at
        the December, 2017 SLAC collaboration meeting (see, e.g.:
        https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=3400)
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
        mask : array_like. True if and only if the object is an LRG
            target.
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
    """
    #----- Luminous Red Galaxies
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    lrg = primary.copy()

    #ADM apply the color and flag selection for all LRGs
    lrg &= isLRG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                 rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr,
                 gflux_ivar=gflux_ivar, primary=primary)

    lrg1pass = lrg.copy()
    lrg2pass = lrg.copy()

    #ADM one-pass LRGs are 18 (the BGS limit) <= z < 20
    lrg1pass &= zflux > 10**((22.5-20.0)/2.5)
    lrg1pass &= zflux <= 10**((22.5-18.0)/2.5)

    #ADM two-pass LRGs are 20 (the BGS limit) <= z < 20
    lrg2pass &= zflux > 10**((22.5-20.4)/2.5)
    lrg2pass &= zflux <= 10**((22.5-20.0)/2.5)

    return lrg, lrg1pass, lrg2pass


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
    """
    #----- Luminous Red Galaxies
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    lrg = primary.copy()

    #ADM apply the color and flag selection for all LRGs
    lrg &= isLRG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                 rflux_snr=rflux_snr, zflux_snr=zflux_snr, w1flux_snr=w1flux_snr,
                 gflux_ivar=gflux_ivar, primary=primary)

    lrg1pass = lrg.copy()
    lrg2pass = lrg.copy()

    #ADM one-pass LRGs are 18 (the BGS limit) <= z < 20
    lrg1pass &= zflux > 10**((22.5-20.0)/2.5)
    lrg1pass &= zflux <= 10**((22.5-18.0)/2.5)

    #ADM two-pass LRGs are 20 (the BGS limit) <= z < 20
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
    #ADM this is the original FDR cut to remove stars and low-z galaxies
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


def isFSTD_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None):
    """Select FSTD targets just based on color cuts. Returns a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : boolean array, True if the object has colors like an FSTD
    """

    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    fstd = primary.copy()

    # Clip to avoid warnings from negative numbers.
    gflux = gflux.clip(0)
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)

    # colors near BD+17
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grcolor = 2.5 * np.log10(rflux / gflux)
        rzcolor = 2.5 * np.log10(zflux / rflux)
        fstd &= (grcolor - 0.26)**2 + (rzcolor - 0.13)**2 < 0.06**2

    return fstd


def isFSTD(gflux=None, rflux=None, zflux=None, primary=None, 
           gfracflux=None, rfracflux=None, zfracflux=None,
           gsnr=None, rsnr=None, zsnr=None,
           objtype=None, obs_rflux=None, bright=False):
    """Select FSTD targets using color cuts and photometric quality cuts (PSF-like
    and fracflux).  See isFSTD_colors() for additional info.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
          The flux in nano-maggies of g, r, z, w1, and w2 bands.
        gfracflux, rfracflux, zfracflux: array_like
          Profile-weight fraction of the flux from other sources divided by the 
          total flux in g, r and z bands.
        gsnr, rsnr, zsnr: array_like
          The signal-to-noise ratio in g, r, and z bands.
        primary: array_like or None
          If given, the BRICK_PRIMARY column of the catalogue.
        bright: apply magnitude cuts for "bright" conditions; otherwise, choose
          "normal" brightness standards.

    Returns:
        mask : boolean array, True if the object has colors like an FSTD
    """
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    fstd = primary.copy()

    # Apply the magnitude and color cuts.
    fstd &= isFSTD_colors(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux)

    # Apply type=PSF, fracflux, and S/N cuts.
    fstd &= _psflike(objtype)

    #ADM probably a more elegant way to do this, coded it like this for
    #ADM data model transition from 2-D to 1-D arrays
    fracflux = [gfracflux, rfracflux, zfracflux]
    snr = [gsnr, rsnr, zsnr]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # fracflux can be Inf/NaN
        for j in (0, 1, 2):  # g, r, z
            fstd &= fracflux[j] < 0.04
            fstd &= snr[j] > 10

    # Observed flux; no Milky Way extinction
    if obs_rflux is None:
        obs_rflux = rflux

    if bright:
        rbright = 14.0
        rfaint = 17.0
    else:
        rbright = 16.0
        rfaint = 19.0

    fstd &= obs_rflux < 10**((22.5 - rbright)/2.5)
    fstd &= obs_rflux > 10**((22.5 - rfaint)/2.5)

    return fstd


def isMWS_main_north(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                     objtype=None, gaia=None, primary=None,
                     pmra=None, pmdec=None, parallax=None, 
                     obs_rflux=None, gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for MAIN MWS targets for the BASS/MzLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like or None
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            The TYPE column of the catalogue to restrict to point sources.
        gaia, boolean array_like or None
            True if there is a match between this object in the Legacy
            Surveys and in Gaia.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        pmra, pmdec, parallax: array_like or None
            Gaia-based proper motion in RA and Dec and parallax
            (same units as the Gaia data model, e.g.:
            https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html).
        obs_rflux: array_like or None
            `rflux` but WITHOUT any Galactic extinction correction
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            (Extinction-corrected) Gaia-based g-, b- and r-band MAGNITUDES
            (same units as the Gaia data model).

    Returns:
        mask0 : array_like. 
            True if and only if the object is a MWS-MAIN target.
        mask1 : array_like. 
            True if the object is a MWS-MAIN-RED target.
        mask2 : array_like. 
            True if the object is a MWS-MAIN-BLUE target.
    """
    #ADM currently no difference between N/S for MWS, so easiest
    #ADM just to use one function
    return isMWS_main_south(gflux=gflux,rflux=rflux,zflux=zflux,w1flux=w1flux,w2flux=w2flux,
                            objtype=objtype,gaia=gaia,primary=primary,
                            pmra=pmra,pmdec=pmdec,parallax=parallax,obs_rflux=obs_rflux, 
                            gaiagmag=gaiagmag,gaiabmag=gaiabmag,gaiarmag=gaiarmag)


def isMWS_main_south(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                     objtype=None, gaia=None, primary=None,
                     pmra=None, pmdec=None, parallax=None, 
                     obs_rflux=None, gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for MAIN MWS targets for the DECaLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like or None
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            The TYPE column of the catalogue to restrict to point sources.
        gaia, boolean array_like or None
            True if there is a match between this object in the Legacy
            Surveys and in Gaia.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        pmra, pmdec, parallax: array_like or None
            Gaia-based proper motion in RA and Dec and parallax
            (same units as the Gaia data model, e.g.:
            https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html).
        obs_rflux: array_like or None
            `rflux` but WITHOUT any Galactic extinction correction
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            (Extinction-corrected) Gaia-based g-, b- and r-band MAGNITUDES
            (same units as the Gaia data model).

    Returns:
        mask0 : array_like. 
            True if and only if the object is a MWS-MAIN target.
        mask1 : array_like. 
            True if the object is a MWS-MAIN-RED target.
        mask2 : array_like. 
            True if the object is a MWS-MAIN-BLUE target.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    #ADM do not target any objects for which entries are NaN
    #ADM and turn off the NaNs for those entries
    nans = (np.isnan(rflux) | np.isnan(gflux) |
               np.isnan(parallax) | np.isnan(pmra) | np.isnan(pmdec))
    w = np.where(nans)[0]
    if len(w) > 0:
        #ADM make copies as we are reassigning values
        rflux, gflux, obs_rflux = rflux.copy(), gflux.copy(), obs_rflux.copy()
        parallax, pmra, pmdec = parallax.copy(), pmra.copy(), pmdec.copy()
        rflux[w], gflux[w], obs_rflux[w] = 0., 0., 0.
        parallax[w], pmra[w], pmdec[w] = 0., 0., 0.
        mws &= ~nans
        log.info('{}/{} NaNs in file...t = {:.1f}s'
                 .format(len(w),len(mws),time()-start))

    #ADM apply the selection for all MWS-MAIN targets
    #ADM main targets match to a Gaia source
    mws &= gaia
    #ADM main targets are point-like
    mws &= _psflike(objtype)
    #ADM main targets are 16 <= r < 19 
    mws &= rflux > 10**((22.5-19.0)/2.5)
    mws &= rflux <= 10**((22.5-16.0)/2.5)
    #ADM main targets are robs < 20
    mws &= obs_rflux > 10**((22.5-20.0)/2.5)

    #ADM calculate the overall proper motion magnitude
    #ADM inexplicably I'm getting a Runtimewarning here for
    #ADM a few values in the sqrt, so I'm catching it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = np.sqrt(pmra**2. + pmdec**2.)
    
    #ADM make a copy of the main bits for a red/blue split
    red = mws.copy()
    blue = mws.copy()

    #ADM MWS-BLUE is g-r < 0.7
    blue &= rflux < gflux * 10**(0.7/2.5)                      # (g-r)<0.7

    #ADM MWS-RED is g-r >= 0.7 and parallax < 1mas...
    red &= parallax < 1.
    red &= rflux >= gflux * 10**(0.7/2.5)                      # (g-r)>=0.7
    #ADM ...and proper motion < 7.
    red &= pm < 7.

    #ADM no further splitting was deemed necessary as of 5 June 2018
    # (version 99 of https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection)
    #ADM make a copy of the red bits for the bright/faint split
#    rbright = red.copy()
#    rfaint = red.copy()

    #ADM the bright, red objects are r < 18 and |pm| < 7 mas/yr
#    rbright &= rflux > 10**((22.5-18.0)/2.5)
#    rbright &= pm < 7.

    #ADM the faint, red objects are r >= 18 and |pm| < 5 mas/yr
#    rfaint &= rflux <= 10**((22.5-18.0)/2.5)
#    rfaint &= pm < 5.

    return mws, red, blue


def isMWS_nearby(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                 objtype=None, gaia=None, primary=None,
                 pmra=None, pmdec=None, parallax=None, 
                 obs_rflux=None, gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for NEARBY Milky Way Survey targets.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like or None
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            The TYPE column of the catalogue to restrict to point sources.
        gaia, boolean array_like or None
            True if there is a match between this object in the Legacy
            Surveys and in Gaia.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        pmra, pmdec, parallax: array_like or None
            Gaia-based proper motion in RA and Dec and parallax
            (same units as the Gaia data model, e.g.:
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

    #ADM do not target any objects for which entries are NaN
    #ADM and turn off the NaNs for those entries
    nans = np.isnan(gaiagmag) | np.isnan(parallax)
    w = np.where(nans)[0]
    if len(w) > 0:
        #ADM make copies as we are reassigning values
        parallax, gaiagmag = parallax.copy(), gaiagmag.copy()
        parallax[w], gaiagmag[w] = 0., 0.
        mws &= ~nans
        log.info('{}/{} NaNs in file...t = {:.1f}s'
                 .format(len(w),len(mws),time()-start))

    #ADM apply the selection for all MWS-NEARBY targets
    #ADM must be a Legacy Surveys object that matches a Gaia source
    mws &= gaia
    #ADM Gaia G mag of less than 20
    mws &= gaiagmag < 20.
    #ADM parallax cut corresponding to 100pc
    mws &= parallax > 10.
    #ADM NOTE TO THE MWS GROUP: There is no bright cut on G. IS THAT THE REQUIRED BEHAVIOR?

    return mws


def isMWS_WD(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
             objtype=None, gaia=None, primary=None,
             pmra=None, pmdec=None, parallax=None, 
             obs_rflux=None, gaiagmag=None, gaiabmag=None, gaiarmag=None):
    """Set bits for WHITE DWARF Milky Way Survey targets.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like or None
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            The TYPE column of the catalogue to restrict to point sources.
        gaia, boolean array_like or None
            True if there is a match between this object in the Legacy
            Surveys and in Gaia.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        pmra, pmdec, parallax: array_like or None
            Gaia-based proper motion in RA and Dec and parallax
            (same units as the Gaia data model, e.g.:
            https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html).
        obs_rflux: array_like or None
            `rflux` but WITHOUT any Galactic extinction correction
        gaiagmag, gaiabmag, gaiarmag: array_like or None
            (Extinction-corrected) Gaia-based g-, b- and r-band MAGNITUDES
            (same units as the Gaia data model).

    Returns:
        mask : array_like. 
            True if and only if the object is a MWS-WD target.
    """
    if primary is None:
        primary = np.ones_like(gaia, dtype='?')
    mws = primary.copy()

    #ADM do not target any objects for which entries are NaN
    #ADM and turn off the NaNs for those entries
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

    #ADM apply the selection for all MWS-WD targets
    #ADM must be a Legacy Surveys object that matches a Gaia source
    mws &= gaia
    #ADM Gaia G mag of less than 20
    mws &= gaiagmag < 20.
    #ADM Color/absolute magnitude cut of G - 5log10(1/pi)+5 > 2.8(Bp-Rp) + 8    
    #ADM We've set parallax=zero for some objects with gaia=False hence the
    #ADM "where" in the divide and log10 (gaia=False objects can't be selected anyway)
    dist = np.divide(1.,parallax,where=parallax > 1e-16)
    #ADM Getting some strange Runtime warnings on a few values
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mws &= gaiagmag - 5.*np.log10(dist,where=dist > 1e-16) + 5. > 2.8*(gaiabmag-gaiarmag) + 8.
    #ADM NOTE TO THE MWS GROUP: If Parallaxes are negative (i.e. presumably "very distant
    #ADM but not measurably so") THEY WILL BE TARGETED BY THIS LOGIC. IS THAT THE REQUIRED BEHAVIOR?

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


def isBGS_faint(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                objtype=None, primary=None, south=True):
    """Convenience function for backwards-compatability prior to north/south split.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            If given, The TYPE column of the catalogue.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to ``True``
            Call isBGS_faint_north if south=False, otherwise call isBGS_faint_south.

    Returns:
        mask : array_like. True if and only if the object is a BGS target.
    """
    if south==False:
        return isBGS_faint_north(gflux, rflux, zflux, w1flux, w2flux, 
                                 objtype=objtype, primary=primary)
    else:
        return isBGS_faint_south(gflux, rflux, zflux, w1flux, w2flux, 
                                 objtype=objtype, primary=primary)


def isBGS_faint_north(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                objtype=None, primary=None):
    """Target Definition of BGS faint targets for the BASS/MzLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            If given, The TYPE column of the catalogue.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only if the object is a BGS target.
    """
    #------ Bright Galaxy Survey
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()
    bgs &= rflux > 10**((22.5-20.0)/2.5)
    bgs &= rflux <= 10**((22.5-19.5)/2.5)
    if objtype is not None:
        bgs &= ~_psflike(objtype)

    return bgs


def isBGS_faint_south(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                objtype=None, primary=None):
    """Target Definition of BGS faint targets for the DECaLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            If given, The TYPE column of the catalogue.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only if the object is a BGS target.
    """
    #------ Bright Galaxy Survey
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()
    bgs &= rflux > 10**((22.5-20.0)/2.5)
    bgs &= rflux <= 10**((22.5-19.5)/2.5)
    if objtype is not None:
        bgs &= ~_psflike(objtype)

    return bgs


def isBGS_bright(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                 objtype=None, primary=None, south=True):
    """Convenience function for backwards-compatability prior to north/south split.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            If given, The TYPE column of the catalogue.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue
        south: boolean, defaults to ``True``
            Call isBGS_bright_north if south=False, otherwise call isBGS_bright_south.

    Returns:
        mask : array_like. True if and only if the object is a BGS target.
    """
    if south==False:
        return isBGS_bright_north(gflux, rflux, zflux, w1flux, w2flux, 
                                  objtype=objtype, primary=primary)
    else:
        return isBGS_bright_south(gflux, rflux, zflux, w1flux, w2flux, 
                                  objtype=objtype, primary=primary)


def isBGS_bright_north(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                       objtype=None, primary=None):
    """Target Definition of BGS bright targets for the BASS/MzLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            If given, The TYPE column of the catalogue.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only if the object is a BGS target.
    """
    #------ Bright Galaxy Survey
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()
    bgs &= rflux > 10**((22.5-19.5)/2.5)
    if objtype is not None:
        bgs &= ~_psflike(objtype)
    return bgs


def isBGS_bright_south(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
                       objtype=None, primary=None):
    """Target Definition of BGS bright targets for the DECaLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            If given, The TYPE column of the catalogue.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only if the object is a BGS target.
    """
    #------ Bright Galaxy Survey
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()
    bgs &= rflux > 10**((22.5-19.5)/2.5)
    if objtype is not None:
        bgs &= ~_psflike(objtype)
    return bgs


def isQSO_colors(gflux, rflux, zflux, w1flux, w2flux, optical=False, south=True):
    """Convenience function for backwards-compatability prior to north/south split.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        optical: boolean, defaults to False
            Just apply optical color-cuts
        south: boolean, defaults to True
            Call isQSO_colors_north if south=False, otherwise call isQSO_colors_south.

    Returns:
        mask : array_like. True if the object has QSO-like colors.
    """
    if south==False:
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


def isQSO_cuts(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr, deltaChi2,
                   release=None, objtype=None, primary=None, south=True):
    """Convenience function for backwards-compatability prior to north/south split.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        w1snr: array_like[ntargets]
            S/N in the W1 band.
        w2snr: array_like[ntargets]
            S/N in the W2 band.
        deltaChi2: array_like[ntargets]
            chi2 difference between PSF and SIMP models,  dchisq_PSF - dchisq_SIMP
        release: array_like[ntargets]
            The Legacy Survey imaging RELEASE (e.g. http://legacysurvey.org/release/)
        objtype (optional): array_like or None
            If given, the TYPE column of the Tractor catalogue.
        primary (optional): array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to True
            Call isQSO_cuts_north if south=False, otherwise call isQSO_cuts_south.

    Returns:
        mask : array_like. True if and only if the object is a QSO
            target.
    """
    if south==False:
        return isQSO_cuts_north(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr, deltaChi2,
                                release=release, objtype=objects, primary=primary)
    else:
        return isQSO_cuts_south(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr, deltaChi2,
                                release=release, objtype=objtype, primary=primary)


def isQSO_cuts_north(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr, deltaChi2, 
               release=None, objtype=None, primary=None):
    """Cuts based QSO target selection for the BASS/MzLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        w1snr: array_like[ntargets]
            S/N in the W1 band.
        w2snr: array_like[ntargets]
            S/N in the W2 band.
        deltaChi2: array_like[ntargets]
            chi2 difference between PSF and SIMP models,  dchisq_PSF - dchisq_SIMP
        release: array_like[ntargets]
            The Legacy Survey imaging RELEASE (e.g. http://legacysurvey.org/release/)
        objtype (optional): array_like or None
            If given, the TYPE column of the Tractor catalogue.
        primary (optional): array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only if the object is a QSO
            target.

    Notes:
        Uses isQSO_colors() to make color cuts first, then applies
            w1snr, w2snr, deltaChi2, and optionally primary and objtype cuts

    """
    qso = isQSO_colors_north(gflux=gflux, rflux=rflux, zflux=zflux,
                             w1flux=w1flux, w2flux=w2flux)

    qso &= w1snr > 4
    qso &= w2snr > 2

    #ADM default to RELEASE of 6000 if nothing is passed
    if release is None:
        release = np.zeros_like(gflux, dtype='?')+6000

    qso &= ((deltaChi2>40.) | (release>=5000) )

    if primary is not None:
        qso &= primary

    if objtype is not None:
        qso &= _psflike(objtype)

    return qso


def isQSO_cuts_south(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr, deltaChi2, 
               release=None, objtype=None, primary=None):
    """Cuts based QSO target selection for the DECaLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        w1snr: array_like[ntargets]
            S/N in the W1 band.
        w2snr: array_like[ntargets]
            S/N in the W2 band.
        deltaChi2: array_like[ntargets]
            chi2 difference between PSF and SIMP models,  dchisq_PSF - dchisq_SIMP
        release: array_like[ntargets]
            The Legacy Survey imaging RELEASE (e.g. http://legacysurvey.org/release/)
        objtype (optional): array_like or None
            If given, the TYPE column of the Tractor catalogue.
        primary (optional): array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only if the object is a QSO
            target.

    Notes:
        Uses isQSO_colors() to make color cuts first, then applies
            w1snr, w2snr, deltaChi2, and optionally primary and objtype cuts

    """
    qso = isQSO_colors_south(gflux=gflux, rflux=rflux, zflux=zflux,
                             w1flux=w1flux, w2flux=w2flux)

    qso &= w1snr > 4
    qso &= w2snr > 2

    #ADM default to RELEASE of 5000 if nothing is passed
    if release is None:
        release = np.zeros_like(gflux, dtype='?')+5000

    qso &= ((deltaChi2>40.) | (release>=5000) )

    if primary is not None:
        qso &= primary

    if objtype is not None:
        qso &= _psflike(objtype)

    return qso


def isQSO_randomforest(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                    objtype=None, release=None, deltaChi2=None, primary=None, south=True):
    """Convenience function for backwards-compatability prior to north/south split.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        objtype: array_like or None
            If given, the TYPE column of the Tractor catalogue.
        release: array_like[ntargets]
            The Legacy Survey imaging RELEASE (e.g. http://legacysurvey.org/release/)
        deltaChi2: array_like or None
             If given, difference in chi2 bteween PSF and SIMP morphology
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.
        south: boolean, defaults to True
            Call isQSO_randomforest_north if south=False, 
            otherwise call isQSO_randomforest_south.

    Returns:
        mask : array_like. True if and only if the object is a QSO
            target.

    """
                               
    if south==False:
        return isQSO_randomforest_north(gflux=gflux, rflux=rflux, zflux=zflux,
                                w1flux=w1flux, w2flux=w2flux, objtype=objtype, 
                                release=release, deltaChi2=deltaChi2, primary=primary)
    else:
        return isQSO_randomforest_south(gflux=gflux, rflux=rflux, zflux=zflux,
                                w1flux=w1flux, w2flux=w2flux, objtype=objtype, 
                                release=release, deltaChi2=deltaChi2, primary=primary)

def isQSO_randomforest_north(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                       objtype=None, release=None, deltaChi2=None, primary=None):
    """
    Target definition of QSO using a random forest for the BASS/MzLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        objtype: array_like or None
            If given, the TYPE column of the Tractor catalogue.
        release: array_like[ntargets]
            The Legacy Survey imaging RELEASE (e.g. http://legacysurvey.org/release/).
        deltaChi2: array_like or None
             If given, difference in chi2 between PSF and SIMP morphology.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask: array_like
            True if and only if the object is a QSO target.
    """
    # BRICK_PRIMARY
    if primary is None :
        primary = np.ones_like(gflux, dtype=bool)

    # RELEASE
    # ADM default to RELEASE of 6000 if nothing is passed
    if release is None :
        release = np.zeros_like(gflux, dtype='?') + 6000
    release = np.atleast_1d( release )

    # Build variables for random forest
    nFeatures = 11 # Number of attributes describing each object to be classified by the rf
    nbEntries = rflux.size
    colors, r, photOK = _getColors(nbEntries, nFeatures, gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d( r )

    # Preselection to speed up the process
    rMax = 22.7 # r < 22.7
    rMin = 17.5 # r > 17.5
    preSelection = ( r < rMax ) & ( r > rMin ) & photOK & primary
    if objtype is not None :
        preSelection &= _psflike( objtype )
    if deltaChi2 is not None :
        deltaChi2 = np.atleast_1d( deltaChi2 )
        preSelection[ release < 5000 ] &= deltaChi2[ release < 5000 ] > 30.
    
    # "qso" mask initialized to "preSelection" mask
    qso = np.copy( preSelection )
    
    if np.any( preSelection ) :
        
        from desitarget.myRF import myRF
        
        # Data reduction to preselected objects
        colorsReduced = colors[ preSelection ]
        releaseReduced = release[ preSelection ]
        r_Reduced = r[ preSelection ]
        colorsIndex =  np.arange(0, nbEntries, dtype=np.int64)
        colorsReducedIndex =  colorsIndex[ preSelection ]
    
        # Path to random forest files
        pathToRF = resource_filename('desitarget', 'data')
        # rf filenames
        rf_DR3_fileName = pathToRF + '/rf_model_dr3.npz'
        rf_DR5_fileName = pathToRF + '/rf_model_dr5.npz'
        rf_DR5_HighZ_fileName = pathToRF + '/rf_model_dr5_HighZ.npz'
        
        tmpReleaseOK = releaseReduced < 6000
        if np.any( tmpReleaseOK ) :
            # rf initialization - colors data duplicated within "myRF"
            rf_DR3 = myRF( colorsReduced[ tmpReleaseOK ] , pathToRF ,
                           numberOfTrees=200 , version=1 )
            # rf loading
            rf_DR3.loadForest( rf_DR3_fileName )
            # Compute rf probabilities
            tmp_rf_proba = rf_DR3.predict_proba()
            # Compute optimized proba cut
            tmp_r_Reduced = r_Reduced[ tmpReleaseOK ]
            pcut = np.where( tmp_r_Reduced > 20.0 ,
                             0.95 - ( tmp_r_Reduced - 20.0 ) * 0.08 , 0.95 )
            # Add rf proba test result to "qso" mask            
            qso[ colorsReducedIndex[ tmpReleaseOK ] ] = tmp_rf_proba >= pcut
        
        tmpReleaseOK = releaseReduced >= 6000
        if np.any( tmpReleaseOK ) :
            # rf initialization - colors data duplicated within "myRF"
            rf_DR5 = myRF( colorsReduced[ tmpReleaseOK ] , pathToRF ,
                           numberOfTrees=500 , version=2 )
            rf_DR5_HighZ = myRF( colorsReduced[ tmpReleaseOK ] , pathToRF ,
                                 numberOfTrees=500 , version=2 )
            # rf loading
            rf_DR5.loadForest( rf_DR5_fileName )
            rf_DR5_HighZ.loadForest( rf_DR5_HighZ_fileName )
            # Compute rf probabilities
            tmp_rf_proba = rf_DR5.predict_proba()
            tmp_rf_HighZ_proba = rf_DR5_HighZ.predict_proba()
            # Compute optimized proba cut
            tmp_r_Reduced = r_Reduced[ tmpReleaseOK ]
            pcut = np.where( tmp_r_Reduced > 21.1 ,
                             0.82 - ( tmp_r_Reduced - 21.1 ) * 0.20 , 0.82 )
            pcut_HighZ = 0.42
            # Add rf proba test result to "qso" mask
            qso[ colorsReducedIndex[ tmpReleaseOK ] ] = \
                ( tmp_rf_proba >= pcut ) | ( tmp_rf_HighZ_proba >= pcut_HighZ )
    
    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "numpy.ndarray"
    if nbEntries == 1 :
        qso = qso[0]
    
    return qso

def isQSO_randomforest_south(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None,
                       objtype=None, release=None, deltaChi2=None, primary=None):
    """
    Target definition of QSO using a random forest for the DECaLS photometric system.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        objtype: array_like or None
            If given, the TYPE column of the Tractor catalogue.
        release: array_like[ntargets]
            The Legacy Survey imaging RELEASE (e.g. http://legacysurvey.org/release/).
        deltaChi2: array_like or None
             If given, difference in chi2 between PSF and SIMP morphology.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask: array_like
            True if and only if the object is a QSO target.
    """
    # BRICK_PRIMARY
    if primary is None :
        primary = np.ones_like(gflux, dtype=bool)

    # RELEASE
    # ADM default to RELEASE of 5000 if nothing is passed
    if release is None :
        release = np.zeros_like(gflux, dtype='?') + 5000
    release = np.atleast_1d( release )

    # Build variables for random forest
    nFeatures = 11 # Number of attributes describing each object to be classified by the rf
    nbEntries = rflux.size
    colors, r, photOK = _getColors(nbEntries, nFeatures, gflux, rflux, zflux, w1flux, w2flux)
    r = np.atleast_1d( r )

    # Preselection to speed up the process
    rMax = 22.7 # r < 22.7
    rMin = 17.5 # r > 17.5
    preSelection = ( r < rMax ) & ( r > rMin ) & photOK & primary
    if objtype is not None :
        preSelection &= _psflike( objtype )
    if deltaChi2 is not None :
        deltaChi2 = np.atleast_1d( deltaChi2 )
        preSelection[ release < 5000 ] &= deltaChi2[ release < 5000 ] > 30.
    
    # "qso" mask initialized to "preSelection" mask
    qso = np.copy( preSelection )
    
    if np.any( preSelection ) :
        
        from desitarget.myRF import myRF
        
        # Data reduction to preselected objects
        colorsReduced = colors[ preSelection ]
        releaseReduced = release[ preSelection ]
        r_Reduced = r[ preSelection ]
        colorsIndex =  np.arange(0, nbEntries, dtype=np.int64)
        colorsReducedIndex =  colorsIndex[ preSelection ]
    
        # Path to random forest files
        pathToRF = resource_filename('desitarget', 'data')
        # rf filenames
        rf_DR3_fileName = pathToRF + '/rf_model_dr3.npz'
        rf_DR5_fileName = pathToRF + '/rf_model_dr5.npz'
        rf_DR5_HighZ_fileName = pathToRF + '/rf_model_dr5_HighZ.npz'
        
        tmpReleaseOK = releaseReduced < 5000
        if np.any( tmpReleaseOK ) :
            # rf initialization - colors data duplicated within "myRF"
            rf_DR3 = myRF( colorsReduced[ tmpReleaseOK ] , pathToRF ,
                           numberOfTrees=200 , version=1 )
            # rf loading
            rf_DR3.loadForest( rf_DR3_fileName )
            # Compute rf probabilities
            tmp_rf_proba = rf_DR3.predict_proba()
            tmp_r_Reduced = r_Reduced[ tmpReleaseOK ]
            # Compute optimized proba cut
            pcut = np.where( tmp_r_Reduced > 20.0 ,
                             0.95 - ( tmp_r_Reduced - 20.0 ) * 0.08 , 0.95 )
            # Add rf proba test result to "qso" mask            
            qso[ colorsReducedIndex[ tmpReleaseOK ] ] = tmp_rf_proba >= pcut
        
        tmpReleaseOK = releaseReduced >= 5000
        if np.any( tmpReleaseOK ) :
            # rf initialization - colors data duplicated within "myRF"
            rf_DR5 = myRF( colorsReduced[ tmpReleaseOK ] , pathToRF ,
                           numberOfTrees=500 , version=2 )
            rf_DR5_HighZ = myRF( colorsReduced[ tmpReleaseOK ] , pathToRF ,
                                 numberOfTrees=500 , version=2 )
            # rf loading
            rf_DR5.loadForest( rf_DR5_fileName )
            rf_DR5_HighZ.loadForest( rf_DR5_HighZ_fileName )
            # Compute rf probabilities
            tmp_rf_proba = rf_DR5.predict_proba()
            tmp_rf_HighZ_proba = rf_DR5_HighZ.predict_proba()
            # Compute optimized proba cut
            tmp_r_Reduced = r_Reduced[ tmpReleaseOK ]
            pcut = np.where( tmp_r_Reduced > 20.8 ,
                             0.88 - ( tmp_r_Reduced - 20.8 ) * 0.025 , 0.88 )
            pcut[ tmp_r_Reduced > 21.5 ] = 0.8625 - 0.05 * ( tmp_r_Reduced[ tmp_r_Reduced > 21.5 ] - 21.5 )
            pcut[ tmp_r_Reduced > 22.3 ] = 0.8225 - 0.53 * ( tmp_r_Reduced[ tmp_r_Reduced > 22.3 ] - 22.3 )
            pcut_HighZ = np.where( tmp_r_Reduced > 20.5 ,
                                   0.55 - ( tmp_r_Reduced - 20.5 ) * 0.025 , 0.55 )
            # Add rf proba test result to "qso" mask
            qso[ colorsReducedIndex[ tmpReleaseOK ] ] = \
                ( tmp_rf_proba >= pcut ) | ( tmp_rf_HighZ_proba >= pcut_HighZ )
    
    # In case of call for a single object passed to the function with scalar arguments
    # Return "numpy.bool_" instead of "numpy.ndarray"
    if nbEntries == 1 :
        qso = qso[0]
    
    return qso

def _psflike(psftype):
    """ If the object is PSF """
    #ADM explicitly checking for NoneType. I can't see why we'd ever want to
    #ADM run this test on empty information. In the past we have had bugs where
    #ADM we forgot to pass objtype=objtype in, e.g., isFSTD
    if psftype is None:
        raise ValueError("NoneType submitted to _psfflike function")

    psftype = np.asarray(psftype)
    #ADM in Python3 these string literals become byte-like
    #ADM so to retain Python2 compatibility we need to check
    #ADM against both bytes and unicode
    #ADM, also 'PSF' for astropy.io.fits; 'PSF ' for fitsio (sigh)
    psflike = ( (psftype == 'PSF') | (psftype == b'PSF') | 
                (psftype == 'PSF ') | (psftype == b'PSF ') )
    return psflike

def _isonnorthphotsys(photsys):
    """ If the object is from the northen photometric system """
    #ADM explicitly checking for NoneType. I can't see why we'd ever want to
    #ADM run this test on empty information. In the past we have had bugs where
    #ADM we forgot to populate variables before passing them
    if photsys is None:
        raise ValueError("NoneType submitted to _isonnorthphotsys function")

    psftype = np.asarray(photsys)
    #ADM in Python3 these string literals become byte-like
    #ADM so to retain Python2 compatibility we need to check
    #ADM against both bytes and unicode
    northern = ( (photsys == 'N') | (photsys == b'N') )
    return northern

def _getColors(nbEntries, nfeatures, gflux, rflux, zflux, w1flux, w2flux):

    limitInf=1.e-04
    gflux = gflux.clip(limitInf)
    rflux = rflux.clip(limitInf)
    zflux = zflux.clip(limitInf)
    w1flux = w1flux.clip(limitInf)
    w2flux = w2flux.clip(limitInf)

    g=np.where( gflux>limitInf,22.5-2.5*np.log10(gflux), 0.)
    r=np.where( rflux>limitInf,22.5-2.5*np.log10(rflux), 0.)
    z=np.where( zflux>limitInf,22.5-2.5*np.log10(zflux), 0.)
    W1=np.where( w1flux>limitInf, 22.5-2.5*np.log10(w1flux), 0.)
    W2=np.where( w2flux>limitInf, 22.5-2.5*np.log10(w2flux), 0.)

    photOK = (g>0.) & (r>0.) & (z>0.) & (W1>0.) # & (W2>0.)

    colors  = np.zeros((nbEntries,nfeatures))
    colors[:,0]=g-r
    colors[:,1]=r-z
    colors[:,2]=g-z
    colors[:,3]=g-W1
    colors[:,4]=r-W1
    colors[:,5]=z-W1
    colors[:,6]=g-W2
    colors[:,7]=r-W2
    colors[:,8]=z-W2
    colors[:,9]=W1-W2
    colors[:,10]=r

    return colors, r, photOK

def _is_row(table):
    '''Return True/False if this is a row of a table instead of a full table

    supports numpy.ndarray, astropy.io.fits.FITS_rec, and astropy.table.Table
    '''
    import astropy.io.fits.fitsrec
    import astropy.table.row
    if isinstance(table, (astropy.io.fits.fitsrec.FITS_record, astropy.table.row.Row)) or \
        np.isscalar(table):
        return True
    else:
        return False

def unextinct_fluxes(objects):
    """Calculate unextincted DECam and WISE fluxes

    Args:
        objects: array or Table with columns FLUX_G, FLUX_R, FLUX_Z, 
            MW_TRANSMISSION_G, MW_TRANSMISSION_R, MW_TRANSMISSION_Z,
            FLUX_W1, FLUX_W2, MW_TRANSMISSION_W1, MW_TRANSMISSION_W2

    Returns:
        array or Table with columns GFLUX, RFLUX, ZFLUX, W1FLUX, W2FLUX

    Output type is Table if input is Table, otherwise numpy structured array
    """
    dtype = [('GFLUX', 'f4'), ('RFLUX', 'f4'), ('ZFLUX', 'f4'),
             ('W1FLUX', 'f4'), ('W2FLUX', 'f4')]
    if _is_row(objects):
        result = np.zeros(1, dtype=dtype)[0]
    else:
        result = np.zeros(len(objects), dtype=dtype)

#ADM This was a hack for DR3 because of some corrupt sweeps/Tractor files,
#ADM the comment can be removed if DR4/DR5 run OK. It's just here as a reminder.
#    dered_decam_flux = np.divide(objects['DECAM_FLUX'] , objects['DECAM_MW_TRANSMISSION'],
#                                 where=objects['DECAM_MW_TRANSMISSION']!=0)
    result['GFLUX'] = objects['FLUX_G'] / objects['MW_TRANSMISSION_G']
    result['RFLUX'] = objects['FLUX_R'] / objects['MW_TRANSMISSION_R']
    result['ZFLUX'] = objects['FLUX_Z'] / objects['MW_TRANSMISSION_Z']
    result['W1FLUX'] = objects['FLUX_W1'] / objects['MW_TRANSMISSION_W1']
    result['W2FLUX'] = objects['FLUX_W2'] / objects['MW_TRANSMISSION_W2']

    if isinstance(objects, Table):
        return Table(result)
    else:
        return result


def apply_cuts(objects, qso_selection='randomforest', match_to_gaia=True,
               gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Perform target selection on objects, returning target mask arrays

    Args:
        objects: numpy structured array with UPPERCASE columns needed for
            target selection, OR a string tractor/sweep filename

    Options:
        qso_selection : algorithm to use for QSO selection; valid options
            are 'colorcuts' and 'randomforest'
        match_to_gaia : defaults to ``True``
            if ``True``, match to Gaia DR2 chunks files and populate 
            Gaia columns to facilitate the MWS selection
        gaiadir : defaults to the the Gaia DR2 path at NERSC
           Root directory of a Gaia Data Release as used by the Legacy Surveys. 

    Returns:
        (desi_target, bgs_target, mws_target) where each element is
        an ndarray of target selection bitmask flags for each object

    Bugs:
        If objects is a astropy Table with lowercase column names, this
        converts them to UPPERCASE in-place, thus modifying the input table.
        To avoid this, pass in objects.copy() instead.

    See desitarget.targetmask for the definition of each bit
    """
    #- Check if objects is a filename instead of the actual data
    if isinstance(objects, str):
        objects = io.read_tractor(objects)

    #ADM add Gaia information, if requested
    if match_to_gaia:
        log.info('Matching Gaia to {} primary objects...t = {:.1f}s'
                 .format(len(objects),time()-start))
        gaiainfo = match_gaia_to_primary(objects, gaiadir=gaiadir)
        log.info('Done with Gaia match for {} primary objects...t = {:.1f}s'
                 .format(len(objects),time()-start))
        #ADM add the Gaia column information to the primary array
        for col in gaiainfo.dtype.names:
            objects[col] = gaiainfo[col]

    #- ensure uppercase column names if astropy Table
    if isinstance(objects, (Table, Row)):
        for col in list(objects.columns.values()):
            if not col.name.isupper():
                col.name = col.name.upper()

    #ADM rewrite the fluxes to shift anything on the northern Legacy Surveys
    #ADM system to approximate the southern system
    photsys_north = _isonnorthphotsys(objects["PHOTSYS"])
    photsys_south = ~_isonnorthphotsys(objects["PHOTSYS"])
    wnorth = np.where(photsys_north)

    if len(wnorth[0]) > 0:
        gshift, rshift, zshift = shift_photo_north(objects["FLUX_G"][wnorth],
                                                   objects["FLUX_R"][wnorth],
                                                   objects["FLUX_Z"][wnorth])
        objects["FLUX_G"][wnorth] = gshift
        objects["FLUX_R"][wnorth] = rshift
        objects["FLUX_Z"][wnorth] = zshift

    #ADM the observed r-band flux (used for F standards and MWS, below)
    #ADM make copies of values that we may reassign due to NaNs
    obs_rflux = objects['FLUX_R']

    #- undo Milky Way extinction
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']
    release = objects['RELEASE']

    gfluxivar = objects['FLUX_IVAR_G']

    gfracflux = objects['FRACFLUX_G'].T # note transpose
    rfracflux = objects['FRACFLUX_R'].T # note transpose
    zfracflux = objects['FRACFLUX_Z'].T # note transpose

    gallmask = objects['ALLMASK_G']
    rallmask = objects['ALLMASK_R']
    zallmask = objects['ALLMASK_Z']

    gsnr = objects['FLUX_G'] * np.sqrt(objects['FLUX_IVAR_G'])
    rsnr = objects['FLUX_R'] * np.sqrt(objects['FLUX_IVAR_R'])
    zsnr = objects['FLUX_Z'] * np.sqrt(objects['FLUX_IVAR_Z'])
    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    w2snr = objects['FLUX_W2'] * np.sqrt(objects['FLUX_IVAR_W2'])

    # Delta chi2 between PSF and SIMP morphologies; note the sign....
    dchisq = objects['DCHISQ']
    deltaChi2 = dchisq[...,0] - dchisq[...,1]

    #ADM remove handful of NaN values from DCHISQ values and make them unselectable
    w = np.where(deltaChi2 != deltaChi2)
    #ADM this is to catch the single-object case for unit tests
    if len(w[0]) > 0:
        deltaChi2[w] = -1e6

    #ADM the Gaia columns
    gaia = objects["REF_ID"] != -1
    pmra = objects['PMRA']
    pmdec = objects['PMDEC']
    parallax = objects['PARALLAX']
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    gaiabmag = objects['GAIA_PHOT_BP_MEAN_MAG']
    gaiarmag = objects['GAIA_PHOT_RP_MEAN_MAG']

    #- DR1 has targets off the edge of the brick; trim to just this brick
    try:
        primary = objects['BRICK_PRIMARY']
    except (KeyError, ValueError):
        if _is_row(objects):
            primary = True
        else:
            primary = np.ones_like(objects, dtype=bool)

    lrg_north, lrg1pass_north, lrg2pass_north = isLRGpass_north(primary=primary, 
                    gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, gflux_ivar=gfluxivar, 
                    rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr)

    lrg_south, lrg1pass_south, lrg2pass_south = isLRGpass_south(primary=primary, 
                    gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, gflux_ivar=gfluxivar, 
                    rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr)
    #ADM combine LRG target bits for an LRG target based on any imaging
    lrg = (lrg_north & photsys_north) | (lrg_south & photsys_south)
    lrg1pass = (lrg1pass_north & photsys_north) | (lrg1pass_south & photsys_south)
    lrg2pass = (lrg2pass_north & photsys_north) | (lrg2pass_south & photsys_south)

    elg_north = isELG_north(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                            gallmask=gallmask, rallmask=rallmask, zallmask=zallmask)
    elg_south = isELG_south(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux)
    #ADM combine ELG target bits for an ELG target based on any imaging
    elg = (elg_north & photsys_north) | (elg_south & photsys_south)

    if qso_selection=='colorcuts' :
        #ADM determine quasar targets in the north and the south separately
        qso_north = isQSO_cuts_north(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                                     w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2, 
                                     objtype=objtype, w1snr=w1snr, w2snr=w2snr, release=release)
        qso_south = isQSO_cuts_south(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                                     w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2, 
                                     objtype=objtype, w1snr=w1snr, w2snr=w2snr, release=release)
    elif qso_selection == 'randomforest':
        #ADM determine quasar targets in the north and the south separately
        qso_north = isQSO_randomforest_north(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                                             w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2, 
                                             objtype=objtype, release=release)
        qso_south = isQSO_randomforest_south(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                                             w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2, 
                                             objtype=objtype, release=release)
    else:
        raise ValueError('Unknown qso_selection {}; valid options are {}'.format(qso_selection,
                                                                                 qso_selection_options))
    #ADM combine quasar target bits for a quasar target based on any imaging
    qso = (qso_north & photsys_north) | (qso_south & photsys_south)

    #ADM Make sure to pass all of the needed columns! At one point we stopped
    #ADM passing objtype, which meant no standards were being returned.
    fstd = isFSTD(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                  gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                  gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                  obs_rflux=obs_rflux, objtype=objtype)
    fstd_bright = isFSTD(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                  gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                  gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                  obs_rflux=obs_rflux, objtype=objtype, bright=True)

    #ADM set the BGS bits
    bgs_bright_north = isBGS_bright_north(primary=primary, rflux=rflux, objtype=objtype)
    bgs_bright_south = isBGS_bright_south(primary=primary, rflux=rflux, objtype=objtype)
    bgs_bright = (bgs_bright_north & photsys_north) | (bgs_bright_south & photsys_south)
    bgs_faint_north = isBGS_faint_north(primary=primary, rflux=rflux, objtype=objtype)
    bgs_faint_south = isBGS_faint_south(primary=primary, rflux=rflux, objtype=objtype)
    bgs_faint = (bgs_faint_north & photsys_north) | (bgs_faint_south & photsys_south)

    #ADM set the MWS bits
    mws_n, mws_red_n, mws_blue_n = isMWS_main_north(primary=primary, gaia=gaia,  
                                gflux=gflux, rflux=rflux, obs_rflux=obs_rflux, 
                                pmra=pmra, pmdec=pmdec, parallax=parallax, objtype=objtype)
    mws_s, mws_red_s, mws_blue_s = isMWS_main_south(primary=primary, gaia=gaia,  
                                gflux=gflux, rflux=rflux, obs_rflux=obs_rflux, 
                                pmra=pmra, pmdec=pmdec, parallax=parallax, objtype=objtype)
    mws_nearby = isMWS_nearby(gaia=gaia, gaiagmag=gaiagmag, parallax=parallax)
    mws_wd = isMWS_WD(gaia=gaia, parallax=parallax, 
                      gaiagmag=gaiagmag, gaiabmag=gaiabmag, gaiarmag=gaiarmag)
    #ADM combine the north/south MWS bits
    mws = (mws_n & photsys_north) | (mws_s & photsys_south)
    mws_blue = (mws_blue_n & photsys_north) | (mws_blue_s & photsys_south)
    mws_red = (mws_red_n & photsys_north) | (mws_red_s & photsys_south)

    # Construct the targetflag bits for DECaLS (i.e. South)
    # This should really be refactored into a dedicated function.
    desi_target  = lrg_south * desi_mask.LRG_SOUTH
    desi_target |= elg_south * desi_mask.ELG_SOUTH
    desi_target |= qso_south * desi_mask.QSO_SOUTH

    # Construct the targetflag bits for MzLS and BASS (i.e. North)
    desi_target |= lrg_north * desi_mask.LRG_NORTH
    desi_target |= elg_north * desi_mask.ELG_NORTH
    desi_target |= qso_north * desi_mask.QSO_NORTH

    # Construct the targetflag bits combining north and south
    desi_target |= lrg * desi_mask.LRG
    desi_target |= elg * desi_mask.ELG
    desi_target |= qso * desi_mask.QSO

    #ADM add the per-pass information in the south...
    desi_target |= lrg1pass_south * desi_mask.LRG_1PASS_SOUTH
    desi_target |= lrg2pass_south * desi_mask.LRG_2PASS_SOUTH
    #ADM ...the north...
    desi_target |= lrg1pass_north * desi_mask.LRG_1PASS_NORTH
    desi_target |= lrg2pass_north * desi_mask.LRG_2PASS_NORTH
    #ADM ...and combined
    desi_target |= lrg1pass * desi_mask.LRG_1PASS
    desi_target |= lrg2pass * desi_mask.LRG_2PASS

    # Standards; still need to set STD_WD
    desi_target |= fstd * desi_mask.STD_FSTAR
    desi_target |= fstd_bright * desi_mask.STD_BRIGHT

    # BGS bright and faint, south
    bgs_target  = bgs_bright_south * bgs_mask.BGS_BRIGHT_SOUTH
    bgs_target |= bgs_faint_south * bgs_mask.BGS_FAINT_SOUTH

    # BGS bright and faint, north
    bgs_target |= bgs_bright_north * bgs_mask.BGS_BRIGHT_NORTH
    bgs_target |= bgs_faint_north * bgs_mask.BGS_FAINT_NORTH

    # BGS combined, bright and faint
    bgs_target |= bgs_bright * bgs_mask.BGS_BRIGHT
    bgs_target |= bgs_faint * bgs_mask.BGS_FAINT

    #ADM MWS main, nearby, and WD
    mws_target  = mws * mws_mask.MWS_MAIN
    mws_target |= mws_wd * mws_mask.MWS_WD
    mws_target |= mws_nearby * mws_mask.MWS_NEARBY

    #ADM MWS main north/south split
    mws_target |= mws_n * mws_mask.MWS_MAIN_NORTH
    mws_target |= mws_s * mws_mask.MWS_MAIN_SOUTH

    #ADM MWS main blue/red split
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
    #ADM set up default logging
    from desiutil.log import get_logger
    log = get_logger()

    #- Convert single file to list of files
    if isinstance(infiles,str):
        infiles = [infiles,]

    #- Sanity check that files exist before going further
    for filename in infiles:
        if not os.path.exists(filename):
            raise ValueError("{} doesn't exist".format(filename))

    #- function to run on every brick/sweep file
    def _check_input_files(filename):
        '''Check for corrupted values in a file'''
        from functools import partial
        from os.path import getsize

        #ADM read in Tractor or sweeps files
        objects = io.read_tractor(filename)
        #ADM if everything is OK the default meassage will be "OK"
        filemessageroot = 'OK'
        filemessageend = ''
        #ADM columns that shouldn't have zero values
        cols = [
            'BRICKID',
#            'RA_IVAR', 'DEC_IVAR',
            'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
#            'WISE_FLUX',
#            'WISE_MW_TRANSMISSION','DCHISQ'
            ]
        #ADM for each of these columnes that shouldn't have zero values,
        #ADM loop through and look for zero values
        for colname in cols:
            if np.min(objects[colname]) == 0:
                filemessageroot = "WARNING...some values are zero for"
                filemessageend += " "+colname

        #ADM now, loop through entries in the file and search for 4096-byte
        #ADM blocks that are all zeros (a sign of corruption in file-writing)
        #ADM Note that fits files are padded by 2880 bytes, so we only want to
        #ADM process the file length (in bytes) - 2880
        bytestop = getsize(filename) -2880

        with open(filename, 'rb') as f:
            for block_number, data in enumerate(iter(partial(f.read, 4096), b'')):
                if not any(data):
                    if block_number*4096 < bytestop:
                        filemessageroot = "WARNING...some values are zero for"
                        filemessageend += ' 4096-byte-block-#{0}'.format(block_number)

        return [filename,filemessageroot+filemessageend]

    # Counter for number of bricks processed;
    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    t0 = time()
    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if nbrick%25 == 0 and nbrick>0:
            elapsed = time() - t0
            rate = nbrick / elapsed
            log.info('{} files; {:.1f} files/sec; {:.1f} total mins elapsed'.format(nbrick, rate, elapsed/60.))
        nbrick[...] += 1    # this is an in-place modification
        return result

    #- Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            fileinfo = pool.map(_check_input_files, infiles, reduce=_update_status)
    else:
        fileinfo = list()
        for fil in infiles:
            fileinfo.append(_update_status(_check_input_files(fil)))

    fileinfo = np.array(fileinfo)
    w = np.where(fileinfo[...,1] != 'OK')

    if len(w[0]) == 0:
        log.info('ALL FILES ARE OK')
    else:
        for fil in fileinfo[w]:
            log.info(fil[0],fil[1])

    return len(w[0])


qso_selection_options = ['colorcuts', 'randomforest']
Method_sandbox_options = ['XD', 'RF_photo', 'RF_spectro']

def select_targets(infiles, numproc=4, qso_selection='randomforest',
            match_to_gaia=True, sandbox=False, FoMthresh=None, Method=None, 
            gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Process input files in parallel to select targets

    Parameters
    ----------
    infiles : :class:`list` or `str` 
        A list of input filenames (tractor or sweep files) OR a single filename
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use
    qso_selection : :class`str`, optional, defaults to `randomforest`
        The algorithm to use for QSO selection; valid options are 
        'colorcuts' and 'randomforest'
    match_to_gaia : :class:`boolean`, optional, defaults to ``True``
        If ``True``, match to Gaia DR2 chunks files and populate Gaia columns
        to facilitate the MWS selection
    sandbox : :class:`boolean`, optional, defaults to ``False``
        If ``True``, use the sample selection cuts in :mod:`desitarget.sandbox.cuts`
    FoMthresh : :class:`float`, optional, defaults to `None`
        If a value is passed then run `apply_XD_globalerror` for ELGs in
        the sandbox. This will write out an "FoM.fits" file for every ELG target
        in the sandbox directory
    Method : :class:`str`, optional, defaults to `None`
        Method used in the sandbox    
    gaiadir : :class:`str`, optional, defaults to Gaia DR2 path at NERSC
        Root directory of a Gaia Data Release as used by the Legacy Surveys.

    Returns
    -------   
    :class:`~numpy.ndarray`
        The subset of input targets which pass the cuts, including extra
        columns for `DESI_TARGET`, `BGS_TARGET`, and `MWS_TARGET` target
        selection bitmasks

    Notes
    -----
        - if numproc==1, use serial code instead of parallel
    """
    from desiutil.log import get_logger
    log = get_logger()

    #- Convert single file to list of files
    if isinstance(infiles,str):
        infiles = [infiles,]

    #- Sanity check that files exist before going further
    for filename in infiles:
        if not os.path.exists(filename):
            raise ValueError("{} doesn't exist".format(filename))

    def _finalize_targets(objects, desi_target, bgs_target, mws_target):
        #- desi_target includes BGS_ANY and MWS_ANY, so we can filter just
        #- on desi_target != 0
        keep = (desi_target != 0)
        objects = objects[keep]
        desi_target = desi_target[keep]
        bgs_target = bgs_target[keep]
        mws_target = mws_target[keep]

        #- Add *_target mask columns
        targets = desitarget.targets.finalize(
            objects, desi_target, bgs_target, mws_target)

        return io.fix_tractor_dr1_dtype(targets)

    #- functions to run on every brick/sweep file
    def _select_targets_file(filename):
        '''Returns targets in filename that pass the cuts'''
        objects = io.read_tractor(filename)
        desi_target, bgs_target, mws_target = apply_cuts(objects,qso_selection,match_to_gaia,gaiadir)

        return _finalize_targets(objects, desi_target, bgs_target, mws_target)

    def _select_sandbox_targets_file(filename):
        '''Returns targets in filename that pass the sandbox cuts'''
        from desitarget.sandbox.cuts import apply_sandbox_cuts
        objects = io.read_tractor(filename)
        desi_target, bgs_target, mws_target = apply_sandbox_cuts(objects,FoMthresh,Method)

        return _finalize_targets(objects, desi_target, bgs_target, mws_target)

    # Counter for number of bricks processed;
    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    t0 = time()
    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if nbrick%50 == 0 and nbrick>0:
            rate = nbrick / (time() - t0)
            log.info('{} files; {:.1f} files/sec'.format(nbrick, rate))

        nbrick[...] += 1    # this is an in-place modification
        return result

    #- Parallel process input files
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

    targets = np.concatenate(targets)

    return targets
