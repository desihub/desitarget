from __future__ import absolute_import, division, print_function
import warnings
import numpy as np

from desitarget.io import read_tractor

"""
    Target Selection for DECALS catalogue data

    https://desi.lbl.gov/trac/wiki/TargetSelection

    A collection of helpful (static) methods to check whether an object's
    flux passes a given selection criterion (e.g. LRG, ELG or QSO).

    These cuts assume we are passed the extinction-corrected fluxes
    (flux/mw_transmission) and are taken from:

    https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection

    These files (together with npyquery, were originally from ImagingLSS (github.com/desihub/imaginglss)

"""

from desitarget import desi_mask, bgs_mask, mws_mask

def select_targets(objects):
    """Perform target selection on objects, returning targetflag array

    Args:
        objects: numpy structured array with UPPERCASE columns needed for
            target selection, OR a string tractor filename
            
    Returns:
        (desi_target, bgs_target, mws_target) where each element is
        an ndarray of target selection bitmask flags for each object
        
    See desitarget.targetmask for the definition of each bit
    """
    #- Check if objects is a filename instead of the actual data
    if isinstance(objects, (str, unicode)):
        objects = read_tractor(objects)
    
    #- construct milky way extinction corrected fluxes
    dered_decam_flux = objects['DECAM_FLUX'] / objects['DECAM_MW_TRANSMISSION']
    gflux = dered_decam_flux[:, 1]
    rflux = dered_decam_flux[:, 2]
    zflux = dered_decam_flux[:, 4]

    dered_wise_flux = objects['WISE_FLUX'] / objects['WISE_MW_TRANSMISSION']
    w1flux = dered_wise_flux[:, 0]
    wflux = 0.75* w1flux + 0.25*dered_wise_flux[:, 1]

    #- DR1 has targets off the edge of the brick; trim to just this brick
    primary = objects['BRICK_PRIMARY']

    #- each of lrg, elg, bgs, ... will be a boolean array of matches
    lrg = primary.copy()
    lrg &= rflux > 10**((22.5-23.0)/2.5)
    lrg &= zflux > 10**((22.5-22.56)/2.5)
    lrg &= w1flux > 10**((22.5-19.35)/2.5)
    lrg &= zflux > rflux * 10**(1.6/2.5)
    #- clip to avoid warnings from negative numbers raised to fractional powers
    lrg &= w1flux * rflux.clip(0)**(1.33-1) > zflux.clip(0)**1.33 * 10**(-0.33/2.5)
    ### lrg &= w1flux * rflux**(1.33-1) > zflux**1.33 * 10**(-0.33/2.5)

    elg = primary.copy()
    elg &= rflux > 10**((22.5-23.4)/2.5)
    elg &= zflux > rflux * 10**(0.3/2.5)
    elg &= zflux < rflux * 10**(1.5/2.5)
    elg &= rflux**2 < gflux * zflux * 10**(-0.2/2.5)
    elg &= zflux < gflux * 10**(1.2/2.5)

    qso = primary.copy()
    qso &= rflux > 10**((22.5-23.0)/2.5)
    qso &= rflux < gflux * 10**(1.0/2.5)
    qso &= zflux > rflux * 10**(-0.3/2.5)
    qso &= zflux < rflux * 10**(1.1/2.5)
    #- clip to avoid warnings from negative numbers raised to fractional powers
    qso &= wflux * gflux.clip(0)**1.2 > rflux.clip(0)**(1+1.2) * 10**(2/2.5)
    ### qso &= wflux * gflux**1.2 > rflux**(1+1.2) * 10**(2/2.5)

    bgs = primary.copy()
    bgs &= objects['TYPE'] != 'PSF'   #- for astropy.io.fits (sigh)
    bgs &= objects['TYPE'] != 'PSF '  #- for fitsio (sigh)
    bgs &= rflux > 10**((22.5-19.35)/2.5)

    fstd = primary.copy()
    fstd &= objects['TYPE'] != 'PSF'   #- for astropy.io.fits (sigh)
    fstd &= objects['TYPE'] != 'PSF '  #- for fitsio (sigh)
    fracflux = objects['DECAM_FRACFLUX'].T
    signal2noise = objects['DECAM_FLUX'] * np.sqrt(objects['DECAM_FLUX_IVAR'])
    for j in (1,2,4):  #- g, r, z
        fstd &= fracflux[j] < 0.04
        fstd &= signal2noise[:, j] > 10
    #- observed flux; no Milky Way extinction
    obs_rflux = objects['DECAM_FLUX'][:, 2]
    fstd &= obs_rflux < 10**((22.5-16.0)/2.5)
    fstd &= obs_rflux > 10**((22.5-19.0)/2.5)
    #- colors near BD+17; ignore warnings about flux<=0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grcolor = 2.5 * np.log10(rflux / gflux)
        rzcolor = 2.5 * np.log10(zflux / rflux)
        fstd &= (grcolor - 0.32)**2 + (rzcolor - 0.13)**2 < 0.06**2

    #-----
    #- construct the targetflag bits
    desi_target  = lrg * desi_mask.LRG
    desi_target |= elg * desi_mask.ELG
    desi_target |= qso * desi_mask.QSO
    desi_target |= fstd * desi_mask.FSTD

    #- Currently our only cuts are DECam based (i.e. South)
    desi_target |= lrg * desi_mask.LRG_SOUTH
    desi_target |= elg * desi_mask.ELG_SOUTH
    desi_target |= qso * desi_mask.QSO_SOUTH

    
    bgs_target = bgs * bgs_mask.BGS_BRIGHT
    bgs_target |= bgs * bgs_mask.BGS_BRIGHT_SOUTH

    mws_target = np.zeros_like(bgs_target)

    return desi_target, bgs_target, mws_target

def calc_numobs(targets, targetflags):
    """
    Return array of number of observations needed for each target.
    
    Args:
        targets: numpy structured array with tractor inputs
        targetflags: array of target selection bit flags 
    
    Returns:
        array of integers of number of observations needed
    """
    #- Default is one observation
    nobs = np.ones(len(targets), dtype='i4')
    
    #- If it wasn't selected by any target class, it gets 0 observations
    #- Normally these would have already been removed, but just in case...
    nobs[targetflags == 0] = 0
    
    #- LRGs get 1, 2, or 3 observations depending upon magnitude
    zflux = targets['DECAM_FLUX'][:,4] / targets['DECAM_MW_TRANSMISSION'][:,4]    
    islrg = (targetflags & targetmask.LRG) != 0
    lrg2 = islrg & (zflux < 10**((22.5-20.36)/2.5))
    lrg3 = islrg & (zflux < 10**((22.5-20.56)/2.5))
    nobs[lrg2] = 2
    nobs[lrg3] = 3
    
    #- TBD: flag QSOs for 4-5 obs ahead of time, or only after confirming
    #- that they are redshift>2.15 (i.e. good for Lyman-alpha)?
    
    return nobs
