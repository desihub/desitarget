from __future__ import absolute_import
import numpy as np

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

from desitarget.internal.npyquery import Column as C
from desitarget.internal.npyquery import Max, Min
from desitarget.targetmask import targetmask

#- Collect the columns to use in the cuts below
DECAM_FLUX = C('DECAM_FLUX')
DECAM_MW_TRANSMISSION = C('DECAM_MW_TRANSMISSION')
WISE_FLUX = C('WISE_FLUX')
WISE_MW_TRANSMISSION = C('WISE_MW_TRANSMISSION')
BRICK_PRIMARY = C('BRICK_PRIMARY')
TYPE = C('TYPE')

SHAPEDEV_R = C('SHAPEDEV_R')
SHAPEEXP_R = C('SHAPEEXP_R')

#- Some new columns are combinations of others
GFLUX = DECAM_FLUX[1] / DECAM_MW_TRANSMISSION[1]
RFLUX = DECAM_FLUX[2] / DECAM_MW_TRANSMISSION[2]
ZFLUX = DECAM_FLUX[4] / DECAM_MW_TRANSMISSION[4]
W1FLUX = WISE_FLUX[0] / WISE_MW_TRANSMISSION[0] 
WFLUX = 0.75 * WISE_FLUX[0] / WISE_MW_TRANSMISSION[0] \
      + 0.25 * WISE_FLUX[1] / WISE_MW_TRANSMISSION[1] 

#-------------------------------------------------------------------------
#- The actual target selection cuts for each object type

LRG =  BRICK_PRIMARY != 0
""" LRG Cut """

LRG &= RFLUX > 10**((22.5-23.0)/2.5)
LRG &= ZFLUX > 10**((22.5-20.56)/2.5)
LRG &= W1FLUX > 10**((22.5-19.35)/2.5)
LRG &= ZFLUX > RFLUX * 10**(1.6/2.5)
LRG &= W1FLUX * RFLUX ** (1.33-1) > ZFLUX**1.33 * 10**(-0.33/2.5)

ELG =  BRICK_PRIMARY != 0
""" ELG Cut """

ELG &= RFLUX > 10**((22.5-23.4)/2.5)
ELG &= ZFLUX > 10**(0.3/2.5) * RFLUX
ELG &= ZFLUX < 10**(1.5/2.5) * RFLUX
ELG &= RFLUX**2 < GFLUX * ZFLUX * 10**(-0.2/2.5)
ELG &= ZFLUX < GFLUX * 10**(1.2/2.5) 

#- This shape cut is not included on the above reference wiki page
ELG &= Max(SHAPEDEV_R, SHAPEEXP_R) < 1.5

QSO =  BRICK_PRIMARY != 0
""" QSO Cut """

QSO &= RFLUX > 10**((22.5-23.0)/2.5)
QSO &= RFLUX < 10**(1.0/2.5) * GFLUX
QSO &= ZFLUX > 10**(-0.3/2.5) * RFLUX
QSO &= ZFLUX < 10**(1.1/2.5) * RFLUX
QSO &= WFLUX * GFLUX**1.2 > 10**(2/2.5) * RFLUX**(1+1.2)

BGS =  BRICK_PRIMARY != 0
""" BGS Cut """

BGS &= TYPE != 'PSF'   #- for astropy.io.fits
BGS &= TYPE != 'PSF '  #- for fitsio  (sigh)
BGS &=  RFLUX > 10**((22.5-19.35)/2.5)

#- A dictionary of the cut types known in this file
TYPES = {
    'LRG': LRG,
    'ELG': ELG,
    'BGS': BGS,
    'QSO': QSO,
}

def select_targets(candidates):
    """Perform target selection on candidates, returning targetflag array

    Args:
        candidates: numpy structured array with UPPERCASE columns needed for
            target selection
            
    Returns:
        targetflag : ndarray of target selection bitmask flags for each object
        
    See desitarget.targetmask for the definition of each bit
    """
    targetflag = np.zeros(len(candidates), dtype='i8')

    for t, cut in TYPES.items():
        bitfield = targetmask.mask(t)
        with np.errstate(all='ignore'):
            mask = cut.apply(candidates)
        targetflag[mask] |= bitfield
        nselected = np.count_nonzero(mask)
        assert np.count_nonzero(targetflag & bitfield) == nselected
        
    return targetflag

#-------------------------------------------------------------------------
