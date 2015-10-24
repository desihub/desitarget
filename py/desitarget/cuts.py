from __future__ import absolute_import

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

#-------------------------------------------------------------------------
#- Make a dictionary of the cut types known in this file
types = {
    'LRG': LRG,
    'ELG': ELG,
    'BGS': BGS,
    'QSO': QSO,
}

#-------------------------------------------------------------------------
#- SJB Alternate TS code

def select_targets(objects):
    '''
    Given an input table from a tractor sweep file, return
    boolean array for whether each object passes LRG cuts or not.

    Assumes columns rflux, zflux, w1flux have been added, and
    brick_primary='T' filter has already been applied
    '''
    dered_decam_flux = objects['DECAM_FLUX'] / objects['DECAM_MW_TRANSMISSION']
    gflux = dered_decam_flux[:, 1]
    rflux = dered_decam_flux[:, 2]
    zflux = dered_decam_flux[:, 4]

    # gflux = objects['decam_flux'][:,1] / objects['decam_mw_transmission'][:,1]
    # rflux = objects['decam_flux'][:,2] / objects['decam_mw_transmission'][:,2]
    # zflux = objects['decam_flux'][:,4] / objects['decam_mw_transmission'][:,4]

    dered_wise_flux = objects['WISE_FLUX'] / objects['WISE_MW_TRANSMISSION']
    w1flux = dered_wise_flux[:, 0]
    wflux = 0.75* w1flux + 0.25*dered_wise_flux[:, 1]

    # w1flux = objects['wise_flux'][:,0] / objects['WISE_MW_TRANSMISSION'][:,0]
    # wflux = 0.75*objects['wise_flux'][:,0] / objects['WISE_MW_TRANSMISSION'][:,0] + \
            #         0.25*objects['WISE_FLUX'][:,1] / objects['WISE_MW_TRANSMISSION'][:,1]

    primary = objects['BRICK_PRIMARY']

    lrg = primary.copy()
    lrg &= rflux > 10**((22.5-23.0)/2.5)
    lrg &= zflux > 10**((22.5-22.56)/2.5)
    lrg &= w1flux > 10**((22.5-19.35)/2.5)
    lrg &= zflux > rflux * 10**(1.6/2.5)
    lrg &= w1flux * rflux**(1.33-1) > zflux**1.33 * 10**(-0.33/2.5)

    elg = primary.copy()
    elg &= rflux > 10**((22.5-23.4)/2.5)
    elg &= zflux > rflux * 10**(0.3/2.5)
    elg &= zflux < rflux * 10**(1.5/2.5)
    elg &= rflux**2 < gflux * zflux * 10**(-0.2/2.5)
    elg &= zflux < gflux * 10**(1.2/2.5)

    bgs = primary.copy()
    bgs &= objects['TYPE'] != 'PSF'   #- for astropy.io.fits
    bgs &= objects['TYPE'] != 'PSF '  #- for fitsio (sigh)
    bgs &= rflux > 10**((22.5-19.35)/2.5)

    qso = primary.copy()
    qso &= rflux > 10**((22.5-23.0)/2.5)
    qso &= rflux < gflux * 10**(1.0/2.5)
    qso &= zflux > rflux * 10**(-0.3/2.5)
    qso &= zflux < rflux * 10**(1.1/2.5)
    qso &= wflux * gflux**1.2 > rflux**(1+1.2) * 10**(2/2.5)

    #- TODO: fix hardcoded bit numbers
    target_mask = (lrg * 2**0) + (elg * 2**1) + (bgs * 2**3) + (qso * 2**4)

    return target_mask


