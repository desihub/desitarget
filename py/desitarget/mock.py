from __future__ import absolute_import

"""
    Target Selection for MOCK catalogue data

    https://desi.lbl.gov/trac/wiki/TargetSelection

    Inputs are r, g, z, W1, W2

    A collection of helpful (static) methods to check whether an object's
    flux passes a given selection criterion (e.g. LRG, ELG or QSO).

      https://desi.lbl.gov/trac/wiki/TargetSelection

"""

__author__ = "Yu Feng and John Moustous"
__version__ = "1.0"
__email__  = "yfeng1@berkeley.edu"

from npyquery import Column as C
from npyquery import Max, Min

GFLUX = 10 ** ((22.5 - C('g')) / 2.5)
RFLUX = 10 ** ((22.5 - C('r')) / 2.5)
ZFLUX = 10 ** ((22.5 - C('z')) / 2.5)
W1FLUX = 10 ** ((22.5 - C('W1')) / 2.5)
W2FLUX = 10 ** ((22.5 - C('W2')) / 2.5)

WFLUX = 0.75 * W1FLUX + 0.25 * W2FLUX[1]

LRG  = RFLUX > 10**((22.5-23.0)/2.5)
""" LRG Cut """
LRG &= ZFLUX > 10**((22.5-20.56)/2.5)
LRG &= W1FLUX > 10**((22.5-19.35)/2.5)
LRG &= ZFLUX > RFLUX * 10**(1.6/2.5)
LRG &= W1FLUX * RFLUX ** (1.33-1) > ZFLUX**1.33 * 10**(-0.33/2.5)


ELG  = RFLUX > 10**((22.5-23.4)/2.5)
""" ELG Cut """ 
ELG &= ZFLUX > 10**(0.3/2.5) * RFLUX
ELG &= ZFLUX < 10**(1.5/2.5) * RFLUX
ELG &= RFLUX**2 < GFLUX * ZFLUX * 10**(-0.2/2.5)
ELG &= ZFLUX < GFLUX * 10**(1.2/2.5) 

QSO  = RFLUX > 10**((22.5-23.0)/2.5)
""" QSO Cut """
QSO &= RFLUX < 10**(1.0/2.5) * GFLUX
QSO &= ZFLUX > 10**(-0.3/2.5) * RFLUX
QSO &= ZFLUX < 10**(1.1/2.5) * RFLUX
QSO &= WFLUX * GFLUX**1.2 > 10**(2/2.5) * RFLUX**(1+1.2)


BGS  =  RFLUX > 10**((22.5-19.35)/2.5)
""" BGS Cut """

__all__ = ['LRG', 'ELG', 'QSO', 'BGS']
