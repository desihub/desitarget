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


### use these lines for mag cuts.

#TOFLUX = lambda mag : 10 ** ((22.5 - mag) / 2.5)
#GFLUX = TOFLUX(C('g'))
#RFLUX = TOFLUX(C('r'))
#ZFLUX = TOFLUX(C('z'))
#W1FLUX = TOFLUX(C('W1'))
#W2FLUX = TOFLUX(C('W2'))

### use these lines for nanomaggie cuts.
GFLUX = C('GFLUX')
RFLUX = C('RFLUX')
ZFLUX = C('ZFLUX')
W1FLUX = C('W1FLUX')
W2FLUX = C('W2FLUX')

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

def test():
    import numpy

    TOFLUX = lambda mag : 10 ** ((22.5 - mag) / 2.5)

    randoms = numpy.random.uniform(size=(4, 1000))
    r = randoms[0] * (24- 16) + 16
    g = randoms[1] * 2.5 - 0.5 + r
    W1 = r - (randoms[2] * 8 - 2)
    W2 = r - (randoms[2] * 8 - 2)
    z = r - (randoms[3] * 3.0 - 0.5)

    GFLUX = TOFLUX(g)
    RFLUX = TOFLUX(r)
    ZFLUX = TOFLUX(z)
    W1FLUX = TOFLUX(W1)
    W2FLUX = TOFLUX(W1)
    
#    print QSO(dict(r=r, g=g, W1=W1, W2=W2, z=z))

    print(QSO(dict(GFLUX=GFLUX, RFLUX=RFLUX, W1FLUX=W1FLUX, W2FLUX=W2FLUX, ZFLUX=ZFLUX)))

    # alternative
    rec = numpy.rec.fromarrays(
            [GFLUX, RFLUX, W1FLUX, W2FLUX, ZFLUX],
            names=['GFLUX', 'RFLUX', 'W1FLUX', 'W2FLUX', 'ZFLUX'],
            )
    print 'QSO kept', len(QSO(rec)), '/', len(rec)
    print 'LRG kept', len(LRG(rec)), '/', len(rec)
    print 'ELG kept', len(ELG(rec)), '/', len(rec)
    print 'BGS kept', len(BGS(rec)), '/', len(rec)

if __name__ == '__main__':
    test()
