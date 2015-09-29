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

from desitarget.npyquery import Column as C
from desitarget.npyquery import Max, Min

from desitarget.decals import *
def make_most_assumptions(cut):
    return cut\
         .assume(BRICK_PRIMARY, 1) \
         .assume(DECAM_MW_TRANSMISSION[1], 1.0) \
         .assume(DECAM_MW_TRANSMISSION[2], 1.0) \
         .assume(DECAM_MW_TRANSMISSION[4], 1.0) \
         .assume(WISE_MW_TRANSMISSION[0], 1.0) \
         .assume(WISE_MW_TRANSMISSION[1], 1.0) \
         .assume(SHAPEDEV_R, 0.0) \
         .assume(SHAPEEXP_R, 0.0)

LRG = make_most_assumptions(LRG)
        
ELG = make_most_assumptions(ELG)

QSO = make_most_assumptions(QSO)

BGS = make_most_assumptions(BGS).assume(TYPE, 'AAA')

MyQSO =  RFLUX > 10**((22.5-23.0)/2.5) 
MyQSO &= RFLUX < 10**(1.0/2.5) * GFLUX
MyQSO &= ZFLUX > 10**(-0.3/2.5) * RFLUX
MyQSO &= ZFLUX < 10**(1.1/2.5) * RFLUX
MyQSO &= WFLUX * GFLUX**1.2 > 10**(2/2.5) * RFLUX**(1+1.2)

MyQSO = make_most_assumptions(MyQSO)

__all__ = ['LRG', 'ELG', 'QSO', 'BGS']

def makedecals(GFLUX, RFLUX, ZFLUX, W1FLUX, W2FLUX):
    import numpy
    decals = numpy.empty(len(GFLUX),
            dtype=[
                ('DECAM_FLUX', ('f4', 6)),
                ('WISE_FLUX', ('f4', 4)),
            ])
    decals['DECAM_FLUX'].T[1] = GFLUX
    decals['DECAM_FLUX'].T[2] = RFLUX
    decals['DECAM_FLUX'].T[4] = ZFLUX
    decals['WISE_FLUX'].T[0] = W1FLUX
    decals['WISE_FLUX'].T[1] = W2FLUX
    return decals
 
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
    
    data = makedecals(GFLUX, RFLUX, ZFLUX, W1FLUX, W2FLUX)

    print 'QSO kept', len(QSO(data)), '/', len(data)
    print 'LRG kept', len(LRG(data)), '/', len(data)
    print 'ELG kept', len(ELG(data)), '/', len(data)
    print 'BGS kept', len(BGS(data)), '/', len(data)
    print 'MyQSO kept', len(MyQSO(data)), '/', len(data)

if __name__ == '__main__':
    test()
