import unittest
import numpy as np
from astropy.table import Table

from desitarget import desi_mask as Mx
from desitarget import desi_mask
from desitarget.targets import calc_numobs
from desitarget.mock import build

class TestBrickFluctuation(unittest.TestCase):
    
    def setUp(self):
        self.targets = Table()
        self.types = np.array(['ELG', 'LRG', 'QSO', 'QSO', 'ELG'])
        self.priorities = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.post_prio = [Mx[t].priorities['MORE_ZGOOD'] for t in self.types]
        self.post_prio[0] = 1  #- ELG
        self.post_prio[2] = 1  #- low-z QSO
        self.targets['DESI_TARGET'] = [Mx[t].mask for t in self.types]
        self.targets['BGS_TARGET'] = np.zeros(len(self.types), dtype=np.int64)
        self.targets['MWS_TARGET'] = np.zeros(len(self.types), dtype=np.int64)
        n = len(self.targets)
        self.targets['ZFLUX'] = 10**((22.5-np.linspace(20, 22, n))/2.5)
        self.targets['TARGETID'] = list(range(n))
        
        #- reverse the order for zcat to make sure joins work
        self.zcat = Table()
        self.zcat['TARGETID'] = self.targets['TARGETID'][-2::-1]
        self.zcat['Z'] = [2.5, 1.0, 0.5, 1.0]
        self.zcat['ZWARN'] = [0, 0, 0, 0]
        self.zcat['NUMOBS'] = [1, 1, 1, 1]

            
    def test_generate_brick(self):
        b = build.generate_brick_info(bounds=(0.0, 1.0, -1.0, 1.0))
        keys = ['BRICKNAME', 'RA', 'DEC', 'RA1', 'RA2', 'DEC1', 'DEC2', 'BRICKAREA']
        for k in b.keys():
            self.assertTrue(np.all(k in keys))
        self.assertTrue(np.all((b['RA']<b['RA2']) & (b['RA']>b['RA1'])))
        self.assertTrue(np.all((b['DEC']<b['DEC2']) & (b['DEC']>b['DEC1'])))



if __name__ == '__main__':
    unittest.main()
