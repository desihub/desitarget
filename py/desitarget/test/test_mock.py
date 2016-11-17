import os
import unittest
import numpy as np
from astropy.table import Table

from desitarget import desi_mask as Mx
from desitarget import obsconditions
from desitarget.mtl import make_mtl
from desitarget.mock import selection

class TestMock(unittest.TestCase):
    
    def setUp(self):
        self.targets = Table()
        self.types = np.array(['ELG', 'LRG', 'QSO', 'QSO', 'ELG'])
        self.priorities = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.post_prio = [Mx[t].priorities['MORE_ZGOOD'] for t in self.types]
        self.post_prio[0] = 0  #- ELG
        self.post_prio[2] = 0  #- low-z QSO
        self.targets['DESI_TARGET'] = [Mx[t].mask for t in self.types]
        self.targets['BGS_TARGET'] = np.zeros(len(self.types), dtype=np.int64)
        self.targets['MWS_TARGET'] = np.zeros(len(self.types), dtype=np.int64)
        n = len(self.targets)
        self.targets['ZFLUX'] = 10**((22.5-np.linspace(20, 22, n))/2.5)
        self.targets['TARGETID'] = list(range(n))
        self.targets['RA'] = np.array([0.0, 20.0, 30.0, 180.0, 220.0])
        self.targets['DEC'] = np.array([20.0, 10.0, 0.0, -20.0, -30.0])
        
        #- reverse the order for zcat to make sure joins work
        self.zcat = Table()
        self.zcat['TARGETID'] = self.targets['TARGETID'][-2::-1]
        self.zcat['Z'] = [2.5, 1.0, 0.5, 1.0]
        self.zcat['ZWARN'] = [0, 0, 0, 0]
        self.zcat['NUMOBS'] = [1, 1, 1, 1]
            
    def test_sample_depths(self):
        depths = selection.sample_depths(self.targets['RA'], self.targets['DEC'])
        expected_keys = ['EBV', 'DEPTH_G', 'DEPTH_R', 'DEPTH_Z', 'GALDEPTH_G',
                         'GALDEPTH_R', 'GALDEPTH_Z']
        for k in expected_keys:
            self.assertIn(k, depths.keys())
        for k in depths.keys():
            self.assertEqual(len(self.targets['RA']), len(depths[k]))
            self.assertEqual(len(self.targets['DEC']), len(depths[k]))
            self.assertTrue(np.all(depths[k]>0.0))
        

if __name__ == '__main__':
    unittest.main()
