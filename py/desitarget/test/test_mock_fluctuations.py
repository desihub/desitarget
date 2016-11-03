import os
import unittest
import numpy as np
from astropy.table import Table

from desitarget.mock import fluctuations
from desitarget import desi_mask as Mx
from desitarget import obsconditions


class TestFluctuations(unittest.TestCase):
    
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
        self.targets['RA'] = [10.0, 10.0, 160.0, 340.0, 340.0]
        self.targets['DEC'] = [-80.0, -80.0, 0.0, 80.0, 80.0]
        
        #- reverse the order for zcat to make sure joins work
        self.zcat = Table()
        self.zcat['TARGETID'] = self.targets['TARGETID'][-2::-1]
        self.zcat['Z'] = [2.5, 1.0, 0.5, 1.0]
        self.zcat['ZWARN'] = [0, 0, 0, 0]
        self.zcat['NUMOBS'] = [1, 1, 1, 1]

        self.mean_density = {}
        self.mean_density['ELG']  = 2400.0
        self.mean_density['LRG'] = 350.0
        self.mean_density['TRACERQSO'] = 120.0
        self.mean_density['LYAQSO'] = 50.0
        self.mean_density['BGS'] = 1400.0
        self.mean_density['MWS'] = 600.0



    def test_density_across_brick_info(self):
        f = fluctuations.density_across_brick(self.targets['RA'], self.targets['DEC'], self.mean_density)        
        self.assertTrue('DENSITY_ELG' in f.keys())
        self.assertTrue('DENSITY_MWS' in f.keys())
        self.assertTrue('DENSITY_TRACERQSO' in f.keys())
        self.assertTrue('DENSITY_LYAQSO' in f.keys())
        self.assertTrue('DENSITY_BGS' in f.keys())
        self.assertTrue('DENSITY_LRG' in f.keys())

    def test_density_across_values(self):
        f = fluctuations.density_across_brick(self.targets['RA'], self.targets['DEC'], self.mean_density)        
        for k in f.keys():            
            #check that densities are within expected values
            self.assertTrue(np.all(f[k] > 0.0))
            self.assertTrue(np.all(f[k] < 50000.0))

    def test_density_across_brick_structure(self):
        f = fluctuations.density_across_brick(self.targets['RA'], self.targets['DEC'], self.mean_density)        
        for k in f.keys():            
            #check that the number of density values is the same as the input ra,dec 
            self.assertEqual(len(f[k]), 5)            
            #check that numbers are equal for equal positions
            self.assertEqual(f[k][0], f[k][1])
            self.assertEqual(f[k][3], f[k][4])
            

if __name__ == '__main__':
    unittest.main()
