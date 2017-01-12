'''
Testing desitarget.mock.build, but only add_mock_shapes_and_fluxes for now
'''

import unittest
import numpy as np
from astropy.table import Table

from desitarget.mock.build import add_mock_shapes_and_fluxes
from desitarget import desi_mask, bgs_mask, mws_mask

class TestMockBuild(unittest.TestCase):
    
    def setUp(self):
        pass
            
    def test_shapes_and_fluxes(self):
        nreal = 40
        real = Table()
        real['DESI_TARGET'] = 2**np.random.randint(0,3,size=nreal)
        real['BGS_TARGET'] = np.zeros(nreal, dtype=int)
        real['BGS_TARGET'][0:5] = bgs_mask.BGS_BRIGHT
        real['BGS_TARGET'][5:10] = bgs_mask.BGS_FAINT
        real['DESI_TARGET'][0:10] = 0
        
        real['DECAM_FLUX'] = np.random.uniform(size=(nreal,6))
        real['SHAPEDEV_R'] = np.random.uniform(size=nreal)
        real['SHAPEEXP_R'] = np.random.uniform(size=nreal)
        
        nmock = 45
        mock = Table()
        mock['DESI_TARGET'] = 2**np.random.randint(0,3,size=nmock)
        mock['BGS_TARGET'] = np.zeros(nmock, dtype=int)
        mock['BGS_TARGET'][10:15] = bgs_mask.BGS_BRIGHT
        mock['BGS_TARGET'][15:20] = bgs_mask.BGS_FAINT
        mock['DESI_TARGET'][10:20] = 0
        
        add_mock_shapes_and_fluxes(mock, real)
        self.assertTrue('DECAM_FLUX' in mock.colnames)
        self.assertTrue('SHAPEDEV_R' in mock.colnames)
        self.assertTrue('SHAPEEXP_R' in mock.colnames)
                
if __name__ == '__main__':
    unittest.main()
