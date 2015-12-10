import unittest
import numpy as np

from desitarget import desi_mask
from desitarget.targets import calc_numobs

class TestNumObs(unittest.TestCase):
    
    def setUp(self):
        dtype = [
            ('DESI_TARGET',np.int64),
            ('DECAM_FLUX', '>f4', (6,)),
            ('DECAM_MW_TRANSMISSION', '>f4', (6,)),
        ]
        self.targets = np.zeros(5, dtype=dtype)
        self.targets['DECAM_MW_TRANSMISSION'] = 1.0
        self.targets['DECAM_FLUX'][:,4] = 10**((22.5-np.linspace(20, 21, 5))/2.5)
            
    def test_numobs(self):
        t = self.targets
        #- default DESI_TARGET=0 should be no observations
        self.assertTrue(np.all(calc_numobs(t) == 0))
        
        #- ELGs and QSOs get one observation
        t['DESI_TARGET'] = desi_mask.ELG
        self.assertTrue(np.all(calc_numobs(t) == 1))
        t['DESI_TARGET'] = desi_mask.QSO
        self.assertTrue(np.all(calc_numobs(t) == 1))
        
        #- LRG numobs depends upon zflux  (DECAM_FLUX index 4)
        t['DESI_TARGET'] = desi_mask.LRG
        nobs = calc_numobs(t)
        self.assertTrue(np.all(nobs == [1, 1, 2, 3, 3]))

        #- this is true even if other targeting bits are set
        t['DESI_TARGET'] |= desi_mask.mask('ELG|BGS_ANY')
        nobs = calc_numobs(t)
        self.assertTrue(np.all(nobs == [1, 1, 2, 3, 3]))
                
if __name__ == '__main__':
    unittest.main()
