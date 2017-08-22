import unittest
import numpy as np
from astropy.table import Table

from desitarget import desi_mask
from desitarget.targets import calc_numobs

class TestNumObs(unittest.TestCase):
    
    def setUp(self):
        dtype = [
            ('DESI_TARGET',np.int64),
            ('BGS_TARGET',np.int64),
            ('MWS_TARGET',np.int64),
            ('NUMOBS',np.int32),
            ('DECAM_FLUX', '>f4', (6,)),
            ('DECAM_MW_TRANSMISSION', '>f4', (6,)),
        ]
        self.targets = np.zeros(5, dtype=dtype)
        self.targets['DECAM_MW_TRANSMISSION'] = 1.0
        self.targets['DECAM_FLUX'][:,4] = 10**((22.5-np.linspace(20, 21, 5))/2.5)
            
    def test_numobs(self):
        t = self.targets

        #- No target bits set is an error
        with self.assertRaises(ValueError):
            calc_numobs(t)
        
        #- ELGs and QSOs get one observation
        t['DESI_TARGET'] = desi_mask.ELG
        self.assertTrue(np.all(calc_numobs(t) == 1))
        t['DESI_TARGET'] = desi_mask.QSO
        self.assertTrue(np.all(calc_numobs(t) == 4))
        
        #- LRG numobs depends upon zflux  (DECAM_FLUX index 4)
        t['DESI_TARGET'] = desi_mask.LRG
        nobs = calc_numobs(t)
        self.assertTrue(np.all(nobs == [1, 1, 2, 3, 3]))

        #- test astropy Table
        t = Table(t)
        nobs = calc_numobs(t)
        self.assertTrue(np.all(nobs == [1, 1, 2, 3, 3]))

        #- LRG numobs also works with ZFLUX instead of DECAM*
        t['ZFLUX'] = t['DECAM_FLUX'][:,4] / t['DECAM_MW_TRANSMISSION'][:,4]
        t.remove_column('DECAM_FLUX')
        t.remove_column('DECAM_MW_TRANSMISSION')
        nobs = calc_numobs(t)
        self.assertTrue(np.all(nobs == [1, 1, 2, 3, 3]))

        #- this is true even if other targeting bits are set
        t['DESI_TARGET'] |= desi_mask.mask('ELG|BGS_ANY')
        nobs = calc_numobs(t)
        self.assertTrue(np.all(nobs == [1, 1, 2, 3, 3]))
        
        #- But if no *FLUX available, default to LRGs with 2 obs
        t.remove_column('ZFLUX')
        nobs = calc_numobs(t)
        self.assertTrue(np.all(nobs == 2))
                
if __name__ == '__main__':
    unittest.main()
