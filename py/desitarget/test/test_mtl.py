import os
import unittest
import numpy as np
from astropy.table import Table

from desitarget import desi_mask as Mx
from desitarget import obsconditions
from desitarget.mtl import make_mtl

class TestMTL(unittest.TestCase):
    
    def setUp(self):
        self.targets = Table()
        self.types = np.array(['ELG', 'LRG', 'QSO', 'QSO', 'ELG'])
        self.priorities = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.post_prio = [Mx[t].priorities['MORE_ZGOOD'] for t in self.types]
        self.post_prio[0] = 2 #ELG
        self.post_prio[2] = 2 #lowz QSO
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
            
    def test_mtl(self):
        mtl = make_mtl(self.targets)
        goodkeys = sorted(set(self.targets.dtype.names) | set(['NUMOBS_MORE', 'PRIORITY', 'OBSCONDITIONS']))
        mtlkeys = sorted(mtl.dtype.names)
        self.assertEqual(mtlkeys, goodkeys)
                    
    def test_numobs(self):
        mtl = make_mtl(self.targets)
        mtl.sort(keys='TARGETID')
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == [1, 2, 4, 4, 1]))
        self.assertTrue(np.all(mtl['PRIORITY'] == self.priorities))
        #- Check that ELGs can be observed in gray conditions but not others
        iselg = (self.types == 'ELG')
        self.assertTrue(np.all((mtl['OBSCONDITIONS'][iselg] & obsconditions.GRAY) != 0))
        self.assertTrue(np.all((mtl['OBSCONDITIONS'][~iselg] & obsconditions.GRAY) == 0))

    def test_zcat(self):
        mtl = make_mtl(self.targets, self.zcat, trim=False)
        mtl.sort(keys='TARGETID')
        self.assertTrue(np.all(mtl['PRIORITY'] == self.post_prio))
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == [0, 1, 0, 3, 1]))

        
        #- change one target to a SAFE (BADSKY) target and confirm priority=0 not 1
        self.targets['DESI_TARGET'][0] = Mx.BADSKY
        mtl = make_mtl(self.targets, self.zcat, trim=False)
        mtl.sort(keys='TARGETID')
        self.assertEqual(mtl['PRIORITY'][0], 0)
 
    def test_mtl_io(self):
        mtl = make_mtl(self.targets, self.zcat, trim=True)
        testfile = 'test-aszqweladfqwezceas.fits'
        mtl.write(testfile, overwrite=True)
        x = mtl.read(testfile)
        os.remove(testfile)
        if x.masked:
            self.assertTrue(np.all(mtl['NUMOBS_MORE'].mask == x['NUMOBS_MORE'].mask))


if __name__ == '__main__':
    unittest.main()
