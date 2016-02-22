import unittest
import numpy as np
from astropy.table import Table

from desitarget import desi_mask as Mx
from desitarget.mtl import make_mtl

class TestMTL(unittest.TestCase):
    
    def setUp(self):
        self.targets = Table()
        self.types = np.array(['ELG', 'LRG', 'QSO'])
        self.priorities = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.targets['DESI_TARGET'] = [Mx[t].mask for t in self.types]
        self.targets['ZFLUX'] = 10**((22.5-np.linspace(20, 21, 3))/2.5)
            
    def test_mtl(self):
        mtl = make_mtl(self.targets)
        goodkeys = set(self.targets.dtype.names) | set(['NUMOBS_MORE', 'PRIORITY', 'LASTPASS'])
        self.assertTrue(set(mtl.dtype.names) == goodkeys, \
                        'colname mismatch: {} vs. {}'.format( \
                            mtl.dtype.names, goodkeys))
                    
    def test_numobs(self):
        mtl = make_mtl(self.targets)
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == [1, 2, 4]))
        self.assertTrue(np.all(mtl['PRIORITY'] == self.priorities))
        iselg = (self.types == 'ELG')
        self.assertTrue(np.all(mtl['LASTPASS'][iselg] != 0))
        self.assertTrue(np.all(mtl['LASTPASS'][~iselg] == 0))
            
