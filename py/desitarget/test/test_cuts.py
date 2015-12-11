import unittest
import os.path
from uuid import uuid4
from astropy.io import fits
from astropy.table import Table
import numpy as np

from desitarget import io
from desitarget import cuts
from desitarget import desi_mask

class TestCuts(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # py/desitarget/test -> etc/datadir
        thisdir, thisfile = os.path.split(__file__)
        cls.datadir = os.path.abspath(thisdir+'/../../../') + '/etc/testdata'
        cls.tractorfiles = io.list_tractorfiles(cls.datadir)
        cls.sweepfiles = io.list_sweepfiles(cls.datadir)

    def test_unextinct_fluxes(self):
        targets = io.read_tractor(self.tractorfiles[0])
        t1 = cuts.unextinct_fluxes(targets)
        self.assertTrue(isinstance(t1, np.ndarray))
        t2 = cuts.unextinct_fluxes(Table(targets))
        self.assertTrue(isinstance(t2, Table))
        for col in ['GFLUX', 'RFLUX', 'ZFLUX', 'W1FLUX', 'W2FLUX', 'WFLUX']:
            self.assertIn(col, t1.dtype.names)
            self.assertIn(col, t2.dtype.names)
            self.assertTrue(np.all(t1[col] == t2[col]))

    def test_cuts1(self):
        #- Cuts work with either data or filenames
        desi, bgs, mws = cuts.apply_cuts(self.tractorfiles[0])
        desi, bgs, mws = cuts.apply_cuts(self.sweepfiles[0])
        data = io.read_tractor(self.tractorfiles[0])
        desi, bgs, mws = cuts.apply_cuts(data)
        data = io.read_tractor(self.sweepfiles[0])
        desi, bgs, mws = cuts.apply_cuts(data)

        # bgs_any1 = (desi & desi_mask.BGS_ANY)
        # bgs_any2 = (bgs != 0)
        # self.assertTrue(np.all(bgs_any1 == bgs_any2))

    def test_select_targets(self):
        #- select targets should work with either data or filenames
        for filelist in [self.tractorfiles, self.sweepfiles]:
            targets = cuts.select_targets(filelist, numproc=1)
            t1 = cuts.select_targets(filelist[0:1], numproc=1)
            t2 = cuts.select_targets(filelist[0], numproc=1)
            for col in t1.dtype.names:
                try:
                    notNaN = ~np.isnan(t1[col])
                except TypeError:  #- can't check string columns for NaN
                    notNan = np.ones(len(t1), dtype=bool)
                    
                self.assertTrue(np.all(t1[col][notNaN]==t2[col][notNaN]))            
            
    def test_missing_files(self):
        with self.assertRaises(ValueError):
            targets = cuts.select_targets(['blat.foo1234',], numproc=1)
        
    def test_parallel_select(self):
        for nproc in [1,2]:
            for filelist in [self.tractorfiles, self.sweepfiles]:
                targets = cuts.select_targets(filelist, numproc=nproc)
                self.assertTrue('DESI_TARGET' in targets.dtype.names)
                self.assertTrue('BGS_TARGET' in targets.dtype.names)
                self.assertTrue('MWS_TARGET' in targets.dtype.names)
                self.assertEqual(len(targets), np.count_nonzero(targets['DESI_TARGET']))
            
                bgs1 = (targets['DESI_TARGET'] & desi_mask.BGS_ANY) != 0
                bgs2 = targets['BGS_TARGET'] != 0
                self.assertTrue(np.all(bgs1 == bgs2))
        
if __name__ == '__main__':
    unittest.main()

