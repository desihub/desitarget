import unittest
import os.path
from uuid import uuid4
from astropy.io import fits
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

    def test_cuts1(self):
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
        for nproc in [1,2]:
            for filelist in [self.tractorfiles, self.sweepfiles]:
                targets = cuts.select_targets(filelist, numproc=nproc)
                self.assertTrue('DESI_TARGET' in targets.dtype.names)
                self.assertTrue('BGS_TARGET' in targets.dtype.names)
                self.assertTrue('MWS_TARGET' in targets.dtype.names)
                self.assertEqual(len(targets), np.count_nonzero(targets['DESI_TARGET']))
            
                bgs1 = targets['DESI_TARGET'] & desi_mask.BGS_ANY
                bgs2 = targets['BGS_TARGET'] != 0
                self.assertTrue(np.all(bgs1 == bgs2))
        
if __name__ == '__main__':
    unittest.main()

