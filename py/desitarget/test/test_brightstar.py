import unittest
from pkg_resources import resource_filename
import os.path
import fitsio
import numpy as np

#ADM this is an old set of tests from when the brightmask
#ADM module only worked with circular sources and was
#ADM therefore called brightstar instead of brightmask
from desitarget import brightmask as brightstar
#ADM these remain useful tests to increase coverage, though
from desitarget.targetmask import desi_mask, targetid_mask

class TestBRIGHTSTAR(unittest.TestCase):

    def setUp(self):
        #ADM some locations of output test files
        self.testbsfile = 'bs.fits'
        self.testmaskfile = 'bsmask.fits'

        #ADM some locations of input files
        self.bsdatadir = resource_filename('desitarget.test', 't2')
        self.datadir = resource_filename('desitarget.test', 't')

    def tearDown(self):
        #ADM remove any existing bright star files in this directory
        if os.path.exists(self.testmaskfile):
            os.remove(self.testmaskfile)
        if os.path.exists(self.testbsfile):
            os.remove(self.testbsfile)

    def test_collect_bright_stars(self):
        """Test the collection of bright stars from the sweeps
        """
        #ADM collect the bright stars from the sweeps in the data directory and write to file...
        bs1 = brightstar.collect_bright_stars('grz',[9,9,9],rootdirname=self.bsdatadir,outfilename=self.testbsfile)
        #ADM ...and read in the file that was written
        bs2 = fitsio.read(self.testbsfile) 
        #ADM the created collection of objects from the sweeps should be the same as the read-in file
        bs1ids = bs1['BRICKID'].astype(np.int64)*1000000 + bs1['OBJID']
        bs2ids = bs2['BRICKID'].astype(np.int64)*1000000 + bs2['OBJID']
        self.assertTrue(np.all(bs1ids == bs2ids))

    def test_make_bright_star_mask(self):
        """Test the construction of a bright star mask
        """
        #ADM create a collection of bright stars and write to file
        bs1 = brightstar.collect_bright_stars('grz',[23,23,23],rootdirname=self.bsdatadir,outfilename=self.testbsfile)
        #ADM create a bright star mask from the collection of bright stars and write to file...
        mask = brightstar.make_bright_star_mask('grz',[23,23,23],infilename=self.testbsfile,outfilename=self.testmaskfile)
        #ADM ...and read it back in
        mask1 = fitsio.read(self.testmaskfile)
        #ADM create the bright star mask from scratch
        mask2 = brightstar.make_bright_star_mask('grz',[23,23,23],rootdirname=self.bsdatadir)
        #ADM the created-from-scratch mask should be the same as the read-in mask
        self.assertTrue(np.all(mask1["TARGETID"] == mask2["TARGETID"]))

if __name__ == '__main__':
    unittest.main()
