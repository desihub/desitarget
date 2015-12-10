import unittest
import os.path
from uuid import uuid4
from astropy.io import fits
import numpy as np

from desitarget import io

class TestIO(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # py/desitarget/test -> etc/datadir
        thisdir, thisfile = os.path.split(__file__)
        cls.datadir = os.path.abspath(thisdir+'/../../../') + '/etc/testdata'

    def setUp(self):
        self.testfile = 'test-{}.fits'.format(uuid4().hex)
        
    def tearDown(self):
        if os.path.exists(self.testfile):
            os.remove(self.testfile)
            
    def test_list_tractorfiles(self):
        files = io.list_tractorfiles(self.datadir)
        self.assertEqual(len(files), 3)
        for x in files:
            self.assertTrue(os.path.basename(x).startswith('tractor'))
            self.assertTrue(os.path.basename(x).endswith('.fits'))

    def test_list_sweepfiles(self):
        files = io.list_sweepfiles(self.datadir)
        self.assertEqual(len(files), 3)
        for x in files:
            self.assertTrue(os.path.basename(x).startswith('sweep'))
            self.assertTrue(os.path.basename(x).endswith('.fits'))
            
    def test_iter(self):
        for x in io.iter_files(self.datadir, prefix='tractor', ext='fits'):
            pass
        #- io.iter_files should also work with a file, not just a directory
        for y in io.iter_files(x, prefix='tractor', ext='fits'):
            self.assertEqual(x, y)
    
    def test_fix_dr1(self):
        '''test the DR1 TYPE dype fix (make everything S4)'''
        #- First, break it
        files = io.list_sweepfiles(self.datadir)
        objects = io.read_tractor(files[0])
        dt = objects.dtype.descr
        for i in range(len(dt)):
            if dt[i][0] == 'TYPE':
                dt[i] = ('TYPE', 'S10')
                break
        badobjects = objects.astype(np.dtype(dt))
        
        newobjects = io.fix_tractor_dr1_dtype(badobjects)
        self.assertEqual(newobjects['TYPE'].dtype, np.dtype('S4'))
        
    
    def test_readwrite_tractor(self):
        tractorfile = io.list_tractorfiles(self.datadir)[0]
        sweepfile = io.list_sweepfiles(self.datadir)[0]
        data = io.read_tractor(sweepfile)
        data = io.read_tractor(tractorfile)
        self.assertEqual(len(data), 6)  #- test data has 6 objects per file
        data, hdr = io.read_tractor(tractorfile, header=True)
        self.assertEqual(len(data), 6)  #- test data has 6 objects per file
        
        io.write_targets(self.testfile, data, indir=self.datadir)
        d2, h2 = fits.getdata(self.testfile, header=True)
        self.assertEqual(h2['DEPVER02'], self.datadir)
        self.assertEqual(data.dtype.names, d2.dtype.names)
        for column in data.dtype.names:
            self.assertTrue(np.all(data[column] == d2[column]))

    def test_brickname(self):
        self.assertEqual(io.brickname_from_filename('tractor-3301m002.fits'), '3301m002')
        self.assertEqual(io.brickname_from_filename('tractor-3301p002.fits'), '3301p002')
        self.assertEqual(io.brickname_from_filename('/a/b/tractor-3301p002.fits'), '3301p002')
        
if __name__ == '__main__':
    unittest.main()
