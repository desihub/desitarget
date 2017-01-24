import unittest
from pkg_resources import resource_filename
import numpy as np

from desitarget import htm

class TestHTM(unittest.TestCase):

    def setUp(self):
        self.ra = np.array([ 287.87029928,  223.95814592,  328.99793884,  127.24251626])
        self.dec = np.array([ 16.22908364, -53.49629596,  24.40112077, -23.14789754])
        self.id = np.array(['N023330222213310122202','S233100021230000131322','N031100100010011323332','S131112210233032311131'],dtype='<U22')
        self.intid = np.array([14015193368226, 12043250829178, 14104942108414, 10813196528989])

    def test_htm_lookup_char(self):
        testid = htm.lookup(self.ra,self.dec,verbose=False)
        self.assertTrue(np.all(testid == self.id))

    def test_htm_lookup_int(self):
        testid = htm.lookup(self.ra,self.dec,charpix=False,verbose=False)
        self.assertTrue(np.all(testid == self.intid))

    def test_one_htm_lookup_char(self):
        testid = htm.lookup(self.ra[0],self.dec[0],verbose=False)
        self.assertTrue(testid == self.id[0])

    def test_one_htm_lookup_int(self):
        testid = htm.lookup(self.ra[0],self.dec[0],charpix=False,verbose=False)
        self.assertTrue(testid == self.intid[0])

if __name__ == '__main__':
    unittest.main()
