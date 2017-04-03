import unittest
import numpy as np

from desitarget import htm

class TestHTM(unittest.TestCase):

    def setUp(self):
        #- Test canonical values from http://skyserver.sdss.org/dr13/en/tools/search/sql.aspx
        #- select dbo.fHtmEq(287.87029928, 16.22908364), dbo.fHtmGetString(14015193368226)
        self.ra = np.array([ 287.87029928,  223.95814592,  328.99793884,  127.24251626])
        self.dec = np.array([ 16.22908364, -53.49629596,  24.40112077, -23.14789754])
        self.id = np.array(['N023330222213310122202','S233100021230000131322','N031100100010011323332','S131112210233032311131'],dtype='<U22')
        self.intid = np.array([14015193368226, 12043250829178, 14104942108414, 10813196528989])
        self.racorner = np.array([0,0,90,180,270,0])
        self.deccorner = np.array([90,0,0,0,0,-90])
        self.idcorner = np.array(['N31000000000', 'N32000000000', 'N22000000000', 'N12000000000','N02000000000', 'S01000000000'],dtype='<U12')

        self.raequator = [10, 45, 350]
        self.decequator = [0, 0, 0]
        self.strid_equator = ['N320010020010020010020', 'N302000000000000000000', 'N000020010020010020010']
        self.intid_equator = [17046860497160, 16630113370112, 13202798690820]

        self.rameridian = [0, 0, 90, 90, 180, 180, 270, 270]
        self.decmeridian = [10, 45, -10, -45, -45, 22.5, 60, -33]
        self.intid_meridian = [17051089388036, 16836271800320, 9900034916616, 9964324126720,
            11063835754496, 14877766713344, 13578968603033, 12202153675017]

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

    def test_corner(self):
        #ADM test some corner cases at the level of a few-millionths of an arcsec
        #ADM note that it is impossible to guarantee exact agreement for corner
        #ADM cases across different implementations, but this suggests acreement
        #ADM with the official release to at least millionths of an arcsec
        testid = htm.lookup(self.racorner,self.deccorner,verbose=False,level=10)
        self.assertTrue(np.all(testid == self.idcorner))

    @unittest.expectedFailure
    def test_equator(self):
        strid = htm.lookup(self.raequator,self.decequator,verbose=False,level=20,charpix=True)
        intid = htm.lookup(self.raequator,self.decequator,verbose=False,level=20,charpix=False)
        # for i in range(len(strid)):
        #     print('{} {} {} {}  {} {}'.format(
        #         self.raequator[i], self.decequator[i],
        #         self.strid_equator[i], strid[i],
        #         self.intid_equator[i], intid[i],
        #     ))
        self.assertTrue(np.all(intid == self.intid_equator))
        self.assertTrue(np.all(strid == self.strid_equator))

    @unittest.expectedFailure
    def test_meridian(self):
        strid = htm.lookup(self.rameridian,self.decmeridian,verbose=False,level=20,charpix=True)
        intid = htm.lookup(self.rameridian,self.decmeridian,verbose=False,level=20,charpix=False)
        # for i in range(len(strid)):
        #     ### print('{} {} {} {}  {} {}'.format(
        #     print('{} {}  {} {}'.format(
        #         self.rameridian[i], self.decmeridian[i],
        #         # self.strid_meridian[i], strid[i],
        #         self.intid_meridian[i], intid[i],
        #     ))
        self.assertTrue(np.all(intid == self.intid_meridian))
        # self.assertTrue(np.all(strid == self.strid_meridian))


if __name__ == '__main__':
    unittest.main()
