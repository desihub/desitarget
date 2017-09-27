import unittest
from pkg_resources import resource_filename
import os.path
import fitsio
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from desitarget import skies, io

from desiutil import brick

class TestSKIES(unittest.TestCase):

    def setUp(self):
        #ADM location of input test file
        self.datadir = resource_filename('desitarget.test', 't')
        self.sweepfile = self.datadir + '/sweep-320m005-330p000.fits'

        #ADM read in the test sweeps file
        objs = io.read_tractor(self.sweepfile)

        #ADM create a "maximum" search distance that is as large as the 
        #ADM diagonal across all objects in the test sweeps file
        cmax = SkyCoord(max(objs["RA"])*u.degree, max(objs["DEC"])*u.degree)
        cmin = SkyCoord(min(objs["RA"])*u.degree, min(objs["DEC"])*u.degree)
        self.maxrad = cmax.separation(cmin).arcsec

        #ADM at this nskymin you always seem to get at least 1 bad position
        #ADM (based on 1000 trials)
        self.nskymin = 50000
        self.navoid = 2.

    
    def test_calculate_separations():
        """
        Test the separation radius for objects are consistent with their shapes
        """
        sep = skies.calculate_separations(objs,navoid)
        
        #ADM are objects with radii of 2 x the seeing PSF-like 
        #ADM (or galaxies that are compact to < 2 arcsecond seeing)?
        w = np.where(sep==2*navoid)
        maxsize = np.fmax(objs["SHAPEEXP_R"][w],objs["SHAPEDEV_R"][w])
        self.assertTrue(np.all(maxsize <= 2))

        #ADM are objects with radii of > 2 x the seeing galaxies
        #ADM with larger half-light radii than 2 arcsec?
        w = np.where(sep!=2*navoid)
        maxsize = np.fmax(objs["SHAPEEXP_R"][w],objs["SHAPEDEV_R"][w])
        self.assertTrue(np.all(maxsize >= 2))


    def test_generate_sky_positions():
        """
        Test that bad sky positions match objects and good sky positions don't
        """
        #ADM generate good and bad sky positions at a high density for testing
        ragood, decgood, rabad, decbad = 
                    skies.generate_sky_positions(objs,navoid=self.navoid,nskymin=self.nskymin)
          
        #ADM navoid x the largest half-light radius of a galaxy in the field
        #ADM or the PSF assuming a seeing of 2"
        nobjs = len(objs)
        sep = navoid*np.max(np.vstack([objs["SHAPEDEV_R"], objs["SHAPEEXP_R"], np.ones(nobjs)*2]).T,axis=1)

        #ADM the object positions from the mock sweeps file
        cobjs = SkyCoord(objs["RA"]*u.degree, objs["DEC"]*u.degree)

        #ADM test that none of the good sky positions match to an object
        #ADM using self.maxrad should capture all possible matches
        cskies = SkyCoord(ragood*u.degree, decgood*u.degree)
        idskies, idobjs, d2d, _ = cobjs.search_around_sky(cskies,self.maxrad*u.arcsec)
        self.assertFalse(np.any(sep[idobjs] > d2d.arcsec))

        #ADM test that all of the bad sky positions match to an object
        cskies = SkyCoord(rabad*u.degree, decbad*u.degree)
        idskies, idobjs, d2d, _ = cobjs.search_around_sky(cskies,self.maxrad*u.arcsec)
        #ADM a list of indices where any sky position matched an object
        w = np.where(sep[idobjs] > d2d.arcsec)
        #ADM do we have the same total of unique (each sky position counted
        #ADM only once) matches as the total number of bad sky positions?
        self.assertEqual(len(np.unique(idskies[w])),len(rabad))


if __name__ == '__main__':
    unittest.main()
