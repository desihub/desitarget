import unittest
from pkg_resources import resource_filename
import os.path
import fitsio
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from desitarget import skies, io, targets

from desiutil import brick

class TestSKIES(unittest.TestCase):

    def setUp(self):
        #ADM location of input test file
        self.datadir = resource_filename('desitarget.test', 't')
        self.sweepfile = self.datadir + '/sweep-320m005-330p000.fits'

        #ADM read in the test sweeps file
        self.objs = io.read_tractor(self.sweepfile)

        #ADM create a "maximum" search distance that is as large as the 
        #ADM diagonal across all objects in the test sweeps file
        cmax = SkyCoord(max(self.objs["RA"])*u.degree, max(self.objs["DEC"])*u.degree)
        cmin = SkyCoord(min(self.objs["RA"])*u.degree, min(self.objs["DEC"])*u.degree)
        self.maxrad = cmax.separation(cmin).arcsec

        #ADM at this nskymin you always seem to get at least 1 bad position
        #ADM (based on 1000 trials)
        self.nskymin = 50000
        self.navoid = 2.

    def test_calculate_separations(self):
        """
        Test the separation radius for objects are consistent with their shapes
        """
        sep = skies.calculate_separations(self.objs,self.navoid)
        
        #ADM are objects with radii of 2 x the seeing PSF-like 
        #ADM (or galaxies that are compact to < 2 arcsecond seeing)?
        w = np.where(sep==2*self.navoid)
        maxsize = np.fmax(self.objs["SHAPEEXP_R"][w],self.objs["SHAPEDEV_R"][w])
        self.assertTrue(np.all(maxsize <= 2))

        #ADM are objects with radii of > 2 x the seeing galaxies
        #ADM with larger half-light radii than 2 arcsec?
        w = np.where(sep!=2*self.navoid)
        maxsize = np.fmax(self.objs["SHAPEEXP_R"][w],self.objs["SHAPEDEV_R"][w])
        self.assertTrue(np.all(maxsize >= 2))

    def test_generate_sky_positions(self):
        """
        Test that bad sky positions match objects and good sky positions don't
        """
        #ADM generate good and bad sky positions at a high density for testing
        ragood, decgood, rabad, decbad = skies.generate_sky_positions(
            self.objs,navoid=self.navoid,nskymin=self.nskymin)
          
        #ADM navoid x the largest half-light radius of a galaxy in the field
        #ADM or the PSF assuming a seeing of 2"
        nobjs = len(self.objs)
        sep = self.navoid*np.max(np.vstack(
            [self.objs["SHAPEDEV_R"], self.objs["SHAPEEXP_R"], np.ones(nobjs)*2]).T,axis=1)

        #ADM the object positions from the mock sweeps file
        cobjs = SkyCoord(self.objs["RA"]*u.degree, self.objs["DEC"]*u.degree)

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

    def test_make_sky_targets_bits(self):
        """
        Check that the output bit formatting from make_sky_targets is correct
        """
        #ADM make the output rec array
        outskies = skies.make_sky_targets(
            self.objs,navoid=self.navoid,nskymin=self.nskymin) 

        #ADM construct the OBJID (which is just the sequential ordering of the sky positions)
        objid = np.arange(len((outskies)))

        #ADM construct the BRICKID
        b = brick.Bricks(bricksize=0.25)
        brickid = b.brickid(outskies["RA"],outskies["DEC"])

        #ADM mock-up the release number from the input objects' information
        release = np.max(self.objs["RELEASE"])

        #ADM check official targetid agrees with the output targetid from make_sky_targets
        targetid = targets.encode_targetid(objid=objid,brickid=brickid,release=release,sky=1)
        self.assertTrue(np.all(targetid==outskies["TARGETID"]))

if __name__ == '__main__':
    unittest.main()
