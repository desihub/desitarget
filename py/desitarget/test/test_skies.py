import unittest
from pkg_resources import resource_filename
import os.path
import fitsio
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from desitarget import skies, io, targets
from desitarget.skies import psfsize
from desitarget.cuts import _psflike

from desiutil import brick

from desitarget.geomask import ellipse_matrix, is_in_ellipse_matrix

class TestSKIES(unittest.TestCase):

    def setUp(self):
        #ADM at this nskymin you always seem to get at least 1 bad position
        self.nskymin = 5000000
        self.navoid = 2.
        self.psfsize = psfsize
        #ADM set the magnitude limits to something extremely faint
        #ADM so that nothing is limited on magnitude
        self.maglim = [100,100,100]

        #ADM location of input test file
        self.datadir = resource_filename('desitarget.test', 't')
        self.sweepfile = self.datadir + '/sweep-320m005-330p000.fits'

        #ADM read in the test sweeps file
        self.objs = io.read_tractor(self.sweepfile)
        
        #ADM need to ensure that one object has a large enough half-light radius
        #ADM to cover matching sky positions to larger objects
        self.objs['SHAPEEXP_R'][0] = 120.
        self.objs['SHAPEEXP_E1'][0] = -0.22389728
        self.objs['SHAPEEXP_E2'][0] = 0.42635256

        #ADM create a "maximum" search distance that is as large as the 
        #ADM diagonal across all objects in the test sweeps file
        #ADM note that this is only close to correct because the test
        #ADM file is near 0o Declination
        cmax = SkyCoord(max(self.objs["RA"])*u.degree, max(self.objs["DEC"])*u.degree)
        cmin = SkyCoord(min(self.objs["RA"])*u.degree, min(self.objs["DEC"])*u.degree)
        self.maxrad = cmax.separation(cmin).arcsec

    def test_calculate_separations(self):
        """
        Test the separation radius for objects are consistent with their shapes
        """
        sep = skies.calculate_separations(self.objs,self.navoid)
        
        #ADM are objects with radii of psfsize x the seeing PSF-like 
        #ADM (or galaxies that are compact to < psfsize arcsecond seeing)?
        w = np.where(sep==self.psfsize*self.navoid)
        maxsize = np.fmax(self.objs["SHAPEEXP_R"][w],self.objs["SHAPEDEV_R"][w])
        self.assertTrue(np.all(maxsize <= self.psfsize))

        #ADM are objects with radii of > psfsize x the seeing galaxies
        #ADM with larger half-light radii than psfsize arcsec?
        w = np.where(sep!=psfsize*self.navoid)
        maxsize = np.fmax(self.objs["SHAPEEXP_R"][w],self.objs["SHAPEDEV_R"][w])
        self.assertTrue(np.all(maxsize >= self.psfsize))

    def test_generate_sky_positions_psf(self):
        """
        Test bad sky positions match objects and good ones don't, using circles on the sky
        """
        #ADM retrieve only the "PSF" type objects
        wpsf = np.where(_psflike(self.objs["TYPE"]))
        psfobjs = self.objs[wpsf]

        #ADM the avoidance radius for PSF-like objects should be this
        sep = self.navoid*psfsize

        #ADM generate good and bad sky positions at a high density for testing
        #ADM of the PSF-like objects
        ragood, decgood, rabad, decbad = skies.generate_sky_positions(
            psfobjs,navoid=self.navoid,nskymin=self.nskymin,maglim=self.maglim)
          
        #ADM the object positions from the mock sweeps file
        cobjs = SkyCoord(psfobjs["RA"]*u.degree, psfobjs["DEC"]*u.degree)
        #ADM the calculated good sky positions from generate_sky_positions
        cskies = SkyCoord(ragood*u.degree, decgood*u.degree)
        
        #ADM test that none of the good sky positions match to a PSF object
        idskies, idobjs, d2d, _ = cobjs.search_around_sky(cskies,self.maxrad*u.arcsec)
        self.assertFalse(np.any(d2d.arcsec < sep))

        #ADM test that all of the bad sky positions match to an object
        #ADM for brevity just perform a circular match on the maximum radius
        cskies = SkyCoord(rabad*u.degree, decbad*u.degree)
        idskies, idobjs, d2d, _ = cobjs.search_around_sky(cskies,self.maxrad*u.arcsec)
        #ADM a list of indices where any sky position matched an object
        w = np.where(sep > d2d.arcsec)
        #ADM do we have the same total of unique (each sky position counted
        #ADM only once) matches as the total number of bad sky positions?
        self.assertEqual(len(np.unique(idskies[w])),len(rabad))

    def test_generate_sky_positions_ellipse(self):
        """
        Test good sky positions don't match elliptical avoidance zones
        """
        #ADM retrieve only the objects that are EXP in type
        morph = self.objs["TYPE"]
        explike = ( (morph == 'EXP')  | (morph == b'EXP') |
                    (morph == 'EXP ') | (morph == b'EXP ') )
        well = np.where(explike)
        ellobjs = self.objs[well]

        #ADM generate good and bad sky positions at a high density for testing
        ragood, decgood, rabad, decbad = skies.generate_sky_positions(
            ellobjs,navoid=self.navoid,nskymin=self.nskymin,maglim=self.maglim)
          
        #ADM the object positions from the mock sweeps file
        cobjs = SkyCoord(ellobjs["RA"]*u.degree, ellobjs["DEC"]*u.degree)
        #ADM the calculated good sky positions from generate_sky_positions
        cskies = SkyCoord(ragood*u.degree, decgood*u.degree)
        
        #ADM test that none of the good sky positions match to 
        #ADM one of the elliptical objects
        TEXP = ellipse_matrix(ellobjs["SHAPEEXP_R"]*self.navoid,
                                    ellobjs["SHAPEEXP_E1"],ellobjs["SHAPEEXP_E2"])

        for i, dum in enumerate(cobjs):
            is_in = is_in_ellipse_matrix(cskies.ra.deg, cskies.dec.deg,
                                         cobjs[i].ra.deg, cobjs[i].dec.deg,
                                         TEXP[...,i])
            self.assertFalse(np.any(is_in))

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
