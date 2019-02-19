# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.skyfibers.
"""
import unittest
from pkg_resources import resource_filename
import numpy as np

from desitarget import skyfibers
from desitarget.targetmask import desi_mask

from desitarget.skyutilities.legacypipe.util import LegacySurveyData

from glob import glob
from os.path import basename


class TestSKYFIBERS(unittest.TestCase):

    @classmethod
    def setUp(self):
        # ADM location of input test survey directory structure
        self.sd = resource_filename('desitarget.test', 'dr6')

        # ADM create the survey object
        self.survey = LegacySurveyData(self.sd)

        # ADM determine which bricks we can access in the test directory
        brickdirs = glob("{}/coadd/*/*".format(self.survey.survey_dir))
        bricknames = [basename(brickdir) for brickdir in brickdirs]
        # ADM just test with one brick
        self.brickname = bricknames[0]

        if self.brickname != '0959p805':
            print("brick name is {} not '0959p805'".format(brickname))
            print("'0959p805' was chosen to as it has good g-band")
            print("images and is relatively small")
            raise ValueError

        # ADM generate a handful (~4) sky locations
        brickarea = 0.25*0.25
        self.nskies = 4
        # ADM as the code ensures a minimum number of skies is
        # ADM generated, we need to pass 3.9999 not 4
        self.nskiespersqdeg = int((self.nskies-1e-6)/brickarea)

        # ADM extract flux in 1 and 2" apertures
        self.ap_arcsec = [1., 2.]

    def test_survey_object(self):
        """
        Test that the survey object is correctly initialized
        """
        self.assertTrue(self.sd == self.survey.survey_dir)

    def test_density_of_sky_fibers(self):
        """
        Test the functions that generate the requisite DESI sky fiber density
        """
        modelhi = skyfibers.model_density_of_sky_fibers(10)
        modello = skyfibers.model_density_of_sky_fibers(1)
        hi = skyfibers.density_of_sky_fibers(5)
        lo = skyfibers.density_of_sky_fibers(1)

        # ADM check that margin behaves correctly (with 10x as much margin)
        # ADM we should always have 10 times as many fibers
        self.assertTrue(modelhi/modello == 10)
        self.assertTrue(hi/lo == 5)

    def test_make_skies_for_a_brick(self):
        """
        Test the production of a few sky locations from a survey object
        """
        # ADM generate the skies as a structured array
        skies = skyfibers.make_skies_for_a_brick(self.survey, self.brickname,
                                                 nskiespersqdeg=self.nskiespersqdeg,
                                                 apertures_arcsec=self.ap_arcsec)

        # ADM check the brick name information is generated correctly
        # ADM remember the default for target output strings is bytes
        self.assertTrue(np.all(
            skies["BRICKNAME"] == np.array(self.brickname).astype('S'))
        )

        # ADM check we've stored the correct information give the numbers of
        # ADM skies requested and the length of the apertures
        self.assertTrue(
            skies["APFLUX_R"].shape == (self.nskies, len(self.ap_arcsec))
        )

        # ADM generate the associated sky table
        skytable = skyfibers.sky_fibers_for_brick(self.survey, self.brickname,
                                                  nskies=self.nskies,
                                                  apertures_arcsec=self.ap_arcsec)

        # ADM check some of the outputs are the same for the table and FITS
        self.assertTrue(
            np.all(skies["APFLUX_G"] == skytable.apflux_g)
        )
        self.assertTrue(
            np.all(skies["APFLUX_IVAR_Z"] == skytable.apflux_ivar_z)
        )

    def test_make_skies_for_a_brick_per_band(self):
        """
        Test aperture fluxes at sky locations are correct for different bands
        """
        # ADM generate the skies as a structured array
        skies = skyfibers.make_skies_for_a_brick(self.survey, self.brickname,
                                                 nskiespersqdeg=self.nskiespersqdeg,
                                                 apertures_arcsec=self.ap_arcsec)

        # ADM generate the skies just in r-band and z-band...
        rzskies = skyfibers.make_skies_for_a_brick(self.survey, self.brickname,
                                                   nskiespersqdeg=self.nskiespersqdeg,
                                                   apertures_arcsec=self.ap_arcsec,
                                                   bands=["r", "z"])

        # ADM ...and just in g-band (which should be THE ONLY GOOD band)!
        # ADM which is why I set up these tests with brick 0959p805
        gskies = skyfibers.make_skies_for_a_brick(self.survey, self.brickname,
                                                  nskiespersqdeg=self.nskiespersqdeg,
                                                  apertures_arcsec=self.ap_arcsec,
                                                  bands=["g"])

        # ADM the r and z bands for brick 0959p805 are bad, so should be the
        # ADM same no matter which bands we extract (they should be all zero)
        self.assertTrue(
            np.all(rzskies["APFLUX_R"] == skies["APFLUX_R"])
        )
        self.assertTrue(
            np.all(rzskies["APFLUX_Z"] == skies["APFLUX_Z"])
        )

        # ADM but the g band is good, so shouldn't be the same if we extract
        # ADM it in concert with the other (r,z) bands
        self.assertFalse(
            np.all(gskies["APFLUX_G"] == skies["APFLUX_G"])
        )

    def test_target_bits(self):
        """
        Test that apertures that are in blobs have the BAD_SKY bit set
        """
        # ADM get sky locations in g-band (which should be THE ONLY GOOD band)!
        # ADM which is why I set up these tests with brick 0959p805.
        gskies = skyfibers.make_skies_for_a_brick(self.survey, self.brickname,
                                                  nskiespersqdeg=self.nskiespersqdeg,
                                                  apertures_arcsec=self.ap_arcsec,
                                                  bands=["g"])

        # ADM note that these only technically work because the IVARs are good
        # ADM good for brick 0959p805 in DR6 of the Legacy Surveys.
        wgood = gskies["BLOBDIST"] > 0.
        wbad = gskies["BLOBDIST"] == 0.

        # ADM check apertures with good/bad flux have good/bad sky bits set.
        self.assertTrue(
            np.all(gskies[wgood]["DESI_TARGET"] == desi_mask.SKY)
        )
        self.assertTrue(
            np.all(gskies[wbad]["DESI_TARGET"] == desi_mask.BAD_SKY)
        )

    def test_bundle_bricks(self):
        """
        Test the bundle_bricks scripting code simply executes without bugs
        """
        dum = skyfibers.bundle_bricks(1, 1, 1, surveydir=self.survey.survey_dir)
        self.assertTrue(dum is None)

    def test_select_skies(self):
        """
        Test the wrapper function for batch selection of skies
        """
        # ADM generate the skies as a structured array
        skies = skyfibers.make_skies_for_a_brick(self.survey, self.brickname,
                                                 nskiespersqdeg=self.nskiespersqdeg,
                                                 apertures_arcsec=self.ap_arcsec)

        # ADM generate skies using the wrapper function
        ss = skyfibers.select_skies(self.survey, numproc=1,
                                    nskiespersqdeg=self.nskiespersqdeg,
                                    apertures_arcsec=self.ap_arcsec)

        # ADM check the wrapper generates a single brick correctly
        self.assertTrue(np.all(ss == skies))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_skyfibers
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
