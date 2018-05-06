import unittest
from pkg_resources import resource_filename
import numpy as np

from desitarget import skyfibers
from desitarget import io, targets

from desitarget.skyutilities.legacypipe.util import LegacySurveyData

from glob import glob
from os.path import basename

class TestSKYFIBERS(unittest.TestCase):

    @classmethod
    def setUp(self):
        #ADM location of input test survey directory structure
        self.sd = resource_filename('desitarget.test', 'dr6')

        #ADM create the survey object
        self.survey = LegacySurveyData(self.sd)

        #ADM determine which bricks we can access in the test directory
        brickdirs = glob("{}/coadd/*/*".format(survey.survey_dir))
        bricknames = [ basename(brickdir) for brickdir in brickdirs ]
        #ADM just test with one brick
        self.brickname = bricknames[0]

        #ADM generate a handfule (~4) sky locations
        brickarea = 0.25*0.25
        self.nskies = 4.
        self.nskiespersqdeg = int(self.nskies/brickarea)

        #ADM extract flux in 1 and 2" apertures
        self.ap_arcsec = [1.,2.]


    def test_survey_object(self):
        """
        Test that the survey object is correctly initialized
        """
        np.assertTrue(self.sd == self.survey.survey_dir)
        

    def test_density_of_sky_fibers(self):
        """
        Test the functions that generate the requisite DESI sky fiber density
        """
        modelhi = skyfibers.model_density_of_sky_fibers(10)
        modello = skyfibers.model_density_of_sky_fibers(1)
        hi = skyfibers.density_of_sky_fibers(5)
        lo = skyfibers.density_of_sky_fibers(1)

        #ADM check that margin behaves correctly (with 10x as much margin)
        #ADM we should always have 10 times as many fibers
        self.assertTrue(modelhi/modello == 10)
        self.assertTrue(hi/lo == 5)

    def test_make_skies_for_a_brick(self):
        """
        Test the production of a few sky locations from a survey object
        """
        #ADM generate the FITS format for the skies
        skies = skyfibers.make_skies_for_a_brick(self.survey, self.brickname, 
                                        nskiespersqdeg=self.nskiespersqdeg,
                                        apertures_arcsec=self.ap_arcsec)

        #ADM check the brick name information is generated correctly
        #ADM remember the default for target output strings is bytes
        np.assertTrue(np.all(
            skies["BRICKNAME"] == np.array(self.brickname).astype('S'))
        )
        
        #ADM check we've stored the correct information give the numbers of
        #ADM skies requested and the length of the apertures
        np.assertTrue(
            skies["APFLUX_R"].shape == (self.nskies,len(self.ap_arcsec)))

        #ADM generate the associated sky table
        skytable = skyfibers.sky_fibers_for_brick(self.survey,self.brickname,
                                            nskies=self.nskies,
                                            apertures_arcsec=self.ap_arcsec)

        #ADM check some of the outputs are the same for the table and FITS
        np.assertTrue(np.all(skies["APFLUX_G"] == skytable.apflux_g))
        np.assertTrue(np.all(skies["APFLUX_IVAR_Z"] == skytable.apflux_ivar_z))


    def test_make_skies_for_a_brick_per_band(self):
        """
        Test sky locations pro
        """
        #ADM generate the FITS format for the skies
        skies = skyfibers.make_skies_for_a_brick(self.survey, self.brickname, 
                                        nskiespersqdeg=self.nskiespersqdeg,
                                        apertures_arcsec=self.ap_arcsec,
                                        bands = ["g","r","z"])




if __name__ == '__main__':
    unittest.main()
