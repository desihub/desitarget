import unittest
from pkg_resources import resource_filename
import numpy as np

from desitarget import skyfibers as skies
from desitarget import io, targets

from desitarget.skyutilities.legacypipe.util import LegacySurveyData

class TestSKIES(unittest.TestCase):

    def setUp(self):
        #ADM location of input test survey directory structure
        self.sd = resource_filename('desitarget.test', 'dr6')
        #ADM create the survey object
        self.survey = LegacySurveyData(self.sd)

    def test_survey_object(self):
        """
        Test that the survey object is correctly initialized
        """
        np.assertTrue(self.sd == self.survey.survey_dir)
        

    def test_density_of_sky_fibers(self):
        """
        Test the functions that generate the requisite DESI sky fiber density
        """
        modelhi = skies.model_density_of_sky_fibers(10)
        modello = skies.model_density_of_sky_fibers(1)
        hi = skies.density_of_sky_fibers(5)
        lo = skies.density_of_sky_fibers(1)

        #ADM check that margin behaves correctly (with 10x as much margin)
        #ADM we should always have 10 times as many fibers
        self.assertTrue(modelhi/modello == 10)
        self.assertTrue(hi/lo == 5)


if __name__ == '__main__':
    unittest.main()
