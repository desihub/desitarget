import unittest
import numpy as np

from pkg_resources import resource_filename
from desitarget.mock import sample


class TestSample(unittest.TestCase):


    def setUp(self):
        self.modelpath = resource_filename('desitarget', "mock/data")
        self.lrg = sample.GaussianMixtureModel.load(self.modelpath + '/lrgMag_gmm.fits')
        self.elg = sample.GaussianMixtureModel.load(self.modelpath + '/elgMag_gmm.fits')
        self.qso = sample.GaussianMixtureModel.load(self.modelpath + '/qsoMag_gmm.fits')


    def test_sample_magnitudes(self):
        seed = 123
        gen = np.random.RandomState(seed)
        n_targets = 8000
        #Sampled magnitudes
        lrg_samp = sample.sample_magnitudes('LRG', n_targets=n_targets, random_state=gen)
        elg_samp = sample.sample_magnitudes('ELG', n_targets=n_targets, random_state=gen)
        qso_samp = sample.sample_magnitudes('QSO', n_targets=n_targets, random_state=gen)

        #g,r,z mean and std deviation for lrg, elg, qso data
        lrgMean = [23.66, 21.72, 19.89]
        elgMean = [23.14, 22.80, 22.05]
        qsoMean = [21.59, 21.18, 20.87]
        lrgStd = [1.17, 0.77, 0.74]
        elgStd = [0.99, 0.93, 0.94]
        qsoStd = [1.39, 1.20, 1.19]


        #Test mean and standard deviation
        mean_threshold = 0.05
        std_threshold = 0.5
        for i in range(0,3):
            self.assertTrue(np.abs(np.mean(lrg_samp[:,i])-lrgMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(elg_samp[:,i])-elgMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(qso_samp[:,i])-qsoMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.std(lrg_samp[:,i])-lrgStd[i]) < std_threshold)
            self.assertTrue(np.abs(np.std(elg_samp[:,i])-elgStd[i]) < std_threshold)
            self.assertTrue(np.abs(np.std(qso_samp[:,i])-qsoStd[i]) < std_threshold)

if __name__ == '__main__':
    unittest.main()
