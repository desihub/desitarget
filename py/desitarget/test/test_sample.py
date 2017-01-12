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
        self.bgs = sample.GaussianMixtureModel.load(self.modelpath + '/bgsMag_gmm.fits')


    def test_sample_magnitudes(self):
        seed = 123
        gen = np.random.RandomState(seed)
        n_targets = 8000
        #Sampled magnitudes
        lrg_samp = sample.sample_magnitudes('LRG', n_targets=n_targets, random_state=gen)
        elg_samp = sample.sample_magnitudes('ELG', n_targets=n_targets, random_state=gen)
        qso_samp = sample.sample_magnitudes('QSO', n_targets=n_targets, random_state=gen)
        bgs_samp = sample.sample_magnitudes('BGS', n_targets=n_targets, random_state=gen)

        #g,r,z mean and std deviation for lrg, elg, qso data
        lrgMean = [23.66, 21.72, 19.89]
        elgMean = [23.14, 22.80, 22.05]
        qsoMean = [21.59, 21.18, 20.87]
        bgsMean = [19.13, 18.06, 17.44]
        lrgStd = [1.17, 0.77, 0.74]
        elgStd = [0.99, 0.93, 0.94]
        qsoStd = [1.39, 1.20, 1.19]
        bgsStd = [2.16, 1.86, 1.78]


        #Test mean and standard deviation
        mean_threshold = 0.05
        std_threshold = 0.5
        for i, index in enumerate(['g', 'r', 'z']):
            self.assertTrue(np.abs(np.mean(lrg_samp[index])-lrgMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(elg_samp[index])-elgMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(qso_samp[index])-qsoMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(bgs_samp[index])-bgsMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.std(lrg_samp[index])-lrgStd[i]) < std_threshold)
            self.assertTrue(np.abs(np.std(elg_samp[index])-elgStd[i]) < std_threshold)
            self.assertTrue(np.abs(np.std(qso_samp[index])-qsoStd[i]) < std_threshold)
            self.assertTrue(np.abs(np.std(bgs_samp[index])-bgsStd[i]) < std_threshold)

if __name__ == '__main__':
    unittest.main()
