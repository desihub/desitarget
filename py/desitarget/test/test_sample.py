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

        #g,r,z,w1,w2,w3,w4 mean and std deviation for lrg, elg, qso, bgs data
        lrgMean = [23.56, 21.69, 19.86, 18.17, 18.68, 18.08, 16.00]
        elgMean = [22.80, 22.46, 21.71, 20.93, 20.84, 18.60, 16.25]
        qsoMean = [21.25, 20.86, 20.54, 19.56, 19.16, 17.91, 16.14]
        bgsMean = [18.80, 17.85, 17.21, 17.20, 17.61, 17.07, 16.03]
        lrgStd = [1.27, 0.86, 0.86, 1.79, 2.00, 2.03, 1.66]
        elgStd = [1.54, 1.48, 1.46, 1.92, 1.91, 1.54, 1.46]
        qsoStd = [1.41, 1.25, 1.22, 1.20, 1.25, 1.37, 1.31]
        bgsStd = [2.32, 2.07, 1.93, 1.91, 1.88, 1.59, 1.42]


        #Test mean and standard deviation
        mean_threshold = 0.05
        std_threshold = 0.5
        for i, index in enumerate(['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']):
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
