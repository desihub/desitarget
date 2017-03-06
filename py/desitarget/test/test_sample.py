import unittest
import numpy as np

from pkg_resources import resource_filename
from desitarget.mock import sample
from desiutil.sklearn import GaussianMixtureModel

class TestSample(unittest.TestCase):


    def setUp(self):
        self.modelpath = resource_filename('desitarget', "mock/data")
        self.lrg = GaussianMixtureModel.load(self.modelpath + '/lrg_gmm.fits')
        self.elg = GaussianMixtureModel.load(self.modelpath + '/elg_gmm.fits')
        self.qso = GaussianMixtureModel.load(self.modelpath + '/qso_gmm.fits')
        self.bgs = GaussianMixtureModel.load(self.modelpath + '/bgs_gmm.fits')


    def test_sample_mag_shape(self):
        seed = 123
        gen = np.random.RandomState(seed)
        n_targets = 8000
        #Sampled magnitudes
        lrg_samp = sample.sample_mag_shape('LRG', n_targets=n_targets, random_state=gen)
        elg_samp = sample.sample_mag_shape('ELG', n_targets=n_targets, random_state=gen)
        qso_samp = sample.sample_mag_shape('QSO', n_targets=n_targets, random_state=gen)
        bgs_samp = sample.sample_mag_shape('BGS', n_targets=n_targets, random_state=gen)

        #g,r,z,w1,w2,w3,w4 mean and std deviation for lrg, elg, qso, bgs data
        lrgMean = [23.55, 21.68, 19.85, 18.14, 18.66, 18.05, 15.99, 1.56, 0.00,
        0.00, 5.01, 0.00, 0.00]
        elgMean = [22.78, 22.45, 21.70, 20.92, 20.87, 18.56, 16.26, 0.60, 0.00,
        0.00, 0.52, 0.00, 0.00]
        qsoMean = [21.22, 20.85, 20.53, 19.58, 19.17, 17.93, 16.18]
        bgsMean = [18.77, 17.81, 17.18, 17.18, 17.59, 17.05, 15.99, 1.82, 0.00,
        0.00, 4.13, 0.00, 0.00]
        lrgStd = [1.27, 0.87, 0.86, 1.78, 1.99, 2.03, 1.67, 10.65, 0.13, 0.13,
        19.95, 0.16, 0.18]
        elgStd = [1.62, 1.56, 1.53, 1.91, 1.93, 1.50, 1.43, 3.36, 0.14, 0.14,
        5.85, 0.06, 0.07]
        qsoStd = [1.39, 1.24, 1.22, 1.22, 1.27, 1.39, 1.35]
        bgsStd = [2.32, 2.07, 1.93, 1.90, 1.88, 1.61, 1.45, 7.44, 0.19, 0.20,
        18.66, 0.19, 0.19]


        #Test mean and standard deviation
        mean_threshold = 0.15
        std_threshold = 2.5
        for i, index in enumerate(['g', 'r', 'z', 'w1', 'w2', 'w3', 'w4']):
            self.assertTrue(np.abs(np.mean(lrg_samp[index])-lrgMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(elg_samp[index])-elgMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(qso_samp[index])-qsoMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(bgs_samp[index])-bgsMean[i]) < mean_threshold)
            self.assertTrue(np.abs(np.std(lrg_samp[index])-lrgStd[i]) < std_threshold)
            self.assertTrue(np.abs(np.std(elg_samp[index])-elgStd[i]) < std_threshold)
            self.assertTrue(np.abs(np.std(qso_samp[index])-qsoStd[i]) < std_threshold)
            self.assertTrue(np.abs(np.std(bgs_samp[index])-bgsStd[i]) < std_threshold)

        for i, index in enumerate(['exp_r', 'exp_e1', 'exp_e2', 'dev_r',
        'dev_e1', 'dev_e2']):
            self.assertTrue(np.abs(np.mean(lrg_samp[index])-lrgMean[i+7]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(elg_samp[index])-elgMean[i+7]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(bgs_samp[index])-bgsMean[i+7]) < mean_threshold)
            self.assertTrue(np.abs(np.std(lrg_samp[index])-lrgStd[i+7]) < std_threshold)
            self.assertTrue(np.abs(np.std(elg_samp[index])-elgStd[i+7]) < std_threshold)
            self.assertTrue(np.abs(np.std(bgs_samp[index])-bgsStd[i+7]) < std_threshold)

if __name__ == '__main__':
    unittest.main()
