import unittest
import numpy as np

from pkg_resources import resource_filename
from desitarget.mock.sample_colors import SampleGMM
from desiutil.sklearn import GaussianMixtureModel

class TestSample(unittest.TestCase):


    def setUp(self):
        seed = 123
        gen = np.random.RandomState(seed)
        self.gmm = SampleGMM(random_state=gen)

    def test_sample(self):
        n_targets = 8000

        lrg_samp = self.gmm.sample(target_type='LRG', n_targets=n_targets)
        elg_samp = self.gmm.sample(target_type='ELG', n_targets=n_targets)
        qso_samp = self.gmm.sample(target_type='QSO', n_targets=n_targets)
        bgs_samp = self.gmm.sample(target_type='BGS', n_targets=n_targets)

        lrgMean = [19.55, 1.54, 1.17]
        elgMean = [22.84, 0.35, 0.77]
        qsoMean = [21.38, 0.43, 0.26]
        bgsMean = [17.71, 1.02, 0.63]

        lrgStd = [0.60, 0.33, 0.37]
        elgStd = [0.52, 0.21, 0.19]
        qsoStd = [1.10, 0.69, 0.49]
        bgsStd = [2.17, 0.66, 0.56]

        mean_threshold = 0.15
        std_threshold = 2.5

        for ii, tt in enumerate(lrg_samp.dtype.names):
            self.assertTrue(np.abs(np.mean(lrg_samp[tt])-lrgMean[ii]) < mean_threshold)
            self.assertTrue(np.abs(np.std(lrg_samp[tt])-lrgStd[ii]) < std_threshold)

        for ii, tt in enumerate(elg_samp.dtype.names):
            self.assertTrue(np.abs(np.mean(elg_samp[tt])-elgMean[ii]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(qso_samp[tt])-qsoMean[ii]) < mean_threshold)
            self.assertTrue(np.abs(np.mean(bgs_samp[tt])-bgsMean[ii]) < mean_threshold)
            self.assertTrue(np.abs(np.std(elg_samp[tt])-elgStd[ii]) < std_threshold)
            self.assertTrue(np.abs(np.std(qso_samp[tt])-qsoStd[ii]) < std_threshold)
            self.assertTrue(np.abs(np.std(bgs_samp[tt])-bgsStd[ii]) < std_threshold)

if __name__ == '__main__':
    unittest.main()
