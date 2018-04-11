"""
desitarget.mock.sample_colors
=============================

Samples magnitudes and shapes for LRG, ELG, QSO, and BGS targets from a Gaussian mixture model.

The model for each object type is fit to DR2 targets that have passed target
selection critera.
"""

from __future__ import print_function, division
import numpy as np

from desiutil.log import get_logger
log = get_logger()

class SampleGMM(object):

    def __init__(self, random_state=None):
        from pkg_resources import resource_filename
        from desiutil.sklearn import GaussianMixtureModel

        gmm_path = 'mock/data/dr5/colors/'

        bgsfile = resource_filename('desitarget', gmm_path+'bgs_colors_gmm.fits')
        elgfile = resource_filename('desitarget', gmm_path+'elg_colors_gmm.fits')
        lrgfile = resource_filename('desitarget', gmm_path+'lrg_colors_gmm.fits')
        qsofile = resource_filename('desitarget', gmm_path+'qso_colors_gmm.fits')

        self.bgsmodel = GaussianMixtureModel.load(bgsfile)
        self.elgmodel = GaussianMixtureModel.load(elgfile)
        self.lrgmodel = GaussianMixtureModel.load(lrgfile)
        self.qsomodel = GaussianMixtureModel.load(qsofile)

        self.random_state = random_state

    def sample(self, target_type='LRG', n_targets=1):
        """Sample colors and one magnitude based on target type (i.e. LRG, ELG,
        QSO, BGS).

        Can sample multiple targets at once and needs only to be called
        once for each target_type.

        Args:
          target_type (str) : One of four object types (LRG, ELG, QSO, BGS).
          n_targets (int) : Number of sampled magntiudes to be returned for the
            specified target_type.

        Returns: np.ndarray length n_targets :
            Structured array with the following columns for each target type:
                BGS,ELG,QSO:
                    r, g-r, r-z
                LRG:
                    z, r-z, z-w1
        """

        if target_type not in ('BGS', 'ELG', 'LRG', 'QSO'):
            log.fatal('Unknown object type {}!'.format(target_type))
            raise ValueError

        # Generate a sample of colors/magnitude of size n_targets.
        if target_type == 'BGS':
            params = self.bgsmodel.sample(n_targets, self.random_state).astype('f4')
        elif target_type == 'ELG':
            params = self.elgmodel.sample(n_targets, self.random_state).astype('f4')
        elif target_type == 'LRG':
            params = self.lrgmodel.sample(n_targets, self.random_state).astype('f4')
        elif target_type == 'QSO':
            params = self.qsomodel.sample(n_targets, self.random_state).astype('f4')

        if target_type != 'LRG':
            tags = ('r', 'g-r', 'r-z')

        else:
            tags = ('z', 'r-z', 'z-w1')

        samp = np.empty( n_targets, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]

        return samp
