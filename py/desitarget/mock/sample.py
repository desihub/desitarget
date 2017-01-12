"""Samples magnitudes for LRGs, ELGs and QSOs from a Gaussian mixture model.

The model for each object type is fit to DR2 targets that have passed target
selection critera.
"""

from __future__ import print_function, division

import numpy as np
import os
from pkg_resources import resource_filename
from astropy.io import fits

class GaussianMixtureModel(object):

    def __init__(self, weights, means, covars, covtype):
        self.weights = weights
        self.means = means
        self.covars = covars
        self.covtype = covtype
        self.n_components, self.n_dimensions = self.means.shape

    @staticmethod
    def save(model, filename):
        hdus = fits.HDUList()
        hdr = fits.Header()
        hdr['covtype'] = model.covariance_type
        hdus.append(fits.ImageHDU(model.weights_, name='weights', header=hdr))
        hdus.append(fits.ImageHDU(model.means_, name='means'))
        hdus.append(fits.ImageHDU(model.covars_, name='covars'))
        hdus.writeto(filename, clobber=True)

    @staticmethod
    def load(filename):
        hdus = fits.open(filename, memmap=False)
        hdr = hdus[0].header
        covtype = hdr['covtype']
        model = GaussianMixtureModel(
            hdus['weights'].data, hdus['means'].data, hdus['covars'].data, covtype)
        hdus.close()
        return model

    def sample(self, n_samples=1, random_state=None):

        if self.covtype != 'full':
            return NotImplementedError(
                'covariance type "{0}" not implemented yet.'.format(self.covtype))

        # Code adapted from sklearn's GMM.sample()
        if random_state is None:
            random_state = np.random.RandomState()

        weight_cdf = np.cumsum(self.weights)
        X = np.empty((n_samples, self.n_dimensions))
        rand = random_state.rand(n_samples)
        # decide which component to use for each sample
        comps = weight_cdf.searchsorted(rand)
        # for each component, generate all needed samples
        for comp in range(self.n_components):
            # occurrences of current component in X
            comp_in_X = (comp == comps)
            # number of those occurrences
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                X[comp_in_X] = random_state.multivariate_normal(
                    self.means[comp], self.covars[comp], num_comp_in_X)
        return X


def sample_magnitudes(target_type, n_targets, random_state):
    """Sample magnitudes based on target type (i.e. LRG, ELG, QSO).

    Can sample multiple targets at once and needs only to be called
    once for each target_type.

    Parameters
    ----------
    target_type : str
        One of three object types (LRG, ELG, QSO).
    n_targets : int
        Number of sampled magntiudes to be returned for the specified
        target_type.


    Returns
    -------
    np.ndarray length n_targets
        Structured array with columns g,r,z of sampled magnitudes
    """

    #Path to model .fits files
    pathToModels = resource_filename('desitarget', "mock/data")

    #Load the mixture model for the specified target_type
    if target_type == 'LRG':
        model = GaussianMixtureModel.load(pathToModels + '/lrgMag_gmm.fits')
    elif target_type == 'ELG':
        model = GaussianMixtureModel.load(pathToModels + '/elgMag_gmm.fits')
    elif target_type == 'QSO':
        model = GaussianMixtureModel.load(pathToModels + '/qsoMag_gmm.fits')
    elif target_type == 'BGS':
        model = GaussianMixtureModel.load(pathToModels + '/bgsMag_gmm.fits')

    #Generate a sample of magnitudes of size n_targets
    mags = model.sample(n_samples=n_targets, random_state=random_state)

    samp = np.empty(n_targets, dtype=[('g', 'f8'), ('r', 'f8'), ('z', 'f8')])
    samp['g'] = mags[:,0]
    samp['r'] = mags[:,1]
    samp['z'] = mags[:,2]

    return samp
