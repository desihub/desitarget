"""
desitarget.mock.sample
======================

Samples magnitudes and shapes for LRG, ELG, QSO, and BGS targets from a Gaussian mixture model.

The model for each object type is fit to DR2 targets that have passed target
selection critera.
"""

from __future__ import print_function, division

from desiutil.log import get_logger
log = get_logger()

class SampleGMM(object):
    """Sample magnitudes based on target type (i.e. LRG, ELG, QSO, BGS).

    Can sample multiple targets at once and needs only to be called
    once for each target_type.

    Args:
      target_type (str) : One of four object types (LRG, ELG, QSO, BGS).
      n_targets (int) : Number of sampled magntiudes to be returned for the
        specified target_type.
    random_state: RandomState or an int seed.  A random number generator.

    Returns: np.ndarray length n_targets :
        Structured array with columns g, r, z, w1, w2, w3, w4, exp_r, exp_e1,
        exp_e2, dev_r, dev_e1, dev_e2 of sampled magnitudes.

    """
    def __init__(self, random_state=None):
        from pkg_resources import resource_filename
        from desiutil.sklearn import GaussianMixtureModel

        bgsfile = resource_filename('desitarget', 'mock/data/bgs_gmm.fits')
        elgfile = resource_filename('desitarget', 'mock/data/elg_gmm.fits')
        lrgfile = resource_filename('desitarget', 'mock/data/lrg_gmm.fits')
        qsofile = resource_filename('desitarget', 'mock/data/qso_gmm.fits')

        self.bgsmodel = GaussianMixtureModel.load(bgsfile)
        self.elgmodel = GaussianMixtureModel.load(elgfile)
        self.lrgmodel = GaussianMixtureModel.load(lrgfile)
        self.qsomodel = GaussianMixtureModel.load(qsofile)

        self.random_state = random_state

    def sample(self, target_type='LRG', n_targets=1):
        import numpy as np

        if target_type not in ('BGS', 'ELG', 'LRG', 'QSO'):
            log.fatal('Unknown object type {}!'.format(target_type))
            raise ValueError

        # Generate a sample of magnitudes/shapes of size n_targets.
        if target_type == 'BGS':
            params = self.bgsmodel.sample(n_targets, self.random_state).astype('f4')
        elif target_type == 'ELG':
            params = self.elgmodel.sample(n_targets, self.random_state).astype('f4')
        elif target_type == 'LRG':
            params = self.lrgmodel.sample(n_targets, self.random_state).astype('f4')
        elif target_type == 'QSO':
            params = self.qsomodel.sample(n_targets, self.random_state).astype('f4')

        tags = ('g', 'r', 'z', 'w1', 'w2', 'w3', 'w4')
        if target_type != 'QSO':
            tags = tags + ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2')

        samp = np.empty( n_targets, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]
            
        return samp

def sample_mag_shape(target_type, n_targets, random_state=None):
    """Sample magnitudes and shapes based on target type (i.e. LRG, ELG, QSO, BGS).

    Can sample multiple targets at once and needs only to be called
    once for each target_type.

    Parameters
    ----------
    target_type : str
        One of four object types (LRG, ELG, QSO, BGS).
    n_targets : int
        Number of sampled magntiudes and shapes to be returned for the specified
        target_type.
    random_state: RandomState or an int seed
        A random number generator.


    Returns
    -------
    np.ndarray length n_targets
        Structured array with columns g,r,z,w1,w2,w3,w4,exp_r,exp_e1, exp_e2,
        dev_r, dev_e1, dev_e2 of sampled magnitudes and shapes. Note that
        target_type='QSO' only returns magnitudes.
    """
    import numpy as np
    from pkg_resources import resource_filename
    from desiutil.sklearn import GaussianMixtureModel

    #Path to model .fits files
    pathToModels = resource_filename('desitarget', "mock/data")

    #Load the mixture model for the specified target_type
    if target_type == 'LRG':
        model = GaussianMixtureModel.load(pathToModels + '/lrg_gmm.fits')
    elif target_type == 'ELG':
        model = GaussianMixtureModel.load(pathToModels + '/elg_gmm.fits')
    elif target_type == 'QSO':
        model = GaussianMixtureModel.load(pathToModels + '/qso_gmm.fits')
    elif target_type == 'BGS':
        model = GaussianMixtureModel.load(pathToModels + '/bgs_gmm.fits')

    #Generate a sample of magnitudes of size n_targets
    params = model.sample(n_samples=n_targets, random_state=random_state)

    if target_type == 'QSO':

        samp = np.empty(n_targets, dtype=[('g', 'f4'), ('r', 'f4'), ('z', 'f4'),
                                          ('w1', 'f4'), ('w2', 'f4'), ('w3', 'f4'),
                                          ('w4', 'f4')])

        samp['g'] = params[:,0]
        samp['r'] = params[:,1]
        samp['z'] = params[:,2]
        samp['w1'] = params[:,3]
        samp['w2'] = params[:,4]
        samp['w3'] = params[:,5]
        samp['w4'] = params[:,6]

    else:

        samp = np.empty(n_targets, dtype=[('g', 'f4'), ('r', 'f4'), ('z', 'f4'), ('w1', 'f4'),
                                          ('w2', 'f4'), ('w3', 'f4'), ('w4', 'f4'),
                                          ('exp_r', 'f4'), ('exp_e1', 'f4'), ('exp_e2', 'f4'),
                                          ('dev_r', 'f4'), ('dev_e1', 'f4'), ('dev_e2', 'f4')])

        samp['g'] = params[:,0]
        samp['r'] = params[:,1]
        samp['z'] = params[:,2]
        samp['w1'] = params[:,3]
        samp['w2'] = params[:,4]
        samp['w3'] = params[:,5]
        samp['w4'] = params[:,6]
        samp['exp_r'] = params[:,7]
        samp['exp_e1'] = params[:,8]
        samp['exp_e2'] = params[:,9]
        samp['dev_r'] = params[:,10]
        samp['dev_e1'] = params[:,11]
        samp['dev_e2'] = params[:,12]

    return samp
