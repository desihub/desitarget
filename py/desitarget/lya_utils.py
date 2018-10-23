import numpy as np
from scipy.interpolate import interp2d
from astropy.table import Table
from desitarget.targetmask import desi_mask
from pkg_resources import resource_filename

# This script is intended as a collection of utilities
# to prioritize ly-alpha targets.

def load_weights():
    """ Convenience function to load the weights for the
    ``cosmological weight`` of Ly-alpha QSOs. Weights were
    provided by A. Font-Ribera.
    """

    table = Table.read(resource_filename('desitarget','data/quasarvalue.txt'), format='ascii')
    z_col, r_col, w_col = table.columns[0], table.columns[1], table.columns[2]

    z_vec = np.unique(z_col)
    z_edges = np.linspace(2.025, 4.025, len(z_vec) + 1)
    assert np.allclose(z_vec, 0.5 * (z_edges[1:] + z_edges[:-1]))

    r_vec = np.unique(r_col)
    r_edges = np.linspace(18.05, 23.05, len(r_vec) + 1)
    assert np.allclose(r_vec, 0.5 * (r_edges[1:] + r_edges[:-1]))

    W = np.empty((len(r_vec), len(z_vec)))
    k = 0
    for j in range(len(z_vec)):
        for i in range(len(r_vec))[::-1]:
            assert r_col[k] == r_vec[i]
            assert z_col[k] == z_vec[j]
            W[i, j] = w_col[k]
            k += 1
    return W, r_edges, r_vec, z_edges, z_vec

def qso_weight(redshift,rmag):
    """
    Function that sets up and evaluates the weight using a 2D bi-linear spline

    Args:
    -----
    redshift: array (n,) with the redshift of the quasars that we want to get a value for.
    rmag: array (n,) with the r-band magnitude of the quasars that we want to get a value for.

    Returns:
    --------
    value: array of len(redshift) with the value of the quasars that we are interested in.
    """
    W, r_edges, r_vec, z_edges, z_vec = load_weights()
    wgt = interp2d(z_vec, r_vec, W, fill_value=0) # If out of the region of interest 0 weight
    try:
        assert(redshift.shape == rmag.shape)
    except:
        ValueError('redshift and rmag should have the same shape')
    try:
        # The 2D spline returns a (n,n) array and reorders the input monotonically
        # To avoid that we loop over the magnitudes and redshifts
        return np.array([wgt(redshift[i], rmag[i])[0] for i in range(len(redshift))])
    except TypeError: # If redshift and rmag are both scalars the above will fail
        return wgt(redshift, rmag)[0]

def lya_priority(redshift,rmag,prob=None):
    """
    Function that prioritizes lya targets using their cosmological value

    Args:
    -----
    redshift: array (n,) with the redshift of the quasars to prioritize.
    rmag: array (n,) with the r-band magnitude of the quasars to prioritize.
    prob: array (n,) NOT IMPLEMENTED: This is a placeholder to add some probabilities
    (of being a quasar or having certain redshift or both) to the calculation as weights.

    Returns:
    --------
    priorities: array (n,) of integers setting the observation priorities.
    """
    if prob is None:
        prob = np.ones(len(redshift))
    vqso = qso_weight(redshift,rmag)
    value = prob*vqso
    if np.max(value) > 0:
        value = value/np.max(value)
    else:
        value = np.zeros(len(redshift))
    min_priority = desi_mask['QSO'].priorities['UNOBS']
    max_priority = desi_mask['QSO'].priorities['MORE_ZGOOD']
    priorities = (min_priority+(max_priority-min_priority)*value).astype(int)
    return priorities
