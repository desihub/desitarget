"""
desitarget.mocks
===================

Utility functions to perform target selection on mock data.
"""


from __future__ import absolute_import, division

import numpy as np
import cuts 
import h5py
import sys
import numpy as np
from astropy.io import fits
import os


def load_light_cone_durham(filename):
    """
    Args:
    filename: filename of the hdf5 file storing lightconedata.
    
    Returns:
    target_id: 1D numpy array, array of unique target IDs associated to the magnitudes. 
    ra : 1D numpy array, Right Ascension
    dec: 1D numpy array, declination
    g_mags: 1D numpy array, magnitudes in g band.
    r_mags: 1D numpy array, magnitudes in g band.
    z_mags: 1D numpy array, magnitudes in g band.
    returns 1D numpy array of a subset from target_id.
    
    """

    try:
        fin = h5py.File(mockfile, "r") 
        data = fin.require_group('/Data') 
        ra = data['ra'].value                                                                                    
        dec = data['dec'].value                                                                                  
        gal_id_string = data['GalaxyID'].value # these are string values, not integers!                               
        g_mags = data['appDgo_tot_ext'].value                                                                        
        r_mags = data['appDro_tot_ext'].value                                                                        
        z_mags = data['appDzo_tot_ext'].value  
        n_gals = 0
        n_gals = ra_data.size
        target_id = np.arange(n_gals)
    except Exception, e:
        import traceback
        print 'ERROR in loadlightconedurham'
        traceback.print_exc()
        raise e
    return target_id, ra, dec, g_mags, r_mags, z_mags, 

