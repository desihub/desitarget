# Licensed under a 4-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*- 

"""
============================
desitarget.mock.fluctuations
============================

Generates density and depth fluctuations
"""

import numpy as np
import desispec.brick

def density_across_brick(ra, dec, mean_density):    
    """
    Computes the number density for mocks across bricks.
    Input:
    
    Parameters
    ---------
    ra : :class:`numpy.ndarray`
    dec: :class: `numpy.ndarray`
    mean_density: dictionary with mean density across different 
         target classes. 
         It includes at least the following keys
         ['ELG','LRG','TRACERQSO','LYAQSO','BGS','MWS'] 
    Returns
    -------
    Dictionary with the brick density at the location of the (ra,dec) input.
        The dictionary includes at least the following keys
        ['DENSITY_ELG','DENSITY_TRACERQSO', 'DENSITY_LYAQSO','DENSITY_BGS', 'DENSITY_MWS', 'DENSITY_LRG']

    """
    
    bricknames = desispec.brick.brickname(ra, dec)
    individual_bricknames = np.array(list(set(bricknames)))

    n_brick = len(individual_bricknames)
    n_obj = len(ra)

    density_lrg = np.ones(n_obj)
    density_elg = np.ones(n_obj)
    density_qso = np.ones(n_obj)
    density_lya = np.ones(n_obj)
    density_bgs = np.ones(n_obj)
    density_mws = np.ones(n_obj)

    # fluctuation factor
    ff  = 10**np.random.normal(scale=0.3, size=n_brick)    
    density_elg_brick = mean_density['ELG'] * ff
    density_lrg_brick = mean_density['LRG'] * ff
    density_qso_brick = mean_density['TRACERQSO'] * ff
    density_lya_brick = mean_density['LYAQSO'] * ff
    density_bgs_brick = mean_density['BGS'] * ff
    density_mws_brick = mean_density['MWS'] * ff
    
    
    for i in range(n_brick):        
        ii = (bricknames == individual_bricknames[i])
        density_elg[ii] = density_elg_brick[i]
        density_qso[ii] = density_qso_brick[i]
        density_lya[ii] = density_lya_brick[i]
        density_bgs[ii] = density_bgs_brick[i]
        density_mws[ii] = density_mws_brick[i]
        density_lrg[ii] = density_lrg_brick[i]

    return {'DENSITY_ELG': density_elg, 'DENSITY_TRACERQSO' : density_qso, 
            'DENSITY_LYAQSO': density_lya, 'DENSITY_BGS' : density_bgs,
            'DENSITY_MWS': density_mws, 'DENSITY_LRG': density_lrg}
