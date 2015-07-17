"""
desitarget.cuts
===================

Utility functions to perform target selection based on 
color cuts on a dataset
"""

from __future__ import absolute_import, division

import numpy as np

def select_target(target_id, g_mags, r_mags, z_mags, target_type=""):
    """
    args:
    target_id : 1D integer array of unique target IDs associated to the magnitudes.
    g_mags : 1D magnitudes in the g-band
    r_mags : 1D magnitudes in the r-band
    z_mags : 1D magnitudes in the z-band
    
    returns 1D numpy array of a subset from target_id.
    
    We compute the conditions that correspond to LRG/ELG/QSO/BGS 
    target selection following the criteria described here:
    
    https://desi.lbl.gov/trac/wiki/TargetSelection
    
    NOTE: requirements on WISE bands are not included yet

    """

    target_types = ["ELG", "LRG", "QSO", "BGS"]


    target_true = np.empty(0, dtype='int')
    if(target_type in target_types):
        if(target_type=="LRG"):
            target_true =  np.where((r_mags < 23.0) & (z_mags < 20.56) & ((r_mags-z_mags)>1.6))

        elif (target_type=="ELG"):
            target_true = np.where((r_mags < 23.4) & 
                                   ((r_mags - z_mags)>0.3) & 
                                   ((r_mags - z_mags)<1.5) & 
                                   ((g_mags - r_mags)<(r_mags - z_mags - 0.2)) & 
                                   ((g_mags - r_mags)< 1.2 - (r_mags - z_mags)))
            
    else:
         raise RuntimeError("Target type %s not recognized"%(target_type))
        


    return target_true
        


    
