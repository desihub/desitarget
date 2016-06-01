# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
========
mocks.io
========

Makes mock target catalogs
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import fitsio
import os, re
from . import __version__ as desitarget_version
from . import gitversion

def estimate_density(ra, dec):
    """Estimate the number density from a small patch
    
    Args:
        ra: array_like
            An array with RA positions.
        dec: array_like
            An array with Dec positions.

    Returns:
        density: float
           Object number density computed over a small patch.
    """
    density = 0.0 

    footprint_area = 20. * 45.* np.sin(45. * np.pi/180.)/(45. * np.pi/180.)
    smalldata = ra[(ra>170.) & (dec<190.) & (dec>0.) & (dec<45.)]
    n_in = len(smalldata)
    density = n_in/footprint_area

    return density

def reduce(ra, dec, z, frac):
    xra = np.array(ra)
    xdec = np.array(dec)
    xzz = np.array(z)
   
    keepornot = np.random.uniform(0.,1.,len(ra))
    limit = np.zeros(len(xra)) + frac
    #create boolean array of which to keep
    #find new length
    kept = keepornot < limit
    yra = xra[kept]
    ydec = xdec[kept]
    yzz = xzz[kept]
    
    return((yra,ydec,yzz))


def select_population(ra, dec, z, goal_density=0.0, min_z=-1.0, max_z=1000.0, true_type='GALAXY', 
                      desi_target_flag=0, bgs_target_flag=0, mws_target_flag=0):
    ii = ((z>=min_z) & (z<=max_z))

    mock_dens = estimate_density(ra[ii], dec[ii])
    frac_keep = min(goal_density/mock_dens , 1.0)
    if mock_dens < goal_density:
        print("WARNING: mock cannot achieve the goal density. Goal {}. Mock {}".format(goal_density, mock_dens))


    ra_pop, dec_pop, z_pop = reduce(ra[ii], dec[ii], z[ii], frac_keep)
    n = len(ra_pop)


    print("keeping total={} fraction={}".format(n, frac_keep))

    desi_target_pop  = np.zeros(n, dtype='i8'); desi_target_pop[:] = desi_target_flag
    bgs_target_pop = np.zeros(n, dtype='i8'); bgs_target_pop[:] = bgs_target_flag
    mws_target_pop = np.zeros(n, dtype='i8'); mws_target_pop[:] = mws_target_flag
    true_type_pop = np.zeros(n, dtype='S10'); true_type_pop[:] = true_type
    sub_prior_pop = np.random.uniform(0, 1, size=n)

    return ((ra_pop, dec_pop, z_pop, desi_target_pop, bgs_target_pop, mws_target_pop, true_type_pop, sub_prior_pop))

