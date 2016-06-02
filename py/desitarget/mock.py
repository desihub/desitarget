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
    """Reduces the size of input RA, DEC, Z arrays.
    
    Args:
        ra: array_like
            An array with RA positions.
        dec: array_like
            An array with Dec positions.
        z: array_like
            An array with redshifts
        frac: float
           Fraction of input arrays to be kept.

    Returns:
        ra_kept: array_like
             Subset of input RA.
        dec_kept: array_like
             Subset of input Dec.
        z_kept: array_like
             Subset of input Z.

    """
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


def select_population(ra, dec, z, **kwargs):

    """Selects points in RA, Dec, Z to assign them a target population.
    
    Args:
        ra: array_like
            An array with RA positions.
        dec: array_like
            An array with Dec positions.
        z: array_like
            An array with redshifts


    **kwargs:
        goal_density: float
            Number density (n/deg^2) desired for this set of points.
goal_density=0.0, 
min_z=-1.0, max_z=1000.0, true_type='UNKNOWN', 
                      desi_target_flag=0, bgs_target_flag=0, mws_target_flag=0):            

    Returns:
        ra_pop: array_like (float)
             Subset of input RA.
        dec_pop: array_like (float)
             Subset of input Dec.
        z_pop: array_like (float)
             Subset of input Z.
        desi_target_pop: array_like (int)
             Array of DESI target flags corresponding to the input desi_target_flag
        bgs_target_pop: array_like (int)
             Array of BGS target flags corresponding to the input bgs_target_flag
        mws_target_pop: array_like (int)
             Array of MWS target flags corresponding to the input mws_target_flag
        true_type_pop: array_like (string)
             Array of true types corresponding to the input true_type.
    """

    ii = ((z>=kwargs['min_z']) & (z<=kwargs['max_z']))

    mock_dens = estimate_density(ra[ii], dec[ii])
    frac_keep = min(kwargs['goal_density']/mock_dens , 1.0)
    if mock_dens < kwargs['goal_density']:
        print("WARNING: mock cannot achieve the goal density. Goal {}. Mock {}".format(kwargs['goal_density'], mock_dens))


    ra_pop, dec_pop, z_pop = reduce(ra[ii], dec[ii], z[ii], frac_keep)
    n = len(ra_pop)


    print("keeping total={} fraction={}".format(n, frac_keep))

    desi_target_pop  = np.zeros(n, dtype='i8'); desi_target_pop[:] = kwargs['desi_target_flag']
    bgs_target_pop = np.zeros(n, dtype='i8'); bgs_target_pop[:] = kwargs['bgs_target_flag']
    mws_target_pop = np.zeros(n, dtype='i8'); mws_target_pop[:] = kwargs['mws_target_flag']
    true_type_pop = np.zeros(n, dtype='S10'); true_type_pop[:] = kwargs['true_type']

    return ((ra_pop, dec_pop, z_pop, desi_target_pop, bgs_target_pop, mws_target_pop, true_type_pop))

