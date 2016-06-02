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
import desitarget.io as dtio
from desitarget import desi_mask
import os
from astropy.table import Table, Column
import desispec.brick

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

def build_mock(**kwargs):
    # Set desired number densities
    goal_density_qsoI = kwargs['qsoI_dens']
    goal_density_qsoII = kwargs['qsoII_dens']
    goal_density_qso_fake = kwargs['qso_fake_dens']
    goal_density_lrg = kwargs['lrg_dens']
    goal_density_lrg_fake = kwargs['lrg_fake_dens']
    goal_density_elg = kwargs['elg_dens']

    # read the mocks on disk
    qso_mock_ra, qso_mock_dec, qso_mock_z = dtio.read_mock_dark_time(kwargs['qso_file'])
    elg_mock_ra, elg_mock_dec, elg_mock_z = dtio.read_mock_dark_time(kwargs['elg_file'])
    lrg_mock_ra, lrg_mock_dec, lrg_mock_z = dtio.read_mock_dark_time(kwargs['lrg_file'])
    random_mock_ra, random_mock_dec, random_mock_z = dtio.read_mock_dark_time(kwargs['random_file'], read_z=False)

    # build lists for the different population types
    ra_list = [qso_mock_ra, qso_mock_ra, random_mock_ra, lrg_mock_ra, random_mock_ra, elg_mock_ra]
    dec_list = [qso_mock_dec, qso_mock_dec, random_mock_dec, lrg_mock_dec, random_mock_dec, elg_mock_dec]
    z_list = [qso_mock_z, qso_mock_z, random_mock_z, lrg_mock_z, random_mock_z, elg_mock_z]
    min_z_list  = [2.1, -1.0, -1.0, -1.0, -1.0, -1.0]
    max_z_list  = [1000.0, 2.1, 1000.0, 1000.0, 1000.0, 1000.0]
    goal_list = [goal_density_qsoI, goal_density_qsoII, goal_density_qso_fake, goal_density_lrg, goal_density_lrg_fake, goal_density_elg]
    true_type_list = ['QSO', 'QSO', 'STAR', 'GALAXY', 'UNKNOWN', 'GALAXY']
    desi_tf_list = [desi_mask.QSO, desi_mask.QSO, desi_mask.QSO, desi_mask.LRG, desi_mask.LRG, desi_mask.ELG]
    bgs_tf_list = [0,0,0,0,0,0]
    mws_tf_list = [0,0,0,0,0,0]

    # arrays for the full target and truth tables
    ra_total = np.empty(0)
    dec_total = np.empty(0)
    z_total = np.empty(0)
    desi_target_total = np.empty(0, dtype='i8')
    bgs_target_total = np.empty(0, dtype='i8')
    mws_target_total = np.empty(0, dtype='i8')
    true_type_total = np.empty(0, dtype='S10')

    # loop over the populations
    for ra, dec, z, min_z, max_z, goal, true_type, desi_tf, bgs_tf, mws_tf in\
            zip(ra_list, dec_list, z_list, min_z_list, max_z_list, goal_list,\
                    true_type_list, desi_tf_list, bgs_tf_list, mws_tf_list):

        # select subpopulation
        ra_, dec_, z_ ,desi_flag_, bgs_flag_, mws_flag_, true_type_ = \
            select_population(ra, dec, z, min_z=min_z, max_z=max_z,\
                              goal_density=goal,\
                              true_type=true_type,\
                              desi_target_flag = desi_tf,\
                              bgs_target_flag = bgs_tf,\
                              mws_target_flag = mws_tf)
        
        # append to the full list
        ra_total = np.append(ra_total, ra_)
        dec_total = np.append(dec_total, dec_)
        z_total = np.append(z_total, z_)
        desi_target_total = np.append(desi_target_total, desi_flag_)
        bgs_target_total = np.append(bgs_target_total, bgs_flag_)
        mws_target_total = np.append(mws_target_total, mws_flag_)
        true_type_total = np.append(true_type_total, true_type_)

    # make up the IDs, subpriorities and bricknames
    n = len(ra_total)
    targetid = np.random.randint(2**62, size=n)
    subprior = np.random.uniform(0., 1., size=n)
    brickname = desispec.brick.brickname(ra_total, dec_total)
    
    print('Total in targetid {}'.format(len(targetid)))
    print('Total in ra {}'.format(len(ra_total)))
    print('Total in dec {}'.format(len(dec_total)))
    print('Total in brickname {}'.format(len(brickname)))
    print('Total in desi {}'.format(len(desi_target_total)))
    print('Total in bgs {}'.format(len(bgs_target_total)))
    print('Total in mws {}'.format(len(mws_target_total)))


    # write the Targets to disk
    targets_filename = os.path.join(kwargs['output_dir'], 'targets.fits')
    targets = Table()
    targets['TARGETID'] = targetid
    targets['BRICKNAME'] = brickname
    targets['RA'] = ra_total
    targets['DEC'] = dec_total
    targets['DESI_TARGET'] = desi_target_total
    targets['BGS_TARGET'] = bgs_target_total
    targets['MWS_TARGET'] = mws_target_total
    targets['SUBPRIORITY'] = subprior
    targets.write(targets_filename, overwrite=True)
    
    # write the Truth to disk
    truth_filename  = os.path.join(kwargs['output_dir'], 'truth.fits')
    truth = Table()
    truth['TARGETID'] = targetid
    truth['BRICKNAME'] = brickname
    truth['RA'] = ra_total
    truth['DEC'] = dec_total
    truth['TRUEZ'] = z_total
    truth['TRUETYPE'] = true_type_total
    truth.write(truth_filename, overwrite=True)
    
    return
    

