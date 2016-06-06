# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
========================
desitarget.mock.darktime
========================

Builds target/truth files from already existing mock data
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import fitsio
import os, re
import desitarget.mock.io 
import desitarget.io
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
        min_z = float
            Minimum redshift to select from the input z.
        max_z = float
            Maximum redshift to select from the input z.
        true_type = string
            Desired label for this population.
        desi_target_flag = 64bit mask
            Kind of DESI target following desitarget.desi_mask
        bgs_target_flag = 64bit mask
            Kind of BGS target following desitarget.desi_mask
        mws_target_flag = 64bit mask
            Kind of MWS target following desitarget.desi_mask

    Returns:
        Dictionary with the following entries:
            'RA': array_like (float)
                Subset of input RA.
            'DEC': array_like (float)
                Subset of input Dec.
            'Z': array_like (float)
                Subset of input Z.
            'DESI_TARGET': array_like (int)
                Array of DESI target flags corresponding to the input desi_target_flag
            'BGS_TARGET': array_like (int)
                Array of BGS target flags corresponding to the input bgs_target_flag
            'MWS_TARGET': array_like (int)
                Array of MWS target flags corresponding to the input mws_target_flag
            'TRUE_TYPE': array_like (string)
                Array of true types corresponding to the input true_type.
    """

    ii = ((z>=kwargs['min_z']) & (z<=kwargs['max_z']))

    mock_dens = estimate_density(ra[ii], dec[ii])
    frac_keep = min(kwargs['goal_density']/mock_dens , 1.0)
    if mock_dens < kwargs['goal_density']:
        print("WARNING: mock cannot achieve the goal density. Goal {}. Mock {}".format(kwargs['goal_density'], mock_dens))


    ra_pop, dec_pop, z_pop = reduce(ra[ii], dec[ii], z[ii], frac_keep)
    n = len(ra_pop)

#    print("keeping total={} fraction={}".format(n, frac_keep))

    desi_target_pop  = np.zeros(n, dtype='i8'); desi_target_pop[:] = kwargs['desi_target_flag']
    bgs_target_pop = np.zeros(n, dtype='i8'); bgs_target_pop[:] = kwargs['bgs_target_flag']
    mws_target_pop = np.zeros(n, dtype='i8'); mws_target_pop[:] = kwargs['mws_target_flag']
    true_type_pop = np.zeros(n, dtype='S10'); true_type_pop[:] = kwargs['true_type']

    return {'RA':ra_pop, 'DEC':dec_pop, 'Z':z_pop, 
            'DESI_TARGET':desi_target_pop, 'BGS_TARGET': bgs_target_pop, 'MWS_TARGET':mws_target_pop, 'TRUE_TYPE':true_type_pop}

def build_mock_target(qsolya_dens=0.0, qsotracer_dens=0.0, qso_fake_dens=0.0, lrg_dens=0.0, lrg_fake_dens=0.0, elg_dens=0.0, elg_fake_dens=0.0,
                      mock_qso_file='', mock_lrg_file='', mock_elg_file='',mock_random_file='', output_dir='', rand_seed=42):
    """Builds a Target and Truth files from a series of mock files
    
    Args:
        qsolya_dens: float
           Desired number density for Lya QSOs.
        qsotracer_dens: float
           Desired number density for tracer QSOs.
        qso_fake_dens: float
           Desired number density for fake (contamination) QSOs.
        lrg_dens: float
           Desired number density for LRGs.
        lrg_fake_dens: float
           Desired number density for fake (contamination) LRGs.
        elg_dens: float
           Desired number density for ELGs.
        elg_fake_dens: float
           Desired number density for fake (contamination) ELGs.
        mock_qso_file: string
           Filename for the mock QSOs.
        mock_lrg_file: string
           Filename for the mock LRGss.
        mock_elg_file: string
           Filename for the mock ELGs.
        mock_random_file: string
           Filename for a random set of points.
        output_dir: string
           Path to write the outputs (targets.fits and truth.fits).
        rand_seed: int
           seed for random number generator
    """
    np.random.seed(seed=rand_seed)

    # read the mocks on disk
    qso_mock_ra, qso_mock_dec, qso_mock_z = desitarget.mock.io.read_mock_dark_time(mock_qso_file)
    elg_mock_ra, elg_mock_dec, elg_mock_z = desitarget.mock.io.read_mock_dark_time(mock_elg_file)
    lrg_mock_ra, lrg_mock_dec, lrg_mock_z = desitarget.mock.io.read_mock_dark_time(mock_lrg_file)
    random_mock_ra, random_mock_dec, random_mock_z = desitarget.mock.io.read_mock_dark_time(mock_random_file, read_z=False)

    # build lists for the different population types
    ra_list = [qso_mock_ra, qso_mock_ra, random_mock_ra, lrg_mock_ra, random_mock_ra, elg_mock_ra, elg_mock_ra]
    dec_list = [qso_mock_dec, qso_mock_dec, random_mock_dec, lrg_mock_dec, random_mock_dec, elg_mock_dec, elg_mock_dec]
    z_list = [qso_mock_z, qso_mock_z, random_mock_z, lrg_mock_z, random_mock_z, elg_mock_z, elg_mock_z]
    min_z_list  = [2.1, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    max_z_list  = [1000.0, 2.1, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    goal_list = [qsolya_dens, qsotracer_dens, qso_fake_dens, lrg_dens, lrg_fake_dens, elg_dens, elg_fake_dens]
    true_type_list = ['QSO', 'QSO', 'STAR', 'GALAXY', 'UNKNOWN', 'GALAXY', 'UNKNOWN']
    desi_tf_list = [desi_mask.QSO, desi_mask.QSO, desi_mask.QSO, desi_mask.LRG, desi_mask.LRG, desi_mask.ELG, desi_mask.ELG]
    bgs_tf_list = [0,0,0,0,0,0,0]
    mws_tf_list = [0,0,0,0,0,0,0]

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
        pop_dict =   select_population(ra, dec, z,\
                                           min_z=min_z,\
                                           max_z=max_z,\
                                           goal_density=goal,\
                                           true_type=true_type,\
                                           desi_target_flag = desi_tf,\
                                           bgs_target_flag = bgs_tf,\
                                           mws_target_flag = mws_tf)
        
        # append to the full list
        ra_total = np.append(ra_total, pop_dict['RA'])
        dec_total = np.append(dec_total, pop_dict['DEC'])
        z_total = np.append(z_total, pop_dict['Z'])
        desi_target_total = np.append(desi_target_total, pop_dict['DESI_TARGET'])
        bgs_target_total = np.append(bgs_target_total, pop_dict['BGS_TARGET'])
        mws_target_total = np.append(mws_target_total, pop_dict['MWS_TARGET'])
        true_type_total = np.append(true_type_total, pop_dict['TRUE_TYPE'])

    # make up the IDs, subpriorities and bricknames
    n = len(ra_total)
    targetid = np.random.randint(2**62, size=n)
    subprior = np.random.uniform(0., 1., size=n)
    brickname = desispec.brick.brickname(ra_total, dec_total)
    
    print('Total in targetid {}'.format(len(targetid)))
#    print('Total in ra {}'.format(len(ra_total)))
#    print('Total in dec {}'.format(len(dec_total)))
#    print('Total in brickname {}'.format(len(brickname)))
#    print('Total in desi {}'.format(len(desi_target_total)))
#    print('Total in bgs {}'.format(len(bgs_target_total)))
#    print('Total in mws {}'.format(len(mws_target_total)))

    targets_filename = os.path.join(output_dir, 'targets.fits')

    # write the Targets to disk
    type_table = [
        ('TARGETID', '>i4'),
        ('BRICKNAME', '|S8'),
        ('RA', '>f8'),
        ('DEC', '>f8'),
        ('DESI_TARGET', 'i8'),
        ('BGS_TARGET', 'i8'),
        ('MWS_TARGET', 'i8'),
        ('SUBPRIORITY', '>f8')
    ]

    targets = np.ndarray(shape=(n), dtype=type_table)
    targets['TARGETID'] = targetid
    targets['BRICKNAME'] = brickname
    targets['RA'] = ra_total
    targets['DEC'] = dec_total
    targets['DESI_TARGET'] = desi_target_total
    targets['BGS_TARGET'] = bgs_target_total
    targets['MWS_TARGET'] = mws_target_total
    targets['SUBPRIORITY'] = subprior

    desitarget.io.write_targets(targets_filename, targets, indir=output_dir)

    # write the Truth to disk
    truth_filename = os.path.join(output_dir, 'truth.fits')
    type_table = [
        ('TARGETID', '>i4'),
        ('BRICKNAME', '|S8'),
        ('RA', '>f8'),
        ('DEC', '>f8'),
        ('TRUEZ', '>f8'),
        ('TRUETYPE', '|S10')
    ]

    truth = np.ndarray(shape=(n), dtype=type_table)
    truth['TARGETID'] = targetid
    truth['BRICKNAME'] = brickname
    truth['RA'] = ra_total
    truth['DEC'] = dec_total
    truth['TRUEZ'] = z_total
    truth['TRUETYPE'] = true_type_total
    desitarget.io.write_targets(truth_filename, truth, indir=output_dir)
    
    return
    


def build_mock_sky_star(std_star_dens=0.0, sky_calib_dens=0.0, mock_random_file='', output_dir='', rand_seed=42):
    """Builds a Sky and StandardStar files from a series of mock files
    
    Args:
        std_star_dens: float
           Desired number density for starndar stars.
        sky_calib_dens: float
           Desired number density for sky calibration locations.
        mock_random_file: string
           Filename for a random set of points.
        output_dir: string
           Path to write the outputs (targets.fits and truth.fits).
        rand_seed: int
           seed for random number generator
    """
    np.random.seed(seed=rand_seed)

    # Set desired number densities
    goal_density_std_star = std_star_dens
    goal_density_sky = sky_calib_dens

    # read the mock on disk
    random_mock_ra, random_mock_dec, random_mock_z = desitarget.mock.io.read_mock_dark_time(mock_random_file, read_z=False)

    true_type_list = ['STAR', 'SKY']
    goal_density_list = [goal_density_std_star, goal_density_sky]
    desi_target_list  = [desi_mask.STD_FSTAR, desi_mask.SKY]
    filename_list = ['stdstar.fits', 'sky.fits']

    for true_type, goal_density, desi_target_flag, filename in\
            zip(true_type_list, goal_density_list, desi_target_list, filename_list):
        pop_dict = select_population(random_mock_ra, random_mock_dec, random_mock_z,\
                                         min_z=-1.0,\
                                         max_z=100,\
                                         goal_density=goal_density,\
                                         true_type=true_type,\
                                         desi_target_flag = desi_target_flag,\
                                         bgs_target_flag = 0,\
                                         mws_target_flag = 0)
        
        # make up the IDs, subpriorities and bricknames
        n = len(pop_dict['RA'])
        targetid = np.random.randint(2**62, size=n)
        subprior = np.random.uniform(0., 1., size=n)
        brickname = desispec.brick.brickname(pop_dict['RA'], pop_dict['DEC'])
    
        print('Total in targetid {}'.format(len(targetid)))

        # write the targets to disk
        targets_filename = os.path.join(output_dir, filename)
        type_table = [
            ('TARGETID', '>i4'),
            ('BRICKNAME', '|S8'),
            ('RA', '>f8'),
            ('DEC', '>f8'),
            ('DESI_TARGET', 'i8'),
            ('BGS_TARGET', 'i8'),
            ('MWS_TARGET', 'i8'),
            ('SUBPRIORITY', '>f8')
            ]

        targets = np.ndarray(shape=(n), dtype=type_table)
        targets['TARGETID'] = targetid
        targets['BRICKNAME'] = brickname
        targets['RA'] = pop_dict['RA']
        targets['DEC'] = pop_dict['DEC']
        targets['DESI_TARGET'] = pop_dict['DESI_TARGET']
        targets['BGS_TARGET'] = pop_dict['BGS_TARGET']
        targets['MWS_TARGET'] = pop_dict['MWS_TARGET']
        targets['SUBPRIORITY'] = subprior
        desitarget.io.write_targets(targets_filename, targets, indir=output_dir)
        
    return
    

