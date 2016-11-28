# Licensed under a 4-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
===========================
desitarget.mock.selection
===========================

Applies selection criteria on mock target catalogs.
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import os, re
import desitarget.mock.io
import desitarget.io
from   desitarget import mws_mask, desi_mask, bgs_mask
import os
from   astropy.table import Table, Column
import fitsio
import desispec.brick
import warnings


def make_lookup_dict(bricks):
    """
    Creates lookup dictionary for a list of bricks.
    
    Parameters:
    -----------
    bricks (array): array of bricknames.

    Output:
    -------
    Lookup Dictionary. l['brickname'] returns a list with all the IDs in the
    input array bricks where bricks=='brickname'
    """
    l = {}
    unique_bricks = list(set(bricks))

    for b in unique_bricks:
        l[b] = []

    for i in range(len(bricks)):
        try:
            l[bricks[i]].append(i)
        except:
            l[bricks[i]] = i
    return l
############################################################
def mag_select(data, sourcename, targetname, truthname, brick_info=None, density_fluctuations=False, **kwargs):
    """
    Apply the selection function to determine the target class of each entry in
    the input catalog.

    Parameters:
    -----------
        data: dict
            Data required for selection
        brick_info: 
            Summary of depths and target fluctuations
        kwargs: dict
            Parameters of sample selection. The requirements will be different
            for different values of source name.

            Should include:

            mag_faintest:    float
                Hard faint limit for inclusion in survey.
            mag_faint_filler:  float
                Magintude fainter than which stars are considered filler, rather
                than part of the main sample.
            mag_bright:      float
                Hard bright limit for inclusion in survey.
    """
    # reference for SDSS values:  (Yuan, Liu, Xiang) https://arxiv.org/pdf/1301.1427v1.pdf
    # reference for other values: The Tractor https://github.com/dstndstn/tractor/blob/39f883c811f0a6b17a44db140d93d4268c6621a1/tractor/sfd.py
    extinctions = {
        'SDSSu': 4.239,
        'SDSSg': 3.30,
        'SDSSr': 2.31,
        'SDSSz': 1.29,
        'DESu': 3.995,
        'DESg': 3.214,
        'DESr': 2.165,
        'DESi': 1.592,
        'DESz': 1.211,
        'DESY': 1.064,
        'WISEW1': 0.184,
        'WISEW2': 0.113,
        'WISEW3': 0.0241,
        'WISEW4': 0.00910,
        }

    target_class = -1

    if(sourcename == 'STD_FSTAR'):
        """
        Apply the selection function to determine the target class of each entry in
        the input catalog.

        This implements standard F-star cuts from:
        https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection

        Optical colors near BD+17 4708
        (GRCOLOR - 0.32)^2 + (RZCOLOR - 0.13)^2 < 0.06^2

        To do:
        - Isolation criterion
        - DECam magnitudes rather than SDSS
        """
        # Parameters
        mag_faint  = kwargs['mag_faint']
        mag_bright = kwargs['mag_bright']
        grcolor    = kwargs['grcolor']
        rzcolor    = kwargs['rzcolor']
        colortol   = kwargs['colortol']

        SELECTION_MAG_NAME = 'SDSSr_obs'
        COLOR_G_NAME       = 'SDSSg_obs'
        COLOR_R_NAME       = 'SDSSr_obs'
        COLOR_Z_NAME       = 'SDSSz_obs'

        # Will populate this array with the bitmask values of each target class
        target_class = np.zeros(len(data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

        fainter_than_bright_limit  = data[SELECTION_MAG_NAME]  >= mag_bright
        brighter_than_faint_limit  = data[SELECTION_MAG_NAME]  <  mag_faint

        gmr            = data[COLOR_G_NAME] - data[COLOR_R_NAME]
        rmz            = data[COLOR_R_NAME] - data[COLOR_Z_NAME]

        select_color     = (gmr - grcolor)**2 + (rmz - rzcolor)**2 < colortol**2
        select_mag       = (fainter_than_bright_limit) & (brighter_than_faint_limit)
        select_std_stars = (select_color) & (select_mag)
        target_class[select_std_stars] = desi_mask.mask('STD_FSTAR')

    if(sourcename == 'MWS_MAIN'):
        mag_bright       = kwargs['mag_bright']
        mag_faintest     = kwargs['mag_faintest']
        mag_faint_filler = kwargs['mag_faint_filler']

        # Parameters
        SELECTION_MAG_NAME = 'SDSSr_obs'

        # Will populate this array with the bitmask values of each target class
        target_class = np.zeros(len(data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

        fainter_than_bright_limit  = data[SELECTION_MAG_NAME]  >= mag_bright
        fainter_than_filler_limit  = data[SELECTION_MAG_NAME]  >= mag_faint_filler
        brighter_than_filler_limit = data[SELECTION_MAG_NAME]  <  mag_faint_filler
        brighter_than_faint_limit  = data[SELECTION_MAG_NAME]  <  mag_faintest

        # MWS mocks enforce a 'hole' of radius 100pc around the sun to avoid
        # overlap with the WD100pc sample.
        DISTANCE_FROM_SUN_NAME         = 'd_helio'
        DISTANCE_FROM_SUN_TO_PC_FACTOR = 1000.0 # In Galaxia mocks, d_helio is in kpc
        DISTANCE_FROM_SUN_CUT          = 100.0/DISTANCE_FROM_SUN_TO_PC_FACTOR
        further_than_100pc             = data[DISTANCE_FROM_SUN_NAME] > DISTANCE_FROM_SUN_CUT

        # Main sample
        select_main_sample               = (fainter_than_bright_limit) & (brighter_than_filler_limit) & (further_than_100pc)
        target_class[select_main_sample] = mws_mask.mask('MWS_MAIN')

        # Faint sample
        select_faint_filler_sample               = (fainter_than_filler_limit) & (brighter_than_faint_limit) & (further_than_100pc)
        target_class[select_faint_filler_sample] = mws_mask.mask('MWS_MAIN_VERY_FAINT')

    if(sourcename == 'MWS_WD'):
        mag_bright = kwargs['mag_bright']
        mag_faint  = kwargs['mag_faint']

        # Parameters
        SELECTION_MAG_NAME = 'magg'

        # Will populate this array with the bitmask values of each target class
        target_class = np.zeros(len(data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

        fainter_than_bright_limit  = data[SELECTION_MAG_NAME]  >= mag_bright
        brighter_than_faint_limit  = data[SELECTION_MAG_NAME]  <  mag_faint
        is_wd                      = data['WD'] == 1

        # WD sample
        select_wd_sample               = (fainter_than_bright_limit) & (brighter_than_faint_limit) & (is_wd)
        target_class[select_wd_sample] = mws_mask.mask('MWS_WD')

        # Nearby ('100pc') sample -- everything in the input table that isn't a WD
        # Expect to refine this in future
        select_nearby_sample               = (fainter_than_bright_limit) & (brighter_than_faint_limit) & (np.invert(is_wd))
        target_class[select_nearby_sample] = mws_mask.mask('MWS_NEARBY')

    if(sourcename == 'BGS'):
        mag_bright = kwargs['mag_bright']
        mag_faintest = kwargs['mag_faintest']
        mag_priority_split = kwargs['mag_priority_split']
        ra = data['RA']
        dec = data['DEC']

        # Parameters
        SELECTION_MAG_NAME = 'SDSSr_true'
        DEPTH_MAG_NAME = 'GALDEPTH_R'

        # Will populate this array with the bitmask values of each target class
        target_class = np.zeros(len(data[SELECTION_MAG_NAME]),dtype=np.int64) - 1
        
        if density_fluctuations:
        #now have to loop over all bricks with some data
            bricks = desispec.brick.brickname(ra, dec)
            unique_bricks = list(set(bricks))
            
            i_brick = 0 
            n_brick = len(unique_bricks)

            # create dictionary with targets per brick
            lookup = make_lookup_dict(bricks)
            for brickname in unique_bricks:
                in_brick = np.array(lookup[brickname])
                i_brick += 1
#                print('brick {} out of {}'.format(i_brick,n_brick))                

                id_binfo  = (brick_info['BRICKNAME'] == brickname)
                if np.count_nonzero(id_binfo) != 1:
                    depth = 0.0
                    extinction = 99.0
                    warnings.warn("Tile is on the border. Extinction = 99.0. Depth = 0.0", RuntimeWarning)
                else:
                    depth = brick_info[DEPTH_MAG_NAME][id_binfo]
                    extinction = brick_info['EBV'][id_binfo] * extinctions['SDSSr']            
                # print('DEPTH {} Ext {}'.format(depth, extinction))

                tmp  = data[SELECTION_MAG_NAME][in_brick] + extinction
                brighter_than_depth        = tmp < depth
                fainter_than_bright_limit  = tmp >= mag_bright
                brighter_than_split_mag    = tmp < mag_priority_split
                fainter_than_split_mag     = tmp >= mag_priority_split
                brighter_than_faint_limit  = tmp  < mag_faintest
                
                # Bright sample
                select_bright_sample               = (fainter_than_bright_limit) & (brighter_than_split_mag) & (brighter_than_depth)
                target_class[in_brick[select_bright_sample]] = bgs_mask.mask('BGS_BRIGHT')
            
                # Faint sample
                select_faint_sample               = (fainter_than_split_mag) & (brighter_than_faint_limit) & (brighter_than_depth)
                target_class[in_brick[select_faint_sample]] = bgs_mask.mask('BGS_FAINT')
        else:
            fainter_than_bright_limit  = data[SELECTION_MAG_NAME]  >= mag_bright
            brighter_than_split_mag    = data[SELECTION_MAG_NAME]   < mag_priority_split
            fainter_than_split_mag     = data[SELECTION_MAG_NAME]  >= mag_priority_split
            brighter_than_faint_limit  = data[SELECTION_MAG_NAME]   < mag_faintest

        # Bright sample
            select_bright_sample               = (fainter_than_bright_limit) & (brighter_than_split_mag)
            target_class[select_bright_sample] = bgs_mask.mask('BGS_BRIGHT')

        # Faint sample
            select_faint_sample               = (fainter_than_split_mag) & (brighter_than_faint_limit)
            target_class[select_faint_sample] = bgs_mask.mask('BGS_FAINT')

    return target_class


def estimate_density(ra, dec, bounds=(170, 190, 0, 35)):
    """Estimate the number density from a small patch
    
    Args:
        ra: array_like
            An array with RA positions.
        dec: array_like
            An array with Dec positions.

    Options:
        bounds: (min_ra, max_ra, min_dec, max_dec) to use for density
            estimation, assuming complete coverage within those bounds [deg]

    Returns:
        density: float
           Object number density computed over a small patch.
    """
    print('ra max min {} {}'.format(ra.max(), ra.min()))
    print('dec max min {} {}'.format(dec.max(), dec.min()))
    density = 0.0 

    min_ra, max_ra, min_dec, max_dec = bounds
    footprint_area = (max_ra-min_ra) * (np.sin(max_dec*np.pi/180.) - np.sin(min_dec*np.pi/180.)) * 180 / np.pi

    n_in = np.count_nonzero((ra>=min_ra) & (ra<max_ra) & (dec>=min_dec) & (dec<max_dec))
    density = n_in/footprint_area
    if(n_in==0):
        density = 1E-6
    return density


def ndens_select(data, sourcename, targetname, truthname, brick_info = None, density_fluctuations = False, **kwargs):

    """Apply selection function based only on number density and redshift criteria.

    """

    ra = data['RA']
    dec = data['DEC']
    z = data['Z']
    
    if ('min_z' in kwargs) & ('max_z' in kwargs):
        in_z = ((z>=kwargs['min_z']) & (z<=kwargs['max_z']))
    else:
        in_z = z>=0.0

    # if we don't have a mean NTARGET in the input file, we fall back to constant mean values from the config file
    constant_density = False
    try:
        mean_density = brick_info['NTARGET_'+sourcename] 
    except:
        message = "Mean number density for target {} is constant and taken from input file: {}".format(sourcename, kwargs['density'])
        warnings.warn(message, RuntimeWarning)
        mean_density = kwargs['density']
        constant_density = True


    try:
        global_density  = kwargs['global_density']
    except:
        global_density = False

    n = len(ra)
    target_class = np.zeros(n,dtype=np.int64) - 1
    keepornot = np.random.uniform(0.,1.,n)

    if density_fluctuations and constant_density == False and global_density == False:
        bricks = desispec.brick.brickname(ra, dec)
        unique_bricks = list(set(bricks))
        n_brick = len(unique_bricks)

        i_brick = 0
        lookup = make_lookup_dict(bricks)
        for brickname in unique_bricks:
            in_brick = np.array(lookup[brickname])
            i_brick += 1
#            print('brick {} out of {}'.format(i_brick,n_brick))                

            n_in_brick = len(in_brick)

            #locate the brick info we need
            id_binfo  = (brick_info['BRICKNAME'] == brickname)

            if np.count_nonzero(id_binfo) != 1:
                num_density = 0.0
                brick_area = 1.0
                warnings.warn("Tile is on the border. NumDensity= 0.0", RuntimeWarning)
            else:
                brick_area = brick_info['BRICKAREA'][id_binfo]
                num_density = brick_info['FLUC_EBV'][sourcename][id_binfo]  * mean_density

            mock_dens = n_in_brick/brick_area
                                           
#            print('in brick mock density {} - desired num density {}'.format(mock_dens, num_density))
            frac_keep = num_density/mock_dens
            if(frac_keep>1.0):
                warnings.warn("target {}: frac_keep>1.0.: frac_keep={} ".format(sourcename, frac_keep), RuntimeWarning)
#                print('num density desired {}, num density in mock {}, frac_keep {} - {}'.format(num_density, mock_dens, frac_keep, n_in_brick))
            
            kept = (keepornot[in_brick] < frac_keep) & (in_z[in_brick])
            
 #           print('len kept {}'.format(np.count_nonzero(select_sample)))
            target_class[in_brick[kept]] = desi_mask.mask(targetname)
    else:
        print('No Fluctuations for this target {}'.format(sourcename))
        try:
            bounds = kwargs['min_ra'], kwargs['max_ra'], kwargs['min_dec'], kwargs['max_dec']
            mock_dens = estimate_density(ra[in_z], dec[in_z], bounds=bounds)
        except KeyError:
            mock_dens = estimate_density(ra[in_z], dec[in_z])

        num_density = mean_density

        print('mock density {} - desired num density {}'.format(mock_dens, num_density))
        frac_keep = num_density/mock_dens
        if(frac_keep>1.0):
            warnings.warn("target {} frac_keep>1.0.: frac_keep={} ".format(sourcename, frac_keep), RuntimeWarning)
#        print('num density desired {}, num density in mock {}, frac_keep {}'.format(num_density, mock_num_density, frac_keep))
        kept = keepornot < frac_keep

            
        select_sample = (in_z) & (kept)             
        target_class[select_sample] = desi_mask.mask(targetname)

    return target_class

