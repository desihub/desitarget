# Licensed under a 4-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
===========================
desitarget.mock.mws_galaxia
===========================

Builds target/truth files from already existing mock data
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

############################################################
def mag_select(data, source_name, **kwargs):
    """
    Apply the selection function to determine the target class of each entry in
    the input catalog.

    Parameters:
    -----------
        data: dict
            Data required for selection

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
    target_class = -1

    if(source_name == 'STD_FSTAR'):
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

    if(source_name == 'MWS_MAIN'):
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

    if(source_name == 'MWS_WD'):
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

    if(source_name == 'BGS'):
        mag_bright = kwargs['mag_bright']
        mag_faintest = kwargs['mag_faintest']
        mag_priority_split = kwargs['mag_priority_split']

        # Parameters
        SELECTION_MAG_NAME = 'SDSSr_true'

        # Will populate this array with the bitmask values of each target class
        target_class = np.zeros(len(data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

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


def ndens_select(data, source_name, **kwargs):

    """Apply selection function based only on number density and redshift criteria.

    """

    ra = data['RA']
    dec = data['DEC']
    z = data['Z']
    
    if ('min_z' in kwargs) & ('max_z' in kwargs):
        in_z = ((z>=kwargs['min_z']) & (z<=kwargs['max_z']))
    else:
        in_z = z>0.0

    try:
        bounds = kwargs['min_ra'], kwargs['max_ra'], kwargs['min_dec'], kwargs['max_dec']
        mock_dens = estimate_density(ra[in_z], dec[in_z], bounds=bounds)
    except KeyError:
        mock_dens = estimate_density(ra[in_z], dec[in_z])

    frac_keep = min(kwargs['number_density']/mock_dens , 1.0)
    if mock_dens < kwargs['number_density']:
        print("WARNING: mock cannot achieve the goal density for source {} Goal {}. Mock {}".format(source_name, kwargs['number_density'], mock_dens))


    n = len(ra)
    keepornot = np.random.uniform(0.,1.,n)
    limit = np.zeros(n) + frac_keep
    kept = keepornot < limit
    select_sample = (in_z) & (kept)

    target_class = np.zeros(n,dtype=np.int64) - 1
    target_class[select_sample] = desi_mask.mask(source_name)

    return target_class

