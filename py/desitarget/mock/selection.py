# Licensed under a 4-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
===========================
desitarget.mock.selection
===========================

Applies selection criteria on mock target catalogs.
"""
from __future__ import (absolute_import, division)


import os
import re
import warnings

import numpy as np
from astropy.table import Table, Column
import fitsio

import desitarget.io
import desitarget.mock.io
import desispec.brick
from desitarget import mws_mask, desi_mask, bgs_mask

def make_lookup_dict(bricks):
    """
    Creates lookup dictionary for a list of bricks.
    
    Parameters:
    -----------
    bricks (array): array of bricknames.

    Output:
    -------
    Lookup Dictionary. lookup['brickname'] returns a list with all the indices
    in the input array bricks where bricks=='brickname'
    """
    lookup = {}
    unique_bricks = list(set(bricks))

    for b in unique_bricks:
        lookup[b] = []

    for i in range(len(bricks)):
        try:
            lookup[bricks[i]].append(i)
        except:
            lookup[bricks[i]] = i
    return lookup

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

    if (sourcename == 'STD_FSTAR'):
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
 
        SELECTION_MAG_NAME = 'DECAMr_obs'
        COLOR_G_NAME       = 'DECAMg_obs'
        COLOR_R_NAME       = 'DECAMr_obs'
        COLOR_Z_NAME       = 'DECAMz_obs'

        # Will populate this array with the bitmask values of each target class
        n = len(data[SELECTION_MAG_NAME])
        target_class = np.zeros(n,dtype=np.int64) - 1

        fainter_than_bright_limit  = data[SELECTION_MAG_NAME]  >= mag_bright
        brighter_than_faint_limit  = data[SELECTION_MAG_NAME]  <  mag_faint

        gmr            = data[COLOR_G_NAME] - data[COLOR_R_NAME]
        rmz            = data[COLOR_R_NAME] - data[COLOR_Z_NAME]

        select_color     = (gmr - grcolor)**2 + (rmz - rzcolor)**2 < colortol**2
        select_mag       = (fainter_than_bright_limit) & (brighter_than_faint_limit)
        select_std_stars = (select_color) & (select_mag)
        target_class[select_std_stars] = desi_mask.mask('STD_FSTAR')

    if (sourcename == 'MWS_MAIN'):
        mag_bright       = kwargs['mag_bright']
        mag_faintest     = kwargs['mag_faintest']
        mag_faint_filler = kwargs['mag_faint_filler']

        # Parameters
        SELECTION_MAG_NAME = 'DECAMr_obs'

        # Will populate this array with the bitmask values of each target class
        n = len(data[SELECTION_MAG_NAME])
        target_class = np.zeros(n, dtype=np.int64) - 1

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


        select_main_sample               = (fainter_than_bright_limit) & (brighter_than_filler_limit) & (further_than_100pc)
        select_faint_filler_sample               = (fainter_than_filler_limit) & (brighter_than_faint_limit) & (further_than_100pc)

        if ('density' in kwargs): #this forces downsampling on the whole sample
            keepornot = np.random.uniform(0.,1.,n)

            mock_area = 0.0
            bricks = desispec.brick.brickname(data['RA'], data['DEC'])
            unique_bricks = list(set(bricks))
        
            for brickname in unique_bricks:
                id_binfo  = (brick_info['BRICKNAME'] == brickname)
                if np.count_nonzero(id_binfo) == 1:
                    mock_area += brick_info['BRICKAREA'][id_binfo]
                
            mock_dens = (np.count_nonzero(select_main_sample) + np.count_nonzero(select_faint_filler_sample))/mock_area
            num_density = kwargs['density']

            frac_keep = num_density/mock_dens
            print('This mock is being downsampled')
            print('mock area {} mock density {} - desired num density {}'.format(mock_area, mock_dens, num_density))

            if(frac_keep>1.0):
                warnings.warn("target {} frac_keep>1.0.: frac_keep={} ".format(sourcename, frac_keep), RuntimeWarning)

            kept = keepornot < frac_keep

            select_main_sample               &= kept
            select_faint_filler_sample       &= kept


        target_class[select_main_sample] = mws_mask.mask('MWS_MAIN')
        target_class[select_faint_filler_sample] = mws_mask.mask('MWS_MAIN_VERY_FAINT')
            
    if (sourcename == 'MWS_WD'):
        mag_bright = kwargs['mag_bright']
        mag_faint  = kwargs['mag_faint']

        # Parameters
        SELECTION_MAG_NAME = 'g_sdss'

        # Will populate this array with the bitmask values of each target class
        target_class = np.zeros(len(data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

        fainter_than_bright_limit  = data[SELECTION_MAG_NAME]  >= mag_bright
        brighter_than_faint_limit  = data[SELECTION_MAG_NAME]  <  mag_faint


        # WD sample
        select_wd_sample               = (fainter_than_bright_limit) & (brighter_than_faint_limit)
        target_class[select_wd_sample] = mws_mask.mask('MWS_WD')

    if (sourcename == 'MWS_NEARBY'):
        mag_bright = kwargs['mag_bright']
        mag_faint  = kwargs['mag_faint']

        # Parameters
        SELECTION_MAG_NAME = 'magg'

        # Will populate this array with the bitmask values of each target class
        target_class = np.zeros(len(data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

        fainter_than_bright_limit  = data[SELECTION_MAG_NAME]  >= mag_bright
        brighter_than_faint_limit  = data[SELECTION_MAG_NAME]  <  mag_faint

        select_nearby_sample               = (fainter_than_bright_limit) & (brighter_than_faint_limit)
        target_class[select_nearby_sample] = mws_mask.mask('MWS_NEARBY')

    if (sourcename == 'BGS'):
        mag_bright = kwargs['mag_bright']
        mag_faintest = kwargs['mag_faintest']
        mag_priority_split = kwargs['mag_priority_split']
        ra = data['RA']
        dec = data['DEC']

        # Parameters
        SELECTION_MAG_NAME = 'DECAMr_true'
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
                    extinction = brick_info['EBV'][id_binfo] * extinctions['DESr']            
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

def ndens_select(data, sourcename, targetname, truthname, brick_info = None, density_fluctuations = False, **kwargs):

    """Apply selection function based only on number density and redshift criteria.

    """

    ra = data['RA']
    dec = data['DEC']
    z = data['Z']

    bricks = desispec.brick.brickname(ra, dec)
    unique_bricks = list(set(bricks))
    n_brick = len(unique_bricks)

    
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
        #compute the whole mock area

        mock_area = 0.0        
        i_brick = 0
        for brickname in unique_bricks:
            i_brick += 1
#            print('{} out of {}'.format(i_brick, n_brick))
            id_binfo  = (brick_info['BRICKNAME'] == brickname)
            if np.count_nonzero(id_binfo) == 1:
                mock_area += brick_info['BRICKAREA'][id_binfo]
                
        mock_dens = len(ra)/mock_area
        num_density = mean_density

        print('mock area {} mock density {} - desired num density {}'.format(mock_area, mock_dens, num_density))
        frac_keep = num_density/mock_dens
        if(frac_keep>1.0):
            warnings.warn("target {} frac_keep>1.0.: frac_keep={} ".format(sourcename, frac_keep), RuntimeWarning)
#        print('num density desired {}, num density in mock {}, frac_keep {}'.format(num_density, mock_num_density, frac_keep))
        kept = keepornot < frac_keep
            
        select_sample = (in_z) & (kept)             
        target_class[select_sample] = desi_mask.mask(targetname)

    return target_class

class SelectTargets(object):
    """Select various types of targets.  Most of this functionality is taken from
    desitarget.cuts but that code has not been factored in a way that is
    convenient at this time.

    """
    def __init__(self):
        from desitarget import desi_mask, bgs_mask, mws_mask, obsconditions
        self.desi_mask = desi_mask
        self.bgs_mask = bgs_mask
        self.mws_mask = mws_mask
        self.obsconditions = obsconditions
        
        self.decam_extcoeff = (3.995, 3.214, 2.165, 1.592, 1.211, 1.064) # extinction coefficients
        self.wise_extcoeff = (0.184, 0.113, 0.0241, 0.00910)
        self.sdss_extcoeff = (4.239, 3.303, 2.285, 1.698, 1.263)

    def bgs_select(self, targets, truth=None):
        """Select BGS targets.  Note that obsconditions for BGS_ANY are set to BRIGHT
        only. Is this what we want?

        """
        from desitarget.cuts import isBGS_bright, isBGS_faint

        rflux = targets['DECAM_FLUX'][..., 2]

        # Select BGS_BRIGHT targets.
        bgs_bright = isBGS_bright(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT_SOUTH
        targets['DESI_TARGET'] |= (bgs_bright != 0) * self.desi_mask.BGS_ANY
        for oo in self.bgs_mask.BGS_BRIGHT.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(oo)

        # Select BGS_FAINT targets.
        bgs_faint = isBGS_faint(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT_SOUTH
        targets['DESI_TARGET'] |= (bgs_faint != 0) * self.desi_mask.BGS_ANY
        for oo in self.bgs_mask.BGS_FAINT.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(oo)
            
        return targets

    def elg_select(self, targets, truth=None):
        """Select ELG targets."""
        from desitarget.cuts import isELG

        gflux = targets['DECAM_FLUX'][..., 1]
        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)

        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        for oo in self.desi_mask.ELG.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(oo)
            
        return targets

    def lrg_select(self, targets, truth=None):
        """Select LRG targets."""
        from desitarget.cuts import isLRG

        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        w1flux = targets['WISE_FLUX'][..., 0]
        lrg = isLRG(rflux=rflux, zflux=zflux, w1flux=w1flux)

        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH
        for oo in self.desi_mask.LRG.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(oo)
            
        return targets

    def mws_main_select(self, targets, truth=None):
        """Select MWS_MAIN, MWS_MAIN_VERY_FAINT, STD_FSTAR, and STD_BRIGHT targets.  The
        selection eventually will be done with Gaia (I think).

        """    
        from desitarget.cuts import isFSTD

        def _isMWS_MAIN(rflux):
            """A function like this should be in desitarget.cuts. Select 15<r<19 stars.""" 
            main = rflux > 10**((22.5-19.0)/2.5)
            main &= rflux <= 10**((22.5-15.0)/2.5)
            return main
        
        def _isMWS_MAIN_VERY_FAINT(rflux):
            """A function like this should be in desitarget.cuts. Select 19<r<20 filler stars."""
            faint = rflux > 10**((22.5-20.0)/2.5)
            faint &= rflux <= 10**((22.5-19.0)/2.5)
            return faint
            
        gflux = targets['DECAM_FLUX'][..., 1]
        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        obs_rflux = rflux * 10**(-0.4 * targets['EBV'] * self.decam_extcoeff[2]) # attenuate for dust

        snr = np.zeros_like(targets['DECAM_FLUX']) + 100      # Hack -- fixed S/N
        fracflux = np.zeros_like(targets['DECAM_FLUX']).T     # No contamination from neighbors.
        objtype = np.repeat('PSF', len(targets)).astype('U3') # Right data type?!?

        # Select MWS_MAIN targets.
        mws_main = _isMWS_MAIN(rflux=rflux)
        targets['MWS_TARGET'] |= (mws_main != 0) * self.mws_mask.mask('MWS_MAIN')
        targets['DESI_TARGET'] |= (mws_main != 0) * self.desi_mask.MWS_ANY
        for oo in self.mws_mask.MWS_MAIN.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (mws_main != 0) * self.obsconditions.mask(oo)

        # Select MWS_MAIN_VERY_FAINT targets.
        mws_very_faint = _isMWS_MAIN_VERY_FAINT(rflux=rflux)
        targets['MWS_TARGET'] |= (mws_very_faint != 0) * self.mws_mask.mask('MWS_MAIN_VERY_FAINT')
        targets['DESI_TARGET'] |= (mws_very_faint != 0) * self.desi_mask.MWS_ANY
        for oo in self.mws_mask.MWS_MAIN_VERY_FAINT.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (mws_very_faint != 0) * self.obsconditions.mask(oo)

        # Select dark-time FSTD targets.
        fstd = isFSTD(gflux=gflux, rflux=rflux, zflux=zflux, objtype=objtype,
                      decam_fracflux=fracflux, decam_snr=snr, obs_rflux=obs_rflux)
        targets['DESI_TARGET'] |= (fstd != 0) * self.desi_mask.STD_FSTAR
        for oo in self.desi_mask.STD_FSTAR.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (fstd != 0) * self.obsconditions.mask(oo)

        # Select bright-time FSTD targets.
        fstd_bright = isFSTD(gflux=gflux, rflux=rflux, zflux=zflux, objtype=objtype,
                             decam_fracflux=fracflux, decam_snr=snr, obs_rflux=obs_rflux,
                             bright=True)
        targets['DESI_TARGET'] |= (fstd_bright != 0) * self.desi_mask.STD_BRIGHT
        for oo in self.desi_mask.STD_BRIGHT.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (fstd_bright != 0) * self.obsconditions.mask(oo)

        return targets

    def mws_nearby_select(self, targets, truth=None):
        """Select MWS_NEARBY targets.  The selection eventually will be done with Gaia,
        so for now just do a "perfect" selection.

        """    
        mws_nearby = np.ones(len(targets)) # select everything!
        #mws_nearby = (truth['MAG'] <= 20.0) * 1 # SDSS g-band!

        targets['MWS_TARGET'] |= (mws_nearby != 0) * self.mws_mask.mask('MWS_NEARBY')
        targets['DESI_TARGET'] |= (mws_nearby != 0) * self.desi_mask.MWS_ANY
        for oo in self.mws_mask.MWS_NEARBY.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (mws_nearby != 0) * self.obsconditions.mask(oo)
        
        return targets

    def mws_wd_select(self, targets, truth=None):
        """Select MWS_WD and STD_WD targets.  The selection eventually will be done with
        Gaia, so for now just do a "perfect" selection here.

        """    
        #mws_wd = np.ones(len(targets)) # select everything!
        mws_wd = ((truth['MAG'] >= 15.0) * (truth['MAG'] <= 20.0)) * 1 # SDSS g-band!

        targets['MWS_TARGET'] |= (mws_wd != 0) * self.mws_mask.mask('MWS_WD')
        targets['DESI_TARGET'] |= (mws_wd != 0) * self.desi_mask.MWS_ANY
        for oo in self.mws_mask.MWS_WD.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (mws_wd != 0) * self.obsconditions.mask(oo)

        # Select STD_WD; cut just on g-band magnitude (not TEMPLATESUBTYPE!)
        std_wd = (truth['MAG'] <= 19.0) * 1 # SDSS g-band!
        targets['DESI_TARGET'] |= (std_wd !=0) * self.desi_mask.mask('STD_WD')
        for oo in self.desi_mask.STD_WD.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (std_wd != 0) * self.obsconditions.mask(oo)
        
        return targets

    def qso_select(self, targets, truth=None):
        """Select QSO targets.  Unfortunately we can't apply the appropriate color-cuts
        because our spectra don't go red enough (i.e., into the WISE bands).  So
        all the QSOs pass for now.

        """
        if False:
            from desitarget.cuts import isQSO
        
            gflux = targets['DECAM_FLUX'][..., 1]
            rflux = targets['DECAM_FLUX'][..., 2]
            zflux = targets['DECAM_FLUX'][..., 4]
            w1flux = targets['WISE_FLUX'][..., 0]
            w2flux = targets['WISE_FLUX'][..., 1]
            qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                               w1flux=w1flux, w2flux=w2flux)
        else:
            qso = np.ones(len(targets)) # select everything!

        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH
        for oo in self.desi_mask.QSO.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (qso != 0) * self.obsconditions.mask(oo)

        return targets

    def sky_select(self, targets, truth=None):
        """Select SKY targets."""

        targets['DESI_TARGET'] |= self.desi_mask.mask('SKY')
        for oo in self.desi_mask.SKY.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= self.obsconditions.mask(oo)
            
        return targets
