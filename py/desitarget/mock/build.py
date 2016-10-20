from __future__ import (absolute_import, division, print_function)

import os
import importlib
import time
import glob
import numpy as np

from   copy import copy

import astropy.io.fits as astropy_fitsio
from   astropy.table import Table, Column
import astropy.table

from   desitarget.targetmask import desi_mask
from   desitarget.mock.io    import decode_rownum_filenum
import desitarget.mock.io as mockio
import desitarget.mock.selection as mockselect
from desitarget import obsconditions
import desispec.brick

############################################################
def targets_truth(params):
    """
    Write

    Args:
        sources:    dict of source definitions.
        output_dir: location for intermediate mtl files.
        reset:      If True, force all intermediate TL files to be remade.

    Returns:
        targets:    
        truth:      

    """

    truth_all       = list()
    source_data_all = dict()
    target_mask_all = dict()

    # prints info about what we will be loading
    source_defs = params['sources']
    print('The following populations and paths are specified:')
    for source_name in sorted(source_defs.keys()):
        source_format = params['sources'][source_name]['format']
        source_path = params['sources'][source_name]['root_mock_dir']
        print('type: {}\n format: {} \n path: {}'.format(source_name, source_format, source_path))

    # load all the mocks
    for source_name in sorted(source_defs.keys()):
        source_format = params['sources'][source_name]['format']
        source_path = params['sources'][source_name]['root_mock_dir']
        source_dict = params['sources'][source_name]


        print('type: {} format: {}'.format(source_name, source_format))
        function = 'read_'+source_format
        if 'mock_name' in source_dict.keys():
            mock_name = source_dict['mock_name']
        else:
            mock_name = None
        result = getattr(mockio, function)(source_path, source_name, mock_name=mock_name)
        source_data_all[source_name] = result

    print('loaded {} mock sources'.format(len(source_data_all)))

    print('Making target selection')
    # runs target selection on every mock
    for source_name in sorted(source_defs.keys()):
        source_selection = params['sources'][source_name]['selection']
        source_dict = params['sources'][source_name]
        source_data = source_data_all[source_name]

        print('type: {} select: {}'.format(source_name, source_selection))
        selection_function = source_selection + '_select'
        result = getattr(mockselect, selection_function.lower())(source_data, source_name, **source_dict)
        target_mask_all[source_name] = result
        
    # consolidates all relevant arrays across mocks
    ra_total = np.empty(0)
    dec_total = np.empty(0)
    z_total = np.empty(0)
    desi_target_total = np.empty(0, dtype='i8')
    bgs_target_total = np.empty(0, dtype='i8')
    mws_target_total = np.empty(0, dtype='i8')
    true_type_total = np.empty(0, dtype='S10')
    true_subtype_total = np.empty(0, dtype='S10')
    obsconditions_total = np.empty(0, dtype='uint16')


    print('Collects information across mock files')
    for source_name in sorted(source_defs.keys()):
        source_data = source_data_all[source_name]
        target_mask = target_mask_all[source_name]

        ii = target_mask >-1 #only targets that passed cuts

        #define all flags
        desi_target = 0 * target_mask[ii]
        bgs_target = 0 * target_mask[ii] 
        mws_target = 0 * target_mask[ii]
        if source_name in ['ELG', 'LRG', 'QSO', 'STD_FSTAR', 'SKY']:
            desi_target = target_mask[ii]
        if source_name in ['BGS']:
            bgs_target = target_mask[ii]
        if source_name in ['MWS_MAIN', 'MWS_WD']:
            mws_target = target_mask[ii]


        # define names that go into Truth
        n = len(source_data['RA'][ii])
        if source_name not in ['STD_FSTAR', 'SKY']:
            true_type = np.zeros(n, dtype='S10'); true_type[:] = source_name
            true_subtype = np.zeros(n, dtype='S10');true_subtype[:] = source_name

                
        #define obsconditions
        source_obsconditions = np.ones(n,dtype='uint16')
        if source_name in ['ELG', 'LRG', 'QSO']:
            source_obsconditions[:] = obsconditions.DARK
        if source_name in ['BGS']:
            source_obsconditions[:] = obsconditions.BRIGHT
        if source_name in ['MWS_MAIN', 'MWS_WD']:
            source_obsconditions[:] = obsconditions.BRIGHT
        if source_name in ['STD_FSTAR', 'SKY']:
            source_obsconditions[:] = obsconditions.DARK|obsconditions.GRAY|obsconditions.BRIGHT 

        #append to the arrays that will go into Targets
        if source_name in ['STD_FSTAR']:
            ra_stars = source_data['RA'][ii].copy()
            dec_stars = source_data['DEC'][ii].copy()
            desi_target_stars = desi_target.copy()
            bgs_target_stars = bgs_target.copy()
            mws_target_stars = mws_target.copy()
            obsconditions_stars = source_obsconditions.copy()
        if source_name in ['SKY']:
            ra_sky = source_data['RA'][ii].copy()
            dec_sky = source_data['DEC'][ii].copy()
            desi_target_sky = desi_target.copy()
            bgs_target_sky = bgs_target.copy()
            mws_target_sky = mws_target.copy()
            obsconditions_sky = source_obsconditions.copy()
        if source_name not in ['SKY', 'STD_FSTAR']:
            ra_total = np.append(ra_total, source_data['RA'][ii])
            dec_total = np.append(dec_total, source_data['DEC'][ii])
            z_total = np.append(z_total, source_data['Z'][ii])
            desi_target_total = np.append(desi_target_total, desi_target)
            bgs_target_total = np.append(bgs_target_total, bgs_target)
            mws_target_total = np.append(mws_target_total, mws_target)
            true_type_total = np.append(true_type_total, true_type)
            true_subtype_total = np.append(true_subtype_total, true_subtype)
            obsconditions_total = np.append(obsconditions_total, source_obsconditions)

            

        print('{}: selected {} out of {}'.format(source_name, len(source_data['RA'][ii]), len(source_data['RA'])))

    # create unique IDs, subpriorities and bricknames across all mock files

    n_target = len(ra_total)     
    n_star = 0
    n_sky = 0
    n  = n_target    
    if 'STD_FSTAR' in source_defs.keys():
        n_star = len(ra_stars)
        n += n_star
    if 'SKY' in source_defs.keys():
        n_sky = len(ra_sky)
        n += n_sky

    print('Great total of {} targets {} stdstars {} sky pos'.format(n_target, n_star, n_sky))
    targetid = np.random.randint(2**62, size=n)


    if 'STD_FSTAR' in source_defs.keys():
        subprior = np.random.uniform(0., 1., size=n_star)
        brickname = desispec.brick.brickname(ra_stars, dec_stars)
        #write the Std Stars to disk
        print('Started writing StdStars file')
        stars_filename = os.path.join(params['output_dir'], 'stdstars.fits')
        stars = Table()
        stars['TARGETID'] = targetid[n_target:n_target+n_star]
        stars['BRICKNAME'] = brickname
        stars['RA'] = ra_stars
        stars['DEC'] = dec_stars
        stars['DESI_TARGET'] = desi_target_stars
        stars['BGS_TARGET'] = bgs_target_stars
        stars['MWS_TARGET'] = mws_target_stars
        stars['SUBPRIORITY'] = subprior
        stars['OBSCONDITIONS'] = obsconditions_stars
        stars.write(stars_filename, overwrite=True)
        print('Finished writing stdstars file')

    if 'SKY' in source_defs.keys():
        subprior = np.random.uniform(0., 1., size=n_sky)
        brickname = desispec.brick.brickname(ra_sky, dec_sky)
        #write the Std Stars to disk
        print('Started writing sky to file')
        sky_filename = os.path.join(params['output_dir'], 'sky.fits')
        sky = Table()
        sky['TARGETID'] = targetid[n_target+n_star:n_target+n_star+n_sky]
        sky['BRICKNAME'] = brickname
        sky['RA'] = ra_sky
        sky['DEC'] = dec_sky
        sky['DESI_TARGET'] = desi_target_sky
        sky['BGS_TARGET'] = bgs_target_sky
        sky['MWS_TARGET'] = mws_target_sky
        sky['SUBPRIORITY'] = subprior
        sky['OBSCONDITIONS'] = obsconditions_sky
        sky.write(sky_filename, overwrite=True)
        print('Finished writing sky file')

    if n_target > 0:
        subprior = np.random.uniform(0., 1., size=n_target)
        brickname = desispec.brick.brickname(ra_total, dec_total)

    # write the Targets to disk
        print('Started writing Targets file')
        targets_filename = os.path.join(params['output_dir'], 'targets.fits')
        targets = Table()
        targets['TARGETID'] = targetid[0:n_target]
        targets['BRICKNAME'] = brickname
        targets['RA'] = ra_total
        targets['DEC'] = dec_total
        targets['DESI_TARGET'] = desi_target_total
        targets['BGS_TARGET'] = bgs_target_total
        targets['MWS_TARGET'] = mws_target_total
        targets['SUBPRIORITY'] = subprior
        targets['OBSCONDITIONS'] = obsconditions_total
        targets.write(targets_filename, overwrite=True)
        print('Finished writing Targets file')

    # write the Truth to disk
        print('Started writing Truth file')
        truth_filename = os.path.join(params['output_dir'], 'truth.fits')
        truth = Table()
        truth['TARGETID'] = targetid[0:n_target]
        truth['BRICKNAME'] = brickname
        truth['RA'] = ra_total
        truth['DEC'] = dec_total
        truth['TRUEZ'] = z_total
        truth['TRUETYPE'] = true_type_total
        truth['TRUESUBTYPE'] = true_subtype_total
        truth.write(truth_filename, overwrite=True)
        print('Finished writing Truth file')
        
        

