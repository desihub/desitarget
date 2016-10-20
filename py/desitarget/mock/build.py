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

        print('type: {} format: {}'.format(source_name, source_format))
        function = 'read_'+source_format
        result = getattr(mockio, function)(source_path, source_name)
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
        if source_name in ['ELG', 'LRG', 'QSO']:
            desi_target = target_mask[ii]
        if source_name in ['BGS']:
            bgs_target = target_mask[ii]
        if source_name in ['MWS_MAIN', 'MWS_WD']:
            mws_target = target_mask[ii]


        # define names that go into Truth
        n = len(source_data['RA'][ii])
        true_type = np.zeros(n, dtype='S10'); true_type[:] = source_name
        true_subtype = np.zeros(n, dtype='S10'); true_subtype[:] = source_name

        #define obsconditions
        source_obsconditions = np.ones(n,dtype='uint16')
        if source_name in ['ELG', 'LRG', 'QSO']:
            source_obsconditions[:] = obsconditions.DARK
        if source_name in ['BGS']:
            source_obsconditions[:] = obsconditions.BRIGHT
        if source_name in ['MWS_MAIN', 'MWS_WD']:
            source_obsconditions[:] = obsconditions.BRIGHT

        #append to the arrays that will go into Targets
        ra_total = np.append(ra_total, source_data['RA'][ii])
        dec_total = np.append(dec_total, source_data['DEC'][ii])
        z_total = np.append(z_total, source_data['Z'][ii])
        desi_target_total = np.append(desi_target_total, desi_target)
        bgs_target_total = np.append(bgs_target_total, bgs_target)
        mws_target_total = np.append(mws_target_total, mws_target)
        true_type_total = np.append(true_type_total, true_type)
        true_subtype_total = np.append(true_subtype_total, true_subtype)
        obsconditions_total = np.append(obsconditions_total, source_obsconditions)

        print('sub', len(source_data['RA'][ii]), len(source_data['RA']))

    # create unique IDs, subpriorities and bricknames across all mock files
    n = len(ra_total)
    print('Great total of {} targets'.format(n))
    targetid = np.random.randint(2**62, size=n)
    subprior = np.random.uniform(0., 1., size=n)
    brickname = desispec.brick.brickname(ra_total, dec_total)


    # write the Targets to disk
    print('Started writing Targets file')
    targets_filename = os.path.join(params['output_dir'], 'targets.fits')
    targets = Table()
    targets['TARGETID'] = targetid
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
    truth['TARGETID'] = targetid
    truth['BRICKNAME'] = brickname
    truth['RA'] = ra_total
    truth['DEC'] = dec_total
    truth['TRUEZ'] = z_total
    truth['TRUETYPE'] = true_type_total
    truth['TRUESUBTYPE'] = true_subtype_total
    truth.write(truth_filename, overwrite=True)
    print('Finished writing Truth file')
