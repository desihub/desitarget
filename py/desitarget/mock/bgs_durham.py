# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
===========================
desitarget.mock.bgs_durham
===========================

Builds target/truth files from already existing mock data
"""

from __future__ import (absolute_import, division)
#
import numpy as np
import os, re
import desitarget.mock.io 
import desitarget.io
from   desitarget import desi_mask
from   desitarget import bgs_mask
import os
from   astropy.table import Table, Column
import astropy.io.fits
import astropy.io.fits.convenience
import desispec.brick

############################################################
def bgs_selection(data, mag_faintest=20.0, mag_priority_split=19.5, mag_bright=15.0):
    """
    Apply the selection function to determine the target class of each entry in
    the input catalog.

    Parameters:
    -----------
        data: dict
            Data required for selection
        mag_faintest:    float
            Hard faint limit for inclusion in survey.
        mag_priority_split:  float
            Magintude fainter than which galaxies have lower priority
        mag_bright:      float 
            Hard bright limit for inclusion in survey.
    """
    # Parameters
    SELECTION_MAG_NAME = 'SDSSr_true'

    # Will populate this array with the bitmask values of each target class
    target_class = np.zeros(len(data[SELECTION_MAG_NAME]),dtype=np.int64) - 1
    priority     = np.zeros(len(data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

    fainter_than_bright_limit  = data[SELECTION_MAG_NAME]  >= mag_bright
    brighter_than_split_mag    = data[SELECTION_MAG_NAME]   < mag_priority_split
    fainter_than_split_mag     = data[SELECTION_MAG_NAME]  >= mag_priority_split
    brighter_than_faint_limit  = data[SELECTION_MAG_NAME]   < mag_faintest

    # Bright sample
    select_bright_sample               = (fainter_than_bright_limit) & (brighter_than_split_mag)
    target_class[select_bright_sample] = bgs_mask.bitnum('BGS_BRIGHT')
    priority[select_bright_sample]     = bgs_mask['BGS_BRIGHT'].priorities['UNOBS']

    # Nearby ('100pc') sample -- everything in the input table that isn't a WD
    # Expect to refine this in future
    select_faint_sample               = (fainter_than_split_mag) & (brighter_than_faint_limit)
    target_class[select_faint_sample] = bgs_mask.bitnum('BGS_FAINT')
    priority[select_faint_sample]     = bgs_mask['BGS_FAINT'].priorities['UNOBS']

    return target_class, priority

############################################################
def build_mock_target(mock_dir='', output_dir='', 
                      targets_name='bgs_durahm_mxxl_targets.fits',
                      truth_name='bgs_durahm_mxxl_truth.fits',
                      selection_name='bgs_durahm_mxxl_selection.fits',
                      mag_faintest=20.0, mag_priority_split=19.5, mag_bright=15.0, 
                      rand_seed=42):
                      
    """Builds a Target and Truth files from a series of mock files.
    
    Parameters:
    ----------    
        rand_seed: int
            seed for random number generator. 
    """
    np.random.seed(seed=rand_seed)

    # Read the mocks on disk. This returns a dict.
    # FIXME should just use table here too?
    data, file_list = desitarget.mock.io.read_mock_bgs_mxxl_brighttime(root_mock_bgs_mxxl_dir=mock_dir)
    data_keys       = list(data.keys())
    n               = len(data[data_keys[0]])
    
    # Allocate target classes and priorities
    target_class, priority = bgs_selection(data,
                                           mag_faintest       = mag_faintest,
                                           mag_priority_split = mag_priority_split,
                                           mag_bright         = mag_bright)
    # Identify all targets
    in_target_list = target_class >= 0
    ii             = np.where(in_target_list)[0]
    n_selected     = len(ii)

    print("Properties read from mock: {}".format(data.keys()))
    print("n_items in full catalog: {}".format(n))

    print 'Selection criteria:'
    for criterion in ['mag_faintest','mag_priority_split','mag_bright']:
        print(" -- {:30s} {}".format(criterion,locals().get(criterion)))

    print("n_items after selection: {}".format(n_selected))
        
    # targetid  = np.random.randint(2**62, size=n_selected)

    # Targetids are row numbers in original input
    # targetid  = np.arange(0,n)[ii]

    # Targetids are brick/object combos from original input
    BRICKID_FACTOR           = 1e10 # Max 10^10 objects per brick
    combined_brick_object_id = data['brickid'][ii]*BRICKID_FACTOR + data['objid'][ii]
    targetid                 = np.asarray(combined_brick_object_id,dtype=np.int64)

    # Assign random subpriorities for now
    subprior  = np.random.uniform(0., 1., size=n_selected)

    # Assign DESI-standard bricknames
    # CAREFUL: These are not the bricknames used by the input catalog!
    brickname = desispec.brick.brickname(data['RA'][ii], data['DEC'][ii])

    # assign target flags and true types
    desi_target_pop   = np.zeros(n_selected, dtype='i8')
    bgs_target_pop    = np.zeros(n_selected, dtype='i8') 
    mws_target_pop    = np.zeros(n_selected, dtype='i8') 
    bgs_target_pop[:] = target_class[ii]

    # APC This is a string? FIXME
    true_type_pop         = np.zeros(n_selected, dtype='S10')
    unique_target_classes = np.unique(target_class[ii])
    for tc in unique_target_classes:
        tc_name = bgs_mask.bitname(tc)
        has_this_target_class                = np.where(target_class[ii] == tc)[0]
        true_type_pop[has_this_target_class] = tc_name 
        print('Target class %s (%d): %d'%(tc_name,tc,len(has_this_target_class)))

    # Write the Targets to disk
    targets_filename = os.path.join(output_dir, targets_name)

    targets = Table()
    targets['TARGETID']    = targetid
    targets['BRICKNAME']   = brickname
    targets['RA']          = data['RA'][ii]
    targets['DEC']         = data['DEC'][ii]
    targets['DESI_TARGET'] = desi_target_pop
    targets['BGS_TARGET']  = bgs_target_pop
    targets['MWS_TARGET']  = mws_target_pop
    targets['NUMOBS_MORE'] = 1
    targets['PRIORITY']    = priority[ii]
    targets['SUBPRIORITY'] = subprior
    targets.write(targets_filename, overwrite=True)

    # Write the Truth to disk
    truth_filename = os.path.join(output_dir, truth_name)
    truth = Table()
    truth['TARGETID']  = targetid
    truth['BRICKNAME'] = brickname
    truth['RA']        = data['RA'][ii]
    truth['DEC']       = data['DEC'][ii]
    truth['TRUEZ']     = data['Z'][ii]

    # True type is just targeted type for now.
    truth['TRUETYPE']  = true_type_pop
    truth.write(truth_filename, overwrite=True)

    # Write the selection data to disk
    selection_filename = os.path.join(output_dir, selection_name)
    selection          = Table()
    selection['ROW']   = data['rownum'][ii]
    selection['FILE']  = data['filenum'][ii]
    selection.write(selection_filename, overwrite=True)
    # Append an extension to the selection file with the file list
    hdulist            = astropy.io.fits.open(selection_filename,mode='append')
    file_list_table    = Table()
    file_list_table['FILENAME'] = file_list
    file_list_hdu               = astropy.io.fits.BinTableHDU.from_columns(np.array(file_list_table))
    # file_list_hdu             = astropy.io.fits.convenience.table_to_hdu(file_list_table)
    hdulist.append(file_list_hdu)
    hdulist.writeto(selection_filename,clobber=True)

    return
    
