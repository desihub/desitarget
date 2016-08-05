# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
from   desitarget import desi_mask
from   desitarget import mws_mask
import os
from   astropy.table import Table, Column
import astropy.io.fits
import astropy.io.fits.convenience
import desispec.brick
    
############################################################
def mws_selection(mws_data, mag_faintest=20.0, mag_faint_filler=19.0, mag_bright=15.0):
    """
    Apply the selection function to determine the target class of each entry in
    the input catalog.

    Parameters:
    -----------
        mws_data: dict
            Data required for selection
        mag_faintest:    float
            Hard faint limit for inclusion in survey.
        mag_faint_filler:  float
            Magintude fainter than which stars are considered filler, rather
            than part of the main sample.
        mag_bright:      float 
            Hard bright limit for inclusion in survey.
    """
    # Parameters
    SELECTION_MAG_NAME = 'SDSSr_obs'

    # Will populate this array with the bitmask values of each target class
    target_class = np.zeros(len(mws_data[SELECTION_MAG_NAME]),dtype=np.int64) - 1
    priority     = np.zeros(len(mws_data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

    fainter_than_bright_limit  = mws_data[SELECTION_MAG_NAME]  >= mag_bright
    fainter_than_filler_limit  = mws_data[SELECTION_MAG_NAME]  >= mag_faint_filler
    brighter_than_filler_limit = mws_data[SELECTION_MAG_NAME]  <  mag_faint_filler
    brighter_than_faint_limit  = mws_data[SELECTION_MAG_NAME]  <  mag_faintest

    # Main sample
    select_main_sample               = (fainter_than_bright_limit) & (brighter_than_filler_limit)    
    target_class[select_main_sample] = mws_mask.bitnum('MWS_MAIN')
    priority[select_main_sample]     = mws_mask['MWS_MAIN'].priorities['UNOBS']

    # Faint sample
    select_faint_filler_sample               = (fainter_than_filler_limit) & (brighter_than_faint_limit)    
    target_class[select_faint_filler_sample] = mws_mask.bitnum('MWS_MAIN_VERY_FAINT')
    priority[select_faint_filler_sample]     = mws_mask['MWS_MAIN_VERY_FAINT'].priorities['UNOBS']

    return target_class, priority

############################################################
def build_mock_target(root_mock_mws_dir='', output_dir='',
                      targets_name='mws_galaxia_targets.fits',
                      truth_name='mws_galaxia_truth.fits',
                      selection_name='mws_galaxia_selection.fits',
                      mag_faintest = 20.0, mag_faint_filler=19.0, mag_bright=15.0, 
                      rand_seed=42,brickname_list=None):
                      
    """Builds Target and Truth files from a series of mock files.
    
    Parameters:
        mock_mws_dir: string
            Root path to a hierarchy of Galaxia mock files in arbitrary bricks
        output_dir: string
            Path to write the outputs (targets.fits and truth.fits).
        mag_bright: float
            Upper limit cut in SDSSr observed magnitude. Default = 15.0.
        mag_faint_filler: float
            Step between filler targets and main sample. Default = 19.0
        mag_faintest: float
            Lower limit cut in SDSSr observed magnitude. Default = 20.0.
        rand_seed: int
            seed for random number generator.

    Note:
        This routine assigns targetIDs that encode the mapping of each row in the
        target outputfile to a filenumber and row in the set of mock input files.
        This targetID will be further modified when all target lists are merged.
    """
    np.random.seed(seed=rand_seed)

    # Read the mocks on disk. This returns a dict.
    # FIXME should just use table here too?
    mws_data, file_list = desitarget.mock.io.read_mock_mws_brighttime(root_mock_mws_dir=root_mock_mws_dir,
                                                                      brickname_list=brickname_list)
    mws_keys = list(mws_data.keys())
    n        = len(mws_data[mws_keys[0]])
    
    # Allocate target classes
    target_class,priority   = mws_selection(mws_data,
                                            mag_faintest     = mag_faintest,
                                            mag_faint_filler = mag_faint_filler,
                                            mag_bright       = mag_bright)
    # Identify all targets
    in_target_list = target_class >= 0
    ii             = np.where(in_target_list)[0]
    n_selected     = len(ii)

    print("Properties read from mock: {}".format(mws_data.keys()))
    print("n_items in full catalog: {}".format(n))

    print('Selection criteria:')
    for criterion in ['mag_faintest','mag_faint_filler','mag_bright']:
        print(" -- {:30s} {}".format(criterion,locals().get(criterion)))

    print("n_items after selection: {}".format(n_selected))
        
    # This routine assigns targetIDs that encode the mapping of each row in the
    # target outputfile to a filenumber and row in the set of mock input files.
    # This targetID will be further modified when all target lists are merged.
    targetid = encode_rownum_filenum(mws_data['rownum'][ii],mws_data['filenum'][ii])
 
    # Assign random subpriorities for now
    subprior = np.random.uniform(0., 1., size=n_selected)

    # Assign DESI-standard bricknames
    # CAREFUL: These are not the bricknames used by the input catalog!
    brickname = desispec.brick.brickname(mws_data['RA'][ii], mws_data['DEC'][ii])

    # Assign target flags and true types
    desi_target_pop   = np.zeros(n_selected, dtype='i8')
    bgs_target_pop    = np.zeros(n_selected, dtype='i8') 
    mws_target_pop    = np.zeros(n_selected, dtype='i8')
    mws_target_pop[:] = target_class[ii]

    # Write target class a a string.
    # FIXME how is this supposed to work for combinations of bitmasks?
    true_type_pop         = np.zeros(n_selected, dtype='S10')
    unique_target_classes = np.unique(target_class[ii])
    for tc in unique_target_classes:
        tc_name = mws_mask.bitname(tc)
        has_this_target_class                = np.where(target_class[ii] == tc)[0]
        true_type_pop[has_this_target_class] = tc_name 
        print('Target class %s (%d): %d'%(tc_name,tc,len(has_this_target_class)))

    # Write the Targets to disk
    targets_filename = os.path.join(output_dir, targets_name)

    targets = Table()
    targets['TARGETID']    = targetid
    targets['BRICKNAME']   = brickname
    targets['RA']          = mws_data['RA'][ii]
    targets['DEC']         = mws_data['DEC'][ii]
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
    truth['RA']        = mws_data['RA'][ii]
    truth['DEC']       = mws_data['DEC'][ii]
    truth['TRUEZ']     = mws_data['Z'][ii]

    # True type is just targeted type for now.
    truth['TRUETYPE']  = true_type_pop
    truth.write(truth_filename, overwrite=True)

    # Write the selection data to disk
    selection_filename = os.path.join(output_dir, selection_name)
    selection          = Table()
    selection['ROW']   = mws_data['rownum'][ii]
    selection['FILE']  = mws_data['filenum'][ii]
    selection.write(selection_filename, overwrite=True)
    # Append an extension to the selection file with the file list
    hdulist            = astropy.io.fits.open(selection_filename,mode='append')
    file_list_table    = Table()
    file_list_table['FILENAME'] = file_list
    file_list_hdu               = astropy.io.fits.BinTableHDU.from_columns(np.array(file_list_table))
    #file_list_hdu              = astropy.io.fits.convenience.table_to_hdu(file_list_table)
    hdulist.append(file_list_hdu)
    hdulist.writeto(selection_filename,clobber=True)

    return targets, truth
    
