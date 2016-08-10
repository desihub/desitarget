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
from   desitarget import mws_mask
import os
from   astropy.table import Table, Column
import astropy.io.fits as fitsio
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
    #priority     = np.zeros(len(mws_data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

    fainter_than_bright_limit  = mws_data[SELECTION_MAG_NAME]  >= mag_bright
    fainter_than_filler_limit  = mws_data[SELECTION_MAG_NAME]  >= mag_faint_filler
    brighter_than_filler_limit = mws_data[SELECTION_MAG_NAME]  <  mag_faint_filler
    brighter_than_faint_limit  = mws_data[SELECTION_MAG_NAME]  <  mag_faintest

    # Main sample
    select_main_sample               = (fainter_than_bright_limit) & (brighter_than_filler_limit)    
    target_class[select_main_sample] = mws_mask.mask('MWS_MAIN')
    #priority[select_main_sample]     = mws_mask['MWS_MAIN'].priorities['UNOBS']

    # Faint sample
    select_faint_filler_sample               = (fainter_than_filler_limit) & (brighter_than_faint_limit)    
    target_class[select_faint_filler_sample] = mws_mask.mask('MWS_MAIN_VERY_FAINT')
    #priority[select_faint_filler_sample]     = mws_mask['MWS_MAIN_VERY_FAINT'].priorities['UNOBS']

    return target_class#, priority

############################################################
def build_mock_target(root_mock_dir='', output_dir='',
                      targets_name='mws_galaxia_targets.fits',
                      truth_name='mws_galaxia_truth.fits',
                      mag_faintest = 20.0, mag_faint_filler=19.0, mag_bright=15.0, 
                      remake_cached_targets=False,
                      rand_seed=42,brickname_list=None):                  
    """Builds Target and Truth files from a series of mock files.
    
    Stores the resulting target and truth tables to disk.

    Parameters:
        root_mock_dir: string
            Root path to a hierarchy of Galaxia mock files in arbitrary bricks
        output_dir: string
            Path to write the outputs (targets.fits and truth.fits).
        mag_bright: float
            Upper limit cut in SDSSr observed magnitude. Default = 15.0.
        mag_faint_filler: float
            Step between filler targets and main sample. Default = 19.0
        mag_faintest: float
            Lower limit cut in SDSSr observed magnitude. Default = 20.0.
        remake_cached_targets: bool (default=False)
            If True, re-reads the mock files and generates new targets and
            truth, which are then saved to disk (replacing those that already
            exist, if they have the same names. If False, attempts to read
            cached files and aborts if they don't exist.
        rand_seed: int
            seed for random number generator.

    Note:
        This routine assigns targetIDs that encode the mapping of each row in the
        target outputfile to a filenumber and row in the set of mock input files.
        This targetID will be further modified when all target lists are merged.

        NUMOBS, PRIORITY also not set in this routine, hence output targets and
        truth are not directly usable as MTLs.
    """
    np.random.seed(seed=rand_seed)

    targets_filename = os.path.join(output_dir, targets_name)
    truth_filename   = os.path.join(output_dir, truth_name)

    target_exists    = os.path.exists(targets_filename)
    truth_exists     = os.path.exists(truth_filename)

    # Do we need to store copies of the targets and truth on disk?
    write_new_files = (remake_cached_targets) or not (target_exists and truth_exists)

    if not write_new_files:
        # Report the size of the existing input files
        targets_filesize = os.path.getsize(targets_filename) / (1024.0**3)
        truth_filesize   = os.path.getsize(targets_filename) / (1024.0**3)

        # Just read from the files we already have. Need to convert to astropy
        # table for consistency of return types.
        print("Reading existing files:")
        print("    Targets: {} ({:4.3f} Gb)".format(targets_filename,targets_filesize))
        targets = Table(fitsio.getdata(targets_filename))
        print("    Truth:   {} ({:4.3f} Gb)".format(truth_filename,truth_filesize))
        truth   = Table(fitsio.getdata(truth_filename))
    else:
        # Read the mocks on disk. This returns a dict.
        # fitsio rather than Table used for speed.
        data, file_list = desitarget.mock.io.read_mock_mws_brighttime(root_mock_dir=root_mock_dir,
                                                                          brickname_list=brickname_list)
        data_keys = list(data.keys())
        n          = len(data[data_keys[0]])
        
        # Allocate target classes but NOT priorities
        target_class  = mws_selection(data,
                                      mag_faintest     = mag_faintest,
                                      mag_faint_filler = mag_faint_filler,
                                      mag_bright       = mag_bright)
        # Identify all targets
        
        # The default null value of target_class is -1, A value of 0 is valid
        # as a target class in the yaml schema.
        in_target_list = target_class >= 0
        ii             = np.where(in_target_list)[0]
        n_selected     = len(ii)

        print("Properties read from mock: {}".format(data.keys()))
        print("n_items in full catalog: {}".format(n))

        print('Selection criteria:')
        for criterion in ['mag_faintest','mag_faint_filler','mag_bright']:
            print(" -- {:30s} {}".format(criterion,locals().get(criterion)))

        print("n_items after selection: {}".format(n_selected))
            
        # This routine assigns targetIDs that encode the mapping of each row in the
        # target outputfile to a filenumber and row in the set of mock input files.
        # This targetID will be further modified when all target lists are merged.
        targetid = desitarget.mock.io.encode_rownum_filenum(data['rownum'][ii],
                                                            data['filenum'][ii])
     
        # Assign random subpriorities for now
        subprior = np.random.uniform(0., 1., size=n_selected)

        # Assign DESI-standard bricknames
        # CAREFUL: These are not the bricknames used by the input catalog!
        brickname = desispec.brick.brickname(data['RA'][ii], data['DEC'][ii])

        # Assign target flags and true types
        desi_target_pop   = np.zeros(n_selected, dtype='i8')
        bgs_target_pop    = np.zeros(n_selected, dtype='i8') 
        mws_target_pop    = np.zeros(n_selected, dtype='i8')
        mws_target_pop[:] = target_class[ii]

        # APC This is a string? 
        # FIXME (APC) This looks totally wrong, especially if the target class
        # encodes a combination of bits such that mask.names() returns a list.
        # The 'true type' should be something totally separate (an LRG is an
        # LRG, regardless of whether it's in the North or South, etc.).
        true_type_pop     = np.asarray(desitarget.targets.target_bitmask_to_string(target_class[ii],mws_mask),dtype='S10')

        # Create targets table
        targets = Table()
        targets['TARGETID']    = targetid
        targets['BRICKNAME']   = brickname
        targets['RA']          = data['RA'][ii]
        targets['DEC']         = data['DEC'][ii]
        targets['DESI_TARGET'] = desi_target_pop
        targets['BGS_TARGET']  = bgs_target_pop
        targets['MWS_TARGET']  = mws_target_pop
        targets['SUBPRIORITY'] = subprior

        # FIXME should not be setting priority here
        #targets['NUMOBS_MORE'] = 1
        #targets['PRIORITY']    = priority[ii]

        # Create Truth table
        truth = Table()
        truth['TARGETID']  = targetid
        truth['BRICKNAME'] = brickname
        truth['RA']        = data['RA'][ii]
        truth['DEC']       = data['DEC'][ii]
        truth['TRUEZ']     = data['Z'][ii]

        # True type is just targeted type for now.
        truth['TRUETYPE']  = true_type_pop

        # Write Targets and Truth 
        fitsio.writeto(targets_filename, targets.as_array(),clobber=True)
        fitsio.writeto(truth_filename, truth.as_array(),clobber=True)

        print("Wrote new files:")
        print("    Targets: {}".format(targets_filename))
        print("    Truth:   {}".format(truth_filename))

    return targets, truth
    
