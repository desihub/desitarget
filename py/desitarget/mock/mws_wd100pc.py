# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
===========================
desitarget.mock.mws_wd100pc
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
def wd100pc_selection(mws_data, mag_faint=20.0, mag_bright=15.0):
    """
    Apply the selection function to determine the target class of each entry in
    the input catalog.

    Parameters:
    -----------
        mws_data: dict
            Data required for selection
        mag_faint:    float
            Hard faint limit for inclusion in survey.
        mag_bright:      float
            Hard bright limit for inclusion in survey.
    """
    # Parameters
    SELECTION_MAG_NAME = 'magg'

    # Will populate this array with the bitmask values of each target class
    target_class = np.zeros(len(mws_data[SELECTION_MAG_NAME]),dtype=np.int64) - 1
    #priority     = np.zeros(len(mws_data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

    fainter_than_bright_limit  = mws_data[SELECTION_MAG_NAME]  >= mag_bright
    brighter_than_faint_limit  = mws_data[SELECTION_MAG_NAME]  <  mag_faint
    is_wd                      = mws_data['WD'] == 1

    # WD sample
    select_wd_sample               = (fainter_than_bright_limit) & (brighter_than_faint_limit) & (is_wd)   
    target_class[select_wd_sample] = mws_mask.mask('MWS_WD')
    #priority[select_wd_sample]     = mws_mask['MWS_WD'].priorities['UNOBS']

    # Nearby ('100pc') sample -- everything in the input table that isn't a WD
    # Expect to refine this in future
    select_nearby_sample               = (fainter_than_bright_limit) & (brighter_than_faint_limit) & (np.invert(is_wd))
    target_class[select_nearby_sample] = mws_mask.mask('MWS_NEARBY')
    #priority[select_nearby_sample]     = mws_mask['MWS_NEARBY'].priorities['UNOBS']

    return target_class#, priority

############################################################
def build_mock_target(root_mock_dir='', output_dir='',
                      targets_name='mws_wd100pc_targets.fits',
                      truth_name='mws_wd100pc_truth.fits',
                      selection_name='mws_wd100pc_selection.fits',
                      mag_faint=20.0, mag_bright=15.0,
                      remake_cached_targets=False,
                      write_cached_targets=True,
                      rand_seed=42):

    """Builds a Target and Truth files from a series of mock files

    Parameters:
    -----------
        mock_dir: string
            Root path to directory with GUMS-based WD and 100pc mock table
        output_dir: string
            Path to write the outputs (targets.fits and truth.fits).
        mag_bright: float
            Upper limit cut in Gaia G observed magnitude. Default = 15.0.
        mag_faint: float
            Lower limit cut in Gaia G observed magnitude. Default = 20.0.
        rand_seed: int
            seed for random number generator.
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
        targets   = Table(fitsio.getdata(targets_filename))
        print("    Truth:   {} ({:4.3f} Gb)".format(truth_filename,truth_filesize))
        truth     = Table(fitsio.getdata(truth_filename,extname='TRUTH'))
        file_list = Table(fitsio.getdata(truth_filename,extname='SOURCES'))
    else:
        # Read the mocks on disk. This returns a dict.
        # FIXME should just use table here too?
        mws_data, file_list = desitarget.mock.io.read_mock_wd100pc_brighttime(root_mock_dir=root_mock_dir)
        mws_keys            = list(mws_data.keys())
        n                   = len(mws_data[mws_keys[0]])
        
        # Allocate target classes
        target_class = wd100pc_selection(mws_data,
                                         mag_faint  = mag_faint,
                                         mag_bright = mag_bright)
        # Identify all targets
        in_target_list = target_class >= 0
        ii             = np.where(in_target_list)[0]
        n_selected     = len(ii)

        print("Properties read from mock: {}".format(mws_data.keys()))
        print("n_items in full catalog: {}".format(n))

        print('Selection criteria:')
        for criterion in ['mag_faint','mag_bright']:
            print(" -- {:30s} {}".format(criterion,locals().get(criterion)))

        print("n_items after selection: {}".format(n_selected))
            
        # targetid  = np.random.randint(2**62, size=n_selected)

        # Targetids are row numbers in original input (since there is only one
        # file)
        targetid  = np.arange(0,n)[ii]

        # Targetids are brick/object combos from original input
        #BRICKID_FACTOR           = 1e10 # Max 10^10 objects per brick
        #combined_brick_object_id = mws_data['brickid'][ii]*BRICKID_FACTOR + mws_data['objid'][ii]
        #targetid                 = np.asarray(combined_brick_object_id,dtype=np.int64)

        # Assign random subpriorities for now
        subprior  = np.random.uniform(0., 1., size=n_selected)

        # Assign DESI-standard bricknames
        # CAREFUL: These are not the bricknames used by the input catalog!
        brickname = desispec.brick.brickname(mws_data['RA'][ii], mws_data['DEC'][ii])

        # assign target flags and true types
        desi_target_pop   = np.zeros(n_selected, dtype='i8')
        bgs_target_pop    = np.zeros(n_selected, dtype='i8') 
        mws_target_pop    = np.zeros(n_selected, dtype='i8')
        mws_target_pop[:] = target_class[ii]

        # APC This is a string? 
        # FIXME (APC) This looks totally wrong, especially if the target class
        # encodes a combination of bits such that mask.names() returns a list.
        # The 'true type' should be something totally separate (an LRG is an
        # LRG, regardless of whether it's in the North or South, etc.).
        true_type_pop     = np.array(desitarget.targets.target_bitmask_to_string(target_class[ii],mws_mask),dtype='S10')

        # Write the Targets to disk
        targets = Table()
        targets['TARGETID']    = targetid
        targets['BRICKNAME']   = brickname
        targets['RA']          = mws_data['RA'][ii]
        targets['DEC']         = mws_data['DEC'][ii]
        targets['DESI_TARGET'] = desi_target_pop
        targets['BGS_TARGET']  = bgs_target_pop
        targets['MWS_TARGET']  = mws_target_pop
        #targets['NUMOBS_MORE'] = 1
        #targets['PRIORITY']    = priority[ii]
        targets['SUBPRIORITY'] = subprior

        # Write the Truth to disk
        truth = Table()
        truth['TARGETID']  = targetid
        truth['BRICKNAME'] = brickname
        truth['RA']        = mws_data['RA'][ii]
        truth['DEC']       = mws_data['DEC'][ii]
        truth['TRUEZ']     = mws_data['Z'][ii]

        # True type is just targeted type for now.
        truth['TRUETYPE']  = true_type_pop
        
        # File list
        file_list = np.array(file_list,dtype=[('FILE','|S500'),('NROWS','i8')])

        # Write Targets and Truth 
        if write_cached_targets:

            # In Py3, no obvious advantage in using astropy.io.fits here and a
            # huge potential slowdown over Py2 when writing brickname column.
            # Symmetry with how the truth table is written would be nice, but
            # have to forego that until this is understood.
            # fitsio.writeto(targets_filename, targets.as_array(),clobber=True)
            targets.write(targets_filename,overwrite=True)

            # Write truth, slightly convoluted because we write two tables
            #fitsio.writeto(truth_filename, truth.as_array(),clobber=True)
            prihdr    = fitsio.Header()
            prihdu    = fitsio.PrimaryHDU(header=prihdr)
            
            mainhdr   = fitsio.Header()
            mainhdr['EXTNAME'] = 'TRUTH'
            mainhdu   = fitsio.BinTableHDU.from_columns(truth.as_array(),header=mainhdr)

            sourcehdr = fitsio.Header()
            sourcehdr['EXTNAME'] = 'SOURCES'
            assert(np.all([len(x[0]) < 500 for x in file_list]))
            sourcehdu = fitsio.BinTableHDU.from_columns(file_list,header=sourcehdr)

            truth_hdu = fitsio.HDUList([prihdu, mainhdu, sourcehdu])
            if write_cached_targets:
                truth_hdu.writeto(truth_filename,clobber=True)

            print("Wrote new files:")
            print("    Targets: {}".format(targets_filename))
            print("    Truth:   {}".format(truth_filename))

    return targets, truth, Table(file_list)
    
        # Write the selection data to disk
        #selection_filename = os.path.join(output_dir, selection_name)
        #selection          = Table()
        #selection['ROW']   = mws_data['rownum'][ii]
        #selection['FILE']  = mws_data['filenum'][ii]
        #selection.write(selection_filename, overwrite=True)
        ## Append an extension to the selection file with the file list
        #hdulist            = astropy.io.fits.open(selection_filename,mode='append')
        #file_list_table    = Table()
        #file_list_table['FILENAME'] = file_list
        #file_list_hdu               = astropy.io.fits.BinTableHDU.from_columns(np.array(file_list_table))
        ##file_list_hdu              = astropy.io.fits.convenience.table_to_hdu(file_list_table)
        #hdulist.append(file_list_hdu)
        #hdulist.writeto(selection_filename,clobber=True)

    return
    
