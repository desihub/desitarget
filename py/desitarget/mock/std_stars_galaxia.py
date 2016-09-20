# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
===========================
desitarget.mock.std_stars_galaxia
===========================

Builds target/truth files from already existing mock data.

Uses the MWS galaxia mocks to construct a list of stdstar targets, which are
treated separately by fibreassign.
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import os, re
import desitarget.mock.io
import desitarget.io
from   desitarget import desi_mask
import os
from   astropy.table import Table, Column
import astropy.io.fits as fitsio
import astropy.io.fits.convenience
import desispec.brick

from   desitarget.mtl import MTL_RESERVED_TARGETID_MIN_SKY, MTL_RESERVED_TARGETID_MIN_STD


############################################################
def std_star_selection(mws_data, mag_faint=19.0, mag_bright=16.0, grcolor=0.32, rzcolor=0.13, colortol=0.06):
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

    Parameters:
    -----------
        mws_data: dict
            Data required for selection
        mag_bright: float
            Upper limit cut in SDSSr observed magnitude.
        mag_faint: float
            Lower limit cut in SDSSr observed magnitude.
        grcolor: float
            Standard star ideal color, g-r. [mag]
        rzcolor: float
            Standard star ideal color, r-z. [mag]
        colortol:
            Acceptable distance in (g-r,r-z) space from ideal color. [mag]
    """
    # Parameters
    SELECTION_MAG_NAME = 'SDSSr_obs'

    COLOR_G_NAME       = 'SDSSg_obs'
    COLOR_R_NAME       = 'SDSSr_obs'
    COLOR_Z_NAME       = 'SDSSz_obs'

    # Will populate this array with the bitmask values of each target class
    target_class = np.zeros(len(mws_data[SELECTION_MAG_NAME]),dtype=np.int64) - 1

    fainter_than_bright_limit  = mws_data[SELECTION_MAG_NAME]  >= mag_bright
    brighter_than_faint_limit  = mws_data[SELECTION_MAG_NAME]  <  mag_faint

    gmr            = mws_data[COLOR_G_NAME] - mws_data[COLOR_R_NAME]
    rmz            = mws_data[COLOR_R_NAME] - mws_data[COLOR_Z_NAME]

    select_color     = (gmr - grcolor)**2 + (rmz - rzcolor)**2 < colortol**2
    select_mag       = (fainter_than_bright_limit) & (brighter_than_faint_limit)
    select_std_stars = (select_color) & (select_mag)
    target_class[select_std_stars] = desi_mask.mask('STD_FSTAR')

    return target_class

############################################################
def build_mock_target(root_mock_dir='', output_dir='',
                      targets_name='std_star_galaxia_targets.fits',
                      truth_name='std_star_galaxia_truth.fits',
                      mag_faint=19.0, mag_bright=16.0,
                      grcolor=0.32, rzcolor=0.13, colortol=0.06,
                      remake_cached_targets=False,
                      write_cached_targets=True,
                      rand_seed=42,brickname_list=None):
    """Builds Target and Truth files from a series of mock files.

    Stores the resulting target and truth tables to disk.

    Parameters:
        root_mock_dir: string
            Root path to a hierarchy of Galaxia mock files in arbitrary bricks
        output_dir: string
            Path to write the outputs (targets.fits and truth.fits).
        mag_bright: float
            Upper limit cut in SDSSr observed magnitude.
        mag_faint: float
            Lower limit cut in SDSSr observed magnitude.
        grcolor: float
            Standard star ideal color, g-r. [mag]
        rzcolor: float
            Standard star ideal color, r-z. [mag]
        colortol:
            Acceptable distance in (g-r,r-z) space from ideal color. [mag]
        remake_cached_targets: bool (default=False)
            If True, re-reads the mock files and generates new targets and
            truth, which are then saved to disk (replacing those that already
            exist, if they have the same names. If False, attempts to read
            cached files and aborts if they don't exist.
        rand_seed: int
            seed for random number generator.

    Note:
        Sets targetids differently to other mock-making routines, because in
        the current setup, the std star targetids are not re-written as part of
        making MTL. These IDs therefore need to be kept in some reserved range.

        Adds an OBSCONDITIONS column to targets. For conventional targets this
        is done by desitarget.mtl.make_mtl, but since standards are not
        included in the MTL this has to be done elsewhere. Currently this is
        done in a hacky way wihtout using the targetmask.yaml.
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
        # fitsio rather than Table used for speed.
        data, file_list = desitarget.mock.io.read_mock_mws_brighttime(root_mock_dir=root_mock_dir,
                                                                      brickname_list=brickname_list,
                                                                      selection='fstar_standards')
        data_keys = list(data.keys())
        n          = len(data[data_keys[0]])

        # Allocate target classes but NOT priorities
        target_class  = std_star_selection(data,
                                           mag_faint    = mag_faint,
                                           mag_bright   = mag_bright,
                                           grcolor  = grcolor,
                                           rzcolor  = rzcolor,
                                           colortol = colortol)
        # Identify all targets

        # The default null value of target_class is -1, A value of 0 is valid
        # as a target class in the yaml schema.
        in_target_list = target_class >= 0
        ii             = np.where(in_target_list)[0]
        n_selected     = len(ii)

        print("Properties read from mock: {}".format(data.keys()))
        print("n_items in full catalog: {}".format(n))

        print('Selection criteria:')
        for criterion in ['mag_faint','mag_bright','grcolor','rzcolor','colortol']:
            print(" -- {:30s} {}".format(criterion,locals().get(criterion)))

        print("n_items after selection: {}".format(n_selected))

        # This routine assigns targetIDs that encode the mapping of each row in the
        # target outputfile to a filenumber and row in the set of mock input files.
        # This targetID will be further modified when all target lists are merged.
        rowfile_id = desitarget.mock.io.encode_rownum_filenum(data['rownum'][ii],
                                                              data['filenum'][ii])

        # Ensure unique targetides in the range specified by desitarget.mtl
        targetid = MTL_RESERVED_TARGETID_MIN_STD + np.arange(0,len(ii))

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
        true_type_pop     = np.asarray(desitarget.targets.target_bitmask_to_string(target_class[ii],desi_mask),dtype='S10')

        # Create targets table. Note this includes OBSCONDITIONS
        # FIXME assuming that OBSCONDITIONS are the same for all standards and never
        # change.

        targets = Table()
        targets['TARGETID']      = targetid
        targets['BRICKNAME']     = brickname
        targets['RA']            = data['RA'][ii]
        targets['DEC']           = data['DEC'][ii]
        targets['DESI_TARGET']   = desi_target_pop
        targets['BGS_TARGET']    = bgs_target_pop
        targets['MWS_TARGET']    = mws_target_pop
        targets['SUBPRIORITY']   = subprior
        targets['OBSCONDITIONS'] = 1 # FIXME A hack, should be done properly.

        #targets['MUDELTA']     = data['pm_RA'][ii]
        #targets['MUALPHASTAR'] = data['pm_DEC'][ii]

        # Create Truth table
        truth = Table()
        truth['TARGETID']  = targetid
        truth['BRICKNAME'] = brickname
        truth['RA']        = data['RA'][ii]
        truth['DEC']       = data['DEC'][ii]
        truth['TRUEZ']     = data['Z'][ii]
        truth['ROWFILEID'] = rowfile_id

        # True type is just targeted type for now.
        truth['TRUETYPE']  = true_type_pop

        # Write Targets and Truth
        if write_cached_targets:
            fitsio.writeto(targets_filename, targets.as_array(),clobber=True)

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
        file_list = np.array(file_list,dtype=[('FILE','|S500'),('NROWS','i8')])
        sourcehdu = fitsio.BinTableHDU.from_columns(file_list,header=sourcehdr)

        truth_hdu = fitsio.HDUList([prihdu, mainhdu, sourcehdu])

        if write_cached_targets:
            truth_hdu.writeto(truth_filename,clobber=True)

            print("Wrote new files:")
            print("    Targets: {}".format(targets_filename))
            print("    Truth:   {}".format(truth_filename))

    return targets, truth, Table(file_list)
