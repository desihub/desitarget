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

############################################################
def build_mock_target(root_mock_dir='', output_dir='',
                      mock_ext='hdf5',
                      targets_name='bgs_durahm_mxxl_targets.fits',
                      truth_name='bgs_durahm_mxxl_truth.fits',
                      selection_name='bgs_durahm_mxxl_selection.fits',
                      mag_faintest=20.0, mag_priority_split=19.5, mag_bright=15.0,
                      remake_cached_targets=False,
                      write_cached_targets=True,
                      rand_seed=42):

    """Builds a Target and Truth files from a series of mock files.

    Parameters:
    ----------
        rand_seed: int
            seed for random number generator.
    """
    desitarget.io.check_fitsio_version()
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
        targets   = Table(desitarget.io.whitespace_fits_read(targets_filename,ext='TARGETS'))
        print("    Truth:   {} ({:4.3f} Gb)".format(truth_filename,truth_filesize))
        truth     = Table(desitarget.io.whitespace_fits_read(truth_filename,ext='TRUTH'))
        file_list = Table(desitarget.io.whitespace_fits_read(truth_filename,ext='SOURCES'))
    else:
        # Read the mocks on disk. This returns a dict.
        # FIXME should just use table here too?
        data, file_list = desitarget.mock.io.read_mock_bgs_mxxl_brighttime(root_mock_dir=root_mock_dir,mock_ext=mock_ext)
        data_keys       = list(data.keys())
        n               = len(data[data_keys[0]])

        # Allocate target classes and priorities
        target_class = bgs_selection(data,
                                     mag_faintest       = mag_faintest,
                                     mag_priority_split = mag_priority_split,
                                     mag_bright         = mag_bright)
        # Identify all targets
        in_target_list = target_class >= 0
        ii             = np.where(in_target_list)[0]
        n_selected     = len(ii)

        print("Properties read from mock: {}".format(data.keys()))
        print("n_items in full catalog: {}".format(n))

        print('Selection criteria:')
        for criterion in ['mag_faintest','mag_priority_split','mag_bright']:
            print(" -- {:30s} {}".format(criterion,locals().get(criterion)))

        print("n_items after selection: {}".format(n_selected))

        # targetid  = np.random.randint(2**62, size=n_selected)

        # Targetids are row numbers in original input
        #targetid  = np.arange(0,n)[ii]

        # Targetids are brick/object combos from original input
        #BRICKID_FACTOR           = 1e10 # Max 10^10 objects per brick
        #combined_brick_object_id = data['brickid'][ii]*BRICKID_FACTOR + data['objid'][ii]
        #targetid                 = np.asarray(combined_brick_object_id,dtype=np.int64)

        # This routine assigns targetIDs that encode the mapping of each row in the
        # target outputfile to a filenumber and row in the set of mock input files.
        # This targetID will be further modified when all target lists are merged.
        targetid = desitarget.mock.io.encode_rownum_filenum(data['rownum'][ii],data['filenum'][ii])

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

        # APC This is a string?
        # FIXME (APC) This looks totally wrong, especially if the target class
        # encodes a combination of bits such that mask.names() returns a list.
        # The 'true type' should be something totally separate (an LRG is an
        # LRG, regardless of whether it's in the North or South, etc.).
        true_type_pop     = np.asarray(desitarget.targets.target_bitmask_to_string(target_class[ii],bgs_mask),dtype='S10')

        # Write the Targets to disk
        targets = Table()
        targets['TARGETID']    = targetid
        targets['BRICKNAME']   = brickname
        targets['RA']          = data['RA'][ii]
        targets['DEC']         = data['DEC'][ii]
        targets['DESI_TARGET'] = desi_target_pop
        targets['BGS_TARGET']  = bgs_target_pop
        targets['MWS_TARGET']  = mws_target_pop
        targets['SUBPRIORITY'] = subprior

        # Write the Truth to disk
        truth = Table()
        truth['TARGETID']  = targetid
        truth['BRICKNAME'] = brickname
        truth['RA']        = data['RA'][ii]
        truth['DEC']       = data['DEC'][ii]
        truth['TRUEZ']     = data['Z'][ii]

        # True type is just targeted type for now.
        truth['TRUETYPE']  = true_type_pop

        # File list
        file_list = np.array(file_list,dtype=[('FILE','|S500'),('NROWS','i8')])
        assert(np.all([len(x[0]) < 500 for x in file_list]))

        # Write Targets
        if write_cached_targets:
            print('Writing target list and truth for this mock...')
            with fitsio.FITS(targets_filename,'rw',clobber=True) as fits:
                fits.write(desiutil.io.encode_table(targets).as_array(),extname='TARGETS',clobber=True)

            # Write truth, slightly convoluted because we write two tables
            with fitsio.FITS(truth_filename,'rw',clobber=True) as fits:
                fits.write(desiutil.io.encode_table(truth).as_array(), extname='TRUTH')
                fits.write(file_list, extname='SOURCES')

            print("Wrote new files:")
            print("    Targets: {}".format(targets_filename))
            print("    Truth:   {}".format(truth_filename))

    return targets, truth, Table(file_list)
    

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

