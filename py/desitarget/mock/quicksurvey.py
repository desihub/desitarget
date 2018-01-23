# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.quicksurvey
===========================

The code in this module is related to quicksurvey but has likely been rendered
obsolete by the latest refactor of select_mock_targets.  It is being placed
here temporarily and will be cleaned up in a second pass.
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
from glob import glob

import fitsio
from scipy import constants

from desitarget.io import check_fitsio_version, iter_files
from desitarget.mock.sample import SampleGMM
from desiutil.brick import brickname as get_brickname_from_radec
from desiutil.brick import Bricks
from desimodel.footprint import radec2pix

from desiutil.log import get_logger, DEBUG
log = get_logger()

#############
# Code previously in desitarget.mock.build
#############

def target_selection(Selection, target_name, targets, truth, nside, healpix_id, seed, rand, log, output_dir):
    """Applies target selection functions to a set of targets and truth tables.
    
    Args:
        Selection: desitarget.mock.Selection 
            This object contains the information from the configuration file
            used to start the construction of the mock target files.
        target_name: string
            Name of the target being processesed, i.e. "QSO"
        targets: astropy.table
            Initial set of targets coming from the input files. 
        truth: astropy.table
            Corresponding Truth to Targets
        nside: int
            nside for healpix
        healpix_id: int
            ID of current healpix
        seed: int
            Seed used for the random number generator
        rand: numpy.random.RandomState
        log: logger object
        output_dir: str
            Directory where the outputs are written.
            
    Output:
        targets: astropy.table
            Final set of targets after target selection. 
        truth: astropy.table
            Corresponding Truth to Targets

    """
    
    selection_function = '{}_select'.format(target_name.lower())
    select_targets_function = getattr(Selection, selection_function)

    select_targets_function(targets, truth, boss_std=1)
    keep = np.where(targets['DESI_TARGET'] != 0)[0]

    targets = targets[keep]
    truth = truth[keep]
    if len(keep) == 0:
        log.warning('All {} targets rejected!'.format(target_name))
    else:
        log.warning('{} targets accepted out of a total of {}'.format(len(keep), len(targets)))
    
    #Make a final check. Some of the TARGET flags must have been selected. 
    no_target_class = np.ones(len(targets), dtype=bool)
    if 'DESI_TARGET' in targets.dtype.names:
        no_target_class &=  targets['DESI_TARGET'] == 0
    if 'BGS_TARGET' in targets.dtype.names:
        no_target_class &= targets['BGS_TARGET']  == 0
    if 'MWS_TARGET' in targets.dtype.names:
        no_target_class &= targets['MWS_TARGET']  == 0

    n_no_target_class = np.sum(no_target_class)
    if n_no_target_class > 0:
        raise ValueError('WARNING: {:d} rows in targets.calc_numobs have no target class'.format(n_no_target_class))
    
    return targets, truth

def estimate_number_density(ra, dec):
    """Estimates the number density of points with positions RA, DEC.
    
    Args:
        ra: numpy.array
            RA positions in degrees.
        dec: numpy.array
            DEC positions in degrees.
            
    Output: 
        average number density: float
    """
    import healpy as hp
    if len(ra) != len(dec):
        raise ValueError('Arrays ra,dec must have same size.')
    
    n_tot = len(ra)
    nside = 64 # this produces a bin area close to 1 deg^2
    bin_area = hp.nside2pixarea(nside, degrees=True)
    pixels = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra), False)
    counts = np.bincount(pixels)
    if  n_tot == 0:
        return 0.0
    else:
        # This is a weighted density that does not take into account empty healpixels
        return np.sum(counts*counts)/np.sum(counts)/bin_area

def downsample_pixel(density, zcut, target_name, targets, truth, nside,
                     healpix_id, seed, rand, log, output_dir, contam=False):
    """Reduces the number of targets to match a desired number density.
    
    Args:
        density: np.array
            Array of number densities desired in different redshift intervals. Units of targets/deg^2
        zcut: np.array
            Array defining the redshift intervals to set the desired number densities.
        target_name: string
            Name of the target being processesed, i.e. "QSO"
        targets: astropy.table
            Final set of Targets. 
        truth: astropy.table
            Corresponding Truth to Targets
        nside: int
            nside for healpix
        healpix_id: int
            ID of current healpix
        seed: int
            Seed used for the random number generator
        rand: numpy.random.RandomState
        log: logger object
        output_dir: str
            Directory where the outputs are written.
        contam: bool
            If True the subsampling is only applied to contaminants. If False it
            is applied only to noncontaminated targets.  Output: Updated
            versions of the tables: targets truth.
            
    Example:
        If density = [300,400] and zcut=[0.0,2.0, 3.0] the first number density cap of 300
        will be applied to all targets with redshifts 0.0 < z < 2.0, while the second cap
        of 400 will be applied to all targets with redshifts 2.0 < z < 3.0

    """
    import healpy as hp
    
    n_cuts = len(density)
    n_obj = len(targets)
        
    r = rand.uniform(0.0, 1.0, size=n_obj)
    keep = r <= 1.0
    
    if contam:
        good_targets = (truth['CONTAM_TARGET'] & contam_mask.mask(target_name+'_CONTAM')) !=0
    else:
        good_targets = (truth['TEMPLATETYPE']==target_name) & (truth['CONTAM_TARGET']==0)
        
    if contam:
        log.info('Downsampling {} contaminants'.format(target_name))
    else:
        log.info('Downsampling pure {} targets'.format(target_name))
        
    for i in range(n_cuts):
        in_z = (truth['TRUEZ'] > zcut[i]) & (truth['TRUEZ']<zcut[i+1]) & (good_targets)
        input_density = estimate_number_density(targets['RA'][in_z], targets['DEC'][in_z])
        if density[i] < input_density:
            frac_keep = density[i]/input_density
            keep[in_z] = (r[in_z]<frac_keep)
            log.info('Downsampling for {}. Going from {} to {} obs/deg^2'.format(target_name, input_density, density[i]))
        else:
            log.info('Cannot go from {} to {} obs/deg^2 for object {}'.format(input_density, density[i], target_name))
            
    targets = targets[keep]
    truth = truth[keep]
    return targets, truth
    
def read_mock_no_spectra(source_name, params, log, rand=None, nproc=1,
                         healpixels=None, nside=16, in_desi=True):
    """Read one specified mock catalog.
    
    Args:
        source_name: str
            Name of the target being processesed, e.g., 'QSO'.
        params: dict
            Dictionary summary of the input configuration file.
        log: desiutil.logger
           Logger object.
        rand: numpy.random.RandomState
           Object for random number generation.
        nproc: int
            Number of processors to be used for reading.
        healpixels : numpy.ndarray or int
            List of healpixels to process. The mocks are cut to match these
            pixels.
        nside: int
            nside for healpix
        in_desi: boolean
            Decides whether the targets will be trimmed to be inside the DESI
            footprint.
            
    Returns:
        source_data : dict
            Parsed source data based on the input mock catalog.

    """
    # Read the mock catalog.
    import desitarget.mock.io as mockio
    
    target_name = params['sources'][source_name]['target_name'] # Target type (e.g., ELG)
    mockformat = params['sources'][source_name]['format']

    mock_dir_name = params['sources'][source_name]['mock_dir_name']
    if 'magcut' in params['sources'][source_name].keys():
        magcut = params['sources'][source_name]['magcut']
    else:
        magcut = None

    log.info('Source: {}, target: {}, format: {}'.format(source_name, target_name.upper(), mockformat))
    log.info('Reading {}'.format(mock_dir_name))

    mockread_function = getattr(mockio, 'read_{}'.format(mockformat))
    if 'LYA' in params['sources'][source_name].keys():
        lya = params['sources'][source_name]['LYA']
    else:
        lya = None
    source_data = mockread_function(mock_dir_name, target_name, rand=rand,
                                    magcut=magcut, nproc=nproc, lya=lya,
                                    healpixels=healpixels, nside=nside)


    # Insert proper density fluctuations model here!  Note that in general
    # healpixels will generally be a scalar (because it's called inside a loop),
    # but also allow for multiple healpixels.
    try:
        npix = healpixels.size
    except:
        npix = len(healpixels)
    skyarea = npix * hp.nside2pixarea(nside, degrees=True)

    # Return only the points that are in the DESI footprint.
    if bool(source_data):
        if in_desi:
            import desimodel.io
            import desimodel.footprint

            n_obj = len(source_data['RA'])
            tiles = desimodel.io.load_tiles()
            if n_obj > 0:
                indesi = desimodel.footprint.is_point_in_desi(tiles, source_data['RA'], source_data['DEC'])
                for k in source_data.keys():
                    if type(source_data[k]) is np.ndarray:
                        if n_obj == len(source_data[k]):
                            source_data[k] = source_data[k][indesi]
                        
    return source_data

def _initialize_no_spectra(params, verbose=False, seed=1, output_dir="./", nproc=1,
                           nside=16, healpixels=None):
    """Initialize various objects needed to generate mock targets (with and without
    spectra).

    Args:
        params : dict
            Source parameters.
        seed: int
            Seed for the random number generator
        output_dir : str
            Output directory (default '.').
        nproc : int
            Number of parallel processes to use (default 1).
        nside : int
            Healpix resolution corresponding to healpixels (default 16).
        healpixels : numpy.ndarray or int
            Restrict the sample of mock targets analyzed to those lying inside
            this set (array) of healpix pixels. Default (None).

    Returns:
        log: desiutil.logger
           Logger object.
        rand: numpy.random.RandomState
           Object for random number generation.
        magnitudes: desitarget.mock.MockMagnitudes    
            Object to assign magnitudes to each target class.
        selection: desitarget.mock.SelectTargets
            Object to select targets from the input mock catalogs.

    """
    from desiutil.log import get_logger, DEBUG
    from desitarget.mock.selection import SelectTargets

    from desitarget.mock.spectra import MockMagnitudes

    # Initialize logger
    if verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()
    
    if params is None:
        log.fatal('Required params input not given!')
        raise ValueError

    # Check for required parameters in the input 'params' dict
    # Initialize the random seed
    rand = np.random.RandomState(seed)

    # Create the output directories
    if os.path.exists(output_dir):
        if os.listdir(output_dir):
            log.warning('Output directory {} is not empty.'.format(output_dir))
    else:
        log.info('Creating directory {}'.format(output_dir))
        os.makedirs(output_dir)    
    log.info('Writing to output directory {}'.format(output_dir))      
        
    # Default set of healpixels is the whole sky (yikes!)
    if healpixels is None:
        healpixels = np.arange(hp.nside2npix(nside))

    areaperpix = hp.nside2pixarea(nside, degrees=True)
    totarea = len(healpixels) * areaperpix
    log.info('Processing {} healpixel(s) (nside = {}, {:.3f} deg2/pixel) spanning {:.3f} deg2.'.format(
        len(healpixels), nside, areaperpix, totarea))

    # Initialize the Classes used to assign magnitudes and to
    # select targets.
    log.info('Initializing the MockMagnitudes and SelectTargets Classes.')
    selection = SelectTargets(verbose=verbose, rand=rand)
    Magnitudes = MockMagnitudes(rand=rand, verbose=verbose, nproc=nproc)
    
    return log, rand, Magnitudes, selection, healpixels    

def finish_catalog_no_spectra(targets, truth, skytargets, skytruth, nside,
                              healpix_id, seed, rand, log, output_dir):
    """Adds TARGETID, SUBPRIORITY and HPXPIXEL to targets.
    
    Args:
        targets: astropy.table
            Final set of Targets. 
        truth: astropy.table
            Corresponding Truth to Targets
        skytargets: astropy.table
            Sky positions
        skytruth: astropy.table
            Corresponding Truth to Sky
        nside: int
            nside for healpix
        healpix_id: int
            ID of current healpix
        seed: int
            Seed used for the random number generator
        rand: numpy.random.RandomState
        log: logger object
        output_dir: str
            Directory where the outputs are written.
            
    Output:
        Updated versions of: targets, truth, skytargets, skytruth
    """
    from desimodel.footprint import radec2pix

    n_obj = len(targets)
    n_sky = len(skytargets)
    log.info('Total number of targets and sky in pixel {}: {} {}'.format(healpix_id, n_obj, n_sky))
    objid = np.arange(n_obj + n_sky)
    
    if n_obj > 0:
        targetid = encode_targetid(objid=objid, brickid=healpix_id*np.ones(n_obj+n_sky, dtype=int), mock=1)
        subpriority = rand.uniform(0.0, 1.0, size=n_obj+n_sky)

        truth['TARGETID'][:] = targetid[:n_obj]
        targets['TARGETID'][:] = targetid[:n_obj]
        targets['SUBPRIORITY'][:] = subpriority[:n_obj]
    
    if n_sky > 0:
        skytargets['TARGETID'][:] = targetid[n_obj:]
        skytargets['SUBPRIORITY'][:] = subpriority[n_obj:]
        
    if n_obj > 0:
        targpix = radec2pix(nside, targets['RA'], targets['DEC'])
        targets['HPXPIXEL'][:] = targpix

    if n_sky > 0:
        targpix = radec2pix(nside, skytargets['RA'], skytargets['DEC'])
        skytargets['HPXPIXEL'][:] = targpix
    
    return targets, truth, skytargets, skytruth

def get_magnitudes_onepixel(Magnitudes, source_data, target_name, rand, log,
                            nside, healpix_id, dust_dir):
    """Assigns magnitudes to set of targets and truth dictionaries located on the pixel healpix_id.
    
    Args:
        Magnitudes: desitarget.mock.Magnitudes
            This object contains the information to assign magnitudes to each kind of targets.
        source_data: dictionary
            This corresponds to the "raw" data coming directly from the mock file.
        target_name: string
            Name of the target being processesed, i.e. "QSO"
        rand: numpy.random.RandomState
        log: logger object
        nside: int
            nside for healpix
        healpix_id: int
            ID of current healpix
        seed: int
            Seed used for the random number generator
        dust_dir: str
            Directory where the E(B-V) information is stored.
            
    Output:
        targets: astropy.table
            Targets on the pixel healpix_id.
        truth: astropy.table
            Corresponding Truth to Targets
    """
    from desimodel.footprint import radec2pix
    from desitarget.mock.mockmaker import empty_targets_table, empty_truth_table

    obj_pix_id = radec2pix(nside, source_data['RA'], source_data['DEC'])
    onpix = np.where(obj_pix_id == healpix_id)[0]
    
    log.info('{} objects of type {} in healpix_id {}'.format(np.count_nonzero(onpix), target_name, healpix_id))
    
    nobj = len(onpix)

    # Initialize the output targets and truth catalogs and populate them with
    # the quantities of interest.
    targets = empty_targets_table(nobj)
    truth = empty_truth_table(nobj)

    for key in ('RA', 'DEC', 'BRICKNAME'):
        targets[key][:] = source_data[key][onpix]

    # Assign unique OBJID values and reddenings. 
    targets['HPXPIXEL'][:] = np.arange(nobj)

    extcoeff = dict(G = 3.214, R = 2.165, Z = 1.221, W1 = 0.184, W2 = 0.113)
    ebv = sfdmap.ebv(targets['RA'], targets['DEC'], mapdir=dust_dir)
    for band in ('G', 'R', 'Z', 'W1', 'W2'):
        targets['MW_TRANSMISSION_{}'.format(band)][:] = 10**(-0.4 * extcoeff[band] * ebv)

    # Hack! Assume a constant 5-sigma depth of g=24.7, r=23.9, and z=23.0 for
    # all bricks: http://legacysurvey.org/dr3/description and a constant depth
    # (W1=22.3-->1.2 nanomaggies, W2=23.8-->0.3 nanomaggies) in the WISE bands
    # for now.
    onesigma = np.hstack([10**(0.4 * (22.5 - np.array([24.7, 23.9, 23.0])) ) / 5,
                10**(0.4 * (22.5 - np.array([22.3, 23.8])) )])
    
    # Add shapes and sizes.
    if 'SHAPEEXP_R' in source_data.keys(): # not all target types have shape information
        for key in ('SHAPEEXP_R', 'SHAPEEXP_E1', 'SHAPEEXP_E2',
                    'SHAPEDEV_R', 'SHAPEDEV_E1', 'SHAPEDEV_E2'):
            targets[key][:] = source_data[key][onpix]

    for key, source_key in zip( ['MOCKID', 'SEED', 'TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TRUESPECTYPE'],
                                ['MOCKID', 'SEED', 'TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TRUESPECTYPE'] ):
        if isinstance(source_data[source_key], np.ndarray):
            truth[key][:] = source_data[source_key][onpix]
        else:
            truth[key][:] = np.repeat(source_data[source_key], nobj)

    if target_name.lower() == 'sky':
        return [targets, truth]
            
    truth['TRUEZ'][:] = source_data['Z'][onpix]

    # Assign the magnitudes
    meta = getattr(Magnitudes, target_name.lower())(source_data, index=onpix, mockformat=source_data['MOCKFORMAT'])

    for key in ('TEMPLATEID', 'MAG', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 
                'OIIFLUX', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'):
        truth[key][:] = meta[key]

    for band, fluxkey in enumerate( ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2') ):
        targets[fluxkey][:] = truth[fluxkey][:] #+ rand.normal(scale=onesigma[band], size=nobj)

    return [targets, truth]

def targets_truth_no_spectra(params, seed=1, output_dir="./", nproc=1, nside=16,
                             healpixels=None, verbose=False, dust_dir="./"): 
    """Generate a catalog of targets and the corresponding truth catalog.
    
    Inputs:
        params: dictionary
            Input parameters read from the configuration files.
        seed: int
            Seed for the random number generation.
        output_dir : str
            Output directory (default '.').
        nproc : int
            Number of parallel processes to use (default 1).
        nside : int
            Healpix resolution corresponding to healpixels (default 16).
        healpixels : numpy.ndarray or int
            Restrict the sample of mock targets analyzed to those lying inside
            this set (array) of healpix pixels. Default (None)
        verbose: bool
            Degree of verbosity.
            
    Returns:
        Files 'targets.fits', 'truth.fits', 'sky.fits', 'standards-dark.fits',
        and 'standards-bright.fits' written to disk for a list of healpixels.
    
    """
    log, rand, Magnitudes, Selection, healpixels = _initialize_no_spectra(params,
                                                                          verbose=verbose,
                                                                          seed=seed, 
                                                                          output_dir=output_dir, 
                                                                          nproc=nproc,
                                                                          nside=nside,
                                                                          healpixels=healpixels)
    
    # Loop over each source / object type.
    for healpix in healpixels:
        alltargets = list()
        alltruth = list()
        allskytargets = list()
        allskytruth = list()
          
        for source_name in sorted(params['sources'].keys()):
            
            # Read the data.
            log.info('Reading  source : {}'.format(source_name))
            source_data = read_mock_no_spectra(source_name, params, log, rand=rand, nproc=nproc,
                                               healpixels=healpix, nside=nside)
        
            # If there are no sources, keep going.
            if not bool(source_data):
                continue
                
            #Initialize variables for density subsampling
            if 'density' in params['sources'][source_name].keys():
                density = [params['sources'][source_name]['density']]
                zcut = [-1000, 1000]
            
            # assign magnitudes for targets in that pixel
            pixel_results = get_magnitudes_onepixel(Magnitudes, source_data, source_name, rand, log, 
                                                    nside, healpix, dust_dir=params['dust_dir'])
            
            targets = pixel_results[0]
            truth = pixel_results[1]
            
            # Make target selection and downsample number density if necessary. SKY is an special case.
            if source_name.upper() == 'SKY':
                if 'density' in params['sources'][source_name].keys():
                    targets, truth = downsample_pixel(density, zcut, source_name, targets, truth,
                                                      nside, healpix, seed, rand, log, output_dir, contam=False)
                allskytargets.append(targets)
                allskytruth.append(truth)                    
            else:
                targets, truth = target_selection(Selection, source_name, targets, truth,
                                                             nside, healpix, seed, rand, log, output_dir)
                
                # Downsample to a global number density if required
                if 'density' in params['sources'][source_name].keys():
                    if source_name == 'QSO':
                        # Distinguish between the Lyman-alpha and tracer QSOs
                        if 'LYA' in params['sources'][source_name].keys():
                            density = [params['sources'][source_name]['density'], 
                                      params['sources'][source_name]['LYA']['density']]
                            zcut = [-1000,params['sources'][source_name]['LYA']['zcut'],1000]
                    
                    targets, truth = downsample_pixel(density, zcut, source_name, targets, truth,
                                                    nside, healpix, seed, rand, log, output_dir, contam=False)
               
                    
                if len(targets)>0:
                    alltargets.append(targets)
                    alltruth.append(truth)
               
        if len(alltargets)==0 and len(allskytargets)==0:
            continue
        
        #Merge all sources
        if len(alltargets):
            targets = vstack(alltargets)
            truth = vstack(alltruth)
            
            # downsample contaminants for each class
            for source_name in sorted(params['sources'].keys()):
                if 'contam' in params['sources'][source_name].keys():
                    # Initialize variables for density subsampling
                    zcut=[-1000,1000]
                    density = [params['sources'][source_name]['contam']['density']]
                    targets, truth = downsample_pixel(density, zcut, source_name, targets, truth,
                                                  nside, healpix, seed, rand, log, output_dir, contam=True)
        else:
            targets = []
            truth = []
                    
        if len(allskytargets):
            skytargets = vstack(allskytargets)
            skytruth = vstack(allskytruth)
        else:
            skytargets = []
            skytruth = []

        #Add some final columns
        targets, truth, skytargets, skytruth = finish_catalog_no_spectra(targets, truth, skytargets, skytruth,
                                                                         nside,healpix, seed, rand, log, output_dir)
        #write the results
        _write_targets_truth(targets, truth, skytargets, skytruth,  
                             nside, healpix, seed, log, output_dir)
    return


#############
# Code previously in desitarget.mock.io
#############

"""How to distribute 52 user bits of targetid.

Used to generate target IDs as combination of input file and row in input file.
Sets the maximum number of rows per input file for mocks using this scheme to
generate target IDs

"""
# First 32 bits are row
ENCODE_ROW_END     = 32
ENCODE_ROW_MASK    = 2**ENCODE_ROW_END - 2**0
ENCODE_ROW_MAX     = ENCODE_ROW_MASK
# Next 20 bits are file
ENCODE_FILE_END    = 52
ENCODE_FILE_MASK   = 2**ENCODE_FILE_END - 2**ENCODE_ROW_END
ENCODE_FILE_MAX    = ENCODE_FILE_MASK >> ENCODE_ROW_END

try:
    C_LIGHT = constants.c/1000.0
except TypeError:
    #
    # This can happen during documentation builds.
    #
    C_LIGHT = 299792458.0/1000.0

def print_all_mocks_info(params):
    """Prints parameters to read mock files.

    Parameters
    ----------
        params : dict
            The different kind of sources are stored under the 'sources' key.

    """
    log.info('Paths and targets:')
    for source_name in params['sources'].keys():
        mockformat = params['sources'][source_name]['format']
        source_path = params['sources'][source_name]['mock_dir_name']
        target_name = params['sources'][source_name]['target_name']
        log.info('source_name: {}\n format: {} \n target_name {} \n path: {}'.format(source_name,
                                                                                  mockformat,
                                                                                  target_name,
                                                                                  source_path))

def load_all_mocks(params, rand=None, bricksize=0.25, nproc=1, healpixels=None, nside=None):
    """Read all the mocks.

    Parameters
    ----------
    params : dict
        The different kind of sources are stored under the 'sources' key.
    rand : numpy.RandomState
        RandomState object used for the random number generation.
    bricksize : float
        Size of each (imaging) brick in deg.
    nproc : int
        Number of cores to use for reading (default 1).
    healpixels : numpy.ndarray or numpy.int64
        Only read objects within this list of healpix pixel numbers.
    nside : int
        Healpix resolution for input healpixels.

    Returns
    -------
    source_data_all : dict
        The keys correspond to the different input 'sources' stored under
        params['sources'].keys()

    """
    if rand is None:
        rand = np.random.RandomState()

    check_fitsio_version() # Make sure fitsio is up to date.

    #loaded_mocks = list()

    source_data_all = {}
    for source_name in sorted(params['sources'].keys()):

        target_name = params['sources'][source_name]['target_name']
        mockformat = params['sources'][source_name]['format']
        mock_dir_name = params['sources'][source_name]['mock_dir_name']
        #bounds = params['sources'][source_name]['bounds']

        if 'magcut' in params['sources'][source_name].keys():
            magcut = params['sources'][source_name]['magcut']
        else:
            magcut = None

        read_function = 'read_{}'.format(mockformat)

        log.info('Source: {}, target: {}, format: {}'.format(source_name, target_name.upper(), mockformat))
        log.info('Reading {} with mock.io.{}'.format(mock_dir_name, read_function))

        func = globals()[read_function]
        if 'LYA' in params['sources'][source_name].keys():
            result = func(mock_dir_name, target_name, rand=rand, bricksize=bricksize,
                          magcut=magcut, nproc=nproc, bounds=bounds, 
                          lya=params['sources'][source_name]['LYA'])
        else:
            result = func(mock_dir_name, target_name, rand=rand, bricksize=bricksize,
                          magcut=magcut, nproc=nproc, healpixels=healpixels, nside=nside)

        source_data_all[source_name] = result
        print()

    log.info('Loaded {} mock catalog(s).'.format(len(source_data_all)))
    return source_data_all

def encode_rownum_filenum(rownum, filenum):
    """Encodes row and file number in 52 packed bits.

    Parameters
    ----------
    rownum : int
        Row in input file.
    filenum : int
        File number in input file set.

    Returns
    -------
    encoded value(s) : int64 numpy.ndarray
        52 packed bits encoding row and file number.

    """
    assert(np.shape(rownum) == np.shape(filenum))
    assert(np.all(rownum  >= 0))
    assert(np.all(rownum  <= int(ENCODE_ROW_MAX)))
    assert(np.all(filenum >= 0))
    assert(np.all(filenum <= int(ENCODE_FILE_MAX)))

    # This should be a 64 bit integer.
    encoded_value = (np.asarray(filenum,dtype=np.uint64) << ENCODE_ROW_END) + np.asarray(rownum, dtype=np.uint64)

    # Note return signed
    return np.asarray(encoded_value, dtype=np.int64)

def decode_rownum_filenum(encoded_values):
    """Inverts encode_rownum_filenum to obtain row number and file number.

    Parameters
    ----------
    encoded_values(s) : int64 ndarray

    Returns
    -------
    filenum : str
        File number.
    rownum : int
        Row number.

    """
    filenum = (np.asarray(encoded_values,dtype=np.uint64) & ENCODE_FILE_MASK) >> ENCODE_ROW_END
    rownum  = (np.asarray(encoded_values,dtype=np.uint64) & ENCODE_ROW_MASK)
    return rownum, filenum

def make_mockid(objid, n_per_file):
    """
    Computes mockid from row and file IDs.

    Parameters
    ----------
    objid : int array
        Row identification number.
    n_per_file : int list
        Number of items per file that went into objid.

    Returns
    -------
    mockid : int array
        Encoded row and file ID.

    """
    n_files = len(n_per_file)
    n_obj = len(objid)

    n_p_file = np.array(n_per_file)
    n_per_file_cumsum = n_p_file.cumsum()

    filenum = np.zeros(n_obj, dtype='int64')
    for n in range(1, n_files):
        filenum[n_per_file_cumsum[n-1]:n_per_file_cumsum[n]] = n

    return encode_rownum_filenum(objid, filenum)

def read_100pc(mock_dir_name, target_name='STAR', rand=None, bricksize=0.25,
               healpixels=None, nside=None, magcut=None, nproc=None, lya=None):
    """Read a single-file GUMS-based mock of nearby (d<100 pc) normal stars (i.e.,
    no white dwarfs).

    Parameters
    ----------
    mock_dir_name : str
        Complete path and filename of the mock catalog.
    target_name : str
        Target name (not used; defaults to `STAR`).
    rand : numpy.RandomState
        RandomState object used for the random number generation.
    bricksize : float
        Size of each brick in deg.
    healpixels : numpy.ndarray or numpy.int64
        Restrict the sample to read objects within this list of healpix pixel numbers.
    nside : int
        Healpix resolution for input healpixels.
    magcut : float
        Magnitude cut to apply to the sample (not used here).
    nproc : int
        Number of cores to use for reading (not used here).

    Returns
    -------
    Dictionary with the following entries.
        'OBJID' : int64 numpy.ndarray
            Object identification number for each file in mock_dir_name.
        'MOCKID': int numpy.ndarray
            Unique mock identification number.
        'RA': numpy.ndarray
            RA positions for the objects in the mock.
        'DEC' : numpy.ndarray
            DEC positions for the objects in the mock.
        'Z' : numpy.ndarray
            Heliocentric radial velocity divided by the speed of light.
        'BRICKNAME' : str numpy.ndarray
            Brick name assigned according to RA, Dec coordinates.
        'SEED' : int numpy.ndarray
            Random seed used in the template-generating code.
        'MAG': numpy.ndarray
            Apparent magnitude in SDSS g-band(???).
        'TEFF': numpy.ndarray
            Effective stellar temperature (K).
        'LOGG': numpy.ndarray
            Surface gravity (cm/s**2).
        'FEH': numpy.ndarray
            Logarithmic iron abundance relative to solar.
        'FILTERNAME': str
            Filter name corresponding to mag (used to normalize the spectra).
        'TRUESPECTYPE': str
            Set to `STAR` for this whole sample.
        'TEMPLATETYPE': str
            Set to `STAR` for this whole sample.
        'TEMPLATESUBTYPE': numpy.ndarray
            Spectral class for each object (e.g., GV) based on the GUMS mock.
        'FILES': str list
            List of all mock file(s) read.
        'N_PER_FILE': int list
            Number of mock targets per file.

    """
    mockfile = mock_dir_name
    try:
        os.stat(mockfile)
    except:
        log.fatal('Mock file {} not found!'.format(mockfile))
        raise IOError

    # Read the ra,dec coordinates, generate mockid, and then restrict to the
    # desired healpixels.
    radec = fitsio.read(mockfile, columns=['RA', 'DEC'], upper=True, ext=1)
    nobj = len(radec)

    files = list()
    n_per_file = list()
    files.append(mockfile)
    n_per_file.append(nobj)

    objid = np.arange(nobj, dtype='i8')
    mockid = make_mockid(objid, n_per_file)

    log.info('Assigning healpix pixels with nside = {}'.format(nside))
    allpix = radec2pix(nside, radec['RA'], radec['DEC'])
    cut = np.where( np.in1d(allpix, healpixels)*1 )[0]

    nobj = len(cut)
    if nobj == 0:
        log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
        return dict()
    else:
        log.info('Trimmed to {} {}s in healpixels {}'.format(nobj, target_name, healpixels))

    objid = objid[cut]
    mockid = mockid[cut]
    ra = radec['RA'][cut].astype('f8') % 360.0 # enforce 0 < ra < 360
    dec = radec['DEC'][cut].astype('f8')
    del radec

    cols = ['RADIALVELOCITY', 'MAGG', 'TEFF', 'LOGG', 'FEH', 'SPECTRALTYPE']
    data = fitsio.read(mockfile, columns=cols, upper=True, ext=1, rows=cut)
    zz = (data['RADIALVELOCITY'] / C_LIGHT).astype('f4')
    mag = data['MAGG'].astype('f4') # SDSS g-band
    teff = data['TEFF'].astype('f4')
    logg = data['LOGG'].astype('f4')
    feh = data['FEH'].astype('f4')
    templatesubtype = data['SPECTRALTYPE']

    brickname = get_brickname_from_radec(ra, dec, bricksize=bricksize)
    seed = rand.randint(2**32, size=nobj)

    return {'OBJID': objid, 'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'Z': zz,
            'BRICKNAME': brickname, 'SEED': seed, 'MAG': mag, 'TEFF': teff, 'LOGG': logg, 'FEH': feh,
            'FILTERNAME': 'sdss2010-g', # ?????
            'TRUESPECTYPE': 'STAR', 'TEMPLATETYPE': 'STAR', 'TEMPLATESUBTYPE': templatesubtype,
            'FILES': files, 'N_PER_FILE': n_per_file}

def read_wd(mock_dir_name, target_name='WD', rand=None, bricksize=0.25,
               healpixels=None, nside=None, magcut=None, nproc=None, lya=None):
    """Read a single-file GUMS-based mock of white dwarfs.

    Parameters
    ----------
    mock_dir_name : str
        Complete path and filename of the mock catalog.
    target_name : str
        Target name (not used; defaults to `WD`).
    rand : numpy.RandomState
        RandomState object used for the random number generation.
    bricksize : float
        Size of each brick in deg.
    healpixels : numpy.ndarray or numpy.int64
        Restrict the sample to read objects within this list of healpix pixel numbers.
    nside : int
        Healpix resolution for input healpixels.
    magcut : float
        Magnitude cut to apply to the sample (not used here).
    nproc : int
        Number of cores to use for reading (not used here).

    Returns
    -------
    Dictionary with the following entries.
        'OBJID' : int64 numpy.ndarray
            Object identification number for each file in mock_dir_name.
        'MOCKID': int numpy.ndarray
            Unique mock identification number.
        'RA': numpy.ndarray
            RA positions for the objects in the mock.
        'DEC' : numpy.ndarray
            DEC positions for the objects in the mock.
        'Z' : numpy.ndarray
            Heliocentric radial velocity divided by the speed of light.
        'BRICKNAME' : str numpy.ndarray
            Brick name assigned according to RA, Dec coordinates.
        'SEED' : int numpy.ndarray
            Random seed used in the template-generating code.
        'MAG': numpy.ndarray
            Apparent magnitude in the SDSS g-band.
        'TEFF': numpy.ndarray
            Effective stellar temperature (K).
        'LOGG': numpy.ndarray
            Surface gravity (cm/s**2).
        'FILTERNAME': str
            Filter name corresponding to mag (used to normalize the spectra).
        'TRUESPECTYPE': str
            Set to `STAR` for this whole sample.
        'TEMPLATETYPE': str
            Set to `WD` for this whole sample.
        'TEMPLATESUBTYPE': numpy.ndarray
            Spectral class for each object (DA vs DB) based on the GUMS mock.
        'FILES': str list
            List of all mock file(s) read.
        'N_PER_FILE': int list
            Number of mock targets per file.

    """
    mockfile = mock_dir_name
    try:
        os.stat(mockfile)
    except:
        log.fatal('Mock file {} not found!'.format(mockfile))
        raise IOError

    # Read the ra,dec coordinates, generate mockid, and then restrict to the
    # desired healpixels.
    radec = fitsio.read(mockfile, columns=['RA', 'DEC'], upper=True, ext=1)
    nobj = len(radec)

    files = list()
    n_per_file = list()
    files.append(mockfile)
    n_per_file.append(nobj)

    objid = np.arange(nobj, dtype='i8')
    mockid = make_mockid(objid, n_per_file)

    log.info('Assigning healpix pixels with nside = {}'.format(nside))
    allpix = radec2pix(nside, radec['RA'], radec['DEC'])
    cut = np.where( np.in1d(allpix, healpixels)*1 )[0]

    nobj = len(cut)
    if nobj == 0:
        log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
        return dict()
    else:
        log.info('Trimmed to {} {}s in healpixels {}'.format(nobj, target_name, healpixels))

    objid = objid[cut]
    mockid = mockid[cut]
    ra = radec['RA'][cut].astype('f8') % 360.0 # enforce 0 < ra < 360
    dec = radec['DEC'][cut].astype('f8')
    del radec

    cols = ['RADIALVELOCITY', 'G_SDSS', 'TEFF', 'LOGG', 'SPECTRALTYPE']
    data = fitsio.read(mockfile, columns=cols, upper=True, ext=1, rows=cut)
    zz = (data['RADIALVELOCITY'] / C_LIGHT).astype('f4')
    mag = data['G_SDSS'].astype('f4') # SDSS g-band
    teff = data['TEFF'].astype('f4')
    logg = data['LOGG'].astype('f4')
    templatesubtype = np.char.upper(data['SPECTRALTYPE'].astype('<U'))

    brickname = get_brickname_from_radec(ra, dec, bricksize=bricksize)
    seed = rand.randint(2**32, size=nobj)

    return {'OBJID': objid, 'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'Z': zz,
            'BRICKNAME': brickname, 'SEED': seed, 'MAG': mag, 'TEFF': teff, 'LOGG': logg,
            'FILTERNAME': 'sdss2010-g',
            'TRUESPECTYPE': 'STAR', 'TEMPLATETYPE': 'WD', 'TEMPLATESUBTYPE': templatesubtype,
            'FILES': files, 'N_PER_FILE': n_per_file}

def _sample_vdisp(logvdisp_meansig, nmodel=1, rand=None):
    """Choose a subset of velocity dispersions."""
    if rand is None:
        rand = np.random.RandomState()

    #fracvdisp = (0.1, 40)
    fracvdisp = (0.1, 1)

    nvdisp = int(np.max( ( np.min( ( np.round(nmodel * fracvdisp[0]), fracvdisp[1] ) ), 1 ) ))
    vvdisp = 10**rand.normal(logvdisp_meansig[0], logvdisp_meansig[1], nvdisp)
    vdisp = rand.choice(vvdisp, nmodel)

    return vdisp

def read_gaussianfield(mock_dir_name, target_name, rand=None, bricksize=0.25,
                       healpixels=None, nside=None, magcut=None, nproc=None, lya=None):
    """Reads the GaussianRandomField mocks for ELGs, LRGs, and QSOs.

    Parameters
    ----------
    mock_dir_name : str
        Complete top-level path to the mock catalogs or the mock filename (for SKY).
    target_name : str
        Target name specifying the mock catalog to read ('LRG', 'ELG', 'QSO', or
        'SKY').
    rand : numpy.RandomState
        RandomState object used for the random number generation.
    bricksize : float
        Size of each brick in deg.
    healpixels : numpy.ndarray or numpy.int64
        Restrict the sample to read objects within this list of healpix pixel numbers.
    nside : int
        Healpix resolution for input healpixels.
    magcut : float
        Magnitude cut to apply to the sample (not used here).
    nproc : int
        Number of cores to use for reading (not used here).
    lya : dictionary
        Information on the Lyman-alpha mock to read.

    Returns
    -------
    Dictionary with the following basic entries (for SKY).
        'OBJID' : int64 numpy.ndarray
            Object identification number for each file in mock_dir_name.
        'MOCKID': int numpy.ndarray
            Unique mock identification number.
        'RA': numpy.ndarray
            RA positions for the objects in the mock.
        'DEC' : numpy.ndarray
            DEC positions for the objects in the mock.
        'Z' : numpy.ndarray
            Heliocentric redshift (equal to zero for SKY).
        'BRICKNAME' : str numpy.ndarray
            Brick name assigned according to RA, Dec coordinates.
        'SEED' : int numpy.ndarray
            Random seed used in the template-generating code.
        'TRUESPECTYPE': str
            Set to one of SKY, GALAXY (for ELG and LRG), or QSO.
        'TEMPLATETYPE': str
            Set to one of SKY, ELG, LRG, or QSO.
        'TEMPLATESUBTYPE': numpy.ndarray
            Not used for now (empty string for all target names).
        'FILES': str list
            List of all mock file(s) read.
        'N_PER_FILE': int list
            Number of mock targets per file.

    The target names ELG, LRG, and QSO have the following additional/optional
    keys.
        'GR': numpy.ndarray
            Apparent g-r color (only for ELG, QSO).
        'RZ': numpy.ndarray
            Apparent r-z color
        'RW1': numpy.ndarray
            Apparent r-W1 color (only for LRG).
        'W1W2': numpy.ndarray
            Apparent W1-W2 color (only for QSO).
        'MAG': numpy.ndarray
            Apparent magnitude in the DECam r-, z-, or g-band (for ELG, LRG, QSO, resp.)
        'FILTERNAME': str
            Filter name corresponding to mag (used to normalize the spectra).
        'VDISP': numpy.ndarray
            Velocity dispersion (km/s) (only for ELG, LRG).

    """
    mockfile = mock_dir_name
    try:
        os.stat(mockfile)
    except:
        log.fatal('Mock file {} not found!'.format(mockfile))
        raise IOError

    # Read the ra,dec coordinates, generate mockid, and then restrict to the
    # desired healpixels.
    radec = fitsio.read(mockfile, columns=['RA', 'DEC'], upper=True, ext=1)
    nobj = len(radec)

    files = list()
    n_per_file = list()
    files.append(mockfile)
    n_per_file.append(nobj)

    objid = np.arange(nobj, dtype='i8')
    mockid = make_mockid(objid, n_per_file)

    log.info('Assigning healpix pixels with nside = {}'.format(nside))
    allpix = radec2pix(nside, radec['RA'], radec['DEC'])
    cut = np.where( np.in1d(allpix, healpixels)*1 )[0]

    nobj = len(cut)
    if nobj == 0:
        log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
        return dict()
    else:
        log.info('Trimmed to {} {}s in healpixels {}'.format(nobj, target_name, healpixels))

    objid = objid[cut]
    mockid = mockid[cut]
    ra = radec['RA'][cut].astype('f8') % 360.0 # enforce 0 < ra < 360
    dec = radec['DEC'][cut].astype('f8')
    del radec
        
    if target_name != 'SKY':
        data = fitsio.read(mockfile, columns=['Z_COSMO', 'DZ_RSD'], upper=True, ext=1, rows=cut)
        zz = (data['Z_COSMO'].astype('f8') + data['DZ_RSD'].astype('f8')).astype('f4')
        mag = np.zeros_like(zz) - 1 # placeholder
        del data

    if target_name == 'QSO':
        nobj_qso = nobj
        lyafiles_qso = np.repeat('', nobj_qso)
        lyahdu_qso = np.repeat([-1], nobj_qso)
        templatetype_qso = np.repeat('QSO', nobj_qso)
        templatesubtype_qso = np.repeat('', nobj_qso)

    # Combine the QSO and Lyman-alpha samples.
    if target_name == 'QSO' and lya:
        log.info('  Adding Lya QSOs.')
        
        new_format = ("nside" in lya)
        
        if new_format :
            log.info('  Using new format with 2D images in healpix files')
        else :
            log.info('  Using old format with 1 HDU per skewer')
        
        mockfile_lya = lya['mock_dir_name']
        try:
            os.stat(mockfile_lya)
        except:
            log.fatal('Mock file {} not found!'.format(mockfile_lya))
            raise IOError
        
        if new_format:
            tmp         = fitsio.read(mockfile_lya, columns=['RA', 'DEC', 'MOCKID' ,'Z','PIXNUM'],
                                      upper=True, ext=1)
            ra_lya      = tmp['RA'].astype('f8') % 360.0 # enforce 0 < ra < 360
            dec_lya     = tmp['DEC'].astype('f8')            
            zz_lya      = tmp['Z'].astype('f4')
            objid_lya   = (tmp['MOCKID'].astype(float)).astype(int) # will change
            mockpix_lya = tmp['PIXNUM']
            mockid_lya  = objid_lya.copy() # the logic to read the spectra is in spectra.py, not here            
            #log.warning("FIXED gmag=22 for now, that's where we should use a QSO luminosity function")
            #mag_lya     = 22.*np.ones(zz_lya.size).astype('f4') # g-band
            mag_lya = 22.*np.ones(zz_lya.size).astype('f4') # g-band magnitude place-holder
            del tmp
        else:
            
            tmp = fitsio.read(mockfile_lya, columns=['RA', 'DEC', 'MOCKFILEID', 'MOCKHDUNUM', 'MAG_G', 'Z'],
                              upper=True, ext=1)
            nobj_lya  = len(tmp)
            
            objid_lya = np.arange(nobj_lya, dtype='i8')
            n_per_file.append(nobj_lya)
            mockid_lya = make_mockid(objid_lya, [n_per_file[1]])
            zz_lya      = tmp['Z'].astype('f4')
            ra_lya      = tmp['RA'].astype('f8') % 360.0 # enforce 0 < ra < 360
            dec_lya     = tmp['DEC'].astype('f8')
            
            mag_lya     = tmp['MAG_G'].astype('f4')
            mockfileid_lya = tmp['MOCKFILEID']
            hdu_lya        = tmp['MOCKHDUNUM']
            del tmp
            
            lyainfo   = fitsio.read(mockfile_lya, upper=True, ext=2)
            files_lya = lyainfo['MOCKFILE'][mockfileid_lya]            
        
        # apply sky cut
        allpix = radec2pix(nside, ra_lya, dec_lya)
        cut = np.where( np.in1d(allpix, healpixels)*1 )[0]

        nobj_lya = len(cut)
        if nobj_lya == 0:
            log.warning('No Lya QSOs in healpixels {}!'.format(healpixels))
            lyafiles = lyafiles_qso
            lyahdu = lyahdu_qso
            templatetype = templatetype_qso
            templatesubtype = templatesubtype_qso
        else:
            log.info('Trimmed to {} Lya QSO skewers in healpixels {}'.format(nobj_lya, healpixels))            
            objid_lya  = objid_lya[cut]
            mockid_lya = mockid_lya[cut]
            ra_lya     = ra_lya[cut]
            dec_lya    = dec_lya[cut]
            zz_lya     = zz_lya[cut]
            mag_lya    = mag_lya[cut] 
            
            # adding file path
            if new_format :
                mockpix_lya = mockpix_lya[cut]
                mockdir = os.path.dirname(mockfile_lya)
                mocknside = lya['nside']
                lyafiles = []
                for mpix in mockpix_lya:
                    lyafiles.append("%s/%d/%d/transmission-%d-%d.fits"%(mockdir,mpix/100,mpix,mocknside,mpix))
                lyafiles = np.hstack((lyafiles_qso, lyafiles))
                
            else:
                mockdir = os.path.dirname(mockfile_lya)
                lyafiles = np.hstack( (lyafiles_qso, np.array([os.path.join(
                    mockdir, ff.decode('utf-8')) for ff in files_lya[cut]])) )
                lyahdu = np.hstack( (lyahdu_qso, hdu_lya[cut]) )
                        
            # Join the QSO + Lya samples
            ra = np.hstack((ra, ra_lya))
            dec = np.hstack((dec, dec_lya))
            zz  = np.hstack((zz, zz_lya))
            mag = np.hstack((mag, mag_lya))
            objid = np.hstack((objid, objid_lya))
            mockid = np.hstack((mockid, mockid_lya))
            nobj = len(ra)
            
            templatetype = np.hstack( (templatetype_qso, np.repeat('QSO', nobj_lya)) )
            templatesubtype = np.hstack( (np.repeat('', nobj_qso), np.repeat('LYA', nobj_lya)) )
            
        log.info('The combined QSO sample has {} targets.'.format(nobj))
        
    brickname = get_brickname_from_radec(ra, dec, bricksize=bricksize)
    seed = rand.randint(2**32, size=nobj)

    # Create a basic dictionary for SKY.
    out = {'OBJID': objid, 'MOCKID': mockid, 'RA': ra, 'DEC': dec, 
           'BRICKNAME': brickname, 'SEED': seed, 'FILES': files,
           'N_PER_FILE': n_per_file}

    # Assign magnitudes / colors based on the appropriate Gaussian mixture model.
    if target_name == 'SKY':
        out.update({'TRUESPECTYPE': 'SKY', 'TEMPLATETYPE': 'SKY', 'TEMPLATESUBTYPE': ''})
    else:
        log.info('Sampling from {} Gaussian mixture model.'.format(target_name))

        GMM = SampleGMM(random_state=rand)
        mags = GMM.sample(target_name, nobj) # [g, r, z, w1, w2, w3, w4]

        # Temporary hack to deal with the lower-than average ELG target densities.
        if False:
            if target_name == 'ELG':
                from desitarget.cuts import isELG
                niter, maxiter, fracelg = 0, 5, 0.0
                while fracelg < 0.98 and niter < maxiter:
                    gflux, rflux, zflux = [10**(0.4*(22.5-mags[b])) for b in 'grz']
                    iselg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)
                    fracelg = np.sum(iselg)/nobj
                    #print(niter, fracelg)
    
                    need = np.where(iselg == False)[0]
                    if len(need) > 0:
                        newmags = GMM.sample(target_name, len(need))
                        mags[need] = newmags
                    
                    niter = niter + 1
    
                #import matplotlib.pyplot as plt
                #plt.scatter(mags['r'] - mags['z'], mags['g'] - mags['r'])
                #plt.xlim(-0.5, 2) ; plt.ylim(-0.5, 2)
                #plt.show()
                #import pdb ; pdb.set_trace()

        out.update({'Z': zz, 'GR': mags['g']-mags['r'], 'RZ': mags['r']-mags['z'],
                    'RW1': mags['r']-mags['w1'], 'W1W2': mags['w1']-mags['w2']})

        if target_name in ('ELG', 'LRG'):
            out.update({
                'SHAPEEXP_R': mags['exp_r'], 'SHAPEEXP_E1': mags['exp_e1'], 'SHAPEEXP_E2': mags['exp_e2'], 
                'SHAPEDEV_R': mags['dev_r'], 'SHAPEDEV_E1': mags['dev_e1'], 'SHAPEDEV_E2': mags['dev_e2']
                })

        if target_name == 'ELG':
            """Selected in the r-band with g-r, r-z colors."""

            #vdisp = _sample_vdisp((1.9, 0.15), nmodel=nobj, rand=rand)
            vdisp = np.zeros(nobj)
            for bb in sorted(set(brickname)):
                these = np.where( bb == brickname )[0]
                vdisp[these] = _sample_vdisp((1.9, 0.15), nmodel=len(these), rand=rand)
            
            out.update({'TRUESPECTYPE': 'GALAXY', 'TEMPLATETYPE': 'ELG', 'TEMPLATESUBTYPE': '',
                        'VDISP': vdisp, 'MAG': mags['r'], 'FILTERNAME': 'decam2014-r'})

        elif target_name == 'LRG':
            """Selected in the z-band with r-z, r-W1 colors."""

            #vdisp = _sample_vdisp((2.3, 0.1), nmodel=nobj, rand=rand)
            vdisp = np.zeros(nobj)
            for bb in sorted(set(brickname)):
                these = np.where( bb == brickname )[0]
                vdisp[these] = _sample_vdisp((2.3, 0.1), nmodel=len(these), rand=rand)
                
            out.update({'TRUESPECTYPE': 'GALAXY', 'TEMPLATETYPE': 'LRG', 'TEMPLATESUBTYPE': '',
                        'VDISP': vdisp, 'MAG': mags['z'], 'FILTERNAME': 'decam2014-z'})

        elif target_name == 'QSO':
            """Selected in the r-band with g-r, r-z, and W1-W2 colors."""
            replace = np.where(mag == -1)[0]
            if len(replace) > 0:
                mag[replace] = mags['g'][replace] # g-band

            if new_format :
                out.update({
                    'TRUESPECTYPE': 'QSO', 'TEMPLATETYPE': templatetype, 'TEMPLATESUBTYPE': templatesubtype, 
                    'LYAFILES': lyafiles,
                    'MAG': mag, 'FILTERNAME': 'decam2014-g'}) # Lya is normalized in the g-band
            else :
                out.update({
                    'TRUESPECTYPE': 'QSO', 'TEMPLATETYPE': templatetype, 'TEMPLATESUBTYPE': templatesubtype, 
                    #'TRUESPECTYPE': truespectype, 'TEMPLATETYPE': templatetype, 'TEMPLATESUBTYPE': templatesubtype,
                    'LYAFILES': lyafiles, 'LYAHDU': lyahdu, 
                    'MAG': mag, 'FILTERNAME': 'decam2014-g'}) # Lya is normalized in the g-band

        else:
            log.fatal('Unrecognized target type {}!'.format(target_name))
            raise ValueError

    return out

def read_durham_mxxl_hdf5(mock_dir_name, target_name='BGS', rand=None, bricksize=0.25,
                          healpixels=None, nside=None, magcut=None, nproc=None, lya=None):
    """ Reads the MXXL mock of BGS galaxies.

    Parameters
    ----------
    mock_dir_name : str
        Complete path and filename of the mock catalog.
    target_name : str
        Target name (not used; defaults to `BGS`).
    rand : numpy.RandomState
        RandomState object used for the random number generation.
    bricksize : float
        Size of each brick in deg.
    healpixels : numpy.ndarray or numpy.int64
        Restrict the sample to read objects within this list of healpix pixel numbers.
    nside : int
        Healpix resolution for input healpixels.
    magcut : float
        Magnitude cut to apply to the sample (not used here).
    nproc : int
        Number of cores to use for reading (not used here).

    Returns
    -------
    Dictionary with the following entries.
        'OBJID' : int64 numpy.ndarray
            Object identification number for each file in mock_dir_name.
        'MOCKID': int numpy.ndarray
            Unique mock identification number.
        'RA': numpy.ndarray
            RA positions for the objects in the mock.
        'DEC' : numpy.ndarray
            DEC positions for the objects in the mock.
        'Z' : numpy.ndarray
            Heliocentric redshift.
        'BRICKNAME' : str numpy.ndarray
            Brick name assigned according to RA, Dec coordinates.
        'SEED' : int numpy.ndarray
            Random seed used in the template-generating code.
        'MAG': numpy.ndarray
            Apparent magnitude in the SDSS r-band.
        'VDISP': numpy.ndarray
            Velocity dispersion (km/s).
        'SDSS_absmag_r01' : numpy.ndarray
            Absolute SDSS r-band magnitude band-shifted to z=0.1.
        'SDSS_01gr' : numpy.ndarray
            SDSS g-r color band-shifted to z=0.1
        'FILTERNAME': str
            Filter name corresponding to mag (used to normalize the spectra).
        'TRUESPECTYPE': str
            Set to `GALAXY` for this whole sample.
        'TEMPLATETYPE': str
            Set to `BGS` for this whole sample.
        'TEMPLATESUBTYPE': numpy.ndarray
            Not used for now (empty string for all target names).
        'FILES': str list
            List of all mock file(s) read.
        'N_PER_FILE': int list
            Number of mock targets per file.


    """
    import h5py

    mockfile = mock_dir_name
    try:
        os.stat(mockfile)
    except:
        log.fatal('Mock file {} not found!'.format(mockfile))
        raise IOError

    # Read the ra,dec coordinates, generate mockid, and then restrict to the
    # desired healpixels.
    f = h5py.File(mockfile)
    ra  = f['Data/ra'][...].astype('f8') % 360.0 # enforce 0 < ra < 360
    dec = f['Data/dec'][...].astype('f8')
    nobj = len(ra)

    files = list()
    files.append(mockfile)
    n_per_file = list()
    n_per_file.append(nobj)

    objid = np.arange(nobj, dtype='i8')
    mockid = make_mockid(objid, n_per_file)

    log.info('Assigning healpix pixels with nside = {}'.format(nside))
    allpix = radec2pix(nside, ra, dec)
    these = np.in1d(allpix, healpixels)
    cut = np.where( these*1 )[0]

    nobj = len(cut)
    if nobj == 0:
        log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
        return dict()
    else:
        log.info('Trimmed to {} {}s in healpixels {}'.format(nobj, target_name, healpixels))

    objid = objid[cut]
    mockid = mockid[cut]
    ra = ra[cut]
    dec = dec[cut]
    
    zz = f['Data/z_obs'][these].astype('f4')
    rmag = f['Data/app_mag'][these].astype('f4')
    absmag = f['Data/abs_mag'][these].astype('f4')
    gr = f['Data/g_r'][these].astype('f4')
    f.close()

    if magcut is not None:
        cut = rmag < magcut
        if np.count_nonzero(cut) == 0:
            log.warning('No objects with r < {}!'.format(magcut))
            return dict()
        else:
            objid = objid[cut]
            mockid = mockid[cut]
            ra = ra[cut]
            dec = dec[cut]
            zz = zz[cut]
            rmag = rmag[cut]
            absmag = absmag[cut]
            gr = gr[cut]
            nobj = len(ra)
            log.info('Trimmed to {} {}s with r < {}.'.format(nobj, target_name, magcut))

    brickname = get_brickname_from_radec(ra, dec, bricksize=bricksize)
    seed = rand.randint(2**32, size=nobj)

    vdisp = np.zeros(nobj)
    for bb in sorted(set(brickname)):
        these = np.where( bb == brickname )[0]
        vdisp[these] = _sample_vdisp((1.9, 0.15), nmodel=len(these), rand=rand)

    return {'OBJID': objid, 'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'Z': zz,
            'BRICKNAME': brickname, 'SEED': seed, 'MAG': rmag, 'VDISP': vdisp,
            'SDSS_absmag_r01': absmag, 'SDSS_01gr': gr, 'FILTERNAME': 'sdss2010-r',
            'TRUESPECTYPE': 'GALAXY', 'TEMPLATETYPE': 'BGS', 'TEMPLATESUBTYPE': '',
            'FILES': files, 'N_PER_FILE': n_per_file}

def _load_galaxia_file(args):
    return load_galaxia_file(*args)

def load_galaxia_file(target_name, mockfile, healpixels, nside):
    """Multiprocessing support routine for read_galaxia.  Read each individual mock
    galaxia file.

    """
    try:
        os.stat(mockfile)
    except:
        log.fatal('Mock file {} not found!'.format(mockfile))
        raise IOError

    log.info('  Reading {}'.format(mockfile))
    radec = fitsio.read(mockfile, columns=['RA', 'DEC'], upper=True, ext=1)
    nobj = len(radec)

    files = list()
    n_per_file = list()
    files.append(mockfile)
    n_per_file.append(nobj)

    objid = np.arange(nobj, dtype='i8')
    mockid = make_mockid(objid, n_per_file)

    allpix = radec2pix(nside, radec['RA'], radec['DEC'])
    cut = np.where( np.in1d(allpix, healpixels)*1 )[0]

    nobj = len(cut)
    if nobj == 0:
        return dict()

    objid = objid[cut]
    mockid = mockid[cut]
    ra = radec['RA'][cut].astype('f8') % 360.0 # enforce 0 < ra < 360
    dec = radec['DEC'][cut].astype('f8')
    del radec

    cols = ['V_HELIO',
            'SDSSU_TRUE_NODUST', 'SDSSG_TRUE_NODUST', 'SDSSR_TRUE_NODUST', 'SDSSI_TRUE_NODUST', 'SDSSZ_TRUE_NODUST',
            'SDSSR_OBS', 'TEFF', 'LOGG', 'FEH']
    data = fitsio.read(mockfile, columns=cols, upper=True, ext=1, rows=cut)
    zz = (data['V_HELIO'].astype('f4') / C_LIGHT).astype('f4')
    mag = data['SDSSR_TRUE_NODUST'].astype('f4') # SDSS r-band, extinction-corrected
    mag_obs = data['SDSSR_OBS'].astype('f4')     # SDSS r-band, observed
    teff = 10**data['TEFF'].astype('f4')         # log10!
    logg = data['LOGG'].astype('f4')
    feh = data['FEH'].astype('f4')

    def select_sdss_std(umag, gmag, rmag, imag, zmag, obs_rmag=None):
        """Select standard stars using SDSS photometry and the BOSS selection.
        
        According to http://www.sdss.org/dr12/algorithms/boss_std_ts the r-band
        magnitude for the magnitude cuts is the extinction corrected magnitude.
    
        """
        umg_cut = ((umag - gmag) - 0.82)**2
        gmr_cut = ((gmag - rmag) - 0.30)**2
        rmi_cut = ((rmag - imag) - 0.09)**2
        imz_cut = ((imag - zmag) - 0.02)**2
    
        is_std = np.sqrt((umg_cut + gmr_cut + rmi_cut + imz_cut)) < 0.08
    
        if obs_rmag is not None:
            is_std &= (15.0 < obs_rmag) & (obs_rmag < 19)
        
        return is_std

    # Use extinction-corrected SDSS mags 
    istd = select_sdss_std(data['SDSSU_TRUE_NODUST'], data['SDSSG_TRUE_NODUST'],
                           data['SDSSR_TRUE_NODUST'], data['SDSSI_TRUE_NODUST'],
                           data['SDSSZ_TRUE_NODUST'], obs_rmag=None)
    
    return {'OBJID': objid, 'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'BOSS_STD': istd, 
            'Z': zz, 'MAG': mag, 'MAG_OBS': mag_obs, 'TEFF': teff,
            'LOGG': logg, 'FEH': feh, 'FILES': files, 'N_PER_FILE': n_per_file}

def read_galaxia(mock_dir_name, target_name='STAR', rand=None, bricksize=0.25,
                 healpixels=None, nside=None, magcut=None, nproc=1, lya=None):
    """ Read and concatenate the MWS_MAIN mock files.

    Parameters
    ----------
    mock_dir_name : str
        Complete top-level path to the mock catalogs.
    target_name : str
        Target name (not used; defaults to `STAR`).
    rand : numpy.RandomState
        RandomState object used for the random number generation.
    bricksize : float
        Size of each brick in deg.
    healpixels : numpy.ndarray or numpy.int64
        Restrict the sample to read objects within this list of healpix pixel numbers.
    nside : int
        Healpix resolution for input healpixels.
    magcut : float
        Magnitude cut to apply to the sample (not used here).
    nproc : int
        Number of cores to use for reading (default 1).

    Returns
    -------
    Dictionary with the following entries.
        'OBJID' : int64 numpy.ndarray
            Object identification number for each file in mock_dir_name.
        'MOCKID': int numpy.ndarray
            Unique mock identification number.
        'RA': numpy.ndarray
            RA positions for the objects in the mock.
        'DEC' : numpy.ndarray
            DEC positions for the objects in the mock.
        'Z' : numpy.ndarray
            Heliocentric radial velocity divided by the speed of light.
        'BRICKNAME' : str numpy.ndarray
            Brick name assigned according to RA, Dec coordinates.
        'SEED' : int numpy.ndarray
            Random seed used in the template-generating code.
        'MAG': numpy.ndarray
            Apparent magnitude (extinction-corrected) in SDSS r-band.
        'MAG_OBS': numpy.ndarray
            Apparent magnitude (including extinction) in SDSS r-band.
        'TEFF': numpy.ndarray
            Effective stellar temperature (K).
        'LOGG': numpy.ndarray
            Surface gravity (cm/s**2).
        'FEH': numpy.ndarray
            Logarithmic iron abundance relative to solar.
        'FILTERNAME': str
            Filter name corresponding to mag (used to normalize the spectra).
        'TRUESPECTYPE': str
            Set to `STAR` for this whole sample.
        'TEMPLATETYPE': str
            Set to `STAR` for this whole sample.
        'TEMPLATESUBTYPE': numpy.ndarray
            Spectral class for each object (e.g., GV) based on the GUMS mock.
        'FILES': str list
            List of all mock file(s) read.
        'N_PER_FILE': int list
            Number of mock targets per file.

    """
    import multiprocessing
    import healpy as hp
    
    # Figure out which mock files to read based on the input healpixels.
    brickfile = os.path.join(mock_dir_name, 'bricks.fits')
    try:
        os.stat(brickfile)
    except:
        log.fatal('Brick information file {} not found!'.format(brickfile))
        raise IOError

    hdr = fitsio.read_header(brickfile, ext=0)
    brickinfo = fitsio.read(brickfile, extname='BRICKS', upper=True,
                            columns=['BRICKNAME', 'RA', 'DEC'])

    radius = np.sqrt(2) * np.radians(hdr['BRICKSIZ']) / 2
    theta, phi = np.radians(90-brickinfo['DEC']), np.radians(brickinfo['RA'])
    vec = hp.ang2vec(theta, phi)
    ipix = [hp.query_disc(nside, vec[i], radius=radius, inclusive=True,
                          nest=True) for i in range(len(brickinfo))]

    these = []
    for ii, thesepix in enumerate(ipix):
        if np.count_nonzero( np.in1d(thesepix, healpixels) ) > 0:
            these.append(ii)
    these = np.unique(np.array(these))

    if len(these) == 0:
        log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
        return dict()

    if target_name.upper() == 'FAINTSTAR':
        suffix = '_superfaint'
    else:
        suffix = ''
        
    file_list = []
    bricks = brickinfo['BRICKNAME'][these]
    for bb in bricks:
        bb = bb.decode('utf-8') # This will probably break in Python2 ??
        ff = os.path.join(mock_dir_name, 'bricks', '???', bb,
                          'allsky_galaxia{}_desi_{}.fits'.format(suffix, bb))
        if len(glob(ff)) == 1:
            file_list.append(glob(ff))
        else:
            log.warning('Missing file {}'.format(ff))

    nfiles = len(file_list)

    if nfiles == 0:
        log.warning('No files found in {}!'.format(mock_dir_name))
        return dict()
    
    file_list = list( np.concatenate(file_list) )
    
    # Multiprocess the I/O
    mpargs = list()
    for ff in file_list:
        mpargs.append((target_name, ff, healpixels, nside))
        
    if nproc > 1:
        p = multiprocessing.Pool(nproc)
        data1 = p.map(_load_galaxia_file, mpargs)
        p.close()
    else:
        data1 = list()
        for args in mpargs:
            data1.append(_load_galaxia_file(args))

    # Remove empty dictionaries and then consolidate.
    data = dict()
    data1 = [dd for dd in data1 if dd]
    if len(data1) == 0:
        log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
        return data

    for k in data1[0].keys():
        data[k] = np.concatenate([dd[k] for dd in data1])
    del data1

    objid = data['OBJID']
    mockid = data['MOCKID']
    ra = data['RA']
    dec = data['DEC']
    boss_std = data['BOSS_STD']
    zz = data['Z']
    mag = data['MAG']
    mag_obs = data['MAG_OBS']
    teff = data['TEFF']
    logg = data['LOGG']
    feh = data['FEH']
    files = data['FILES']
    n_per_file = data['N_PER_FILE']
    nobj = len(ra)
    del data

    log.info('Read {} {}s from {} files in healpixels {}.'.format(nobj, target_name, nfiles, healpixels))

    # Debugging plot that I would like to keep here for now.
    if False:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        brickinfo = fitsio.read(brickfile, extname='BRICKS', upper=True,
                                columns=['BRICKNAME', 'RA', 'DEC', 'RA1', 'RA2', 'DEC1', 'DEC2'])
        fig, ax = plt.subplots()
        ax.scatter(ra, dec, alpha=0.2)
        for pix in healpixels:
            corners = hp.boundaries(nside, pix, step=1, nest=True)
            corner_theta, corner_phi = hp.vec2ang(corners.T)
            corner_ra, corner_dec = np.degrees(corner_phi), np.degrees(np.pi/2 - corner_theta)
            min_ra, max_ra, min_dec, max_dec = corner_ra.min(), corner_ra.max(), corner_dec.min(), corner_dec.max()
            verts = np.vstack( (corner_ra, corner_dec) ).T
            ax.add_patch(Polygon(verts, fill=False, ls='--'))
        for tt in these:
            verts = [
                (brickinfo['RA1'][tt], brickinfo['DEC1'][tt]), (brickinfo['RA2'][tt], brickinfo['DEC1'][tt]),
                (brickinfo['RA2'][tt], brickinfo['DEC2'][tt]), (brickinfo['RA1'][tt], brickinfo['DEC2'][tt])
                ]
            #print(verts)
            ax.add_patch(Polygon(verts, fill=False, color='green', ls='--'))
        mm = np.arange(len(brickinfo))
        for tt in mm:
            verts = [
                (brickinfo['RA1'][tt], brickinfo['DEC1'][tt]), (brickinfo['RA2'][tt], brickinfo['DEC1'][tt]),
                (brickinfo['RA2'][tt], brickinfo['DEC2'][tt]), (brickinfo['RA1'][tt], brickinfo['DEC2'][tt])
                ]
            ax.add_patch(Polygon(verts, fill=False, color='black', ls='-', lw=2))
        ax.set_xlim(min_ra-5, max_ra+5)
        ax.set_ylim(min_dec-5, max_dec+5)
        ax.margins(0.05)
        plt.show(block=False)
        import pdb ; pdb.set_trace()

    if magcut is not None:
        cut = mag < magcut
        if np.count_nonzero(cut) == 0:
            log.warning('No objects with r < {}!'.format(magcut))
            return dict()
        else:
            mockid = mockid[cut]
            objid = objid[cut]
            ra = ra[cut]
            dec = dec[cut]
            boss_std = boss_std[cut]
            zz = zz[cut]
            mag = mag[cut]
            mag_obs = mag_obs[cut]
            teff = teff[cut]
            logg = logg[cut]
            feh = feh[cut]
            nobj = len(ra)
            log.info('Trimmed to {} {}s with r < {}.'.format(nobj, target_name, magcut))

    seed = rand.randint(2**32, size=nobj)
    brickname = get_brickname_from_radec(ra, dec, bricksize=bricksize)

    return {'OBJID': objid, 'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'BOSS_STD': boss_std, 
            'Z': zz, 'BRICKNAME': brickname, 'SEED': seed, 'MAG': mag, 'TEFF': teff, 'LOGG': logg, 'FEH': feh,
            'MAG_OBS': mag_obs, 'FILTERNAME': 'sdss2010-r',
            'TRUESPECTYPE': 'STAR', 'TEMPLATETYPE': 'STAR', 'TEMPLATESUBTYPE': '',
            'FILES': files, 'N_PER_FILE': n_per_file}

def _load_lya_file(lyafile, hdu):
    """Multiprocessing support routine to read the individual mock Lyman-alpha
    files.

    """
    try:
        os.stat(lyafile)
    except:
        log.fatal('Lya file {} not found!'.format(lyafile))
        raise IOError

    log.info('Reading HDU {} from {}.'.format(lyafile))
    
    
    h = fitsio.FITS(lyafile)
    heads = [head.read_header() for head in h]

    nn = len(heads) - 1 # the first item in heads is empty
    zz = np.zeros(nn).astype('f4')
    ra = np.zeros(nn).astype('f8')
    dec = np.zeros(nn).astype('f8')
    mag_g = np.zeros(nn).astype('f4')

    for ii in range(nn):
        zz[ii] = heads[ii+1]['ZQSO']
        ra[ii] = heads[ii+1]['RA']
        dec[ii] = heads[ii+1]['DEC']
        mag_g[ii] = heads[ii+1]['MAG_G']

    objid = np.arange(len(ra), dtype='i8')
    ra = ra * 180.0 / np.pi
    ra = ra % 360.0 #enforce 0 < ra < 360
    dec = dec * 180.0 / np.pi

    return {'OBJID': objid, 'RA': ra, 'DEC': dec, 'Z': zz, 'MAG_G': mag_g}

def read_lya(mock_dir_name, target_name='QSO', rand=None, bricksize=0.25,
             bounds=(0.0, 360.0, -90.0, 90.0), magcut=None, nproc=1,
             lya=None):
    """ Read and concatenate the LYA mock files.

    Parameters
    ----------
    mock_dir_name : str
        Complete top-level path to the mock catalogs.
    target_name : str
        Target name (not used; defaults to `QSO`).
    rand : numpy.RandomState
        RandomState object used for the random number generation.
    bricksize : float
        Size of each brick in deg.
    bounds : 4-element tuple
        Restrict the sample to bounds = (min_ra, max_ra, min_dec, max_dec).
    magcut : float
        Magnitude cut to apply to the sample (not used here).
    nproc : int
        Number of cores to use for reading (default 1).

    Returns
    -------
    Dictionary with the following entries.
        'OBJID' : int64 numpy.ndarray
            Object identification number for each file in mock_dir_name.
        'MOCKID': int numpy.ndarray
            Unique mock identification number.
        'RA': numpy.ndarray
            RA positions for the objects in the mock.
        'DEC' : numpy.ndarray
            DEC positions for the objects in the mock.
        'Z' : numpy.ndarray
            Heliocentric radial velocity divided by the speed of light.
        'BRICKNAME' : str numpy.ndarray
            Brick name assigned according to RA, Dec coordinates.
        'SEED' : int numpy.ndarray
            Random seed used in the template-generating code.
        'MAG': numpy.ndarray
            Apparent magnitude (extinction-corrected) in SDSS r-band.
        'FILTERNAME': str
            Filter name corresponding to mag (used to normalize the spectra).
        'TRUESPECTYPE': str
            Set to `QSO` for this whole sample.
        'TEMPLATETYPE': str
            Set to `QSO` for this whole sample.
        'TEMPLATESUBTYPE': numpy.ndarray
            Spectral class for each object (set to `LYA` for this whole sample).
        'FILES': str list
            List of all mock file(s) read.
        'N_PER_FILE': int list
            Number of mock targets per file.

    """
    import multiprocessing
    #nproc = max(1, multiprocessing.cpu_count() // 2)

    if False:
        iter_mock_files = iter_files(mock_dir_name, '', ext='fits.gz')
    else:
        from glob import glob
        log.warning('Temporary hack using glob because I am having problems with iter_files.')
        iter_mock_files = glob(mock_dir_name+'/*.fits.gz')

    file_list = list(iter_mock_files)
    nfiles = len(iter_mock_files)

    if nfiles == 0:
        log.fatal('Unable to find files in {}'.format(mock_dir_name))
        raise ValueError

    if nproc > 1:
        p = multiprocessing.Pool(nproc)
        target_list = p.map(_load_lya_file, file_list)
        p.close()
    else:
        target_list = list()
        for mock_file in iter_mock_files:
            target_list.append(_load_lya_file(mock_file))

    # Concatenate all the dictionaries into a single dictionary, in an order
    # determined by np.argsort applied to the base name of each path in
    # file_list.
    file_order = np.argsort([os.path.basename(x) for x in file_list])

    log.info('Combining mock files.')
    full_data   = dict()
    if len(target_list) > 0:
        for k in list(target_list[0]):  # iterate over keys
            log.info(' -- {}'.format(k))
            data_list_this_key = list()
            for itarget in file_order:  # append all the arrays corresponding to a given key
                data_list_this_key.append(target_list[itarget][k])

            full_data[k] = np.concatenate(data_list_this_key) # consolidate data dictionary

        # Count number of points per file
        k          = list(target_list[0])[0] # pick the first available column
        n_per_file = [len(target_list[itarget][k]) for itarget in file_order]
        ofile_list = [file_list[itarget] for itarget in file_order]
        #bb = [file_list[itarget] for itarget in file_order]

    objid = full_data['OBJID']
    ra = full_data['RA']
    dec = full_data['DEC']
    zz = full_data['Z']
    mag_g = full_data['MAG_G']
    nobj = len(ra)
    log.info('Read {} objects from {} mock files.'.format(nobj, nfiles))

    mockid = make_mockid(objid, n_per_file)

    if bounds is not None:
        min_ra, max_ra, min_dec, max_dec = bounds
        cut = (ra >= min_ra) * (ra <= max_ra) * (dec >= min_dec) * (dec <= max_dec)
        if np.count_nonzero(cut) == 0:
            log.fatal('No objects in range RA={}, {}, Dec={}, {}!'.format(nobj, min_ra, max_ra, min_dec, max_dec))
            raise ValueError
        objid = objid[cut]
        mockid = mockid[cut]
        ra = ra[cut]
        dec = dec[cut]
        zz = zz[cut]
        mag_g = mag_g[cut]
        nobj = len(ra)
        log.info('Trimmed to {} objects in range RA={}, {}, Dec={}, {}'.format(nobj, min_ra, max_ra, min_dec, max_dec))

    seed = rand.randint(2**32, size=nobj)
    brickname = get_brickname_from_radec(ra, dec, bricksize=bricksize)

    # Sample from the GMM to get magnitudes and colors.
    #log.info('Sampling from Gaussian mixture model.')
    #GMM = SampleGMM(random_state=rand)
    #mags = GMM.sample(target_name, nobj) # [g, r, z, w1, w2, w3, w4]

    return {'OBJID': objid, 'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'Z': zz,
            'BRICKNAME': brickname, 'SEED': seed, 'FILTERNAME': 'sdss2010-g',
            'TRUESPECTYPE': 'QSO', 'TEMPLATETYPE': 'QSO', 'TEMPLATESUBTYPE': 'LYA',
            'MAG': mag_g, 'FILES': ofile_list, 'N_PER_FILE': n_per_file}


#############
# Code previously in desitarget.mock.spectra
#############

import numpy as np
from desisim.io import read_basis_templates, empty_metatable

class TemplateKDTree(object):
    """Build a KD Tree for each object type.

    """
    def __init__(self, nproc=1, verbose=False):
        from speclite import filters
        from scipy.spatial import cKDTree as KDTree
        from desiutil.log import get_logger, DEBUG

        if verbose:
            self.log = get_logger(DEBUG)
        else:
            self.log = get_logger()
            
        self.nproc = nproc
        self.verbose = verbose

        self.bgs_meta = read_basis_templates(objtype='BGS', onlymeta=True)#, verbose=False)
        self.elg_meta = read_basis_templates(objtype='ELG', onlymeta=True)#, verbose=False)
        self.lrg_meta = read_basis_templates(objtype='LRG', onlymeta=True)#, verbose=False)
        self.qso_meta = read_basis_templates(objtype='QSO', onlymeta=True)#, verbose=False)
        self.wd_da_meta = read_basis_templates(objtype='WD', subtype='DA', onlymeta=True)#, verbose=False)
        self.wd_db_meta = read_basis_templates(objtype='WD', subtype='DB', onlymeta=True)#, verbose=False)

        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')

        # Read all the stellar spectra and synthesize DECaLS/WISE fluxes.
        self.star_phot()

        self.bgs_tree = KDTree(self._bgs())
        self.elg_tree = KDTree(self._elg())
        #self.lrg_tree = KDTree(self._lrg())
        #self.qso_tree = KDTree(self._qso())
        self.star_tree = KDTree(self._star())
        self.wd_da_tree = KDTree(self._wd_da())
        self.wd_db_tree = KDTree(self._wd_db())

    def star_phot(self, normfilter='decam2014-r'):
        """Synthesize photometry for the full set of stellar templates."""

        star_flux, star_wave, star_meta = read_basis_templates(objtype='STAR')#, verbose=False)
        star_maggies_table = self.decamwise.get_ab_maggies(star_flux, star_wave, mask_invalid=True)

        star_maggies = dict()
        for key in star_maggies_table.columns:
            star_maggies[key] = star_maggies_table[key] / star_maggies_table[normfilter] # normalized maggies
        self.star_flux_g = star_maggies['decam2014-g']
        self.star_flux_r = star_maggies['decam2014-r']
        self.star_flux_z = star_maggies['decam2014-z']
        self.star_flux_w1 = star_maggies['wise2010-W1']
        self.star_flux_w2 = star_maggies['wise2010-W1']
        
        self.star_meta = star_meta

    def _bgs(self):
        """Quantities we care about: redshift (z), M_0.1r, and 0.1(g-r).  This needs to
        be generalized to accommodate other mocks!

        """
        zobj = self.bgs_meta['Z'].data
        mabs = self.bgs_meta['SDSS_UGRIZ_ABSMAG_Z01'].data
        rmabs = mabs[:, 2]
        gr = mabs[:, 1] - mabs[:, 2]
        return np.vstack((zobj, rmabs, gr)).T

    def _elg(self):
        """Quantities we care about: redshift, g-r, r-z."""
        
        zobj = self.elg_meta['Z'].data
        gr = self.elg_meta['DECAM_G'].data - self.elg_meta['DECAM_R'].data
        rz = self.elg_meta['DECAM_R'].data - self.elg_meta['DECAM_Z'].data
        #W1W2 = self.elg_meta['W1'].data - self.elg_meta['W2'].data
        return np.vstack((zobj, gr, rz)).T

    def _lrg(self):
        """Quantities we care about: r-z, r-W1."""
        
        zobj = self.elg_meta['Z'].data
        gr = self.elg_meta['DECAM_G'].data - self.elg_meta['DECAM_R'].data
        rz = self.elg_meta['DECAM_R'].data - self.elg_meta['DECAM_Z'].data
        #W1W2 = self.elg_meta['W1'].data - self.elg_meta['W2'].data
        return np.vstack((zobj, gr, rz)).T

    def _star(self):
        """Quantities we care about: Teff, logg, and [Fe/H].

        """
        teff = self.star_meta['TEFF'].data
        logg = self.star_meta['LOGG'].data
        feh = self.star_meta['FEH'].data
        return np.vstack((teff, logg, feh)).T

    #def qso(self):
    #    """Quantities we care about: redshift, XXX"""
    #    pass 

    def _wd_da(self):
        """DA white dwarf.  Quantities we care about: Teff and logg.

        """
        teff = self.wd_da_meta['TEFF'].data
        logg = self.wd_da_meta['LOGG'].data
        return np.vstack((teff, logg)).T

    def _wd_db(self):
        """DB white dwarf.  Quantities we care about: Teff and logg.

        """
        teff = self.wd_db_meta['TEFF'].data
        logg = self.wd_db_meta['LOGG'].data
        return np.vstack((teff, logg)).T

    def query(self, objtype, matrix, subtype=''):
        """Return the nearest template number based on the KD Tree.

        Args:
          objtype (str): object type
          matrix (numpy.ndarray): (M,N) array (M=number of properties,
            N=number of objects) in the same format as the corresponding
            function for each object type (e.g., self.bgs).
          subtype (str, optional): subtype (only for white dwarfs)

        Returns:
          dist: distance to nearest template
          indx: index of nearest template
        
        """
        if objtype.upper() == 'BGS':
            dist, indx = self.bgs_tree.query(matrix)
            
        elif objtype.upper() == 'ELG':
            dist, indx = self.elg_tree.query(matrix)
            
        elif objtype.upper() == 'LRG':
            dist, indx = self.lrg_tree.query(matrix)
            
        elif objtype.upper() == 'STAR':
            dist, indx = self.star_tree.query(matrix)
            
        elif objtype.upper() == 'QSO':
            dist, indx = self.qso_tree.query(matrix)
            
        elif objtype.upper() == 'WD':
            if subtype.upper() == 'DA':
                dist, indx = self.wd_da_tree.query(matrix)
            elif subtype.upper() == 'DB':
                dist, indx = self.wd_db_tree.query(matrix)
            else:
                self.log.warning('Unrecognized SUBTYPE {}!'.format(subtype))
                raise ValueError
                
        return dist, indx

class MockSpectra(object):
    """Generate spectra for each type of mock.  Currently just choose the closest
    template; we can get fancier later.

    ToDo (@moustakas): apply Galactic extinction.

    """
    def __init__(self, wavemin=None, wavemax=None, dw=0.2, nproc=1,
                 rand=None, verbose=False):

        from desimodel.io import load_throughput

        self.tree = TemplateKDTree(nproc=nproc, verbose=verbose)

        # Build a default (buffered) wavelength vector.
        if wavemin is None:
            wavemin = load_throughput('b').wavemin - 10.0
        if wavemax is None:
            wavemax = load_throughput('z').wavemax + 10.0
            
        self.wavemin = wavemin
        self.wavemax = wavemax
        self.dw = dw
        self.wave = np.arange(round(wavemin, 1), wavemax, dw)

        self.rand = rand
        self.verbose = verbose

        #self.__normfilter = 'decam2014-r' # default normalization filter

        # Initialize the templates once:
        from desisim.templates import BGS, ELG, LRG, QSO, SIMQSO, STAR, WD
        self.bgs_templates = BGS(wave=self.wave, normfilter='sdss2010-r') # Need to generalize this!
        self.elg_templates = ELG(wave=self.wave, normfilter='decam2014-r')
        self.lrg_templates = LRG(wave=self.wave, normfilter='decam2014-z')
        self.qso_templates = QSO(wave=self.wave, normfilter='decam2014-g')
        self.simqso_templates = SIMQSO(wave=self.wave, normfilter='decam2014-g')
        self.lya_templates = QSO(wave=self.wave, normfilter='decam2014-g')
        self.star_templates = STAR(wave=self.wave, normfilter='decam2014-r')
        self.wd_da_templates = WD(wave=self.wave, normfilter='decam2014-g', subtype='DA')
        self.wd_db_templates = WD(wave=self.wave, normfilter='decam2014-g', subtype='DB')
        
    def bgs(self, data, index=None, mockformat='durham_mxxl_hdf5'):
        """Generate spectra for BGS.

        Currently only the MXXL (durham_mxxl_hdf5) mock is supported.  DATA
        needs to have Z, SDSS_absmag_r01, SDSS_01gr, VDISP, and SEED, which are
        assigned in mock.io.read_durham_mxxl_hdf5.  See also
        TemplateKDTree.bgs().

        """
        objtype = 'BGS'
        if index is None:
            index = np.arange(len(data['Z']))
            
        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'durham_mxxl_hdf5':
            alldata = np.vstack((data['Z'][index],
                                 data['SDSS_absmag_r01'][index],
                                 data['SDSS_01gr'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.bgs_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, novdisp=False,
                                                          verbose=self.verbose)

        return flux, meta

    def elg(self, data, index=None, mockformat='gaussianfield'):
        """Generate spectra for the ELG sample.

        Currently only the GaussianField mock sample is supported.  DATA needs
        to have Z, GR, RZ, VDISP, and SEED, which are assigned in
        mock.io.read_gaussianfield.  See also TemplateKDTree.elg().

        """
        objtype = 'ELG'
        if index is None:
            index = np.arange(len(data['Z']))

        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'gaussianfield':
            alldata = np.vstack((data['Z'][index],
                                 data['GR'][index],
                                 data['RZ'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        if False:
            import matplotlib.pyplot as plt
            def elg_colorbox(ax):
                """Draw the ELG selection box."""
                from matplotlib.patches import Polygon
                grlim = ax.get_ylim()
                coeff0, coeff1 = (1.15, -0.15), (-1.2, 1.6)
                rzmin, rzpivot = 0.3, (coeff1[1] - coeff0[1]) / (coeff0[0] - coeff1[0])
                verts = [(rzmin, grlim[0]),
                         (rzmin, np.polyval(coeff0, rzmin)),
                         (rzpivot, np.polyval(coeff1, rzpivot)),
                         ((grlim[0] - 0.1 - coeff1[1]) / coeff1[0], grlim[0] - 0.1)
                         ]
                ax.add_patch(Polygon(verts, fill=False, ls='--', color='k'))

            fig, ax = plt.subplots()
            ax.scatter(data['RZ'][index], data['GR'][index])
            ax.set_xlim(-0.5, 2) ; plt.ylim(-0.5, 2)
            elg_colorbox(ax)
            plt.show()
            import pdb ; pdb.set_trace()

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.elg_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, novdisp=False,
                                                          verbose=self.verbose)

        return flux, meta

    def elg_test(self, data, index=None, mockformat='gaussianfield'):
        """Test script -- generate spectra for the ELG sample.

        Currently only the GaussianField mock sample is supported.  DATA needs
        to have Z, GR, RZ, VDISP, and SEED, which are assigned in
        mock.io.read_gaussianfield.  See also TemplateKDTree.elg().

        """
        objtype = 'ELG'
        if index is None:
            index = np.arange(len(data['Z']))

        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'gaussianfield':
            alldata = np.vstack((data['Z'][index],
                                 data['GR'][index],
                                 data['RZ'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d

        f1 = interp1d(np.squeeze(self.tree.elg_kcorr['REDSHIFT']), np.squeeze(self.tree.elg_kcorr['GR']), axis=0)
        gr = f1(data['Z'][index])
        plt.plot(np.squeeze(self.tree.elg_kcorr['REDSHIFT']), np.squeeze(self.tree.elg_kcorr['GR'])[:, 500])
        plt.scatter(data['Z'][index], gr[:, 500], marker='x', color='red', s=15)
        plt.show()

        def elg_colorbox(ax):
            """Draw the ELG selection box."""
            from matplotlib.patches import Polygon
            grlim = ax.get_ylim()
            coeff0, coeff1 = (1.15, -0.15), (-1.2, 1.6)
            rzmin, rzpivot = 0.3, (coeff1[1] - coeff0[1]) / (coeff0[0] - coeff1[0])
            verts = [(rzmin, grlim[0]),
                     (rzmin, np.polyval(coeff0, rzmin)),
                     (rzpivot, np.polyval(coeff1, rzpivot)),
                     ((grlim[0] - 0.1 - coeff1[1]) / coeff1[0], grlim[0] - 0.1)
                     ]
            ax.add_patch(Polygon(verts, fill=False, ls='--', color='k'))

        fig, ax = plt.subplots()
        ax.scatter(data['RZ'][index], data['GR'][index])
        ax.set_xlim(-0.5, 2) ; plt.ylim(-0.5, 2)
        elg_colorbox(ax)
        plt.show()
        import pdb ; pdb.set_trace()

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.elg_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, novdisp=False,
                                                          verbose=self.verbose)

        return flux, meta

    def lrg(self, data, index=None, mockformat='gaussianfield'):
        """Generate spectra for the LRG sample.

        Currently only the GaussianField mock sample is supported.  DATA needs
        to have Z, GR, RZ, VDISP, and SEED, which are assigned in
        mock.io.read_gaussianfield.  See also TemplateKDTree.lrg().

        """
        objtype = 'LRG'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'gaussianfield':
            # This is wrong: choose a template with equal probability.
            templateid = self.rand.choice(self.tree.lrg_meta['TEMPLATEID'], len(index))
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.lrg_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, novdisp=False,
                                                          verbose=self.verbose)

        return flux, meta

    def mws(self, data, index=None, mockformat='galaxia'):
        """Generate spectra for the MWS_NEARBY and MWS_MAIN samples.

        """
        objtype = 'STAR'
        if index is None:
            index = np.arange(len(data['Z']))

        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'FEH'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'FEH')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == '100pc':
            alldata = np.vstack((data['TEFF'][index],
                                 data['LOGG'][index],
                                 data['FEH'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
            
        elif mockformat.lower() == 'galaxia':
            alldata = np.vstack((data['TEFF'][index],
                                 data['LOGG'][index],
                                 data['FEH'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.star_templates.make_templates(input_meta=input_meta,
                                                          verbose=self.verbose) # Note! No colorcuts.

        return flux, meta

    def mws_nearby(self, data, index=None, mockformat='100pc'):
        """Generate spectra for the MWS_NEARBY sample.

        """
        flux, meta = self.mws(data, index=index, mockformat=mockformat)
        return flux, meta

    def mws_main(self, data, index=None, mockformat='galaxia'):
        """Generate spectra for the MWS_MAIN sample.

        """
        flux, meta = self.mws(data, index=index, mockformat=mockformat)
        return flux, meta

    def faintstar(self, data, index=None, mockformat='galaxia'):
        """Generate spectra for the FAINTSTAR (faint stellar) sample.

        """
        flux, meta = self.mws(data, index=index, mockformat=mockformat)
        return flux, meta

    def mws_wd(self, data, index=None, mockformat='wd'):
        """Generate spectra for the MWS_WD sample.  Deal with DA vs DB white dwarfs
        separately.

        """
        objtype = 'WD'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        input_meta = empty_metatable(nmodel=nobj, objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'SUBTYPE'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'TEMPLATESUBTYPE')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'wd':
            meta = empty_metatable(nmodel=nobj, objtype=objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')
            
            for subtype in ('DA', 'DB'):
                these = np.where(input_meta['SUBTYPE'] == subtype)[0]
                if len(these) > 0:
                    alldata = np.vstack((data['TEFF'][index][these],
                                         data['LOGG'][index][these])).T
                    _, templateid = self.tree.query(objtype, alldata, subtype=subtype)

                    input_meta['TEMPLATEID'][these] = templateid
                    
                    template_function = 'wd_{}_templates'.format(subtype.lower())
                    flux1, _, meta1 = getattr(self, template_function).make_templates(input_meta=input_meta[these],
                                                          verbose=self.verbose)
                    
                    meta[these] = meta1
                    flux[these, :] = flux1
            
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        return flux, meta

    def qso(self, data, index=None, mockformat='gaussianfield'):
        """Generate spectra for the QSO or QSO/LYA samples.

        Note: We need to make sure NORMFILTER matches!

        """
        from desisim.lya_spectra import get_spectra
        from desisim.lya_spectra import read_lya_skewers,apply_lya_transmission
        import fitsio

        objtype = 'QSO'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        if mockformat.lower() == 'gaussianfield':
            input_meta = empty_metatable(nmodel=nobj, objtype=objtype)
            for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT'),
                                      ('SEED', 'MAG', 'Z')):
                input_meta[inkey] = data[datakey][index]

            # Build the tracer and Lya forest QSO spectra separately.
            meta = empty_metatable(nmodel=nobj, objtype=objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')

            lya = np.where( data['TEMPLATESUBTYPE'][index] == 'LYA' )[0]
            tracer = np.where( data['TEMPLATESUBTYPE'][index] == '' )[0]

            if len(tracer) > 0:
                flux1, _, meta1 = self.qso_templates.make_templates(input_meta=input_meta[tracer],
                                                                    lyaforest=False,
                                                                    nocolorcuts=True,
                                                                    verbose=self.verbose)
                meta[tracer] = meta1
                flux[tracer, :] = flux1

            if len(lya) > 0:
                ilya = index[lya].astype(int)
                nqso = ilya.size
                                
                if 'LYAHDU' in data : 
                    # this is the old format with one HDU per spectrum
                    alllyafile = data['LYAFILES'][ilya]
                    alllyahdu = data['LYAHDU'][ilya]

                    for lyafile in sorted(set(alllyafile)):
                        these = np.where( lyafile == alllyafile )[0]

                        templateid = alllyahdu[these] - 1 # templateid is 0-indexed
                        flux1, _, meta1 = get_spectra(lyafile, templateid=templateid, normfilter=data['FILTERNAME'],
                                                      rand=self.rand, qso=self.lya_templates, nocolorcuts=True)
                        meta1['SUBTYPE'] = 'LYA'
                        meta[lya[these]] = meta1
                        flux[lya[these], :] = flux1
                else : # new format
                    # Read skewers.
                    skewer_wave = None
                    skewer_trans = None
                    skewer_meta = None
                    
                    # All the files that contain at least one QSO skewer.
                    alllyafile = data['LYAFILES'][ilya]
                    uniquelyafiles = sorted(set(alllyafile))
                                        
                    for lyafile in uniquelyafiles:
                        these = np.where( alllyafile == lyafile )[0]
                        objid_in_data = data['OBJID'][ilya][these]
                        objid_in_mock = (fitsio.read(lyafile, columns=['MOCKID'], upper=True,
                                                     ext=1).astype(float)).astype(int)
                        o2i = dict()
                        for i, o in enumerate(objid_in_mock):
                            o2i[o] = i
                        indices_in_mock_healpix = np.zeros(objid_in_data.size).astype(int)
                        for i, o in enumerate(objid_in_data):
                            if not o in o2i:
                                self.log.error("No MOCKID={} in {}. It's a bug, should never happen".format(o,lyafile))
                                raise(KeyError("No MOCKID={} in {}. It's a bug, should never happen".format(o,lyafile)))
                            indices_in_mock_healpix[i] = o2i[o]
                        
                        tmp_wave, tmp_trans, tmp_meta = read_lya_skewers(lyafile, indices=indices_in_mock_healpix) 
                                                
                        if skewer_wave is None:
                            skewer_wave = tmp_wave
                            dw = skewer_wave[1]-skewer_wave[0] # this is just to check same wavelength
                            skewer_trans = np.zeros((nqso,skewer_wave.size)) # allocate skewer_array
                            skewer_meta = dict()
                            for k in tmp_meta.dtype.names:
                                skewer_meta[k] = np.zeros(nqso).astype(tmp_meta[k].dtype)
                        else :
                            # check wavelength is the same for all skewers
                            assert(np.max(np.abs(wave-tmp_wave))<0.001*dw)
                        
                        skewer_trans[these] = tmp_trans
                        for k in skewer_meta.keys():
                            skewer_meta[k][these] = tmp_meta[k]
                    
                    # Check we matched things correctly.
                    assert(np.max(np.abs(skewer_meta["Z"]-data['Z'][ilya]))<0.000001)
                    assert(np.max(np.abs(skewer_meta["RA"]-data['RA'][ilya]))<0.000001)
                    assert(np.max(np.abs(skewer_meta["DEC"]-data['DEC'][ilya]))<0.000001)
                    
                    # Now we create a series of QSO spectra all at once, which
                    # is faster than calling each one at a time.
                    
                    #seed = self.rand.randint(2**32)
                    #qso  = self.lya_templates
                    qso_flux, qso_wave, qso_meta = self.simqso_templates.make_templates(
                        nmodel=nqso, redshift=data['Z'][ilya], #seed=seed,
                        lyaforest=False, nocolorcuts=True)
                    
                    # apply transmission to QSOs
                    qso_flux = apply_lya_transmission(qso_wave, qso_flux, skewer_wave, skewer_trans)
                    
                    qso_meta['SUBTYPE'] = 'LYA'
                    meta[lya] = qso_meta
                    flux[lya, :] = qso_flux
                    
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            
        return flux, meta

    def sky(self, data, index=None, mockformat=None):
        """Generate spectra for SKY.

        """
        objtype = 'SKY'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)
            
        meta = empty_metatable(nmodel=nobj, objtype=objtype)
        for inkey, datakey in zip(('SEED', 'REDSHIFT'),
                                  ('SEED', 'Z')):
            meta[inkey] = data[datakey][index]
        flux = np.zeros((nobj, len(self.wave)), dtype='i1')

        return flux, meta

class MockMagnitudes(object):
    """
    Generate mock magnitudes for each mock.
    """
    def __init__(self, nproc=1, rand=None, verbose=False):
        self.rand = rand
        self.verbose = verbose
    def bgs(self, data, index=None, mockformat='durham_mxxl_hdf5'):
        """Generate magnitudes for BGS.
        Currently only the MXXL (durham_mxxl_hdf5) mock is supported.  DATA
        needs to have Z, SDSS_absmag_r01, SDSS_01gr, VDISP, and SEED, which are
        assigned in mock.io.read_durham_mxxl_hdf5.  
        """
        objtype = 'BGS'
        if index is None:
            index = np.arange(len(data['Z']))
        meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            meta[inkey] = data[datakey][index]

        if mockformat.lower() != 'durham_mxxl_hdf5':
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta
        meta['FLUX_R'][:] = 10**((22.5 - data['MAG'][index])/2.5) # r-band flux
        meta['FLUX_G'][:] = 10**((22.5-(data['SDSS_01gr'][index] + data['MAG'][index]))/2.5) # g-band flux
        return meta
    
    def mws(self, data, index=None, mockformat='galaxia'):
        """Generate magnitudes for the MWS_NEARBY and MWS_MAIN samples.
        """
        objtype = 'STAR'
        if index is None:
            index = np.arange(len(data['Z']))

        meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'FEH'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'FEH')):
            meta[inkey] = data[datakey][index]

        if not (mockformat.lower() in ['100pc','galaxia']):
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta

        meta['FLUX_R'][:] = 10**((22.5 - data['MAG'][index])/2.5) # r-band flux
        return meta
                
    def mws_nearby(self, data, index=None, mockformat='100pc'):
        """Generate magnitudes for the MWS_NEARBY sample.
        """
        meta = self.mws(data, index=index, mockformat=mockformat)
        return meta

    def mws_main(self, data, index=None, mockformat='galaxia'):
        """Generate magnitudes for the MWS_MAIN sample.
        """
        meta = self.mws(data, index=index, mockformat=mockformat)
        return meta

    def faintstar(self, data, index=None, mockformat='galaxia'):
        """Generate magnitudes for the FAINTSTAR (faint stellar) sample.
        """
        meta = self.mws(data, index=index, mockformat=mockformat)
        return meta
        
    def elg(self, data, index=None, mockformat='gaussianfield'):
        """Generate magnitudes for the ELG sample.
        Currently only the GaussianField mock sample is supported.  DATA needs
        to have Z, GR, RZ, VDISP, and SEED, which are assigned in
        mock.io.read_gaussianfield.  
        """
        objtype = 'ELG'
        if index is None:
            index = np.arange(len(data['Z']))

        meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            meta[inkey] = data[datakey][index]

        if mockformat.lower() != 'gaussianfield':
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta
        
        meta['FLUX_G'][:] = 10**((22.5 - (data['GR'][index] + data['MAG'][index]))/2.5) # g-band flux
        meta['FLUX_R'][:] = 10**((22.5 - data['MAG'][index])/2.5) # r-band flux
        meta['FLUX_Z'][:] = 10**((22.5 - (data['MAG'][index] - data['RZ'][index]))/2.5) # z-band flux
        return meta
    
    def lrg(self, data, index=None, mockformat='gaussianfield'):
        """Generate magnitudes for the LRG sample.
        Currently only the GaussianField mock sample is supported.  DATA needs
        to have Z, GR, RZ, VDISP, and SEED, which are assigned in
        mock.io.read_gaussianfield.  
        """
        objtype = 'LRG'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            meta[inkey] = data[datakey][index]

        if mockformat.lower() != 'gaussianfield':
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta
        
        meta['FLUX_G'][:] = 10**((22.5 - (data['GR'][index] + data['RZ'][index] + data['MAG'][index]))/2.5) # g-band flux  
        meta['FLUX_R'][:] =  10**((22.5 - (data['MAG'][index] + data['RZ'][index]))/2.5) # r-band flux
        meta['FLUX_Z'][:] = 10**((22.5 - data['MAG'][index])/2.5) # z-band flux
        meta['FLUX_W1'][:] = 10**((22.5 - (data['RZ'][index] + data['MAG'][index] - data['RW1'][index]))/2.5)# wise flux 1
        meta['FLUX_W2'][:] = 10**((22.5 - (data['RZ'][index] + data['MAG'][index] - data['RW1'][index] - data['W1W2'][index]))/2.5)# wise flux 2
        
        return meta
    
    def qso(self, data, index=None, mockformat='gaussianfield'):
        """Generate magnitudes for the QSO or QSO/LYA samples.
        """
        from desisim.lya_spectra import get_spectra
                
        objtype = 'QSO'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)
        meta = empty_metatable(nmodel=nobj, objtype=objtype)

        if mockformat.lower() != 'gaussianfield':
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta
        
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT'),
                                      ('SEED', 'MAG', 'Z')):
            meta[inkey] = data[datakey][index]
        
        meta['FLUX_G'][:] = 10**((22.5 - data['MAG'][index])/2.5) # g-band flux
        meta['FLUX_R'][:] = 10**((22.5 - (data['MAG'][index] - data['GR'][index]))/2.5) # r-band flux
        meta['FLUX_Z'][:] = 10**((22.5 - (data['MAG'][index] - data['GR'][index] - data['RZ'][index]))/2.5) # z-band flux
        meta['FLUX_W1'][:] = 10**((22.5 - (data['MAG'][index] - data['GR'][index] - data['RW1'][index]))/2.5)# wise flux 1
        meta['FLUX_W2'][:] = 10**((22.5 - (data['MAG'][index] - data['GR'][index] - data['RW1'][index] - data['W1W2'][index]))/2.5)# wise flux 2
        
        lya = np.where( data['TEMPLATESUBTYPE'][index] == 'LYA' )[0]                 
        if len(lya) > 0:
            meta['SUBTYPE'][lya] = 'LYA'
                                       
        return meta
    def mws_wd(self, data, index=None, mockformat='wd'):
        """Generate magnitudes for the MWS_WD sample.  Deal with DA vs DB white dwarfs
        separately.
        """
        objtype = 'WD'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        meta = empty_metatable(nmodel=nobj, objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'SUBTYPE'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'TEMPLATESUBTYPE')):
            meta[inkey] = data[datakey][index]

        if mockformat.lower() != 'wd':
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta
            
        for subtype in ('DA', 'DB'):
            these = np.where(meta['SUBTYPE'] == subtype)[0]
            if len(these) > 0:
                meta['SUBTYPE'][these] = subtype    
                meta['TEMPLATEID'][:] = -1
    
        meta['FLUX_G'][:] = 10**((22.5 - data['MAG'][index])/2.5) # g-band flux
        return meta
    
    def sky(self, data, index=None, mockformat=None):
        """Generate spectra for SKY.

        """
        objtype = 'SKY'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)
            
        meta = empty_metatable(nmodel=nobj, objtype=objtype)
        for inkey, datakey in zip(('SEED', 'REDSHIFT'),
                                  ('SEED', 'Z')):
            meta[inkey] = data[datakey][index]
            
        return meta


#############
# Code previously in desitarget.mock.selection
#############

import numpy as np

class SelectTargets(object):
    """Select various types of targets.  Most of this functionality is taken from
    desitarget.cuts but that code has not been factored in a way that is
    convenient at this time.

    """
    def __init__(self, rand=None, brick_info=None, verbose=False):
        
        from desitarget import (desi_mask, bgs_mask, mws_mask,
                                contam_mask, obsconditions)
        from desiutil.log import get_logger, DEBUG

        if verbose:
            self.log = get_logger(DEBUG)
        else:
            self.log = get_logger()
        
        self.desi_mask = desi_mask
        self.bgs_mask = bgs_mask
        self.mws_mask = mws_mask
        self.contam_mask = contam_mask
        self.obsconditions = obsconditions
        
        self.rand = rand
        self.brick_info = brick_info

    def _star_select(self, targets, truth):
        """Select stellar (faint and bright) contaminants for the extragalactic
        targets.

        """ 
        from desitarget.cuts import isBGS_faint, isELG, isLRG_colors, isQSO_colors

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']

        # Select stellar contaminants for BGS_FAINT targets.
        bgs_faint = isBGS_faint(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT_SOUTH
        targets['DESI_TARGET'] |= (bgs_faint != 0) * self.desi_mask.BGS_ANY
        
        targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
            self.bgs_mask.BGS_FAINT.obsconditions)
        targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
            self.bgs_mask.BGS_FAINT_SOUTH.obsconditions)
        targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
            self.desi_mask.BGS_ANY.obsconditions)
        
        truth['CONTAM_TARGET'] |= (bgs_faint != 0) * self.contam_mask.BGS_IS_STAR
        truth['CONTAM_TARGET'] |= (bgs_faint != 0) * self.contam_mask.BGS_CONTAM

        # Select stellar contaminants for ELG targets.
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        
        targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(
            self.desi_mask.ELG.obsconditions)
        targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(
            self.desi_mask.ELG_SOUTH.obsconditions)
        
        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_IS_STAR
        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_CONTAM

        # Select stellar contaminants for LRG targets.
        lrg = isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux)
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH

        targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(
            self.desi_mask.LRG.obsconditions)
        targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(
            self.desi_mask.LRG_SOUTH.obsconditions)
        
        truth['CONTAM_TARGET'] |= (lrg != 0) * self.contam_mask.LRG_IS_STAR
        truth['CONTAM_TARGET'] |= (lrg != 0) * self.contam_mask.LRG_CONTAM

        # Select stellar contaminants for QSO targets.
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                           w2flux=w2flux, optical=False) # Note optical=False!
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH

        targets['OBSCONDITIONS'] |= (qso != 0) * self.obsconditions.mask(
            self.desi_mask.QSO.obsconditions)
        targets['OBSCONDITIONS'] |= (qso != 0) * self.obsconditions.mask(
            self.desi_mask.QSO_SOUTH.obsconditions)
        
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_IS_STAR
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_CONTAM

    def _std_select(self, targets, truth, boss_std=None):
        """Select bright- and dark-time standard stars."""
        from desitarget.cuts import isFSTD

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']

        obs_rflux = rflux / targets['MW_TRANSMISSION_R'] # attenuate for Galactic dust

        gsnr, rsnr, zsnr = gflux*0+100, rflux*0+100, zflux*0+100    # Hack -- fixed S/N
        gfracflux, rfracflux, zfracflux = gflux*0, rflux*0, zflux*0 # # No contamination from neighbors.
        objtype = np.repeat('PSF', len(targets)).astype('U3') # Right data type?!?

        # Select dark-time FSTD targets.  Temporary hack to use the BOSS
        # standard-star selection algorith.
        if boss_std is None:
            fstd = isFSTD(gflux=gflux, rflux=rflux, zflux=zflux, objtype=objtype,
                          gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, 
                          gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux, 
                          obs_rflux=obs_rflux)
        else:
            rbright, rfaint = 16, 19
            fstd = boss_std * ( obs_rflux < 10**((22.5 - rbright)/2.5) ) * ( obs_rflux > 10**((22.5 - rfaint)/2.5) )

        targets['DESI_TARGET'] |= (fstd != 0) * self.desi_mask.STD_FSTAR
        targets['OBSCONDITIONS'] |= (fstd != 0) * self.obsconditions.mask(
            self.desi_mask.STD_FSTAR.obsconditions)

        # Select bright-time FSTD targets.  Temporary hack to use the BOSS
        # standard-star selection algorith.
        if boss_std is None:
            fstd_bright = isFSTD(gflux=gflux, rflux=rflux, zflux=zflux, objtype=objtype,
                                 gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, 
                                 gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux, 
                                 obs_rflux=obs_rflux, bright=True)
        else:
            rbright, rfaint = 14, 17
            fstd_bright = boss_std * ( obs_rflux < 10**((22.5 - rbright)/2.5) ) * ( obs_rflux > 10**((22.5 - rfaint)/2.5) )

        targets['DESI_TARGET'] |= (fstd_bright != 0) * self.desi_mask.STD_BRIGHT
        targets['OBSCONDITIONS'] |= (fstd_bright != 0) * self.obsconditions.mask(
            self.desi_mask.STD_BRIGHT.obsconditions)
        
    def bgs_select(self, targets, truth, boss_std=None):
        """Select BGS targets.

        """
        from desitarget.cuts import isBGS_bright, isBGS_faint

        rflux = targets['FLUX_R']

        # Select BGS_BRIGHT targets.
        bgs_bright = isBGS_bright(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT_SOUTH
        targets['DESI_TARGET'] |= (bgs_bright != 0) * self.desi_mask.BGS_ANY

        targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(
            self.bgs_mask.BGS_BRIGHT.obsconditions)
        targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(
            self.bgs_mask.BGS_BRIGHT_SOUTH.obsconditions)
        targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(
            self.desi_mask.BGS_ANY.obsconditions)
        
        # Select BGS_FAINT targets.
        bgs_faint = isBGS_faint(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT_SOUTH
        targets['DESI_TARGET'] |= (bgs_faint != 0) * self.desi_mask.BGS_ANY
        
        targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
            self.bgs_mask.BGS_FAINT.obsconditions)
        targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
            self.bgs_mask.BGS_FAINT_SOUTH.obsconditions)
        targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
            self.desi_mask.BGS_ANY.obsconditions)

    def elg_select(self, targets, truth, boss_std=None):
        """Select ELG targets and contaminants."""
        from desitarget.cuts import isELG, isQSO_colors

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']
        
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)

        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(
            self.desi_mask.ELG.obsconditions)
        targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(
            self.desi_mask.ELG_SOUTH.obsconditions)

        # Select ELG contaminants for QSO targets.  There should be a morphology
        # cut here, too, so we're going to overestimate the number of
        # contaminants.  To make sure we don't reduce the number density of true
        # ELGs, demand that the QSO contaminants are not also selected as ELGs.
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                           w2flux=w2flux, optical=False) # Note optical=False!
        targets['DESI_TARGET'] |= (qso != 0) * (elg == 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * (elg == 0) * self.desi_mask.QSO_SOUTH
        targets['OBSCONDITIONS'] |= (qso != 0) * (elg == 0) * self.obsconditions.mask(
            self.desi_mask.QSO.obsconditions)
        targets['OBSCONDITIONS'] |= (qso != 0) * (elg == 0) * self.obsconditions.mask(
            self.desi_mask.QSO_SOUTH.obsconditions)

        truth['CONTAM_TARGET'] |= (qso != 0) * (elg == 0) * self.contam_mask.QSO_IS_ELG
        truth['CONTAM_TARGET'] |= (qso != 0) * (elg == 0) * self.contam_mask.QSO_IS_GALAXY
        truth['CONTAM_TARGET'] |= (qso != 0) * (elg == 0) * self.contam_mask.QSO_CONTAM

    def faintstar_select(self, targets, truth, boss_std=None):
        """Select faint stellar contaminants for the extragalactic targets.""" 

        self._star_select(targets, truth)

    def lrg_select(self, targets, truth, boss_std=None):
        """Select LRG targets."""
        from desitarget.cuts import isLRG_colors, isQSO_colors

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']

        lrg = isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux)

        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH
        targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(
            self.desi_mask.LRG.obsconditions)
        targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(
            self.desi_mask.LRG_SOUTH.obsconditions)
        
        # Select LRG contaminants for QSO targets.  There should be a morphology
        # cut here, too, so we're going to overestimate the number of
        # contaminants.  To make sure we don't reduce the number density of true
        # LRGs, demand that the QSO contaminants are not also selected as LRGs.
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                           w2flux=w2flux, optical=False) # Note optical=False!
        targets['DESI_TARGET'] |= (qso != 0) * (lrg == 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * (lrg == 0) * self.desi_mask.QSO_SOUTH
        targets['OBSCONDITIONS'] |= (qso != 0) * (lrg == 0)  * self.obsconditions.mask(
            self.desi_mask.QSO.obsconditions)
        targets['OBSCONDITIONS'] |= (qso != 0) * (lrg == 0)  * self.obsconditions.mask(
            self.desi_mask.QSO_SOUTH.obsconditions)
        
        truth['CONTAM_TARGET'] |= (qso != 0) * (lrg == 0) * self.contam_mask.QSO_IS_LRG
        truth['CONTAM_TARGET'] |= (qso != 0) * (lrg == 0) * self.contam_mask.QSO_IS_GALAXY
        truth['CONTAM_TARGET'] |= (qso != 0) * (lrg == 0) * self.contam_mask.QSO_CONTAM

    def mws_main_select(self, targets, truth, boss_std=None):
        """Select MWS_MAIN, MWS_MAIN_VERY_FAINT, standard stars, and (bright)
        contaminants for extragalactic targets.  The selection here eventually
        will be done with Gaia (I think).

        """
        def _isMWS_MAIN(rflux):
            """A function like this should be in desitarget.cuts. Select 15<r<19 stars."""
            main = rflux > 10**( (22.5 - 19.0) / 2.5 )
            main &= rflux <= 10**( (22.5 - 15.0) / 2.5 )
            return main

        def _isMWS_MAIN_VERY_FAINT(rflux):
            """A function like this should be in desitarget.cuts. Select 19<r<20 filler stars."""
            faint = rflux > 10**( (22.5 - 20.0) / 2.5 )
            faint &= rflux <= 10**( (22.5 - 19.0) / 2.5 )
            return faint
        
        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']

        # Select MWS_MAIN targets.
        mws_main = _isMWS_MAIN(rflux=rflux)
        #mws_main = np.ones(len(targets)) # select everything!
        
        targets['MWS_TARGET'] |= (mws_main != 0) * self.mws_mask.mask('MWS_MAIN')
        targets['DESI_TARGET'] |= (mws_main != 0) * self.desi_mask.MWS_ANY
        targets['OBSCONDITIONS'] |= (mws_main != 0)  * self.obsconditions.mask(
            self.mws_mask.MWS_MAIN.obsconditions)
        targets['OBSCONDITIONS'] |= (mws_main != 0)  * self.obsconditions.mask(
            self.desi_mask.MWS_ANY.obsconditions)
        
        mws_main_very_faint = _isMWS_MAIN_VERY_FAINT(rflux=rflux)
        targets['MWS_TARGET'] |= (mws_main_very_faint != 0) * self.mws_mask.mask('MWS_MAIN_VERY_FAINT')
        targets['DESI_TARGET'] |= (mws_main_very_faint != 0) * self.desi_mask.MWS_ANY
        targets['OBSCONDITIONS'] |= (mws_main_very_faint != 0)  * self.obsconditions.mask(
            self.mws_mask.MWS_MAIN_VERY_FAINT.obsconditions)
        targets['OBSCONDITIONS'] |= (mws_main_very_faint != 0)  * self.obsconditions.mask(
            self.desi_mask.MWS_ANY.obsconditions)

        # Select standard stars.
        self._std_select(targets, truth, boss_std=boss_std)
        
        # Select bright stellar contaminants for the extragalactic targets.
        self._star_select(targets, truth)

    def mws_nearby_select(self, targets, truth, boss_std=None):
        """Select MWS_NEARBY targets.  The selection eventually will be done with Gaia,
        so for now just do a "perfect" selection.

        """
        mws_nearby = np.ones(len(targets)) # select everything!
        #mws_nearby = (truth['MAG'] <= 20.0) * 1 # SDSS g-band!

        targets['MWS_TARGET'] |= (mws_nearby != 0) * self.mws_mask.mask('MWS_NEARBY')
        targets['DESI_TARGET'] |= (mws_nearby != 0) * self.desi_mask.MWS_ANY
        targets['OBSCONDITIONS'] |= (mws_nearby != 0)  * self.obsconditions.mask(
            self.mws_mask.MWS_NEARBY.obsconditions)
        targets['OBSCONDITIONS'] |= (mws_nearby != 0)  * self.obsconditions.mask(
            self.desi_mask.MWS_ANY.obsconditions)

    def mws_wd_select(self, targets, truth, boss_std=None):
        """Select MWS_WD and STD_WD targets.  The selection eventually will be done with
        Gaia, so for now just do a "perfect" selection here.

        """
        #mws_wd = np.ones(len(targets)) # select everything!
        mws_wd = ((truth['MAG'] >= 15.0) * (truth['MAG'] <= 20.0)) * 1 # SDSS g-band!

        targets['MWS_TARGET'] |= (mws_wd != 0) * self.mws_mask.mask('MWS_WD')
        targets['DESI_TARGET'] |= (mws_wd != 0) * self.desi_mask.MWS_ANY
        targets['OBSCONDITIONS'] |= (mws_wd != 0)  * self.obsconditions.mask(
            self.mws_mask.MWS_WD.obsconditions)
        targets['OBSCONDITIONS'] |= (mws_wd != 0)  * self.obsconditions.mask(
            self.desi_mask.MWS_ANY.obsconditions)

        # Select STD_WD; cut just on g-band magnitude (not TEMPLATESUBTYPE!)
        std_wd = (truth['MAG'] <= 19.0) * 1 # SDSS g-band!
        targets['DESI_TARGET'] |= (std_wd !=0) * self.desi_mask.mask('STD_WD')
        targets['OBSCONDITIONS'] |= (std_wd != 0)  * self.obsconditions.mask(
            self.desi_mask.STD_WD.obsconditions)

    def qso_select(self, targets, truth, boss_std=None):
        """Select QSO targets and contaminants."""
        from desitarget.cuts import isQSO_colors, isELG

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']
          
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux, optical=True)

        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH
        targets['OBSCONDITIONS'] |= (qso != 0)  * self.obsconditions.mask(
            self.desi_mask.QSO.obsconditions)
        targets['OBSCONDITIONS'] |= (qso != 0)  * self.obsconditions.mask(
            self.desi_mask.QSO_SOUTH.obsconditions)

        # Select QSO contaminants for ELG targets.  There'd be no morphology cut
        # and we're missing the WISE colors, so the density won't be right.
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        targets['OBSCONDITIONS'] |= (elg != 0)  * self.obsconditions.mask(
            self.desi_mask.ELG.obsconditions)
        targets['OBSCONDITIONS'] |= (elg != 0)  * self.obsconditions.mask(
            self.desi_mask.ELG_SOUTH.obsconditions)

        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_IS_QSO
        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_CONTAM

    def sky_select(self, targets, truth, boss_std=None):
        """Select SKY targets."""

        targets['DESI_TARGET'] |= self.desi_mask.mask('SKY')
        targets['OBSCONDITIONS'] |= self.obsconditions.mask(
            self.desi_mask.SKY.obsconditions)

    def density_select(self, targets, truth, source_name, target_name, density=None, subset=None):
        """Downsample a target sample to a desired number density in targets/deg2."""
        nobj = len(targets)

        unique_bricks = list(set(targets['BRICKNAME']))
        n_brick = len(unique_bricks)

        for thisbrick in unique_bricks:
            brickindx = np.where(self.brick_info['BRICKNAME'] == thisbrick)[0]
            brick_area = float(self.brick_info['AREA'][brickindx][0])

            if subset is None:
                onbrick = np.where( (targets['BRICKNAME'] == thisbrick) )[0]
            else:
                onbrick = np.where( (targets['BRICKNAME'] == thisbrick) * subset )[0]
            #onbrick = np.where((targets['BRICKNAME'] == thisbrick) * (truth['CONTAM_TARGET'] == 0))[0]

            n_in_brick = len(onbrick)
            if n_in_brick == 0:
                self.log.info('No {}s on brick {}.'.format(target_name, thisbrick))
                
            if density and n_in_brick > 0:
                mock_density = n_in_brick / brick_area
                desired_density = float(self.brick_info['FLUC_EBV_{}'.format(source_name)][brickindx] * density)
                if desired_density < 0.0:
                    desired_density = 0.0

                frac_keep = desired_density / mock_density
                if frac_keep > 1.0:
                    self.log.warning('Density {:.0f}/deg2 (N={:g}) lower than desired {:.0f}/deg2 on brick {}.'.format(
                        mock_density, n_in_brick, desired_density, thisbrick))
                else:
                    frac_toss = 1.0 - frac_keep
                    ntoss = int( np.ceil( n_in_brick * frac_toss ) )

                    self.log.info('Downsampling from {:.0f} to {:.0f} targets/deg2 (N={:g} to N={:g}) on brick {}.'.format(
                        mock_density, desired_density, n_in_brick, n_in_brick - ntoss, thisbrick))
                    
                    toss = self.rand.choice(onbrick, ntoss, replace=False)
                    targets['DESI_TARGET'][toss] = 0

    def contaminants_select(self, targets, truth, source_name, target_name, contam):
        """Downsample contaminants to a desired number density in targets/deg2."""
        nobj = len(targets)

        unique_bricks = list(set(targets['BRICKNAME']))
        n_brick = len(unique_bricks)

        for thisbrick in unique_bricks:
            brickindx = np.where(self.brick_info['BRICKNAME'] == thisbrick)[0]
            brick_area = float(self.brick_info['AREA'][brickindx][0])

            for contam_name in contam.keys():
                onbrick = np.where(
                    (targets['BRICKNAME'] == thisbrick) *
                    #(targets['DESI_TARGET'] == 0) *
                    (truth['CONTAM_TARGET'] & self.contam_mask.mask('{}_IS_{}'.format(target_name, contam_name)) != 0)
                    )[0]
                n_in_brick = len(onbrick)

                if n_in_brick == 0:
                    self.log.debug('No {}_IS_{} contaminants on brick {}.'.format(target_name, contam_name, thisbrick))
                
                if n_in_brick > 0:
                    contam_density = len(onbrick) / brick_area
                    desired_density = float(self.brick_info['FLUC_EBV_{}'.format(source_name)][brickindx] * contam[contam_name])
                    if desired_density < 0.0:
                        desired_density = 0.0

                    frac_keep = desired_density / contam_density
                    if frac_keep > 1.0:
                        self.log.warning('Density {:.0f}/deg2 (N={:g}) of {} contaminants lower than desired {:.0f}/deg2 on brick {}.'.format(
                            contam_density, n_in_brick, contam_name.upper(), desired_density, thisbrick))

                    else:
                        frac_toss = 1.0 - frac_keep
                        ntoss = int( np.ceil( n_in_brick * frac_toss ) )

                        self.log.info('Downsampling {}_IS_{} contaminants from {:.0f} to {:.0f} targets/deg2 (N={:g} to N={:g}) on brick {}.'.format(
                            target_name, contam_name, contam_density, desired_density, n_in_brick, n_in_brick - ntoss, thisbrick))

                        # This isn't quite right because we occassionally throw
                        # away too many other "real" targets.
                        toss = self.rand.choice(onbrick, ntoss, replace=False)
                        targets['DESI_TARGET'][toss] = 0 
