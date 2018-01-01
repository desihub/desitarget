# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.build
=====================

Build a truth catalog (including spectra) and a targets catalog for the mocks.

"""
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import healpy as hp
from astropy.table import Table, Column, vstack, hstack

from desimodel.footprint import radec2pix
from desitarget.targets import encode_targetid
import desitarget.mock.io as mockio
from desitarget import desi_mask, bgs_mask, mws_mask, contam_mask

def empty_targets_table(nobj=1):
    """Initialize an empty 'targets' table.

    """
    targets = Table()

    # RELEASE
    targets.add_column(Column(name='BRICKID', length=nobj, dtype='i4'))
    targets.add_column(Column(name='BRICKNAME', length=nobj, dtype='U8'))
    targets.add_column(Column(name='BRICK_OBJID', length=nobj, dtype='i4'))
    # TYPE
    targets.add_column(Column(name='RA', length=nobj, dtype='f8', unit='degree'))
    targets.add_column(Column(name='DEC', length=nobj, dtype='f8', unit='degree'))
    # RA_IVAR
    # DEC_IVAR
    # DCHISQ
    targets.add_column(Column(name='FLUX_G', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_R', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_Z', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_W1', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_W2', length=nobj, dtype='f4', unit='nanomaggies'))
    # FLUX_W3
    # FLUX_W4
    # FLUX_IVAR_G
    # FLUX_IVAR_R
    # FLUX_IVAR_Z
    # FLUX_IVAR_W1
    # FLUX_IVAR_W2
    # FLUX_IVAR_W3
    # FLUX_IVAR_W4
    targets.add_column(Column(name='MW_TRANSMISSION_G', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_Z', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_W1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_W2', length=nobj, dtype='f4'))
    # MW_TRANSMISSION_W3
    # MW_TRANSMISSION_W4
    # NOBS_G
    # NOBS_R
    # NOBS_Z
    # FRACFLUX_G
    # FRACFLUX_R
    # FRACFLUX_Z
    targets.add_column(Column(name='PSFDEPTH_G', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='PSFDEPTH_R', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='PSFDEPTH_Z', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='GALDEPTH_G', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='GALDEPTH_R', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='GALDEPTH_Z', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    # The following two columns do not appear in the data targets catalog.
    targets.add_column(Column(name='PSFDEPTH_W1', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='PSFDEPTH_W2', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='SHAPEDEV_R', length=nobj, dtype='f4', unit='arcsec'))
    # SHAPEDEV_R_IVAR
    targets.add_column(Column(name='SHAPEDEV_E1', length=nobj, dtype='f4'))
    # SHAPEDEV_E1_IVAR
    targets.add_column(Column(name='SHAPEDEV_E2', length=nobj, dtype='f4'))
    # SHAPEDEV_E2_IVAR    
    targets.add_column(Column(name='SHAPEEXP_R', length=nobj, dtype='f4', unit='arcsec'))
    # SHAPEEXP_R_IVAR
    targets.add_column(Column(name='SHAPEEXP_E1', length=nobj, dtype='f4'))
    # SHAPEEXP_E1_IVAR
    targets.add_column(Column(name='SHAPEEXP_E2', length=nobj, dtype='f4'))
    # SHAPEEXP_E2_IVAR
    targets.add_column(Column(name='SUBPRIORITY', length=nobj, dtype='f8'))
    targets.add_column(Column(name='TARGETID', length=nobj, dtype='int64'))
    targets.add_column(Column(name='DESI_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='BGS_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='MWS_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='HPXPIXEL', length=nobj, dtype='i8'))
    # PHOTSYS
    # Do we need obsconditions or not?!?
    targets.add_column(Column(name='OBSCONDITIONS', length=nobj, dtype='i8'))

    return targets

def empty_truth_table(nobj=1):
    """Initialize an empty 'truth' table.

    """
    truth = Table()
    truth.add_column(Column(name='TARGETID', length=nobj, dtype='int64'))
    truth.add_column(Column(name='MOCKID', length=nobj, dtype='int64'))
    truth.add_column(Column(name='CONTAM_TARGET', length=nobj, dtype='i8'))

    truth.add_column(Column(name='TRUEZ', length=nobj, dtype='f4', data=np.zeros(nobj)))
    truth.add_column(Column(name='TRUESPECTYPE', length=nobj, dtype='U10')) # GALAXY, QSO, STAR, etc.
    truth.add_column(Column(name='TEMPLATETYPE', length=nobj, dtype='U10')) # ELG, BGS, STAR, WD, etc.
    truth.add_column(Column(name='TEMPLATESUBTYPE', length=nobj, dtype='U10')) # DA, DB, etc.

    truth.add_column(Column(name='TEMPLATEID', length=nobj, dtype='i4', data=np.zeros(nobj)-1))
    truth.add_column(Column(name='SEED', length=nobj, dtype='int64', data=np.zeros(nobj)-1))
    truth.add_column(Column(name='MAG', length=nobj, dtype='f4', data=np.zeros(nobj)+99, unit='mag'))
    
    truth.add_column(Column(name='FLUX_G', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_R', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_Z', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_W1', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_W2', length=nobj, dtype='f4', unit='nanomaggies'))

    truth.add_column(Column(name='OIIFLUX', length=nobj, dtype='f4',
                            data=np.zeros(nobj)-1, unit='erg/(s*cm2)'))
    truth.add_column(Column(name='HBETAFLUX', length=nobj, dtype='f4',
                            data=np.zeros(nobj)-1, unit='erg/(s*cm2)'))

    truth.add_column(Column(name='TEFF', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='K'))
    truth.add_column(Column(name='LOGG', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='m/(s**2)'))
    truth.add_column(Column(name='FEH', length=nobj, dtype='f4', data=np.zeros(nobj)-1))

    return truth

def _initialize_targets_truth(source_data, indx=None):
    """Given a source_data dictionary, initialize the 'targets' and 'truth' tables
    and populate them with various quantities of interest.

    """
    if indx is None:
        indx = np.arange(len(source_data['RA']))
    nobj = len(indx)

    # Initialize the tables.
    targets = empty_targets_table(nobj)
    truth = empty_truth_table(nobj)
    
    for key in ('RA', 'DEC', 'BRICKNAME'):
        targets[key][:] = source_data[key][indx]

    # Add dust and depth.
    for band in ('G', 'R', 'Z', 'W1', 'W2'):
        key = 'MW_TRANSMISSION_{}'.format(band)
        targets[key][:] = source_data[key][indx]

    for band in ('G', 'R', 'Z'):
        for prefix in ('PSF', 'GAL'):
            key = '{}DEPTH_{}'.format(prefix, band)
            targets[key][:] = source_data[key][indx]

    for band in ('W1', 'W2'):
        key = 'PSFDEPTH_{}'.format(band)
        targets[key][:] = source_data[key][indx]

    # Add shapes and sizes.
    if 'SHAPEEXP_R' in source_data.keys(): # not all target types have shape information
        for key in ('SHAPEEXP_R', 'SHAPEEXP_E1', 'SHAPEEXP_E2',
                    'SHAPEDEV_R', 'SHAPEDEV_E1', 'SHAPEDEV_E2'):
            targets[key][:] = source_data[key][indx]

    for key, source_key in zip( ['MOCKID', 'SEED', 'TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TRUESPECTYPE'],
                                ['MOCKID', 'SEED', 'TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TRUESPECTYPE'] ):
        if isinstance(source_data[source_key], np.ndarray):
            truth[key][:] = source_data[source_key][indx]
        else:
            truth[key][:] = np.repeat(source_data[source_key], nobj)

    # Sky targets do not have redshifts.
    if 'Z' in source_data.keys():
        truth['TRUEZ'][:] = source_data['Z'][indx]

    return targets, truth

def _initialize(params, verbose=False, seed=1, output_dir="./", 
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
        selection: desitarget.mock.SelectTargets
            Object to select targets from the input mock catalogs.

    """
    from desiutil.log import get_logger, DEBUG
    #from desitarget.mock.selection import SelectTargets

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

    # Initialize the Classes used to assign spectra (or magnitudes) and to
    # select targets.  Note: The default wavelength array gets initialized in
    # MockSpectra.
    #selection = SelectTargets(verbose=verbose, rand=rand)

    return log, rand, healpixels
    
def read_mock(source_name, params, log, seed=None, healpixels=None,
              nside=16, nside_chunk=128, in_desi=True):
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
    from desitarget.mock import mockmaker

    target_name = params['sources'][source_name]['target_name'].upper() # Target type (e.g., ELG)
    mockformat = params['sources'][source_name]['format']
    mockfile = params['sources'][source_name]['mockfile']

    if 'magcut' in params['sources'][source_name].keys():
        magcut = params['sources'][source_name]['magcut']
    else:
        magcut = None

    log.info('Source: {}, target: {}, format: {}'.format(source_name, target_name, mockformat))
    #log.info('Reading {}'.format(mockfile))
    
    MakeMock = getattr(mockmaker, '{}Maker'.format(target_name))(seed=seed)

    source_data = MakeMock.read(mockfile=mockfile, mockformat=mockformat,
                                healpixels=healpixels, nside=nside,
                                nside_chunk=nside_chunk, magcut=magcut,
                                dust_dir=params['dust_dir'])

    # --------------------------------------------------
    # push this to its own thing
    nobj = len(source_data['RA'])

    psfdepth_mag = np.array((24.65, 23.61, 22.84)) # 5-sigma, mag
    galdepth_mag = np.array((24.7, 23.9, 23.0))    # 5-sigma, mag

    psfdepth_ivar = (1 / 10**(-0.4 * (psfdepth_mag - 22.5)))**2 # 5-sigma, 1/nanomaggies**2
    galdepth_ivar = (1 / 10**(-0.4 * (galdepth_mag - 22.5)))**2 # 5-sigma, 1/nanomaggies**2

    for ii, band in enumerate(('G', 'R', 'Z')):
        source_data['PSFDEPTH_{}'.format(band)] = np.repeat(psfdepth_ivar[ii], nobj)
        source_data['GALDEPTH_{}'.format(band)] = np.repeat(galdepth_ivar[ii], nobj)

    wisedepth_mag = np.array((22.3, 23.8)) # 1-sigma, mag
    wisedepth_ivar = 1 / (5 * 10**(-0.4 * (wisedepth_mag - 22.5)))**2 # 5-sigma, 1/nanomaggies**2
    # --------------------------------------------------

    for ii, band in enumerate(('W1', 'W2')):
        source_data['PSFDEPTH_{}'.format(band)] = np.repeat(wisedepth_ivar[ii], nobj)
    
    # Insert proper density fluctuations model here!  Note that in general
    # healpixels will generally be a scalar (because it's called inside a loop),
    # but also allow for multiple healpixels.
    try:
        npix = healpixels.size
    except:
        npix = len(healpixels)
    skyarea = npix * hp.nside2pixarea(nside, degrees=True)

    #if 'density' in params['sources'][source_name].keys():
    #    density = params['sources'][source_name]['density']
    #    ntarget = density * skyarea
    #    ntarget_split = np.repeat(ntarget, nproc) * rand.normal(loc=1.0, scale=0.02, size=nproc)
    #    source_data['TARGET_DENSITY'] = np.round(ntarget_split).astype('int')

    #if 'contam' in params['sources'][source_name].keys():
    #    contam = params['sources'][source_name]['contam']
    #    for contamtype in contam.keys():
    #        source_data['TARGET_DENSITY'] = np.round(density * skyarea).astype('int')

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
                        
    return source_data, MakeMock

def _scatter_photometry(targname, source_data, truth, targets,
                        rand, meta=None, indx=None, qaplot=False):
    """Add noise to the photometry based on the depth.

    """
    if indx is None:
        indx = np.arange(len(source_data['RA']))
    nobj = len(indx)

    # Optionally populate from a metadata table.
    if meta is not None:
        for key in ('TEMPLATEID', 'MAG', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1',
                    'FLUX_W2', 'OIIFLUX', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'):
            truth[key][:] = meta[key]

    if 'elg' in targname or 'lrg' in targname or 'bgs' in targname:
        depthprefix = 'GAL'
    else:
        depthprefix = 'PSF'

    factor = 5 # -- should this be 1 or 5???

    for band in ('G', 'R', 'Z'):
        fluxkey = 'FLUX_{}'.format(band)
        depthkey = '{}DEPTH_{}'.format(depthprefix, band)
            
        sigma = 1 / np.sqrt(source_data[depthkey][indx]) / 5 # nanomaggies, 1-sigma
        targets[fluxkey][:] = truth[fluxkey] + rand.normal(scale=sigma)

    for band in ('W1', 'W2'):
        fluxkey = 'FLUX_{}'.format(band)
        depthkey = 'PSFDEPTH_{}'.format(band)
            
        sigma = 1 / np.sqrt(source_data[depthkey][indx]) / 5 # nanomaggies, 1-sigma
        targets[fluxkey][:] = truth[fluxkey] + rand.normal(scale=sigma)

    if qaplot:
        import matplotlib.pyplot as plt
        gr1 = -2.5 * np.log10( truth['FLUX_G'] / truth['FLUX_R'] )
        rz1 = -2.5 * np.log10( truth['FLUX_R'] / truth['FLUX_Z'] )
        gr = -2.5 * np.log10( targets['FLUX_G'] / targets['FLUX_R'] )
        rz = -2.5 * np.log10( targets['FLUX_R'] / targets['FLUX_Z'] )
        plt.scatter(rz1, gr1, color='red', alpha=0.5, edgecolor='none', 
                    label='Noiseless Photometry')
        plt.scatter(rz, gr, alpha=0.5, color='green', edgecolor='none',
                    label='Noisy Photometry')
        plt.xlim(-0.5, 2) ; plt.ylim(-0.5, 2)
        plt.legend(loc='upper left')
        plt.show()
        import pdb ; pdb.set_trace()

def _faintstar_targets_truth(source_data, indx, Spectra, select_targets_function, log,
                             rand, mockformat='galaxia', qaplot=False):
    """Preselect stars that are going to pass target selection cuts without actually
    generating spectra, in order to save memory and time.

    """
    if mockformat.lower() == 'galaxia':
        alldata = np.vstack((source_data['TEFF'][indx],
                             source_data['LOGG'][indx],
                             source_data['FEH'][indx])).T
        _, templateid = Spectra.tree.query('STAR', alldata)
        templateid = templateid.flatten()
    else:
        log.warning('Unrecognized mockformat {}!'.format(mockformat))
        raise ValueError

    normmag = 1e9 * 10**(-0.4 * source_data['MAG'][indx]) # nanomaggies

    # Initialize dummy targets and truth tables.
    targets, truth = _initialize_targets_truth(source_data, indx=indx)

    # Pack the noiseless photometry in the truth table, generate noisy
    # photometry, and then select targets.
    for band in ('G', 'R', 'Z', 'W1', 'W2'):
        truth['FLUX_{}'.format(band)] = getattr( Spectra.tree, 'star_flux_{}'.format(
            band.lower()) )[templateid] * normmag
        
    _scatter_photometry('faintstar', source_data, truth, targets, rand, indx=indx)

    select_targets_function(targets, truth)#, boss_std=boss_std)

    keep = np.where(targets['DESI_TARGET'] != 0)[0]
    log.info('Pre-selected {} FAINTSTAR targets.'.format(len(keep)))
    
    if len(keep) > 0:
        targets = targets[keep]
        truth = truth[keep]
        
        flux, meta = Spectra.faintstar(source_data, index=indx[keep],
                                       mockformat=mockformat)
        
        for key in ('TEMPLATEID', 'MAG', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1',
                    'FLUX_W2', 'OIIFLUX', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'):
            truth[key][:] = meta[key]
    else:
        flux, meta = [], []
    
    if qaplot:
        import matplotlib.pyplot as plt
        gr1 = -2.5 * np.log10( truth['FLUX_G'] / truth['FLUX_R'] )
        rz1 = -2.5 * np.log10( truth['FLUX_R'] / truth['FLUX_Z'] )
        gr = -2.5 * np.log10( targets['FLUX_G'] / targets['FLUX_R'] )
        rz = -2.5 * np.log10( targets['FLUX_R'] / targets['FLUX_Z'] )
        plt.scatter(rz1, gr1, color='red', alpha=0.5, edgecolor='none', 
                    label='Noiseless Photometry')
        plt.scatter(rz, gr, alpha=0.5, color='green', edgecolor='none',
                    label='Noisy Photometry')
        if len(keep) > 0:
            plt.scatter(rz1[keep], gr1[keep], color='red', edgecolor='k')
            plt.scatter(rz[keep], gr[keep], color='green', edgecolor='k')
        plt.xlim(-0.5, 2) ; plt.ylim(-0.5, 2)
        plt.legend(loc='upper left')
        plt.show()
    
    return targets, truth, flux

def _get_spectra_onepixel(specargs):
    """Filler function for the multiprocessing."""
    return get_spectra_onepixel(*specargs)

def get_spectra_onepixel(source_data, indx, MakeMock, rand, log, ntarget):
    """Wrapper function to generate spectra for all targets on a single healpixel.

    Args:
        source_data : dict
            Dictionary with all the mock data (candidate mock targets).
        indx : int or np.ndarray
            Indices of candidate mock targets to consider.
        Spectra : desitarget.mock.spectra.MockSpectra
            Object to assign spectra to each target class.
        select_targets_function : desitarget.mock.selection.SelectTargets.*_select
            Object to assign bits and select targets.
        rand : numpy.random.RandomState
           Object for random number generation.
        log : desiutil.logger
           Logger object.
        ntarget : int
           Desired number of targets to generate.

    Returns:
        targets : astropy.table.Table
            Table of mock targets.
        truth : astropy.table.Table
            Corresponding truth table.
        trueflux : numpy.ndarray
            Array [npixel, ntarget] of observed-frame spectra.  Only computed
            and returned for non-sky targets.

    """
    targname = source_data['TARGET_NAME'].lower()
    #mockformat = source_data['MOCKFORMAT'].lower()

    if len(indx) < ntarget:
        log.warning('Too few candidate targets ({}) than desired ({}).'.format(
            len(indx), ntarget))

    # Skies are a special case -- no need to chunk.
    if targname == 'sky':
        these = rand.choice(len(indx), ntarget, replace=False)
        targets, truth = _initialize_targets_truth(source_data, these)
        MakeMock.select_targets(targets, truth)
        return [targets, truth]

    # Build spectra in chunks and stop when we have enough.
    nchunk = np.ceil(len(indx) / ntarget).astype('int')
    
    targets = list()
    truth = list()
    trueflux = list()

    ntot = 0
    for ii, chunkindx in enumerate(np.array_split(indx, nchunk)):

        # Faintstar targets are a special case.
        if targname == 'faintstar':
            _targets, _truth, chunkflux = _faintstar_targets_truth(source_data, chunkindx, Spectra,
                                                                   select_targets_function,
                                                                   log, rand, mockformat=mockformat)

        else:
            _targets, _truth = _initialize_targets_truth(source_data, chunkindx)

            # Generate the spectra.
            chunkflux, _, chunkmeta = MakeMock.make_spectra(source_data, index=chunkindx)
        
            # Scatter the photometry based on the depth.
            _scatter_photometry(targname, source_data, _truth, _targets,
                                rand, meta=chunkmeta, indx=chunkindx)

            # Select targets.
            MakeMock.select_targets(_targets, _truth)#, boss_std=boss_std)

        keep = np.where(_targets['DESI_TARGET'] != 0)[0]
        nkeep = len(keep)

        log.debug('Selected {} / {} targets on chunk {} / {}.'.format(
            nkeep, len(chunkindx), ii+1, nchunk))

        if nkeep > 0:
            targets.append(_targets[keep])
            truth.append(_truth[keep])
            trueflux.append(chunkflux[keep, :])

        # If we have enough, get out!
        ntot += nkeep
        if ntot >= ntarget:
            break

    targets = vstack(targets)
    truth = vstack(truth)
    trueflux = np.concatenate(trueflux)

    # Only keep as many targets as we need.
    targets = targets[:ntarget]
    truth = truth[:ntarget]
    trueflux = trueflux[:ntarget, :]
        
    return [targets, truth, trueflux]

def _healpixel_chunks(nside, nside_chunk, log):
    """Chunk each healpixel into a smaller set of healpixels, for
    parallelization.

    """
    if nside >= nside_chunk:
        nside_chunk = nside
        
    areaperpixel = hp.nside2pixarea(nside, degrees=True)
    areaperchunk = hp.nside2pixarea(nside_chunk, degrees=True)

    nchunk = 4**np.int(np.log2(nside_chunk) - np.log2(nside))
    log.info('Dividing each nside={} healpixel into {} nside={} healpixel(s).'.format(
        nside, nchunk, nside_chunk))

    return areaperpixel, areaperchunk, nchunk

def targets_truth(params, output_dir='./', seed=None, nproc=1, nside=16,
                  nside_chunk=128, healpixels=None, verbose=False):
    """Generate a catalog of targets, spectra, and the corresponding "truth" catalog
    (with, e.g., the true redshift) for use in simulations.

    Args:
        params : dict
            Source parameters.
        seed: int
            Seed for the random number generation.
        output_dir : str
            Output directory (default '.').
        nproc : int
            Number of parallel processes to use (default 1).
        nside : int
            Healpix resolution corresponding to healpixels (default 16).
        nside_chunk : int
            Healpix resolution for chunking the sample (NB: nside_chunk must be
            <= nside).
        healpixels : numpy.ndarray or int
            Restrict the sample of mock targets analyzed to those lying inside
            this set (array) of healpix pixels.  (Default: None)
        verbose: bool
            Be verbose. (Default: False)

    Returns:
        Files 'targets.fits', 'truth.fits', 'sky.fits', 'standards-dark.fits',
        and 'standards-bright.fits' written to disk for a list of healpixels.

    """
    from time import time
    import healpy as hp
    from desitarget.internal import sharedmem

    # Initialize a bunch of objects we need.
    log, rand, healpixels = _initialize(params, verbose=verbose, seed=seed,
                                        output_dir=output_dir, nside=nside,
                                        healpixels=healpixels)

    # Chunk the sample for the multiprocessing.    
    areaperpix, areaperchunk, nchunk = _healpixel_chunks(nside, nside_chunk, log)

    # Loop over each source / object type.
    for healpix in healpixels:
        alltargets = list()
        alltruth = list()
        alltrueflux = list()
        allskytargets = list()
        allskytruth = list()

        for source_name in sorted(params['sources'].keys()):
            targets, truth, skytargets, skytruth = [], [], [], []

            # Read the data.
            log.info('Reading source : {}'.format(source_name))
            source_data, MakeMock = read_mock(source_name, params, log, seed=seed, 
                                              healpixels=healpix, nside=nside,
                                              nside_chunk=nside_chunk)

            # If there are no sources, keep going.
            if not bool(source_data):
                continue

            # Instantiate the target selection function.
            #selection_function = '{}_select'.format(source_name.lower())
            #select_targets_function = getattr(Selection, selection_function)

            # Target density -- need a proper fluctuations model here.
            # NTARGETPERCHUNK needs to be an array with the targets divided
            # among the chunk healpixels.
            if 'density' in params['sources'][source_name].keys():
                density = params['sources'][source_name]['density']
                ntarget = np.round(density * areaperpix).astype('int')
            else:
                ntarget = len(source_data['RA'])
                density = ntarget / areaperpix
            ntargetperchunk = np.repeat(np.round(ntarget / nchunk).astype('int'), nchunk)

            log.info('Goal: generate spectra for {} {} targets ({:.2f} / deg2).'.format(
                ntarget, source_name, density))

            # Generate the spectra in chunks of smaller healpixels.
            healpix_chunk = radec2pix(nside_chunk, source_data['RA'], source_data['DEC'])

            specargs = list()
            for pixchunk, ntarg in zip( set(healpix_chunk), ntargetperchunk ):
                indx = np.where( np.in1d(healpix_chunk, pixchunk)*1 )[0]
                if len(indx) > 0:
                    if len(indx) < ntarg:
                        ntarg = len(indx)
                    specargs.append( (source_data, indx, MakeMock, rand, log, ntarg) )
                    #specargs.append( (source_data, indx, Spectra, select_targets_function,
                    #                  rand, log, ntarg) )

            # Multiprocessing.
            nn = np.zeros((), dtype='i8')
            t0 = time()
            def _update_spectra_status(result):
                if nn % 2 == 0 and nn > 0:
                    rate = (time() - t0) / nn
                    log.info('Healpixel chunk {} / {} ({:.1f} sec / chunk)'.format(nn, nchunk, rate))
                nn[...] += 1    # in-place modification
                return result

            if nproc > 1:
                pool = sharedmem.MapReduce(np=nproc)
                with pool:
                    pixel_results = pool.map(_get_spectra_onepixel, specargs,
                                             reduce=_update_spectra_status)
            else:
                pixel_results = list()
                for args in specargs:
                    pixel_results.append( _update_spectra_status( _get_spectra_onepixel(args) ) )

            # Unpack the results; sky targets are a special case.
            pixel_results = list(zip(*pixel_results))

            if source_name.upper() == 'SKY':
                skytargets = vstack(pixel_results[0])
                skytruth = vstack(pixel_results[1])
                log.info('Generated {} sky targets.'.format(len(skytargets)))

                allskytargets.append(skytargets)
                allskytruth.append(skytruth)
            else:
                targets = vstack(pixel_results[0])
                truth = vstack(pixel_results[1])
                trueflux = np.concatenate(pixel_results[2])
                log.info('Done: Generated spectra for {} {} targets ({:.2f} / deg2).'.format(
                    len(targets), source_name, len(targets) / areaperpix))

                alltargets.append(targets)
                alltruth.append(truth)
                alltrueflux.append(trueflux)

            # Contaminants here?

        if len(alltargets) == 0 and len(allskytargets) == 0: # all done
            continue

        if len(alltargets) > 0:
            targets = vstack(alltargets)
            truth = vstack(alltruth)
            trueflux = np.concatenate(alltrueflux)
        else:
            targets = []

        if len(allskytargets) > 0:
            skytargets = vstack(allskytargets)
            skytruth = vstack(allskytruth)
        else:
            skytargets = []

        # Add some final columns.
        targets, truth, skytargets, skytruth = _finish_catalog(targets, truth, skytargets, skytruth,
                                                               nside, healpix, rand, log)

        # Finally, write the results.
        _write_targets_truth(targets, truth, skytargets, skytruth,  
                             nside, healpix, seed, log, output_dir)
        
def _finish_catalog(targets, truth, skytargets, skytruth, nside,
                    healpix, rand, log, use_brickid=False):
    """Adds TARGETID, SUBPRIORITY and HPXPIXEL to targets.
    
    Args:
        targets: astropy.table
            Final set of Targets. 
        truth: astropy.table
            Corresponding Truth to Targets
        skytargets: astropy.table
            Sky positions
        nside: int
            nside for healpix
        healpix: int
            healpixel
        rand: numpy.random.RandomState
           Object for random number generation.
        log: desiutil.logger
           Logger object.
        use_brickid: bool
           Assign brickid based on the brickname rather than the healpix
           numbers.  (Deprecated because it results in duplicated targetid.)
            
    Returns:
        Updated versions of: targets, truth, and skytargets.

    """
    nobj = len(targets)
    nsky = len(skytargets)
    log.info('Summary: ntargets = {}, nsky = {} in pixel {}.'.format(nobj, nsky, healpix))

    if use_brickid:

        # Assign the correct BRICKID and unique OBJIDs for every object on this brick.
        from desiutil.brick import Bricks

        brick_info = Bricks().to_table()

        if nobj > 0 and nsky > 0:
            allbrickname = set(targets['BRICKNAME']) | set(skytargets['BRICKNAME'])
        if nobj > 0 and nsky == 0:
            allbrickname = set(targets['BRICKNAME'])
        if nobj == 0 and nsky > 0:
            allbrickname = set(skytargets['BRICKNAME'])

        for brickname in allbrickname:
            iinfo = np.where(brickname == brick_info['BRICKNAME'])[0]

            nobj_brick = 0
            if nobj > 0:
                itarg = np.where(brickname == targets['BRICKNAME'])[0]
                nobj_brick = len(itarg)
                targets['BRICKID'][itarg] = brick_info['BRICKID'][iinfo]
                targets['BRICK_OBJID'][itarg] = np.arange(nobj_brick)

            if nsky > 0:
                iskytarg = np.where(brickname == skytargets['BRICKNAME'])[0]
                nsky_brick = len(iskytarg)
                skytargets['BRICKID'][iskytarg] = brick_info['BRICKID'][iinfo]
                skytargets['BRICK_OBJID'][iskytarg] = np.arange(nsky_brick) + nobj_brick

        subpriority = rand.uniform(0.0, 1.0, size=nobj+nsky)

        if nobj > 0:
            targetid = encode_targetid(objid=targets['BRICK_OBJID'], brickid=targets['BRICKID'], mock=1)
            truth['TARGETID'][:] = targetid
            targets['TARGETID'][:] = targetid
            targets['SUBPRIORITY'][:] = subpriority[:nobj]

            targets['HPXPIXEL'][:] = radec2pix(nside, targets['RA'], targets['DEC'])

        if nsky > 0:
            skytargetid = encode_targetid(objid=skytargets['BRICK_OBJID'], brickid=skytargets['BRICKID'], mock=1, sky=1)
            skytargets['TARGETID'][:] = skytargetid
            skytargets['SUBPRIORITY'][:] = subpriority[nobj:]

            skytargets['HPXPIXEL'][:] = radec2pix(nside, skytargets['RA'], skytargets['DEC'])
                
    else:
        objid = np.arange(nobj + nsky)
        targetid = encode_targetid(objid=objid, brickid=healpix, mock=1)
        subpriority = rand.uniform(0.0, 1.0, size=nobj + nsky)

        if nobj > 0:
            targets['BRICKID'][:] = healpix
            targets['BRICK_OBJID'][:] = objid[:nobj]
            targets['TARGETID'][:] = targetid[:nobj]
            targets['SUBPRIORITY'][:] = subpriority[:nobj]
            truth['TARGETID'][:] = targetid[:nobj]

        if nsky > 0:
            skytargets['BRICKID'][:] = healpix
            skytargets['BRICK_OBJID'][:] = objid[nobj:]
            skytargets['TARGETID'][:] = targetid[nobj:]
            skytargets['SUBPRIORITY'][:] = subpriority[nobj:]
            skytruth['TARGETID'][:] = targetid[nobj:]
        
    return targets, truth, skytargets, skytruth

def _write_targets_truth(targets, truth, skytargets, skytruth, nside,
                         healpix_id, seed, log, output_dir):
    """Writes targets to disk.
    
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
        log: logger object
        output_dir: str
            Directory where the outputs are written.
            
    Output:
        Files "targets", "truth", "sky", "standards" written to disk.
    
    """
    from astropy.io import fits
    from desiutil import depend
    from desispec.io.util import fitsheader, write_bintable
    
    nobj = len(targets)
    nsky = len(skytargets)
    
    if seed is None:
        seed1 = 'None'
    else:
        seed1 = seed
    truthhdr = fitsheader(dict(
        SEED = (seed1, 'initial random seed')
        ))

    targetshdr = fitsheader(dict(
        SEED = (seed1, 'initial random seed')
        ))
    targetshdr['HPXNSIDE'] = (nside, 'HEALPix nside')
    targetshdr['HPXNEST'] = (True, 'HEALPix nested (not ring) ordering')

    outdir = mockio.get_healpix_dir(nside, healpix_id, basedir=output_dir)
    os.makedirs(outdir, exist_ok=True)

    # Write out the sky catalog.
    skyfile = mockio.findfile('sky', nside, healpix_id, basedir=output_dir)
    if nsky > 0:
        log.info('Writing {} SKY targets to {}'.format(nsky, skyfile))
        write_bintable(skyfile+'.tmp', skytargets, extname='SKY',
                               header=targetshdr, clobber=True)
        os.rename(skyfile+'.tmp', skyfile)
    else:
        log.info('No sky targets generated; {} not written.'.format(skyfile))
        log.info('  Sky file {} not written.'.format(skyfile))

    if nobj > 0:
    # Write out the dark- and bright-time standard stars.
        for stdsuffix, stdbit in zip(('dark', 'bright'), ('STD_FSTAR', 'STD_BRIGHT')):
            stdfile = mockio.findfile('standards-{}'.format(stdsuffix), nside, healpix_id, basedir=output_dir)
            istd   = (((targets['DESI_TARGET'] & desi_mask.mask(stdbit)) | 
                   (targets['DESI_TARGET'] & desi_mask.mask('STD_WD')) ) != 0)

            if np.count_nonzero(istd) > 0:
                log.info('Writing {} {} standards to {}'.format(np.sum(istd), stdsuffix.upper(), stdfile))
                write_bintable(stdfile+'.tmp', targets[istd], extname='STD',
                               header=targetshdr, clobber=True)
                os.rename(stdfile+'.tmp', stdfile)
            else:
                log.info('No {} standards stars selected.'.format(stdsuffix))
                log.info('  Standard star file {} not written.'.format(stdfile))

        # Finally write out the rest of the targets.
        targetsfile = mockio.findfile('targets', nside, healpix_id, basedir=output_dir)
        truthfile = mockio.findfile('truth', nside, healpix_id, basedir=output_dir)
   
        log.info('Writing {} targets to:'.format(nobj))
        log.info('  {}'.format(targetsfile))
        targets.meta['EXTNAME'] = 'TARGETS'
        write_bintable(targetsfile+'.tmp', targets, extname='TARGETS',
                          header=targetshdr, clobber=True)
        os.rename(targetsfile+'.tmp', targetsfile)

        log.info('  {}'.format(truthfile))
        hx = fits.HDUList()
        hdu = fits.convenience.table_to_hdu(truth)
        hdu.header['EXTNAME'] = 'TRUTH'
        hx.append(hdu)

        try:
            hx.writeto(truthfile+'.tmp', overwrite=True)
        except:
            hx.writeto(truthfile+'.tmp', clobber=True)
        os.rename(truthfile+'.tmp', truthfile)

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

    source_data['SOURCE_NAME'] = source_name
    source_data['MOCKFORMAT'] = mockformat

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

def merge_file_tables(fileglob, ext, outfile=None, comm=None):
    '''
    parallel merge tables from individual files into an output file

    Args:
        comm: MPI communicator object
        fileglob (str): glob of files to combine (e.g. '*/blat-*.fits')
        ext (str or int): FITS file extension name or number
        outfile (str): output file to write

    Returns merged table as np.ndarray
    '''
    import fitsio
    import glob
    from desiutil.log import get_logger
    
    if comm is not None:
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        size = 1
        rank = 0

    if rank == 0:
        infiles = sorted(glob.glob(fileglob))
    else:
        infiles = None

    if comm is not None:
        infiles = comm.bcast(infiles, root=0)
 
    if len(infiles)==0:
        log = get_logger()
        log.info('Zero pixel files for extension {}. Skipping.'.format(ext))
        return
    
    #- Each rank reads and combines a different set of files
    data = np.hstack( [fitsio.read(x, ext) for x in infiles[rank::size]] )

    if comm is not None:
        data = comm.gather(data, root=0)
        if rank == 0 and size>1:
            data = np.hstack(data)

    if rank == 0 and outfile is not None:
        log = get_logger()
        log.info('Writing {}'.format(outfile))
        header = fitsio.read_header(infiles[0], ext)
        tmpout = outfile + '.tmp'
        
        # Find duplicates
        vals, idx_start, count = np.unique(data['TARGETID'], return_index=True, return_counts=True)
        if len(vals) != len(data):
            log.warning('Non-unique TARGETIDs found!')
            raise ValueError
        
        fitsio.write(tmpout, data, header=header, extname=ext, clobber=True)
        os.rename(tmpout, outfile)

    return data

def join_targets_truth(mockdir, outdir=None, force=False, comm=None):
    '''
    Join individual healpixel targets and truth files into combined tables

    Args:
        mockdir: top level mock target directory

    Options:
        outdir: output directory, default to mockdir
        force: rewrite outputs even if they already exist
        comm: MPI communicator; if not None, read data in parallel
    '''
    import fitsio
    if outdir is None:
        outdir = mockdir

    if comm is not None:
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        comm = None
        size = 1
        rank = 0
    
    #- Use rank 0 to check pre-existing files to avoid N>>1 ranks hitting the disk
    if rank == 0:
        todo = dict()
        todo['sky'] = not os.path.exists(outdir+'/sky.fits') or force
        todo['stddark'] = not os.path.exists(outdir+'/standards-dark.fits') or force
        todo['stdbright'] = not os.path.exists(outdir+'/standards-bright.fits') or force
        todo['targets'] = not os.path.exists(outdir+'/targets.fits') or force
        todo['truth'] = not os.path.exists(outdir+'/truth.fits') or force
        todo['mtl'] = not os.path.exists(outdir+'/mtl.fits') or force
    else:
        todo = None

    if comm is not None:
        todo = comm.bcast(todo, root=0)

    if todo['sky']:
        merge_file_tables(mockdir+'/*/*/sky-*.fits', 'SKY',
                    outfile=outdir+'/sky.fits', comm=comm)

    if todo['stddark']:
        merge_file_tables(mockdir+'/*/*/standards-dark*.fits', 'STD',
                    outfile=outdir+'/standards-dark.fits', comm=comm)

    if todo['stdbright']:
        merge_file_tables(mockdir+'/*/*/standards-bright*.fits', 'STD',
                    outfile=outdir+'/standards-bright.fits', comm=comm)

    if todo['targets']:
        merge_file_tables(mockdir+'/*/*/targets-*.fits', 'TARGETS',
                    outfile=outdir+'/targets.fits', comm=comm)

    if todo['truth']:
        merge_file_tables(mockdir+'/*/*/truth-*.fits', 'TRUTH',
                    outfile=outdir+'/truth.fits', comm=comm)

    #- Make initial merged target list (MTL) using rank 0
    if rank == 0 and todo['mtl']:
        from desitarget import mtl
        from desiutil.log import get_logger
        log = get_logger()
        out_mtl = os.path.join(outdir, 'mtl.fits')
        log.info('Generating merged target list {}'.format(out_mtl))
        targets = fitsio.read(outdir+'/targets.fits')
        mtl = mtl.make_mtl(targets)
        tmpout = out_mtl+'.tmp'
        mtl.meta['EXTNAME'] = 'MTL'
        mtl.write(tmpout, overwrite=True, format='fits')
        os.rename(tmpout, out_mtl)

