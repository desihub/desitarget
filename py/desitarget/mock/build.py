# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.build
=====================

Build truth and targets catalogs, including spectra, for the mocks.

"""
from __future__ import absolute_import, division, print_function

import os, time
import numpy as np
import healpy as hp

from astropy.table import vstack

from desimodel.footprint import radec2pix

def initialize_targets_truth(params, healpixels=None, nside=None, output_dir='.', 
                             seed=None, verbose=False):
    """Initialize various objects needed to generate mock targets.

    Parameters
    ----------
    params : :class:`dict`
        Dictionary defining the mock from which to generate targets.
    healpixels : :class:`numpy.ndarray` or :class:`int`
        Generate mock targets within this set of healpix pixels.
    nside : :class:`int`
        Healpix resolution corresponding to healpixels.
    output_dir : :class:`str`, optional.
        Output directory.  Defaults to '.' (current directory).
    seed: :class:`int`, optional
        Seed for the random number generator.  Defaults to None.
    verbose: :class:`bool`, optional
        Be verbose. Defaults to False.

    Returns
    -------
    log : :class:`desiutil.logger`
       Logger object.
    healpixseeds : :class:`numpy.ndarray` or :class:`int`
       Array of random number seeds (one per healpixels pixel) needed to ensure
       reproducibility.

    Raises
    ------
    ValueError
        If params, healpixels, or nside are not defined.  A ValueError is also
        raised if nside > 256, since this exceeds the number of bits that can be
        accommodated by desitarget.targets.encode_targetid.

    """
    from desiutil.log import get_logger, DEBUG

    if params is None:
        log.fatal('PARAMS input is required.')
        raise ValueError

    if healpixels is None:
        log.fatal('HEALPIXELS input is required.')
        raise ValueError
        
    if nside is None:
        log.fatal('NSIDE input is required.')
        raise ValueError

    if nside > 256:
        log.warning('NSIDE = {} exceeds the number of bits available for BRICKID in targets.encode_targetid.')
        raise ValueError

    if verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()
    
    npix = len(np.atleast_1d(healpixels))

    # Initialize the random seed
    rand = np.random.RandomState(seed)
    healpixseeds = rand.randint(2**31, size=npix)

    # Create the output directories
    if os.path.exists(output_dir):
        if os.listdir(output_dir):
            log.warning('Output directory {} is not empty.'.format(output_dir))
    else:
        log.info('Creating directory {}'.format(output_dir))
        os.makedirs(output_dir)    
    log.info('Writing to output directory {}'.format(output_dir))      
        
    areaperpix = hp.nside2pixarea(nside, degrees=True)
    log.info('Processing {} healpixel(s) (nside = {}, {:.3f} deg2/pixel) spanning {:.3f} deg2.'.format(
        len(healpixels), nside, areaperpix, npix * areaperpix))

    return log, healpixseeds
    
def read_mock(params, log=None, target_name='', seed=None, healpixels=None,
              nside=None, nside_chunk=128, MakeMock=None):
    """Read a mock catalog.
    
    Parameters
    ----------
    params : :class:`dict`
        Dictionary summary of the input configuration file, restricted to a
        particular target (e.g., 'QSO').
    log : :class:`desiutil.logger`
        Logger object.
    target_name : :class:`str`
        Target name; mock.mockmaker.[TARGET_NAME]Maker class to instantiate. 
    seed: :class:`int`, optional
        Seed for the random number generator.  Defaults to None.
    healpixels : :class:`numpy.ndarray` or `int`
        List of healpixels to read.
    nside : :class:`int`
        Healpix resolution corresponding to healpixels.
    nside_chunk : :class:`int`, optional
        Healpix resolution for chunking the sample to avoid memory problems.
        (NB: nside_chunk must be <= nside).  Defaults to 128.
            
    Returns
    -------
    data : :class:`dict`
        Parsed target data based on the input mock catalog (to be documented).

    Raises
    ------
    ValueError
        If the mock_density was not returned when expected.

    """
    from desitarget.mock import mockmaker

    target_type = params.get('target_type')
    mockfile = params.get('mockfile')
    mockformat = params.get('format')
    magcut = params.get('magcut')
    nside_galaxia = params.get('nside_galaxia')
    calib_only = params.get('calib_only', False)

    # QSO/Lya parameters
    nside_lya = params.get('nside_lya')
    zmin_lya = params.get('zmin_lya')
    zmax_qso = params.get('zmax_qso')
    use_simqso = params.get('use_simqso', True)
    balprob = params.get('balprob', 0.0)
    add_dla = params.get('add_dla', False)

    if 'density' in params.keys():
        mock_density = True
    else:
        mock_density = False

    log.info('Target: {}, type: {}, format: {}, mockfile: {}'.format(
        target_name, target_type, mockformat, mockfile))

    if MakeMock is None:
        MakeMock = getattr(mockmaker, '{}Maker'.format(target_name))(seed=seed, nside_chunk=nside_chunk,
                                                                     calib_only=calib_only,
                                                                     use_simqso=use_simqso,
                                                                     balprob=balprob,
                                                                     add_dla=add_dla)
    else:
        MakeMock.seed = seed # updated seed
        
    data = MakeMock.read(mockfile=mockfile, mockformat=mockformat,
                         healpixels=healpixels, nside=nside,
                         magcut=magcut, nside_lya=nside_lya,
                         zmin_lya=zmin_lya, zmax_qso=zmax_qso,
                         nside_galaxia=nside_galaxia, mock_density=mock_density)
    if not bool(data):
        return data, MakeMock

    # Add the information we need to incorporate density fluctuations.
    if 'density' in params.keys():
        if 'MOCK_DENSITY' not in data.keys():
            log.warning('Expected mock_density value not found!')
            raise ValueError
        
        data['DENSITY'] = params['density']
        data['DENSITY_FACTOR'] = data['DENSITY'] / data['MOCK_DENSITY']
        if data['DENSITY_FACTOR'] > 1:
            log.warning('Density factor {} should not be > 1!'.format(data['DENSITY_FACTOR']))
            data['DENSITY_FACTOR'] = 1.0
        
        data['MAXITER'] = 5

        log.info('Computed median mock density for {}s of {:.2f} targets/deg2.'.format(
            target_name, data['MOCK_DENSITY']))
        log.info('Target density = {:.2f} targets/deg2 (downsampling factor = {:.3f}).'.format(
            data['DENSITY'], data['DENSITY_FACTOR']))
    else:
        data['DENSITY_FACTOR'] = 1.0 # keep them all
        data['MAXITER'] = 1

    return data, MakeMock

def _get_spectra_onepixel(specargs):
    """Filler function for the multiprocessing."""
    return get_spectra_onepixel(*specargs)

def get_spectra_onepixel(data, indx, MakeMock, seed, log, ntarget,
                         maxiter=1, no_spectra=False, calib_only=False):
    """Wrapper function to generate spectra for all targets on a single healpixel.

    Parameters
    ----------
    data : :class:`dict`
        Dictionary with all the mock data (candidate mock targets).
    indx : :class:`int` or :class:`numpy.ndarray`
        Indices of candidate mock targets to consider.
    MakeMock : :class:`desitarget.mock.mockmaker` object
        Object to assign spectra to each target class.
    seed: :class:`int`
        Seed for the random number generator.
    log : :class:`desiutil.logger`
       Logger object.
    ntarget : :class:`int`
       Desired number of targets to generate.
    maxiter : :class:`int`
       Maximum number of iterations to generate targets.
    no_spectra : :class:`bool`, optional
        Do not generate spectra, e.g., for use with quicksurvey.  Defaults to False.
    calib_only : :class:`bool`, optional
        Use targets as calibration (standard star) targets, only. Defaults to False.

    Returns
    -------
    targets : :class:`astropy.table.Table`
        Target catalog.
    truth : :class:`astropy.table.Table`
        Corresponding truth table.
    objtruth : :class:`astropy.table.Table`
        Corresponding objtype-specific truth table (if applicable).
    trueflux : :class:`numpy.ndarray`
        Array [npixel, ntarget] of observed-frame spectra.  Only computed
        and returned for non-sky targets and if no_spectra=False.

    """
    targname = data['TARGET_NAME']

    rand = np.random.RandomState(seed)

    targets = list()
    truth = list()
    objtruth = list()
    trueflux = list()

    if ntarget == 0:
        return [targets, truth, objtruth, trueflux]

    # Generate the spectra iteratively until we achieve the required target
    # density.  Randomly divide the possible targets into each iteration.
    iterseeds = rand.randint(2**31, size=maxiter)
    rand.shuffle(indx)
    iterindx = np.array_split(indx, maxiter)

    makemore, itercount, ntot = True, 0, 0
    while makemore:
        chunkflux, _, chunktargets, chunktruth, chunkobjtruth = MakeMock.make_spectra(
            data, indx=iterindx[itercount], seed=iterseeds[itercount], no_spectra=no_spectra)

        MakeMock.select_targets(chunktargets, chunktruth, targetname=data['TARGET_NAME'])
        
        keep = np.where(chunktargets['DESI_TARGET'] != 0)[0]
        #if 'CONTAM_NAME' in data.keys():
        #    import pdb ; pdb.set_trace()

        nkeep = len(keep)
        if nkeep > 0:
            ntot += nkeep
            log.debug('Generated {} / {} ({} / {} total) {} targets on iteration {} / {}.'.format(
                nkeep, len(chunktargets), ntot, ntarget, targname, itercount+1, maxiter))

            targets.append(chunktargets[keep])
            truth.append(chunktruth[keep])
            if len(chunkobjtruth) > 0: # skies have no objtruth
                objtruth.append(chunkobjtruth[keep])
            if not no_spectra:
                trueflux.append(chunkflux[keep, :])

        itercount += 1
        if itercount == maxiter or ntot >= ntarget:
            if maxiter > 1:
                log.debug('Generated {} / {} {} targets after {} iterations.'.format(
                    ntot, ntarget, targname, itercount))
            makemore = False
        else:
            need = np.where(chunktargets['DESI_TARGET'] == 0)[0]

            #import matplotlib.pyplot as plt
            #noneed = np.where(chunktargets['DESI_TARGET'] != 0)[0]
            #gr = -2.5 * np.log10( chunktargets['FLUX_G'] / chunktargets['FLUX_R'] )
            #rz = -2.5 * np.log10( chunktargets['FLUX_R'] / chunktargets['FLUX_Z'] )
            #plt.scatter(rz[noneed], gr[noneed], color='red', alpha=0.5, edgecolor='none', label='Made Cuts')
            #plt.scatter(rz[need], gr[need], alpha=0.5, color='green', edgecolor='none', label='Failed Cuts')
            #plt.legend(loc='upper left')
            #plt.show()

            if len(need) > 0:
                # Distribute the objects that didn't pass target selection
                # to the remaining iterations.
                iterneed = np.array_split(iterindx[itercount - 1][need], maxiter - itercount)
                for ii in range(maxiter - itercount):
                    iterindx[ii + itercount] = np.hstack( (iterindx[itercount:][ii], iterneed[ii]) )

    if len(targets) > 0:
        targets = vstack(targets)
        truth = vstack(truth)
        if ntot > ntarget: # Only keep up to the number of desired targets.
            log.debug('Removing {} extraneous targets.'.format(ntot - ntarget))
            keep = rand.choice(ntot, size=ntarget, replace=False)
            targets = targets[keep]
            truth = truth[keep]
        if len(objtruth) > 0: # skies have no objtruth
            objtruth = vstack(objtruth)
            if ntot > ntarget:
                objtruth = objtruth[keep]
        if not no_spectra:
            trueflux = np.concatenate(trueflux)
            if ntot > ntarget:
                trueflux = trueflux[keep, :]

    return [targets, truth, objtruth, trueflux]

def density_fluctuations(data, log, nside, nside_chunk, seed=None):
    """Determine the density of targets to generate, accounting for fluctuations due
    to reddening, imaging systematics, and large-scale structure.

    Parameters
    ----------
    data : :class:`dict`
        Data on the input mock targets (to be documented).
    log : :class:`desiutil.logger`
        Logger object.
    nside : :class:`int`
        Healpix resolution.
    nside_chunk : :class:`int`
        Healpix resolution for chunking the sample.
    seed: :class:`int`, optional
        Seed for the random number generator.  Defaults to None.

    Returns
    -------
    indxperchunk : :class:`list`
        Indices (in data) of the mock targets to generate per healpixel chunk. 
    ntargperchunk : :class:`numpy.ndarray`
        Number of targets to generate per healpixel chunk.
    areaperpixel : :class:`float`
        Area per healpixel (used to construct useful log messages).
    
    """
    rand = np.random.RandomState(seed)

    ## Fluctuations model coefficients from --
    ##   https://github.com/desihub/desitarget/blob/master/doc/nb/target-fluctuations.ipynb
    #model = dict()
    #model['LRG'] = (0.27216, 2.631, 0.145) # slope, intercept, and scatter
    #model['ELG'] = (-0.55792, 3.380, 0.081)
    #model['QSO'] = (0.33321, 3.249, 0.112)
    #coeff = model.get(data['TARGET_NAME'])
    
    # Chunk each healpixel into a smaller set of healpixels, for
    # parallelization.
    if nside >= nside_chunk:
        log.warning('Nside must be <= nside_chunk.')
        nside_chunk = nside
        
    areaperpixel = hp.nside2pixarea(nside, degrees=True)
    areaperchunk = hp.nside2pixarea(nside_chunk, degrees=True)

    nchunk = 4**np.int(np.log2(nside_chunk) - np.log2(nside))
    log.info('Dividing each nside={} healpixel ({:.2f} deg2) into {} nside={} healpixel(s) ({:.2f} deg2).'.format(
        nside, areaperpixel, nchunk, nside_chunk, areaperchunk))

    # Assign targets to healpix chunks.
    #ntarget = len(data['RA'])
    healpix_chunk = radec2pix(nside_chunk, data['RA'], data['DEC'])

    #if 'CONTAM_FACTOR' in data.keys():
    #    # density model here!
    #    density_factor = data.get('CONTAM_FACTOR')
    #else:
    #    density_factor = data.get('DENSITY_FACTOR')

    density_factor = data.get('DENSITY_FACTOR')        

    indxperchunk, ntargperchunk = list(), list()
    for pixchunk in set(healpix_chunk):

        # Subsample the targets on this mini healpixel.
        allindxthispix = np.where( np.in1d(healpix_chunk, pixchunk)*1 )[0]

        if 'CONTAM_NUMBER' in data.keys():
            ntargthispix = np.round( data['CONTAM_NUMBER'] / nchunk ).astype(int)
            indxthispix = rand.choice(allindxthispix, size=5 * ntargthispix, replace=False) # fudge factor!
        else:
            ntargthispix = np.round( len(allindxthispix) * density_factor ).astype('int')
            indxthispix = allindxthispix
        #indxthispix = rand.choice(allindxthispix, size=ntargthispix, replace=False)

        indxperchunk.append(indxthispix)
        ntargperchunk.append(ntargthispix)

        #print(pixchunk, ntargthispix, ntargthispix / areaperchunk)
        #if coeff:
        #    # Number of targets in this chunk, based on the fluctuations model.
        #    denschunk = density * 10**( np.polyval(coeff[:2], data['EBV'][indx]) - np.polyval(coeff[:2], 0) +
        #                                rand.normal(scale=coeff[2]) )            # [ntarget/deg2]
        #    ntarg = np.rint( np.median(denschunk) * areaperchunk ).astype('int') # [ntarget]
        #    ntargetperchunk.append(ntarg)
        #else:
        #    # Divide the targets evenly among chunks.
        #    ntargetperchunk = np.repeat(np.round(ntarget / nchunk).astype('int'), nchunk)

    ntargperchunk = np.array(ntargperchunk)

    # Special case when the number of targets is very small.
    if np.sum(ntargperchunk) == 0:
        ntargperchunk[0] = np.round( len(data['RA']) * density_factor ).astype('int')
        
    return indxperchunk, ntargperchunk, areaperpixel

def get_spectra(data, MakeMock, log, nside, nside_chunk, seed=None,
                nproc=1, sky=False, no_spectra=False, calib_only=False,
                contaminants=False):
    """Generate spectra (in parallel) for a set of targets.

    Parameters
    ----------
    data : :class:`dict`
        Data on the input mock targets (to be documented).
    MakeMock : :class:`desitarget.mock.mockmaker` object
        Object to assign spectra to each target class.
    log : :class:`desiutil.logger`
       Logger object.
    nside : :class:`int`
        Healpix resolution corresponding to healpixels.
    nside_chunk : :class:`int`
        Healpix resolution for chunking the sample to avoid memory problems.
    seed: :class:`int`, optional
        Seed for the random number generator.  Defaults to None.
    nproc : :class:`int`, optional
        Number of parallel processes to use.  Defaults to 1.
    sky : :class:`bool`
        Processing sky targets (which are a special case).  Defaults to False.
    no_spectra : :class:`bool`, optional
        Do not generate spectra, e.g., for use with quicksurvey.  Defaults to False.
    calib_only : :class:`bool`, optional
        Use targets as calibration (standard star) targets, only. Defaults to False.
    contaminants : :class:`bool`, optional
        Generate spectra for contaminants (mostly affects the log
        messages). Defaults to False.

    Returns
    -------
    targets : :class:`astropy.table.Table`
        Target catalog.
    truth : :class:`astropy.table.Table`
        Corresponding truth table.
    objtruth : :class:`astropy.table.Table`
        Corresponding objtype-specific truth table (if applicable).
    trueflux : :class:`numpy.ndarray`
        Corresponding noiseless spectra.

    """
    from time import time
    from desitarget.internal import sharedmem
    
    # Parallelize by chunking the sample into smaller healpixels and
    # determine the number of targets per chunk.
    indxperchunk, ntargperchunk, area = density_fluctuations(
        data, log, nside=nside, nside_chunk=nside_chunk, seed=seed)

    maxiter = data.get('MAXITER')

    nchunk = len(indxperchunk)
    nalltarget = np.sum(ntargperchunk)
    if contaminants:
        log.info('Goal: Generate spectra for {} {} is {} contaminants ({:.2f} / deg2).'.format(
            nalltarget, data['TARGET_NAME'], data['CONTAM_NAME'], nalltarget / area))
    else:
        log.info('Goal: Generate spectra for {} {} targets ({:.2f} / deg2).'.format(
            nalltarget, data['TARGET_NAME'], nalltarget / area))

    rand = np.random.RandomState(seed)
    chunkseeds = rand.randint(2**31, size=nchunk)

    # Set up the multiprocessing.
    specargs = list()
    for indx, ntarg, chunkseed in zip( indxperchunk, ntargperchunk, chunkseeds ):
        if len(indx) > 0:
            specargs.append( (data, indx, MakeMock, chunkseed, log,
                              ntarg, maxiter, no_spectra, calib_only) )

    nn = np.zeros((), dtype='i8')
    t0 = time()
    def _update_spectra_status(result):
        """Status update."""
        if nn % 2 == 0 and nn > 0:
            rate = (time() - t0) / nn
            log.debug('Healpixel chunk {} / {} ({:.1f} sec / chunk)'.format(nn, nchunk, rate))
        nn[...] += 1    # in-place modification
        return result
    
    if nproc > 1:
        pool = sharedmem.MapReduce(np=nproc)
        with pool:
            results = pool.map(_get_spectra_onepixel, specargs,
                               reduce=_update_spectra_status)
    else:
        results = list()
        for args in specargs:
            results.append( _update_spectra_status( _get_spectra_onepixel(args) ) )
    ttime = time() - t0

    # Unpack the results and return; note that sky targets are a special case.
    results = list(zip(*results))

    targets, truth, objtruth, good = [], [], [], []
    for ii, (targ, tru, objtru) in enumerate( zip(results[0], results[1], results[2]) ):
        if len(targ) != len(tru):
            log.warning('Mismatching targets and truth tables!')
            raise ValueError
        if len(targ) > 0:
            good.append(ii)
            targets.append(targ)
            truth.append(tru)
            if len(objtru) > 0: # skies have no objtruth
                if len(targ) != len(objtru):
                    log.warning('Mismatching targets and objtruth tables!')
                    raise ValueError
                objtruth.append(objtru)
               
    if len(targets) > 0:
        targets = vstack(targets)
        truth = vstack(truth)
        if len(objtruth) > 0: # skies have no objtruth
            objtruth = vstack(objtruth)
        good = np.array(good)

    if sky:
        trueflux = []
    else:
        if no_spectra:
            trueflux = []
        else:
            if len(good) > 0:
                trueflux = np.concatenate(np.array(results[3])[good])
            else:
                trueflux = []

    if contaminants:
        log.info('Done: Generated spectra for {} {} is {} targets ({:.2f} / deg2).'.format(
            len(targets), data['TARGET_NAME'], data['CONTAM_NAME'], len(targets) / area))
        log.info('Total time for {} is {}s = {:.3f} minutes ({:.3f} cpu minutes/deg2).'.format(
            data['TARGET_NAME'], data['CONTAM_NAME'], ttime / 60, (ttime*nproc) / area ))
    else:
        log.info('Done: Generated spectra for {} {} targets ({:.2f} / deg2).'.format(
            len(targets), data['TARGET_NAME'], len(targets) / area))
        log.info('Total time for {}s = {:.3f} minutes ({:.3f} cpu minutes/deg2).'.format(
            data['TARGET_NAME'], ttime / 60, (ttime*nproc) / area ))

    return targets, truth, objtruth, trueflux

def get_contaminants_onepixel(params, healpix, nside, seed, nproc, log,
                              nside_chunk, targets, truth, objtruth, trueflux,
                              ContamStarsMock=None, ContamGalaxiesMock=None,
                              no_spectra=False):
    """Generate spectra (in parallel) for a set of targets.

    Parameters
    ----------
    params : :class:`dict`
        Dictionary defining the type and number of contaminants.
    healpix : : :class:`int`
        Healpixel number.
    nside : :class:`int`
        Nside corresponding to healpix.
    seed : :class:`int`, optional
        Seed for the random number generation.
    nproc : :class:`int`, optional
        Number of parallel processes to use.
    log : :class:`desiutil.logger`
       Logger object.
    nside_chunk : :class:`int`
        Healpix resolution for chunking the sample to avoid memory problems.
    targets : :class:`astropy.table.Table`
        Target catalog.
    truth : :class:`astropy.table.Table`
        Corresponding truth table.
    objtruth : :class:`astropy.table.Table`
        Corresponding objtype-specific truth table (if applicable).
    trueflux : :class:`numpy.ndarray`
        Array [npixel, ntarget] of observed-frame spectra.  Only computed
        and returned for non-sky targets and if no_spectra=False.
    ContamStarsMock : :class:`desitarget.mock.mockmaker` object
        Maker Class for generating stellar contaminants.
    ContamGalaxiesMock : :class:`desitarget.mock.mockmaker` object
        Maker Class for generating extragalactic contaminants.
    no_spectra : :class:`bool`, optional
        Do not generate spectra, e.g., for use with quicksurvey.  Defaults to False.

    Returns
    -------
    targets : :class:`astropy.table.Table`
        Target catalog.
    truth : :class:`astropy.table.Table`
        Corresponding truth table.
    objtruth : :class:`astropy.table.Table`
        Corresponding objtype-specific truth table (if applicable).
    trueflux : :class:`numpy.ndarray`
        Corresponding noiseless spectra.

    """
    # Stars--
    stars_targets, stars_truth = list(), list()
    if ContamStarsMock is not None:
        _, star_params = list(params['contaminants']['stars'].items())[0]

        # Read and cache the candidate stellar contaminants.
        if 'FAINTSTAR' in star_params:
            faintstar_mockfile = star_params['FAINTSTAR']['mockfile']
            faintstar_magcut = star_params['FAINTSTAR'].get('magcut', None)
        else:
            faintstar_mockfile, faintstar_magcut = None, None

        star_data = ContamStarsMock.read(mockfile=star_params['mockfile'],
                                         mockformat=star_params['format'],
                                         healpixels=healpix, nside=nside,
                                         magcut=star_params.get('magcut', None),
                                         nside_galaxia=star_params['nside_galaxia'],
                                         faintstar_mockfile=faintstar_mockfile,
                                         faintstar_magcut=faintstar_magcut,
                                         target_name='CONTAM_STAR', seed=seed)
        nobj = len(star_data['RA'])
        star_data['MAXITER'] = 5
        star_data['CONTAM_FACTOR'] = 0.0

        # Now iterate over every target class.
        for target_type in params['contaminants']['targets']:
            cparams = params['contaminants']['targets'][target_type]

            if target_type in params['targets'] and 'stars' in cparams.keys():
                log.info('Generating {:.1f}% stellar contaminants for target class {}.'.format(
                    100*cparams['stars'], target_type))

                # BGS have TYPE!=PSF so make the stellar contaminants TYPE=REX
                if target_type == 'BGS':
                    morph = 'REX'
                    mask_type = 'BGS_ANY'
                else:
                    morph = None
                    mask_type = target_type
                ntarg = np.sum(targets['DESI_TARGET'] & ContamStarsMock.desi_mask.mask(mask_type) != 0)

                if ntarg > 0:
                    star_data['TARGET_NAME'] = target_type
                    star_data['CONTAM_NAME'] = 'CONTAM_STAR'

                    # ToDo: Modulate the contamination with Galactic latitude...
                    star_data['CONTAM_NUMBER'] = np.round( cparams['stars'] * ntarg ).astype(int)
                    star_data['CONTAM_FACTOR'] = cparams['stars'] * ntarg / len(star_data['RA'])

                    # Sample from the appropriate Gaussian mixture model and
                    # then generate the spectra.
                    if target_type == 'LRG':
                        mag = star_data['ZMAG']
                    else:
                        mag = star_data['MAG']
                    
                    gmmout = ContamStarsMock.sample_GMM(nobj, target=target_type, morph=morph,
                                                        isouth=star_data['SOUTH'],
                                                        seed=seed, prior_mag=mag)
                    star_data.update(gmmout)

                    contamtargets, contamtruth, contamobjtruth, contamtrueflux = get_spectra(
                        star_data, ContamStarsMock, log, nside=nside, nside_chunk=nside_chunk,
                        seed=seed, nproc=nproc, no_spectra=no_spectra, contaminants=True)

                    if len(contamtargets) > 0:
                        stars_targets.append(contamtargets)
                        stars_truth.append(contamtruth)
                        if 'STAR' in objtruth.keys():
                            objtruth['STAR'] = vstack( (objtruth['STAR'], contamobjtruth) )
                        else:
                            objtruth['STAR'] = contamobjtruth
                        trueflux = np.vstack( (trueflux, contamtrueflux) )

        if len(stars_targets) > 0:
            stars_targets = vstack(stars_targets)
            stars_truth = vstack(stars_truth)

    # Galaxies--
    galaxies_targets, galaxies_truth = list(), list()
    if ContamGalaxiesMock is not None:
        _, galaxy_params = list(params['contaminants']['galaxies'].items())[0]

        galaxy_data = ContamGalaxiesMock.read(mockfile=galaxy_params['mockfile'],
                                              mockformat=galaxy_params['format'],
                                              healpixels=healpix, nside=nside,
                                              magcut=galaxy_params.get('magcut', None),
                                              nside_galaxia=galaxy_params['nside_buzzard'],
                                              target_name='CONTAM_GALAXY', seed=seed)
        nobj = len(galaxy_data['RA'])
        galaxy_data['MAXITER'] = 5
        galaxy_data['CONTAM_FACTOR'] = 0.0 # fraction of candidate contaminants to keep

        # Now iterate over every target class.
        for target_type in params['contaminants']['targets']:
            cparams = params['contaminants']['targets'][target_type]

            if target_type in params['targets'] and 'galaxies' in cparams.keys():
                log.info('Generating {:.1f}% extragalactic contaminants for target class {}.'.format(
                    100*cparams['galaxies'], target_type))

                morph = None
                mask_type = target_type
                ntarg = np.sum(targets['DESI_TARGET'] & ContamGalaxiesMock.desi_mask.mask(mask_type) != 0)

                if ntarg > 0:
                    galaxy_data['TARGET_NAME'] = target_type
                    galaxy_data['CONTAM_NAME'] = 'CONTAM_GALAXY'

                    # ToDo: Modulate the contamination fraction...
                    galaxy_data['CONTAM_NUMBER'] = np.round( cparams['galaxies'] * ntarg ).astype(int)
                    galaxy_data['CONTAM_FACTOR'] = cparams['galaxies'] * ntarg / len(galaxy_data['RA'])

                    # Sample from the appropriate Gaussian mixture model and
                    # then generate the spectra.
                    if target_type == 'LRG':
                        mag = galaxy_data['ZMAG']
                    else:
                        mag = galaxy_data['MAG']
                    
                    gmmout = ContamGalaxiesMock.sample_GMM(nobj, target=target_type, morph=morph,
                                                           isouth=galaxy_data['SOUTH'],
                                                           seed=seed, prior_mag=mag)
                    galaxy_data.update(gmmout)

                    contamtargets, contamtruth, contamobjtruth, contamtrueflux = get_spectra(
                        galaxy_data, ContamGalaxiesMock, log, nside=nside, nside_chunk=nside_chunk,
                        seed=seed, nproc=nproc, no_spectra=no_spectra, contaminants=True)

                    if len(contamtargets) > 0:
                        galaxies_targets.append(contamtargets)
                        galaxies_truth.append(contamtruth)
                        # We use BGS spectral templates as contaminants.
                        if 'BGS' in objtruth.keys():
                            objtruth['BGS'] = vstack( (objtruth['BGS'], contamobjtruth) )
                        else:
                            objtruth['BGS'] = contamobjtruth
                        trueflux = np.vstack( (trueflux, contamtrueflux) )

        if len(galaxies_targets) > 0:
            galaxies_targets = vstack(galaxies_targets)
            galaxies_truth = vstack(galaxies_truth)

    # Now merge all the contaminants into the output targets catalog.
    if len(stars_targets) > 0:
        targets = vstack( (targets, stars_targets) )
        truth = vstack( (truth, stars_truth) )
        
    if len(galaxies_targets) > 0:
        targets = vstack( (targets, galaxies_targets) )
        truth = vstack( (truth, galaxies_truth) )

    return targets, truth, objtruth, trueflux

def targets_truth(params, healpixels=None, nside=None, output_dir='.',
                  seed=None, nproc=1, nside_chunk=128, verbose=False,
                  no_spectra=False):
    """Generate truth and targets catalogs, and noiseless spectra.

    Parameters
    ----------
    params : :class:`dict`
        Target parameters.
    healpixels : :class:`numpy.ndarray` or :class:`int`
        Restrict the sample of mock targets analyzed to those lying inside
        this set (array) of healpix pixels.
    nside : :class:`int`
        Healpix resolution corresponding to healpixels.
    output_dir : :class:`str`, optional
        Output directory.    Defaults to '.' (current directory).
    seed : :class:`int`
        Seed for the random number generation.  Defaults to None.
    nproc : :class:`int`, optional
        Number of parallel processes to use.  Defaults to 1 (i.e., no
        multiprocessing).
    nside_chunk : :class:`int`, optional
        Healpix resolution for chunking the sample to avoid memory problems.
        (NB: nside_chunk must be <= nside).  Defaults to 128.
    verbose : :class:`bool`, optional
        Be verbose. Defaults to False.
    no_spectra : :class:`bool`, optional
        Do not generate spectra, e.g., for use with quicksurvey.

    Returns
    -------
    Files 'targets.fits', 'truth.fits', 'sky.fits', 'standards-dark.fits', and
    'standards-bright.fits' written to output_dir for the given list of
    healpixels.

    """
    from desitarget.mock import mockmaker

    log, healpixseeds = initialize_targets_truth(
        params, verbose=verbose, seed=seed, nside=nside,
        output_dir=output_dir, healpixels=healpixels)

    # Read (and cache) the MockMaker classes we need.
    log.info('Initializing and caching all MockMaker classes.')
    AllMakeMock = []
    for target_name in sorted(params['targets'].keys()):
        target_type = params['targets'][target_name].get('target_type')
        calib_only = params['targets'][target_name].get('calib_only', False)
        use_simqso = params['targets'][target_name].get('use_simqso', True)
        balprob = params['targets'][target_name].get('balprob', 0.0)
        add_dla = params['targets'][target_name].get('add_dla', False)

        AllMakeMock.append(getattr(mockmaker, '{}Maker'.format(target_name))(
            seed=seed, nside_chunk=nside_chunk, calib_only=calib_only,
            use_simqso=use_simqso, balprob=balprob, add_dla=add_dla,
            no_spectra=no_spectra))

    # Are we adding contaminants?  If so, cache the relevant classes here.
    if 'contaminants' in params.keys():
        if 'stars' in params['contaminants']:
            log.info('Initializing and caching MockMaker class for stellar contaminants.')
            if len(params['contaminants']['stars'].keys()) > 1:
                log.fatal('Multiple stellar contamination classes are not supported!')
                raise ValueError
            star_name, _ = list(params['contaminants']['stars'].items())[0]
            ContamStarsMock = getattr(mockmaker, '{}Maker'.format(star_name))(
                seed=seed, nside_chunk=nside_chunk, no_spectra=no_spectra)
        else:
            ContamStarsMock = None
                
        if 'galaxies' in params['contaminants']:
            log.info('Initializing and caching MockMaker class for extragalactic contaminants.')
            if len(params['contaminants']['galaxies'].keys()) > 1:
                log.fatal('Multiple stellar contamination classes are not supported!')
                raise ValueError
            galaxies_name, _ = list(params['contaminants']['galaxies'].items())[0]
            ContamGalaxiesMock = getattr(mockmaker, '{}Maker'.format(galaxies_name))(
                seed=seed, nside_chunk=nside_chunk, no_spectra=no_spectra,
                target_name='CONTAM_GALAXY')
        else:
            ContamGalaxiesMock = None
            
    # Loop over each source / object type.
    for healpix, healseed in zip(healpixels, healpixseeds):
        log.info('Working on healpixel {}'.format(healpix))

        alltargets = list()
        alltruth = list()
        allobjtruth = dict()
        alltrueflux = list()
        allskytargets = list()
        allskytruth = list()

        for ii, target_name in enumerate(sorted(params['targets'].keys())):
            targets, truth, skytargets, skytruth = [], [], [], []

            # Read the data and ithere are no targets, keep going.
            log.info('Working on target class {} on healpixel {}'.format(target_name, healpix))
            data, MakeMock = read_mock(params['targets'][target_name], log, target_name,
                                       seed=healseed, healpixels=healpix,
                                       nside=nside, nside_chunk=nside_chunk,
                                       MakeMock=AllMakeMock[ii])
            
            if not bool(data):
                continue

            # Generate targets in parallel; SKY targets are special.
            target_type = params['targets'][target_name]['target_type'].upper()
            sky = target_type == 'SKY'
            calib_only = params['targets'][target_name].get('calib_only', False)
            targets, truth, objtruth, trueflux = get_spectra(data, MakeMock, log, nside=nside,
                                                             nside_chunk=nside_chunk, seed=healseed,
                                                             nproc=nproc, sky=sky, no_spectra=no_spectra,
                                                             calib_only=calib_only)
            del data
            
            if sky:
                allskytargets.append(targets)
                allskytruth.append(truth)
            else:
                if len(targets) > 0:
                    alltargets.append(targets)
                    alltruth.append(truth)
                    if target_type in allobjtruth.keys(): # e.g., QSO
                        allobjtruth[target_type] = vstack( (allobjtruth[target_type], objtruth) )
                    else:
                        allobjtruth[target_type] = objtruth
                    alltrueflux.append(trueflux)

        if len(alltargets) == 0 and len(allskytargets) == 0: # all done
            continue

        # Pack it all together and then add some final columns.
        if len(alltargets) > 0:
            targets = vstack(alltargets) 
            truth = vstack(alltruth)
            objtruth = allobjtruth
            trueflux = np.concatenate(alltrueflux)
        else:
            targets = []

        if len(allskytargets) > 0:
            skytargets = vstack(allskytargets)
            skytruth = vstack(allskytruth)
        else:
            skytargets = []

        # Now add contaminants.  Should probably push this to its own function.
        if len(targets) > 0 and 'contaminants' in params.keys():
            log.info('Working on contaminants.')
            targets, truth, objtruth, trueflux = get_contaminants_onepixel(
                params, healpix, nside, healseed, nproc, log, nside_chunk,
                targets, truth, objtruth, trueflux, no_spectra=no_spectra,
                ContamStarsMock=ContamStarsMock,
                ContamGalaxiesMock=ContamGalaxiesMock)

        # Finish up.
        targets, truth, objtruth, skytargets, skytruth = finish_catalog(
            targets, truth, objtruth, skytargets, skytruth, healpix,
            nside, log, seed=healseed)

        # Finally, write the results to disk.
        write_targets_truth(targets, truth, objtruth, trueflux, MakeMock.wave,
                            skytargets, skytruth,  healpix, nside, log, output_dir, 
                            seed=healseed)
        
def finish_catalog(targets, truth, objtruth, skytargets, skytruth, healpix,
                   nside, log, seed=None, survey='main'):
    """Add hpxpixel, brick_objid, targetid, subpriority, priority, and numobs to the
    target catalog.
    
    Parameters
    ----------
    targets : :class:`astropy.table.Table`
        Final set of targets. 
    truth : :class:`astropy.table.Table`
        Corresponding truth table for targets.
    objtruth : :class:`astropy.table.Table`
        Corresponding objtype-specific truth table (if applicable).
    skytargets : :class:`astropy.table.Table`
        Sky targets.
    skytruth : :class:`astropy.table.Table`
        Corresponding truth table for sky targets.
    healpix : : :class:`int`
        Healpixel number.
    nside : :class:`int`
        Nside corresponding to healpix.
    log : :class:`desiutil.logger`
       Logger object.
    seed : :class:`int`, optional
        Seed for the random number generation.  Defaults to None.
    survey : :class:`str`
        Specifies which target masks yaml file to use. Options are `main`, `cmx`
        and `sv` for the main survey, commissioning and SV, respectively.
        Defaults to `main`.
            
    Returns
    -------
    Updated versions of targets, truth, objtruth, skytargets, and skytruth.

    """
    from desitarget.targets import encode_targetid, initial_priority_numobs

    rand = np.random.RandomState(seed)
    
    nobj = len(targets)
    nsky = len(skytargets)
    area = hp.nside2pixarea(nside, degrees=True)
    log.info('Summary: ntargets = {} ({:.2f} targets/deg2), nsky = {} ({:.2f} targets/deg2) in pixel {}.'.format(
        nobj, nobj / area, nsky, nsky / area, healpix))

    # Assign TARGETID using the healpixel number, not BRICKID, otherwise we'll
    # end up with duplicate TARGETID values.
    objid = np.arange(nobj + nsky)
    targetid = encode_targetid(objid=objid, brickid=healpix, mock=1)

    #objid = np.zeros(nobj+nsky).astype('i4')
    #for brickid in set(targets['BRICKID']):
    #    indx = brickid == targets['BRICKID']
    #    objid[indx] = np.arange(np.sum(indx))
    #targetid = encode_targetid(objid=objid, brickid=targets['BRICKID'], mock=1)

    subpriority = rand.uniform(0.0, 1.0, size=nobj + nsky)

    if nobj > 0:
        #targets['BRICKID'][:] = healpix # use the derived BRICKID values
        targets['HPXPIXEL'][:] = healpix
        targets['BRICK_OBJID'][:] = objid[:nobj]
        targets['TARGETID'][:] = targetid[:nobj]
        targets['SUBPRIORITY'][:] = subpriority[:nobj]
        truth['TARGETID'][:] = targetid[:nobj]

        # Assign the appropriate TARGETID values to the objtruth tables.
        for obj in set(truth['TEMPLATETYPE']):
            these = obj == truth['TEMPLATETYPE']
            objtruth[obj]['TARGETID'][:] = truth['TARGETID'][these]

        # Check.
        for obj in set(truth['TEMPLATETYPE']):
            these = obj == truth['TEMPLATETYPE']
            if not np.all( (objtruth[obj]['TARGETID'] == truth['TARGETID'][these]) ) or \
              not np.all( (objtruth[obj]['TARGETID'] == targets['TARGETID'][these]) ):
                log.warning('Mismatching TARGETIDs!')
                raise ValueError                
                    
        targets['PRIORITY_INIT'], targets['NUMOBS_INIT'] = initial_priority_numobs(
            targets, survey=survey)

        # Rename TYPE --> MORPHTYPE
        targets.rename_column('TYPE', 'MORPHTYPE')

        assert(len(targets['TARGETID'])==len(np.unique(targets['TARGETID'])))

    if nsky > 0:
        skytargets['HPXPIXEL'][:] = healpix
        skytargets['BRICK_OBJID'][:] = objid[nobj:]
        skytargets['TARGETID'][:] = targetid[nobj:]
        skytargets['SUBPRIORITY'][:] = subpriority[nobj:]
        skytruth['TARGETID'][:] = targetid[nobj:]

        skytargets['PRIORITY_INIT'], skytargets['NUMOBS_INIT'] = initial_priority_numobs(
            skytargets, survey=survey)

        # Rename TYPE --> MORPHTYPE
        skytargets.rename_column('TYPE', 'MORPHTYPE')

    return targets, truth, objtruth, skytargets, skytruth

def write_targets_truth(targets, truth, objtruth, trueflux, truewave, skytargets,
                        skytruth, healpix, nside, log, output_dir, seed=None):
    """Writes all the final catalogs to disk.
    
    Parameters
    ----------
    targets : :class:`astropy.table.Table`
        Final target catalog.
    truth : :class:`astropy.table.Table`
        Corresponding truth table for targets.
    objtruth : :class:`astropy.table.Table`
        Corresponding objtype-specific truth table (if applicable).
    trueflux : :class:`numpy.ndarray`
        Array [npixel, ntarget] of observed-frame spectra.  
    truewave : :class:`numpy.ndarray`
        Wavelength array corresponding to trueflux.
    skytargets : :class:`astropy.table.Table`
        Sky targets.
    skytruth : :class:`astropy.table.Table`
        Corresponding truth table for sky targets.
    healpix : : :class:`int`
        Healpixel number.
    nside : :class:`int`
        Nside corresponding to healpix.
    log : :class:`desiutil.logger`
       Logger object.
    seed : :class:`int`, optional
        Seed for the random number generation.  Defaults to None.
    output_dir : :class:`str`
        Output directory.
            
    Returns
    -------
    Files targets-{nside}-{healpix}.fits, truth-{nside}-{healpix}.fits,
    sky-{nside}-{healpix}.fits, standards-bright-{nside}-{healpix}.fits, and
    standards-dark-{nside}-{healpix}.fits are all written to disk.

    Raises
    ------
    ValueError
        If there are duplicate targetid ids.
    """
    from astropy.io import fits
    from desiutil import depend
    from desispec.io.util import fitsheader, write_bintable
    import desitarget.mock.io as mockio
    from ..targetmask import desi_mask
    
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

    outdir = mockio.get_healpix_dir(nside, healpix, basedir=output_dir)
    os.makedirs(outdir, exist_ok=True)

    # Write out the sky catalog.
    skyfile = mockio.findfile('sky', nside, healpix, basedir=output_dir)
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
        for stdsuffix, stdbit in zip(('dark', 'bright'), ('STD_FAINT', 'STD_BRIGHT')):
            stdfile = mockio.findfile('standards-{}'.format(stdsuffix), nside, healpix, basedir=output_dir)
            istd = ( (targets['DESI_TARGET'] & desi_mask.mask(stdbit)) |
                     (targets['DESI_TARGET'] & desi_mask.mask('STD_WD')) ) != 0

            if np.count_nonzero(istd) > 0:
                log.info('Writing {} {} standards to {}'.format(np.sum(istd), stdsuffix.upper(), stdfile))
                write_bintable(stdfile+'.tmp', targets[istd], extname='STD',
                               header=targetshdr, clobber=True)
                os.rename(stdfile+'.tmp', stdfile)
            else:
                log.info('No {} standards stars selected.'.format(stdsuffix))
                log.info('  Standard star file {} not written.'.format(stdfile))

        # Finally write out the rest of the targets.
        targetsfile = mockio.findfile('targets', nside, healpix, basedir=output_dir)
        truthfile = mockio.findfile('truth', nside, healpix, basedir=output_dir)
   
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

        if len(trueflux) > 0:
            hdu = fits.ImageHDU(truewave.astype(np.float32),
                                name='WAVE', header=truthhdr)
            hdu.header['BUNIT'] = 'Angstrom'
            hdu.header['AIRORVAC'] = 'vac'
            hx.append(hdu)

            hdu = fits.ImageHDU(trueflux.astype(np.float32), name='FLUX')
            hdu.header['BUNIT'] = '1e-17 erg/s/cm2/Angstrom'
            hx.append(hdu)

        if len(objtruth) > 0:
            for obj in sorted(set(truth['TEMPLATETYPE'])):
                hdu = fits.convenience.table_to_hdu(objtruth[obj])
                hdu.header['EXTNAME'] = 'TRUTH_{}'.format(obj)
                hx.append(hdu)

        try:
            hx.writeto(truthfile+'.tmp', overwrite=True)
        except:
            hx.writeto(truthfile+'.tmp', clobber=True)
        os.rename(truthfile+'.tmp', truthfile)

def _merge_file_tables(fileglob, ext, outfile=None, comm=None, addcols=None, overwrite=False):
    '''
    parallel merge tables from individual files into an output file

    Args:
        fileglob (str): glob of files to combine (e.g. '*/blat-*.fits')
        ext (str or int): FITS file extension name or number

    Options:
        outfile (str): output file to write
        comm: MPI communicator object
        addcols: dict extra columns to add with fill values, e.g. dict(OBSCONDITIONS=1)

    Returns merged table as np.ndarray
    '''
    import fitsio
    import glob
    from desiutil.log import get_logger
    log = get_logger()
    
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
    data = list()
    for filename in infiles[rank::size]:
        try:
            data.append(fitsio.read(filename, ext))
        except OSError:  #- yep, OSError not IOError
            log.info('Extension {} not found in {}'.format(ext, filename))
            pass

    if len(data) > 0:
        data = np.hstack(data)
    else:
        # some ranks may not touch files that have this ext
        data = None

    if comm is not None:
        data = comm.gather(data, root=0)
        if rank == 0 and size>1:
            data = [d for d in data if d is not None]
            data = np.hstack(data)

    if rank == 0 and outfile is not None:
        if (data is None) or len(data) == 0:
            message = '{} not found in any input files; skipping'.format(ext)
            log.warning(message)
            return None

        log.info('Writing {} {}'.format(outfile, ext))
        header = fitsio.read_header(infiles[0], ext)

        #- Use tmpout name so interupted I/O doesn't leave a corrupted file
        #- of the correct name
        tmpout = outfile + '.tmp'

        #- If appending, move file back to tmpout name
        if (not overwrite) and os.path.exists(outfile):
            os.rename(outfile, tmpout)
        
        # Find duplicates
        vals, idx_start, count = np.unique(data['TARGETID'], return_index=True, return_counts=True)
        if len(vals) != len(data):
            log.warning('Non-unique TARGETIDs found!')
            raise ValueError

        if addcols is not None:
            numrows = len(data)
            colnames = list()
            coldata = list()
            for colname, value in addcols.items():
                colnames.append(colname)
                coldata.append(np.full(numrows, value))

            data = np.lib.recfunctions.append_fields(data, colnames, coldata,
                                                     usemask=False)

        fitsio.write(tmpout, data, header=header, extname=ext, clobber=overwrite)
        os.rename(tmpout, outfile)

    return data

def join_targets_truth(mockdir, outdir=None, overwrite=False, comm=None):
    '''
    Join individual healpixel targets and truth files into combined tables

    Args:
        mockdir: top level mock target directory

    Options:
        outdir: output directory, default to mockdir
        overwrite: rewrite outputs even if they already exist
        comm: MPI communicator; if not None, read data in parallel
    '''
    import fitsio
    from desitarget.targetmask import obsconditions as obsmask
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
        todo['sky'] = not os.path.exists(outdir+'/sky.fits') or overwrite
        todo['stddark'] = not os.path.exists(outdir+'/standards-dark.fits') or overwrite
        todo['stdbright'] = not os.path.exists(outdir+'/standards-bright.fits') or overwrite
        todo['targets'] = not os.path.exists(outdir+'/targets.fits') or overwrite
        todo['truth'] = not os.path.exists(outdir+'/truth.fits') or overwrite
        todo['mtl'] = not os.path.exists(outdir+'/mtl.fits') or overwrite
    else:
        todo = None

    if comm is not None:
        todo = comm.bcast(todo, root=0)

    if todo['sky']:
        _merge_file_tables(mockdir+'/*/*/sky-*.fits', 'SKY',
                           outfile=outdir+'/sky.fits', comm=comm,
                           overwrite=overwrite,
                           addcols=dict(OBSCONDITIONS=obsmask.mask('DARK|GRAY|BRIGHT')))

    if todo['stddark']:
        _merge_file_tables(mockdir+'/*/*/standards-dark*.fits', 'STD',
                           outfile=outdir+'/standards-dark.fits', comm=comm,
                           overwrite=overwrite,
                           addcols=dict(OBSCONDITIONS=obsmask.mask('DARK|GRAY')))

    if todo['stdbright']:
        _merge_file_tables(mockdir+'/*/*/standards-bright*.fits', 'STD',
                           outfile=outdir+'/standards-bright.fits', comm=comm,
                           overwrite=overwrite,
                           addcols=dict(OBSCONDITIONS=obsmask.mask('BRIGHT')))

    if todo['targets']:
        _merge_file_tables(mockdir+'/*/*/targets-*.fits', 'TARGETS',
                           overwrite=overwrite,
                           outfile=outdir+'/targets.fits', comm=comm)

    if todo['truth']:
        _merge_file_tables(mockdir+'/*/*/truth-*.fits', 'TRUTH',
                           overwrite=overwrite,
                           outfile=outdir+'/truth.fits', comm=comm)
        # append, not overwrite other per-subclass truth tables
        for templatetype in ['BGS', 'ELG', 'LRG', 'QSO', 'STAR', 'WD']:
            extname = 'TRUTH_' + templatetype
            _merge_file_tables(mockdir+'/*/*/truth-*.fits', extname,
                               overwrite=False,
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



