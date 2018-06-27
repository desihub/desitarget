# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.build
=====================

Build truth and targets catalogs, including spectra, for the mocks.

"""
from __future__ import absolute_import, division, print_function

import os
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

    # Parse DUST_DIR.
    if 'dust_dir' in params.keys():
        dust_dir = params['dust_dir']    
        try:
            dust_dir = dust_dir.format(**os.environ)
        except KeyError as e:
            log.warning('Environment variable not set for DUST_DIR: {}'.format(e))
            raise ValueError
    else:
        log.warning('DUST_DIR parameter not found in configuration file.')
        raise ValueError

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

    return log, healpixseeds, dust_dir
    
def read_mock(params, log, dust_dir=None, seed=None, healpixels=None,
              nside=None, nside_chunk=128, MakeMock=None):
    """Read a mock catalog.
    
    Parameters
    ----------
    params : :class:`dict`
        Dictionary defining the mock from which to generate targets.
    log : :class:`desiutil.logger`
        Logger object.
    params : :class:`dict`
        Dictionary summary of the input configuration file, restricted to a
        particular source_name (e.g., 'QSO').
    dust_dir : :class:`str`
        Full path to the dust maps.
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
        Parsed source data based on the input mock catalog (to be documented).

    Raises
    ------
    ValueError
        If the mock_density was not returned when expected.

    """
    from desitarget.mock import mockmaker

    target_name = params.get('target_name')
    mockfile = params.get('mockfile')
    mockformat = params.get('format')
    magcut = params.get('magcut')
    nside_lya = params.get('nside_lya')
    nside_galaxia = params.get('nside_galaxia')
    calib_only = params.get('calib_only', False)

    if 'density' in params.keys():
        mock_density = True
    else:
        mock_density = False

    log.info('Target: {}, format: {}, mockfile: {}'.format(target_name, mockformat, mockfile))

    if MakeMock is None:
        MakeMock = getattr(mockmaker, '{}Maker'.format(target_name))(seed=seed, nside_chunk=nside_chunk,
                                                                     calib_only=calib_only)
    else:
        MakeMock.seed = seed # updated seed
        
    data = MakeMock.read(mockfile=mockfile, mockformat=mockformat,
                         healpixels=healpixels, nside=nside,
                         magcut=magcut, nside_lya=nside_lya,
                         nside_galaxia=nside_galaxia,
                         dust_dir=dust_dir, mock_density=mock_density)
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
    trueflux : :class:`numpy.ndarray`
        Array [npixel, ntarget] of observed-frame spectra.  Only computed
        and returned for non-sky targets and if no_spectra=False.

    """
    targname = data['TARGET_NAME']

    rand = np.random.RandomState(seed)

    targets = list()
    truth = list()
    trueflux = list()

    boss_std = None

    # Temporary hack to use BOSS standard stars.
    if 'BOSS_STD' in data.keys():
        boss_std = data['BOSS_STD'][indx]

        if calib_only:
            calib = np.where(boss_std)[0]
            ntarget = len(calib)
            if ntarget == 0:
                log.debug('No (flux) calibration star(s) on this healpixel.')
                return [targets, truth, trueflux]
            else:
                log.debug('Generating spectra for {} candidate (flux) calibration stars.'.format(ntarget))
                indx = indx[calib]
                boss_std = boss_std[calib]

    # Faintstar targets are a special case.
    if targname.lower() == 'faintstar':
        chunkflux, _, chunkmeta, chunktargets, chunktruth = MakeMock.make_spectra(
            data, indx=indx, boss_std=boss_std)
        
        if len(chunktargets) > 0:
            keep = np.where(chunktargets['DESI_TARGET'] != 0)[0]
            nkeep = len(keep)
        else:
            nkeep = 0

        log.debug('Selected {} / {} {} targets'.format(nkeep, ntarget, targname))

        if nkeep > 0:
            targets.append(chunktargets[keep])
            truth.append(chunktruth[keep])
            if not no_spectra:
                trueflux.append(chunkflux[keep, :])
    else:
        # Generate the spectra iteratively until we achieve the required
        # target density.
        iterseeds = rand.randint(2**31, size=maxiter)

        makemore, itercount, ntot = True, 0, 0
        while makemore:
            chunkflux, _, chunkmeta, chunktargets, chunktruth = MakeMock.make_spectra(
                data, indx=indx, seed=iterseeds[itercount], no_spectra=no_spectra)

            MakeMock.select_targets(chunktargets, chunktruth, boss_std=boss_std)

            keep = np.where(chunktargets['DESI_TARGET'] != 0)[0]
            nkeep = len(keep)
            if nkeep > 0:
                log.debug('Generated {} / {} {} targets on iteration {} / {}.'.format(
                    nkeep, ntarget, targname, itercount, maxiter))
                ntot += nkeep

                targets.append(chunktargets[keep])
                truth.append(chunktruth[keep])
                if not no_spectra:
                    trueflux.append(chunkflux[keep, :])

            itercount += 1
            if itercount == maxiter:
                if maxiter > 1:
                    log.warning('Generated {} / {} {} targets after {} iterations.'.format(
                        ntot, ntarget, targname, maxiter))
                makemore = False
            else:
                need = np.where(chunktargets['DESI_TARGET'] == 0)[0]
                if len(need) > 0:
                    indx = indx[need]
                else:
                    makemore = False

    if len(targets) > 0:
        targets = vstack(targets)
        truth = vstack(truth)
        if not no_spectra:
            trueflux = np.concatenate(trueflux)
        
    return [targets, truth, trueflux]

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

    # Assign sources to healpix chunks.
    #ntarget = len(data['RA'])
    healpix_chunk = radec2pix(nside_chunk, data['RA'], data['DEC'])

    density_factor = data.get('DENSITY_FACTOR')

    indxperchunk, ntargperchunk = list(), list()
    for pixchunk in set(healpix_chunk):

        # Subsample the targets on this mini healpixel.
        allindxthispix = np.where( np.in1d(healpix_chunk, pixchunk)*1 )[0]

        ntargthispix = np.ceil( len(allindxthispix) * density_factor ).astype('int')
        indxthispix = rand.choice(allindxthispix, size=ntargthispix, replace=False)

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

    return indxperchunk, ntargperchunk, areaperpixel

def get_spectra(data, MakeMock, log, nside, nside_chunk, seed=None,
                nproc=1, sky=False, no_spectra=False, calib_only=False):
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

    Returns
    -------
    targets : :class:`astropy.table.Table`
        Target catalog.
    truth : :class:`astropy.table.Table`
        Corresponding truth table.
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

    #targets = [targ for targ in results[0] if len(targ) > 0]
    #truth = [tru for tru in results[1] if len(tru) > 0]
    
    targets, truth, good = [], [], []
    for ii, (targ, tru) in enumerate( zip(results[0], results[1]) ):
        if len(targ) != len(tru):
            log.warning('Mismatching argets and truth tables!')
            raise ValueError
        if len(targ) > 0:
            good.append(ii)
            targets.append(targ)
            truth.append(tru)
               
    if len(targets) > 0:
        targets = vstack(targets)
        truth = vstack(truth)
        good = np.array(good)

    if sky:
        trueflux = []
    else:
        if no_spectra:
            trueflux = []
        else:
            if len(good) > 0:
                trueflux = np.concatenate(np.array(results[2])[good])
            else:
                trueflux = []
                
    log.info('Done: Generated spectra for {} {} targets ({:.2f} / deg2).'.format(
        len(targets), data['TARGET_NAME'], len(targets) / area))

    log.info('Total time for {}s = {:.3f} minutes ({:.3f} cpu minutes/deg2).'.format(
        data['TARGET_NAME'], ttime / 60, (ttime*nproc) / area ))

    return targets, truth, trueflux

def targets_truth(params, healpixels=None, nside=None, output_dir='.',
                  seed=None, nproc=1, nside_chunk=128, verbose=False,
                  no_spectra=False):
    """Generate truth and targets catalogs, and noiseless spectra.

    Parameters
    ----------
    params : :class:`dict`
        Source parameters.
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

    log, healpixseeds, dust_dir = initialize_targets_truth(
        params, verbose=verbose, seed=seed, nside=nside,
        output_dir=output_dir, healpixels=healpixels)

    # Read (and cache) the MockMaker classes we need.
    AllMakeMock = []
    for source_name in sorted(params['sources'].keys()):
        target_name = params['sources'][source_name].get('target_name')
        calib_only = params['sources'][source_name].get('calib_only', False)
        AllMakeMock.append(getattr(mockmaker, '{}Maker'.format(target_name))(
            seed=seed, nside_chunk=nside_chunk, calib_only=calib_only))

    # Loop over each source / object type.
    for healpix, healseed in zip(healpixels, healpixseeds):
        log.info('Working on healpixel {}'.format(healpix))

        alltargets = list()
        alltruth = list()
        alltrueflux = list()
        allskytargets = list()
        allskytruth = list()

        for ii, source_name in enumerate(sorted(params['sources'].keys())):
            targets, truth, skytargets, skytruth = [], [], [], []

            # Read the data and ithere are no targets, keep going.
            log.info('Working on target class: {}'.format(source_name))
            data, MakeMock = read_mock(params['sources'][source_name],
                                       log, dust_dir=dust_dir,
                                       seed=healseed, healpixels=healpix,
                                       nside=nside, nside_chunk=nside_chunk,
                                       MakeMock=AllMakeMock[ii])
            
            if not bool(data):
                continue

            # Generate targets in parallel; SKY targets are special. 
            sky = source_name.upper() == 'SKY'
            calib_only = params['sources'][source_name].get('calib_only', False)
            targets, truth, trueflux = get_spectra(data, MakeMock, log, nside=nside,
                                                   nside_chunk=nside_chunk, seed=healseed,
                                                   nproc=nproc, sky=sky, no_spectra=no_spectra,
                                                   calib_only=calib_only)
            
            if sky:
                allskytargets.append(targets)
                allskytruth.append(truth)
            else:
                if len(targets) > 0:
                    alltargets.append(targets)
                    alltruth.append(truth)
                    alltrueflux.append(trueflux)

            # Contaminants here?

        if len(alltargets) == 0 and len(allskytargets) == 0: # all done
            continue

        # Pack it all together and then add some final columns.
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

        targets, truth, skytargets, skytruth = finish_catalog(
            targets, truth, skytargets, skytruth, healpix,
            nside, log, seed=healseed)

        # Finally, write the results to disk.
        write_targets_truth(targets, truth, trueflux, MakeMock.wave, skytargets,
                            skytruth,  healpix, nside, log, output_dir, 
                            seed=healseed)
        
def finish_catalog(targets, truth, skytargets, skytruth, healpix,
                   nside, log, seed=None):
    """Add brickid, brick_objid, targetid, and subpriority to target catalog.
    
    Parameters
    ----------
    targets : :class:`astropy.table.Table`
        Final set of targets. 
    truth : :class:`astropy.table.Table`
        Corresponding truth table for targets.
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
            
    Returns
    -------
    Updated versions of targets, truth, skytargets, and skytruth.

    """
    from desitarget.targets import encode_targetid

    rand = np.random.RandomState(seed)
    
    nobj = len(targets)
    nsky = len(skytargets)
    log.info('Summary: ntargets = {}, nsky = {} in pixel {}.'.format(nobj, nsky, healpix))

    objid = np.arange(nobj + nsky)
    targetid = encode_targetid(objid=objid, brickid=healpix, mock=1)
    subpriority = rand.uniform(0.0, 1.0, size=nobj + nsky)

    if nobj > 0:
        targets['BRICKID'][:] = healpix
        targets['HPXPIXEL'][:] = healpix
        targets['BRICK_OBJID'][:] = objid[:nobj]
        targets['TARGETID'][:] = targetid[:nobj]
        targets['SUBPRIORITY'][:] = subpriority[:nobj]
        truth['TARGETID'][:] = targetid[:nobj]

    if nsky > 0:
        skytargets['BRICKID'][:] = healpix
        skytargets['HPXPIXEL'][:] = healpix
        skytargets['BRICK_OBJID'][:] = objid[nobj:]
        skytargets['TARGETID'][:] = targetid[nobj:]
        skytargets['SUBPRIORITY'][:] = subpriority[nobj:]
        skytruth['TARGETID'][:] = targetid[nobj:]
        
    return targets, truth, skytargets, skytruth

def write_targets_truth(targets, truth, trueflux, truewave, skytargets,
                        skytruth, healpix, nside, log, output_dir,
                        seed=None):
    """Writes all the final catalogs to disk.
    
    Parameters
    ----------
    targets : :class:`astropy.table.Table`
        Final target catalog.
    truth : :class:`astropy.table.Table`
        Corresponding truth table for targets.
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
        for stdsuffix, stdbit in zip(('dark', 'bright'), ('STD_FSTAR', 'STD_BRIGHT')):
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

        try:
            hx.writeto(truthfile+'.tmp', overwrite=True)
        except:
            hx.writeto(truthfile+'.tmp', clobber=True)
        os.rename(truthfile+'.tmp', truthfile)

def _merge_file_tables(fileglob, ext, outfile=None, comm=None, addcols=None):
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

        if addcols is not None:
            numrows = len(data)
            colnames = list()
            coldata = list()
            for colname, value in addcols.items():
                colnames.append(colname)
                coldata.append(np.full(numrows, value))

            data = np.lib.recfunctions.append_fields(data, colnames, coldata,
                                                     usemask=False)

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
        _merge_file_tables(mockdir+'/*/*/sky-*.fits', 'SKY',
                           outfile=outdir+'/sky.fits', comm=comm,
                           addcols=dict(OBSCONDITIONS=obsmask.mask('DARK|GRAY|BRIGHT')))

    if todo['stddark']:
        _merge_file_tables(mockdir+'/*/*/standards-dark*.fits', 'STD',
                           outfile=outdir+'/standards-dark.fits', comm=comm,
                           addcols=dict(OBSCONDITIONS=obsmask.mask('DARK|GRAY')))

    if todo['stdbright']:
        _merge_file_tables(mockdir+'/*/*/standards-bright*.fits', 'STD',
                           outfile=outdir+'/standards-bright.fits', comm=comm,
                           addcols=dict(OBSCONDITIONS=obsmask.mask('BRIGHT')))

    if todo['targets']:
        _merge_file_tables(mockdir+'/*/*/targets-*.fits', 'TARGETS',
                           outfile=outdir+'/targets.fits', comm=comm)

    if todo['truth']:
        _merge_file_tables(mockdir+'/*/*/truth-*.fits', 'TRUTH',
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



