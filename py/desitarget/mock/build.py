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

def _initialize(params, verbose=False, seed=1, output_dir='./', 
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
    from desimodel.footprint import tiles2pix

    # Initialize logger
    if verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()
    
    if params is None:
        log.fatal('PARAMS input is required.')
        raise ValueError

    if nside is None:
        log.fatal('NSIDE input is required.')
        raise ValueError

    if nside > 256:
        log.warning('NSIDE = {} exceeds the number of bits available for BRICKID in targets.encode_targetid.')
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
        
    # Default set of healpixels is the whole DESI footprint (yikes!)
    if healpixels is None:
        log.warning('List of healpixels not provided; processing the whole DESI footprint!')
        healpixels = tiles2pix(nside)

    areaperpix = hp.nside2pixarea(nside, degrees=True)
    totarea = len(healpixels) * areaperpix
    log.info('Processing {} healpixel(s) (nside = {}, {:.3f} deg2/pixel) spanning {:.3f} deg2.'.format(
        len(healpixels), nside, areaperpix, totarea))

    return log, rand, healpixels, areaperpix
    
def _density_fluctuations(params, data, log, nside=16, nside_chunk=128, rand=None):
    """Density fluctuations model."""

    if rand is None:
        rand = np.random.RandomState()

    # Fluctuations model coefficients from --
    #   https://github.com/desihub/desitarget/blob/master/doc/nb/target-fluctuations.ipynb
    model = dict()
    model['LRG'] = (0.27216, 2.631, 0.145) # slope, intercept, and scatter
    model['ELG'] = (-0.55792, 3.380, 0.081)
    model['QSO'] = (0.33321, 3.249, 0.112)

    coeff = model.get(params['target_name'])
    
    # Chunk each healpixel into a smaller set of healpixels, for
    # parallelization.
    if nside >= nside_chunk:
        nside_chunk = nside
        
    areaperpixel = hp.nside2pixarea(nside, degrees=True)
    areaperchunk = hp.nside2pixarea(nside_chunk, degrees=True)

    nchunk = 4**np.int(np.log2(nside_chunk) - np.log2(nside))
    log.info('Dividing each nside={} healpixel ({:.2f} deg2) into {} nside={} healpixel(s) ({:.2f} deg2).'.format(
        nside, areaperpixel, nchunk, nside_chunk, areaperchunk))

    # Get the requested target density, if any.
    ntarget = len(data['RA'])
    if 'density' in params.keys():
        density = params['density']
    else:
        density = ntarget / areaperpixel

    # Assign sources to healpix chunks.
    healpix_chunk = radec2pix(nside_chunk, data['RA'], data['DEC'])

    indxperchunk, ntargetperchunk = list(), list()
    for pixchunk in set(healpix_chunk):
        indx = np.where( np.in1d(healpix_chunk, pixchunk)*1 )[0]
        indxperchunk.append(indx)

        if coeff:
            # Number of targets in this chunk, based on the fluctuations model.
            denschunk = density * 10**( np.polyval(coeff[:2], data['EBV'][indx]) - np.polyval(coeff[:2], 0) +
                                        rand.normal(scale=coeff[2]) )            # [ntarget/deg2]
            ntarg = np.rint( np.median(denschunk) * areaperchunk ).astype('int') # [ntarget]
            ntargetperchunk.append(ntarg)
        else:
            # Divide the targets evenly among chunks.
            ntargetperchunk = np.repeat(np.round(ntarget / nchunk).astype('int'), nchunk)

    return indxperchunk, np.array(ntargetperchunk)

def read_mock(params, log, dust_dir=None, seed=None, healpixels=None,
              nside=16, nside_chunk=128, in_desi=True):
    """Read one specified mock catalog.
    
    Args:
        params: dict
            Dictionary summary of the input configuration file, restricted to a
            particular source_name (e.g., 'QSO').
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

    target_name = params.get('target_name')
    mockfile = params.get('mockfile')
    mockformat = params.get('format')
    magcut = params.get('magcut')
    nside_lya = params.get('nside_lya')
    nside_galaxia = params.get('nside_galaxia')

    log.info('Target: {}, format: {}, mockfile: {}'.format(target_name, mockformat, mockfile))

    MakeMock = getattr(mockmaker, '{}Maker'.format(target_name))(seed=seed)
    source_data = MakeMock.read(mockfile=mockfile, mockformat=mockformat,
                                healpixels=healpixels, nside=nside,
                                nside_chunk=nside_chunk, magcut=magcut,
                                nside_lya=nside_lya, nside_galaxia=nside_galaxia,
                                dust_dir=dust_dir)

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
    if len(indx) < ntarget:
        log.warning('Too few candidate targets ({}) than desired ({}).'.format(
            len(indx), ntarget))        

    # Build spectra in chunks and stop when we have enough.
    nchunk = np.ceil(len(indx) / ntarget).astype('int')
    
    targets = list()
    truth = list()
    trueflux = list()

    boss_std = None
    
    ntot = 0
    for ii, chunkindx in enumerate(np.array_split(indx, nchunk)):

        # Temporary hack to use BOSS standard stars.
        if 'BOSS_STD' in source_data.keys():
            boss_std = source_data['BOSS_STD'][chunkindx]

        # Faintstar targets are a special case.
        if source_data['TARGET_NAME'].lower() == 'faintstar':
            chunkflux, _, chunkmeta, chunktargets, chunktruth = MakeMock.make_spectra(
                source_data, indx=chunkindx, boss_std=boss_std)

        else:
            # Generate the spectra.
            chunkflux, _, chunkmeta, chunktargets, chunktruth = MakeMock.make_spectra(
                source_data, indx=chunkindx)

            # Select targets.
            MakeMock.select_targets(chunktargets, chunktruth, boss_std=boss_std)

        keep = np.where(chunktargets['DESI_TARGET'] != 0)[0]
        nkeep = len(keep)

        log.debug('Selected {} / {} targets on chunk {} / {}'.format(
            nkeep, len(chunkindx), ii+1, nchunk))

        if nkeep > 0:
            targets.append(chunktargets[keep])
            truth.append(chunktruth[keep])
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

def targets_truth(params, output_dir='.', seed=None, nproc=1, nside=None,
                  healpixels=None, nside_chunk=128, verbose=False):
    """Generate a catalog of targets, spectra, and the corresponding "truth" catalog
    (with, e.g., the true redshift) for use in simulations.

    Args:
        params : dict
            Source parameters.
        output_dir : str
            Output directory (default '.').
        seed: int
            Seed for the random number generation.
        nproc : int
            Number of parallel processes to use (default 1).
        nside : int
            Healpix resolution corresponding to healpixels (default 16).
        healpixels : numpy.ndarray or int
            Restrict the sample of mock targets analyzed to those lying inside
            this set (array) of healpix pixels.  (Default: None)
        nside_chunk : int
            Healpix resolution for chunking the sample (NB: nside_chunk must be
            <= nside).
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
    log, rand, healpixels, areaperpix = _initialize(params, verbose=verbose,
                                                    seed=seed, output_dir=output_dir,
                                                    nside=nside, healpixels=healpixels)
    
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
            source_data, MakeMock = read_mock(params['sources'][source_name],
                                              log, dust_dir=params['dust_dir'],
                                              seed=seed, healpixels=healpix,
                                              nside=nside, nside_chunk=nside_chunk)

            # If there are no sources, keep going.
            if not bool(source_data):
                continue

            # Parallelize by chunking the sample into smaller healpixels and
            # determine the number of targets per chunk.
            indxperchunk, ntargetperchunk = _density_fluctuations(
                params['sources'][source_name], source_data, log, nside=nside,
                nside_chunk=nside_chunk, rand=rand)

            nchunk = len(indxperchunk)
            ntarget = np.sum(ntargetperchunk)
            density = ntarget / areaperpix
            
            log.info('Goal: generate spectra for {} {} targets ({:.2f} / deg2).'.format(
                ntarget, source_name, density))

            specargs = list()
            for indx, ntarg in zip( indxperchunk, ntargetperchunk ):
                if len(indx) > 0:
                    specargs.append( (source_data, indx, MakeMock, rand, log, ntarg) )

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
    from desitarget.targets import encode_targetid
    
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
    import desitarget.mock.io as mockio
    from desitarget import desi_mask
    
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

def _merge_file_tables(fileglob, ext, outfile=None, comm=None):
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
        _merge_file_tables(mockdir+'/*/*/sky-*.fits', 'SKY',
                           outfile=outdir+'/sky.fits', comm=comm)

    if todo['stddark']:
        _merge_file_tables(mockdir+'/*/*/standards-dark*.fits', 'STD',
                           outfile=outdir+'/standards-dark.fits', comm=comm)

    if todo['stdbright']:
        _merge_file_tables(mockdir+'/*/*/standards-bright*.fits', 'STD',
                           outfile=outdir+'/standards-bright.fits', comm=comm)

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



