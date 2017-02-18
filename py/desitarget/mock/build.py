# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=====================
desitarget.mock.build
=====================

Build a truth catalog (including spectra) and a targets catalog for the mocks. 

python -m cProfile -o bgsmock.dat /usr/local/repos/desihub/desitarget/bin/select_mock_targets -c mock_bright_bgs.yaml -s 444 --nproc 4
pyprof2calltree -k -i bgsmock.dat

"""
from __future__ import (absolute_import, division, print_function)

import os
import warnings
from time import time

import yaml
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column, vstack

from desispec.log import get_logger, DEBUG
from desispec.io.util import fitsheader, write_bintable

import desitarget.mock.io as mockio
import desitarget.mock.selection as mockselect
from desitarget.mock.spectra import MockSpectra
from desitarget.internal import sharedmem

log = get_logger(DEBUG)

class BrickInfo(object):
    """Gather information on all the bricks.

    """
    def __init__(self, random_state=None, dust_dir=None, bounds=(0.0, 359.99, -89.99, 89.99),
                 bricksize=0.25, decals_brick_info=None, target_names=None):
        """Initialize the class.

        Args:
          random_state : random number generator object
          dust_dir : path where the E(B-V) maps are stored
          bounds : brick boundaries
          bricksize : brick size (default 0.25 deg, square)
          decals_brick_info : filename of the DECaLS brick information structure
          target_names : list of targets (e.g., BGS, ELG, etc.)

        """
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

        self.dust_dir = dust_dir
        self.bounds = bounds
        self.bricksize = bricksize
        self.decals_brick_info = decals_brick_info
        self.target_names = target_names

    def generate_brick_info(self):
        """Generate the brick dictionary in the ragion (min_ra, max_ra, min_dec,
           max_dec).

           [Doesn't this functionality exist elsewhere?!?]

        """
        from desispec.brick import Bricks
        min_ra, max_ra, min_dec, max_dec = self.bounds

        B = Bricks(bricksize=self.bricksize)
        brick_info = {}
        brick_info['BRICKNAME'] = []
        brick_info['RA'] = []
        brick_info['DEC'] =  []
        brick_info['RA1'] =  []
        brick_info['RA2'] =  []
        brick_info['DEC1'] =  []
        brick_info['DEC2'] =   []
        brick_info['BRICKAREA'] =  [] 

        i_rows = np.where((B._edges_dec < (max_dec+B._bricksize/2)) &
                          (B._edges_dec > (min_dec-B._bricksize/2)))[0]
        
        for i_row in i_rows:
            j_col_min = int((min_ra )/360 * B._ncol_per_row[i_row])
            j_col_max = int((max_ra )/360 * B._ncol_per_row[i_row])
            
            for j_col in range(j_col_min, j_col_max+1):
                brick_info['BRICKNAME'].append(B._brickname[i_row][j_col])
                    
                brick_info['RA'].append(B._center_ra[i_row][j_col])
                brick_info['DEC'].append(B._center_dec[i_row])
        
                brick_info['RA1'].append(B._edges_ra[i_row][j_col])
                brick_info['DEC1'].append(B._edges_dec[i_row])
                    
                brick_info['RA2'].append(B._edges_ra[i_row][j_col+1])
                brick_info['DEC2'].append(B._edges_dec[i_row+1])
        
                brick_area = (brick_info['RA2'][-1]- brick_info['RA1'][-1]) 
                brick_area *= (np.sin(brick_info['DEC2'][-1]*np.pi/180.) -
                               np.sin(brick_info['DEC1'][-1]*np.pi/180.)) * 180 / np.pi
                brick_info['BRICKAREA'].append(brick_area)

        for k in brick_info.keys():
            brick_info[k] = np.array(brick_info[k])

        log.info('Generating brick information for {} brick(s) with boundaries RA={}, {}, Dec={}, {} and bricksize {} deg.'.\
                 format(len(brick_info['BRICKNAME']), self.bounds[0], self.bounds[1],
                        self.bounds[2], self.bounds[3], self.bricksize))
            
        return brick_info

    def extinction_across_bricks(self, brick_info):
        """Estimates E(B-V) across bricks.
    
        Args:
          brick_info : dictionary gathering brick information. It must have at
            least two keys 'RA' and 'DEC'.

        """
        from desitarget.mock import sfdmap

        #log.info('Generated extinction for {} bricks'.format(len(brick_info['RA'])))
        a = {}
        a['EBV'] = sfdmap.ebv(brick_info['RA'], brick_info['DEC'], mapdir=self.dust_dir)
        
        return a

    def depths_across_bricks(self, brick_info):
        """
        Generates a sample of magnitud dephts for a set of bricks.
    
        This model was built from the Data Release 3 of DECaLS.
    
        Args:
        -----------
            brick_info(Dictionary). Containts at least the following keys:
                RA (float): numpy array of RA positions
                DEC (float): numpy array of Dec positions
    
        Return:
        ------
            depths (dictionary). keys include
                'DEPTH_G', 'DEPTH_R', 'DEPTH_Z',
                'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z'.
                The values ofr each key ar numpy arrays (float) with size equal to 
                the input ra, dec arrays.
    
        """
        ra = brick_info['RA']
        dec = brick_info['DEC']
    
        n_to_generate = len(ra)
        #mean and std deviation of the difference between DEPTH and GALDEPTH in the DR3 data.
        differences = {}
        differences['DEPTH_G'] = [0.22263251, 0.059752077]
        differences['DEPTH_R'] = [0.26939404, 0.091162138]
        differences['DEPTH_Z'] = [0.34058815, 0.056099825]
        
        # (points, fractions) provide interpolation to the integrated probability distributions from DR3 data
        
        points = {}
        points['DEPTH_G'] = np.array([ 12.91721153,  18.95317841,  20.64332008,  23.78604698,  24.29093361,
                      24.4658947,   24.55436325,  24.61874771,  24.73129845,  24.94996071])
        points['DEPTH_R'] = np.array([ 12.91556168,  18.6766777,   20.29519463,  23.41814804,  23.85244179,
                      24.10131454,  24.23338318,  24.34066582,  24.53495026,  24.94865227])
        points['DEPTH_Z'] = np.array([ 13.09378147,  21.06531525,  22.42395782,  22.77471352,  22.96237755,
                      23.04913139,  23.43119431,  23.69817734,  24.1913662,   24.92163849])
    
        fractions = {}
        fractions['DEPTH_G'] = np.array([0.0, 0.01, 0.02, 0.08, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
        fractions['DEPTH_R'] = np.array([0.0, 0.01, 0.02, 0.08, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
        fractions['DEPTH_Z'] = np.array([0.0, 0.01, 0.03, 0.08, 0.2, 0.3, 0.7, 0.9, 0.99, 1.0])
    
        names = ['DEPTH_G', 'DEPTH_R', 'DEPTH_Z']
        depths = {}
        for name in names:
            fracs = self.random_state.random_sample(n_to_generate)
            depths[name] = np.interp(fracs, fractions[name], points[name])
    
            depth_minus_galdepth = self.random_state.normal(
                loc=differences[name][0], 
                scale=differences[name][1], size=n_to_generate)
            depth_minus_galdepth[depth_minus_galdepth<0] = 0.0
            
            depths['GAL'+name] = depths[name] - depth_minus_galdepth
            #log.info('Generated {} and GAL{} for {} bricks'.format(name, name, len(ra)))
            
        return depths

    def fluctuations_across_bricks(self, brick_info):
        """
        Generates number density fluctuations.
    
        Args:
          decals_brick_info (string). file summarizing tile statistics Data Release 3 of DECaLS. 
          brick_info(Dictionary). Containts at least the following keys:
            DEPTH_G(float) : array of depth magnitudes in the G band.
    
        Returns:
          fluctuations (dictionary) with keys 'FLUC+'depth, each one with values
            corresponding to a dictionary with keys ['ALL','LYA','MWS','BGS','QSO','ELG','LRG'].
            i.e. fluctuation[FLUC_DEPTH_G]['MWS'] holds the number density as a funtion 
            is a dictionary with keys corresponding to the different galaxy types.
        
        """
        from desitarget.QA import generate_fluctuations
    
        fluctuation = {}
        
        depth_available = []
    #   for k in brick_info.keys():        
        for k in ['GALDEPTH_R', 'EBV']:        
            if ('DEPTH' in k or 'EBV' in k):
                depth_available.append(k)
    
        for depth in depth_available:        
            fluctuation['FLUC_'+depth] = {}
            for ttype in self.target_names:
                fluctuation['FLUC_'+depth][ttype] = generate_fluctuations(self.decals_brick_info,
                                                                          ttype,
                                                                          depth,
                                                                          brick_info[depth],
                                                                          random_state=self.random_state)
                #log.info('Generated target fluctuation for type {} using {} as input for {} bricks'.format(
                #    ttype, depth, len(fluctuation['FLUC_'+depth][ttype])))
                
        return fluctuation

    def targetinfo(self):
        """Read target info from DESIMODEL, change all the keys to upper case, and
        append into brick_info.

        """ 
        filein = open(os.getenv('DESIMODEL')+'/data/targets/targets.dat')
        td = yaml.load(filein)
        target_desimodel = {}
        for t in td.keys():
            target_desimodel[t.upper()] = td[t]

        return target_desimodel

    def build_brickinfo(self):
        """Build the complete information structure."""

        brick_info = self.generate_brick_info()
        brick_info.update(self.extinction_across_bricks(brick_info))   # add extinction
        brick_info.update(self.depths_across_bricks(brick_info))       # add depths
        brick_info.update(self.fluctuations_across_bricks(brick_info)) # add number density fluctuations
        brick_info.update(self.targetinfo())                           # add nominal target densities

        return brick_info

def add_mock_shapes_and_fluxes(mocktargets, realtargets=None, random_state=None):
    '''Add SHAPEDEV_R and SHAPEEXP_R from a real target catalog.'''
    from desitarget.targetmask import desi_mask, bgs_mask
    
    if random_state is None:
        random_state = np.random.RandomState()
        
    n = len(mocktargets)
    
    for objtype in ('ELG', 'LRG', 'QSO'):
        mask = desi_mask.mask(objtype)
        #- indices where mock (ii) and real (jj) match the mask
        ii = np.where((mocktargets['DESI_TARGET'] & mask) != 0)[0]
        jj = np.where((realtargets['DESI_TARGET'] & mask) != 0)[0]
        if len(jj) == 0:
            log.warning('Real target catalog missing {}'.format(objtype))
            raise ValueError

        #- Which random jj should be used to fill in values for ii?
        kk = jj[random_state.randint(0, len(jj), size=len(ii))]
        mocktargets['SHAPEDEV_R'][ii] = realtargets['SHAPEDEV_R'][kk]
        mocktargets['SHAPEEXP_R'][ii] = realtargets['SHAPEEXP_R'][kk]

    for objtype in ('BGS_FAINT', 'BGS_BRIGHT'):
        mask = bgs_mask.mask(objtype)
        #- indices where mock (ii) and real (jj) match the mask
        ii = np.where((mocktargets['BGS_TARGET'] & mask) != 0)[0]
        jj = np.where((realtargets['BGS_TARGET'] & mask) != 0)[0]
        if len(jj) == 0:
            log.warning('Real target catalog missing {}'.format(objtype))
            raise ValueError

        #- Which jj should be used to fill in values for ii?
        #- NOTE: not filling in BGS or MWS fluxes, only shapes
        kk = jj[random_state.randint(0, len(jj), size=len(ii))]
        mocktargets['SHAPEDEV_R'][ii] = realtargets['SHAPEDEV_R'][kk]
        mocktargets['SHAPEEXP_R'][ii] = realtargets['SHAPEEXP_R'][kk]

def fileid_filename(source_data, output_dir):
    '''
    Outputs text file with mapping between mock filenum and file on disk

    returns mapping dictionary map[mockanme][filenum] = filepath
    
    '''
    out = open(os.path.join(output_dir, 'map_id_filename.txt'), 'w')
    map_id_name = {}
    for k in source_data.keys():
        map_id_name[k] = {}
        data = source_data[k]
        filenames = data['FILES']
        n_files = len(filenames)
        for i in range(n_files):
            map_id_name[k][i] = filenames[i]
            out.write('{} {} {}\n'.format(k,i, map_id_name[k][i]))
    out.close()

    return map_id_name

def empty_targets_table(nobj=1):
    """Initialize an empty 'targets' table.  The required output columns in order
    for fiberassignment to work are: TARGETID, RA, DEC, DESI_TARGET, BGS_TARGET,
    MWS_TARGET, SUBPRIORITY and OBSCONDITIONS.  Everything else is gravy.

    """
    from astropy.table import Table, Column

    targets = Table()

    # Columns required for fiber assignment:
    targets.add_column(Column(name='TARGETID', length=nobj, dtype='int64'))
    targets.add_column(Column(name='RA', length=nobj, dtype='f8'))
    targets.add_column(Column(name='DEC', length=nobj, dtype='f8'))
    targets.add_column(Column(name='DESI_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='BGS_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='MWS_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='SUBPRIORITY', length=nobj, dtype='f8'))
    targets.add_column(Column(name='OBSCONDITIONS', length=nobj, dtype='i4'))

    # Quantities mimicking a true targeting catalog (or inherited from the
    # mocks).
    targets.add_column(Column(name='BRICKNAME', length=nobj, dtype='S10'))
    targets.add_column(Column(name='DECAM_FLUX', shape=(6,), length=nobj, dtype='f4'))
    targets.add_column(Column(name='WISE_FLUX', shape=(2,), length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEEXP_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='DECAM_DEPTH', shape=(6,), length=nobj, data=np.zeros((nobj, 6))+99, dtype='f4'))
    targets.add_column(Column(name='DECAM_GALDEPTH', shape=(6,), length=nobj, data=np.zeros((nobj, 6))+99, dtype='f4'))

    return targets

def empty_truth_table(nobj=1):
    """Initialize the truth table for each mock object, with spectra.
    
    """
    from astropy.table import Table, Column

    truth = Table()
    truth.add_column(Column(name='TARGETID', length=nobj, dtype='int64'))
    truth.add_column(Column(name='MOCKID', length=nobj, dtype='int64'))

    truth.add_column(Column(name='TRUEZ', length=nobj, dtype='f4', data=np.zeros(nobj)))
    truth.add_column(Column(name='TRUESPECTYPE', length=nobj, dtype=(str, 10))) # GALAXY, QSO, STAR, etc.
    truth.add_column(Column(name='TEMPLATETYPE', length=nobj, dtype=(str, 10))) # ELG, BGS, STAR, WD, etc.
    truth.add_column(Column(name='TEMPLATESUBTYPE', length=nobj, dtype=(str, 10))) # DA, DB, etc.

    truth.add_column(Column(name='TEMPLATEID', length=nobj, dtype='i4', data=np.zeros(nobj)-1))
    truth.add_column(Column(name='SEED', length=nobj, dtype='int64', data=np.zeros(nobj)-1))
    truth.add_column(Column(name='MAG', length=nobj, dtype='f4', data=np.zeros(nobj)+99))
    truth.add_column(Column(name='DECAM_FLUX', shape=(6,), length=nobj, dtype='f4'))
    truth.add_column(Column(name='WISE_FLUX', shape=(2,), length=nobj, dtype='f4'))

    truth.add_column(Column(name='OIIFLUX', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='erg/(s*cm2)'))
    truth.add_column(Column(name='HBETAFLUX', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='erg/(s*cm2)'))

    truth.add_column(Column(name='TEFF', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='K'))
    truth.add_column(Column(name='LOGG', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='m/(s**2)'))
    truth.add_column(Column(name='FEH', length=nobj, dtype='f4', data=np.zeros(nobj)-1))

    return truth

def _get_spectra_onebrick(specargs):
    """Filler function for the multiprocessing."""
    return get_spectra_onebrick(*specargs)

def get_spectra_onebrick(target_name, mockformat, thisbrick, brick_info, Spectra, source_data, rand):
    """Wrapper function to generate spectra for all the objects on a single brick."""

    brickindx = np.where(brick_info['BRICKNAME'] == thisbrick)[0]
    nbrick = len(brickindx)

    onbrick = np.where(source_data['BRICKNAME'] == thisbrick)[0]
    #print('HACK!!!!!!!!!!!!!!!')
    #onbrick = onbrick[:100]
    nobj = len(onbrick)

    #log.info('{}, {} objects'.format(thisbrick, nobj))
    
    if (nbrick != 1) or (nobj == 0):
        log.warning('No matching brick or no matching objects in brick {}!'.format(thisbrick))
        # warnings.warn("Tile is on the border. DEPTH_R = 99.0. GALDEPTH_R = 99.0", RuntimeWarning)
        _targets = empty_targets_table()
        _truth = empty_truth_table()
        _trueflux = np.zeros((1, len(Spectra.wave)), dtype='f4')
        _onbrick = np.array([], dtype=int)
        #import pdb ; pdb.set_trace()
        return [_targets, _truth, _trueflux, _onbrick]
        
    targets = empty_targets_table(nobj)
    truth = empty_truth_table(nobj)

    # Generate spctra.    
    trueflux, meta = getattr(Spectra, target_name.lower())(source_data, index=onbrick, mockformat=mockformat)

    for key in ('TEMPLATEID', 'SEED', 'MAG', 'DECAM_FLUX', 'WISE_FLUX',
                'OIIFLUX', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'):
        truth[key] = meta[key]
        
    for band, depthkey in zip((1, 2, 4), ('DEPTH_G', 'DEPTH_R', 'DEPTH_Z')):
        targets['DECAM_DEPTH'][:, band] = brick_info[depthkey][brickindx]
    for band, depthkey in zip((1, 2, 4), ('GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z')):
        targets['DECAM_GALDEPTH'][:, band] = brick_info[depthkey][brickindx]

    # Perturb the photometry based on the variance on this brick.  Hack!  Assume
    # a constant depth (22.3-->1.2 nanomaggies, 23.8-->0.3 nanomaggies) in the
    # WISE bands for now.
    wise_onesigma = np.zeros((nobj, 2))
    wise_onesigma[:, 0] = 1.2
    wise_onesigma[:, 1] = 0.3
    targets['WISE_FLUX'] = truth['WISE_FLUX'] + rand.normal(scale=wise_onesigma)
    
    for band in (1, 2, 4):
        targets['DECAM_FLUX'][:, band] = truth['DECAM_FLUX'][:, band] + \
          rand.normal(scale=1.0/np.sqrt(targets['DECAM_DEPTH'][:, band]))
          
    return [targets, truth, trueflux, onbrick]

def _write_onebrick(writeargs):
    """Filler function for the multiprocessing."""
    return write_onebrick(*writeargs)

def write_onebrick(thisbrick, targets, truth, trueflux, truthhdr, wave, output_dir):
    """Wrapper function to write out files on a single brick."""

    onbrick = np.where(targets['BRICKNAME'] == thisbrick)[0]

    radir = os.path.join(output_dir, thisbrick[:3])
    targetsfile = os.path.join(radir, 'targets-{}.fits'.format(thisbrick))
    truthfile = os.path.join(radir, 'truth-{}.fits'.format(thisbrick))
    log.info('Writing {}.'.format(truthfile))

    targets[onbrick].write(targetsfile, overwrite=True)
            
    hx = fits.HDUList()
    hdu = fits.ImageHDU(wave.astype(np.float32), name='WAVE', header=truthhdr)
    hx.append(hdu)

    hdu = fits.ImageHDU(trueflux.astype(np.float32), name='FLUX')
    hdu.header['BUNIT'] = '1e-17 erg/s/cm2/A'
    hx.append(hdu)

    hx.writeto(truthfile, overwrite=True)
    write_bintable(truthfile, truth[onbrick], extname='TRUTH')
    
    #import pdb ; pdb.set_trace()

def targets_truth(params, output_dir, realtargets=None, seed=None,
                  bricksize=0.25, nproc=4, verbose=True):
    """
    Write

    Args:
        params: dict of source definitions.
        output_dir: location for intermediate mtl files.

    Options:
        realtargets: real target catalog table, e.g. from DR3
        nproc : number of parallel processes to use (default 4)

    Returns:
      targets:
      truth:

    Notes:
      If nproc == 1 use serial instead of parallel code.
    
    """
    rand = np.random.RandomState(seed)

    # Build the brick information structure.
    if ('subset' in params.keys()) & (params['subset']['ra_dec_cut'] == True):
        bounds = (params['subset']['min_ra'], params['subset']['max_ra'],
                  params['subset']['min_dec'], params['subset']['max_dec'])
    else:
        bounds=(0.0, 359.99, -89.99, 89.99)
        
    brick_info = BrickInfo(random_state=rand, dust_dir=params['dust_dir'], bounds=bounds,
                           bricksize=bricksize, decals_brick_info=params['decals_brick_info'],
                           target_names=list(params['sources'].keys())).build_brickinfo()

    # Initialize the Classes used to assign spectra and select targets.  Note:
    # The default wavelength array gets initialized here, too.
    log.info('Initializing the MockSpectra and SelectTargets classes.')
    Spectra = MockSpectra()
    SelectTargets = mockselect.SelectTargets()

    # Print info about the mocks we will be loading and then load them.
    if verbose:
        mockio.print_all_mocks_info(params)
    source_data_all = mockio.load_all_mocks(params, rand=rand, bricksize=bricksize)
    # map_fileid_filename = fileid_filename(source_data_all, output_dir)

    #import pdb ; pdb.set_trace()

    # Loop over each source / object type.
    alltargets = list()
    alltruth = list()
    alltrueflux = list()

    source_defs = params['sources']
    for source_name in sorted(source_defs.keys()):
        log.info('Assigning spectra and selecting targets for source {}.'.format(source_name))
        
        target_name = params['sources'][source_name]['target_name'] # Target type (e.g., ELG, BADQSO)
        mockformat = params['sources'][source_name]['format']
        source_data = source_data_all[source_name]     # data (ra, dec, etc.)

        #getSpectra_function = 'getspectra_{}_{}'.format(target_name.lower())
        #log.info('Generating spectra using function {}.'.format(getSpectra_function))

        # Assign spectra by parallel-processing the bricks.
        #brickname = get_brickname_from_radec(source_data['RA'], source_data['DEC'])#, bricksize=bricksize)
        brickname = source_data['BRICKNAME']
        unique_bricks = list(set(brickname))
        #unique_bricks = list(set(brickname[:5]))
        #print('HACK!!!!!!!!!!!!!!!!!!!!!!!!')
        log.info('Assigned objects to {} unique bricks.'.format(len(unique_bricks)))

        nbrick = np.zeros((), dtype='i8')
        t0 = time()
        def _update_spectra_status(result):
            if nbrick % 5 == 0 and nbrick > 0:
            #if verbose and nbrick % 5 == 0 and nbrick > 0:
                rate = nbrick / (time() - t0)
                log.info('{} bricks; {:.1f} bricks / sec'.format(nbrick, rate))
                #rate = (time() - t0) / nbrick
                #print('{} bricks; {:.1f} sec / brick'.format(nbrick, rate))
            nbrick[...] += 1    # this is an in-place modification
            return result
    
        specargs = list()
        for thisbrick in unique_bricks:
            specargs.append((target_name, mockformat, thisbrick, brick_info, Spectra, source_data, rand))
                
        if nproc > 1:
            pool = sharedmem.MapReduce(np=nproc)
            with pool:
                out = pool.map(_get_spectra_onebrick, specargs, reduce=_update_spectra_status)
        else:
            out = list()
            for ii in range(len(unique_bricks)):
                out.append(_update_spectra_status(_get_spectra_onebrick(specargs[ii])))

        # Initialize and then populate the truth and targets tables. 
        nobj = len(source_data['RA'])
        targets = empty_targets_table(nobj)
        truth = empty_truth_table(nobj)
        trueflux = np.zeros((nobj, len(Spectra.wave)), dtype='f4')

        for ii in range(len(unique_bricks)):
            targets[out[ii][3]] = out[ii][0]
            truth[out[ii][3]] = out[ii][1]
            trueflux[out[ii][3], :] = out[ii][2]

        targets['RA'] = source_data['RA']
        targets['DEC'] = source_data['DEC']
        targets['BRICKNAME'] = brickname
        
        truth['MOCKID'] = source_data['MOCKID']
        truth['TRUEZ'] = source_data['Z'].astype('f4')
        truth['TEMPLATETYPE'] = source_data['TEMPLATETYPE']
        truth['TEMPLATESUBTYPE'] = source_data['TEMPLATESUBTYPE']
        truth['TRUESPECTYPE'] = source_data['TRUESPECTYPE']

        # Select targets and get the targeting bits.
        selection_function = '{}_select'.format(target_name.lower())
        log.info('Selecting {} targets using {} function.'.format(source_name, selection_function))

        getattr(SelectTargets, selection_function)(targets, truth)
        #import pdb ; pdb.set_trace()
        
        keep = targets['DESI_TARGET'] != 0

        alltargets.append(targets[keep])
        alltruth.append(truth[keep])
        alltrueflux.append(trueflux[keep, :])

    # Consolidate across all the mocks and then assign TARGETIDs, subpriorities,
    # and shapes and fluxes.
    targets = vstack(alltargets)
    truth = vstack(alltruth)
    trueflux = np.concatenate(alltrueflux)
    ntarget = len(targets)

    targetid = rand.randint(2**62, size=ntarget)
    truth['TARGETID'] = targetid
    targets['TARGETID'] = targetid
    targets['SUBPRIORITY'] = rand.uniform(0.0, 1.0, size=ntarget)

    if realtargets is not None:
        add_mock_shapes_and_fluxes(targets, realtargets, random_state=rand)

    log.info('DO A FINAL CHECK OF THE DENSITIES AND SUBSAMPLE IF NECESSARY!!!')


    # Write out.
    log.info('Writing out.')

    # Create the RA-slice directories, if necessary.
    radir = np.array(['{}'.format(os.path.join(output_dir, name[:3])) for name in targets['BRICKNAME']])
    for thisradir in list(set(radir)):
        try:
            os.stat(thisradir)
        except:
            os.makedirs(thisradir)

    # Initialize the output header.
    if seed is None:
        seed1 = 'None'
    else:
        seed1 = seed
    truthhdr = fitsheader(dict(
        SEED = (seed1, 'initial random seed'),
        BRICKSZ = (bricksize, 'bricksize'),
        BUNIT = ('Angstrom', 'wavelength units'),
        AIRORVAC = ('vac', 'vacuum wavelengths')
        ))

    nbrick = np.zeros((), dtype='i8')
    t0 = time()
    def _update_write_status(result):
        if verbose and nbrick % 5 == 0 and nbrick > 0:
            rate = nbrick / (time() - t0)
            #rate = (time() - t0) / nbrick
            print('Writing {} bricks; {:.1f} bricks / sec'.format(nbrick, rate))
        nbrick[...] += 1    # this is an in-place modification
        return result

    unique_bricks = list(set(targets['BRICKNAME']))
    
    writeargs = list()
    for thisbrick in unique_bricks:
        writeargs.append((thisbrick, targets, truth, trueflux, truthhdr, Spectra.wave, output_dir))
                
    if nproc > 1:
        pool = sharedmem.MapReduce(np=nproc)
        with pool:
            pool.map(_write_onebrick, writeargs, reduce=_update_write_status)
    else:
        for ii in range(len(unique_bricks)):
            _update_write_status(_write_onebrick(writeargs[ii]))



#    # consolidates all relevant arrays across mocks
#    ra_total = np.empty(0)
#    dec_total = np.empty(0)
#    z_total = np.empty(0)
#    mockid_total = np.empty(0, dtype='int64')
#    desi_target_total = np.empty(0, dtype='i8')
#    bgs_target_total = np.empty(0, dtype='i8')
#    mws_target_total = np.empty(0, dtype='i8')
#    true_type_total = np.empty(0, dtype='S10')
#    source_type_total = np.empty(0, dtype='S10')
#    obsconditions_total = np.empty(0, dtype='uint16')
#    decam_flux = np.empty((0,6), dtype='f4')
#
#    print('Collects information across mock files')
#    for source_name in sorted(source_defs.keys()):
#        target_name = params['sources'][source_name]['target_name']
#        truth_name = params['sources'][source_name]['truth_name']
#        source_data = source_data_all[source_name]
#        target_mask = target_mask_all[source_name]
#
#        ii = target_mask >-1 #only targets that passed cuts
#
#        #define all flags
#        desi_target = 0 * target_mask[ii]
#        bgs_target = 0 * target_mask[ii] 
#        mws_target = 0 * target_mask[ii]
#        if target_name in ['ELG', 'LRG', 'QSO', 'STD_FSTAR', 'SKY']:
#            desi_target = target_mask[ii]
#        if target_name in ['BGS']:
#            bgs_target = target_mask[ii]
#            desi_target |= desi_mask.BGS_ANY
#        if target_name in ['MWS_MAIN', 'MWS_WD' ,'MWS_NEARBY']:
#            mws_target = target_mask[ii]
#            desi_target |= desi_mask.MWS_ANY
#
#
#        # define names that go into Truth
#        n = len(source_data['RA'][ii])
#        #if source_name not in ['STD_FSTAR', 'SKY']:
#        #    true_type_map = {
#        #        'STD_FSTAR': 'STAR',
#        #        'ELG': 'GALAXY',
#        #        'LRG': 'GALAXY',
#        #        'BGS': 'GALAXY',
#        #        'QSO': 'QSO',
#        #        'STD_FSTAR': 'STAR',
#        #        'MWS_MAIN': 'STAR',
#        #        'MWS_WD': 'STAR',
#        #        'MWS_NEARBY': 'STAR',
#        #        'SKY': 'SKY',
#        #    }
#        #    source_type = np.zeros(n, dtype='S10')
#        #    source_type[:] = target_name
#        #    true_type = np.zeros(n, dtype='S10')
#        #    true_type[:] = true_type_map[truth_name]
#
#        ##define obsconditions
#        #source_obsconditions = np.ones(n,dtype='uint16')
#        #if target_name in ['LRG', 'QSO']:
#        #    source_obsconditions[:] = obsconditions.DARK
#        #if target_name in ['ELG']:
#        #    source_obsconditions[:] = obsconditions.DARK|obsconditions.GRAY
#        #if target_name in ['BGS']:
#        #    source_obsconditions[:] = obsconditions.BRIGHT
#        #if target_name in ['MWS_MAIN', 'MWS_WD', 'MWS_NEARBY']:
#        #    source_obsconditions[:] = obsconditions.BRIGHT
#        #if target_name in ['STD_FSTAR', 'SKY']:
#        #    source_obsconditions[:] = obsconditions.DARK|obsconditions.GRAY|obsconditions.BRIGHT 
#
#        #append to the arrays that will go into Targets
#        if source_name in ['STD_FSTAR']:
#            ra_stars = source_data['RA'][ii].copy()
#            dec_stars = source_data['DEC'][ii].copy()
#            desi_target_stars = desi_target.copy()
#            bgs_target_stars = bgs_target.copy()
#            mws_target_stars = mws_target.copy()
#            obsconditions_stars = source_obsconditions.copy()
#        if source_name in ['SKY']:
#            ra_sky = source_data['RA'][ii].copy()
#            dec_sky = source_data['DEC'][ii].copy()
#            desi_target_sky = desi_target.copy()
#            bgs_target_sky = bgs_target.copy()
#            mws_target_sky = mws_target.copy()
#            obsconditions_sky = source_obsconditions.copy()
#        if source_name not in ['SKY', 'STD_FSTAR']:
#            ra_total = np.append(ra_total, source_data['RA'][ii])
#            dec_total = np.append(dec_total, source_data['DEC'][ii])
#            z_total = np.append(z_total, source_data['Z'][ii])
#            desi_target_total = np.append(desi_target_total, desi_target)
#            bgs_target_total = np.append(bgs_target_total, bgs_target)
#            mws_target_total = np.append(mws_target_total, mws_target)
#            true_type_total = np.append(true_type_total, true_type)
#            source_type_total = np.append(source_type_total, source_type)
#            obsconditions_total = np.append(obsconditions_total, source_obsconditions)
#            mockid_total = np.append(mockid_total, source_data['MOCKID'][ii])
#
#            # --------------------------------------------------
#            # HERE!!  Need to read in more of the rest-frame properties so we can do the mapping to spectra.
#
#            ##- Add fluxes, which default to 0 if the mocks don't have them
#            #if 'DECAMr_true' in source_data and 'DECAMr_obs' not in source_data:
#            #    from desitarget.mock import sfdmap
#            #    ra = source_data['RA']
#            #    dec = source_data['DEC']
#            #    ebv = sfdmap.ebv(ra, dec, mapdir=params['dust_dir'])
#            #    #- Magic number for extinction coefficient from https://github.com/dstndstn/tractor/blob/39f883c811f0a6b17a44db140d93d4268c6621a1/tractor/sfd.py
#            #    source_data['DECAMr_obs'] = source_data['DECAMr_true'] + ebv*2.165
#            #
#            #if 'DECAM_FLUX' in source_data:
#            #    decam_flux = np.append(decam_flux, source_data['DECAM_FLUX'][ii])
#            #else:
#            #    n = len(desi_target)
#            #    tmpflux = np.zeros((n,6), dtype='f4')
#            #    if 'DECAMg_obs' in source_data:
#            #        tmpflux[:,1] = 10**(0.4*(22.5-source_data['DECAMg_obs'][ii]))
#            #    if 'DECAMr_obs' in source_data:
#            #        tmpflux[:,2] = 10**(0.4*(22.5-source_data['DECAMr_obs'][ii]))
#            #    if 'DECAMz_obs' in source_data:
#            #        tmpflux[:,4] = 10**(0.4*(22.5-source_data['DECAMz_obs'][ii]))
#            #    decam_flux = np.vstack([decam_flux, tmpflux])
#            ## --------------------------------------------------
#
#        print('source {} target {} truth {}: selected {} out of {}'.format(
#                source_name, target_name, truth_name, len(source_data['RA'][ii]), len(source_data['RA'])))
#
#    # create unique IDs, subpriorities and bricknames across all mock files
#    n_target = len(ra_total)     
#    n_star = 0
#    n_sky = 0
#    n  = n_target    
#    if 'STD_FSTAR' in source_defs.keys():
#        n_star = len(ra_stars)
#        n += n_star
#    if 'SKY' in source_defs.keys():
#        n_sky = len(ra_sky)
#        n += n_sky
#    print('Great total of {} targets {} stdstars {} sky pos'.format(n_target, n_star, n_sky))
#    targetid = rand.randint(2**62, size=n)
#
#    # write to disk
#    if 'STD_FSTAR' in source_defs.keys():
#        subprior = rand.uniform(0., 1., size=n_star)
#        #write the Std Stars to disk
#        print('Started writing StdStars file')
#        stars_filename = os.path.join(output_dir, 'stdstars.fits')
#        stars = Table()
#        stars['TARGETID'] = targetid[n_target:n_target+n_star]
#        stars['RA'] = ra_stars
#        stars['DEC'] = dec_stars
#        stars['DESI_TARGET'] = desi_target_stars
#        stars['BGS_TARGET'] = bgs_target_stars
#        stars['MWS_TARGET'] = mws_target_stars
#        stars['SUBPRIORITY'] = subprior
#        stars['OBSCONDITIONS'] = obsconditions_stars
#        # if ('subset' in params.keys()) & (params['subset']['ra_dec_cut']==True):
#        #     stars = stars[ii_stars]
#        #     print('subsetting in std_stars data done')
#        brickname = get_brickname_from_radec(stars['RA'], stars['DEC'])
#        stars['BRICKNAME'] = brickname
#        stars.write(stars_filename, overwrite=True)
#        print('Finished writing stdstars file')
#
#    if 'SKY' in source_defs.keys():
#        subprior = rand.uniform(0., 1., size=n_sky)
#        #write the Std Stars to disk
#        print('Started writing sky to file')
#        sky_filename = os.path.join(output_dir, 'sky.fits')
#        sky = Table()
#        sky['TARGETID'] = targetid[n_target+n_star:n_target+n_star+n_sky]
#        sky['RA'] = ra_sky
#        sky['DEC'] = dec_sky
#        sky['DESI_TARGET'] = desi_target_sky
#        sky['BGS_TARGET'] = bgs_target_sky
#        sky['MWS_TARGET'] = mws_target_sky
#        sky['SUBPRIORITY'] = subprior
#        sky['OBSCONDITIONS'] = obsconditions_sky
#        brickname = get_brickname_from_radec(sky['RA'], sky['DEC'])
#        sky['BRICKNAME'] = brickname
#        sky.write(sky_filename, overwrite=True)
#        print('Finished writing sky file')
#
#    if n_target > 0:
#        subprior = rand.uniform(0., 1., size=n_target)
#        # write the Targets to disk
#        print('Started writing Targets file')
#        targets_filename = os.path.join(output_dir, 'targets.fits')
#        targets = Table()
#        targets['TARGETID'] = targetid[0:n_target]
#        targets['RA'] = ra_total
#        targets['DEC'] = dec_total
#        targets['DESI_TARGET'] = desi_target_total
#        targets['BGS_TARGET'] = bgs_target_total
#        targets['MWS_TARGET'] = mws_target_total
#        targets['SUBPRIORITY'] = subprior
#        targets['OBSCONDITIONS'] = obsconditions_total
#        brickname = get_brickname_from_radec(targets['RA'], targets['DEC'])
#        targets['BRICKNAME'] = brickname          
#
#        targets['DECAM_FLUX'] = decam_flux
#        add_mock_shapes_and_fluxes(targets, realtargets)
#        add_galdepths(targets, brick_info)
#        targets.write(targets_filename, overwrite=True)
#        print('Finished writing Targets file')
#
#        # write the Truth to disk
#        print('Started writing Truth file')
#        truth_filename = os.path.join(output_dir, 'truth.fits')
#        truth = Table()
#        truth['TARGETID'] = targetid[0:n_target]
#        truth['RA'] = ra_total
#        truth['DEC'] = dec_total
#        truth['TRUEZ'] = z_total
#        truth['TRUETYPE'] = true_type_total
#        truth['SOURCETYPE'] = source_type_total
#        truth['BRICKNAME'] = brickname
#        truth['MOCKID'] = mockid_total
#
#        add_OIIflux(targets, truth, random_state=rand)
#        
#        truth.write(truth_filename, overwrite=True)
#        print('Finished writing Truth file')
#
#def add_galdepths(mocktargets, brickinfo):
#    '''
#    Add GALDEPTH_R and DEPTH_R.
#    Modifies mocktargets by adding columns.
#    DEPTHS are constant across bricks.
#    '''
#    n = len(mocktargets)
#    if 'DEPTH_R' not in mocktargets.dtype.names:
#        mocktargets['DEPTH_R'] = 99.0*np.ones(n, dtype='f4')
#
#    if 'GALDEPTH_R' not in mocktargets.dtype.names:
#        mocktargets['GALDEPTH_R'] = 99.0*np.ones(n, dtype='f4')
#
#    # create dictionary with targets per brick
#    
#    bricks = get_brickname_from_radec(mocktargets['RA'], mocktargets['DEC'])
#    unique_bricks = list(set(bricks))
#    lookup = mockselect.make_lookup_dict(bricks)
#    n_brick = len(unique_bricks)
#    i_brick = 0
#    for brickname in unique_bricks:
#        in_brick = np.array(lookup[brickname])
#        i_brick += 1
##       print('brick {} out of {}'.format(i_brick,n_brick))                
#        id_binfo  = (brickinfo['BRICKNAME'] == brickname)
#        if np.count_nonzero(id_binfo) == 1:
#            mocktargets['DEPTH_R'][in_brick] = brickinfo['DEPTH_R'][id_binfo]
#            mocktargets['GALDEPTH_R'][in_brick] = brickinfo['GALDEPTH_R'][id_binfo]
#        else:
#            warnings.warn("Tile is on the border. DEPTH_R = 99.0. GALDEPTH_R = 99.0", RuntimeWarning)
#
#def add_OIIflux(targets, truth, random_state=None):
#    '''
#    PLACEHOLDER: add fake OIIFLUX entries to truth for ELG targets
#    
#    Args:
#        targets: target selection catalog Table or structured array
#        truth: target selection catalog Table
#    
#    Note: Modifies truth table in place by adding OIIFLUX column
#
#    '''
#    if random_state is None:
#        random_state = np.random.RandomState()
#        
#    assert np.all(targets['TARGETID'] == truth['TARGETID'])
#
#    ntargets = len(targets)
#    truth['OIIFLUX'] = np.zeros(ntargets, dtype=float)
#    
#    isELG = (targets['DESI_TARGET'] & desi_mask.ELG) != 0
#    nELG = np.count_nonzero(isELG)
#
#    #- TODO: make this a meaningful distribution
#    #- At low redshift and low flux, r-band flux sets an approximate
#    #- upper limit on [OII] flux, but no lower limit; treat as uniform
#    #- within a r-flux dependent upper limit
#    rflux = targets['DECAM_FLUX'][isELG][:,2]
#    maxflux = np.clip(3e-16*rflux, 0, 7e-16)
#    truth['OIIFLUX'][isELG] = maxflux * random_state.uniform(0,1.0,size=nELG)
#
#
