# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.build
=====================

Build a truth catalog (including spectra) and a targets catalog for the mocks.

"""
from __future__ import (absolute_import, division, print_function)

import os

import numpy as np
from astropy.table import Table, Column, vstack, hstack

from desiutil.log import get_logger, DEBUG
from desitarget import desi_mask, bgs_mask, mws_mask, contam_mask, targetid_mask, obsconditions
import desitarget.mock.io as mockio
from desitarget.mock import sfdmap

from desitarget.targets import encode_targetid

def fileid_filename(source_data, output_dir, log):
    '''
    Outputs text file with mapping between mock filenum and file on disk

    returns mapping dictionary map[mockanme][filenum] = filepath

    '''
    outfile = os.path.join(output_dir, 'map_id_filename.txt')
    log.info('Writing {}'.format(outfile))
    
    out = open(outfile, 'w')
    map_id_name = {}
    for k in source_data.keys():
        map_id_name[k] = {}
        data = source_data[k]
        if 'FILES' in data.keys():
            filenames = data['FILES']
            n_files = len(filenames)
            for i in range(n_files):
                map_id_name[k][i] = filenames[i]
                out.write('{} {} {}\n'.format(k, i, map_id_name[k][i]))
    out.close()

    return map_id_name

class BrickInfo(object):
    """Gather information on all the bricks.

    """
    def __init__(self, random_state=None, dust_dir=None, bricksize=0.25,
                 decals_brick_info=None, target_names=None, log=None):
        """Initialize the class.

        Args:
          random_state : random number generator object
          dust_dir : path where the E(B-V) maps are stored
          bricksize : brick size (default 0.25 deg, square)
          decals_brick_info : filename of the DECaLS brick information structure
          target_names : list of targets (e.g., BGS, ELG, etc.)

        """
        if log:
            self.log = log
        else:
            self.log = get_logger()
            
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

        self.dust_dir = dust_dir
        self.bricksize = bricksize
        self.decals_brick_info = decals_brick_info
        self.target_names = target_names

    def generate_brick_info(self):
        """Generate the brick dictionary in the region (min_ra, max_ra, min_dec,
        max_dec).

        [Doesn't this functionality exist elsewhere?!?]

        """
        from desiutil.brick import Bricks

        brick_info = Bricks(bricksize=self.bricksize).to_table()
        nbrick = len(brick_info)

        self.log.info('Generated brick information for {} brick(s) with bricksize {:g} deg.'.\
                      format(nbrick, self.bricksize))

        return brick_info

    def extinction_across_bricks(self, brick_info):
        """Estimates E(B-V) across bricks.

        Args:
          brick_info : dictionary gathering brick information. It must have at
            least two keys 'RA' and 'DEC'.

        """
        #log.info('Generated extinction for {} bricks'.format(len(brick_info['RA'])))
        a = Table()
        a['EBV'] = sfdmap.ebv(brick_info['RA'], brick_info['DEC'], mapdir=self.dust_dir)

        return a

    def depths_across_bricks(self, brick_info):
        """
        Generates a sample of magnitud dephts for a set of bricks.

        This model was built from the Data Release 3 of DECaLS.

        Args:
            brick_info(Dictionary). Containts at least the following keys:
                RA (float): numpy array of RA positions
                DEC (float): numpy array of Dec positions

        Returns:
            depths (dictionary). keys include
                'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z'.
                The values ofr each key ar numpy arrays (float) with size equal to
                the input ra, dec arrays.

        """
        ra = brick_info['RA']
        dec = brick_info['DEC']

        n_to_generate = len(ra)
        #mean and std deviation of the difference between PSFDEPTH and GALDEPTH in the DR3 data.
        differences = {}
        differences['PSFDEPTH_G'] = [0.22263251, 0.059752077]
        differences['PSFDEPTH_R'] = [0.26939404, 0.091162138]
        differences['PSFDEPTH_Z'] = [0.34058815, 0.056099825]

        # (points, fractions) provide interpolation to the integrated probability distributions from DR3 data

        points = {}
        points['PSFDEPTH_G'] = np.array([ 12.91721153,  18.95317841,  20.64332008,  23.78604698,  24.29093361,
                      24.4658947,   24.55436325,  24.61874771,  24.73129845,  24.94996071])
        points['PSFDEPTH_R'] = np.array([ 12.91556168,  18.6766777,   20.29519463,  23.41814804,  23.85244179,
                      24.10131454,  24.23338318,  24.34066582,  24.53495026,  24.94865227])
        points['PSFDEPTH_Z'] = np.array([ 13.09378147,  21.06531525,  22.42395782,  22.77471352,  22.96237755,
                      23.04913139,  23.43119431,  23.69817734,  24.1913662,   24.92163849])

        fractions = {}
        fractions['PSFDEPTH_G'] = np.array([0.0, 0.01, 0.02, 0.08, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
        fractions['PSFDEPTH_R'] = np.array([0.0, 0.01, 0.02, 0.08, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
        fractions['PSFDEPTH_Z'] = np.array([0.0, 0.01, 0.03, 0.08, 0.2, 0.3, 0.7, 0.9, 0.99, 1.0])

        names = ['PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z']
        depths = Table()
        for name in names:
            fracs = self.random_state.random_sample(n_to_generate)
            depths[name] = np.interp(fracs, fractions[name], points[name])

            depth_minus_galdepth = self.random_state.normal(
                loc=differences[name][0],
                scale=differences[name][1], size=n_to_generate)
            depth_minus_galdepth[depth_minus_galdepth<0] = 0.0

            depths[name.replace('PSF', 'GAL')] = depths[name] - depth_minus_galdepth
            #log.info('Generated {} and GAL{} for {} bricks'.format(name, name, len(ra)))

        return depths

    def fluctuations_across_bricks(self, brick_info):
        """
        Generates number density fluctuations.

        Args:
          decals_brick_info (string). file summarizing tile statistics Data Release 3 of DECaLS.
          brick_info(Dictionary). Containts at least the following keys:
            PSFDEPTH_G(float) : array of depth magnitudes in the G band.

        Returns:
          fluctuations (dictionary) with keys 'FLUC+'depth, each one with values
            corresponding to a dictionary with keys ['ALL','LYA','MWS','BGS','QSO','ELG','LRG'].
            i.e. fluctuation[FLUC_DEPTH_G_MWS] holds the number density as a funtion
            is a dictionary with keys corresponding to the different galaxy types.

        """
        from desitarget.QA import generate_fluctuations

        fluctuation = Table()

        depth_available = []
    #   for k in brick_info.keys():
        for k in ['GALDEPTH_R', 'EBV']:
            if ('PSFDEPTH' in k or 'EBV' in k):
                depth_available.append(k)

        for depth in depth_available:
            for ttype in self.target_names:
                fluctuation['FLUC_{}_{}'.format(depth, ttype)] = generate_fluctuations(
                    self.decals_brick_info, ttype, depth, brick_info[depth].data,
                    random_state=self.random_state
                    )
                #log.info('Generated target fluctuation for type {} using {} as input for {} bricks'.format(
                #    ttype, depth, len(fluctuation['FLUC_'+depth][ttype])))
        return fluctuation

    def targetinfo(self):
        """Read target info from DESIMODEL, change all the keys to upper case, and
        append into brick_info.

        """
        import yaml
        with open(os.path.join( os.getenv('DESIMODEL'), 'data', 'targets', 'targets.yaml' ), 'r') as filein:
            td = yaml.load(filein)
        target_desimodel = Table()
        for t in td.keys():
            if 'ntarget' in t.upper():
                target_desimodel[t.upper()] = td[t]

        return target_desimodel

    def build_brickinfo(self):
        """Build the complete information structure."""
        from astropy.table import hstack

        brick_info = self.generate_brick_info()
        brick_info = hstack( (brick_info, self.extinction_across_bricks(brick_info)) )   # add extinction
        brick_info = hstack( (brick_info, self.depths_across_bricks(brick_info)) )       # add depths
        brick_info = hstack( (brick_info, self.fluctuations_across_bricks(brick_info)) ) # add number density fluctuations
        #brick_info = hstack( (brick_info, self.targetinfo()) )                          # add nominal target densities

        return brick_info

def add_mock_shapes_and_fluxes(mocktargets, realtargets=None, random_state=None):
    '''Add SHAPEDEV_R and SHAPEEXP_R from a real target catalog.'''

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

def empty_targets_table(nobj=1):
    """Initialize an empty 'targets' table.  The required output columns in order
    for fiberassignment to work are: TARGETID, RA, DEC, DESI_TARGET, BGS_TARGET,
    MWS_TARGET, and SUBPRIORITY.  Everything else is gravy.

    """
    targets = Table()

    # RELEASE
    targets.add_column(Column(name='BRICKID', length=nobj, dtype='i4'))
    targets.add_column(Column(name='BRICKNAME', length=nobj, dtype='U8'))
    targets.add_column(Column(name='BRICK_OBJID', length=nobj, dtype='i4'))
    # TYPE
    targets.add_column(Column(name='RA', length=nobj, dtype='f8'))
    targets.add_column(Column(name='DEC', length=nobj, dtype='f8'))
    # RA_IVAR
    # DEC_IVAR
    targets.add_column(Column(name='FLUX_G', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FLUX_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FLUX_Z', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FLUX_W1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FLUX_W2', length=nobj, dtype='f4'))
    # FLUX_W3
    # FLUX_W4
    # FLUX_IVAR_G
    # FLUX_IVAR_R
    # FLUX_IVAR_Z
    # FLUX_IVAR_W1
    # FLUX_IVAR_W2
    # FLUX_IVAR_W3
    # FLUX_IVAR_W4
    targets.add_column(Column(name='SHAPEEXP_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEEXP_E1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEEXP_E2', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_E1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_E2', length=nobj, dtype='f4'))
    targets.add_column(Column(name='PSFDEPTH_G', length=nobj, dtype='f4'))
    targets.add_column(Column(name='PSFDEPTH_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='PSFDEPTH_Z', length=nobj, dtype='f4'))
    targets.add_column(Column(name='GALDEPTH_G', length=nobj, dtype='f4'))
    targets.add_column(Column(name='GALDEPTH_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='GALDEPTH_Z', length=nobj, dtype='f4'))

    targets.add_column(Column(name='MW_TRANSMISSION_G', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_Z', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_W1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_W2', length=nobj, dtype='f4'))

    targets.add_column(Column(name='TARGETID', length=nobj, dtype='int64'))
    targets.add_column(Column(name='DESI_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='BGS_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='MWS_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='HPXPIXEL', length=nobj, dtype='i8'))
    targets.add_column(Column(name='SUBPRIORITY', length=nobj, dtype='f8'))
    targets.add_column(Column(name='OBSCONDITIONS', length=nobj, dtype='i8'))


    return targets

def empty_truth_table(nobj=1):
    """Initialize the truth table for each mock object, with spectra.

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
    truth.add_column(Column(name='MAG', length=nobj, dtype='f4', data=np.zeros(nobj)+99))
    
    truth.add_column(Column(name='FLUX_G', length=nobj, dtype='f4'))
    truth.add_column(Column(name='FLUX_R', length=nobj, dtype='f4'))
    truth.add_column(Column(name='FLUX_Z', length=nobj, dtype='f4'))
    truth.add_column(Column(name='FLUX_W1', length=nobj, dtype='f4'))
    truth.add_column(Column(name='FLUX_W2', length=nobj, dtype='f4'))

    truth.add_column(Column(name='OIIFLUX', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='erg/(s*cm2)'))
    truth.add_column(Column(name='HBETAFLUX', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='erg/(s*cm2)'))

    truth.add_column(Column(name='TEFF', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='K'))
    truth.add_column(Column(name='LOGG', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='m/(s**2)'))
    truth.add_column(Column(name='FEH', length=nobj, dtype='f4', data=np.zeros(nobj)-1))

    return truth

def _get_spectra_onebrick(specargs):
    """Filler function for the multiprocessing."""
    return get_spectra_onebrick(*specargs)

def get_spectra_onebrick(target_name, mockformat, thisbrick, brick_info, Spectra, dust_dir,
                         select_targets_function, source_data, rand, log):
    """Wrapper function to generate spectra for all the objects on a single brick."""

    brickindx = np.where(brick_info['BRICKNAME'] == thisbrick)[0]
    onbrick = np.where(source_data['BRICKNAME'] == thisbrick)[0]
    nobj = len(onbrick)

    # Initialize the output targets and truth catalogs and populate them with
    # the quantities of interest.
    targets = empty_targets_table(nobj)
    truth = empty_truth_table(nobj)

    for key in ('RA', 'DEC', 'BRICKNAME'):
        targets[key][:] = source_data[key][onbrick]

    for key in ('BRICKID', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z'):
        targets[key][:] = brick_info[key][brickindx]

    # Assign unique OBJID values and reddenings.  See
    #   http://legacysurvey.org/dr4/catalogs
    targets['BRICK_OBJID'][:] = np.arange(nobj)

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
            targets[key][:] = source_data[key][onbrick]

    for key, source_key in zip( ['MOCKID', 'SEED', 'TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TRUESPECTYPE'],
                                ['MOCKID', 'SEED', 'TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TRUESPECTYPE'] ):
        if isinstance(source_data[source_key], np.ndarray):
            truth[key][:] = source_data[source_key][onbrick]
        else:
            truth[key][:] = np.repeat(source_data[source_key], nobj)

    # Sky targets are a special case without redshifts.
    if target_name == 'sky':
        select_targets_function(targets, truth)
        return [targets, truth]

    truth['TRUEZ'][:] = source_data['Z'][onbrick]

    # For FAINTSTAR targets, preselect stars that are going to pass target
    # selection cuts without actually generating spectra, in order to save
    # memory and time.
    if target_name == 'faintstar':
        if mockformat.lower() == 'galaxia':
            alldata = np.vstack((source_data['TEFF'][onbrick],
                                 source_data['LOGG'][onbrick],
                                 source_data['FEH'][onbrick])).T
            _, templateid = Spectra.tree.query('STAR', alldata)
            templateid = templateid.flatten()
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        normmag = 1E9 * 10**(-0.4 * source_data['MAG'][onbrick]) # nanomaggies

        for band, fluxkey in enumerate( ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2') ):
            truth[fluxkey][:] = getattr( Spectra.tree, 'star_{}'.format(fluxkey.lower()) )[templateid] * normmag
            targets[fluxkey][:] = truth[fluxkey][:] + \
              rand.normal(scale=onesigma[band], size=nobj)

        select_targets_function(targets, truth)

        keep = np.where(targets['DESI_TARGET'] != 0)[0]
        nobj = len(keep)

        # Temporary debugging plot.
        if False:
            import matplotlib.pyplot as plt
            gr1 = -2.5 * np.log10( truth['FLUX_G'] / truth['FLUX_R'] )
            rz1 = -2.5 * np.log10( truth['FLUX_R'] / truth['FLUX_Z'] )
            gr = -2.5 * np.log10( targets['FLUX_G'] / targets['FLUX_R'] )
            rz = -2.5 * np.log10( targets['FLUX_R'] / targets['FLUX_Z'] )
            plt.scatter(rz1, gr1, color='red', alpha=0.5, edgecolor='none', edgewidth=2, label='Noiseless Photometry')
            plt.scatter(rz1[keep], gr1[keep], color='red', edgecolor='k')
            plt.scatter(rz, gr, alpha=0.5, color='green', edgecolor='none', label='Noisy Photometry')
            plt.scatter(rz[keep], gr[keep], color='green', edgecolor='k', edgewidth=2)
            plt.xlim(-0.5, 2) ; plt.ylim(-0.5, 2)
            plt.legend(loc='upper left')
            plt.show()
            import pdb ; pdb.set_trace()
        
        if nobj == 0:
            log.warning('No {} targets identified!'.format(target_name.upper()))
            return [empty_targets_table(1), empty_truth_table(1), np.zeros( [1, len(Spectra.wave)], dtype='f4' )]
        else:
            onbrick = onbrick[keep]
            truth = truth[keep]
            targets = targets[keep]

    # Finally build the spectra and select targets.
    trueflux, meta = getattr(Spectra, target_name)(source_data, index=onbrick, mockformat=mockformat)

    for key in ('TEMPLATEID', 'MAG', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 
                'OIIFLUX', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'):
        truth[key][:] = meta[key]

    # Perturb the photometry based on the variance on this brick and apply
    # target selection.
    for band, fluxkey in enumerate( ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2') ):
        targets[fluxkey][:] = truth[fluxkey][:] + rand.normal(scale=onesigma[band], size=nobj)

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
        gr1 = -2.5 * np.log10( truth['FLUX_G'] / truth['FLUX_R'] )
        rz1 = -2.5 * np.log10( truth['FLUX_R'] / truth['FLUX_Z'] )
        gr = -2.5 * np.log10( targets['FLUX_G'] / targets['FLUX_R'] )
        rz = -2.5 * np.log10( targets['FLUX_R'] / targets['FLUX_Z'] )
        
        fig, ax = plt.subplots()
        ax.scatter(rz1, gr1, color='red', label='Noiseless Photometry')
        ax.scatter(rz, gr, alpha=0.5, color='green', label='Noisy Photometry')
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(-0.5, 2)
        elg_colorbox(ax)
        ax.legend(loc='upper left')
        plt.show()
        import pdb ; pdb.set_trace()
        
    if 'BOSS_STD' in source_data.keys():
        boss_std = source_data['BOSS_STD'][onbrick]
    else:
        boss_std = None
    select_targets_function(targets, truth, boss_std=boss_std)

    keep = np.where(targets['DESI_TARGET'] != 0)[0]
    nobj = len(keep)

    if nobj == 0:
        log.warning('No {} targets identified!'.format(target_name.upper()))
    else:
        log.debug('Selected {} {}s on brick {}.'.format(nobj, target_name.upper(), thisbrick))
        targets = targets[keep]
        truth = truth[keep]
        trueflux = trueflux[keep, :]

    return [targets, truth, trueflux]

#from memory_profiler import profile
#@profile
def targets_truth(params, output_dir='.', realtargets=None, seed=None, verbose=False,
                  bricksize=0.25, nproc=1, nside=16, healpixels=None):
    """Generate a catalog of targets, spectra, and the corresponding "truth" catalog
    (with, e.g., the true redshift) for use in simulations.

    Args:
        params : dict
            Source parameters.
        output_dir : str
            Output directory (default '.').
        realtargets : astropy.table
            Real target catalog table, e.g. from DR3 (deprecated!).
        bricksize : float
            Brick size for assigning bricknames, which should match the imaging
            team value (default 0.25 deg).
        nproc : int
            Number of parallel processes to use (default 1).
        nside : int
            Healpix resolution corresponding to healpixels (default 16).
        healpixels : numpy.ndarray or int
            Restrict the sample of mock targets analyzed to those lying inside
            this set (array) of healpix pixels.

    Returns:
        A variety of fits files are written to output_dir.

    """
    import healpy as hp
    from time import time

    from astropy.io import fits

    from desiutil import depend
    from desispec.io.util import fitsheader, write_bintable
    from desitarget.mock.selection import SelectTargets
    from desitarget.mock.spectra import MockSpectra
    from desitarget.internal import sharedmem
    from desimodel.footprint import radec2pix
    from desitarget import obsconditions

    if verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    if params is None:
        log.fatal('Required params input not given!')
        raise ValueError

    # Initialize the random seed
    rand = np.random.RandomState(seed)

    # Create the output directories and clean them up if necessary.
    if os.path.exists(output_dir):
        if os.listdir(output_dir):
            log.warning('Output directory {} is not empty.'.format(output_dir))
    else:
        log.info('Creating directory {}'.format(output_dir))
        os.makedirs(output_dir)
        
    log.info('Writing to output directory {}'.format(output_dir))
    print()

    # Default set of healpixels is the whole sky (yikes!)
    if healpixels is None:
        healpixels = np.arange(hp.nside2npix(nside))

    areaperpix = hp.nside2pixarea(nside, degrees=True)
    skyarea = len(healpixels) * areaperpix
    log.info('Grouping into {} healpixel(s) (nside = {}, {:.3f} deg2/pixel) spanning {:.3f} deg2.'.format(
        len(healpixels), nside, areaperpix, skyarea))

    brick_info = BrickInfo(random_state=rand, dust_dir=params['dust_dir'], bricksize=bricksize,
                           decals_brick_info=params['decals_brick_info'], log=log,
                           target_names=list(params['sources'].keys())).build_brickinfo()

    # Initialize the Classes used to assign spectra and select targets.  Note:
    # The default wavelength array gets initialized here, too.
    log.info('Initializing the MockSpectra and SelectTargets Classes.')
    Spectra = MockSpectra(rand=rand, verbose=verbose, nproc=nproc)
    Selection = SelectTargets(logger=log, rand=rand,
                              brick_info=brick_info)
    print()

    current_pixel_ID = 0 
    # Loop over each source / object type.
    alltargets = list()
    alltruth = list()
    alltrueflux = list()

    for source_name in sorted(params['sources'].keys()):
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
        source_data = mockread_function(mock_dir_name, target_name, rand=rand, bricksize=bricksize,
                                        magcut=magcut, nproc=nproc, lya=lya,
                                        healpixels=healpixels, nside=nside)

        # If there are no sources, keep going.
        if not bool(source_data):
            continue

        selection_function = '{}_select'.format(target_name.lower())
        select_targets_function = getattr(Selection, selection_function)

        # Assign spectra by parallel-processing the bricks.
        brickname = source_data['BRICKNAME']
        unique_bricks = sorted(set(brickname))

        # Quickly check that all the brick info is here.
        if False:
            for thisbrick in unique_bricks:
                brickindx = np.where(brick_info['BRICKNAME'] == thisbrick)[0]
                if (len(brickindx) != 1):
                    log.fatal('One or too many matching brick(s) {}! This should not happen...'.format(thisbrick))
                    ww = np.where( (brick_info['RA'] > source_data['RA'].min()) *
                                   (brick_info['RA'] < source_data['RA'].max()) *
                                   (brick_info['DEC'] > source_data['DEC'].min()) *
                                   (brick_info['DEC'] < source_data['DEC'].max()) )[0]
                    print(brick_info['BRICKNAME'][ww])
                    import matplotlib.pyplot as plt
                    plt.scatter(source_data['RA'], source_data['DEC'])
                    plt.scatter(brick_info['RA'], brick_info['DEC'])
                    plt.xlim(source_data['RA'].min()-0.2, source_data['RA'].max()+0.2)
                    plt.ylim(source_data['DEC'].min()-0.2, source_data['DEC'].max()+0.3)
                    plt.show()
                    import pdb ; pdb.set_trace()
        
        log.info('Assigned {} {}s to {} unique {}x{} deg2 bricks.'.format(
            len(brickname), source_name, len(unique_bricks), bricksize, bricksize))

        nbrick = np.zeros((), dtype='i8')
        t0 = time()
        def _update_spectra_status(result):
            if nbrick % 10 == 0 and nbrick > 0:
                rate = (time() - t0) / nbrick
                log.info('{} bricks; {:.1f} sec / brick'.format(nbrick, rate))
            nbrick[...] += 1    # this is an in-place modification
            return result

        specargs = list()
        for thisbrick in unique_bricks:
            specargs.append( (target_name.lower(), mockformat, thisbrick, brick_info,
                              Spectra, params['dust_dir'], select_targets_function,
                              source_data, rand, log) )

        if nproc > 1:
            pool = sharedmem.MapReduce(np=nproc)
            with pool:
                out = pool.map(_get_spectra_onebrick, specargs,
                               reduce=_update_spectra_status)
        else:
            out = list()
            for ii in range(len(unique_bricks)):
                out.append( _update_spectra_status( _get_spectra_onebrick(specargs[ii]) ) )

        del source_data # memory clean-up

        # Unpack the results removing any possible bricks without targets.
        out = list(zip(*out))

        # SKY are a special case of targets without truth or trueflux.
        targets = vstack(out[0])
        truth = vstack(out[1])
        if target_name.upper() != 'SKY':
            trueflux = np.concatenate(out[2])
        
            keep = np.where(targets['DESI_TARGET'] != 0)[0]
            if len(keep) == 0:
                continue

            targets = targets[keep]
            truth = truth[keep]
            trueflux = trueflux[keep, :]
        del out

        # Finally downsample based on the desired number density.
        if 'density' in params['sources'][source_name].keys():
            density = params['sources'][source_name]['density']
            if target_name != 'QSO':
                log.info('Downsampling {}s to desired target density of {} targets/deg2.'.format(target_name, density))
                
            if target_name == 'QSO':
                # Distinguish between the Lyman-alpha and tracer QSOs
                if 'LYA' in params['sources'][source_name].keys():
                    density_lya = params['sources'][source_name]['LYA']['density']
                    zcut = params['sources'][source_name]['LYA']['zcut']
                    tracer = truth['TRUEZ'] < zcut
                    lya = truth['TRUEZ'] >= zcut
                    if len(tracer) > 0:
                        log.info('Downsampling tracer {}s to desired target density of {} targets/deg2.'.format(target_name, density))
                        Selection.density_select(targets, truth, source_name=source_name, target_name=target_name,
                                                 density=density, subset=tracer)
                        print()
                    if len(lya) > 0:
                        log.info('Downsampling Lya {}s to desired target density of {} targets/deg2.'.format(target_name, density_lya))
                        Selection.density_select(targets, truth, source_name=source_name, target_name=target_name,
                                                 density=density_lya, subset=lya)

                else:
                    Selection.density_select(targets, truth, source_name=source_name,
                                             target_name=target_name, density=density)
                    
            else:
                Selection.density_select(targets, truth, source_name=source_name,
                                         target_name=target_name, density=density)            

            keep = np.where(targets['DESI_TARGET'] != 0)[0]
            #keep = np.where( np.any( ((targets['DESI_TARGET'] != 0), (truth['CONTAM_TARGET'] != 0)), axis=0) )[0]

            if len(keep) == 0:
                log.warning('All {} targets rejected!'.format(target_name))
            else:
                targets = targets[keep]
                truth = truth[keep]
                if target_name.upper() != 'SKY':
                    trueflux = trueflux[keep, :]

        #rewrite the target ID to be sure that it is unique in this brick
        if len(targets):
            new_pixel_ID = current_pixel_ID + len(targets)
            targets['BRICK_OBJID'][:]= np.arange(current_pixel_ID,new_pixel_ID)
            current_pixel_ID = new_pixel_ID
            
        if target_name.upper() == 'SKY':
            skytruth = truth.copy()
            skytargets = targets.copy()
        else:
            alltargets.append(targets)
            alltruth.append(truth)
            alltrueflux.append(trueflux)
        print()

    del brick_info # memory clean-up

    # Consolidate across all the mocks.  Note that the code quits if alltargets
    #is zero-length, even if skytargets is non-zero length. In other words, if
    #the parameter file only contains SKY the code will (wrongly) quit anyway.
    if len(alltargets) == 0:
        log.info('No targets; all done.')
        return

    
    targets = vstack(alltargets)
    truth = vstack(alltruth)
    trueflux = np.concatenate(alltrueflux)

    # Finally downsample contaminants.  The way this is being done isn't ideal
    # because in principle an object could be a contaminant in one target class
    # (and be tossed) but be a contaminant for another target class and be kept.
    # But I think this is mostly OK.
    for source_name in sorted(params['sources'].keys()):
        target_name = params['sources'][source_name]['target_name'] # Target type (e.g., ELG)
        
        if 'contam' in params['sources'][source_name].keys():
            log.info('Downsampling {} contaminant(s) to desired target density.'.format(target_name))
            contam = params['sources'][source_name]['contam']

            Selection.contaminants_select(targets, truth, source_name=source_name,
                                          target_name=target_name, contam=contam)
            
            keep = np.where(targets['DESI_TARGET'] != 0)[0]
            if len(keep) == 0:
                log.warning('All {} contaminants rejected!'.format(target_name))
            else:
                targets = targets[keep]
                truth = truth[keep]
                trueflux = trueflux[keep, :]

    # Write out the fileid-->filename mapping.  This doesn't work right now.
    #map_fileid_filename = fileid_filename(source_data_all, output_dir, log)

    # Deprecated:  add mock shapes and fluxes from the real target catalog.
    if realtargets is not None:
        add_mock_shapes_and_fluxes(targets, realtargets, random_state=rand)

    # Finally assign TARGETIDs and subpriorities.
    ntarget = len(targets)
    try:
        nsky = len(skytargets)
    except:
        nsky = 0

    #compute target healpixel number
    targpix = radec2pix(nside, targets['RA'], targets['DEC'])
    targetid = encode_targetid(objid=targets['BRICK_OBJID'],
                               brickid=targpix, mock=1)
    truth['TARGETID'][:] = targetid
    targets['TARGETID'][:] = targetid
    del targetid

    
    
    if nsky > 0:
        skypix = radec2pix(nside, skytargets['RA'], skytargets['DEC'])
        skytargets['TARGETID'][:] = encode_targetid(objid=skytargets['BRICK_OBJID'],
                                                    brickid=skypix, mock=1, sky=1)

    subpriority = rand.uniform(0.0, 1.0, size=ntarget + nsky)
    targets['SUBPRIORITY'][:] = subpriority[:ntarget]
    if nsky > 0:
        skytargets['SUBPRIORITY'][:] = subpriority[ntarget:ntarget+nsky]

    # Write the final catalogs out by healpixel.
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
    depend.setdep(targetshdr, 'HPXNSIDE', nside)
    depend.setdep(targetshdr, 'HPXNEST', True)

    targets['HPXPIXEL'][:] = targpix

    if nsky > 0:
        skypix = radec2pix(nside, skytargets['RA'], skytargets['DEC'])
        skytargets['HPXPIXEL'][:] = skypix

    for pixnum in healpixels:
        # healsuffix = '{}-{}.fits'.format(nside, pixnum)
        outdir = mockio.get_healpix_dir(nside, pixnum, basedir=output_dir)
        os.makedirs(outdir, exist_ok=True)

        # Write out the sky catalog.
        if nsky > 0:
            isky = pixnum == skypix
            if np.count_nonzero(isky) > 0:
                # skyfile = os.path.join(output_dir, 'sky-{}.fits'.format(healsuffix))
                skyfile = mockio.findfile('sky', nside, pixnum, basedir=output_dir)
            
                log.info('Writing {} SKY targets to {}'.format(np.sum(isky), skyfile))
                write_bintable(skyfile+'.tmp', skytargets[isky], extname='SKY',
                               header=targetshdr, clobber=True)
                os.rename(skyfile+'.tmp', skyfile)

        # Select targets in this pixel
        inpixel     = (pixnum == targpix)
        npixtargets = np.count_nonzero(inpixel)

        # Write out the dark- and bright-time standard stars.
        for stdsuffix in ('dark', 'bright'):
            # Standards taken from targets

            stdfile = mockio.findfile('standards-{}'.format(stdsuffix), nside, pixnum, basedir=output_dir)

            # Select all targets with each of the standard star types and
            # matching obsconditions
            if stdsuffix == 'dark':
                dark_std_mask = desi_mask.STD_FSTAR | desi_mask.STD_WD
                istd = inpixel & ((targets['DESI_TARGET'] & dark_std_mask) != 0)
            elif stdsuffix == 'bright':
                bright_std_mask = desi_mask.STD_BRIGHT | desi_mask.STD_WD
                istd = inpixel & ((targets['DESI_TARGET'] & bright_std_mask) != 0)
            else:
                raise RuntimeError('unknown stdsuffix {}'.format(stdsuffix))

            if np.count_nonzero(istd) > 0:
                log.info('Writing {} {} standards on healpix {} to {}'.format(np.sum(istd), stdsuffix, pixnum, stdfile))
                write_bintable(stdfile+'.tmp', targets[istd], extname='STD',
                               header=targetshdr, clobber=True)
                os.rename(stdfile+'.tmp', stdfile)
            else:
                log.error('No {} standards on healpix {}, {} not written.'.format(stdsuffix, pixnum, stdfile))

        # Finally write out the rest of the targets.
        targetsfile = mockio.findfile('targets', nside, pixnum, basedir=output_dir)
        truthfile = mockio.findfile('truth', nside, pixnum, basedir=output_dir)

        if npixtargets > 0:
            log.info('Writing {} targets to {}'.format(npixtargets, targetsfile))
            targets.meta['EXTNAME'] = 'TARGETS'
            write_bintable(targetsfile+'.tmp', targets[inpixel], extname='TARGETS',
                           header=targetshdr, clobber=True)
            os.rename(targetsfile+'.tmp', targetsfile)

            log.info('Writing {}'.format(truthfile))
            hx = fits.HDUList()
            hdu = fits.convenience.table_to_hdu(truth[inpixel])
            hdu.header['EXTNAME'] = 'TRUTH'
            hx.append(hdu)

            hdu = fits.ImageHDU(Spectra.wave.astype(np.float32), name='WAVE', header=truthhdr)
            hdu.header['BUNIT'] = 'Angstrom'
            hdu.header['AIRORVAC'] = 'vac'
            hx.append(hdu)

            hdu = fits.ImageHDU(trueflux[inpixel, :].astype(np.float32), name='FLUX')
            hdu.header['BUNIT'] = '1e-17 erg/s/cm2/Angstrom'
            hx.append(hdu)

            try:
                hx.writeto(truthfile+'.tmp', overwrite=True)
            except:
                hx.writeto(truthfile+'.tmp', clobber=True)
            os.rename(truthfile+'.tmp', truthfile)

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
        assert len(vals) == len(data)
        
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
        log = get_logger()
        out_mtl = os.path.join(outdir, 'mtl.fits')
        log.info('Generating merged target list {}'.format(out_mtl))
        targets = fitsio.read(outdir+'/targets.fits')
        mtl = mtl.make_mtl(targets)
        tmpout = out_mtl+'.tmp'
        mtl.meta['EXTNAME'] = 'MTL'
        mtl.write(tmpout, overwrite=True, format='fits')
        os.rename(tmpout, out_mtl)


def initialize(params, verbose=False, seed=1, output_dir="./", nproc=1, nside=16, healpixels=None):
    """Initializes variables to prepare mock target generation.
    Args:
        params : dict
            Source parameters.
        seed: int
            Seed for the random number generatios
        output_dir : str
            Output directory (default '.').
        nproc : int
            Number of parallel processes to use (default 1).
        nside : int
            Healpix resolution corresponding to healpixels (default 16).
        healpixels : numpy.ndarray or int
            Restrict the sample of mock targets analyzed to those lying inside
            this set (array) of healpix pixels. Default (None)
        
    Returns:
        log: logger object
        rand: numpy.random.RandomState
            seed: int
            Seed used for the random number generator
        magnitudes: desitarget.mock.Magnitudes
            This object contains the information to assign magnitudes to each kind of targets.
        selection: desitarget.mock.Selection 
            This object contains the information from the configuration file
            used to start the construction of the mock target files.
        output_dir: str
            Directory where the outputs are written.
        healpixels : numpy.ndarray or int
            Restrict the sample of mock targets analyzed to those lying inside
            this set (array) of healpix pixels. Default (None).
            If the input is None, the output are the pixels on the whole sky.
    """
    from desitarget.mock.spectra import MockMagnitudes
    from desitarget.mock.selection import SelectTargets
    import healpy as hp

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
    skyarea = len(healpixels) * areaperpix
    log.info('Grouping into {} healpixel(s) (nside = {}, {:.3f} deg2/pixel) spanning {:.3f} deg2.'.format(
        len(healpixels), nside, areaperpix, skyarea))

    # Initialize the Classes used to assign spectra and select targets.  Note:
    # The default wavelength array gets initialized here, too.
    log.info('Initializing the MockMagnitudes and SelectTargets Classes.')
    magnitudes = MockMagnitudes(rand=rand, verbose=verbose, nproc=nproc)
    selection = SelectTargets(logger=log, rand=rand)
    
    print()
    current_pixel_ID = 0
    # Loop over each source / object type.
    alltargets = list()
    alltruth = list()
    
    return log, rand, magnitudes, selection, healpixels

def read_catalog(source_name, params, log, rand=None, nproc=1, healpixels=None, nside=16, in_desi=True):
    """Reads a mock file.
    Args.
        source_name: string
            Name of the target being processesed, i.e. "QSO"
        params: dictionary
            Gathers the information of the input configuration file.
        log: logger object
        nproc: int
            Number of processors to be used for reading.
        rand: numpy.random.RandomState
        healpixels: numpy.array
            List of healpixels to process. The mocks are cut to match these pixels.
        nside: int
            nside for healpix
        in_desi: boolean
            Decides whether the targets will be trimmed to be inside the DESI footprint.
            
        Magnitudes: desitarget.mock.Magnitudes
            This object contains the information to assign magnitudes to each kind of targets.
        source_data: dictionary
            This corresponds to the "raw" data coming directly from the mock file.

        mockformat: string
            Format of the mock files used to read the data.
    Output.
        source_data: dictionary
            Raw information from the input mock file.

    """
    import desimodel.io
    import desimodel.footprint
    
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

    #returns only the points that are in DESI footprint
    if bool(source_data):
        if in_desi:
            n_obj = len(source_data['RA'])
            tiles = desimodel.io.load_tiles()
            if n_obj>0:
                indesi = desimodel.footprint.is_point_in_desi(tiles, source_data['RA'], source_data['DEC'])
                for k in source_data.keys():
                    if (n_obj == len(source_data[k])) and (type(source_data[k]) is np.ndarray):
                        source_data[k] = source_data[k][indesi]
    return source_data

def get_magnitudes_onepixel(Magnitudes, source_data, target_name, mockformat, 
                            rand, log, nside, healpix_id, dust_dir):
    """Assigns magnitudes to set of targets and truth dictionaries located on the pixel healpix_id.
    Args.
        Magnitudes: desitarget.mock.Magnitudes
            This object contains the information to assign magnitudes to each kind of targets.
        source_data: dictionary
            This corresponds to the "raw" data coming directly from the mock file.
        target_name: string
            Name of the target being processesed, i.e. "QSO"
        mockformat: string
            Format of the mock files used to read the data.
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
    Output.
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
    meta = getattr(Magnitudes, target_name.lower())(source_data, index=onpix, mockformat=mockformat)

    for key in ('TEMPLATEID', 'MAG', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 
                'OIIFLUX', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'):
        truth[key][:] = meta[key]

    for band, fluxkey in enumerate( ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2') ):
        targets[fluxkey][:] = truth[fluxkey][:] #+ rand.normal(scale=onesigma[band], size=nobj)

    return [targets, truth]

def target_selection(Selection, target_name, targets, truth, nside, healpix_id, seed, rand, log, output_dir):
    """Applies target selection functions to a set of targets and truth tables.
    Args.
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
    Output.
         targets: astropy.table
            Final set of targets after target selection. 
        truth: astropy.table
            Corresponding Truth to Targets
    """
    

    selection_function = '{}_select'.format(target_name.lower())
    select_targets_function = getattr(Selection, selection_function)

    select_targets_function(targets, truth)
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
    """Estimates the number density of points with positions RA, DEC
    Args.
        ra: numpy.array
            RA positions in degrees.
        dec: numpy.array
            DEC positions in degrees.
    Output. 
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

def downsample_pixel(density, zcut, target_name, targets, truth, nside, healpix_id, seed, rand, log,output_dir):
    """Reduces the number of targets to match a desired number density.
    Args.
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
    Output.
        Updated versions of the tables:
            targets
            truth.
    Example.
        If density = [300,400] and zcut=[0.0,2.0, 3.0] the first number density cap of 300
        will be applied to all targets with redshifts 0.0 < z < 2.0, while the second cap
        of 400 will be applied to all targets with redshifts 2.0 < z < 3.0
    """
    import healpy as hp
    
    n_cuts = len(density)
    n_obj = len(targets)
    

    r = rand.uniform(0.0, 1.0, size=n_obj)
    keep = r < 1.0
    for i in range(n_cuts):
        in_z = (truth['TRUEZ'] > zcut[i]) & (truth['TRUEZ']<zcut[i+1])
        input_density = estimate_number_density(targets['RA'][in_z], targets['DEC'][in_z])
        if density[i] < input_density:
            frac_keep = density[i]/input_density
            keep = keep & (r<frac_keep) & in_z
            log.info('Downsampling for {}. Going from {} to {} obs/deg^2'.format(target_name, input_density, density[i]))
        else:
            log.info('Cannot go from {} to {} obs/deg^2 for object {}'.format(input_density, density[i], target_name))
            
    targets = targets[keep]
    truth = truth[keep]
    return targets, truth

    
def finish_catalog(targets, truth, skytargets, skytruth, nside, healpix_id, seed, rand, log, output_dir):
    """Adds TARGETID, SUBPRIORITY and HPXPIXEL to targets.
    Args.
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
    Output.
        Updated versions of:
        targets
        truth
        skytargets
        skytruth
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

def write_to_disk(targets, truth, skytargets, skytruth, nside, healpix_id, seed, rand, log, output_dir):
    """Writes targets to disk.
    Args.
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
    Output.
        None. 
        Files are only written to disk.
    """
    from desispec.io.util import fitsheader, write_bintable
    from desiutil import depend
    from astropy.io import fits
    
    
    n_obj = len(targets)
    n_sky = len(skytargets)
    
    # Write the final catalogs
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
    depend.setdep(targetshdr, 'HPXNSIDE', nside)
    depend.setdep(targetshdr, 'HPXNEST', True)

        
    outdir = mockio.get_healpix_dir(nside, healpix_id, basedir=output_dir)
    os.makedirs(outdir, exist_ok=True)

    # Write out the sky catalog.
    if n_sky > 0:
        skyfile = mockio.findfile('sky', nside, healpix_id, basedir=output_dir)
        log.info('Writing {} SKY targets to {}'.format(n_sky, skyfile))
        write_bintable(skyfile+'.tmp', skytargets, extname='SKY',
                               header=targetshdr, clobber=True)
        os.rename(skyfile+'.tmp', skyfile)

    if n_obj > 0:
    # Write out the dark- and bright-time standard stars.
        for stdsuffix, stdbit in zip(('dark', 'bright'), ('STD_FSTAR', 'STD_BRIGHT')):
            stdfile = mockio.findfile('standards-{}'.format(stdsuffix), nside, healpix_id, basedir=output_dir)
            istd   = (((targets['DESI_TARGET'] & desi_mask.mask(stdbit)) | 
                   (targets['DESI_TARGET'] & desi_mask.mask('STD_WD')) ) != 0)

            if np.count_nonzero(istd) > 0:
                log.info('Writing {} {} standards on healpix {} to {}'.format(np.sum(istd), stdsuffix, healpix_id, stdfile))
                write_bintable(stdfile+'.tmp', targets[istd], extname='STD',
                               header=targetshdr, clobber=True)
                os.rename(stdfile+'.tmp', stdfile)
            else:
                log.info('No {} standards on healpix {}, {} not written.'.format(stdsuffix, healpix_id, stdfile))

        # Finally write out the rest of the targets.
        targetsfile = mockio.findfile('targets', nside, healpix_id, basedir=output_dir)
        truthfile = mockio.findfile('truth', nside, healpix_id, basedir=output_dir)

   
        log.info('Writing {} targets to {}'.format(n_obj, targetsfile))
        targets.meta['EXTNAME'] = 'TARGETS'
        write_bintable(targetsfile+'.tmp', targets, extname='TARGETS',
                          header=targetshdr, clobber=True)
        os.rename(targetsfile+'.tmp', targetsfile)

        log.info('Writing {}'.format(truthfile))
        hx = fits.HDUList()
        hdu = fits.convenience.table_to_hdu(truth)
        hdu.header['EXTNAME'] = 'TRUTH'
        hx.append(hdu)

        try:
            hx.writeto(truthfile+'.tmp', overwrite=True)
        except:
            hx.writeto(truthfile+'.tmp', clobber=True)
        os.rename(truthfile+'.tmp', truthfile)


def targets_truth_no_spectra(params, seed=1, output_dir="./", nproc=1, nside=16, healpixels=None, verbose=False, dust_dir="./"):
    """Generate a catalog of targets and the corresponding truth catalog.
    
    Inputs.
    """
    log, rand, Magnitudes, Selection, healpixels = initialize(params, verbose = verbose, 
                                       seed = seed, 
                                       output_dir = output_dir, 
                                       nproc = nproc, 
                                       healpixels = healpixels)
    
    # Loop over each source / object type.
   
    
    for healpix in healpixels:
        alltargets = list()
        alltruth = list()
        allskytargets = list()
        allskytruth = list()
          
        for source_name in sorted(params['sources'].keys()):
            
            #Read the data
            mockformat = params['sources'][source_name]['format']
            log.info('Reading  source : {}'.format(source_name))
            source_data = read_catalog(source_name, params, log, 
                                    rand=rand, nproc=nproc, healpixels=healpix, nside=nside)
        
            # If there are no sources, keep going.
            if not bool(source_data):
                continue
                
            #Initialize variables for density subsampling
            if 'density' in params['sources'][source_name].keys():
                density = [params['sources'][source_name]['density']]
                zcut=[-1000,1000]
            
            # assign magnitudes for targets in that pixel
            pixel_results = get_magnitudes_onepixel(Magnitudes, source_data, source_name, mockformat, rand, log, 
                                        nside, healpix, dust_dir=params['dust_dir'])
            
            targets = pixel_results[0]
            truth = pixel_results[1]
            
            # Make target selection and downsample number density if necessary. SKY is an special case.
            if source_name.upper() == 'SKY':
                if 'density' in params['sources'][source_name].keys():
                    targets, truth = downsample_pixel(density, zcut, source_name, targets, truth,
                                                    nside, healpix, seed, rand, log, output_dir)
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
                                                    nside, healpix, seed, rand, log, output_dir)
                                                    
                    
                if len(targets)>0:
                    alltargets.append(targets)
                    alltruth.append(truth)
               
        if len(alltargets)==0 and len(allskytargets)==0:
            continue
        
        #Merge all sources
        if len(alltargets):
            targets = vstack(alltargets)
            truth = vstack(alltruth)
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
        targets, truth, skytargets, skytruth = finish_catalog(targets, truth, skytargets, skytruth,
                                                              nside,healpix, seed, rand, log, output_dir)
        #write the results
        write_to_disk(targets, truth, skytargets, skytruth,  
                          nside, healpix, seed, rand, log, output_dir)
    return
