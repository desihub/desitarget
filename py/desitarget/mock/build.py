# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.build
=====================

Build a truth catalog (including spectra) and a targets catalog for the mocks.

time python -m cProfile -o mock.dat /usr/local/repos/desihub/desitarget/bin/select_mock_targets -c mock_moustakas.yaml -s 333 --nproc 1 --output_dir proftest
pyprof2calltree -k -i mock.dat &

/usr/bin/time -l select_mock_targets -c qatargets_input.yaml --output_dir new --nproc 4 --seed 111 --verbose --clobber

"""
from __future__ import (absolute_import, division, print_function)

import os

import numpy as np
from astropy.table import Table, Column, vstack

from desiutil.log import get_logger, DEBUG
from desitarget import desi_mask, bgs_mask, mws_mask, contam_mask

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
    def __init__(self, random_state=None, dust_dir=None, bounds=(0.0, 360.0, -90.0, 90.0),
                 bricksize=0.25, decals_brick_info=None, target_names=None, log=None):
        """Initialize the class.

        Args:
          random_state : random number generator object
          dust_dir : path where the E(B-V) maps are stored
          bounds : brick boundaries
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
        self.bounds = bounds
        self.bricksize = bricksize
        self.decals_brick_info = decals_brick_info
        self.target_names = target_names

    def generate_brick_info(self):
        """Generate the brick dictionary in the region (min_ra, max_ra, min_dec,
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

        i_rows = np.where(((B._edges_dec+self.bricksize) >= min_dec) & ((B._edges_dec-self.bricksize) <= max_dec))[0]
        for i_row in i_rows:
            j_col_min = int((min_ra)/360 * B._ncol_per_row[i_row])
            j_col_max = int((max_ra)/360 * B._ncol_per_row[i_row])

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

        self.log.info('Generating brick information for {} brick(s) with boundaries RA={:g}, {:g}, Dec={:g}, {:g} and bricksize {:g} deg.'.\
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
            brick_info(Dictionary). Containts at least the following keys:
                RA (float): numpy array of RA positions
                DEC (float): numpy array of Dec positions

        Returns:
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
        import yaml
        filein = open(os.getenv('DESIMODEL')+'/data/targets/targets.dat')
        td = yaml.load(filein)
        target_desimodel = {}
        for t in td.keys():
            if 'ntarget' in t.upper():
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
    MWS_TARGET, SUBPRIORITY and OBSCONDITIONS.  Everything else is gravy.

    """
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
    targets.add_column(Column(name='BRICKNAME', length=nobj, dtype='U10'))
    targets.add_column(Column(name='DECAM_FLUX', shape=(6,), length=nobj, dtype='f4'))
    targets.add_column(Column(name='WISE_FLUX', shape=(2,), length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEEXP_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEEXP_E1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEEXP_E2', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_E1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_E2', length=nobj, dtype='f4'))
    targets.add_column(Column(name='DECAM_DEPTH', shape=(6,), length=nobj,
                              data=np.zeros((nobj, 6)), dtype='f4'))
    targets.add_column(Column(name='DECAM_GALDEPTH', shape=(6,), length=nobj,
                              data=np.zeros((nobj, 6)), dtype='f4'))
    targets.add_column(Column(name='EBV', length=nobj, dtype='f4'))

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

def get_spectra_onebrick(target_name, mockformat, thisbrick, brick_info, Spectra,
                         select_targets_function, source_data, rand, log):
    """Wrapper function to generate spectra for all the objects on a single brick."""

    brickindx = np.where(brick_info['BRICKNAME'] == thisbrick)[0]
    onbrick = np.where(source_data['BRICKNAME'] == thisbrick)[0]
    nobj = len(onbrick)

    brickarea = brick_info['BRICKAREA'][brickindx][0]

    # Initialize the output targets and truth catalogs and populate them with
    # the quantities of interest.
    targets = empty_targets_table(nobj)
    truth = empty_truth_table(nobj)

    for key in ('RA', 'DEC', 'BRICKNAME'):
        targets[key] = source_data[key][onbrick]

    for band, depthkey in zip((1, 2, 4), ('DEPTH_G', 'DEPTH_R', 'DEPTH_Z')):
        targets['DECAM_DEPTH'][:, band] = brick_info[depthkey][brickindx]
    for band, depthkey in zip((1, 2, 4), ('GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z')):
        targets['DECAM_GALDEPTH'][:, band] = brick_info[depthkey][brickindx]
    targets['EBV'] = brick_info['EBV'][brickindx]

    # Use the point-source depth for point sources, although this should really
    # be tied to the morphology.
    if 'star' in target_name or 'qso' in target_name:
        depthkey = 'DECAM_DEPTH'
    else:
        depthkey = 'DECAM_GALDEPTH'
    with np.errstate(divide='ignore'):                        
        #decam_onesigma = 1.0 / np.sqrt(targets[depthkey][0, :]) # grab the first object
        decam_onesigma = 10**(0.4 * (22.5 - targets[depthkey][0, :]) ) / 5

    # Hack! Assume a constant 5-sigma depth of g=24.7, r=23.9, and z=23.0 for
    #   all bricks:  http://legacysurvey.org/dr3/description
    decam_onesigma = 10**(0.4 * (22.5 - np.array([0.0, 24.7, 23.9, 0.0, 23.0, 0.0])) ) / 5

    # Hack! Assume a constant depth (W1=22.3-->1.2 nanomaggies, W2=23.8-->0.3
    # nanomaggies) in the WISE bands for now.
    wise_onesigma = 10**(0.4 * (22.5 - np.array([22.3, 23.8])) )

    # Add shapes and sizes.
    if 'SHAPEEXP_R' in source_data.keys(): # not all target types have shape information
        for key in ('SHAPEEXP_R', 'SHAPEEXP_E1', 'SHAPEEXP_E2',
                    'SHAPEDEV_R', 'SHAPEDEV_E1', 'SHAPEDEV_E2'):
            targets[key] = source_data[key][onbrick]

    for key, source_key in zip( ['MOCKID', 'SEED', 'TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TRUESPECTYPE'],
                                ['MOCKID', 'SEED', 'TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TRUESPECTYPE'] ):
        if isinstance(source_data[source_key], np.ndarray):
            truth[key] = source_data[source_key][onbrick]
        else:
            truth[key] = np.repeat(source_data[source_key], nobj)

    # Sky targets are a special case without redshifts.
    if target_name == 'sky':
        select_targets_function(targets, truth)
        return [targets, truth]

    truth['TRUEZ'] = source_data['Z'][onbrick]

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

        for band in (0, 1):
            truth['WISE_FLUX'][:, band] = Spectra.tree.star_wise_flux[templateid, band] * normmag
            targets['WISE_FLUX'][:, band] = truth['WISE_FLUX'][:, band] + \
              rand.normal(scale=wise_onesigma[band], size=nobj)
            
        for band in (1, 2, 4):
            truth['DECAM_FLUX'][:, band] = Spectra.tree.star_decam_flux[templateid, band] * normmag
            targets['DECAM_FLUX'][:, band] = truth['DECAM_FLUX'][:, band] + \
              rand.normal(scale=decam_onesigma[band], size=nobj)

        select_targets_function(targets, truth)

        keep = np.where(targets['DESI_TARGET'] != 0)[0]
        nobj = len(keep)

        # Temporary debugging plot.
        if False:
            import matplotlib.pyplot as plt
            gr1 = -2.5 * np.log10( truth['DECAM_FLUX'][:, 1] / truth['DECAM_FLUX'][:, 2] )
            rz1 = -2.5 * np.log10( truth['DECAM_FLUX'][:, 2] / truth['DECAM_FLUX'][:, 4] )
            gr = -2.5 * np.log10( targets['DECAM_FLUX'][:, 1] / targets['DECAM_FLUX'][:, 2] )
            rz = -2.5 * np.log10( targets['DECAM_FLUX'][:, 2] / targets['DECAM_FLUX'][:, 4] )
            plt.scatter(rz1, gr1, color='red', alpha=0.5, edgecolor='none')
            plt.scatter(rz1[keep], gr1[keep], color='red', edgecolor='k')
            plt.scatter(rz, gr, alpha=0.5, color='green', edgecolor='none')
            plt.scatter(rz[keep], gr[keep], color='green', edgecolor='k')
            plt.xlim(-0.5, 2) ; plt.ylim(-0.5, 2)
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

    for key in ('TEMPLATEID', 'MAG', 'DECAM_FLUX', 'WISE_FLUX',
                'OIIFLUX', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'):
        truth[key] = meta[key]

    # Perturb the photometry based on the variance on this brick and apply
    # target selection.
    for band in (0, 1):
        targets['WISE_FLUX'][:, band] = truth['WISE_FLUX'][:, band] + \
          rand.normal(scale=wise_onesigma[band], size=nobj)
    for band in (1, 2, 4):
        targets['DECAM_FLUX'][:, band] = truth['DECAM_FLUX'][:, band] + \
          rand.normal(scale=decam_onesigma[band], size=nobj)

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
        gr1 = -2.5 * np.log10( truth['DECAM_FLUX'][:, 1] / truth['DECAM_FLUX'][:, 2] )
        rz1 = -2.5 * np.log10( truth['DECAM_FLUX'][:, 2] / truth['DECAM_FLUX'][:, 4] )
        gr = -2.5 * np.log10( targets['DECAM_FLUX'][:, 1] / targets['DECAM_FLUX'][:, 2] )
        rz = -2.5 * np.log10( targets['DECAM_FLUX'][:, 2] / targets['DECAM_FLUX'][:, 4] )
        fig, ax = plt.subplots()
        ax.scatter(rz1, gr1, color='red')
        ax.scatter(rz, gr, alpha=0.5, color='green')
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(-0.5, 2)
        elg_colorbox(ax)
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

def _write_onebrick(writeargs):
    """Filler function for the multiprocessing."""
    return write_onebrick(*writeargs)

def write_onebrick(thisbrick, targets, truth, trueflux, truthhdr, wave, output_dir, log):
    """Wrapper function to write out files on a single brick.

    """
    from astropy.io import fits

    onbrick = np.where(targets['BRICKNAME'] == thisbrick)[0]

    radir = os.path.join(output_dir, thisbrick[:3])
    targetsfile = os.path.join(radir, 'targets-{}.fits'.format(thisbrick))
    truthfile = os.path.join(radir, 'truth-{}.fits'.format(thisbrick))
    truthspecfile = os.path.join(radir, 'truth-spectra-{}.fits'.format(thisbrick))

    log.info('Writing {}'.format(targetsfile))
    try:
        targets[onbrick].write(targetsfile, overwrite=True)
    except:
        targets[onbrick].write(targetsfile, clobber=True)

    #log.info('Writing {}'.format(truthfile))
    try:
        truth[onbrick].write(truthfile, overwrite=True)
    except:
        truth[onbrick].write(truthfile, clobber=True)

    #log.info('Writing {}'.format(truthspecfile))
    hx = fits.HDUList()
    hdu = fits.ImageHDU(wave.astype(np.float32), name='WAVE', header=truthhdr)
    hdu.header['BUNIT'] = 'Angstrom'
    hdu.header['AIRORVAC'] = 'vac'
    hx.append(hdu)

    hdu = fits.ImageHDU(trueflux[onbrick, :].astype(np.float32), name='FLUX')
    hdu.header['BUNIT'] = '1e-17 erg/s/cm2/A'
    hx.append(hdu)
    
    try:
        hx.writeto(truthspecfile, overwrite=True)
    except:
        hx.writeto(truthspecfile, clobber=True)

    #write_bintable(truthfile, truth[onbrick], extname='TRUTH')

def _write_onehealpix(writeargs):
    """Filler function for the multiprocessing."""
    return write_onehealpix(*writeargs)

def write_onehealpix(subdir, pixnum, inpixel, nside, targets, truth,
                     trueflux, truthhdr, wave, output_dir, log):
    """Wrapper function to write out files in a single healpix pixel.

    """
    from astropy.io import fits

    targetsfile = os.path.join(subdir, 'targets-{}-{}.fits'.format(nside, pixnum))
    truthfile = os.path.join(subdir, 'truth-{}-{}.fits'.format(nside, pixnum))
    truthspecfile = os.path.join(subdir, 'spectra-truth-{}-{}.fits'.format(nside, pixnum))

    log.info('Writing {}'.format(targetsfile))
    try:
        targets[inpixel].write(targetsfile, overwrite=True)
    except:
        targets[inpixel].write(targetsfile, clobber=True)

    #log.info('Writing {}'.format(truthfile))
    try:
        truth[inpixel].write(truthfile, overwrite=True)
    except:
        truth[inpixel].write(truthfile, clobber=True)

    #log.info('Writing {}'.format(truthspecfile))
    hx = fits.HDUList()
    hdu = fits.ImageHDU(wave.astype(np.float32), name='WAVE', header=truthhdr)
    hdu.header['BUNIT'] = 'Angstrom'
    hdu.header['AIRORVAC'] = 'vac'
    hx.append(hdu)

    hdu = fits.ImageHDU(trueflux[inpixel, :].astype(np.float32), name='FLUX')
    hdu.header['BUNIT'] = '1e-17 erg/s/cm2/A'
    hx.append(hdu)
    
    try:
        hx.writeto(truthspecfile, overwrite=True)
    except:
        hx.writeto(truthspecfile, clobber=True)

    #write_bintable(truthfile, truth[inpixel], extname='TRUTH')

def _create_raslices(output_dir, ioutput_dir, brickname):
    """Create the RA-slice directories, if necessary."""
    
    for odir in (output_dir, ioutput_dir):
        radir = np.array(['{}'.format(os.path.join(odir, name[:3])) for name in brickname])
        for thisradir in list(set(radir)):
            try:
                os.stat(thisradir)
            except:
                os.makedirs(thisradir)
                
#from memory_profiler import profile
#@profile
def targets_truth(params, output_dir, realtargets=None, seed=None, verbose=True,
                  clobber=False, bricksize=0.25, nside=64, nproc=1):
    """
    Write

    Args:
        params: dict of source definitions.
        output_dir: location for intermediate mtl files.
        realtargets (optional): real target catalog table, e.g. from DR3
        clobber (optional): delete files in the output directory (mandatory if not empty)
        nproc (optional): number of parallel processes to use (default 4)

    Returns:
      targets:
      truth:

    Notes:
      If nproc == 1 use serial instead of parallel code.

    """
    import healpy
    import shutil
    from time import time

    from desispec.io.util import fitsheader, write_bintable
    import desitarget.mock.io as mockio
    from desitarget.mock.selection import SelectTargets
    from desitarget.mock.spectra import MockSpectra
    from desitarget.internal import sharedmem
    
    if params is None or output_dir is None:
        log.fatal('Required inputs params and output_dir not given!')
        raise ValueError

    if verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()
    
    # Initialize the random seed
    rand = np.random.RandomState(seed)

    # Create the output directories and clean them up if necessary.
    #ioutput_dir = os.path.normpath(output_dir)+'-i'

    #for odir in (output_dir, ioutput_dir):
    for odir in np.atleast_1d(output_dir):
        try:
            os.stat(odir)
            if os.listdir(odir):
                if clobber:
                    shutil.rmtree(odir)
                    #log.info('Cleaning directory {}'.format(odir))
                    os.makedirs(odir)
                else:
                    log.warning('Output directory {} is not empty; please set clobber=True.'.format(odir))
                    return
        except:
            log.info('Creating directory {}'.format(odir))
            os.makedirs(odir)
    log.info('Writing to output directory {}'.format(output_dir))
    #log.info('Writing to output directory {} and intermediate output directory {}'.format(output_dir, ioutput_dir))
    print()

    # Add the ra,dec boundaries to the parameters dictionary for each source, so
    # we can check the target densities, below.
    if ('subset' in params.keys()) & (params['subset']['ra_dec_cut'] == True):
        bounds = (params['subset']['min_ra'], params['subset']['max_ra'],
                  params['subset']['min_dec'], params['subset']['max_dec'])
    else:
        bounds = (0.0, 360.0, -90.0, 90.0)

    for src in params['sources'].keys():
        params['sources'][src].update({'bounds': bounds})

    # Build the brick information structure.
    brick_info = BrickInfo(random_state=rand, dust_dir=params['dust_dir'], bounds=bounds,
                           bricksize=bricksize, decals_brick_info=params['decals_brick_info'],
                           target_names=list(params['sources'].keys()), log=log).build_brickinfo()

    # Initialize the Classes used to assign spectra and select targets.  Note:
    # The default wavelength array gets initialized here, too.
    log.info('Initializing the MockSpectra and SelectTargets Classes.')
    Spectra = MockSpectra(rand=rand, verbose=verbose, nproc=nproc)
    Selection = SelectTargets(logger=log, rand=rand,
                              brick_info=brick_info)
    print()

    # Loop over each source / object type.
    alltargets = list()
    alltruth = list()
    alltrueflux = list()
    for source_name in params['sources'].keys():
        # Read the mock catalog.
        target_name = params['sources'][source_name]['target_name'] # Target type (e.g., ELG)
        mockformat = params['sources'][source_name]['format']
        #source_data = source_data_all[source_name]     # data (ra, dec, etc.)

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
                                        bounds=bounds, magcut=magcut, nproc=nproc, lya=lya)

        # If there are no sources, keep going.
        if not bool(source_data):
            continue

        selection_function = '{}_select'.format(target_name.lower())
        select_targets_function = getattr(Selection, selection_function)

        # Assign spectra by parallel-processing the bricks.
        brickname = source_data['BRICKNAME']
        unique_bricks = list(set(brickname))

        # Quickly check that all the brick info is here.
        skyarea = 0.0
        for thisbrick in unique_bricks:
            brickindx = np.where(brick_info['BRICKNAME'] == thisbrick)[0]
            skyarea = skyarea + brick_info['BRICKAREA'][brickindx][0]
            if (len(brickindx) != 1):
                log.fatal('One or too many matching brick(s) {}! This should not happen...'.format(thisbrick))
                raise ValueError
        #skyarea = brick_info['BRICKAREA'][0] * len(unique_bricks)
        
        log.info('Assigned {} {}s to {} unique {}x{} deg2 bricks spanning (approximately) {:.4g} deg2.'.format(
            len(brickname), source_name, len(unique_bricks), bricksize, bricksize, skyarea))

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
                              Spectra, select_targets_function, source_data, rand, log) )

        if nproc > 1:
            pool = sharedmem.MapReduce(np=nproc)
            with pool:
                out = pool.map(_get_spectra_onebrick, specargs,
                               reduce=_update_spectra_status)
        else:
            out = list()
            for ii in range(len(unique_bricks)):
                out.append( _update_spectra_status( _get_spectra_onebrick(specargs[ii]) ) )

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

        if target_name.upper() == 'SKY':
            skytruth = truth.copy()
            skytargets = targets.copy()
        else:
            alltargets.append(targets)
            alltruth.append(truth)
            alltrueflux.append(trueflux)
        print()

    # Consolidate across all the mocks.  Note that the code quits if alltargets
    #is zero-length, even if skytargets is non-zero length. In other words, if
    #the parameter file only contains SKY the code will quit anyway.
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
    for source_name in params['sources'].keys():
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

    #print( np.sum( (targets['DESI_TARGET'] & desi_mask.MWS_ANY) != 0) )
    #import pdb ; pdb.set_trace()
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

    targetid = rand.randint(2**62, size=ntarget + nsky)
    subpriority = rand.uniform(0.0, 1.0, size=ntarget + nsky)

    truth['TARGETID'] = targetid[:ntarget]
    targets['TARGETID'] = targetid[:ntarget]
    targets['SUBPRIORITY'] = subpriority[:ntarget]

    # Write out the sky catalog.
    if nsky > 0:
        skyfile = os.path.join(output_dir, 'sky.fits')
        
        skytargets['TARGETID'] = targetid[ntarget:ntarget+nsky]
        skytargets['SUBPRIORITY'] = subpriority[ntarget:ntarget+nsky]

        if np.sum((skytargets['DESI_TARGET'] & desi_mask.SKY) != 0) != nsky:
            log.fatal('Lost SKY targets somewhere!')
            raise ValueError

        log.info('Writing {} SKY targets to {}'.format(nsky, skyfile))
        write_bintable(skyfile, skytargets, extname='SKY', clobber=True)
        print()

    # Write out the dark- and bright-time standard stars.  White dwarf standards
    # not yet supported.
    for suffix, stdbit in zip(('dark', 'bright'), ('STD_FSTAR', 'STD_BRIGHT')):
        stdfile = os.path.join(output_dir, 'standards-{}.fits'.format(suffix))
        istd = ((targets['DESI_TARGET'] & desi_mask.mask(stdbit)) |
                (targets['DESI_TARGET'] & desi_mask.mask('STD_WD'))) != 0
        #istd = (targets['DESI_TARGET'] & desi_mask.mask(stdbit)) != 0
        if np.count_nonzero(istd) > 0:
            log.info('Writing {}'.format(stdfile))
            write_bintable(stdfile, targets[istd], extname='STD', clobber=True)
        else:
            log.info('No {} standards found, {} not written.'.format(suffix.upper(), stdfile))
    print()

    if seed is None:
        seed1 = 'None'
    else:
        seed1 = seed
    truthhdr = fitsheader(dict(
        SEED = (seed1, 'initial random seed')
        #BRICKSZ = (outbricksize, 'brick size (deg)')
        #BUNIT = ('Angstrom', 'wavelength units'),
        #AIRORVAC = ('vac', 'vacuum wavelengths')
        ))

    # Write out targets by healpix pixels based on the grouping
    #   {output_dir}/{subdir}/truth-{nside}-{ipix}.fits
    # where subdir = ipix // (nside // 2)
    pixels = healpy.ang2pix(nside, np.radians(90-targets['DEC']),
                            np.radians(targets['RA']), nest=True)
    unique_pixels = list(set(pixels))

    """basedir/8-{superpix}/64-{pixnum}/filename-64-{pixnum}.fits

    where pixnum is the nside=64 nested pixel number, and superpix = pixnum /
    4**3.

    I'm not completely convinced that will be user friendly, but it does have
    the advantages that:

    * <1000 subdirectories at any level
    
    * everything under a 8-{superpix} subdirectory is grouped on the sky (unlike
      the case if we just did subdir = pixnum//100 or something like that that
      is easier to calculate in your head but breaks spatial grouping).

    * nside=8 is 53.7 deg2 which seems likely for a viable sub-unit to process
      at NERSC per job

    * nside=64 is 0.84 deg2 which seems like a viable sized sub-unit for
      grouping targets
    """

    writeargs = list()
    for pixnum in unique_pixels:
        superpix = pixnum // 4**3
        subdir = os.path.join( output_dir, '8-{}'.format(superpix), '64-{}'.format(pixnum) )
        #subdir = os.path.join( output_dir, str(pixnum // (nside // 2)) )
        try:
            os.stat(subdir)
        except:
            os.makedirs(subdir)

        inpixel = np.where(pixnum == pixels)[0]
        writeargs.append((subdir, pixnum, inpixel, nside, targets, truth, trueflux,
                          truthhdr, Spectra.wave, output_dir, log))

    if nproc > 1:
        pool = sharedmem.MapReduce(np=nproc)
        with pool:
            pool.map(_write_onehealpix, writeargs)
    else:
        for args in writeargs:
            _write_onehealpix(args)

def join_targets_truth(output_dir, nside=64, verbose=True, clobber=False):
    """Combine all the target and truth catalogs generated by targets_truth into a
    monolithic targets.fits and truth.fits files

    time select_mock_targets --output_dir debug --join

    """
    import fitsio
    from glob import glob

    if verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    targets, truth = [], []

    subdirs = [dd for dd in os.listdir(output_dir) if os.path.isdir( os.path.join(output_dir, dd) ) ]
    
    for subdir in np.atleast_1d(subdirs):
        alltargfile = np.array( glob(os.path.join(output_dir, subdir, '*-*', 'targets-*-*.fits') ) )
        alltruthfile = np.array( glob(os.path.join(output_dir, subdir, '*-*', 'truth-*-*.fits') ) )

        for targfile, truthfile in zip( np.atleast_1d(alltargfile), np.atleast_1d(alltruthfile) ):
            log.info('Reading {}'.format(targfile))
            targets.append( Table(fitsio.read(targfile, ext=1)) )
            truth.append( Table(fitsio.read(truthfile, ext=1)) )
        
    targets = vstack( targets )
    truth = vstack( truth )

    for outfile, cat in zip( (os.path.join(output_dir, 'targets.fits'),
                              os.path.join(output_dir, 'truth.fits')), (targets, truth) ):
        log.info('Writing {}'.format(outfile))
        try:
            cat.write(outfile, overwrite=True)
        except:
            cat.write(outfile, clobber=True)

