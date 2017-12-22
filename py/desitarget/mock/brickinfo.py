# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.brickinfo
=========================

Obsolete class for computing statistics on bricks.

"""
from __future__ import absolute_import, division, print_function

import os
import numpy as np

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

