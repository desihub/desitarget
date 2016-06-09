# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
===========================
desitarget.mock.brighttime
===========================

Builds target/truth files from already existing mock data
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import fitsio
import os, re
import desitarget.mock.io 
import desitarget.io
from desitarget import desi_mask
import os
from astropy.table import Table, Column
import desispec.brick
def build_mock_target(mock_bgs_file='', mock_mws_file='', output_dir='', rand_seed=42):
                      
    """Builds a Target and Truth files from a series of mock files
    
    Args:
        mock_bgs_file: string
           Filename for the mock BGS galaxies.
        mock_mws_file: string
           Filename for the mock MWS stars.
        output_dir: string
           Path to write the outputs (targets.fits and truth.fits).
        rand_seed: int
           seed for random number generator
    """
    np.random.seed(seed=rand_seed)

    # read the mocks on disk


    # build lists for the different population types

    # arrays for the full target and truth tables

    # loop over the populations
        
    # make up the IDs, subpriorities and bricknames
    n = 1
    targetid = np.random.randint(2**62, size=n)
    subprior = np.random.uniform(0., 1., size=n)
    brickname = desispec.brick.brickname(ra_total, dec_total)

    # write the Targets to disk
#    targets_filename = os.path.join(output_dir, 'targets.fits')
#    targets = Table()
#    targets['TARGETID'] = targetid
#    targets['BRICKNAME'] = brickname
#    targets['RA'] = ra_total
#    targets['DEC'] = dec_total
#    targets['DESI_TARGET'] = desi_target_total
#    targets['BGS_TARGET'] = bgs_target_total
#    targets['MWS_TARGET'] = mws_target_total
#    targets['SUBPRIORITY'] = subprior
#    targets.write(targets_filename, overwrite=True)

    # write the Truth to disk
#    truth_filename = os.path.join(output_dir, 'truth.fits')
#    truth = Table()
#    truth['TARGETID'] = targetid
#    truth['BRICKNAME'] = brickname
#    truth['RA'] = ra_total
#    truth['DEC'] = dec_total
#    truth['TRUEZ'] = z_total
#    truth['TRUETYPE'] = true_type_total
#    truth.write(truth_filename, overwrite=True)

    return
    

