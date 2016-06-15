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
from desitarget import mws_mask
import os
from astropy.table import Table, Column
import desispec.brick



def build_mock_target(root_mock_mws_dir='', output_dir='', 
                      mag_faint = 20.0, mag_bright=15.0, 
                      rand_seed=42):
                      
    """Builds a Target and Truth files from a series of mock files
    
    Parameters:
    ----------    
        mock_mws_dir: string
            Root path to all the 'desi_galfast' files.
        output_dir: string
            Path to write the outputs (targets.fits and truth.fits).
        mag_bright: float
            Upper limit cut in SDSSr observed magnitude. Default = 15.0.
        mag_faint: float
            Lower limit cut in SDSSr observed magnitude. Default = 20.0.
        rand_seed: int
            seed for random number generator. 
    """
    np.random.seed(seed=rand_seed)

    # read the mocks on disk
    mws_data = desitarget.mock.io.read_mock_mws_brighttime(root_mock_mws_dir=root_mock_mws_dir)
    mws_keys = mws_data.keys()
    n = len(mws_data[mws_keys[0]])
    
    # perform magnitude cut 
    ii = ((mws_data['SDSSr_obs'] > mag_bright ) & (mws_data['SDSSr_obs'] < mag_faint))
    print("Keys: {}".format(mws_data.keys()))
    print("n_items in full catalog: {}".format(n))
    n = len(mws_data['SDSSr_obs'][ii])
    print("mag_faint {} mag_bright {}".format(mag_faint, mag_bright))
    print("n_items after selection : {}".format(n))
        
    # make up the IDs, subpriorities and bricknames
    targetid = np.random.randint(2**62, size=n)
    subprior = np.random.uniform(0., 1., size=n)
    brickname = desispec.brick.brickname(mws_data['RA'][ii], mws_data['DEC'][ii])

    # assign target flags and true types
    desi_target_pop  = np.zeros(n, dtype='i8'); 
    bgs_target_pop = np.zeros(n, dtype='i8'); 
    mws_target_pop = np.zeros(n, dtype='i8'); mws_target_pop[:] = mws_mask.MWS_PLX
    true_type_pop = np.zeros(n, dtype='S10'); true_type_pop[:] = 'MWS_PLX'

    # write the Targets to disk
    targets_filename = os.path.join(output_dir, 'targets.fits')
    targets = Table()
    targets['TARGETID'] = targetid
    targets['BRICKNAME'] = brickname
    targets['RA'] = mws_data['RA'][ii]
    targets['DEC'] = mws_data['DEC'][ii]
    targets['DESI_TARGET'] = desi_target_pop
    targets['BGS_TARGET'] = bgs_target_pop
    targets['MWS_TARGET'] = mws_target_pop
    targets['SUBPRIORITY'] = subprior
    targets.write(targets_filename, overwrite=True)

    # write the Truth to disk
    truth_filename = os.path.join(output_dir, 'truth.fits')
    truth = Table()
    truth['TARGETID'] = targetid
    truth['BRICKNAME'] = brickname
    truth['RA'] = mws_data['RA'][ii]
    truth['DEC'] = mws_data['DEC'][ii]
    truth['TRUEZ'] = mws_data['Z'][ii]
    truth['TRUETYPE'] = true_type_pop
    truth.write(truth_filename, overwrite=True)

    return
    
