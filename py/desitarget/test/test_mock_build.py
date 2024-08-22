# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.mock.build, but only add_mock_shapes_and_fluxes for now.
"""
import unittest
import tempfile
import os
import shutil
from importlib import resources
import numpy as np
from astropy.table import Table
import healpy as hp
import fitsio

import desimodel.footprint

from desitarget.mock.sky import random_sky
from desitarget.mock.build import targets_truth
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask

class TestMockBuild(unittest.TestCase):

    def setUp(self):
        self.outdir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    @unittest.skipUnless('DESITARGET_RUN_MOCK_UNITTEST' in os.environ, '$DESITARGET_RUN_MOCK_UNITTEST not set; skipping expensive mock tests')
    def test_targets_truth(self):
        configfile = resources.files('desitarget').joinpath('mock/data/select-mock-targets.yaml')

        import yaml
        with open(configfile) as fx:
            params = yaml.safe_load(fx)

        for targettype in params['targets'].keys():
            mockfile = params['targets'][targettype]['mockfile'].format(**os.environ)
            self.assertTrue(os.path.exists(mockfile), 'Missing {}'.format(mockfile))

        #- Test without spectra
        targets_truth(params, healpixels=[99737,], nside=256, output_dir=self.outdir, no_spectra=True)
        for obscon in ['bright', 'dark']:
            targetfile = self.outdir + '/997/99737/' + obscon + '/targets-{}-256-99737.fits'.format(obscon)
            truthfile = self.outdir + '/997/99737/' + obscon + '/truth-{}-256-99737.fits'.format(obscon)
            self.assertTrue(os.path.exists(targetfile))
            self.assertTrue(os.path.exists(truthfile))

            with fitsio.FITS(truthfile) as fx:
                self.assertTrue('TRUTH' in fx)
                #- WAVE is there, and FLUX is there but with strange shape (n,0)
                self.assertTrue('WAVE' not in fx)
                self.assertTrue('FLUX' not in fx)

        #- Test with spectra
        shutil.rmtree(self.outdir+'/997')

        targets_truth(params, healpixels=[99737,], nside=256, output_dir=self.outdir, no_spectra=False)
        for obscon in ['bright', 'dark']:
            targetfile = self.outdir + '/997/99737/' + obscon + '/targets-{}-256-99737.fits'.format(obscon)
            truthfile = self.outdir + '/997/99737/' + obscon + '/truth-{}-256-99737.fits'.format(obscon)
            self.assertTrue(os.path.exists(targetfile))
            self.assertTrue(os.path.exists(truthfile))

            with fitsio.FITS(truthfile) as fx:
                self.assertTrue('TRUTH' in fx)
                self.assertTrue('WAVE' in fx)
                self.assertTrue('FLUX' in fx)

    @unittest.skip('This test is deprecated, so skip for now.')
    def test_shapes_and_fluxes(self):
        from desitarget.mock.build import add_mock_shapes_and_fluxes
        nreal = 40
        real = Table()
        real['DESI_TARGET'] = 2**np.random.randint(0,3,size=nreal)
        real['BGS_TARGET'] = np.zeros(nreal, dtype=int)
        real['BGS_TARGET'][0:5] = bgs_mask.BGS_BRIGHT
        real['BGS_TARGET'][5:10] = bgs_mask.BGS_FAINT
        real['DESI_TARGET'][0:10] = 0
        
        real['DECAM_FLUX'] = np.random.uniform(size=(nreal,6))
        real['SHAPEDEV_R'] = np.random.uniform(size=nreal)
        real['SHAPEEXP_R'] = np.random.uniform(size=nreal)
        
        nmock = 45
        mock = Table()
        mock['DESI_TARGET'] = 2**np.random.randint(0,3,size=nmock)
        mock['BGS_TARGET'] = np.zeros(nmock, dtype=int)
        mock['BGS_TARGET'][10:15] = bgs_mask.BGS_BRIGHT
        mock['BGS_TARGET'][15:20] = bgs_mask.BGS_FAINT
        mock['DESI_TARGET'][10:20] = 0
        
        add_mock_shapes_and_fluxes(mock, real)
        self.assertTrue('DECAM_FLUX' in mock.colnames)
        self.assertTrue('SHAPEDEV_R' in mock.colnames)
        self.assertTrue('SHAPEEXP_R' in mock.colnames)

    def test_sky(self):
        nside = 256
        ra, dec, pix = random_sky(nside, allsky=False)
        self.assertEqual(len(ra), len(dec))
        surveypix = desimodel.footprint.tiles2pix(nside)
        theta = np.radians(90 - dec)
        phi = np.radians(ra)
        skypix = hp.ang2pix(nside, theta, phi, nest=True)
        self.assertEqual(set(surveypix), set(skypix))

