# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.mock.build, but only add_mock_shapes_and_fluxes for now.
"""
import unittest
import numpy as np
from astropy.table import Table
import healpy as hp

import desimodel.footprint

from desitarget.mock.sky import random_sky
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask

class TestMockBuild(unittest.TestCase):
    
    def setUp(self):
        pass

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

if __name__ == '__main__':
    unittest.main()

def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_mock_build
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
