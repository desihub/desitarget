# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget brick fluctuations (mostly deprecated).
"""

import unittest
import numpy as np
from astropy.table import Table

from desitarget.targetmask import desi_mask as Mx
from desitarget.targetmask import desi_mask
from desitarget.mock import build


class TestBrickFluctuation(unittest.TestCase):

    def setUp(self):
        self.targets = Table()
        self.types = np.array(['ELG', 'LRG', 'QSO', 'QSO', 'ELG'])
        self.priorities = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.post_prio = [Mx[t].priorities['MORE_ZGOOD'] for t in self.types]
        self.post_prio[0] = 1  # - ELG
        self.post_prio[2] = 1  # - low-z QSO
        self.targets['DESI_TARGET'] = [Mx[t].mask for t in self.types]
        self.targets['BGS_TARGET'] = np.zeros(len(self.types), dtype=np.int64)
        self.targets['MWS_TARGET'] = np.zeros(len(self.types), dtype=np.int64)
        n = len(self.targets)
        self.targets['ZFLUX'] = 10**((22.5-np.linspace(20, 22, n))/2.5)
        self.targets['TARGETID'] = list(range(n))

        # - reverse the order for zcat to make sure joins work
        self.zcat = Table()
        self.zcat['TARGETID'] = self.targets['TARGETID'][-2::-1]
        self.zcat['Z'] = [2.5, 1.0, 0.5, 1.0]
        self.zcat['ZWARN'] = [0, 0, 0, 0]
        self.zcat['NUMOBS'] = [1, 1, 1, 1]

        # self.brick_info = build.BrickInfo()
        # self.b = self.brick_info.generate_brick_info()
        # self.depth = self.brick_info.depths_across_bricks(self.b)

    @unittest.skip('This test is deprecated, so skip for now.')
    def test_generate_brick(self):
        keys = ['BRICKNAME', 'BRICKID', 'BRICKQ', 'BRICKROW', 'BRICKCOL',
                'RA', 'DEC', 'RA1', 'RA2', 'DEC1', 'DEC2', 'AREA']
        for k in self.b.keys():
            self.assertTrue(k in keys)
            self.assertTrue(isinstance(self.b[k], np.ndarray))
        self.assertTrue(np.all((self.b['RA'] < self.b['RA2']) & (self.b['RA'] > self.b['RA1'])))
        self.assertTrue(np.all((self.b['DEC'] <= self.b['DEC2']) & (self.b['DEC'] >= self.b['DEC1'])))

    @unittest.skip('This test is deprecated, so skip for now.')
    def test_generate_depths(self):
        keys = ['PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z']
        for k in self.depth.keys():
            self.assertTrue(k in keys)
            self.assertTrue(isinstance(self.depth[k], np.ndarray))
            self.assertEqual(len(self.depth[k]), len(self.b['RA']))
            self.assertEqual(len(self.depth[k]), len(self.b['DEC']))
