# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.targets.calc_priority.
"""
import unittest
import numpy as np

from astropy.table import Table

from desitarget.targetmask import desi_mask, bgs_mask, mws_mask, obsmask
from desitarget.targets import calc_priority
from desitarget.targets import initial_priority_numobs
from desitarget.mtl import make_mtl


class TestPriorities(unittest.TestCase):

    def setUp(self):
        targdtype = [
            ('DESI_TARGET', np.int64),
            ('BGS_TARGET', np.int64),
            ('MWS_TARGET', np.int64),
            ('PRIORITY_INIT', np.int64),
            ('NUMOBS_INIT', np.int64)
        ]
        zdtype = [
            ('Z', np.float32),
            ('ZWARN', np.float32),
            ('NUMOBS', np.float32),
        ]

        n = 3

        self.targets = Table(np.zeros(n, dtype=targdtype))
        self.targets['TARGETID'] = list(range(n))

        self.zcat = Table(np.zeros(n, dtype=zdtype))
        self.zcat['TARGETID'] = list(range(n))

    def test_priorities(self):
        t = self.targets
        z = self.zcat
        # - No targeting bits set is priority=0
        self.assertTrue(np.all(calc_priority(t, z) == 0))

        # - test QSO > (LRG_1PASS | LRG_2PASS) > ELG
        t['DESI_TARGET'] = desi_mask.ELG
        self.assertTrue(np.all(calc_priority(t, z) == desi_mask.ELG.priorities['UNOBS']))
        t['DESI_TARGET'] |= desi_mask.LRG_1PASS
        self.assertTrue(np.all(calc_priority(t, z) == desi_mask.LRG.priorities['UNOBS']))
        t['DESI_TARGET'] |= desi_mask.LRG_2PASS
        self.assertTrue(np.all(calc_priority(t, z) == desi_mask.LRG.priorities['UNOBS']))
        t['DESI_TARGET'] |= desi_mask.QSO
        self.assertTrue(np.all(calc_priority(t, z) == desi_mask.QSO.priorities['UNOBS']))

        # - different states -> different priorities

        # - Done is Done, regardless of ZWARN.
        t['DESI_TARGET'] = desi_mask.ELG
        t["PRIORITY_INIT"], t["NUMOBS_INIT"] = initial_priority_numobs(t, survey='main')
        z['NUMOBS'] = [0, 1, 1]
        z['ZWARN'] = [1, 1, 0]
        p = make_mtl(t, z)["PRIORITY"]

        self.assertEqual(p[0], desi_mask.ELG.priorities['UNOBS'])
        self.assertEqual(p[1], desi_mask.ELG.priorities['DONE'])
        self.assertEqual(p[2], desi_mask.ELG.priorities['DONE'])

        # - BGS FAINT targets are never DONE, only MORE_ZGOOD.
        t['DESI_TARGET'] = desi_mask.BGS_ANY
        t['BGS_TARGET'] = bgs_mask.BGS_FAINT
        t["PRIORITY_INIT"], t["NUMOBS_INIT"] = initial_priority_numobs(t, survey='main')
        z['NUMOBS'] = [0, 1, 1]
        z['ZWARN'] = [1, 1, 0]
        p = make_mtl(t, z)["PRIORITY"]

        self.assertEqual(p[0], bgs_mask.BGS_FAINT.priorities['UNOBS'])
        self.assertEqual(p[1], bgs_mask.BGS_FAINT.priorities['MORE_ZWARN'])
        self.assertEqual(p[2], bgs_mask.BGS_FAINT.priorities['MORE_ZGOOD'])
        # BGS_FAINT: {UNOBS: 2000, MORE_ZWARN: 2000, MORE_ZGOOD: 1000, DONE: 2, OBS: 1, DONOTOBSERVE: 0}

        # - BGS BRIGHT targets are never DONE, only MORE_ZGOOD.
        t['DESI_TARGET'] = desi_mask.BGS_ANY
        t['BGS_TARGET'] = bgs_mask.BGS_BRIGHT
        t["PRIORITY_INIT"], t["NUMOBS_INIT"] = initial_priority_numobs(t, survey='main')
        z['NUMOBS'] = [0, 1, 1]
        z['ZWARN'] = [1, 1, 0]
        p = make_mtl(t, z)["PRIORITY"]

        self.assertEqual(p[0], bgs_mask.BGS_BRIGHT.priorities['UNOBS'])
        self.assertEqual(p[1], bgs_mask.BGS_BRIGHT.priorities['MORE_ZWARN'])
        self.assertEqual(p[2], bgs_mask.BGS_BRIGHT.priorities['MORE_ZGOOD'])
        # BGS_BRIGHT: {UNOBS: 2100, MORE_ZWARN: 2100, MORE_ZGOOD: 1000, DONE: 2, OBS: 1, DONOTOBSERVE: 0}

        # BGS targets are NEVER done even after 100 observations
        t['DESI_TARGET'] = desi_mask.BGS_ANY
        t['BGS_TARGET'] = bgs_mask.BGS_BRIGHT
        t["PRIORITY_INIT"], t["NUMOBS_INIT"] = initial_priority_numobs(t, survey='main')
        z['NUMOBS'] = [0, 100, 100]
        z['ZWARN'] = [1,   1,   0]
        p = calc_priority(t, z)

        self.assertEqual(p[0], bgs_mask.BGS_BRIGHT.priorities['UNOBS'])
        self.assertEqual(p[1], bgs_mask.BGS_BRIGHT.priorities['MORE_ZWARN'])
        self.assertEqual(p[2], bgs_mask.BGS_BRIGHT.priorities['MORE_ZGOOD'])

        # BGS ZGOOD targets always have lower priority than MWS targets that
        # are not DONE.
        # ADM first discard N/S informational bits from bitmask as these
        # ADM should never trump the other bits.
        bgs_names = [name for name in bgs_mask.names() if 'NORTH' not in name and 'SOUTH' not in name]
        mws_names = [name for name in mws_mask.names() if 'NORTH' not in name and 'SOUTH' not in name]

        lowest_bgs_priority_zgood = min([bgs_mask[n].priorities['MORE_ZGOOD'] for n in bgs_names])

        lowest_mws_priority_unobs = min([mws_mask[n].priorities['UNOBS'] for n in mws_names])
        lowest_mws_priority_zwarn = min([mws_mask[n].priorities['MORE_ZWARN'] for n in mws_names])
        lowest_mws_priority_zgood = min([mws_mask[n].priorities['MORE_ZGOOD'] for n in mws_names])

        lowest_mws_priority = min(lowest_mws_priority_unobs,
                                  lowest_mws_priority_zwarn,
                                  lowest_mws_priority_zgood)

        self.assertLess(lowest_bgs_priority_zgood, lowest_mws_priority)

    def test_bright_mask(self):
        t = self.targets
        z = self.zcat
        t['DESI_TARGET'][0] = desi_mask.ELG
        t['DESI_TARGET'][1] = desi_mask.ELG | desi_mask.NEAR_BRIGHT_OBJECT
        t['DESI_TARGET'][2] = desi_mask.ELG | desi_mask.IN_BRIGHT_OBJECT
        p = calc_priority(t, z)
        self.assertEqual(p[0], p[1], "NEAR_BRIGHT_OBJECT shouldn't impact priority but {} != {}".format(p[0], p[1]))
        self.assertEqual(p[2], -1, "IN_BRIGHT_OBJECT priority not -1")

    def test_mask_priorities(self):
        for mask in [desi_mask, bgs_mask, mws_mask]:
            for name in mask.names():
                if name.startswith('STD') or name in ['BGS_ANY', 'MWS_ANY', 'SECONDARY_ANY',
                                                      'IN_BRIGHT_OBJECT', 'NEAR_BRIGHT_OBJECT',
                                                      'BRIGHT_OBJECT', 'SKY', 'SV', 'NO_TARGET']:
                    self.assertEqual(mask[name].priorities, {}, 'mask.{} has priorities?'.format(name))
                else:
                    for state in obsmask.names():
                        self.assertIn(state, mask[name].priorities,
                                      '{} not in mask.{}.priorities'.format(state, name))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_priorities
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
