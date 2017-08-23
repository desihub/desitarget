import unittest
import numpy as np

from astropy.table import Table

from desitarget import desi_mask, bgs_mask, mws_mask, obsmask
from desitarget.targets import calc_priority

class TestPriorities(unittest.TestCase):

    def setUp(self):
        dtype = [
            ('DESI_TARGET',np.int64),
            ('BGS_TARGET',np.int64),
            ('MWS_TARGET',np.int64),
            ('Z',np.float32),
            ('ZWARN',np.float32),
        ]
        self.targets = Table(np.zeros(3, dtype=dtype))

    def test_priorities(self):
        t = self.targets
        #- No targeting bits set is priority=0
        self.assertTrue(np.all(calc_priority(t) == 0))

        #- test QSO > LRG > ELG
        t['DESI_TARGET'] = desi_mask.ELG
        self.assertTrue(np.all(calc_priority(t) == desi_mask.ELG.priorities['UNOBS']))
        t['DESI_TARGET'] |= desi_mask.LRG
        self.assertTrue(np.all(calc_priority(t) == desi_mask.LRG.priorities['UNOBS']))
        t['DESI_TARGET'] |= desi_mask.QSO
        self.assertTrue(np.all(calc_priority(t) == desi_mask.QSO.priorities['UNOBS']))

        #- different states -> different priorities

        #- Done is Done, regardless of ZWARN.
        t['DESI_TARGET'] = desi_mask.ELG
        t['NUMOBS'] = [0, 1, 1]
        t['ZWARN']  = [1, 1, 0]
        p = calc_priority(t)

        self.assertEqual(p[0], desi_mask.ELG.priorities['UNOBS'])
        self.assertEqual(p[1], desi_mask.ELG.priorities['DONE'])
        self.assertEqual(p[2], desi_mask.ELG.priorities['DONE'])

        #- BGS FAINT targets are never DONE, only MORE_ZGOOD.
        t['DESI_TARGET'] = desi_mask.BGS_ANY
        t['BGS_TARGET']  = bgs_mask.BGS_FAINT
        t['NUMOBS'] = [0, 1, 1]
        t['ZWARN']  = [1, 1, 0]
        p = calc_priority(t)

        self.assertEqual(p[0], bgs_mask.BGS_FAINT.priorities['UNOBS'])
        self.assertEqual(p[1], bgs_mask.BGS_FAINT.priorities['MORE_ZWARN'])
        self.assertEqual(p[2], bgs_mask.BGS_FAINT.priorities['MORE_ZGOOD'])
        ### BGS_FAINT: {UNOBS: 2000, MORE_ZWARN: 2000, MORE_ZGOOD: 1000, DONE: 2, OBS: 1, DONOTOBSERVE: 0}

        #- BGS BRIGHT targets are never DONE, only MORE_ZGOOD.
        t['DESI_TARGET']  = desi_mask.BGS_ANY
        t['BGS_TARGET']   = bgs_mask.BGS_BRIGHT
        t['NUMOBS'] = [0, 1, 1]
        t['ZWARN']  = [1, 1, 0]
        p = calc_priority(t)

        self.assertEqual(p[0], bgs_mask.BGS_BRIGHT.priorities['UNOBS'])
        self.assertEqual(p[1], bgs_mask.BGS_BRIGHT.priorities['MORE_ZWARN'])
        self.assertEqual(p[2], bgs_mask.BGS_BRIGHT.priorities['MORE_ZGOOD'])
        ### BGS_BRIGHT: {UNOBS: 2100, MORE_ZWARN: 2100, MORE_ZGOOD: 1000, DONE: 2, OBS: 1, DONOTOBSERVE: 0}

        # BGS targets are NEVER done even after 100 observations
        t['DESI_TARGET'] = desi_mask.BGS_ANY
        t['BGS_TARGET']  = bgs_mask.BGS_BRIGHT
        t['NUMOBS'] = [0, 100, 100]
        t['ZWARN']  = [1,   1,   0]
        p = calc_priority(t)

        self.assertEqual(p[0], bgs_mask.BGS_BRIGHT.priorities['UNOBS'])
        self.assertEqual(p[1], bgs_mask.BGS_BRIGHT.priorities['MORE_ZWARN'])
        self.assertEqual(p[2], bgs_mask.BGS_BRIGHT.priorities['MORE_ZGOOD'])

        # BGS ZGOOD targets always have lower priority than MWS targets that
        # are not DONE.
        lowest_bgs_priority_zgood = min([bgs_mask[n].priorities['MORE_ZGOOD'] for n in bgs_mask.names()])

        lowest_mws_priority_unobs = min([mws_mask[n].priorities['UNOBS'] for n in mws_mask.names()])
        lowest_mws_priority_zwarn = min([mws_mask[n].priorities['MORE_ZWARN'] for n in mws_mask.names()])
        lowest_mws_priority_zgood = min([mws_mask[n].priorities['MORE_ZGOOD'] for n in mws_mask.names()])

        lowest_mws_priority = min(lowest_mws_priority_unobs,
                                  lowest_mws_priority_zwarn,
                                  lowest_mws_priority_zgood)

        self.assertLess(lowest_bgs_priority_zgood,lowest_mws_priority)

    def test_bright_mask(self):
        t = self.targets
        t['DESI_TARGET'][0] = desi_mask.ELG
        t['DESI_TARGET'][1] = desi_mask.ELG | desi_mask.NEAR_BRIGHT_OBJECT
        t['DESI_TARGET'][2] = desi_mask.ELG | desi_mask.IN_BRIGHT_OBJECT
        p = calc_priority(t)
        self.assertEqual(p[0], p[1], "NEAR_BRIGHT_OBJECT shouldn't impact priority but {} != {}".format(p[0], p[1]))
        self.assertEqual(p[2], -1, "IN_BRIGHT_OBJECT priority not -1")

    def test_mask_priorities(self):
        for mask in [desi_mask, bgs_mask, mws_mask]:
            for name in mask.names():
                if name == 'SKY' or name.startswith('STD') \
                    or name in ['BGS_ANY', 'MWS_ANY', 'ANCILLARY_ANY',
                                'IN_BRIGHT_OBJECT', 'NEAR_BRIGHT_OBJECT',
                                'BRIGHT_OBJECT']:
                    self.assertEqual(mask[name].priorities, {}, 'mask.{} has priorities?'.format(name))
                else:
                    for state in obsmask.names():
                        self.assertIn(state, mask[name].priorities,
                            '{} not in mask.{}.priorities'.format(state, name))

if __name__ == '__main__':
    unittest.main()
