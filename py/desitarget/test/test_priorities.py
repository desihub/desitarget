# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.targets.calc_priority.
"""
import unittest
import numpy as np

from astropy.table import Table

from desitarget.targetmask import desi_mask, bgs_mask, mws_mask, obsmask
from desitarget.targets import calc_priority, main_cmx_or_sv
from desitarget.targets import initial_priority_numobs
from desitarget.mtl import make_mtl, mtldatamodel


class TestPriorities(unittest.TestCase):

    def setUp(self):
        zdtype = [
            ('Z', np.float32),
            ('ZWARN', np.float32),
            ('NUMOBS', np.float32),
            ('SPECTYPE', np.str)
        ]

        n = 3

        self.targets = Table(np.zeros(n, dtype=mtldatamodel.dtype))
        self.targets['TARGETID'] = list(range(n))

        self.zcat = Table(np.zeros(n, dtype=zdtype))
        self.zcat['TARGETID'] = list(range(n))

    def test_priorities(self):
        """Test that priorities are set correctly for both the main survey and SV.
        """
        # ADM loop through once for SV and once for the main survey.
        for prefix in ["", "SV1_"]:
            t = self.targets.copy()
            z = self.zcat.copy()

            main_names = ['DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET']
            for name in main_names:
                t.rename_column(name, prefix+name)

            # ADM retrieve the mask and column names for this survey flavor.
            colnames, masks, _ = main_cmx_or_sv(t)
            desi_target, bgs_target, mws_target = colnames
            desi_mask, bgs_mask, mws_mask = masks

            # - No targeting bits set is priority=0
            self.assertTrue(np.all(calc_priority(t, z, "BRIGHT") == 0))

            # ADM test QSO > LRG > ELG for main survey and SV.
            t[desi_target] = desi_mask.ELG
            self.assertTrue(np.all(calc_priority(
                t, z, "GRAY|DARK") == desi_mask.ELG.priorities['UNOBS']))
            t[desi_target] |= desi_mask.LRG
            self.assertTrue(np.all(calc_priority(
                t, z, "GRAY|DARK") == desi_mask.LRG.priorities['UNOBS']))
            t[desi_target] |= desi_mask.QSO
            self.assertTrue(np.all(calc_priority(
                t, z, "GRAY|DARK") == desi_mask.QSO.priorities['UNOBS']))

            # - different states -> different priorities
            # - Done is Done, regardless of ZWARN.
            t[desi_target] = desi_mask.ELG
            t["PRIORITY_INIT"], t["NUMOBS_INIT"] = initial_priority_numobs(t)
            z['NUMOBS'] = [0, 1, 1]
            z['ZWARN'] = [1, 1, 0]
            p = make_mtl(t, "GRAY|DARK", zcat=z)["PRIORITY"]

            self.assertEqual(p[0], desi_mask.ELG.priorities['UNOBS'])
            self.assertEqual(p[1], desi_mask.ELG.priorities['DONE'])
            self.assertEqual(p[2], desi_mask.ELG.priorities['DONE'])

            # ADM In BRIGHT conditions BGS FAINT targets are
            # ADM never DONE, only MORE_ZGOOD.
            t[desi_target] = desi_mask.BGS_ANY
            t[bgs_target] = bgs_mask.BGS_FAINT
            t["PRIORITY_INIT"], t["NUMOBS_INIT"] = initial_priority_numobs(t)
            z['NUMOBS'] = [0, 1, 1]
            z['ZWARN'] = [1, 1, 0]
            p = make_mtl(t, "BRIGHT", zcat=z)["PRIORITY"]

            self.assertEqual(p[0], bgs_mask.BGS_FAINT.priorities['UNOBS'])
            self.assertEqual(p[1], bgs_mask.BGS_FAINT.priorities['MORE_ZWARN'])
            self.assertEqual(p[2], bgs_mask.BGS_FAINT.priorities['MORE_ZGOOD'])
            # BGS_FAINT: {UNOBS: 2000, MORE_ZWARN: 2000, MORE_ZGOOD: 1000, DONE: 2, OBS: 1, DONOTOBSERVE: 0}

            # ADM but in DARK conditions, BGS_FAINT should behave as
            # ADM for other target classes.
            z = self.zcat.copy()
            z['NUMOBS'] = [0, 1, 1]
            z['ZWARN'] = [1, 1, 0]
            p = make_mtl(t, "DARK|GRAY", zcat=z)["PRIORITY"]

            self.assertEqual(p[0], bgs_mask.BGS_FAINT.priorities['UNOBS'])
            self.assertEqual(p[1], bgs_mask.BGS_FAINT.priorities['DONE'])
            self.assertEqual(p[2], bgs_mask.BGS_FAINT.priorities['DONE'])

            # ADM In BRIGHT conditions BGS BRIGHT targets are
            # ADM never DONE, only MORE_ZGOOD.
            t[desi_target] = desi_mask.BGS_ANY
            t[bgs_target] = bgs_mask.BGS_BRIGHT
            t["PRIORITY_INIT"], t["NUMOBS_INIT"] = initial_priority_numobs(t)
            z['NUMOBS'] = [0, 1, 1]
            z['ZWARN'] = [1, 1, 0]
            p = make_mtl(t, "BRIGHT", zcat=z)["PRIORITY"]

            self.assertEqual(p[0], bgs_mask.BGS_BRIGHT.priorities['UNOBS'])
            self.assertEqual(p[1], bgs_mask.BGS_BRIGHT.priorities['MORE_ZWARN'])
            self.assertEqual(p[2], bgs_mask.BGS_BRIGHT.priorities['MORE_ZGOOD'])
            # BGS_BRIGHT: {UNOBS: 2100, MORE_ZWARN: 2100, MORE_ZGOOD: 1000, DONE: 2, OBS: 1, DONOTOBSERVE: 0}

            # ADM In BRIGHT conditions BGS targets are
            # ADM NEVER done even after 100 observations
            t[desi_target] = desi_mask.BGS_ANY
            t[bgs_target] = bgs_mask.BGS_BRIGHT
            t["PRIORITY_INIT"], t["NUMOBS_INIT"] = initial_priority_numobs(t)
            z['NUMOBS'] = [0, 100, 100]
            z['ZWARN'] = [1,   1,   0]
            p = calc_priority(t, z, "BRIGHT")

            self.assertEqual(p[0], bgs_mask.BGS_BRIGHT.priorities['UNOBS'])
            self.assertEqual(p[1], bgs_mask.BGS_BRIGHT.priorities['MORE_ZWARN'])
            self.assertEqual(p[2], bgs_mask.BGS_BRIGHT.priorities['MORE_ZGOOD'])

            # BGS ZGOOD targets always have lower priority than MWS targets that
            # are not DONE. Exempting the MWS "BACKUP" targets.
            # ADM first discard N/S informational bits from bitmask as these
            # ADM should never trump the other bits.
            bgs_names = [name for name in bgs_mask.names() if 'NORTH' not in name
                         and 'SOUTH' not in name]
            mws_names = [name for name in mws_mask.names() if
                         'NORTH' not in name and 'SOUTH' not in name and
                         'BACKUP' not in name and 'STD' not in name]

            lowest_mws_priority_unobs = [mws_mask[n].priorities['UNOBS']
                                         for n in mws_names]

            lowest_bgs_priority_zgood = np.min(
                [bgs_mask[n].priorities['MORE_ZGOOD'] for n in bgs_names])

            # ADM MORE_ZGOOD and MORE_ZWARN are only meaningful if a
            # ADM target class requests more than 1 observation (except
            # ADM for BGS, which has a numobs=infinity exception)
            lowest_mws_priority_zwarn = [mws_mask[n].priorities['MORE_ZWARN']
                                         for n in mws_names
                                         if mws_mask[n].numobs > 1]
            lowest_mws_priority_zgood = [mws_mask[n].priorities['MORE_ZGOOD']
                                         for n in mws_names
                                         if mws_mask[n].numobs > 1]

            lowest_mws_priority = np.min(np.concatenate([
                lowest_mws_priority_unobs,
                lowest_mws_priority_zwarn,
                lowest_mws_priority_zgood]))

            self.assertLess(lowest_bgs_priority_zgood, lowest_mws_priority)

    def test_bright_mask(self):
        t = self.targets
        z = self.zcat
        t['DESI_TARGET'][0] = desi_mask.ELG
        t['DESI_TARGET'][1] = desi_mask.ELG | desi_mask.NEAR_BRIGHT_OBJECT
        t['DESI_TARGET'][2] = desi_mask.ELG | desi_mask.IN_BRIGHT_OBJECT
        p = calc_priority(t, z, "BRIGHT|GRAY|DARK")
        self.assertEqual(p[0], p[1], "NEAR_BRIGHT_OBJECT shouldn't impact priority but {} != {}".format(p[0], p[1]))
        self.assertEqual(p[2], -1, "IN_BRIGHT_OBJECT priority not -1")

    def test_mask_priorities(self):
        for mask in [desi_mask, bgs_mask, mws_mask]:
            for name in mask.names():
                if (
                        'STD' in name or name.endswith('BRIGHT_OBJECT') or name in
                        ['BGS_ANY', 'MWS_ANY', 'SCND_ANY', 'SKY', 'SUPP_SKY', 'NO_TARGET']
                ):
                    self.assertEqual(mask[name].priorities, {}, 'mask.{} has priorities?'.format(name))
                else:
                    for state in obsmask.names():
                        self.assertIn(state, mask[name].priorities,
                                      '{} not in mask.{}.priorities'.format(state, name))

    def test_cmx_priorities(self):
        """Test that priority calculation can handle commissioning files.
        """
        t = self.targets.copy()
        z = self.zcat

        # ADM restructure the table to look like a commissioning table.
        t.rename_column('DESI_TARGET', 'CMX_TARGET')
        t.remove_column('BGS_TARGET')
        t.remove_column('MWS_TARGET')

        # - No targeting bits set is priority=0
        self.assertTrue(np.all(calc_priority(t, z, "GRAY|DARK") == 0))

        # ADM retrieve the cmx_mask.
        colnames, masks, _ = main_cmx_or_sv(t)
        cmx_mask = masks[0]

        # ADM test handling of unobserved SV0_BGS and SV0_MWS
        for name, obscon in [("SV0_BGS", "BRIGHT"), ("SV0_MWS", "POOR")]:
            t['CMX_TARGET'] = cmx_mask[name]
            self.assertTrue(np.all(calc_priority(
                t, z, obscon) == cmx_mask[name].priorities['UNOBS']))

        # ADM done is Done, regardless of ZWARN.
        for name, obscon in [("SV0_BGS", "BRIGHT"), ("SV0_MWS", "POOR")]:
            t['CMX_TARGET'] = cmx_mask[name]
            t["PRIORITY_INIT"], t["NUMOBS_INIT"] = initial_priority_numobs(t)

            # APC: Use NUMOBS_INIT here to avoid hardcoding NOBS corresponding to "done".
            numobs_done = t['NUMOBS_INIT'][0]
            z['NUMOBS'] = [0, numobs_done, numobs_done]
            z['ZWARN'] = [1, 1, 0]
            p = make_mtl(t, obscon, zcat=z)["PRIORITY"]

            self.assertEqual(p[0], cmx_mask[name].priorities['UNOBS'])
            self.assertEqual(p[1], cmx_mask[name].priorities['DONE'])
            self.assertEqual(p[2], cmx_mask[name].priorities['DONE'])

        # BGS ZGOOD targets always have lower priority than MWS targets that
        # are not DONE.
        lowest_bgs_priority_zgood = cmx_mask['SV0_BGS'].priorities['MORE_ZGOOD']

        lowest_mws_priority_unobs = cmx_mask['SV0_MWS'].priorities['UNOBS']
        lowest_mws_priority_zwarn = cmx_mask['SV0_MWS'].priorities['MORE_ZWARN']
        lowest_mws_priority_zgood = cmx_mask['SV0_MWS'].priorities['MORE_ZGOOD']

        lowest_mws_priority = min(lowest_mws_priority_unobs,
                                  lowest_mws_priority_zwarn,
                                  lowest_mws_priority_zgood)

        self.assertLess(lowest_bgs_priority_zgood, lowest_mws_priority)


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_priorities
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
