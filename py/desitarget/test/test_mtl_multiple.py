# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.mtl.
"""
import os
import unittest
import numpy as np
from astropy.table import Table, join

from desitarget.targetmask import desi_mask as Mx
from desitarget.targetmask import obsconditions
from desitarget.mtl import make_mtl, mtldatamodel
from desitarget.targets import initial_priority_numobs, main_cmx_or_sv
from desitarget.targets import switch_main_cmx_or_sv


class TestMTL(unittest.TestCase):

    def setUp(self):
        self.targets = Table()

        # This is a dual identity case. In all cases the target is both QSO and ELG.
        # ADM The first case is a true QSO with low z.
        # ADM The second case is a true QSO with mid z.
        # ADM The third case is a QSO with a z warning.
        # ADM The fourth case is a true QSO with high z.
        # ADM The fifth case is a z=1.5 ELG but that was a QSO target.

        self.type_A = np.array(['QSO', 'QSO', 'QSO', 'QSO', 'ELG'])
        self.type_B = np.array(['ELG', 'ELG', 'ELG', 'ELG', 'QSO'])
        self.priorities_A = np.array([Mx[t].priorities['UNOBS'] for t in self.type_A])
        self.priorities_B = np.array([Mx[t].priorities['UNOBS'] for t in self.type_B])
        self.priorities = np.maximum(self.priorities_A, self.priorities_B)  # get the maximum between the two.

        nt = len(self.type_A)
        # ADM add some "extra" columns that are needed for observations.
        for col in ["RA", "DEC", "PARALLAX", "PMRA", "PMDEC", "REF_EPOCH"]:
            self.targets[col] = np.zeros(nt, dtype=mtldatamodel[col].dtype)
        Amx = np.array([Mx[t].mask for t in self.type_A])
        Bmx = np.array([Mx[t].mask for t in self.type_B])
        self.targets['DESI_TARGET'] = Amx | Bmx
        for col in ['BGS_TARGET', 'MWS_TARGET', 'SUBPRIORITY']:
            self.targets[col] = np.zeros(nt, dtype=mtldatamodel[col].dtype)

        n = len(self.targets)
        self.targets['ZFLUX'] = 10**((22.5-np.linspace(20, 22, n))/2.5)
        self.targets['TARGETID'] = list(range(n))
        pinit, ninit = initial_priority_numobs(self.targets)
        self.targets["PRIORITY_INIT"] = pinit
        self.targets["NUMOBS_INIT"] = ninit

        # - reverse the order for zcat to make sure joins work.
        self.zcat = Table()
        self.zcat['TARGETID'] = self.targets['TARGETID'][::-1]
        self.zcat['Z'] = [1.5, 2.5, 1.2, 1.5, 0.5]
        self.zcat['ZWARN'] = [0, 0, 1, 0, 0]
        self.zcat['NUMOBS'] = [1, 1, 1, 1, 1]
        self.zcat['SPECTYPE'] = ['GALAXY', 'QSO', 'QSO', 'QSO', 'QSO']
        self.zcat['ZTILEID'] = [-1, -1, -1, -1, -1]

        # priorities and numobs more after measuring redshifts.
        self.post_prio = [0 for t in self.type_A]
        self.post_numobs_more = [0 for t in self.type_A]
        self.post_prio[0] = Mx['QSO'].priorities['DONE']  # low-z QSO, DONE.
        self.post_prio[1] = Mx['QSO'].priorities['MORE_MIDZQSO']  # mid-z QSO, reobserve at low priority.
        self.post_prio[2] = Mx['QSO'].priorities['MORE_ZWARN']  # QSO, ZWARNING, reobserve.
        self.post_prio[3] = Mx['QSO'].priorities['MORE_ZGOOD']  # high-z QSO, reobserve at high priority.
        self.post_prio[4] = Mx['QSO'].priorities['MORE_MIDZQSO']  # mid-z QSO or galaxy, reobserve at low priority.

        self.post_numobs_more[0] = 0
        self.post_numobs_more[1] = 3
        self.post_numobs_more[2] = 3
        self.post_numobs_more[3] = 3
        self.post_numobs_more[4] = 3

    def test_mtl(self):
        """Test output from MTL has the correct column names.
        """
        # ADM loop through once each for the main survey, commissioning and SV.
        # t = self.reset_targets(prefix)
        mtl = make_mtl(self.targets, "GRAY|DARK", trimcols=True)
        mtldm = switch_main_cmx_or_sv(mtldatamodel, mtl)
        refnames = sorted(mtldm.dtype.names)
        mtlnames = sorted(mtl.dtype.names)
        self.assertEqual(refnames, mtlnames)

    def test_numobs(self):
        """Test priorities, numobs and obsconditions are set correctly with no zcat.
        """
        # ADM loop through once for SV and once for the main survey.
        mtl = make_mtl(self.targets, "GRAY|DARK")
        mtl.sort(keys='TARGETID')
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == [4, 4, 4, 4, 4]))
        self.assertTrue(np.all(mtl['PRIORITY'] == self.priorities))

    def test_zcat(self):
        """Test priorities, numobs and obsconditions are set correctly after zcat.
        """
        # ADM loop through once for SV and once for the main survey.
        mtl = make_mtl(self.targets, "DARK|GRAY", zcat=self.zcat, trim=False)
        mtl.sort(keys='TARGETID')
        self.assertTrue(np.all(mtl['PRIORITY'] == self.post_prio))
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == self.post_numobs_more))

    def test_numobs(self):
        """Check that LRGs and ELGs only request one observation.
        """
        # ADM How sources are prioritized is set purely by Z and ZWARN.
        # ADM So, we need to check that LRGs and ELGs only request one
        # ADM observation. If they request more than 1, then presumably
        # ADM they will have values of MORE_ZGOOD that exceed DONE. For
        # ADM dual targets that are both, say ELGs and QSOs, the 4 QSO
        # ADM observations could then adopt the high ELG priorities
        # ADM rather than the low MORE_MIDZQSO priorities. Basically, we
        # ADM need more careful logic if numobs > 1 for galaxies.
        for bitname in "LRG", "ELG":
            self.assertEqual(Mx[bitname].numobs, 1)


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_mtl
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
