# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.targets.calc_numobs.
"""
import unittest
import numpy as np
from astropy.table import Table

from desitarget.targetmask import desi_mask
from desitarget.targets import calc_numobs


class TestNumObs(unittest.TestCase):

    def setUp(self):
        dtype = [
            ('DESI_TARGET', np.int64),
            ('BGS_TARGET', np.int64),
            ('MWS_TARGET', np.int64),
            ('NUMOBS', np.int32),
        ]
        self.targets = np.zeros(5, dtype=dtype)

    def test_numobs(self):
        t = self.targets

        # - No target bits set is an error
        with self.assertRaises(ValueError):
            calc_numobs(t)

        # - ELGs and QSOs get one/four observations
        t['DESI_TARGET'] = desi_mask.ELG
        self.assertTrue(np.all(calc_numobs(t) == 1))
        t['DESI_TARGET'] = desi_mask.QSO
        self.assertTrue(np.all(calc_numobs(t) == 4))

        # ADM LRG NUMOBS is defined using per-pass target bits
        # ADM the desi_mask.LRG reference tests the default, which
        # ADM should correspond to 2 observations
        t['DESI_TARGET'] = [desi_mask.LRG,
                            desi_mask.LRG_1PASS, desi_mask.LRG_2PASS,
                            desi_mask.LRG_1PASS, desi_mask.LRG_2PASS]
        nobs = calc_numobs(t)
        self.assertTrue(np.all(nobs == [2, 1, 2, 1, 2]))

        # - test astropy Table
        t = Table(t)
        nobs = calc_numobs(t)
        self.assertTrue(np.all(nobs == [2, 1, 2, 1, 2]))

        # - this is true even if other targeting bits are set
        t['DESI_TARGET'] |= desi_mask.mask('ELG|BGS_ANY')
        nobs = calc_numobs(t)
        self.assertTrue(np.all(nobs == [2, 1, 2, 1, 2]))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_numobs
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
