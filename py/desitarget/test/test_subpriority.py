# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.subpriority
"""
import unittest
from pkg_resources import resource_filename
import numpy as np
import os
from astropy.table import Table

import desitarget.subpriority

class TestSubpriority(unittest.TestCase):

    @classmethod
    def setUp(self):
        pass

    def test_override(self):
        n = 10
        targets = Table()
        ids = np.arange(n)
        np.random.shuffle(ids)
        targets['TARGETID'] = ids
        orig_subpriority = np.random.random(n)
        targets['SUBPRIORITY'] = orig_subpriority.copy()

        override = Table()
        override['TARGETID'] = np.array([3,2,20])
        override['SUBPRIORITY'] = np.array([10.0, 20.0, 30.0])

        desitarget.subpriority.override_subpriority(targets, override)

        #- Check that we overrode correctly; don't juse geomask.match
        #- to avoid circularity of code and test
        for i, tid in enumerate(targets['TARGETID']):
            in_override = np.where(override['TARGETID'] == tid)[0]
            if len(in_override) > 0:
                j = in_override[0]
                self.assertEqual(targets['SUBPRIORITY'][i], override['SUBPRIORITY'][j])
            else:
                self.assertEqual(targets['SUBPRIORITY'][i], orig_subpriority[i])


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_geomask
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
