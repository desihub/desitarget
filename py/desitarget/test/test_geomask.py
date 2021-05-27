# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.geomask.
"""
import unittest
from pkg_resources import resource_filename
import numpy as np
import os

from desitarget import geomask


class TestGEOMASK(unittest.TestCase):

    @classmethod
    def setUp(self):
        drdir = '/blat/foo'  # doesn't have to exist, just for paths
        self.surveydir = os.path.join(drdir, 'decam')
        self.surveydir2 = os.path.join(drdir, '90prime-mosaic')

    def test_bundle_bricks(self):
        """
        Test the bundle_bricks scripting code executes without bugs
        """
        blat = geomask.bundle_bricks(1, 1, 1,
                                     surveydirs=[self.surveydir])
        self.assertTrue(blat is None)

        foo = geomask.bundle_bricks(1, 1, 1,
                                    surveydirs=[self.surveydir, self.surveydir2])
        self.assertTrue(foo is None)

    def test_match(self):
        a = np.array([1,2,3,4])
        b = np.array([4,3])
        iia, iib = geomask.match(a, b)

        #- returned indices match
        self.assertTrue(np.all(a[iia] == b[iib]))

        #- ... and are complete
        ainb = np.isin(a, b)
        bina = np.isin(b, a)
        self.assertTrue(np.all( np.isin(a[ainb], a[iia]) ) )
        self.assertTrue(np.all( np.isin(b[bina], b[iib]) ) )

        #- Check corner cases
        a = np.array([1,2,3,4])
        b = np.array([5,6])
        iia, iib = geomask.match(a, b)
        self.assertEqual(len(iia), 0)
        self.assertEqual(len(iib), 0)

        a = np.array([1,2,3,4])
        b = np.array([4,3,2,1])
        iia, iib = geomask.match(a, b)
        self.assertTrue(np.all(a[iia] == b[iib]))
        self.assertTrue(len(iia), len(a))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_geomask
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
