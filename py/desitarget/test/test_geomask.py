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
        drdir = '/global/project/projectdirs/cosmo/work/legacysurvey/dr8b/'
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


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_geomask
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
