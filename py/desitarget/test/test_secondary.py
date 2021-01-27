# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.mtl.
"""
import unittest
from importlib import import_module

class TestSECONDARY(unittest.TestCase):

    def setUp(self):
        # ADM these are the allowed types of observations of secondaries.
        self.flavors = {"SPARE", "DEDICATED", "SSV", "QSO", "TOO"}

    def test_flavors(self):
        """Test that secondary masks only have the allowed flavors.
        """
        # ADM loop over flavors of SV and the main survey to get masks.
        from desitarget.targetmask import scnd_mask as Mx
        Mxs = [Mx]
        svs = 2
        for i in range(1, svs):
            targmask = import_module(
                "desitarget.sv{}.sv{}_targetmask".format(i, i))
            Mxs.append(targmask.scnd_mask)

        # ADM for each mask...
        for Mx in Mxs:
            # ADM ...if we've already defined the flavor property...
            if "flavor" in dir(Mx[Mx[0]]):
                flavs =set([Mx[bitname].flavor for bitname in Mx.names()])
                # ADM ...all of the included flavors are allowed flavors.
                self.assertTrue(flavs.issubset(self.flavors))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_mtl
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
