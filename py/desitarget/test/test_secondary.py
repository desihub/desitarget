# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget secondary targets.
"""
import os
import numpy as np
import unittest
from importlib import import_module
from glob import glob
from importlib import resources


class TestSECONDARY(unittest.TestCase):

    def setUp(self):
        # ADM these are the allowed types of observations of secondaries.
        self.flavors = {"SPARE", "DEDICATED", "SSV", "QSO", "TOO"}
        # ADM this is the list of defined directories for SV.
        fns = sorted(glob(str(resources.files('desitarget').joinpath('sv*'))))
        svlist = [os.path.basename(fn) for fn in fns if os.path.isdir(fn)]
        # ADM loop over all SV flavors and main survey to get all masks.
        from desitarget.targetmask import scnd_mask as Mx
        self.Mxs = [Mx]
        for sv in svlist:
            targmask = import_module(
                "desitarget.{}.{}_targetmask".format(sv, sv))
            self.Mxs.append(targmask.scnd_mask)

    def test_flavors(self):
        """Test that secondary masks only have the allowed flavors.
        """
        # ADM for each mask...
        self.assertEqual(len(self.Mxs), 4)
        for Mx in self.Mxs:
            # ADM ...if we've already defined the flavor property...
            if "flavor" in dir(Mx[Mx[0]]):
                flavs = set([Mx[bitname].flavor for bitname in Mx.names()])
                # ADM ...all of the included flavors are allowed flavors.
                self.assertTrue(flavs.issubset(self.flavors))

    def test_updatemws(self):
        """Test that secondary masks have updatemws=True|False
        """
        # ADM for each mask...
        for Mx in self.Mxs:
            # ADM ...if we've already defined the updatemws property...
            if "updatemws" in dir(Mx[Mx[0]]):
                for bitname in Mx.names():
                    self.assertTrue(isinstance(Mx[bitname].updatemws, bool))

    def test_downsample(self):
        """Test that secondary masks all have downsample defined and <= 1.
        """
        # ADM for each mask...
        for Mx in self.Mxs:
            try:
                # ADM create a list of all of the downsample values.
                ds = []
                for bitname in Mx.names():
                    if "TOO" not in bitname:
                        ds.append(Mx[bitname].downsample)
            except AttributeError:
                # ADM check downsample is defined for all scnd_mask bits.
                msg = "downsample missing for bit {} in mask {}".format(
                    bitname, Mx._name)
                raise AttributeError(msg)
        # ADM check downsample is always less than 1 (< 100%).
        self.assertTrue(np.all(np.array(ds) <= 1))

