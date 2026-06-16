# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test veto aspects of desitarget.io and desitarget.mtl.
"""
import unittest
import shutil
import os
import astropy
from importlib import resources
from uuid import uuid4
import numpy as np

from desitarget import io
from desitarget.mtl import process_vetoes


class TestVETO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datadir = str(resources.files("desitarget").joinpath("test/t/main"))

    def setUp(self):
        # ADM make an entire copy of the test MTL directory structure.
        self.testdir = "test-{}".format(uuid4().hex)
        shutil.copytree(self.datadir, os.path.join(self.testdir, "main"))
        # ADM we need to remove the bad test file from the MTL mock
        # ADM directory, otherwise updates will fail on that file.
        shutil.rmtree(os.path.join(self.testdir,
                                   "main", "veto", "bright1b", "bad"))
        self.test_goodfn = os.path.join(
            self.datadir, "veto/bright1b/good/good-veto-bright1b.ecsv")
        self.test_badfn = os.path.join(
            self.datadir, "veto/bright1b/bad/bad-veto-bright1b.ecsv")

    def tearDown(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir, ignore_errors=True)

    def test_veto_file(self):
        """Test a good veto file can be read but a bad one can't"""
        # ADM check the good data reads as expected.
        data = io.read_mtl_veto_file(self.test_goodfn)
        self.assertTrue(type(data), astropy.table.table.Table)

        # ADM check the bad data fails all of the ways.
        badmsg = "good so far!"
        try:
            data = io.read_mtl_veto_file(self.test_badfn)
        except ValueError as msg:
            badmsg = str(msg)

        errorcount = len(badmsg.split(" AND "))
        print('adasdas')
        self.assertEqual(errorcount, 3)

    def test_process_vetoes(self):
        """Test the end-to-end veto process"""
        # ADM Process the vetos working in the test directory.
        process_vetoes("bright1b", survey="main", mtldir=self.testdir)

        amtl1 = io.read_mtl_ledger(os.path.join(self.testdir, "main", "bright1b",
                                                "mtl-bright1b-hp-661.ecsv"),
                                   initial=True)
        amtl2 = io.read_mtl_ledger(os.path.join(self.testdir, "main", "bright1b",
                                                "mtl-bright1b-hp-704.ecsv"),
                                   initial=True)
        bmtl1 = io.read_mtl_ledger(os.path.join(self.testdir, "main", "bright1b",
                                                "mtl-bright1b-hp-661.ecsv"))
        bmtl2 = io.read_mtl_ledger(os.path.join(self.testdir, "main", "bright1b",
                                                "mtl-bright1b-hp-704.ecsv"))

        # ADM test all the initial states...
        self.assertTrue(np.all(amtl1["TARGET_STATE"] == ["M31_GIANT|UNOBS"]))
        self.assertTrue(np.all(amtl2["TARGET_STATE"] == ["M31_GIANT|UNOBS"]))
        # ADM ...have turned into the vetoed states.
        self.assertTrue(np.all(bmtl1["TARGET_STATE"] == ["VETO|DONE"]))
        self.assertTrue(np.all(bmtl2["TARGET_STATE"] == ["VETO|DONE"]))

        # ADM also check that the final priorities are the DONE priority
        # ADM and the initial priorities were higher than that.
        # ADM there are two objects in the first file, so the sum is 4.
        self.assertTrue(np.sum(amtl1["PRIORITY"]) > 4)
        self.assertTrue(np.sum(amtl2["PRIORITY"]) > 2)
        self.assertEqual(np.sum(bmtl1["PRIORITY"]), 4)
        self.assertEqual(np.sum(bmtl2["PRIORITY"]), 2)

        # ADM also check that no more observations are required
        # ADM but that more observations were originally required.
        self.assertTrue(np.all(amtl1["NUMOBS_MORE"] > 0))
        self.assertTrue(np.sum(amtl2["NUMOBS_MORE"] > 0))
        self.assertEqual(np.sum(bmtl1["NUMOBS_MORE"]), 0)
        self.assertEqual(np.sum(bmtl2["NUMOBS_MORE"]), 0)

        # ADM finally, check the mtl-done-vetoes.ecsv file is made.
        self.assertTrue('mtl-done-vetoes.ecsv' in os.listdir(self.testdir))
