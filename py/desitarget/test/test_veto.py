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


class TestIO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datadir = str(resources.files("desitarget").joinpath("test/t/main"))

    def setUp(self):
        # ADM make an entire copy of the test MTL directory structure.
        self.testdir = "test-{}".format(uuid4().hex)
        shutil.copytree(self.datadir, self.testdir)
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
        self.assertTrue(type(data),  astropy.table.table.Table)

        # ADM check the bad data fails all of the ways.
        badmsg = "good so far!"
        try:
            data = io.read_mtl_veto_file(self.test_badfn)
        except ValueError as msg:
            badmsg = str(msg)

        errorcount = len(badmsg.split(" AND "))

        self.assertEqual(errorcount, 3)

