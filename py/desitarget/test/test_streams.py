# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget units.
"""
import unittest
import yaml
import numpy as np
from importlib import resources

class TestSTREAMS(unittest.TestCase):

    def setUp(self):
        # ADM load the streams and dwarfs yaml files.
        fn = resources.files('desitarget').joinpath('data/streams.yaml')
        with open(fn) as f:
            self.streams = yaml.safe_load(f)
        fn = resources.files('desitarget').joinpath('data/dwarfs.yaml')
        with open(fn) as f:
            self.dwarfs = yaml.safe_load(f)

    def test_stream_ranks(self):
        """Test stream rankings aren't duplicated and increment by 1.
        """
        # ADM gather the ranks from the streams yaml file.
        ranks = []
        for stream in self.streams:
            ranks.append(self.streams[stream]['TARGMWEXT_RANK'])
        ranks = np.array(ranks)

        # ADM the expectation is that these ranks should increment by 1
        # ADM and always increase.
        expect_ranks = np.arange(len(ranks)) + 2

        # ADM check the ranks match with expectation.
        msg = f"Ranks in streams file are: {ranks} but should be {expect_ranks}"
        self.assertTrue(np.all(expect_ranks == ranks), msg=msg)

    def test_dwarf_ranks(self):
        """Test dwarf rankings are all 1.
        """
        # ADM gather the ranks from the dwarfs yaml file.
        ranks = []
        for dwarf in self.dwarfs:
            ranks.append(self.dwarfs[dwarf]['TARGMWEXT_RANK'])
        ranks = np.array(ranks)

        # ADM check all the ranks are 1.
        self.assertTrue(np.all(ranks==1))
