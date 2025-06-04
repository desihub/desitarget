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

    def test_ranks(self):
        """Test stream/dwarf rankings aren't duplicated and increment.
        """
        # ADM gather the ranks from the streams yaml file.
        stream_ranks = []
        for stream in self.streams:
            stream_ranks.append(self.streams[stream]['TARGMWEXT_RANK'])
        stream_ranks = np.array(stream_ranks)

        # ADM gather the ranks from the dwarfs yaml file.
        dwarf_ranks = []
        for dwarf in self.dwarfs:
            dwarf_ranks.append(self.dwarfs[dwarf]['TARGMWEXT_RANK'])
        dwarf_ranks = np.array(dwarf_ranks)

        # ADM the combined ranks, in order.
        ranks = sorted(np.concatenate([dwarf_ranks, stream_ranks]))

        # ADM the expectation is that these ranks should include 5 1s for
        # ADM the initial dwarf galaxies, and then any new ranks should
        # ADM always increment by 1.
        expect_ranks = np.concatenate([np.ones(5, dtype='int'),
                                       np.arange(2, len(ranks)-5+2, dtype='int')
                                       ])

        # ADM check the ranks match with expectation.
        msg = f"Ranks in streams file are: {ranks} but should be {expect_ranks}"
        self.assertTrue(np.all(expect_ranks == ranks), msg=msg)
