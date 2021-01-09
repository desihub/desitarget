# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.sv.
"""
import unittest
import fitsio
import os
import numpy as np
import healpy as hp
from glob import glob

from pkg_resources import resource_filename
from desitarget import io, cuts
from desitarget.targetmask import desi_mask


class TestSV(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datadir = resource_filename('desitarget.test', 't')
        cls.tractorfiles = sorted(io.list_tractorfiles(cls.datadir))
        cls.sweepfiles = sorted(io.list_sweepfiles(cls.datadir))

        # ADM find which HEALPixels are covered by test sweeps files.
        cls.nside = 32
        pixlist = []
        for fn in cls.sweepfiles:
            objs = fitsio.read(fn)
            theta, phi = np.radians(90-objs["DEC"]), np.radians(objs["RA"])
            pixels = hp.ang2pix(cls.nside, theta, phi, nest=True)
            pixlist.append(pixels)
        cls.pix = np.unique(pixlist)

        # ADM set up the GAIA_DIR environment variable.
        cls.gaiadir_orig = os.getenv("GAIA_DIR")
        os.environ["GAIA_DIR"] = resource_filename('desitarget.test', 't4')

    @classmethod
    def tearDownClass(cls):
        # ADM reset GAIA_DIR environment variable.
        if cls.gaiadir_orig is not None:
            os.environ["GAIA_DIR"] = cls.gaiadir_orig

    def test_sv_cuts(self):
        """Test SV cuts work.
        """
        # ADM find all svX sub-directories in the desitarget directory.
        fns = glob(resource_filename('desitarget', 'sv*'))
        svlist = [os.path.basename(fn) for fn in fns if os.path.isdir(fn)]

        for survey in svlist:
            desicol, bgscol, mwscol = ["{}_{}_TARGET".format(survey.upper(), tc)
                                       for tc in ["DESI", "BGS", "MWS"]]
            for filelist in [self.tractorfiles, self.sweepfiles]:
                # ADM set backup to False as the Gaia unit test
                # ADM files only cover a limited pixel range.
                targets = cuts.select_targets(filelist, survey=survey,
                                              backup=False, test=True)
                for col in desicol, bgscol, mwscol:
                    self.assertTrue(col in targets.dtype.names)
                self.assertEqual(len(targets), np.count_nonzero(targets[desicol]))

                # ADM this test should be fine as long as the main survey BGS
                # ADM bits don't get divorced from the SV survey bits.
                bgs1 = (targets[desicol] & desi_mask.BGS_ANY) != 0
                bgs2 = targets[bgscol] != 0
                self.assertTrue(np.all(bgs1 == bgs2))

            # ADM backup targets can only be run on sweep files.
            for filelist in self.sweepfiles:
                # ADM also test the backup targets in the pixels covered
                # ADM by the Gaia unit test files.
                targets = cuts.select_targets(filelist, survey=survey, test=True,
                                              nside=self.nside, pixlist=self.pix)
                for col in desicol, bgscol, mwscol:
                    self.assertTrue(col in targets.dtype.names)
                self.assertEqual(len(targets), np.count_nonzero(targets[desicol]))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_sv
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
