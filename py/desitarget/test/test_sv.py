# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.sv.
"""
import unittest
import numpy as np

from pkg_resources import resource_filename
from desitarget import io, cuts
from desitarget.targetmask import desi_mask


class TestSV(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datadir = resource_filename('desitarget.test', 't')
        cls.tractorfiles = sorted(io.list_tractorfiles(cls.datadir))
        cls.sweepfiles = sorted(io.list_sweepfiles(cls.datadir))

    def test_sv_cuts(self):
        """Test SV cuts work.
        """
        for filelist in [self.tractorfiles, self.sweepfiles]:
            # ADM increase maxsv as we add more iterations of SV!!!
            maxsv = 1
            svlist = ['sv{}'.format(i) for i in range(1, maxsv+1)]
            for survey in svlist:
                desicol, bgscol, mwscol = ["{}_{}_TARGET".format(survey.upper(), tc)
                                           for tc in ["DESI", "BGS", "MWS"]]
                targets = cuts.select_targets(filelist, survey=survey)
                for col in desicol, bgscol, mwscol:
                    self.assertTrue(col in targets.dtype.names)
                self.assertEqual(len(targets), np.count_nonzero(targets[desicol]))

                # ADM this test should be fine as long as the main survey BGS
                # ADM bits don't get divorced from the SV survey bits.
                bgs1 = (targets[desicol] & desi_mask.BGS_ANY) != 0
                bgs2 = targets[bgscol] != 0
                self.assertTrue(np.all(bgs1 == bgs2))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_sv
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
