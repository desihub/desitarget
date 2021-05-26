# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.QA.
"""
import unittest
import os
import shutil
import tempfile
import warnings
import numpy as np
import healpy as hp
from pkg_resources import resource_filename
from glob import glob
from desitarget.QA import make_qa_page, _load_systematics
from desitarget.QA import _parse_tcnames, _in_desi_footprint
from desiutil.log import get_logger
log = get_logger()


class TestQA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datadir = resource_filename('desitarget.test', 't/')
        cls.targfile = os.path.join(cls.datadir, 'targets.fits')
        cls.mocktargfile = os.path.join(cls.datadir, 'targets-mocks.fits')
        cls.cmxfile = os.path.join(cls.datadir, 'cmx-targets.fits')
        cls.pixmapfile = os.path.join(cls.datadir, 'pixweight.fits')
        cls.origdir = os.getcwd()
        cls.testdir = tempfile.mkdtemp()
        log.info("working in {}...".format(cls.testdir))
        os.chdir(cls.testdir)

    @classmethod
    def tearDownClass(cls):
        # - Remove all test input and output files.
        os.chdir(cls.origdir)
        if os.path.exists(cls.testdir):
            shutil.rmtree(cls.testdir)

    def setUp(self):
        # Treat some specific warnings as errors so that we can find and fix
        # warnings.filterwarnings('error', '.*Mean of empty slice.*')
        # warnings.filterwarnings('error', '.*Using or importing the ABCs.*')
        warnings.filterwarnings('error', '.*invalid value encountered.*')

        # SJB Always make sure we start in the test directory
        os.chdir(self.testdir)

    def tearDown(self):
        # ADM Remove the output files.
        # SJB only in testdir, just in case something else did a chdir
        os.chdir(self.testdir)
        for filelist in [sorted(glob("*png")), sorted(glob("*html")), sorted(glob("*dat"))]:
            for filename in filelist:
                if os.path.exists(filename):
                    os.remove(filename)

    def test_qa_main(self):
        """Test plots/pages made for some main survey target types.
        """
        # ADM note that these might not all be in the test files
        # ADM but this also tests passing via tcnames.
        tcnames = ["ALL", "BGS_FAINT"]

        # ADM the large max_bin_area helps speed the tests.
        make_qa_page(self.targfile, qadir=self.testdir, max_bin_area=99.,
                     imaging_map_file=self.pixmapfile, tcnames=tcnames)

        pngs, htmls = len(glob("*png")), len(glob("*html"))
        dats, alls = len(glob("*dat")), len(glob("./*"))
        sysplots = len(_load_systematics())

        # ADM one webpage is made per tc, plus the index.html.
        self.assertEqual(htmls, len(tcnames)+1)
        # ADM 4 N(m) plots are made per tc.
        self.assertEqual(dats, 4*len(tcnames))
        # ADM 11 plots made per tc. plus 2 lots of systematics plots.
        self.assertEqual(pngs, 11*len(tcnames)+2*sysplots)
        # ADM there are only .html, .dat and .png files.
        self.assertEqual(pngs+htmls+dats, alls)

    def test_qa_cmx(self):
        """Test plots/pages are made for some commissioning targets.
        """
        # ADM the large max_bin_area helps speed the tests.
        make_qa_page(self.cmxfile, qadir=self.testdir, max_bin_area=99.,
                     systematics=False)

        pngs, htmls = len(glob("*png")), len(glob("*html"))
        dats, alls = len(glob("*dat")), len(glob("./*"))

        # ADM there are only .html, .dat and .png files.
        self.assertEqual(pngs+htmls+dats, alls)

    def test_qa_mocks(self):
        """Test mock QA plots/pages
        """
        make_qa_page(self.mocktargfile, qadir=self.testdir,
                     makeplots=True, numproc=1, mocks=True,
                     max_bin_area=53.7148, systematics=False)

        pngs, htmls = len(glob("*png")), len(glob("*html"))
        dats, alls = len(glob("*dat")), len(glob("./*"))

        # pngs, htmls, and dats exist, and nothing else
        self.assertGreater(pngs, 0)
        self.assertGreater(htmls, 0)
        self.assertGreater(dats, 0)
        self.assertEqual(pngs+htmls+dats, alls)

    def test_parse_tc_names(self):
        """Test target class strings are parsed into lists.
        """
        # ADM the defaults list of target classes without "ALL".
        no_all = _parse_tcnames(add_all=False)
        # ADM passing the string instead of defaulting.
        no_all2 = _parse_tcnames(tcstring=",".join(no_all), add_all=False)
        # ADM the default list of target classes with "ALL".
        with_all = _parse_tcnames()
        # ADM you shouldn't be able to pass gobbledygook.
        failed = False
        try:
            fooblat = _parse_tcnames(tcstring="blat,foo")
        except ValueError:
            failed = True

        self.assertTrue(no_all == no_all2)
        self.assertTrue(set(with_all)-set(no_all) == {'ALL'})
        self.assertTrue(failed)

    def test_in_footprint(self):
        """Test target class strings are parsed into lists.
        """
        # ADM a location that's definitely in DESI (38.5,7.5).
        targs = np.zeros(1, dtype=[('RA', '>f8'), ('DEC', '>f8')])
        targs["RA"], targs["DEC"] = 38.5, 7.5
        tin = _in_desi_footprint(targs)
        # ADM shift to a location definitely out of DESI (38.5,-60).
        targs["DEC"] = -60.
        tout = _in_desi_footprint(targs)

        self.assertEqual(len(tin[0]), 1)
        self.assertEqual(len(tout[0]), 0)


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_qa
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
