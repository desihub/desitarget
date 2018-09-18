# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.QA.
"""
import unittest
import os
import shutil
import tempfile
import numpy as np
from pkg_resources import resource_filename
from desitarget.QA import make_qa_page
from glob import glob

class TestQA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datadir = resource_filename('desitarget.test', 't/')
        cls.targfile = os.path.join(cls.datadir,'targets.fits')
        cls.cmxfile = os.path.join(cls.datadir,'cmx-targets.fits')
        cls.origdir = os.getcwd()
        cls.testdir = tempfile.mkdtemp()
        print("working in {}...".format(cls.testdir))
        os.chdir(cls.testdir)

        # ADM make absolutely sure that Tk is not the back-end.
        # ADM initially unset the matplotlib back-end.
        cls.mpl = os.environ.get('MPLBACKEND')
        print("setting matplotlib back-end to Agg...")
        os.environ["MPLBACKEND"] = 'Agg'

    @classmethod
    def tearDownClass(cls):
        #- Remove all test input and output files.
        os.chdir(cls.origdir)
        if os.path.exists(cls.testdir):
            shutil.rmtree(cls.testdir)

        # ADM reset the matplotlib back-end.
        print("setting matplotlib back-end to original value...")
        if cls.mpl is not None:
            os.environ["MPLBACKEND"] = cls.mpl

    def setUp(self):
        pass

    def tearDown(self):
        pass
        # ADM Remove the output files.
        for filelist in [glob("*png"), glob("*html"), glob("*dat")]:
            for filename in filelist:
                if os.path.exists(filename):
                    os.remove(filename)

    def test_qa_main(self):
        """Test plots/pages made for some main survey target types.
        """
        # ADM note that these might not all be in the test files
        # ADM but this also tests passing via tcnames.
        tcnames = ["ALL","BGS_FAINT"]

        make_qa_page(self.targfile, qadir=self.testdir, 
                     systematics=False, tcnames=tcnames)

        pngs, htmls = len(glob("*png")), len(glob("*html"))
        dats, alls = len(glob("*dat")), len(glob("./*"))
        
        # ADM one webpage is made per tc, plus the index.html.
        self.assertEqual(htmls, len(tcnames)+1)
        # ADM 4 N(m) plots are made per tc.
        self.assertEqual(dats, 4*len(tcnames))
        # ADM 11 plots are made per tc.
        self.assertEqual(pngs, 11*len(tcnames))
        # ADM there are only .html, .dat and .png files.
        self.assertEqual(pngs+htmls+dats, alls)

    def test_qa_cmx(self):
        """Test plots/pages are made for some commissioning targets.
        """

        make_qa_page(self.cmxfile, qadir=self.testdir, 
                     systematics=False)

        pngs, htmls = len(glob("*png")), len(glob("*html"))
        dats, alls = len(glob("*dat")), len(glob("./*"))
        
        self.assertEqual(pngs+htmls+dats, alls)

if __name__ == '__main__':
    unittest.main()

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m desitarget.test.test_qa
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)


