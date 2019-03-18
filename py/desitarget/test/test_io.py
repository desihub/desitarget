# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.io.
"""
import unittest
from pkg_resources import resource_filename
import os.path
from uuid import uuid4
from astropy.io import fits
import numpy as np
import fitsio

from desitarget import io


class TestIO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datadir = resource_filename('desitarget.test', 't')

    def setUp(self):
        self.testfile = 'test-{}.fits'.format(uuid4().hex)

    def tearDown(self):
        if os.path.exists(self.testfile):
            os.remove(self.testfile)

    def test_list_tractorfiles(self):
        files = io.list_tractorfiles(self.datadir)
        self.assertEqual(len(files), 3)
        for x in files:
            self.assertTrue(os.path.basename(x).startswith('tractor'))
            self.assertTrue(os.path.basename(x).endswith('.fits'))

    def test_list_sweepfiles(self):
        files = io.list_sweepfiles(self.datadir)
        self.assertEqual(len(files), 3)
        for x in files:
            self.assertTrue(os.path.basename(x).startswith('sweep'))
            self.assertTrue(os.path.basename(x).endswith('.fits'))

    def test_iter(self):
        for x in io.iter_files(self.datadir, prefix='tractor', ext='fits'):
            pass
        # - io.iter_files should also work with a file, not just a directory
        for y in io.iter_files(x, prefix='tractor', ext='fits'):
            self.assertEqual(x, y)

    def test_fix_dr1(self):
        '''test the DR1 TYPE dype fix (make everything S4)'''
        # - First, break it
        files = io.list_sweepfiles(self.datadir)
        objects = io.read_tractor(files[0])
        dt = objects.dtype.descr
        for i in range(len(dt)):
            if dt[i][0] == 'TYPE':
                dt[i] = ('TYPE', 'S10')
                break
        badobjects = objects.astype(np.dtype(dt))

        newobjects = io.fix_tractor_dr1_dtype(badobjects)
        self.assertEqual(newobjects['TYPE'].dtype, np.dtype('S4'))

    def test_tractor_columns(self):
        # ADM Gaia columns that get added on input.
        from desitarget.gaiamatch import gaiadatamodel
        from desitarget.gaiamatch import pop_gaia_coords, pop_gaia_columns
        # ADM have to remove the GAIA_RA, GAIA_DEC columns used for matching.
        gaiadatamodel = pop_gaia_coords(gaiadatamodel)
        # ADM prior to DR8, we're also missing some other Gaia columns.
        gaiadatamodel = pop_gaia_columns(
            gaiadatamodel,
            ['REF_CAT', 'GAIA_PHOT_BP_RP_EXCESS_FACTOR',
             'GAIA_ASTROMETRIC_SIGMA5D_MAX', 'GAIA_ASTROMETRIC_PARAMS_SOLVED']
        )
        # ADM BRICK_PRIMARY, PHOTSYS get added on input.
        tscolumns = list(io.tsdatamodel.dtype.names)     \
            + ['BRICK_PRIMARY', 'PHOTSYS']               \
            + list(gaiadatamodel.dtype.names)            \
            + list(io.dr8datamodel.dtype.names)
        tractorfile = io.list_tractorfiles(self.datadir)[0]
        data = io.read_tractor(tractorfile)
        self.assertEqual(set(data.dtype.names), set(tscolumns))
#        columns = ['BX', 'BY']
        columns = ['RA', 'DEC']
        data = io.read_tractor(tractorfile, columns=columns)
        self.assertEqual(set(data.dtype.names), set(columns))
        data = io.read_tractor(tractorfile, columns=tuple(columns))
        self.assertEqual(set(data.dtype.names), set(columns))

    def test_readwrite_tractor(self):
        tractorfile = io.list_tractorfiles(self.datadir)[0]
        sweepfile = io.list_sweepfiles(self.datadir)[0]
        data = io.read_tractor(sweepfile)
        self.assertEqual(len(data), 6)  # - test data has 6 objects per file
        data = io.read_tractor(tractorfile)
        self.assertEqual(len(data), 6)  # - test data has 6 objects per file
        data, hdr = io.read_tractor(tractorfile, header=True)
        self.assertEqual(len(data), 6)  # - test data has 6 objects per file

        # ADM check that input and output columns are the same.
        io.write_targets(self.testfile, data, indir=self.datadir)
        # ADM use fits read wrapper in io to correctly handle whitespace.
        d2, h2 = io.whitespace_fits_read(self.testfile, header=True)
        self.assertEqual(list(data.dtype.names), list(d2.dtype.names))

        # ADM check HPXPIXEL got added writing targets with NSIDE request.
        io.write_targets(self.testfile, data, nside=64, indir=self.datadir)
        # ADM use fits read wrapper in io to correctly handle whitespace.
        d2, h2 = io.whitespace_fits_read(self.testfile, header=True)
        self.assertEqual(list(data.dtype.names)+["HPXPIXEL"], list(d2.dtype.names))

        for column in data.dtype.names:
            self.assertTrue(np.all(data[column] == d2[column]))

    def test_brickname(self):
        self.assertEqual(io.brickname_from_filename('tractor-3301m002.fits'), '3301m002')
        self.assertEqual(io.brickname_from_filename('tractor-3301p002.fits'), '3301p002')
        self.assertEqual(io.brickname_from_filename('/a/b/tractor-3301p002.fits'), '3301p002')


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_io
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
