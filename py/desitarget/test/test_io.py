# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.io.
"""
import unittest
from importlib import resources
import shutil
import os.path
from uuid import uuid4
from astropy.io import fits
import numpy as np
import fitsio

from desitarget import io
from desitarget.targetmask import obsconditions
from desitarget import __version__


class TestIO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datadir = resources.files('desitarget').joinpath('test/t')

    def setUp(self):
        self.testdir = 'test-{}'.format(uuid4().hex)

    def tearDown(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir, ignore_errors=True)

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
        # ADM remove the GAIA_RA, GAIA_DEC columns used for matching.
        gaiadatamodel = pop_gaia_coords(gaiadatamodel)
        tractorfile = io.list_tractorfiles(self.datadir)[0]
        data = io.read_tractor(tractorfile)
        # ADM form the final data model in a manner that maintains
        # ADM backwards-compatability with DR8.
        if "FRACDEV" in data.dtype.names:
            tsdatamodel = np.array(
                [], dtype=io.basetsdatamodel.dtype.descr +
                io.dr8addedcols.dtype.descr)
        else:
            tsdatamodel = np.array(
                [], dtype=io.basetsdatamodel.dtype.descr +
                io.dr9addedcols.dtype.descr)
        # ADM PHOTSYS gets added on input.
        tscolumns = list(tsdatamodel.dtype.names)           \
            + ['PHOTSYS']                                   \
            + list(gaiadatamodel.dtype.names)
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
        _, filename = io.write_targets(self.testdir, data, indir=self.datadir)
        # ADM use fits read wrapper in io to correctly handle whitespace.
        d2, h2 = io.whitespace_fits_read(filename, header=True)
        self.assertEqual(list(data.dtype.names), list(d2.dtype.names))

        # ADM check HPXPIXEL got added writing targets with NSIDE request.
        _, filename = io.write_targets(self.testdir, data, nside=64,
                                       indir=self.datadir)
        # ADM use fits read wrapper in io to correctly handle whitespace.
        d2, h2 = io.whitespace_fits_read(filename, header=True)
        self.assertEqual(list(data.dtype.names)+["HPXPIXEL"], list(d2.dtype.names))
        for column in data.dtype.names:
            kind = data[column].dtype.kind
            # ADM whitespace_fits_read() doesn't convert string data
            # ADM types identically for every version of fitsio.
            if kind == 'U' or kind == 'S':
                self.assertTrue(np.all(data[column] == d2[column].astype(data[column].dtype)))
            else:
                self.assertTrue(np.all(data[column] == d2[column]))

    def test_brickname(self):
        self.assertEqual(io.brickname_from_filename('tractor-3301m002.fits'), '3301m002')
        self.assertEqual(io.brickname_from_filename('tractor-3301p002.fits'), '3301p002')
        self.assertEqual(io.brickname_from_filename('/a/b/tractor-3301p002.fits'), '3301p002')

    def test_write_targets(self):
        """test io.write_targets, including SUBPRIORITY overrides
        """
        from desimodel.footprint import radec2pix
        from astropy.table import Table

        targets = Table()
        targets['TARGETID'] = [1, 2, 3, 4, 5, 6]
        D = obsconditions.DARK
        G = obsconditions.GRAY
        B = obsconditions.BRIGHT
        targets['OBSCONDITIONS'] = [D, D, G, G, B, B]
        iidark = targets['OBSCONDITIONS'] == obsconditions.DARK
        iibright = targets['OBSCONDITIONS'] == obsconditions.BRIGHT

        # - some RA,DEC within nested nside=8 healpix 123
        targets['RA'] = np.random.uniform(105, 106, len(targets))
        targets['DEC'] = np.random.uniform(72, 73, len(targets))
        nside = 8
        hpix = 123

        # - Set some but not all SUBPRIORITY
        targets['SUBPRIORITY'] = np.zeros(len(targets))
        targets['SUBPRIORITY'][0::2] = 2.0

        targets = np.asarray(targets)

        io.write_targets(self.testdir, targets, obscon='DARK',
                         nsidefile=nside, hpxlist=[hpix, ])
        io.write_targets(self.testdir, targets, obscon='BRIGHT',
                         nsidefile=nside, hpxlist=[hpix, ])

        darkfile = f'{self.testdir}/drX/{__version__}/targets/main/resolve/dark/targets-dark-hp-{hpix}.fits'
        brightfile = f'{self.testdir}/drX/{__version__}/targets/main/resolve/bright/targets-bright-hp-{hpix}.fits'

        self.assertTrue(os.path.exists(darkfile))
        self.assertTrue(os.path.exists(brightfile))

        # - Each dark,bright file should only have the targets for that obscon
        # - and should only override the zero SUBPRIORITY
        dt = Table.read(darkfile)
        self.assertEqual(len(dt), 2)
        self.assertTrue(np.all(dt['TARGETID'] == targets['TARGETID'][iidark]))
        self.assertTrue(np.all(dt['SUBPRIORITY'] > 0.0))
        self.assertEqual(dt['SUBPRIORITY'][0], 2.0)
        self.assertNotEqual(dt['SUBPRIORITY'][1], 0.0)

        bt = Table.read(brightfile)
        self.assertEqual(len(bt), 2)
        self.assertTrue(np.all(bt['TARGETID'] == targets['TARGETID'][iibright]))
        self.assertTrue(np.all(bt['SUBPRIORITY'] > 0.0))
        self.assertEqual(bt['SUBPRIORITY'][0], 2.0)
        self.assertNotEqual(bt['SUBPRIORITY'][1], 0.0)


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_io
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
