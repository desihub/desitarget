# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.cuts.
"""
import unittest
from pkg_resources import resource_filename
import os.path
from uuid import uuid4
import numbers

from astropy.io import fits
from astropy.table import Table
import fitsio
import numpy as np

from desitarget import io, cuts
from desitarget.targetmask import desi_mask


class TestCuts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datadir = resource_filename('desitarget.test', 't')
#        cls.gaiadir = resource_filename('desitarget.test', 'tgaia')
        cls.tractorfiles = sorted(io.list_tractorfiles(cls.datadir))
        cls.sweepfiles = sorted(io.list_sweepfiles(cls.datadir))

    def test_unextinct_fluxes(self):
        """Test function that unextincts fluxes
        """
        targets = io.read_tractor(self.tractorfiles[0])
        t1 = cuts.unextinct_fluxes(targets)
        self.assertTrue(isinstance(t1, np.ndarray))
        t2 = cuts.unextinct_fluxes(Table(targets))
        self.assertTrue(isinstance(t2, Table))
        for col in ['GFLUX', 'RFLUX', 'ZFLUX', 'W1FLUX', 'W2FLUX']:
            self.assertIn(col, t1.dtype.names)
            self.assertIn(col, t2.dtype.names)
            self.assertTrue(np.all(t1[col] == t2[col]))

    def test_cuts_basic(self):
        """Test cuts work with either data or filenames
        """
        # ADM only test the "BGS" class for speed.
        # ADM with one run of all target classes for coverage.
        tc = ["BGS"]
        desi, bgs, mws = cuts.apply_cuts(self.tractorfiles[0], tcnames=tc)
        desi, bgs, mws = cuts.apply_cuts(self.sweepfiles[0], tcnames=tc)
        data = io.read_tractor(self.tractorfiles[0])
        desi, bgs, mws = cuts.apply_cuts(data, tcnames=tc)
        data = io.read_tractor(self.sweepfiles[0])
        desi, bgs, mws = cuts.apply_cuts(data)

        bgs_any1 = (desi & desi_mask.BGS_ANY != 0)
        bgs_any2 = (bgs != 0)
        self.assertTrue(np.all(bgs_any1 == bgs_any2))

    def test_cuts_noprimary(self):
        """Test cuts work with or without "primary"
        """
        # - BRICK_PRIMARY was removed from the sweeps in dr3 (@moustakas).
        targets = Table.read(self.sweepfiles[0])
        if 'BRICK_PRIMARY' in targets.colnames:
            desi1, bgs1, mws1 = cuts.apply_cuts(targets)
            targets.remove_column('BRICK_PRIMARY')
            desi2, bgs2, mws2 = cuts.apply_cuts(targets)
            self.assertTrue(np.all(desi1 == desi2))
            self.assertTrue(np.all(bgs1 == bgs2))
            self.assertTrue(np.all(mws1 == mws2))

    def test_single_cuts(self):
        """Test cuts of individual target classes
        """
        targets = Table.read(self.sweepfiles[0])
        flux = cuts.unextinct_fluxes(targets)
        gflux = flux['GFLUX']
        rflux = flux['RFLUX']
        zflux = flux['ZFLUX']
        w1flux = flux['W1FLUX']
        w2flux = flux['W2FLUX']

        gfluxivar = targets['FLUX_IVAR_G']
        rfluxivar = targets['FLUX_IVAR_R']
        zfluxivar = targets['FLUX_IVAR_Z']

        gsnr = targets['FLUX_G'] * np.sqrt(targets['FLUX_IVAR_G'])
        rsnr = targets['FLUX_R'] * np.sqrt(targets['FLUX_IVAR_R'])
        zsnr = targets['FLUX_Z'] * np.sqrt(targets['FLUX_IVAR_Z'])
        w1snr = targets['FLUX_W1'] * np.sqrt(targets['FLUX_IVAR_W1'])
        w2snr = targets['FLUX_W2'] * np.sqrt(targets['FLUX_IVAR_W2'])

        dchisq = targets['DCHISQ']
        deltaChi2 = dchisq[..., 0] - dchisq[..., 1]

        gnobs, rnobs, znobs = targets['NOBS_G'], targets['NOBS_R'], targets['NOBS_Z']
        gallmask = targets['ALLMASK_G']
        rallmask = targets['ALLMASK_R']
        zallmask = targets['ALLMASK_Z']
        gfracflux = targets['FRACFLUX_G']
        rfracflux = targets['FRACFLUX_R']
        zfracflux = targets['FRACFLUX_Z']
        gfracmasked = targets['FRACMASKED_G']
        rfracmasked = targets['FRACMASKED_R']
        zfracmasked = targets['FRACMASKED_Z']
        gfracin = targets['FRACIN_G']
        rfracin = targets['FRACIN_R']
        zfracin = targets['FRACIN_Z']
        brightstarinblob = targets['BRIGHTSTARINBLOB']

        gaiagmag = targets['GAIA_PHOT_G_MEAN_MAG']
        Grr = gaiagmag - 22.5 + 2.5*np.log10(targets['FLUX_R'])

        if 'BRICK_PRIMARY' in targets.colnames:
            primary = targets['BRICK_PRIMARY']
        else:
            primary = np.ones_like(gflux, dtype='?')

        lrg1 = cuts.isLRG(primary=primary, gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                          gflux_ivar=gfluxivar, rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr)
        lrg2 = cuts.isLRG(primary=None, gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                          gflux_ivar=gfluxivar, rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr)
        self.assertTrue(np.all(lrg1 == lrg2))
        # ADM also check that the color selections alone work. This tripped us up once
        # ADM with the mocks part of the code calling a non-existent LRG colors function.
        lrg1 = cuts.isLRG_colors(primary=primary, gflux=gflux, rflux=rflux, zflux=zflux,
                                 w1flux=w1flux, w2flux=w2flux)
        lrg2 = cuts.isLRG_colors(primary=None, gflux=gflux, rflux=rflux, zflux=zflux,
                                 w1flux=w1flux, w2flux=w2flux)
        self.assertTrue(np.all(lrg1 == lrg2))

        elg1 = cuts.isELG(gflux=gflux, rflux=rflux, zflux=zflux,
                          gallmask=gallmask, rallmask=rallmask, zallmask=zallmask,
                          brightstarinblob=brightstarinblob, primary=primary)
        elg2 = cuts.isELG(gflux=gflux, rflux=rflux, zflux=zflux,
                          gallmask=gallmask, rallmask=rallmask, zallmask=zallmask,
                          brightstarinblob=brightstarinblob, primary=None)
        self.assertTrue(np.all(elg1 == elg2))

        elg1 = cuts.isELG_colors(gflux=gflux, rflux=rflux, zflux=zflux, primary=primary)
        elg2 = cuts.isELG_colors(gflux=gflux, rflux=rflux, zflux=zflux, primary=None)
        self.assertTrue(np.all(elg1 == elg2))

        for targtype in ["bright", "faint", "wise"]:
            bgs = []
            for primary in [primary, None]:
                bgs.append(
                    cuts.isBGS(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                               gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                               gfracmasked=gfracmasked, rfracmasked=rfracmasked, zfracmasked=zfracmasked,
                               gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                               gfracin=gfracin, rfracin=rfracin, zfracin=zfracin,
                               gfluxivar=gfluxivar, rfluxivar=rfluxivar, zfluxivar=zfluxivar,
                               brightstarinblob=brightstarinblob, Grr=Grr, w1snr=w1snr, gaiagmag=gaiagmag,
                               primary=primary, targtype=targtype)
                )
            self.assertTrue(np.all(bgs[0] == bgs[1]))

        # ADM need to include RELEASE for quasar cuts, at least.
        release = targets['RELEASE']
        # - Test that objtype and primary are optional
        psftype = targets['TYPE']
        qso1 = cuts.isQSO_cuts(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                               deltaChi2=deltaChi2, brightstarinblob=brightstarinblob,
                               w1snr=w1snr, w2snr=w2snr, objtype=psftype, primary=primary,
                               release=release)
        qso2 = cuts.isQSO_cuts(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                               deltaChi2=deltaChi2, brightstarinblob=brightstarinblob,
                               w1snr=w1snr, w2snr=w2snr, objtype=None, primary=None,
                               release=release)
        self.assertTrue(np.all(qso1 == qso2))
        # ADM also check that the color selections alone work. This tripped us up once
        # ADM with the mocks part of the code calling a non-existent LRG colors function.
        qso1 = cuts.isQSO_colors(gflux, rflux, zflux, w1flux, w2flux, optical=False)
        qso2 = cuts.isQSO_colors(gflux, rflux, zflux, w1flux, w2flux, optical=None)
        self.assertTrue(np.all(qso1 == qso2))

        fstd1 = cuts.isSTD_colors(gflux=gflux, rflux=rflux, zflux=zflux, primary=None)
        fstd2 = cuts.isSTD_colors(gflux=gflux, rflux=rflux, zflux=zflux, primary=primary)
        self.assertTrue(np.all(fstd1 == fstd2))

    def _test_table_row(self, targets):
        """Test cuts work with tables from several I/O libraries
        """
        # ADM only test the ELG cuts for speed. There's a
        # ADM full run through all classes in test_cuts_basic.
        tc = ["ELG"]

        self.assertFalse(cuts._is_row(targets))
        self.assertTrue(cuts._is_row(targets[0]))

        desi, bgs, mws = cuts.apply_cuts(targets, tcnames=tc)
        self.assertEqual(len(desi), len(targets))
        self.assertEqual(len(bgs), len(targets))
        self.assertEqual(len(mws), len(targets))

        desi, bgs, mws = cuts.apply_cuts(targets[0], tcnames=tc)
        self.assertTrue(isinstance(desi, numbers.Integral), 'DESI_TARGET mask not an int')
        self.assertTrue(isinstance(bgs, numbers.Integral), 'BGS_TARGET mask not an int')
        self.assertTrue(isinstance(mws, numbers.Integral), 'MWS_TARGET mask not an int')

    def test_astropy_fits(self):
        """Test astropy.fits I/O library
        """
        targets = fits.getdata(self.tractorfiles[0])
        self._test_table_row(targets)

    def test_astropy_table(self):
        """Test astropy tables I/O library
        """
        targets = Table.read(self.tractorfiles[0])
        self._test_table_row(targets)

    def test_numpy_ndarray(self):
        """Test fitsio I/O library
        """
        targets = fitsio.read(self.tractorfiles[0], upper=True)
        self._test_table_row(targets)

    def test_select_targets(self):
        """Test select targets works with either data or filenames
        """
        # ADM only test the ELG cuts for speed. There's a
        # ADM full run through all classes in test_cuts_basic.
        tc = ["LRG"]

        for filelist in [self.tractorfiles, self.sweepfiles]:
            targets = cuts.select_targets(filelist, numproc=1, tcnames=tc)
            t1 = cuts.select_targets(filelist[0:1], numproc=1, tcnames=tc)
            t2 = cuts.select_targets(filelist[0], numproc=1, tcnames=tc)
            for col in t1.dtype.names:
                try:
                    notNaN = ~np.isnan(t1[col])
                except TypeError:  # - can't check string columns for NaN
                    notNaN = np.ones(len(t1), dtype=bool)

                self.assertTrue(np.all(t1[col][notNaN] == t2[col][notNaN]))

    @unittest.skip("The sandbox isn't used much, we will probably deprecate it.")
    def test_select_targets_sandbox(self):
        """Test sandbox cuts at least don't crash
        """
        from desitarget import sandbox
        ntot = 0
        for filename in self.tractorfiles+self.sweepfiles:
            targets = cuts.select_targets(filename, numproc=1, sandbox=True)
            objects = Table.read(filename)
            if 'BRICK_PRIMARY' in objects.colnames:
                objects.remove_column('BRICK_PRIMARY')
            desi_target, bgs_target, mws_target = \
                sandbox.cuts.apply_sandbox_cuts(objects)
            n = np.count_nonzero(desi_target) + \
                np.count_nonzero(bgs_target) + \
                np.count_nonzero(mws_target)
            self.assertEqual(len(targets), n)
            ntot += n
        self.assertGreater(ntot, 0, 'No targets selected by sandbox.cuts')

    def test_check_targets(self):
        """Test code that checks files for corruption
        """
        for filelist in self.tractorfiles:
            nbadfiles = cuts.check_input_files(filelist, numproc=1)

            self.assertTrue(nbadfiles == 0)

    def test_qso_selection_options(self):
        """Test the QSO selection options are passed correctly
        """
        tc = ["QSO"]

        targetfile = self.tractorfiles[0]
        for qso_selection in cuts.qso_selection_options:
            results = cuts.select_targets(targetfile,
                                          tcnames=tc, qso_selection=qso_selection)

        with self.assertRaises(ValueError):
            results = cuts.select_targets(targetfile, numproc=1,
                                          tcnames=tc, qso_selection='blatfoo')

    def test_bgs_target_types(self):
        """Test that incorrect BGS target types are caught
        """
        with self.assertRaises(ValueError):
            dum = cuts.isBGS_colors(targtype='blatfoo')

        with self.assertRaises(ValueError):
            dum = cuts.notinBGS_mask(targtype='blatfoo')

    def test_missing_files(self):
        """Test the code will die gracefully if input files are missing
        """
        with self.assertRaises(ValueError):
            targets = cuts.select_targets(['blat.foo1234', ], numproc=1)

    def test_parallel_select(self):
        """Test multiprocessing parallelization works
        """
        # ADM only test the ELG, BGS cuts for speed. There's a
        # ADM full run through all classes in test_cuts_basic.
        tc = ["ELG", "BGS"]

        for nproc in [1, 2]:
            for filelist in [self.tractorfiles, self.sweepfiles]:
                targets = cuts.select_targets(filelist,
                                              numproc=nproc, tcnames=tc)
                self.assertTrue('DESI_TARGET' in targets.dtype.names)
                self.assertTrue('BGS_TARGET' in targets.dtype.names)
                self.assertTrue('MWS_TARGET' in targets.dtype.names)
                self.assertEqual(len(targets), np.count_nonzero(targets['DESI_TARGET']))

                bgs1 = (targets['DESI_TARGET'] & desi_mask.BGS_ANY) != 0
                bgs2 = targets['BGS_TARGET'] != 0
                self.assertTrue(np.all(bgs1 == bgs2))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_cuts
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
