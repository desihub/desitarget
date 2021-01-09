# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.cuts.
"""
import unittest
from pkg_resources import resource_filename
import os.path
from uuid import uuid4
import numbers
import warnings

from astropy.io import fits
from astropy.table import Table
import fitsio
import numpy as np
import healpy as hp

from desitarget import io, cuts
from desitarget.targetmask import desi_mask
from desitarget.geomask import hp_in_box, pixarea2nside, box_area


class TestCuts(unittest.TestCase):

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

    def setUp(self):
        # treat some specific warnings as errors so we can find and fix
        # (could turn off if this becomes problematic)
        warnings.filterwarnings('error', '.*Calling nonzero on 0d arrays.*')

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
        zfiberflux = flux['ZFIBERFLUX']
        rfiberflux = flux['RFIBERFLUX']

        gfluxivar = targets['FLUX_IVAR_G']
        rfluxivar = targets['FLUX_IVAR_R']
        zfluxivar = targets['FLUX_IVAR_Z']
        w1fluxivar = targets['FLUX_IVAR_W1']

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
        maskbits = targets['MASKBITS']

        gaiagmag = targets['GAIA_PHOT_G_MEAN_MAG']
        Grr = gaiagmag - 22.5 + 2.5*np.log10(targets['FLUX_R'])

        if 'BRICK_PRIMARY' in targets.colnames:
            primary = targets['BRICK_PRIMARY']
        else:
            primary = np.ones_like(gflux, dtype='?')

        # ADM check for both defined fiberflux and fiberflux of None.
        for ff in zfiberflux, None:
            lrg1 = cuts.isLRG(primary=primary, gflux=gflux, rflux=rflux,
                              zflux=zflux, w1flux=w1flux, zfiberflux=ff,
                              gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                              maskbits=maskbits, rfluxivar=rfluxivar,
                              zfluxivar=zfluxivar, w1fluxivar=w1fluxivar)
            lrg2 = cuts.isLRG(primary=None, gflux=gflux, rflux=rflux, zflux=zflux,
                              w1flux=w1flux, zfiberflux=ff,
                              gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                              maskbits=maskbits, rfluxivar=rfluxivar,
                              zfluxivar=zfluxivar, w1fluxivar=w1fluxivar)

            self.assertTrue(np.all(lrg1 == lrg2))

            # ADM also check that the color selections alone work. This tripped us up once
            # ADM with the mocks part of the code calling a non-existent LRG colors function.
            lrg1 = cuts.isLRG_colors(primary=primary, gflux=gflux, rflux=rflux,
                                     zflux=zflux, zfiberflux=ff,
                                     w1flux=w1flux, w2flux=w2flux)
            lrg2 = cuts.isLRG_colors(primary=None, gflux=gflux, rflux=rflux,
                                     zflux=zflux, zfiberflux=ff,
                                     w1flux=w1flux, w2flux=w2flux)
            self.assertTrue(np.all(lrg1 == lrg2))

        elg1 = cuts.isELG(gflux=gflux, rflux=rflux, zflux=zflux,
                          gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                          gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                          maskbits=maskbits, primary=primary)
        elg2 = cuts.isELG(gflux=gflux, rflux=rflux, zflux=zflux,
                          gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                          gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                          maskbits=maskbits, primary=None)
        self.assertTrue(np.all(elg1 == elg2))

        elg1 = cuts.isELG_colors(gflux=gflux, rflux=rflux, zflux=zflux, primary=primary)
        elg2 = cuts.isELG_colors(gflux=gflux, rflux=rflux, zflux=zflux, primary=None)
        self.assertTrue(np.all(elg1 == elg2))

        # ADM check for both defined fiberflux and fiberflux of None.
        for ff in rfiberflux, None:
            for targtype in ["bright", "faint", "wise"]:
                bgs = []
                for prim in [primary, None]:
                    bgs.append(
                        cuts.isBGS(
                            rfiberflux=ff, gflux=gflux, rflux=rflux,
                            zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                            gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                            gfracmasked=gfracmasked, rfracmasked=rfracmasked,
                            zfracmasked=zfracmasked, gfracflux=gfracflux,
                            rfracflux=rfracflux, zfracflux=zfracflux,
                            gfracin=gfracin, rfracin=rfracin, zfracin=zfracin,
                            gfluxivar=gfluxivar, rfluxivar=rfluxivar,
                            zfluxivar=zfluxivar, maskbits=maskbits,
                            Grr=Grr, w1snr=w1snr, gaiagmag=gaiagmag,
                            primary=prim, targtype=targtype)
                    )
                self.assertTrue(np.all(bgs[0] == bgs[1]))

        # ADM need to include RELEASE for quasar cuts, at least.
        release = targets['RELEASE']
        # - Test that objtype and primary are optional
        psftype = targets['TYPE']
        qso1 = cuts.isQSO_cuts(gflux=gflux, rflux=rflux, zflux=zflux,
                               w1flux=w1flux, w2flux=w2flux,
                               gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                               deltaChi2=deltaChi2, maskbits=maskbits,
                               w1snr=w1snr, w2snr=w2snr, objtype=psftype, primary=primary,
                               release=release)
        qso2 = cuts.isQSO_cuts(gflux=gflux, rflux=rflux, zflux=zflux,
                               w1flux=w1flux, w2flux=w2flux,
                               gnobs=gnobs, rnobs=rnobs, znobs=znobs,
                               deltaChi2=deltaChi2, maskbits=maskbits,
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
        # ADM add the DR7/DR8 data columns if they aren't there yet.
        # ADM can remove this once DR8 is finalized.
        if "MASKBITS" not in targets.dtype.names:
            targets = io.add_dr8_columns(targets)

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
        # ADM only test the LRG cuts for speed. There's a
        # ADM full run through all classes in test_cuts_basic.
        tc = ["LRG"]

        for filelist in [self.tractorfiles, self.sweepfiles]:
            # ADM set backup to False as the Gaia unit test
            # ADM files only cover a limited pixel range.
            targets = cuts.select_targets(filelist, numproc=1, tcnames=tc,
                                          backup=False)
            t1 = cuts.select_targets(filelist[0:1], numproc=1, tcnames=tc,
                                     backup=False)
            t2 = cuts.select_targets(filelist[0], numproc=1, tcnames=tc,
                                     backup=False)
            for col in t1.dtype.names:
                try:
                    notNaN = ~np.isnan(t1[col])
                except TypeError:  # - can't check string columns for NaN
                    notNaN = np.ones(len(t1), dtype=bool)

                self.assertTrue(np.all(t1[col][notNaN] == t2[col][notNaN]))

    def test_qso_selection_options(self):
        """Test the QSO selection options are passed correctly
        """
        tc = ["QSO"]

        targetfile = self.tractorfiles[0]
        for qso_selection in cuts.qso_selection_options:
            # ADM set backup to False as the Gaia unit test
            # ADM files only cover a limited pixel range.
            results = cuts.select_targets(targetfile, backup=False,
                                          tcnames=tc, qso_selection=qso_selection)

        with self.assertRaises(ValueError):
            results = cuts.select_targets(targetfile, numproc=1, backup=False,
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
                # ADM set backup to False as the Gaia unit test
                # ADM files only cover a limited pixel range.
                targets = cuts.select_targets(filelist, backup=False,
                                              numproc=nproc, tcnames=tc)
                self.assertTrue('DESI_TARGET' in targets.dtype.names)
                self.assertTrue('BGS_TARGET' in targets.dtype.names)
                self.assertTrue('MWS_TARGET' in targets.dtype.names)
                self.assertEqual(len(targets), np.count_nonzero(targets['DESI_TARGET']))

                bgs1 = (targets['DESI_TARGET'] & desi_mask.BGS_ANY) != 0
                bgs2 = targets['BGS_TARGET'] != 0
                self.assertTrue(np.all(bgs1 == bgs2))

    def test_backup(self):
        """Test BACKUP targets are selected.
        """
        # ADM only test the ELG, BGS cuts for speed. There's a
        # ADM full run through all classes in test_cuts_basic.
        tc = ["ELG", "BGS"]

        # ADM BACKUP targets can only run on the sweep files.
        for filelist in self.sweepfiles:
            # ADM limit to pixels covered in the Gaia unit test files.
            targets = cuts.select_targets(
                filelist, numproc=1, tcnames=tc, test=True, nside=self.nside,
                pixlist=self.pix)
            self.assertTrue('DESI_TARGET' in targets.dtype.names)
            self.assertTrue('BGS_TARGET' in targets.dtype.names)
            self.assertTrue('MWS_TARGET' in targets.dtype.names)
            self.assertEqual(len(targets), np.count_nonzero(targets['DESI_TARGET']))

            bgs1 = (targets['DESI_TARGET'] & desi_mask.BGS_ANY) != 0
            bgs2 = targets['BGS_TARGET'] != 0
            self.assertTrue(np.all(bgs1 == bgs2))

    def test_targets_spatial(self):
        """Test applying RA/Dec/HEALpixel inputs to sweeps recovers same targets
        """
        # ADM only test some of the galaxy cuts for speed. There's a
        # ADM full run through all classes in test_cuts_basic.
        tc = ["LRG", "ELG", "BGS"]
        infiles = self.sweepfiles[2]

        # ADM set backup to False as the Gaia unit test
        # ADM files only cover a limited pixel range.
        targets = cuts.select_targets(infiles, numproc=1, tcnames=tc,
                                      backup=False)

        # ADM test the RA/Dec box input.
        radecbox = [np.min(targets["RA"])-0.01, np.max(targets["RA"])+0.01,
                    np.min(targets["DEC"])-0.01, np.max(targets["DEC"]+0.01)]
        t1 = cuts.select_targets(infiles, numproc=1, tcnames=tc,
                                 radecbox=radecbox, backup=False)

        # ADM test the RA/Dec/radius cap input.
        centra, centdec = 0.5*(radecbox[0]+radecbox[1]), 0.5*(radecbox[2]+radecbox[3])
        # ADM 20 degrees should be a large enough radius for the sweeps.
        maxrad = 20.
        radecrad = centra, centdec, maxrad
        t2 = cuts.select_targets(infiles, numproc=1, tcnames=tc,
                                 radecrad=radecrad, backup=False)

        # ADM test the pixel input.
        nside = pixarea2nside(box_area(radecbox))
        pixlist = hp_in_box(nside, radecbox)
        t3 = cuts.select_targets(infiles, numproc=1, tcnames=tc,
                                 nside=nside, pixlist=pixlist, backup=False)

        # ADM sort each set of targets on TARGETID to compare them.
        targets = targets[np.argsort(targets["TARGETID"])]
        t1 = t1[np.argsort(t1["TARGETID"])]
        t2 = t2[np.argsort(t2["TARGETID"])]
        t3 = t3[np.argsort(t3["TARGETID"])]

        # ADM test the same targets were recovered and that
        # ADM each recovered target has the same bits set.
        for targs in t1, t2, t3:
            for col in "TARGETID", "DESI_TARGET", "BGS_TARGET", "MWS_TARGET":
                self.assertTrue(np.all(targs[col] == targets[col]))

    def test_targets_spatial_inputs(self):
        """Test the code fails if more than one spatial input is passed
        """
        # ADM set up some fake inputs.
        pixlist = [0, 1]
        radecbox = [2, 3, 4, 5]
        radecrad = [6, 7, 8]
        # ADM we should throw an error every time we pass 2 inputs that aren't NoneType.
        timesthrown = 0
        for i in range(3):
            inputs = [pixlist, radecbox, radecrad]
            inputs[i] = None
            try:
                cuts.select_targets(self.sweepfiles, numproc=1,
                                    pixlist=inputs[0], radecbox=inputs[1], radecrad=inputs[2])
            except ValueError:
                timesthrown += 1

        self.assertEqual(timesthrown, 3)


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_cuts
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
