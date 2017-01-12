import unittest
from pkg_resources import resource_filename
import os.path
from uuid import uuid4
import numbers

from astropy.io import fits
from astropy.table import Table
import fitsio
import numpy as np

from desitarget import io
from desitarget import cuts
from desitarget import desi_mask

class TestCuts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datadir = resource_filename('desitarget.test', 't')
        cls.tractorfiles = sorted(io.list_tractorfiles(cls.datadir))
        cls.sweepfiles = sorted(io.list_sweepfiles(cls.datadir))

    def test_unextinct_fluxes(self):
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
        #- Cuts work with either data or filenames
        desi, bgs, mws = cuts.apply_cuts(self.tractorfiles[0])
        desi, bgs, mws = cuts.apply_cuts(self.sweepfiles[0])
        data = io.read_tractor(self.tractorfiles[0])
        desi, bgs, mws = cuts.apply_cuts(data)
        data = io.read_tractor(self.sweepfiles[0])
        desi, bgs, mws = cuts.apply_cuts(data)

        # bgs_any1 = (desi & desi_mask.BGS_ANY)
        # bgs_any2 = (bgs != 0)
        # self.assertTrue(np.all(bgs_any1 == bgs_any2))

    def test_cuts_noprimary(self):
        #- cuts should work with or without "primary"
        targets = Table.read(self.sweepfiles[0])
        desi1, bgs1, mws1 = cuts.apply_cuts(targets)
        targets.remove_column('BRICK_PRIMARY')
        desi2, bgs2, mws2 = cuts.apply_cuts(targets)
        self.assertTrue(np.all(desi1==desi2))
        self.assertTrue(np.all(bgs1==bgs2))
        self.assertTrue(np.all(mws1==mws2))

    def test_single_cuts(self):
        #- test cuts of individual target classes
        targets = Table.read(self.sweepfiles[0])
        flux = cuts.unextinct_fluxes(targets)
        gflux = flux['GFLUX']
        rflux = flux['RFLUX']
        zflux = flux['ZFLUX']
        w1flux = flux['W1FLUX']
        w2flux = flux['W2FLUX']
        wise_snr = targets['WISE_FLUX'] * np.sqrt(targets['WISE_FLUX_IVAR'])
        dchisq = targets['DCHISQ'] 
        deltaChi2 = dchisq[...,0] - dchisq[...,1]
        primary = targets['BRICK_PRIMARY']

        lrg1 = cuts.isLRG(rflux=rflux, zflux=zflux, w1flux=w1flux, primary=None)
        lrg2 = cuts.isLRG(rflux=rflux, zflux=zflux, w1flux=w1flux, primary=primary)
        self.assertTrue(np.all(lrg1==lrg2))

        elg1 = cuts.isELG(gflux=gflux, rflux=rflux, zflux=zflux, primary=primary)
        elg2 = cuts.isELG(gflux=gflux, rflux=rflux, zflux=zflux, primary=None)
        self.assertTrue(np.all(elg1==elg2))

        psftype = targets['TYPE']
        bgs1 = cuts.isBGS_bright(rflux=rflux, objtype=psftype, primary=primary)
        bgs2 = cuts.isBGS_bright(rflux=rflux, objtype=None, primary=None)
        self.assertTrue(np.all(bgs1==bgs2))

        bgs1 = cuts.isBGS_faint(rflux=rflux, objtype=psftype, primary=primary)
        bgs2 = cuts.isBGS_faint(rflux=rflux, objtype=None, primary=None)
        self.assertTrue(np.all(bgs1==bgs2))

        #- Test that objtype and primary are optional
        qso1 = cuts.isQSO_cuts(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                          deltaChi2=deltaChi2, wise_snr=wise_snr, objtype=psftype, primary=primary)
        qso2 = cuts.isQSO_cuts(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                          deltaChi2=deltaChi2, wise_snr=wise_snr, objtype=None, primary=None)
        self.assertTrue(np.all(qso1==qso2))

        fstd1 = cuts.isFSTD_colors(gflux=gflux, rflux=rflux, zflux=zflux, primary=None)
        fstd2 = cuts.isFSTD_colors(gflux=gflux, rflux=rflux, zflux=zflux, primary=primary)
        self.assertTrue(np.all(fstd1==fstd2))

    #- cuts should work with tables from several I/O libraries
    def _test_table_row(self, targets):
        self.assertFalse(cuts._is_row(targets))
        self.assertTrue(cuts._is_row(targets[0]))

        desi, bgs, mws = cuts.apply_cuts(targets)
        self.assertEqual(len(desi), len(targets))
        self.assertEqual(len(bgs), len(targets))
        self.assertEqual(len(mws), len(targets))

        desi, bgs, mws = cuts.apply_cuts(targets[0])
        self.assertTrue(isinstance(desi, numbers.Integral), 'DESI_TARGET mask not an int')
        self.assertTrue(isinstance(bgs, numbers.Integral), 'BGS_TARGET mask not an int')
        self.assertTrue(isinstance(mws, numbers.Integral), 'MWS_TARGET mask not an int')

    def test_astropy_fits(self):
        targets = fits.getdata(self.tractorfiles[0])
        self._test_table_row(targets)

    def test_astropy_table(self):
        targets = Table.read(self.tractorfiles[0])
        self._test_table_row(targets)

    def test_numpy_ndarray(self):
        targets = fitsio.read(self.tractorfiles[0], upper=True)
        self._test_table_row(targets)

    #- select targets should work with either data or filenames
    def test_select_targets(self):
        for filelist in [self.tractorfiles, self.sweepfiles]:
            targets = cuts.select_targets(filelist, numproc=1)
            t1 = cuts.select_targets(filelist[0:1], numproc=1)
            t2 = cuts.select_targets(filelist[0], numproc=1)
            for col in t1.dtype.names:
                try:
                    notNaN = ~np.isnan(t1[col])
                except TypeError:  #- can't check string columns for NaN
                    notNan = np.ones(len(t1), dtype=bool)

                self.assertTrue(np.all(t1[col][notNaN]==t2[col][notNaN]))

    def test_check_targets(self):
        for filelist in self.tractorfiles:
            nbadfiles = cuts.check_input_files(filelist, numproc=1)

            self.assertTrue(nbadfiles==0)

    def test_qso_selection_options(self):
        targetfile = self.tractorfiles[0]
        for qso_selection in cuts.qso_selection_options:
            results = cuts.select_targets(targetfile, qso_selection=qso_selection)
            
        with self.assertRaises(ValueError):
            results = cuts.select_targets(targetfile, numproc=1, qso_selection='blatfoo')

    def test_missing_files(self):
        with self.assertRaises(ValueError):
            targets = cuts.select_targets(['blat.foo1234',], numproc=1)

    def test_parallel_select(self):
        for nproc in [1,2]:
            for filelist in [self.tractorfiles, self.sweepfiles]:
                targets = cuts.select_targets(filelist, numproc=nproc)
                self.assertTrue('DESI_TARGET' in targets.dtype.names)
                self.assertTrue('BGS_TARGET' in targets.dtype.names)
                self.assertTrue('MWS_TARGET' in targets.dtype.names)
                self.assertEqual(len(targets), np.count_nonzero(targets['DESI_TARGET']))

                bgs1 = (targets['DESI_TARGET'] & desi_mask.BGS_ANY) != 0
                bgs2 = targets['BGS_TARGET'] != 0
                self.assertTrue(np.all(bgs1 == bgs2))

if __name__ == '__main__':
    unittest.main()
