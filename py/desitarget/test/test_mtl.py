# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.mtl.
"""

# ADM only run unit tests that require redrock if its installed.
try:
    # ADM import the zwarn mask from redrock.
    from redrock.zwarning import ZWarningMask as rrMx
    norr = False
except ImportError:
    from desiutil.log import get_logger
    log = get_logger()
    log.info('redrock not installed; skipping ZWARN consistency check')
    norr = True

import os
import unittest
import numpy as np
from astropy.table import Table, join

from desitarget.targetmask import desi_mask as Mx
from desitarget.targetmask import bgs_mask, obsconditions
from desitarget.mtl import make_mtl, mtldatamodel
from desitarget.targets import initial_priority_numobs, main_cmx_or_sv
from desitarget.targets import switch_main_cmx_or_sv


class TestMTL(unittest.TestCase):

    def setUp(self):
        self.targets = Table()
        self.types = np.array(['ELG', 'LRG', 'QSO', 'QSO', 'QSO', 'ELG'])
        self.priorities = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.post_prio = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.post_prio[0] = Mx['ELG'].priorities['DONE']  # ELG
        self.post_prio[1] = Mx['LRG'].priorities['DONE']  # LRG...all one-pass
        self.post_prio[2] = Mx['QSO'].priorities['DONE']  # lowz QSO
        self.post_prio[3] = Mx['QSO'].priorities['MORE_MIDZQSO']  # midz QSO
        self.post_prio[4] = Mx['QSO'].priorities['MORE_ZGOOD']  # highz QSO
        nt = len(self.types)
        # ADM add some "extra" columns that are needed for observations.
        for col in ["RA", "DEC", "PARALLAX", "PMRA", "PMDEC", "REF_EPOCH"]:
            self.targets[col] = np.zeros(nt, dtype=mtldatamodel[col].dtype)
        self.targets['DESI_TARGET'] = [Mx[t].mask for t in self.types]
        for col in ['BGS_TARGET', 'MWS_TARGET', 'SUBPRIORITY']:
            self.targets[col] = np.zeros(nt, dtype=mtldatamodel[col].dtype)
        n = len(self.targets)
        self.targets['ZFLUX'] = 10**((22.5-np.linspace(20, 22, n))/2.5)
        self.targets['TARGETID'] = list(range(n))
        # ADM determine the initial PRIORITY and NUMOBS.
        pinit, ninit = initial_priority_numobs(self.targets, obscon="DARK|GRAY")
        self.targets["PRIORITY_INIT"] = pinit
        self.targets["NUMOBS_INIT"] = ninit

        # - reverse the order for zcat to make sure joins work
        self.zcat = Table()
        self.zcat['TARGETID'] = self.targets['TARGETID'][-2::-1]
        self.zcat['Z'] = [2.5, 1.9, 0.5, 0.5, 1.0]
        self.zcat['ZWARN'] = [0, 0, 0, 0, 0]
        self.zcat['NUMOBS'] = [1, 1, 1, 1, 1]
        self.zcat['SPECTYPE'] = ['QSO', 'QSO', 'QSO', 'GALAXY', 'GALAXY']
        self.zcat['ZTILEID'] = [-1, -1, -1, -1, -1]

    def reset_targets(self, prefix):
        """Add prefix to TARGET columns"""

        t = self.targets.copy()
        main_names = ['DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET']

        if prefix == 'CMX_':
            # ADM restructure the table to look like a commissioning table.
            t.rename_column('DESI_TARGET', 'CMX_TARGET')
            t.remove_column('BGS_TARGET')
            t.remove_column('MWS_TARGET')
        else:
            for name in main_names:
                t.rename_column(name, prefix+name)

        return t

    def test_mtl(self):
        """Test output from MTL has the correct column names.
        """
        # ADM loop through once each for the main survey, commissioning and SV.
        for prefix in ["", "CMX_", "SV1_"]:
            t = self.reset_targets(prefix)
            mtl = make_mtl(t, "BRIGHT|GRAY|DARK", trimcols=True)
            mtldm = switch_main_cmx_or_sv(mtldatamodel, mtl)
            refnames = sorted(mtldm.dtype.names)
            mtlnames = sorted(mtl.dtype.names)
            self.assertEqual(refnames, mtlnames)

    def test_numobs(self):
        """Test priorities, numobs and obsconditions are set correctly with no zcat.
        """
        # ADM loop through once for SV and once for the main survey.
        for prefix in ["", "SV1_"]:
            t = self.reset_targets(prefix)
            mtl = make_mtl(t, "GRAY|DARK")
            mtl.sort(keys='TARGETID')
            self.assertTrue(np.all(mtl['NUMOBS_MORE'] == [1, 1, 4, 4, 4, 1]))
            self.assertTrue(np.all(mtl['PRIORITY'] == self.priorities))
            # - Check that ELGs can be observed in gray conditions but not others
            iselg = (self.types == 'ELG')
            self.assertTrue(np.all((mtl['OBSCONDITIONS'][iselg] & obsconditions.GRAY) != 0))
            self.assertTrue(np.all((mtl['OBSCONDITIONS'][~iselg] & obsconditions.GRAY) == 0))

    def test_zcat(self):
        """Test priorities, numobs and obsconditions are set correctly after zcat.
        """
        # ADM loop through once for SV and once for the main survey.
        for prefix in ["", "SV1_"]:
            t = self.reset_targets(prefix)
            mtl = make_mtl(t, "DARK|GRAY", zcat=self.zcat, trim=False)
            mtl.sort(keys='TARGETID')
            pp = self.post_prio.copy()
            nom = [0, 0, 0, 3, 3, 1]
            # ADM in SV, all quasars get all observations.
#            if prefix == "SV1_":
#                pp[2], nom[2] = pp[3], nom[3]
            self.assertTrue(np.all(mtl['PRIORITY'] == pp))
            self.assertTrue(np.all(mtl['NUMOBS_MORE'] == nom))
            # - change one target to a SAFE (BADSKY) target and confirm priority=0 not 1
            t[prefix+'DESI_TARGET'][0] = Mx.BAD_SKY
            mtl = make_mtl(t, "DARK|GRAY", zcat=self.zcat, trim=False)
            mtl.sort(keys='TARGETID')
            self.assertEqual(mtl['PRIORITY'][0], 0)

    def test_mtl_io(self):
        """Test MTL correctly handles masked NUMOBS quantities.
        """
        # ADM loop through once for SV and once for the main survey.
        for prefix in ["", "SV1_"]:
            t = self.reset_targets(prefix)
            mtl = make_mtl(t, "BRIGHT", zcat=self.zcat, trim=True)
            testfile = 'test-aszqweladfqwezceas.fits'
            mtl.write(testfile, overwrite=True)
            x = mtl.read(testfile)
            os.remove(testfile)
            if x.masked:
                self.assertTrue(np.all(mtl['NUMOBS_MORE'].mask == x['NUMOBS_MORE'].mask))

    def test_merged_qso(self):
        """Test QSO tracers that are also other target types get 1 observation.
        """
        # ADM there are other tests of this kind in test_multiple_mtl.py.

        # ADM create a set of targets that are QSOs and
        # ADM (perhaps) also another target class.
        qtargets = self.targets.copy()
        qtargets["DESI_TARGET"] |= Mx["QSO"]

        # ADM give them all a "tracer" redshift (below a mid-z QSO).
        qzcat = self.zcat.copy()
        qzcat["Z"] = 0.5

        # ADM set their initial conditions to be that of a QSO.
        pinit, ninit = initial_priority_numobs(qtargets, obscon="DARK|GRAY")
        qtargets["PRIORITY_INIT"] = pinit
        qtargets["NUMOBS_INIT"] = ninit

        # ADM run through MTL.
        mtl = make_mtl(qtargets, obscon="DARK|GRAY", zcat=qzcat)

        # ADM all confirmed tracer quasars should have NUMOBS_MORE=0.
        self.assertTrue(np.all(qzcat["NUMOBS_MORE"] == 0))

    def test_endless_bgs(self):
        """Test BGS targets always get another observation in bright time.
        """
        # ADM create a set of BGS FAINT/BGS_BRIGHT targets
        # ADM (perhaps) also another target class.
        bgstargets = self.targets.copy()
        bgstargets["DESI_TARGET"] = Mx["BGS_ANY"]
        bgstargets["BGS_TARGET"] = bgs_mask["BGS_FAINT"] | bgs_mask["BGS_BRIGHT"]

        # ADM set their initial conditions for the bright-time survey.
        pinit, ninit = initial_priority_numobs(bgstargets, obscon="BRIGHT")
        bgstargets["PRIORITY_INIT"] = pinit
        bgstargets["NUMOBS_INIT"] = ninit

        # ADM create a copy of the zcat.
        bgszcat = self.zcat.copy()

        # ADM run through MTL.
        mtl = make_mtl(bgstargets, obscon="BRIGHT", zcat=bgszcat)

        # ADM all BGS targets should always have NUMOBS_MORE=1.
        self.assertTrue(np.all(bgszcat["NUMOBS_MORE"] == 1))

    @unittest.skipIf(norr, 'redrock not installed; skip ZWARN consistency check')
    def test_zwarn_in_sync(self):
        """Check redrock and desitarget ZWARN bit-masks are consistent.
        """
        # ADM import the zwarn mask from desitarget.
        from desitarget.targetmask import zwarn_mask as dtMx
        dtbitnames = dtMx.names()
        for (bitname, bitval) in rrMx.flags():
            if "RESERVED" not in bitname:
                # ADM check bit-names are consistent.
                self.assertTrue(bitname in dtbitnames,
                                "missing ZWARN bit {} from redrock".format(
                                    bitname))
                # ADM check bit-values are consistent.
                self.assertTrue(dtMx[bitname] == bitval,
                                "ZWARN bit value mismatch for {}".format(
                                    bitname))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_mtl
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
