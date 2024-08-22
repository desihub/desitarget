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
from desitarget.mtl import make_mtl, mtldatamodel, survey_data_model
from desitarget.targets import initial_priority_numobs, main_cmx_or_sv
from desitarget.targets import switch_main_cmx_or_sv, zcut, midzcut


class TestMTL(unittest.TestCase):

    def setUp(self):
        self.lyaz = zcut + 0.4  # ADM an appropriate redshift for a LyA QSO.
        self.midz = 0.5*(midzcut + zcut)  # ADM a redshift for a true mid-z QSO.
        self.lowz = 0.5  # ADM an appropriate redshift for a tracer QSO.
        self.targets = Table()
        self.types = np.array(['ELG_LOP', 'LRG', 'QSO', 'QSO', 'QSO', 'ELG_LOP'])
        self.priorities = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.nom = [Mx[t].numobs for t in self.types]  # ADM the initial values of NUMOBS_MORE.

        # ADM checked-by-hand priorities and numbers of observations.
        # ADM priorities after one pass through MTL.
        self.post_prio = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.post_prio[0] = Mx['ELG_LOP'].priorities['DONE']  # ELG.
        self.post_prio[1] = Mx['LRG'].priorities['DONE']  # LRG...all one-pass.
        self.post_prio[2] = Mx['QSO'].priorities['MORE_MIDZQSO']  # lowz/tracer QSO.
        self.post_prio[3] = Mx['QSO'].priorities['MORE_MIDZQSO']  # true midz QSO.
        self.post_prio[4] = Mx['QSO'].priorities['MORE_ZGOOD']  # highz QSO.
        # ADM numobs_more after one pass through MTL.
        self.post_nom = [Mx[t].numobs for t in self.types]
        self.post_nom[0] = 0  # ELG gets 1 total observation.
        self.post_nom[1] = 0  # LRG gets 1 total observation.
        self.post_nom[2] = 1  # lowz/tracer QSO gets 2 total observations.
        self.post_nom[3] = 3  # true midz QSO gets 4 total observations.
        self.post_nom[4] = 3  # LyA QSO gets 4 total observations.
        # ADM priorities after two passes through MTL.
        self.post_prio_duo = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.post_prio_duo[0] = Mx['ELG_LOP'].priorities['DONE']  # ELG after second pass.
        self.post_prio_duo[1] = Mx['LRG'].priorities['DONE']  # LRG...all one-pass.
        self.post_prio_duo[2] = Mx['QSO'].priorities['DONE']  # lowz QSO after second pass.
        self.post_prio_duo[3] = Mx['QSO'].priorities['MORE_MIDZQSO']  # true midz QSO after second pass.
        self.post_prio_duo[4] = Mx['QSO'].priorities['MORE_ZGOOD']  # highz QSO no change after second pass.
        # ADM numobs_more after two passes through MTL.
        self.post_nom_duo = [Mx[t].numobs for t in self.types]
        self.post_nom_duo[0] = 0  # ELG gets 1 total observation.
        self.post_nom_duo[1] = 0  # LRG gets 1 total observation.
        self.post_nom_duo[2] = 0  # lowz/tracer QSO gets 2 total observations.
        self.post_nom_duo[3] = 2  # true midz QSO gets 4 total observations.
        self.post_nom_duo[4] = 2  # LyA QSO gets 4 total observations.
        # ADM priorities after everything is done.
        self.post_prio_done = [Mx[t].priorities['UNOBS'] for t in self.types]
        self.post_prio_done[0] = Mx['ELG_LOP'].priorities['DONE']  # ELG.
        self.post_prio_done[1] = Mx['LRG'].priorities['DONE']  # LRG.
        self.post_prio_done[2] = Mx['QSO'].priorities['DONE']  # lowz QSO.
        self.post_prio_done[3] = Mx['QSO'].priorities['DONE']  # true midz.
        self.post_prio_done[4] = Mx['QSO'].priorities['DONE']  # highz QSO.
        # ADM numobs_more after two passes through MTL.
        self.post_nom_done = [Mx[t].numobs for t in self.types]
        self.post_nom_done[0] = 0
        self.post_nom_done[1] = 0
        self.post_nom_done[2] = 0
        self.post_nom_done[3] = 0
        self.post_nom_done[4] = 0

        nt = len(self.types)
        # ADM add some "extra" columns that are needed for observations.
        for col in ["RA", "DEC", "PARALLAX", "PMRA", "PMDEC", "REF_EPOCH"]:
            self.targets[col] = np.zeros(nt, dtype=mtldatamodel[col].dtype)
        self.targets['DESI_TARGET'] = [Mx[t].mask for t in self.types]
        for col in ['BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET',
                    'SUBPRIORITY', "PRIORITY"]:
            self.targets[col] = np.zeros(nt, dtype=mtldatamodel[col].dtype)
        n = len(self.targets)
        self.targets['TARGETID'] = list(range(n))
        # ADM determine the initial PRIORITY and NUMOBS.
        pinit, ninit = initial_priority_numobs(self.targets, obscon="DARK")
        self.targets["PRIORITY_INIT"] = pinit
        self.targets["NUMOBS_INIT"] = ninit

        # - reverse the order for zcat to make sure joins work
        self.zcat = Table()
        self.zcat['TARGETID'] = self.targets['TARGETID'][-2::-1]
        # ADM in update_data_model, below, we set QN redshifts ('Z_QN')
        # ADM to mimic redrock redshifts ('Z').
        self.zcat['Z'] = [self.lyaz, self.midz, self.lowz, 0.6, 1.0]
        self.zcat['ZWARN'] = [0, 0, 0, 0, 0]
        self.zcat['NUMOBS'] = [1, 1, 1, 1, 1]
        self.zcat['ZTILEID'] = [-1, -1, -1, -1, -1]

    def update_data_model(self, cat):
        """Catalogs have a different data model for the Main Survey.
        """
        _, _, survey = main_cmx_or_sv(cat)
        truedm = survey_data_model(cat, survey=survey)
        addedcols = list(set(truedm.dtype.names) - set(cat.dtype.names))
        # ADM We set Main Survey QN columns in the SetUp. Add any others.
        for col in addedcols:
            cat[col] = [-1] * len(cat)
        # ADM Set QN redshifts ('Z_QN') to mimic redrock redshifts ('Z').
        if 'Z' in cat.dtype.names:
            cat['Z_QN'] = cat['Z']
            cat['IS_QSO_QN'] = 1

        return cat

    def reset_targets(self, prefix):
        """Add prefix to TARGET columns"""

        t = self.targets.copy()
        main_names = ['DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', "SCND_TARGET"]

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
        for prefix in ["", "CMX_", "SV3_"]:
            t = self.reset_targets(prefix)
            t = self.update_data_model(t)
            col, Mx, survey = main_cmx_or_sv(t)
            mtl = make_mtl(t, "BRIGHT|DARK", trimcols=True)
            mtldm = switch_main_cmx_or_sv(mtldatamodel, mtl)
            _, _, survey = main_cmx_or_sv(mtldm)
            mtldm = survey_data_model(mtldm, survey=survey)
            refnames = sorted(mtldm.dtype.names)
            mtlnames = sorted(mtl.dtype.names)
            self.assertEqual(refnames, mtlnames)

    def test_numobs(self):
        """Test priorities, numobs, set correctly with no zcat.
        """
        t = self.reset_targets("")
        t = self.update_data_model(t)
        mtl = make_mtl(t, "DARK")
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == self.nom))
        self.assertTrue(np.all(mtl['PRIORITY'] == self.priorities))

    def test_zcat(self):
        """Test priorities, numobs, set correctly after zcat.
        """
        t = self.reset_targets("")
        t = self.update_data_model(t)
        zcat = self.update_data_model(self.zcat.copy())
        mtl = make_mtl(t, "DARK", zcat=zcat, trim=False)
        pp = self.post_prio
        nom = self.post_nom
        self.assertTrue(np.all(mtl['PRIORITY'] == pp))
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == nom))
        # - change one target to a SAFE (BADSKY) target and confirm priority=0 not 1
        t['DESI_TARGET'][0] = Mx.BAD_SKY
        zcat = self.update_data_model(self.zcat.copy())
        mtl = make_mtl(t, "DARK", zcat=zcat, trim=False)
        self.assertEqual(mtl['PRIORITY'][0], 0)

    def test_multiple_passes(self):
        """Test priorities, numobs, correct after two or more MTL passes.
        """
        # ADM set up the MTL as for test_zcat.
        t = self.reset_targets("")
        t = self.update_data_model(t)
        zcat = self.update_data_model(self.zcat.copy())
        mtl = make_mtl(t, "DARK", zcat=zcat, trim=False)

        # ADM add an observation.
        zcat["NUMOBS"] += 1

        # ADM repeat MTL to check that numobs and priorities are correct.
        mtl = make_mtl(mtl, "DARK", zcat=zcat, trim=False)
        pp = self.post_prio_duo
        nom = self.post_nom_duo
        self.assertTrue(np.all(mtl['PRIORITY'] == pp))
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == nom))

        # ADM repeat until QSOs should be done, check everything IS done.
        passes = int(np.unique(zcat["NUMOBS"]))
        for i in range(Mx["QSO"].numobs - passes):
            zcat["NUMOBS"] += 1
            mtl = make_mtl(mtl, "DARK", zcat=zcat, trim=False)
        pp = self.post_prio_done
        nom = self.post_nom_done
        self.assertTrue(np.all(mtl['PRIORITY'] == pp))
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == nom))

    def test_lya_lock_in(self):
        """Test LyA QSOs remain LyA QSOs, even when the zcat changes.
        """
        # ADM set up the MTL as for test_zcat.
        t = self.reset_targets("")
        t = self.update_data_model(t)
        zcat = self.update_data_model(self.zcat.copy())
        mtl = make_mtl(t, "DARK", zcat=zcat, trim=False)

        # ADM record the location of the mid-z and LyA QSO in the MTL.
        iilowzmtl = mtl['Z'] == self.lowz
        iimidzmtl = mtl['Z'] == self.midz
        iilyazmtl = mtl['Z'] == self.lyaz

        # ADM add an observation.
        zcat["NUMOBS"] += 1

        # ADM now update the zcat so the mid-z QSO is LyA and vice-versa.
        # ADM also update a tracer QSO to be LyA.
        modzcat = zcat.copy()
        if not np.all(modzcat["Z"] == modzcat["Z_QN"]):
            msg = "Z_QN should always equal Z for this test!!!"
            log.error(msg)
            raise ValueError(msg)
        iilowz = modzcat["Z"] == self.lowz
        iimidz = modzcat["Z"] == self.midz
        iilyaz = modzcat["Z"] == self.lyaz
        for zcol in "Z", "Z_QN":
            modzcat[zcol][iilowz] = self.lyaz
            modzcat[zcol][iimidz] = self.lyaz
            modzcat[zcol][iilyaz] = self.midz

        # ADM run MTL.
        mtl = make_mtl(mtl, "DARK", zcat=modzcat, trim=False)

        # ADM the result should leave the LyA QSO unchanged (it's "locked
        # ADM in"), but promote the mid-z/tracer QSOs to being LyA QSOs.
        pp = np.array(self.post_prio_duo)
        pp[iilowzmtl] = pp[iilyazmtl]
        pp[iimidzmtl] = pp[iilyazmtl]
        # ADM numbers of observations should follow suit.
        nom = np.array(self.post_nom_duo)
        nom[iilowzmtl] = nom[iilyazmtl]
        nom[iimidzmtl] = nom[iilyazmtl]

        self.assertTrue(np.all(mtl['PRIORITY'] == pp))
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == nom))

        # ADM add an observation.
        modzcat["NUMOBS"] += 1

        # ADM reverse things, returning the mid-z/tracer QSOs to their
        # ADM original redshifts and the LyA QSO to the Ly-A redshift.
        for zcol in "Z", "Z_QN":
            modzcat[zcol][iilowz] = self.lowz
            modzcat[zcol][iimidz] = self.midz
            modzcat[zcol][iilyaz] = self.lyaz

        # ADM run MTL.
        mtl = make_mtl(mtl, "DARK", zcat=modzcat, trim=False)

        # ADM the reverted mid-z/tracer QSOs should remain "locked in" as
        # ADM LyA quasars. So, their priorities should remain unchanged
        # ADM until they're DONE.
        self.assertTrue(np.all(mtl['PRIORITY'] == pp))

        # ADM repeat MTL until the QSOs should be DONE. Everything should
        # ADM be DONE as usual, regardless of previous redshift changes.
        passes = int(np.unique(modzcat["NUMOBS"]))
        for i in range(Mx["QSO"].numobs - passes):
            modzcat["NUMOBS"] += 1
            mtl = make_mtl(mtl, "DARK", zcat=modzcat, trim=False)
        pp = self.post_prio_done
        nom = self.post_nom_done
        self.assertTrue(np.all(mtl['PRIORITY'] == pp))
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == nom))

    def test_midz_no_lock_in(self):
        """Test true mid-z QSOs revert to tracers when redshifts change.
        """
        # ADM set up the MTL as for test_zcat.
        t = self.reset_targets("")
        t = self.update_data_model(t)
        zcat = self.update_data_model(self.zcat.copy())
        mtl = make_mtl(t, "DARK", zcat=zcat, trim=False)

        # ADM grab the location of the mid-z QSO and a tracer in the MTL.
        iimidzmtl = mtl['Z'] == self.midz
        iilowzmtl = mtl['Z'] == self.lowz

        # ADM add an observation.
        zcat["NUMOBS"] += 1

        # ADM now update the zcat so the mid-z QSO is a low-z QSO.
        modzcat = zcat.copy()
        iimidz = modzcat["Z"] == self.midz
        for zcol in "Z", "Z_QN":
            modzcat[zcol][iimidz] = self.lowz

        # ADM run MTL.
        mtl = make_mtl(mtl, "DARK", zcat=modzcat, trim=False)

        # ADM the mid-z QSO should now look exactly like a tracer.
        pp = np.array(self.post_prio_duo)
        nom = np.array(self.post_nom_duo)
        pp[iimidzmtl] = pp[iilowzmtl]
        nom[iimidzmtl] = nom[iilowzmtl]

        self.assertTrue(np.all(mtl['PRIORITY'] == pp))
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == nom))

    def test_mtl_io(self):
        """Test MTL correctly handles masked NUMOBS quantities.
        """
        # ADM loop through once for SV and once for the main survey.
        t = self.reset_targets("")
        t = self.update_data_model(t)
        zcat = self.update_data_model(self.zcat.copy())
        mtl = make_mtl(t, "BRIGHT", zcat=zcat, trim=True)
        testfile = 'test-aszqweladfqwezceas.fits'
        mtl.write(testfile, overwrite=True)
        x = mtl.read(testfile)
        os.remove(testfile)
        if x.masked:
            self.assertTrue(np.all(
                mtl['NUMOBS_MORE'].mask == x['NUMOBS_MORE'].mask))

    def test_merged_qso(self):
        """Test QSO tracers merged with other targets get 2 observations.
        """
        # ADM there are other tests of this kind in test_multiple_mtl.py.

        # ADM create a set of targets that are QSOs and
        # ADM (perhaps) also another target class.
        qtargets = self.targets.copy()
        qtargets["DESI_TARGET"] |= Mx["QSO"]

        # ADM give them all a "tracer" redshift (below a mid-z QSO).
        qzcat = self.update_data_model(self.zcat.copy())
        # ADM that this is a tracer should hold regardless of IS_QSO_QN.
        qzcat["Z"] = 0.5
        qzcat["Z_QN"] = 0.5

        # ADM set their initial conditions to be that of a QSO.
        pinit, ninit = initial_priority_numobs(qtargets, obscon="DARK")
        qtargets["PRIORITY_INIT"] = pinit
        qtargets["NUMOBS_INIT"] = ninit

        # ADM run through MTL, only return entries updated by the zcat.
        mtl = make_mtl(qtargets, obscon="DARK", zcat=qzcat, trimtozcat=True)

        # ADM all confirmed tracer quasars should have NUMOBS_MORE=1.
        self.assertTrue(np.all(mtl["NUMOBS_MORE"] == 1))

    @unittest.skip('This test is deprecated.')
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
        bgszcat = self.update_data_model(self.zcat.copy())

        # ADM run through MTL, only return entries updated by the zcat.
        mtl = make_mtl(bgstargets, obscon="BRIGHT", zcat=bgszcat,
                       trimtozcat=True)

        # ADM all BGS targets should always have NUMOBS_MORE=1.
        self.assertTrue(np.all(mtl["NUMOBS_MORE"] == 1))

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

