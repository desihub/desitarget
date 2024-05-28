# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.mtl specifically for secondary/stream programs.
"""

import os
import unittest
import numpy as np
from astropy.table import Table, join

from desitarget.targetmask import desi_mask as Mx
from desitarget.targetmask import scnd_mask as sMx

from desitarget.mtl import make_mtl, mtldatamodel, survey_data_model
from desitarget.targets import initial_priority_numobs, main_cmx_or_sv

from desiutil.log import get_logger
log = get_logger()


class TestMTLStreams(unittest.TestCase):

    def setUp(self):
        self.targs = Table()
        # ADM two copies of each of the GD1-style targets.
        self.types = np.array(['GD1_BRIGHT_PM', 'GD1_FAINT_NO_PM', 'GD1_FILLER',
                               'GD1_BRIGHT_PM', 'GD1_FAINT_NO_PM', 'GD1_FILLER'])
        # ADM the initial values of PRIORITY.
        self.priorities = [sMx[t].priorities['UNOBS'] for t in self.types]
        # ADM the initial values of NUMOBS_MORE.
        self.nom = [sMx[t].numobs for t in self.types]

        nt = len(self.types)
        # ADM add some "extra" columns that are needed for observations.
        for col in ["RA", "DEC", "PARALLAX", "PMRA", "PMDEC", "REF_EPOCH"]:
            self.targs[col] = np.zeros(nt, dtype=mtldatamodel[col].dtype)
        self.targs['DESI_TARGET'] = Mx["SCND_ANY"].mask
        self.targs['SCND_TARGET'] = [sMx[t].mask for t in self.types]
        for col in ['BGS_TARGET', 'MWS_TARGET', 'SUBPRIORITY', "PRIORITY"]:
            self.targs[col] = np.zeros(nt, dtype=mtldatamodel[col].dtype)

        n = len(self.targs)
        self.targs['TARGETID'] = list(range(n))

        # ADM determine the initial PRIORITY and NUMOBS.
        pinit, ninit = initial_priority_numobs(self.targs, obscon="BRIGHT",
                                               scnd=True)
        self.targs["PRIORITY_INIT"] = pinit
        self.targs["NUMOBS_INIT"] = ninit

        # ADM set up an ersatz redshift catalog.
        self.zcat = Table()
        # ADM reverse the TARGETIDs to check joins.
        self.zcat['TARGETID'] = np.flip(self.targs['TARGETID'])

        self.zcat['Z'] = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
        # ADM set ZWARN for half of the objects to test both MORE_ZWARN
        # ADM and MORE_ZGOOD.
        self.zcat['ZWARN'] = [0, 0, 0, 1, 1, 1]
        self.zcat['NUMOBS'] = [1, 1, 1, 1, 1, 1]
        self.zcat['ZTILEID'] = [-1, -1, -1, -1, -1, -1]

        # ADM expected progression in priorities and numbers of observations.
        # ADM hand-code to some extent to better check for discrepancies.
        iigood = self.zcat["ZWARN"] == 0
        zgood = [sMx[t].priorities['MORE_ZGOOD'] for t in self.types[iigood]]
        zwarn = [sMx[t].priorities['MORE_ZWARN'] for t in self.types[~iigood]]
        # ADM PRIORITY after zero, one, two, three passes through MTL.
        self.post_prio = pinit
        # ADM scalar version of initial numbers of observations. Should
        # ADM (deliberately) fail if classes have different NUMOBS_INIT.
        self.ninit_int = int(np.unique(ninit))
        # ADM loop through the numbers of observations, retain priority.
        for i in range(self.ninit_int - 1):
            self.post_prio = np.vstack([self.post_prio, zgood + zwarn])
        self.post_prio = np.vstack(
            [self.post_prio, [sMx[t].priorities['DONE'] for t in self.types]])
        # ADM NUMOBS after zero, one, two, three passes through MTL.
        self.post_nom = ninit
        for numobs in np.arange(1, self.ninit_int + 1):
            self.post_nom = np.vstack([self.post_nom,
                                       np.array(self.nom) - numobs])

    def flesh_out_data_model(self, cat):
        """Flesh out columns to produce full Main Survey data model.
        """
        truedm = survey_data_model(cat, survey="main")
        addedcols = list(set(truedm.dtype.names) - set(cat.dtype.names))
        for col in addedcols:
            cat[col] = [-1] * len(cat)
        # ADM Set QN redshifts ('Z_QN') to mimic redrock redshifts ('Z').
        if 'Z' in cat.dtype.names:
            cat['Z_QN'] = cat['Z']
            cat['IS_QSO_QN'] = 1

        return cat

    def test_numobs(self):
        """Test priorities, numobs, set correctly with no zcat.
        """
        t = self.targs.copy()
        t = self.flesh_out_data_model(t)
        mtl = make_mtl(t, "BRIGHT")
        log.info(f"Initial: {mtl['PRIORITY']}, {self.post_prio[0]}")
        log.info(f"Initial: {mtl['NUMOBS_MORE']}, {self.post_nom[0]}")
        self.assertTrue(np.all(mtl['NUMOBS_MORE'] == self.post_nom[0]))
        self.assertTrue(np.all(mtl['PRIORITY'] == self.post_prio[0]))

    def test_zcat(self):
        """Test priorities/numobs correct after zcat/multiple passes.
        """
        t = self.targs.copy()
        t = self.flesh_out_data_model(t)

        zc = self.zcat.copy()
        zc = self.flesh_out_data_model(zc)

        for numobs in range(1, self.ninit_int + 1):
            zc["NUMOBS"] = numobs
            mtl = make_mtl(t, "BRIGHT", zcat=zc, trim=False)
            log.info(f"{numobs}, {mtl['PRIORITY']}, {self.post_prio[numobs]}")
            log.info(f"{numobs}, {mtl['NUMOBS_MORE']}, {self.post_nom[numobs]}")
            self.assertTrue(np.all(mtl['PRIORITY'] == self.post_prio[numobs]))
            self.assertTrue(np.all(mtl['NUMOBS_MORE'] == self.post_nom[numobs]))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_mtl_streams
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
