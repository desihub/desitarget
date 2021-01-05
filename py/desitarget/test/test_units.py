# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget units.
"""
import os
import unittest
import yaml
import astropy.version as astropyversion
import astropy.units as u
import numpy as np
from astropy.table import Table
from pkg_resources import resource_filename
from desitarget.brightmask import maskdatamodel as dma
from desitarget.gaiamatch import gaiadatamodel as dmb
from desitarget.gfa import gfadatamodel as dmc
from desitarget.io import basetsdatamodel as dmd
from desitarget.io import dr8addedcols as dme
from desitarget.io import dr9addedcols as dmf
from desitarget.mtl import mtldatamodel as dmg
from desitarget.skyfibers import skydatamodel as dmh


class TestUNITS(unittest.TestCase):

    def setUp(self):
        # ADM load the units yaml file.
        basefn = os.path.join('data', 'units.yaml')
        self.fn = resource_filename('desitarget', basefn)
        with open(self.fn) as f:
            self.units = yaml.safe_load(f)

        # ADM combine the unique quantities from the various data models.
        dmnames = dma.dtype.names
        for dm in dmb, dmc, dmd, dme, dmf, dmg, dmh:
            dmnames += dm.dtype.names
        self.dmnames = list(set(dmnames))

    def test_fits_units(self):
        """Test the units meet the FITS standard (via astropy).
        """
        # ADM unique units from the yaml file with NoneType removed.
        uniq = set(self.units.values())
        uniq.remove(None)

        # ADM nmgy isn't an allowed unit in earlier versions of astropy.
        if astropyversion.major < 4:
            uniq = set([i for i in list(uniq) if 'nanomaggy' not in i])

        # ADM parse the units to check they're allowed astropy units.
        parsed = [u.Unit(unit) for unit in uniq]

        # ADM these should be equivalent, even though, formally, parsed
        # ADM contains items of type astropy.units.core.Unit.
        self.assertEqual(list(uniq), parsed)

    def test_quantities(self):
        """Test all data model quantities are in the units yaml file.
        """
        missing = [dmn for dmn in self.dmnames if dmn not in self.units]
        msg = 'These quantities are missing in {}'.format(self.fn)

        self.assertEqual([], missing, msg=msg)

    def test_assigning(self):
        """Test all data model quantities can be assigned units.
        """
        # ADM loop through the data model and create a list of units
        # ADM that would be suitable for writing using fitsio.
        unitlist = []
        for col in self.dmnames:
            if self.units[col] is None:
                unitlist.append("")
            else:
                unitlist.append(self.units[col])
        unitlist = np.array(unitlist)

        # ADM also test assigning units directly to an astropy Table.
        data = Table(np.zeros(len(self.dmnames)), names=self.dmnames)
        for col in data.columns:
            data[col].unit = self.units[col]

        # ADM recover the units from the Table.
        tabunits = np.array([data[col].unit for col in data.columns])

        # ADM the only discrepancies should be where "" in the unitlist
        # ADM correspond to None in the astropy Table.
        msg = "\n*** Unit list is:\n {} \n*** But Table units are:\n {}".format(
            unitlist, tabunits)
        self.assertTrue(np.all(unitlist[tabunits != unitlist] == ""), msg=msg)


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_mtl
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
