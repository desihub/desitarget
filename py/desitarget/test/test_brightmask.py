# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.brightmask.
"""
import unittest
from pkg_resources import resource_filename
import os.path
import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.coordinates import SkyCoord
from astropy import units as u
from glob import glob
import healpy as hp

from desitarget import brightmask, io
from desitarget.targetmask import desi_mask, targetid_mask

from desiutil import brick


class TestBRIGHTMASK(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # ADM set up the necessary environment variables.
        cls.gaiadir_orig = os.getenv("GAIA_DIR")
        os.environ["GAIA_DIR"] = resource_filename('desitarget.test', 't4')
        cls.tychodir_orig = os.getenv("TYCHO_DIR")
        os.environ["TYCHO_DIR"] = resource_filename('desitarget.test', 't4/tycho')
        cls.uratdir_orig = os.getenv("URAT_DIR")
        os.environ["URAT_DIR"] = resource_filename('desitarget.test', 't4/urat')

        # ADM some locations of input files.
        cls.bsdatadir = resource_filename('desitarget.test', 't2')
        cls.datadir = resource_filename('desitarget.test', 't')
        maskablefile = cls.bsdatadir + '/sweep-190m005-200p000.fits'
        unmaskablefile = cls.datadir + '/sweep-320m005-330p000.fits'

        # ADM allowed HEALPixels in the Tycho directory.
        pixnum = []
        fns = glob(os.path.join(os.environ["TYCHO_DIR"], 'healpix', '*fits'))
        for fn in fns:
            data, hdr = fitsio.read(fn, "TYCHOHPX", header=True)
            nside = hdr["HPXNSIDE"]
            theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
            pixnum.append(list(set(hp.ang2pix(nside, theta, phi, nest=True))))
        cls.pixnum = [i for eachlist in pixnum for i in eachlist]
        cls.nside = nside

        # ADM read in the "maskable targets" (targets that ARE in masks)
        masktargs = fitsio.read(maskablefile)
        # ADM because the input file has not been through Target Selection we need to add DESI_TARGET and TARGETID
        zeros = np.zeros(len(masktargs))
        masktargs = rfn.append_fields(masktargs, ["DESI_TARGET", "TARGETID"],
                                      [zeros, zeros], usemask=False, dtypes='>i8')
        # ADM As the sweeps file is also doubling as a targets file, we have to duplicate the
        # ADM column "BRICK_OBJID" and include it as the new column "OBJID"
        cls.masktargs = rfn.append_fields(masktargs, "BRICK_OBJID", zeros, usemask=False, dtypes='>i4')
        cls.masktargs["BRICK_OBJID"] = cls.masktargs["OBJID"]

        # ADM read in the "unmaskable targets" (targets that are NOT in masks)
        unmasktargs = fitsio.read(unmaskablefile)
        # ADM because the input file has not been through Target Selection we need to add DESI_TARGET and TARGETID
        zeros = np.zeros(len(unmasktargs))
        unmasktargs = rfn.append_fields(unmasktargs, ["DESI_TARGET", "TARGETID"],
                                        [zeros, zeros], usemask=False, dtypes='>i8')
        # ADM As the sweeps file is also doubling as a targets file, we have to duplicate the
        # ADM column "BRICK_OBJID" and include it as the new column "OBJID"
        cls.unmasktargs = rfn.append_fields(unmasktargs, "BRICK_OBJID", zeros, usemask=False, dtypes='>i4')
        cls.unmasktargs["BRICK_OBJID"] = cls.unmasktargs["OBJID"]

        # ADM invent a mask with differing radii (1' and 20') and declinations
        cls.mask = np.zeros(3, dtype=[('RA', '>f8'), ('DEC', '>f8'), ('IN_RADIUS', '>f4'),
                                      ('E1', '>f4'), ('E2', '>f4'), ('TYPE', 'S4')])
        cls.mask["DEC"] = [0, 70, 35]
        cls.mask["IN_RADIUS"] = [1, 20, 10]
        cls.mask["E1"] = [0., 0., -0.3]
        cls.mask["E2"] = [0., 0., 0.5]
        cls.mask["TYPE"] = ['REX', b'PSF ', 'EXP ']

    @classmethod
    def tearDownClass(cls):
        # ADM remove any existing test files in this directory
        if os.path.exists(cls.testmaskfile):
            os.remove(cls.testmaskfile)
        if os.path.exists(cls.testtargfile):
            os.remove(cls.testtargfile)
        if os.path.exists(cls.testbsfile):
            os.remove(cls.testbsfile)
        # ADM reset the environment variables.
        if cls.gaiadir_orig is not None:
            os.environ["GAIA_DIR"] = cls.gaiadir_orig
        if cls.tychodir_orig is not None:
            os.environ["TYCHO_DIR"] = cls.tychodir_orig
        if cls.uratdir_orig is not None:
            os.environ["URAT_DIR"] = cls.uratdir_orig

    def test_make_bright_star_mask(self):
        """Test the construction of a bright star mask.
        """
        mask = make_bright_star_mask_in_hp(cls.nside, cls.pixnum[0], maglim=20.)

        self.assertTrue(np.all(mask1["TARGETID"] == mask2["TARGETID"]))

    def test_mask_targets(self):
        """Test that targets in masks are flagged as being in masks
        """
        # ADM mask the targets, creating the mask
        targs = brightmask.mask_targets(self.masktargs, maglim=[8, 10], numproc=1,
                                        rootdirname=self.bsdatadir, outfilename=self.testmaskfile)
        self.assertTrue(np.any(targs["DESI_TARGET"] != 0))

    def test_non_mask_targets(self):
        """Test that targets that are NOT in masks are flagged as not being in masks
        """
        # ADM write targs to file in order to check input file method
        fitsio.write(self.testtargfile, self.unmasktargs, clobber=True)
        # ADM create the mask and write it to file
        mask = brightmask.make_bright_source_mask('RZ', [8, 10],
                                                  rootdirname=self.bsdatadir, outfilename=self.testmaskfile)
        # ADM mask the targets, reading in the mask
        targs = brightmask.mask_targets(self.testtargfile, inmaskfile=self.testmaskfile)

        # ADM none of the targets should have been masked
        self.assertTrue(np.all((targs["DESI_TARGET"] == 0) | ((targs["DESI_TARGET"] & desi_mask.BAD_SKY) != 0)))

    def test_safe_locations(self):
        """Test that SAFE/BADSKY locations are equidistant from mask centers
        """
        # ADM append SAFE (BADSKY) locations around the perimeter of the mask
        safes = brightmask.get_safe_targets(self.unmasktargs, self.mask)
        targs = np.concatenate([self.unmasktargs, safes])
        # ADM restrict to just SAFE (BADSKY) locations
        skybitset = ((targs["TARGETID"] & targetid_mask.SKY) != 0)
        safes = targs[np.where(skybitset)]
        # ADM for each mask location check that every safe location is equidistant from the mask center
        c = SkyCoord(safes["RA"]*u.deg, safes["DEC"]*u.deg)
        for i in range(2):
            cent = SkyCoord(self.mask[i]["RA"]*u.deg, self.mask[i]["DEC"]*u.deg)
            sep = cent.separation(c)
            # ADM only things close to mask i
            w = np.where(sep < np.min(sep)*1.002)
            # ADM are these all the same distance to a very high precision?
            print("mask information", self.mask[i])
            self.assertTrue(np.max(sep[w] - sep[w[0]]) < 1e-15*u.deg)

    def test_targetid(self):
        """Test SKY/RELEASE/BRICKID/OBJID are set correctly in TARGETID and DESI_TARGET for SAFE/BADSKY locations
        """
        # ADM append SAFE (BADSKY) locations around the periphery of the mask
        safes = brightmask.get_safe_targets(self.unmasktargs, self.mask)
        targs = np.concatenate([self.unmasktargs, safes])

        # ADM first check that the SKY bit and BADSKY bits are appropriately set
        skybitset = ((targs["TARGETID"] & targetid_mask.SKY) != 0)
        badskybitset = ((targs["DESI_TARGET"] & desi_mask.BAD_SKY) != 0)
        self.assertTrue(np.all(skybitset == badskybitset))

        # ADM now check that the other bits are in the correct locations
        # ADM first restrict to just things in BRICK 330368
        w = np.where(targs["BRICKID"] == 330368)
        targs = targs[w]

        # ADM check that the TARGETIDs are unique
        self.assertEqual(len(set(targs["TARGETID"])), len(targs["TARGETID"]))

        # ADM the targetids as a binary string
        bintargids = [np.binary_repr(targid) for targid in targs["TARGETID"]]

        # ADM check that the data release is set (in a way unlike the normal bit-setting in brightmask.py)
        # ADM note that release should be zero for SAFE LOCATIONS
        rmostbit = targetid_mask.RELEASE.bitnum
        lmostbit = targetid_mask.RELEASE.bitnum + targetid_mask.RELEASE.nbits
        drbitset = int(bintargids[0][-lmostbit:-rmostbit], 2)
        drbitshould = targs["RELEASE"][0]
        self.assertEqual(drbitset, drbitshould)
        self.assertEqual(drbitset, 0)

        # ADM finally check that the BRICKIDs are all 330368
        rmostbit = targetid_mask.BRICKID.bitnum
        lmostbit = targetid_mask.BRICKID.bitnum + targetid_mask.BRICKID.nbits
        brickidset = np.array([int(bintargid[-lmostbit:-rmostbit], 2) for bintargid in bintargids])
        self.assertTrue(np.all(brickidset == 330368))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_brightmask
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
