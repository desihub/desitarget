# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.brightmask.
"""
import unittest
from pkg_resources import resource_filename
import os
import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.coordinates import SkyCoord
from astropy import units as u
from glob import glob
import healpy as hp
import tempfile
import shutil

from desitarget import brightmask, io
from desitarget.targetmask import desi_mask, targetid_mask

from desiutil import brick


class TestBRIGHTMASK(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # ADM set up the necessary environment variables.
        cls.gaiadir_orig = os.getenv("GAIA_DIR")
        testdir = 'desitarget.test'
        os.environ["GAIA_DIR"] = resource_filename(testdir, 't4')
        cls.tychodir_orig = os.getenv("TYCHO_DIR")
        os.environ["TYCHO_DIR"] = resource_filename(testdir, 't4/tycho')
        cls.uratdir_orig = os.getenv("URAT_DIR")
        os.environ["URAT_DIR"] = resource_filename(testdir, 't4/urat')

        # ADM a temporary output directory to test writing masks.
        cls.maskdir = tempfile.mkdtemp()

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

        # ADM pick a faint maglim (as unit tests deal with few objects).
        cls.maglim = 20.
        # ADM also pick a reasonable epoch at which to make the mask.
        cls.maskepoch = 2025.5

        # ADM an example mask, made from all of the test HEALPixels.
        cls.allmx = brightmask.make_bright_star_mask(
            numproc=1, nside=cls.nside, pixels=cls.pixnum,
            maglim=cls.maglim, maskepoch=cls.maskepoch)

        # ADM read in some targets.
        targdir = resource_filename(testdir, 't')
        fn = os.path.join(targdir, 'sweep-320m005-330p000.fits')
        ts = fitsio.read(fn)
        # ADM targets are really sweeps objects, so add target fields.
        zs = np.zeros(len(ts))
        targs = rfn.append_fields(ts, ["DESI_TARGET", "TARGETID"], [zs, zs],
                                  usemask=False, dtypes='>i8')
        cls.targs = rfn.append_fields(targs, "BRICK_OBJID", zs, usemask=False,
                                      dtypes='>i4')
        cls.targs["BRICK_OBJID"] = cls.targs["OBJID"]

        # ADM mask_targets checks for unique TARGETIDs, so create some.
        cls.targs["TARGETID"] = np.arange(len(cls.targs))

        # ADM invent a mask with various testing properties.
        cls.mask = np.zeros(3, dtype=brightmask.maskdatamodel.dtype)
        cls.mask["DEC"] = [0, 70, 35]
        cls.mask["IN_RADIUS"] = [1, 20, 10]
        cls.mask["E1"] = [0., 0., -0.3]
        cls.mask["E2"] = [0., 0., 0.5]
        cls.mask["TYPE"] = ['PSF', b'PSF ', 'PSF ']

    @classmethod
    def tearDownClass(cls):
        # ADM remove the temporary output directory.
        if os.path.exists(cls.maskdir):
            shutil.rmtree(cls.maskdir)

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
        # ADM test making the mask in an individual pixel.
        mx = brightmask.make_bright_star_mask_in_hp(
            self.nside, self.pixnum[0],
            maglim=self.maglim, maskepoch=self.maskepoch)

        # ADM check that running across all pixels contains the subset
        # ADM of masks in the single pixel.
        self.assertTrue(len(set(mx["REF_ID"]) - set(self.allmx["REF_ID"])) == 0)
        self.assertTrue(len(set(self.allmx["REF_ID"]) - set(mx["REF_ID"])) > 0)

    def test_make_bright_star_mask_parallel(self):
        """Check running the mask-making code in parallel.
        """
        # ADM run on two processors.
        two = brightmask.make_bright_star_mask(
            numproc=2, nside=self.nside, pixels=self.pixnum,
            maglim=self.maglim, maskepoch=self.maskepoch)

        # ADM check that running in parallel recovers the same masks as
        # ADM running on one processor.
        one = self.allmx[np.argsort(self.allmx["REF_ID"])]
        two = two[np.argsort(two["REF_ID"])]
        self.assertTrue(np.all(one == two))

    def test_mask_write(self):
        """Test that masks are written to file correctly.
        """
        # ADM some meaningless magnitude limits and mask epochs.
        ml, me = 62.3, 2062.3
        # ADM a keyword dictionary to write to the output file header.
        extra = {'BLAT': 'blat', 'FOO': 'foo'}

        # ADM test writing without HEALPixel-split.
        _, mxdir = io.write_masks(self.maskdir, self.allmx, maglim=ml,
                                  maskepoch=me, extra=extra)

        # ADM test writing with HEALPixel-split.
        _, mxdir = io.write_masks(self.maskdir, self.allmx, maglim=ml,
                                  maskepoch=me, extra=extra, nside=self.nside)

        # ADM construct the output directory and file name.
        mxd = io.find_target_files(self.maskdir, flavor="masks",
                                   maglim=ml, epoch=me)
        mxfn = io.find_target_files(self.maskdir, flavor="masks",
                                    maglim=ml, epoch=me, hp=self.pixnum[0])

        # ADM check the output directory is as expected.
        self.assertEqual(mxdir, mxd)

        # ADM check all of the files were made in the correct place.
        fns = glob(os.path.join(mxdir, "masks-hp*fits"))
        self.assertEqual(len(fns), len(self.pixnum)+1)

        # ADM check the extra kwargs were written to the header.
        for key in extra:
            hdr = fitsio.read_header(mxfn, "MASKS")
            self.assertEqual(hdr[key].rstrip(), extra[key])

    def test_mask_targets(self):
        """Test that targets in masks are flagged accordingly.
        """
        # ADM create the output mask directory.
        _, mxdir = io.write_masks(self.maskdir, self.allmx, maglim=self.maglim,
                                  maskepoch=self.maskepoch, nside=self.nside)

        # ADM make targets with the same coordinates as the masks.
        # ADM remembering to select masks that actually have a radius.
        ii = self.allmx["IN_RADIUS"] > 0
        targs = self.targs.copy()
        targs["RA"] = self.allmx["RA"][ii][:len(targs)]
        targs["DEC"] = self.allmx["DEC"][ii][:len(targs)]

        # ADM add mask information to DESI_TARGET.
        mxt = brightmask.mask_targets(targs, mxdir, nside=self.nside,
                                      pixlist=self.pixnum)

        # ADM all the targs should have been masked.
        nmasked = np.sum(mxt["DESI_TARGET"] & desi_mask["IN_BRIGHT_OBJECT"] != 0)
        self.assertEqual(nmasked, len(targs))

        # ADM and we should have added some safe targets that will be
        # ADM "near" bright objects.
        is_nbo = mxt["DESI_TARGET"] & desi_mask["NEAR_BRIGHT_OBJECT"] != 0
        self.assertTrue(np.all(is_nbo))

    def test_non_mask_targets(self):
        """Test targets that are NOT in masks are flagged accordingly.
        """
        # ADM create the output mask directory.
        _, mxdir = io.write_masks(self.maskdir, self.allmx, maglim=self.maglim,
                                  maskepoch=self.maskepoch, nside=self.nside)

        # ADM update DESI_TARGET for any targets in masks.
        mxtargs = brightmask.mask_targets(self.targs, mxdir, nside=self.nside,
                                          pixlist=self.pixnum)

        # ADM none of the targets should be in a mask.
        self.assertTrue(np.all(mxtargs["DESI_TARGET"] == 0))

    def test_safe_locations(self):
        """Test SAFE/BADSKY locations are equidistant from mask centers.
        """
        # ADM append SAFE locations around the perimeter of the mask.
        safes = brightmask.get_safe_targets(self.targs, self.mask)
        targs = np.concatenate([self.targs, safes])
        # ADM restrict to just SAFE locations.
        skybitset = ((targs["TARGETID"] & targetid_mask.SKY) != 0)
        safes = targs[np.where(skybitset)]
        # ADM for each mask location check that every safe location is
        # ADM equidistant from the mask center.
        c = SkyCoord(safes["RA"]*u.deg, safes["DEC"]*u.deg)
        for i in range(2):
            cent = SkyCoord(self.mask[i]["RA"]*u.deg, self.mask[i]["DEC"]*u.deg)
            sep = cent.separation(c)
            # ADM only things close to mask i
            w = np.where(sep < np.min(sep)*1.002)
            # ADM are these all the same distance to high precision?
            self.assertTrue(np.max(sep[w] - sep[w[0]]) < 1e-15*u.deg)

    def test_targetid(self):
        """Test SKY/RELEASE/BRICKID/OBJID are set correctly in TARGETID
        and DESI_TARGET for SAFE/BADSKY locations.
        """
        # ADM append SAFE locations around the perimeter of the mask.
        safes = brightmask.get_safe_targets(self.targs, self.mask)
        targs = np.concatenate([self.targs, safes])

        # ADM first check the SKY and BADSKY bits are appropriately set.
        skybitset = ((targs["TARGETID"] & targetid_mask.SKY) != 0)
        badskybitset = ((targs["DESI_TARGET"] & desi_mask.BAD_SKY) != 0)
        self.assertTrue(np.all(skybitset == badskybitset))

        # ADM now check that the other bits are in the correct locations
        # ADM first restrict to the ~half-dozen targets in BRICK 521233.
        bid = 521233
        ii = targs["BRICKID"] == bid
        targs = targs[ii]

        # ADM check that the TARGETIDs are unique.
        s = set(targs["TARGETID"])
        self.assertEqual(len(s), len(targs["TARGETID"]))

        # ADM the TARGETIDs as a binary string.
        bintargids = [np.binary_repr(targid) for targid in targs["TARGETID"]]

        # ADM check the DR is set (in a way unlike the normal bit-setting
        # in brightmask.py). Release should be zero for SAFE locations.
        rmostbit = targetid_mask.RELEASE.bitnum
        lmostbit = targetid_mask.RELEASE.bitnum + targetid_mask.RELEASE.nbits
        drbitset = int(bintargids[0][-lmostbit:-rmostbit], 2)
        drbitshould = targs["RELEASE"][0]
        self.assertEqual(drbitset, drbitshould)
        self.assertEqual(drbitset, 0)

        # ADM check that the BRICKIDs are as restricted/requested.
        rmostbit = targetid_mask.BRICKID.bitnum
        lmostbit = targetid_mask.BRICKID.bitnum + targetid_mask.BRICKID.nbits
        brickidset = np.array(
            [int(bintargid[-lmostbit:-rmostbit], 2) for bintargid in bintargids])
        self.assertTrue(np.all(brickidset == bid))


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_brightmask
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
