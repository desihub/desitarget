import unittest
from pkg_resources import resource_filename
import os.path
import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.coordinates import SkyCoord
from astropy import units as u

from desitarget import brightstar, desi_mask, targetid_mask, io

from desiutil import brick

class TestBRIGHTSTAR(unittest.TestCase):

    def setUp(self):
        #ADM some locations of output test files
        self.testbsfile = 'bs.fits'
        self.testmaskfile = 'bsmask.fits'
        self.testtargfile = 'bstargs.fits'

        #ADM some locations of input files
        self.bsdatadir = resource_filename('desitarget.test', 't2')
        self.datadir = resource_filename('desitarget.test', 't')
        maskablefile = self.bsdatadir + '/sweep-190m005-200p000.fits'
        unmaskablefile = self.datadir + '/sweep-320m005-330p000.fits'

        #ADM read in the "maskable targets" (targets that ARE in masks)
        masktargs = fitsio.read(maskablefile)
        #ADM because the input file has not been through Target Selection we need to add DESI_TARGET and TARGETID
        zeros = np.zeros(len(masktargs))
        masktargs = rfn.append_fields(masktargs,["DESI_TARGET","TARGETID"],
                                      [zeros,zeros],usemask=False,dtypes='>i8')
        #ADM As the sweeps file is also doubling as a targets file, we have to duplicate the
        #ADM column "BRICK_OBJID" and include it as the new column "OBJID"
        self.masktargs = rfn.append_fields(masktargs,"BRICK_OBJID",zeros,usemask=False,dtypes='>i4')
        self.masktargs["BRICK_OBJID"] = self.masktargs["OBJID"]

        #ADM read in the "unmaskable targets" (targets that are NOT in masks)
        unmasktargs = fitsio.read(unmaskablefile)
        #ADM because the input file has not been through Target Selection we need to add DESI_TARGET and TARGETID
        zeros = np.zeros(len(unmasktargs))
        unmasktargs = rfn.append_fields(unmasktargs,["DESI_TARGET","TARGETID"],
                                        [zeros,zeros],usemask=False,dtypes='>i8')
        #ADM As the sweeps file is also doubling as a targets file, we have to duplicate the
        #ADM column "BRICK_OBJID" and include it as the new column "OBJID"
        self.unmasktargs = rfn.append_fields(unmasktargs,"BRICK_OBJID",zeros,usemask=False,dtypes='>i4')
        self.unmasktargs["BRICK_OBJID"] = self.unmasktargs["OBJID"]

        #ADM set up brick information for just the brick with brickID 330368 (bricksize=0.25)
        self.drbricks = np.zeros(1,dtype=[('ra', '>f8'), ('dec', '>f8'), ('nobjs', '>i2')])
        self.drbricks["ra"] = 0.125
        self.drbricks["dec"] = 0.0
        self.drbricks["nobjs"] = 1000

        #ADM invent a mask with differing radii (1' and 20') and declinations
        self.mask = np.zeros(2, dtype=[('RA', '>f8'), ('DEC', '>f8'), ('IN_RADIUS', '>f8')])
        self.mask["DEC"] = [0,70]
        self.mask["IN_RADIUS"] = [1,20]

    def tearDown(self):
        #ADM remove any existing bright star files in this directory
        if os.path.exists(self.testmaskfile):
            os.remove(self.testmaskfile)
        if os.path.exists(self.testtargfile):
            os.remove(self.testtargfile)
        if os.path.exists(self.testbsfile):
            os.remove(self.testbsfile)

    def test_collect_bright_stars(self):
        """Test the collection of bright stars from the sweeps
        """
        #ADM collect the bright stars from the sweeps in the data directory and write to file...
        bs1 = brightstar.collect_bright_stars('grz',[9,9,9],rootdirname=self.bsdatadir,outfilename=self.testbsfile)
        #ADM ...and read in the file that was written
        bs2 = fitsio.read(self.testbsfile) 
        #ADM the created collection of objects from the sweeps should be the same as the read-in file
        bs1ids = bs1['BRICKID'].astype(np.int64)*1000000 + bs1['OBJID']
        bs2ids = bs2['BRICKID'].astype(np.int64)*1000000 + bs2['OBJID']
        self.assertTrue(np.all(bs1ids == bs2ids))

    def test_make_bright_star_mask(self):
        """Test the construction of a bright star mask
        """
        #ADM create a collection of bright stars and write to file
        bs1 = brightstar.collect_bright_stars('grz',[9,9,9],rootdirname=self.bsdatadir,outfilename=self.testbsfile)
        #ADM create a bright star mask from the collection of bright stars and write to file...
        mask = brightstar.make_bright_star_mask('grz',[9,9,9],infilename=self.testbsfile,outfilename=self.testmaskfile)
        #ADM ...and read it back in
        mask1 = fitsio.read(self.testmaskfile)
        #ADM create the bright star mask from scratch
        mask2 = brightstar.make_bright_star_mask('grz',[9,9,9],rootdirname=self.bsdatadir)
        #ADM the created-from-scratch mask should be the same as the read-in mask
        self.assertTrue(np.all(mask1["TARGETID"] == mask2["TARGETID"]))

    def test_mask_targets(self):
        """Test that targets in masks are flagged as being in masks
        """
        #ADM mask the targets, creating the mask
        targs = brightstar.mask_targets(self.masktargs,bands="RZ",maglim=[8,10],numproc=1,
                                        rootdirname=self.bsdatadir,outfilename=self.testmaskfile,drbricks=self.drbricks)
        self.assertTrue(np.any(targs["DESI_TARGET"] != 0))

    def test_non_mask_targets(self):
        """Test that targets that are NOT in masks are flagged as not being in masks
        """
        #ADM write targs to file in order to check input file method
        fitsio.write(self.testtargfile, self.unmasktargs, clobber=True)
        #ADM create the mask and write it to file
        mask = brightstar.make_bright_star_mask('RZ',[8,10],rootdirname=self.bsdatadir,outfilename=self.testmaskfile)
        #ADM mask the targets, reading in the mask
        targs = brightstar.mask_targets(self.testtargfile,instarmaskfile=self.testmaskfile,drbricks=self.drbricks)
        #ADM none of the targets should have been masked
        self.assertTrue(np.all((targs["DESI_TARGET"] == 0) | ((targs["DESI_TARGET"] & desi_mask.BADSKY) != 0)))

    def test_safe_locations(self):
        """Test that SAFE/BADSKY locations are equidistant from mask centers
        """
        #ADM append SAFE (BADSKY) locations around the perimeter of the mask
        targs = brightstar.append_safe_targets(self.unmasktargs,self.mask,drbricks=self.drbricks)
        #ADM restrict to just SAFE (BADSKY) locations
        skybitset = ((targs["TARGETID"] & targetid_mask.SKY) != 0)
        safes = targs[np.where(skybitset)]
        #ADM for each mask location check that every safe location is equidistant from the mask center
        c = SkyCoord(safes["RA"]*u.deg,safes["DEC"]*u.deg)
        for i in range(2):
            cent = SkyCoord(self.mask[i]["RA"]*u.deg, self.mask[i]["DEC"]*u.deg)
            sep = cent.separation(c)
            #ADM only things close to mask i
            w = np.where(sep < np.min(sep)*1.002)
            #ADM are these all the same distance to a very high precision?
            print("mask position and radius (arcmin)",self.mask[i])
            self.assertTrue(np.max(sep[w] - sep[w[0]]) < 1e-15*u.deg)

    def test_targetid(self):
        """Test SKY/RELEASE/BRICKID/OBJID are set correctly in TARGETID and DESI_TARGET for SAFE/BADSKY locations
        """
        #ADM append SAFE (BADSKY) locations around the periphery of the mask
        targs = brightstar.append_safe_targets(self.unmasktargs,self.mask,drbricks=self.drbricks)

        #ADM first check that the SKY bit and BADSKY bits are appropriately set
        skybitset = ((targs["TARGETID"] & targetid_mask.SKY) != 0)
        badskybitset = ((targs["DESI_TARGET"] & desi_mask.BADSKY) != 0)
        self.assertTrue(np.all(skybitset == badskybitset))

        #ADM now check that the other bits are in the correct locations
        #ADM first restrict to just things in BRICK 330368
        w = np.where(targs["BRICKID"] == 330368)
        targs = targs[w]

        #ADM check that the TARGETIDs are unique
        self.assertEqual(len(set(targs["TARGETID"])),len(targs["TARGETID"]))

        #ADM the targetids as a binary string
        bintargids = [ np.binary_repr(targid) for targid in targs["TARGETID"] ]        

        #ADM check that the data release is set (in a way unlike the normal bit-setting in brightstar.py)
        #ADM note that release should be zero for SAFE LOCATIONS
        rmostbit = targetid_mask.RELEASE.bitnum
        lmostbit = targetid_mask.RELEASE.bitnum + targetid_mask.RELEASE.nbits
        drbitset = int(bintargids[0][-lmostbit:-rmostbit],2)
        drbitshould = targs["RELEASE"][0]
        self.assertEqual(drbitset,drbitshould)
        self.assertEqual(drbitset,0)

        #ADM check that the OBJIDs proceed from "nobjs" in self.drbricks
        rmostbit = targetid_mask.OBJID.bitnum
        lmostbit = targetid_mask.OBJID.bitnum + targetid_mask.OBJID.nbits
        #ADM guard against the fact that when written the rmostbit for OBJID is 0
        if rmostbit == 0:
            objidset = np.array([ int(bintargid[-lmostbit:],2) for bintargid in bintargids ])
        else:
            objidset = np.array([ int(bintargid[-lmostbit:-rmostbit],2) for bintargid in bintargids ])
        objidshould = self.drbricks["nobjs"]+np.arange(len(objidset))+1
        self.assertTrue(np.all(objidset == objidshould))

        #ADM finally check that the BRICKIDs are all 330368
        rmostbit = targetid_mask.BRICKID.bitnum
        lmostbit = targetid_mask.BRICKID.bitnum + targetid_mask.BRICKID.nbits
        brickidset = np.array([ int(bintargid[-lmostbit:-rmostbit],2) for bintargid in bintargids ])
        self.assertTrue(np.all(brickidset == 330368))

if __name__ == '__main__':
    unittest.main()
