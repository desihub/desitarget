import unittest
from pkg_resources import resource_filename
import os.path
import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.coordinates import SkyCoord
from astropy import units as u

from desitarget import brightstar, desi_mask, targetid_mask

class TestBRIGHTSTAR(unittest.TestCase):

    def setUp(self):
        self.datadir = resource_filename('desitarget.test', 't')
        self.bsdatadir = resource_filename('desitarget.test', 't2')
        self.testbsfile = 'bs.fits'
        self.testmaskfile = 'bsmask.fits'
        self.testtargfile = 'bstargs.fits'
        self.maskablefile = 'sweep-190m005-200p000.fits'
        self.unmaskablefile = 'sweep-320m005-330p000.fits'

    def tearDown(self):
        #ADM remove any existing bright star files in this directory
        if os.path.exists(self.testmaskfile):
            os.remove(self.testmaskfile)
        if os.path.exists(self.testtargfile):
            os.remove(self.testtargfile)
        if os.path.exists(self.testbsfile):
            os.remove(self.testbsfile)

    def test_collect_bright_stars(self):
        #ADM collect the bright stars from the sweeps in the data directory and write to file...
        bs1 = brightstar.collect_bright_stars('grz',[9,9,9],rootdirname=self.bsdatadir,outfilename=self.testbsfile)
        #ADM ...and read in the file that was written
        bs2 = fitsio.read(self.testbsfile) 
        #ADM the created collection of objects from the sweeps should be the same as the read-in file
        bs1ids = bs1['BRICKID'].astype(np.int64)*1000000 + bs1['OBJID']
        bs2ids = bs2['BRICKID'].astype(np.int64)*1000000 + bs2['OBJID']
        self.assertTrue(np.all(bs1ids == bs2ids))

    def test_make_bright_star_mask(self):
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
        maskablefile = self.bsdatadir+'/'+self.maskablefile
        maskabletargs = fitsio.read(maskablefile)
        #ADM because the input file has not been through Target Selection we need to add DESI_TARGET and TARGETID
        ntargs = len(maskabletargs)
        intargs = rfn.append_fields(maskabletargs,["DESI_TARGET","TARGETID"],
                                              [np.zeros(ntargs),np.zeros(ntargs)],usemask=False,dtypes='>i8')
        #ADM As the sweeps file is also doubling as a targets file, we have to duplicate the
        #ADM column "BRICK_OBJID" and include it as the new column "OBJID"
        intargs = rfn.append_fields(intargs,"BRICK_OBJID",np.zeros(ntargs),usemask=False,dtypes='>i4')
        intargs["BRICK_OBJID"] = intargs["OBJID"]

        #ADM mask the targets, creating the mask
        targs = brightstar.mask_targets(intargs,bands="RZ",maglim=[8,10],numproc=1,
                                        rootdirname=self.bsdatadir,outfilename=self.testmaskfile)
        self.assertTrue(np.any(targs["DESI_TARGET"] != 0))

    def test_non_mask_targets(self):
        unmaskablefile = self.datadir+'/'+self.unmaskablefile
        unmaskabletargs = fitsio.read(unmaskablefile)
        #ADM because the input file has not been through Target Selection we need to add DESI_TARGET and TARGETID
        ntargs = len(unmaskabletargs)
        intargs = rfn.append_fields(unmaskabletargs,["DESI_TARGET","TARGETID"],
                                              [np.zeros(ntargs),np.zeros(ntargs)],usemask=False,dtypes='>i8')
        #ADM As the sweeps file is also doubling as a targets file, we have to duplicate the
        #ADM column "BRICK_OBJID" and include it as the new column "OBJID"
        intargs = rfn.append_fields(intargs,"BRICK_OBJID",np.zeros(ntargs),usemask=False,dtypes='>i4')
        intargs["BRICK_OBJID"] = intargs["OBJID"]

        #ADM write targs to file in order to check input file method
        fitsio.write(self.testtargfile, intargs, clobber=True)
        #ADM create the mask and write it to file
        mask = brightstar.make_bright_star_mask('RZ',[8,10],rootdirname=self.bsdatadir,outfilename=self.testmaskfile)
        #ADM mask the targets, reading in the mask
        targs = brightstar.mask_targets(self.testtargfile,instarmaskfile=self.testmaskfile)
        #ADM none of the targets should have been masked
        self.assertTrue(np.all((targs["DESI_TARGET"] == 0) | ((targs["DESI_TARGET"] & desi_mask.SAFE) != 0)))

    def test_safe_locations(self):
        unmaskablefile = self.datadir+'/'+self.unmaskablefile
        unmaskabletargs = fitsio.read(unmaskablefile)
        #ADM because the input file has not been through Target Selection we need to add DESI_TARGET and TARGETID
        ntargs = len(unmaskabletargs)
        targs = rfn.append_fields(unmaskabletargs,["DESI_TARGET","TARGETID"],
                                  [np.zeros(ntargs),np.zeros(ntargs)],usemask=False,dtypes='>i8')
        #ADM invent a mask with wildly differing radii (5' and 2o) and declinations
        mask = np.zeros(2, dtype=[('RA', '>f8'), ('DEC', '>f8'), ('IN_RADIUS', '>f8')])
        mask["DEC"] = [0,70]
        mask["IN_RADIUS"] = [5.,60.*2]
        #ADM append SAFE locations around the periphery of the mask
        targs = brightstar.append_safe_targets(targs,mask)
        #ADM first check that the SKY bit and BADSKY bits are appropriately set
        skybitset = ((targs["TARGETID"] & targetid_mask.SKY) != 0)
        badskybitset = ((targs["DESI_TARGET"] & desi_mask.SAFE) != 0)
        self.assertTrue(np.all(skybitset == badskybitset))
        #ADM restrict to just SAFE locations
        safes = targs[np.where(skybitset)]
        #ADM for each mask location check that every safe location is equidistant from the mask center
        c = SkyCoord(safes["RA"]*u.deg,safes["DEC"]*u.deg)
        for i in range(2):
            cent = SkyCoord(mask[i]["RA"]*u.deg, mask[i]["DEC"]*u.deg)
            sep = cent.separation(c)
            #ADM only things close to mask i
            w = np.where(sep < np.min(sep)*1.002)
            #ADM are these all the same distance to a very high precision?
            print("mask position and radius (arcmin)",mask[i])
            self.assertTrue(np.max(sep[w] - sep[w[0]]) < 1e-15*u.deg)

if __name__ == '__main__':
    unittest.main()
