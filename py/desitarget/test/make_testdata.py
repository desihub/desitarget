#- For the record (and future updates):
#- This code was used to generate tractor and sweep file subsets for testing.
#- The hardcoded paths are for NERSC, but you can swap out any
#- legacy survey data release path as needed.
#ADM as of DR4, we read in DR3 files and use desitarget.io 
#ADM to transform the format to the post-DR3 data model.
#ADM Should eventually update to read in DR5 files directly

from os.path import basename
import numpy as np
#from astropy.io import fits
from desitarget.cuts import apply_cuts
from desitarget.io import read_tractor
import fitsio
from time import time
from desitarget.gaiamatch import find_gaia_files

start = time()
tractordir = '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/tractor/330'
#tractordir = '/data/legacysurvey/dr3.1/tractor/330/'
for brick in ['3301m002', '3301m007', '3303p000']:
    filepath = '{}/tractor-{}.fits'.format(tractordir, brick)
    desi_target, bgs_target, mws_target = apply_cuts(filepath)
    #ADM as nobody is testing the MWS in the sandbox, yet, we need to
    #ADM ensure we ignore MWS targets for testing the main algorithms
    yes = np.where( (desi_target != 0) & (mws_target == 0) )[0]
    no = np.where(desi_target == 0)[0]
    keep = np.concatenate([yes[0:3], no[0:3]])
#    data, hdr = fits.getdata(filepath, header=True)
#    fits.writeto('t/'+basename(filepath), data[keep], header=hdr)
    data, hdr = read_tractor(filepath, header=True)
    #ADM the FRACDEV and FRACDEV_IVAR columns can 
    #ADM contain some NaNs, which break testing
    wnan = np.where(data["FRACDEV"] != data["FRACDEV"])
    if len(wnan[0]) > 0:
        data["FRACDEV"][wnan] = 0.
    wnan = np.where(data["FRACDEV_IVAR"] != data["FRACDEV_IVAR"])
    if len(wnan[0]) > 0:
        data["FRACDEV_IVAR"][wnan] = 0.
    fitsio.write('t/'+basename(filepath), data[keep], header=hdr, clobber=True)
    print('made Tractor file for brick {}...t={:.2f}s'.format(brick,time()-start))

sweepdir = '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1'
#sweepdir = '/data/legacysurvey/dr2p/sweep/'
for radec in ['310m005-320p000', '320m005-330p000', '330m005-340p000']:
    filepath = '{}/sweep-{}.fits'.format(sweepdir, radec)
    desi_target, bgs_target, mws_target = apply_cuts(filepath)
    yes = np.where( (desi_target != 0) & (mws_target == 0) )[0]
    #ADM as nobody is testing the MWS in the sandbox, yet, we need to
    #ADM ensure we ignore MWS targets for testing the main algorithms
    no = np.where(desi_target == 0)[0]
    keep = np.concatenate([yes[0:3], no[0:3]])
#    data, hdr = fits.getdata(filepath, header=True)
#    fits.writeto('t/'+basename(filepath), data[keep], header=hdr)
    data, hdr = read_tractor(filepath, header=True)
    fitsio.write('t/'+basename(filepath), data[keep], header=hdr, clobber=True)
    print('made sweeps file for range {}...t={:.2f}s'.format(radec,time()-start))

#ADM adding Gaia files to which to match 
for brick in ['3301m002', '3301m007', '3303p000']:
    filepath = '{}/tractor-{}.fits'.format(tractordir, brick)
    data = fitsio.read('t/'+basename(filepath))
    #ADM use find_gaia_files to determine which Gaia files potentially
    #ADM match the sweeps objects of interest
    for gaiafile in find_gaia_files(data):
        #ADM for each of the relevant Gaia files, read the first 5 rows
        gaiadata = fitsio.read(gaiafile, rows=range(5))
        #ADM and write them to a special Gaia directory
        fitsio.write('tgaia/'+basename(gaiafile), gaiadata, clobber=True)

for radec in ['310m005-320p000', '320m005-330p000', '330m005-340p000']:
    filepath = '{}/sweep-{}.fits'.format(sweepdir, radec)
    data = fitsio.read('t/'+basename(filepath))
    #ADM use find_gaia_files to determine which Gaia files potentially
    #ADM match the sweeps objects of interest
    for gaiafile in find_gaia_files(data):
        #ADM for each of the relevant Gaia files, read the first 5 rows
        gaiadata = fitsio.read(gaiafile, rows=range(5))
        #ADM and write them to a special Gaia directory
        fitsio.write('tgaia/'+basename(gaiafile), gaiadata, clobber=True)

#ADM adding a file to make a mask for bright stars
#ADM this should go in its own directory /t2 (others are in t1)
filepath = '{}/sweep-{}.fits'.format(sweepdir, '190m005-200p000')
data, hdr = read_tractor(filepath, header=True)
keep = np.where(data["FLUX_Z"] > 100000)
fitsio.write('t2/'+basename(filepath), data[keep], header=hdr, clobber=True)
print('Done...t={:.2f}s'.format(time()-start))
