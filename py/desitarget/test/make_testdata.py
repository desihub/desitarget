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

tractordir = '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/tractor/330'
#tractordir = '/data/legacysurvey/dr3.1/tractor/330/'
for brick in ['3301m002', '3301m007', '3303p000']:
    filepath = '{}/tractor-{}.fits'.format(tractordir, brick)
    desi_target = apply_cuts(filepath)[0]
    yes = np.where(desi_target != 0)[0]
    no = np.where(desi_target == 0)[0]
    keep = np.concatenate([yes[0:3], no[0:3]])
#    data, hdr = fits.getdata(filepath, header=True)
#    fits.writeto('t/'+basename(filepath), data[keep], header=hdr)
    data, hdr = read_tractor(filepath, header=True)
    fitsio.write('t/'+basename(filepath), data[keep], header=hdr)

sweepdir = '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1'
#sweepdir = '/data/legacysurvey/dr2p/sweep/'
for radec in ['310m005-320p000', '320m005-330p000', '330m005-340p000']:
    filepath = '{}/sweep-{}.fits'.format(sweepdir, radec)
    desi_target = apply_cuts(filepath)[0]
    yes = np.where(desi_target != 0)[0]
    no = np.where(desi_target == 0)[0]
    keep = np.concatenate([yes[0:3], no[0:3]])
#    data, hdr = fits.getdata(filepath, header=True)
#    fits.writeto('t/'+basename(filepath), data[keep], header=hdr)
    data, hdr = read_tractor(filepath, header=True)
    fitsio.write('t/'+basename(filepath), data[keep], header=hdr)

#ADM adding a file to make a mask for bright stars
#ADM this should go in its own directory /t2 (others are in t1)
filepath = '{}/sweep-{}.fits'.format(sweepdir, '190m005-200p000')
data, hdr = read_tractor(filepath, header=True)
keep = np.where(data["FLUX_Z"] > 100000)
fitsio.write('t2/'+basename(filepath), data[keep], header=hdr)
