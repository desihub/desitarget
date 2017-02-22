#- For the record (and future updates):
#- This code was used to generate tractor and sweep file subsets for testing.
#- The hardcoded paths are for Stephen's laptop, but you can swap out any
#- legacy survey data release path as needed.

from os.path import basename
import numpy as np
from astropy.io import fits
from desitarget.cuts import apply_cuts

tractordir = '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/tractor/330/'
#tractordir = '/data/legacysurvey/dr3.1/tractor/330/'
for brick in ['3301m002', '3301m007', '3303p000']:
    filepath = '{}/tractor-{}.fits'.format(tractordir, brick)
    desi_target = apply_cuts(filepath)[0]
    yes = np.where(desi_target != 0)[0]
    no = np.where(desi_target == 0)[0]
    keep = np.concatenate([yes[0:3], no[0:3]])
    data, hdr = fits.getdata(filepath, header=True)
    fits.writeto(basename(filepath), data[keep], header=hdr)

sweepdir = '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1/'
#sweepdir = '/data/legacysurvey/dr2p/sweep/'
for radec in ['190m005-200p000', '310m005-320p000', '320m005-330p000', '330m005-340p000']:
    filepath = '{}/sweep-{}.fits'.format(sweepdir, radec)
    desi_target = apply_cuts(filepath)[0]
    yes = np.where(desi_target != 0)[0]
    no = np.where(desi_target == 0)[0]
    keep = np.concatenate([yes[0:3], no[0:3]])
    data, hdr = fits.getdata(filepath, header=True)
    fits.writeto(basename(filepath), data[keep], header=hdr)

#ADM adding a file to make a mask for bright stars
#ADM this should go in its own directory /t/brighstar
filepath = '{}/sweep-{}.fits'.format(sweepdir, '190m005-200p000')
data, hdr = fits.getdata(filepath, header=True)
keep = np.where(data["DECAM_FLUX"][:,4] > 100000)
fits.writeto('brightstar/'+basename(filepath), data[keep], header=hdr)
