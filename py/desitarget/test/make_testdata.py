#- For the record (and future updates):
#- This code was used to generate tractor and sweep file subsets for testing.
#- The hardcoded paths are for Stephen's laptop, but you can swap out any
#- legacy survey data release path as needed.

from os.path import basename
from desitarget.cuts import apply_cuts

tractordir = '/data/legacysurvey/dr1/tractor/330/'
for brick in ['3301m002', '3301m007', '3303p000']:
    filepath = '{}/tractor-{}.fits'.format(tractordir, brick)
    desi_target = apply_cuts(filepath)[0]
    yes = np.where(desi_target != 0)[0]
    no = np.where(desi_target == 0)[0]
    keep = np.concatenate([yes[0:3], no[0:3]])
    data, hdr = fits.getdata(filepath, header=True)
    fits.writeto(basename(filepath), data[keep], header=hdr)

sweepdir = '/data/legacysurvey/dr2p/sweep/'
for radec in ['310m005-320p000', '320m005-330p000', '330m005-340p000']:
    filepath = '{}/sweep-{}.fits'.format(sweepdir, radec)
    desi_target = apply_cuts(filepath)[0]
    yes = np.where(desi_target != 0)[0]
    no = np.where(desi_target == 0)[0]
    keep = np.concatenate([yes[0:3], no[0:3]])
    data, hdr = fits.getdata(filepath, header=True)
    fits.writeto(basename(filepath), data[keep], header=hdr)


