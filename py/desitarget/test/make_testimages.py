# - For the record (and future updates):
# - This code was used to generate tractor and sweep file subsets for testing.
# - The hardcoded paths are for NERSC, but you can swap out any
# - legacy survey data release path as needed.
# ADM Currently use DR6 files. Should update to DR8 at some point.

# SJB prevent import from running this code
if __name__ == "__main__":
    import os
    import numpy as np
    from time import time
    from glob import glob
    import fitsio

    start = time()

    # ADM to test the skies code we need to mimic a survey directory for a brick
    sd = '/global/project/projectdirs/cosmo/data/legacysurvey'
    dr = 'dr6'
    codir = 'coadd'
    blobdir = 'metrics'
    brick = '0959p805'
    prebrick = brick[:3]
    bands = ['g', 'z']

    # ADM tear everything down, first
    os.system('rm -rf {}'.format(dr))

    rootdir = dr
    os.system('mkdir {}'.format(rootdir))
    rootdir += '/{}'.format(blobdir)
    os.system('mkdir {}'.format(rootdir))
    rootdir += '/{}'.format(prebrick)
    os.system('mkdir {}'.format(rootdir))
    os.system('cp {}/{}/blobs*{}* {}'.format(sd, rootdir, brick, rootdir))

    rootdir = dr
    rootdir += '/{}'.format(codir)
    os.system('mkdir {}'.format(rootdir))
    rootdir += '/{}'.format(prebrick)
    os.system('mkdir {}'.format(rootdir))
    rootdir += '/{}'.format(brick)
    os.system('mkdir {}'.format(rootdir))

    for band in bands:
        if (band != 'g' or band != 'z') and brick != '0959p805':
            msg = "brick 0959p805, bands g,z chosen as their (DR6) files are small!"
            raise ValueError(msg)
        os.system('cp {}/{}/*{}-image-{}* {}'.format(sd, rootdir, brick, band, rootdir))
        os.system('cp {}/{}/*{}-invvar-{}* {}'.format(sd, rootdir, brick, band, rootdir))
        os.system('cp {}/{}/*{}-nexp-{}* {}'.format(sd, rootdir, brick, band, rootdir))

    # ADM make a simplified survey bricks file for this data release
    brickfile = '{}/survey-bricks-{}.fits.gz'.format(dr, dr)
    sbfile = '{}/{}'.format(sd, brickfile)
    brickinfo = fitsio.read(sbfile)
    # ADM remember that fitsio reads things in as bytes, so convert to unicode
    bricknames = brickinfo['brickname'].astype('U')
    wbrick = np.where(bricknames == brick)[0]
    fitsio.write(brickfile, brickinfo[wbrick])

    # ADM make a simplified survey bricks file for the whole sky
    brickfile = '{}/survey-bricks.fits.gz'.format(dr)
    sbfile = '{}/{}'.format(sd, brickfile)
    brickinfo = fitsio.read(sbfile)
    # ADM remember that fitsio reads things in as bytes, so convert to unicode
    bricknames = brickinfo['BRICKNAME'].astype('U')
    wbrick = np.where(bricknames == brick)[0]
    fitsio.write(brickfile, brickinfo[wbrick])

    print('Done...t={:.2f}s'.format(time()-start))
