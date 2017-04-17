#!/usr/bin/env python

"""Extract

"""
import os
import numpy as np
from glob import glob
import fitsio

from astropy.table import Table, Column, vstack

from desispec.io.util import fitsheader, write_bintable
from desiutil.log import get_logger
log = get_logger()

import multiprocessing
nproc = max(1, multiprocessing.cpu_count() // 2)
    
def _lyapath():
    return os.path.join(os.getenv('DESI_ROOT'), 'mocks', 'lya_forest', 'v0.0.2')

def _read_lya(lyafile):
    """Read the metadata from a single Lya file."""
    log.info('Reading {}'.format(lyafile))
    ra, dec, z, mag = [], [], [], []
    ff = fitsio.FITS(lyafile)
    ff = ff[1:len(ff)]
    for h in ff:
        head = h.read_header()
        z.append(head['ZQSO'])
        ra.append(head['RA'])
        dec.append(head['DEC'])
        mag.append(head['MAG_G'])

    ra = np.array(ra)
    dec = np.array(dec)
    z = np.array(z).astype('f4')
    ra = ra * 180.0 / np.pi
    ra = ra % 360.0 #enforce 0 < ra < 360
    dec = dec * 180.0 / np.pi

    dat = Table()
    dat.add_column(Column(name='RA', data=ra, dtype='f8'))
    dat.add_column(Column(name='DEC', data=dec, dtype='f8'))
    dat.add_column(Column(name='Z', data=z, dtype='f4'))
    dat.add_column(Column(name='MAG_G', data=mag, dtype='f4'))

    return dat

def read_lya(nread=None):
    """Read the Lyman-alpha mocks."""

    lyafiles = glob(os.path.join(_lyapath(), 'simpleSpec_*.fits.gz'))
    if nread:
        lyafiles = lyafiles[:nread]
    log.info('Reading metadata for {} Lya files'.format(len(lyafiles)))

    p = multiprocessing.Pool(nproc)
    dat = p.map(_read_lya, lyafiles)
    p.close()
    
    return vstack(dat)

if __name__ == '__main__':

    data = read_lya()

    metafile = os.path.join(_lyapath(), 'metadata-simpleSpec.fits')
    log.info('Writing {}'.format(metafile))
    write_bintable(metafile, data, extname='METADATA', clobber=True)
