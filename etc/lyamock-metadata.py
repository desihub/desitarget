#!/usr/bin/env python

"""Extract

"""
import os
import numpy as np
from glob import glob
import multiprocessing

import fitsio

from astropy.table import Table, Column, vstack

from desispec.io.util import fitsheader, write_bintable
from desiutil.log import get_logger
log = get_logger()

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

def read_lya(indir, nproc, nread=None):
    """Read the Lyman-alpha mocks.
    
    Returns filelist, metadata table
    """

    lyafiles = sorted(glob(os.path.join(indir, 'simpleSpec_*.fits.gz')))
    if nread:
        lyafiles = lyafiles[:nread]
    log.info('Reading metadata for {} Lya files'.format(len(lyafiles)))

    p = multiprocessing.Pool(nproc)
    dat = p.map(_read_lya, lyafiles)
    p.close()
    
    #- Add mapping of target to -> mockfile,rownum
    filemap = Table()
    filemap['MOCKFILE'] = [os.path.basename(x) for x in lyafiles]
    filemap['MOCKFILEID'] = np.arange(len(lyafiles), dtype=np.int16)
    for i, xdat in enumerate(dat):
        xdat['MOCKFILEID'] = np.ones(len(xdat), dtype=np.int16) * i
        xdat['MOCKHDUNUM'] = np.arange(1, len(xdat)+1, dtype=np.int32)
    
    return vstack(dat), filemap

if __name__ == '__main__':
    import argparse
    
    _nproc = multiprocessing.cpu_count() // 2
    parser = argparse.ArgumentParser(usage = "%(prog)s [options]")
    parser.add_argument("-i", "--indir", type=str,  help="input data")
    parser.add_argument("-o", "--output", type=str,  help="output file")
    parser.add_argument("--nproc", type=int,  help="output file", default=_nproc)
    args = parser.parse_args()

    if args.indir is None:
        args.indir = _lyapath()

    if args.output is None:
        args.output = 'metadata-simpleSpec.fits'

    log.info('Reading mocks from {}'.format(args.indir))
    data, filemap = read_lya(args.indir, nproc=args.nproc)

    log.info('Writing {}'.format(args.output))
    write_bintable(args.output, data, extname='METADATA', clobber=True)
    header = dict(MOCKDIR = os.path.abspath(args.indir))
    write_bintable(args.output, filemap, extname='FILEMAP', header=header)
