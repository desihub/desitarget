#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import desitarget
import desitarget.targets
from desitarget.io import read_tractor, write_targets
from desitarget.io import iter_tractor, map_tractor
from desitarget.io import fix_tractor_dr1_dtype

from argparse import ArgumentParser
import os, sys
from time import time

import fitsio

ap = ArgumentParser()
### ap.add_argument("--type", choices=["tractor"], default="tractor", help="Assume a type for src files")
ap.add_argument("src", help="Tractor file or root directory with tractor files")
ap.add_argument("dest", help="Output sweep file")
ap.add_argument('-v', "--verbose", action='store_true')
ap.add_argument('-n', "--nbricks", type=int, help='number of bricks in a sweep', default=100)
ap.add_argument('-b', "--bricklist", help='filename with list of bricknames to include')
ap.add_argument("--numproc", type=int, help='number of concurrent processes to use', default=1)

SWEEP_DTYPE = np.dtype([
    ('BRICKID', '>i4'), 
    ('BRICKNAME', 'S8'), 
    ('OBJID', '>i4'), 
    ('BRICK_PRIMARY', '?'), 
    ('TYPE', 'S4'), 
    ('RA', '>f8'), 
    ('RA_IVAR', '>f4'), 
    ('DEC', '>f8'), 
    ('DEC_IVAR', '>f4'), 
    ('DECAM_FLUX', '>f4', (6,)), 
    ('DECAM_FLUX_IVAR', '>f4', (6,)), 
    ('DECAM_MW_TRANSMISSION', '>f4', (6,)), 
    ('DECAM_NOBS', 'u1', (6,)), 
    ('DECAM_RCHI2', '>f4', (6,)), 
    ('DECAM_FRACFLUX', '>f4', (6,)), 
    ('DECAM_FRACMASKED', '>f4', (6,)), 
    ('DECAM_FRACIN', '>f4', (6,)), 
    ('OUT_OF_BOUNDS', '?'), 
    ('DECAM_ANYMASK', '>i2', (6,)), 
    ('DECAM_ALLMASK', '>i2', (6,)), 
    ('WISE_FLUX', '>f4', (4,)), 
    ('WISE_FLUX_IVAR', '>f4', (4,)), 
    ('WISE_MW_TRANSMISSION', '>f4', (4,)), 
    ('WISE_NOBS', '>i2', (4,)), 
    ('WISE_FRACFLUX', '>f4', (4,)), 
    ('WISE_RCHI2', '>f4', (4,)), 
    ('DCHISQ', '>f4', (4,)), 
    ('FRACDEV', '>f4'), 
    ('EBV', '>f4')]
)


def main():
    ns = ap.parse_args()
            
    #- Load list of bricknames to use
    if ns.bricklist is not None:
        bricklist = np.loadtxt(ns.bricklist, dtype='S8')
    else:
        bricklist = None
        
    #- Loop over bricks collecting target selection flags
    #- and targets that passed the cuts
    t0 = time()

    def filter(filename):
        '''Wrapper function for performing target selection on a single brick file
        
        Used by _map_tractor() for parallel processing'''
        objects = fitsio.read(filename, 1, upper=True)
        chunk = np.empty(len(objects), dtype=SWEEP_DTYPE)

        for colname in SWEEP_DTYPE.names:
            if colname not in objects.dtype.names:
                # skip missing columns read_tractor
                continue
            chunk[colname][...] = objects[colname][...]
            
        return chunk

    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')
    fileid = np.zeros((), dtype='i8')
    chunks = []
    def collect_results(chunk):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if ns.verbose and nbrick % ns.nbricks == 0 and nbrick != 0:
            rate = nbrick / (time() - t0)
            data = np.concatenate(chunks)

            filename = os.path.join(ns.dest, 'sweep-%08d' % fileid)
            hdr = {}
            fitsio.write(filename, data, extname='SWEEP', header=hdr, clobber=True)

            # these are an in-place modifications
            while(chunks): chunks.pop()
            fileid[...] += 1
            
            print('{} bricks; {:.1f} bricks/sec'.format(nbrick, rate))

        nbrick[...] += 1
        chunks.append(chunk)
        return None

    map_tractor(filter, ns.src, \
            bricklist=bricklist, numproc=ns.numproc, reduce=collect_results)

    if ns.verbose:
        print ('written to', ns.dest)

if __name__ == "__main__":
    main()
