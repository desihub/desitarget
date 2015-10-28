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

ap = ArgumentParser()
### ap.add_argument("--type", choices=["tractor"], default="tractor", help="Assume a type for src files")
ap.add_argument("src", help="Tractor file or root directory with tractor files")
ap.add_argument("dest", help="Output sweep file")
ap.add_argument('-v', "--verbose", action='store_true')
ap.add_argument('-b', "--bricklist", help='filename with list of bricknames to include')
ap.add_argument("--numproc", type=int, help='number of concurrent processes to use', default=1)

def write_sweep(filename, data):
    import fitsio
    hdr = {}
    fitsio.write(filename, data, extname='SWEEP', header=hdr, clobber=True)

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
        objects = read_tractor(filename)

        return fix_tractor_dr1_dtype(objects)

    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')
    fileid = np.zeros((), dtype='i8')
    objects = []
    def collect_results(data):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if ns.verbose and nbrick % 100 == 0 and nbrick != 0:
            rate = nbrick / (time() - t0)
            c = np.concatenate(objects)
            write_sweep(os.path.join(ns.dest, 'sweep-%08d.fits' % fileid), c)  

            # these are an in-place modifications
            while(objects): objects.pop()
            fileid[...] += 1
            
            print('{} bricks; {:.1f} bricks/sec'.format(nbrick, rate))

        nbrick[...] += 1
        objects.append(data)
        return None

    map_tractor(filter, ns.src, \
            bricklist=bricklist, numproc=ns.numproc, reduce=collect_results)

    if ns.verbose:
        print ('written to', ns.dest)

if __name__ == "__main__":
    main()
