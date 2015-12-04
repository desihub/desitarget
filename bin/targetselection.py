#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import desitarget
import desitarget.targets
from desitarget.io import read_tractor, write_targets
from desitarget.io import iter_tractor, map_tractor
from desitarget.io import fix_tractor_dr1_dtype
from desitarget.cuts import calc_numobs
from desitarget.cuts import select_targets
### from desitarget.cuts_npyquery import select_targets

import warnings
warnings.simplefilter('error')

from argparse import ArgumentParser
import os, sys
from time import time

ap = ArgumentParser()
### ap.add_argument("--type", choices=["tractor"], default="tractor", help="Assume a type for src files")
ap.add_argument("src", help="Tractor file or root directory with tractor files")
ap.add_argument("dest", help="Output target selection file")
ap.add_argument('-v', "--verbose", action='store_true')
ap.add_argument('-b', "--bricklist", help='filename with list of bricknames to include')
ap.add_argument("--numproc", type=int, help='number of concurrent processes to use', default=1)

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

    def _select_targets_brickfile(filename):
        '''Wrapper function for performing target selection on a single brick file
        
        Used by _map_tractor() for parallel processing'''
        objects = read_tractor(filename)
        desi_target, bgs_target, mws_target = select_targets(objects)
        
        #- desi_target includes BGS_ANY and MWS_ANY, so we can filter just
        #- on desi_target != 0
        keep = (desi_target != 0)
        objects = objects[keep]
        desi_target = desi_target[keep]
        bgs_target = bgs_target[keep]
        mws_target = mws_target[keep]

        targets = desitarget.targets.finalize(objects, desi_target, bgs_target, mws_target)

        return fix_tractor_dr1_dtype(targets)

    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    def collect_results(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if ns.verbose and nbrick%50 == 0:
            rate = nbrick / (time() - t0)
            print('{} bricks; {:.1f} bricks/sec'.format(nbrick, rate))

        # this is an in-place modification
        nbrick[...] += 1

        return result

    bnames, bfiles, targets = \
        map_tractor(_select_targets_brickfile, ns.src, \
            bricklist=bricklist, numproc=ns.numproc, reduce=collect_results)

    #- convert list of per-brick items to single arrays across all bricks
    t1 = time()
    targets = np.concatenate(targets)
    t2 = time()

    write_targets(ns.dest, targets, indir=ns.src)
    t3 = time()
    if ns.verbose:
        print ('written to', ns.dest)
        print('Target selection {:.1f} sec'.format(t1-t0))
        print('Combine results  {:.1f} sec'.format(t2-t1))
        print('Write output     {:.1f} sec'.format(t3-t2))

if __name__ == "__main__":
    main()
