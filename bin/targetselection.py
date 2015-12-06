#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import desitarget
import desitarget.targets
from desitarget.io import read_tractor, write_targets
from desitarget.io import map_tractor, map_sweep
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
ap.add_argument("src", help="Tractor file or root directory with tractor files")
ap.add_argument("dest", help="Output target selection file")
ap.add_argument('-v', "--verbose", action='store_true')
ap.add_argument("--sweep", action='store_true', help='look for sweep files instead of tractor files')
ap.add_argument('-b', "--bricklist", help='filename with list of bricknames to include')
ap.add_argument("--numproc", type=int, help='number of concurrent processes to use', default=1)

def newmain():
    infiles = io.list_sweepfiles(src)
    if len(infiles) == 0:
        infiles = io.list_tractorfiles(src)
    if len(infiles) == 0:
        print('FATAL: no sweep or tractor files found')
        sys.exit(1)
        
    targets = select_targets(infiles)

def main():
    ns = ap.parse_args()
            
    #- Load list of bricknames to use
    if ns.bricklist is not None:
        bricklist = np.loadtxt(ns.bricklist, dtype='S8')
    else:
        bricklist = None
        
    #- function to run on every brick/sweep file
    def _select_targets_brickfile(filename):
        '''Returns targets in filename that pass the cuts
        
        Used by map_tractor / map_sweep for parallel processing'''
        objects = read_tractor(filename)
        desi_target, bgs_target, mws_target = select_targets(objects)
        
        #- desi_target includes BGS_ANY and MWS_ANY, so we can filter just
        #- on desi_target != 0
        keep = (desi_target != 0)
        objects = objects[keep]
        desi_target = desi_target[keep]
        bgs_target = bgs_target[keep]
        mws_target = mws_target[keep]

        #- Add *_target mask columns
        targets = desitarget.targets.finalize(
            objects, desi_target, bgs_target, mws_target)

        return fix_tractor_dr1_dtype(targets)

    # Counter for number of bricks processed;
    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    def update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if ns.verbose and nbrick%50 == 0 and nbrick>0:
            rate = nbrick / (time() - t0)
            print('{} files; {:.1f} files/sec'.format(nbrick, rate))

        nbrick[...] += 1    # this is an in-place modification
        return result

    #- Parallel loop over bricks collecting target selection flags
    #- and targets that passed the cuts
    t0 = time()
    if ns.sweep:
        infiles, targets = \
            map_sweep(_select_targets_brickfile, ns.src, \
                numproc=ns.numproc, reduce=update_status)
    else:
        bnames, infiles, targets = \
            map_tractor(_select_targets_brickfile, ns.src, \
                bricklist=bricklist, numproc=ns.numproc, reduce=update_status)

    #- convert list of per-brick items to single arrays across all bricks
    targets = np.concatenate(targets)

    t1 = time()
    write_targets(ns.dest, targets, indir=ns.src)
    t2 = time()
    print('{} targets written to {}'.format(len(targets), ns.dest))
    if ns.verbose:
        print('Timing:')
        print('  Target selection {:.1f} sec'.format(t1-t0))
        print('  Write output     {:.1f} sec'.format(t2-t1))

if __name__ == "__main__":
    main()
