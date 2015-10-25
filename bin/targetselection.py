#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import desitarget
from desitarget.io import read_tractor, write_targets
from desitarget.io import iter_tractor, map_tractor
from desitarget.io import fix_tractor_dr1_dtype
from desitarget.cuts import calc_numobs
from desitarget.cuts import select_targets
import desitarget.targets
### from desitarget.cuts_npyquery import select_targets

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

def do_one(filename):
    objects = read_tractor(filename)
    targetflag = select_targets(objects)
    keep = (targetflag != 0)
    return fix_tractor_dr1_dtype(objects[keep]), targetflag[keep]

def main():
    ns = ap.parse_args()
    
    #- Load list of bricknames to use
    if ns.bricklist is not None:
        bricklist = np.loadtxt(ns.bricklist, dtype='S8')
    else:
        bricklist = None
        
    #- Loop over bricks collecting target selection flags
    #- and targets that passed the cuts
    targetflags = list()
    targets = list()
    t0 = time()
    nbrick = 0
    if ns.numproc > 1:
        print('Using {} concurrent processes'.format(ns.numproc))
        results = map_tractor(do_one, ns.src, bricklist=bricklist, numproc=ns.numproc)
        #- unpack list of tuples into tuple of lists
        targets, targetflags = zip(*results)
    else:
        for brickname, filename in iter_tractor(ns.src):
            if (bricklist is not None) and (brickname not in bricklist):
                continue
            
            nbrick += 1
            xtargets, xflags = do_one(filename)
            targets.append(xtargets)
            targetflags.append(xflags)

            if nbrick % 50 == 0:
                rate = nbrick / (time() - t0)
                print('{} bricks; {:.1f} bricks/sec'.format(nbrick, rate))
    
    rate = len(targets) / (time() - t0)
    print('--> {:.1f} bricks/sec'.format(rate))
    
    #- convert list of per-brick items to single arrays across all bricks
    t1 = time()
    targetflags = np.concatenate(targetflags)
    targets = np.concatenate(targets)
    t2 = time()

    numobs = calc_numobs(targets, targetflags)
    t3 = time()
    targets = desitarget.targets.finalize(targets, targetflags, numobs)
    t4 = time()

    write_targets(ns.dest, targets, indir=ns.src)
    t5 = time()
    print ('written to', ns.dest)
    print(1, t1-t0)
    print(2, t2-t1)
    print(3, t3-t2)
    print(4, t4-t3)
    print(5, t5-t4)

if __name__ == "__main__":
    main()
