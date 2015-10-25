#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import desitarget
from desitarget.io import read_tractor, iter_tractor, write_targets
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
    for brickname, filename in iter_tractor(ns.src):
        if (bricklist is not None) and (brickname not in bricklist):
            continue
            
        nbrick += 1
        objects = read_tractor(filename)
        targetflag = select_targets(objects)
        keep = (targetflag != 0)
        targetflags.append(targetflag[keep])
        targets.append(fix_tractor_dr1_dtype(objects[keep]))

        if nbrick % 50 == 0:
            rate = nbrick / (time() - t0)
            print('{} bricks; {:.1f} bricks/sec'.format(nbrick, rate))
                
    #- convert list of per-brick items to single arrays across all bricks
    targetflags = np.concatenate(targetflags)
    targets = np.concatenate(targets)

    numobs = calc_numobs(targets, targetflags)
    targets = desitarget.targets.finalize(targets, targetflags, numobs)

    write_targets(ns.dest, targets)
    print ('written to', ns.dest)

if __name__ == "__main__":
    main()
