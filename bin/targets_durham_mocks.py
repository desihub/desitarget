#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import desitarget
import desitarget.targets
from desitarget.io import read_tractor, write_targets
from desitarget.io import read_mock_durham
from desitarget.cuts import calc_numobs
from desitarget.cuts import select_targets
### from desitarget.cuts_npyquery import select_targets

from argparse import ArgumentParser
import os, sys
from time import time

ap = ArgumentParser()
ap.add_argument("src_core", help="Mock lightcone 'core' file")
ap.add_argument("src_photo", help="Mock lightcone 'photometry' file")
ap.add_argument("dest", help="Output target selection file")
ap.add_argument('-v', "--verbose", action='store_true')

def main():
    ns = ap.parse_args()

    t0 = time()
    objects = read_mock_durham(ns.src_core, ns.src_photo)

    t1 = time()
    targetflags = select_targets(objects)
    keep = (targetflags !=0)

    targets = objects[keep]
    targetflag = targetflags[keep]

    t2 = time()
    numobs = calc_numobs(targets, targetflag)
    targets = desitarget.targets.finalize(targets, targetflag, numobs)

    t3 = time()
    write_targets(ns.dest, targets)
    t4 = time()
    if ns.verbose:
        print ('written {} targets to {}'.format(len(targets), ns.dest))
        print('Read mock file {:.1f} sec'.format(t1-t0))
        print('Make target selection  {:.1f} sec'.format(t2-t1))
        print('Compute numobs, finalize target selection {:.1f} sec'.format(t3-t2))
        print('Write output  {:.1f} sec'.format(t4-t3))

if __name__ == "__main__":
    main()
