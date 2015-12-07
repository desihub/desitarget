#!/usr/bin/env python

from __future__ import print_function, division

import os, sys
import numpy as np

from desitarget import io
from desitarget.cuts import select_targets

import warnings
warnings.simplefilter('error')

from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("src", help="Tractor file or root directory with tractor files")
ap.add_argument("dest", help="Output target selection file")
ap.add_argument('-v', "--verbose", action='store_true')
ap.add_argument("--sweep", action='store_true', help='look for sweep files instead of tractor files')
ap.add_argument('-b', "--bricklist", help='filename with list of bricknames to include')
ap.add_argument("--numproc", type=int, help='number of concurrent processes to use', default=1)

ns = ap.parse_args()
infiles = io.list_sweepfiles(ns.src)
if len(infiles) == 0:
    infiles = io.list_tractorfiles(ns.src)
if len(infiles) == 0:
    print('FATAL: no sweep or tractor files found')
    sys.exit(1)
    
targets = select_targets(infiles, numproc=ns.numproc, verbose=ns.verbose)
io.write_targets(ns.dest, targets, indir=ns.src)

print('{} targets written to {}'.format(len(targets), ns.dest))

