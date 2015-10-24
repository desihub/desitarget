#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.table import Table

import desitarget
from desitarget.io import read_tractor, iter_tractor, write_targets
import desitarget.cuts
from desitarget import targetmask 

from argparse import ArgumentParser
import os, sys
from time import time

default_outfile = 'desi-targets-{}.fits'.format(desitarget.__version__)

ap = ArgumentParser()
### ap.add_argument("--type", choices=["tractor"], default="tractor", help="Assume a type for src files")
ap.add_argument("src", help="File that stores Candidates/Objects. Ending with a '/' will be a directory")
ap.add_argument("dest", help="File that stores targets. A directory if src is a directory.")

def main():
    verbose = False
    ns = ap.parse_args()
    if os.path.isdir(ns.src):
        #- Loop over bricks, collecting target selection bitmask (tsbits)
        #- and candidates that pass the cuts
        tsbits = list()
        candidates = list()
        t0 = time()
        nbrick = 0
        for brickname, filename in iter_tractor(ns.src):
            nbrick += 1
            if verbose:
                print(brickname, ':', end=' ')
            ### brick_tsbits, brick_candidates = do_one(filename, verbose=verbose)
            brick_tsbits, brick_candidates = do_sjb(filename)
            tsbits.append(brick_tsbits)

            #- Hack to work around tractor datamodel inconsistency
            if brick_candidates['TYPE'].dtype != 'S4':
                print("fixing TYPE dtype for brick", brickname)
                dt = brick_candidates.dtype.descr
                for i in range(len(dt)):
                    if dt[i][0] == 'TYPE':
                        dt[i] = ('TYPE', '|S4')
                brick_candidates = brick_candidates.astype(np.dtype(dt))

            candidates.append(brick_candidates)
            if nbrick % 50 == 0:
                rate = nbrick / (time() - t0)
                print('{} bricks; {:.1f} bricks/sec'.format(nbrick, rate))
            if brick_candidates.dtype != candidates[0].dtype:
                print('ERROR: incompatible dtypes in brick', brickname)
                print(brick_candidates.dtype)
                print(candidates[0].dtype)
                for name in brick_candidates.dtype.names:
                    if brick_candidates[name].dtype != candidates[0][name].dtype:
                        print(name, brick_candidates[name].dtype, candidates[0][name].dtype)
                sys.exit(1)
                    
        #- convert list of per-brick items to single arrays across all bricks
        tsbits = np.concatenate(tsbits)
        candidates = np.concatenate(candidates)
    else:
        tsbits, candidates = do_one(ns.src, ns.dest)

    write_targets(ns.dest, candidates, tsbits)
    print ('written to', ns.dest)

def do_sjb(src):
    objects = read_tractor(src)
    tsbits = desitarget.cuts.select_targets(objects)
    keep = (tsbits != 0)
    return tsbits[keep], objects[keep]

def do_one(src, verbose=False):
    candidates = read_tractor(src)

    # FIXME: fits doesn't like u8; there must be a workaround
    # but lets stick with i8 for now.
    tsbits = np.zeros(len(candidates), dtype='i8')

    for t, cut in desitarget.cuts.types.items():
        bitfield = targetmask.mask(t)
        with np.errstate(all='ignore'):
            mask = cut.apply(candidates)
        tsbits[mask] |= bitfield
        nselected = np.count_nonzero(mask)
        assert np.count_nonzero(tsbits & bitfield) == nselected
        # print (' ', t, 'selected', np.count_nonzero(mask))
        if verbose:
            print('{:5d} {:s}'.format(nselected, t), end='')
    
    if verbose:
        print()

    keep = (tsbits != 0)
    return tsbits[keep], candidates[keep]

if __name__ == "__main__":
    main()
