#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

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
ap.add_argument("src", help="Tractor file or root directory with tractor files")
ap.add_argument("dest", help="Output target selection file")
ap.add_argument('-v', "--verbose", action='store_true')

def main():
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
            if ns.verbose:
                print(brickname, ':', end=' ')
            ### brick_tsbits, brick_candidates = do_one(filename, verbose=ns.verbose)
            brick_tsbits, brick_candidates = do_sjb(filename)
            tsbits.append(brick_tsbits)

            #- Hack to work around DR1 tractor datamodel inconsistency
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
