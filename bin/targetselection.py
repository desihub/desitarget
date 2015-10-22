#!/usr/bin/env python

from __future__ import print_function, division

import numpy
from astropy.table import Table

import desitarget
from desitarget.io import read_tractor, iter_tractor, write_targets
from desitarget.cuts import LRG, ELG, BGS, QSO
from desitarget import targetmask 

from argparse import ArgumentParser
import os

default_outfile = 'desi-targets-{}.fits'.format(desitarget.__version__)

ap = ArgumentParser()
ap.add_argument("--type", choices=["tractor"], default="tractor", help="Assume a type for src files")
# ap.add_argument("--infile", help="File that stores Candidates/Objects")
# ap.add_argument("--indir", help="Base Legacy Survey data release directory")
# ap.add_argument("--output", help="File that stores targets", default=default_outfile)
ap.add_argument("src", help="File that stores Candidates/Objects. Ending with a '/' will be a directory")
ap.add_argument("dest", help="File that stores targets. A directory if src is a directory.")

TYPES = {
    'LRG': LRG,
    'ELG': ELG,
    'BGS': BGS,
    'QSO': QSO,
}

def main():
    ns = ap.parse_args()

    # if ns.indir is not None:
    #     allfiles = get_tractor_files(ns.indir)
    # else:
    #     allfiles = [ns.infile, ]
    # 
    # targets = list()
    # desi_targetmask = list()
    # for infile in allfiles:
    #     candidates = read_tractor(infile)
    # 
    #     # FIXME: fits doesn't like u8; there must be a workaround
    #     # but lets stick with i8 for now.
    #     tsbits = numpy.zeros(len(candidates), dtype='i8')
    # 
    #     for t in TYPES.keys():
    #         cut = TYPES[t]
    #         bitfield = targetmask.mask(t)
    #         with numpy.errstate(all='ignore'):
    #             mask = cut.apply(candidates)
    #         tsbits[mask] |= bitfield
    #         assert ((tsbits & bitfield) != 0).sum() == mask.sum()
    #         print (t, 'selected', mask.sum())
    if ns.src.endswith('/'):
        for brickname, filename in iter_tractor(ns.src):
            dest = os.path.join(ns.dest, os.path.relpath(filename, ns.src))
            try:
                os.makedirs(os.path.dirname(dest))
            except:
                pass
            do_one(filename, dest)
    else:
        do_one(ns.src, ns.dest)

def do_one(src, dest):
    candidates = read_tractor(src)

        keep = (tsbits != 0)
        targets.append(candidates[keep])
        desi_targetmask.append(tsbits[keep])

    #- Merge individual bricks into one
    targets = numpy.hstack(targets)
    desi_targetmask = numpy.concatenate(desi_targetmask)

    write_targets(dest, candidates, tsbits)
    print ('written to', dest)

if __name__ == "__main__":
    main()
