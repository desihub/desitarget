import numpy
from desitarget.io import read_tractor, write_targets
from desitarget.cuts import LRG, ELG, BGS, QSO
from desitarget import targetmask 

from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--type", choices=["tractor"], default="tractor", help="Assume a type for src files")
ap.add_argument("src", help="File that stores Candidates/Objects")
ap.add_argument("dest", help="File that stores targets")

TYPES = {
    # FIXME: this is verbose,
    # but since the list will change, may better leave it this
    # ugly for now.
    'LRG': LRG,
    'ELG': ELG,
    'BGS': BGS,
    'QSO': QSO,
}

def main():
    ns = ap.parse_args()
    candidates = read_tractor(ns.src)

    # lets not set the bits yet.
    tsbits = numpy.zeros(len(candidates), dtype=('u1', 8))

    for t in TYPES.keys():
        cut = TYPES[t]
        bitnum = targetmask.bitnum(t)
        mask = cut.apply(candidates)
        tsbits[:, bitnum] = mask
 
    print ('selected', tsbits.sum(axis=0, dtype='i4'))
    write_targets(ns.dest, candidates, tsbits)
    print ('written to', ns.dest)

if __name__ == "__main__":
    main()
