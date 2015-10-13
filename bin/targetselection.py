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
    'LRG': LRG,
    'ELG': ELG,
    'BGS': BGS,
    'QSO': QSO,
}

def main():
    ns = ap.parse_args()

    candidates = read_tractor(ns.src)

    # FIXME: fits doesn't like u8; there must be a workaround
    # but lets stick with i8 for now.
    tsbits = numpy.zeros(len(candidates), dtype='i8')

    for t in TYPES.keys():
        cut = TYPES[t]
        bitfield = targetmask.mask(t)
        with numpy.errstate(all='ignore'):
            mask = cut.apply(candidates)
        tsbits[mask] |= bitfield
        assert ((tsbits & bitfield) != 0).sum() == mask.sum()
        print (t, 'selected', mask.sum())

    write_targets(ns.dest, candidates, tsbits)
    print ('written to', ns.dest)

if __name__ == "__main__":
    main()
