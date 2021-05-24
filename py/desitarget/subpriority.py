#!/usr/bin/env python

"""
Get what subpriority was used by fiberassign
"""

import os.path
import numpy as np
import fitsio
from desiutil.log import get_logger

def override_subpriority(targets, override):
    """
    Override SUBPRIORITY column in targets for those in override table

    Args:
        targets: table with columns TARGETID and SUBPRIORITY
        override: table with columns TARGETID and SUBPRIORITY

    Returns:
        indices of targets table rows that were changed

    Modifies ``targets`` table in-place without copying memory.
    Rows in ``targets`` that aren't in ``override`` are unchanged.
    Rows in ``override`` that aren't in ``targets`` are ignored.
    """
    log = get_logger()
    ii = np.where(np.isin(targets['TARGETID'], override['TARGETID']))[0]
    n = len(ii)
    if n > 0:
        subprio_dict = dict()
        for tid, subprio in zip(override['TARGETID'], override['SUBPRIORITY']):
            subprio_dict[tid] = subprio

        for i in ii:
            tid = targets['TARGETID'][i]
            targets['SUBPRIORITY'][i] = subprio_dict[tid]

    return ii


def get_fiberassign_subpriorities(fiberassignfiles,
        survey, program=None, expect_unique=False):
    """
    TODO: document
    """
    log = get_logger()

    #- allow duplicate inputs, but don't process multiple tiles
    processed = set()

    subpriorities = list()
    for filename in fiberassignfiles:
        #- Have we already processed this file (e.g. from an earlier expid)?
        basename = os.path.basename(filename)
        if basename in processed:
            continue
        else:
            processed.add(basename)

        with fitsio.FITS(filename) as fx:
            hdr = fx[0].read_header()

            if 'SURVEY' not in hdr:
                log.warning(f"Skipping {filename} missing SURVEY keyword")
                continue

            if 'FAPRGRM' not in hdr:
                log.warning(f"Skipping {filename} missing FAPRGRM keyword")
                continue

            if survey and (hdr['SURVEY'].lower() != survey.lower()):
                log.info(f"Skipping {filename} with SURVEY {hdr['SURVEY']} != {survey}")
                continue

            if program and (hdr['FAPRGRM'].lower() != program.lower()):
                log.info(f"Skipping {filename} with FAPRGRM {hdr['FAPRGRM']} != {program}")
                continue

            log.info(f'Reading {filename}')
            sp = fx['TARGETS'].read(columns=['TARGETID', 'SUBPRIORITY'])

        subpriorities.append(sp)

    log.info('Stacking individual fiberassign inputs')
    subpriorities = np.hstack(subpriorities)

    #- QA checks on basic assumptions
    log.info('Checking assumptions about TARGETID:SUBPRIORITY uniqueness')
    tid, sortedidx = np.unique(subpriorities['TARGETID'], return_index=True)
    if len(tid) != len(subpriorities):
        if expect_unique:
            log.warning('Some TARGETIDs appear multiple times')

        subpriodict = dict()
        for targetid, subprio in zip(
                subpriorities['TARGETID'], subpriorities['SUBPRIORITY']):
            if targetid in subpriodict:
                if subprio != subpriodict[targetid]:
                    log.error(f'TARGETID {targetid} has multiple subpriorities')
            else:
                subpriodict[targetid] = subprio

    log.info('Sorting by TARGETID')
    subpriorities = subpriorities[sortedidx]

    return subpriorities

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('-i', '--infiles', nargs='+', required=True,
            help='Input fiberassign files with TARGETS HDU')
    p.add_argument('-o', '--outfile', required=True,
            help='Output FITS file to keep TARGETID SUBPRIORITY')
    p.add_argument('--survey', default='main',
            help='SURVEY survey filter')
    p.add_argument('--faprgrm', required=True,
            help='FAPRGRM program filter')
    
    args = p.parse_args()
    log = get_logger()

    nfiles = len(args.infiles)
    log.info(f'Getting target subpriorities from {nfiles} fiberassign files')
    subpriorities = get_fiberassign_subpriorities(
            args.infiles, args.survey, args.faprgrm, expect_unique=True)
    log.info(f'{len(subpriorities)} targets')

    if 'DESI_ROOT' in os.environ:
        desiroot = os.path.normpath(os.getenv('DESI_ROOT'))
    else:
        desiroot = None

    hdr = fitsio.FITSHDR()
    for i, filename in enumerate(args.infiles):
        if desiroot and filename.startswith(desiroot):
            filename = filename.replace(desiroot, '$DESI_ROOT')

        hdr[f'INFIL{i:03d}'] = filename

    fitsio.write(args.outfile, subpriorities, extname='SUBPRIORITY', header=hdr, clobber=True)
    log.info(f'Wrote {args.outfile}')



