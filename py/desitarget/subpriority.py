"""
Get what subpriority was used by fiberassign
"""

import os.path
import numpy as np
import fitsio

from desiutil.log import get_logger

from desitarget.targetmask import desi_mask
from desitarget.geomask import match


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
    # SB Use geomask.match if input targets are unique,
    # SB otherwise use slower code that supports duplicates (e.g. secondaries)
    if len(np.unique(targets['TARGETID'])) == len(targets['TARGETID']):
        ii_targ, ii_over = match(targets['TARGETID'], override['TARGETID'])
        if len(ii_targ) > 0:
            targets['SUBPRIORITY'][ii_targ] = override['SUBPRIORITY'][ii_over]
        return np.sort(ii_targ)
    else:
        ii = np.where(np.isin(targets['TARGETID'], override['TARGETID']))[0]
        n = len(ii)
        if n > 0:
            # SB create TARGETID->SUBPRIORITY dict only for TARGETID in targets
            subprio_dict = dict()
            jj = np.where(np.isin(override['TARGETID'], targets['TARGETID']))[0]
            for tid, subprio in zip(
                    override['TARGETID'][jj], override['SUBPRIORITY'][jj]):
                subprio_dict[tid] = subprio

            for i in ii:
                tid = targets['TARGETID'][i]
                targets['SUBPRIORITY'][i] = subprio_dict[tid]

        return ii


def get_fiberassign_subpriorities(fiberassignfiles):
    """
    Return table of TARGETID, SUBPRIORITY used in input fiberassign files

    Args:
        fiberassignfiles: list of input fiberassign files

    Returns: dict[dark|bright|sky] = ndarray with columns TARGETID, SUBPRIORITY
    for targets matching that observing condition (or sky targets)
    """
    log = get_logger()

    # - allow duplicate inputs, but don't process multiple tiles
    processed = set()

    subprio_tables = dict(dark=list(), bright=list(), sky=list())

    for filename in fiberassignfiles:
        # - Have we already processed this file (e.g. from an earlier expid)?
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

            program = hdr['FAPRGRM'].lower()
            if program not in ('dark', 'bright'):
                log.warning(f"Skipping {filename} with FAPRGRM={program}")
                continue

            if hdr['SURVEY'].lower() != 'main':
                log.info(f"Skipping {filename} with SURVEY {hdr['SURVEY']} != main")
                continue

            log.info(f'Reading {filename}')
            sp = fx['TARGETS'].read(columns=['TARGETID', 'SUBPRIORITY', 'DESI_TARGET'])

        # - Separate skies from non-skies
        skymask = desi_mask.mask('SKY|SUPP_SKY|BAD_SKY')
        iisky = (sp['DESI_TARGET'] & skymask) != 0

        subprio_tables['sky'].append(sp[iisky])
        subprio_tables[program].append(sp[~iisky])

    log.info('Stacking individual fiberassign inputs')
    for program in subprio_tables.keys():
        subprio_tables[program] = np.hstack(subprio_tables[program])

    # - QA checks on basic assumptions about uniqueness
    log.info('Checking assumptions about TARGETID:SUBPRIORITY uniqueness')
    for program in ['dark', 'bright', 'sky']:
        subprio = subprio_tables[program]
        tid, sortedidx = np.unique(subprio['TARGETID'], return_index=True)

        # - sky can appear multiple times, but with same SUBPRIORITY
        if program == 'sky':
            subpriodict = dict()
            for targetid, sp in zip(
                    subprio['TARGETID'], subprio['SUBPRIORITY']):
                if targetid in subpriodict:
                    if sp != subpriodict[targetid]:
                        log.error(f'{program} TARGETID {targetid} has multiple subpriorities')
                else:
                    subpriodict[targetid] = sp
        # - but other programs should have each TARGETID exactly once
        else:
            if len(tid) != len(subprio):
                log.error(f'Some {program} TARGETIDs appear multiple times')

        log.info(f'Sorting {program} targets by TARGETID')
        subprio_tables[program] = subprio[sortedidx]

    return subprio_tables


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('-i', '--infiles', nargs='+', required=True,
                   help='Input fiberassign files with TARGETS HDU')
    p.add_argument('-o', '--outdir', required=True,
                   help='Output directory to keep dark/bright/sky TARGETID SUBPRIORITY tables')

    args = p.parse_args()
    log = get_logger()

    nfiles = len(args.infiles)
    log.info(f'Getting target subpriorities from {nfiles} fiberassign files')
    subprio_tables = get_fiberassign_subpriorities(args.infiles)

    if 'DESI_ROOT' in os.environ:
        desiroot = os.path.normpath(os.getenv('DESI_ROOT'))
    else:
        desiroot = None

    for program, subprio in subprio_tables.items():
        hdr = fitsio.FITSHDR()
        hdr['FAPRGRM'] = program
        for i, filename in enumerate(args.infiles):
            if desiroot and filename.startswith(desiroot):
                filename = filename.replace(desiroot, '$DESI_ROOT')

            hdr[f'INFIL{i:03d}'] = filename

        outfile = os.path.join(args.outdir, f'subpriorities-{program}.fits')
        fitsio.write(outfile, subprio, extname='SUBPRIORITY', header=hdr, clobber=True)
        log.info(f'Wrote {outfile}')
