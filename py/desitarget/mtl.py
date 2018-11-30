"""
desispec.mtl
============

Merged target lists?
"""

import numpy as np
import sys
from astropy.table import Table, join

from desitarget.targetmask import desi_mask, bgs_mask, mws_mask, obsmask, obsconditions
from desitarget.targets import calc_numobs, calc_priority


############################################################
@profile
def make_mtl(targets, zcat=None, trim=False):
    """Adds NUMOBS, PRIORITY, and OBSCONDITIONS columns to a targets table.

    Parameters
    ----------
    targets : :class:`~numpy.array`
        A numpy rec array with at least columns ``TARGETID``, ``DESI_TARGET``.
    zcat : :class:`~astropy.table.Table`, optional
        Redshift catalog table with columns ``TARGETID``, ``NUMOBS``, ``Z``,
        ``ZWARN``.
    trim : :class:`bool`, optional
        If ``True`` (default), don't include targets that don't need
        any more observations.  If ``False``, include every input target.

    Returns
    -------
    :class:`~astropy.table.Table`
        MTL Table with targets columns plus:

        * NUMOBS_MORE    - number of additional observations requested
        * PRIORITY       - target priority (larger number = higher priority)
        * OBSCONDITIONS  - replaces old GRAYLAYER
    """
    # ADM set up the default logger.
    from desiutil.log import get_logger
    log = get_logger()

    n = len(targets)
    # ADM if the input target columns were incorrectly called NUMOBS or priority
    # ADM rename them to NUMOBS_INIT or PRIORITY_INIT.
    for name in ['NUMOBS', 'PRIORITY']:
        targets.dtype.names = [name+'_INIT' if col==name else col for col in targets.dtype.names]

    # ADM if a redshift catalog was passed, order it to match the input targets
    # ADM catalog on 'TARGETID'.
    if zcat is not None:
        # ADM there might be a quicker way to do this?
        # ADM set up a dictionary of the indexes of each target id.
        d = dict(tuple(zip(targets["TARGETID"],np.arange(n))))
        # ADM loop through the zcat and look-up the index in the dictionary.
        zmatcher = np.array([d[tid] for tid in zcat["TARGETID"]])
        ztargets = zcat
        if ztargets.masked:
            unobs = ztargets['NUMOBS'].mask
            ztargets['NUMOBS'][unobs] = 0
            unobsz = ztargets['Z'].mask
            ztargets['Z'][unobsz] = -1
            unobszw = ztargets['ZWARN'].mask
            ztargets['ZWARN'][unobszw] = -1

    else:
        ztargets = Table()
        ztargets['TARGETID'] = targets['TARGETID']
        ztargets['NUMOBS'] = np.zeros(n, dtype=np.int32)
        ztargets['Z']      = -1 * np.ones(n, dtype=np.float32)
        ztargets['ZWARN']  = -1 * np.ones(n, dtype=np.int32)
        # ADM if zcat wasn't passed, there is a one-to-one correspondence
        # ADM between the targets and the zcat.
        zmatcher = np.arange(n)

    # ADM use passed value of NUMOBS_INIT instead of calling the memory-heavy calc_numobs.
    # ztargets['NUMOBS_MORE'] = np.maximum(0, calc_numobs(ztargets) - ztargets['NUMOBS'])
    ztargets['NUMOBS_MORE'] = np.maximum(0, targets[zmatcher]['NUMOBS_INIT'] - ztargets['NUMOBS'])

    # ADM assign priorities, note that only things in the zcat can have changed priorities.
    priority = calc_priority_no_table(targets[zmatcher], ztargets)

    # If priority went to 0==DONOTOBSERVE or 1==OBS or 2==DONE, then NUMOBS_MORE should also be 0
    ### mtl['NUMOBS_MORE'] = ztargets['NUMOBS_MORE']
    ii = (mtl['PRIORITY'] <= 2)
    log.info('{:d} of {:d} targets have priority zero, setting N_obs=0.'.format(np.sum(ii),len(mtl)))

    mtl['NUMOBS_MORE'][ii] = 0

    # Remove extra zcat columns from join(targets, zcat) that are not needed
    # for final MTL output
    for name in ['NUMOBS', 'Z', 'ZWARN']:
        mtl.remove_column(name)

    #- Set the OBSCONDITIONS mask for each target bit
    mtl['OBSCONDITIONS'] = np.zeros(n, dtype='i4')

    for mask, xxx_target in [
        (desi_mask, 'DESI_TARGET'),
        (mws_mask, 'MWS_TARGET'),
        (bgs_mask, 'BGS_TARGET') ]:
        for name in mask.names():
            #- which targets have this bit for this mask set?
            ii = (mtl[xxx_target] & mask[name]) != 0
            #- under what conditions can that bit be observed?
            if np.any(ii):
                mtl['OBSCONDITIONS'][ii] |= obsconditions.mask(mask[name].obsconditions)

    # Filter out any targets marked as done.
    if trim:
        notdone = mtl['NUMOBS_MORE'] > 0
        log.info('{:d} of {:d} targets are done, trimming these'.format(len(mtl) - np.sum(notdone),
                                                                     len(mtl)))
        mtl     = mtl[notdone]


    # Filtering can reset the fill_value, which is just wrong wrong wrong
    # See https://github.com/astropy/astropy/issues/4707
    # and https://github.com/astropy/astropy/issues/4708
    mtl['NUMOBS_MORE'].fill_value = -1


    return mtl


if __name__ == "__main__":

    import fitsio
    objs = fitsio.read("/project/projectdirs/desi/target/catalogs/dr7.1/PR372/targets-dr7.1-PR372.fits")
    print(len(objs))
    objs = objs[:10000000]
    print(len(objs))
    make_mtl(objs)



