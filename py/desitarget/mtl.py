"""
desispec.mtl
============

Merged target lists?
"""

import numpy as np
import sys
from astropy.table import Table, join

from .targetmask import desi_mask, bgs_mask, mws_mask, obsmask, obsconditions
from .targets    import calc_numobs, calc_priority, encode_mtl_targetid


############################################################
def make_mtl(targets, zcat=None, trim=False):
    """Adds NUMOBS, PRIORITY, and OBSCONDITIONS columns to a targets table.

    Parameters
    ----------
    targets : :class:`~astropy.table.Table`
        A table with columns ``TARGETID``, ``DESI_TARGET``.
    zcat : :class:`~astropy.table.Table`, optional
        Redshift catalog table with columns ``TARGETID``, ``NUMOBS``, ``Z``,
        ``ZWARN``.
    trim : :class:`bool`, optional
        If ``True`` (default), don't include targets that don't need
        any more observations.  If ``False``, include every input target.

    Returns
    -------
    :class:`~astropy.table.Table`
        MTL Table with targets columns plus

        * NUMOBS_MORE    - number of additional observations requested
        * PRIORITY       - target priority (larger number = higher priority)
        * OBSCONDITIONS  - replaces old GRAYLAYER

    Notes
    -----
        TODO: Check if input targets is ever altered (it shouldn't...).
    """
    n       = len(targets)
    targets = Table(targets)

    # Create redshift catalog
    if zcat is not None:

        ztargets = join(targets, zcat['TARGETID', 'NUMOBS', 'Z', 'ZWARN'],
                            keys='TARGETID', join_type='outer')
        if ztargets.masked:
            unobs = ztargets['NUMOBS'].mask
            ztargets['NUMOBS'][unobs] = 0
            unobsz = ztargets['Z'].mask
            ztargets['Z'][unobsz] = -1
            unobszw = ztargets['ZWARN'].mask
            ztargets['ZWARN'][unobszw] = -1


    else:
        ztargets           = targets.copy()
        ztargets['NUMOBS'] = np.zeros(n, dtype=np.int32)
        ztargets['Z']      = -1 * np.ones(n, dtype=np.float32)
        ztargets['ZWARN']  = -1 * np.ones(n, dtype=np.int32)

    ztargets['NUMOBS_MORE'] = np.maximum(0, calc_numobs(ztargets) - ztargets['NUMOBS'])

    # Create MTL
    mtl = ztargets.copy()

    # Assign priorities
    mtl['PRIORITY'] = calc_priority(ztargets)

    # If priority went to 0==DONOTOBSERVE or 1==OBS or 2==DONE, then NUMOBS_MORE should also be 0
    ### mtl['NUMOBS_MORE'] = ztargets['NUMOBS_MORE']
    ii = (mtl['PRIORITY'] <= 2)
    print('{:d} of {:d} targets have priority zero, setting N_obs=0.'.format(np.sum(ii),len(mtl)))

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
        print('{:d} of {:d} targets are done, trimming these'.format(len(mtl) - np.sum(notdone),
                                                                     len(mtl)))
        mtl     = mtl[notdone]


    # Filtering can reset the fill_value, which is just wrong wrong wrong
    # See https://github.com/astropy/astropy/issues/4707
    # and https://github.com/astropy/astropy/issues/4708
    mtl['NUMOBS_MORE'].fill_value = -1


    return mtl
