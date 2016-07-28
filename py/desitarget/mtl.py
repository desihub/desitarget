import numpy as np
from astropy.table import Table, join

from .targetmask import desi_mask, obsmask
from .targets import calc_numobs, calc_priority

def make_mtl(targets, zcat=None, trim=False):
    """Adds NUMOBS, PRIORITY, and GRAYLAYER columns to a targets table.

    Parameters
    ----------
    targets : :class:`~astropy.table.Table`
        A table with columns ``TARGETID``, ``DESI_TARGET``.
    zcat : :class:`~astropy.table.Table`, optional
        Redshift catalog table with columns ``TARGETID``, ``NUMOBS``, ``Z``,
        ``ZWARN``.
    trim : :class:`bool`, optional
        If ``True``, don't include targets that don't need
        any more observations.  If ``False`` (default), include every input target.

    Returns
    -------
    :class:`~astropy.table.Table`
        MTL Table with targets columns plus

        * NUMOBS_MORE - number of additional observations requested
        * PRIORITY - target priority (larger number = higher priority)
        * GRAYLAYER - can this be observed during gray time?

    Notes
    -----
        TODO: Check if input targets is ever altered (it shouldn't...).
    """
    n = len(targets)
    targets = Table(targets)
    if zcat is not None:
        ztargets = join(targets, zcat['TARGETID', 'NUMOBS', 'Z', 'ZWARN'],
                            keys='TARGETID', join_type='outer')
        if ztargets.masked:
            unobs = ztargets['NUMOBS'].mask
            ztargets['NUMOBS'][unobs] = 0
    else:
        ztargets = targets.copy()
        ztargets['NUMOBS'] = np.zeros(n, dtype=np.int32)
        ztargets['Z'] = -1 * np.ones(n, dtype=np.float32)
        ztargets['ZWARN'] = -1  * np.ones(n, dtype=np.int32)

    ztargets['NUMOBS_MORE'] = np.maximum(0, calc_numobs(ztargets) - ztargets['NUMOBS'])

    mtl = ztargets.copy()
    ### mtl['NUMOBS_MORE'] = ztargets['NUMOBS_MORE']
    mtl['PRIORITY'] = calc_priority(ztargets)

    #- If priority went to 0, then NUMOBS_MORE should also be 0
    ii = (mtl['PRIORITY'] == 0)
    mtl['NUMOBS_MORE'][ii] = 0

    #- remove extra zcat columns from join(targets, zcat) that are not needed
    #- for final MTL output
    for name in ['NUMOBS', 'Z', 'ZWARN']:
        mtl.remove_column(name)

    #- ELGs can be observed during gray time
    graylayer = np.zeros(n, dtype='i4')
    iselg = (mtl['DESI_TARGET'] & desi_mask.ELG) != 0
    graylayer[iselg] = 1
    mtl['GRAYLAYER'] = graylayer

    if trim:
        notdone = mtl['NUMOBS_MORE'] > 0
        mtl = mtl[notdone]

    #- filtering can reset the fill_value, which is just wrong wrong wrong
    #- See https://github.com/astropy/astropy/issues/4707
    #- and https://github.com/astropy/astropy/issues/4708
    mtl['NUMOBS_MORE'].fill_value = -1

    return mtl
