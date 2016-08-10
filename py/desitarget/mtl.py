import numpy as np
from astropy.table import Table, join

from .targetmask import desi_mask, obsmask
from .targets    import calc_numobs, calc_priority, encode_mtl_targetid

############################################################
def make_mtl(targets, zcat=None, trim=True, truth=None):
    """Adds NUMOBS, PRIORITY, and GRAYLAYER columns to a targets table.

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

        * NUMOBS_MORE - number of additional observations requested
        * PRIORITY    - target priority (larger number = higher priority)
        * GRAYLAYER   - can this be observed during gray time?

    Notes
    -----
        TODO: Check if input targets is ever altered (it shouldn't...).
    """
    n       = len(targets)
    targets = Table(targets)

    # FIXME (APC): NUMOBS vs. NUMOBS_MORE?

    # Create redshift catalog
    if zcat is not None:
        ztargets = join(targets, zcat['TARGETID', 'NUMOBS', 'Z', 'ZWARN'],
                            keys='TARGETID', join_type='outer')
        if ztargets.masked:
            unobs = ztargets['NUMOBS'].mask
            ztargets['NUMOBS'][unobs] = 0
    else:
        ztargets           = targets.copy()
        ztargets['NUMOBS'] = np.zeros(n, dtype=np.int32)
        ztargets['Z']      = -1 * np.ones(n, dtype=np.float32)
        ztargets['ZWARN']  = -1 * np.ones(n, dtype=np.int32)

    ztargets['NUMOBS_MORE'] = np.maximum(0, calc_numobs(ztargets) - ztargets['NUMOBS'])

    # Create MTL
    print('DEBUG: ztargets before copy: %d'%(len(ztargets)))
    mtl = ztargets.copy()
    print('DEBUG: mtl after copy: %d'%(len(mtl)))

    if truth is not None:
        mtl_truth = truth.copy()

    # Assign priorities
    mtl['PRIORITY'] = calc_priority(ztargets)

    # If priority went to 0, then NUMOBS_MORE should also be 0
    ### mtl['NUMOBS_MORE'] = ztargets['NUMOBS_MORE']
    ii = (mtl['PRIORITY'] == 0)
    print('{:d} of {:d} targets have priority zero, setting N_obs=0.'.format(np.sum(ii),len(mtl)))
    mtl['NUMOBS_MORE'][ii] = 0

    # Remove extra zcat columns from join(targets, zcat) that are not needed
    # for final MTL output
    for name in ['NUMOBS', 'Z', 'ZWARN']:
        mtl.remove_column(name)

    # FIXME (APC): special-case fixes and use of GRAYLAYER seem awkward here.
    #              Will replace this with calc_obsconditions?

    # ELGs can be observed during gray time
    graylayer        = np.zeros(n, dtype='i4')
    iselg            = (mtl['DESI_TARGET'] & desi_mask.ELG) != 0
    graylayer[iselg] = 1
    mtl['GRAYLAYER'] = graylayer

    # Filter out any targets marked as done
    if trim:
        notdone = mtl['NUMOBS_MORE'] > 0
        print('{:d} of {:d} targets are done, trimming these'.format(len(mtl) - np.sum(notdone),
                                                                     len(mtl)))
        mtl     = mtl[notdone]

        # Filter truth in the same way
        if truth is not None:
            mtl_truth = mtl_truth[notdone]

    # Filtering can reset the fill_value, which is just wrong wrong wrong
    # See https://github.com/astropy/astropy/issues/4707
    # and https://github.com/astropy/astropy/issues/4708
    mtl['NUMOBS_MORE'].fill_value = -1

    # FIXME (APC): is this the place to do this?
    # Assign targetids last and use mtl; might encode some ordering info that
    # would be affected by filtering.

    # mtl['TARGETID'] = encode_mtl_targetid(mtl)
    # Not ideal, just use row number for fast inverse from tile maps, and unpack
    # back to original targetids when making the catalog. Could also dump the
    # original or 'souped up' target id in the truth table.

    mtl['TARGETID'] = np.arange(0,len(mtl),dtype=np.int64)

    if truth is not None:
        mtl_truth['TARGETID'] = mtl['TARGETID']

    if truth is None:
        return mtl
    else:
        return mtl, mtl_truth
