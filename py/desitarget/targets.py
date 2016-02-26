import numpy as np
import numpy.lib.recfunctions as rfn

from astropy.table import Table

from desitarget import desi_mask, bgs_mask, mws_mask
from desitarget import obsmask

def calc_priority(targets):
    '''
    Calculate target priorities given observation state and target masks

    Args:
        targets: numpy structured array or astropy Table of targets, including
            columns DESI_TARGET, BGS_TARGET, and MWS_TARGET
    
    Returns:
        integer array of priorities

    Notes:
        If a target passes more than one selection, the highest priority wins
    '''
    targets = Table(targets).copy()
    if 'NUMOBS' not in targets.colnames:
        targets['NUMOBS'] = np.zeros(len(targets), dtype=np.int32)
    
    #- default is 0 priority, i.e. do not observe
    priority = np.zeros(len(targets), dtype='i8')

    #- Determine which targets have been observed
    #- TODO: this doesn't distinguish between really unobserved vs not yet processed
    unobs = (targets['NUMOBS'] == 0)
    if np.all(unobs):
        done  = np.zeros(len(targets), dtype=bool)
        zgood = np.zeros(len(targets), dtype=bool)
        zwarn = np.zeros(len(targets), dtype=bool)
    else:
        nmore = np.maximum(0, calc_numobs(targets) - targets['NUMOBS'])
        assert np.all(nmore >= 0)
        done = ~unobs & (nmore == 0)
        zgood = ~unobs & (nmore > 0) & (targets['ZWARN'] == 0)
        zwarn = ~unobs & (nmore > 0) & (targets['ZWARN'] != 0)

    #- zgood, zwarn, done, and unobs should be mutually exclusive and cover all targets
    assert not np.any(unobs & zgood)
    assert not np.any(unobs & zwarn)
    assert not np.any(unobs & done)
    assert not np.any(zgood & zwarn)
    assert not np.any(zgood & done)
    assert not np.any(zwarn & done)
    assert np.all(unobs | done | zgood | zwarn)

    #- DESI dark time targets
    if 'DESI_TARGET' in targets.colnames:
        for name in ('ELG', 'LRG'):
            ii = (targets['DESI_TARGET'] & desi_mask[name]) != 0
            priority[ii & unobs] = np.maximum(priority[ii & unobs], desi_mask[name].priorities['UNOBS'])
            priority[ii & done] = np.maximum(priority[ii & done], desi_mask[name].priorities['DONE'])
            priority[ii & zgood] = np.maximum(priority[ii & zgood], desi_mask[name].priorities['MORE_ZGOOD'])
            priority[ii & zwarn] = np.maximum(priority[ii & zwarn], desi_mask[name].priorities['MORE_ZWARN'])
    
        #- QSO could be Lyman-alpha or Tracer
        name = 'QSO'
        ii = (targets['DESI_TARGET'] & desi_mask[name]) != 0
        good_hiz = zgood & (targets['Z'] >= 2.15) & (targets['ZWARN'] == 0)    
        priority[ii & unobs] = np.maximum(priority[ii & unobs], desi_mask[name].priorities['UNOBS'])
        priority[ii & done] = np.maximum(priority[ii & done], desi_mask[name].priorities['DONE'])
        priority[ii & good_hiz] = np.maximum(priority[ii & good_hiz], desi_mask[name].priorities['MORE_ZGOOD'])
        priority[ii & ~good_hiz] = np.maximum(priority[ii & ~good_hiz], desi_mask[name].priorities['DONE'])
        priority[ii & zwarn] = np.maximum(priority[ii & zwarn], desi_mask[name].priorities['MORE_ZWARN'])

    #- BGS targets
    if 'BGS_TARGET' in targets.colnames:
        for name in bgs_mask.names():
            ii = (targets['BGS_TARGET'] & bgs_mask[name]) != 0
            priority[ii & unobs] = np.maximum(priority[ii & unobs], bgs_mask[name].priorities['UNOBS'])
            priority[ii & done] = np.maximum(priority[ii & done], bgs_mask[name].priorities['DONE'])
            priority[ii & zgood] = np.maximum(priority[ii & zgood], bgs_mask[name].priorities['MORE_ZGOOD'])
            priority[ii & zwarn] = np.maximum(priority[ii & zwarn], bgs_mask[name].priorities['MORE_ZWARN'])

    #- MWS targets
    if 'MWS_TARGET' in targets.colnames:
        for name in mws_mask.names():
            ii = (targets['MWS_TARGET'] & mws_mask[name]) != 0
            priority[ii & unobs] = np.maximum(priority[ii & unobs], mws_mask[name].priorities['UNOBS'])
            priority[ii & done] = np.maximum(priority[ii & done], mws_mask[name].priorities['DONE'])
            priority[ii & zgood] = np.maximum(priority[ii & zgood], mws_mask[name].priorities['MORE_ZGOOD'])
            priority[ii & zwarn] = np.maximum(priority[ii & zwarn], mws_mask[name].priorities['MORE_ZWARN'])

    return priority

def calc_numobs(targets):
    """
    Calculates the requested number of observations needed for each target
    
    Args:
        targets: numpy structured array or astropy Table of targets, including
            columns DESI_TARGET, DECAM_FLUX, and DECAM_MW_TRANSMISSION
            
    Returns:
        array of integers of requested number of observations
        
    Notes:
        if ZFLUX or (DECAM_FLUX and DECAM_MW_TRANSMISSION) are in targets,
            then LRG numobs depends upon zmag, else defaults to 2
    """
    #- Default is one observation
    nobs = np.ones(len(targets), dtype='i4')
    
    #- If it wasn't selected by any target class, it gets 0 observations
    #- Normally these would have already been removed, but just in case...
    nobs[targets['DESI_TARGET'] == 0] = 0

    #- LRGs get 1, 2, or 3 observations depending upon magnitude
    zflux = None
    if 'ZFLUX' in targets.dtype.names:
        zflux = targets['ZFLUX']
    elif 'DECAM_FLUX' in targets.dtype.names:
        if 'DECAM_MW_TRANSMISSION' in targets.dtype.names:
            zflux = targets['DECAM_FLUX'][:,4] / targets['DECAM_MW_TRANSMISSION'][:,4]
        else:
            zflux = targets['DECAM_FLUX'][:,4]

    islrg = (targets['DESI_TARGET'] & desi_mask.LRG) != 0
    if zflux is not None:
        lrg2 = islrg & (zflux < 10**((22.5-20.36)/2.5))
        lrg3 = islrg & (zflux < 10**((22.5-20.56)/2.5))
        nobs[lrg2] = 2
        nobs[lrg3] = 3
    else:
        nobs[islrg] = 2

    #- TBD: flag QSOs for 4 obs ahead of time, or only after confirming
    #- that they are redshift>2.15 (i.e. good for Lyman-alpha)?
    isqso = (targets['DESI_TARGET'] & desi_mask.QSO) != 0
    nobs[isqso] = 4

    #- TBD: BGS Faint = 2 observations
    if 'BGS_TARGET' in targets.dtype.names:
        ii = (targets['BGS_TARGET'] & bgs_mask.BGS_FAINT) != 0
        nobs[ii] = np.maximum(nobs[ii], 2)

    return nobs

def finalize(targets, desi_target, bgs_target, mws_target):
    """Return new targets array with added/renamed columns
    
    Args:
        targets: numpy structured array of targets
        kwargs: colname=array of columns to add
        desi_target: 1D array of target selection bit flags
        bgs_target: 1D array of target selection bit flags
        mws_target: 1D array of target selection bit flags
        
    Returns new targets structured array with those changes
    
    Finalize target list by:
      * renaming OBJID -> BRICK_OBJID (it is only unique within a brick)
      * Adding new columns:
    
        - TARGETID: unique ID across all bricks
        - DESI_TARGET: target selection flags
        - MWS_TARGET: target selection flags
        - BGS_TARGET: target selection flags        
    """
    ntargets = len(targets)
    assert ntargets == len(desi_target)
    assert ntargets == len(bgs_target)
    assert ntargets == len(mws_target)
    
    #- OBJID in tractor files is only unique within the brick; rename and
    #- create a new unique TARGETID
    targets = rfn.rename_fields(targets, {'OBJID':'BRICK_OBJID'})
    targetid = targets['BRICKID'].astype(np.int64)*1000000 + targets['BRICK_OBJID']

    #- Add new columns: TARGETID, TARGETFLAG, NUMOBS
    targets = rfn.append_fields(targets,
        ['TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET'],
        [targetid, desi_target, bgs_target, mws_target], usemask=False)

    return targets