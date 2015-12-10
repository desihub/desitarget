import numpy as np
import numpy.lib.recfunctions as rfn

from desitarget import desi_mask, bgs_mask, mws_mask
from desitarget import obsstate

def calc_priority(targets, targetstate=None):
    '''
    Calculate target priorities given observation state and target masks

    Args:
        targets: numpy structured array or astropy Table of targets, including
            columns DESI_TARGET, BGS_TARGET, and MWS_TARGET
            
    Optional:
        targetstate: array of integers with the obstate mask for each target.
            If None, treat as desitarget.obsstate.UNOBS
                        
    Returns:
        integer array of priorities
        
    Notes:
        If a target passes more than one selection, the highest priority wins
    '''
    if targetstate is None:
        targetstate = obsstate.UNOBS
    
    #- default is 0 priority, i.e. do not observe
    priority = np.zeros(len(targets), dtype='i8')
    
    #- Cache what targets are in what states
    targetstate = np.asarray(targetstate)
    isstate = dict()
    for x in obsstate.names():
        isstate[x] = (targetstate & obsstate[x]) != 0

    for xxx_target, xxx_mask in [
            (targets['DESI_TARGET'], desi_mask),
            (targets['BGS_TARGET'], bgs_mask),
            (targets['MWS_TARGET'], mws_mask),
        ]:
        for objtype in xxx_mask.names():
            #- targets of this objtype
            thistype = (xxx_target & xxx_mask[objtype]) != 0
            for state, p in xxx_mask[objtype].priorities.items():
                #- targets of this type and in this obsstate
                ii = isstate[state] & thistype
                priority[ii] = np.maximum(priority[ii], p)
                ### print objtype, state, ii, priority

    return priority

def calc_numobs(targets):
    """
    Calculates the requested number of observations needed for each target
    
    Args:
        targets: numpy structured array or astropy Table of targets, including
            columns DESI_TARGET, DECAM_FLUX, and DECAM_MW_TRANSMISSION
            
    Returns:
        array of integers of requested number of observations
    """
    #- Default is one observation
    nobs = np.ones(len(targets), dtype='i4')
    
    #- If it wasn't selected by any target class, it gets 0 observations
    #- Normally these would have already been removed, but just in case...
    nobs[targets['DESI_TARGET'] == 0] = 0
    
    #- LRGs get 1, 2, or 3 observations depending upon magnitude
    zflux = targets['DECAM_FLUX'][:,4] / targets['DECAM_MW_TRANSMISSION'][:,4]    
    islrg = (targets['DESI_TARGET'] & desi_mask.LRG) != 0
    lrg2 = islrg & (zflux < 10**((22.5-20.36)/2.5))
    lrg3 = islrg & (zflux < 10**((22.5-20.56)/2.5))
    nobs[lrg2] = 2
    nobs[lrg3] = 3
    
    #- TBD: flag QSOs for 4-5 obs ahead of time, or only after confirming
    #- that they are redshift>2.15 (i.e. good for Lyman-alpha)?
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