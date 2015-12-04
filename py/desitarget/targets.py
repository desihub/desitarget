import numpy as np
import numpy.lib.recfunctions as rfn

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