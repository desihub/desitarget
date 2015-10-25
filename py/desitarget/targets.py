import numpy as np
import numpy.lib.recfunctions as rfn

def finalize(targets, targetflags, numobs):
    """Return new targets array with added/renamed columns
    
    Args:
        targets: numpy structured array of targets
        targetflags: 1D array of target selection bit flags
        numobs: 1D integer array of number of observations needed
        
    Returns new targets structured array with those changes
    
    Finalize target list by:
      * renaming OBJID -> BRICK_OBJID (it is only unique within a brick)
      * Adding new columns:
    
        - TARGETID: unique ID across all bricks
        - TARGETFLAG: target selection flags from targetflags input
        - NUMOBS: number of observations needed, from numobs input
        
    """    
    #- OBJID in tractor files is only unique within the brick; rename and
    #- create a new unique TARGETID
    targets = rfn.rename_fields(targets, {'OBJID':'BRICK_OBJID'})
    targetid = targets['BRICKID'].astype(np.int64)*1000000 + targets['BRICK_OBJID']

    #- Add new columns: TARGETID, TARGETFLAG, NUMOBS
    targets = rfn.append_fields(targets,
        ['TARGETID', 'TARGETFLAG', 'NUMOBS'],
        [targetid, targetflags, numobs], usemask=False)

    return targets