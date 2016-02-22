import numpy as np
from astropy.table import Table, join

from desitarget import desi_mask, obsmask
from desitarget.targets import calc_numobs, calc_priority

def make_mtl(targets, zcat=None):
    '''
    Adds NUMOBS, PRIORITY, and LASTPASS columns to a targets table
    
    Args:
        targets : Table with columns DESI_TARGET
    
    Returns:
        MTL Table with targets columns plus NUMOBS, PRIORITY, LASTPASS

    TODO:
        Check if targets is ever altered (it shouldn't...)
    '''
    n = len(targets)
    targets = Table(targets)
    if zcat is not None:
        ztargets = join(targets, zcat, keys='TARGETID', join_type='outer')
        if ztargets.masked:
            unobs = ztargets['NUMOBS'].mask
            ztargets['NUMOBS'][unobs] = 0
    else:
        ztargets = targets.copy()
        ztargets['NUMOBS'] = np.zeros(n, dtype=np.int32)
        ztargets['Z'] = -1 * np.ones(n, dtype=np.float32)
        ztargets['ZWARN'] = -1  * np.ones(n, dtype=np.int32)
    
    ztargets['NUMOBS_MORE'] = np.maximum(0, calc_numobs(ztargets) - ztargets['NUMOBS'])

    mtl = targets.copy()
    mtl['NUMOBS_MORE'] = ztargets['NUMOBS_MORE']
    mtl['PRIORITY'] = calc_priority(ztargets)

    #- ELGs can be observed during gray time (the "last pass")
    lastpass = np.zeros(n, dtype='i4')
    iselg = (mtl['DESI_TARGET'] & desi_mask.ELG) != 0
    lastpass[iselg] = 1    
    mtl['LASTPASS'] = lastpass
    
    return mtl
