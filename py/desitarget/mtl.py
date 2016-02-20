import numpy as np

from desitarget import desi_mask
from desitarget.targets import calc_numobs

def make_mtl(targets):
    '''
    Adds NUMOBS, PRIORITY, and LASTPASS columns to a targets table
    
    Args:
        targets : Table with columns DESI_TARGET
    
    Returns:
        MTL Table with targets columns plus NUMOBS, PRIORITY, LASTPASS

    Note:
        v0 - no zcatalog information used yet
    '''
    n = len(targets)
    mtl = targets.copy(copy_data=False)
    
    priority = np.zeros(n, dtype='i4')
    for name in desi_mask.names():
        if name.startswith('STD_') or \
           name in ('SKY', 'BGS_ANY', 'MWS_ANY', 'ANCILLARY_ANY'):
            continue
        ii = (mtl['DESI_TARGET'] & desi_mask[name]) != 0
        if np.any(ii):
            priority[ii] = np.maximum(priority[ii], desi_mask[name].priorities['UNOBS'])
                
    mtl['PRIORITY'] = priority

    mtl['NUMOBS'] = calc_numobs(targets)

    #- ELGs can be observed during gray time (the "last pass")
    lastpass = np.zeros(n, dtype='i4')
    iselg = (mtl['DESI_TARGET'] & desi_mask.ELG) != 0
    lastpass[iselg] = 1    
    mtl['LASTPASS'] = lastpass
    
    return mtl
