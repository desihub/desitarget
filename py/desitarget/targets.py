import numpy as np
import numpy.lib.recfunctions as rfn

from astropy.table import Table

from desitarget import desi_mask, bgs_mask, mws_mask
from desitarget import obsmask

############################################################
# TARGETID bit packing

# Of 64 bits total:

# First 52 bits available for propagated targetid, giving a max value of
# 2**52 - 1 = 4503599627370495. The invididual sources are free to
# distribute these bits however they like

# For the case where the propagated ID encodes a filenumber and rownumber
# this would allow, for example, 1048575 files with 4294967295 rows per
# file.

# Higher order bits encode source file and survey. Seems pointless since source
# and survey are implcitiy in the *_TARGET columns which are propagated through
# fibreassign anyway, so this is just a toy example.

# Number of bits allocated to each section
USER_END   = 52 # Free to use
SOURCE_END = 60 # Source class
SURVEY_END = 64 # Survey

# Bitmasks
ENCODE_MTL_USER_MASK   = 2**USER_END   - 2**0           # 0x000fffffffffffff
ENCODE_MTL_SOURCE_MASK = 2**SOURCE_END - 2**USER_END    # 0x0ff0000000000000
ENCODE_MTL_SURVEY_MASK = 2**SURVEY_END - 2**SOURCE_END  # 0xf000000000000000

# Maximum number of unique values
USER_MAX   = ENCODE_MTL_USER_MASK                  # 4503599627370495
SOURCE_MAX = ENCODE_MTL_SOURCE_MASK >> USER_END    # 255
SURVEY_MAX = ENCODE_MTL_SURVEY_MASK >> SOURCE_END  # 15

TARGETID_SURVEY_INDEX = {'desi': 0, 'bgs':  1, 'mws':  2}

############################################################
def target_bitmask_to_string(target_class,mask):
    """Converts integer values of target bitmasks to strings.

    Where multiple bits are set, joins the names of each contributing bit with
    '+'.
    """
    target_class_names = np.zeros(len(target_class),dtype=np.object)
    unique_target_classes = np.unique(target_class)
    for tc in unique_target_classes:
        # tc is the encoded integer value of the target bitmask
        has_this_target_class = np.where(target_class == tc)[0]

        tc_name = '+'.join(mask.names(tc))
        target_class_names[has_this_target_class] = tc_name 
        print('Target class %s (%d): %d'%(tc_name,tc,len(has_this_target_class)))

    return target_class_names

############################################################
def encode_mtl_targetid(targets):
    """
    Sets targetid used in MTL, which encode both the target class and
    arbitrary tracibility data propagated from individual input sources.

    Allows rows in final MTL (and hence fibre map) to be mapped to input
    sources.
    """
    encoded_targetid = targets['TARGETID'].copy()
    
    # Validate incoming target ids
    if not np.all(encoded_targetid <= ENCODE_MTL_USER_MASK):
        print('Invalid range of user-specfied targetid: cannot exceed {}'.format(ENCODE_MTL_USER_MASK))
        raise Exception
    
    desi_target = targets['DESI_TARGET'] != 0
    bgs_target  = targets['BGS_TARGET']  != 0
    mws_target  = targets['MWS_TARGET']  != 0

    # Assumes surveys are mutually exclusive.
    assert(np.max(np.sum([desi_target,bgs_target,mws_target],axis=0)) == 1)
    
    # Set the survey bits
    #encoded_targetid[desi_target] += TARGETID_SURVEY_INDEX['desi'] << SOURCE_END
    #encoded_targetid[bgs_target ] += TARGETID_SURVEY_INDEX['bgs']  << SOURCE_END
    #encoded_targetid[mws_target]  += TARGETID_SURVEY_INDEX['mws']  << SOURCE_END
 
    encoded_targetid[desi_target] += encode_survey_source(TARGETID_SURVEY_INDEX['desi'],0,0)
    encoded_targetid[bgs_target ] += encode_survey_source(TARGETID_SURVEY_INDEX['bgs'],0,0)
    encoded_targetid[mws_target]  += encode_survey_source(TARGETID_SURVEY_INDEX['mws'],0,0) 
     
    # Set the source bits. Will be different for each survey.
    desi_sources = ['ELG','LRG','QSO']
    bgs_sources  = ['BGS_FAINT','BGS_BRIGHT']
    mws_sources  = ['MWS_MAIN','MWS_WD','MWS_NEARBY']

    for name in desi_sources:
        ii  = (targets['DESI_TARGET'] & desi_mask[name]) != 0
        assert(desi_mask[name] <= SOURCE_MAX)
        encoded_targetid[ii] += encode_survey_source(0,desi_mask[name],0)
 
    for name in bgs_sources:
        ii  = (targets['BGS_TARGET'] & bgs_mask[name]) != 0
        assert(bgs_mask[name] <= SOURCE_MAX)
        encoded_targetid[ii] += encode_survey_source(0,bgs_mask[name],0)
    
    for name in mws_sources:
        ii  = (targets['MWS_TARGET'] & mws_mask[name]) != 0
        assert(mws_mask[name] <= SOURCE_MAX)
        encoded_targetid[ii] += encode_survey_source(0,mws_mask[name],0)
   
    # FIXME (APC): expensive...
    assert(len(np.unique(encoded_targetid)) == len(encoded_targetid))
    return encoded_targetid

############################################################
def encode_survey_source(survey,source,original_targetid):
    """
    """
    return (survey << SOURCE_END) + (source << USER_END) + original_targetid

############################################################
def decode_survey_source(encoded_values):
    """
    Returns
    -------
        survey[:], source[:], original_targetid[:]
    """
    _encoded_values = np.asarray(np.atleast_1d(encoded_values),dtype=np.uint64)
    survey = (_encoded_values & ENCODE_MTL_SURVEY_MASK) >> SOURCE_END
    source = (_encoded_values & ENCODE_MTL_SOURCE_MASK) >> USER_END

    original_targetid = (encoded_values & ENCODE_MTL_USER_MASK)

    return survey, source, original_targetid

############################################################
def calc_priority(targets):
    """
    Calculate target priorities given observation state and target masks

    Args:
        targets: numpy structured array or astropy Table of targets, including
            columns DESI_TARGET, BGS_TARGET, and MWS_TARGET

    Returns:
        integer array of priorities

    Notes:
        If a target passes more than one selection, the highest priority wins.
    """
    # FIXME (APC): Blimey, full copy and conversion to table!
    import time
    t0 = time.time()
    print('DEBUG: before targets.calc_priority slow copy')
    targets = Table(targets).copy()
    t1 = time.time()
    print('DEBUG: seconds for targets.calc_priority slow copy: {}'.format(t1-t0))

    # If no NUMOBS, assume no targets have been observed. Requires copy above.
    if 'NUMOBS' not in targets.colnames:
        targets['NUMOBS'] = np.zeros(len(targets), dtype=np.int32)

    # Default is 0 priority, i.e. do not observe
    priority = np.zeros(len(targets), dtype='i8')

    # Determine which targets have been observed
    # TODO: this doesn't distinguish between really unobserved vs not yet
    # processed.
    unobs = (targets['NUMOBS'] == 0)
    print('DEBUG: calc_priority has %d unobserved targets'%(np.sum(unobs)))
    if np.all(unobs):
        done  = np.zeros(len(targets), dtype=bool)
        zgood = np.zeros(len(targets), dtype=bool)
        zwarn = np.zeros(len(targets), dtype=bool)
    else:
        nmore = np.maximum(0, calc_numobs(targets) - targets['NUMOBS'])
        assert np.all(nmore >= 0)
        done  = ~unobs & (nmore == 0)
        zgood = ~unobs & (nmore > 0) & (targets['ZWARN'] == 0)
        zwarn = ~unobs & (nmore > 0) & (targets['ZWARN'] != 0)

    # zgood, zwarn, done, and unobs should be mutually exclusive and cover all
    # targets.
    assert not np.any(unobs & zgood)
    assert not np.any(unobs & zwarn)
    assert not np.any(unobs & done)
    assert not np.any(zgood & zwarn)
    assert not np.any(zgood & done)
    assert not np.any(zwarn & done)
    assert np.all(unobs | done | zgood | zwarn)

    # DESI dark time targets
    if 'DESI_TARGET' in targets.colnames:
        for name in ('ELG', 'LRG'):
            ii                   = (targets['DESI_TARGET'] & desi_mask[name]) != 0
            priority[ii & unobs] = np.maximum(priority[ii & unobs], desi_mask[name].priorities['UNOBS'])
            priority[ii & done]  = np.maximum(priority[ii & done],  desi_mask[name].priorities['DONE'])
            priority[ii & zgood] = np.maximum(priority[ii & zgood], desi_mask[name].priorities['MORE_ZGOOD'])
            priority[ii & zwarn] = np.maximum(priority[ii & zwarn], desi_mask[name].priorities['MORE_ZWARN'])

        # QSO could be Lyman-alpha or Tracer
        name = 'QSO'
        ii                       = (targets['DESI_TARGET'] & desi_mask[name]) != 0
        good_hiz                 = zgood & (targets['Z'] >= 2.15) & (targets['ZWARN'] == 0)    
        priority[ii & unobs]     = np.maximum(priority[ii & unobs], desi_mask[name].priorities['UNOBS'])
        priority[ii & done]      = np.maximum(priority[ii & done], desi_mask[name].priorities['DONE'])
        priority[ii & good_hiz]  = np.maximum(priority[ii & good_hiz], desi_mask[name].priorities['MORE_ZGOOD'])
        priority[ii & ~good_hiz] = np.maximum(priority[ii & ~good_hiz], desi_mask[name].priorities['DONE'])
        priority[ii & zwarn]     = np.maximum(priority[ii & zwarn], desi_mask[name].priorities['MORE_ZWARN'])

    # BGS targets
    if 'BGS_TARGET' in targets.colnames:
        for name in bgs_mask.names():
            ii                   = (targets['BGS_TARGET'] & bgs_mask[name]) != 0
            priority[ii & unobs] = np.maximum(priority[ii & unobs], bgs_mask[name].priorities['UNOBS'])
            priority[ii & done]  = np.maximum(priority[ii & done],  bgs_mask[name].priorities['DONE'])
            priority[ii & zgood] = np.maximum(priority[ii & zgood], bgs_mask[name].priorities['MORE_ZGOOD'])
            priority[ii & zwarn] = np.maximum(priority[ii & zwarn], bgs_mask[name].priorities['MORE_ZWARN'])

    # MWS targets
    if 'MWS_TARGET' in targets.colnames:
        for name in mws_mask.names():
            ii                   = (targets['MWS_TARGET'] & mws_mask[name]) != 0
            priority[ii & unobs] = np.maximum(priority[ii & unobs], mws_mask[name].priorities['UNOBS'])
            priority[ii & done]  = np.maximum(priority[ii & done],  mws_mask[name].priorities['DONE'])
            priority[ii & zgood] = np.maximum(priority[ii & zgood], mws_mask[name].priorities['MORE_ZGOOD'])
            priority[ii & zwarn] = np.maximum(priority[ii & zwarn], mws_mask[name].priorities['MORE_ZWARN'])

    return priority

############################################################
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
        This is NUMOBS desired before any spectroscopic observations; i.e.
            it does *not* take redshift into consideration (which is relevant
            for interpreting low-z vs high-z QSOs)
    """
    # Default is one observation
    nobs = np.ones(len(targets), dtype='i4')

    # If it wasn't selected by any target class, it gets 0 observations
    # Normally these would have already been removed, but just in case...

    no_target_class = np.ones(len(targets), dtype=bool)
    if 'DESI_TARGET' in targets.dtype.names:
        no_target_class &=  targets['DESI_TARGET'] == 0
    if 'BGS_TARGET' in targets.dtype.names:
        no_target_class &= targets['BGS_TARGET']  == 0
    if 'MWS_TARGET' in targets.dtype.names:
        no_target_class &= targets['MWS_TARGET']  == 0

    n_no_target_class = np.sum(no_target_class)
    if n_no_target_class > 0:
        raise ValueError('WARNING: {:d} rows in targets.calc_numobs have no target class'.format(n_no_target_class))

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

    # FIXME (APC): Better not to hardcode all this here? Took out the following
    # for compatibility with earlier MWS tests
    # SJB: better to not hardcode (for BGS and LRGs too), but until that is
    # refactored we still need to request 2 observations for BGS_FAINT
    #- TBD: BGS Faint = 2 observations
    if 'BGS_TARGET' in targets.dtype.names:
       ii       = (targets['BGS_TARGET'] & bgs_mask.BGS_FAINT) != 0
       nobs[ii] = np.maximum(nobs[ii], 2)

    return nobs

############################################################
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
