"""
desitarget.targets
==================

Presumably this defines targets.
"""
import numpy as np
import numpy.lib.recfunctions as rfn

from astropy.table import Table

from desitarget import desi_mask, bgs_mask, mws_mask, targetid_mask

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
def encode_targetid(objid=None,brickid=None,release=None,mock=None,sky=None):
    """Create the DESI TARGETID from input source and imaging information

    Parameters
    ----------
    objid : :class:`int` or :class:`~numpy.ndarray`, optional
        The OBJID from Legacy Survey imaging (e.g. http://legacysurvey.org/dr4/catalogs/)
    brickid : :class:`int` or :class:`~numpy.ndarray`, optional
        The BRICKID from Legacy Survey imaging (e.g. http://legacysurvey.org/dr4/catalogs/)
    release : :class:`int` or :class:`~numpy.ndarray`, optional
        The RELEASE from Legacy Survey imaging (e.g. http://legacysurvey.org/dr4/catalogs/)
    mock : :class:`int` or :class:`~numpy.ndarray`, optional
        1 if this object is a mock object (generated from 
        mocks, not from real survey data), 0 otherwise
    sky : :class:`int` or :class:`~numpy.ndarray`, optional
        1 if this object is a blank sky object, 0 otherwise

    Returns
    -------
    :class:`int` or `~numpy.ndarray` 
        The TARGETID for DESI, encoded according to the bits listed in
        :meth:`desitarget.targetid_mask`. If an integer is passed, then an
        integer is returned, otherwise an array is returned

    Notes
    -----
        - This is set up with maximum flexibility so that mixes of integers 
          and arrays can be passed, in case some value like BRICKID or SKY 
          is the same for a set of objects. Consider, e.g.:

              print(
                  targets.decode_targetid(
                      targets.encode_targetid(objid=np.array([234,12]),
                                              brickid=np.array([234,12]),
                                              release=4,
                                              sky=[1,0]))
                                              )

        (array([234,12]), array([234,12]), array([4,4]), array([0,0]), array([1,0]))

        - See also https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=2348
    """

    #ADM a flag that tracks whether the main inputs were integers
    intpassed = True

    #ADM determine the length of whichever value was passed that wasn't None
    #ADM default to an integer (length 1)
    nobjs = 1
    inputs = [objid, brickid, release, sky, mock]
    goodpar = [ input is not None for input in inputs ]
    firstgoodpar = np.where(goodpar)[0][0]
    if isinstance(inputs[firstgoodpar],np.ndarray):
        nobjs = len(inputs[firstgoodpar])
        intpassed = False

    #ADM set parameters that weren't passed to zerod arrays
    #ADM set integers that were passed to at least 1D arrays
    if objid is None:
        objid = np.zeros(nobjs,dtype='int64')
    else:
        objid = np.atleast_1d(objid)
    if brickid is None:
        brickid = np.zeros(nobjs,dtype='int64')
    else:
        brickid = np.atleast_1d(brickid)
    if release is None:
        release = np.zeros(nobjs,dtype='int64')
    else:
        release = np.atleast_1d(release)
    if mock is None:
        mock = np.zeros(nobjs,dtype='int64')
    else:
        mock = np.atleast_1d(mock)
    if sky is None:
        sky = np.zeros(nobjs,dtype='int64')
    else:
        sky = np.atleast_1d(sky)

    #ADM check none of the passed parameters exceed their bit-allowance
    if not np.all(objid <= 2**targetid_mask.OBJID.nbits):
        print('Invalid range when creating targetid: OBJID cannot exceed {}'.format(2**targetid_mask.OBJID.nbits))
        raise Exception
    if not np.all(brickid <= 2**targetid_mask.BRICKID.nbits):
        print('Invalid range when creating targetid: BRICKID cannot exceed {}'.format(2**targetid_mask.BRICKID.nbits))
        raise Exception
    if not np.all(release <= 2**targetid_mask.RELEASE.nbits):
        print('Invalid range when creating targetid: RELEASE cannot exceed {}'.format(2**targetid_mask.RELEASE.nbits))
        raise Exception
    if not np.all(mock <= 2**targetid_mask.MOCK.nbits):
        print('Invalid range when creating targetid: MOCK cannot exceed {}'.format(2**targetid_mask.MOCK.nbits))
        raise Exception
    if not np.all(sky <= 2**targetid_mask.SKY.nbits):
        print('Invalid range when creating targetid: SKY cannot exceed {}'.format(2**targetid_mask.SKY.nbits))
        raise Exception

    #ADM set up targetid as an array of 64-bit integers
    targetid = np.zeros(nobjs,('int64'))
    #ADM populate TARGETID based on the passed columns and desitarget.targetid_mask
    #ADM remember to shift to type integer 64 to avoid casting
    targetid |= objid.astype('int64') << targetid_mask.OBJID.bitnum
    targetid |= brickid.astype('int64') << targetid_mask.BRICKID.bitnum
    targetid |= release.astype('int64') << targetid_mask.RELEASE.bitnum
    targetid |= mock.astype('int64') << targetid_mask.MOCK.bitnum
    targetid |= sky.astype('int64') << targetid_mask.SKY.bitnum

    #ADM if the main inputs were integers, return an integer
    if intpassed:
        return targetid[0]
    return targetid

############################################################
def decode_targetid(targetid):
    """break a DESI TARGETID into its constituent parts

    Parameters
    ----------
    :class:`int` or :class:`~numpy.ndarray` 
        The TARGETID for DESI, encoded according to the bits listed in
        :meth:`desitarget.targetid_mask`        

    Returns
    -------
    objid : :class:`int` or `~numpy.ndarray`
        The OBJID from Legacy Survey imaging (e.g. http://legacysurvey.org/dr4/catalogs/)
    brickid : :class:`int` or `~numpy.ndarray`
        The BRICKID from Legacy Survey imaging (e.g. http://legacysurvey.org/dr4/catalogs/)
    release : :class:`int` or `~numpy.ndarray`
        The RELEASE from Legacy Survey imaging (e.g. http://legacysurvey.org/dr4/catalogs/)
    mock : :class:`int` or `~numpy.ndarray`
        1 if this object is a mock object (generated from 
        mocks, not from real survey data), 0 otherwise
    sky : :class:`int` or `~numpy.ndarray`
        1 if this object is a blank sky object, 0 otherwise

    Notes
    -----
        - if a 1-D array is passed, then an integer is returned. Otherwise an array
          is returned
        - see also https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=2348
    """

    #ADM retrieve each constituent value by left-shifting by the number of bits that comprise
    #ADM the value, to the left-end of the value, and then right-shifting to the right-end
    objid = (targetid & (2**targetid_mask.OBJID.nbits - 1 
                         << targetid_mask.OBJID.bitnum)) >> targetid_mask.OBJID.bitnum
    brickid = (targetid & (2**targetid_mask.BRICKID.nbits - 1 
                           << targetid_mask.BRICKID.bitnum)) >> targetid_mask.BRICKID.bitnum
    release = (targetid & (2**targetid_mask.RELEASE.nbits - 1 
                           << targetid_mask.RELEASE.bitnum)) >> targetid_mask.RELEASE.bitnum
    mock = (targetid & (2**targetid_mask.MOCK.nbits - 1 
                        << targetid_mask.MOCK.bitnum)) >> targetid_mask.MOCK.bitnum
    sky = (targetid & (2**targetid_mask.SKY.nbits - 1 
                       << targetid_mask.SKY.bitnum)) >> targetid_mask.SKY.bitnum

    return objid, brickid, release, mock, sky

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

    # Special case: IN_BRIGHT_OBJECT means priority=-1 no matter what
    ii = (targets['DESI_TARGET'] & desi_mask.IN_BRIGHT_OBJECT) != 0
    priority[ii] = -1

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

    # BGS: observe both BGS target classes once (and once only) on every epoch,
    # regardless of how many times it has been observed on previous epochs.

    # Priorities for MORE_ZWARN and MORE_ZGOOD are set in targetmask.yaml such
    # that targets are reobserved at the same priority until they have a good
    # redshift. Targets with good redshifts are still observed on subsequent
    # epochs but with a priority below all other BGS and MWS targets. 

    if 'BGS_TARGET' in targets.dtype.names:
        # This forces the calculation of nmore in targets.calc_priority (and
        # ztargets['NOBS_MORE'] in mtl.make_mtl) to give nmore = 1 regardless
        # of targets['NUMOBS']
        ii       = (targets['BGS_TARGET'] & bgs_mask.BGS_FAINT) != 0
        nobs[ii] = targets['NUMOBS'][ii]+1
        ii       = (targets['BGS_TARGET'] & bgs_mask.BGS_BRIGHT) != 0
        nobs[ii] = targets['NUMOBS'][ii]+1

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
    targetid = encode_targetid(objid=targets['BRICK_OBJID'],
                               brickid=targets['BRICKID'],
                               release=targets['RELEASE'])

    #- Add new columns: TARGETID, TARGETFLAG, NUMOBS
    targets = rfn.append_fields(targets,
        ['TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET'],
        [targetid, desi_target, bgs_target, mws_target], usemask=False)

    return targets
