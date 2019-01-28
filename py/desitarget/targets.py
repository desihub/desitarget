"""
desitarget.targets
==================

Presumably this defines targets.
"""
import numpy as np
import numpy.lib.recfunctions as rfn

from astropy.table import Table

from desitarget.targetmask import desi_mask, bgs_mask, mws_mask, targetid_mask

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
USER_END = 52    # Free to use
SOURCE_END = 60  # Source class
SURVEY_END = 64  # Survey

# Bitmasks
ENCODE_MTL_USER_MASK = 2**USER_END - 2**0               # 0x000fffffffffffff
ENCODE_MTL_SOURCE_MASK = 2**SOURCE_END - 2**USER_END    # 0x0ff0000000000000
ENCODE_MTL_SURVEY_MASK = 2**SURVEY_END - 2**SOURCE_END  # 0xf000000000000000

# Maximum number of unique values
USER_MAX = ENCODE_MTL_USER_MASK                    # 4503599627370495
SOURCE_MAX = ENCODE_MTL_SOURCE_MASK >> USER_END    # 255
SURVEY_MAX = ENCODE_MTL_SURVEY_MASK >> SOURCE_END  # 15

TARGETID_SURVEY_INDEX = {'desi': 0, 'bgs': 1, 'mws': 2}


def target_bitmask_to_string(target_class, mask):
    """Converts integer values of target bitmasks to strings.

    Where multiple bits are set, joins the names of each contributing bit with
    '+'.
    """
    # ADM set up the default logger
    from desiutil.log import get_logger
    log = get_logger()

    target_class_names = np.zeros(len(target_class), dtype=np.object)
    unique_target_classes = np.unique(target_class)
    for tc in unique_target_classes:
        # tc is the encoded integer value of the target bitmask
        has_this_target_class = np.where(target_class == tc)[0]

        tc_name = '+'.join(mask.names(tc))
        target_class_names[has_this_target_class] = tc_name
        log.info('Target class %s (%d): %d' % (tc_name, tc, len(has_this_target_class)))

    return target_class_names


def encode_mtl_targetid(targets):
    """
    Sets targetid used in MTL, which encode both the target class and
    arbitrary tracibility data propagated from individual input sources.

    Allows rows in final MTL (and hence fibre map) to be mapped to input
    sources.
    """
    # ADM set up the default logger
    from desiutil.log import get_logger
    log = get_logger()

    encoded_targetid = targets['TARGETID'].copy()

    # Validate incoming target ids
    if not np.all(encoded_targetid <= ENCODE_MTL_USER_MASK):
        log.error('Invalid range of user-specfied targetid: cannot exceed {}'
                  .format(ENCODE_MTL_USER_MASK))

    desi_target = targets['DESI_TARGET'] != 0
    bgs_target = targets['BGS_TARGET'] != 0
    mws_target = targets['MWS_TARGET'] != 0

    # Assumes surveys are mutually exclusive.
    assert(np.max(np.sum([desi_target, bgs_target, mws_target], axis=0)) == 1)

    # Set the survey bits
    # encoded_targetid[desi_target] += TARGETID_SURVEY_INDEX['desi'] << SOURCE_END
    # encoded_targetid[bgs_target ] += TARGETID_SURVEY_INDEX['bgs']  << SOURCE_END
    # encoded_targetid[mws_target]  += TARGETID_SURVEY_INDEX['mws']  << SOURCE_END

    encoded_targetid[desi_target] += encode_survey_source(TARGETID_SURVEY_INDEX['desi'], 0, 0)
    encoded_targetid[bgs_target] += encode_survey_source(TARGETID_SURVEY_INDEX['bgs'], 0, 0)
    encoded_targetid[mws_target] += encode_survey_source(TARGETID_SURVEY_INDEX['mws'], 0, 0)

    # Set the source bits. Will be different for each survey.
    desi_sources = ['ELG', 'LRG', 'QSO']
    bgs_sources = ['BGS_FAINT', 'BGS_BRIGHT', 'BGS_WISE']
    mws_sources = ['MWS_MAIN', 'MWS_WD', 'MWS_NEARBY']

    for name in desi_sources:
        ii = (targets['DESI_TARGET'] & desi_mask[name]) != 0
        assert(desi_mask[name] <= SOURCE_MAX)
        encoded_targetid[ii] += encode_survey_source(0, desi_mask[name], 0)

    for name in bgs_sources:
        ii = (targets['BGS_TARGET'] & bgs_mask[name]) != 0
        assert(bgs_mask[name] <= SOURCE_MAX)
        encoded_targetid[ii] += encode_survey_source(0, bgs_mask[name], 0)

    for name in mws_sources:
        ii = (targets['MWS_TARGET'] & mws_mask[name]) != 0
        assert(mws_mask[name] <= SOURCE_MAX)
        encoded_targetid[ii] += encode_survey_source(0, mws_mask[name], 0)

    # FIXME (APC): expensive...
    assert(len(np.unique(encoded_targetid)) == len(encoded_targetid))
    return encoded_targetid


def encode_targetid(objid=None, brickid=None, release=None, mock=None, sky=None):
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
    # ADM set up the default logger
    from desiutil.log import get_logger
    log = get_logger()

    # ADM a flag that tracks whether the main inputs were integers
    intpassed = True

    # ADM determine the length of whichever value was passed that wasn't None
    # ADM default to an integer (length 1)
    nobjs = 1
    inputs = [objid, brickid, release, sky, mock]
    goodpar = [input is not None for input in inputs]
    firstgoodpar = np.where(goodpar)[0][0]
    if isinstance(inputs[firstgoodpar], np.ndarray):
        nobjs = len(inputs[firstgoodpar])
        intpassed = False

    # ADM set parameters that weren't passed to zerod arrays
    # ADM set integers that were passed to at least 1D arrays
    if objid is None:
        objid = np.zeros(nobjs, dtype='int64')
    else:
        objid = np.atleast_1d(objid)
    if brickid is None:
        brickid = np.zeros(nobjs, dtype='int64')
    else:
        brickid = np.atleast_1d(brickid)
    if release is None:
        release = np.zeros(nobjs, dtype='int64')
    else:
        release = np.atleast_1d(release)
    if mock is None:
        mock = np.zeros(nobjs, dtype='int64')
    else:
        mock = np.atleast_1d(mock)
    if sky is None:
        sky = np.zeros(nobjs, dtype='int64')
    else:
        sky = np.atleast_1d(sky)

    # ADM check none of the passed parameters exceed their bit-allowance
    if not np.all(objid <= 2**targetid_mask.OBJID.nbits):
        log.error('Invalid range when creating targetid: OBJID cannot exceed {}'
                  .format(2**targetid_mask.OBJID.nbits))
    if not np.all(brickid <= 2**targetid_mask.BRICKID.nbits):
        log.error('Invalid range when creating targetid: BRICKID cannot exceed {}'
                  .format(2**targetid_mask.BRICKID.nbits))
    if not np.all(release <= 2**targetid_mask.RELEASE.nbits):
        log.error('Invalid range when creating targetid: RELEASE cannot exceed {}'
                  .format(2**targetid_mask.RELEASE.nbits))
    if not np.all(mock <= 2**targetid_mask.MOCK.nbits):
        log.error('Invalid range when creating targetid: MOCK cannot exceed {}'
                  .format(2**targetid_mask.MOCK.nbits))
    if not np.all(sky <= 2**targetid_mask.SKY.nbits):
        log.error('Invalid range when creating targetid: SKY cannot exceed {}'
                  .format(2**targetid_mask.SKY.nbits))

    # ADM set up targetid as an array of 64-bit integers
    targetid = np.zeros(nobjs, ('int64'))
    # ADM populate TARGETID based on the passed columns and desitarget.targetid_mask
    # ADM remember to shift to type integer 64 to avoid casting
    targetid |= objid.astype('int64') << targetid_mask.OBJID.bitnum
    targetid |= brickid.astype('int64') << targetid_mask.BRICKID.bitnum
    targetid |= release.astype('int64') << targetid_mask.RELEASE.bitnum
    targetid |= mock.astype('int64') << targetid_mask.MOCK.bitnum
    targetid |= sky.astype('int64') << targetid_mask.SKY.bitnum

    # ADM if the main inputs were integers, return an integer
    if intpassed:
        return targetid[0]
    return targetid


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

    # ADM retrieve each constituent value by left-shifting by the number of bits that comprise
    # ADM the value, to the left-end of the value, and then right-shifting to the right-end
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


def initial_priority_numobs(targets, survey='main'):
    """highest initial priority and numobs for an array of target bits

    Parameters
    ----------
    targets : :class:`~numpy.ndarray`
        An array of targets generated by, e.g., :mod:`~desitarget.cuts`
        must include at least (all of) the columns `DESI_TARGET`, `MWS_TARGET` and
        `BGS_TARGET`
    survey : :class:`str`, defaults to `main`
        Specifies which target masks yaml file to use. Options are `main`,
        `cmx` and `svX` (where X = 1, 2, 3 etc.) for the main survey,
        commissioning and an iteration of SV.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of integers corresponding to the highest initial priority for each of the
        passed bit masks
    :class:`~numpy.ndarray`
        The number of observations corresponding to the highest-priority target class

    Notes
    -----
        - the initial priority that should be used for each target bit is in the file
          data/targetmask.yaml and is called `UNOBS`. It can be retrieved from the
          targeting masks using, e.g., `desi_mask["ELG"].priorities["UNOBS"]`
    """
    # ADM set up the default logger.
    from desiutil.log import get_logger
    log = get_logger()

    # ADM if `survey` is 'main', default to the main survey masks. If it's `cmx`
    # ADM import the commissioning mask. If it's `sv` import the SV masks.
    if survey == 'main':
        colnames = ["DESI_TARGET", "BGS_TARGET", "MWS_TARGET"]
        masks = [desi_mask, bgs_mask, mws_mask]
    elif survey == 'cmx':
        from desitarget.cmx.cmx_targetmask import cmx_mask
        colnames = ["CMX_TARGET"]
        masks = [cmx_mask]
    elif survey[0:2] == 'sv':
        if survey == 'sv1':
            import desitarget.sv1.sv1_targetmask as targmask
        if survey == 'sv2':
            import desitarget.sv2.sv2_targetmask as targmask
        colnames = ["{}_{}_TARGET".format(survey.upper(), tc) for tc in ["DESI", "BGS", "MWS"]]
        masks = [targmask.desi_mask, targmask.bgs_mask, targmask.mws_mask]
    else:
        log.critical("survey must be either 'main', 'cmx' or 'sv', not {}!!!"
                     .format(survey))

    # ADM set up the output arrays
    outpriority = np.zeros(len(targets), dtype='int')
    outnumobs = np.zeros(len(targets), dtype='int')

    for colname, mask in zip(colnames, masks):
        # ADM first determine which bits actually have priorities
        bitnames = []
        for name in mask.names():
            try:
                _ = mask[name].priorities["UNOBS"]
                bitnames.append(name)
            except KeyError:
                pass

        # ADM loop through the relevant bits updating with the highest priority
        # ADM and the largest value of NUMOBS.
        for name in bitnames:
            # ADM indexes in the DESI/MWS/BGS_TARGET column that have this bit set
            istarget = (targets[colname] & mask[name]) != 0
            # ADM for each index, determine where this bit is set and the priority
            # ADM for this bit is > than the currently stored priority.
            w = np.where((mask[name].priorities['UNOBS'] >= outpriority) & istarget)[0]
            # ADM where a larger priority trumps the stored priority, update the priority
            if len(w) > 0:
                outpriority[w] = mask[name].priorities['UNOBS']
            # ADM for each index, determine where this bit is set and whether NUMOBS
            # ADM for this bit is > than the currently stored NUMOBS.
            w = np.where((mask[name].numobs >= outnumobs) & istarget)[0]
            # ADM where a larger NUMOBS trumps the stored NUMOBS, update NUMOBS.
            if len(w) > 0:
                outnumobs[w] = mask[name].numobs

    return outpriority, outnumobs


def encode_survey_source(survey, source, original_targetid):
    """
    """
    return (survey << SOURCE_END) + (source << USER_END) + original_targetid


def decode_survey_source(encoded_values):
    """
    Returns
    -------
        survey[:], source[:], original_targetid[:]
    """
    _encoded_values = np.asarray(np.atleast_1d(encoded_values), dtype=np.uint64)
    survey = (_encoded_values & ENCODE_MTL_SURVEY_MASK) >> SOURCE_END
    source = (_encoded_values & ENCODE_MTL_SOURCE_MASK) >> USER_END

    original_targetid = (encoded_values & ENCODE_MTL_USER_MASK)

    return survey, source, original_targetid


def calc_priority(targets, zcat):
    """
    Calculate target priorities given target masks and observation/redshift status.

    Parameters
    ----------
    targets : :class:`~numpy.ndarray`
        numpy structured array or astropy Table of targets. Must include columns
        `DESI_TARGET`, `BGS_TARGET`, `MWS_TARGET`.
    zcat : :class:`~numpy.ndarray`
        numpy structured array or Table of redshift information. Must include 'Z',
        `ZWARN`, `NUMOBS` and be the same length as `targets`. May also contain
        `NUMOBS_MORE` if this isn't the first time through MTL and `NUMOBS > 0`.

    Returns
    -------
    :class:`~numpy.array`
        integer array of priorities.

    Notes
    -----
        - If a target passes more than one selection, the highest priority wins.
    """
    # ADM set up default DESI logger.
    from desiutil.log import get_logger
    log = get_logger()

    # ADM check the input arrays are the same length.
    assert len(targets) == len(zcat)

    # Default is 0 priority, i.e. do not observe.
    priority = np.zeros(len(targets), dtype='i8')

    # Determine which targets have been observed.
    # TODO: this doesn't distinguish between really unobserved vs not yet
    # processed.
    unobs = (zcat["NUMOBS"] == 0)
    log.debug('calc_priority has %d unobserved targets' % (np.sum(unobs)))
    if np.all(unobs):
        done = np.zeros(len(targets), dtype=bool)
        zgood = np.zeros(len(targets), dtype=bool)
        zwarn = np.zeros(len(targets), dtype=bool)
    else:
        nmore = zcat["NUMOBS_MORE"]
        assert np.all(nmore >= 0)
        done = ~unobs & (nmore == 0)
        zgood = ~unobs & (nmore > 0) & (zcat['ZWARN'] == 0)
        zwarn = ~unobs & (nmore > 0) & (zcat['ZWARN'] != 0)

    # zgood, zwarn, done, and unobs should be mutually exclusive and cover all
    # targets.
    assert not np.any(unobs & zgood)
    assert not np.any(unobs & zwarn)
    assert not np.any(unobs & done)
    assert not np.any(zgood & zwarn)
    assert not np.any(zgood & done)
    assert not np.any(zwarn & done)
    assert np.all(unobs | done | zgood | zwarn)

    # DESI dark time targets.
    if 'DESI_TARGET' in targets.dtype.names:
        for name in ('ELG', 'LRG_1PASS', 'LRG_2PASS'):
            ii = (targets['DESI_TARGET'] & desi_mask[name]) != 0
            priority[ii & unobs] = np.maximum(priority[ii & unobs], desi_mask[name].priorities['UNOBS'])
            priority[ii & done] = np.maximum(priority[ii & done],  desi_mask[name].priorities['DONE'])
            priority[ii & zgood] = np.maximum(priority[ii & zgood], desi_mask[name].priorities['MORE_ZGOOD'])
            priority[ii & zwarn] = np.maximum(priority[ii & zwarn], desi_mask[name].priorities['MORE_ZWARN'])

        # QSO could be Lyman-alpha or Tracer.
        name = 'QSO'
        ii = (targets['DESI_TARGET'] & desi_mask[name]) != 0
        good_hiz = zgood & (zcat['Z'] >= 2.15) & (zcat['ZWARN'] == 0)
        priority[ii & unobs] = np.maximum(priority[ii & unobs], desi_mask[name].priorities['UNOBS'])
        priority[ii & done] = np.maximum(priority[ii & done], desi_mask[name].priorities['DONE'])
        priority[ii & good_hiz] = np.maximum(priority[ii & good_hiz], desi_mask[name].priorities['MORE_ZGOOD'])
        priority[ii & ~good_hiz] = np.maximum(priority[ii & ~good_hiz], desi_mask[name].priorities['DONE'])
        priority[ii & zwarn] = np.maximum(priority[ii & zwarn], desi_mask[name].priorities['MORE_ZWARN'])

    # BGS targets.
    if 'BGS_TARGET' in targets.dtype.names:
        for name in bgs_mask.names():
            ii = (targets['BGS_TARGET'] & bgs_mask[name]) != 0
            priority[ii & unobs] = np.maximum(priority[ii & unobs], bgs_mask[name].priorities['UNOBS'])
            priority[ii & done] = np.maximum(priority[ii & done],  bgs_mask[name].priorities['DONE'])
            priority[ii & zgood] = np.maximum(priority[ii & zgood], bgs_mask[name].priorities['MORE_ZGOOD'])
            priority[ii & zwarn] = np.maximum(priority[ii & zwarn], bgs_mask[name].priorities['MORE_ZWARN'])

    # MWS targets.
    if 'MWS_TARGET' in targets.dtype.names:
        for name in mws_mask.names():
            ii = (targets['MWS_TARGET'] & mws_mask[name]) != 0
            priority[ii & unobs] = np.maximum(priority[ii & unobs], mws_mask[name].priorities['UNOBS'])
            priority[ii & done] = np.maximum(priority[ii & done],  mws_mask[name].priorities['DONE'])
            priority[ii & zgood] = np.maximum(priority[ii & zgood], mws_mask[name].priorities['MORE_ZGOOD'])
            priority[ii & zwarn] = np.maximum(priority[ii & zwarn], mws_mask[name].priorities['MORE_ZWARN'])

    # Special case: IN_BRIGHT_OBJECT means priority=-1 no matter what
    ii = (targets['DESI_TARGET'] & desi_mask.IN_BRIGHT_OBJECT) != 0
    priority[ii] = -1

    return priority


def calc_numobs(targets):
    """
    Calculates the requested number of observations needed for each target

    Args:
        targets: numpy structured array or astropy Table of targets, including
            columns `DESI_TARGET`, `BGS_TARGET` or `MWS_TARGET`

    Returns:
        array of integers of requested number of observations

    Notes:
        This is `NUMOBS` desired before any spectroscopic observations; i.e.
            it does *not* take redshift into consideration (which is relevant
            for interpreting low-z vs high-z QSOs)
    """
    # ADM set up the default logger
    from desiutil.log import get_logger
    log = get_logger()

    # Default is one observation
    nobs = np.ones(len(targets), dtype='i4')

    # If it wasn't selected by any target class, it gets 0 observations
    # Normally these would have already been removed, but just in case...
    no_target_class = np.ones(len(targets), dtype=bool)
    if 'DESI_TARGET' in targets.dtype.names:
        no_target_class &= targets['DESI_TARGET'] == 0
    if 'BGS_TARGET' in targets.dtype.names:
        no_target_class &= targets['BGS_TARGET'] == 0
    if 'MWS_TARGET' in targets.dtype.names:
        no_target_class &= targets['MWS_TARGET'] == 0

    n_no_target_class = np.sum(no_target_class)
    if n_no_target_class > 0:
        raise ValueError('WARNING: {:d} rows in targets.calc_numobs have no target class'.format(n_no_target_class))

    # - LRGs get 1, 2, or (perhaps) 3 observations depending upon magnitude
    # ADM set this using the LRG_1PASS/2PASS and maybe even 3PASS bits
    islrg = (targets['DESI_TARGET'] & desi_mask.LRG) != 0
    # ADM default to 2 passes for LRGs
    nobs[islrg] = 2
    # ADM for redundancy in case the defaults change, explicitly set
    # ADM NOBS for 1PASS and 2PASS LRGs
    try:
        lrg1 = (targets['DESI_TARGET'] & desi_mask.LRG_1PASS) != 0
        lrg2 = (targets['DESI_TARGET'] & desi_mask.LRG_2PASS) != 0
        nobs[lrg1] = 1
        nobs[lrg2] = 2
    except AttributeError:
        log.error('per-pass LRG bits not set in {}'.format(desi_mask))
    # ADM also reserve a setting for LRG_3PASS, but fail gracefully for now
    try:
        lrg3 = (targets['DESI_TARGET'] & desi_mask.LRG_3PASS) != 0
        nobs[lrg3] = 3
    except AttributeError:
        pass

    # - TBD: flag QSOs for 4 obs ahead of time, or only after confirming
    # - that they are redshift>2.15 (i.e. good for Lyman-alpha)?
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
        ii = (targets['BGS_TARGET'] & bgs_mask.BGS_FAINT) != 0
        nobs[ii] = targets['NUMOBS'][ii]+1
        ii = (targets['BGS_TARGET'] & bgs_mask.BGS_BRIGHT) != 0
        nobs[ii] = targets['NUMOBS'][ii]+1
        ii = (targets['BGS_TARGET'] & bgs_mask.BGS_WISE) != 0
        nobs[ii] = targets['NUMOBS'][ii]+1

    return nobs


def finalize(targets, desi_target, bgs_target, mws_target,
             sky=0, survey='main'):
    """Return new targets array with added/renamed columns

    Parameters
    ----------
    targets : :class:`~numpy.ndarray`
        numpy structured array of targets.
    desi_target : :class:`~numpy.ndarray`
        1D array of target selection bit flags.
    bgs_target : :class:`~numpy.ndarray`
        1D array of target selection bit flags.
    mws_target : :class:`~numpy.ndarray`
        1D array of target selection bit flags.
    sky : :class:`int`, defaults to 0
        Pass `1` to indicate these are blank sky targets, `0` otherwise.
    survey : :class:`str`, defaults to `main`
        Specifies which target masks yaml file to use. Options are `main`,
        `cmx` and `svX` (where X = 1, 2, 3 etc.) for the main survey,
        commissioning and an iteration of SV.

    Returns
    -------
    :class:`~numpy.ndarray`
       new targets structured array with the following additions:
          * renaming OBJID -> BRICK_OBJID (it is only unique within a brick).
          * renaming TYPE -> MORPHTYPE (used downstream in other contexts).
          * Adding new columns:
              - TARGETID: unique ID across all bricks.
              - DESI_TARGET: dark time survey target selection flags.
              - MWS_TARGET: bright time MWS target selection flags.
              - BGS_TARGET: bright time BGS target selection flags.
              - PRIORITY: initial priority at which to observe target.
              - SUBPRIORITY: a placeholder column that is set to zero.
              - NUMOBS: initial number of observations for target.

    Notes
    -----
        - SUBPRIORITY is the only column that isn't populated. This is
          because it's easier to populate it in a reproducible fashion
          when collecting targets rather than on a per-brick basis
          when this function is called. It's set to all zeros.
    """
    ntargets = len(targets)
    assert ntargets == len(desi_target)
    assert ntargets == len(bgs_target)
    assert ntargets == len(mws_target)

    # - OBJID in tractor files is only unique within the brick; rename and
    # - create a new unique TARGETID
    targets = rfn.rename_fields(targets,
                                {'OBJID': 'BRICK_OBJID', 'TYPE': 'MORPHTYPE'})
    targetid = encode_targetid(objid=targets['BRICK_OBJID'],
                               brickid=targets['BRICKID'],
                               release=targets['RELEASE'],
                               sky=sky)

    nodata = np.zeros(ntargets, dtype='int')-1
    subpriority = np.zeros(ntargets, dtype='float')

    # ADM add new columns, which are different depending on SV/cmx/main survey.
    if survey == 'main':
        targets = rfn.append_fields(
            targets,
            ['TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'PRIORITY_INIT', 'SUBPRIORITY', 'NUMOBS_INIT'],
            [targetid, desi_target, bgs_target, mws_target, nodata, subpriority, nodata], usemask=False
        )
    elif survey == 'cmx':
        targets = rfn.append_fields(
            targets,
            ['TARGETID', 'CMX_TARGET', 'PRIORITY_INIT', 'SUBPRIORITY', 'NUMOBS_INIT'],
            [targetid, desi_target, nodata, subpriority, nodata], usemask=False
        )
    elif survey[0:2] == 'sv':
        dt, bt, mt = ["{}_{}_TARGET".format(survey.upper(), tc) for tc in ["DESI", "BGS", "MWS"]]
        targets = rfn.append_fields(
            targets,
            ['TARGETID', dt, bt, mt, 'PRIORITY_INIT', 'SUBPRIORITY', 'NUMOBS_INIT'],
            [targetid, desi_target, bgs_target, mws_target, nodata, subpriority, nodata], usemask=False
        )
    else:
        msg = "survey must be either 'main', 'cmx' or begin with 'sv', not {}!!!".format(survey)
        log.critical(msg)
        raise ValueError(msg)

    # ADM determine the initial priority and number of observations.
    targets["PRIORITY_INIT"], targets["NUMOBS_INIT"] = initial_priority_numobs(targets, survey=survey)

    # ADM some final checks that the targets conform to expectations...
    # ADM check that each target has a unique ID.
    if len(targets["TARGETID"]) != len(set(targets["TARGETID"])):
        msg = 'TARGETIDs are not unique!'
        log.critical(msg)
        raise AssertionError(msg)

    # ADM check that we have no LRG targets that don't have LRG_1PASS/2PASS set.
    if survey == 'main':
        lrgset = targets["DESI_TARGET"] & desi_mask.LRG != 0
        pass1lrgset = targets["DESI_TARGET"] & desi_mask.LRG_1PASS != 0
        pass2lrgset = targets["DESI_TARGET"] & desi_mask.LRG_2PASS != 0
        if not np.all(lrgset == pass1lrgset | pass2lrgset):
            msg = 'Some LRG targets do not have 1PASS/2PASS set!'
            log.critical(msg)
            raise AssertionError(msg)

    return targets
