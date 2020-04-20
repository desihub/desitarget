"""
desitarget.targets
==================

Presumably this defines targets.

.. _`DocDB 2348`: https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=2348
"""
import numpy as np
import healpy as hp
import numpy.lib.recfunctions as rfn

from astropy.table import Table

from desitarget.targetmask import desi_mask, bgs_mask, mws_mask
from desitarget.targetmask import scnd_mask, targetid_mask
from desitarget.targetmask import obsconditions

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()


def encode_targetid(objid=None, brickid=None, release=None,
                    mock=None, sky=None, gaiadr=None):
    """Create the DESI TARGETID from input source and imaging info.

    Parameters
    ----------
    objid : :class:`int` or :class:`~numpy.ndarray`, optional
        The OBJID from Legacy Surveys imaging or the row within
        a Gaia HEALPixel file in $GAIA_DIR/healpix if
        `gaia` is not ``None``.
    brickid : :class:`int` or :class:`~numpy.ndarray`, optional
        The BRICKID from Legacy Surveys imaging.
        or the Gaia HEALPixel chunk number for files in
        $GAIA_DIR/healpix if `gaia` is not ``None``.
    release : :class:`int` or :class:`~numpy.ndarray`, optional
        The RELEASE from Legacy Surveys imaging. Or, if < 1000,
        the secondary target class bit flag number from
        'data/targetmask.yaml'. Or, if < 1000 and `sky` is not
        ``None``, the HEALPixel processing number for SUPP_SKIES.
    mock : :class:`int` or :class:`~numpy.ndarray`, optional
        1 if this object is a mock object (generated from
        mocks, not from real survey data), 0 otherwise
    sky : :class:`int` or :class:`~numpy.ndarray`, optional
        1 if this object is a blank sky object, 0 otherwise
    gaiadr : :class:`int` or :class:`~numpy.ndarray`, optional
        The Gaia Data Release number (e.g. send 2 for Gaia DR2).
        A value of 1 does NOT mean DR1. Rather it has the specific
        meaning of a DESI first-light commissioning target.

    Returns
    -------
    :class:`int` or `~numpy.ndarray`
        The TARGETID for DESI, encoded according to the bits listed in
        :meth:`desitarget.targetid_mask`. If an integer is passed, then
        an integer is returned, otherwise an array is returned.

    Notes
    -----
        - Has maximum flexibility so that mixes of integers and arrays
          can be passed, in case some value like BRICKID or SKY
          is the same for a set of objects. Consider, e.g.:

          print(
              targets.decode_targetid(
                  targets.encode_targetid(objid=np.array([234,12]),
                                          brickid=np.array([234,12]),
                                          release=4000,
                                          sky=[1,0]))
                                          )

        (array([234,12]), array([234,12]), array([4000,4000]),
         array([0,0]), array([1,0]))

        - See also `DocDB 2348`_.
    """
    # ADM a flag that tracks whether the main inputs were integers.
    intpassed = True

    # ADM the names of the bits with RESERVED removed.
    bitnames = targetid_mask.names()
    if "RESERVED" in bitnames:
        bitnames.remove("RESERVED")

    # ADM determine the length of passed values that aren't None.
    # ADM default to an integer (length 1).
    nobjs = 1
    inputs = [objid, brickid, release, mock, sky, gaiadr]
    goodpar = [param is not None for param in inputs]
    firstgoodpar = np.where(goodpar)[0][0]
    if isinstance(inputs[firstgoodpar], np.ndarray):
        nobjs = len(inputs[firstgoodpar])
        intpassed = False

    # ADM set parameters that weren't passed to zerod arrays
    # ADM set integers that were passed to at least 1D arrays
    for i, param in enumerate(inputs):
        if param is None:
            inputs[i] = np.zeros(nobjs, dtype='int64')
        else:
            inputs[i] = np.atleast_1d(param)

    # ADM check passed parameters don't exceed their bit-allowance.
    for param, bitname in zip(inputs, bitnames):
        if not np.all(param < 2**targetid_mask[bitname].nbits):
            msg = 'Invalid range when making targetid: {} '.format(bitname)
            msg += 'cannot exceed {}'.format(2**targetid_mask[bitname].nbits - 1)
            log.critical(msg)
            raise IOError(msg)

    # ADM set up targetid as an array of 64-bit integers.
    targetid = np.zeros(nobjs, ('int64'))
    # ADM populate TARGETID. Shift to type integer 64 to avoid casting.
    for param, bitname in zip(inputs, bitnames):
        targetid |= param.astype('int64') << targetid_mask[bitname].bitnum

    # ADM if the main inputs were integers, return an integer.
    if intpassed:
        return targetid[0]
    return targetid


def decode_targetid(targetid):
    """break a DESI TARGETID into its constituent parts.

    Parameters
    ----------
    :class:`int` or :class:`~numpy.ndarray`
        The TARGETID for DESI, encoded according to the bits listed in
        :meth:`desitarget.targetid_mask`.

    Returns
    -------
    :class:`int` or :class:`~numpy.ndarray`
        The OBJID from Legacy Surveys imaging or the row within
        a Gaia HEALPixel file in $GAIA_DIR/healpix if
        `gaia` is not ``None``.
    :class:`int` or :class:`~numpy.ndarray`
        The BRICKID from Legacy Surveys imaging.
        or the Gaia HEALPixel chunk number for files in
        $GAIA_DIR/healpix if `gaia` is not ``None``.
    :class:`int` or :class:`~numpy.ndarray`
        The RELEASE from Legacy Surveys imaging. Or, if < 1000,
        the secondary target class bit flag number from
        'data/targetmask.yaml'. Or, if < 1000 and `sky` is not
        ``None``, the HEALPixel processing number for SUPP_SKIES.
    :class:`int` or :class:`~numpy.ndarray`
        1 if this object is a mock object (generated from
        mocks, not from real survey data), 0 otherwise
    :class:`int` or :class:`~numpy.ndarray`
        1 if this object is a blank sky object, 0 otherwise
    :class:`int` or :class:`~numpy.ndarray`
        The Gaia Data Release number (e.g. will be 2 for Gaia DR2).
        A value of 1 does NOT mean DR1. Rather it has the specific
        meaning of a DESI first-light commissioning target.

    Notes
    -----
        - if a 1-D array is passed, then an integer is returned.
          Otherwise an array is returned.
        - see also `DocDB 2348`_.
    """
    # ADM the names of the bits with RESERVED removed.
    bitnames = targetid_mask.names()
    if "RESERVED" in bitnames:
        bitnames.remove("RESERVED")

    # ADM retrieve each value by left-shifting by the number of bits
    # ADM that comprise the value, to the left-end of the value, and
    # ADM then right-shifting to the right-end.
    outputs = []
    for bitname in bitnames:
        bitnum = targetid_mask[bitname].bitnum
        val = (targetid & (2**targetid_mask[bitname].nbits - 1
                           << targetid_mask[bitname].bitnum)) >> bitnum
        outputs.append(val)

    return outputs


def main_cmx_or_sv(targets, rename=False, scnd=False):
    """determine whether a target array is main survey, commissioning, or SV

    Parameters
    ----------
    targets : :class:`~numpy.ndarray`
        An array of targets generated by, e.g., :mod:`~desitarget.cuts`
        must include at least (all of) the columns `DESI_TARGET`, `MWS_TARGET` and
        `BGS_TARGET` or the corresponding commissioning or SV columns.
    rename : :class:`bool`, optional, defaults to ``False``
        If ``True`` then also return a copy of `targets` with the input `_TARGET`
        columns renamed to reflect the main survey format.
    scnd : :class:`bool`, optional, defaults to ``False``
        If ``True``, then add the secondary target information to the output.

    Returns
    -------
    :class:`list`
        A list of strings corresponding to the target columns names. For the main survey
        this would be [`DESI_TARGET`, `BGS_TARGET`, `MWS_TARGET`], for commissioning it
        would just be [`CMX_TARGET`], for SV1 it would be [`SV1_DESI_TARGET`,
        `SV1_BGS_TARGET`, `SV1_MWS_TARGET`]. Also includes, e.g. `SCND_TARGET`, if
        `scnd` is passed as ``True``.
    :class:`list`
        A list of the masks that correspond to each column from the relevant main/cmx/sv
        yaml file. Also includes the relevant SCND_MASK, if `scnd` is passed as True.
    :class:`str`
        The string 'main', 'cmx' or 'svX' (where X = 1, 2, 3 etc.) for the main survey,
        commissioning and an iteration of SV. Specifies which type of file was sent.
    :class:`~numpy.ndarray`, optional, if `rename` is ``True``
        A copy of the input targets array with the `_TARGET` columns renamed to
        `DESI_TARGET`, and (if they exist) `BGS_TARGET`, `MWS_TARGET`.
    """
    # ADM default to the main survey.
    maincolnames = ["DESI_TARGET", "BGS_TARGET", "MWS_TARGET", "SCND_TARGET"]
    outcolnames = maincolnames.copy()
    masks = [desi_mask, bgs_mask, mws_mask, scnd_mask]
    survey = 'main'

    # ADM set survey to correspond to commissioning or SV if those columns exist
    # ADM and extract the column names of interest.
    incolnames = np.array(targets.dtype.names)
    notmain = np.array(['SV' in name or 'CMX' in name for name in incolnames])
    if np.any(notmain):
        outcolnames = list(incolnames[notmain])
        survey = outcolnames[0].split('_')[0].lower()
    if survey[0:2] == 'sv':
        outcolnames = ["{}_{}".format(survey.upper(), col) for col in maincolnames]

    # ADM retrieve the correct masks, depending on the survey type.
    if survey == 'cmx':
        from desitarget.cmx.cmx_targetmask import cmx_mask
        masks = [cmx_mask]
    elif survey[0:2] == 'sv':
        if survey == 'sv1':
            import desitarget.sv1.sv1_targetmask as targmask
        if survey == 'sv2':
            import desitarget.sv2.sv2_targetmask as targmask
        masks = [targmask.desi_mask, targmask.bgs_mask,
                 targmask.mws_mask, targmask.scnd_mask]
    elif survey != 'main':
        msg = "input target file must be 'main', 'cmx' or 'sv', not {}!!!".format(survey)
        log.critical(msg)
        raise ValueError(msg)

    if not scnd:
        outcolnames = outcolnames[:3]
        masks = masks[:3]

    # ADM if requested, rename the columns.
    if rename:
        mapper = {}
        for i, col in enumerate(outcolnames):
            mapper[col] = maincolnames[i]
        return outcolnames, masks, survey, rfn.rename_fields(targets, mapper)

    return outcolnames, masks, survey


def set_obsconditions(targets, scnd=False):
    """set the OBSCONDITIONS mask for each target bit.

    Parameters
    ----------
    targets : :class:`~numpy.ndarray`
        An array of targets generated by, e.g., :mod:`~desitarget.cuts`.
        Must include at least (all of) the columns `DESI_TARGET`,
        `BGS_TARGET`, `MWS_TARGET` or corresponding cmx or SV columns.
    scnd : :class:`bool`, optional, defaults to ``False``
        If ``True`` then make all of the comparisons on the `SCND_TARGET`
        column instead of `DESI_TARGET`, `BGS_TARGET` and `MWS_TARGET`.

    Returns
    -------
    :class:`~numpy.ndarray`
        The OBSCONDITIONS bitmask for the passed targets.

    Notes
    -----
        - the OBSCONDITIONS for each target bit is in the file, e.g.
          data/targetmask.yaml. It can be retrieved using, for example,
          `obsconditions.mask(desi_mask["ELG"].obsconditions)`.
    """
    colnames, masks, _ = main_cmx_or_sv(targets, scnd=scnd)
    # ADM if we requested secondary targets, the needed information
    # ADM was returned as the last part of each array.
    if scnd:
        colnames, masks = colnames[-1:], masks[-1:]

    n = len(targets)
    obscon = np.zeros(n, dtype='i4')
    for mask, xxx_target in zip(masks, colnames):
        for name in mask.names():
            # ADM which targets have this bit for this mask set?
            ii = (targets[xxx_target] & mask[name]) != 0
            # ADM under what conditions can that bit be observed?
            if np.any(ii):
                obscon[ii] |= obsconditions.mask(mask[name].obsconditions)

    return obscon


def initial_priority_numobs(targets, scnd=False,
                            obscon="DARK|GRAY|BRIGHT|POOR|TWILIGHT12|TWILIGHT18"):
    """highest initial priority and numobs for an array of target bits.

    Parameters
    ----------
    targets : :class:`~numpy.ndarray`
        An array of targets generated by, e.g., :mod:`~desitarget.cuts`.
        Must include at least (all of) the columns `DESI_TARGET`,
        `BGS_TARGET`, `MWS_TARGET` or corresponding cmx or SV columns.
    scnd : :class:`bool`, optional, defaults to ``False``
        If ``True`` then make all of the comparisons on the `SCND_TARGET`
        column instead of `DESI_TARGET`, `BGS_TARGET` and `MWS_TARGET`.
    obscon : :class:`str`, optional, defaults to almost all OBSCONDITIONS
        A combination of strings that are in the desitarget bitmask yaml
        file (specifically in `desitarget.targetmask.obsconditions`).

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of integers corresponding to the highest initial
        priority for each target consistent with the constraints
        on observational conditions imposed by `obscon`.
    :class:`~numpy.ndarray`
        An array of integers corresponding to the largest number of
        observations for each target consistent with the constraints
        on observational conditions imposed by `obscon`.

    Notes
    -----
        - the initial priority for each target bit is in the file, e.g.,
          data/targetmask.yaml. It can be retrieved using, for example,
          `desi_mask["ELG"].priorities["UNOBS"]`.
        - the input obscon string can be converted to a bitmask using
          `desitarget.targetmask.obsconditions.mask(blat)`.
    """
    colnames, masks, _ = main_cmx_or_sv(targets, scnd=scnd)
    # ADM if we requested secondary targets, the needed information
    # ADM was returned as the last part of each array.
    if scnd:
        colnames, masks = colnames[-1:], masks[-1:]

    # ADM set up the output arrays.
    outpriority = np.zeros(len(targets), dtype='int')
    # ADM remember that calibs have NUMOBS of -1.
    outnumobs = np.zeros(len(targets), dtype='int')-1

    # ADM convert the passed obscon string to bits.
    obsbits = obsconditions.mask(obscon)

    # ADM loop through the masks to establish all bitnames of interest.
    for colname, mask in zip(colnames, masks):
        # ADM first determine which bits actually have priorities.
        bitnames = []
        for name in mask.names():
            try:
                _ = mask[name].priorities["UNOBS"]
                # ADM also only consider bits with correct OBSCONDITIONS.
                obsforname = obsconditions.mask(mask[name].obsconditions)
                if (obsforname & obsbits) != 0:
                    bitnames.append(name)
            except KeyError:
                pass

        # ADM loop through the relevant bits updating with the highest
        # ADM priority and the largest value of NUMOBS.
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


def calc_priority(targets, zcat, obscon):
    """
    Calculate target priorities from masks, observation/redshift status.

    Parameters
    ----------
    targets : :class:`~numpy.ndarray`
        numpy structured array or astropy Table of targets. Must include
        the columns `DESI_TARGET`, `BGS_TARGET`, `MWS_TARGET`
        (or their SV/cmx equivalents).
    zcat : :class:`~numpy.ndarray`
        numpy structured array or Table of redshift information. Must
        include 'Z', `ZWARN`, `NUMOBS` and be the same length as
        `targets`. May also contain `NUMOBS_MORE` if this isn't the
        first time through MTL and `NUMOBS > 0`.
    obscon : :class:`str`
        A combination of strings that are in the desitarget bitmask yaml
        file (specifically in `desitarget.targetmask.obsconditions`), e.g.
        "DARK|GRAY". Governs the behavior of how priorities are set based
        on "obsconditions" in the desitarget bitmask yaml file.

    Returns
    -------
    :class:`~numpy.array`
        integer array of priorities.

    Notes
    -----
        - If a target passes multiple selections, highest priority wins.
        - Will automatically detect if the passed targets are main
          survey, commissioning or SV and behave accordingly.
    """
    # ADM check the input arrays are the same length.
    assert len(targets) == len(zcat)

    # ADM determine whether the input targets are main survey, cmx or SV.
    colnames, masks, survey = main_cmx_or_sv(targets, scnd=True)
    # ADM the target bits/names should be shared between main survey and SV.
    if survey != 'cmx':
        desi_target, bgs_target, mws_target, scnd_target = colnames
        desi_mask, bgs_mask, mws_mask, scnd_mask = masks
    else:
        cmx_mask = masks[0]

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
    if survey != 'cmx':
        if desi_target in targets.dtype.names:
            # ADM 'LRG' is the guiding column in SV and the main survey
            # ADM (once, it was 'LRG_1PASS' and 'LRG_2PASS' in the MS).
            # names = ('ELG', 'LRG_1PASS', 'LRG_2PASS')
            # if survey[0:2] == 'sv':
            names = ('ELG', 'LRG')
            for name in names:
                # ADM only update priorities for passed observing conditions.
                pricon = obsconditions.mask(desi_mask[name].obsconditions)
                if (obsconditions.mask(obscon) & pricon) != 0:
                    ii = (targets[desi_target] & desi_mask[name]) != 0
                    priority[ii & unobs] = np.maximum(priority[ii & unobs], desi_mask[name].priorities['UNOBS'])
                    priority[ii & done] = np.maximum(priority[ii & done],  desi_mask[name].priorities['DONE'])
                    priority[ii & zgood] = np.maximum(priority[ii & zgood], desi_mask[name].priorities['MORE_ZGOOD'])
                    priority[ii & zwarn] = np.maximum(priority[ii & zwarn], desi_mask[name].priorities['MORE_ZWARN'])

            # QSO could be Lyman-alpha or Tracer.
            name = 'QSO'
            # ADM only update priorities for passed observing conditions.
            pricon = obsconditions.mask(desi_mask[name].obsconditions)
            if (obsconditions.mask(obscon) & pricon) != 0:
                ii = (targets[desi_target] & desi_mask[name]) != 0
                # ADM all redshifts require more observations in SV.
                good_hiz = zgood & (zcat['Z'] >= 2.15) & (zcat['ZWARN'] == 0)
                if survey[0:2] == 'sv':
                    good_hiz = zgood & (zcat['ZWARN'] == 0)
                priority[ii & unobs] = np.maximum(priority[ii & unobs], desi_mask[name].priorities['UNOBS'])
                priority[ii & done] = np.maximum(priority[ii & done], desi_mask[name].priorities['DONE'])
                priority[ii & good_hiz] = np.maximum(priority[ii & good_hiz], desi_mask[name].priorities['MORE_ZGOOD'])
                priority[ii & ~good_hiz] = np.maximum(priority[ii & ~good_hiz], desi_mask[name].priorities['DONE'])
                priority[ii & zwarn] = np.maximum(priority[ii & zwarn], desi_mask[name].priorities['MORE_ZWARN'])

        # BGS targets.
        if bgs_target in targets.dtype.names:
            for name in bgs_mask.names():
                # ADM only update priorities for passed observing conditions.
                pricon = obsconditions.mask(bgs_mask[name].obsconditions)
                if (obsconditions.mask(obscon) & pricon) != 0:
                    ii = (targets[bgs_target] & bgs_mask[name]) != 0
                    priority[ii & unobs] = np.maximum(priority[ii & unobs], bgs_mask[name].priorities['UNOBS'])
                    priority[ii & done] = np.maximum(priority[ii & done],  bgs_mask[name].priorities['DONE'])
                    priority[ii & zgood] = np.maximum(priority[ii & zgood], bgs_mask[name].priorities['MORE_ZGOOD'])
                    priority[ii & zwarn] = np.maximum(priority[ii & zwarn], bgs_mask[name].priorities['MORE_ZWARN'])

        # MWS targets.
        if mws_target in targets.dtype.names:
            for name in mws_mask.names():
                # ADM only update priorities for passed observing conditions.
                pricon = obsconditions.mask(mws_mask[name].obsconditions)
                if (obsconditions.mask(obscon) & pricon) != 0:
                    ii = (targets[mws_target] & mws_mask[name]) != 0
                    priority[ii & unobs] = np.maximum(priority[ii & unobs], mws_mask[name].priorities['UNOBS'])
                    priority[ii & done] = np.maximum(priority[ii & done],  mws_mask[name].priorities['DONE'])
                    priority[ii & zgood] = np.maximum(priority[ii & zgood], mws_mask[name].priorities['MORE_ZGOOD'])
                    priority[ii & zwarn] = np.maximum(priority[ii & zwarn], mws_mask[name].priorities['MORE_ZWARN'])

        # ADM Secondary targets.
        if scnd_target in targets.dtype.names:
            # APC Secondary target bits only drive updates to targets with specific DESI_TARGET bits
            # APC See https://github.com/desihub/desitarget/pull/530
            scnd_update = (targets[desi_target] & desi_mask['SCND_ANY']) != 0
            if np.any(scnd_update):
                # APC Allow changes to primaries if the DESI_TARGET bitmask has any of the
                # APC following bits set, but not any other bits.
                update_from_scnd_bits = (desi_mask['SCND_ANY'] | desi_mask['MWS_ANY'] |
                                         desi_mask['STD_BRIGHT'] | desi_mask['STD_FAINT'] | desi_mask['STD_WD'])
                scnd_update &= ((targets[desi_target] & ~update_from_scnd_bits) == 0)
                print('{} scnd targets to be updated'.format(scnd_update.sum()))

            for name in scnd_mask.names():
                # ADM only update priorities for passed observing conditions.
                pricon = obsconditions.mask(scnd_mask[name].obsconditions)
                if (obsconditions.mask(obscon) & pricon) != 0:
                    ii = (targets[scnd_target] & scnd_mask[name]) != 0
                    ii &= scnd_update
                    priority[ii & unobs] = np.maximum(priority[ii & unobs], scnd_mask[name].priorities['UNOBS'])
                    priority[ii & done] = np.maximum(priority[ii & done],  scnd_mask[name].priorities['DONE'])
                    priority[ii & zgood] = np.maximum(priority[ii & zgood], scnd_mask[name].priorities['MORE_ZGOOD'])
                    priority[ii & zwarn] = np.maximum(priority[ii & zwarn], scnd_mask[name].priorities['MORE_ZWARN'])

        # Special case: IN_BRIGHT_OBJECT means priority=-1 no matter what
        ii = (targets[desi_target] & desi_mask.IN_BRIGHT_OBJECT) != 0
        priority[ii] = -1

    # ADM Special case: SV-like commissioning targets.
    if 'CMX_TARGET' in targets.dtype.names:
        priority = _cmx_calc_priority(targets, priority, obscon,
                                      unobs, done, zgood, zwarn, cmx_mask, obsconditions)

    return priority


def _cmx_calc_priority(targets, priority, obscon, unobs, done, zgood, zwarn, cmx_mask, obsconditions):
    """Special-case logic for target priorities in CMX.

    Parameters
    ----------
    targets : :class:`~numpy.ndarray`
        numpy structured array or astropy Table of targets. Must include
        the column `CMX_TARGET`.
    priority : :class:`~numpy.ndarray`
        Initial priority values set, in calc_priorities().
    obscon : :class:`str`
        A combination of strings that are in the desitarget bitmask yaml
        file (specifically in `desitarget.targetmask.obsconditions`), e.g.
        "DARK|GRAY". Governs the behavior of how priorities are set based
        on "obsconditions" in the desitarget bitmask yaml file.
    unobs : :class:`~numpy.ndarray`
        Boolean flag on targets indicating state UNOBS.
    done : :class:`~numpy.ndarray`
        Boolean flag on targets indicating state DONE.
    zgood : :class:`~numpy.ndarray`
        Boolean flag on targets indicating state ZGOOD.
    zwarn : :class:`~numpy.ndarray`
        Boolean flag on targets indicating state ZWARN.
    cmx_mask : :class:`~desiutil.bitmask.BitMask`
        The CMX target bitmask.
    obscondtions : :class:`~desiutil.bitmask.BitMask`
        The CMX obsconditions bitmask.

    Returns
    -------
    :class:`~numpy.ndarray`
        The updated priority values.

    Notes
    -----
        - Intended to be called only from within calc_priority(), where any
          pre-processing of the target state flags (uobs, done, zgood, zwarn) is
          handled.

    """
    # Build a whitelist of targets to update
    names_to_update = ['SV0_' + label for label in ('STD_FAINT', 'STD_BRIGHT',
                                                    'BGS', 'MWS', 'WD', 'MWS_FAINT',
                                                    'MWS_CLUSTER', 'MWS_CLUSTER_VERYBRIGHT')]
    names_to_update.extend(['BACKUP_BRIGHT', 'BACKUP_FAINT'])

    for name in names_to_update:
        pricon = obsconditions.mask(cmx_mask[name].obsconditions)
        if (obsconditions.mask(obscon) & pricon) != 0:
            ii = (targets['CMX_TARGET'] & cmx_mask[name]) != 0
            priority[ii & unobs] = np.maximum(priority[ii & unobs], cmx_mask[name].priorities['UNOBS'])
            priority[ii & done] = np.maximum(priority[ii & done],  cmx_mask[name].priorities['DONE'])
            priority[ii & zgood] = np.maximum(priority[ii & zgood], cmx_mask[name].priorities['MORE_ZGOOD'])
            priority[ii & zwarn] = np.maximum(priority[ii & zwarn], cmx_mask[name].priorities['MORE_ZWARN'])

    return priority


def resolve(targets):
    """Resolve which targets are primary in imaging overlap regions.

    Parameters
    ----------
    targets : :class:`~numpy.ndarray`
        Rec array of targets. Must have columns "RA" and "DEC" and
        either "RELEASE" or "PHOTSYS".

    Returns
    -------
    :class:`~numpy.ndarray`
        The original target list trimmed to only objects from the "northern"
        photometry in the northern imaging area and objects from "southern"
        photometry in the southern imaging area.
    """
    # ADM retrieve the photometric system from the RELEASE.
    from desitarget.io import release_to_photsys, desitarget_resolve_dec
    if 'PHOTSYS' in targets.dtype.names:
        photsys = targets["PHOTSYS"]
    else:
        photsys = release_to_photsys(targets["RELEASE"])

    # ADM a flag of which targets are from the 'N' photometry.
    from desitarget.cuts import _isonnorthphotsys
    photn = _isonnorthphotsys(photsys)

    # ADM grab the declination used to resolve targets.
    split = desitarget_resolve_dec()

    # ADM determine which targets are north of the Galactic plane. As
    # ADM a speed-up, bin in ~1 sq.deg. HEALPixels and determine
    # ADM which of those pixels are north of the Galactic plane.
    # ADM We should never be as close as ~1o to the plane.
    from desitarget.geomask import is_in_gal_box, pixarea2nside
    nside = pixarea2nside(1)
    theta, phi = np.radians(90-targets["DEC"]), np.radians(targets["RA"])
    pixnum = hp.ang2pix(nside, theta, phi, nest=True)
    # ADM find the pixels north of the Galactic plane...
    allpix = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, allpix, nest=True)
    ra, dec = np.degrees(phi), 90-np.degrees(theta)
    pixn = is_in_gal_box([ra, dec], [0., 360., 0., 90.], radec=True)
    # ADM which targets are in pixels north of the Galactic plane.
    galn = pixn[pixnum]

    # ADM which targets are in the northern imaging area.
    arean = (targets["DEC"] >= split) & galn

    # ADM retain 'N' targets in 'N' area and 'S' in 'S' area.
    keep = (photn & arean) | (~photn & ~arean)

    return targets[keep]


def finalize(targets, desi_target, bgs_target, mws_target,
             sky=0, survey='main', darkbright=False, gaiadr=None,
             targetid=None):
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
    darkbright : :class:`bool`, optional, defaults to ``False``
        If sent, then split `NUMOBS_INIT` and `PRIORITY_INIT` into
        `NUMOBS_INIT_DARK`, `NUMOBS_INIT_BRIGHT`, `PRIORITY_INIT_DARK`
        and `PRIORITY_INIT_BRIGHT` and calculate values appropriate
        to "BRIGHT" and "DARK|GRAY" observing conditions.
    gaiadr : :class:`int`, optional, defaults to ``None``
        If passed and not ``None``, then build the `TARGETID` from the
        "GAIA_OBJID" and "GAIA_BRICKID" columns in the passed `targets`,
        and set the `gaiadr` part of `TARGETID` to whatever is passed.
    targetid : :class:`int64`, optional, defaults to ``None``
        In the mocks we compute `TARGETID` outside this function.

    Returns
    -------
    :class:`~numpy.ndarray`
       new targets structured array with the following additions:
          * renaming OBJID -> BRICK_OBJID (it is only unique within a brick).
          * renaming TYPE -> MORPHTYPE (used downstream in other contexts).
          * Adding new columns:
              - TARGETID: unique ID across all bricks or Gaia files.
              - DESI_TARGET: dark time survey target selection flags.
              - MWS_TARGET: bright time MWS target selection flags.
              - BGS_TARGET: bright time BGS target selection flags.
              - PRIORITY_INIT: initial priority for observing target.
              - SUBPRIORITY: a placeholder column that is set to zero.
              - NUMOBS_INIT: initial number of observations for target.
              - OBSCONDITIONS: bitmask of observation conditions.

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

    # allow TARGETID to be passed as an input (specifically for the mocks).
    if targetid is None:
        if gaiadr is not None:
            targetid = encode_targetid(objid=targets['GAIA_OBJID'],
                                       brickid=targets['GAIA_BRICKID'],
                                       release=0,
                                       sky=sky,
                                       gaiadr=gaiadr)
        else:
            targetid = encode_targetid(objid=targets['BRICK_OBJID'],
                                       brickid=targets['BRICKID'],
                                       release=targets['RELEASE'],
                                       sky=sky)
    assert ntargets == len(targetid)

    nodata = np.zeros(ntargets, dtype='int')-1
    subpriority = np.zeros(ntargets, dtype='float')

    # ADM new columns are different depending on SV/cmx/main survey.
    if survey == 'main':
        colnames = ['DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET']
    elif survey == 'cmx':
        colnames = ['CMX_TARGET']
    elif survey[0:2] == 'sv':
        colnames = ["{}_{}_TARGET".format(survey.upper(), tc)
                    for tc in ["DESI", "BGS", "MWS"]]
    else:
        msg = "survey must be 'main', 'cmx' or 'svX' (X=1,2..etc.), not {}!"   \
            .format(survey)
        log.critical(msg)
        raise ValueError(msg)

    # ADM the columns to write out and their values and formats.
    cols = ["TARGETID"] + colnames + ['SUBPRIORITY', 'OBSCONDITIONS']
    vals = [targetid] + [desi_target, bgs_target, mws_target][:len(colnames)]  \
        + [subpriority, nodata]
    forms = ['>i8'] + ['>i8', '>i8', '>i8'][:len(colnames)] + ['>f8', '>i8']

    # ADM set the initial PRIORITY and NUMOBS.
    if darkbright:
        # ADM populate bright/dark if splitting by survey OBSCONDITIONS.
        ender, obscon = ["_DARK", "_BRIGHT"], ["DARK|GRAY", "BRIGHT"]
    else:
        ender, obscon = [""], ["DARK|GRAY|BRIGHT|POOR|TWILIGHT12|TWILIGHT18"]
    for edr, oc in zip(ender, obscon):
        cols += ["{}_INIT{}".format(pn, edr) for pn in ["PRIORITY", "NUMOBS"]]
        vals += [nodata, nodata]
        forms += ['>i8', '>i8']

    # ADM write the output array.
    newdt = [dt for dt in zip(cols, forms)]
    done = np.array(np.zeros(len(targets)), dtype=targets.dtype.descr+newdt)
    for col in targets.dtype.names:
        done[col] = targets[col]
    for col, val in zip(cols, vals):
        done[col] = val

    # ADM add PRIORITY/NUMOBS columns.
    for edr, oc in zip(ender, obscon):
        pc, nc = "PRIORITY_INIT"+edr, "NUMOBS_INIT"+edr
        done[pc], done[nc] = initial_priority_numobs(done, obscon=oc)

    # ADM set the OBSCONDITIONS.
    done["OBSCONDITIONS"] = set_obsconditions(done)

    # ADM some final checks that the targets conform to expectations...
    # ADM check that each target has a unique ID.
    if len(done["TARGETID"]) != len(set(done["TARGETID"])):
        msg = 'TARGETIDs are not unique!'
        log.critical(msg)
        raise AssertionError(msg)

    # ADM check all LRG targets have LRG_1PASS/2PASS set.
    # ADM we've moved away from LRG PASSes this so deprecate for now.
#    if survey == 'main':
#        lrgset = done["DESI_TARGET"] & desi_mask.LRG != 0
#        pass1lrgset = done["DESI_TARGET"] & desi_mask.LRG_1PASS != 0
#        pass2lrgset = done["DESI_TARGET"] & desi_mask.LRG_2PASS != 0
#        if not np.all(lrgset == pass1lrgset | pass2lrgset):
#            msg = 'Some LRG targets do not have 1PASS/2PASS set!'
#            log.critical(msg)
#            raise AssertionError(msg)

    return done
