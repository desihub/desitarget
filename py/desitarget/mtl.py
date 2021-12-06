"""
desitarget.mtl
==============

Merged target lists.

.. _`STRICT ISO format`: https://stackoverflow.com/a/55157458
"""

import os
import numpy as np
import healpy as hp
import numpy.lib.recfunctions as rfn
import sys
from astropy.table import Table, hstack, vstack
from astropy.io import ascii
import fitsio
from time import time, sleep
from datetime import datetime, timezone
from glob import glob, iglob

from . import __version__ as dt_version
from desitarget.targetmask import obsmask, obsconditions, zwarn_mask
from desitarget.targets import calc_priority, calc_numobs_more
from desitarget.targets import main_cmx_or_sv, switch_main_cmx_or_sv
from desitarget.targets import set_obsconditions, decode_targetid
from desitarget.geomask import match, match_to
from desitarget.internal import sharedmem
from desitarget import io
from desimodel.footprint import is_point_in_desi, tiles2pix

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()

# ADM the data model for MTL. Note that the _TARGET columns will have
# ADM to be changed on the fly for SV1_, SV2_, etc. files.
# ADM OBSCONDITIONS is formatted as just 'i4' for backward compatibility.
mtldatamodel = np.array([], dtype=[
    ('RA', '>f8'), ('DEC', '>f8'), ('PARALLAX', '>f4'),
    ('PMRA', '>f4'), ('PMDEC', '>f4'), ('REF_EPOCH', '>f4'),
    ('DESI_TARGET', '>i8'), ('BGS_TARGET', '>i8'), ('MWS_TARGET', '>i8'),
    ('SCND_TARGET', '>i8'), ('TARGETID', '>i8'),
    ('SUBPRIORITY', '>f8'), ('OBSCONDITIONS', 'i4'),
    ('PRIORITY_INIT', '>i8'), ('NUMOBS_INIT', '>i8'), ('PRIORITY', '>i8'),
    ('NUMOBS', '>i8'), ('NUMOBS_MORE', '>i8'), ('Z', '>f8'), ('ZWARN', '>i8'),
    ('TIMESTAMP', 'U25'), ('VERSION', 'U14'), ('TARGET_STATE', 'U30'),
    ('ZTILEID', '>i4')
    ])

# ADM columns to add to the mtl/zcat data models for the Main Survey.
msaddcols = np.array([], dtype=[
    ('Z_QN', '>f8'), ('IS_QSO_QN', '>i2'), ('DELTACHI2', '>f8'),
    ])

zcatdatamodel = np.array([], dtype=[
    ('RA', '>f8'), ('DEC', '>f8'), ('TARGETID', '>i8'),
    ('NUMOBS', '>i4'), ('Z', '>f8'), ('ZWARN', '>i8'), ('ZTILEID', '>i4')
    ])

mtltilefiledm = np.array([], dtype=[
    ('TILEID', '>i4'), ('TIMESTAMP', 'U25'), ('VERSION', 'U14'),
    ('PROGRAM', 'U6'), ('ZDATE', '>i8'), ('ARCHIVEDATE', '>i8')
    ])


# ADM when using basic or csv ascii writes, specifying the formats of
# ADM float32 columns can make things easier on the eye.
mtlformatdict = {"PARALLAX": '%16.8f', 'PMRA': '%16.8f', 'PMDEC': '%16.8f'}


def survey_data_model(dm, survey='main'):
    """Construct the appropriate data model for a given survey.

    Parameters
    ----------
    dm : :class:`~numpy.array`
        A data model related to MTL. Typically one of `zcatdatamodel` or
        `mtldatamodel`.
    survey : :class:`str`, optional, defaults to "main"
        Used to construct the right data model for an iteration of DESI.
        Options are ``'main'`` ``'cmx'``, ``'svX``' (for X of 1, 2, etc.)
        for the main survey, commissioning and iterations of SV.

    Returns
    -------
    :class:`~numpy.array`
        The approriate data model. If `survey` is `'main'` this will be
        the passed `dm` with any columns from `msaddcols` added. If
        `survey` is sv-like, this will just be the input `dm`.
    """
    if survey[:2] == 'sv' or survey == 'cmx':
        return dm
    elif survey == 'main':
        return np.array([],
                        dtype=dm.dtype.descr + msaddcols.dtype.descr)
    else:
        msg = "Allowed 'survey' inputs are sv(X) or main, not {}".format(survey)
        log.critical(msg)
        raise ValueError(msg)


def check_timestamp(timestamp):
    """Check whether a timestamp is in a valid datetime format.

    Parameters
    ----------
    timestamp : :class:`str`
        A string that should be a valid datetime string with a timezone.

    Returns
    -------
    :class:`str`
        The input `timestamp` string.

    Notes
    -----
    - Triggers an exception if the string is not a valid datetime string
      or if the timezone was not included in the string.
    """
    from dateutil.parser import parse

    try:
        check = parse(timestamp)
    except ValueError:
        msg = "{} is not a valid timestamp!!!".format(timestamp)
        log.critical(msg)
        raise ValueError(msg)

    if check.tzinfo is None:
        msg = "{} does not include timezone information!!!".format(timestamp)
        log.critical(msg)
        raise ValueError(msg)

    return timestamp


def utc_date_to_night(timestamp):
    """Convert a UTC/ISO date into a DESI night (YYYYMMDD) format.

    Parameters
    ----------
    timestamp : :class:`~numpy.array` or :class:`str`
        A date-like string or array of strings in either UTC or strict
        UTC/ISO format.

    Returns
    -------
    :class:`~numpy.array`
        An array of nights in the DESI YYYYMMDD format, where each entry
        corresponds to each entry in `timestamp`.

    Notes
    -----
    - This function is sufficiently sloppy that it can handle both
      `STRICT ISO format`_ (as generated by, e.g.,
      :func:`desitarget.mtl.get_utc_iso_date()`) and general UTC format
      (as generated by, e.g., :func:`desitarget.mtl.get_utc_date()`).
    """
    # ADM in case a single string was passed.
    if isinstance(timestamp, str):
        timestamp = [timestamp]

    # ADM convert each input timestamp to a datetime object.
    dts = [datetime.fromisoformat(ts) for ts in timestamp]
    # ADM convert the datetime object to DESI night format.
    yyyymmdd = [int(f'{dt.year:04}{dt.month:02}{dt.day:02}') for dt in dts]

    return np.array(yyyymmdd)


def get_utc_date(survey="sv3"):
    """Convenience function to grab the UTC date.

    Parameters
    ----------
    survey : :class:`str`, optional, defaults to "sv3"
        Used to construct the right ISO format for an iteration of DESI.
        Options are ``'main'`` ``'cmx'``, ``'svX``' (for X of 1, 2, etc.)
        for the main survey, commissioning and iterations of SV.

    Returns
    -------
    :class:`str`
        The UTC date, appropriate for making a TIMESTAMP.

    Notes
    -----
    - This is spun off into its own function to have a consistent way to
      record time across the entire desitarget package.
    - The `survey` input defaults to `"sv3"` for backwards compatibility
      (we became stricter about the format for the main survey).
    """
    if survey[:2] == 'sv' or survey == 'cmx':
        return datetime.utcnow().isoformat(timespec='seconds')
    elif survey == 'main':
        return get_utc_iso_date()
    else:
        msg = "Allowed 'survey' inputs are sv(X) or main, not {}".format(survey)
        log.critical(msg)
        raise ValueError(msg)


def get_utc_iso_date():
    """Convenience function to grab the UTC date in STRICT ISO format.

    Returns
    -------
    :class:`str`
        UTC date in `STRICT ISO format`_, appropriate for a TIMESTAMP.
    """
    return datetime.now(tz=timezone.utc).isoformat(timespec='seconds')


def get_mtl_dir(mtldir=None):
    """Convenience function to grab the $MTL_DIR environment variable.

    Parameters
    ----------
    mtldir : :class:`str`, optional, defaults to $MTL_DIR
        If `mtldir` is passed, it is returned from this function. If it's
        not passed, the $MTL_DIR environment variable is returned.

    Returns
    -------
    :class:`str`
        If `mtldir` is passed, it is returned from this function. If it's
        not passed, the directory stored in the $MTL_DIR environment
        variable is returned.
    """
    if mtldir is None:
        mtldir = os.environ.get('MTL_DIR')
        # ADM check that the $MTL_DIR environment variable is set.
        if mtldir is None:
            msg = "Pass mtldir or set $MTL_DIR environment variable!"
            log.critical(msg)
            raise ValueError(msg)

    return mtldir


def get_zcat_dir(zcatdir=None):
    """Convenience function to grab the $ZCAT_DIR environment variable.

    Parameters
    ----------
    zcatdir : :class:`str`, optional, defaults to $ZCAT_DIR
        If `zcatdir` is passed, it is returned from this function. If it's
        not passed, the $ZCAT_DIR environment variable is returned.

    Returns
    -------
    :class:`str`
        If `zcatdir` is passed, it is returned from this function. If it's
        not passed, the directory stored in the $ZCAT_DIR environment
        variable is returned.
    """
    if zcatdir is None:
        zcatdir = os.environ.get('ZCAT_DIR')
        # ADM check that the $ZCAT_DIR environment variable is set.
        if zcatdir is None:
            msg = "Pass zcatdir or set $ZCAT_DIR environment variable!"
            log.critical(msg)
            raise ValueError(msg)

    return zcatdir


def get_mtl_tile_file_name(secondary=False):
    """Convenience function to grab the name of the MTL tile file.

    Parameters
    ----------
    secondary : :class:`bool`, optional, defaults to ``False``
        If ``True`` return the name of the MTL tile file for secondary
        targets instead of the standard, primary MTL tile file.

    Returns
    -------
    :class:`str`
        The name of the MTL tile file.
    """
    fn = "mtl-done-tiles.ecsv"
    if secondary:
        fn = "scnd-mtl-done-tiles.ecsv"

    return fn


def get_ztile_file_name(survey='main'):
    """Convenience function to grab the name of the ZTILE file.

    survey : :class:`str`, optional, defaults to "main"
        To look up the correct ZTILE filename. Options are ``'main'`` and
        ``'svX``' (where X is 1, 2, 3 etc.) for the main survey and
        different iterations of SV, respectively.

    Returns
    -------
    :class:`str`
        The name of the ZTILE file.
    """
    # ADM tile file name used to be different for sv and main.
    if survey[:2] == 'sv' or survey == 'main':
        fn = "tiles-specstatus.ecsv"
    else:
        msg = "Allowed 'survey' inputs are sv(X) or main, not {}".format(survey)
        log.critical(msg)
        raise ValueError(msg)

    return fn


def _get_mtl_nside():
    """Grab the HEALPixel nside to be used for MTL ledger files.

    Returns
    -------
    :class:`int`
        The HEALPixel nside number for MTL file creation and retrieval.
    """
    # from desitarget.geomask import pixarea2nside
    # ADM the nside (16) appropriate to a 7 sq. deg. field.
    # nside = pixarea2nside(7)
    nside = 32

    return nside


def get_mtl_ledger_format():
    """Grab the file format for MTL ledger files.

    Returns
    -------
    :class:`str`
        The file format for MTL ledgers. Should be "ecsv" or "fits".
    """
    # ff = "fits"
    ff = "ecsv"

    return ff


def make_mtl(targets, obscon, zcat=None, scnd=None,
             trim=False, trimcols=False, trimtozcat=False):
    """Add zcat columns to a targets table, update priorities and NUMOBS.

    Parameters
    ----------
    targets : :class:`~numpy.array` or `~astropy.table.Table`
        A numpy rec array or astropy Table with at least the columns
        `TARGETID`, `DESI_TARGET`, `BGS_TARGET`, `MWS_TARGET` (or the
        corresponding columns for SV or commissioning) `NUMOBS_INIT` and
        `PRIORITY_INIT`. `targets` must also contain `PRIORITY` if `zcat`
        is not ``None`` (i.e. if this isn't the first time through MTL
        and/or if `targets` is itself an mtl array). `PRIORITY` is needed
        to "lock in" the state of Ly-Alpha QSOs. `targets` may also
        contain `SCND_TARGET` (or the corresponding columns for SV) if
        secondary targets are under consideration.
    obscon : :class:`str`
        A combination of strings that are in the desitarget bitmask yaml
        file (specifically in `desitarget.targetmask.obsconditions`), e.g.
        "DARK|GRAY". Governs the behavior of how priorities are set based
        on "obsconditions" in the desitarget bitmask yaml file.
    zcat : :class:`~astropy.table.Table`, optional
        Redshift catalog table with columns ``TARGETID``, ``NUMOBS``,
        ``Z``, ``ZWARN``, ``ZTILEID``, and possibly the extra columns in
        ``msaddcols`` at the top of the module.
    scnd : :class:`~numpy.array`, `~astropy.table.Table`, optional
        TYPICALLY, we have a separate secondary targets (they have their
        own "ledger"). So passing associated secondaries is DEPRECATED
        (but still works). `scnd` is kept for backwards compatibility.
        A set of secondary targets associated with the `targets`. As with
        the `target` must include at least ``TARGETID``, ``NUMOBS_INIT``,
        ``PRIORITY_INIT`` or the corresponding SV columns.
        The secondary targets will be padded to have the same columns
        as the targets, and concatenated with them.
    trim : :class:`bool`, optional
        If ``True`` (default), don't include targets that don't need
        any more observations.  If ``False``, include every input target.
    trimcols : :class:`bool`, optional, defaults to ``False``
        Only pass through columns in `targets` that are actually needed
        for fiberassign (see `desitarget.mtl.mtldatamodel`).
    trimtozcat : :class:`bool`, optional, defaults to ``False``
        Only return targets that have been UPDATED (i.e. the targets with
        a match in `zcat`). Returns all targets if `zcat` is ``None``.

    Returns
    -------
    :class:`~astropy.table.Table`
        MTL Table with targets columns plus:

        * NUMOBS_MORE    - number of additional observations requested
        * PRIORITY       - target priority (larger number = higher priority)
        * TARGET_STATE   - the observing state that corresponds to PRIORITY
        * OBSCONDITIONS  - replaces old GRAYLAYER
        * TIMESTAMP      - time that (this) make_mtl() function was run
        * VERSION        - version of desitarget used to run make_mtl()

    Notes
    -----
    - Sources in the zcat with `ZWARN` of `NODATA` are always ignored.
    - The input `zcat` WILL BE MODIFIED. So, if a desire is that `zcat`
      remains unaltered, make sure to copy `zcat` before passing it.
    """
    start = time()
    # ADM set up the default logger.
    from desiutil.log import get_logger
    log = get_logger()

    # ADM if trimcols was passed, reduce input target columns to minimal.
    if trimcols:
        mtldm = switch_main_cmx_or_sv(mtldatamodel, targets)
        # ADM the data model for mtl depends on the survey type.
        _, _, survey = main_cmx_or_sv(mtldm)
        mtldm = survey_data_model(mtldm, survey=survey)
        cullcols = list(set(targets.dtype.names) - set(mtldm.dtype.names))
        if isinstance(targets, Table):
            targets.remove_columns(cullcols)
        else:
            targets = rfn.drop_fields(targets, cullcols)

    # ADM determine whether the input targets are main survey, cmx or SV.
    colnames, masks, survey = main_cmx_or_sv(targets, scnd=True)
    # ADM set the first column to be the "desitarget" column
    desi_target, desi_mask = colnames[0], masks[0]
    scnd_target, scnd_mask = colnames[-1], masks[-1]

    # ADM if secondaries were passed, concatenate them with the targets.
    if scnd is not None:
        nrows = len(scnd)
        log.info('Pad {} primary targets with {} secondaries...t={:.1f}s'.format(
            len(targets), nrows, time()-start))
        padit = np.zeros(nrows, dtype=targets.dtype)
        sharedcols = set(targets.dtype.names).intersection(set(scnd.dtype.names))
        for col in sharedcols:
            padit[col] = scnd[col]
        targets = np.concatenate([targets, padit])
        # APC Propagate a flag on which targets came from scnd
        is_scnd = np.repeat(False, len(targets))
        is_scnd[-nrows:] = True
        log.info('Done with padding...t={:.1f}s'.format(time()-start))

    # Trim targets from zcat that aren't in original targets table.
    # ADM or that didn't actually obtain an observation.
    if zcat is not None:
        ok = np.in1d(zcat['TARGETID'], targets['TARGETID'])
        num_extra = np.count_nonzero(~ok)
        if num_extra > 0:
            log.info("Ignoring {} z entries that aren't in the input target list"
                     " (e.g. likely skies, secondaries-when-running-primary, "
                     "primaries-when-running-secondary, etc.)".format(num_extra))
            zcat = zcat[ok]
        # ADM also ignore anything with NODATA set in ZWARN.
        nodata = zcat["ZWARN"] & zwarn_mask["NODATA"] != 0
        num_nod = np.sum(nodata)
        if num_nod > 0:
            log.info("Ignoring a further {} zcat entries with NODATA set".format(
                num_nod))
            zcat = zcat[~nodata]
        # SB ignore targets that failed QA: ZWARN bits BAD_SPECQA|BAD_PETALQA
        badqa = zcat["ZWARN"] & zwarn_mask.mask("BAD_SPECQA|BAD_PETALQA") != 0
        num_badqa = np.sum(badqa)
        if num_badqa > 0:
            log.info(f"Ignoring a further {num_badqa} zcat entries with BAD_SPECQA or BAD_PETALQA set")
            zcat = zcat[~badqa]
        # ADM simulations (I think) and some unit tests expect zcat to
        # ADM be modified by make_mtl().
        if num_extra > 0 or num_nod > 0 or num_badqa > 0:
            msg = "The size of the zcat has changed, so it won't be modified!"
            log.warning(msg)

    n = len(targets)
    # ADM if a redshift catalog was passed, order it to match the input targets
    # ADM catalog on 'TARGETID'.
    if zcat is not None:
        # ADM find where zcat matches target array.
        zmatcher = match_to(targets["TARGETID"], zcat["TARGETID"])
        ztargets = zcat
        if ztargets.masked:
            unobs = ztargets['NUMOBS'].mask
            ztargets['NUMOBS'][unobs] = 0
            unobsz = ztargets['Z'].mask
            ztargets['Z'][unobsz] = -1
            unobszw = ztargets['ZWARN'].mask
            ztargets['ZWARN'][unobszw] = -1
    else:
        ztargets = Table()
        ztargets['TARGETID'] = targets['TARGETID']
        ztargets['NUMOBS'] = np.zeros(n, dtype=np.int32)
        ztargets['Z'] = -1 * np.ones(n, dtype=np.float32)
        ztargets['ZWARN'] = -1 * np.ones(n, dtype=np.int32)
        # ADM a catch all for added zcat columns.
        xtradescr = [dt for dt in mtldatamodel.dtype.descr if dt[0] == "ZTILEID"]
        if survey == 'main':
            xtradescr += msaddcols.dtype.descr
        for xctuple in xtradescr:
            xtracol, dt = xctuple
            # ADM default to "_" instead of -1 for strings.
            if isinstance(np.empty(1, dtype=dt).item(), str):
                ztargets[xtracol] = np.full(n, "-", dtype=dt)
            else:
                ztargets[xtracol] = np.full(n, -1, dtype=dt)
        # ADM if zcat wasn't passed, there is a one-to-one correspondence
        # ADM between the targets and the zcat.
        zmatcher = np.arange(n)

    # ADM extract just the targets that match the input zcat.
    targets_zmatcher = targets[zmatcher]

    # ADM special cases for SV3.
    if survey == "sv3":
        if zcat is not None:
            # ADM a necessary hack as we created ledgers for SV3 with
            # ADM NUMOBS_INIT==9 then later decided on NUMOBS_INIT==3.
            ii = targets_zmatcher["NUMOBS_INIT"] == 9
            targets_zmatcher["NUMOBS_INIT"][ii] = 3
            # ADM make sure to also force a permanent change of state for
            # ADM the actual *targets* that will be returned as the mtl.
            targets["NUMOBS_INIT"][zmatcher[ii]] = 3
        if (obsconditions.mask(obscon) & obsconditions.mask("DARK")) != 0:
            # ADM In dark time, if a QSO target is above feasible galaxy
            # ADM redshifts, NUMOBS should behave like a QSO, not an ELG.
            ii = targets_zmatcher[desi_target] & desi_mask["QSO"] != 0
            # ADM the secondary bit-names that correspond to primary QSOs.
            sns = [bn for bn in scnd_mask.names() if scnd_mask[bn].flavor == 'QSO']
            for sn in sns:
                ii |= targets_zmatcher[scnd_target] & scnd_mask[sn] != 0
            # ADM above feasible galaxy redshifts (with no warning).
            ii &= ztargets['Z'] > 1.6
            ii &= ztargets['ZWARN'] == 0
            targets_zmatcher["NUMOBS_INIT"][ii] = desi_mask["QSO"].numobs

    # ADM update the number of observations for the targets.
    ztargets['NUMOBS_MORE'] = calc_numobs_more(targets_zmatcher, ztargets, obscon)

    # ADM assign priorities. Only things in the zcat can have changed
    # ADM priorities. Anything else is assigned PRIORITY_INIT, below.
    priority, target_state = calc_priority(
        targets_zmatcher, ztargets, obscon, state=True)

    # If priority went to 0==DONOTOBSERVE or 1==OBS or 2==DONE, then
    # NUMOBS_MORE should also be 0.
    # ## mtl['NUMOBS_MORE'] = ztargets['NUMOBS_MORE']
    ii = (priority <= 2)
    log.info('{:d} of {:d} targets have priority <=2, setting N_obs=0.'.format(
        np.sum(ii), n))
    ztargets['NUMOBS_MORE'][ii] = 0

    # - Set the OBSCONDITIONS mask for each target bit.
    obsconmask = set_obsconditions(targets)

    # APC obsconmask will now be incorrect for secondary-only targets. Fix this
    # APC using the mask on secondary targets.
    if scnd is not None:
        obsconmask[is_scnd] = set_obsconditions(targets[is_scnd], scnd=True)

    # ADM set up the output mtl table.
    mtl = Table(targets)
    mtl.meta['EXTNAME'] = 'MTL'

    # ADM use the Main Survey data model, if appropriate.
    mtldm = survey_data_model(mtldatamodel, survey=survey)

    # ADM add a placeholder for the secondary bit-mask, if it isn't there.
    if scnd_target not in mtl.dtype.names:
        mtl[scnd_target] = np.zeros(len(mtl),
                                    dtype=mtldm["SCND_TARGET"].dtype)

    # ADM initialize columns to avoid zero-length/missing/format errors.
    zcols = ["NUMOBS_MORE", "NUMOBS", "Z", "ZWARN", "ZTILEID"]
    if survey == 'main':
        zcols += list(msaddcols.dtype.names)
    for col in zcols + ["TARGET_STATE", "TIMESTAMP", "VERSION"]:
        mtl[col] = np.empty(len(mtl), dtype=mtldm[col].dtype)

    # ADM any target that wasn't matched to the ZCAT should retain its
    # ADM original (INIT) value of PRIORITY and NUMOBS.
    mtl['NUMOBS_MORE'] = mtl['NUMOBS_INIT']
    mtl['PRIORITY'] = mtl['PRIORITY_INIT']
    mtl['TARGET_STATE'] = "UNOBS"
    # ADM add the time and version of the desitarget code that was run.
    mtl["TIMESTAMP"] = get_utc_date(survey=survey)
    mtl["VERSION"] = dt_version

    # ADM now populate the new mtl columns with the updated information.
    mtl['OBSCONDITIONS'] = obsconmask
    mtl['PRIORITY'][zmatcher] = priority
    mtl['TARGET_STATE'][zmatcher] = target_state
    for col in zcols:
        mtl[col][zmatcher] = ztargets[col]
    # ADM add ZTILEID, other columns, if passed, otherwise we're likely
    # ADM to be working with non-ledger-based mocks and can let it slide.
    xtradescr = [dt for dt in mtldatamodel.dtype.descr if dt[0] == "ZTILEID"]
    if survey == 'main':
        xtradescr += msaddcols.dtype.descr
    for xctuple in xtradescr:
        xtracol, dt = xctuple
        if xtracol in ztargets.dtype.names:
            mtl[xtracol][zmatcher] = ztargets[xtracol]
        else:
            # ADM default to "_" instead of -1 for strings.
            if isinstance(np.empty(1, dtype=dt).item(), str):
                mtl[xtracol] = "-"
            else:
                mtl[xtracol] = -1

    # Filter out any targets marked as done.
    if trim:
        notdone = mtl['NUMOBS_MORE'] > 0
        log.info('{:d} of {:d} targets are done, trimming these'.format(
            len(mtl) - np.sum(notdone), len(mtl))
        )
        mtl = mtl[notdone]

    # Filtering can reset the fill_value, which is just wrong wrong wrong
    # See https://github.com/astropy/astropy/issues/4707
    # and https://github.com/astropy/astropy/issues/4708
    mtl['NUMOBS_MORE'].fill_value = -1

    log.info('Done...t={:.1f}s'.format(time()-start))

    if trimtozcat:
        return mtl[zmatcher]
    return mtl


def find_non_overlap_tiles(obscon, mtldir=None, isodate=None, check=False):
    """
    Create (or append to) a ledger to override the standard ledgers.

    Parameters
    ----------
    obscon : :class:`str`
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "DARK"
        Used to construct the directory to find the Main Survey ledgers.
    mtldir : :class:`str`, optional, defaults to ``None``
        Full path to the directory that hosts the MTL ledgers and the MTL
        tile file. If ``None``, then look up the MTL directory from the
        $MTL_DIR environment variable.
    isodate : :class:`str`, optional, defaults to ``None``
        A date in ISO format, such as returned by
        :func:`desitarget.mtl.get_utc_date()`. Only tiles processed
        AFTER OR EXACTLY ON `isodate` are considered. If ``None`` then
        no date restrictions are applied.
    check : :class:`bool`, optional, defaults to ``False``
        If ``True``, then instead of a list of non-overlapping tiles,
        return a dictionary whose keys are the list of non-overlapping
        tiles and whose values are dictionaries who, in turn, have keys
        corresponding to any past overlapping tiles and values that are
        the `TIMESTAMP` for that past overlapping tile.

    Returns
    -------
    :class:`~astropy.table.Table` or `dict`
        A table of tiles (in the standard DESI format) that weren't
        covered by a future, overlapping tile, sorted by `TILEID`. A
        dictionary is returned instead if `check` is passed as ``True``.

    Notes
    -----
    - Requires both the ops and mtl parts of the operations SVN trunk to
      be checked out in the standard location relative to `mtldir`.
    - Only works for Main Survey tiles, because we only look in the
      tiles-main.ecsv file and in $MTL_DIR/main.
    """
    t0 = time()
    # ADM grab the MTL directory (in case we're relying on $MTL_DIR).
    mtldir = get_mtl_dir(mtldir)
    # ADM construct the full path to the mtl tile file.
    mtltilefn = os.path.join(mtldir, get_mtl_tile_file_name())
    # ADM read the tiles in reverse chronological order.
    tiles = Table.read(mtltilefn)
    ii = np.flip(np.argsort(tiles["TIMESTAMP"]))
    tiles = tiles[ii]
    # ADM and restrict to the obscon of interest.
    ii = tiles["PROGRAM"] == obscon.upper()
    log.info("{} {} tiles in MTL done file...t={:.1f}s".format(
        np.sum(ii), obscon, time()-t0))
    tiles = tiles[ii]

    # ADM grab the ops directory to retrieve the full tile information.
    opsdir = os.path.join(mtldir[:-3], 'ops')
    opstilefn = os.path.join(opsdir, "tiles-main.ecsv")
    alltileinfo = Table.read(opstilefn)
    # ADM restrict to tiles that have been observed in specified obscon.
    ii = alltileinfo["PROGRAM"] == obscon.upper()
    ii &= alltileinfo["STATUS"] != "unobs"
    # ADM remove any potentially retired tiles.
    ii &= alltileinfo["IN_DESI"]
    alltileinfo = alltileinfo[ii]

    # ADM retrieve observed tiles that have not been processed by MTL.
    alltilematch, tilematch = match(alltileinfo["TILEID"], tiles["TILEID"])
    alltilenomatch = np.array(list(
        set(np.arange(len(alltileinfo))) - set(alltilematch)))
    log.info("{} observed {} tiles not yet processed by MTL...t={:.1f}s".format(
        len(alltilenomatch), obscon, time()-t0))
    obstileinfo = alltileinfo[alltilenomatch]

    # ADM restrict MTL tiles to just the dates of interest, if requested.
    if isodate is not None:
        ii = tiles["TIMESTAMP"] >= isodate
        log.info("{} {} tiles MTL-processed on or after {}...t={:.1f}s".format(
            np.sum(ii), obscon, isodate, time()-t0))
        tiles = tiles[ii]

    # ADM retrieve the full tile information for MTL tiles of interest.
    aii = match_to(alltileinfo["TILEID"], tiles["TILEID"])
    log.info("matched {} ops tiles to mtl tiles...t={:.1f}s".format(
        len(aii), time()-t0))
    alltileinfo = alltileinfo[aii]

    # ADM read in the potential additional targets that could have been
    # ADM observed since MTL was last run...
    ledgerdir = os.path.join(mtldir, "main", obscon.lower())
    obstargs = io.read_targets_in_tiles(ledgerdir, tiles=obstileinfo, mtl=True)
    # ADM ...and restrict to just overlapping targets that have been
    # ADM processed through MTL at least once before.
    ii = obstargs["ZTILEID"] != -1
    obstargs = obstargs[ii]
    log.info("Read {} potentially observed targets in {} tiles...t={:.1f}s"
             .format(np.sum(ii), len(obstileinfo), time()-t0))

    # ADM read in targets in the tiles that have been processed by MTL...
    # ADM (unique=False returns targets observed on > 1 tile).
    targs = io.read_targets_in_tiles(ledgerdir,
                                     tiles=alltileinfo, mtl=True, unique=False)
    # ADM ...and restrict to just the MTL tiles of interest.
    tileset = set(tiles["TILEID"])
    ii = np.array([tile in tileset for tile in targs["ZTILEID"]])
    log.info("Read {} MTL-processed targets in {} tiles...t={:.1f}s".format(
        np.sum(ii), len(alltileinfo), time()-t0))
    targs = targs[ii]

    # ADM now we have the targets and tiles, loop through the tiles in
    # ADM reverse chronological order, store the targets, and flag any
    # ADM tiles that have some targets that are also on a later tile.
    nooverlap = []  # ADM to hold tiles that do NOT overlap a later tile.
    # ADM this holds TARGETIDs for targets that were observed "later".
    aftertargs = list(obstargs["TARGETID"])
    timestore = "9999-01-01T00:00:00+00:00"
    for ntile, tile in enumerate(tiles):
        # ADM just to log progress.
        if ntile % (len(tiles)//5+1) == 0 and ntile > 0:
            elapsed = time() - t0
            rate = elapsed / ntile
            msg = "{}/{} tiles checked for overlaps ".format(ntile, len(tiles))
            log.info(msg + "{:.1f} secs/tile; {:.1f} total mins elapsed".format(
                rate, elapsed/60.))
        # ADM check the chronology is, indeed, reversed.
        if tile["TIMESTAMP"] > timestore:
            msg = "Tiles not in reverse chronological order!!!"
            log.critical(msg)
            raise ValueError(msg)
        timestore = tile["TIMESTAMP"]

        # ADM look up the full tile info for this tile.
        ii = alltileinfo["TILEID"] == tile["TILEID"]
        # ADM read in potential targets touched by this tile.
        intile = is_point_in_desi(alltileinfo[ii], targs["RA"], targs["DEC"])
        # ADM restrict to just unique TARGETIDs (as a potential target
        # ADM could have been observed on multiple tiles).
        targsintile = list(set(targs[intile]["TARGETID"]))
        # ADM match to check if targets were observed on a later tile.
        tii, aii = match(targsintile, aftertargs)
        # ADM if there IS a match, this is an overlapping tile.
        if len(tii) == 0:
            nooverlap.append(tile["TILEID"])
        # ADM append all of the TARGETIDs to the "observed later" list.
        aftertargs += targsintile
        # ADM only retain unique TARGETIDs, for later matching.
        aftertargs = list(set(aftertargs))

    # ADM return an informative dictionary if `check` was passed.
    # ADM this might be slow for large numbers of tiles.
    if check:
        checkdict = {}
        # ADM we'll need to work with all of the tiles, again.
        tiles = Table.read(mtltilefn)
        log.info("Making checker dictionary...")
        # ADM loop through the tiles that have no "future" overlaps.
        for ntile, tile in enumerate(nooverlap):
            # ADM just to log progress.
            if ntile % (len(nooverlap)//5+1) == 0 and ntile > 0:
                elapsed = time() - t0
                rate = elapsed / ntile
                msg = "Made for {}/{} ".format(ntile, len(nooverlap))
                log.info(msg + "{:.1f} secs/tile; {:.1f} mins elapsed".format(
                    rate, elapsed/60.))
            # ADM read in the targets in this no-future-overlaps tile.
            ii = alltileinfo["TILEID"] == tile
            targs = io.read_targets_in_tiles(ledgerdir, tiles=alltileinfo[ii],
                                             mtl=True, unique=False)
            # ADM find all of the past tiles touched by the targets.
            tilespast = list(set(targs["ZTILEID"]) - {-1})
            # ADM match back to the MTL tiles to retrieve the TIMESTAMP.
            ii = match_to(tiles["TILEID"], tilespast)
            # ADM the MTL tile file is oredered chronologically, so if we
            # ADM sort on index we'll recover the TIMESTAMPs in order.
            ii = sorted(ii)
            checkdict[tile] = {k: v for k, v in
                               zip(tiles[ii]["TILEID"], tiles[ii]["TIMESTAMP"])}
        return checkdict

    # ADM return the non-overlapping tiles.
    ii = match_to(alltileinfo["TILEID"], sorted(nooverlap))
    return alltileinfo[ii]


def purge_tiles(tiles, obscon, mtldir=None, secondary=False, verbose=True):
    """
    Utterly remove tiles from MTL ledgers and associated mtl-done files.

    Parameters
    ----------
    tiles : :class:`~astropy.table.Table`
        A Table of tiles (in the standard DESI format) to be removed from
        the MTL ledgers and associated mtl-done files.
    obscon : :class:`str`
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "DARK"
        Used to construct the directory to find the Main Survey ledgers.
    mtldir : :class:`str`, optional, defaults to ``None``
        Full path to the directory that hosts the MTL ledgers and the MTL
        tile file. If ``None``, then look up the MTL directory from the
        $MTL_DIR environment variable.
    secondary : :class:`bool`, optional, defaults to ``False``
        If ``True`` then purge secondary targets instead of primaries for
        passed `obscon`.
    verbose : :class:`bool`, optional, defaults to ``True``
        If ``True`` then log extra information.

    Returns
    -------
    :class:`~astropy.table.Table`
        A Table of all of the entries removed from the ledgers.
    :class:`~astropy.table.Table`
        A Table of all of the entries removed from an mtl-done file.

    Notes
    -----
    - Requires the mtl part of the operations SVN trunk to be checked out
      in the standard location relative to `mtldir`.
    - Only works for Main Survey tiles, as we only look in $MTL_DIR/main.
    """
    t0 = time()
    # ADM a quick check that the observing conditions match the program.
    if not np.all(tiles["PROGRAM"] == obscon.upper()):
        msg = "PROGRAM is {} but passed obscon is {}!!!".format(
            set(tiles["PROGRAM"]), obscon)
        log.error(msg)
        raise RuntimeError(msg)

    # ADM grab the MTL directory (in case we're relying on $MTL_DIR).
    mtldir = get_mtl_dir(mtldir)
    # ADM construct the full path to the mtl tile file.
    mtltilefn = os.path.join(mtldir, get_mtl_tile_file_name(secondary=secondary))

    # ADM construct the full path to the ledger directory.
    resolve = True
    msg = "running on {} ledger with obscon={} (and survey={})"
    if secondary:
        log.info(msg.format("SECONDARY", obscon, "main"))
        resolve = None
    else:
        log.info(msg.format("PRIMARY", obscon, "main"))
    ledgerdir = io.find_target_files(mtldir, flavor="mtl", resolve=resolve,
                                     obscon=obscon)

    # ADM the filename format and the HEALPixel nside for the ledgers.
    fileform = io.find_mtl_file_format_from_header(ledgerdir)
    nside = io.read_keyword_from_mtl_header(ledgerdir, "FILENSID")

    # ADM generate the ledger names from the input tiles.
    pixlist = tiles2pix(nside, tiles=tiles)

    # ADM first, remove the targets from the ledger files.
    # ADM to store the removed targets.
    gonetargs = []
    # ADM a set of all of the to-be-removed TILEIDs.
    s = set(tiles["TILEID"])
    # ADM read in each ledger and remove the relevant tiles.
    for pix in pixlist:
        # ADM read the header and the data.
        fn = fileform.format(pix)
        hdr = io.read_ecsv_header(fn, cleanup=True)
        targs = io.read_mtl_ledger(fn, unique=False)
        # ADM remove any targets on to-be-purged tiles...
        iibad = np.array([tileid in s for tileid in targs["ZTILEID"]])
        # ADM ...but retain these targets to return.
        gonetargs.append(targs[iibad])
        # ADM write the targets on the not-to-be-purged tiles.
        io.write_with_units(fn, targs[~iibad],
                            extname='MTL', header=hdr, ecsv=True)
        if verbose:
            msg = "Removed {} targets and retained {} targets in {}".format(
                np.sum(iibad), np.sum(~iibad), fn)
            log.info(msg)

    # ADM second, remove the tiles from the done file.
    # ADM to store the removed tiles.
    inmtltiles = io.read_mtl_tile_file(mtltilefn, unique=False)
    # ADM guarantee that the output Table will be in the required format.
    mtltiles = np.empty_like(mtltilefiledm, shape=len(inmtltiles))
    for col in mtltilefiledm.dtype.names:
        mtltiles[col] = inmtltiles[col]
    mtltiles = Table(mtltiles)
    # ADM now find and purge the actual tiles.
    iibad = np.array([tileid in s for tileid in mtltiles["TILEID"]])
    gonetiles = mtltiles[iibad]
    io.write_with_units(mtltilefn, mtltiles[~iibad], extname='MTLTILE',
                        ecsv=True)

    return Table(np.concatenate(gonetargs)), gonetiles


def make_ledger_in_hp(targets, outdirname, nside, pixlist, obscon="DARK",
                      indirname=None, verbose=True, scnd=False,
                      timestamp=None, append=False):
    """
    Make an initial MTL ledger file for targets in a set of HEALPixels.

    Parameters
    ----------
    targets : :class:`~numpy.array`
        Targets made by, e.g. `desitarget.cuts.select_targets()`.
    outdirname : :class:`str`
        Output directory to which to write the MTLs (the file names are
        constructed on the fly).
    nside : :class:`int`
        (NESTED) HEALPixel nside that corresponds to `pixlist`.
    pixlist : :class:`list` or `int`
        HEALPixels at `nside` at which to write the MTLs.
    obscon : :class:`str`, optional, defaults to "DARK"
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "GRAY"
        Governs how priorities are set when merging targets. Also governs
        the sub-directory to which the ledger is written.
    indirname : :class:`str`
        A directory associated with the targets. Written to the headers
        of the output MTL files.
    verbose : :class:`bool`, optional, defaults to ``True``
        If ``True`` then log target and file information.
    scnd : :class:`bool`, defaults to ``False``
        If ``True`` then this is a ledger of secondary targets.
    timestamp : :class:`str`, optional
        A timestamp to use in place of that assigned by `make_mtl`.
    append : :class:`bool`, optional, defaults to ``False``
        If ``True`` then append to any existing ledgers rather than
        creating new ones. In this mode, if a ledger exists it will be
        appended to and if it doesn't exist it will be created.

    Returns
    -------
    Nothing, but writes the `targets` out to `outdirname` split across
    each HEALPixel in `pixlist`.
    """
    t0 = time()
    # ADM a dictionary to hold header keywords for the ouput file.
    hdr = {}

    # ADM in case an integer was passed.
    pixlist = np.atleast_1d(pixlist)

    # ADM execute MTL.
    mtl = make_mtl(targets, obscon, trimcols=True)

    # ADM if requested, substitute a bespoke timestamp.
    if timestamp is not None:
        # ADM check the timestamp is valid.
        _ = check_timestamp(timestamp)
        hdr["TSFORCED"] = timestamp
        mtl["TIMESTAMP"] = timestamp

    # ADM the HEALPixel within which each target in the MTL lies.
    theta, phi = np.radians(90-mtl["DEC"]), np.radians(mtl["RA"])
    mtlpix = hp.ang2pix(nside, theta, phi, nest=True)

    # ADM write the MTLs.
    _, _, survey = main_cmx_or_sv(mtl)
    for pix in pixlist:
        inpix = mtlpix == pix
        ecsv = get_mtl_ledger_format() == "ecsv"
        if np.any(inpix):
            nt, fn = io.write_mtl(
                outdirname, mtl[inpix].as_array(), indir=indirname, ecsv=ecsv,
                survey=survey, obscon=obscon, nsidefile=nside, hpxlist=pix,
                scnd=scnd, extra=hdr, append=append)
            if verbose:
                writ = int(append)*"appended" + int(not(append))*"written"
                log.info('{} targets {} to {}...t={:.1f}s'.format(
                    nt, writ, fn, time()-t0))

    return


def make_ledger(hpdirname, outdirname, pixlist=None, obscon="DARK",
                numproc=1, timestamp=None, append=False):
    """
    Make initial MTL ledger files for HEALPixels, in parallel.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to either a directory containing targets that
        have been partitioned by HEALPixel (i.e. as made by
        `select_targets` with the `bundle_files` option). Or the
        name of a single file of targets.
    outdirname : :class:`str`
        Output directory to which to write the MTL (the file name is
        constructed on the fly).
    pixlist : :class:`list` or `int`, defaults to ``None``
        (Nested) HEALPixels for which to write the MTLs at the default
        `nside` (which is `_get_mtl_nside()`). Defaults to ``None``,
        which runs all of the pixels at `_get_mtl_nside()`.
    obscon : :class:`str`, optional, defaults to "DARK"
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "GRAY"
        Governs how priorities are set based on "obsconditions". Also
        governs the sub-directory to which the ledger is written.
    numproc : :class:`int`, optional, defaults to 1 for serial
        Number of processes to parallelize across.
    timestamp : :class:`str`, optional
        A timestamp to use in place of that assigned by `make_mtl`.
    append : :class:`bool`, optional, defaults to ``False``
        If ``True`` then append to any existing ledgers rather than
        creating new ones. In this mode, if a ledger exists it will be
        appended to and if it doesn't exist it will be created.

    Returns
    -------
    Nothing, but writes the full HEALPixel-split ledger to `outdirname`.

    Notes
    -----
    - For _get_mtl_nside()=32, takes about 25 minutes with `numproc=12`.
      `numproc>12` can run into memory issues.
    - For _get_mtl_nside()=16, takes about 50 minutes with `numproc=8`.
      `numproc>8` can run into memory issues.
    """
    # ADM grab information regarding how the targets were constructed.
    hdr, dt = io.read_targets_header(hpdirname, dtype=True)
    # ADM check the obscon for which the targets were made is
    # ADM consistent with the requested obscon.
    oc = hdr["OBSCON"]
    if obscon not in oc:
        msg = "File is type {} but requested behavior is {}".format(oc, obscon)
        log.critical(msg)
        raise ValueError(msg)

    # ADM check whether this is a file of standalone secondary targets.
    scnd = False
    if hdr["EXTNAME"] == 'SCND_TARGETS':
        scnd = True

    # ADM the MTL datamodel must reflect the target flavor (SV, etc.)...
    mtldm = switch_main_cmx_or_sv(mtldatamodel, np.array([], dt))
    # ADM ...and the data model can differ with survey type..
    _, _, survey = main_cmx_or_sv(mtldm)
    mtldm = survey_data_model(mtldm, survey=survey)

    # ADM speed-up by only reading the necessary columns.
    cols = list(set(mtldm.dtype.names).intersection(dt.names))

    # ADM optimal nside for reading in the targeting files.
    if "FILENSID" in hdr:
        nside = hdr["FILENSID"]
    else:
        # ADM if a file was passed instead of a HEALPixel-split directory
        # ADM use nside=4 as 196 pixels often balances well across CPUs.
        nside = 4
    npixels = hp.nside2npix(nside)
    pixels = np.arange(npixels)

    # ADM the nside at which to write the MTLs.
    mtlnside = _get_mtl_nside()
    # ADM default to running all pixels.
    if pixlist is None:
        mtlnpixels = hp.nside2npix(mtlnside)
        pixlist = np.arange(mtlnpixels)

    # ADM check that the nside for writing MTLs is not at a lower
    # ADM resolution than the nside at which the files are stored.
    msg = "Ledger nside ({}) must be higher than file nside ({})!!!".format(
        mtlnside, nside)
    assert mtlnside >= nside, msg

    from desitarget.geomask import nside2nside

    # ADM the common function that is actually parallelized across.
    def _make_ledger_in_hp(pixnum):
        """make initial ledger in a single HEALPixel"""
        # ADM construct a list of all pixels in pixnum at the MTL nside.
        setpix = set(nside2nside(nside, mtlnside, pixnum))
        pix = [p for p in pixlist if p in setpix]
        if len(pix) == 0:
            return
        # ADM read in the needed columns from the targets.
        targs = io.read_targets_in_hp(hpdirname, nside, pixnum, columns=cols)
        if len(targs) == 0:
            return
        # ADM the secondary targeting files don't include the BGS_TARGET
        # ADM and MWS_TARGET columns, which are needed for MTL.
        neededcols, _, _ = main_cmx_or_sv(targs)
        misscols = set(neededcols) - set(cols)
        if len(misscols) > 0:
            # ADM the data type for the DESI_TARGET column.
            dtdt = mtldatamodel["DESI_TARGET"].dtype
            zerod = [np.zeros(len(targs)), np.zeros(len(targs))]
            targs = rfn.append_fields(
                targs, misscols, data=zerod, dtypes=[dtdt, dtdt], usemask=False)
        # ADM write MTLs for the targs split over HEALPixels in pixlist.
        return make_ledger_in_hp(
            targs, outdirname, mtlnside, pix, obscon=obscon,
            indirname=hpdirname, verbose=False, scnd=scnd,
            timestamp=timestamp, append=append)

    # ADM this is just to count pixels in _update_status.
    npix = np.ones((), dtype='i8')
    t0 = time()
    # ADM this is just to log whether we're in write or append mode.
    writ = int(append)*"Appending" + int(not(append))*"Writing"

    def _update_status(result):
        """wrap key reduction operation on the main parallel process"""
        if npix % 2 == 0 and npix > 0:
            rate = (time() - t0) / npix
            log.info('{} {}/{} HEALPixels; {:.1f} secs/pixel...t = {:.1f} mins'.
                     format(writ, npix, npixels, rate, (time()-t0)/60.))
        npix[...] += 1
        return result

    # ADM Parallel process across HEALPixels.
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            pool.map(_make_ledger_in_hp, pixels, reduce=_update_status)
    else:
        for pixel in pixels:
            _update_status(_make_ledger_in_hp(pixel))

    log.info("Done writing ledger to {}...t = {:.1f} mins".format(
        outdirname, (time()-t0)/60.))

    return


def standard_override_columns(mtl):
    """
    Add some standard column entries to an mtl Table.

    Parameters
    ----------
    mtl : :class:`~astropy.table.Table`
        An astropy Table. Must contain the columns TIMESTAMP,
        TARGET_STATE, VERSION and ZTILEID.

    Returns
    -------
    :class:`~astropy.table.Table`
        The input table with IMESTAMP updated to now, the second part of
        TARGET_STATE updated to be OVERRIDE, the git VERSION updated, and
        ZTILEID set to -1.
    """
    mtl["TIMESTAMP"] = get_utc_date(survey="main")
    newts = ["{}|OVERRIDE".format(t.split("|")[0]) for t in mtl["TARGET_STATE"]]
    mtl["TARGET_STATE"] = np.array(newts)
    mtl["VERSION"] = dt_version
    mtl["ZTILEID"] = -1

    return mtl


def process_overrides(ledgerfn):
    """
    Recover MTL entries from override ledgers and update those ledgers.

    Parameters
    ----------
    ledgerfn : :class:`str`
        Name of the override ledger to process.

    Returns
    -------
    :class:`~astropy.table.Table`
        MTL entries from override ledger with NUMOVERRIDE column removed,
        TIMESTAMP updated to now, the second part of TARGET_STATE
        updated to be OVERRIDE, and the git VERSION updated.

    Notes
    -----
    - Rewrites entries to the override ledger with NUMOVERRIDE updated to
      be NUMOVERRIDE - 1, TIMESTAMP updated to now, the second part of
      TARGET_STATE updated to OVERRIDE, the git VERSION updated, and the
      ZTILEID updated to -1.
    """
    log.info("Processing override ledgers")

    # ADM read in the relevant entries in the override ledger.
    mtl = Table(io.read_mtl_ledger(ledgerfn))

    # ADM indicate that we've already overrode once.
    mtl["NUMOVERRIDE"] -= 1

    # ADM update the standard information for override ledgers.
    mtl = standard_override_columns(mtl)

    # ADM append the updated mtl entry to the override ledger.
    f = open(ledgerfn, "a")
    ascii.write(mtl, f, format='no_header', formats=mtlformatdict)
    f.close()

    # ADM return the entry without the NUMOVERRIDE column.
    del mtl["NUMOVERRIDE"]
    return mtl


def ledger_overrides(overfn, obscon, colsub=None, valsub=None,
                     mtldir=None, secondary=False, numoverride=999):
    """
    Create (or append to) a ledger to override the standard ledgers.

    Parameters
    ----------
    overfn : :class:`str`
        Full path to the filename that contains the override information.
        Must contain at least `RA`, `DEC`, `TARGETID` and be in a format
        that can be read automatically by astropy.table.Table.read().
    obscon : :class:`str`
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "DARK"
        Used to construct the directory to find the Main Survey ledgers.
    colsub : :class:`dict`, optional
        If passed, each key should correspond to the name of a ledger
        column and each value should correspond to the name of a column
        in `overfn`. The ledger columns are overwritten by the
        corresponding column in `overfn` (for the appropriate TARGETID).
    valsub : :class:`dict`, optional
        If passed, each key should correspond to the name of a ledger
        column and each value to a single number or string. The "value"
        will be overwritten into the "key" column of the ledger. Takes
        precedence over colsub.
    mtldir : :class:`str`, optional, defaults to ``None``
        Full path to the directory that hosts the MTL ledgers and the MTL
        tile file. If ``None``, then look up the MTL directory from the
        $MTL_DIR environment variable.
    secondary : :class:`bool`, optional, defaults to ``False``
        If ``True`` then process secondary targets instead of primaries
        for passed `obscon`.
    numoverride : :class:`int`, optional, defaults to 999
        The override ledger is read every time the MTL loop is run. This
        is the number of times to override the standard results in the
        MTL loop. Defaults to 999, i.e. essentially "always override."

    Returns
    -------
    :class:`str`
        The directory containing the ledgers that were updated.

    Notes
    -----
    - Regardless of the inputs, the TIMESTAMP in the output override
      ledger is always updated to now, the second part of TARGET_STATE is
      always updated to OVERRIDE, the git VERSION is always updated and
      the ZTILEID is always set to -1.
    """
    # ADM grab the MTL directory (in case we're relying on $MTL_DIR).
    mtldir = get_mtl_dir(mtldir)

    # ADM construct the relevant sub-directory for this survey and
    # ADM set of observing conditions.
    resolve = True
    msg = "running on {} ledgers with obscon={} and survey=main"
    if secondary:
        log.info(msg.format("SECONDARY", obscon))
        resolve = None
    else:
        log.info(msg.format("PRIMARY", obscon))
    outdir = io.find_target_files(mtldir, flavor="mtl", obscon=obscon,
                                  survey="main", resolve=resolve, override=True)
    # ADM and create the sub-directory if it doesn't exist.
    os.makedirs(outdir, exist_ok=True)

    # ADM read in the file with override information.
    objs = Table.read(overfn)
    # ADM be somewhat forgiving and convert lower-case columns.
    for colname in ["RA", "DEC", "TARGETID"]:
        try:
            objs.rename_column(colname.lower(), colname)
        except KeyError:
            pass
        # ADM check all required columns are present.
        if colname not in objs.colnames:
            msg = "{} must be a column in {}!".format(colname, overfn)
            log.error(msg)
            raise IOError(msg)

    # ADM retrieve the current ledger entry for each target to be overrode.
    nside = _get_mtl_nside()
    theta, phi = np.radians(90-objs["DEC"]), np.radians(objs["RA"])
    pixnum = hp.ang2pix(nside, theta, phi, nest=True)
    for tid, pix in zip(objs["TARGETID"], pixnum):
        # ADM construct the input/output ledger filenames.
        infn = io.find_target_files(mtldir, flavor="mtl", survey="main", hp=pix,
                                    resolve=resolve, obscon=obscon, ender="ecsv")
        outfn = io.find_target_files(
            mtldir, flavor="mtl", survey="main", hp=pix, resolve=resolve,
            obscon=obscon, override=True, ender="ecsv")

        ledger = Table(io.read_mtl_ledger(infn))
        ii = ledger["TARGETID"] == tid
        # ADM warn if one of the TARGETIDs seems incorrect.
        if not np.any(ii):
            msg = "TARGETID {} from {} not in file {}!".format(tid, overfn, infn)
            log.warning(msg)
        entry = ledger[ii]

        # ADM substitute the column entries, where requested.
        ii = objs["TARGETID"] == tid
        if colsub is not None:
            for col in colsub:
                if col not in ledger.colnames:
                    msg = "column {} from colsub not in ledger!!!".format(col)
                    log.error(msg)
                    raise ValueError(msg)
                entry[col] = objs[ii][colsub[col]]
        # ADM overwrite ledger entries, where requested.
        if valsub is not None:
            for col, val in valsub.items():
                if col not in ledger.colnames:
                    msg = "column {} from valsub not in ledger!!!".format(col)
                    log.error(msg)
                    raise ValueError(msg)
                entry[col] = val

        # ADM add the number of times to override to the ledger entry.
        entry["NUMOVERRIDE"] = numoverride
        # ADM add some other standardized column information.
        entry = standard_override_columns(entry)

        # ADM finally write out the override ledger entry, after first
        # ADM checking if the file exists.
        if os.path.exists(outfn):
            f = open(outfn, "a")
            ascii.write(entry, f, format='no_header', formats=mtlformatdict)
            f.close()
            checkfn = outfn
        else:
            _, checkfn = io.write_mtl(
                mtldir, entry.as_array(), indir=infn, ecsv=True, survey="main",
                obscon=obscon, nsidefile=nside, hpxlist=pix, scnd=secondary,
                override=True)
        log.info('Wrote target {} from {} to {}'.format(tid, overfn, checkfn))

    log.info("Touched pixels {}".format(",".join([str(pix) for pix in pixnum])))

    return outdir


def force_overrides(hpdirname, pixlist):
    """
    Force override ledgers to be processed and added to the MTL ledgers.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to a directory containing an MTL ledger that has been
        partitioned by HEALPixel (i.e. as made by `make_ledger`).
    pixlist : :class:`list`
        A list of HEALPixels corresponding to the ledgers to be updated.

    Returns
    -------
    :class:`str`
        The directory containing the ledgers that were updated.
    """
    # ADM find the general format for the ledger files in `hpdirname`.
    fileform = io.find_mtl_file_format_from_header(hpdirname)
    # ADM this is the format for any associated override ledgers.
    overrideff = io.find_mtl_file_format_from_header(hpdirname, override=True)

    # ADM before making updates, check all suggested ledgers exist.
    for pix in pixlist:
        overfn = overrideff.format(pix)
        fn = fileform.format(pix)
        for f in overfn, fn:
            if not os.path.exists(f):
                msg = "no ledger exists at: {}".format(f)
                log.error(msg)
                raise OSError

    for pix in pixlist:
        # ADM the correct filenames for this pixel number.
        fn = fileform.format(pix)
        overfn = overrideff.format(pix)

        # ADM update override ledger and recover relevant MTL entries.
        overmtl = process_overrides(overfn)

        # ADM append override entries to the ledger.
        f = open(fn, "a")
        ascii.write(overmtl, f, format='no_header', formats=mtlformatdict)
        f.close()

    return hpdirname


def remove_overrides(mtl):
    """Remove all targets that share a TARGETID with an OVERRIDE target

    Parameters
    ----------
    mtl : :class:`~numpy.array` or `~astropy.table.Table`
        An array of targets from a Merged Target List. Must contain at
        least the column TARGET_STATE. Can contain duplicate TARGETIDs.

    Returns
    -------
    :class:`~numpy.array` or `~astropy.table.Table`
        The original `mtl` but all targets that have a TARGETID for which
        one entry has the string "OVERRIDE" in TARGET_STATE are removed.
    """
    # ADM find OVERRIDE entries.
    ii = np.array(["OVERRIDE" in ts for ts in mtl["TARGET_STATE"]])

    # ADM find the set of TARGETIDs that have an OVERRIDE entry.
    s = set(mtl[ii]["TARGETID"])

    # ADM find the entries that do NOT have an OVERRIDE TARGETID.
    ii = np.array([tid not in s for tid in mtl["TARGETID"]])

    log.info("removing {} override entries".format(len(mtl)-np.sum(ii)))

    # ADM return the input mtl without the OVERRIDE TARGETIDs.
    return mtl[ii]


def reprocess_ledger(hpdirname, zcat, obscon="DARK"):
    """
    Reprocess HEALPixel-split ledgers for targets with new redshifts.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to a directory containing an MTL ledger that has been
        partitioned by HEALPixel (i.e. as made by `make_ledger`).
    zcat : :class:`~astropy.table.Table`, optional
        Redshift catalog table with columns ``TARGETID``, ``NUMOBS``,
        ``Z``, ``ZWARN``, ``ZTILEID``, and ``msaddcols`` at the top of
        the code for the Main Survey.
    obscon : :class:`str`, optional, defaults to "DARK"
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "DARK"
        Governs how priorities are set using "obsconditions". Basically a
        check on whether the files in `hpdirname` are as expected.

    Returns
    -------
    Nothing, but relevant ledger files are updated.
    """
    # ADM find the general format for the ledger files in `hpdirname`.
    # ADM also returning the obsconditions.
    fileform, oc = io.find_mtl_file_format_from_header(hpdirname, returnoc=True)
    # ADM this is the format for any associated override ledgers.
    overrideff = io.find_mtl_file_format_from_header(hpdirname, override=True)

    # ADM check the obscondition is as expected.
    if obscon != oc:
        msg = "File is type {} but requested behavior is {}".format(oc, obscon)
        log.critical(msg)
        raise RuntimeError(msg)

    # ADM read targets from the relevant ledgers.
    nside = _get_mtl_nside()
    theta, phi = np.radians(90-zcat["DEC"]), np.radians(zcat["RA"])
    pixnum = hp.ang2pix(nside, theta, phi, nest=True)
    pixnum = list(set(pixnum))

    # ADM first, gather unique targets to find any unobserved targets.
    # ADM we'll read in too many targets, here, but that's OK as
    # ADM we'll soon match to just the relevant targets.
    targets = io.read_mtl_in_hp(hpdirname, nside, pixnum, unique=True)
    # ADM remove OVERRIDE entries, which should never need reprocessed.
    targets = remove_overrides(targets)

    # ADM match the zcat to the targets and restrict to just the
    # ADM relevant targets and zcat entries.
    tii, zii = match(targets["TARGETID"], zcat["TARGETID"])
    msg = "Found {}/{} targets from zcat in ledgers".format(len(zii), len(zcat))
    msg += " (remainder likely primaries when running secondaries or vice versa)"
    log.info(msg)
    targets, zcat = targets[tii], zcat[zii]

    # ADM run through various cases. FIRST, as-yet-unobserved targets.
    log.info("Reprocessing as-yet-unobserved targets")
    isunobs = targets["ZTILEID"] == -1
    unobs = targets[isunobs]
    # ADM a quick check for strangeness.
    kewl = np.array(["UNOBS" in state or "CALIB" in state
                     for state in unobs["TARGET_STATE"]])
    if not np.all(kewl):
        msg = "Some targets have ZTILEID of -1 but STATE isn't UNOBS or CALIB:"
        log.critical(msg)
        log.info(unobs[~kewl])
        raise RuntimeError

    # ADM run MTL as usual for the unobserved targets.
    mtl = make_mtl(unobs, oc, zcat=zcat[isunobs], trimtozcat=True, trimcols=True)

    # ADM now we're done with unobserved targets, read all observations
    # ADM of all targets (unique=False) and work tile-by-tile.
    targets = io.read_mtl_in_hp(hpdirname, nside, pixnum, unique=False)
    # ADM remove OVERRIDE entries, which should never need reprocessed.
    targets = remove_overrides(targets)
    # ADM sort by TIMESTAMP to ensure tiles are listed chronologically.
    targets = targets[np.argsort(targets["TIMESTAMP"])]

    # ADM need to know which targets were first observed on which tiles.
    obs = targets[targets["ZTILEID"] != -1]
    _, ii = np.unique(obs["TARGETID"], return_index=True)
    first = obs[ii]
    # ADM a few quick sanity checks.
    # ADM all of the first observations indeed had one observation.
    assert np.all(first["NUMOBS"] == 1)
    # ADM all of the first-observation targets are in first.
    assert np.sum(targets["NUMOBS"] == 1) == len(first)
    # ADM all TARGETIDs in the set of first observations are unique.
    assert len(set(first["TARGETID"])) == len(first)

    # ADM to hold the final list of updates:
    donemtl = [mtl]
    # ADM work separately on each tile one-by-one.
    for tileid in set(zcat["ZTILEID"]):
        ztile = zcat[zcat["ZTILEID"] == tileid]
        ttile = targets[targets["ZTILEID"] == tileid]

        # ADM find the most recent observations on this tile...
        tflipped = np.flip(ttile)
        _, ii = np.unique(tflipped["TARGETID"], return_index=True)
        ttile = tflipped[ii]

        # ADM match the new and the most recent MTL observations.
        tii, zii = match(ttile["TARGETID"], ztile["TARGETID"])
        msg = "{}/{} recently observed targets from zcat on tile {}".format(
            len(zii), len(ztile), tileid)
        log.info(msg)
        ttile, ztile = ttile[tii], ztile[zii]

        # ADM find previous good and bad observations based on zwarn.
        Mxbad = "BAD_SPECQA|BAD_PETALQA|NODATA"
        tbad = ttile["ZWARN"] & zwarn_mask.mask(Mxbad) != 0
        zbad = ztile["ZWARN"] & zwarn_mask.mask(Mxbad) != 0
        tgood, zgood = ~tbad, ~zbad

        # ADM check for "first" observations on this tile.
        fii, zii = match(first["TARGETID"], ztile["TARGETID"])
        firstseen = np.zeros(len(ztile), dtype="?")
        firstseen[zii] = True

        # ADM recover the initial ("UNOBS") mtl for this tile.
        initmtl = make_mtl(ttile, oc, trimcols=True)
        # ADM also recover the once-through-mtl updated state.
        firstmtl = make_mtl(initmtl, oc, zcat=ztile,
                            trimtozcat=True, trimcols=True)

        # ADM there are 4 possible cases, but there's nothing to do when
        # ADM the new and old processing were bad & bad or good & good
        # ADM except update the z information for targets on the tile.
        reprocmtl = np.zeros(len(ttile), dtype=mtl.dtype)
        for col in reprocmtl.dtype.names:
            reprocmtl[col] = ttile[col]
        for col in ["Z", "ZWARN"] + list(msaddcols.dtype.names):
            reprocmtl[col] = ztile[col]

        # ADM CASE: Most recent processing was bad, new one is good.
        case = tbad & zgood
        # ADM generically, just acknowledge we have a good observation.
        reprocmtl[case]["NUMOBS"] += 1
        reprocmtl[case]["NUMOBS_MORE"] -= 1
        # ADM if the first observation was on this TILE, update to the
        # ADM first-observation-after-the-initial state.
        firstcase = case & firstseen
        for col in reprocmtl.dtype.names:
            reprocmtl[col][firstcase] = firstmtl[col][firstcase]

        # ADM CASE: Most recent processing was good, new one is bad.
        case = tgood & zbad
        # ADM generically, just acknowledge we have a bad observation.
        reprocmtl[case]["NUMOBS"] -= 1
        reprocmtl[case]["NUMOBS_MORE"] += 1
        # ADM if the first observation was on this TILE, update to the
        # ADM unobserved state.
        firstcase = case & firstseen
        for col in reprocmtl.dtype.names:
            reprocmtl[col][firstcase] = initmtl[col][firstcase]

        # ADM append this reprocessing to the MTL updates.
        donemtl.append(reprocmtl)

    # ADM collect the results.
    mtl = np.concatenate(donemtl)

    # ADM update the TIMESTAMP to now for all cases.
    mtl["TIMESTAMP"] = get_utc_date(survey="main")

    # ADM re-collect everything on pixels for writing to ledgers.
    nside = _get_mtl_nside()
    theta, phi = np.radians(90-mtl["DEC"]), np.radians(mtl["RA"])
    pixnum = hp.ang2pix(nside, theta, phi, nest=True)

    # ADM loop through the pixels and update the ledger, depending
    # ADM on whether we're working with .fits or .ecsv files.
    ender = get_mtl_ledger_format()
    for pix in set(pixnum):
        # ADM grab the targets in the pixel.
        ii = pixnum == pix
        mtlpix = mtl[ii]

        # ADM sorting on TARGETID is neater (although not strictly
        # ADM necessary when using io.read_mtl_ledger(unique=True)).
        mtlpix = mtlpix[np.argsort(mtlpix["TARGETID"])]

        # ADM the correct filenames for this pixel number.
        fn = fileform.format(pix)
        overfn = overrideff.format(pix)

        # ADM if an override ledger exists, update it and recover its
        # ADM relevant MTL entries.
        if os.path.exists(overfn):
            overmtl = process_overrides(overfn)
            # ADM add any override entries TO THE END OF THE LEDGER.
            mtlpix = vstack([mtlpix, overmtl])

        # ADM if we're working with .ecsv, simply append to the ledger.
        if ender == 'ecsv':
            f = open(fn, "a")
            ascii.write(mtlpix, f, format='no_header', formats=mtlformatdict)
            f.close()
        # ADM otherwise, for FITS, we'll have to read in the whole file.
        else:
            ledger, hd = fitsio.read(fn, extname="MTL", header=True)
            done = np.concatenate([ledger, mtlpix.as_array()])
            fitsio.write(fn+'.tmp', done, extname='MTL', header=hd, clobber=True)
            os.rename(fn+'.tmp', fn)

    return


def update_ledger(hpdirname, zcat, targets=None, obscon="DARK",
                  numobs_from_ledger=False):
    """
    Update relevant HEALPixel-split ledger files for some targets.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to a directory containing an MTL ledger that has been
        partitioned by HEALPixel (i.e. as made by `make_ledger`).
    zcat : :class:`~astropy.table.Table`, optional
        Redshift catalog table with columns ``TARGETID``, ``NUMOBS``,
        ``Z``, ``ZWARN``, ``ZTILEID``, and ``msaddcols`` at the top of
        the code for the Main Survey.
    targets : :class:`~numpy.array` or `~astropy.table.Table`, optional
        A numpy rec array or astropy Table with at least the columns
        ``RA``, ``DEC``, ``TARGETID``, ``DESI_TARGET``, ``NUMOBS_INIT``,
        and ``PRIORITY_INIT``. If ``None``, then assume the `zcat`
        includes ``RA`` and ``DEC`` and look up `targets` in the ledger.
    obscon : :class:`str`, optional, defaults to "DARK"
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "DARK"
        Governs how priorities are set using "obsconditions". Basically a
        check on whether the files in `hpdirname` are as expected.
    numobs_from_ledger : :class:`bool`, optional, defaults to ``True``
        If ``True`` then inherit the number of observations so far from
        the ledger rather than expecting it to have a reasonable value
        in the `zcat.`

    Returns
    -------
    Nothing, but relevant ledger files are updated.
    """
    # ADM find the general format for the ledger files in `hpdirname`.
    # ADM also returning the obsconditions.
    fileform, oc = io.find_mtl_file_format_from_header(hpdirname, returnoc=True)
    # ADM this is the format for any associated override ledgers.
    overrideff = io.find_mtl_file_format_from_header(hpdirname, override=True)

    # ADM check the obscondition is as expected.
    if obscon != oc:
        msg = "File is type {} but requested behavior is {}".format(oc, obscon)
        log.critical(msg)
        raise RuntimeError(msg)

    # ADM if targets wasn't sent, that means the zcat includes
    # ADM coordinates and we can read relevant targets from the ledger.
    if targets is None:
        nside = _get_mtl_nside()
        theta, phi = np.radians(90-zcat["DEC"]), np.radians(zcat["RA"])
        pixnum = hp.ang2pix(nside, theta, phi, nest=True)
        pixnum = list(set(pixnum))
        # ADM we'll read in too many targets, here, but that's OK as
        # ADM make_mtl(trimtozcat=True) only returns the updated targets.
        targets = io.read_mtl_in_hp(hpdirname, nside, pixnum, unique=True)

    # ADM if requested, use the previous values in the ledger to set
    # ADM NUMOBS in the zcat.
    if numobs_from_ledger:
        # ADM match the zcat to the targets.
        tii, zii = match(targets["TARGETID"], zcat["TARGETID"])
        # ADM update NUMOBS in the zcat for matches.
        zcat["NUMOBS"][zii] = targets["NUMOBS"][tii] + 1

    # ADM run MTL, only returning the targets that are updated.
    mtl = make_mtl(targets, oc, zcat=zcat, trimtozcat=True, trimcols=True)

    # ADM this is redundant if targets wasn't sent, but it's quick.
    nside = _get_mtl_nside()
    theta, phi = np.radians(90-mtl["DEC"]), np.radians(mtl["RA"])
    pixnum = hp.ang2pix(nside, theta, phi, nest=True)

    # ADM loop through the pixels and update the ledger, depending
    # ADM on whether we're working with .fits or .ecsv files.
    ender = get_mtl_ledger_format()
    for pix in set(pixnum):
        # ADM grab the targets in the pixel.
        ii = pixnum == pix
        mtlpix = mtl[ii]

        # ADM sorting on TARGETID is neater (although not strictly
        # ADM necessary when using io.read_mtl_ledger(unique=True)).
        mtlpix = mtlpix[np.argsort(mtlpix["TARGETID"])]

        # ADM the correct filenames for this pixel number.
        fn = fileform.format(pix)
        overfn = overrideff.format(pix)

        # ADM if an override ledger exists, update it and recover its
        # ADM relevant MTL entries.
        if os.path.exists(overfn):
            overmtl = process_overrides(overfn)
            # ADM add any override entries TO THE END OF THE LEDGER.
            mtlpix = vstack([mtlpix, overmtl])

        # ADM if we're working with .ecsv, simply append to the ledger.
        if ender == 'ecsv':
            f = open(fn, "a")
            ascii.write(mtlpix, f, format='no_header', formats=mtlformatdict)
            f.close()
        # ADM otherwise, for FITS, we'll have to read in the whole file.
        else:
            ledger, hd = fitsio.read(fn, extname="MTL", header=True)
            done = np.concatenate([ledger, mtlpix.as_array()])
            fitsio.write(fn+'.tmp', done, extname='MTL', header=hd, clobber=True)
            os.rename(fn+'.tmp', fn)

    return


def match_ledger_to_targets(mtl, targets):
    """Add a full set of target columns to an MTL array.

    Parameters
    ----------
    mtl : :class:`~numpy.array` or `~astropy.table.Table`
        A Merged Target List in array or Table form. Must contain the
        column "TARGETID".
    targets : :class:`str`
        An array of DESI targets, as made by, e.g., `select_targets`.
        Must contain the column "TARGETID".

    Returns
    -------
    :class:`~numpy.array`
        The passed MTL array with all target columns added.

    Notes
    -----
    - See also :func:`~desitarget.mtl.inflate_ledger()`.
    - All TARGETIDs in the `mtl` must be present in `targets`, otherwise
      an exception is thrown.
    - Speed-ups to inflate_ledger() suggested by Anand Raichoor.
    """
    # ADM match the mtl back to the targets on TARGETID.
    ii = match_to(targets["TARGETID"], mtl["TARGETID"])

    # ADM sanity check that everything in the mtl is in the targets.
    if len(ii) != len(mtl):
        msg = "MTL contains {} objects, but only {} match targets".format(
            len(mtl), len(ii))
        log.error(msg)
        raise IOError(msg)

    # ADM extract just the targets that match the mtl.
    targets = targets[ii]

    # AR now turning to Table, and removing TARGETID.
    t = Table(targets)
    t.remove_column("TARGETID")

    # ADM warn about duplicate columns.
    dupcols = set(t.columns).intersection(set(mtl.dtype.names))
    if len(dupcols) > 0:
        msg = "{} columns will be duplicated in output array.".format(dupcols)
        msg += " Target columns are denoted by _1 and mtl columns by _2."
        log.warning(msg)

    # AR starting with the added columns to preserve the same column
    # AR ordering as in inflate_ledger().
    done = Table(mtl)
    # AR putting TARGETID first, for the same reason.
    keys = ["TARGETID"] + [key for key in done.dtype.names if key != "TARGETID"]
    done = done[keys]

    # AR and stacking horizontally.
    done = hstack([t, done])
    done = done.as_array()

    return done


def inflate_ledger(mtl, hpdirname, columns=None, header=False, strictcols=False,
                   quick=False):
    """Add a fuller set of target columns to an MTL array.

    Parameters
    ----------
    mtl : :class:`~numpy.array` or `~astropy.table.Table`
        A Merged Target List in array or Table form. Must contain the
        columns "RA", "DEC" and "TARGETID"
    hpdirname : :class:`str`
        Full path to a directory containing targets that have been
        partitioned by HEALPixel (i.e. as made by `select_targets`
        with the `bundle_files` option).
    columns : :class:`list` or :class:`str`, optional
        Only return these target columns. If ``None`` or not passed
        then return all of the target columns.
    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then also return the header of the last file read
        from the `hpdirname` directory.
    strictcols : :class:`bool`, optional, defaults to ``False``
        If ``True`` then strictly return only the columns in `columns`,
        otherwise, inflate the ledger with the new columns. Ignored if
        `columns` is set to ``None``.
    quick : :class:`bool`, optional, defaults to ``False``
        If ``True``, assume the fidelity of the data model. This is much
        faster but makes fewer error checks. If `quick` is ``True`` then
        `strictcols` is ignored and any columns shared by `mtl` and the
        target files in `hpdirname` are BOTH returned using the astropy
        Table convention. e.g., if "RA" is in both the target files and
        `mtl`, the returned array will have "RA_1" for the added column
        from the target files and "RA_2" for the original `mtl` column.

    Returns
    -------
    :class:`~numpy.array`
        The original MTL with the fuller set of columns.

    Notes
    -----
    - For most uses, you WILL want to pass `quick`=``True``.
    - Will run more quickly if the targets in `mtl` are clustered.
    - "TARGETID" is ALWAYS returned, even if it isn't in `columns`,
      unless `strictcols`==``True``.
    - All TARGETIDs in the `mtl` must be present in the `hpdirname`,
      directory, otherwise an exception is thrown.
    - If `quick`=``False`` then any column in `mtl` that is also in
      `columns` will be OVERWRITTEN. So (for `quick`=``False``), be
      careful not to pass `columns=None` if you only intend to ADD
      columns to `mtl` rather than also SUBSTITUTING some columns.
    """
    # ADM if a table was passed convert it to a numpy array.
    if isinstance(mtl, Table):
        mtl = mtl.as_array()

    # ADM if a string was passed for the columns, convert it to a list.
    if isinstance(columns, str):
        columns = [columns]

    # ADM we have to have TARGETID, even if it wasn't a passed column.
    if columns is not None:
        origcols = columns.copy()
        if "TARGETID" not in columns:
            columns.append("TARGETID")

    # ADM look up the optimal nside for reading targets.
    if quick:
        fns = iglob(os.path.join(hpdirname, "*fits"))
        fn = next(fns)
        # ADM grab the FILENSID from one of the files.
        nside = fitsio.read_header(fn, 1)["FILENSID"]
    else:
        nside, _ = io.check_hp_target_dir(hpdirname)

    # ADM which pixels do we need to read.
    theta, phi = np.radians(90-mtl["DEC"]), np.radians(mtl["RA"])
    pixnums = hp.ang2pix(nside, theta, phi, nest=True)
    pixlist = list(set(pixnums))

    # ADM read in targets in the required pixels.
    targs = io.read_targets_in_hp(hpdirname, nside, pixlist, columns=columns,
                                  header=header, quick=quick)
    if header:
        targs, hdr = targs

    if quick:
        # ADM the quick code ignores column parsing/the strictcols input.
        done = match_ledger_to_targets(mtl, targs)
    else:
        # ADM match the mtl back to the targets on TARGETID.
        ii = match_to(targs["TARGETID"], mtl["TARGETID"])

        # ADM sanity check that everything in the mtl is in the targets.
        if len(ii) != len(mtl):
            msg = "MTL contains {} objects, but only {} match targets".format(
                len(mtl), len(ii))

        # ADM extract just the targets that match the mtl.
        targs = targs[ii]

        # ADM create an array to contain the fuller set of target columns.
        # ADM start with the data model for the target columns.
        dt = targs.dtype.descr
        # ADM add the unique columns from the mtl.
        xtracols = [nam for nam in mtl.dtype.names if nam not in targs.dtype.names]
        for col in xtracols:
            dt.append((col, mtl[col].dtype.str))
        # ADM remove columns from the data model that weren't requested.
        if columns is not None and strictcols:
            dt = [(name, form) for name, form in dt if name in origcols]
        # ADM create the output array.
        done = np.empty(len(mtl), dtype=dt)
        # ADM populate the output array with the fuller target columns.
        for col in targs.dtype.names:
            if col in done.dtype.names:
                done[col] = targs[col]
        # ADM populate the output array with the unique MTL columns.
        for col in xtracols:
            if col in done.dtype.names:
                done[col] = mtl[col]

    if header:
        return done, hdr
    return done


def tiles_to_be_processed(zcatdir, mtltilefn, obscon, survey, reprocess=False,
                          batch=None):
    """Find tiles that are "done" but aren't yet in the MTL tile record.

    Parameters
    ----------
    zcatdir : :class:`str`
        Full path to the directory that hosts redshift catalogs.
    mtltilefn : :class:`str`
        Full path to the file of tiles that have been processed by MTL.
    obscon : :class:`str`
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "DARK"
        Governs how priorities are set when merging targets.
    survey : :class:`str`
        Used to look up the correct ledger, in combination with `obscon`.
        Options are ``'main'`` and ``'svX``' (where X is 1, 2, 3 etc.)
        for the main survey and different iterations of SV, respectively.
    reprocess : :class:`bool`, optional, defaults to ``False``
        If ``True`` find reprocessed tiles (tiles with an ARCHIVEDATE in
        the tiles-specstatus file later than their TIMESTAMP in the
        mtl-done-tiles file) instead of tiles that are newly done.
    batch : :class:`int`, optional, defaults to ``None``
        Only return the first `batch` tiles. The ``None`` default returns
        all tiles. Useful for performing MTL updates in small batches. If
        `batch` exceeds the total number of unprocessed tiles, all tiles
        are returned. Tiles are sorted by TILEID for reproducibility.

    Returns
    -------
    :class:`~numpy.array`
        An array of tiles that are yet to be processed and written to the
        mtl tile file. A subset is returned if `batch` is not ``None``.

    Notes
    -----
    - If `survey` is `'main'` the code assumes the file with zdone for
      spectro tiles is `mtltilefn`/../../ops/tiles-specstatus.ecsv
      (i.e. it is in the ops directory parallel to the mtl directory).
      If `survey` is `'svX'` the code assumes it is `zcatdir`/tiles.csv.
    """
    # ADM read in the ZTILE file.
    ztilefn = get_ztile_file_name(survey=survey)
    # ADM directory structure used to be different for sv and main.
    if survey[:2] == 'sv' or survey == 'main':
        if os.path.dirname(mtltilefn)[-3:] == 'mtl':
            opsdir = os.path.join(os.path.dirname(mtltilefn)[:-3], 'ops')
        else:
            opsdir = os.path.dirname(mtltilefn)
        ztilefn = os.path.join(opsdir, ztilefn)
    else:
        msg = "Allowed 'survey' inputs are sv(X) or main, not {}".format(survey)
        log.critical(msg)
        raise ValueError(msg)

    tilelookup = Table.read(ztilefn)

    # ADM the ZDONE column is a string, convert to a Boolean.
    zdone = np.array(["true" in row for row in tilelookup["ZDONE"]])
    # ADM redshift processing must be complete.
    ii = zdone
    alltiles = tilelookup[ii]

    # ADM read in the MTL tile file, guarding against it not having being
    # ADM created yet.
    donetiles = None
    if os.path.isfile(mtltilefn):
        donetiles = io.read_mtl_tile_file(mtltilefn)
        # ADM check loop from ZTILES to MTL_DONE_TILES isn't out-of-sync.
        ncheck = len(set(donetiles["TILEID"]) - set(alltiles["TILEID"]))
        if ncheck > 0:
            msg = "{} tile(s) that have been processed by MTL".format(ncheck)
            msg += " are missing from the ZTILES file!!!"
            log.warning(msg)
            msg = "This might be because some zdone=true tiles have been set to "
            msg += "false in tiles-specstatus.ecsv for reprocessing. Tiles are:"
            log.info(msg)
            log.info(set(donetiles["TILEID"]) - set(alltiles["TILEID"]))

    # ADM extract the updated tiles.
    if donetiles is None:
        # ADM first time through, all tiles have yet to be processed...
        tiles = alltiles
    elif reprocess:
        # ADM match the done tiles to tiles-specstatus file on TILEID.
        aii, dii = match(alltiles["TILEID"], donetiles["TILEID"])
        # ADM quick check that we've matched everything.
        if len(donetiles) != len(dii):
            msg = "{} tiles in {} only matched to {} tiles in {}!!!".format(
                len(donetiles), mtltilefn, len(dii), ztilefn)
            log.warning(msg)
            msg = "Again, this might be as some zdone=true tiles have been set "
            msg += "to false for further reprocessing (see above)."
            log.info(msg)
        matchedtiles, donetiles = alltiles[aii], donetiles[dii]
        # ADM convert the TIMESTAMP for the done tiles to a DESI night.
        donenight = utc_date_to_night(donetiles["TIMESTAMP"])
        # ADM check for tiles that were archived on a date later than
        # ADM their most recent timestamp.
        ii = matchedtiles["ARCHIVEDATE"] > donenight
        tiles = matchedtiles[ii]
    else:
        # ADM ...else, we want tiles that uniquely appear in the combined
        # ADM "alltiles" and "donetiles" (as they aren't in "donetiles").
        newtids = set(alltiles["TILEID"]) - set(donetiles["TILEID"])
        ii = np.array([tid in newtids for tid in alltiles["TILEID"]])
        tiles = alltiles[ii]

    # ADM restrict the tiles to be processed to the correct survey.
    ii = tiles["SURVEY"] == survey
    # ADM also must match the correct lower- or upper-case (OBSCON).
    ii &= ((tiles["FAPRGRM"] == obscon.lower()) |
           (tiles["FAPRGRM"] == obscon.upper()))
    tiles = tiles[ii]

    # ADM initialize the output array and add the tiles.
    newtiles = np.zeros(len(tiles), dtype=mtltilefiledm.dtype)
    newtiles["TILEID"] = tiles["TILEID"]
    # ADM look up the time.
    newtiles["TIMESTAMP"] = get_utc_date(survey=survey)
    # ADM add the version of desitarget.
    newtiles["VERSION"] = dt_version
    # ADM add the program/obscon.
    newtiles["PROGRAM"] = obscon
    # ADM the final processed date for the redshifts.
    newtiles["ZDATE"] = tiles["LASTNIGHT"]
    # ADM the date the tile was archived.
    newtiles["ARCHIVEDATE"] = tiles["ARCHIVEDATE"]

    # ADM sort on TILEID, just in case.
    ii = np.argsort(newtiles["TILEID"])
    newtiles = newtiles[ii]
    # ADM if batching was requested, return the first n tiles.
    if batch is not None and batch < len(newtiles):
        return newtiles[:batch]

    return newtiles


def make_zcat(zcatdir, tiles, obscon, survey):
    """Make a catalog of redshifts used to inform the MTL loop.

    Parameters
    ----------
    zcatdir : :class:`str`
        Full path to the "daily" directory that hosts redshift catalogs.
    tiles : :class:`~numpy.array`
        Numpy array of tiles to be processed. Must contain at least:
        * TILEID - unique tile identifier.
        * ZDATE - final night processed to complete the tile (YYYYMMDD).
    obscon : :class:`str`
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. `desitarget.targetmask.obsconditions`), e.g. "DARK".
        Governs how ZWARN is updated using `DELTACHI2` when `survey` is
        "sv3" (in :func:`~desitarget.mtl.make_zcat_rr_backstop()`).
    survey : :class:`str`, optional, defaults to "main"
        Used to update `ZWARN` using `DELTACHI2` for a given survey type.
        Options are ``'main'`` and ``'svX``' (where X is 1, 2, 3 etc.)
        for the main survey and different iterations of SV, respectively.

    Returns
    -------
    :class:`~astropy.table.Table`
        A zcat in the official format (`zcatdatamodel`) compiled from
        the `tiles` in `zcatdir`.

    Notes
    -----
    - For surveys prior to "main" this is just a wrapper on
      :func:`~desitarget.mtl.make_zcat_rr_backstop()`.
    """
    if survey != "main":
        return make_zcat_rr_backstop(zcatdir, tiles, obscon, survey)

    # ADM the root directory in the data model.
    rootdir = os.path.join(zcatdir, "tiles", "archive")

    # ADM for each tile, read in the spectroscopic and targeting info.
    allzs = []
    for tile in tiles:
        # ADM build the correct directory structure.
        tiledir = os.path.join(rootdir, str(tile["TILEID"]))
        ymdir = os.path.join(tiledir, str(tile["ARCHIVEDATE"]))
        # ADM explicitly check if the directory has been created.
        if not os.path.isdir(ymdir):
            msg = "{} is pointed to in the tiles-specstatus.ecsv".format(ymdir)
            msg += " file but does not exist!!!"
            log.critical(msg)
            raise RuntimeError(msg)
        # ADM and retrieve the zmtl catalog (orig name zqso/lyazcat)
        qsozcatfns = sorted(glob(os.path.join(ymdir, "zmtl*fits")))
        for qsozcatfn in qsozcatfns:
            zz = fitsio.read(qsozcatfn, "ZMTL")
            allzs.append(zz)
            # ADM check the correct TILEID was written in the fibermap.
            if set(zz["ZTILEID"]) != set([tile["TILEID"]]):
                msg = "Directory and fibermap don't match for tile".format(tile)
                log.critical(msg)
                raise ValueError(msg)
    zs = np.concatenate(allzs)

    # ADM remove -ve TARGETIDs which should correspond to sky fibers.
    zs = zs[zs["TARGETID"] >= 0]

    # ADM check the TARGETIDs are unique. If they aren't the likely
    # ADM explanation is that overlapping tiles (which could include
    # ADM duplicate targets) are being processed.
    if len(zs) != len(set(zs["TARGETID"])):
        msg = "a target is duplicated!!! You are likely trying to process "
        msg += "overlapping tiles when one of these tiles should already have "
        msg += "been processed and locked in mtl-done-tiles.ecsv"
        log.critical(msg)
        raise ValueError(msg)

    # ADM write out the zcat as a file with the correct data model.
    dm = survey_data_model(zcatdatamodel, survey=survey)
    qsozcat = Table(np.zeros(len(zs), dtype=dm.dtype))
    for col in qsozcat.dtype.names:
        qsozcat[col] = zs[col]

    return qsozcat


def make_zcat_rr_backstop(zcatdir, tiles, obscon, survey):
    """Make a simple zcat using only redrock outputs.

    Parameters
    ----------
    zcatdir : :class:`str`
        Full path to the "daily" directory that hosts redshift catalogs.
    tiles : :class:`~numpy.array`
        Numpy array of tiles to be processed. Must contain at least:
        * TILEID - unique tile identifier.
        * ZDATE - final night processed to complete the tile (YYYYMMDD).
    obscon : :class:`str`
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. `desitarget.targetmask.obsconditions`), e.g. "DARK".
        Governs how ZWARN is updated using `DELTACHI2`.
    survey : :class:`str`, optional, defaults to "main"
        Used to update `ZWARN` using `DELTACHI2` for a given survey type.
        Options are ``'main'`` and ``'svX``' (where X is 1, 2, 3 etc.)
        for the main survey and different iterations of SV, respectively.

    Returns
    -------
    :class:`~astropy.table.Table`
        A zcat in the official format (`zcatdatamodel`) compiled from
        the `tiles` in `zcatdir`.

    Notes
    -----
    - How the `zcat` is constructed could certainly change once we have
      the final schema in place.
    """
    # ADM the root directory in the data model.
    rootdir = os.path.join(zcatdir, "tiles", "cumulative")

    # ADM for each tile, read in the spectroscopic and targeting info.
    allzs = []
    allfms = []
    for tile in tiles:
        # ADM build the correct directory structure.
        tiledir = os.path.join(rootdir, str(tile["TILEID"]))
        ymdir = os.path.join(tiledir, str(tile["ZDATE"]))
        # ADM and retrieve the redshifts.
        zbestfns = sorted(glob(os.path.join(ymdir, "zbest*")))
        for zbestfn in zbestfns:
            zz = fitsio.read(zbestfn, "ZBEST")
            allzs.append(zz)
            # ADM read in all of the exposures in the fibermap.
            fm = fitsio.read(zbestfn, "FIBERMAP")
            # ADM in the transition between SV3 and the Main Survey, the
            # ADM fibermap data model changed. New columns may need to be
            # ADM removed to concatenate old- and new-style fibermaps.
            if "PLATE_RA" in fm.dtype.names:
                fm = rfn.drop_fields(fm, ["PLATE_RA", "PLATE_DEC"])
            # ADM recover the information for unique targets based on the
            # ADM first entry for each TARGETID.
            _, ii = np.unique(fm['TARGETID'], return_index=True)
            allfms.append(fm[ii])
            # ADM check the correct TILEID was written in the fibermap.
            if set(fm["TILEID"]) != set([tile["TILEID"]]):
                msg = "Directory and fibermap don't match for tile".format(tile)
                log.critical(msg)
                raise ValueError(msg)
    zs = np.concatenate(allzs)
    fms = np.concatenate(allfms)

    # ADM remove -ve TARGETIDs which should correspond to sky fibers.
    zs = zs[zs["TARGETID"] >= 0]
    fms = fms[fms["TARGETID"] >= 0]

    # ADM check the TARGETIDs are unique. If they aren't the likely
    # ADM explanation is that overlapping tiles (which could include
    # ADM duplicate targets) are being processed.
    if len(zs) != len(set(zs["TARGETID"])):
        msg = "a target is duplicated!!! You are likely trying to process "
        msg += "overlapping tiles when one of these tiles should already have "
        msg += "been processed and locked in mtl-done-tiles.ecsv"
        log.critical(msg)
        raise ValueError(msg)

    # ADM currently, the spectroscopic files aren't coadds, so aren't
    # ADM unique. We therefore need to look up (any) coordinates for
    # ADM each z in the fibermap.
    zid = match_to(fms["TARGETID"], zs["TARGETID"])

    # ADM write out the zcat as a file with the correct data model.
    zcatdm = survey_data_model(zcatdatamodel, survey=survey)
    zcat = Table(np.zeros(len(zs), dtype=zcatdm.dtype))

    zcat["RA"] = fms[zid]["TARGET_RA"]
    zcat["DEC"] = fms[zid]["TARGET_DEC"]
    zcat["ZTILEID"] = fms[zid]["TILEID"]
    zcat["NUMOBS"] = zs["NUMTILE"]
    for col in set(zcat.dtype.names) - set(['RA', 'DEC', 'NUMOBS', 'ZTILEID']):
        zcat[col] = zs[col]

    # ADM Finally, flag the ZWARN bit if DELTACHI2 is too low (for sv3).
    if survey == "sv3":
        from desitarget.sv3.sv3_targetmask import desi_mask
        desi_target = fms[zid]["SV3_DESI_TARGET"]
        # ADM set ZWARN flag for everything with DELTACHI2 < 25.
        lodc2 = zs["DELTACHI2"] < 25
        zcat["ZWARN"] |= lodc2*zwarn_mask["LOW_DEL_CHI2"]
        if obscon == "BRIGHT":
            lodc2 = zs["DELTACHI2"] < 40
            bgs = desi_target & desi_mask["BGS_ANY"] != 0
            lodc2bgs = bgs & lodc2
            zcat["ZWARN"] |= lodc2bgs*zwarn_mask["LOW_DEL_CHI2_BGS"]

    return zcat


def loop_ledger(obscon, survey='main', zcatdir=None, mtldir=None,
                numobs_from_ledger=True, secondary=False, reprocess=False,
                batch=None):
    """Execute full MTL loop, including reading files, updating ledgers.

    Parameters
    ----------
    obscon : :class:`str`
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "DARK"
        Governs how priorities are set when merging targets.
    survey : :class:`str`, optional, defaults to "main"
        Used to look up the correct ledger, in combination with `obscon`.
        Options are ``'main'`` and ``'svX``' (where X is 1, 2, 3 etc.)
        for the main survey and different iterations of SV, respectively.
    zcatdir : :class:`str`, optional, defaults to ``None``
        Full path to the directory that hosts redshift catalogs. If this
        is ``None``, look up the redshift catalog directory from the
        $ZCAT_DIR environment variable.
    mtldir : :class:`str`, optional, defaults to ``None``
        Full path to the directory that hosts the MTL ledgers and the MTL
        tile file. If ``None``, then look up the MTL directory from the
        $MTL_DIR environment variable.
    numobs_from_ledger : :class:`bool`, optional, defaults to ``True``
        If ``True`` then inherit the number of observations so far from
        the ledger rather than expecting it to have a reasonable value
        in the `zcat.`
    secondary : :class:`bool`, optional, defaults to ``False``
        If ``True`` then process secondary targets instead of primaries
        for passed `survey` and `obscon`.
    reprocess : :class:`bool`, optional, defaults to ``False``
        If ``True`` find reprocessed tiles (tiles with an ARCHIVEDATE in
        the tiles-specstatus file later than their TIMESTAMP in the
        mtl-done-tiles file) instead of tiles that are newly done and
        process using special reprocessing logic.
    batch : :class:`int`, optional, defaults to ``None``
        Only process the first `batch` tiles. The ``None`` default returns
        all tiles. Useful for performing MTL updates in small batches. If
        `batch` exceeds the total number of unprocessed tiles, all tiles
        are returned. Tiles are sorted by TILEID for reproducibility.

    Returns
    -------
    :class:`str`
        The directory containing the ledger that was updated.
    :class:`str`
        The name of the MTL tile file that was updated.
    :class:`str`
        Name of ZTILE file used to link TILEIDs to observing conditions
        to determine if tiles were "done" (that they had zdone=True).
    :class:`~numpy.array`
        Information for the tiles that were processed.

    Notes
    -----
    - Assumes all of the relevant ledgers have already been made by,
      e.g., :func:`~desitarget.mtl.make_ledger()`.
    - If `survey` is `'main'` the code assumes the file with the zdone
      status for spectro tiles is `mtldir`/../ops/tiles-specstatus.ecsv
      (i.e. it is in the ops directory parallel to the mtl directory).
      If `survey` is `'svX'` the code assumes it is `zcatdir`/tiles.csv.
    """
    # ADM first grab all of the relevant files.
    # ADM grab the MTL directory (in case we're relying on $MTL_DIR).
    mtldir = get_mtl_dir(mtldir)
    # ADM construct the full path to the mtl tile file.
    mtltilefn = os.path.join(mtldir, get_mtl_tile_file_name(secondary=secondary))

    # ADM construct the relevant sub-directory for this survey and
    # ADM set of observing conditions..
    form = get_mtl_ledger_format()
    resolve = True
    msg = "running on {} ledger with obscon={} and survey={}"
    if secondary:
        log.info(msg.format("SECONDARY", obscon, survey))
        resolve = None
    else:
        log.info(msg.format("PRIMARY", obscon, survey))
    hpdirname = io.find_target_files(mtldir, flavor="mtl", resolve=resolve,
                                     survey=survey, obscon=obscon, ender=form)
    # ADM grab the zcat directory (in case we're relying on $ZCAT_DIR).
    zcatdir = get_zcat_dir(zcatdir)

    # ADM grab an array of tiles that are yet to be processed.
    tiles = tiles_to_be_processed(zcatdir, mtltilefn, obscon, survey,
                                  reprocess=reprocess, batch=batch)

    # ADM contruct the ZTILE filename, for logging purposes.
    ztilefn = get_ztile_file_name(survey=survey)
    # ADM directory structure used to be different for sv and main.
    if survey[:2] == 'sv' or survey == 'main':
        if os.path.dirname(mtltilefn)[-3:] == 'mtl':
            opsdir = os.path.join(os.path.dirname(mtltilefn)[:-3], 'ops')
        else:
            opsdir = os.path.dirname(mtltilefn)
        ztilefn = os.path.join(opsdir, ztilefn)

    # ADM stop if there are no tiles to process.
    if len(tiles) == 0:
        return hpdirname, mtltilefn, ztilefn, tiles

    # ADM create the catalog of updated redshifts.
    zcat = make_zcat(zcatdir, tiles, obscon, survey)

    # ADM insist that for an MTL loop with real observations, the zcat
    # ADM must conform to the data model. In particular, it must include
    # ADM ZTILEID, and other columns added for the Main Survey. These
    # ADM columns may not be needed for non-ledger simulations.
    # ADM Note that the data model differs with survey type.
    zcatdm = survey_data_model(zcatdatamodel, survey=survey)
    if zcat.dtype.descr != zcatdm.dtype.descr:
        msg = "zcat data model must be {} not {}!".format(
            zcatdm.dtype.descr, zcat.dtype.descr)
        log.critical(msg)
        raise ValueError(msg)
    # ADM useful to know how many targets were updated.
    _, _, _, _, sky, _ = decode_targetid(zcat["TARGETID"])
    ntargs, nsky = np.sum(sky == 0), np.sum(sky)
    msg = "Update state for {} targets".format(ntargs)
    msg += " (the zcats also contain {} skies with +ve TARGETIDs)".format(nsky)
    log.info(msg)

    # ADM update the appropriate ledgers.
    if reprocess:
        reprocess_ledger(hpdirname, zcat, obscon=obscon)
    else:
        update_ledger(hpdirname, zcat, obscon=obscon,
                      numobs_from_ledger=numobs_from_ledger)

    # ADM for the main survey "holding pen" method, ensure the TIMESTAMP
    # ADM in the mtl-done-tiles file is always later than in the ledgers.
    if survey == "main":
        sleep(1)
        tiles["TIMESTAMP"] = get_utc_date(survey=survey)

    # ADM write the processed tiles to the MTL tile file.
    io.write_mtl_tile_file(mtltilefn, tiles)

    return hpdirname, mtltilefn, ztilefn, tiles
