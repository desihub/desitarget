"""
desitarget.streams.targets
===========================

Manipulate/add target bitmasks/priorities/numbers of observations for
stream targets.
"""
import numpy as np
import numpy.lib.recfunctions as rfn

from desitarget.targets import initial_priority_numobs, set_obsconditions, \
    encode_targetid

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()


def finalize(targets, desi_target, bgs_target, mws_target, scnd_target):
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
    scnd_target : :class:`~numpy.ndarray`
        1D array of target selection bit flags.

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
              - SCND_TARGET: stream secondary target selection flags.
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
        - NUMOBS_INIT and PRIORITY_INIT are split into DARK/BRIGHT/BACKUP
          versions, as these surveys are effectively separate.
    """
    # ADM some straightforward checks that inputs are the same length.
    ntargets = len(targets)
    assert ntargets == len(desi_target)
    assert ntargets == len(bgs_target)
    assert ntargets == len(mws_target)
    assert ntargets == len(scnd_target)

    # - OBJID in tractor files is only unique within the brick; rename and
    # - create a new unique TARGETID
    targets = rfn.rename_fields(targets,
                                {'OBJID': 'BRICK_OBJID', 'TYPE': 'MORPHTYPE'})


    targetid = encode_targetid(objid=targets['BRICK_OBJID'],
                               brickid=targets['BRICKID'],
                               release=targets['RELEASE'])


    nodata = np.zeros(ntargets, dtype='int')-1
    subpriority = np.zeros(ntargets, dtype='float')

    # ADM the columns to write out and their values and formats.
    cols = ["TARGETID", "DESI_TARGET", "BGS_TARGET", "MWS_TARGET", "SCND_TARGET",
            "SUBPRIORITY", "OBSCONDITIONS"]
    vals = [targetid, desi_target, bgs_target, mws_target, scnd_target,
            subpriority, nodata]
    forms = [">i8", ">i8", ">i8", ">i8", ">i8", ">f8", ">i8"]

    # ADM set the initial PRIORITY and NUMOBS.
    # ADM populate bright/dark/backup separately.
    ender = ["_DARK", "_BRIGHT", "_BACKUP"]
    obscon = ["DARK|GRAY", "BRIGHT", "BACKUP"]
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
        done[pc], done[nc] = initial_priority_numobs(done, obscon=oc, scnd=True)

    # ADM set the OBSCONDITIONS.
    done["OBSCONDITIONS"] = set_obsconditions(done, scnd=True)

    # ADM some final checks that the targets conform to expectations...
    # ADM check that each target has a unique ID.
    if len(done["TARGETID"]) != len(np.unique(done["TARGETID"])):
        msg = ("Targets are not unique. The code might need updated to read the "
               "sweep files one-by-one (as in desitarget.cuts.select_targets()) "
               "rather than caching each individual stream")
        log.critical(msg)

    return done
