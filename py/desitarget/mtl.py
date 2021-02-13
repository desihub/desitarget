"""
desitarget.mtl
==============

Merged target lists.
"""

import os
import numpy as np
import healpy as hp
import numpy.lib.recfunctions as rfn
import sys
from astropy.table import Table
from astropy.io import ascii
import fitsio
from time import time
from datetime import datetime

from . import __version__ as dt_version
from desitarget.targetmask import obsmask, obsconditions
from desitarget.targets import calc_priority, calc_numobs_more
from desitarget.targets import main_cmx_or_sv, switch_main_cmx_or_sv
from desitarget.targets import set_obsconditions
from desitarget.internal import sharedmem
from desitarget import io

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
    ('TIMESTAMP', 'S19'), ('VERSION', 'S14'), ('TARGET_STATE', 'S16')
    ])

# ADM when using basic or csv ascii writes, specifying the formats of
# ADM float32 columns can make things easier on the eye.
mtlformatdict = {"PARALLAX": '%16.8f', 'PMRA': '%16.8f', 'PMDEC': '%16.8f'}


def get_mtl_dir():
    """Convenience function to grab the MTL_DIR environment variable.

    Returns
    -------
    :class:`str`
        The directory stored in the $MTL_DIR environment variable.
    """
    # ADM check that the $MTL_DIR environment variable is set.
    mtldir = os.environ.get('MTL_DIR')
    if mtldir is None:
        msg = "Set $MTL_DIR environment variable!"
        log.critical(msg)
        raise ValueError(msg)

    return mtldir


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
    """Adds fiberassign and zcat columns to a targets table.

    Parameters
    ----------
    targets : :class:`~numpy.array` or `~astropy.table.Table`
        A numpy rec array or astropy Table with at least the columns
        ``TARGETID``, ``DESI_TARGET``, ``NUMOBS_INIT``, ``PRIORITY_INIT``.
        or the corresponding columns for SV or commissioning.
    obscon : :class:`str`
        A combination of strings that are in the desitarget bitmask yaml
        file (specifically in `desitarget.targetmask.obsconditions`), e.g.
        "DARK|GRAY". Governs the behavior of how priorities are set based
        on "obsconditions" in the desitarget bitmask yaml file.
    zcat : :class:`~astropy.table.Table`, optional
        Redshift catalog table with columns ``TARGETID``, ``NUMOBS``, ``Z``,
        ``ZWARN``.
    scnd : :class:`~numpy.array`, `~astropy.table.Table`, optional
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
    """
    start = time()
    # ADM set up the default logger.
    from desiutil.log import get_logger
    log = get_logger()

    # ADM if trimcols was passed, reduce input target columns to minimal.
    if trimcols:
        mtldm = switch_main_cmx_or_sv(mtldatamodel, targets)
        cullcols = list(set(targets.dtype.names) - set(mtldm.dtype.names))
        if isinstance(targets, Table):
            targets.remove_columns(cullcols)
        else:
            targets = rfn.drop_fields(targets, cullcols)

    # ADM determine whether the input targets are main survey, cmx or SV.
    colnames, masks, survey = main_cmx_or_sv(targets, scnd=True)
    # ADM set the first column to be the "desitarget" column
    desi_target, desi_mask = colnames[0], masks[0]
    scnd_target = colnames[-1]

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
    if zcat is not None:
        ok = np.in1d(zcat['TARGETID'], targets['TARGETID'])
        num_extra = np.count_nonzero(~ok)
        if num_extra > 0:
            log.warning("Ignoring {} zcat entries that aren't "
                        "in the input target list".format(num_extra))
            zcat = zcat[ok]

    n = len(targets)
    # ADM if a redshift catalog was passed, order it to match the input targets
    # ADM catalog on 'TARGETID'.
    if zcat is not None:
        # ADM there might be a quicker way to do this?
        # ADM set up a dictionary of the indexes of each target id.
        d = dict(tuple(zip(targets["TARGETID"], np.arange(n))))
        # ADM loop through the zcat and look-up the index in the dictionary.
        zmatcher = np.array([d[tid] for tid in zcat["TARGETID"]])
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
        # ADM if zcat wasn't passed, there is a one-to-one correspondence
        # ADM between the targets and the zcat.
        zmatcher = np.arange(n)

    # ADM extract just the targets that match the input zcat.
    targets_zmatcher = targets[zmatcher]

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
    log.info('{:d} of {:d} targets have priority zero, setting N_obs=0.'.format(
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

    # ADM add a placeholder for the secondary bit-mask, if it isn't there.
    if scnd_target not in mtl.dtype.names:
        mtl[scnd_target] = np.zeros(len(mtl),
                                    dtype=mtldatamodel["SCND_TARGET"].dtype)

    # ADM initialize columns to avoid zero-length/missing/format errors.
    zcols = ["NUMOBS_MORE", "NUMOBS", "Z", "ZWARN"]
    for col in zcols + ["TARGET_STATE", "TIMESTAMP", "VERSION"]:
        mtl[col] = np.empty(len(mtl), dtype=mtldatamodel[col].dtype)

    # ADM any target that wasn't matched to the ZCAT should retain its
    # ADM original (INIT) value of PRIORITY and NUMOBS.
    mtl['NUMOBS_MORE'] = mtl['NUMOBS_INIT']
    mtl['PRIORITY'] = mtl['PRIORITY_INIT']
    mtl['TARGET_STATE'] = "UNOBS"
    # ADM add the time and version of the desitarget code that was run.
    utc = datetime.utcnow().isoformat(timespec='seconds')
    mtl["TIMESTAMP"] = utc
    mtl["VERSION"] = dt_version

    # ADM now populate the new mtl columns with the updated information.
    mtl['OBSCONDITIONS'] = obsconmask
    mtl['PRIORITY'][zmatcher] = priority
    mtl['TARGET_STATE'][zmatcher] = target_state
    for col in zcols:
        mtl[col][zmatcher] = ztargets[col]

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

    # ADM assert the data model is complete.
    # ADM turning this off for now, useful for testing.
#    mtltypes = [mtl[i].dtype.type for i in mtl.dtype.names]
#    mtldmtypes = [mtldm[i].dtype.type for i in mtl.dtype.names]
#    assert set(mtl.dtype.names) == set(mtldm.dtype.names)
#    assert mtltypes == mtldmtypes

    log.info('Done...t={:.1f}s'.format(time()-start))

    if trimtozcat:
        return mtl[zmatcher]
    return mtl


def make_ledger_in_hp(targets, outdirname, nside, pixlist,
                      obscon="DARK", indirname=None, verbose=True):
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
        (NESTED) HEALPixel nside that corresponds to `pixnum`.
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

    Returns
    -------
    Nothing, but writes the `targets` out to `outdirname` split across
    each HEALPixel in `pixlist`.
    """
    t0 = time()

    # ADM in case an integer was passed.
    pixlist = np.atleast_1d(pixlist)

    # ADM execute MTL.
    mtl = make_mtl(targets, obscon, trimcols=True)

    # ADM the HEALPixel within which each target in the MTL lies.
    theta, phi = np.radians(90-mtl["DEC"]), np.radians(mtl["RA"])
    mtlpix = hp.ang2pix(nside, theta, phi, nest=True)

    # ADM write the MTLs.
    _, _, survey = main_cmx_or_sv(mtl)
    for pix in pixlist:
        inpix = mtlpix == pix
        ecsv = get_mtl_ledger_format() == "ecsv"
        nt, fn = io.write_mtl(
            outdirname, mtl[inpix].as_array(), indir=indirname, ecsv=ecsv,
            survey=survey, obscon=obscon, nsidefile=nside, hpxlist=pix)
        if verbose:
            log.info('{} targets written to {}...t={:.1f}s'.format(
                nt, fn, time()-t0))

    return


def make_ledger(hpdirname, outdirname, obscon="DARK", numproc=1):
    """
    Make initial MTL ledger files for all HEALPixels.

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
    obscon : :class:`str`, optional, defaults to "DARK"
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "GRAY"
        Governs how priorities are set based on "obsconditions". Also
        governs the sub-directory to which the ledger is written.
    numproc : :class:`int`, optional, defaults to 1 for serial
        Number of processes to parallelize across.

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

    # ADM the MTL datamodel must reflect the target flavor (SV, etc.).
    mtldm = switch_main_cmx_or_sv(mtldatamodel, np.array([], dt))
    # ADM speed-up by only reading the necessary columns.
    cols = list(set(mtldm.dtype.names).intersection(dt.names))

    # ADM optimal nside for reading in the targeting files.
    nside = hdr["FILENSID"]
    npixels = hp.nside2npix(nside)
    pixels = np.arange(npixels)

    # ADM the nside at which to write the MTLs.
    mtlnside = _get_mtl_nside()

    # ADM check that the nside for writing MTLs is not at a lower
    # ADM resolution than the nside at which the files are stored.
    msg = "Ledger nside ({}) must be higher than file nside ({})!!!".format(
        mtlnside, nside)
    assert mtlnside >= nside, msg

    from desitarget.geomask import nside2nside

    # ADM the common function that is actually parallelized across.
    def _make_ledger_in_hp(pixnum):
        """make initial ledger in a single HEALPixel"""
        # ADM read in the needed columns from the targets.
        targs = io.read_targets_in_hp(hpdirname, nside, pixnum, columns=cols)
        if len(targs) == 0:
            return
        # ADM construct a list of all pixels in pixnum at the MTL nside.
        pixlist = nside2nside(nside, mtlnside, pixnum)
        # ADM write MTLs for the targs split over HEALPixels in pixlist.
        return make_ledger_in_hp(
            targs, outdirname, mtlnside, pixlist,
            obscon=obscon, indirname=hpdirname, verbose=False)

    # ADM this is just to count pixels in _update_status.
    npix = np.ones((), dtype='i8')
    t0 = time()

    def _update_status(result):
        """wrap key reduction operation on the main parallel process"""
        if npix % 2 == 0 and npix > 0:
            rate = (time() - t0) / npix
            log.info('{}/{} HEALPixels; {:.1f} secs/pixel...t = {:.1f} mins'.
                     format(npix, npixels, rate, (time()-t0)/60.))
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

    log.info("Done writing ledger...t = {:.1f} mins".format((time()-t0)/60.))

    return


def update_ledger(hpdirname, targets, zcat, obscon="DARK"):
    """
    Update relevant HEALPixel-split ledger files for some targets.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to a directory containing an MTL ledger that has been
        partitioned by HEALPixel (i.e. as made by `make_ledger`).
    targets : :class:`~numpy.array` or `~astropy.table.Table`
        A numpy rec array or astropy Table with at least the columns
        ``RA``, ``DEC``, ``TARGETID``, ``DESI_TARGET``, ``NUMOBS_INIT``,
        and ``PRIORITY_INIT``.
    zcat : :class:`~astropy.table.Table`, optional
        Redshift catalog table with columns ``TARGETID``, ``NUMOBS``,
        ``Z``, ``ZWARN``.
    obscon : :class:`str`, optional, defaults to "DARK"
        A string matching ONE obscondition in the desitarget bitmask yaml
        file (i.e. in `desitarget.targetmask.obsconditions`), e.g. "GRAY"
        Governs how priorities are set using "obsconditions". Basically a
        check on whether the files in `hpdirname` are as expected.

    Returns
    -------
    Nothing, but relevant ledger files are updated.
    """
# ADM in theory, here, fiberassign wouldn't need to carry much around at
# ADM all. We could, instead, simply read the relevant MTL pixel-ledgers
# ADM and match on TARGETID to recover everything we'd need. Better yet,
# ADM if the zcat included RA/Dec, we wouldn't even need `targets`..e.g.:
    # ADM read the relevant pixel-ledger (and record the files we read).
#    mtltargs, fndict = io.read_mtl_in_hp(hpdirname, nside, pixnum,
#                                         unique=True, returnfn=True)
    # ADM then match between mtltargs and targets on TARGETID, etc.

    # ADM find the general format for the ledger files in `hpdirname`.
    # ADM also returning the obsconditions.
    fileform, oc = io.find_mtl_file_format_from_header(hpdirname, returnoc=True)

    # ADM check the obscondition is as expected.
    if obscon != oc:
        msg = "File is type {} but requested behavior is {}".format(oc, obscon)
        log.critical(msg)
        raise ValueError(msg)

    # ADM run MTL, only returning the targets that are updated.
    mtl = make_mtl(targets, oc, zcat=zcat, trimtozcat=True, trimcols=True)

    # ADM look up which HEALPixels are represented in the updated MTL.
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

        # ADM sorting on TARGETID is important for
        # ADM io.read_mtl_ledger(unique=True)
        mtlpix = mtlpix[np.argsort(mtlpix["TARGETID"])]

        # ADM the correct filename for this pixel number.
        fn = fileform.format(pix)

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


def inflate_ledger(mtl, hpdirname, columns=None, header=False, strictcols=False):
    """Add a fuller set of target columns to an MTL.

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
        otherwise, inflate the ledger with the new columns.

    Returns
    -------
    :class:`~numpy.array`
        The original MTL with the fuller set of columns.

    Notes
    -----
        - Will run more quickly if the targets in `mtl` are clustered.
        - TARGETID is always returned, even if it isn't in `columns`.
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
    nside, _ = io.check_hp_target_dir(hpdirname)

    # ADM which pixels do we need to read.
    theta, phi = np.radians(90-mtl["DEC"]), np.radians(mtl["RA"])
    pixnums = hp.ang2pix(nside, theta, phi, nest=True)
    pixlist = list(set(pixnums))

    # ADM read in targets in the required pixels.
    targs = io.read_targets_in_hp(hpdirname, nside, pixlist, columns=columns,
                                  header=header)
    if header:
        targs, hdr = targs

    # ADM match the mtl back to the targets on TARGETID.
    etargids, smtlids = enumerate(targs["TARGETID"]), set(mtl["TARGETID"])
    ii = [i for i, tid in etargids if tid in smtlids]

    # ADM reorder targets to match MTL on TARGETID.
    targs = targs[ii]
    targsort = np.argsort(targs["TARGETID"])
    mtlsort = np.argsort(mtl["TARGETID"])
    targs = targs[targsort][mtlsort]

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
