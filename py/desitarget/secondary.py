"""
desitarget.secondary
====================

Modules dealing with target selection for secondary targets.

Note that the environment variable SCND_DIR should be set, and that
directory should contain files of two flavors:

(1) In $SCND_DIR/indata: .fits or .txt files defining the secondary
    targets with columns corresponding to secondary.indatamodel.

    For .txt files the first N columns must correspond to the N columns
    in secondary.indatamodel, and other columns can be anything. The #
    may be used as a comment card.

    For .fits files, a subset of the columns must correspond to the
    columns in secondary.indatamodel, other columns can be anything.

(2) In $SCND_DIR/docs: .ipynb (notebook) or .txt files containing
    a description of how each input data file was constructed.

Only one secondary target file of each type should be in the indata and
docs directories. So, if, e.g. $SCND_DIR/indata/blat.fits exists
then $SCND_DIR/indata/blat.txt should not.

Example files can be found in the NERSC directory:

/project/projectdirs/desi/target/secondary

Note that the OVERRIDE column in the data model means "do not just
accept an existing target, override it and make a new TARGETID." It
should be True (override) or False (do not override) for each target.
In .txt files it should be 1 or 0 instead of True/False, and will be
loaded from the text file as the corresponding Boolean.
"""
import os
import re
import fitsio
import itertools
import numpy as np
import healpy as hp

import numpy.lib.recfunctions as rfn

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Row

from time import time
from glob import glob
from importlib import import_module

from collections import defaultdict

from desitarget.internal import sharedmem
from desitarget.geomask import radec_match_to, add_hp_neighbors, is_in_hp
from desitarget.gaiamatch import gaiadatamodel

from desitarget.targets import encode_targetid, main_cmx_or_sv, resolve
from desitarget.targets import set_obsconditions, initial_priority_numobs
from desitarget.targetmask import obsconditions

from desiutil import brick
from desiutil.log import get_logger

# ADM set up the Legacy Surveys bricks object.
bricks = brick.Bricks(bricksize=0.25)
# ADM set up the default DESI logger.
log = get_logger()
start = time()

indatamodel = np.array([], dtype=[
    ('RA', '>f8'), ('DEC', '>f8'), ('PMRA', '>f4'), ('PMDEC', '>f4'),
    ('REF_EPOCH', '>f4'), ('OVERRIDE', '?')
])

# ADM the columns not in the primary target files are:
#  OVERRIDE - If True/1 force as a target even if there is a primary.
#           - If False/0 allow this to be replaced by a primary target.
#  SCND_TARGET - Corresponds to the bit mask from data/targetmask.yaml
#                or svX/data/svX_targetmask.yaml (scnd_mask).
# ADM Note that TARGETID for secondary-only targets is unique because
# ADM RELEASE is < 1000 (before DR1) for secondary-only targets.
# ADM also add needed columns for fiberassign from the Gaia data model.
gaiacols = ["PARALLAX", "GAIA_PHOT_G_MEAN_MAG", 'GAIA_ASTROMETRIC_EXCESS_NOISE']
gaiadt = [(gaiadatamodel[gaiacols].dtype.names[i],
           gaiadatamodel[gaiacols].dtype[i].str) for i in range(len(gaiacols))]

outdatamodel = np.array([], dtype=[
    ('RA', '>f8'), ('DEC', '>f8'), ('PMRA', '>f4'), ('PMDEC', '>f4'),
    ('REF_EPOCH', '>f4'), ('OVERRIDE', '?')] + gaiadt + [
    ('TARGETID', '>i8'), ('DESI_TARGET', '>i8'), ('SCND_TARGET', '>i8'),
    ('PRIORITY_INIT', '>i8'), ('SUBPRIORITY', '>f8'),
    ('NUMOBS_INIT', '>i8'), ('OBSCONDITIONS', '>i8')])

# ADM extra columns that are used during processing but are
# ADM not an official part of the input or output data model.
# ADM PRIM_MATCH records whether a secondary matches a primary TARGET.
suppdatamodel = np.array([], dtype=[
    ('SCND_TARGET_INIT', '>i8'), ('SCND_ORDER', '>i4'), ('PRIM_MATCH', '?')
])


def duplicates(seq):
    """Locations of duplicates in an array or list.

    Parameters
    ----------
    seq : :class:`list` or `~numpy.ndarray` or `str`
        A sequence, e.g., [1, 2, 3, 2] or "adfgtarga"

    Returns
    -------
    :class:`generator`
        A generator of the duplicated values in the sequence
        and an array of the indexes for each duplicate, e.g.
        for i in duplicates("adfgtarga"):
            print(i)
        returns
            ('a', array([0, 5, 8]))
            ('g', array([3, 7]))

    Notes
    -----
        - h/t https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    """
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)

    return ((key, np.array(locs)) for key, locs in tally.items() if len(locs) > 1)


def _get_scxdir(scxdir=None, survey=""):
    """Retrieve the base secondary directory with error checking.

    Parameters
    ----------
    scxdir : :class:`str`, optional, defaults to :envvar:`SCND_DIR`
        Directory containing secondary target files to which to match.
        If not specified, the directory is taken to be the value of
        the :envvar:`SCND_DIR` environment variable.
    survey : :class:`str`, optional, defaults to "" for the Main Survey.
        Flavor of survey that we're processing, e.g., "sv1". Don't pass
        anything for "main" (the Main Survey).

    Returns
    -------
    :class:`str`
        The base secondary directory, after checking it's format.
    """
    # ADM if scxdir was not passed, default to environment variable.
    if scxdir is None:
        scxdir = os.environ.get('SCND_DIR')

    # ADM fail if the scx directory is not set or passed.
    if scxdir is None or not os.path.exists(scxdir):
        log.info('pass scxdir (or --scnddir or --nosecondary) or set $SCND_DIR')
        msg = 'Secondary target files not found in {}'.format(scxdir)
        log.critical(msg)
        raise ValueError(msg)

    # ADM also fail if the indata, outdata and docs directories don't
    # ADM exist in the secondary directory.
    checkdir = [os.path.join(survey, d) for d in ["docs", "indata", "outdata"]]
    for subdir in checkdir:
        if not os.path.isdir(os.path.join(scxdir, subdir)):
            msg = '{} directory not found in {}'.format(subdir, scxdir)
            log.critical(msg)
            raise ValueError(msg)

    return scxdir


def _check_files(scxdir, scnd_mask):
    """Retrieve input files from the scx directory with error checking.

    Parameters
    ----------
    scxdir : :class:`str`
        Directory produced by :func:`~secondary._check_files()`.
    scnd_mask : :class:`desiutil.bitmask.BitMask`
        A mask corresponding to a set of secondary targets, e.g, could
        be ``from desitarget.targetmask import scnd_mask`` for the
        main survey mask.

    Returns
    -------
    Nothing.

    Notes
    -----
        - Checks that each file name has one corresponding bit in the
          secondary_mask.
        - Checks for only valid file extensions in the input directories.
        - Checks that there are no duplicate files in the `scxdir`/indata
          or in the `scxdir`/docs directory.
        - Checks that every file in the `scxdir`/indata directory has a
          corresponding informational file in `scxdir`/docs.
    """
    # ADM the allowed extensions in each directory.
    extdic = {'indata': {'.txt', '.fits'},
              'docs': {'.txt', '.ipynb'}}
    # ADM the full paths to the indata/docs directories.
    dirdic = {'indata': os.path.join(scxdir, 'indata'),
              'docs': os.path.join(scxdir, 'docs')}

    # ADM setdic will contain the set of file names, without
    # ADM extensions in each of the indata and docs directories.
    setdic = {}
    for subdir in 'indata', 'docs':
        # ADM retrieve the full file names. Ignore directories.
        fnswext = [fn for fn in os.listdir(os.path.join(scxdir, subdir)) if
                   os.path.isfile(os.path.join(scxdir, subdir, fn))]
        # ADM split off the extensions.
        exts = [os.path.splitext(fn)[1] for fn in fnswext]
        # ADM check they're all allowed extensions.
        if len(set(exts) - extdic[subdir]) > 0:
            msg = 'bad extension(s) {} in {}; must be one of {}'.format(
                set(exts) - extdic[subdir], dirdic[subdir], extdic[subdir]
            )
            log.critical(msg)
            raise ValueError(msg)

        # ADM retrieve the filenames without the extensions.
        fns = [os.path.splitext(fn)[0] for fn in fnswext]
        # ADM check for any duplicate files in either directory.
        if len(fns) > len(set(fns)):
            uniq, cnt = np.unique(fns, return_counts=True)
            dups = uniq[cnt > 1]
            msg = 'duplicate file(s) like {} in {}'.format(
                dups, dirdic[subdir])
            log.critical(msg)
            raise ValueError(msg)
        setdic[subdir] = set(fns)

    # ADM check for bit correspondence.
    setbitfns = set([scnd_mask[name].filename for name in scnd_mask.names()])
    if setbitfns != setdic['indata']:
        msg = "files in yaml file don't match files in {}\n".format(
            dirdic['indata'])
        msg += "files with bits in yaml file: {}\nfiles in {}: {}".format(
            list(setbitfns), dirdic[subdir], list(setdic['indata']))
        log.critical(msg)
        raise ValueError(msg)

    # ADM now check that files correspond between the directories.
    # ADM this ^ returns all elements not in both sets.
    missing = setdic['indata'] ^ setdic['docs']
    if len(missing) > 0:
        msg = ""
        for subdir in 'indata', 'docs':
            # ADM grab the docs dir when working with the indata dir
            # ADM and vice versa.
            otherdir = list(setdic.keys())
            otherdir.remove(subdir)
            # ADM print which files were missed in this specific dir.
            missed = list(missing.intersection(setdic[subdir]))
            msg += '{} file(s) not in {}: {}\n'.format(
                dirdic[subdir], dirdic[otherdir[0]], missed)
        log.critical(msg)
        raise ValueError(msg)

    return


def read_files(scxdir, scnd_mask):
    """Read in all secondary files and concatenate them into one array.

    Parameters
    ----------
    scxdir : :class:`str`
        Directory produced by :func:`~secondary._check_files()`.
    scnd_mask : :class:`desiutil.bitmask.BitMask`, optional
        A mask corresponding to a set of secondary targets, e.g, could
        be ``from desitarget.targetmask import scnd_mask`` for the
        main survey mask.

    Returns
    -------
    :class:`~numpy.ndarray`
        All secondary targets concatenated as one array with columns
        that correspond to `desitarget.secondary.outdatamodel`.
    """
    # ADM the full directory name for the input data files.
    fulldir = os.path.join(scxdir, 'indata')

    scxall = []
    # ADM loop through all of the scx bits.
    for name in scnd_mask.names():
        log.debug('SCND target: {}'.format(name))
        # ADM the full file path without the extension.
        fn = os.path.join(fulldir, scnd_mask[name].filename)
        log.debug('     path:   {}'.format(fn))
        # ADM if the relevant file is a .txt file, read it in.
        if os.path.exists(fn+'.txt'):
            try:
                scxin = np.loadtxt(fn+'.txt', usecols=[0, 1, 2, 3, 4, 5],
                                   dtype=indatamodel.dtype)
            except (ValueError, IndexError):
                msg = "First 6 columns don't correspond to {} in {}.txt".format(
                    indatamodel.dtype, fn)
                # ADM perhaps people provided .csv files as .txt files.
                try:
                    scxin = np.loadtxt(fn+'.txt', usecols=[0, 1, 2, 3, 4, 5],
                                       dtype=indatamodel.dtype, delimiter=",")
                except (ValueError, IndexError):
                    log.error(msg)
                    raise IOError(msg)

        # ADM otherwise it's a fits file, read it in.
        else:
            scxin = fitsio.read(fn+'.fits',
                                columns=indatamodel.dtype.names)

        # ADM ensure this is a properly constructed numpy array.
        scxin = np.atleast_1d(scxin)

        # ADM assert the data model.
        msg = "Data model doesn't match {} in {}".format(indatamodel.dtype, fn)
        for col in indatamodel.dtype.names:
            assert scxin[col].dtype == indatamodel[col].dtype, msg

        # ADM check RA/Dec are reasonable.
        outofbounds = ((scxin["RA"] >= 360.) | (scxin["RA"] < 0) |
                       (scxin["DEC"] > 90) | (scxin["DEC"] < -90))
        if np.any(outofbounds):
            msg = "RA/Dec outside of range in {}; RA={}, Dec={}".format(
                fn, scxin["RA"][outofbounds], scxin["DEC"][outofbounds])
            log.error(msg)
            raise IOError(msg)

        # ADM the default is 2015.5 for the REF_EPOCH.
        ii = scxin["REF_EPOCH"] == 0
        scxin["REF_EPOCH"][ii] = 2015.5

        # ADM add the other output columns.
        dt = outdatamodel.dtype.descr + suppdatamodel.dtype.descr
        scxout = np.zeros(len(scxin), dtype=dt)
        for col in indatamodel.dtype.names:
            scxout[col] = scxin[col]
        scxout["SCND_TARGET"] = scnd_mask[name]
        scxout["SCND_TARGET_INIT"] = scnd_mask[name]
        scxout["SCND_ORDER"] = np.arange(len(scxin))
        scxout["PRIORITY_INIT"] = scnd_mask[name].priorities['UNOBS']
        scxout["NUMOBS_INIT"] = scnd_mask[name].numobs
        scxout["TARGETID"] = -1
        scxout["OBSCONDITIONS"] =     \
            obsconditions.mask(scnd_mask[name].obsconditions)
        scxout["PRIM_MATCH"] = False
        scxall.append(scxout)

    return np.concatenate(scxall)


def add_primary_info(scxtargs, priminfodir):
    """Add TARGETIDs to secondaries from directory of primary matches.

    Parameters
    ----------
    scxtargs : :class:`~numpy.ndarray`
        An array of secondary targets, must contain the columns
        `SCND_TARGET`, `SCND_ORDER` and `TARGETID`.
    priminfodir : :class:`list` or `str`
        Location of the directory that has previously matched primary
        and secondary targets to recover the unique primary TARGETIDs.

    Returns
    -------
    :class:`~numpy.ndarray`
        The array of secondary targets, with the `TARGETID` column
        populated for matches to secondary targets

    Notes
    -----
        - The input `scxtargs` is modified, so be careful to make
          a copy if you want that variable to remain unchanged!
    """
    log.info("Begin matching primaries in {} to secondaries...t={:.1f}s"
             .format(priminfodir, time()-start))
    # ADM read all of the files from the priminfodir.
    primfns = glob(os.path.join(priminfodir, "*fits"))
    # ADM if there are no files, there were no primary matches
    # ADM and we can just return.
    if len(primfns) == 0:
        log.warning("No secondary target matches a primary target!!!")
        return scxtargs

    log.info("Begin reading files from {}...t={:.1f}s".format(
        priminfodir, time()-start))
    primtargs = []
    for primfn in primfns:
        prim = fitsio.read(primfn)
        primtargs.append(prim)
    primtargs = np.concatenate(primtargs)
    log.info("Done reading files...t={:.1f}s".format(time()-start))

    # ADM make a unique look-up for the target sets.
    scxbitnum = np.log2(scxtargs["SCND_TARGET"]).astype('int')
    primbitnum = np.log2(primtargs["SCND_TARGET"]).astype('int')
    # ADM SCND_ORDER can potentially run into the tens-of-millions, so
    # ADM we need to use int64 type to get as high as 1000 x SCND_ORDER.
    scxids = 1000 * scxtargs["SCND_ORDER"].astype('int64') + scxbitnum
    primids = 1000 * primtargs["SCND_ORDER"].astype('int64') + primbitnum

    # ADM matches could have occurred ACROSS HEALPixels, producing
    # ADM duplicated secondary targets with different TARGETIDs...
    alldups = []
    for _, dups in duplicates(primids):
        # ADM...resolve these on which matched a primary target (rather
        # ADM than just a source from a sweeps files) and then ALSO on
        # ADM which has the highest priority. The combination in the code
        # ADM is, e.g., True/False (meaning 1/0) * PRIORITY_INIT.
        am = np.argmax(primtargs[dups]["PRIM_MATCH"]*primtargs[dups]["PRIORITY_INIT"])
        dups = np.delete(dups, am)
        alldups.append(dups)
    # ADM catch cases where there are no duplicates.
    if len(alldups) != 0:
        alldups = np.hstack(alldups)
        primtargs = np.delete(primtargs, alldups)
        primids = np.delete(primids, alldups)
    log.info("Remove {} cases where a secondary matched several primaries".
             format(len(alldups)))

    # ADM we already know that all primaries match a secondary, so,
    # ADM for speed, we can reduce to the matching set.
    sprimids = set(primids)
    scxii = [scxid in sprimids for scxid in scxids]
    assert len(sprimids) == len(primids)
    assert set(scxids[scxii]) == sprimids

    # ADM sort-to-match sxcid and primid.
    primii = np.zeros_like(primids)
    primii[np.argsort(scxids[scxii])] = np.argsort(primids)
    assert np.all(primids[primii] == scxids[scxii])

    # ADM now we have the matches, update the secondary targets
    # ADM with the primary TARGETIDs.
    scxtargs["TARGETID"][scxii] = primtargs["TARGETID"][primii]
    scxtargs["PRIM_MATCH"][scxii] = primtargs["PRIM_MATCH"][primii]

    # APC Secondary targets that don't match to a primary target.
    # APC all still have TARGETID = -1 at this point. They
    # APC get removed in finalize_secondary().
    log.info("Done matching primaries in {} to secondaries...t={:.1f}s"
             .format(priminfodir, time()-start))

    return scxtargs


def match_secondary(primtargs, scxdir, scndout, sep=1.,
                    pix=None, nside=None, swfiles=None):
    """Match secondary targets to primary targets and update bits.

    Parameters
    ----------
    primtargs : :class:`~numpy.ndarray`
        An array of primary targets.
    scndout : :class`~numpy.ndarray`
        Name of a sub-directory to which to write the information in
        `desitarget.secondary.outdatamodel` with `TARGETID` and (the
        highest) `PRIORITY_INIT` updated with matching primary info.
    scxdir : :class:`str`, optional, defaults to `None`
        Name of the directory that hosts secondary targets.
    sep : :class:`float`, defaults to 1 arcsecond
        The separation at which to match in ARCSECONDS.
    pix : :class:`list`, optional, defaults to `None`
        Limit secondary targets to (NESTED) HEALpixels that touch
        pix at the supplied `nside`, as a speed-up.
    nside : :class:`int`, optional, defaults to `None`
        The (NESTED) HEALPixel nside to be used with `pixlist`.
    swfiles : :class:`list`, optional, defaults to `None`
        A list of files (typically sweep files). If passed and not `None`
        then once all of the primary TARGETS have been matched and the
        relevant bit information updated, use these files to find
        additional sources from which to derive a primary TARGETID.

    Returns
    -------
    :class:`~numpy.ndarray`
        The array of primary targets, with the `SCND_TARGET` bit
        populated for matches to secondary targets
    """
    # ADM add a SCND_TARGET column to the primary targets.
    dt = primtargs.dtype.descr
    dt.append(('SCND_TARGET', '>i8'))
    targs = np.zeros(len(primtargs), dtype=dt)
    for col in primtargs.dtype.names:
        targs[col] = primtargs[col]

    # ADM check if this is an SV or main survey file.
    cols, mx, surv = main_cmx_or_sv(targs, scnd=True)
    log.info('running on the {} survey...'.format(surv))
    if surv != 'main':
        scxdir = os.path.join(scxdir, surv)

    # ADM read in non-OVERRIDE secondary targets.
    scxtargs = read_files(scxdir, mx[3])
    scxtargs = scxtargs[~scxtargs["OVERRIDE"]]

    # ADM match primary targets to non-OVERRIDE secondary targets.
    inhp = np.ones(len(scxtargs), dtype="?")
    # ADM as a speed-up, save memory by limiting the secondary targets
    # ADM to just HEALPixels that could touch the primary targets.
    if nside is not None and pix is not None:
        # ADM remember to grab adjacent pixels in case of edge effects.
        allpix = add_hp_neighbors(nside, pix)
        inhp = is_in_hp(scxtargs, nside, allpix)
        # ADM it's unlikely that the matching separation is comparable
        # ADM to the HEALPixel resolution, but guard against that anyway.
        halfpix = np.degrees(hp.max_pixrad(nside))*3600.
        if sep > halfpix:
            msg = 'sep ({}") exceeds (half) HEALPixel size ({}")'.format(
                sep, halfpix)
            log.critical(msg)
            raise ValueError(msg)

    # ADM warn the user if the secondary and primary samples are "large".
    big = 1e6
    if np.sum(inhp) > big and len(primtargs) > big:
        log.warning('Large secondary (N={}) and primary (N={}) samples'
                    .format(np.sum(inhp), len(primtargs)))
        log.warning('The code may run slowly')

    # ADM for each secondary target, determine if there is a match
    # ADM with a primary target. Note that sense is important, here
    # ADM (the primary targets must be passed first).
    log.info('Matching primary and secondary targets for {} at {}"...t={:.1f}s'
             .format(scndout, sep, time()-start))
    mtargs, mscx = radec_match_to(targs, scxtargs[inhp], sep=sep)
    # ADM recast the indices to the full set of secondary targets,
    # ADM instead of just those that were in the relevant HEALPixels.
    mscx = np.where(inhp)[0][mscx]

    # ADM loop through the matches and update the SCND_TARGET
    # ADM column in the primary target list. The np.unique is a
    # ADM speed-up to assign singular matches first.
    umtargs, inv, cnt = np.unique(mtargs,
                                  return_inverse=True, return_counts=True)
    # ADM number of times each primary target was matched, ordered
    # ADM the same as mtargs, i.e. n(mtargs) for each entry in mtargs.
    nmtargs = cnt[inv]
    # ADM assign anything with nmtargs = 1 directly.
    singular = nmtargs == 1
    targs["SCND_TARGET"][mtargs[singular]] = scxtargs["SCND_TARGET"][mscx[singular]]
    # ADM loop through things with nmtargs > 1 and combine the bits.
    for i in range(len((mtargs[~singular]))):
        targs["SCND_TARGET"][mtargs[~singular][i]] |= scxtargs["SCND_TARGET"][mscx[~singular][i]]
    # ADM also assign the SCND_ANY bit to the primary targets.
    desicols, desimasks, _ = main_cmx_or_sv(targs, scnd=True)
    desi_mask = desimasks[0]

    targs[desicols[0]][umtargs] |= desi_mask.SCND_ANY

    # ADM rename the SCND_TARGET column, in case this is an SV file.
    targs = rfn.rename_fields(targs, {'SCND_TARGET': desicols[3]})

    # APC Secondary target bits only affect PRIORITY, NUMOBS and
    # APC obsconditions for specific DESI_TARGET bits
    # APC See https://github.com/desihub/desitarget/pull/530

    # APC Only consider primary targets with secondary bits set
    scnd_update = (targs[desicols[0]] & desi_mask['SCND_ANY']) != 0
    if np.any(scnd_update):
        # APC Allow changes to primaries if the DESI_TARGET bitmask has
        # APC only the following bits set, in any combination.
        log.info('Test if secondaries can update {} matched primaries'.format(
            scnd_update.sum()))
        update_from_scnd_bits = (desi_mask['SCND_ANY'] | desi_mask['MWS_ANY'] |
                                 desi_mask['STD_BRIGHT'] | desi_mask['STD_FAINT']
                                 | desi_mask['STD_WD'])
        scnd_update &= ((targs[desicols[0]] & ~update_from_scnd_bits) == 0)
        log.info('New priority, numobs, obscon for {} matched primaries'.format(
            scnd_update.sum()))

        # APC Primary and secondary obsconditions are or'd
        scnd_obscon = set_obsconditions(targs[scnd_update], scnd=True)
        targs['OBSCONDITIONS'][scnd_update] &= scnd_obscon

        # APC bit of a hack here
        # APC Check for _BRIGHT, _DARK split in column names
        darkbright = 'NUMOBS_INIT_DARK' in targs.dtype.names
        if darkbright:
            ender, obscon = ["_DARK", "_BRIGHT"], ["DARK|GRAY", "BRIGHT"]
        else:
            ender, obscon = [""], ["DARK|GRAY|BRIGHT|POOR|TWILIGHT12|TWILIGHT18"]

        # APC secondaries can increase priority and numobs
        for edr, oc in zip(ender, obscon):
            pc, nc = "PRIORITY_INIT"+edr, "NUMOBS_INIT"+edr
            scnd_priority, scnd_numobs = initial_priority_numobs(
                targs[scnd_update], obscon=oc, scnd=True)
            targs[nc][scnd_update] = np.maximum(
                targs[nc][scnd_update], scnd_numobs)
            targs[pc][scnd_update] = np.maximum(
                targs[pc][scnd_update], scnd_priority)

    # ADM update the secondary targets with the primary information.
    scxtargs["TARGETID"][mscx] = targs["TARGETID"][mtargs]
    # ADM the maximum priority will be used to break ties in the
    # ADM unlikely event that a secondary matches two primaries.
    hipri = np.maximum(targs["PRIORITY_INIT_DARK"],
                       targs["PRIORITY_INIT_BRIGHT"])
    scxtargs["PRIORITY_INIT"][mscx] = hipri[mtargs]
    # ADM record that we have a match to a primary.
    scxtargs["PRIM_MATCH"][mscx] = True

    # ADM now we're done matching the primary and secondary targets, also
    # ADM match the secondary targets to sweep files, if passed, to find
    # ADM TARGETIDs.
    notid = scxtargs["TARGETID"] == -1
    if swfiles is not None and np.sum(notid) > 0:
        log.info('Reading input sweep files...t={:.1f}s'.format(time()-start))
        # ADM first read in all of the sweeps files.
        swobjs = []
        for ifil, swfile in enumerate(swfiles):
            swobj = fitsio.read(swfile, columns=["RELEASE", "BRICKID", "OBJID",
                                                 "RA", "DEC"])
            # ADM limit to just sources in the healpix of interest.
            # ADM remembering to grab adjacent pixels for edge effects.
            inhp = np.ones(len(swobj), dtype="?")
            if nside is not None and pix is not None:
                inhp = is_in_hp(swobj, nside, allpix)
            swobjs.append(swobj[inhp])
            log.info("Read {} sources from {}/{} sweep files...t={:.1f}s".format(
                np.sum(inhp), ifil+1, len(swfiles), time()-start))
        swobjs = np.concatenate(swobjs)
        # ADM resolve so there are no duplicates across the N/S boundary.
        swobjs = resolve(swobjs)
        log.info("Total sources read: {}".format(len(swobjs)))

        # ADM continue if there are sources in the pixels of interest.
        if len(swobjs) > 0:
            # ADM limit to just secondaries in the healpix of interest.
            inhp = np.ones(len(scxtargs), dtype="?")
            if nside is not None and pix is not None:
                inhp = is_in_hp(scxtargs, nside, pix)

            # ADM now perform the match.
            log.info('Matching secondary targets to sweep files...t={:.1f}s'
                     .format(time()-start))
            mswobjs, mscx = radec_match_to(swobjs,
                                           scxtargs[inhp & notid], sep=sep)
            # ADM recast the indices to the full set of secondaries,
            # ADM instead of just those that were in the relevant pixels.
            mscx = np.where(inhp & notid)[0][mscx]
            log.info('Found {} additional matches...t={:.1f}s'.format(
                len(mscx), time()-start))

            if len(mscx) > 0:
                # ADM construct the targetid from the sweeps information.
                targetid = encode_targetid(objid=swobjs['OBJID'],
                                           brickid=swobjs['BRICKID'],
                                           release=swobjs['RELEASE'])
                # ADM and add the targetid to the secondary targets.
                scxtargs["TARGETID"][mscx] = targetid[mswobjs]

    # ADM write the secondary targets that have updated TARGETIDs.
    ii = scxtargs["TARGETID"] != -1
    nmatches = np.sum(ii)
    log.info('Writing {} secondary target matches to {}...t={:.1f}s'
             .format(nmatches, scndout, time()-start))
    if nmatches > 0:
        hdr = fitsio.FITSHDR()
        hdr["SURVEY"] = surv
        fitsio.write(scndout, scxtargs[ii],
                     extname='SCND_TARG', header=hdr, clobber=True)

    log.info('Done...t={:.1f}s'.format(time()-start))

    return targs


def finalize_secondary(scxtargs, scnd_mask, survey='main', sep=1.,
                       darkbright=False):
    """Assign secondary targets a realistic TARGETID, finalize columns.

    Parameters
    ----------
    scxtargs : :class:`~numpy.ndarray`
        An array of secondary targets, must contain the columns `RA`,
        `DEC` and `TARGETID`. `TARGETID` should be -1 for objects
        that lack a `TARGETID`.
    scnd_mask : :class:`desiutil.bitmask.BitMask`
        A mask corresponding to a set of secondary targets, e.g, could
        be ``from desitarget.targetmask import scnd_mask`` for the
        main survey mask.
    survey : :class:`str`, optional, defaults to "main"
        string indicating whether we are working in the context of the
        Main Survey (`main`) or SV (e.g. `sv1`, `sv2` etc.). Used to
        set the `RELEASE` number in the `TARGETID` (see Notes).
    sep : :class:`float`, defaults to 1 arcsecond
        The separation at which to match secondary targets to
        themselves in ARCSECONDS.
    darkbright : :class:`bool`, optional, defaults to ``False``
        If sent, then split `NUMOBS_INIT` and `PRIORITY_INIT` into
        `NUMOBS_INIT_DARK`, `NUMOBS_INIT_BRIGHT`, `PRIORITY_INIT_DARK`
        and `PRIORITY_INIT_BRIGHT` and calculate values appropriate
        to "BRIGHT" and "DARK|GRAY" observing conditions.

    Returns
    -------
    :class:`~numpy.ndarray`
        The array of secondary targets, with the `TARGETID` bit updated
        to be unique and reasonable and the `SCND_TARGET` column renamed
        based on the flavor of `scnd_mask`.

    Notes
    -----
        - Secondaries without `OVERRIDE` are also matched to themselves
        Such matches are given the same `TARGETID` (that of the primary
        if they match a primary) and the bitwise or of `SCND_TARGET` and
        `OBSCONDITIONS` bits across matches. The highest `PRIORITY_INIT`
        is retained, and others are set to -1. Only secondaries with
        priorities that are not -1 are written to the main file. If
        multiple matching secondary targets have the same (highest)
        priority, the first one encountered retains its `PRIORITY_INIT`
        - The secondary `TARGETID` is designed to be reproducible. It
        combines `BRICKID` based on location, `OBJID` based on the
        order of the targets in the secondary file (`SCND_ORDER`) and
        `RELEASE` from the secondary bit number (`SCND_TARGET`) and the
        input `survey`. `RELEASE` is set to ((X-1)*100)+np.log2(scnd_bit)
        with X from the `survey` string survey=svX and scnd_bit from
        `SCND_TARGET`. For the main survey (survey="main") X-1 is 5.

    Notes
    -----
        - The input `scxtargs` is modified, so be careful to make
          a copy if you want that variable to remain unchanged!
    """
    # ADM assign new TARGETIDs to targets without a primary match.
    nomatch = scxtargs["TARGETID"] == -1

    # ADM get the BRICKIDs for each source.
    brxid = bricks.brickid(scxtargs["RA"], scxtargs["DEC"])

    # ADM ensure unique secondary bits for different iterations of SV
    # ADM and the Main Survey.
    if survey == 'main':
        Xm1 = 5
    elif survey[0:2] == 'sv':
        # ADM the re.search just extracts the numbers in the string.
        Xm1 = int(re.search(r'\d+', survey).group())-1
        # ADM we've allowed a max of up to sv5 (!). Fail if surpassed.
        if Xm1 >= 5:
            msg = "Only coded for up to 'sv5', not {}!!!".format(survey)
            log.critical(msg)
            raise ValueError(msg)
    else:
        msg = "allowed surveys: 'main', 'svX', not {}!!!".format(survey)
        log.critical(msg)
        raise ValueError(msg)

    # ADM the RELEASE for each source is the `SCND_TARGET` bit NUMBER.
    release = (Xm1*100)+np.log2(scxtargs["SCND_TARGET_INIT"]).astype('int')

    # ADM build the OBJIDs based on the values of SCND_ORDER.
    t0 = time()
    log.info("Begin assigning OBJIDs to bricks...")
    # ADM So as not to overwhelm the bit-limits for OBJID
    # ADM rank by SCND_ORDER for each brick and bit combination.
    # ADM First, create a unique ID based on brxid and release.
    scnd_order = scxtargs["SCND_ORDER"]
    sorter = (1000*brxid) + release
    # ADM sort the unique IDs and split based on where they change.
    argsort = np.argsort(sorter)
    w = np.where(np.diff(sorter[argsort]))[0]
    soperbrxbit = np.split(scnd_order[argsort], w+1)
    # ADM loop through each (brxid, release) and sort on scnd_order.
    # ADM double argsort returns the ascending ranked order of the entry
    # ADM (whereas a single argsort returns the indexes for ordering).
    sortperbrxbit = [np.argsort(np.argsort(so)) for so in soperbrxbit]
    # ADM finally unroll the (brxid, release) combinations...
    sortedobjid = np.array(list(itertools.chain.from_iterable(sortperbrxbit)))
    # ADM ...and reorder based on the initial argsort.
    objid = np.zeros_like(sortedobjid)-1
    objid[argsort] = sortedobjid
    log.info("Assigned OBJIDs to bricks in {:.1f}s".format(time()-t0))

    # ADM check that the objid array was entirely populated.
    assert np.all(objid != -1)

    # ADM assemble the TARGETID, SCND objects.
    targetid = encode_targetid(objid=objid, brickid=brxid, release=release)

    # ADM a check that the generated TARGETIDs are unique.
    if len(set(targetid)) != len(targetid):
        msg = "duplicate TARGETIDs generated for secondary targets!!!"
        log.critical(msg)
        raise ValueError(msg)

    # ADM assign the unique TARGETIDs to the secondary objects.
    scxtargs["TARGETID"][nomatch] = targetid[nomatch]
    log.debug("Assigned {} targetids to unmatched secondaries".format(
        len(targetid[nomatch])))

    # ADM match secondaries to themselves, to ensure duplicates
    # ADM share a TARGETID. Don't match special (OVERRIDE) targets
    # ADM or sources that have already been matched to a primary.
    w = np.where(~scxtargs["OVERRIDE"] & nomatch)[0]
    if len(w) > 0:
        log.info("Matching {} secondary targets to themselves...t={:.1f}s"
                 .format(len(scxtargs), time()-t0))
        # ADM use astropy for the matching. At NERSC, astropy matches
        # ADM ~20M objects to themselves in about 10 minutes.
        c = SkyCoord(scxtargs["RA"][w]*u.deg, scxtargs["DEC"][w]*u.deg)
        m1, m2, _, _ = c.search_around_sky(c, sep*u.arcsec)
        log.info("Done with matching...t={:.1f}s".format(time()-t0))
        # ADM restrict only to unique matches (and exclude self-matches).
        uniq = m1 > m2
        m1, m2 = m1[uniq], m2[uniq]
        # ADM set same TARGETID for any matches. m2 must come first, here.
        scxtargs["TARGETID"][w[m2]] = scxtargs["TARGETID"][w[m1]]

    # ADM Ensure secondary targets with matching TARGETIDs have all the
    # ADM relevant SCND_TARGET bits set. By definition, targets with
    # ADM OVERRIDE set never have matching TARGETIDs.
    wnoov = np.where(~scxtargs["OVERRIDE"])[0]
    if len(wnoov) > 0:
        for _, inds in duplicates(scxtargs["TARGETID"][wnoov]):
            scnd_targ = 0
            for ind in inds:
                scnd_targ |= scxtargs["SCND_TARGET"][wnoov[ind]]
            scxtargs["SCND_TARGET"][wnoov[inds]] = scnd_targ
    log.info("Done checking SCND_TARGET...t={:.1f}s".format(time()-t0))

    # ADM change the data model depending on whether the mask
    # ADM is an SVX (X = 1, 2, etc.) mask or not. Nothing will
    # ADM change if the mask has no preamble.
    prepend = scnd_mask._name[:-9].upper()
    scxtargs = rfn.rename_fields(
        scxtargs, {'SCND_TARGET': prepend+'SCND_TARGET'}
    )

    # APC same thing for DESI_TARGET
    scxtargs = rfn.rename_fields(
        scxtargs, {'DESI_TARGET': prepend+'DESI_TARGET'}
    )

    # APC Remove duplicate targetids from secondary-only targets
    alldups = []
    for _, dups in duplicates(scxtargs['TARGETID']):
        # Retain the duplicate with highest priority, breaking ties
        # on lowest index in list of duplicates
        dups = np.delete(dups, np.argmax(scxtargs['PRIORITY_INIT'][dups]))
        alldups.append(dups)
    # ADM guard against the case that there are no duplicates.
    if len(alldups) == 0:
        alldups = [alldups]
    alldups = np.hstack(alldups)
    log.debug("Flagging {} duplicate secondary targetids with PRIORITY_INIT=-1".format(len(alldups)))

    # ADM and remove the INIT fields in prep for a dark/bright split.
    scxtargs = rfn.drop_fields(scxtargs, ["PRIORITY_INIT", "NUMOBS_INIT"])

    # ADM set initial priorities, numobs and obsconditions for both
    # ADM BRIGHT and DARK|GRAY conditions, if requested.
    nscx = len(scxtargs)
    nodata = np.zeros(nscx, dtype='int')-1
    if darkbright:
        ender, obscon = ["_DARK", "_BRIGHT"], ["DARK|GRAY", "BRIGHT"]
    else:
        ender, obscon = [""], ["DARK|GRAY|BRIGHT|POOR|TWILIGHT12|TWILIGHT18"]
    cols, vals, forms = [], [], []
    for edr, oc in zip(ender, obscon):
        cols += ["{}_INIT{}".format(pn, edr) for pn in ["PRIORITY", "NUMOBS"]]
        vals += [nodata, nodata]
        forms += ['>i8', '>i8']

    # ADM write the output array.
    newdt = [dt for dt in zip(cols, forms)]
    done = np.array(np.zeros(nscx), dtype=scxtargs.dtype.descr+newdt)
    for col in scxtargs.dtype.names:
        done[col] = scxtargs[col]
    for col, val in zip(cols, vals):
        done[col] = val

    # ADM add the actual PRIORITY/NUMOBS values.
    for edr, oc in zip(ender, obscon):
        pc, nc = "PRIORITY_INIT"+edr, "NUMOBS_INIT"+edr
        done[pc], done[nc] = initial_priority_numobs(done, obscon=oc, scnd=True)

        # APC Flagged duplicates are removed in io.write_secondary
        if len(alldups) > 0:
            done[pc][alldups] = -1

    # APC add secondary flag in DESI_TARGET
    cols, mx, surv = main_cmx_or_sv(done, scnd=True)
    done[cols[0]] = mx[0]['SCND_ANY']

    # ADM set the OBSCONDITIONS.
    done["OBSCONDITIONS"] = set_obsconditions(done, scnd=True)

    return done


def select_secondary(priminfodir, sep=1., scxdir=None, darkbright=False):
    """Process secondary targets and update relevant bits.

    Parameters
    ----------
    priminfodir : :class:`list` or `str`
        Location of the directory that has previously matched primary
        and secondary targets to recover the unique primary TARGETIDs.
        The first file in this directory should have a header keyword
        SURVEY indicating whether we are working in the context of the
        Main Survey (`main`) or SV (e.g. `sv1`, `sv2` etc.).
    sep : :class:`float`, defaults to 1 arcsecond
        The separation at which to match in ARCSECONDS.
    scxdir : :class:`str`, optional, defaults to :envvar:`SCND_DIR`
        The name of the directory that hosts secondary targets.
    darkbright : :class:`bool`, optional, defaults to ``False``
        If sent, then split `NUMOBS_INIT` and `PRIORITY_INIT` into
        `NUMOBS_INIT_DARK`, `NUMOBS_INIT_BRIGHT`, `PRIORITY_INIT_DARK`
        and `PRIORITY_INIT_BRIGHT` and calculate values appropriate
        to "BRIGHT" and "DARK|GRAY" observing conditions.

    Returns
    -------
    :class:`~numpy.ndarray`
        All secondary targets from `scxdir` with columns ``TARGETID``,
        ``SCND_TARGET``, ``PRIORITY_INIT``, ``SUBPRIORITY`` and
        ``NUMOBS_INIT`` added. These columns are also populated,
        excepting ``SUBPRIORITY``.
    """
    # ADM Sanity check that priminfodir exists.
    if not os.path.exists(priminfodir):
        msg = "{} doesn't exist".format(priminfodir)
        log.critical(msg)
        raise ValueError(msg)

    # ADM read in the SURVEY from the first file in priminfodir.
    fns = glob(os.path.join(priminfodir, "*fits"))
    hdr = fitsio.read_header(fns[0], 'SCND_TARG')
    survey = hdr["SURVEY"].rstrip()

    # ADM load the correct mask.
    from desitarget.targetmask import scnd_mask
    if survey[:2] == 'sv':
        try:
            targmask = import_module("desitarget.{}.{}_targetmask".format(
                survey, survey))
        except ModuleNotFoundError:
            msg = 'Bitmask yaml does not exist for survey type {}'.format(survey)
            log.critical(msg)
            raise ModuleNotFoundError(msg)
        scnd_mask = targmask.scnd_mask
    elif survey != 'main':
        msg = "allowed surveys: 'main', 'svX', not {}!!!".format(survey)
        log.critical(msg)
        raise ValueError(msg)

    log.info("Reading secondary files...t={:.1f}m".format((time()-start)/60.))
    # ADM retrieve the scxdir, check it's structure and fidelity...
    scxdir = _get_scxdir(scxdir)
    _check_files(scxdir, scnd_mask)
    # ADM ...and read in all of the secondary targets.
    scxtargs = read_files(scxdir, scnd_mask)

    # ADM only non-override targets could match a primary.
    scxover = scxtargs[scxtargs["OVERRIDE"]]
    scxtargs = scxtargs[~scxtargs["OVERRIDE"]]

    log.info("Adding primary TARGETIDs...t={:.1f}m".format((time()-start)/60.))
    # ADM add in the primary TARGETIDs where we have them.
    scxtargs = add_primary_info(scxtargs, priminfodir)

    # ADM now we're done matching, bring the override targets back...
    scxout = np.concatenate([scxtargs, scxover])

    log.info("Finalizing secondaries...t={:.1f}m".format((time()-start)/60.))
    # ADM assign TARGETIDs to secondaries that did not match a primary.
    scxout = finalize_secondary(scxout, scnd_mask, survey=survey,
                                sep=sep, darkbright=darkbright)

    return scxout
