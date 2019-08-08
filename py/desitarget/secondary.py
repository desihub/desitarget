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
import fitsio
import itertools
import numpy as np
import healpy as hp

import numpy.lib.recfunctions as rfn

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Row

from time import time

from collections import defaultdict

from desitarget.internal import sharedmem
from desitarget.geomask import radec_match_to, add_hp_neighbors, is_in_hp

from desitarget.targets import encode_targetid, main_cmx_or_sv
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
#                or sv1/data/sv1_targetmask.yaml (scnd_mask).
# ADM Note that TARGETID for secondary-only targets is unique because
# ADM RELEASE is 0 for secondary-only targets.
outdatamodel = np.array([], dtype=[
    ('RA', '>f8'), ('DEC', '>f8'), ('PMRA', '>f4'), ('PMDEC', '>f4'),
    ('REF_EPOCH', '>f4'), ('OVERRIDE', '?'),
    ('TARGETID', '>i8'), ('SCND_TARGET', '>i8'),
    ('PRIORITY_INIT', '>i8'), ('SUBPRIORITY', '>f8'),
    ('NUMOBS_INIT', '>i8'), ('OBSCONDITIONS', '>i8')
])

# ADM extra columns that are used during processing but are
# ADM not an official part of the input or output data model.
suppdatamodel = np.array([], dtype=[
    ('SCND_TARGET_INIT', '>i8'), ('SCND_ORDER', '>i4')
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


def _get_scxdir(scxdir=None):
    """Retrieve the base secondary directory with error checking.

    Parameters
    ----------
    scxdir : :class:`str`, optional, defaults to :envvar:`SCND_DIR`
        Directory containing secondary target files to which to match.
        If not specified, the directory is taken to be the value of
        the :envvar:`SCND_DIR` environment variable.

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
        log.info('pass scxdir or set $SCND_DIR...')
        msg = 'Secondary target files not found in {}'.format(scxdir)
        log.critical(msg)
        raise ValueError(msg)

    # ADM also fail if the indata, outdata and docs directories don't
    # ADM exist in the secondary directory.
    for subdir in "docs", "indata", "outdata":
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
        # ADM retrieve the full file names.
        fnswext = os.listdir(os.path.join(scxdir, subdir))
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


def read_files(scxdir, scnd_mask,
               obscon="DARK|GRAY|BRIGHT|POOR|TWILIGHT12|TWILIGHT18"):
    """Read in all secondary files and concatenate them into one array.

    Parameters
    ----------
    scxdir : :class:`str`
        Directory produced by :func:`~secondary._check_files()`.
    scnd_mask : :class:`desiutil.bitmask.BitMask`, optional
        A mask corresponding to a set of secondary targets, e.g, could
        be ``from desitarget.targetmask import scnd_mask`` for the
        main survey mask. Defaults to the main survey mask.
    obscon : :class:`str`, optional, defaults to almost all conditions
        An OBSCONDITIONS string that can be understood by the desitarget
        mask parser (e.g. 'GRAY|DARK'). Only secondary targets that have
        a match to these OBSCONDITIONS in the yaml file are processed.

    Returns
    -------
    :class:`~numpy.ndarray`
        All secondary targets concatenated as one array with columns
        that correspond to `desitarget.secondary.outdatamodel`.
    """
    # ADM the bits that correspond to the passed obscon.
    goodobsbits = obsconditions.mask(obscon)

    # ADM the full directory name for the input data files.
    fulldir = os.path.join(scxdir, 'indata')

    scxall = []
    # ADM loop through all of the scx bits.
    for name in scnd_mask.names():
        # ADM only process if the conditions are correct.
        obsbits = obsconditions.mask(scnd_mask[name].obsconditions)
        if (goodobsbits & obsbits) != 0:
            # ADM the full file path without the extension.
            fn = os.path.join(fulldir, scnd_mask[name].filename)
            # ADM if the relevant file is a .txt file, read it in.
            if os.path.exists(fn+'.txt'):
                scxin = np.loadtxt(fn+'.txt', usecols=[0, 1, 2, 3, 4, 5],
                                   dtype=indatamodel.dtype)
            # ADM otherwise it's a fits file, read it in.
            else:
                scxin = fitsio.read(fn+'.fits',
                                    columns=indatamodel.dtype.names)

            # ADM ensure this is a properly constructed numpy array.
            scxin = np.atleast_1d(scxin)

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

            scxall.append(scxout)

    return np.concatenate(scxall)


def match_secondary(infile, scxtargs, sep=1., scxdir=None):
    """Match secondary targets to primary targets and update bits.

    Parameters
    ----------
    infile : :class:`str`
        The full path to a file containing primary targets.
    scxtargs : :class:`~numpy.ndarray`
        An array of secondary targets.
    sep : :class:`float`, defaults to 1 arcsecond
        The separation at which to match in ARCSECONDS.
    scxdir : :class:`str`, optional, defaults to `None`
        Name of the directory that hosts secondary targets. If passed,
        this is written to the output primary file header as `SCNDDIR`.

    Returns
    -------
    :class:`~numpy.ndarray`
        The array of secondary targets, with the `TARGETID` bit
        updated with any matching primary targets from `infile`.

    Notes
    -----
        - The primary target `infiles` are written back to their original
          path with `FILE.fits` changed to `wscnd/FILE-wscnd.fits` and
          the `SCND_TARGET` bit populated for matching targets.
    """
    # ADM just the file name for logging.
    fn = os.path.basename(infile)
    # ADM read in the primary targets.
    log.info('Reading primary targets file {}...t={:.1f}s'
             .format(infile, time()-start))
    intargs, hdr = fitsio.read(infile, extension="TARGETS", header=True)

    # ADM fail if file's already been matched to secondary targets.
    if "SCNDDIR" in hdr:
        msg = "{} already matched to secondary targets".format(fn) \
              + " (did you mean to remove {}?)!!!".format(fn)
        log.critical(msg)
        raise ValueError(msg)
    # ADM add the SCNDDIR to the primary targets file header.
    hdr["SCNDDIR"] = scxdir
    # ADM add a SCND_TARGET column to the primary targets.
    dt = intargs.dtype.descr
    dt.append(('SCND_TARGET', '>i8'))
    targs = np.zeros(len(intargs), dtype=dt)
    for col in intargs.dtype.names:
        targs[col] = intargs[col]

    # ADM match to all secondary targets for non-custom primary files.
    inhp = np.ones(len(scxtargs), dtype="?")
    # ADM as a speed-up, save memory by limiting the secondary targets
    # ADM to just HEALPixels that could touch the primary targets.
    if 'FILEHPX' in hdr:
        nside, pix = hdr['FILENSID'], hdr['FILEHPX']
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
    big = 500000
    if np.sum(inhp) > big and len(intargs) > big:
        log.warning('Large secondary (N={}) and primary (N={}) samples'
                    .format(np.sum(inhp), len(intargs)))
        log.warning('The code may run slowly')

    # ADM for each secondary target, determine if there is a match
    # ADM with a primary target. Note that sense is important, here
    # ADM (the primary targets must be passed first).
    log.info('Matching primary and secondary targets for {} at {}"...t={:.1f}s'
             .format(fn, sep, time()-start))
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
    targs[desicols[0]][umtargs] |= desimasks[0].SCND_ANY

    # ADM rename the SCND_TARGET column, in case this is an SV file.
    targs = rfn.rename_fields(targs, {'SCND_TARGET': desicols[3]})

    # ADM update the secondary targets with the primary TARGETID.
    scxtargs["TARGETID"][mscx] = targs["TARGETID"][mtargs]

    # ADM form the output primary file name and write the file.
    base, ext = os.path.splitext(infile)
    dirn, fn = os.path.split(base)
    dirn = os.path.join(dirn, "wscnd")
    if not os.path.exists(dirn):
        os.mkdir(dirn)
    outfile = "{}-wscnd{}".format(os.path.join(dirn, fn), ext)
    log.info('Writing updated primary targets to {}...t={:.1f}s'
             .format(outfile, time()-start))
    fitsio.write(outfile, targs, extname='TARGETS', header=hdr, clobber=True)

    log.info('Done for {}...t={:.1f}s'.format(fn, time()-start))

    return scxtargs


def finalize_secondary(scxtargs, scnd_mask, sep=1.):
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
    sep : :class:`float`, defaults to 1 arcsecond
        The separation at which to match secondary targets to
        themselves in ARCSECONDS.

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
        order of the target in the secondary file (`SCND_ORDER`) and
        `RELEASE` from the secondary bit number (`SCND_TARGET`).
    """
    # ADM assign new TARGETIDs to targets without a primary match.
    nomatch = scxtargs["TARGETID"] == -1

    # ADM get the BRICKIDs for each source.
    brxid = bricks.brickid(scxtargs["RA"][nomatch],
                           scxtargs["DEC"][nomatch])

    # ADM the RELEASE for each source is the `SCND_TARGET` bit NUMBER.
    release = np.log2(scxtargs["SCND_TARGET_INIT"][nomatch]).astype('int')

    # ADM build the OBJIDs based on the values of SCND_ORDER for each
    # ADM brick and bit combination. First, so as not to overwhelm
    # ADM the bit-limits for OBJID, find the minimum SCND_ORDER for
    # ADM each brick and bit combination.
    # ADM create a unique ID based on brxid and release.
    scnd_order = scxtargs["SCND_ORDER"][nomatch]
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

    # ADM assemble the TARGETID, SCND objects have RELEASE==0.
    targetid = encode_targetid(objid=objid, brickid=brxid, release=release)

    # ADM a check that the generated TARGETIDs are unique.
    if len(set(targetid)) != len(targetid):
        msg = "duplicate TARGETIDs generated for secondary targets!!!"
        log.critical(msg)
        raise ValueError(msg)

    # ADM assign the unique TARGETIDs to the secondary objects.
    scxtargs["TARGETID"][nomatch] = targetid

    # ADM match secondaries to themselves, to ensure duplicates
    # ADM share a TARGETID. Don't match special (OVERRIDE) targets
    # ADM or sources that have already been matched to a primary.
    w = np.where(~scxtargs["OVERRIDE"] & nomatch)[0]
    if len(w) > 0:
        log.info("Matching secondary targets to themselves...t={:.1f}s"
                 .format(time()-t0))
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
    # ADM relevant OBSCONDITIONS and SCND_TARGET bits set. By definition,
    # ADM targets with OVERRIDE set never have matching TARGETIDs.
    wnoov = np.where(~scxtargs["OVERRIDE"])[0]
    if len(wnoov) > 0:
        for _, inds in duplicates(scxtargs["TARGETID"][wnoov]):
            scnd_targ = 0
            obs_con = 0
            for ind in inds:
                scnd_targ |= scxtargs["SCND_TARGET"][wnoov[ind]]
                obs_con |= scxtargs["OBSCONDITIONS"][wnoov[ind]]
                scxtargs["SCND_TARGET"][wnoov[inds]] = scnd_targ
                scxtargs["OBSCONDITIONS"][wnoov[inds]] = obs_con
            # ADM only keep the priority for the highest-priority match.
            maxi = np.max(scxtargs["PRIORITY_INIT"][wnoov[inds]])
            argmaxi = np.argmax(scxtargs["PRIORITY_INIT"][wnoov[inds]])
            scxtargs["PRIORITY_INIT"][wnoov[inds]] = -1
            scxtargs["PRIORITY_INIT"][wnoov[inds[argmaxi]]] = maxi

    # ADM change the data model depending on whether the mask
    # ADM is an SVX (X = 1, 2, etc.) mask or not. Nothing will
    # ADM change if the mask has no preamble.
    prepend = scnd_mask._name[:-9].upper()
    scxtargs = rfn.rename_fields(
        scxtargs, {'SCND_TARGET': prepend+'SCND_TARGET'}
        )

    return scxtargs


def select_secondary(infiles, numproc=4, sep=1., obscon=None,
                     scxdir=None, scnd_mask=None):
    """Process secondary targets and update relevant bits.

    Parameters
    ----------
    infiles : :class:`list` or `str`
        A list of input primary target file names OR a single file name.
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    sep : :class:`float`, defaults to 1 arcsecond
        The separation at which to match in ARCSECONDS.
    obscon : :class:`str`, optional, defaults to None
        An OBSCONDITIONS string that can be understood by the desitarget
        mask parser (e.g. 'GRAY|DARK'). Only secondary targets that have
        a match to these OBSCONDITIONS in the yaml file are processed.
        If ``None``, the code will attempt to read it from the OBSCON
        keyword in the header of the first primary file it encounters.
    scxdir : :class:`str`, optional, defaults to :envvar:`SCND_DIR`
        The name of the directory that hosts secondary targets.
    scnd_mask : :class:`desiutil.bitmask.BitMask`, optional
        A mask corresponding to a set of secondary targets, e.g, could
        be ``from desitarget.targetmask import scnd_mask`` for the
        main survey mask. Defaults to the main survey mask.

    Returns
    -------
    :class:`~numpy.ndarray`
        All secondary targets from `scxdir` with columns ``TARGETID``,
        ``SCND_TARGET``, ``PRIORITY_INIT``, ``SUBPRIORITY`` and
        ``NUMOBS_INIT`` added. These columns are also populated,
        excepting ``SUBPRIORITY``.

    Notes
    -----
        - In addition, the primary target `infiles` are written back to
          their original path with `.fits` changed to `-wscnd.fits` and
          the ``SCND_TARGET`` and ``SCND_ANY`` columns
          populated for matching targets.
    """
    # ADM import the default (main survey) mask.
    if scnd_mask is None:
        from desitarget.targetmask import scnd_mask

    # ADM if a single primary file was passed, convert it to a list.
    if isinstance(infiles, str):
        infiles = [infiles, ]
    nfiles = len(infiles)

    # - Sanity check that files exist before going further.
    for filename in infiles:
        if not os.path.exists(filename):
            msg = "{} doesn't exist".format(filename)
            log.critical(msg)
            raise ValueError(msg)

    # ADM if OBSCON wasn't sent, read it from the primary targets.
    hdr = fitsio.read_header(infiles[0], extension='TARGETS')
    if obscon is None:
        log.info('Trying to read OBSCON from header of {}'.format(infiles[0]))
        obscon = hdr["OBSCON"]

    # ADM retrieve the scxdir, check it's structure and fidelity...
    scxdir = _get_scxdir(scxdir)
    _check_files(scxdir, scnd_mask)
    # ADM ...and read in all of the secondary targets.
    scxtargs = read_files(scxdir, scnd_mask, obscon=obscon)

    # ADM split off any scx targets that have requested an OVERRIDE.
    scxover = scxtargs[scxtargs["OVERRIDE"]]
    scxtargs = scxtargs[~scxtargs["OVERRIDE"]]

    # ADM function to run on every input file.
    def _match_scx_file(fn):
        """wrapper on match_secondary() given a file name"""
        # ADM for one of the input primary target files, match to the
        # ADM non-override scx targets and update bits and TARGETID.
        return match_secondary(fn, scxtargs, sep=sep, scxdir=scxdir)

    # ADM this is just to count files in _update_status.
    nfile = np.array(1)
    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if nfile % 1 == 0 and nfile > 0:
            elapsed = (time()-t0)/60.
            rate = nfile/elapsed/60.
            log.info('{}/{} files; {:.1f} sec/file...t = {:.1f} mins'
                     .format(nfile, nfiles, 1./rate, elapsed))
        nfile[...] += 1    # this is an in-place modification.
        return result

    # - Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            scxall = pool.map(_match_scx_file, infiles, reduce=_update_status)
        # ADM if we ran with numproc==1, then the TARGETID in the view of
        # ADM scxtargs will have naturally updated during the loop. This could
        # ADM be solved with an expensive copy, if it was necessary. For the
        # ADM numproc > 1 case, though, we need to find TARGETIDs that have
        # ADM been set across the scxall outputs.
        targetids = np.max(np.vstack([scxt['TARGETID'] for scxt in scxall]), axis=0)
        scxtargs = scxall[-1]
        scxtargs["TARGETID"] = targetids
    else:
        scxall = []
        for infile in infiles:
            scxall.append(_update_status(_match_scx_file(infile)))
        scxtargs = scxall[-1]

    # ADM now we're done matching, bring the override targets back...
    scxout = np.concatenate([scxtargs, scxover])

    # ADM ...and assign TARGETIDs to non-matching secondary targets.
    scxout = finalize_secondary(scxout, scnd_mask, sep=sep)

    return scxout
