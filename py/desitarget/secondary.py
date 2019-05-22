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
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Row

from time import time

from desitarget.internal import sharedmem
from desitarget.targetmask import scnd_mask
from desitarget.geomask import radec_match_to
from desitarget.targets import main_cmx_or_sv, encode_targetid

from desiutil import brick
from desiutil.log import get_logger

# ADM set up the Legacy Surveys bricks object.
bricks = brick.Bricks(bricksize=0.25)
# ADM set up the default DESI logger.
log = get_logger()
start = time()

indatamodel = np.array([], dtype=[
    ('RA', '>f8'), ('DEC', '>f8'), ('OVERRIDE', '?')
])

# ADM the columns not in the primary target files are:
#  OVERRIDE - If True/1 force as a target even if there is a primary.
#           - If False/0 allow this to be replaced by a primary target.
#  SCND_TARGET - The bit mask from data/targetmask.yaml (scnd_mask).
#  SCND_ORDER - Row number in the input secondary file for this target.
# ADM Note that TARGETID for secondary-only targets is unique because
# ADM RELEASE is 0 for secondary-only targets.
outdatamodel = np.array([], dtype=[
    ('RA', '>f8'), ('DEC', '>f8'), ('OVERRIDE', '?'),
    ('TARGETID', '>i8'), ('SCND_TARGET', '>i8'),
    ('PRIORITY_INIT', '>i8'), ('SUBPRIORITY', '>f8'),
    ('NUMOBS_INIT', '>i8'), ('SCND_ORDER', '>i4')
])


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


def _check_files(scxdir):
    """Retrieve input files from the scx directory with error checking.

    Parameters
    ----------
    scxdir : :class:`str`
        Directory produced by :func:`~secondary._check_files()`.

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


def read_files(scxdir):
    """Read in all secondary files and concatenate them into one array.

    Parameters
    ----------
    scxdir : :class:`str`
        Directory produced by :func:`~secondary._check_files()`.

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
        # ADM the full file path without the extension.
        fn = os.path.join(fulldir, scnd_mask[name].filename)
        # ADM if the relevant file is a .txt file, read it in.
        if os.path.exists(fn+'.txt'):
            scxin = np.loadtxt(fn+'.txt', usecols=[0, 1, 2],
                             dtype=indatamodel.dtype)
        # ADM otherwise it's a fits file, read it in.
        else:
            scxin = fitsio.read(fn+'.fits',
                              columns=indatamodel.dtype.names)
        # ADM ensure this is a properly constructed numpy array.
        scxin = np.atleast_1d(scxin)
        # ADM add the other output columns.
        scxout = np.zeros(len(scxin), dtype=outdatamodel.dtype)
        for col in indatamodel.dtype.names:
            scxout[col] = scxin[col]
        scxout["SCND_TARGET"] = scnd_mask[name]
        scxout["PRIORITY_INIT"] = scnd_mask[name].priorities['UNOBS']
        scxout["NUMOBS_INIT"] = scnd_mask[name].numobs
        scxout["TARGETID"] = -1
        scxout["SCND_ORDER"] = np.arange(len(scxin))

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
        - The primary target `infiles` are written back to their
          original path with `.fits` changed to `-wscnd.fits` and the
          `SCND_TARGET` bit populated for matching targets.
    """
    # ADM read in the primary targets.
    log.info('Reading primary targets file {}...t={:.1f}s'
             .format(infile, time()-start))
    intargs, hdr = fitsio.read(infile, "TARGETS", header=True)

#    log.info('Adding "SCND_TARGET" column to {}...t={:.1f}s'
#             .format(infile, time()-start))
    # ADM fail if file's already been matched to secondary targets.
    if "SCNDDIR" in hdr:
        msg = "{} already matched to secondary targets!!!".format(infile)
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

    # ADM for every secondary target, determine if there is a match
    # ADM with a primary target. Note that sense is important, here
    # ADM (the primary targets must be passed first).
    log.info('Matching primary and secondary targets at {}"...t={:.1f}s'
             .format(sep, time()-start))
    mtargs, mscx = radec_match_to(targs, scxtargs, sep=sep)

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
    desicols, desimasks, _ = main_cmx_or_sv(targs)
    targs[desicols[0]][umtargs] |= desimasks[0].SCND_ANY

    # ADM update the secondary targets with the primary TARGETID.
    scxtargs["TARGETID"][mscx] = targs["TARGETID"][mtargs]

    # ADM form the output primary file name and write the file.
    base, ext = os.path.splitext(infile)
    outfile = "{}{}{}".format(base, '-wscnd', ext)
    log.info('Writing updated primary targets to {}...t={:.1f}s'
             .format(outfile, time()-start))
    fitsio.write(outfile, targs, extname='TARGETS', header=hdr, clobber=True)

    log.info('Done...t={:.1f}s'.format(time()-start))

    return scxtargs


def finalize_secondary(scxtargs):
    """Assign secondary targets a realistic TARGETID.

    Parameters
    ----------
    scxtargs : :class:`~numpy.ndarray`
        An array of secondary targets, must contain the columns `RA`,
        `DEC` and `TARGETID`. `TARGETID` should be -1 for objects 
        that lack a `TARGETID`.

    Returns
    -------
    :class:`~numpy.ndarray`
        The array of secondary targets, with the `TARGETID` bit
        updated to a unique and reasonable `TARGETID`.

    Notes
    -----
        - The resulting values of `TARGETID` will be unique across
          the input `scxtargs` list. They will not share a `TARGETID`
          with a primary target beacuse they will have `RELEASE`==0.
    """
    # ADM assign new TARGETIDs to targets without a primary match.
    nomatch = scxtargs["TARGETID"] == -1

    # ADM get BRICKIDs, retrieve the list of unique bricks and the 
    # ADM number of sources in each unique brick.
    brxid = bricks.brickid(scxtargs["RA"][nomatch],
                           scxtargs["DEC"][nomatch])
    ubrx, un = np.unique(brxid, return_counts=True)

    # ADM build the OBJIDs from the number of sources per brick.
    objid = np.zeros_like(brxid)
    for brx, nobjs in zip(ubrx, un):
        isinbrx = brxid == brx
        objid[isinbrx] = np.arange(nobjs)

    # ADM assemble the TARGETID, SCND objects have RELEASE==0.
    targetid = encode_targetid(objid=objid, brickid=brxid)

    # ADM a check that the generated TARGETIDs are unique.
    if len(set(targetid)) != len(targetid):
        msg = "duplicate TARGETIDs for secondary targets!!!"
        log.critical(msg)
        raise ValueError(msg)

    # ADM assign the TARGETIDs to the secondary objects
    scxtargs["TARGETID"][nomatch] = targetid

    return scxtargs


def select_secondary(infiles, numproc=4, sep=1., scxdir=None):
    """Process secondary targets and update relevant bits.

    Parameters
    ----------
    infiles : :class:`list` or `str`
        A list of input primary target file names OR a single file name.
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    sep : :class:`float`, defaults to 1 arcsecond
        The separation at which to match in ARCSECONDS.
    scxdir : :class:`str`, optional, defaults to :envvar:`SCND_DIR`
        The name of the directory that hosts secondary targets.

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

    # ADM retrieve the scxdir, check it's structure and fidelity...
    scxdir = _get_scxdir(scxdir)
    _check_files(scxdir)
    # ADM ...and read in all of the secondary targets.
    scxtargs = read_files(scxdir)

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
            log.info('{}/{} files; {:.1f} files/sec...t = {:.1f} mins'
                     .format(nfile, nfiles, rate, elapsed))
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
    scxout = finalize_secondary(scxout)

    return scxout
