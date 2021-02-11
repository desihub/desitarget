"""
desitarget.too
==============

Targets of Opportunity.
"""
import os
import numpy as np
from astropy.table import Table

from desiutil.log import get_logger
log = get_logger()

from desitarget import io
# ADM the data model for ToO, similar to that of secondary targets...
from desitarget.secondary import indatamodel
from desitarget.secondary import outdatamodel
# ADM ...but the OVERRIDE columns isn't necessary...
indtype = [tup for tup in indatamodel.dtype.descr if "OVERRIDE" not in tup]
outdtype = [tup for tup in outdatamodel.dtype.descr if "OVERRIDE" not in tup]
# ADM ...and some extra columns are necessary.
indatamodel = np.array([], dtype=indtype + [
    ('CHECKER', '>U3'), ('TOO_TYPE', '>U5'), ('OCLAYER', '>U6'),
    ('MJD_BEGIN', '>f8'), ('MJD_END', '>f8')])
outdatamodel = np.array([], dtype=outdtype + [
    ('CHECKER', '>U3'), ('TOO_TYPE', '>U5'), ('OCLAYER', '>U6'),
    ('MJD_BEGIN', '>f8'), ('MJD_END', '>f8')])

# ADM when using basic or csv ascii writes, specifying the formats of
# ADM float32 columns can make things easier on the eye.
tooformatdict = {"PARALLAX": '%16.8f', 'PMRA': '%16.8f', 'PMDEC': '%16.8f'}

# ADM This RELEASE means Target of Opportunity in TARGETID.
release = 9999

# ADM Constraints on how many ToOs are allowed in a given time period.
# ADM constraints on fiber overrides in units of total fibers.
constraints = {"FIBER": {"overrides_per_night": 2,
                         "overrides_per_month": 50,
                         "overrides_per_year": 500},
# ADM constraints on field overrides. ALSO in units of TOTAL FIBERS.
               "TILE": {"overrides_per_night": 5000,
                        "overrides_per_month": 5000,
                        "overrides_per_year": 10000}
}

def get_filename(toodir=None, ender="ecsv", outname=False):
    """Construct the input/output ToO filenames (with full directory path).

    Parameters
    ----------
    toodir : :class:`str`, optional, defaults to ``None``
        The directory to treat as the Targets of Opportunity I/O directory.
        If ``None`` then look up from the $TOO_DIR environment variable.
    ender : :class:`str`, optional, defaults to "ecsv"
        File format (in file name), likely either "ecsv" or "fits".
    outname : :class:`bool`, optional, defaults to ``False``
        If ``True`` return the output ToO filename. Otherwise return
        the input ToO filename.

    Returns
    -------
    :class:`str`
        The directory to treat as the Targets of Opportunity I/O directory.
    """
    # ADM retrieve the $TOO_DIR variable, if toodir wasn't passed.
    tdir = get_too_dir(toodir)

    dr = release//1000
    fn = io.find_target_files(tdir, flavor="ToO", ender=ender, nohp=True, dr=dr)
    # ADM change the name slightly to make this the "input" ledger.

    if outname:
        return fn
    return fn.replace(".{}".format(ender), "-input.{}".format(ender))


def _write_too_files(filename, data, ecsv=False):
    """Write ToO ledgers and files.

    Parameters
    ----------
    filename : :class:`str`
        Full path to filename to which to write Targets of Opportunity.
    data : :class:`~numpy.ndarray` or `~astropy.table.Table`
        Table or array of Targets of Opportunity to write.
    ecsv : :class:`bool`, optional, defaults to ``False``
        If ``True`` then write as a .ecsv file, if ``False`` then write
        as a .fits file.

    Returns
    -------
    None
        But `data` is written to `filename` with standard ToO formalism.
    """
    log.info("Writing ToO file to {}".format(filename))

    # ADM grab the standard header.
    hdr = _get_too_header()

    # ADM create necessary directories, if they don't exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # ADM io.write_with_units expects an array, not a Table.
    if isinstance(data, Table):
        data = data.as_array()

    # ADM write the file.
    io.write_with_units(filename, data, extname="TOO", header=hdr, ecsv=ecsv)

    return


def _get_too_header():
    """Convenience function that returns a standard header for ToO files.
    """
    from . import __version__ as desitarget_version
    from desiutil import depend
    hdr = {}
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', io.gitversion())
    hdr["RELEASE"] = release

    return hdr


def get_too_dir(toodir=None):
    """Convenience function to grab the TOO_DIR environment variable.

    Parameters
    ----------
    toodir : :class:`str`, optional, defaults to ``None``
        The directory to treat as the Targets of Opportunity I/O directory.
        If ``None`` then look up from the $TOO_DIR environment variable.

    Returns
    -------
    :class:`str`
        The directory to treat as the Targets of Opportunity I/O directory.
    """
    if toodir is None:
        toodir = os.environ.get("TOO_DIR")

    msg = "Pass toodir or set $TOO_DIR."
    if toodir is None:
        log.critical(msg)
        raise ValueError(msg)

    msg = "{} does not exist. Make it or..." .format(toodir) + msg
    if not os.path.exists(toodir):
        log.critical(msg)
        raise ValueError(msg)

    return toodir


def make_initial_ledger(toodir=None):
    """Set up the initial ToO ledger with one ersatz observation.

    Parameters
    ----------
    toodir : :class:`str`, optional, defaults to ``None``
        The directory to treat as the Targets of Opportunity I/O directory.
        If ``None`` then look up from the $TOO_DIR environment variable.

    Returns
    -------
    :class:`~astropy.table.Table`
        A Table of the initial, example values for the ToO ledger.
        The initial (.ecsv) ledger is also written to toodir or $TOO_DIR.
    """
    # ADM get the ToO directory (or check it exists).
    tdir = get_too_dir(toodir)

    # ADM retrieve the file name to which to write.
    fn = get_filename(tdir)

    # ADM make a single line of the ledger with some indicative values.
    data = np.zeros(2, dtype=indatamodel.dtype)
    data["RA"] = 359.999999, 101.000001
    data["DEC"] = -89.999999, -89.999999
    data["PMRA"] = 13.554634, 4.364553
    data["PMDEC"] = 10.763842, -10.763842
    data["REF_EPOCH"] = 2015.5, 2015.5
    data["CHECKER"] = "ADM", "AM"
    data["TOO_TYPE"] = "TILE", "FIBER"
    data["MJD_BEGIN"] = 40811.04166667, 41811.14166667
    data["MJD_END"] = 40811.95833333, 41811.85833333
    data["OCLAYER"] = "BRIGHT", "DARK"

    # ADM write out the results.
    _write_too_files(fn, data, ecsv=True)

    return data


def _check_ledger(inledger):
    """Perform checks that the ledger conforms to requirements.

    Parameters
    ----------
    inledger : :class:`~astropy.table.Table`
        A Table of input Targets of Opportunity from the ToO ledger.

    None
        But a series of checks of the ledger are conducted.
    """
    # ADM check that every entry has been vetted by-eye.
    checkers = np.array(list(set(inledger["CHECKER"])))
    checkergood = np.array([len(checker) > 1 for checker in checkers])
    if not np.all(checkergood):
        msg = "An entry in the ToO ledger ({}) has not been checked!".format(
            checkers[~checkergood])
        log.critical(msg)
        raise ValueError(msg)

    # ADM check that TOO_TYPE's are all either TILE or FIBER.
    # ADM and the observing condiations are all either DARK or BRIGHT.
    allowed = {"TOO_TYPE": {'FIBER', 'TILE'},
               "OCLAYER": {'BRIGHT', 'DARK'}}
    for col in allowed:
        if not set(inledger[col]).issubset(allowed[col]):
            msg = "Some {} entries in the ToO ledger are not one of {}!".format(
                col, allowed[col])
            log.critical(msg)
            raise ValueError(msg)

    # ADM basic check that the dates are formatted correctly.
    if np.any(inledger["MJD_BEGIN"] > inledger["MJD_END"]):
        msg = "Some MJD_BEGINs are later than their associated MJD_END!"
        log.critical(msg)
        raise ValueError(msg)

    # ADM check that the requested ToOs don't exceed allocations.
    # ADM there are different constraints for the types of observations.
    ii = inledger["TOO_TYPE"] == "TILE"
    # ADM work with discretized days that run from noon until noon so
    # ADM each night of observations is encompassed by an integer day.
    jdbegin, jdend = inledger["MJD_BEGIN"][ii]+0.5, inledger["MJD_END"][ii]+0.5
    jdbegin, jdend = jdbegin.astype(int), jdend.astype(int)
    # ADM establish the range of days to loop over.
    start, fin = jdbegin.astype(int).min(), jdend.astype(int).max()+1

    return


def finalize_too(inledger, survey="main"):
    """Add necessary targeting columns to a ToO ledger.

    Parameters
    ----------
    inledger : :class:`~astropy.table.Table`
        A Table of input Targets of Opportunity from the ToO ledger.
    survey : :class:`str`, optional, defaults to ``'main'``
        Specifies which target masks yaml file to use for bits, and which
        column names to add in the output file. Options are ``'main'``
        and ``'svX``' (where X is 1, 2, 3 etc.) for the main survey and
        different iterations of SV, respectively.

    Returns
    -------
    :class:`~astropy.table.Table`
        A Table of targets generated from the ToO ledger, with the
        necessary columns added for fiberassign to run.
    """
    # ADM create the output data table.
    outdata = Table(np.zeros(len(inledger), dtype=outdatamodel.dtype))
    # ADM change column names to reflect the survey.
    if survey[:2] == "sv":
        bitcols = [col for col in outdata.dtype.names if "_TARGET" in col]
        for col in bitcols:
            outdata.rename_column(col, "{}_{}".format(survey.upper(), col))

    # ADM grab the appropriate masks and column names.
    from desitarget.targets import main_cmx_or_sv
    cols, Mxs, surv = main_cmx_or_sv(outdata, scnd=True)
    dcol, bcol, mcol, scol = cols
    dMx, bMx, mMx, sMx = Mxs

    # ADM add the input columns to the output table.
    for col in inledger.dtype.names:
        outdata[col] = inledger[col]

    # ADM add the output columns.
    ntoo = len(outdata)
    # ADM assign a TARGETID for each input targets.
    from desiutil import brick
    from desitarget.targets import encode_targetid
    bricks = brick.Bricks(bricksize=0.25)
    brickid = bricks.brickid(outdata["RA"], outdata["DEC"])
    objid = np.arange(ntoo)
    targetid = encode_targetid(objid=objid, brickid=brickid, release=release)
    outdata["TARGETID"] = targetid

    # ADM assign the target bitmasks and observing condition for
    # ADM each of the possible observing conditions.
    from desitarget.targetmask import obsconditions
    outdata[dcol] = dMx["SCND_ANY"]
    for oc in set(outdata["OCLAYER"]):
        ii = outdata["OCLAYER"] == oc
        bitname = "{}_TOO".format(oc)
        outdata[scol][ii] = sMx[bitname]
        outdata["PRIORITY_INIT"][ii] = sMx[bitname].priorities["UNOBS"]
        outdata["NUMOBS_INIT"][ii] = sMx[bitname].numobs
        outdata["OBSCONDITIONS"][ii] = obsconditions.mask(
            sMx[bitname].obsconditions)

    # ADM assign a SUBPRIORITY.
    np.random.seed(616)
    outdata["SUBPRIORITY"] = np.random.random(ntoo)

    return outdata


def ledger_to_targets(toodir=None, survey="main", ecsv=False, outdir=None):
    """Convert a ToO ledger to a file of ToO targets.

    Parameters
    ----------
    survey : :class:`str`, optional, defaults to ``'main'``
        Specifies which target masks yaml file to use for bits, and which
        column names to add in the output file. Options are ``'main'``
        and ``'svX``' (where X is 1, 2, 3 etc.) for the main survey and
        different iterations of SV, respectively.
    toodir : :class:`str`, optional, defaults to ``None``
        The directory to treat as the Targets of Opportunity I/O directory.
        If ``None`` then look up from the $TOO_DIR environment variable.
    ecsv : :class:`bool`, optional, defaults to ``False``
        If ``True`` then write as a .ecsv file, if ``False`` then write
        as a .fits file.
    outdir : :class:`str`, optional, defaults to ``None``
        If passed and not ``None``, then read the input ledger from
        `toodir` but write the file of targets to `outdir`.

    Returns
    -------
    :class:`~astropy.table.Table`
        A Table of targets generated from the ToO ledger. The output
        targets are also written to `toodir` (or $TOO_DIR) or `outdir`.

    Notes
    -----
    - One purpose of this code is to add all of the necessary columns for
      fiberassign to run.
    - Another purpose is to run some simple checks that the ToO targets
      do not exceed allowed specifications.
    """
    # ADM get the ToO directory (or check it exists).
    tdir = get_too_dir(toodir)

    # ADM read in the ToO ledger.
    fn = get_filename(tdir)
    indata = Table.read(fn, comment='#', format='ascii.basic', guess=False)

    # ADM check the ledger conforms to requirements.
    _check_ledger(indata)

    # ADM add the output targeting columns.
    outdata = finalize_too(indata, survey=survey)

    # ADM determine the output filename.
    # ADM set output format to ecsv if passed, or fits otherwise.
    form = 'ecsv'*ecsv + 'fits'*(not(ecsv))
    if outdir is None:
        fn = get_filename(tdir, outname=True, ender=form)
    else:
        fn = get_filename(outdir, outname=True, ender=form)

    # ADM write out the results.
    _write_too_files(fn, outdata, ecsv=ecsv)

    return outdata
