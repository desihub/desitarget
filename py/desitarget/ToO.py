"""
desitarget.too
==============

Targets of Opportunity.
"""
import os
import numpy as np

from desiutil.log import get_logger
log = get_logger()

# ADM the data model for ToO, similar to that of secondary targets...
from desitarget.secondary import indatamodel
from desitarget.secondary import outdatamodel
# ADM ...but the OVERRIDE columns isn't necessary...
indtype = [tup for tup in indatamodel.dtype.descr if "OVERRIDE" not in tup]
outdtype = [tup for tup in outdatamodel.dtype.descr if "OVERRIDE" not in tup]
# ADM ...and some extra columns are necessary.
indatamodel = np.array([], dtype=indtype + [
    ('CHECKER', '>U3'), ('MJD_BEGIN', '>f8'), ('MJD_END', '>f8')])
outdatamodel = np.array([], dtype=outdtype + [
    ('CHECKER', '>U3'), ('MJD_BEGIN', '>f8'), ('MJD_END', '>f8')])

# ADM when using basic or csv ascii writes, specifying the formats of
# ADM float32 columns can make things easier on the eye.
tooformatdict = {"PARALLAX": '%16.8f', 'PMRA': '%16.8f', 'PMDEC': '%16.8f'}

# ADM This RELEASE means Target of Opportunity in TARGETID.
release = 9999


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

    from desitarget import io
    dr = release//1000
    fn = io.find_target_files(tdir, flavor="ToO", ender=ender, nohp=True, dr=dr)
    # ADM change the name slightly to make this the "input" ledger.

    if outname:
        return fn
    return fn.replace(".{}".format(ender), "-input.{}".format(ender))


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
    :class:`array_like`
        An array of the initial, example values for the ToO ledger.
        The initial (.ecsv) ledger is also written to toodir or $TOO_DIR.
    """
    # ADM get the ToO directory (or check it exists).
    tdir = get_too_dir(toodir)

    # ADM retrieve the file name to which to write.
    fn = get_filename(tdir)

    # ADM make a single line of the ledger with some indicative values.
    data = np.zeros(1, dtype=indatamodel.dtype)
    data["RA"], data["DEC"] = 359.999999, -89.999999
    data["PMRA"], data["PMDEC"] = 13.554634, -10.763842
    data["REF_EPOCH"] = 2015.5
    data["CHECKER"] = "ADM"
    data["MJD_BEGIN"], data["MJD_END"] = 40811.04166667, 40811.95833333

    # ADM Add a header for the ledger.
    from . import __version__ as desitarget_version
    from desitarget import io
    from desiutil import depend
    hdr = {}
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', io.gitversion())
    hdr["RELEASE"] = release

    log.info("Writing initial ledger to {}".format(fn))
    # ADM create necessary directories, if they don't exist.
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    # ADM and write the initial ledger.
    io.write_with_units(fn, data, extname="TOO", header=hdr, ecsv=True)

    return data


def ledger_to_targets(toodir=None):
    """Process the initial ledger

    Parameters
    ----------
    toodir : :class:`str`, optional, defaults to ``None``
        The directory to treat as the Targets of Opportunity I/O directory.
        If ``None`` then look up from the $TOO_DIR environment variable.

    Returns
    -------
    :class:`array_like`
        An array of the initial, example values for the ToO ledger.
        The initial (.ecsv) ledger is also written to toodir or $TOO_DIR.
    """
