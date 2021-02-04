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
from desitarget.secondary import outdatamodel as toodatamodel
# ADM ...but with some extra columns.
toodatamodel = np.array([], dtype=toodatamodel.dtype.descr + [
    ('CHECKER', '>S3'), ('MJD_START', '>f4'), ('MJD_END', '>f4')])

# ADM when using basic or csv ascii writes, specifying the formats of
# ADM float32 columns can make things easier on the eye.
tooformatdict = {"PARALLAX": '%16.8f', 'PMRA': '%16.8f', 'PMDEC': '%16.8f'}


def get_too_dir(toodir=None):
    """Convenience function to grab the TOO_DIR environment variable.

    Parameters
    ----------
    :class:`str`, optional, defaults to ``None``
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

    msg = "{} does not exist. " .format(toodir) + msg
    if not os.path.exists(toodir):
        log.critical(msg)
        raise ValueError(msg)

    return mtldir
