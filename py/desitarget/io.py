# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==================
desitarget.io
==================

This file knows how to write a TS catalogue.
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import fitsio
import os, re
from . import __version__ as desitarget_version
from . import gitversion

from desiutil import depend

tscolumns = [
    'RELEASE', 'BRICKID', 'BRICKNAME', 'OBJID', 'TYPE',
    'RA', 'RA_IVAR', 'DEC', 'DEC_IVAR',
    'FLUX_G', 'FLUX_R', 'FLUX_Z', 
    'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 
    'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
    'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z',
    'NOBS_G', 'NOBS_R', 'NOBS_Z', 
    'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z'
    'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z'
    'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4'
    'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FLUX_IVAR_W3', 'FLUX_IVAR_W4'
    'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2', 
    'MW_TRANSMISSION_W3', 'MW_TRANSMISSION_W4'
    'SHAPEDEV_R', 'SHAPEEXP_R', 'DCHISQ',
    ]

def read_tractor(filename, header=False, columns=None):
    """Read a tractor catalogue file.

    Parameters
    ----------
    filename : :class:`str`
        File name of one tractor file.
    header : :class:`bool`, optional
        If ``True``, return (data, header) instead of just data.
    columns: :class:`list`, optional
        Specify the desired Tractor catalog columns to read; defaults to
        desitarget.io.tscolumns.

    Returns
    -------
    :class:`numpy.ndarray`
        Array with the tractor schema, uppercase field names.
    """
    check_fitsio_version()

    if columns is None:
        readcolumns = list(tscolumns)
    else:
        readcolumns = list(columns)

    fx = fitsio.FITS(filename, upper=True)
    #- tractor files have BRICK_PRIMARY; sweep files don't
    fxcolnames = fx[1].get_colnames()
    if (columns is None) and \
       (('BRICK_PRIMARY' in fxcolnames) or ('brick_primary' in fxcolnames)):
        readcolumns.append('BRICK_PRIMARY')

    hdr = fx[1].read_header()
    if (columns is None) and ('RELEASE' not in fxcolnames):
        #ADM if RELEASE doesn't exist, then we're pre-DR3, so rewrite the input
        #ADM columns completely to correspond to the DR4+ data model

    data = fx[1].read(columns=readcolumns)

    #ADM Empty (length 0) files have dtype='>f8' instead of 'S8' for brickname
    if len(data) == 0:
        print('WARNING: Empty file>', filename)
        dt = data.dtype.descr
        dt[1] = ('BRICKNAME', 'S8')
        data = data.astype(np.dtype(dt))

    #ADM To circumvent whitespace bugs on I/O from fitsio
    #ADM need to strip any white space from string columns
    for colname in data.dtype.names:
        kind = data[colname].dtype.kind
        if kind == 'U' or kind == 'S':
            data[colname] = np.char.rstrip(data[colname])

    if header:
        fx.close()
        return data, hdr
    else:
        fx.close()
        return data


def fix_tractor_dr1_dtype(objects):
    """DR1 tractor files have inconsitent dtype for the TYPE field.  Fix this.

    Args:
        objects : numpy structured array from target file

    Returns:
        structured array with TYPE.dtype = 'S4' if needed

    If the type was already correct, returns the original array
    """
    if objects['TYPE'].dtype == 'S4':
        return objects
    else:
        dt = objects.dtype.descr
        for i in range(len(dt)):
            if dt[i][0] == 'TYPE':
                dt[i] = ('TYPE', 'S4')
                break
        return objects.astype(np.dtype(dt))


def write_targets(filename, data, indir=None, qso_selection=None, sandboxcuts=False):
    """Write a target catalogue.

    Args:
        filename : output target selection file
        data     : numpy structured array of targets to save

    """
    # FIXME: assert data and tsbits schema
    #ADM use RELEASE to determine the release string for the input targets

    drint = np.max(data['RELEASE']//1000)
    #ADM check the targets all have the same release                                                                                                                        
    checker = np.min(data['RELEASE']//1000)
    if drint != checker:
        raise IOError('Objects from multiple data releases in same input numpy array?!')
    drstring = 'dr'+str(drint)

    #- Create header to include versions, etc.
    hdr = fitsio.FITSHDR()
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())
    depend.setdep(hdr, 'sandboxcuts', sandboxcuts)
    depend.setdep(hdr, 'photcat  ', drstring)

    if indir is not None:
        depend.setdep(hdr, 'tractor-files', indir)

    if qso_selection is None:
        print('ERROR: qso_selection method not specified for output file')
        depend.setdep(hdr, 'qso-selection', 'unknown')
    else:
        depend.setdep(hdr, 'qso-selection', qso_selection)

    fitsio.write(filename, data, extname='TARGETS', header=hdr, clobber=True)


def iter_files(root, prefix, ext='fits'):
    """Iterator over files under in `root` directory with given `prefix` and
    extension.
    """
    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            for filename in filenames:
                if filename.startswith(prefix) and filename.endswith('.'+ext):
                    yield os.path.join(dirpath, filename)
    else:
        filename = os.path.basename(root)
        if filename.startswith(prefix) and filename.endswith('.'+ext):
            yield root


def list_sweepfiles(root):
    """Return a list of sweep files found under `root` directory.
    """
    return [x for x in iter_sweepfiles(root)]


def iter_sweepfiles(root):
    """Iterator over all sweep files found under root directory.
    """
    return iter_files(root, prefix='sweep', ext='fits')


def list_tractorfiles(root):
    """Return a list of tractor files found under `root` directory.
    """
    return [x for x in iter_tractorfiles(root)]


def iter_tractorfiles(root):
    """Iterator over all tractor files found under `root` directory.

    Parameters
    ----------
    root : :class:`str`
        Path to start looking.  Can be a directory or a single file.

    Returns
    -------
    iterable
        An iterator of (brickname, filename).

    Examples
    --------
    >>> for brickname, filename in iter_tractor('./'):
    >>>     print(brickname, filename)
    """
    return iter_files(root, prefix='tractor', ext='fits')


def brickname_from_filename(filename):
    """Parse `filename` to check if this is a tractor brick file.

    Parameters
    ----------
    filename : :class:`str`
        Name of a tractor brick file.

    Returns
    -------
    :class:`str`
        Name of the brick in the file name.

    Raises
    ------
    ValueError
        If the filename does not appear to be a valid tractor brick file.
    """
    if not filename.endswith('.fits'):
        raise ValueError("Invalid tractor brick file: {}!".format(filename))
    #
    # Match filename tractor-0003p027.fits -> brickname 0003p027.
    # Also match tractor-00003p0027.fits, just in case.
    #
    match = re.search('tractor-(\d{4,5}[pm]\d{3,4})\.fits',
                      os.path.basename(filename))

    if match is None:
        raise ValueError("Invalid tractor brick file: {}!".format(filename))
    return match.group(1)

############################################################
def brickname_from_filename_with_prefix(filename,prefix=''):
    """Parse `filename` to check if this is a brick file with a given prefix.

    Parameters
    ----------
    filename : :class:`str`
        Full name of a brick file.
    prefix : :class:`str`
        Optional part of filename immediately preceding the brickname

    Returns
    -------
    :class:`str`
        Name of the brick in the file name.

    Raises
    ------
    ValueError
        If the filename does not appear to be a valid brick file.
    """
    if not filename.endswith('.fits'):
        raise ValueError("Invalid galaxia mock brick file: {}!".format(filename))
    #
    # Match filename tractor-0003p027.fits -> brickname 0003p027.
    # Also match tractor-00003p0027.fits, just in case.
    #
    match = re.search('%s_(\d{4,5}[pm]\d{3,4})\.fits'%(prefix),
                      os.path.basename(filename))

    if match is None:
        raise ValueError("Invalid galaxia mock brick file: {}!".format(filename))
    return match.group(1)


def check_fitsio_version(version='0.9.8'):
    """fitsio_ prior to 0.9.8rc1 has a bug parsing boolean columns.

    .. _fitsio: https://pypi.python.org/pypi/fitsio

    Parameters
    ----------
    version : :class:`str`, optional
        Default '0.9.8'.  Having this parameter allows future-proofing and
        easier testing.

    Raises
    ------
    ImportError
        If the fitsio version is insufficiently recent.
    """
    from distutils.version import LooseVersion
    #
    # LooseVersion doesn't handle rc1 as we want, so also check for 0.9.8xxx.
    #
    if (LooseVersion(fitsio.__version__) < LooseVersion(version) and
        not fitsio.__version__.startswith(version)):
        raise ImportError(('ERROR: fitsio >{0}rc1 required ' +
                           '(not {1})!').format(version, fitsio.__version__))

def whitespace_fits_read(filename, **kwargs):
    """Use fitsio_ to read in a file and strip whitespace from all string columns

    .. _fitsio: https://pypi.python.org/pypi/fitsio

    Parameters
    ----------
    filename : :class:`str`
        Name of the file to be read in by fitsio
    kwargs: arguments that will be passed directly to fitsio
    """
    fitout = fitsio.read(filename, **kwargs)
    #ADM if the header=True option was passed then
    #ADM the output is the header and the data
    data = fitout
    if 'header' in kwargs:
        data, header = fitout

    #ADM guard against the zero-th extension being read by fitsio
    if data is not None:
        #ADM strip any whitespace from string columns
        for colname in data.dtype.names:
            kind = data[colname].dtype.kind
            if kind == 'U' or kind == 'S':
                data[colname] = np.char.rstrip(data[colname])

    if 'header' in kwargs:
        return data, header

    return data
