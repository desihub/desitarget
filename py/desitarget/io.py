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
import numpy.lib.recfunctions as rfn
import healpy as hp

from desiutil import depend

#ADM this is a lookup dictionary to map RELEASE to a simpler "North" or "South" 
#ADM photometric system. This will expand with the definition of RELEASE in the 
#ADM Data Model (e.g. https://desi.lbl.gov/trac/wiki/DecamLegacy/DR4sched) 
releasedict = {3000: 'S', 4000: 'N', 5000: 'S'}

oldtscolumns = [
    'BRICKID', 'BRICKNAME', 'OBJID', 'TYPE',
    'RA', 'RA_IVAR', 'DEC', 'DEC_IVAR',
    'DECAM_FLUX', 'DECAM_MW_TRANSMISSION',
    'DECAM_FRACFLUX', 'DECAM_FLUX_IVAR', 'DECAM_NOBS', 'DECAM_DEPTH', 'DECAM_GALDEPTH',
    'WISE_FLUX', 'WISE_MW_TRANSMISSION',
    'WISE_FLUX_IVAR',
    'SHAPEDEV_R', 'SHAPEEXP_R','DCHISQ',
    ]

#ADM this is an empty array of the full TS data model columns and dtypes
tsdatamodel = np.array([], dtype=[
    ('RELEASE', '>i4'), ('BRICKID', '>i4'), ('BRICKNAME', 'S8'), 
    ('OBJID', '<i4'), ('TYPE', 'S4'), ('RA', '>f8'), ('RA_IVAR', '>f4'), 
    ('DEC', '>f8'), ('DEC_IVAR', '>f4'), 
    ('FLUX_G', '>f4'), ('FLUX_R', '>f4'), ('FLUX_Z', '>f4'), 
    ('FLUX_IVAR_G', '>f4'), ('FLUX_IVAR_R', '>f4'), ('FLUX_IVAR_Z', '>f4'), 
    ('MW_TRANSMISSION_G', '>f4'), ('MW_TRANSMISSION_R', '>f4'), ('MW_TRANSMISSION_Z', '>f4'), 
    ('FRACFLUX_G', '>f4'), ('FRACFLUX_R', '>f4'), ('FRACFLUX_Z', '>f4'), 
    ('NOBS_G', '>i2'), ('NOBS_R', '>i2'), ('NOBS_Z', '>i2'), 
    ('PSFDEPTH_G', '>f4'), ('PSFDEPTH_R', '>f4'), ('PSFDEPTH_Z', '>f4'), 
    ('GALDEPTH_G', '>f4'), ('GALDEPTH_R', '>f4'), ('GALDEPTH_Z', '>f4'), 
    ('FLUX_W1', '>f4'), ('FLUX_W2', '>f4'), ('FLUX_W3', '>f4'), ('FLUX_W4', '>f4'), 
    ('FLUX_IVAR_W1', '>f4'), ('FLUX_IVAR_W2', '>f4'), ('FLUX_IVAR_W3', '>f4'), ('FLUX_IVAR_W4', '>f4'), 
    ('MW_TRANSMISSION_W1', '>f4'), ('MW_TRANSMISSION_W2', '>f4'), 
    ('MW_TRANSMISSION_W3', '>f4'), ('MW_TRANSMISSION_W4', '>f4'), 
    ('SHAPEDEV_R', '>f4'), ('SHAPEEXP_R', '>f4'), ('DCHISQ', '>f4', (5,))
    ])


def convert_from_old_data_model(fx,columns=None):
    """Read data from open Tractor/sweeps file and convert to DR4+ data model

    Parameters
    ----------
    fx : :class:`str`
        Open file object corresponding to one Tractor or sweeps file.
    columns: :class:`list`, optional
        the desired Tractor catalog columns to read

    Returns
    -------
    :class:`numpy.ndarray`
        Array with the tractor schema, uppercase field names.

    Notes
    -----
        - Anything pre-DR3 is assumed to be DR3 (we'd already broken
          backwards-compatability with DR1 because of DECAM_DEPTH but
          this now breaks backwards-compatability with DR2)
    """
    indata = fx[1].read(columns=columns)

    #ADM the number of objects in the input rec array
    nrows = len(indata)

    #ADM the column names that haven't changed between the current and the old data model
    tscolumns = list(tsdatamodel.dtype.names)
    sharedcols = list(set(tscolumns).intersection(oldtscolumns))

    #ADM the data types for the new data model
    dt = tsdatamodel.dtype

    #ADM need to add BRICKPRIMARY and its data type, if it was passed as a column of interest
    if ('BRICK_PRIMARY' in columns):
        sharedcols.append('BRICK_PRIMARY')
        dd = dt.descr
        dd.append(('BRICK_PRIMARY', '?'))
        dt = np.dtype(dd)

    #ADM create a new numpy array with the fields from the new data model...
    outdata = np.empty(nrows, dtype=dt)
    
    #ADM ...and populate them with the passed columns of data
    for col in sharedcols:
        outdata[col] = indata[col]

    #ADM change the DECAM columns from the old (2-D array) to new (named 1-D array) data model
    decamcols = ['FLUX','MW_TRANSMISSION','FRACFLUX','FLUX_IVAR','NOBS','GALDEPTH']
    decambands = 'UGRIZ'
    for bandnum in [1,2,4]:
        for colstring in decamcols:
            outdata[colstring+"_"+decambands[bandnum]] = indata["DECAM_"+colstring][:,bandnum]
        #ADM treat DECAM_DEPTH separately as the syntax is slightly different
        outdata["PSFDEPTH_"+decambands[bandnum]] = indata["DECAM_DEPTH"][:,bandnum]

    #ADM change the WISE columns from the old (2-D array) to new (named 1-D array) data model
    wisecols = ['FLUX','MW_TRANSMISSION','FLUX_IVAR']
    for bandnum in [1,2,3,4]:
        for colstring in wisecols:
            outdata[colstring+"_W"+str(bandnum)] = indata["WISE_"+colstring][:,bandnum-1]

    #ADM we also need to include the RELEASE, which we'll always assume is DR3
    #ADM (deprecating anything from before DR3)
    outdata['RELEASE'] = 3000

    return outdata
    

def read_tractor(filename, header=False, columns=None):
    """Read a tractor catalogue file.

    Parameters
    ----------
    filename : :class:`str`
        File name of one Tractor or sweeps file.
    header : :class:`bool`, optional
        If ``True``, return (data, header) instead of just data.
    columns: :class:`list`, optional
        Specify the desired Tractor catalog columns to read; defaults to
        desitarget.io.tsdatamodel.dtype.names

    Returns
    -------
    :class:`numpy.ndarray`
        Array with the tractor schema, uppercase field names.
    """
    check_fitsio_version()

    fx = fitsio.FITS(filename, upper=True)
    fxcolnames = fx[1].get_colnames()
    hdr = fx[1].read_header()

    if columns is None:
        readcolumns = list(tsdatamodel.dtype.names)
        #ADM if RELEASE doesn't exist, then we're pre-DR3 and need the old data model
        if (('RELEASE' not in fxcolnames) and ('release' not in fxcolnames)):
            readcolumns = list(oldtscolumns)
    else:
        readcolumns = list(columns)
        
    #- tractor files have BRICK_PRIMARY; sweep files don't
    if (columns is None) and \
       (('BRICK_PRIMARY' in fxcolnames) or ('brick_primary' in fxcolnames)):
        readcolumns.append('BRICK_PRIMARY')

    if (columns is None) and \
       (('RELEASE' not in fxcolnames) and ('release' not in fxcolnames)):
        #ADM Rewrite the data completely to correspond to the DR4+ data model.
        #ADM we default to writing RELEASE = 3000 ("DR3 or before data')
        data = convert_from_old_data_model(fx,columns=readcolumns)
    else:
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
    """DR1 tractor files have inconsistent dtype for the TYPE field.  Fix this.

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


def release_to_photsys(release):
    """Convert RELEASE to PHOTSYS using the releasedict lookup table.

    Parameters
    ----------
    objects : :class:`int` or :class:`~numpy.ndarray`
        RELEASE column from a numpy rec array of targets

    Returns
    -------
    :class:`str` or :class:`~numpy.ndarray`
        'N' if the RELEASE corresponds to the northern photometric
        system (MzLS+BASS) and 'S' if it's the southern system (DECaLS)
        
    Notes
    -----
    Defaults to 'U' if the system is not recognized
    """
    #ADM arrays of the key (RELEASE) and value (PHOTSYS) entries in the releasedict
    releasenums = np.array(list(releasedict.keys()))
    photstrings = np.array(list(releasedict.values()))

    #ADM an array with indices running from 0 to the maximum release number + 1
    r2p = np.empty(np.max(releasenums)+1, dtype='|S1')

    #ADM set each entry to 'U' for an unidentified photometric system
    r2p[:] = 'U'

    #ADM populate where the release numbers exist with the PHOTSYS
    r2p[releasenums] = photstrings

    #ADM return the PHOTSYS string that corresponds to each passed release number
    return r2p[release]


def write_targets(filename, data, indir=None, qso_selection=None, 
                  sandboxcuts=False, nside=None):
    """Write a target catalogue.

    Parameters
    ----------
    filename : output target selection file
    data     : numpy structured array of targets to save
    nside: :class:`int`
        If passed, add a column to the targets array popluated 
        with HEALPix pixels at resolution nside
    """
    # FIXME: assert data and tsbits schema

    #ADM use RELEASE to determine the release string for the input targets
    if len(data) == 0:
        #ADM if there are no targets, then we don't know the Data Release
        drstring = 'unknowndr'
    else:
        drint = np.max(data['RELEASE']//1000)
        drstring = 'dr'+str(drint)

    #- Create header to include versions, etc.
    hdr = fitsio.FITSHDR()
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())
    depend.setdep(hdr, 'sandboxcuts', sandboxcuts)
    depend.setdep(hdr, 'photcat', drstring)

    if indir is not None:
        depend.setdep(hdr, 'tractor-files', indir)

    if qso_selection is None:
        print('WARNING: qso_selection method not specified for output file')
        depend.setdep(hdr, 'qso-selection', 'unknown')
    else:
        depend.setdep(hdr, 'qso-selection', qso_selection)

    #ADM add HEALPix column, if requested by input
    if nside is not None:
        theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
        hppix = hp.ang2pix(nside, theta, phi, nest=True)
        data = rfn.append_fields(data, 'HPXPIXEL', hppix, usemask=False)
        depend.setdep(hdr, 'HPXNSIDE', nside)
        depend.setdep(hdr, 'HPXNEST', True)

    #ADM add PHOTSYS column, mapped from RELEASE
    photsys = release_to_photsys(data["RELEASE"])
    data = rfn.append_fields(data, 'PHOTSYS', photsys, usemask=False)    

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
