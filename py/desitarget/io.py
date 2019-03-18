# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=============
desitarget.io
=============

Functions for reading, writing and manipulating files related to targeting.
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import fitsio
import os
import re
from . import __version__ as desitarget_version
import numpy.lib.recfunctions as rfn
import healpy as hp
from glob import glob

from desiutil import depend
from desitarget.geomask import hp_in_box, box_area, is_in_box
from desitarget.geomask import hp_in_cap, cap_area, is_in_cap
from desitarget.geomask import is_in_hp, nside2nside, pixarea2nside

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM this is a lookup dictionary to map RELEASE to a simpler "North" or "South".
# ADM photometric system. This will expand with the definition of RELEASE in the
# ADM Data Model (e.g. https://desi.lbl.gov/trac/wiki/DecamLegacy/DR4sched).
# ADM 7999 were the dr8a test reductions, for which only 'S' surveys were processed.
releasedict = {3000: 'S', 4000: 'N', 5000: 'S', 6000: 'N', 7000: 'S', 7999: 'S',
               8000: 'S', 8001: 'N'}

oldtscolumns = [
    'BRICKID', 'BRICKNAME', 'OBJID', 'TYPE',
    'RA', 'RA_IVAR', 'DEC', 'DEC_IVAR',
    'DECAM_FLUX', 'DECAM_MW_TRANSMISSION',
    'DECAM_FRACFLUX', 'DECAM_FLUX_IVAR', 'DECAM_NOBS', 'DECAM_DEPTH', 'DECAM_GALDEPTH',
    'WISE_FLUX', 'WISE_MW_TRANSMISSION',
    'WISE_FLUX_IVAR',
    'SHAPEDEV_R', 'SHAPEDEV_E1', 'SHAPEDEV_E2',
    'SHAPEDEV_R_IVAR', 'SHAPEDEV_E1_IVAR', 'SHAPEDEV_E2_IVAR',
    'SHAPEEXP_R', 'SHAPEEXP_E1', 'SHAPEEXP_E2',
    'SHAPEEXP_R_IVAR', 'SHAPEEXP_E1_IVAR', 'SHAPEEXP_E2_IVAR',
    'DCHISQ'
    ]

# ADM this is an empty array of the full TS data model columns and dtypes
# ADM other columns can be added in read_tractor.
tsdatamodel = np.array([], dtype=[
    ('RELEASE', '>i2'), ('BRICKID', '>i4'), ('BRICKNAME', 'S8'),
    ('OBJID', '>i4'), ('TYPE', 'S4'), ('RA', '>f8'), ('RA_IVAR', '>f4'),
    ('DEC', '>f8'), ('DEC_IVAR', '>f4'), ('DCHISQ', '>f4', (5,)), ('EBV', '>f4'),
    ('FLUX_G', '>f4'), ('FLUX_R', '>f4'), ('FLUX_Z', '>f4'),
    ('FLUX_IVAR_G', '>f4'), ('FLUX_IVAR_R', '>f4'), ('FLUX_IVAR_Z', '>f4'),
    ('MW_TRANSMISSION_G', '>f4'), ('MW_TRANSMISSION_R', '>f4'), ('MW_TRANSMISSION_Z', '>f4'),
    ('FRACFLUX_G', '>f4'), ('FRACFLUX_R', '>f4'), ('FRACFLUX_Z', '>f4'),
    ('FRACMASKED_G', '>f4'), ('FRACMASKED_R', '>f4'), ('FRACMASKED_Z', '>f4'),
    ('FRACIN_G', '>f4'), ('FRACIN_R', '>f4'), ('FRACIN_Z', '>f4'),
    ('NOBS_G', '>i2'), ('NOBS_R', '>i2'), ('NOBS_Z', '>i2'),
    ('PSFDEPTH_G', '>f4'), ('PSFDEPTH_R', '>f4'), ('PSFDEPTH_Z', '>f4'),
    ('GALDEPTH_G', '>f4'), ('GALDEPTH_R', '>f4'), ('GALDEPTH_Z', '>f4'),
    ('FLUX_W1', '>f4'), ('FLUX_W2', '>f4'), ('FLUX_W3', '>f4'), ('FLUX_W4', '>f4'),
    ('FLUX_IVAR_W1', '>f4'), ('FLUX_IVAR_W2', '>f4'),
    ('FLUX_IVAR_W3', '>f4'), ('FLUX_IVAR_W4', '>f4'),
    ('MW_TRANSMISSION_W1', '>f4'), ('MW_TRANSMISSION_W2', '>f4'),
    ('MW_TRANSMISSION_W3', '>f4'), ('MW_TRANSMISSION_W4', '>f4'),
    ('ALLMASK_G', '>i2'), ('ALLMASK_R', '>i2'), ('ALLMASK_Z', '>i2'),
    ('FRACDEV', '>f4'), ('FRACDEV_IVAR', '>f4'),
    ('SHAPEDEV_R', '>f4'), ('SHAPEDEV_E1', '>f4'), ('SHAPEDEV_E2', '>f4'),
    ('SHAPEDEV_R_IVAR', '>f4'), ('SHAPEDEV_E1_IVAR', '>f4'), ('SHAPEDEV_E2_IVAR', '>f4'),
    ('SHAPEEXP_R', '>f4'), ('SHAPEEXP_E1', '>f4'), ('SHAPEEXP_E2', '>f4'),
    ('SHAPEEXP_R_IVAR', '>f4'), ('SHAPEEXP_E1_IVAR', '>f4'), ('SHAPEEXP_E2_IVAR', '>f4')
    ])

dr7datamodel = np.array([], dtype=[
    ('FIBERFLUX_G', '>f4'), ('FIBERFLUX_R', '>f4'), ('FIBERFLUX_Z', '>f4'),
    ('FIBERTOTFLUX_G', '>f4'), ('FIBERTOTFLUX_R', '>f4'), ('FIBERTOTFLUX_Z', '>f4'),
    ('BRIGHTSTARINBLOB', '?')
    ])

dr8datamodel = np.array([], dtype=[
    ('FIBERFLUX_G', '>f4'), ('FIBERFLUX_R', '>f4'), ('FIBERFLUX_Z', '>f4'),
    ('FIBERTOTFLUX_G', '>f4'), ('FIBERTOTFLUX_R', '>f4'), ('FIBERTOTFLUX_Z', '>f4'),
    ('BRIGHTBLOB', '>i2')
    ])


def desitarget_nside():
    """Default HEALPix Nside for all target selection algorithms."""
    nside = 64
    return nside


def desitarget_resolve_dec():
    """Default Dec cut to separate targets in BASS/MzLS from DECaLS."""
    dec = 32.375
    return dec


def convert_from_old_data_model(fx, columns=None):
    """Read data from open Tractor/sweeps file and convert to DR4+ data model.

    Parameters
    ----------
    fx : :class:`str`
        Open file object corresponding to one Tractor or sweeps file.
    columns: :class:`list`, optional
        the desired Tractor catalog columns to read

    Returns
    -------
    :class:`~numpy.ndarray`
        Array with the tractor schema, uppercase field names.

    Notes
    -----
        - Anything pre-DR3 is assumed to be DR3 (we'd already broken
          backwards-compatability with DR1 because of DECAM_DEPTH but
          this now breaks backwards-compatability with DR2)
    """
    indata = fx[1].read(columns=columns)

    # ADM the number of objects in the input rec array.
    nrows = len(indata)

    # ADM the column names that haven't changed between the current and the old data model.
    tscolumns = list(tsdatamodel.dtype.names)
    sharedcols = list(set(tscolumns).intersection(oldtscolumns))

    # ADM the data types for the new data model.
    dt = tsdatamodel.dtype

    # ADM need to add BRICKPRIMARY and its data type, if it was passed as a column of interest.
    if ('BRICK_PRIMARY' in columns):
        sharedcols.append('BRICK_PRIMARY')
        dd = dt.descr
        dd.append(('BRICK_PRIMARY', '?'))
        dt = np.dtype(dd)

    # ADM create a new numpy array with the fields from the new data model...
    outdata = np.empty(nrows, dtype=dt)

    # ADM ...and populate them with the passed columns of data.
    for col in sharedcols:
        outdata[col] = indata[col]

    # ADM change the DECAM columns from the old (2-D array) to new (named 1-D array) data model.
    decamcols = ['FLUX', 'MW_TRANSMISSION', 'FRACFLUX', 'FLUX_IVAR', 'NOBS', 'GALDEPTH']
    decambands = 'UGRIZ'
    for bandnum in [1, 2, 4]:
        for colstring in decamcols:
            outdata[colstring+"_"+decambands[bandnum]] = indata["DECAM_"+colstring][:, bandnum]
        # ADM treat DECAM_DEPTH separately as the syntax is slightly different.
        outdata["PSFDEPTH_"+decambands[bandnum]] = indata["DECAM_DEPTH"][:, bandnum]

    # ADM change the WISE columns from the old (2-D array) to new (named 1-D array) data model.
    wisecols = ['FLUX', 'MW_TRANSMISSION', 'FLUX_IVAR']
    for bandnum in [1, 2, 3, 4]:
        for colstring in wisecols:
            outdata[colstring+"_W"+str(bandnum)] = indata["WISE_"+colstring][:, bandnum-1]

    # ADM we also need to include the RELEASE, which we'll always assume is DR3
    # ADM (deprecating anything from before DR3).
    outdata['RELEASE'] = 3000

    return outdata


def add_gaia_columns(indata):
    """Add columns needed for MWS targeting to a sweeps-style array.

    Parameters
    ----------
    indata : :class:`~numpy.ndarray`
        Numpy structured array to which to add Gaia-relevant columns.

    Returns
    -------
    :class:`~numpy.ndarray`
        Input array with the Gaia columns added.

    Notes
    -----
        - Gaia columns resemble the data model in :mod:`desitarget.gaiamatch`
          but with "GAIA_RA" and "GAIA_DEC" removed.
    """
    # ADM remove the Gaia coordinates from the Gaia data model as they aren't
    # ADM in the imaging surveys data model.
    from desitarget.gaiamatch import gaiadatamodel, pop_gaia_coords
    gaiadatamodel = pop_gaia_coords(gaiadatamodel)

    # ADM create the combined data model.
    dt = indata.dtype.descr + gaiadatamodel.dtype.descr

    # ADM create a new numpy array with the fields from the new data model...
    nrows = len(indata)
    outdata = np.zeros(nrows, dtype=dt)

    # ADM ...and populate them with the passed columns of data.
    for col in indata.dtype.names:
        outdata[col] = indata[col]

    # ADM set REF_ID to -1 to indicate nothing has a Gaia match (yet).
    outdata['REF_ID'] = -1

    return outdata


def add_dr8_columns(indata):
    """Add columns that are in dr7/dr8 that weren't in dr6.

    Parameters
    ----------
    indata : :class:`~numpy.ndarray`
        Numpy structured array to which to add DR7/DR8 columns.

    Returns
    -------
    :class:`~numpy.ndarray`
        Input array with DR7/DR8 columns added.

    Notes
    -----
        - DR7 columns are stored in :mod:`desitarget.io.dr7datamodel`.
        - DR8 columns are stored in :mod:`desitarget.io.dr8datamodel`.
        - The returned columns are set to all ``0`` or ``False``.
    """
    # ADM if BRIGHSTARINBLOB was sent (the dr7 version of BRIGHTBLOB)
    # ADM then we need to update that column.
    if 'BRIGHTSTARINBLOB' in indata.dtype.names:
        newt = dr8datamodel["BRIGHTBLOB"].dtype.str
        newdt = ("BRIGHTBLOB", newt)
        dt = [fld if fld[0] != 'BRIGHTSTARINBLOB' else newdt
              for fld in indata.dtype.descr]
    else:
        # ADM otherwise, create the combined data model.
        dt = indata.dtype.descr + dr8datamodel.dtype.descr

    # ADM create a new numpy array with the fields from the new data model...
    nrows = len(indata)
    outdata = np.zeros(nrows, dtype=dt)

    # ADM ...and populate them with the passed columns of data.
    for col in indata.dtype.names:
        if col == "BRIGHTSTARINBLOB":
            outdata["BRIGHTBLOB"] = indata["BRIGHTSTARINBLOB"].astype(newt)
        else:
            outdata[col] = indata[col]

    return outdata


def add_photsys(indata):
    """Add the PHOTSYS column to a sweeps-style array.

    Parameters
    ----------
    indata : :class:`~numpy.ndarray`
        Numpy structured array to which to add PHOTSYS column.

    Returns
    -------
    :class:`~numpy.ndarray`
        Input array with PHOTSYS added (and set using RELEASE).

    Notes
    -----
        - The PHOTSYS column is only added if the RELEASE column
          is available in the passed `indata`.
    """
    # ADM only add the PHOTSYS column if RELEASE exists.
    if 'RELEASE' in indata.dtype.names:
        # ADM add PHOTSYS to the data model.
        pdt = [('PHOTSYS', '|S1')]
        dt = indata.dtype.descr + pdt

        # ADM create a new numpy array with the fields from the new data model...
        nrows = len(indata)
        outdata = np.empty(nrows, dtype=dt)

        # ADM ...and populate them with the passed columns of data.
        for col in indata.dtype.names:
            outdata[col] = indata[col]

        # ADM add the PHOTSYS column.
        photsys = release_to_photsys(indata["RELEASE"])
        outdata['PHOTSYS'] = photsys
    else:
        outdata = indata

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
        desitarget.io.tsdatamodel.dtype.names.

    Returns
    -------
    :class:`~numpy.ndarray`
        Array with the tractor schema, uppercase field names.
    """
    check_fitsio_version()

    fx = fitsio.FITS(filename, upper=True)
    fxcolnames = fx[1].get_colnames()
    hdr = fx[1].read_header()

    if columns is None:
        readcolumns = list(tsdatamodel.dtype.names)
        # ADM if RELEASE doesn't exist, then we're pre-DR3 and need the old data model.
        if (('RELEASE' not in fxcolnames) and ('release' not in fxcolnames)):
            readcolumns = list(oldtscolumns)
    else:
        readcolumns = list(columns)

    # - tractor files have BRICK_PRIMARY; sweep files don't
    if (columns is None) and \
       (('BRICK_PRIMARY' in fxcolnames) or ('brick_primary' in fxcolnames)):
        readcolumns.append('BRICK_PRIMARY')

    # ADM if BRIGHTSTARINBLOB exists (it does for DR7, not for DR6) add it and
    # ADM the other DR6->DR7 data model updates.
    if (columns is None) and \
       (('BRIGHTSTARINBLOB' in fxcolnames) or ('brightstarinblob' in fxcolnames)):
        for col in dr7datamodel.dtype.names:
            readcolumns.append(col)
    # ADM if BRIGHTBLOB exists (it does for DR8, not for DR7) add it and
    # ADM the other DR6->DR8 data model updates.
    else:
        if (columns is None) and \
           (('BRIGHTBLOB' in fxcolnames) or ('brightblob' in fxcolnames)):
            for col in dr8datamodel.dtype.names:
                readcolumns.append(col)

    # ADM if Gaia information was passed, add it to the columns to read.
    if (columns is None):
        if (('REF_ID' in fxcolnames) or ('ref_id' in fxcolnames)):
            # ADM remove the Gaia coordinates as they aren't in the imaging data model.
            from desitarget.gaiamatch import gaiadatamodel, pop_gaia_coords, pop_gaia_columns
            gaiadatamodel = pop_gaia_coords(gaiadatamodel)
            # ADM the DR7 sweeps don't contain these columns, but DR8 should.
            if 'REF_CAT' not in fxcolnames:
                gaiadatamodel = pop_gaia_columns(
                    gaiadatamodel,
                    ['REF_CAT', 'GAIA_PHOT_BP_RP_EXCESS_FACTOR',
                     'GAIA_ASTROMETRIC_SIGMA5D_MAX', 'GAIA_ASTROMETRIC_PARAMS_SOLVED']
                )
            gaiacols = gaiadatamodel.dtype.names
            readcolumns += gaiacols

    if (columns is None) and \
       (('RELEASE' not in fxcolnames) and ('release' not in fxcolnames)):
        # ADM Rewrite the data completely to correspond to the DR4+ data model.
        # ADM we default to writing RELEASE = 3000 ("DR3, or before, data")
        data = convert_from_old_data_model(fx, columns=readcolumns)
    else:
        data = fx[1].read(columns=readcolumns)

    # ADM add Gaia columns if not passed.
    if (columns is None) and \
       (('REF_ID' not in fxcolnames) and ('ref_id' not in fxcolnames)):
        data = add_gaia_columns(data)

    # ADM add DR8 data model updates (with zero/False) columns if not passed.
    if (columns is None) and \
       (('BRIGHTBLOB' not in fxcolnames) and ('brightblob' not in fxcolnames)):
        data = add_dr8_columns(data)

    # ADM Empty (length 0) files have dtype='>f8' instead of 'S8' for brickname.
    if len(data) == 0:
        log.warning('WARNING: Empty file>', filename)
        dt = data.dtype.descr
        dt[1] = ('BRICKNAME', 'S8')
        data = data.astype(np.dtype(dt))

    # ADM To circumvent whitespace bugs on I/O from fitsio.
    # ADM need to strip any white space from string columns.
    for colname in data.dtype.names:
        kind = data[colname].dtype.kind
        if kind == 'U' or kind == 'S':
            data[colname] = np.char.rstrip(data[colname])

    # ADM add the PHOTSYS column to unambiguously check whether we're using imaging
    # ADM from the "North" or "South".
    data = add_photsys(data)

    if header:
        fx.close()
        return data, hdr
    else:
        fx.close()
        return data


def fix_tractor_dr1_dtype(objects):
    """DR1 tractor files have inconsistent dtype for the TYPE field.  Fix this.

    Args:
        objects : numpy structured array from target file.

    Returns:
        structured array with TYPE.dtype = 'S4' if needed.

    If the type was already correct, returns the original array.
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
        RELEASE column from a numpy rec array of targets.

    Returns
    -------
    :class:`str` or :class:`~numpy.ndarray`
        'N' if the RELEASE corresponds to the northern photometric
        system (MzLS+BASS) and 'S' if it's the southern system (DECaLS).

    Notes
    -----
    Defaults to 'U' if the system is not recognized.
    """
    # ADM arrays of the key (RELEASE) and value (PHOTSYS) entries in the releasedict.
    releasenums = np.array(list(releasedict.keys()))
    photstrings = np.array(list(releasedict.values()))

    # ADM an array with indices running from 0 to the maximum release number + 1.
    r2p = np.empty(np.max(releasenums)+1, dtype='|S1')

    # ADM set each entry to 'U' for an unidentified photometric system.
    r2p[:] = 'U'

    # ADM populate where the release numbers exist with the PHOTSYS.
    r2p[releasenums] = photstrings

    # ADM return the PHOTSYS string that corresponds to each passed release number.
    return r2p[release]


def write_targets(filename, data, indir=None, qso_selection=None,
                  sandboxcuts=False, nside=None, survey="?",
                  nsidefile=None, hpxlist=None):
    """Write a target catalogue.

    Parameters
    ----------
    filename : :class:`str`
        output target selection file.
    data : :class:`~numpy.ndarray`
        numpy structured array of targets to save.
    indir, qso_selection : :class:`str`, optional, default to `None`
        If passed, note these as the input directory and
        quasar selection method in the output file header.
    sandboxcuts : :class:`bool`, optional, defaults to ``False``
        If passed, note this whether we ran target seletion
        in the sandbox in the output file header.
    nside : :class:`int`, optional, defaults to `None`
        If passed, add a column to the targets array popluated
        with HEALPixels at resolution `nside`.
    survey : :class:`str`, optional, defaults to "?"
        Written to output file header as the keyword `SURVEY`.
    nsidefile : :class:`int`, optional, defaults to `None`
        Passed to indicate in the output file header that the targets
        have been limited to only certain HEALPixels at a given
        nside. Used in conjunction with `hpxlist`.
    hpxlist : :class:`list`, optional, defaults to `None`
        Passed to indicate in the output file header that the targets
        have been limited to only this list of HEALPixels. Used in
        conjunction with `nsidefile`.
    """
    # FIXME: assert data and tsbits schema

    # ADM use RELEASE to determine the release string for the input targets.
    ntargs = len(data)
    if ntargs == 0:
        # ADM if there are no targets, then we don't know the Data Release.
        drstring = 'unknowndr'
    else:
        drint = np.max(data['RELEASE']//1000)
        drstring = 'dr'+str(drint)

    # - Create header to include versions, etc.
    hdr = fitsio.FITSHDR()
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())
    depend.setdep(hdr, 'sandboxcuts', sandboxcuts)
    depend.setdep(hdr, 'photcat', drstring)

    if indir is not None:
        depend.setdep(hdr, 'tractor-files', indir)

    if qso_selection is None:
        log.warning('qso_selection method not specified for output file')
        depend.setdep(hdr, 'qso-selection', 'unknown')
    else:
        depend.setdep(hdr, 'qso-selection', qso_selection)

    # ADM add HEALPix column, if requested by input.
    if nside is not None:
        theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
        hppix = hp.ang2pix(nside, theta, phi, nest=True)
        data = rfn.append_fields(data, 'HPXPIXEL', hppix, usemask=False)
        hdr['HPXNSIDE'] = nside
        hdr['HPXNEST'] = True

    # ADM populate SUBPRIORITY with a reproducible random float.
    if "SUBPRIORITY" in data.dtype.names:
        np.random.seed(616)
        data["SUBPRIORITY"] = np.random.random(ntargs)

    # ADM add the type of survey (main, commissioning; or "cmx", sv) to the header.
    hdr["SURVEY"] = survey

    # ADM record whether this file has been limited to only certain HEALPixels.
    if hpxlist is not None or nsidefile is not None:
        # ADM hpxlist and nsidefile need to be passed together.
        if hpxlist is None or nsidefile is None:
            msg = 'Both hpxlist (={}) and nsidefile (={}) need to be set' \
                .format(hpxlist, nsidefile)
            log.critical(msg)
            raise ValueError(msg)
        hdr['FILENSID'] = nsidefile
        hdr['FILENEST'] = True
        hdr['FILEHPX'] = hpxlist

    fitsio.write(filename, data, extname='TARGETS', header=hdr, clobber=True)


def write_skies(filename, data, indir=None, apertures_arcsec=None,
                nskiespersqdeg=None, nside=None):
    """Write a target catalogue of sky locations.

    Parameters
    ----------
    filename : :class:`str`
        Output target selection file name
    data  : :class:`~numpy.ndarray`
        Array of skies to write to file.
    indir : :class:`str`, optional
        Name of input Legacy Survey Data Release directory, write to header
        of output file if passed (and if not None).
    apertures_arcsec : :class:`list` or `float`, optional
        list of aperture radii in arcseconds to write each aperture as an
        individual line in the header, if passed (and if not None).
    nskiespersqdeg : :class:`float`, optional
        Number of sky locations generated per sq. deg., write to header
        of output file if passed (and if not None).
    nside: :class:`int`, optional
        If passed, add a column to the skies array popluated with HEALPixels
        at resolution `nside`.
    """
    nskies = len(data)

    # ADM force OBSCONDITIONS to be 65535
    # ADM (see https://github.com/desihub/desitarget/pull/313).
    data["OBSCONDITIONS"] = 2**16-1

    # - Create header to include versions, etc.
    hdr = fitsio.FITSHDR()
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())

    if indir is not None:
        depend.setdep(hdr, 'input-data-release', indir)
        # ADM note that if 'dr' is not in the indir DR
        # ADM directory structure, garbage will
        # ADM be rewritten gracefully in the header.
        drstring = 'dr'+indir.split('dr')[-1][0]
        depend.setdep(hdr, 'photcat', drstring)

    if apertures_arcsec is not None:
        for i, ap in enumerate(apertures_arcsec):
            apname = "AP{}".format(i)
            apsize = ap
            hdr[apname] = apsize

    if nskiespersqdeg is not None:
        hdr['NPERSDEG'] = nskiespersqdeg

    # ADM add HEALPix column, if requested by input.
    if nside is not None:
        theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
        hppix = hp.ang2pix(nside, theta, phi, nest=True)
        data = rfn.append_fields(data, 'HPXPIXEL', hppix, usemask=False)
        hdr['HPXNSIDE'] = nside
        hdr['HPXNEST'] = True

    # ADM populate SUBPRIORITY with a reproducible random float.
    if "SUBPRIORITY" in data.dtype.names:
        np.random.seed(616)
        data["SUBPRIORITY"] = np.random.random(nskies)

    fitsio.write(filename, data, extname='SKY_TARGETS', header=hdr, clobber=True)


def write_gfas(filename, data, indir=None, nside=None, survey="?",
               gaiaepoch=None):
    """Write a catalogue of Guide/Focus/Alignment targets.

    Parameters
    ----------
    filename : :class:`str`
        Output file name.
    data  : :class:`~numpy.ndarray`
        Array of GFAs to write to file.
    indir : :class:`str`, optional, defaults to None.
        Name of input Legacy Survey Data Release directory, write to header
        of output file if passed (and if not None).
    nside: :class:`int`, defaults to None.
        If passed, add a column to the GFAs array popluated with HEALPixels
        at resolution `nside`.
    survey : :class:`str`, optional, defaults to "?"
        Written to output file header as the keyword `SURVEY`.
    gaiaepoch: :class:`float`, defaults to None
        Gaia proper motion reference epoch. If not None, write to header of
        output file. If None, default to an epoch of 2015.5.
    """
    # ADM rename 'TYPE' to 'MORPHTYPE'.
    data = rfn.rename_fields(data, {'TYPE': 'MORPHTYPE'})

    # ADM create header to include versions, etc.
    hdr = fitsio.FITSHDR()
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())

    if indir is not None:
        depend.setdep(hdr, 'input-data-release', indir)
        # ADM note that if 'dr' is not in the indir DR
        # ADM directory structure, garbage will
        # ADM be rewritten gracefully in the header.
        drstring = 'dr'+indir.split('dr')[-1][0]
        depend.setdep(hdr, 'photcat', drstring)

    # ADM add HEALPix column, if requested by input.
    if nside is not None:
        theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
        hppix = hp.ang2pix(nside, theta, phi, nest=True)
        data = rfn.append_fields(data, 'HPXPIXEL', hppix, usemask=False)
        hdr['HPXNSIDE'] = nside
        hdr['HPXNEST'] = True

    # ADM add the type of survey (main, or commissioning "cmx") to the header.
    hdr["SURVEY"] = survey

    # ADM add the Gaia reference epoch, or pass 2015.5 if not included.
    hdr['REFEPOCH'] = {'name': 'REFEPOCH',
                       'value': 2015.5,
                       'comment': "Gaia Proper Motion Reference Epoch"}
    if gaiaepoch is not None:
        hdr['REFEPOCH'] = gaiaepoch

    fitsio.write(filename, data, extname='GFA_TARGETS', header=hdr, clobber=True)


def write_randoms(filename, data, indir=None, hdr=None, nside=None, density=None):
    """Write a catalogue of randoms and associated pixel-level information.

    Parameters
    ----------
    filename : :class:`str`
        Output file name.
    data  : :class:`~numpy.ndarray`
        Array of randoms to write to file.
    indir : :class:`str`, optional, defaults to None
        Name of input Legacy Survey Data Release directory, write to header
        of output file if passed (and if not None).
    hdr : :class:`str`, optional, defaults to `None`
        If passed, use this header to start the header of the output `filename`.
    nside: :class:`int`
        If passed, add a column to the randoms array popluated with HEALPixels
        at resolution `nside`.
    density: :class:`int`
        Number of points per sq. deg. at which the catalog was generated,
        write to header of the output file if not None.
    """
    # ADM create header to include versions, etc. If a `hdr` was
    # ADM passed, then use it, if not then create a new header.
    if hdr is None:
        hdr = fitsio.FITSHDR()
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())

    if indir is not None:
        depend.setdep(hdr, 'input-data-release', indir)
        # ADM note that if 'dr' is not in the indir DR
        # ADM directory structure, garbage will
        # ADM be rewritten gracefully in the header.
        drstring = 'dr'+indir.split('dr')[-1][0]
        depend.setdep(hdr, 'photcat', drstring)
        # ADM also write the mask bits header information
        # ADM from a mask bits file in this DR.
        from glob import iglob
        files = iglob(indir+'/coadd/*/*/*maskbits*')
        # ADM we built an iterator over mask bits files for speed
        # ADM if there are no such files to iterate over, just pass.
        try:
            fn = next(files)
            mbhdr = fitsio.read_header(fn)
            # ADM extract the keys that include the string 'BITNM'.
            bncols = [key for key in mbhdr.keys() if 'BITNM' in key]
            for col in bncols:
                hdr[col] = {'name': col,
                            'value': mbhdr[col],
                            'comment': mbhdr.get_comment(col)}
        except StopIteration:
            pass

    # ADM add HEALPix column, if requested by input.
    if nside is not None:
        theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
        hppix = hp.ang2pix(nside, theta, phi, nest=True)
        data = rfn.append_fields(data, 'HPXPIXEL', hppix, usemask=False)
        hdr['HPXNSIDE'] = nside
        hdr['HPXNEST'] = True

    # ADM add density of points if requested by input.
    if density is not None:
        hdr['DENSITY'] = density

    fitsio.write(filename, data, extname='RANDOMS', header=hdr, clobber=True)


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
    # ADM check for duplicate files in case the listing was run
    # ADM at too low a level in the directory structure.
    check = [os.path.basename(x) for x in iter_sweepfiles(root)]
    if len(check) != len(set(check)):
        log.error("Duplicate sweep files in root directory!")

    return [x for x in iter_sweepfiles(root)]


def iter_sweepfiles(root):
    """Iterator over all sweep files found under root directory.
    """
    return iter_files(root, prefix='sweep', ext='fits')


def list_tractorfiles(root):
    """Return a list of tractor files found under `root` directory.
    """
    # ADM check for duplicate files in case the listing was run
    # ADM at too low a level in the directory structure.
    check = [os.path.basename(x) for x in iter_tractorfiles(root)]
    if len(check) != len(set(check)):
        log.error("Duplicate Tractor files in root directory!")

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


def brickname_from_filename_with_prefix(filename, prefix=''):
    """Parse `filename` to check if this is a brick file with a given prefix.

    Parameters
    ----------
    filename : :class:`str`
        Full name of a brick file.
    prefix : :class:`str`
        Optional part of filename immediately preceding the brickname.

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
    if (
        LooseVersion(fitsio.__version__) < LooseVersion(version) and
        not fitsio.__version__.startswith(version)
    ):
        raise ImportError(('ERROR: fitsio >{0}rc1 required ' +
                           '(not {1})!').format(version, fitsio.__version__))


def whitespace_fits_read(filename, **kwargs):
    """Use fitsio_ to read in a file and strip whitespace from all string columns.

    .. _fitsio: https://pypi.python.org/pypi/fitsio

    Parameters
    ----------
    filename : :class:`str`
        Name of the file to be read in by fitsio.
    kwargs: arguments that will be passed directly to fitsio.
    """
    fitout = fitsio.read(filename, **kwargs)
    # ADM if the header=True option was passed then
    # ADM the output is the header and the data.
    data = fitout
    if 'header' in kwargs:
        data, header = fitout

    # ADM guard against the zero-th extension being read by fitsio.
    if data is not None:
        # ADM strip any whitespace from string columns.
        for colname in data.dtype.names:
            kind = data[colname].dtype.kind
            if kind == 'U' or kind == 'S':
                data[colname] = np.char.rstrip(data[colname])

    if 'header' in kwargs:
        return data, header

    return data


def load_pixweight(inmapfile, nside, pixmap=None):
    """Loads a pixel map from file and resamples to a different HEALPixel resolution (nside)

    Parameters
    ----------
    inmapfile : :class:`str`
        Name of the file containing the pixel weight map.
    nside : :class:`int`
        After loading, the array will be resampled to this HEALPix nside.
    pixmap: `~numpy.array`, optional, defaults to None
        Pass a pixel map instead of loading it from file.

    Returns
    -------
    :class:`~numpy.array`
        HEALPixel weight map resampled to the requested nside.
    """
    if pixmap is not None:
        log.debug('Using input pixel weight map of length {}.'.format(len(pixmap)))
    else:
        # ADM read in the pixel weights file.
        if not os.path.exists(inmapfile):
            log.fatal('Input directory does not exist: {}'.format(inmapfile))
            raise ValueError
        pixmap = fitsio.read(inmapfile)

    # ADM determine the file's nside, and flag a warning if the passed nside exceeds it.
    npix = len(pixmap)
    truenside = hp.npix2nside(len(pixmap))
    if truenside < nside:
        log.warning("downsampling is fuzzy...Passed nside={}, but file {} is stored at nside={}"
                    .format(nside, inmapfile, truenside))

    # ADM resample the map.
    return hp.pixelfunc.ud_grade(pixmap, nside, order_in='NESTED', order_out='NESTED')


def load_pixweight_recarray(inmapfile, nside, pixmap=None):
    """Like load_pixweight but for a structured array map with multiple columns

    Parameters
    ----------
    inmapfile : :class:`str`
        Name of the file containing the pixel weight map.
    nside : :class:`int`
        After loading, the array will be resampled to this HEALPix nside.
    pixmap: `~numpy.array`, optional, defaults to None
        Pass a pixel map instead of loading it from file.

    Returns
    -------
    :class:`~numpy.array`
        HEALPixel weight map with all columns resampled to the requested nside.

    Notes
    -----
        - Assumes that tha passed map is in the NESTED scheme, and outputs to
          the NESTED scheme.
        - All columns are resampled as the mean of the relevant pixels, except
          if a column `HPXPIXEL` is passed. That column is reassigned the appropriate
          pixel number at the new nside.
    """
    if pixmap is not None:
        log.debug('Using input pixel weight map of length {}.'.format(len(pixmap)))
    else:
        # ADM read in the pixel weights file.
        if not os.path.exists(inmapfile):
            log.fatal('Input directory does not exist: {}'.format(inmapfile))
            raise ValueError
        pixmap = fitsio.read(inmapfile)

    # ADM determine the file's nside, and flag a warning if the passed nside exceeds it.
    npix = len(pixmap)
    truenside = hp.npix2nside(len(pixmap))
    if truenside < nside:
        log.warning("downsampling is fuzzy...Passed nside={}, but file {} is stored at nside={}"
                    .format(nside, inmapfile, truenside))

    # ADM set up an output array.
    nrows = hp.nside2npix(nside)
    outdata = np.zeros(nrows, dtype=pixmap.dtype)

    # ADM resample the map for each column.
    for col in pixmap.dtype.names:
        outdata[col] = hp.pixelfunc.ud_grade(pixmap[col], nside, order_in='NESTED', order_out='NESTED')

    # ADM if one column was the HEALPixel number, recalculate for the new resolution.
    if 'HPXPIXEL' in pixmap.dtype.names:
        outdata["HPXPIXEL"] = np.arange(nrows)

    return outdata


def gitversion():
    """Returns `git describe --tags --dirty --always`,
    or 'unknown' if not a git repo"""
    import os
    from subprocess import Popen, PIPE, STDOUT
    origdir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    try:
        p = Popen(['git', "describe", "--tags", "--dirty", "--always"], stdout=PIPE, stderr=STDOUT)
    except EnvironmentError:
        return 'unknown'

    os.chdir(origdir)
    out = p.communicate()[0]
    if p.returncode == 0:
        # - avoid py3 bytes and py3 unicode; get native str in both cases
        return str(out.rstrip().decode('ascii'))
    else:
        return 'unknown'


def read_external_file(filename, header=False, columns=["RA", "DEC"]):
    """Read FITS file with loose requirements on upper-case columns and EXTNAME.

    Parameters
    ----------
    filename : :class:`str`
        File name with full directory path included.
    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return (data, header) instead of just data.
    columns: :class:`list`, optional, defaults to ["RA", "DEC"]
        Specify the desired columns to read.

    Returns
    -------
    :class:`~numpy.ndarray`
        The output data array.
    :class:`~numpy.ndarray`, optional
        The output file header, if input `header` was ``True``.

    Notes
    -----
        - Intended to be used with externally supplied files such as locations
          to be matched for commissioning or secondary targets.
    """
    # ADM check we aren't going to have an epic fail on the the version of fitsio.
    check_fitsio_version()

    # ADM prepare to read in the data by reading in columns.
    fx = fitsio.FITS(filename, upper=True)
    fxcolnames = fx[1].get_colnames()
    hdr = fx[1].read_header()

    # ADM convert the columns to upper case...
    colnames = [colname.upper() for colname in fxcolnames]
    # ADM ...and fail if RA and DEC aren't columns.
    if not ("RA" in colnames and "DEC" in colnames):
        msg = 'Input file {} must contain both "RA" and "DEC" columns' \
             .format(filename)
        log.critical(msg)
        raise ValueError(msg)

    # ADM read in the RA/DEC columns.
    outdata = fx[1].read(columns=["RA", "DEC"])

    # ADM return data read in from file, with the header if requested.
    fx.close()
    if header:
        return outdata, hdr
    else:
        return outdata


def decode_sweep_name(sweepname, nside=None, inclusive=True, fact=4):
    """Retrieve RA/Dec edges from a full directory path to a sweep file

    Parameters
    ----------
    sweepname : :class:`str`
        Full path to a sweep file, e.g., /a/b/c/sweep-350m005-360p005.fits
    nside : :class:`int`, optional, defaults to None
        (NESTED) HEALPixel nside
    inclusive : :class:`book`, optional, defaults to ``True``
        see documentation for `healpy.query_polygon()`
    fact : :class:`int`, optional defaults to 4
        see documentation for `healpy.query_polygon()`

    Returns
    -------
    :class:`list` (if nside is None)
        A 4-entry list of the edges of the region covered by the sweeps file
        in the form [RAmin, RAmax, DECmin, DECmax]
        For the above example this would be [350., 360., -5., 5.]
    :class:`list` (if nside is not None)
        A list of HEALPixels that touch the  files at the passed `nside`
        For the above example this would be [16, 17, 18, 19]
    """
    # ADM extract just the file part of the name.
    sweepname = os.path.basename(sweepname)

    # ADM the RA/Dec edges.
    ramin, ramax = float(sweepname[6:9]), float(sweepname[14:17])
    decmin, decmax = float(sweepname[10:13]), float(sweepname[18:21])

    # ADM flip the signs on the DECs, if needed.
    if sweepname[9] == 'm':
        decmin *= -1
    if sweepname[17] == 'm':
        decmax *= -1

    if nside is None:
        return [ramin, ramax, decmin, decmax]

    pixnum = hp_in_box(nside, [ramin, ramax, decmin, decmax],
                       inclusive=inclusive, fact=fact)

    return pixnum


def check_hp_target_dir(hpdirname):
    """Check fidelity of a directory of HEALPixel-partitioned targets.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to a directory containing targets that have been
        split by HEALPixel.

    Returns
    -------
    :class:`int`
        The HEALPixel NSIDE for the files in the passed directory.
    :class:`dict`
        A dictionary where the keys are each HEALPixel covered in the
        passed directory and the values are the file that includes
        that HEALPixel.

    Notes
    -----
        - Checks that all files are at the same NSIDE.
        - Checks that no two files contain the same HEALPixels.
        - Checks that HEALPixel numbers are consistent with NSIDE.
    """
    # ADM glob all the files in the directory, read the pixel
    # ADM numbers and NSIDEs.
    nside = []
    pixlist = []
    fns = glob(os.path.join(hpdirname, "*fits"))
    pixdict = {}
    for fn in fns:
        hdr = fitsio.read_header(fn, "TARGETS")
        nside.append(hdr["FILENSID"])
        pixels = hdr["FILEHPX"]
        # ADM create a look-up dictionary of file-for-each-pixel.
        for pix in pixels:
            pixdict[pix] = fn
        pixlist.append(pixels)
    nside = np.array(nside)
    # ADM as well as having just an array of all the pixels.
    pixlist = np.hstack(pixlist)

    msg = None
    # ADM check all NSIDEs are the same.
    if not np.all(nside == nside[0]):
        msg = 'Not all files in {} are at the same NSIDE'     \
            .format(hpdirname)

    # ADM check that no two files contain the same HEALPixels.
    if not len(set(pixlist)) == len(pixlist):
        dup = set([pix for pix in pixlist if list(pixlist).count(pix) > 1])
        msg = 'Duplicate pixel ({}) in files in {}'           \
            .format(dup, hpdirname)

    # ADM check that the pixels are consistent with the nside.
    goodpix = np.arange(hp.nside2npix(nside[0]))
    badpix = set(pixlist) - set(goodpix)
    if len(badpix) > 0:
        msg = 'Pixel ({}) not allowed at NSIDE={} in {}'.     \
              format(badpix, nside[0], hpdirname)

    if msg is not None:
        log.critical(msg)
        raise AssertionError(msg)

    return nside[0], pixdict


def read_targets_in_hp(hpdirname, nside, pixlist, columns=None):
    """Read in targets in a set of HEALPixels.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to either a directory containing targets that
        have been partitioned by HEALPixel (i.e. as made by
        `select_targets` with the `bundle_files` option). Or the
        name of a single file of targets.
    nside : :class:`int`
        The (NESTED) HEALPixel nside.
    pixlist : :class:`list` or `int` or `~numpy.ndarray`
        Return targets in these HEALPixels at the passed `nside`.
    columns : :class:`list`, optional
        Only read in these target columns.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of targets in the passed pixels.
    """
    # ADM we'll need RA/Dec for final cuts, so ensure they're read.
    addedcols = []
    columnscopy = None
    if columns is not None:
        # ADM make a copy of columns, as it's a kwarg we'll modify.
        columnscopy = columns.copy()
        for radec in ["RA", "DEC"]:
            if radec not in columnscopy:
                columnscopy.append(radec)
                addedcols.append(radec)

    # ADM if a directory was passed, do fancy HEALPixel parsing...
    if os.path.isdir(hpdirname):
        # ADM check, and grab information from, the target directory.
        filenside, filedict = check_hp_target_dir(hpdirname)

        # ADM change the passed pixels to the nside of the file schema.
        filepixlist = nside2nside(nside, filenside, pixlist)

        # ADM only consider pixels for which we have a file.
        isindict = [pix in filedict for pix in filepixlist]
        filepixlist = filepixlist[isindict]

        # ADM make sure each file is only read once.
        infiles = set([filedict[pix] for pix in filepixlist])

        # ADM read in the files and concatenate the resulting targets.
        targets = []
        for infile in infiles:
            targets.append(fitsio.read(infile, columns=columnscopy))
        targets = np.concatenate(targets)
    # ADM ...otherwise just read in the targets.
    else:
        targets = fitsio.read(hpdirname, columns=columnscopy)

    # ADM restrict the targets to the actual requested HEALPixels...
    ii = is_in_hp(targets, nside, pixlist)
    # ADM ...and remove RA/Dec columns if we added them.
    targets = rfn.drop_fields(targets[ii], addedcols)

    return targets


def read_targets_in_box(hpdirname, radecbox=[0., 360., -90., 90.],
                        columns=None):
    """Read in targets in an RA/Dec box.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to either a directory containing targets that
        have been partitioned by HEALPixel (i.e. as made by
        `select_targets` with the `bundle_files` option). Or the
        name of a single file of targets.
    radecbox : :class:`list`, defaults to the entire sky
        4-entry list of coordinates [ramin, ramax, decmin, decmax]
        forming the edges of a box in RA/Dec (degrees).
    columns : :class:`list`, optional
        Only read in these target columns.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of targets in the passed RA/Dec box.
    """
    # ADM we'll need RA/Dec for final cuts, so ensure they're read.
    addedcols = []
    columnscopy = None
    if columns is not None:
        # ADM make a copy of columns, as it's a kwarg we'll modify.
        columnscopy = columns.copy()
        for radec in ["RA", "DEC"]:
            if radec not in columnscopy:
                columnscopy.append(radec)
                addedcols.append(radec)

    # ADM if a directory was passed, do fancy HEALPixel parsing...
    if os.path.isdir(hpdirname):
        # ADM approximate nside for area of passed box.
        nside = pixarea2nside(box_area(radecbox))

        # ADM HEALPixels that touch the box for that nside.
        pixlist = hp_in_box(nside, radecbox)

        # ADM read in targets in these HEALPixels.
        targets = read_targets_in_hp(hpdirname, nside, pixlist,
                                     columns=columnscopy)
    # ADM ...otherwise just read in the targets.
    else:
        targets = fitsio.read(hpdirname, columns=columnscopy)

    # ADM restrict only to targets in the requested RA/Dec box...
    ii = is_in_box(targets, radecbox)
    # ADM ...and remove RA/Dec columns if we added them.
    targets = rfn.drop_fields(targets[ii], addedcols)

    return targets


def read_targets_in_cap(hpdirname, radecrad, columns=None):
    """Read in targets in an RA, Dec, radius cap.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to either a directory containing targets that
        have been partitioned by HEALPixel (i.e. as made by
        `select_targets` with the `bundle_files` option). Or the
        name of a single file of targets.
    radecrad : :class:`list`
        3-entry list of coordinates [ra, dec, radius] forming a cap or
        "circle" on the sky. ra, dec and radius are all in degrees.
    columns : :class:`list`, optional
        Only read in these target columns.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of targets in the passed RA/Dec box.
    """
    # ADM we'll need RA/Dec for final cuts, so ensure they're read.
    addedcols = []
    columnscopy = None
    if columns is not None:
        # ADM make a copy of columns, as it's a kwarg we'll modify.
        columnscopy = columns.copy()
        for radec in ["RA", "DEC"]:
            if radec not in columnscopy:
                columnscopy.append(radec)
                addedcols.append(radec)

    # ADM if a directory was passed, do fancy HEALPixel parsing...
    if os.path.isdir(hpdirname):
        # ADM approximate nside for area of passed cap.
        nside = pixarea2nside(cap_area(np.array(radecrad[2])))

        # ADM HEALPixels that touch the cap for that nside.
        pixlist = hp_in_cap(nside, radecrad)

        # ADM read in targets in these HEALPixels.
        targets = read_targets_in_hp(hpdirname, nside, pixlist,
                                     columns=columnscopy)
    # ADM ...otherwise just read in the targets.
    else:
        targets = fitsio.read(hpdirname, columns=columnscopy)

    # ADM restrict only to targets in the requested cap...
    ii = is_in_cap(targets, radecrad)
    # ADM ...and remove RA/Dec columns if we added them.
    targets = rfn.drop_fields(targets[ii], addedcols)

    return targets
