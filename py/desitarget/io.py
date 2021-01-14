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
# import pandas as pd
import fitsio
from astropy.table import Table
import os
import re
from . import __version__ as desitarget_version
import numpy.lib.recfunctions as rfn
import healpy as hp
from glob import glob, iglob
from time import time
from pkg_resources import resource_filename
import yaml
import hashlib

from desiutil import depend
from desitarget.geomask import hp_in_box, box_area, is_in_box
from desitarget.geomask import hp_in_cap, cap_area, is_in_cap, add_hp_neighbors
from desitarget.geomask import is_in_hp, nside2nside, pixarea2nside
from desitarget.targets import main_cmx_or_sv, decode_targetid
from desimodel.footprint import is_point_in_desi, tiles2pix

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM this is a lookup dictionary to map RELEASE to a simpler "North" or "South".
# ADM photometric system. This will expand with the definition of RELEASE in the
# ADM Data Model (e.g. https://desi.lbl.gov/trac/wiki/DecamLegacy/DR4sched).
# ADM 7999 were the dr8a test reductions, for which only 'S' surveys were processed.
releasedict = {3000: 'S', 4000: 'N', 5000: 'S', 6000: 'N', 7000: 'S', 7999: 'S',
               8000: 'S', 8001: 'N', 9000: 'S', 9001: 'N', 9002: 'S', 9003: 'N',
               9004: 'S', 9005: 'N', 9006: 'S', 9007: 'N', 9008: 'S', 9009: 'N',
               9010: 'S', 9011: 'N', 9012: 'S', 9013: 'N'}

# ADM This is an empty array of most of the TS data model columns and
# ADM dtypes. Note that other columns are added in read_tractor and
# ADM from the "addedcols" data models below.
basetsdatamodel = np.array([], dtype=[
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
    ('FIBERFLUX_G', '>f4'), ('FIBERFLUX_R', '>f4'), ('FIBERFLUX_Z', '>f4'),
    ('FIBERTOTFLUX_G', '>f4'), ('FIBERTOTFLUX_R', '>f4'), ('FIBERTOTFLUX_Z', '>f4'),
    ('REF_EPOCH', '>f4'), ('WISEMASK_W1', '|u1'), ('WISEMASK_W2', '|u1'),
    ('MASKBITS', '>i2')
    ])

# ADM columns that are new for the DR9 data model.
dr9addedcols = np.array([], dtype=[
    ('LC_FLUX_W1', '>f4', (15,)), ('LC_FLUX_W2', '>f4', (15,)),
    ('LC_FLUX_IVAR_W1', '>f4', (15,)), ('LC_FLUX_IVAR_W2', '>f4', (15,)),
    ('LC_NOBS_W1', '>i2', (15,)), ('LC_NOBS_W2', '>i2', (15,)),
    ('LC_MJD_W1', '>f8', (15,)), ('LC_MJD_W2', '>f8', (15,)),
    ('SHAPE_R', '>f4'), ('SHAPE_E1', '>f4'), ('SHAPE_E2', '>f4'),
    ('SHAPE_R_IVAR', '>f4'), ('SHAPE_E1_IVAR', '>f4'), ('SHAPE_E2_IVAR', '>f4'),
    ('SERSIC', '>f4'), ('SERSIC_IVAR', '>f4')
])

# ADM columns that were deprecated in the DR8 data model.
dr8addedcols = np.array([], dtype=[
    ('FRACDEV', '>f4'), ('FRACDEV_IVAR', '>f4'),
    ('SHAPEDEV_R', '>f4'), ('SHAPEDEV_E1', '>f4'), ('SHAPEDEV_E2', '>f4'),
    ('SHAPEDEV_R_IVAR', '>f4'), ('SHAPEDEV_E1_IVAR', '>f4'), ('SHAPEDEV_E2_IVAR', '>f4'),
    ('SHAPEEXP_R', '>f4'), ('SHAPEEXP_E1', '>f4'), ('SHAPEEXP_E2', '>f4'),
    ('SHAPEEXP_R_IVAR', '>f4'), ('SHAPEEXP_E1_IVAR', '>f4'), ('SHAPEEXP_E2_IVAR', '>f4'),
    ])


def desitarget_nside():
    """Default HEALPix Nside for all target selection algorithms."""
    nside = 64
    return nside


def desitarget_resolve_dec():
    """Default Dec cut to separate targets in BASS/MzLS from DECaLS."""
    dec = 32.375
    return dec


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
        # ADM the fitsio check is a hack for the v0.9 to v1.0 transition
        # ADM (v1.0 now converts all byte strings to unicode strings).
        from distutils.version import LooseVersion
        if LooseVersion(fitsio.__version__) >= LooseVersion('1'):
            pdt = [('PHOTSYS', '<U1')]
        else:
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
    """Read a tractor catalogue or sweeps file.

    Parameters
    ----------
    filename : :class:`str`
        File name of one Tractor or sweeps file.
    header : :class:`bool`, optional
        If ``True``, return (data, header) instead of just data.
    columns: :class:`list`, optional
        Specify the desired Tractor catalog columns to read; defaults to
        desitarget.io.tsdatamodel.dtype.names + most of the columns in
        desitarget.gaiamatch.gaiadatamodel.dtype.names, where
        tsdatamodel is, e.g., basetsdatamodel + dr9addedcols.

    Returns
    -------
    :class:`~numpy.ndarray`
        Array with the tractor schema, uppercase field names.
    """
    check_fitsio_version()

    # ADM read in the file information. Due to fitsio header bugs
    # ADM near v1.0.0, make absolutely sure the user wants the header.
    if header:
        indata, hdr = fitsio.read(filename, upper=True, header=True,
                                  columns=columns)
    else:
        indata = fitsio.read(filename, upper=True, columns=columns)

    # ADM form the final data model in a manner that maintains
    # ADM backwards-compatability with DR8.
    if "FRACDEV" in indata.dtype.names:
        tsdatamodel = np.array(
            [], dtype=basetsdatamodel.dtype.descr + dr8addedcols.dtype.descr)
    else:
        tsdatamodel = np.array(
            [], dtype=basetsdatamodel.dtype.descr + dr9addedcols.dtype.descr)

    # ADM the full data model including Gaia columns.
    from desitarget.gaiamatch import gaiadatamodel
    from desitarget.gaiamatch import pop_gaia_coords, pop_gaia_columns
    gaiadatamodel = pop_gaia_coords(gaiadatamodel)

    # ADM special handling of the pre-DR7 Data Model.
    for gaiacol in ['GAIA_PHOT_BP_RP_EXCESS_FACTOR',
                    'GAIA_ASTROMETRIC_SIGMA5D_MAX',
                    'GAIA_ASTROMETRIC_PARAMS_SOLVED', 'REF_CAT']:
        if gaiacol not in indata.dtype.names:
            gaiadatamodel = pop_gaia_columns(gaiadatamodel, [gaiacol])
    dt = tsdatamodel.dtype.descr + gaiadatamodel.dtype.descr
    dtnames = tsdatamodel.dtype.names + gaiadatamodel.dtype.names
    # ADM limit to just passed columns.
    if columns is not None:
        dt = [d for d, name in zip(dt, dtnames) if name in columns]

    # ADM set-up the output array.
    nrows = len(indata)
    data = np.zeros(nrows, dtype=dt)
    # ADM if REF_ID was requested, set it to -1 in case there is no Gaia data.
    if "REF_ID" in data.dtype.names:
        data['REF_ID'] = -1

    # ADM populate the common input/output columns.
    for col in set(indata.dtype.names).intersection(set(data.dtype.names)):
        data[col] = indata[col]

    # ADM MASKBITS used to be BRIGHTSTARINBLOB which was set to True/False
    # ADM and which represented the SECOND bit of MASKBITS.
    if "BRIGHTSTARINBLOB" in indata.dtype.names:
        if "MASKBITS" in data.dtype.names:
            data["MASKBITS"] = indata["BRIGHTSTARINBLOB"] << 1

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
        return data, hdr
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
    Flags an error if the system is not recognized.
    """
    # ADM arrays of the key (RELEASE) and value (PHOTSYS) entries in the releasedict.
    releasenums = np.array(list(releasedict.keys()))
    photstrings = np.array(list(releasedict.values()))

    # ADM explicitly check no unknown release numbers were passed.
    unknown = set(release) - set(releasenums)
    if bool(unknown):
        msg = 'Unknown release number {}'.format(unknown)
        log.critical(msg)
        raise ValueError(msg)

    # ADM an array with indices running from 0 to the maximum release number + 1.
    r2p = np.empty(np.max(releasenums)+1, dtype='|S1')

    # ADM populate where the release numbers exist with the PHOTSYS.
    r2p[releasenums] = photstrings

    # ADM return the PHOTSYS string that corresponds to each passed release number.
    return r2p[release]


def _bright_or_dark(filename, hdr, data, obscon, mockdata=None):
    """modify data/file name for BRIGHT or DARK survey OBSCONDITIONS

    Parameters
    ----------
    filename : :class:`str`
        output target selection file.
    hdr : class:`str`
        header of the output target selection file.
    data : :class:`~numpy.ndarray`
        numpy structured array of targets.
    obscon : :class:`str`
        Can be "DARK" or "BRIGHT" to only write targets appropriate for
        "DARK|GRAY" or "BRIGHT" observing conditions. The relevant
        `PRIORITY_INIT` and `NUMOBS_INIT` columns will be derived from
        `PRIORITY_INIT_DARK`, etc. and `filename` will have "bright" or
        "dark" appended to the lowest DIRECTORY in the input `filename`.
    mockdata : :class:`dict`, optional, defaults to `None`
        Dictionary of mock data to write out (only used in
        `desitarget.mock.build.targets_truth` via `select_mock_targets`).

    Returns
    -------
    :class:`str`
        The modified file name.
    :class:`data`
        The modified data.
    """
    # ADM determine the bits for the OBSCONDITIONS.
    from desitarget.targetmask import obsconditions
    if obscon == "DARK":
        obsbits = obsconditions.mask("DARK|GRAY")
        hdr["OBSCON"] = "DARK|GRAY"
    else:
        # ADM will flag an error if obscon is not, now BRIGHT.
        obsbits = obsconditions.mask(obscon)
        hdr["OBSCON"] = obscon
    # ADM only retain targets appropriate to the conditions.
    ii = (data["OBSCONDITIONS"] & obsbits) != 0
    data = data[ii]

    # Optionally subselect the mock data.
    if len(data) > 0 and mockdata is not None:
        truthdata, trueflux, _objtruth = mockdata['truth'], mockdata['trueflux'], mockdata['objtruth']
        truthdata = truthdata[ii]

        objtruth = {}
        for obj in sorted(set(truthdata['TEMPLATETYPE'])):
            objtruth[obj] = _objtruth[obj]
        for key in objtruth.keys():
            keep = np.where(np.isin(objtruth[key]['TARGETID'], truthdata['TARGETID']))[0]
            if len(keep) > 0:
                objtruth[key] = objtruth[key][keep]

        if len(trueflux) > 0 and trueflux.shape[1] > 0:
            trueflux = trueflux[ii, :]

        mockdata['truth'] = truthdata
        mockdata['trueflux'] = trueflux
        mockdata['objtruth'] = objtruth

    # ADM construct the name for the bright or dark directory.
    newdir = os.path.join(os.path.dirname(filename), obscon.lower())
    filename = os.path.join(newdir, os.path.basename(filename))
    # ADM modify the filename with an obscon prefix.
    filename = filename.replace("targets-", "targets-{}-".format(obscon.lower()))

    # ADM change the name to PRIORITY_INIT, NUMOBS_INIT.
    for col in "NUMOBS_INIT", "PRIORITY_INIT":
        rename = {"{}_{}".format(col, obscon.upper()): col}
        data = rfn.rename_fields(data, rename)

    # ADM remove the other BRIGHT/DARK NUMOBS, PRIORITY columns.
    names = np.array(data.dtype.names)
    dropem = list(names[['_INIT_' in col for col in names]])
    data = rfn.drop_fields(data, dropem)

    if mockdata is not None:
        return filename, hdr, data, mockdata
    else:
        return filename, hdr, data


def write_with_units(filename, data, extname=None, header=None, ecsv=False):
    """Write a FITS file with units from the desitarget data model.

    Parameters
    ----------
    filename : :class:`str`
        The output file.
    data : :class:`~numpy.ndarray`
        The numpy structured array of data to write.
    extname, header optional
        Passed through to `fitsio.write()`. `header` can be either
        a FITShdr object or a dictionary.
    ecsv : :class:`bool`, optional, defaults to ``False``
        If ``True`` then write as a .ecsv file instead of FITS.

    Returns
    -------
    Nothing, but writes the `data` to the `filename` in chunks with units
    added from the desitarget units yaml file (see `/data/units.yaml`).

    Notes
    -----
        - Always OVERWRITES existing files!
        - Writes atomically. Any files that died mid-write will be
          appended by ".tmp".
        - If `ecsv` is ``True`` then a (potentially slow) Table
          conversion is applied to `data`.
    """
    # ADM read the desitarget units yaml file.
    fn = resource_filename('desitarget', os.path.join('data', 'units.yaml'))
    with open(fn) as f:
        unitdict = yaml.safe_load(f)

    if ecsv:
        data = Table(data)
    # ADM loop through the data and create a list of units.
    units = []
    for col in data.dtype.names:
        try:
            if unitdict[col] is None:
                units.append("")
            else:
                units.append(unitdict[col])
            if ecsv:
                data[col].unit = unitdict[col]
        except KeyError:
            units.append("")

    # ADM write the file for either ecsv or fits..
    if ecsv:
        data.meta = dict(header)
        data.meta['EXTNAME'] = extname
        data.write(filename+'.tmp', format='ascii.ecsv', overwrite=True)
    else:
        fitsio.write(filename+'.tmp', data, units=units, extname=extname,
                     header=header, clobber=True)
    os.rename(filename+'.tmp', filename)

    return


def write_targets(targdir, data, indir=None, indir2=None, nchunks=None,
                  qso_selection=None, nside=None, survey="main", nsidefile=None,
                  hpxlist=None, scndout=None, resolve=True, maskbits=True,
                  obscon=None, mockdata=None, supp=False, extra=None,
                  infiles=None):
    """Write target catalogues.

    Parameters
    ----------
    targdir : :class:`str`
        Path to output target selection directory (the directory
        structure and file name are built on-the-fly from other inputs).
    data : :class:`~numpy.ndarray`
        numpy structured array of targets to save.
    indir, indir2, qso_selection : :class:`str`, optional, default to `None`
        If passed, note these as the input directory, an additional input
        directory, and the QSO selection method in the output file header.
    nchunks : :class`int`, optional, defaults to `None`
        The number of chunks in which to write the output file, to save
        memory. Send `None` to write everything at once.
    nside : :class:`int`, optional, defaults to `None`
        If passed, add a column to the targets array popluated
        with HEALPixels at resolution `nside`.
    survey : :class:`str`, optional, defaults to "main"
        Written to output file header as the keyword `SURVEY`.
    nsidefile : :class:`int`, optional, defaults to `None`
        Passed to indicate in the output file header that the targets
        have been limited to only certain HEALPixels at a given
        nside. Used in conjunction with `hpxlist`.
    hpxlist : :class:`list`, optional, defaults to `None`
        Passed to indicate in the output file header that the targets
        have been limited to only this list of HEALPixels. Used in
        conjunction with `nsidefile`.
    resolve, maskbits : :class:`bool`, optional, default to ``True``
        Written to the output file header as `RESOLVE`, `MASKBITS`.
    scndout : :class:`str`, optional, defaults to `None`
        If passed, add to output header as SCNDOUT.
    obscon : :class:`str`, optional, defaults to `None`
        Can pass one of "DARK" or "BRIGHT". If passed, don't write the
        full set of data, rather only write targets appropriate for
        "DARK|GRAY" or "BRIGHT" observing conditions. The relevant
        `PRIORITY_INIT` and `NUMOBS_INIT` columns will be derived from
        `PRIORITY_INIT_DARK`, etc. and `filename` will have "bright" or
        "dark" appended to the lowest DIRECTORY in the input `filename`.
    mockdata : :class:`dict`, optional, defaults to `None`
        Dictionary of mock data to write out (only used in
        `desitarget.mock.build.targets_truth` via `select_mock_targets`).
    supp : :class:`bool`, optional, defaults to ``False``
        Written to the header of the output file to indicate whether
        this is a file of supplemental targets (targets that are
        outside the Legacy Surveys footprint).
    extra : :class:`dict`, optional
        If passed (and not None), write these extra dictionary keys and
        values to the output header.
    infiles : :class:`list` or `~numpy.ndarray`, optional
        If passed (and not None), write a second extension "INFILES" that
        contains the files in `infiles` and their SHA-256 checksums. If
        `infiles` is a list, func:`~desitarget.io.get_checksums()` is
        called to look-up the checksums, if `infiles` is a numpy array
        it's assumed to be in the format returned by `get_checksums()`.

    Returns
    -------
    :class:`int`
        The number of targets that were written to file.
    :class:`str`
        The name of the file to which targets were written.
    """
    # ADM create header.
    hdr = fitsio.FITSHDR()

    # ADM limit to just BRIGHT or DARK targets, if requested.
    # ADM Ignore the filename output, we'll build that on-the-fly.
    if obscon is not None:
        if mockdata is not None:
            _, hdr, data, mockdata = _bright_or_dark(
                targdir, hdr, data, obscon, mockdata=mockdata)
        else:
            _, hdr, data = _bright_or_dark(
                targdir, hdr, data, obscon)

    # ADM if passed, use the indir to determine the Data Release
    # ADM integer and string for the input targets.
    drint = None
    if supp and len(data) > 0:
        _, _, _, _, _, gaiadr = decode_targetid(data["TARGETID"])
        # ADM cmx targets can have the First Light targets, which
        # ADM have an invented Gaia DR.
        if survey != "cmx":
            if len(set(gaiadr)) != 1:
                msg = "Targets are based on multiple Gaia DRs:".format(set(gaiadr))
                log.critical(msg)
                raise ValueError(msg)
            gaiadr = gaiadr[0]
        else:
            gaiadr = np.max(gaiadr)
        drstring = "gaiadr{}".format(gaiadr)
    else:
        try:
            drint = int(indir.split("dr")[1][0])
            drstring = "dr{}".format(drint)
        except (ValueError, IndexError, AttributeError):
            drstring = "X"

    # ADM catch cases where we're writing-to-file and there's no hpxlist.
    hpx = hpxlist
    if hpxlist is None:
        hpx = "X"

    # ADM construct the output file name.
    if mockdata is not None:
        filename = find_target_files(targdir, flavor="targets", obscon=obscon,
                                     hp=hpx, nside=nside, mock=True)
        truthfile = find_target_files(targdir, flavor="truth", obscon=obscon,
                                      hp=hpx, nside=nside, mock=True)
    else:
        filename = find_target_files(targdir, dr=drstring, flavor="targets",
                                     survey=survey, obscon=obscon, hp=hpx,
                                     resolve=resolve, supp=supp)

    ntargs = len(data)
    # ADM die if there are no targets to write.
    if ntargs == 0:
        return ntargs, filename

    # ADM write versions, etc. to the header.
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())
    depend.setdep(hdr, 'photcat', drstring)

    if indir is not None:
        depend.setdep(hdr, 'tractor-files', indir)
    if indir2 is not None:
        depend.setdep(hdr, 'tractor-files-2', indir2)

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
        hdr.add_record(dict(name='HPXNSIDE', value=nside,
                            comment="HEALPix nside"))
        hdr.add_record(dict(name='HPXNEST', value=True,
                            comment="HEALPix nested (not ring) ordering"))

    # ADM populate SUBPRIORITY with a reproducible random float.
    if "SUBPRIORITY" in data.dtype.names and mockdata is None:
        np.random.seed(616)
        data["SUBPRIORITY"] = np.random.random(ntargs)

    # ADM add the type of survey (main, commissioning; or "cmx", sv) to the header.
    hdr["SURVEY"] = survey
    # ADM add whether or not the targets were resolved to the header.
    hdr["RESOLVE"] = resolve
    # ADM add whether or not MASKBITS was applied to the header.
    hdr["MASKBITS"] = maskbits
    # ADM indicate whether this is a supplemental file.
    hdr["SUPP"] = supp
    # ADM add the Data Release to the header.
    if supp:
        hdr["GAIADR"] = gaiadr
    else:
        hdr["DR"] = drint

    # ADM add the extra dictionary to the header.
    if extra is not None:
        for key in extra:
            hdr[key] = extra[key]

    if scndout is not None:
        hdr["SCNDOUT"] = scndout

    # ADM record whether this file has been limited to only certain HEALPixels.
    if hpxlist is not None or nsidefile is not None:
        # ADM hpxlist and nsidefile need to be passed together.
        check_both_set(hpxlist, nsidefile)
        hdr['FILENSID'] = nsidefile
        hdr['FILENEST'] = True
        # ADM warn if we've stored a pixel string that is too long.
        _check_hpx_length(hpxlist, warning=True)
        hdr['FILEHPX'] = hpxlist

    # ADM create necessary directories, if they don't exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # ADM write in a series of chunks to save memory.
    if nchunks is None:
        write_with_units(filename, data, extname='TARGETS', header=hdr)
    else:
        write_in_chunks(filename, data, nchunks, extname='TARGETS', header=hdr)

    # Optionally write out mock catalog data.
    if mockdata is not None:
        # truthfile = filename.replace('targets-', 'truth-')
        truthdata, trueflux, objtruth = mockdata['truth'], mockdata['trueflux'], mockdata['objtruth']

        hdr['SEED'] = (mockdata['seed'], 'initial random seed')
        # ADM TODO: the mock fitsio.writes could use write_with_units()?
        fitsio.write(truthfile+'.tmp', truthdata.as_array(), extname='TRUTH', header=hdr, clobber=True)

        if len(trueflux) > 0 and trueflux.shape[1] > 0:
            wavehdr = fitsio.FITSHDR()
            wavehdr['BUNIT'] = 'Angstrom'
            wavehdr['AIRORVAC'] = 'vac'
            fitsio.write(truthfile+'.tmp', mockdata['truewave'].astype(np.float32),
                         extname='WAVE', header=wavehdr, append=True)

            fluxhdr = fitsio.FITSHDR()
            fluxhdr['BUNIT'] = '1e-17 erg/s/cm2/Angstrom'
            fitsio.write(truthfile+'.tmp', trueflux.astype(np.float32),
                         extname='FLUX', header=fluxhdr, append=True)

        if len(objtruth) > 0:
            for obj in sorted(set(truthdata['TEMPLATETYPE'])):
                out = objtruth[obj]

                # TODO: fix desitarget #529, double check with #603, then remove this
                # Temporarily remove the `TRANSIENT_` columns--
                # see https://github.com/desihub/desitarget/issues/603#issuecomment-612678359 and
                # https://github.com/desihub/desisim/issues/529
                for col in out.colnames.copy():
                    if 'TRANSIENT_' in col:
                        out.remove_column(col)

                fitsio.write(truthfile+'.tmp', out.as_array(), append=True, extname='TRUTH_{}'.format(obj))
        os.rename(truthfile+'.tmp', truthfile)

    # ADM If input files were passed, write them to a second extension.
    if infiles is not None:
        if isinstance(infiles, list):
            shatab = get_checksums(infiles, verbose=True)
        elif isinstance(infiles, np.ndarray):
            shatab = infiles
        fitsio.write(filename, shatab, extname="INFILES")

    return ntargs, filename


def write_mtl(mtldir, data, indir=None, survey="main", obscon=None,
              nsidefile=None, hpxlist=None, extra=None, ecsv=True, mixed=False):
    """Write Merged Target List ledgers or files.

    Parameters
    ----------
    mtldir : :class:`str`
        Path to output MTL directory (the directory structure and file
        name are built on-the-fly from other inputs).
    data : :class:`~numpy.ndarray`
        numpy structured array of merged targets to write.
    indir : :class:`str`, optional, defaults to `None`
        If passed, note as the input directory in the output file header.
    survey : :class:`str`, optional, defaults to "main"
        Written to output file header as the keyword `SURVEY`.
    obscon : :class:`str`, optional
        Name of the `OBSCONDITIONS` used to make the file (e.g. DARK).
    nsidefile : :class:`int`, optional, defaults to `mtl.get_mtl_dir()`
        Passed to indicate in the output file header that the targets
        have been limited to only certain HEALPixels at a given
        nside. Used in conjunction with `hpxlist`.
    hpxlist : :class:`list`, optional, defaults to `None`
        Passed to indicate in the output file header that the targets
        have been limited to only this list of HEALPixels. Used in
        conjunction with `nsidefile`.
    extra : :class:`dict`, optional
        If passed (and not None), write these extra dictionary keys and
        values to the output header.
    ecsv : :class:`bool`, defaults to ``True``
        If ``True`` write a .ecsv file, if ``False`` with a .fits file.
    mixed : :class:`bool`, defaults to ``False``
        If ``True`` allow `data` to be from different Data Releases and
        write out the largest data release integer to the file headers.
        Useful when writing targets from, e.g., DR9 of the Legacy Surveys
        and DR2 of Gaia to the same file.

    Returns
    -------
    :class:`int`
        The number of targets that were written to file.
    :class:`str`
        The name of the file to which targets were written.
    """
    # ADM begin to construct a dictionary of header keys and values.
    keys, vals = ["INDIR", "SURVEY", "OBSCON"], [indir, survey, obscon]

    # ADM hpxlist and nsidefile need to be passed together.
    if hpxlist is not None or nsidefile is not None:
        check_both_set(hpxlist, nsidefile)
        # ADM warn if we've stored a pixel string that is too long.
        _check_hpx_length(hpxlist, warning=True)
        # ADM add to the header dictionary.
        keys += ["FILENSID", "FILENEST", "FILEHPX"]
        vals += [nsidefile, True, hpxlist]

    # ADM catch cases where hpxlist wasn't passed.
    hpx = hpxlist
    if hpxlist is None:
        hpx = "X"

    # ADM determine the data release from a TARGETID.
    _, _, release, _, _, _ = decode_targetid(data["TARGETID"])
    # ADM if the mixed kwarg was sent, allow multiple data releases.
    if mixed:
        dr = np.atleast_1d(np.max(release//1000))
    else:
        dr = np.unique(release//1000)
    if len(dr) == 0:
        drint = 'X'
    else:
        try:
            drint = int(dr)
        except TypeError:
            msg = "Multiple data releases in MTL ({})".format(dr)
            log.error(msg)
            raise TypeError(msg)
    keys += ["DR"]
    vals += [drint]

    # ADM finalize the header dictionary.
    hdrdict = {key: val for key, val in zip(keys, vals) if val is not None}
    if extra is not None:
        hdrdict = {**hdrdict, **extra}

    # ADM set output format to ecsv if passed, or fits otherwise.
    form = 'ecsv'*ecsv + 'fits'*(not(ecsv))
    fn = find_target_files(mtldir, dr=drint, flavor="mtl", survey=survey,
                           obscon=obscon, hp=hpx, ender=form)
    # ADM create necessary directories, if they don't exist.
    os.makedirs(os.path.dirname(fn), exist_ok=True)

    ntargs = len(data)
    # ADM die if there are no targets to write.
    if ntargs == 0:
        return ntargs, fn

    # ADM sort the output file on TARGETID.
    data = data[np.argsort(data["TARGETID"])]

    write_with_units(fn, data, extname='MTL', header=hdrdict, ecsv=ecsv)

    return ntargs, fn


def write_in_chunks(filename, data, nchunks, extname=None, header=None):
    """Write a FITS file in chunks to save memory.

    Parameters
    ----------
    filename : :class:`str`
        The output file.
    data : :class:`~numpy.ndarray`
        The numpy structured array of data to write.
    nchunks : :class`int`, optional, defaults to `None`
        The number of chunks in which to write the output file.
    extname, header, clobber, optional
        Passed through to fitsio.write().

    Returns
    -------
    Nothing, but writes the `data` to the `filename` in chunks.

    Notes
    -----
        - Always OVERWRITES existing files!
        - Mostly deprecated, so was never updated to write units.
    """
    # ADM ensure that files are always overwritten.
    if os.path.isfile(filename):
        os.remove(filename)
    start = time()
    # ADM open a file for writing.
    outy = fitsio.FITS(filename, 'rw')
    # ADM write the chunks one-by-one.
    chunk = len(data)//nchunks
    for i in range(nchunks):
        log.info("Writing chunk {}/{} from index {} to {}...t = {:.1f}s"
                 .format(i+1, nchunks, i*chunk, (i+1)*chunk-1, time()-start))
        datachunk = data[i*chunk:(i+1)*chunk]
        # ADM if this is the first chunk, write the data and header...
        if i == 0:
            outy.write(datachunk, extname='TARGETS', header=header, clobber=True)
        # ADM ...otherwise just append to the existing file object.
        else:
            outy[-1].append(datachunk)
    # ADM append any remaining data.
    datachunk = data[nchunks*chunk:]
    log.info("Writing final partial chunk from index {} to {}...t = {:.1f}s"
             .format(nchunks*chunk, len(data)-1, time()-start))
    outy[-1].append(datachunk)
    outy.close()
    return


def write_secondary(targdir, data, primhdr=None, scxdir=None, obscon=None,
                    drint='X'):
    """Write a catalogue of secondary targets.

    Parameters
    ----------
    targdir : :class:`str`
        Path to output target selection directory (the directory
        structure and file name are built on-the-fly from other inputs).
    data : :class:`~numpy.ndarray`
        numpy structured array of secondary targets to write.
    primhdr : :class:`str`, optional, defaults to `None`
        If passed, added to the header of the output `filename`.
    scxdir : :class:`str`, optional, defaults to :envvar:`SCND_DIR`
        Name of the directory that hosts secondary targets.  The
        secondary targets are written back out to this directory in the
        sub-directory "outdata" and the `scxdir` is added to the
        header of the output `filename`.
    obscon : :class:`str`, optional, defaults to `None`
        Can pass one of "DARK" or "BRIGHT". If passed, don't write the
        full set of secondary targets that do not match a primary,
        rather only write targets appropriate for "DARK|GRAY" or
        "BRIGHT" observing conditions. The relevant `PRIORITY_INIT`
        and `NUMOBS_INIT` columns will be derived from
        `PRIORITY_INIT_DARK`, etc. and `filename` will have "bright" or
        "dark" appended to the lowest DIRECTORY in the input `filename`.
    drint : :class:`int`, optional, defaults to `X`
        The data release ("dr"`drint`"-") in the output filename.

    Returns
    -------
    :class:`int`
        The number of secondary targets that do not match a primary
        target that were written to file.
    :class:`str`
        The name of the file to which such targets were written.

    Notes
    -----
    Two sets of files are written:
        - The file of secondary targets that do not match a primary
          target is written to `targdir`. Such secondary targets
          are determined from having "PRIM_MATCH"=``False`` in `data`.
          Only targets with `PRIORITY_INIT > -1` are written to this file
          (this allows duplicates to be resolved in, e.g.,
          :func:`~desitarget.secondary.finalize()`.
        - Each secondary target that, presumably, was initially drawn
          from the "indata" subdirectory of `scxdir` is written to
          the "outdata" subdirectory of `scxdir`.
    """
    # ADM grab the scxdir, it it wasn't passed.
    from desitarget.secondary import _get_scxdir
    scxdir = _get_scxdir(scxdir)

    # ADM if the primary header was passed, use it, if not
    # ADM then create a new header.
    hdr = primhdr
    if primhdr is None:
        hdr = fitsio.FITSHDR()
    # ADM add the SCNDDIR to the file header.
    hdr["SCNDDIR"] = scxdir

    # ADM limit to just BRIGHT or DARK targets, if requested.
    # ADM ignore the filename output, we'll build that on-the-fly.
    if obscon is not None:
        log.info("Observational conditions are {}".format(obscon))
        _, hdr, data = _bright_or_dark(targdir, hdr, data, obscon)
    else:
        log.info("Observational conditions are ALL")

    # ADM add the secondary dependencies to the file header.
    depend.setdep(hdr, 'scnd-desitarget', desitarget_version)
    depend.setdep(hdr, 'scnd-desitarget-git', gitversion())

    # ADM populate SUBPRIORITY with a reproducible random float.
    if "SUBPRIORITY" in data.dtype.names:
        ntargs = len(data)
        np.random.seed(616)
        data["SUBPRIORITY"] = np.random.random(ntargs)

    # ADM remove the SCND_TARGET_INIT, SCND_ORDER and PRIM_MATCH columns.
    scnd_target_init = data["SCND_TARGET_INIT"]
    scnd_order = data["SCND_ORDER"]
    prim_match = data["PRIM_MATCH"]

    data = rfn.drop_fields(data,
                           ["SCND_TARGET_INIT", "SCND_ORDER", "PRIM_MATCH"])
    # ADM we only need a subset of the columns where we match a primary.
    smalldata = rfn.drop_fields(data, ["PRIORITY_INIT", "SUBPRIORITY",
                                       "NUMOBS_INIT", "OBSCONDITIONS"])

    # ADM load the correct mask.
    _, mx, survey = main_cmx_or_sv(data, scnd=True)
    log.info("Loading mask for {} survey".format(survey))
    scnd_mask = mx[3]

    # ADM construct the output full and reduced file name.
    filename = find_target_files(targdir, dr=drint, flavor="targets", nohp=True,
                                 survey=survey, obscon=obscon, resolve=None)

    # ADM write out the file of matches for every secondary bit.
    scxoutdir = os.path.join(scxdir, 'outdata', desitarget_version)
    if obscon is not None:
        scxoutdir = os.path.join(scxoutdir, obscon.lower())
    else:
        scxoutdir = os.path.join(scxoutdir, "no-obscon")
    os.makedirs(scxoutdir, exist_ok=True)

    # ADM and write out the information for each bit.
    for name in scnd_mask.names():
        # ADM construct the output file name.
        fn = "{}.fits".format(scnd_mask[name].filename)
        scxfile = os.path.join(scxoutdir, fn)
        # ADM retrieve just the data with this bit set.
        ii = (scnd_target_init & scnd_mask[name]) != 0
        # ADM only proceed to the write stage if there are targets.
        if np.sum(ii) > 0:
            # ADM to reorder to match the original input order.
            order = np.argsort(scnd_order[ii])
            # ADM write to file.
            write_with_units(scxfile, smalldata[ii][order], extname='TARGETS',
                             header=hdr)
            log.info('Info for {} secondaries written to {}'
                     .format(np.sum(ii), scxfile))

    # ADM make necessary directories for the file, if they don't exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # ADM standalone secondaries have PRIORITY_INIT > -1 and
    # ADM don't have PRIM_MATCH set.
    ii = ~prim_match & (data["PRIORITY_INIT"] > -1)

    # ADM ...write them out.
    write_with_units(filename, data[ii], extname='SCND_TARGETS', header=hdr)

    return np.sum(ii), filename


def write_skies(targdir, data, indir=None, indir2=None, supp=False,
                apertures_arcsec=None, nskiespersqdeg=None, nside=None,
                nsidefile=None, hpxlist=None, extra=None, mock=False):
    """Write a target catalogue of sky locations.

    Parameters
    ----------
    targdir : :class:`str`
        Path to output target selection directory (the directory
        structure and file name are built on-the-fly from other inputs).
    data  : :class:`~numpy.ndarray`
        Array of skies to write to file.
    indir, indir2 : :class:`str`, optional
        Name of input Legacy Survey Data Release directory/directories,
        write to header of output file if passed (and if not None).
    supp : :class:`bool`, optional, defaults to ``False``
        Written to the header of the output file to indicate whether
        this is a file of supplemental skies (sky locations that are
        outside the Legacy Surveys footprint).
    apertures_arcsec : :class:`list` or `float`, optional
        list of aperture radii in arcseconds to write each aperture as an
        individual line in the header, if passed (and if not None).
    nskiespersqdeg : :class:`float`, optional
        Number of sky locations generated per sq. deg., write to header
        of output file if passed (and if not None).
    nside: :class:`int`, optional
        If passed, add a column to the skies array popluated with
        HEALPixels at resolution `nside`.
    nsidefile : :class:`int`, optional, defaults to `None`
        Passed to indicate in the output file header that the targets
        have been limited to only certain HEALPixels at a given
        nside. Used in conjunction with `hpxlist`.
    hpxlist : :class:`list`, optional, defaults to `None`
        Passed to indicate in the output file header that the targets
        have been limited to only this list of HEALPixels. Used in
        conjunction with `nsidefile`.
    extra : :class:`dict`, optional
        If passed (and not None), write these extra dictionary keys and
        values to the output header.
    mock : :class:`bool`, optional, defaults to ``False``.
        If ``True`` then construct the file path for mock sky target catalogs.

    Returns
    -------
    :class:`int`
        The number of skies that were written to file.
    :class:`str`
        The name of the file to which skies were written.
    """
    nskies = len(data)

    # ADM find the data release string for the input skies.
    drint = None
    if supp and len(data) > 0:
        _, _, _, _, _, gaiadr = decode_targetid(data["TARGETID"])
        if len(set(gaiadr)) != 1:
            msg = "Skies are based on multiple Gaia DRs:".format(set(gaiadr))
            log.critical(msg)
            raise ValueError(msg)
        gaiadr = gaiadr[0]
        drstring = "gaiadr{}".format(gaiadr)
    else:
        try:
            drint = np.max(data['RELEASE']//1000)
            drstring = 'dr'+str(drint)
        except (ValueError, IndexError, AttributeError):
            drstring = "X"

    # - Create header to include versions, etc.
    hdr = fitsio.FITSHDR()
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())
    depend.setdep(hdr, 'photcat', drstring)

    if indir is not None:
        depend.setdep(hdr, 'input-data-release', indir)
    if indir2 is not None:
        depend.setdep(hdr, 'input-data-release-2', indir2)

    if apertures_arcsec is not None:
        for i, ap in enumerate(apertures_arcsec):
            apname = "AP{}".format(i)
            apsize = ap
            hdr[apname] = apsize

    hdr['SUPP'] = supp
    if supp:
        hdr["GAIADR"] = gaiadr
    else:
        hdr["DR"] = drint

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
        # ADM ensure different SUBPRIORITIES for supp/standard files.
        if supp:
            np.random.seed(626)
        else:
            np.random.seed(616)
        data["SUBPRIORITY"] = np.random.random(nskies)

    # ADM add the extra dictionary to the header.
    if extra is not None:
        for key in extra:
            hdr[key] = extra[key]

    # ADM record whether this file has been limited to only certain HEALPixels.
    if hpxlist is not None or nsidefile is not None:
        # ADM hpxlist and nsidefile need to be passed together.
        check_both_set(hpxlist, nsidefile)
        hdr['FILENSID'] = nsidefile
        hdr['FILENEST'] = True
        # ADM warn if we've stored a pixel string that is too long.
        _check_hpx_length(hpxlist, warning=True)
        hdr['FILEHPX'] = hpxlist
    else:
        # ADM set the hp part of the output file name to "X".
        hpxlist = "X"

    # ADM construct the output file name.
    if mock:
        filename = find_target_files(targdir, flavor='sky', hp=hpxlist,
                                     mock=mock, nside=nside)
    else:
        filename = find_target_files(targdir, dr=drstring, flavor="skies",
                                     hp=hpxlist, supp=supp, mock=mock,
                                     nside=nside)

    # ADM create necessary directories, if they don't exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    write_with_units(filename, data, extname='SKY_TARGETS', header=hdr)

    return len(data), filename


def write_gfas(targdir, data, indir=None, indir2=None, nside=None,
               nsidefile=None, hpxlist=None, extra=None):
    """Write a catalogue of Guide/Focus/Alignment targets.

    Parameters
    ----------
    targdir : :class:`str`
        Path to output target selection directory (the directory
        structure and file name are built on-the-fly from other inputs).
    data  : :class:`~numpy.ndarray`
        Array of GFAs to write to file.
    indir, indir2 : :class:`str`, optional, defaults to None.
        Legacy Survey Data Release directory or directories, write to
        header of output file if passed (and if not None).
    nside: :class:`int`, defaults to None.
        If passed, add a column to the GFAs array popluated with
        HEALPixels at resolution `nside`.
    nsidefile : :class:`int`, optional, defaults to `None`
        Passed to indicate in the output file header that the targets
        have been limited to only certain HEALPixels at a given
        nside. Used in conjunction with `hpxlist`.
    hpxlist : :class:`list`, optional, defaults to `None`
        Passed to indicate in the output file header that the targets
        have been limited to only this list of HEALPixels. Used in
        conjunction with `nsidefile`.
    extra : :class:`dict`, optional
        If passed (and not None), write these extra dictionary keys and
        values to the output header.

    Returns
    -------
    :class:`int`
        The number of gfas that were written to file.
    :class:`str`
        The name of the file to which gfas were written.
    """
    # ADM if passed, use the indir to determine the Data Release
    # ADM integer and string for the input targets.
    try:
        drint = int(indir.split("dr")[1][0])
        drstring = 'dr'+str(drint)
    except (ValueError, IndexError, AttributeError):
        drint = None
        drstring = "X"

    # ADM rename 'TYPE' to 'MORPHTYPE'.
    data = rfn.rename_fields(data, {'TYPE': 'MORPHTYPE'})

    # ADM create header to include versions, etc.
    hdr = fitsio.FITSHDR()
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())
    hdr["DR"] = drint

    if indir is not None:
        depend.setdep(hdr, 'input-data-release', indir)
        # ADM note that if 'dr' is not in the indir DR
        # ADM directory structure, garbage will
        # ADM be rewritten gracefully in the header.
        drstring = 'dr'+indir.split('dr')[-1][0]
        depend.setdep(hdr, 'photcat', drstring)
    if indir2 is not None:
        depend.setdep(hdr, 'input-data-release-2', indir2)

    # ADM add the extra dictionary to the header.
    if extra is not None:
        for key in extra:
            hdr[key] = extra[key]

    # ADM add HEALPix column, if requested by input.
    if nside is not None:
        theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
        hppix = hp.ang2pix(nside, theta, phi, nest=True)
        data = rfn.append_fields(data, 'HPXPIXEL', hppix, usemask=False)
        hdr['HPXNSIDE'] = nside
        hdr['HPXNEST'] = True

    # ADM record whether this file has been limited to only certain HEALPixels.
    if hpxlist is not None or nsidefile is not None:
        # ADM hpxlist and nsidefile need to be passed together.
        check_both_set(hpxlist, nsidefile)
        hdr['FILENSID'] = nsidefile
        hdr['FILENEST'] = True
        # ADM warn if we've stored a pixel string that is too long.
        _check_hpx_length(hpxlist, warning=True)
        hdr['FILEHPX'] = hpxlist
    else:
        # ADM set the hp part of the output file name to "X".
        hpxlist = "X"

    # ADM construct the output file name.
    filename = find_target_files(targdir, dr=drint, flavor="gfas", hp=hpxlist)

    # ADM create necessary directories, if they don't exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    write_with_units(filename, data, extname='GFA_TARGETS', header=hdr)

    return len(data), filename


def write_randoms(targdir, data, indir=None, hdr=None, nside=None, supp=False,
                  nsidefile=None, hpxlist=None, resolve=True, north=None,
                  extra=None):
    """Write a catalogue of randoms and associated pixel-level info.

    Parameters
    ----------
    targdir : :class:`str`
        Path to output target selection directory (the directory
        structure and file name are built on-the-fly from other inputs).
    data  : :class:`~numpy.ndarray`
        Array of randoms to write to file.
    indir : :class:`str`, optional, defaults to None
        Name of input Legacy Survey Data Release directory, write to
        header of output file if passed (and if not ``None``).
    hdr : :class:`str`, optional, defaults to ``None``
        If passed, use this header to start the header for `filename`.
    nside: :class:`int`
        If passed, add a column to the randoms array popluated with
        HEALPixels at resolution `nside`.
    supp : :class:`bool`, optional, defaults to ``False``
        Written to the header of the output file to indicate whether
        this is a supplemental file (i.e. random locations that are
        outside the Legacy Surveys footprint).
    nsidefile : :class:`int`, optional, defaults to ``None``
        Passed to indicate in the output file header that the targets
        have been limited to only certain HEALPixels at a given
        nside. Used in conjunction with `hpxlist`.
    hpxlist : :class:`list`, optional, defaults to ``None``
        Passed to indicate in the output file header that the targets
        have been limited to only this list of HEALPixels. Used in
        conjunction with `nsidefile`.
    resolve : :class:`bool`, optional, defaults to ``True``
        Written to the output file header as `RESOLVE`. If ``True``
        (``False``) output directory includes "resolve" ("noresolve").
    north : :class:`bool`, optional
        If passed (and not ``None``), then, if ``True`` (``False``),
        REGION=north (south) is written to the output header and the
        output directory name is appended by "north" ("south").
    extra : :class:`dict`, optional
        If passed (and not ``None``), write these extra dictionary keys
        and values to the output header.

    Returns
    -------
    :class:`int`
        The number of randoms that were written to file.
    :class:`str`
        The name of the file to which randoms were written.
    """
    # ADM create header to include versions, etc. If a `hdr` was
    # ADM passed, then use it, if not then create a new header.
    if hdr is None:
        hdr = fitsio.FITSHDR()
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())

    if indir is not None:
        if supp:
            depend.setdep(hdr, 'input-random-catalog', indir)
        else:
            depend.setdep(hdr, 'input-data-release', indir)
        # ADM use input directory to (try to) determine the Data Release.
        try:
            drint = int(indir.split("dr")[1][0])
            drstring = 'dr'+str(drint)
            depend.setdep(hdr, 'photcat', drstring)
        except (ValueError, IndexError, AttributeError):
            drint = None

    # ADM whether this is a north-specific or south-specific file.
    region = None
    if north is not None:
        region = ["south", "north"][north]
        hdr["REGION"] = region

    # ADM record whether this file has been limited to only certain HEALPixels.
    nohp = False
    if hpxlist is not None or nsidefile is not None:
        # ADM hpxlist and nsidefile need to be passed together.
        check_both_set(hpxlist, nsidefile)
        hdr['FILENSID'] = nsidefile
        hdr['FILENEST'] = True
        # ADM warn if we've stored a pixel string that is too long.
        _check_hpx_length(hpxlist, warning=True)
        hdr['FILEHPX'] = hpxlist
    else:
        # ADM set the hp part of the output file name to "X".
        hpxlist = "X"
        # ADM if these are supplemental radons, ignore HEALPIxels.
        if supp:
            hpxlist = None
            nohp = True

    # ADM add the extra keywords to the header.
    hdr["DR"] = drint
    if extra is not None:
        for key in extra:
            hdr[key] = extra[key]

    # ADM retrieve the seed or seeds, if known.
    seed = None
    if extra is not None:
        for seedy in "seed", "SEED":
            if seedy in extra:
                seed = extra[seedy]
        # ADM we may have two seeds if the file is supplemental: The
        # ADM original seed that was used to run the randoms and a second
        # ADM seed that was used to supplement those randoms.
        if supp:
            for seedy in "origseed", "ORIGSEED":
                if seedy in extra:
                    seed = "{}-{}".format(extra[seedy], seed)

    # ADM construct the output file name.
    filename = find_target_files(targdir, dr=drint, flavor="randoms",
                                 hp=hpxlist, resolve=resolve, supp=supp,
                                 region=region, seed=seed, nohp=True)

    nrands = len(data)
    # ADM die if there are no targets to write.
    if nrands == 0:
        return nrands, filename

    # ADM add HEALPix column, if requested by input.
    if nside is not None:
        theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
        hppix = hp.ang2pix(nside, theta, phi, nest=True)
        data = rfn.append_fields(data, 'HPXPIXEL', hppix, usemask=False)
        hdr['HPXNSIDE'] = nside
        hdr['HPXNEST'] = True

    # ADM note if this is a supplemental (outside-of-footprint) file.
    hdr['SUPP'] = supp

    # ADM add whether or not the randoms were resolved to the header.
    hdr["RESOLVE"] = resolve

    # ADM create necessary directories, if they don't exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    write_with_units(filename, data, extname='RANDOMS', header=hdr)

    return nrands, filename


def write_masks(maskdir, data,
                maglim=None, maskepoch=None, nside=None, extra=None):
    """Write a catalogue of masks and associated pixel-level info.

    Parameters
    ----------
    maskdir : :class:`str`
        Path to output mask directory (the file names are built
        on-the-fly from other inputs).
    data  : :class:`~numpy.ndarray`
        Array of masks to write to file. Must contain at least the
        columns "RA" and "DEC".
    maglim : :class:`float`, optional, defaults to ``None``
        Magnitude limit to which the mask was made.
    maskepoch : :class:`float`, optional, defaults to ``None``
        Epoch at which the mask was made.
    nside: :class:`int`, defaults to not splitting by HEALPixel.
        The HEALPix nside at which to write the output files.
    extra : :class:`dict`, optional
        If passed (and not ``None``), write these extra dictionary keys
        and values to the output header.

    Returns
    -------
    :class:`int`
        The total number of masks that were written.
    :class:`str`
        The name of the directory to which masks were written.
    """
    # ADM create header to include versions, etc.
    hdr = fitsio.FITSHDR()
    depend.setdep(hdr, 'desitarget', desitarget_version)
    depend.setdep(hdr, 'desitarget-git', gitversion())
    # ADM add the magnitude and epoch to the header.
    if maglim is not None:
        hdr["MAGLIM"] = maglim
    if maskepoch is not None:
        hdr["MASKEPOC"] = maskepoch
    # ADM add the extra dictionary to the header.
    if extra is not None:
        for key in extra:
            hdr[key] = extra[key]
    # ADM add the HEALPixel information to the header.
    hdr["FILENSID"] = nside
    hdr["FILENEST"] = True

    nmasks = len(data)
    # ADM die if there are no masks to write.
    if nmasks == 0:
        return nmasks, None

    # ADM write across HEAPixels at input nside.
    if nside is not None:
        npix = hp.nside2npix(nside)
        theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
        hpx = hp.ang2pix(nside, theta, phi, nest=True)
        for pix in range(npix):
            outdata = data[hpx == pix]
            outhdr = dict(hdr).copy()
            outhdr["FILEHPX"] = pix
            # ADM construct the output file name.
            fn = find_target_files(maskdir, flavor="masks",
                                   hp=pix, maglim=maglim, epoch=maskepoch)
            # ADM create necessary directory, if it doesn't exist.
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            # ADM write the output file.
            if len(outdata) > 0:
                write_with_units(fn, outdata, extname='MASKS', header=outhdr)
                log.info('wrote {} masks to {}'.format(len(outdata), fn))
    else:
        fn = find_target_files(maskdir, flavor="masks", hp="X",
                               maglim=maglim, epoch=maskepoch)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        if len(data) > 0:
            write_with_units(fn, data, extname='MASKS', header=hdr)
            log.info('wrote {} masks to {}'.format(len(data), fn))

    return nmasks, os.path.dirname(fn)


def is_sky_dir_official(skydirname):
    """Check a sky file or directory has the correct HEALPixel structure.

    Parameters
    ----------
    skydirname : :class:`str`
        Full path to either a directory containing skies that have been
        partitioned by HEALPixel (i.e. as made by `select_skies` with the
        `bundle_files` option). Or the name of a single file of skies.

    Returns
    -------
    :class:`bool`
        ``True`` if the passed sky file or (the first sky file in the
        passed directory) is structured so that the list of healpixels in
        the file header ("FILEHPX") at the file nside ("FILENSID") in the
        file nested (or ring) scheme ("FILENEST") is a true reflection of
        the HEALPixels in the file.

    Notes
    -----
        - A necessary check because although the targets and GFAs are
          parallelized to run in the exact boundaries of HEALPixels, the
          skies are parallelized across bricks that have CENTERS in a
          given HEALPixel.
        - If this function returns ``False`` the remedy is typically to
          run `bin/repartition_skies`
        - If a directory is passed, this isn't an exhaustive check as
          only the first file is tested. That's enough for just checking
          the output of `select_skies`, though.
    """
    # ADM if skydirname is a directory, just work with one file.
    if os.path.isdir(skydirname):
        gen = iglob(os.path.join(skydirname, '*fits'))
        skydirname = next(gen)

    # ADM read the locations from the file and grab the header.
    data, hdr = read_target_files(skydirname, columns=["RA", "DEC"],
                                  header=True, verbose=False)

    # ADM determine which HEALPixels are in the file.
    theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
    pixinfile = hp.ang2pix(hdr["FILENSID"], theta, phi, nest=hdr["FILENEST"])

    # ADM determine which HEALPixels are in the header.
    hdrpix = hdr["FILEHPX"]
    if isinstance(hdrpix, int):
        hdrpix = [hdrpix]

    return set(pixinfile) == set(hdrpix)


def iter_files(root, prefix, ext='fits', ignore=None):
    """Iterator over files under in `root` directory with given `prefix` and
    extension. `ignore` is a list of strings that will be skipped in the
    directory or file names (for both a speed-up and trimming of files).
    """
    ignorable = False
    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            if ignore is not None:
                for ig in ignore:
                    if ig in dirpath:
                        del dirnames[:]
            for filename in filenames:
                if ignore is not None:
                    ignorable = np.any([ig in filename for ig in ignore])
                if filename.startswith(prefix) and filename.endswith('.'+ext):
                    if not ignorable:
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
    ignoredirs = ['metric', 'coadd', 'log', 'pz', 'external', 'tractor']
    ignorefiles = ['ex.fits', 'lc.fits']
    ignore = ignoredirs + ignorefiles
    return iter_files(root, prefix='sweep', ext='fits', ignore=ignore)


def list_targetfiles(root):
    """Return a list of target files found under `root` directory.
    """
    # ADM catch case where a file was sent instead of a directory.
    if os.path.isfile(root):
        return [root]
    allfns = glob(os.path.join(root, '*target*fits'))
    fns, nfns = np.unique(allfns, return_counts=True)
    if np.any(nfns > 1):
        badfns = fns[nfns > 1]
        msg = "Duplicate target files ({}) beneath root directory {}:".format(
            badfns, root)
        log.error(msg)
        raise SyntaxError(msg)

    return allfns


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
    ignore = ['metric', 'coadd', 'log', 'pz', 'external', 'sweep']
    return iter_files(root, prefix='tractor', ext='fits', ignore=ignore)


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
    match = re.search(r"tractor-(\d{4,5}[pm]\d{3,4})\.fits",
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
    match = re.search(r"%s_(\d{4,5}[pm]\d{3,4})\.fits" % (prefix),
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


def get_sha256sum(infile):
    """Get the sha256 checksum for a single file

    Parameters
    ----------
    infile : :class:`str`
        The full path name to a file.

    Returns
    -------
    :class:`str`
        The sha256 checksum for the passed `infile`.

    Notes
    -----
        - h/t https://stackoverflow.com/questions/61229719
    """
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(infile, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def get_checksums(infiles, verbose=False, check_existing=True):
    """Get the sha256 checksums for a list of files.

    Parameters
    ----------
    infiles : :class:`list` or `str`
        The full paths to a file or files.
    verbose : :class:`bool`, optional, defaults to ``False``
        If ``True`` then log progress and times.
    check_existing : :class:`bool`, optional, defaults to ``True``
        If ``True`` check if any of the `infiles` is in a directory in
        which a .sha256sum file exists, and, if so, check generated
        checksums for each file against the corresponding entry in the
        relevant .sha256sum file (or *files* for `infiles` that span
        multiple directories). An exception is raised for a mismatch.

    Returns
    -------
    :class:`~numpy.ndarray`
        A recarray with two columns "FILENAME" and "SHA256".
    """
    t0 = time()
    # ADM in case a single string was passed.
    infiles = np.atleast_1d(infiles)

    # ADM we'll first populate a dictionary with the checksums.
    shadict = {}
    nf = len(infiles)
    # ADM if verbose is True, write out info for 20 blocks of files.
    block = nf // 20 if nf // 20 else 1
    for ifn, infile in enumerate(infiles):
        shafn = [get_sha256sum(infile), infile]
        # ADM add the filename, sha combination to the dict.
        shadict[shafn[1]] = shafn[0]
        if verbose and ifn % block == 0:
            log.info("Calculated checksum for {}/{} files...t={:.1f}s".format(
                ifn+1, nf, time()-t0))

    # ADM the string data types for the output array should have a
    # ADM length that corresponds to the longest filename/shasum.
    fntype = "U{}".format(np.max([len(k) for k in shadict.keys()]))
    shatype = "U{}".format(np.max([len(k) for k in shadict.values()]))

    # ADM construct the output array.
    shatab = np.zeros(nf, dtype=[('FILENAME', fntype), ('SHA256', shatype)])
    shatab['FILENAME'] = list(shadict.keys())
    shatab['SHA256'] = list(shadict.values())

    # ADM grab the unique directories that host files.
    ldir = set([os.path.dirname(fn) for fn in infiles])
    # ADM loop through each directory and build a dictionary of the
    # ADM expected files and their associated SHA checksums.
    checkdict = {}
    for ld in ldir:
        # ADM look for a shasum file.
        shalist = glob(os.path.join(ld, "*.sha256sum"))
        if len(shalist) > 0 and check_existing:
            shafn = shalist[0]
            if verbose:
                log.info("Comparing checksums to {}".format(shafn))
            # ADM read the checksum file and construct a dictionary
            # ADM of file paths and checksums.
            with open(shafn) as f:
                for line in f:
                    sha256, filename = line.split()
                    fullpath = os.path.join(ld, filename)
                    checkdict[fullpath] = sha256

    # ADM check the existing checksum file against the
    # ADM calculated checksums, if any SHA checksum files existed.
    if len(checkdict) > 0 and check_existing:
        for st in shatab:
            try:
                if checkdict[st["FILENAME"]] != st["SHA256"]:
                    msg = "Checksum issue: {} differs in checksum file {}"  \
                          .format(st, shafn)
                    log.critical(msg)
                    raise IOError(msg)
            except KeyError:
                msg = "Filename {} isn't in checksum file {}".format(
                    st["FILENAME"], shafn)
                log.critical(msg)
                raise IOError(msg)

    return shatab


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
        hdr = read_targets_header(fn)
        nside.append(hdr["FILENSID"])
        pixels = hdr["FILEHPX"]
        # ADM hdr["FILEHPX"] could be a str, depending on fitsio version.
        if isinstance(pixels, str):
            pixels = list(map(int, pixels.split(',')))
        # ADM if this is a one-pixel file, or interpreted as a tuple,
        # ADM convert to a list.
        else:
            pixels = list(np.atleast_1d(pixels))
        # ADM check we haven't stored a pixel string that is too long.
        _check_hpx_length(pixels)
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


def _get_targ_dir():
    """Convenience function to grab the TARGDIR environment variable.

    Returns
    -------
    :class:`str`
        The directory stored in the $GAIA_DIR environment variable.
    """
    # ADM check that the $GAIA_DIR environment variable is set.
    targdir = os.environ.get('TARG_DIR')
    if targdir is None:
        msg = "Set $TARG_DIR environment variable!"
        log.critical(msg)
        raise ValueError(msg)

    return targdir


def find_target_files(targdir, dr='X', flavor="targets", survey="main",
                      obscon=None, hp=None, nside=None, resolve=True, supp=False,
                      mock=False, nohp=False, seed=None, region=None, epoch=None,
                      maglim=None, ender="fits"):
    """Build the name of an output target file (or directory).

    Parameters
    ----------
    targdir : :class:`str`
        Name of a based directory for output target catalogs.
    dr : :class:`str` or :class:`int`, optional, defaults to "X"
        Name of a Legacy Surveys Data Release (e.g. 8). If this is an
        integer or a 1-character string it is prepended by "dr".
    flavor : :class:`str`, optional, defaults to `targets`
        Options: "skies", "gfas", "targets", "randoms", "masks", "mtl".
    survey : :class:`str`, optional, defaults to `main`
        Options include "main", "cmx", "svX" (where X is 1, 2 etc.).
        Only relevant if `flavor` is "targets".
    obscon : :class:`str`, optional
        Name of the `OBSCONDITIONS` used to make the file (e.g. DARK).
    hp : :class:`list` or :class:`int` or :class:`str`, optional
        HEALPixel numbers used to make the file (e.g. 42 or [12, 37]
        or "42" or "12,37"). Required if mock=`True`.
    nside : :class:`int`, optional unless mock=`True`
        Nside corresponding to healpixel `hp`.
    resolve : :class:`bool`, optional, defaults to ``True``
        If ``True`` then find the `resolve` file. Otherwise find the
        `noresolve` file. Relevant if `flavor` is `targets` or `randoms`.
        Pass ``None`` to substitute `resolve` with "secondary".
    supp : :class:`bool`, optional, defaults to ``False``
        If ``True`` then find the supplemental targets file. Overrides
        the `obscon` option.
    mock : :class:`bool`, optional, defaults to ``False``
        If ``True`` then construct the file path for mock target
        catalogs and return (most other inputs are ignored).
    nohp : :class:`bool`, optional, defaults to ``False``
        If ``True``, override the normal behavior for `hp`=``None`` and
        instead construct a filename that omits the `-hpX-` part.
    seed : :class:`int` or `str`, optional
        If `seed` is not ``None``, then it is added to the file name just
        before the ".fits" extension (i.e. "-8.fits" for `seed` of 8).
        Only relevant if `flavor` is "randoms".
    region : :class:`int`, optional
        If `region` is not ``None``, then it is added to the directory
        name after `resolve`. Only relevant if `flavor` is "randoms".
    epoch : :class:`float`
        Epoch at which the mask was made. Only relevant if `flavor` is
        "masks". Must be passed if `flavor` is "masks".
    maglim : :class:`float`, optional
        Magnitude limit to which the mask was made. Only relevant if
        `flavor` is "masks". Must be passed if `flavor` is "masks".
    ender : :class:`str`, optional, defaults to "fits"
        File format (in file name).

    Returns
    -------
    :class:`str`
        The name of the output target file (or directory).

    Notes
    -----
        - If `hp` is passed, the full file name is returned. If `hp`
          is ``None``, the directory name where all of the `hp` files
          are stored is returned. The directory name is the expected
          input for the `desitarget.io.read*` convenience functions
          (:func:`desitarget.io.read_targets_in_hp()`, etc.).
        - On the other hand, if `hp` is ``None`` and `nohp` is ``True``
          then a filename is returned that just omits the `-hp-X-` part.
    """
    # ADM some preliminaries for correct formatting.
    version = desitarget_version
    if obscon is not None:
        obscon = obscon.lower()
    if survey not in ["main", "cmx"] and survey[:2] != "sv":
        msg = "survey must be main, cmx or svX, not {}".format(survey)
        log.critical(msg)
        raise ValueError(msg)
    if mock:
        allowed = ["targets", "truth", "sky"]
    else:
        allowed = ["targets", "skies", "gfas", "randoms", "masks", "mtl"]
    if flavor not in allowed:
        msg = "flavor must be {}, not {}".format(' or '.join(allowed), flavor)
        log.critical(msg)
        raise ValueError(msg)
    res = "noresolve"
    if resolve is None:
        res = "secondary"
    else:
        if resolve:
            res = "resolve"
    resdir = ""
    if flavor in ["targets", "randoms"]:
        resdir = res
    if isinstance(dr, int) or len(dr) == 1:
        drstr = "dr{}".format(dr)
    else:
        drstr = str(dr)

    # If seeking a mock target (or sky) catalog, construct the filepath and then
    # bail.
    if mock:
        if hp is None and nside is None:
            path = targdir
            if obscon is not None:
                fn = '{flavor}-{obscon}.fits'.format(flavor=flavor.lower(), obscon=obscon)
            else:
                fn = '{flavor}.fits'.format(flavor=flavor.lower())
        else:
            if (hp is None and nside is not None) or (hp is not None and nside is None):
                msg = 'Must specify nside and hp to locate the mock target catalogs!'
                log.critical(msg)
                raise ValueError(msg)
            subdir = str(hp // 100)
            path = os.path.abspath(os.path.join(targdir, subdir, str(hp)))
            if obscon is not None:
                path = os.path.join(path, obscon)
                fn = '{flavor}-{obscon}-{nside}-{hp}.fits'.format(
                    flavor=flavor.lower(), obscon=obscon, nside=nside, hp=hp)
            else:
                fn = '{flavor}-{nside}-{hp}.fits'.format(
                    flavor=flavor.lower(), nside=nside, hp=hp)
        return os.path.join(path, fn)

    # ADM build up the name of the file (or directory).
    surv = survey
    if survey[0:2] == "sv":
        surv = survey[0:2]
    if obscon is None:
        obscon = "no-obscon"
    if supp:
        obscon = "supp"
    prefix = flavor

    # ADM the generic directory structure beneath $TARG_DIR or $MTL_DIR.
    fn = os.path.join(targdir, drstr, version, flavor)

    # ADM masks are a special case beneath $MASK_DIR.
    if flavor == "masks":
        maskdir = "maglim-{}-epoch-{}".format(maglim, epoch)
        fn = os.path.join(targdir, version, maskdir)

    # ADM now a case-by-case basis.
    if flavor in ["targets", "mtl"]:
        fn = os.path.join(fn, survey, resdir, obscon)
        prefix = "{}-{}".format(flavor, obscon)
        if not resolve and flavor != "mtl":
            prefix = "{}-{}".format(prefix, res)
        if survey != "main":
            prefix = "{}{}".format(survey, prefix)

    if flavor == "randoms":
        fn = os.path.join(fn, resdir)
        if region is not None:
            fn = os.path.join(fn, region)
        if not resolve:
            prefix = "{}-{}".format(prefix, res)

    if flavor == "skies" and supp:
        fn = "{}-supp".format(fn)
        prefix = "{}-supp".format(prefix)

    # ADM if a HEALPixel number was passed, we want the filename.
    if hp is not None:
        hpstr = ",".join([str(pix) for pix in np.atleast_1d(hp)])
        backend = "{}-hp-{}.{}".format(prefix, hpstr, ender)
        if flavor == "masks":
            backend = "{}-hp-{}.{}".format(prefix, hpstr, ender)
        fn = os.path.join(fn, backend)
    else:
        if nohp:
            backend = "{}.{}".format(prefix, ender)
            fn = os.path.join(fn, backend)

    if flavor == "randoms":
        # ADM note that these clauses won't do anything
        # ADM unless a file name was already constructed.
        if seed is not None:
            fn = fn.replace(".{}".format(ender), "-{}.{}".format(seed, ender))
        if supp:
            fn = fn.replace("randoms", "randoms-outside")

    return fn


def read_mtl_ledger(filename, unique=True):
    """Wrapper to read individual MTL ledger files.

    Parameters
    ----------
    filename : :class:`str`
        Name of a ledger file containing a Merged Target List. If the
        filename contains ".ecsv" then it will be read as an ECSV file.
        If it contains ".fits" then it will be read as a FITS file.
    unique : :class:`bool`, optional, defaults to ``True``
        If ``True`` then only read targets with unique `TARGETID`, where
        the last occurrence of the target in the ledger is the one that
        is retained. If ``False`` then read the entire ledger.

    Returns
    -------
    :class:`~numpy.ndarray`
        A structured numpy array of the MTL.
    """
    if ".ecsv" in filename:
        # ADM infer the column names and types (for the dtype).
        # ADM (this snippet is much quicker than a Table read).
        from desitarget.mtl import mtldatamodel as mtldm
        names, forms = [], []
        with open(filename) as f:
            for line in f:
                if "name" in line:
                    l = line.split()
                    name, form = l[3][:-1], l[5][:-1]
                    names.append(name)
                    if 'string' in form:
                        forms.append(mtldm[name].dtype.str)
                    else:
                        forms.append(form)
                elif '#' not in line:
                    break
        dt = list(zip(names, forms))
        # ADM pandas seems the quickest way to read .csv-like files.
        # ADM it's ~1.5x faster than the basic Table read, but isn't
        # ADM generally supported in the desihub codebase.
#        prelim = pd.read_csv(filename, dtype=dt, comment="#", delimiter=" ")
        # ADM faster for astropy 4; although pandas is still faster.
#        prelim = Table.read(filename, comment="#", delimiter=" ", format='pandas.csv', dtype=dt)
        prelim = Table.read(filename, comment='#', format='ascii.basic',
                            guess=False)
        mtl = np.zeros(len(prelim), dtype=dt)
        for col in prelim.columns:
            mtl[col] = prelim[col]
    elif ".fits" in filename:
        mtl = fitsio.read(filename, extension="MTL")
    else:
        msg = "File not parsed ({}). Should be .fits or .ecsv".format(filename)
        log.error(msg)
        raise IOError(msg)

    if unique:
        # ADM the reverse is because np.unique retains the FIRST unique
        # ADM entry and we want the LAST unique entry.
        mtl = np.flip(mtl)
        _, ii = np.unique(mtl["TARGETID"], return_index=True)
        return mtl[ii]
    else:
        return mtl


def read_target_files(filename, columns=None, rows=None, header=False,
                      downsample=None, verbose=False):
    """Wrapper to cycle through allowed extensions to read target files.

    Parameters
    ----------
    filename : :class:`str`
        Name of a target file of any type. Target file types include
        "TARGETS", "GFA_TARGETS" and "SKY_TARGETS".
    columns : :class:`list`, optional
        Only read in these target columns.
    rows : :class:`list`, optional
        Only read in these rows from the target file.
    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return the header of the file.
    downsample : :class:`int`, optional, defaults to `None`
        If not `None`, downsample the file by this integer value, e.g.
        for `downsample=10` a file with 900 rows would have 90 random
        rows read in. Overrode by the `rows` kwarg if it is not `None`.
    verbose : :class:`bool`, optional, defaults to ``False``
        If ``True`` then log the file and extension that was read.
    """
    start = time()
    # ADM start with some checking that this is a target file.
    targtypes = ["TARGETS", "GFA_TARGETS", "SKY_TARGETS",
                 "MASKS", "MTL", "SCND_TARGETS"]
    # ADM read in the FITS extension info.
    f = fitsio.FITS(filename)
    if len(f) != 2:
        # ADM target files have an extra extension.
        if not f[1].get_extname() == 'TARGETS' and len(f) == 3:
            log.info(f)
            msg = "targeting files should only have 2 extensions?!"
            log.error(msg)
            raise IOError(msg)
    # ADM check for allowed extensions.
    extname = f[1].get_extname()
    if extname not in targtypes:
        log.info(f)
        msg = "unrecognized target file type: {}".format(extname)
        log.error(msg)
        raise IOError(msg)

    if downsample is not None and rows is None:
        np.random.seed(616)
        ntargs = fitsio.read_header(filename, extname)["NAXIS2"]
        rows = np.random.choice(ntargs, ntargs//downsample, replace=False)

    targs, hdr = fitsio.read(filename, extname,
                             columns=columns, rows=rows, header=True)

    if verbose:
        log.info("Read {} targets from {}, extension {}...Took {:.1f}s".format(
            len(targs), os.path.basename(filename), extname, time()-start))

    if header:
        return targs, hdr

    return targs


def read_keyword_from_mtl_header(hpdirname, keyword):
    """Read in a header value from a Merget Target List ledger file.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to either a directory containing MTLs that have been
        partitioned by HEALPixel (i.e. as made by
        :func:`desitarget.mtl.make_ledger_in_hp`). Or the name of a
        single MTL ledger.
    keyword : :class:`str`
        A single header keyword.

    Returns
    -------
    :class:`str`
        The value of `keyword` from the header of `hpdirname` if it is a
        file, or the value from the first file encountered in `hpdirname`
    """
    # ADM for FITS files, our standard targets header-read will work.
    try:
        kw = read_targets_header(hpdirname, verbose=False)[keyword]
        if isinstance(kw, str):
            kw = kw.rstrip()
        return kw
    except OSError:
        if os.path.isdir(hpdirname):
            try:
                gen = iglob(os.path.join(hpdirname, '*ecsv'))
                hpdirname = next(gen)
            except StopIteration:
                msg = "no FITS or ECSV files in {}...?!".format(hpdirname)
                log.info(msg)

    # ADM this (rapidly) reads a single keyword from an ecsv file.
    with open(hpdirname) as f:
        for line in f:
            if keyword in line and 'name' not in line:
                break
        return line.split(": ")[-1].split("}")[0]


def find_mtl_file_format_from_header(hpdirname, returnoc=False):
    """Construct an MTL filename just from the header in the file

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to either a directory containing targets that
        have been partitioned by HEALPixel (i.e. as made by
        `select_targets` with the `bundle_files` option). Or the
        name of a single file of targets.
    returnoc : :class:`bool`, optional, defaults to ``False``
        If ``True`` then also return the OBSCON header keyword
        for files in this directory.

    Returns
    -------
    :class:`str`
        The file form such that output.format(pixel) returns the
        full HEALPixel-dependent filename for a give pixel.
    :class:`str`
        The OBSCON header keyword. Only returned if `returnoc` is
        ``True``.

    Notes
    -----
        - Should work for both .ecsv and .fits files.
    """
    # ADM grab information from the target directory.
    dr = read_keyword_from_mtl_header(hpdirname, "DR")
    surv = read_keyword_from_mtl_header(hpdirname, "SURVEY")
    oc = read_keyword_from_mtl_header(hpdirname, "OBSCON")
    from desitarget.mtl import get_mtl_ledger_format
    ender = get_mtl_ledger_format()

    # ADM construct the full directory path.
    hugefn = find_target_files(hpdirname, flavor="mtl", hp="{}", dr=dr,
                               survey=surv, ender=ender, obscon=oc)
    # ADM return the filename.
    fileform = os.path.join(hpdirname, os.path.basename(hugefn))
    if returnoc:
        return fileform, oc
    return fileform


def read_mtl_in_hp(hpdirname, nside, pixlist, unique=True, returnfn=False):
    """Read Merged Target List ledgers in a set of HEALPixels.

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
    unique : :class:`bool`, optional, defaults to ``True``
        If ``True`` then only read targets with unique `TARGETID`, where
        the last occurrence of the target in the ledger is the one that
        is retained. If ``False`` then read the entire ledger.
    returnfn : :class:`bool`, optional, defaults to ``False``
        If ``True`` then also return a dictionary of the filename
        that had to be read in each pixel to retrieve the MTL(s).

    Returns
    -------
    :class:`~numpy.ndarray`
        A numpy structured array of the MTL(s).
    :class:`dict`
        A dictionary where the keys are pixels and values are filenames
        that were read (only returned if `returnfn` is ``True``).

    Notes
    -----
        - In general, this will be quicker if `pixlist` contains closely
          grouped HEALPixels, as fewer files will need to be read.
    """
    # ADM allow an integer instead of a list to be passed.
    if isinstance(pixlist, int):
        pixlist = [pixlist]

    # ADM if a directory was passed, do fancy HEALPixel parsing...
    outfns = {}
    fileform = find_mtl_file_format_from_header(hpdirname)
    filenside = int(read_keyword_from_mtl_header(hpdirname, "FILENSID"))
    if os.path.isdir(hpdirname):
        # ADM change the passed pixels to the nside of the file schema.
        filepixlist = nside2nside(nside, filenside, pixlist)

        # ADM read in the files and concatenate the resulting targets.
        mtls = []
        outfns = {}
        for pix in filepixlist:
            fn = fileform.format(pix)
            try:
                targs = read_mtl_ledger(fn, unique=unique)
                mtls.append(targs)
                outfns[pix] = fn
            except FileNotFoundError:
                pass

        # ADM if no mtls, look up the data model, return an empty array.
        if len(mtls) == 0:
            fns = iglob(os.path.join(hpdirname, '*.{}'.format(ender)))
            fn = next(fns)
            mtl = read_mtl_ledger(fn)
            outly = np.zeros(0, dtype=mtl.dtype)
            if returnfn:
                return outly, outfns
            return outly

        mtl = np.concatenate(mtls)
    # ADM ...if a directory wasn't passed, just read in the targets.
    else:
        mtl = read_mtl_ledger(hpdirname, unique=unique)

    # ADM restrict the targets to the actual requested HEALPixels...
    ii = is_in_hp(mtl, nside, pixlist)
    mtl = mtl[ii]

    if returnfn:
        return mtl, outfns
    return mtl


def read_targets_in_hp(hpdirname, nside, pixlist, columns=None, header=False,
                       quick=False, downsample=None, verbose=False,
                       mtl=False, unique=True):
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
        Only return these target columns.
    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return the header of either the `hpdirname`
        file, or the last file read from the `hpdirname` directory.
    quick : :class:`bool`, optional, defaults to ``False``
        If ``True``, call :func:`desitarget.io.read_targets_in_quick()`.
        That version of the code assumes that `hpdirname` is a directory,
        which contains files that follow a strict data model. ``True``
        overrides the `mtl`, `unique`, `downsample` and `verbose` inputs.
    downsample : :class:`int`, optional, defaults to `None`
        If not `None`, downsample targets by (roughly) this value, e.g.
        for `downsample=10` a set of 900 targets would have ~90 random
        targets returned.
    verbose : :class:`bool`, optional, defaults to ``False``
        Passed to :func:`read_target_files()`.
    mtl : :class:`bool`, optional, defaults to ``False``
        If ``True`` then read an MTL ledger file/directory instead
        of targets. If ``True`` then the `columns`, `header` and
        `downsample` kwargs are ignored and the full ledger is returned.
    unique : :class:`bool`, optional, defaults to ``True``
        If ``True`` then only read targets with unique `TARGETID` from
        MTL ledgers. Only used if `mtl` is ``True``.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of targets in the passed pixels.

    Notes
    -----
        - If `header` is ``True``, then a second output (the file
          header is returned).
        - In general, this will be quicker if `pixlist` contains closely
          grouped HEALPixels, as fewer files will need to be read.
        - If `mtl` is ``True`` then this is just a wrapper on
          read_mtl_in_hp().
    """
    # ADM if quick is True, use the quick-code.
    if quick:
        return read_targets_in_quick(
            hpdirname, shape='hp', nside=nside,
            pixlist=pixlist, columns=columns, header=header)

    if mtl:
        return read_mtl_in_hp(hpdirname, nside, pixlist, unique=unique)

    # ADM allow an integer instead of a list to be passed.
    if isinstance(pixlist, int):
        pixlist = [pixlist]

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
        # ADM read in the first file to grab the data model for
        # ADM cases where we find no targets in the passed pixlist.
        fn0 = list(filedict.values())[0]
        notargs, nohdr = read_target_files(
            fn0, columns=columnscopy, rows=0, header=True,
            downsample=downsample, verbose=verbose)
        notargs = np.zeros(0, dtype=notargs.dtype)

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
            targs, hdr = read_target_files(
                infile, columns=columnscopy, header=True,
                downsample=downsample, verbose=verbose)
            targets.append(targs)
        # ADM if targets is empty, return no targets.
        if len(targets) == 0:
            if header:
                return notargs, nohdr
            else:
                return notargs
        targets = np.concatenate(targets)
    # ADM ...otherwise just read in the targets.
    else:
        targets, hdr = read_target_files(
            hpdirname, columns=columnscopy, header=True,
            downsample=downsample, verbose=verbose)

    # ADM restrict the targets to the actual requested HEALPixels...
    ii = is_in_hp(targets, nside, pixlist)
    targets = targets[ii]

    # ADM ...and remove RA/Dec columns if we added them.
    if len(addedcols) > 0:
        targets = rfn.drop_fields(targets, addedcols)

    if header:
        return targets, hdr
    return targets


def read_targets_in_quick(hpdirname, shape=None,
                          tiles=None,
                          nside=None, pixlist=None,
                          radecbox=[0., 360., -90., 90.],
                          radecrad=None,
                          columns=None, header=False):
    """Read targets in various shapes, assuming a "standard" data model.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to a directory containing targets that
        have been partitioned by HEALPixel (e.g. as made by
        `select_targets` with the `bundle_files` option).
    shape : :class:`str`
        Type of geometric constraint being passed, options are "tiles",
        "box", "cap", "hp".
    tiles : :class:`~numpy.ndarray`, optional
        Array of tiles in the desimodel format, or ``None`` for all tiles
        from :func:`desimodel.io.load_tiles`. Only used if `shape=tiles`.
    nside : :class:`int`
        The (NESTED) HEALPixel nside. Only used if `shape=hp`.
    pixlist : :class:`list` or `int` or `~numpy.ndarray`
        HEALPixels at the passed `nside`. Only used if `shape=hp`.
    radecbox : :class:`list`, defaults to the entire sky
        4-entry list of coordinates [ramin, ramax, decmin, decmax]
        forming box edges in RA/Dec (degrees). Only used if `shape=box`.
    radecrad : :class:`list`
        3-entry list of coordinates [ra, dec, radius] forming a cap or
        on the sky. ra, dec, radius in degrees. Only used if `shape=cap`.
    columns : :class:`list`, optional
        Only read in these target columns.
    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return the header of either the `hpdirname`
        file, or the last file read from the `hpdirname` directory.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of targets in the passed geometric constraint.

    Notes
    -----
        - If `header` is ``True``, then a second output (the file
          header is returned).
        - $DESIMODEL must be set if `shape="tiles"` and `tiles=None`.
        - Assumes that the data model has these characteristics:
            - one HEALPixel per file.
            - no extraneous FITS files in the directory.
            - every file is formatted to finish "hp-{}.fits".
            - every file has the same, correct FILENSID in its header.
            - the data, and related header, are in FITS extension 1.
        - If you aren't sure if the data model you have will work, or if
          running this code triggers an exception, instead try running the
          relevant "slow" function with quick=``False`` as a check, e.g.
          :func:`desitarget.io.read_targets_in_tiles()`. The output
          TARGETIDs from this function and that approach should be
          identical, although the output may be ordered differently.
        - Based on a suggestion from Anand Raichoor.
    """
    allowed_shapes = ["tiles", "box", "cap", "hp"]
    if shape not in allowed_shapes:
        msg = "shape must be one of {}!!!".format(allowed_shapes)
        log.critical(msg)
        raise IOError(msg)

    if shape == "tiles":
        if tiles is None:
            # ADM check that the DESIMODEL environment variable is set.
            if os.environ.get('DESIMODEL') is None:
                msg = "DESIMODEL environment variable must be set!!!"
                log.critical(msg)
                raise ValueError(msg)
            # ADM if no tiles were sent, default to the entire footprint.
            import desimodel.io as dmio
            tiles = dmio.load_tiles()

    # ADM generator for the FITS files in the passed directory.
    fns = iglob(os.path.join(hpdirname, "*fits"))
    fn = next(fns)
    # ADM grab the FILENSID from one of the files.
    filenside = fitsio.read_header(fn, 1)["FILENSID"]
    # ADM "standard" format formatter for a file:
    formatter = fn.split("hp-")[0]+"hp-{}.fits"
    # ADM grab the data model for cases where we find no targets.
    notargs, nohdr = fitsio.read(fn, columns=columns, rows=0, header=True)
    notargs = np.zeros(0, dtype=notargs.dtype)

    if shape == 'tiles':
        # ADM closest nside to DESI tile area of ~7 deg.
        nside = pixarea2nside(7.)
        # ADM determine the pixels that touch the tiles.
        pixlist = tiles2pix(nside, tiles=tiles)
    if shape == 'box':
        # ADM approximate nside for area of passed box.
        nside = pixarea2nside(box_area(radecbox))
        # ADM HEALPixels that touch the box for that nside.
        pixlist = hp_in_box(nside, radecbox)
    if shape == 'cap':
        # ADM approximate nside for area of passed cap.
        nside = pixarea2nside(cap_area(np.array(radecrad[2])))
        # ADM HEALPixels that touch the cap for that nside.
        pixlist = hp_in_cap(nside, radecrad)

    # ADM determine the relevant HEALPixels for the file NSIDE.
    filepixlist = nside2nside(nside, filenside, pixlist)

    targets = []
    for pix in filepixlist:
        infile = formatter.format(pix)
        try:
            radec = fitsio.read(infile, columns=["RA", "DEC"])
            if shape == 'hp':
                # ADM restrict to only targets in the passed pixels.
                ii = is_in_hp(radec, nside, pixlist)
            if shape == 'tiles':
                # ADM restrict to only targets in the passed tiles.
                ii = is_point_in_desi(tiles, radec["RA"], radec["DEC"])
            elif shape == 'box':
                # ADM restrict to only targets in the passed box.
                ii = is_in_box(radec, radecbox)
            elif shape == 'cap':
                # ADM restrict to only targets in the passed cap.
                ii = is_in_cap(radec, radecrad)
            if np.sum(ii) > 0:
                targs, hdr = fitsio.read(infile, rows=np.where(ii)[0],
                                         columns=columns, header=True)
                targets.append(targs)
        except OSError:
            msg = "passed shape lies partially beyond the footprint of targets"
            log.warning(msg)
    # ADM if targets is empty, return no targets.
    if len(targets) == 0:
        targets, hdr = notargs, nohdr
    else:
        targets = np.concatenate(targets)

    if header:
        return targets, hdr
    return targets


def read_targets_in_tiles(hpdirname, tiles=None, columns=None, header=False,
                          quick=False, mtl=False, unique=True):
    """Read targets in DESI tiles, assuming the "standard" data model.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to either a directory containing targets that
        have been partitioned by HEALPixel (e.g. as made by
        `select_targets` with the `bundle_files` option). Or the
        name of a single file of targets.
    tiles : :class:`~numpy.ndarray`, optional
        Array of tiles in the desimodel format, or ``None`` to use all
        DESI tiles from :func:`desimodel.io.load_tiles`.
    columns : :class:`list`, optional
        Only read in these target columns.
    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return the header of either the `hpdirname`
        file, or the last file read from the `hpdirname` directory.
    quick : :class:`bool`, optional, defaults to ``False``
        If ``True``, call :func:`desitarget.io.read_targets_in_quick()`.
        That version of the code assumes that `hpdirname` is a directory,
        which contains files that follow a strict data model. Passing
        quick=``True`` overrides the `mtl` and `unique` inputs.
    mtl : :class:`bool`, optional, defaults to ``False``
        If ``True`` then read an MTL ledger file/directory instead
        of a target file/directory. If ``True`` then the `columns`
        and `header` kwargs are ignored and the full ledger is returned.
    unique : :class:`bool`, optional, defaults to ``True``
        If ``True`` then only read targets with unique `TARGETID` from
        MTL ledgers. Only used if `mtl` is ``True``.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of targets in the passed tiles.

    Notes
    -----
        - If `header` is ``True``, then a second output (the file
          header is returned).
        - The environment variable $DESIMODEL must be set.
    """
    # ADM if quick is True, use the quick-code.
    if quick:
        return read_targets_in_quick(hpdirname, shape='tiles', tiles=tiles,
                                     columns=columns, header=header)

    # ADM check that the DESIMODEL environment variable is set.
    if os.environ.get('DESIMODEL') is None:
        msg = "DESIMODEL environment variable must be set!!!"
        log.critical(msg)
        raise ValueError(msg)

    # ADM if no tiles were sent, default to the entire footprint.
    if tiles is None:
        import desimodel.io as dmio
        tiles = dmio.load_tiles()

    # ADM we'll need RA/Dec for final cuts, so ensure they're read.
    addedcols = []
    columnscopy = None
    if columns is not None and not mtl:
        # ADM make a copy of columns, as it's a kwarg we'll modify.
        columnscopy = columns.copy()
        for radec in ["RA", "DEC"]:
            if radec not in columnscopy:
                columnscopy.append(radec)
                addedcols.append(radec)

    # ADM if a directory was passed, do fancy HEALPixel parsing...
    if os.path.isdir(hpdirname) or mtl:
        # ADM closest nside to DESI tile area of ~7 deg.
        nside = pixarea2nside(7.)

        # ADM determine the pixels that touch the tiles.
        pixlist = tiles2pix(nside, tiles=tiles)

        # ADM read in targets in these HEALPixels.
        targets = read_targets_in_hp(hpdirname, nside, pixlist,
                                     columns=columnscopy, header=header,
                                     mtl=mtl, unique=unique)
    # ADM ...otherwise just read in the targets.
    else:
        targets = read_target_files(hpdirname, columns=columnscopy,
                                    header=header)

    # ADM if we read a header, targets is now a two-entry list.
    if header and not mtl:
        targets, hdr = targets

    # ADM restrict only to targets in the requested tiles...
    ii = is_point_in_desi(tiles, targets["RA"], targets["DEC"])
    targets = targets[ii]

    # ADM ...and remove RA/Dec columns if we added them.
    if not mtl and len(addedcols) > 0:
        targets = rfn.drop_fields(targets, addedcols)

    if header and not mtl:
        return targets, hdr
    return targets


def read_targets_in_box(hpdirname, radecbox=[0., 360., -90., 90.],
                        columns=None, header=False, quick=False, downsample=None,
                        mtl=False, unique=True):
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
    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return the header of either the `hpdirname`
        file, or the last file read from the `hpdirname` directory.
    quick : :class:`bool`, optional, defaults to ``False``
        If ``True``, call :func:`desitarget.io.read_targets_in_quick()`.
        That version of the code assumes that `hpdirname` is a directory,
        which contains files that follow a strict data model. ``True``
        overrides the `mtl` and `unique` inputs.
    downsample : :class:`int`, optional, defaults to `None`
        If not `None`, downsample targets by (roughly) this value, e.g.
        for `downsample=10` a set of 900 targets would have ~90 random
        targets returned.
    mtl : :class:`bool`, optional, defaults to ``False``
        If ``True`` then read an MTL ledger file/directory instead
        of a target file/directory. If ``True`` then the `columns`
        and `header` kwargs are ignored and the full ledger is returned.
    unique : :class:`bool`, optional, defaults to ``True``
        If ``True`` then only read targets with unique `TARGETID` from
        MTL ledgers. Only used if `mtl` is ``True``.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of targets in the passed RA/Dec box.

    Notes
    -----
        - If `header` is ``True``, then a second output (the file
          header is returned).
    """
    # ADM if quick is True, use the quick-code.
    if quick:
        return read_targets_in_quick(hpdirname, shape='box', radecbox=radecbox,
                                     columns=columns, header=header)

    # ADM we'll need RA/Dec for final cuts, so ensure they're read.
    addedcols = []
    columnscopy = None
    if columns is not None and not mtl:
        # ADM make a copy of columns, as it's a kwarg we'll modify.
        columnscopy = columns.copy()
        for radec in ["RA", "DEC"]:
            if radec not in columnscopy:
                columnscopy.append(radec)
                addedcols.append(radec)

    # ADM if a directory was passed, do fancy HEALPixel parsing...
    if os.path.isdir(hpdirname) or mtl:
        # ADM approximate nside for area of passed box.
        nside = pixarea2nside(box_area(radecbox))
        # ADM HEALPixels that touch the box for that nside.
        pixlist = hp_in_box(nside, radecbox)
        # ADM read in targets in these HEALPixels.
        targets = read_targets_in_hp(hpdirname, nside, pixlist, mtl=mtl,
                                     columns=columnscopy, header=header,
                                     downsample=downsample, unique=unique)
    # ADM ...otherwise just read in the targets.
    else:
        targets = read_target_files(hpdirname, columns=columnscopy,
                                    header=header, downsample=downsample)

    # ADM if we read a header, targets is now a two-entry list.
    if header and not mtl:
        targets, hdr = targets

    # ADM restrict only to targets in the requested RA/Dec box...
    ii = is_in_box(targets, radecbox)
    targets = targets[ii]

    # ADM ...and remove RA/Dec columns if we added them.
    if not mtl and len(addedcols) > 0:
        targets = rfn.drop_fields(targets, addedcols)

    if header and not mtl:
        return targets, hdr
    return targets


def read_targets_in_cap(hpdirname, radecrad, columns=None, header=False,
                        quick=False, mtl=False, unique=True):
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
    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return the header of either the `hpdirname`
        file, or the last file read from the `hpdirname` directory.
    quick : :class:`bool`, optional, defaults to ``False``
        If ``True``, call :func:`desitarget.io.read_targets_in_quick()`.
        That version of the code assumes that `hpdirname` is a directory,
        which contains files that follow a strict data model. ``True``
        overrides the `mtl` and `unique` inputs.
    mtl : :class:`bool`, optional, defaults to ``False``
        If ``True`` then read an MTL ledger file/directory instead
        of a target file/directory. If ``True`` then the `columns`
        kwarg is ignored and the full ledger is returned.
    unique : :class:`bool`, optional, defaults to ``True``
        If ``True`` then only read targets with unique `TARGETID` from
        MTL ledgers. Only used if `mtl` is ``True``.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of targets in the passed cap.
    """
    # ADM if quick is True, use the quick-code.
    if quick:
        return read_targets_in_quick(hpdirname, shape='cap', radecrad=radecrad,
                                     columns=columns, header=header)

    # ADM we'll need RA/Dec for final cuts, so ensure they're read.
    addedcols = []
    columnscopy = None
    if columns is not None and not mtl:
        # ADM make a copy of columns, as it's a kwarg we'll modify.
        columnscopy = columns.copy()
        for radec in ["RA", "DEC"]:
            if radec not in columnscopy:
                columnscopy.append(radec)
                addedcols.append(radec)

    # ADM if a directory was passed, do fancy HEALPixel parsing...
    if os.path.isdir(hpdirname) or mtl:
        # ADM approximate nside for area of passed cap.
        nside = pixarea2nside(cap_area(np.array(radecrad[2])))

        # ADM HEALPixels that touch the cap for that nside.
        pixlist = hp_in_cap(nside, radecrad)

        # ADM read in targets in these HEALPixels.
        targets = read_targets_in_hp(hpdirname, nside, pixlist, mtl=mtl,
                                     columns=columnscopy, header=header,
                                     unique=unique)
    # ADM ...otherwise just read in the targets.
    else:
        targets = read_target_files(hpdirname, columns=columnscopy,
                                    header=header)

    # ADM if we read a header, targets is now a two-entry list.
    if header and not mtl:
        targets, hdr = targets

    # ADM restrict only to targets in the requested cap.
    ii = is_in_cap(targets, radecrad)
    targets = targets[ii]

    # ADM Remove the RA/Dec columns if we added them.
    if not mtl and len(addedcols) > 0:
        targets = rfn.drop_fields(targets, addedcols)

    if header and not mtl:
        return targets, hdr
    return targets


def read_targets_header(hpdirname, dtype=False, verbose=True):
    """Read in header of a targets file.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to either a directory containing targets that
        have been partitioned by HEALPixel (i.e. as made by
        `select_targets` with the `bundle_files` option). Or the
        name of a single file of targets.
    dtype : :class:`bool`, optional, defaults to ``False``
        if ``True``, also return the data model (dtype) of the targets.
    verbose : :class:`bool`, optional, defaults to ``False``
        If ``True`` then log messages and warnings.

    Returns
    -------
    :class:`FITSHDR`
        The header of `hpdirname` if it is a file, or the header
        of the first file encountered in `hpdirname`
    :class:`FITSHDR`
        The dtype of the file that corresponds to the header. Only
        returned if `dtype` is ``True``.
    """
    if os.path.isdir(hpdirname):
        try:
            gen = iglob(os.path.join(hpdirname, '*fits'))
            hpdirname = next(gen)
        except StopIteration:
            if verbose:
                msg = "no FITS files in {}?!".format(hpdirname)
                log.info(msg)

    # ADM rows=[0] here, speeds up read_target_files retrieval
    # ADM of the header.
    row, hdr = read_target_files(hpdirname, rows=[0], header=True, verbose=False)

    if dtype:
        return hdr, row.dtype
    return hdr


def target_columns_from_header(hpdirname):
    """Grab the _TARGET column names from a TARGETS file or directory.

    Parameters
    ----------
    hpdirname : :class:`str`
        Full path to either a directory containing targets that
        have been partitioned by HEALPixel (i.e. as made by
        `select_targets` with the `bundle_files` option). Or the
        name of a single file of targets.

    Returns
    -------
    :class:`list`
        The names of the _TARGET columns, notably whether they are
        SV, main, or cmx _TARGET columns.
    """
    # ADM determine whether we're dealing with a file or directory.
    fn = hpdirname
    if os.path.isdir(hpdirname):
        fn = next(iglob(os.path.join(hpdirname, '*fits')))

    # ADM read in the header and find any columns matching _TARGET.
    allcols = np.array(fitsio.FITS(fn)["TARGETS"].get_colnames())
    targcols = allcols[['_TARGET' in col for col in allcols]]

    return list(targcols)


def _check_hpx_length(hpxlist, length=68, warning=False):
    """Check a list expressed as a csv string won't exceed a length."""
    pixstring = ",".join([str(i) for i in np.atleast_1d(hpxlist)])
    if len(pixstring) > length:
        msg = "Pixel string {} is too long. Maximum is length-{} strings. "  \
              "If making files, try reducing nside or the bundling integer."  \
              .format(pixstring, length)
        if warning:
            log.warning(msg)
        else:
            log.critical(msg)
            raise ValueError(msg)


def check_both_set(hpxlist, nside):
    """Check that if one of two variables is set, the other is too"""
    if hpxlist is not None or nside is not None:
        if hpxlist is None or nside is None:
            msg = 'Both hpxlist (={}) and nside (={}) need to be set' \
                .format(hpxlist, nside)
            log.critical(msg)
            raise ValueError(msg)


def hpx_filename(hpx):
    """Return the standard name for HEALPixel-split input files

    Parameters
    ----------
    hpx : :class:`str` or `int`
        A HEALPixel integer.

    Returns
    -------
    :class: `str`
        Filename in the format used throughout desitarget for
        HEALPixel-split input databases.
    """

    return 'healpix-{:05d}.fits'.format(hpx)


def find_star_files(objs, hpxdir, nside, neighbors=True, radec=False,
                    strict=False):
    """Full paths to HEALPixel-split star files for objects by RA/Dec.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Array of objects. Must contain at least columns "RA" and "DEC".
    hpxdir : :class:`str`
        Name of the directory that hosts the HEALPixel-split files. Most
        likely this directory ends in "/healpix".
    nside : :class:`int`
        The (NESTED) HEALPixel nside integer.
    neighbors : :class:`bool`, optional, defaults to ``True``
        Also return all neighboring pixels that touch the files of
        interest to prevent edge effects (e.g. if a, say, Gaia source is
        1 arcsec away from a primary source and so in an adjacent pixel).
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` then the passed `objs` is an [RA, Dec] list instead
        of a rec array.
    strict : :class:`bool`, optional, defaults to ``False``
        Only return files that actually exist. This is useful for, e.g.,
        URAT files, which don't cover the whole sky and so don't have
        files for every HEALPixel.

    Returns
    -------
    :class:`list`
        A list of all files that need to be read in to account for
        objects at the passed locations.

    Notes
    -----
        - "star" files, here might be Gaia, Tycho or URAT files.
    """
    # ADM which flavor of RA/Dec was passed.
    if radec:
        ra, dec = objs
        dec = np.array(dec)
    else:
        ra, dec = objs["RA"], objs["DEC"]

    # ADM convert RA/Dec to co-latitude and longitude in radians.
    theta, phi = np.radians(90-dec), np.radians(ra)

    # ADM retrieve the pixels in which the locations lie.
    pixnum = hp.ang2pix(nside, theta, phi, nest=True)

    # ADM retrieve only the UNIQUE pixel numbers. It's possible that only
    # ADM one pixel was produced, so ensure pixnum is iterable.
    if not isinstance(pixnum, np.integer):
        pixnum = list(set(pixnum))
    else:
        pixnum = [pixnum]

    # ADM if neighbors was sent, then retrieve all pixels that touch each
    # ADM pixel covered by the passed ras/decs, to prevent edge effects...
    if neighbors:
        pixnum = add_hp_neighbors(nside, pixnum)

    # ADM reformat in the general healpix format used by desitarget.
    fns = [os.path.join(hpxdir, hpx_filename(pn)) for pn in pixnum]

    # ADM restrict to only files/HEALPixels actually covered.
    if strict:
        fns = [fn for fn in fns if os.path.exists(fn)]

    return fns
