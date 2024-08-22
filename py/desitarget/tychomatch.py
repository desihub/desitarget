# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.tychomatch
=====================

Useful Tycho catalog matching and manipulation routines.
"""
import os
import numpy as np
import fitsio
import requests
import pickle
from datetime import datetime

from time import time
from astropy.io import ascii
from glob import glob
import healpy as hp

from desitarget import io
from desitarget.internal import sharedmem
from desimodel.footprint import radec2pix
from desitarget.geomask import add_hp_neighbors, radec_match_to, nside2nside
from desitarget.mtl import get_utc_date

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()

# ADM columns contained in our version of the Tycho fits files.
tychodatamodel = np.array([], dtype=[
    ('TYC1', '>i2'), ('TYC2', '>i2'), ('TYC3', '|u1'),
    ('RA', '>f8'), ('DEC', '>f8'),
    ('MEAN_RA', '>f8'), ('MEAN_DEC', '>f8'),
    ('SIGMA_RA', '>f4'), ('SIGMA_DEC', '>f4'),
    # ADM these are converted to be in mas/yr for consistency with Gaia.
    ('PM_RA', '>f4'), ('PM_DEC', '>f4'),
    ('SIGMA_PM_RA', '>f4'), ('SIGMA_PM_DEC', '>f4'),
    ('EPOCH_RA', '>f4'), ('EPOCH_DEC', '>f4'),
    ('MAG_BT', '>f4'), ('MAG_VT', '>f4'), ('MAG_HP', '>f4'), ('ISGALAXY', '|u1'),
    ('JMAG', '>f4'), ('HMAG', '>f4'), ('KMAG', '>f4'), ('ZGUESS', '>f4')
])


def get_tycho_dir():
    """Convenience function to grab the Tycho environment variable.

    Returns
    -------
    :class:`str`
        The directory stored in the $TYCHO_DIR environment variable.
    """
    # ADM check that the $TYCHO_DIR environment variable is set.
    tychodir = os.environ.get('TYCHO_DIR')
    if tychodir is None:
        msg = "Set $TYCHO_DIR environment variable!"
        log.critical(msg)
        raise ValueError(msg)

    return tychodir


def get_tycho_nside():
    """Grab the HEALPixel nside to be used throughout this module.

    Returns
    -------
    :class:`int`
        The HEALPixel nside number for Tycho file creation and retrieval.
    """
    nside = 4

    return nside


def grab_tycho(cosmodir="/global/cfs/cdirs/cosmo/staging/tycho2/"):
    """Retrieve the cosmo versions of the Tycho files at NERSC.

    Parameters
    ----------
    cosmodir : :class:`str`
        The NERSC directory that hosts the Tycho files.

    Returns
    -------
    None
        But the Tycho fits file, README are written to $TYCHO_DIR/fits.

    Notes
    -----
    - The environment variable $TYCHO_DIR must be set.
    - The fits file is "cleaned up" to conform to DESI Data Systems
      standards (e.g. all columns are converted to upper-case).
    """
    # ADM check that the TYCHO_DIR is set and retrieve it.
    tychodir = get_tycho_dir()

    # ADM construct the directory to which to write files.
    fitsdir = os.path.join(tychodir, 'fits')
    # ADM the directory better be empty for the copy!
    if os.path.exists(fitsdir):
        if len(os.listdir(fitsdir)) > 0:
            msg = "{} should be empty to get TYCHO FITS file!".format(fitsdir)
            log.critical(msg)
            raise ValueError(msg)
    # ADM make the directory, if needed.
    else:
        log.info('Making TYCHO directory for storing FITS files')
        os.makedirs(fitsdir)

    # ADM the actual name of the Tycho file and the associated README.
    tychofn = "tycho2.kd.fits"
    cosmofile = os.path.join(cosmodir, tychofn)
    rfile = os.path.join(cosmodir, "README")

    # ADM the associated output files.
    outfile = os.path.join(fitsdir, tychofn)
    routfile = os.path.join(fitsdir, "README")

    # ADM read in the Tycho file and header in upper-case.
    objs, hdr = fitsio.read(cosmofile, header=True, upper=True)
    nobjs = len(objs)
    done = np.zeros(nobjs, dtype=tychodatamodel.dtype)
    for col in tychodatamodel.dtype.names:
        # ADM proper motions need converted to mas/yr.
        if "PM" in col:
            done[col] = objs[col]*1000
        else:
            done[col] = objs[col]

    # ADM add some information to the header
    hdr["COPYDATE"] = get_utc_date()
    hdr["COSMODIR"] = cosmodir

    # ADM write the data.
    fitsio.write(outfile, done, extname='TYCHOFITS', header=hdr)

    # ADM also update the README.
    msg = "\nCopied from: {}\non: {}\nthe specific file being: {}\n".format(
        cosmodir, copydate, cosmofile)
    with open(rfile) as f:
        readme = f.read()
    with open(routfile, 'w') as f:
        f.write(readme+msg)

    log.info('Wrote Tycho FITS file...t={:.1f}s'.format(time()-start))

    return


def tycho_fits_to_healpix():
    """Convert files in $TYCHO_DIR/fits to files in $TYCHO_DIR/healpix.

    Returns
    -------
    None
        But the archived Tycho FITS files in $TYCHO_DIR/fits are
        rearranged by HEALPixel in the directory $TYCHO_DIR/healpix.
        The HEALPixel sense is nested with nside=get_tycho_nside(), and
        each file in $TYCHO_DIR/healpix is called healpix-xxxxx.fits,
        where xxxxx corresponds to the HEALPixel number.

    Notes
    -----
    - The environment variable $TYCHO_DIR must be set.
    """
    # ADM the resolution at which the Tycho HEALPix files are stored.
    nside = get_tycho_nside()
    npix = hp.nside2npix(nside)

    # ADM check that the TYCHO_DIR is set.
    tychodir = get_tycho_dir()

    # ADM construct the directories for reading/writing files.
    fitsdir = os.path.join(tychodir, "fits")
    tychofn = os.path.join(fitsdir, "tycho2.kd.fits")
    hpxdir = os.path.join(tychodir, "healpix")

    # ADM make sure the output directory is empty.
    if os.path.exists(hpxdir):
        if len(os.listdir(hpxdir)) > 0:
            msg = "{} must be empty to make Tycho HEALPix files!".format(hpxdir)
            log.critical(msg)
            raise ValueError(msg)
    # ADM make the output directory, if needed.
    else:
        log.info("Making Tycho directory for storing HEALPix files")
        os.makedirs(hpxdir)

    # ADM read in the Tycho file and assing Tycho objects to HEALPixels.
    objs, allhdr = fitsio.read(tychofn, header=True, upper=True)
    pix = radec2pix(nside, objs["RA"], objs["DEC"])

    # ADM loop through the pixels and write out the files.
    for pixnum in range(npix):
        # ADM construct the name of the output file.
        outfilename = io.hpx_filename(pixnum)
        outfile = os.path.join(hpxdir, outfilename)
        # ADM update the header with new information.
        hdr = dict(allhdr).copy()
        hdr["HPXNSIDE"] = nside
        hdr["HPXNEST"] = True
        hdr["HPXDATE"] = get_utc_date()

        # ADM determine which objects are in this pixel and write out.
        done = objs[pix == pixnum]

        fitsio.write(outfile, done, extname="TYCHOHPX", header=hdr)

    log.info('Wrote Tycho HEALPix files...t={:.1f}s'.format(time()-start))

    return


def make_tycho_files():
    """Make the HEALPix-split Tycho files in one fell swoop.

    Returns
    -------
    None
        But produces:

        - A FITS file with appropriate header and columns from
          `tychodatamodel`, and a README in $TYCHO_DIR/fits.
        - FITS files reorganized by HEALPixel in $TYCHO_DIR/healpix.

        The HEALPixel sense is nested with nside=get_tycho_nside(), and
        each file in $TYCHO_DIR/healpix is called healpix-xxxxx.fits,
        where xxxxx corresponds to the HEALPixel number.

    Notes
    -----
    - The environment variable $TYCHO_DIR must be set.
    """
    t0 = time()
    log.info('Begin making Tycho files...t={:.1f}s'.format(time()-t0))

    # ADM check that the TYCHO_DIR is set.
    tychodir = get_tycho_dir()

    # ADM a quick check that the fits and healpix directories are empty
    # ADM before embarking on the slower parts of the code.
    fitsdir = os.path.join(tychodir, 'fits')
    hpxdir = os.path.join(tychodir, 'healpix')
    for direc in [fitsdir, hpxdir]:
        if os.path.exists(direc):
            if len(os.listdir(direc)) > 0:
                msg = "{} should be empty to make Tycho files!".format(direc)
                log.critical(msg)
                raise ValueError(msg)

    grab_tycho()
    log.info('Copied Tycho FITS file from cosmo...t={:.1f}s'.format(time()-t0))

    tycho_fits_to_healpix()
    log.info('Rearranged FITS files by HEALPixel...t={:.1f}s'.format(time()-t0))

    return


def find_tycho_files(objs, neighbors=True, radec=False):
    """Find full paths to Tycho healpix files for objects by RA/Dec.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Array of objects. Must contain the columns "RA" and "DEC".
    neighbors : :class:`bool`, optional, defaults to ``True``
        Also return all pixels that touch the files of interest
        to prevent edge effects (e.g. if a Tycho source is 1 arcsec
        away from a primary source and so in an adjacent pixel).
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` then the passed `objs` is an [RA, Dec] list
        instead of a rec array that contains "RA" and "DEC".

    Returns
    -------
    :class:`list`
        A list of all Tycho files to read to account for objects at
        the passed locations.

    Notes
    -----
    - The environment variable $TYCHO_DIR must be set.
    """
    # ADM the resolution at which the Tycho HEALPix files are stored.
    nside = get_tycho_nside()

    # ADM check that the TYCHO_DIR is set and retrieve it.
    tychodir = get_tycho_dir()
    hpxdir = os.path.join(tychodir, 'healpix')

    return io.find_star_files(objs, hpxdir, nside,
                              neighbors=neighbors, radec=radec)


def find_tycho_files_hp(nside, pixlist, neighbors=True):
    """Find full paths to Tycho healpix files in a set of HEALPixels.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    pixlist : :class:`list` or `int`
        A set of HEALPixels at `nside`.
    neighbors : :class:`bool`, optional, defaults to ``True``
        Also return files corresponding to all neighbors that touch the
        pixels in `pixlist` to prevent edge effects (e.g. a Tycho source
        is 1 arcsec outside of `pixlist` and so in an adjacent pixel).

    Returns
    -------
    :class:`list`
        A list of all Tycho files that need to be read in to account for
        objects in the passed list of pixels.

    Notes
    -----
    - The environment variable $TYCHO_DIR must be set.
    """
    # ADM the resolution at which the healpix files are stored.
    filenside = get_tycho_nside()

    # ADM check that the TYCHO_DIR is set and retrieve it.
    tychodir = get_tycho_dir()
    hpxdir = os.path.join(tychodir, 'healpix')

    # ADM work with pixlist as an array.
    pixlist = np.atleast_1d(pixlist)

    # ADM determine the pixels that touch the passed pixlist.
    pixnum = nside2nside(nside, filenside, pixlist)

    # ADM if neighbors was sent, then retrieve all pixels that touch each
    # ADM pixel covered by the provided locations, to prevent edge effects...
    if neighbors:
        pixnum = add_hp_neighbors(filenside, pixnum)

    # ADM reformat in the healpix format used by desitarget.
    tychofiles = [os.path.join(hpxdir, io.hpx_filename(pn)) for pn in pixnum]

    return tychofiles


def match_to_tycho(objs, matchrad=1., radec=False):
    """Match objects to Tycho healpixel files.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Must contain at least "RA" and "DEC".
    matchrad : :class:`float`, optional, defaults to 1 arcsec
        The radius at which to match in arcseconds.
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` then the passed `objs` is an [RA, Dec] list instead of
        a rec array.

    Returns
    -------
    :class:`~numpy.ndarray`
        The matching Tycho information for each object. The returned
        format is as for desitarget.tychomatch.tychodatamodel with
        an extra column "TYCHO_SEP" which is the matching distance
        in ARCSECONDS.

    Notes
    -----
    - For objects with NO match in Tycho, the "TYC1", "TYC2" and
      "TYCHO_SEP" columns are -1, and other columns are zero.
    - Retrieves the CLOSEST match to Tycho for each passed object.
    - Because this reads in HEALPixel split files, it's (far) faster
      for objects that are clumped rather than widely distributed.
    """
    # ADM parse whether a structure or coordinate list was passed.
    if radec:
        ra, dec = objs
    else:
        ra, dec = objs["RA"], objs["DEC"]

    # ADM set up an array of Tycho information for the output.
    nobjs = len(ra)
    done = np.zeros(nobjs, dtype=tychodatamodel.dtype)

    # ADM objects without matches should have TYC1/2/3, TYCHO_SEP of -1.
    for col in "TYC1", "TYC2":
        done[col] = -1
    tycho_sep = np.zeros(nobjs) - 1

    # ADM determine which Tycho files need to be scraped.
    tychofiles = find_tycho_files([ra, dec], radec=True)
    nfiles = len(tychofiles)

    # ADM catch the case of no matches to Tycho.
    if nfiles > 0:
        # ADM loop through the Tycho files and find matches.
        for ifn, fn in enumerate(tychofiles):
            if ifn % 500 == 0 and ifn > 0:
                log.info('{}/{} files; {:.1f} total mins elapsed'
                         .format(ifn, nfiles, (time()-start)/60.))
            tycho = fitsio.read(fn)
            idtycho, idobjs, dist = radec_match_to(
                [tycho["RA"], tycho["DEC"]], [ra, dec],
                sep=matchrad, radec=True, return_sep=True)

            # ADM update matches whenever we have a CLOSER match.
            ii = (tycho_sep[idobjs] == -1) | (tycho_sep[idobjs] > dist)
            done[idobjs[ii]] = tycho[idtycho[ii]]
            tycho_sep[idobjs[ii]] = dist[ii]

    # ADM add the separation distances to the output array.
    dt = tychodatamodel.dtype.descr + [("TYCHO_SEP", ">f4")]
    output = np.zeros(nobjs, dtype=dt)
    for col in tychodatamodel.dtype.names:
        output[col] = done[col]
    output["TYCHO_SEP"] = tycho_sep

    return output
