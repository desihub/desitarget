# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
====================
desitarget.gaiamatch
====================

Useful Gaia matching and manipulation routines.
"""
import os
import sys
import numpy as np
import numpy.lib.recfunctions as rfn
import fitsio
import requests
import pickle
from glob import glob
from time import time
import healpy as hp
from os.path import basename
from desitarget import io
from desitarget.io import check_fitsio_version
from desitarget.internal import sharedmem
from desitarget.geomask import hp_in_box, add_hp_neighbors, pixarea2nside
from desitarget.geomask import hp_beyond_gal_b, nside2nside
from desimodel.footprint import radec2pix
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import ascii

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()

# ADM the current data model for Gaia columns for READING from Gaia files
ingaiadatamodel = np.array([], dtype=[
            ('SOURCE_ID', '>i8'), ('REF_CAT', 'S2'), ('RA', '>f8'), ('DEC', '>f8'),
            ('PHOT_G_MEAN_MAG', '>f4'), ('PHOT_G_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('PHOT_BP_MEAN_MAG', '>f4'), ('PHOT_BP_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('PHOT_RP_MEAN_MAG', '>f4'), ('PHOT_RP_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('PHOT_BP_RP_EXCESS_FACTOR', '>f4'),
            ('ASTROMETRIC_EXCESS_NOISE', '>f4'), ('DUPLICATED_SOURCE', '?'),
            ('ASTROMETRIC_SIGMA5D_MAX', '>f4'), ('ASTROMETRIC_PARAMS_SOLVED', '>i1'),
            ('PARALLAX', '>f4'), ('PARALLAX_ERROR', '>f4'),
            ('PMRA', '>f4'), ('PMRA_ERROR', '>f4'),
            ('PMDEC', '>f4'), ('PMDEC_ERROR', '>f4')
                                   ])

# ADM the current data model for Gaia columns for WRITING to target files
gaiadatamodel = np.array([], dtype=[
            ('REF_ID', '>i8'), ('REF_CAT', 'S2'), ('GAIA_RA', '>f8'), ('GAIA_DEC', '>f8'),
            ('GAIA_PHOT_G_MEAN_MAG', '>f4'), ('GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('GAIA_PHOT_BP_MEAN_MAG', '>f4'), ('GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('GAIA_PHOT_RP_MEAN_MAG', '>f4'), ('GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('GAIA_PHOT_BP_RP_EXCESS_FACTOR', '>f4'),
            ('GAIA_ASTROMETRIC_EXCESS_NOISE', '>f4'), ('GAIA_DUPLICATED_SOURCE', '?'),
            ('GAIA_ASTROMETRIC_SIGMA5D_MAX', '>f4'), ('GAIA_ASTROMETRIC_PARAMS_SOLVED', '>i1'),
            ('PARALLAX', '>f4'), ('PARALLAX_IVAR', '>f4'),
            ('PMRA', '>f4'), ('PMRA_IVAR', '>f4'),
            ('PMDEC', '>f4'), ('PMDEC_IVAR', '>f4')
                                   ])


def get_gaia_dir():
    """Convenience function to grab the Gaia environment variable.

    Returns
    -------
    :class:`str`
        The directory stored in the $GAIA_DIR environment variable.
    """
    # ADM check that the $GAIA_DIR environment variable is set.
    gaiadir = os.environ.get('GAIA_DIR')
    if gaiadir is None:
        msg = "Set $GAIA_DIR environment variable!"
        log.critical(msg)
        raise ValueError(msg)

    return gaiadir


def _get_gaia_nside():
    """Grab the HEALPixel nside to be used throughout this module.

    Returns
    -------
    :class:`int`
        The HEALPixel nside number for Gaia file creation and retrieval.
    """
    nside = 32

    return nside


def get_gaia_nside_brick(bricksize=0.25):
    """Grab the HEALPixel nside that corresponds to a brick.

    Parameters
    ----------
    bricksize : :class:`float`, optional, defaults to 0.25
        Size of the brick, default is the Legacy Surveys standard.

    Returns
    -------
    :class:`int`
        The HEALPixel nside number that corresponds to a brick.
    """

    return pixarea2nside(bricksize*bricksize)


def gaia_psflike(aen, g):
    """Whether an objects is PSF-like based on Gaia quantities.

    Parameters
    ----------
    aen : :class:`array_like` or :class`float`
        Gaia Astrometric Excess Noise.
    g : :class:`array_like` or :class`float`
        Gaia-based g MAGNITUDE (not Galactic-extinction-corrected).

    Returns
    -------
    :class:`array_like` or :class`float`
        A boolean that is ``True`` for objects that are psf-like
        based on Gaia quantities.

    Notes
    -----
        - Input quantities are the same as in `the Gaia data model`_.
    """
    psflike = np.logical_or(
        (g <= 19.) * (aen < 10.**0.5),
        (g >= 19.) * (aen < 10.**(0.5 + 0.2*(g - 19.)))
    )

    return psflike


def is_in_Galaxy(objs, radec=False):
    """An (l, b) cut developed by Boris Gaensicke to avoid the Galaxy.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Array of objects. Must contain at least the columns "RA" and "DEC".
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` then the passed `objs` is an [RA, Dec] list instead of
        a rec array.

    Returns
    -------
    :class:`~numpy.ndarray`
        A boolean array that is ``True`` for objects that are close to
        the Galaxy and ``False`` for objects that aren't.
    """
    # ADM which flavor of RA/Dec was passed.
    if radec:
        ra, dec = objs
    else:
        ra, dec = objs["RA"], objs["DEC"]

    # ADM convert to Galactic coordinates.
    c = SkyCoord(ra*u.degree, dec*u.degree)
    gal = c.galactic

    # ADM and limit to (l, b) ranges.
    ii = np.abs(gal.b.value) < np.abs(gal.l.value*0.139-25)

    return ii


def gaia_dr_from_ref_cat(refcat):
    """Determine the Gaia DR from an array of values, check it's unique.

    Parameters
    ----------
    ref_cat : :class:`~numpy.ndarray` or `str`
        A `REF_CAT` string or an array of `REF_CAT` strings (e.g. b"G2").

    Returns
    -------
    :class:`~numpy.ndarray`
        The corresponding Data Release number (e.g. 2)

    Notes
    -----
        - In reality, only strips the final integer off strings like
          "X3". So, can generically be used for that purpose.
    """
    # ADM if an integer was passed.
    refcat = np.atleast_1d(refcat)
    # ADM in case old-style byte strings were passed.
    if isinstance(refcat[0], bytes):
        return np.array([int(i.decode()[-1]) for i in refcat])
    else:
        return np.array([int(i[-1]) for i in refcat])

    return gaiadr


def scrape_gaia(url="http://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/", nfiletest=None):
    """Retrieve the bulk CSV files released by the Gaia collaboration.

    Parameters
    ----------
    url : :class:`str`
        The web directory that hosts the archived Gaia CSV files.
    nfiletest : :class:`int`, optional, defaults to ``None``
        If an integer is sent, only retrieve this number of files, for testing.

    Returns
    -------
    Nothing
        But the archived Gaia CSV files are written to $GAIA_DIR/csv.

    Notes
    -----
        - The environment variable $GAIA_DIR must be set.
        - Runs in about 26 hours for 60,000 Gaia files.
    """
    # ADM check that the GAIA_DIR is set and retrieve it.
    gaiadir = get_gaia_dir()

    # ADM construct the directory to which to write files.
    csvdir = os.path.join(gaiadir, 'csv')
    # ADM the directory better be empty for the wget!
    if os.path.exists(csvdir):
        if len(os.listdir(csvdir)) > 0:
            msg = "{} should be empty to wget Gaia csv files!".format(csvdir)
            log.critical(msg)
            raise ValueError(msg)
    # ADM make the directory, if needed.
    else:
        log.info('Making Gaia directory for storing CSV files')
        os.makedirs(csvdir)

    # ADM pull back the index.html from the url.
    index = requests.get(url)

    # ADM retrieve any file name that starts with GaiaSource.
    # ADM the [1::2] pulls back just the odd lines from the split list.
    filelist = index.text.split("GaiaSource")[1::2]

    # ADM if nfiletest was passed, just work with that number of files.
    test = nfiletest is not None
    if test:
        filelist = filelist[:nfiletest]
    nfiles = len(filelist)

    # ADM loop through the filelist.
    t0 = time()
    for nfile, fileinfo in enumerate(filelist):
        # ADM make the wget command to retrieve the file and issue it.
        cmd = 'wget -q {}/GaiaSource{} -P {}'.format(url, fileinfo[:-2], csvdir)
        os.system(cmd)
        if nfile % 100 == 0 or test:
            elapsed = time() - t0
            rate = nfile / elapsed
            log.info(
                '{}/{} files; {:.1f} files/sec; {:.1f} total mins elapsed'
                .format(nfile+1, nfiles, rate, elapsed/60.)
            )

    log.info('Done...t={:.1f}s'.format(time()-t0))

    return


def gaia_csv_to_fits(numproc=4):
    """Convert files in $GAIA_DIR/csv to files in $GAIA_DIR/fits.

    Parameters
    ----------
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.

    Returns
    -------
    Nothing
        But the archived Gaia CSV files in $GAIA_DIR/csv are converted
        to FITS files in the directory $GAIA_DIR/fits. Also, a look-up
        table is written to $GAIA_DIR/fits/hpx-to-files.pickle for which
        each index is an nside=_get_gaia_nside(), nested scheme HEALPixel
        and each entry is a list of the FITS files that touch that HEAPixel.

    Notes
    -----
        - The environment variable $GAIA_DIR must be set.
        - if numproc==1, use the serial code instead of the parallel code.
        - Runs in 1-3 hours (depending on node) with numproc=32 for 60,000 files.
    """
    # ADM the resolution at which the Gaia HEALPix files should be stored.
    nside = _get_gaia_nside()

    # ADM check that the GAIA_DIR is set.
    gaiadir = get_gaia_dir()
    log.info("running on {} processors".format(numproc))

    # ADM construct the directories for reading/writing files.
    csvdir = os.path.join(gaiadir, 'csv')
    fitsdir = os.path.join(gaiadir, 'fits')

    # ADM make sure the output directory is empty.
    if os.path.exists(fitsdir):
        if len(os.listdir(fitsdir)) > 0:
            msg = "{} should be empty to make Gaia FITS files!".format(fitsdir)
            log.critical(msg)
            raise ValueError(msg)
    # ADM make the output directory, if needed.
    else:
        log.info('Making Gaia directory for storing FITS files')
        os.makedirs(fitsdir)

    # ADM construct the list of input files.
    infiles = glob("{}/*csv*".format(csvdir))
    nfiles = len(infiles)

    # ADM the critical function to run on every file.
    def _write_gaia_fits(infile):
        """read an input name for a csv file and write it to FITS"""
        outbase = os.path.basename(infile)
        outfilename = "{}.fits".format(outbase.split(".")[0])
        outfile = os.path.join(fitsdir, outfilename)
        fitstable = ascii.read(infile, format='csv')

        # ADM need to convert 5-string values to boolean.
        cols = np.array(fitstable.dtype.names)
        boolcols = cols[np.hstack(fitstable.dtype.descr)[1::2] == '<U5']
        for col in boolcols:
            fitstable[col] = fitstable[col] == 'true'

        # ADM only write out the columns we need for targeting.
        nobjs = len(fitstable)
        done = np.zeros(nobjs, dtype=ingaiadatamodel.dtype)
        for col in done.dtype.names:
            if col == 'REF_CAT':
                done[col] = 'G2'
            else:
                done[col] = fitstable[col.lower()]
        fitsio.write(outfile, done, extname='GAIAFITS')

        # ADM return the HEALPixels that this file touches.
        pix = set(radec2pix(nside, fitstable["ra"], fitstable["dec"]))
        return [pix, os.path.basename(outfile)]

    # ADM this is just to count processed files in _update_status.
    nfile = np.zeros((), dtype='i8')
    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if nfile % 100 == 0 and nfile > 0:
            rate = nfile / (time() - t0)
            elapsed = time() - t0
            log.info(
                '{}/{} files; {:.1f} files/sec; {:.1f} total mins elapsed'
                .format(nfile, nfiles, rate, elapsed/60.)
            )
        nfile[...] += 1    # this is an in-place modification
        return result

    # - Parallel process input files...
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            pixinfile = pool.map(_write_gaia_fits, infiles, reduce=_update_status)
    # ADM ...or run in serial.
    else:
        pixinfile = list()
        for file in infiles:
            pixinfile.append(_update_status(_write_gaia_fits(file)))

    # ADM create a list for which each index is a HEALPixel and each
    # ADM entry is a list of files that touch that HEALPixel.
    npix = hp.nside2npix(nside)
    pixlist = [[] for i in range(npix)]
    for pixels, file in pixinfile:
        for pix in pixels:
            pixlist[pix].append(file)

    # ADM write out the HEALPixel->files look-up table.
    outfilename = os.path.join(fitsdir, "hpx-to-files.pickle")
    outfile = open(outfilename, "wb")
    pickle.dump(pixlist, outfile)
    outfile.close()

    log.info('Done...t={:.1f}s'.format(time()-t0))

    return


def gaia_fits_to_healpix(numproc=4):
    """Convert files in $GAIA_DIR/fits to files in $GAIA_DIR/healpix.

    Parameters
    ----------
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.

    Returns
    -------
    Nothing
        But the archived Gaia FITS files in $GAIA_DIR/fits are
        rearranged by HEALPixel in the directory $GAIA_DIR/healpix.
        The HEALPixel sense is nested with nside=_get_gaia_nside(), and
        each file in $GAIA_DIR/healpix is called healpix-xxxxx.fits,
        where xxxxx corresponds to the HEALPixel number.

    Notes
    -----
        - The environment variable $GAIA_DIR must be set.
        - if numproc==1, use the serial code instead of the parallel code.
        - Runs in about 1-3 hours with numproc=32 for 60,000 files.
    """
    # ADM the resolution at which the Gaia HEALPix files should be stored.
    nside = _get_gaia_nside()

    # ADM check that the GAIA_DIR is set.
    gaiadir = get_gaia_dir()

    # ADM construct the directories for reading/writing files.
    fitsdir = os.path.join(gaiadir, 'fits')
    hpxdir = os.path.join(gaiadir, 'healpix')

    # ADM make sure the output directory is empty.
    if os.path.exists(hpxdir):
        if len(os.listdir(hpxdir)) > 0:
            msg = "{} should be empty to make Gaia HEALPix files!".format(hpxdir)
            log.critical(msg)
            raise ValueError(msg)
    # ADM make the output directory, if needed.
    else:
        log.info('Making Gaia directory for storing HEALPix files')
        os.makedirs(hpxdir)

    # ADM read the pixel -> file look-up table.
    infilename = os.path.join(fitsdir, "hpx-to-files.pickle")
    infile = open(infilename, "rb")
    pixlist = pickle.load(infile)
    npixels = len(pixlist)
    # ADM include the pixel number explicitly in the look-up table.
    pixlist = list(zip(np.arange(npixels), pixlist))

    # ADM the critical function to run on every file.
    def _write_hpx_fits(pixlist):
        """from files that touch a pixel, write out objects in each pixel"""
        pixnum, files = pixlist
        # ADM only proceed if some files touch a pixel.
        if len(files) > 0:
            # ADM track if it's our first time through the files loop.
            first = True
            # ADM Read in files that touch a pixel.
            for file in files:
                filename = os.path.join(fitsdir, file)
                objs = fitsio.read(filename)
                # ADM only retain objects in the correct pixel.
                pix = radec2pix(nside, objs["RA"], objs["DEC"])
                if first:
                    done = objs[pix == pixnum]
                    first = False
                else:
                    done = np.hstack([done, objs[pix == pixnum]])
            # ADM construct the name of the output file.
            outfilename = io.hpx_filename(pixnum)
            outfile = os.path.join(hpxdir, outfilename)
            # ADM write out the file.
            hdr = fitsio.FITSHDR()
            hdr['HPXNSIDE'] = nside
            hdr['HPXNEST'] = True
            fitsio.write(outfile, done, extname='GAIAHPX', header=hdr)

        return

    # ADM this is just to count processed files in _update_status.
    npix = np.zeros((), dtype='i8')
    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if npix % 100 == 0 and npix > 0:
            rate = npix / (time() - t0)
            elapsed = time() - t0
            log.info(
                '{}/{} files; {:.1f} files/sec; {:.1f} total mins elapsed'
                .format(npix, npixels, rate, elapsed/60.)
            )
        npix[...] += 1    # this is an in-place modification
        return result

    # - Parallel process input files...
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            _ = pool.map(_write_hpx_fits, pixlist, reduce=_update_status)
    # ADM ...or run in serial.
    else:
        for pix in pixlist:
            _update_status(_write_hpx_fits(pix))

    log.info('Done...t={:.1f}s'.format(time()-t0))

    return


def make_gaia_files(numproc=4, download=False):
    """Make the HEALPix-split Gaia DR2 files used by desitarget.

    Parameters
    ----------
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    download : :class:`bool`, optional, defaults to ``False``
        If ``True`` then wget the Gaia DR2 csv files from ESA.

    Returns
    -------
    Nothing
        But produces:
        - Full Gaia DR2 CSV files in $GAIA_DIR/csv.
        - FITS files with columns from `ingaiadatamodel` in $GAIA_DIR/fits.
        - FITS files reorganized by HEALPixel in $GAIA_DIR/healpix.

        The HEALPixel sense is nested with nside=_get_gaia_nside(), and
        each file in $GAIA_DIR/healpix is called healpix-xxxxx.fits,
        where xxxxx corresponds to the HEALPixel number.

    Notes
    -----
        - The environment variable $GAIA_DIR must be set.
        - if numproc==1, use the serial code instead of the parallel code.
        - Runs in about 26 hours if download is ``True``.
        - Runs in 1-3 hours with numproc=32 if download is ``False``.
    """
    t0 = time()
    log.info('Begin making Gaia files...t={:.1f}s'.format(time()-t0))

    # ADM check that the GAIA_DIR is set.
    gaiadir = get_gaia_dir()

    # ADM a quick check that the fits and healpix directories are empty
    # ADM before embarking on the slower parts of the code.
    fitsdir = os.path.join(gaiadir, 'fits')
    hpxdir = os.path.join(gaiadir, 'healpix')
    for direc in [fitsdir, hpxdir]:
        if os.path.exists(direc):
            if len(os.listdir(direc)) > 0:
                msg = "{} should be empty to make Gaia files!".format(direc)
                log.critical(msg)
                raise ValueError(msg)

    if download:
        scrape_gaia()
        log.info('Retrieved Gaia files from ESA...t={:.1f}s'.format(time()-t0))

    gaia_csv_to_fits(numproc=numproc)
    log.info('Converted CSV files to FITS...t={:.1f}s'.format(time()-t0))

    gaia_fits_to_healpix(numproc=numproc)
    log.info('Rearranged FITS files by HEALPixel...t={:.1f}s'.format(time()-t0))

    return


def pop_gaia_coords(inarr):
    """Convenience function to pop GAIA_RA and GAIA_DEC columns off an array

    Parameters
    ----------
    inarr : :class:`~numpy.ndarray`
        Structured array with various column names.

    Returns
    -------
    :class:`~numpy.ndarray`
        Input array with columns called "GAIA_RA" and/or "GAIA_DEC" removed.
    """

    return rfn.drop_fields(inarr, ['GAIA_RA', 'GAIA_DEC'])


def pop_gaia_columns(inarr, popcols):
    """Convenience function to pop columns off an input array.

    Parameters
    ----------
    inarr : :class:`~numpy.ndarray`
        Structured array with various column names.
    popcols : :class:`list`
        List of columns to remove from the input array.

    Returns
    -------
    :class:`~numpy.ndarray`
        Input array with columns in cols removed.
    """

    return rfn.drop_fields(inarr, popcols)


def read_gaia_file(filename, header=False, addobjid=False):
    """Read in a Gaia healpix file in the appropriate format for desitarget.

    Parameters
    ----------
    filename : :class:`str`
        File name of a single Gaia "healpix-" file.
    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return (data, header) instead of just data.
    addobjid : :class:`bool`, optional, defaults to ``False``
        Include, in the output, two additional columns. A column
        "GAIA_OBJID" that is the integer number of each row read from
        file and a column "GAIA_BRICKID" that is the integer number of
        the file itself.

    Returns
    -------
    :class:`~numpy.ndarray`
        Gaia data translated to targeting format (upper-case etc.) with the
        columns corresponding to `desitarget.gaiamatch.gaiadatamodel`

    Notes
    -----
        - A better location for this might be in `desitarget.io`?
    """
    # ADM check for an epic fail on the the version of fitsio.
    check_fitsio_version()

    # ADM prepare to read in the Gaia data by reading in columns.
    fx = fitsio.FITS(filename, upper=True)
    fxcolnames = fx[1].get_colnames()
    hdr = fx[1].read_header()

    # ADM the default list of columns.
    readcolumns = list(ingaiadatamodel.dtype.names)
    # ADM read 'em in.
    outdata = fx[1].read(columns=readcolumns)
    # ADM change the data model to what we want for each column.
    outdata.dtype.names = gaiadatamodel.dtype.names

    # ADM the proper motion ERRORS need to be converted to IVARs.
    # ADM remember to leave 0 entries as 0.
    for col in ['PMRA_IVAR', 'PMDEC_IVAR', 'PARALLAX_IVAR']:
        w = np.where(outdata[col] != 0)[0]
        outdata[col][w] = 1./(outdata[col][w]**2.)

    # ADM if requested, add an object identifier for each file row.
    if addobjid:
        newdt = outdata.dtype.descr
        for tup in ('GAIA_BRICKID', '>i4'), ('GAIA_OBJID', '>i4'):
            newdt.append(tup)
        nobjs = len(outdata)
        newoutdata = np.zeros(nobjs, dtype=newdt)
        for col in outdata.dtype.names:
            newoutdata[col] = outdata[col]
        newoutdata['GAIA_OBJID'] = np.arange(nobjs)
        nside = _get_gaia_nside()
        hpnum = radec2pix(nside, outdata["GAIA_RA"], outdata["GAIA_DEC"])
        # ADM int should fail if HEALPix in the file aren't unique.
        newoutdata['GAIA_BRICKID'] = int(np.unique(hpnum))
        outdata = newoutdata

    # ADM return data from the Gaia file, with the header if requested.
    if header:
        fx.close()
        return outdata, hdr
    else:
        fx.close()
        return outdata


def find_gaia_files(objs, neighbors=True, radec=False):
    """Find full paths to Gaia healpix files for objects by RA/Dec.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Array of objects. Must contain at least the columns "RA" and "DEC".
    neighbors : :class:`bool`, optional, defaults to ``True``
        Also return all neighboring pixels that touch the files of interest
        in order to prevent edge effects (e.g. if a Gaia source is 1 arcsec
        away from a primary source and so in an adjacent pixel).
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` then the passed `objs` is an [RA, Dec] list instead of
        a rec array.

    Returns
    -------
    :class:`list`
        A list of all Gaia files that need to be read in to account for objects
        at the passed locations.

    Notes
    -----
        - The environment variable $GAIA_DIR must be set.
    """
    # ADM the resolution at which the Gaia HEALPix files are stored.
    nside = _get_gaia_nside()

    # ADM check that the GAIA_DIR is set and retrieve it.
    gaiadir = get_gaia_dir()
    hpxdir = os.path.join(gaiadir, 'healpix')

    return io.find_star_files(objs, hpxdir, nside,
                              neighbors=neighbors, radec=radec)


def find_gaia_files_hp(nside, pixlist, neighbors=True):
    """Find full paths to Gaia healpix files in a set of HEALPixels.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    pixlist : :class:`list` or `int`
        A set of HEALPixels at `nside`.
    neighbors : :class:`bool`, optional, defaults to ``True``
        Also return files corresponding to all neighbors that touch the
        pixels in `pixlist` to prevent edge effects (e.g. a Gaia source
        is 1 arcsec outside of `pixlist` and so in an adjacent pixel).

    Returns
    -------
    :class:`list`
        A list of all Gaia files that need to be read in to account for
        objects in the passed list of pixels.

    Notes
    -----
        - The environment variable $GAIA_DIR must be set.
    """
    # ADM the resolution at which the healpix files are stored.
    filenside = _get_gaia_nside()

    # ADM check that the GAIA_DIR is set and retrieve it.
    gaiadir = get_gaia_dir()
    hpxdir = os.path.join(gaiadir, 'healpix')

    # ADM work with pixlist as an array.
    pixlist = np.atleast_1d(pixlist)

    # ADM determine the pixels that touch the passed pixlist.
    pixnum = nside2nside(nside, filenside, pixlist)

    # ADM if neighbors was sent, then retrieve all pixels that touch each
    # ADM pixel covered by the provided locations, to prevent edge effects...
    if neighbors:
        pixnum = add_hp_neighbors(filenside, pixnum)

    # ADM reformat in the Gaia healpix format used by desitarget.
    gaiafiles = [os.path.join(hpxdir, io.hpx_filename(pn)) for pn in pixnum]

    return gaiafiles


def find_gaia_files_box(gaiabounds, neighbors=True):
    """Find full paths to Gaia healpix files in an RA/Dec box.

    Parameters
    ----------
    gaiabounds : :class:`list`
        A region of the sky bounded by RA/Dec. Pass as a 4-entry list to
        represent an area bounded by [RAmin, RAmax, DECmin, DECmax]
    neighbors : :class:`bool`, optional, defaults to ``True``
        Also return files corresponding to all neighboring pixels that touch
        the files that touch the box in order to prevent edge effects (e.g. if a Gaia
        source might be 1 arcsec outside of the box and so in an adjacent pixel)

    Returns
    -------
    :class:`list`
        A list of all Gaia files that need to be read in to account for objects
        in the passed box.

    Notes
    -----
        - Uses the `healpy` routines that rely on `fact`, so the usual
          warnings about returning different pixel sets at different values
          of `fact` apply. See:
          https://healpy.readthedocs.io/en/latest/generated/healpy.query_polygon.html
        - The environment variable $GAIA_DIR must be set.
    """
    # ADM the resolution at which the healpix files are stored.
    nside = _get_gaia_nside()

    # ADM check that the GAIA_DIR is set and retrieve it.
    gaiadir = get_gaia_dir()
    hpxdir = os.path.join(gaiadir, 'healpix')

    # ADM determine the pixels that touch the box.
    pixnum = hp_in_box(nside, gaiabounds, inclusive=True, fact=4)

    # ADM if neighbors was sent, then retrieve all pixels that touch each
    # ADM pixel covered by the provided locations, to prevent edge effects...
    if neighbors:
        pixnum = add_hp_neighbors(nside, pixnum)

    # ADM reformat in the Gaia healpix format used by desitarget.
    gaiafiles = [os.path.join(hpxdir, io.hpx_filename(pn)) for pn in pixnum]

    return gaiafiles


def find_gaia_files_beyond_gal_b(mingalb, neighbors=True):
    """Find full paths to Gaia healpix files beyond a Galactic b.

    Parameters
    ----------
    mingalb : :class:`float`
        Closest latitude to Galactic plane to return HEALPixels
        (e.g. send 10 to limit to pixels beyond -10o <= b < 10o).
    neighbors : :class:`bool`, optional, defaults to ``True``
        Also return files corresponding to neighboring pixels that touch
        in order to prevent edge effects (e.g. if a Gaia source might be
        1 arcsec beyond mingalb and so in an adjacent pixel).

    Returns
    -------
    :class:`list`
        All Gaia files that need to be read in to account for objects
        further from the Galactic plane than `mingalb`.

    Notes
    -----
        - The environment variable $GAIA_DIR must be set.
        - :func:`desitarget.geomask.hp_beyond_gal_b()` is already quite
          inclusive, so you may retrieve some extra files along the
          `mingalb` boundary.
    """
    # ADM the resolution at which the healpix files are stored.
    nside = _get_gaia_nside()

    # ADM check that the GAIA_DIR is set and retrieve it.
    gaiadir = get_gaia_dir()
    hpxdir = os.path.join(gaiadir, 'healpix')

    # ADM determine the pixels beyond mingalb.
    pixnum = hp_beyond_gal_b(nside, mingalb, neighbors=True)

    # ADM if neighbors was sent, retrieve all pixels that touch each
    # ADM retrieved, to prevent edge effects...
    if neighbors:
        pixnum = add_hp_neighbors(nside, pixnum)

    # ADM reformat in the Gaia healpix format used by desitarget.
    gaiafiles = [os.path.join(hpxdir, io.hpx_filename(pn)) for pn in pixnum]

    return gaiafiles


def find_gaia_files_tiles(tiles=None, neighbors=True):
    """
    Parameters
    ----------
    tiles : :class:`~numpy.ndarray`
        Array of tiles, or ``None`` to use all DESI tiles from
        :func:`desimodel.io.load_tiles`.
    neighbors : :class:`bool`, optional, defaults to ``True``
        Also return all neighboring pixels that touch the files of interest
        in order to prevent edge effects (e.g. if a Gaia source is 1 arcsec
        away from a primary source and so in an adjacent pixel).

    Returns
    -------
    :class:`list`
        A list of all Gaia files that touch the passed tiles.

    Notes
    -----
        - The environment variables $GAIA_DIR and $DESIMODEL must be set.
    """
    # ADM check that the DESIMODEL environment variable is set.
    if os.environ.get('DESIMODEL') is None:
        msg = "DESIMODEL environment variable must be set!!!"
        log.critical(msg)
        raise ValueError(msg)

    # ADM the resolution at which the healpix files are stored.
    nside = _get_gaia_nside()

    # ADM check that the GAIA_DIR is set and retrieve it.
    gaiadir = get_gaia_dir()
    hpxdir = os.path.join(gaiadir, 'healpix')

    # ADM determine the pixels that touch the tiles.
    from desimodel.footprint import tiles2pix
    pixnum = tiles2pix(nside, tiles=tiles)

    # ADM if neighbors was sent, then retrieve all pixels that touch each
    # ADM pixel covered by the provided locations, to prevent edge effects...
    if neighbors:
        pixnum = add_hp_neighbors(nside, pixnum)

    # ADM reformat in the Gaia healpix format used by desitarget.
    gaiafiles = [os.path.join(hpxdir, io.hpx_filename(pn)) for pn in pixnum]

    return gaiafiles


def match_gaia_to_primary(objs, matchrad=1., retaingaia=False,
                          gaiabounds=[0., 360., -90., 90.]):
    """Match a set of objects to Gaia healpix files and return the Gaia information.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Must contain at least "RA" and "DEC".
    matchrad : :class:`float`, optional, defaults to 1 arcsec
        The matching radius in arcseconds.
    retaingaia : :class:`float`, optional, defaults to False
        If set, return all of the Gaia information in the "area" occupied by
        the passed objects (whether a Gaia object matches a passed RA/Dec
        or not.) THIS ASSUMES THAT THE PASSED OBJECTS ARE FROM A SWEEPS file
        and that the integer values nearest the maximum and minimum passed RAs
        and Decs fairly represent the areal "edges" of that file.
    gaiabounds : :class:`list`, optional, defaults to the whole sky
        Used in conjunction with `retaingaia` to determine over what area to
        retrieve Gaia objects that don't match a sweeps object. Pass a 4-entry
        list to represent an area bounded by [RAmin, RAmax, DECmin, DECmax]

    Returns
    -------
    :class:`~numpy.ndarray`
        The matching Gaia information for each object, where the returned format and
        columns correspond to `desitarget.gaiamatch.gaiadatamodel`

    Notes
    -----
        - The first len(objs) objects correspond row-by-row to the passed objects.
        - For objects that do NOT have a match in the Gaia files, the "REF_ID"
          column is set to -1, and all other columns are zero.
        - If `retaingaia` is True then objects after the first len(objs) objects are
          Gaia objects that do not have a sweeps match but that are in the area
          bounded by `gaiabounds`
    """
    # ADM I'm getting this old Cython RuntimeWarning on search_around_sky ****:
    # RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
    # ADM but it doesn't seem malicious, so I'm filtering. I think its caused
    # ADM by importing a scipy compiled against an older numpy than is installed
    # ADM e.g. https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
    import warnings

    # ADM if retaingaia is True, retain all Gaia objects in a sweeps-like box.
    if retaingaia:
        ramin, ramax, decmin, decmax = gaiabounds

    # ADM convert the coordinates of the input objects to a SkyCoord object.
    cobjs = SkyCoord(objs["RA"]*u.degree, objs["DEC"]*u.degree)
    nobjs = cobjs.size

    # ADM deal with the special case that only a single object was passed.
    if nobjs == 1:
        return match_gaia_to_primary_single(objs, matchrad=matchrad)

    # ADM set up a zerod array of Gaia information for the passed objects.
    gaiainfo = np.zeros(nobjs, dtype=gaiadatamodel.dtype)

    # ADM a supplemental (zero-length) array to hold Gaia objects that don't
    # ADM match a sweeps object, in case retaingaia was set.
    suppgaiainfo = np.zeros(0, dtype=gaiadatamodel.dtype)

    # ADM objects without matches should have REF_ID of -1.
    gaiainfo['REF_ID'] = -1

    # ADM determine which Gaia files need to be considered.
    if retaingaia:
        gaiafiles = find_gaia_files_box(gaiabounds)
    else:
        gaiafiles = find_gaia_files(objs)

    # ADM loop through the Gaia files and match to the passed objects.
    for file in gaiafiles:
        gaia = read_gaia_file(file)
        cgaia = SkyCoord(gaia["GAIA_RA"]*u.degree, gaia["GAIA_DEC"]*u.degree)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ADM ****here's where the warning occurs...
            idobjs, idgaia, _, _ = cgaia.search_around_sky(cobjs, matchrad*u.arcsec)
        # ADM assign the Gaia info to the array that corresponds to the passed objects.
        gaiainfo[idobjs] = gaia[idgaia]

        # ADM if retaingaia was set, also build an array of Gaia objects that
        # ADM don't have sweeps matches, but are within the RA/Dec bounds.
        if retaingaia:
            # ADM find the Gaia IDs that didn't match the passed objects.
            nomatch = set(np.arange(len(gaia)))-set(idgaia)
            noidgaia = np.array(list(nomatch))
            # ADM which Gaia objects with these IDs are within the bounds.
            if len(noidgaia) > 0:
                suppg = gaia[noidgaia]
                winbounds = np.where(
                    (suppg["GAIA_RA"] >= ramin) & (suppg["GAIA_RA"] < ramax)
                    & (suppg["GAIA_DEC"] >= decmin) & (suppg["GAIA_DEC"] < decmax)
                )[0]
                # ADM Append those Gaia objects to the suppgaiainfo array.
                if len(winbounds) > 0:
                    suppgaiainfo = np.hstack([suppgaiainfo, suppg[winbounds]])

    if retaingaia:
        gaiainfo = np.hstack([gaiainfo, suppgaiainfo])

    return gaiainfo


def match_gaia_to_primary_single(objs, matchrad=1.):
    """Match ONE object to Gaia "chunks" files and return the Gaia information.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Must contain at least "RA" and "DEC". MUST BE A SINGLE ROW.
    matchrad : :class:`float`, optional, defaults to 1 arcsec
        The matching radius in arcseconds.

    Returns
    -------
    :class:`~numpy.ndarray`
        The matching Gaia information for the object, where the returned format and
        columns correspond to `desitarget.secondary.gaiadatamodel`

    Notes
    -----
        - If the object does NOT have a match in the Gaia files, the "REF_ID"
          column is set to -1, and all other columns are zero
    """
    # ADM I'm getting this old Cython RuntimeWarning on search_around_sky ****:
    # RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
    # ADM but it doesn't seem malicious, so I'm filtering. I think its caused
    # ADM by importing a scipy compiled against an older numpy than is installed
    # ADM e.g. https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
    import warnings

    # ADM convert the coordinates of the input objects to a SkyCoord object.
    cobjs = SkyCoord(objs["RA"]*u.degree, objs["DEC"]*u.degree)
    nobjs = cobjs.size
    if nobjs > 1:
        log.error("Only matches one row but {} rows were sent".format(nobjs))

    # ADM set up a zerod array of Gaia information for the passed object.
    gaiainfo = np.zeros(nobjs, dtype=gaiadatamodel.dtype)

    # ADM an object without matches should have REF_ID of -1.
    gaiainfo['REF_ID'] = -1

    # ADM determine which Gaia files need to be considered.
    gaiafiles = find_gaia_files(objs)

    # ADM loop through the Gaia files and match to the passed object.
    for file in gaiafiles:
        gaia = read_gaia_file(file)
        cgaia = SkyCoord(gaia["GAIA_RA"]*u.degree, gaia["GAIA_DEC"]*u.degree)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ADM ****here's where the warning occurs...
            sep = cobjs.separation(cgaia)
            idgaia = np.where(sep < matchrad*u.arcsec)[0]
        # ADM assign the Gaia info to the array that corresponds to the passed object.
        if len(idgaia) > 0:
            gaiainfo = gaia[idgaia]

    return gaiainfo


def write_gaia_matches(infiles, numproc=4, outdir="."):
    """Match sweeps files to Gaia and rewrite with the Gaia columns added

    Parameters
    ----------
    infiles : :class:`list` or `str`
        A list of input filenames (sweep files) OR a single filename.
        Arrays in the files must contain at least the columns "RA" and "DEC".
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    outdir : :class:`str`, optional, default to the current directory
        The directory to write the files.

    Returns
    -------
    :class:`~numpy.ndarray`
        The original sweeps files with the columns in `gaiadatamodel`
        added (except for the columns `GAIA_RA` and `GAIA_DEC`) are
        written to file. The filename is the same as the input
        filename with the ".fits" replaced by "-gaia$DRmatch.fits"
        where $DR is extracted from the $GAIA_DIR environment variable.

    Notes
    -----
        - if numproc==1, use the serial code instead of the parallel code.
        - The environment variable $GAIA_DIR must be set.
    """
    # ADM check that the GAIA_DIR is set and retrieve it.
    gaiadir = get_gaia_dir()

    # ADM convert a single file, if passed to a list of files.
    if isinstance(infiles, str):
        infiles = [infiles, ]

    # ADM check that files exist before proceeding.
    for filename in infiles:
        if not os.path.exists(filename):
            raise ValueError("{} doesn't exist".format(filename))

    nfiles = len(infiles)

    # ADM extract a reasonable name for output files from the Gaia directory.
    drloc = gaiadir.find("dr")
    # ADM if we didn't find the substring "dr" go generic.
    if drloc == -1:
        ender = '-gaiamatch.fits'
    else:
        ender = '-gaia{}match.fits'.format(gaiadir[drloc:drloc+3])

    # ADM the critical function to run on every file.
    def _get_gaia_matches(fnwdir):
        '''wrapper on match_gaia_to_primary() given a file name'''
        # ADM extract the output file name.
        fn = os.path.basename(fnwdir)
        outfile = '{}/{}'.format(outdir, fn.replace(".fits", ender))

        # ADM read in the objects.
        objs, hdr = io.read_tractor(fnwdir, header=True)

        # ADM match to Gaia sources.
        gaiainfo = match_gaia_to_primary(objs)
        log.info('Done with Gaia match for {} primary objects...t = {:.1f}s'
                 .format(len(objs), time()-start))

        # ADM remove the GAIA_RA, GAIA_DEC columns as they aren't
        # ADM in the imaging surveys data model.
        gaiainfo = pop_gaia_coords(gaiainfo)

        # ADM add the Gaia column information to the sweeps array.
        for col in gaiainfo.dtype.names:
            objs[col] = gaiainfo[col]

        fitsio.write(outfile, objs, extname='SWEEP', header=hdr, clobber=True)
        return True

    # ADM this is just to count sweeps files in _update_status.
    nfile = np.zeros((), dtype='i8')

    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if nfile % 50 == 0 and nfile > 0:
            rate = nfile / (time() - t0)
            log.info('{}/{} files; {:.1f} files/sec'.format(nfile, nfiles, rate))
        nfile[...] += 1    # this is an in-place modification.
        return result

    # - Parallel process input files.
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            _ = pool.map(_get_gaia_matches, infiles, reduce=_update_status)
    else:
        for file in infiles:
            _ = _update_status(_get_gaia_matches(file))

    return
