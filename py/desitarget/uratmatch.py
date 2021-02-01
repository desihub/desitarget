# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.uratmatch
====================

Useful `URAT`_ matching and manipulation routines.

.. _`URAT`: http://cdsarc.u-strasbg.fr/viz-bin/cat/I/329
"""
import os
import numpy as np
import fitsio
import requests
import pickle

from pkg_resources import resource_filename
from time import time
from astropy.io import ascii
from glob import glob
import healpy as hp

from desitarget.internal import sharedmem
from desimodel.footprint import radec2pix
from desitarget.geomask import add_hp_neighbors, radec_match_to
from desitarget import io

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()

# ADM columns contained in our version of the URAT fits files.
uratdatamodel = np.array([], dtype=[
    ('URAT_ID', '>i8'), ('RA', '>f8'), ('DEC', '>f8'),
    ('APASS_G_MAG', '>f4'), ('APASS_G_MAG_ERROR', '>f4'),
    ('APASS_R_MAG', '>f4'), ('APASS_R_MAG_ERROR', '>f4'),
    ('APASS_I_MAG', '>f4'), ('APASS_I_MAG_ERROR', '>f4'),
    ('PMRA', '>f4'), ('PMDEC', '>f4'), ('PM_ERROR', '>f4')
])


def get_urat_dir():
    """Convenience function to grab the URAT environment variable.

    Returns
    -------
    :class:`str`
        The directory stored in the $URAT_DIR environment variable.
    """
    # ADM check that the $URAT_DIR environment variable is set.
    uratdir = os.environ.get('URAT_DIR')
    if uratdir is None:
        msg = "Set $URAT_DIR environment variable!"
        log.critical(msg)
        raise ValueError(msg)

    return uratdir


def _get_urat_nside():
    """Grab the HEALPixel nside to be used throughout this module.

    Returns
    -------
    :class:`int`
        The HEALPixel nside number for URAT file creation and retrieval.
    """
    nside = 32

    return nside


def scrape_urat(url="http://cdsarc.u-strasbg.fr/ftp/I/329/URAT1/v12/",
                nfiletest=None):
    """Retrieve the binary versions of the URAT files.

    Parameters
    ----------
    url : :class:`str`
        The web directory that hosts the archived binary URAT files.
    nfiletest : :class:`int`, optional, defaults to ``None``
        If an integer is sent, only retrieve this number of files, for testing.

    Returns
    -------
    Nothing
        But the archived URAT files are written to $URAT_DIR/binary.

    Notes
    -----
        - The environment variable $URAT_DIR must be set.
        - Runs in about 50 minutes for 575 URAT files.
    """
    # ADM check that the URAT_DIR is set and retrieve it.
    uratdir = get_urat_dir()

    # ADM construct the directory to which to write files.
    bindir = os.path.join(uratdir, 'binary')
    # ADM the directory better be empty for the wget!
    if os.path.exists(bindir):
        if len(os.listdir(bindir)) > 0:
            msg = "{} should be empty to wget URAT binary files!".format(bindir)
            log.critical(msg)
            raise ValueError(msg)
    # ADM make the directory, if needed.
    else:
        log.info('Making URAT directory for storing binary files')
        os.makedirs(bindir)

    index = requests.get(url)

    # ADM retrieve any file name that starts with z.
    # ADM the [1::2] pulls back just the odd lines from the split list.
    garbled = index.text.split("z")[1::2]
    filelist = ["z{}".format(g[:3]) for g in garbled]

    # ADM if nfiletest was passed, just work with that number of files.
    test = nfiletest is not None
    if test:
        filelist = filelist[:nfiletest]
    nfiles = len(filelist)

    # ADM loop through the filelist.
    start = time()
    for nfile, fileinfo in enumerate(filelist):
        # ADM make the wget command to retrieve the file and issue it.
        cmd = 'wget -q {} -P {}'.format(os.path.join(url, fileinfo), bindir)
        print(cmd)
        os.system(cmd)
        if nfile % 25 == 0 or test:
            elapsed = time() - start
            rate = nfile / elapsed
            log.info(
                '{}/{} files; {:.1f} files/sec; {:.1f} total mins elapsed'
                .format(nfile+1, nfiles, rate, elapsed/60.)
            )

    log.info('Done...t={:.1f}s'.format(time()-start))

    return


def urat_binary_to_csv():
    """Convert files in $URAT_DIR/binary to files in $URAT_DIR/csv.

    Returns
    -------
    Nothing
        But the archived URAT binary files in $URAT_DIR/binary are
        converted to CSV files in the $URAT_DIR/csv.

    Notes
    -----
        - The environment variable $URAT_DIR must be set.
        - Relies on the executable urat/fortran/v1dump, which is only
          tested at NERSC and might need compiled by the user.
        - Runs in about 40 minutes for 575 files.
    """
    # ADM check that the URAT_DIR is set.
    uratdir = get_urat_dir()

    # ADM a quick check that the csv directory is empty before writing.
    csvdir = os.path.join(uratdir, 'csv')
    if os.path.exists(csvdir):
        if len(os.listdir(csvdir)) > 0:
            msg = "{} should be empty to make URAT files!".format(csvdir)
            log.critical(msg)
            raise ValueError(msg)
    # ADM make the directory, if needed.
    else:
        log.info('Making URAT directory for storing CSV files')
        os.makedirs(csvdir)

    log.info('Begin converting URAT files to CSV...t={:.1f}s'
             .format(time()-start))

    # ADM check the v1dump executable has been compiled.
    readme = resource_filename('desitarget', 'urat/fortran/README')
    cmd = resource_filename('desitarget', 'urat/fortran/v1dump')
    if not (os.path.exists(cmd) and os.access(cmd, os.X_OK)):
        msg = "{} must have been compiled (see {})".format(cmd, readme)
        log.critical(msg)
        raise ValueError(msg)

    # ADM execute v1dump.
    os.system(cmd)

    log.info('Done...t={:.1f}s'.format(time()-start))

    return


def urat_csv_to_fits(numproc=5):
    """Convert files in $URAT_DIR/csv to files in $URAT_DIR/fits.

    Parameters
    ----------
    numproc : :class:`int`, optional, defaults to 5
        The number of parallel processes to use.

    Returns
    -------
    Nothing
        But the archived URAT CSV files in $URAT_DIR/csv are converted
        to FITS files in the directory $URAT_DIR/fits. Also, a look-up
        table is written to $URAT_DIR/fits/hpx-to-files.pickle for which
        each index is an nside=_get_urat_nside(), nested scheme HEALPixel
        and each entry is a list of the FITS files that touch that HEAPixel.

    Notes
    -----
        - The environment variable $URAT_DIR must be set.
        - if numproc==1, use the serial code instead of the parallel code.
        - Runs in about 10 minutes with numproc=25 for 575 files.
    """
    # ADM the resolution at which the URAT HEALPix files should be stored.
    nside = _get_urat_nside()

    # ADM check that the URAT_DIR is set.
    uratdir = get_urat_dir()
    log.info("running on {} processors".format(numproc))

    # ADM construct the directories for reading/writing files.
    csvdir = os.path.join(uratdir, 'csv')
    fitsdir = os.path.join(uratdir, 'fits')

    # ADM make sure the output directory is empty.
    if os.path.exists(fitsdir):
        if len(os.listdir(fitsdir)) > 0:
            msg = "{} should be empty to make URAT FITS files!".format(fitsdir)
            log.critical(msg)
            raise ValueError(msg)
    # ADM make the output directory, if needed.
    else:
        log.info('Making URAT directory for storing FITS files')
        os.makedirs(fitsdir)

    # ADM construct the list of input files.
    infiles = glob("{}/*csv*".format(csvdir))
    nfiles = len(infiles)

    # ADM the critical function to run on every file.
    def _write_urat_fits(infile):
        """read an input name for a csv file and write it to FITS"""
        outbase = os.path.basename(infile)
        outfilename = "{}.fits".format(outbase.split(".")[0])
        outfile = os.path.join(fitsdir, outfilename)
        # ADM astropy understands without specifying format='csv'.
        fitstable = ascii.read(infile)

        # ADM map the ascii-read csv to typical DESI quantities.
        nobjs = len(fitstable)
        done = np.zeros(nobjs, dtype=uratdatamodel.dtype)
        # ADM have to do this one-by-one, given the format.
        done["RA"] = fitstable['col1']/1000./3600.
        done["DEC"] = fitstable['col2']/1000./3600. - 90.
        done["PMRA"] = fitstable['col16']/10.
        done["PMDEC"] = fitstable['col17']/10.
        done["PM_ERROR"] = fitstable['col18']/10.
        done["APASS_G_MAG"] = fitstable['col36']/1000.
        done["APASS_R_MAG"] = fitstable['col37']/1000.
        done["APASS_I_MAG"] = fitstable['col38']/1000.
        done["APASS_G_MAG_ERROR"] = fitstable['col41']/1000.
        done["APASS_R_MAG_ERROR"] = fitstable['col42']/1000.
        done["APASS_I_MAG_ERROR"] = fitstable['col43']/1000.
        done["URAT_ID"] = fitstable['col46']

        fitsio.write(outfile, done, extname='URATFITS')

        # ADM return the HEALPixels that this file touches.
        pix = set(radec2pix(nside, done["RA"], done["DEC"]))
        return [pix, os.path.basename(outfile)]

    # ADM this is just to count processed files in _update_status.
    nfile = np.zeros((), dtype='i8')
    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if nfile % 25 == 0 and nfile > 0:
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
            pixinfile = pool.map(_write_urat_fits, infiles, reduce=_update_status)
    # ADM ...or run in serial.
    else:
        pixinfile = list()
        for file in infiles:
            pixinfile.append(_update_status(_write_urat_fits(file)))

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


def urat_fits_to_healpix(numproc=5):
    """Convert files in $URAT_DIR/fits to files in $URAT_DIR/healpix.

    Parameters
    ----------
    numproc : :class:`int`, optional, defaults to 5
        The number of parallel processes to use.

    Returns
    -------
    Nothing
        But the archived URAT FITS files in $URAT_DIR/fits are
        rearranged by HEALPixel in the directory $URAT_DIR/healpix.
        The HEALPixel sense is nested with nside=_get_urat_nside(), and
        each file in $URAT_DIR/healpix is called healpix-xxxxx.fits,
        where xxxxx corresponds to the HEALPixel number.

    Notes
    -----
        - The environment variable $URAT_DIR must be set.
        - if numproc==1, use the serial code instead of the parallel code.
        - Runs in about 10 minutes with numproc=25.
    """
    # ADM the resolution at which the URAT HEALPix files should be stored.
    nside = _get_urat_nside()

    # ADM check that the URAT_DIR is set.
    uratdir = get_urat_dir()

    # ADM construct the directories for reading/writing files.
    fitsdir = os.path.join(uratdir, 'fits')
    hpxdir = os.path.join(uratdir, 'healpix')

    # ADM make sure the output directory is empty.
    if os.path.exists(hpxdir):
        if len(os.listdir(hpxdir)) > 0:
            msg = "{} should be empty to make URAT HEALPix files!".format(hpxdir)
            log.critical(msg)
            raise ValueError(msg)
    # ADM make the output directory, if needed.
    else:
        log.info('Making URAT directory for storing HEALPix files')
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
            fitsio.write(outfile, done, extname='URATHPX', header=hdr)

        return

    # ADM this is just to count processed files in _update_status.
    npix = np.zeros((), dtype='i8')
    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if npix % 500 == 0 and npix > 0:
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


def make_urat_files(numproc=5, download=False):
    """Make the HEALPix-split URAT files in one fell swoop.

    Parameters
    ----------
    numproc : :class:`int`, optional, defaults to 5
        The number of parallel processes to use.
    download : :class:`bool`, optional, defaults to ``False``
        If ``True`` then wget the URAT binary files from Vizier.

    Returns
    -------
    Nothing
        But produces:
        - URAT DR1 binary files in $URAT_DIR/binary (if download=True).
        - URAT CSV files with all URAT columns in $URAT_DIR/csv.
        - FITS files with columns from `uratdatamodel` in $URAT_DIR/fits.
        - FITS files reorganized by HEALPixel in $URAT_DIR/healpix.

        The HEALPixel sense is nested with nside=_get_urat_nside(), and
        each file in $URAT_DIR/healpix is called healpix-xxxxx.fits,
        where xxxxx corresponds to the HEALPixel number.

    Notes
    -----
        - The environment variable $URAT_DIR must be set.
        - if numproc==1, use the serial, instead of the parallel, code.
        - Runs in about 2 hours with numproc=25 if download is ``True``.
        - Runs in about 1 hour with numproc=25 if download is ``False``.
    """
    t0 = time()
    log.info('Begin making URAT files...t={:.1f}s'.format(time()-t0))

    # ADM check that the URAT_DIR is set.
    uratdir = get_urat_dir()

    # ADM a quick check that the fits and healpix directories are empty
    # ADM before embarking on the slower parts of the code.
    csvdir = os.path.join(uratdir, 'csv')
    fitsdir = os.path.join(uratdir, 'fits')
    hpxdir = os.path.join(uratdir, 'healpix')
    for direc in [csvdir, fitsdir, hpxdir]:
        if os.path.exists(direc):
            if len(os.listdir(direc)) > 0:
                msg = "{} should be empty to make URAT files!".format(direc)
                log.critical(msg)
                raise ValueError(msg)

    if download:
        scrape_urat()
        log.info('Retrieved URAT files from Vizier...t={:.1f}s'
                 .format(time()-t0))

    urat_binary_to_csv()
    log.info('Converted binary files to CSV...t={:.1f}s'.format(time()-t0))

    urat_csv_to_fits(numproc=numproc)
    log.info('Converted CSV files to FITS...t={:.1f}s'.format(time()-t0))

    urat_fits_to_healpix(numproc=numproc)
    log.info('Rearranged FITS files by HEALPixel...t={:.1f}s'.format(time()-t0))

    return


def find_urat_files(objs, neighbors=True, radec=False):
    """Find full paths to URAT healpix files for objects by RA/Dec.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Array of objects. Must contain the columns "RA" and "DEC".
    neighbors : :class:`bool`, optional, defaults to ``True``
        Also return all pixels that touch the files of interest
        to prevent edge effects (e.g. if a URAT source is 1 arcsec
        away from a primary source and so in an adjacent pixel).
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` then the passed `objs` is an [RA, Dec] list
        instead of a rec array that contains "RA" and "DEC".

    Returns
    -------
    :class:`list`
        A list of all URAT files to read to account for objects at
        the passed locations.

    Notes
    -----
        - The environment variable $URAT_DIR must be set.
    """
    # ADM the resolution at which the URAT HEALPix files are stored.
    nside = _get_urat_nside()

    # ADM check that the URAT_DIR is set and retrieve it.
    uratdir = get_urat_dir()
    hpxdir = os.path.join(uratdir, 'healpix')

    # ADM remember to pass "strict", as URAT doesn't cover the whole sky.
    return io.find_star_files(objs, hpxdir, nside, strict=True,
                              neighbors=neighbors, radec=radec)


def match_to_urat(objs, matchrad=1., radec=False):
    """Match objects to URAT healpix files and return URAT information.

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
        The matching URAT information for each object. The returned
        format is as for desitarget.uratmatch.uratdatamodel with
        an extra column "URAT_SEP" which is the matching distance
        in ARCSECONDS.

    Notes
    -----
        - For objects that do NOT have a match in URAT, the "URAT_ID"
          and "URAT_SEP" columns are -1, and other columns are zero.
        - Retrieves the CLOSEST match to URAT for each passed object.
        - Because this reads in HEALPixel split files, it's (far) faster
          for objects that are clumped rather than widely distributed.
    """
    # ADM parse whether a structure or coordinate list was passed.
    if radec:
        ra, dec = objs
    else:
        ra, dec = objs["RA"], objs["DEC"]

    # ADM set up an array of URAT information for the output.
    nobjs = len(ra)
    done = np.zeros(nobjs, dtype=uratdatamodel.dtype)

    # ADM objects without matches should have URAT_ID, URAT_SEP of -1.
    done["URAT_ID"] = -1
    urat_sep = np.zeros(nobjs) - 1

    # ADM determine which URAT files need to be scraped.
    uratfiles = find_urat_files([ra, dec], radec=True)
    nfiles = len(uratfiles)

    # ADM catch the case of no matches to URAT.
    if nfiles > 0:
        # ADM loop through the URAT files and find matches.
        for ifn, fn in enumerate(uratfiles):
            if ifn % 500 == 0 and ifn > 0:
                log.info('{}/{} files; {:.1f} total mins elapsed'
                         .format(ifn, nfiles, (time()-start)/60.))
            urat = fitsio.read(fn)
            idurat, idobjs, dist = radec_match_to(
                [urat["RA"], urat["DEC"]], [ra, dec],
                sep=matchrad, radec=True, return_sep=True)

            # ADM update matches whenever we have a CLOSER match.
            ii = (urat_sep[idobjs] == -1) | (urat_sep[idobjs] > dist)
            done[idobjs[ii]] = urat[idurat[ii]]
            urat_sep[idobjs[ii]] = dist[ii]

    # ADM add the separation distances to the output array.
    dt = uratdatamodel.dtype.descr + [("URAT_SEP", ">f4")]
    output = np.zeros(nobjs, dtype=dt)
    for col in uratdatamodel.dtype.names:
        output[col] = done[col]
    output["URAT_SEP"] = urat_sep

    return output
