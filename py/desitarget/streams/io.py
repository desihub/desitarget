"""
desitarget.streams.io
=====================

Reading/writing data for the MWS Stellar Stream programs.
"""
from time import time

import os
import fitsio
import numpy as np
import healpy as hp
import astropy.coordinates as acoo
import astropy.units as auni

from desitarget import io
from desitarget.geomask import pixarea2nside, add_hp_neighbors, sweep_files_touch_hp
from desitarget.gaiamatch import match_gaia_to_primary_post_dr3
from desitarget.targets import resolve
from desitarget.streams.utilities import betw, ivars_to_errors
from desitarget.internal import sharedmem

from desiutil import depend

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()

# ADM the Legacy Surveys part of the data model for working with streams.
streamcolsLS = np.array([], dtype=[
    ('RELEASE', '>i2'), ('BRICKID', '>i4'), ('TYPE', 'S4'),
    ('OBJID', '>i4'), ('RA', '>f8'), ('DEC', '>f8'), ('EBV', '>f4'),
    ('FLUX_G', '>f4'), ('FIBERTOTFLUX_G', '>f4'),
    ('FLUX_R', '>f4'), ('FIBERTOTFLUX_R', '>f4'),
    ('FLUX_Z', '>f4'), ('FIBERTOTFLUX_Z', '>f4'),
])

# ADM the Gaia part of the data model for working with streams.
streamcolsGaia = np.array([], dtype=[
    ('REF_EPOCH', '>f4'),  ('REF_ID', '>i8'),
    ('PARALLAX', '>f4'), ('PARALLAX_ERROR', '>f4'),
    ('PMRA', '>f4'), ('PMRA_ERROR', '>f4'),
    ('PMDEC', '>f4'), ('PMDEC_ERROR', '>f4'),
    ('ASTROMETRIC_PARAMS_SOLVED', '>i1'), ('NU_EFF_USED_IN_ASTROMETRY', '>f4'),
    ('PSEUDOCOLOUR', '>f4'), ('ECL_LAT', '>f8'), ('PHOT_G_MEAN_MAG', '>f4'),
    ('PHOT_BP_MEAN_MAG', '>f4'), ('PHOT_RP_MEAN_MAG', '>f4')
])

# ADM the Gaia Data Release for matching throughout this module.
gaiadr = "dr3"


def read_data_per_stream_one_file(filename, rapol, decpol, mind, maxd):
    """Assemble the data needed for a stream program from one file

    Parameters
    ----------
    swdir : :class:`str`
        Name of a Legacy Surveys sweep file.
    rapol, decpol : :class:`float`
        Pole in the stream coordinate system in DEGREES.
    mind, maxd : :class:`float` or `int`
        Minimum and maximum angular distance from the pole of the stream
        coordinate system to search for members in DEGREES.

    Returns
    -------
    :class:`array_like`
        An array of objects from the filename that are in the stream,
        with matched Gaia information.
    """
    objs = io.read_tractor(filename)

    # ADM codinates of the stream.
    cstream = acoo.SkyCoord(rapol*auni.degree, decpol*auni.degree)
    cobjs = acoo.SkyCoord(objs["RA"]*auni.degree, objs["DEC"]*auni.degree)

    # ADM separation between the objects of interest and the stream.
    sep = cobjs.separation(cstream)

    # ADM only retain objects in the stream...
    ii = betw(sep.value, mind, maxd)

    # ADM ...at a declination of > -20o...
    ii &= objs["DEC"] > -20.

    # ADM ...that aren't very faint (> 22.5 mag in r).
    ii &= objs["FLUX_R"] > 1
    # ADM Also guard against negative fluxes in g/r.
    # ii &= objs["FLUX_G"] > 0.
    # ii &= objs["FLUX_Z"] > 0.

    objs = objs[ii]

    # ADM limit to northern objects in northern imaging and southern
    # ADM objects in southern imaging.
    LSobjs = resolve(objs)

    # ADM set up an an output array, and only retain critical columns
    # ADM from the global data model.
    data = np.zeros(len(LSobjs), dtype=streamcolsLS.dtype.descr +
                    streamcolsGaia.dtype.descr)

    # ADM catch the case where there are no objects meeting the cuts.
    if len(LSobjs) > 0:
        gaiaobjs = match_gaia_to_primary_post_dr3(LSobjs, matchrad=1., dr=gaiadr)

        # ADM to try and better resemble Gaia data, set zero
        # ADM magnitudes and proper motions to NaNs and change
        # ADM IVARs back to errors.
        gaiaobjs = ivars_to_errors(
            gaiaobjs, colnames=["PARALLAX_IVAR", "PMRA_IVAR", "PMDEC_IVAR"])

        for col in ["PHOT_G_MEAN_MAG", "PHOT_BP_MEAN_MAG", "PHOT_RP_MEAN_MAG"
                    , "PARALLAX", "PMRA", "PMDEC"]:
            ii = gaiaobjs[col] < 1e-16
            ii &= gaiaobjs[col] > -1e-16
            gaiaobjs[col][ii] = np.nan

        # ADM a (probably unnecessary) sanity check.
        assert(len(gaiaobjs) == len(LSobjs))

        # ADM add data for the Legacy Surveys columns.
        for col in streamcolsLS.dtype.names:
            data[col] = LSobjs[col]
        # ADM add data for the Gaia columns.
        for col in streamcolsGaia.dtype.names:
            data[col] = gaiaobjs[col]

    return data


def read_data_per_stream(swdir, rapol, decpol, mind, maxd, stream_name,
                         readcache=True, addnors=True, test=False, numproc=1):
    """Assemble the data needed for a particular stream program.

    Parameters
    ----------
    swdir : :class:`str`
        Root directory of Legacy Surveys sweep files for a given data
        release for ONE of EITHER north or south, e.g.
        "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0".
    rapol, decpol : :class:`float`
        Pole in the stream coordinate system in DEGREES.
    mind, maxd : :class:`float` or `int`
        Minimum and maximum angular distance from the pole of the stream
        coordinate system to search for members in DEGREES.
    stream_name : :class:`str`
        Name of a stream. Used to make the cached filename, e.g. "GD1".
    readcache : :class:`bool`
        If ``True`` read from a previously constructed and cached file
        automatically, IF such a file exists. If ``False`` don't read
        from the cache AND OVERWRITE the cached file, if it exists. The
        cached file is $TARG_DIR/streamcache/streamname-drX-cache.fits,
        where streamname is the lower-case passed `stream_name` and drX
        is the Legacy Surveys Data Release (parsed from `swdir`).
    addnors : :class:`bool`
        If ``True`` then if `swdir` contains "north" add sweep files from
        the south by substituting "south" in place of "north" (and vice
        versa, i.e. if `swdir` contains "south" add sweep files from the
        north by substituting "north" in place of "south").
    test : :class:`bool`
        Read a subset of the data for testing purposes.
    numproc : :class:`int`, optional, defaults to 1 for serial
        The number of parallel processes to use. `numproc` of 16 is a
        good balance between speed and file I/O.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for stream members.

    Notes
    -----
    - Example values for, e.g., GD1:
        swdir = "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0"
        rapol, decpol = 34.5987, 29.7331
        mind, maxd = 80, 100
    - The $TARG_DIR environment variable must be set to read/write from
      a cache. If $TARG_DIR is not set, caching is completely ignored.
    - This is useful for a single stream. The :func:`~read_data` function
      is likely a better choice for looping over the entire LS sweeps
      data when targeting multiple streams.
    """
    # ADM start the clock.
    start = time()

    # ADM check whether $TARG_DIR exists. If it does, agree to read from
    # ADM and write to the cache.
    writecache = True
    targdir = os.environ.get("TARG_DIR")
    if targdir is None:
        msg = "Set $TARG_DIR environment variable to use the cache!"
        log.info(msg)
        readcache = False
        writecache = False
    else:
        # ADM retrieve the data release from the passed sweep directory.
        dr = [i for i in swdir.split(os.sep) if "dr" in i]
        # ADM fail if this doesn't look like a standard sweep directory.
        if len(dr) != 1:
            msg = 'swdir not parsed: should include a construction like '
            msg += '"dr9" or "dr10"'
            raise ValueError(msg)
        cachefile = os.path.join(os.getenv("TARG_DIR"), "streamcache",
                                 f"{stream_name.lower()}-{dr[0]}-cache.fits")

    # ADM if we have a cache, read it if requested and return the data.
    if readcache:
        if os.path.isfile(cachefile):
            objs = fitsio.read(cachefile, ext="STREAMCACHE")
            msg = f"Read {len(objs)} objects from {cachefile} cache file"
            log.info(msg)
            return objs
        else:
            msg = f"{cachefile} cache file doesn't exist. "
            msg += f"Proceeding as if readcache=False"
            log.info(msg)

    # ADM read in the sweep files.
    infiles = io.list_sweepfiles(swdir)

    # ADM read both the north and south directories, if requested.
    if addnors:
        if "south" in swdir:
            swdir2 = swdir.replace("south", "north")
        elif "north" in swdir:
            swdir2 = swdir.replace("north", "south")
        else:
            msg = "addnors passed but swdir does not contain north or south!"
            raise ValueError(msg)
        infiles += io.list_sweepfiles(swdir2)

    # ADM calculate nside for HEALPixel of approximately 1o to limit
    # ADM number of sweeps files that need to be read.
    nside = pixarea2nside(1)

    # ADM determine RA, Dec of all HEALPixels at this nside.
    allpix = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, allpix, nest=True)
    ra, dec = np.degrees(phi), 90-np.degrees(theta)

    # ADM only retain HEALPixels in the stream, based on mind and maxd.
    cpix = acoo.SkyCoord(ra*auni.degree, dec*auni.degree)
    cstream = acoo.SkyCoord(rapol*auni.degree, decpol*auni.degree)
    sep = cpix.separation(cstream)
    ii = betw(sep.value, mind, maxd)
    pixlist = allpix[ii]

    # ADM pad with neighboring pixels to ensure stream is fully covered.
    newpixlist = add_hp_neighbors(nside, pixlist)

    # ADM determine which sweep files touch the relevant HEALPixels.
    filesperpixel, _, _ = sweep_files_touch_hp(nside, pixlist, infiles)
    infiles = list(np.unique(np.hstack([filesperpixel[pix] for pix in pixlist])))

    # ADM read a subset of the data for testing purposes, if requested.
    if test:
        msg = "Limiting data to first 20 files for testing purposes"
        log.info(msg)
        infiles = infiles[:20]

    def _read_data_per_stream_one_file(filename):
        """Determine the stream objects for a single sweep file"""
        return read_data_per_stream_one_file(filename, rapol, decpol, mind, maxd)

    nbrick = np.zeros((), dtype='i8')
    t0 = time()
    def _update_status(result):
        """wrapper for critical reduction operation on main parallel process"""
        if nbrick % 5 == 0 and nbrick > 0:
            elapsed = time() - t0
            rate = elapsed / nbrick
            log.info('{}/{} files; {:.1f} secs/file; {:.1f} total mins elapsed'
                     .format(nbrick, len(infiles), rate, elapsed/60.))

        nbrick[...] += 1
        return result

    # ADM parallel process sweep files, limit to objects in the stream.
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            allobjs = pool.map(_read_data_per_stream_one_file, infiles,
                               reduce=_update_status)
    else:
        allobjs = list()
        for fn in infiles:
            allobjs.append(_update_status(_read_data_per_stream_one_file(fn)))


    # ADM assemble all of the relevant objects.
    allobjs = np.concatenate(allobjs)
    log.info(f"Found {len(allobjs)} total objects...t={time()-start:.1f}s")

    # ADM if cache was passed and $TARG_DIR was set then write the data.
    if writecache:
        # ADM if the file doesn't exist we may need to make the directory.
        log.info(f"Writing cache to {cachefile}...t={time()-start:.1f}s")
        os.makedirs(os.path.dirname(cachefile), exist_ok=True)
        # ADM at least add the Gaia DR used to the header.
        hdr = fitsio.FITSHDR()
        hdr.add_record(dict(name="GAIADR", value=gaiadr,
                            comment="GAIA Data Release matched to"))
        io.write_with_units(cachefile, allobjs,
                            header=hdr, extname="STREAMCACHE")

    return allobjs


def write_targets(dirname, targs, header, streamnames="", obscon=None,
                  subpriority=True):
    """Write stream targets to a FITS file.

    Parameters
    ----------
    dirname : :class:`str`
        The output directory name. Filenames are constructed from other
        inputs.
    targs : :class:`~numpy.ndarray`
        The numpy structured array of data to write.
    header : :class:`dict`
        Header for output file. Can be a FITShdr object or dictionary.
        Pass {} if you have no additional header information.
    streamnames : :class:`str, optional
        Information about stream names that correspond to the targets.
        Included in the output filename.
    obscon : :class:`str`, optional, defaults to `None`
        Can pass one of "DARK" or "BRIGHT". If passed, don't write the
        full set of data, rather only write targets appropriate for
        "DARK" or "BRIGHT" observing conditions. The relevant
        `PRIORITY_INIT` and `NUMOBS_INIT` columns will be derived from
        `PRIORITY_INIT_DARK`, etc. and `filename` will have "bright" or
        "dark" appended to the lowest DIRECTORY in the input `filename`.
   subpriority : :class:`bool`, optional, defaults to ``True``
        If ``True`` and a `SUBPRIORITY` column is in the input `targs`,
        then `SUBPRIORITY==0.0` entries are overwritten by a random float
        in the range 0 to 1, using a seed of 816.

    Returns
    -------
    :class:`int`
        The number of targets that were written to file.
    :class:`str`
        The name of the file to which targets were written.

    Notes
    -----
    - Must contain at least the columns:
        PHOT_G_MEAN_MAG, PHOT_BP_MEAN_MAG, PHOT_RP_MEAN_MAG and
        FIBERTOTFLUX_G, FIBERTOTFLUX_R, FIBERTOTFLUX_Z, RELEASE
    - Always OVERWRITES existing files!
    - Writes atomically. Any output files that died mid-write will be
      appended by ".tmp".
    - Units are automatically added from the desitarget units yaml file
      (see `/data/units.yaml`).
    - Mostly wraps :func:`~desitarget.io.write_with_units`.
    """
    # ADM limit to just BRIGHT or DARK targets, if requested.
    # ADM Ignore the filename output, we'll build that on-the-fly.
    if obscon is not None:
        _, header, targs = io._bright_or_dark(dirname, header, targs, obscon)

    # ADM construct the output filename.
    drs = list(set(targs["RELEASE"]//1000))
    if len(drs) == 1:
        drint = drs[0]
        drstr = f"dr{drint}"
    else:
        log.info("Couldn't parse LS data release. Defaulting to drX.")
        drint = "X"
        drstr = "drX"
    outfn = f"streamtargets-{streamnames.lower()}-bright.fits"
    outfn = os.path.join(dirname, drstr, io.desitarget_version,
                         "streamtargets", "main", "resolve", "bright", outfn)

    # ADM check if any targets are too bright.
    maglim = 15
    fluxlim = 10**((22.5-maglim)/2.5)
    toobright = np.zeros(len(targs), dtype="?")
    for col in ["PHOT_G_MEAN_MAG", "PHOT_BP_MEAN_MAG", "PHOT_RP_MEAN_MAG"]:
        toobright |= (targs[col] != 0) & (targs[col] < maglim)
    for col in ["FIBERTOTFLUX_G", "FIBERTOTFLUX_R", "FIBERTOTFLUX_Z"]:
        toobright |= (targs[col] != 0) & (targs[col] > fluxlim)
    if np.any(toobright):
        tids = targs["TARGETID"][toobright]
        log.warning(f"Targets TOO BRIGHT to be written to {outfn}: {tids}")
        # ADM remove the targets that are too bright.
        targs = targs[~toobright]

    # ADM populate SUBPRIORITY with a reproducible random float.
    if "SUBPRIORITY" in targs.dtype.names and subpriority:
        subpseed = 816
        np.random.seed(subpseed)
        # SB only set subpriorities that aren't already set, but keep
        # original full random sequence order.
        ii = targs["SUBPRIORITY"] == 0.0
        targs["SUBPRIORITY"][ii] = np.random.random(len(targs))[ii]
        header["SUBPSEED"] = subpseed

    # ADM add the DESI dependencies.
    depend.add_dependencies(header)
    # ADM some other useful header information.
    depend.setdep(header, 'desitarget', io.desitarget_version)
    depend.setdep(header, 'desitarget-git', io.gitversion())
    depend.setdep(header, 'photcat', drstr)

    # ADM add information to construct the filename to the header.
    header["OBSCON"] = "bright"
    header["SURVEY"] = "main"
    header["RESOLVE"] = True
    header["DR"] = drint
    header["GAIADR"] = gaiadr

    # ADM create necessary directories, if they don't exist.
    os.makedirs(os.path.dirname(outfn), exist_ok=True)
    # ADM and, finally, write out the targets.
    io.write_with_units(outfn, targs, extname="STREAMTARGETS", header=header)

    return len(targs), outfn
