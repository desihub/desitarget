"""
desitarget.streamutilties
=========================

Utilities for the DESI MWS Stellar Stream programs.

Borrows heavily from Sergey Koposov's `astrolibpy routines`_.

.. _`astrolibpy routines`: https://github.com/segasai/astrolibpy/blob/master/my_utils
"""
import yaml
import os
import fitsio
import numpy as np
import healpy as hp
import astropy.coordinates as acoo
import astropy.units as auni
from pkg_resources import resource_filename
from scipy.interpolate import UnivariateSpline
from time import time
from zero_point import zero_point as gaia_zpt

from desitarget import io
from desitarget.geomask import pixarea2nside, add_hp_neighbors, sweep_files_touch_hp
from desitarget.gaiamatch import match_gaia_to_primary
from desitarget.targets import resolve

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock.
start = time()

# ADM load the Gaia zeropoints.
gaia_zpt.zpt.load_tables()

# ADM Galactic reference frame. Use astropy v4.0 defaults.
GCPARAMS = acoo.galactocentric_frame_defaults.get_from_registry(
    "v4.0")['parameters']

# ADM some standard units.
kms = auni.km / auni.s
masyr = auni.mas / auni.year

# ADM the standard data model for working with streams.
streamcols = np.array([], dtype=[
    ('RELEASE', '>i2'), ('BRICKID', '>i4'), ('TYPE', 'S4'),
    ('OBJID', '>i4'), ('RA', '>f8'), ('DEC', '>f8'), ('EBV', '>f4'),
    ('FLUX_G', '>f4'), ('FLUX_R', '>f4'), ('FLUX_Z', '>f4'),
    ('REF_EPOCH', '>f4'), ('PARALLAX', '>f4'), ('PARALLAX_IVAR', '>f4'),
    ('PMRA', '>f4'), ('PMRA_IVAR', '>f4'),
    ('PMDEC', '>f4'), ('PMDEC_IVAR', '>f4'),
    ('ASTROMETRIC_PARAMS_SOLVED', '>i1'), ('NU_EFF_USED_IN_ASTROMETRY', '>f4'),
    ('PSEUDOCOLOUR', '>f4'), ('PHOT_G_MEAN_MAG', '>f4'), ('ECL_LAT', '>f8')
])


def cosd(x):
    """Return cos(x) for an angle x in degrees.
    """
    return np.cos(np.deg2rad(x))


def sind(x):
    """Return sin(x) for an angle x in degrees.
    """
    return np.sin(np.deg2rad(x))


def betw(x, x1, x2):
    """Whether x lies in the range x1 <= x < x2.

    Parameters
    ----------
    x : :class:`~numpy.ndarray` or `int` or `float`
        Value(s) that need checked against x1, x2.
    x1 : :class:`~numpy.ndarray` or `int` or `float`
        Lower range to check against (inclusive).
    x2 : :class:`~numpy.ndarray` or `int` or `float`
        Upper range to check against (exclusive).

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for values of `x` that lie in the range x1 <= x < x2.
        If any input is an array then the output will be a Boolean array.

    Notes
    -----
    - Very permissive. Arrays can be checked against other arrays,
      scalars against scalars and arrays against arrays. For example, if
      all the inputs are arrays the calculation will be element-wise. If
      `x1` and `x2` are floats and `x` is array-like then each element of
      `x` will be checked against the range. If `x` and `x2` are floats
      and `x1` is an array, all possible x1->x2 ranges will be checked.
    """
    return (x >= x1) & (x < x2)


def torect(ra, dec):
    """Convert equatorial coordinates to Cartesian coordinates.

    Parameters
    ----------
    ra : :class:`~numpy.ndarray` or `float`
        Right Ascension in DEGREES.
    dec : :class:`~numpy.ndarray` or `float`
        Declination in DEGREES.

    Returns
    -------
    :class:`tuple`
        A tuple of the x, y, z converted values. If `ra`, `dec` are
        passed as arrays this will be a tuple of x, y, z, arrays.
    """
    x = cosd(ra) * cosd(dec)
    y = sind(ra) * cosd(dec)
    z = sind(dec)

    return x, y, z


def fromrect(x, y, z):
    """Convert equatorial coordinates to Cartesian coordinates.

    Parameters
    ----------
    x, y, z : :class:`~numpy.ndarray` or `float`
        Cartesian coordinates.

    Returns
    -------
    :class:`tuple`
        A tuple of the RA, Dec converted values in DEGREES. If `x`, `y`,
        `z` are passed as arrays this will be a tuple of RA, Dec arrays.
    """
    ra = np.arctan2(y, x) * 57.295779513082323
    dec = 57.295779513082323 * np.arctan2(z, np.sqrt(x**2 + y**2))

    return ra, dec


def rotation_matrix(rapol, decpol, ra0):
    """Return the rotation matrix corresponding to the pole of rapol,
    decpol and with the zero of the new latitude corresponding to ra=ra0.
    The resulting matrix needs to be np.dot'ed with a vector to forward
    transform that vector.

    Parameters
    ----------
    rapol, decpol : :class:`float`
        Pole of the new coordinate system in DEGREES.
    ra0 : :class:`float`
        Zero latitude of the new coordinate system in DEGREES.

    Returns
    -------
    :class:`~numpy.ndarray`
        3x3 Rotation matrix.
    """
    tmppol = np.array(torect(rapol, decpol))  # pole axis
    tmpvec1 = np.array(torect(ra0, 0))  # x axis
    tmpvec1 = np.array(tmpvec1)

    tmpvec1[2] = (-tmppol[0] * tmpvec1[0] - tmppol[1] * tmpvec1[1]) / tmppol[2]
    tmpvec1 /= np.sqrt((tmpvec1**2).sum())
    tmpvec2 = np.cross(tmppol, tmpvec1)  # y axis
    M = np.array([tmpvec1, tmpvec2, tmppol])

    return M


def sphere_rotate(ra, dec, rapol, decpol, ra0, revert=False):
    """Rotate ra, dec to a new spherical coordinate system.

    Parameters
    ----------
    ra : :class:`~numpy.ndarray` or `float`
        Right Ascension in DEGREES.
    dec : :class:`~numpy.ndarray` or `float`
        Declination in DEGREES.
    rapol, decpol : :class:`float`
        Pole of the new coordinate system in DEGREES.
    ra0 : :class:`float`
        Zero latitude of the new coordinate system in DEGREES.
    revert : :class:`bool`, optional, defaults to ``False``
        Reverse the rotation.

    Returns
    -------
    :class:`tuple`
        A tuple of the the new RA, Dec values in DEGREES. If `ra`, `dec`
        are passed as arrays this will be a tuple of RA, Dec arrays.
    """
    x, y, z = torect(ra, dec)
    M = rotation_matrix(rapol, decpol, ra0)

    if not revert:
        Axx, Axy, Axz = M[0]
        Ayx, Ayy, Ayz = M[1]
        Azx, Azy, Azz = M[2]
    else:
        Axx, Ayx, Azx = M[0]
        Axy, Ayy, Azy = M[1]
        Axz, Ayz, Azz = M[2]
    xnew = x * Axx + y * Axy + z * Axz
    ynew = x * Ayx + y * Ayy + z * Ayz
    znew = x * Azx + y * Azy + z * Azz
    del x, y, z
    tmp = fromrect(xnew, ynew, znew)

    return (tmp[0], tmp[1])


def rotate_pm(ra, dec, pmra, pmdec, rapol, decpol, ra0, revert=False):
    """
    Rotate proper motions to a new spherical coordinate system.

    Parameters
    ----------
    ra, dec : :class:`~numpy.ndarray` or `float`
        Right Ascension, Declination in DEGREES.
    pmra, pmdec : :class:`~numpy.ndarray` or `float`
        Proper motion in Right Ascension, Declination in mas/yr.
    pmdec : :class:`~numpy.ndarray` or `float`
        Proper motion in Declination in mas/yr.
    rapol, decpol : :class:`float`
        Pole of the new coordinate system in DEGREES.
    ra0 : :class:`float`
        Zero latitude of the new coordinate system in DEGREES.
    revert : :class:`bool`, optional, defaults to ``False``
        Reverse the rotation.

    Returns
    -------
    :class:`tuple`
        A tuple of the the new pmra, pmdec values in DEGREES. If `ra`,
        `dec`, etc. are passed as arrays this will be a tuple of arrays.
    """
    ra, dec, pmra, pmdec = [np.atleast_1d(_) for _ in [ra, dec, pmra, pmdec]]
    M = rotation_matrix(rapol, decpol, ra0)
    if revert:
        M = M.T
    # unit vectors
    e_mura = np.array([-sind(ra), cosd(ra), ra * 0])
    e_mudec = np.array(
        [-sind(dec) * cosd(ra), -sind(dec) * sind(ra),
         cosd(dec)])
    # velocity vector in arbitrary units
    V = pmra * e_mura + pmdec * e_mudec
    del e_mura, e_mudec
    # apply rotation to velocity
    V1 = M @ V
    del V
    X = np.array([cosd(ra) * cosd(dec), sind(ra) * cosd(dec), sind(dec)])
    # apply rotation to position
    X1 = M @ X
    del X
    # rotated coordinates in radians
    lon = np.arctan2(X1[1, :], X1[0, :])
    lat = np.arctan2(X1[2, :], np.sqrt(X1[0, :]**2 + X1[1, :]**2))
    del X1
    # unit vectors in rotated coordinates
    e_mura = np.array([-np.sin(lon), np.cos(lon), lon * 0])
    e_mudec = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),
         np.cos(lat)])
    del lon, lat

    return np.sum(e_mura * V1, axis=0), np.sum(e_mudec * V1, axis=0)


def correct_pm(ra, dec, pmra, pmdec, dist):
    """Correct proper motions for the Sun's motion.

    Parameters
    ----------
    ra, dec : :class:`~numpy.ndarray` or `float`
        Right Ascension, Declination in DEGREES.
    pmra, pmdec : :class:`~numpy.ndarray` or `float`
        Proper motion in Right Ascension, Declination in mas/yr.
        `pmra` includes the cosine term.
    dist : :class:`float`
        Distance in kpc.

    Returns
    -------
    :class:`tuple`
        A tuple of the the new (pmra, pmdec) values in DEGREES. If `ra`,
        `dec`, etc. are passed as arrays this will be a tuple of arrays.
    """
    C = acoo.ICRS(ra=ra * auni.deg,
                  dec=dec * auni.deg,
                  radial_velocity=0 * kms,
                  distance=dist * auni.kpc,
                  pm_ra_cosdec=pmra * masyr,
                  pm_dec=pmdec * masyr)
    frame = acoo.Galactocentric(**GCPARAMS)
    Cg = C.transform_to(frame)
    Cg1 = acoo.Galactocentric(x=Cg.x,
                              y=Cg.y,
                              z=Cg.z,
                              v_x=Cg.v_x * 0,
                              v_y=Cg.v_y * 0,
                              v_z=Cg.v_z * 0,
                              **GCPARAMS)
    C1 = Cg1.transform_to(acoo.ICRS())

    return ((C.pm_ra_cosdec - C1.pm_ra_cosdec).to_value(masyr),
            (C.pm_dec - C1.pm_dec).to_value(masyr))


def get_stream_parameters(stream_name):
    """Look up information for a given stream.

    Parameters
    ----------
    stream_name : :class:`str`
        Name of a stream that appears in the ../data/streams.yaml file.
        Possibilities include 'GD1'.

    Returns
    -------
    :class:`~dict`
        A dictionary of stream parameters for the passed `stream_name`.
        Includes isochrones and positional information.

    Notes
    -----
    - Parameters for each stream are in the ../data/streams.yaml file.
    """
    # ADM guard against stream being passed as lower-case.
    stream_name = stream_name.upper()

    # ADM open and load the parameter yaml file.
    fn = resource_filename('desitarget', os.path.join('data', 'streams.yaml'))
    with open(fn) as f:
        stream = yaml.safe_load(f)

    return stream[stream_name]


def get_CMD_interpolator(stream_name):
    """Isochrones via interpolating over points in color-magnitude space.

    Parameters
    ----------
    stream_name : :class:`str`
        Name of a stream that appears in the ../data/streams.yaml file.
        Possibilities include 'GD1'.

    Returns
    -------
    A scipy interpolated UnivariateSpline.
    """
    # ADM get information for the stream of interest.
    stream = get_stream_parameters(stream_name)

    # ADM retrieve the color and magnitude offsets.
    coloff = stream["COLOFF"]
    magoff = stream["MAGOFF"]

    # ADM the isochrones to interpolate over.
    iso_dartmouth_g = np.array(stream["ISO_G"])
    iso_dartmouth_r = np.array(stream["ISO_R"])

    # ADM UnivariateSpline is from scipy.interpolate.
    CMD_II = UnivariateSpline(iso_dartmouth_r[::-1] + magoff,
                              (iso_dartmouth_g - iso_dartmouth_r - coloff)[::-1],
                              s=0)

    return CMD_II


def pm12_sel_func(pm1track, pm2track, pmfi1, pmfi2, pm_err, pad=2, mult=2.5):
    """Select stream members using proper motion, padded by some error.

    Parameters
    ----------
    pm1track : :class:`~numpy.ndarray` or `float`
        Allowed proper motions of stream targets, RA-sense.
    pm2track : :class:`~numpy.ndarray` or `float`
        Allowed proper motions of stream targets, Dec-sense.
    pmfi1 : :class:`~numpy.ndarray` or `float`
        Proper motion in stream coordinates of possible targets, derived
        from RA.
    pmfi2 : :class:`~numpy.ndarray` or `float`
        Proper motion in stream coordinates of possible targets, derived
        from Dec.
    pm_err : :class:`~numpy.ndarray` or `float`
        Proper motion error in stream coordinates of possible targets,
        combined across `pmfi1` and `pmfi2` errors.
    pad: : :class:`float` or `int`, defaults to 2
        Extra offset with which to pad `mult`*proper_motion_error.
    mult : :class:`float` or `int`, defaults to 2.5
        Multiple of the proper motion error to use for padding.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for stream members.
    """

    return np.sqrt((pmfi2 - pm2track)**2 +
                   (pmfi1 - pm1track)**2) < pad + mult * pm_err


def plx_sel_func(dist, D, mult, plx_sys=0.05):
    """Select stream members using parallax, padded by some error.

    Parameters
    ----------
    dist : :class:`~numpy.ndarray` or `float`
        Distance of possible stream members.
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains at least
        the columns `RA`, `ASTROMETRIC_PARAMS_SOLVED`, `PHOT_G_MEAN_MAG`,
        `NU_EFF_USED_IN_ASTRONOMY`, `PSEUDOCOLOUR`, `ECL_LAT`, `PARALLAX`
        `PARALLAX_ERROR`. `PARALLAX_IVAR` will be used instead of
        `PARALLAX_ERROR` if `PARALLAX_ERROR` is not present.
    mult : :class:`float` or `int`
        Multiple of the parallax error to use for padding.
    plx_sys : :class:`float`
        Extra offset with which to pad `mult`*parallax_error.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for stream members.
    """
    # extra plx systematic error padding
    plx_sys = 0.05
    subset = np.in1d(D['ASTROMETRIC_PARAMS_SOLVED'], [31, 95])
    plx_zpt_tmp = gaia_zpt.zpt.get_zpt(D['PHOT_G_MEAN_MAG'][subset],
                                       D['NU_EFF_USED_IN_ASTROMETRY'][subset],
                                       D['PSEUDOCOLOUR'][subset],
                                       D['ECL_LAT'][subset],
                                       D['ASTROMETRIC_PARAMS_SOLVED'][subset],
                                       _warnings=False)
    plx_zpt = np.zeros(len(D['RA']))
    plx_zpt_tmp[~np.isfinite(plx_zpt_tmp)] = 0
    plx_zpt[subset] = plx_zpt_tmp
    plx = D['PARALLAX'] - plx_zpt
    dplx = 1 / dist - plx

    if 'PARALLAX_ERROR' in D.dtype.names:
        parallax_error = D['PARALLAX_ERROR']
    elif 'PARALLAX_IVAR' in D.dtype.names:
        parallax_error = 1./np.sqrt(D['PARALLAX_IVAR'])
    else:
        msg = "Either PARALLAX_ERROR or PARALLAX_IVAR must be passed!"
        log.error(msg)

    return np.abs(dplx) < plx_sys + mult * parallax_error


def stream_distance(fi1, stream_name):
    """The distance to members of a stellar stream.

    Parameters
    ----------
    fi1 : :class:`~numpy.ndarray` or `float`
        Phi1 stream coordinate of possible targets, derived from RA.
    stream_name : :class:`str`
        Name of a stream, e.g. "GD1".

    Returns
    -------
    :class:`array_like` or `float`
        The distance to the passed members of the stream.

    Notes
    -----
    - Output type is the same as that of the passed `fi1`.
    """
    if stream_name.upper() == "GD1":
        # ADM The distance to GD1 (similar to Koposov et al. 2010 paper).
        dm = 18.82 + ((fi1 + 48) / 57)**2 - 4.45
        return 10**(dm / 5. - 2)
    else:
        msg = f"stream name {stream_name} not recognized"
        log.error(msg)


def read_data(swdir, rapol, decpol, ra_ref, mind, maxd, stream_name,
              readcache=True, addnors=True, test=False):
    """Assemble the data needed for a particular stream program.

    Example values for GD1:
    swdir = "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0"
    rapol, decpol, ra_ref = 34.5987, 29.7331, 200
    mind, maxd = 80, 100

    Parameters
    ----------
    swdir : :class:`str`
        Root directory of Legacy Surveys sweep files for a given data
        release for ONE of EITHER north or south, e.g.
        "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0".
    rapol, decpol : :class:`float`
        Pole in the stream coordinate system in DEGREES.
    ra_ref : :class:`float`
        Zero latitude in the stream coordinate system in DEGREES.
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

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for stream members.

    Notes
    -----
    - The $TARG_DIR environment variable must be set to read/write from
      a cache. If $TARG_DIR is not set, caching is completely ignored.
    """
    # ADM The Gaia DR to which to match.
    gaiadr = "dr3"

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
            infiles2 = swdir.replace("south", "north")
        elif "north" in swdir:
            infiles2 = swdir.replace("north", "south")
        else:
            msg = "addnors passed but swdir does not contain north or south!"
            raise ValueError(msg)
        infiles += io.list_sweepfiles(infiles2)

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

    # ADM loop through the sweep files and limit to objects in the stream.
    allobjs = []
    for i, filename in enumerate(infiles):
        objs = io.read_tractor(filename)
        cobjs = acoo.SkyCoord(objs["RA"]*auni.degree, objs["DEC"]*auni.degree)
        sep = cobjs.separation(cstream)

        # ADM only retain objects in the stream...
        ii = betw(sep.value, mind, maxd)

        # ADM ...that aren't very faint (> 22.5 mag)...
        ii &= objs["FLUX_R"] > 1
        objs = objs[ii]

        # ADM limit to northern objects in northern imaging and southern
        # ADM objects in southern imaging.
        LSobjs = resolve(objs)

        # ADM catch the case where there are no objects meeting the cuts.
        if len(LSobjs) > 0:
            gaiaobjs = match_gaia_to_primary(LSobjs, matchrad=1., dr=gaiadr)
        else:
            gaiaobjs = LSobjs

        # ADM a (probably unnecessary) sanity check.
        assert(len(gaiaobjs) == len(LSobjs))

        # ADM only retain critical columns from the global data model.
        data = np.zeros(len(LSobjs), dtype=streamcols.dtype)
        # ADM for both Gaia and Legacy Surveys, overwriting with Gaia.
        for objs in LSobjs, gaiaobjs:
            sharedcols = set(data.dtype.names).intersection(set(objs.dtype.names))
            for col in sharedcols:
                data[col] = objs[col]

        # ADM retain the data from this part of the loop.
        allobjs.append(data)
        if i % 10 == 9:
            log.info(f"Ran {i+1}/{len(infiles)} files...t={time()-start:.1f}s")

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
