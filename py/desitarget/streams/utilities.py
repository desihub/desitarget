"""
desitarget.streams.utilities
============================

Utilities for the DESI MWS Stellar Stream programs.

Borrows heavily from Sergey Koposov's `astrolibpy routines`_.

.. _`astrolibpy routines`: https://github.com/segasai/astrolibpy/blob/master/my_utils
"""
import yaml
import os
import numpy as np
import astropy.coordinates as acoo
import astropy.units as auni
from importlib import resources
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from time import time
import desitarget.streams.gaia_dr3_parallax_zero_point.zpt as gaia_zpt
from numpy.lib import recfunctions as rfn

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock.
start = time()

# ADM load the Gaia zero points.
gaia_zpt.load_tables()

# ADM Galactic reference frame. Use astropy v4.0 defaults.
GCPARAMS = acoo.galactocentric_frame_defaults.get_from_registry(
    "v4.0")['parameters']

# ADM some standard units.
kms = auni.km / auni.s
masyr = auni.mas / auni.year


def ivars_to_errors(objs, colnames=[]):
    """
    Convert inverse variances to errors without dividing by zero.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Array that contains the columns to be converted from inverse
        variances to errors.
    colnames : :class:`list`
        The names of the columns to convert.

    Returns
    -------
    :class:`~numpy.ndarray`
        The input `objs`, modified so that any columns in `colnames` are
        converted from inverse variances to errors and occurrences of
        "IVAR" in column names are converted to "ERROR".

    Notes
    -----
    - Column names are assumed to be upper case for the conversion of
      IVAR->ERROR in the column names.
    - No copy is made to save memory. So `objs` will be modified in place
      and calls like a = ivars_to_errors(b, colnames=["X"]) will alter
      values in b as well as returning a.
    """
    for colname in colnames:
        # ADM guard against dividing by zero.
        error = np.zeros_like(objs[colname]) + np.nan
        ii = objs[colname] != 0
        error[ii] = 1./np.sqrt(objs[ii][colname])
        objs[colname] = error
        newcolname = colname.replace("IVAR", "ERROR")
        # ADM rename any IVAR columns.
        objs = rfn.rename_fields(objs, {colname: newcolname})

    return objs


def errors_to_ivars(objs, colnames=[]):
    """
    Convert errors to ivars, correctly dealing with NaNs.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Array that contains the columns to be converted from errors
        to inverse variances.
    colnames : :class:`list`
        The names of the columns to convert.

    Returns
    -------
    :class:`~numpy.ndarray`
        The input `objs`, modified so that any columns in `colnames` are
        converted from errors to inverse variances and occurrences of
        "ERROR" in column names are converted to "IVAR".

    Notes
    -----
    - Column names are assumed to be upper case for the conversion of
      ERROR->IVAR in the column names.
    - No copy is made to save memory. So `objs` will be modified in place
      and calls like a = errors_to_ivars(b, colnames=["X"]) will alter
      values in b as well as returning a.
    """
    for colname in colnames:
        # ADM remove any NaNs.
        ivar = np.zeros_like(objs[colname])
        ii = ~np.isnan(objs[colname])
        ivar[ii] = 1./objs[ii][colname]**2
        objs[colname] = ivar
        newcolname = colname.replace("ERROR", "IVAR")
        # ADM rename any IVAR columns.
        objs = rfn.rename_fields(objs, {colname: newcolname})

    return objs


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
    del Cg
    del C
    del frame
    C1 = Cg1.transform_to(acoo.ICRS())
    del Cg1
    # CMR modified to free up some memory, otherwise fails
    pmracorrfac = C1.pm_ra_cosdec.to_value(masyr)
    pmdeccorrfac = C1.pm_dec.to_value(masyr)
    del C1
    #return ((C.pm_ra_cosdec - C1.pm_ra_cosdec).to_value(masyr),
    #        (C.pm_dec - C1.pm_dec).to_value(masyr))
    return pmra - pmracorrfac, pmdec-pmdeccorrfac

def get_targmwext_parameters(targmwext_name):
    """Look up information for a given stream or dwarf

    Parameters
    ----------
    targmwext_name : :class:`str`
        Name of a stream that appears in the ../data/streams.yaml file
        or ../data/dwarf.yaml
        Possibilities include 'GD1'.

    Returns
    -------
    :class:`~dict`
        A dictionary of stream parameters for the passed `targmwext_name`.
        Includes isochrones and positional information.

    Notes
    -----
    - Parameters for each stream are in the ../data/streams.yaml file.
    - Parameters for each dwarf are in the ../data/dwarfs.yaml file.
    """
    # ADM guard against stream being passed as lower-case.
    targmwext_name = targmwext_name.upper()

    # ADM open and load the stream parameter yaml file.
    fn = resources.files('desitarget').joinpath('data/streams.yaml')
    with open(fn) as f:
        stream = yaml.safe_load(f)
    streamlist = list(stream.keys())
    fn2 = resources.files('desitarget').joinpath('data/dwarfs.yaml')
    with open(fn2) as f:
        dwarf = yaml.safe_load(f)
    dwarflist = list(dwarf.keys())

    if targmwext_name in streamlist:
        return stream[targmwext_name]
    elif targmwext_name in dwarflist:
        return dwarf[targmwext_name]


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
    stream = get_targmwext_parameters(stream_name)

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
    """Select stream members in stream coordinates, using proper motion, padded by some error.

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


def pm12_distdep_sel_func(pm1track, pm2track, pmfi1, pmfi2, pm_err, dist, velpad, mult=2.5):
    """Select stream members in stream coordinate, using a proper motion range defined by a 
       velocity width, padded by some multiple of the error.

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
    pad: : :class:`float` or `int`
        Width of PM selection in km/s, with `mult`*proper_motion_error
    mult : :class:`float` or `int`, defaults to 2.5
        Multiple of the proper motion error to use for padding.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for stream members.
    """

    dpmtot = np.sqrt((pmfi2 - pm2track)**2 + (pmfi1 - pm1track)**2)
    dpmlim = velpad/(4.74*dist)
    return dpmtot < (dpmlim + mult * pm_err)

def pm0_sel_func(pmra0, pmdec0, D, pad=2, mult=2.5):
    """Select dwarf members using proper motion, using a PM range
        around the systemtic PM of the dwarf padded by some error.

    Parameters
    ----------
    pmra0, pmdec0 : :class:`~numpy.ndarray` or `float`
        Systemic proper motions of dwarf galaxy.
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains at least
        the columns `PMRA`, `PMDEC`, `PMRA_ERROR`, and `PMDEC_ERROR`.
    pad: : :class:`float` or `int`, defaults to 2
        Extra offset with which to pad `mult`*proper_motion_error.
    mult : :class:`float` or `int`, defaults to 2.5
        Multiple of the proper motion error to use for padding.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for dwarf members.
    """
    # combine proper motion errors in RA and Dec
    pm_err = np.sqrt(0.5 * (D['PMRA_ERROR']**2 + D['PMDEC_ERROR']**2))
    return np.sqrt((D['PMRA'] - pmra0)**2 + (D['PMDEC'] - pmdec0)**2) < pad + mult * pm_err

def apply_plx_zpt(D):
    """Apply zero-point correction to Gaia parallax

    Parameters
    ----------
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains at least
        the columns `RA`, `ASTROMETRIC_PARAMS_SOLVED`, `PHOT_G_MEAN_MAG`,
        `NU_EFF_USED_IN_ASTRONOMY`, `PSEUDOCOLOUR`, `ECL_LAT`, `PARALLAX`
        `PARALLAX_ERROR`. `PARALLAX_IVAR` will be used instead of
        `PARALLAX_ERROR` if `PARALLAX_ERROR` is not present.

    Returns
    -------
    :class:`array_like` or `float`
        Zero-point corrected parallax.
    """
    subset = np.in1d(D['ASTROMETRIC_PARAMS_SOLVED'], [31, 95])
    plx_zpt_tmp = gaia_zpt.get_zpt(D['PHOT_G_MEAN_MAG'][subset],
                                   D['NU_EFF_USED_IN_ASTROMETRY'][subset],
                                   D['PSEUDOCOLOUR'][subset],
                                   D['ECL_LAT'][subset],
                                   D['ASTROMETRIC_PARAMS_SOLVED'][subset],
                                   _warnings=False)
    plx_zpt = np.zeros(len(D['RA']))
    plx_zpt_tmp[~np.isfinite(plx_zpt_tmp)] = 0
    plx_zpt[subset] = plx_zpt_tmp
    plx = D['PARALLAX'] - plx_zpt
    return plx

def get_plx_error(D):
    """Gets parallax error from inverse variance if necessary.

    Parameters
    ----------
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains either
        the column `PARALLAX_ERROR` or the column `PARALLAX_IVAR`.
    Returns
    -------
    :class:`array_like` or `float`
        Parallax error.
    """
    if 'PARALLAX_ERROR' in D.dtype.names:
        parallax_error = D['PARALLAX_ERROR']
    elif 'PARALLAX_IVAR' in D.dtype.names:
        parallax_error = np.zeros_like(D["PARALLAX_IVAR"]) + 1e8
        ii = D['PARALLAX_IVAR'] != 0
        parallax_error[ii] = 1./np.sqrt(D[ii]['PARALLAX_IVAR'])
    else:
        msg = "Either PARALLAX_ERROR or PARALLAX_IVAR must be passed!"
        log.error(msg)
    return parallax_error


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
    # get parallaxes and apply zero point correction where appropriate
    plx = apply_plx_zpt(D)
    # get errors from IVARS or ERRORS
    parallax_error = get_plx_error(D)

    dplx = 1 / dist - plx
    return np.abs(dplx) < plx_sys + mult * parallax_error

def simple_plx_sel(dist, D, multfac, plxlim, plx_sys=0.05):

    """Select stream members using a parallax upper limit, padded by some error.

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
    plxlim : :class:`float` select possible stream members with plx < plx_lim, plus pad

    Returns
    -------
    :class:`array_like` or `boolean`
	``True`` for stream members.
    """
    # CMR first block of code to fix zpt and ivars is identical to plx_sel
    plx = apply_plx_zpt(D)
    parallax_error = get_plx_error(D)

    # CMR select plx < the upper  limit give by plxlim
    psel = plx < (multfac*parallax_error + plx_sys + plxlim)
    return psel

def dwarf_plx_sel_func(dist, D, plx_sys=0.05, mult=2.5, keep_all_neg=False, min_plx_plxerr=-5):
    """Select dwarf members using parallax, padded by some error.
    NOTE: This is a slight alteration on the GD1 parallax selection,
    as it will select anything with a negative parallax
    regardless of the distance to the galaxy

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
    plx_sys : :class:`float`
        Extra offset with which to pad `mult`*parallax_error.
    mult : :class:`float` or `int`
        Multiple of the parallax error to use for padding.
    keep_all_neg : :class:`bool`
        If true, keep all targets with negative parallaxes.
    min_plx_plxerr : :class:`float`
        Minimum allowable parallax/parallax error, defaults to -5.
        Useful if keep_all_neg = True.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for stream members.
    """
    # Apply zero-point correction to parallax
    plx = apply_plx_zpt(D)
    # Get parallax error
    parallax_error = get_plx_error(D)
    # Apply systemic error padding
    dplx = plx - 1 / dist
    if keep_all_neg:
        sel = (dplx < plx_sys + mult * parallax_error) \
            & (plx / parallax_error > min_plx_plxerr)
    else:
        sel = (np.abs(dplx) < plx_sys + mult * parallax_error) \
            & (plx / parallax_error > min_plx_plxerr)
    return sel

def stream_distance(fi1, stream_name, stream):
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
    #    # ADM The distance to GD1 (similar to Koposov et al. 2010 paper).
    #    dm = 18.82 + ((fi1 + 48) / 57)**2 - 4.45
    #    return 10**(dm / 5. - 2)
        DISTSP = UnivariateSpline(stream['DIST_PHI1T'],stream['DISTT'],s=0)
        return DISTSP(fi1)
    if stream_name.upper() == "ORPHAN":
        DISTSP = UnivariateSpline(stream['DIST_PHI1T'],stream['DISTT'])
        return DISTSP(fi1)
    else:
        msg = f"stream name {stream_name} not recognized"
        log.error(msg)

# select BHB for an old metal-poor population
# this uses an M92 horizontal branch
# distance to the stream, in kpc, at the phi1 location of each star.
# i.e., dist ance to each star if it were in the stream
def oldpop_bhb_sel(gmag, rmag, distance):

    """Select stream members with the CMD properties of BHBs belonging to the stream.

    Parameters
    ----------
    distance : :class:`~numpy.ndarray` or `float`
        Distance of possible stream members, in kpc.
    gmag : :class:`~numpy.ndarray` or `float`
        extinction-corrected g magnitude of candidates
    rmag:  :class:`~numpy.ndarray` or `float`
        extinction-corrected r magnitude of candidates

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for stream members.
    """

    # convert target distance to abs mag
    dm =  5 * np.log10(distance * 1e3) - 5
    absr = rmag - dm

    # define the horizontal branch in absolute mag
    # based in the M92 horizontal branch
    hb_r = np.array([2.72079, 1.21161, 0.78141, 0.48101, 0.42081, 0.36061,
                     0.30041, 0.24021])
    hb_g = np.array([2.297328, 0.877968, 0.547568, 0.446768, 0.486368,
                     0.525968, 0.565568, 0.605168])

    # Now make CMD cut for BHB
    grw_bhb = 0.5 # BHB width in gr
    rw_bhb = 0.5  # BHB width in r
    grmin_bhb = -0.45 # min g-r of BHB
    grmax_bhb = 0.4 # max g-r of BHB
    # first select in the correct gmr range
    gmr_range_bhb = (gmag - rmag < grmax_bhb) & (gmag - rmag > grmin_bhb)

    # interpolate gmr vs r using the HB
    gr_bhb = np.interp(absr, hb_r[::-1] , hb_g[::-1] - hb_r[::-1],
                       left=np.nan, right=np.nan)
    # interpolate r vs. gmr using the HB
    absr_bhb = np.interp(gmag - rmag, hb_g - hb_r, hb_r, left=np.nan,
                         right=np.nan)
    rr_bhb = absr_bhb + dm
    # data_gmr - interp_gmr
    del_color_cmd_bhb = gmag - rmag - gr_bhb
    # data_r - interp_r
    del_r_cmd_bhb = rmag - rr_bhb

    # select if abs(data_gmr - interp_gmr_) < grw_bhb or
    # abs(data_r - interp_r) < rw_bhb
    cmdsel_bhb = (gmr_range_bhb)&((abs(del_color_cmd_bhb) < grw_bhb) | (abs(del_r_cmd_bhb) < rw_bhb))
    return cmdsel_bhb

def cmd_sel_func(
    dwarf_name, D,
    rgb_color_tol=0.2, rgb_mag_tol=0.0,
    rgb_grmin=-0.5, rgb_grmax=1.5,
    rgb_rmin=16, rgb_rmax=23,
    hb_color_tol=0.1, hb_mag_tol=0.5,
    hb_grmin=-0.5, hb_grmax=0.5,
    hb_rmin=16, hb_rmax=23
):
    """Select dwarf galaxy members using CMD cuts.

    Parameters
    ----------
    dwarf_name : :class:`str`
        Name of a dwarf galaxy that appears in the ../data/dwarfs.yaml file.
        Possibilities include 'DRACO_1'.
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains at least
        the columns `FLUX_G`, `FLUX_R`, `FLUX_Z`,
        `FLUX_IVAR_G`, `FLUX_IVAR_R`, `FLUX_IVAR_Z`,
        and `EBV`.
    rgb_color_tol : :class:`float` or `int`
        Width of RGB cmd selection, defaults to 0.2.
    rgb_mag_tol : :class:`float` or `int`
        Height of RGB cmd selection, defaults to 0.0.
        WARNING: Non-zero values lead to unexpected behavior at the tip of the RGB
    rgb_grmin : :class:`float` or `int`
        Minimum extinction-corrected g-r for RGB selection, defaults to -0.5.
    rgb_grmax : :class:`float` or `int`
        Maximum extinction-corrected g-r for RGB selection, defaults to 1.5.
    rgb_rmin : :class:`float` or `int`
        Bright limit for RGB selection, defaults to 16.
    rgb_rmax : :class:`float` or `int`
        Faint limit for RGB selection, defaults to 23.
    hb_color_tol : :class:`float` or `int`
        Width of HB cmd selection, defaults to 0.1.
    hb_mag_tol : :class:`float` or `int`
        Height of HB cmd selection, defaults to 0.5.
    hb_grmin : :class:`float` or `int`
        Minimum extinction-corrected g-r for HB selection, defaults to -0.5.
    hb_grmax : :class:`float` or `int`
        Maximum extinction-corrected g-r for HB selection, defaults to 0.5.
    hb_rmin : :class:`float` or `int`
        Bright limit for HB selection, defaults to 16.
    hb_rmax : :class:`float` or `int`
        Faint limit for HB selection, defaults to 23.
    show_magerr_plot : :class:`bool`
        Show log10(rmag  error) vs. rmag relationship

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for dwarf members.
    """
    # look up the defining parameters of the dwarf.
    dwarf = get_targmwext_parameters(dwarf_name)
    # distance to the dwarf
    dist = dwarf["DIST"]
    # distance modulus of the dwarf
    dm = 5 * np.log10(dist * 1e3) - 5
    # the RGB isochrone of the dwarf
    iso_rgb_g = np.array(dwarf["ISO_RGB_G"])
    iso_rgb_r = np.array(dwarf["ISO_RGB_R"])
    # the HB isochrone of the dwarf
    iso_hb_g = np.array(dwarf["ISO_HB_G"])
    iso_hb_r = np.array(dwarf["ISO_HB_R"])
    # retrieve the color and magnitude offsets.
    coloff = dwarf["COLOFF"]
    magoff = dwarf["MAGOFF"]
    # retrieve coefficients for the rmag - rmagerr linear fit
    rmag_rmagerr_coeffs = np.array(dwarf["RMAG_RMAGERR_COEFFS"])
    def log10_error_func(x, a, b):
        return a * x + b
    
    # apply isochrone offsets
    iso_rgb_gr = iso_rgb_g - iso_rgb_r
    iso_hb_gr = iso_hb_g - iso_hb_r
    iso_rgb_r += magoff
    iso_hb_r += magoff
    iso_rgb_gr -= coloff
    iso_hb_gr -= coloff

    # dust correction.
    ext_coeff = dict(g=3.237, r=2.176, z=1.217)
    eg, er, ez = [ext_coeff[_] * D['EBV'] for _ in 'grz']
    g, r, z = [22.5 - 2.5 * np.log10(D['FLUX_' + _]) for _ in 'GRZ']
    gerr, rerr, zerr = [2.5 / np.log(10) * (np.sqrt(1./D['FLUX_IVAR_'+_]) / D['FLUX_' + _]) for _ in 'GRZ']
    g0 = g - eg
    r0 = r - er
    z0 = z - ez

    # magnitude cut along RGB
    mag_sel_rgb = betw(r0, np.min(iso_rgb_r + dm) - 0.5, rgb_rmax) & betw(g0 - r0, rgb_grmin, rgb_grmax)
    # color cut along RGB
    rgb_color_tol = np.sqrt(rgb_color_tol**2 + (3*10**log10_error_func(iso_rgb_r + dm, *rmag_rmagerr_coeffs))**2)
    grmax1 = np.interp(r0, iso_rgb_r[::-1] + dm, iso_rgb_gr[::-1] + rgb_color_tol[::-1], left=None, right=None)
    grmax2 = np.interp(r0, iso_rgb_r[::-1] + dm + rgb_mag_tol, iso_rgb_gr[::-1] + rgb_color_tol[::-1], left=None, right=None)
    grmax3 = np.interp(r0, iso_rgb_r[::-1] + dm - rgb_mag_tol, iso_rgb_gr[::-1] + rgb_color_tol[::-1], left=None, right=None)
    rgb_grmax_cmd = np.max(np.array([grmax1, grmax2, grmax3]), axis=0)
    grmin1 = np.interp(r0, iso_rgb_r[::-1] + dm, iso_rgb_gr[::-1] - rgb_color_tol[::-1], left=None, right=None)
    grmin2 = np.interp(r0, iso_rgb_r[::-1] + dm - rgb_mag_tol, iso_rgb_gr[::-1] - rgb_color_tol[::-1], left=None, right=None)
    grmin3 = np.interp(r0, iso_rgb_r[::-1] + dm + rgb_mag_tol, iso_rgb_gr[::-1] - rgb_color_tol[::-1], left=None, right=None)
    rgb_grmin_cmd = np.min(np.array([grmin1, grmin2, grmin3]), axis=0)
    color_sel_rgb = betw(g0 - r0, rgb_grmin_cmd, rgb_grmax_cmd)
    # final RGB selection
    rgb_sel = mag_sel_rgb & color_sel_rgb

    # magnitude cut along HB
    mag_sel_hb = betw(r0, hb_rmin, hb_rmax) & betw(g0 - r0, hb_grmin, hb_grmax)
    # color cut along HB
    gr_hb = np.interp(r0, iso_hb_r[::-1] + dm , iso_hb_gr[::-1], left=np.nan, right=np.nan)
    rr_hb = np.interp(g0 - r0, iso_hb_gr, iso_hb_r + dm, left=np.nan, right=np.nan)
    color_sel_hb = mag_sel_hb & ((abs((g0 - r0) - gr_hb) < hb_color_tol) | (abs(r0 - rr_hb) < hb_mag_tol))
    # final HB selection
    hb_sel = mag_sel_hb & color_sel_hb

    return rgb_sel | hb_sel

def spatial_sel_func(ra0, dec0, maxd, D):
    """Selects dwarf members within a given radius.
    Currently redundant with the pre-selection used in generating the input catalog.

    Parameters
    ----------
    ra0, dec0 : :class:`float`
        Central coordinates of dwarf galaxy in DEGREES.
    D : :class:`~numpy.ndarray`
        Numpy structured array of Gaia information that contains at least
        the columns `RA` and `DEC`.
    maxd : :class:`~numpy.ndarray` or `float`
        Maximum distance of possible targets to dwarf.

    Returns
    -------
    :class:`array_like` or `boolean`
        ``True`` for dwarf members.
    """
    # coordinates of the dwarf.
    cdwarf = acoo.SkyCoord(ra0 * auni.degree, dec0 * auni.degree)
    cobjs = acoo.SkyCoord(D['RA'] * auni.degree, D['DEC'] * auni.degree)
    # separation between the objects of interest and the dwarf.
    sep = cobjs.separation(cdwarf)
    # lies in the the dwarf
    return sep.value < maxd


def sort_targmwext_by_rank(in_targmwextlist):
    """Return a list of all the stream and dwarf names in in_targmwextlist
       ordered by their priority as given in TARGMWEXT_RANK. Smaller numbers
       are higher priority.
       
    Parameters
    ----------
    in_targmwextlist : :class:string` 
        List of dwarf and stream objects to be sorted

    Returns:
    :class:`string`
        Sorted list of dwarf and stream objects
    """
    fn = resources.files('desitarget').joinpath('data/streams.yaml')
    with open(fn) as f:
        streaminfo = yaml.safe_load(f)
    all_stream_names = list(streaminfo.keys())
    fn2 = resources.files('desitarget').joinpath('data/dwarfs.yaml')
    with open(fn2) as f:
        dwarfinfo = yaml.safe_load(f)
    all_dwarf_names = list(dwarfinfo.keys())

    in_ranklist = []  # CMR in_prilist is 1-to-1 matched to names in in_targmwextlist 
    for targmwext in in_targmwextlist:
        if targmwext in all_stream_names:
            sinfo = streaminfo[targmwext]            
            rankval = sinfo['TARGMWEXT_RANK']
        elif targmwext in all_dwarf_names:
            dinfo = dwarfinfo[targmwext]
            rankval = dinfo['TARGMWEXT_RANK']
        in_ranklist.append(rankval)

    in_ranklist = np.array(in_ranklist)
    sortrankidx = np.argsort(in_ranklist)
    sort_targmwextlist = []
    out_ranklist = in_ranklist[sortrankidx]
    out_targmwextlist = []
    for i in range(len(sortrankidx)):
        nextname = in_targmwextlist[sortrankidx[i]]
        out_targmwextlist.append(nextname)
    
    return out_targmwextlist
    
def targmwext_resolve(targmwext_name, mws_target, ibright_pm1, ibright_pm2, ibright_pm3, ipm_only,
                      ifaint_no_pm, ifaint_cmd, ifiller):
    """Resolve ambiguity with target subclass bits in the mwstarget mask 
       using TARGMWEXT_RANK from the yaml file. Smaller numbers are  higher priority.
       Streams and dwarfs are selected in rank order and targets selected for lower ranking
       streams that are also selected in higher ranking dwarfs are only selected if their
       target subclass outranks the target subclass they were selected as for the higher ranking
       object.  All dwarfs outrank streams. GD1 is the highest ranking stream, Orphan the second.
       Ranking of subtarget classes: bright_pm1, bright_pm2, bright_pm3, pm_only, faint_no_pm, faint_cmd, filler
       We make assumptions, which avoid the brute-force implementation of these priorities:
       - brightpm[123] never overlap in magnitude, so an object can't be, e.g., pm1 and pm2 in different stream/dwarfs
       - pm_only can only overlap with brightpm[12]
       - faint_no_pm and faint_cmd are never used in the same stream/dwarf
       - faint_no_pm, faint_cmd and filler can only overlap with each other and bright_pm3 

       The bright_pm1, bright_pm2 and bright_pm3 and pm_only outrank faint_cmd, faint_no_pm and filler. 
       The only overlaps possible and their relative rankings are: 
       1) bright_pm3 outranks faint_no_pm, faint_cmd and filler. bright_pm1 and bright_pm2 are too bright
          to overlap with either of the faint selections. 
       2) faint_no_pm and faint_cmd outrank filler.
       3) pm_only can only overlap bright_pm1 and bright_pm2 (and it is only used for dwarfs)
       Note that the mangnitude ranges of bright_pm[123] are always the same so that, e.g., 
       bright_pm1 and  bright_pm2 can neve be set for the same object.  
       See <insert pointer to Nathan's document>.

    Parameters
    ----------
    targmwext_name : :class:`~numpy.ndarray` or `float`  CMR FIX THIS IS A STRING
        Name of the stream or dwarf galaxy being selected
    mws_target : :class:`~numpy.ndarray` or `float`
        Numpy 1d array of target bits as set by PREVIOUS selections
    ibright_pm1 : :class:`array_like` or `boolean`
        Numpy 1d array, ``True`` for objects that pass the bright_pm1 selection criteria 
    ibright_pm2 : :class:`array_like` or `boolean`
        Numpy 1d array, ``True`` for objects that pass the bright_pm2 selection criteria 
    ibright_pm3 : :class:`array_like` or `boolean`
        Numpy 1d array, ``True`` for objects that pass the bright_pm3 selection criteria 
    ipm_only : :class:`array_like` or `boolean`
        Numpy 1d array, ``True`` for objects that pass pm_only. Use for dwarfs but not streams
    ifaint_no_pm : :class:`array_like` or `boolean`
        Numpy 1d array, ``True`` for objects that pass faint_no_pm. Use for dwarfs but not streams
    ifaint_cmd : :class:`array_like` or `boolean`
        Numpy 1d array, ``True`` for objects that pass faint_cmd. Use for streams but not dwarfs
    ifiller : :class:`array_like` or `boolean`
        Numpy 1d array, ``True`` for objects that pass the filler selection criteria  

    Returns
    -------
   :class:`array_like`
        ``True`` if the object is a "BRIGHT_PM1" target and has priorty for duplicates
    :class:`array_like`
        ``True`` if the object is a "BRIGHT_PM2" target and has priorty for duplicates
    :class:`array_like`
        ``True`` if the object is a "BRIGHT_PM3" target and has priorty for duplicates
    :class:`array_like`
        ``True`` if the object is a "PM_ONLY" target and has priorty for duplicates
    :class:`array_like`
        ``True`` if the object is a "FAINT_NO_PM" target and has priorty for duplicates
    :class:`array_like`
        ``True`` if the object is a "FAINT_CMD" target and has priorty for duplicates
    :class:`array_like`
        ``True`` if the object is a "FILLER" target and has priorty for duplicates
    Notes
    -----
    - See ../data/targetmask.yaml for the definition of the target bits and targmwext_priority
    """

    from desitarget.targetmask import mws_mask

    # target bits set selecting the current stream or dwarf 
    any_set_here = ibright_pm1 | ibright_pm2 | ibright_pm3 | ipm_only | ifaint_no_pm | ifaint_cmd | ifiller
    # target bits set when selecting streams and dwarfs that rank higher than this one
    any_set_prev = mws_target & mws_mask['MWS_EXT']
    # look for objects selected in this stream that were also previously targeted
    # for a higher ranking stream or dwarf
    iialldups = any_set_here & any_set_prev
    ndups = np.sum(iialldups != 0)
    if ndups > 0:
        log.info(f"Found {ndups} duplicates")
        #idxall = np.arange(len(mws_targ))
        #dupidx = idxall[tuple(iialldups)]
        # get the yaml file info for all streams and dwarfs so we can get TARGMWEXT_RANK
        fn = resources.files('desitarget').joinpath('data/streams.yaml')
        with open(fn) as f:
            streaminfo = yaml.safe_load(f)
            all_stream_names = list(streaminfo.keys())
        fn2 = resources.files('desitarget').joinpath('data/dwarfs.yaml')
        with open(fn2) as f:
            dwarfinfo = yaml.safe_load(f)
            all_dwarf_names = list(dwarfinfo.keys())
        # get the yaml file info for our current stream or dwarf
        if targmwext_name in all_stream_names:
            thisinfo = streaminfo[targmwext_name]
        elif targmwext_name in all_dwarf_names:
            thisinfo = dwarfinfo[targmwext_name]
        thisrank = thisinfo['TARGMWEXT_RANK']
        # CMR go through all higher ranking streams and dwarfs and check for dups,
        # find and resolve by subtarget_class.
        # We should not have duplicates between dwarfs, but it is easy and useful to check
        allchecknames = np.concatenate((all_dwarf_names, all_stream_names))
        for icheckname in allchecknames:
            if icheckname in all_dwarf_names:
                iinfo = dwarfinfo[icheckname]
            elif icheckname in all_stream_names:
                iinfo = streaminfo[icheckname]
            icheckrank = iinfo['TARGMWEXT_RANK']
            icheckbit_name = f"MWS_{icheckname}"
            # now figure out which bright_pm bits are set
            target_bit_set = targmwextpar["TARGET_BIT_SET"]
            if target_bit_set == "STREAM":
                brightpm1bit = mws_mask.MWS_STREAM_PM1
                brightpm2bit = mws_mask.MWS_STREAM_PM2
                brightpm3bit = mws_mask.MWS_STREAM_PM3
            elif target_bit_set == "DSPH":
                brightpm1bit = mws_mask.MWS_DSPH_PM1
                brightpm2bit = mws_mask.MWS_DSPH_PM2
                brightpm3bit = mws_mask.MWS_DSPH_PM3
            elif target_bit_set == "UFD":
                brightpm1bit = mws_mask.MWS_UFD_PM1
                brightpm2bit = mws_mask.MWS_UFD_PM2
                brightpm3bit = mws_mask.MWS_UFD_PM3
                
            if icheckrank < thisrank:  # we should not have new dups with a lower ranked object
                continue               # as lower-ranked objects get selected later
            # if input stream or dwarf is lower ranking than the object we are checking
            # we need to remove target subclasses that are lower ranked
            iicheckdups = iialldups & mws_mask[bit_name]  # dups for the specifc stream or dwarf we are checking
            if np.sum(iicheckdups) > 0:
                log.info(f"Duplicate targets found in {icheckname} when selecting targets for {targmwext_name}.")
                # Now resolve. See assumption list in the comments about possible duplicate target combinations
                # bright_pm1 and bright_pm2 for the higher ranking object outranks pm_only for the lower rank obj
                # so can't set pm_only for this object
                iduppmonlybrightpm1 = iicheckdups & brightpm1bit & ipm_only 
                if np.sum(iduppmonlybrightpm1) > 0:
                    ipmonly[iduppmonlybrightpm1 != 0] = 0
                iduppmonlybrightpm2 = iicheckdups & ibrightpm2bit & ipm_only
                if np.sum(iduppmonlybrightpm2) > 0:
                    ipmonly[iduppmonlybrightpm2 != 0] = 0
                # our lower ranked object can't set faint_cmd or faint_no_pm or filler 
                # if the higher ranked one has set bright_pm3
                idupbrightpm3faintcmd = iicheckdups & brightpm3bit & ifaint_cmd  
                if np.sum(idupbrightpm3faintcmd) > 0:
                    ifaint_cmd[idupbrightpm3faintcmd != 0] = 0
                idupbrightpm3faintnopm = iicheckdups & brightpm3bit & ifaint_no_pm  
                if np.sum(idupbrightpm3faintnopm) > 0:
                    ifaint_no_pm[idupbrightpm3faintnopm != 0] = 0
                idupbrightpm3filler = iicheckdups & brigthpm3bit & ifiller  
                if np.sum(idupbrightpm3filler) > 0:
                    ifiller[idupbrightpm3filler != 0] = 0
                # our lower ranked objectd cannot set filler if the higher ranked one set faint_no_pm or faint_cmd
                idupfaintnopmfiller = iicheckdups & mws_mask['MWS_FAINT_NO_PM'] & ifiller
                if np.sum(idupfaintnopmfiller) > 0:
                    ifiller[idupfaintnopmfiller !=0] = 0
                idupfaintcmdfiller = iicheckdups & mws_mask['FAINT_NO_PM'] & ifiller
                if np.sum(idupfaintcmdfiller) > 0:
                    ifiller[idupfaintcmdfiller !=0] = 0
                    
    return ibright_pm1, ibright_pm2, ibright_pm3, ipm_only, ifaint_no_pm, ifaint_cmd, ifiller
