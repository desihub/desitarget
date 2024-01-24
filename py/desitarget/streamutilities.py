"""
desitarget.streamutilties
=========================

Utilities for the DESI MWS Stellar Stream programs.

Borrows heavily from Sergey Koposov's `astrolibpy routines`_.

.. _`astrolibpy routines`: https://github.com/segasai/astrolibpy/blob/master/my_utils
"""
import numpy as np
import astropy.coordinates as acoo
import astropy.units as auni

# ADM Galactic reference frame. Use astropy v4.0 defaults.
GCPARAMS = acoo.galactocentric_frame_defaults.get_from_registry(
    "v4.0")['parameters']

# ADM some standard units.
kms = auni.km / auni.s
masyr = auni.mas / auni.year


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
    """Corrects proper motions for the Sun's motion.

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
