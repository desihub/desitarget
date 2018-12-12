# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==================
desitarget.geomask
==================

Utility module with functions for masking circles and ellipses on the sky

"""
from __future__ import (absolute_import, division)
#
import numpy as np
import fitsio
from time import time

from astropy.coordinates import SkyCoord
from astropy import units as u

from desiutil import depend, brick
from desitarget import io
from desitarget.targetmask import desi_mask, targetid_mask
from desitarget.targets import encode_targetid, finalize
from desitarget.internal import sharedmem

import numpy.lib.recfunctions as rfn

from . import __version__ as desitarget_version

import healpy as hp

# ADM fake the matplotlib display so it doesn't die on allocated nodes.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Circle, Ellipse, Rectangle  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402


def ellipse_matrix(r, e1, e2):
    """Calculate transformation matrix from half-light-radius to ellipse

    Parameters
    ----------
    r : :class:`float` or `~numpy.ndarray`
        Half-light radius of the ellipse (ARCSECONDS)
    e1 : :class:`float` or `~numpy.ndarray`
        First ellipticity component of the ellipse
    e2 : :class:`float` or `~numpy.ndarray`
        Second ellipticity component of the ellipse

    Returns
    -------
    :class:`~numpy.ndarray`
        A 2x2 matrix to transform points measured in coordinates of the
        effective-half-light-radius to RA/Dec offset coordinates

    Notes
    -----
        - If a float is passed then the output shape is (2,2,1)
             otherwise it's (2,2,len(r))
        - The parametrization is explained at
             http://legacysurvey.org/dr4/catalogs/
        - Much of the math is taken from:
             https://github.com/dstndstn/tractor/blob/master/tractor/ellipses.py
    """

    # ADM derive the eccentricity from the ellipticity
    # ADM guarding against the option that floats were passed
    e = np.atleast_1d(np.hypot(e1, e2))

    # ADM the position angle in radians and its cos/sin
    theta = np.atleast_1d(np.arctan2(e2, e1) / 2.)
    ct = np.cos(theta)
    st = np.sin(theta)

    # ADM ensure there's a maximum ratio of the semi-major
    # ADM to semi-minor axis, and calculate that ratio
    maxab = 1000.
    ab = np.zeros(len(e))+maxab
    w = np.where(e < 1)
    ab[w] = (1.+e[w])/(1.-e[w])
    w = np.where(ab > maxab)
    ab[w] = maxab

    # ADM convert the half-light radius to degrees
    r_deg = r / 3600.

    # ADM the 2x2 matrix to transform points measured in
    # ADM effective-half-light-radius to RA/Dec offsets
    T = r_deg * np.array([[ct / ab, st], [-st / ab, ct]])

    return T


def ellipse_boundary(RAcen, DECcen, r, e1, e2, nloc=50):
    """Return RA/Dec of an elliptical boundary on the sky

    Parameters
    ----------
    RAcen : :class:`float`
        Right Ascension of the center of the ellipse (DEGREES)
    DECcen : :class:`float`
        Declination of the center of the ellipse (DEGREES)
    r : :class:`float`
        Half-light radius of the ellipse (ARCSECONDS)
    e1 : :class:`float`
        First ellipticity component of the ellipse
    e2 : :class:`float`
        Second ellipticity component of the ellipse
    nloc : :class:`int`, optional, defaults to 50
        the number of locations to generate, equally spaced around the
        periphery of the ellipse

    Returns
    -------
    :class:`~numpy.ndarray`
        Right Ascensions along the boundary of (each) ellipse
    :class:`~numpy.ndarray`
        Declinations along the boundary of (each) ellipse

    Notes
    -----
        - The parametrization is explained at
             http://legacysurvey.org/dr4/catalogs/
        - Much of the math is taken from:
             https://github.com/dstndstn/tractor/blob/master/tractor/ellipses.py
    """
    # ADM Retrieve the 2x2 matrix to transform points measured in
    # ADM effective-half-light-radius to RA/Dec offsets
    T = ellipse_matrix(r, e1, e2)

    # ADM create a circle in effective-half-light-radius
    angle = np.linspace(0, 2.*np.pi, nloc)
    vv = np.vstack([np.sin(angle), np.cos(angle)])

    # ADM transform circle to elliptical boundary
    dra, ddec = np.dot(T[..., 0], vv)

    # ADM return the RA, Dec of the boundary, remembering to correct for
    # ADM spherical projection in Dec
    decs = DECcen + ddec
    # ADM note that this is only true for the small angle approximation
    # ADM but that's OK to < 0.3" for a < 3o diameter galaxy at dec < 60o
    ras = RAcen + (dra/np.cos(np.radians(decs)))

    return ras, decs


def is_in_ellipse(ras, decs, RAcen, DECcen, r, e1, e2):
    """Determine whether points lie within an elliptical mask on the sky

    Parameters
    ----------
    ras : :class:`~numpy.ndarray`
        Array of Right Ascensions to test
    decs : :class:`~numpy.ndarray`
        Array of Declinations to test
    RAcen : :class:`float`
        Right Ascension of the center of the ellipse (DEGREES)
    DECcen : :class:`float`
        Declination of the center of the ellipse (DEGREES)
    r : :class:`float`
        Half-light radius of the ellipse (ARCSECONDS)
    e1 : :class:`float`
        First ellipticity component of the ellipse
    e2 : :class:`float`
        Second ellipticity component of the ellipse

    Returns
    -------
    :class:`boolean`
        An array that is the same length as RA/Dec that is True
        for points that are in the mask and False for points that
        are not in the mask

    Notes
    -----
        - The parametrization is explained at
             http://legacysurvey.org/dr4/catalogs/
        - Much of the math is taken from:
             https://github.com/dstndstn/tractor/blob/master/tractor/ellipses.py
    """

    # ADM Retrieve the 2x2 matrix to transform points measured in
    # ADM effective-half-light-radius to RA/Dec offsets...
    G = ellipse_matrix(r, e1, e2)
    # ADM ...and invert it
    Ginv = np.linalg.inv(G[..., 0])

    # ADM remember to correct for the spherical projection in Dec
    # ADM note that this is only true for the small angle approximation
    # ADM but that's OK to < 0.3" for a < 3o diameter galaxy at dec < 60o
    dra = (ras - RAcen)*np.cos(np.radians(decs))
    ddec = decs - DECcen

    # ADM test whether points are larger than the effective
    # ADM circle of radius 1 generated in half-light-radius coordinates
    dx, dy = np.dot(Ginv, [dra, ddec])

    return np.hypot(dx, dy) < 1


def is_in_ellipse_matrix(ras, decs, RAcen, DECcen, G):
    """Determine whether points lie within an elliptical mask on the sky

    Parameters
    ----------
    ras : :class:`~numpy.ndarray`
        Array of Right Ascensions to test
    decs : :class:`~numpy.ndarray`
        Array of Declinations to test
    RAcen : :class:`float`
        Right Ascension of the center of the ellipse (DEGREES)
    DECcen : :class:`float`
        Declination of the center of the ellipse (DEGREES)
    G : :class:`~numpy.ndarray`
        A 2x2 matrix to transform points measured in coordinates of the
        effective-half-light-radius to RA/Dec offset coordinates
        as generated by, e.g., :mod:`desitarget.geomask.ellipse_matrix`

    Returns
    -------
    :class:`boolean`
        An array that is the same length as ras/decs that is True
        for points that are in the mask and False for points that
        are not in the mask

    Notes
    -----
        - The parametrization is explained at
             http://legacysurvey.org/dr4/catalogs/
        - Much of the math is taken from:
             https://github.com/dstndstn/tractor/blob/master/tractor/ellipses.py
        - G should have a shape of (2,2) (i.e. np.shape(G) gives (2,2)
    """

    # ADM Invert the transformation matrix
    Ginv = np.linalg.inv(G)

    # ADM remember to correct for the spherical projection in Dec
    # ADM note that this is only true for the small angle approximation
    # ADM but that's OK to < 0.3" for a < 3o diameter galaxy at dec < 60o
    dra = (ras - RAcen)*np.cos(np.radians(decs))
    ddec = decs - DECcen

    # ADM test whether points are larger than the effective
    # ADM circle of radius 1 generated in half-light-radius coordinates
    dx, dy = np.dot(Ginv, [dra, ddec])

    return np.hypot(dx, dy) < 1


def is_in_circle(ras, decs, RAcens, DECcens, r):
    """Determine whether a set of points is in a set of circular masks on the sky

    Parameters
    ----------
    ras : :class:`~numpy.ndarray`
        Array of Right Ascensions to test
    decs : :class:`~numpy.ndarray`
        Array of Declinations to test
    RAcen : :class:`~numpy.ndarray`
        Right Ascension of the centers of the circles (DEGREES)
    DECcen : :class:`~numpy.ndarray`
        Declination of the centers of the circles (DEGREES)
    r : :class:`~numpy.ndarray`
        Radius of the circles (ARCSECONDS)

    Returns
    -------
    :class:`boolean`
        An array that is the same length as RA/Dec that is True
        for points that are in any of the masks and False for points that
        are not in any of the masks
    """

    # ADM initialize an array of all False (nothing is yet in a circular mask)
    in_mask = np.zeros(len(ras), dtype=bool)

    # ADM turn the coordinates of the masks and the targets into SkyCoord objects
    ctargs = SkyCoord(ras*u.degree, decs*u.degree)
    cstars = SkyCoord(RAcens*u.degree, DECcens*u.degree)

    # ADM this is the largest search radius we should need to consider
    # ADM in the future an obvious speed up is to split on radius
    # ADM as large radii are rarer but take longer
    maxrad = max(r)*u.arcsec

    # ADM coordinate match the star masks and the targets
    idtargs, idstars, d2d, d3d = cstars.search_around_sky(ctargs, maxrad)

    # ADM catch the case where nothing fell in a mask
    if len(idstars) == 0:
        return in_mask

    # ADM for a matching star mask, find the angular separations that are less than the mask radius
    w_in = np.where(d2d.arcsec < r[idstars])

    # ADM any matching idtargs that meet this separation criterion are in a mask (at least one)
    in_mask[idtargs[w_in]] = 'True'

    return in_mask


def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """Make a scatter plot of circles. Similar to plt.scatter, but the size of circles are in data scale

    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    >>> a = np.arange(11)
    >>> circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    >>> plt.colorbar()

    References
    ----------
    With thanks to https://gist.github.com/syrte/592a062c562cd2a98a83
    """

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    patches = [Circle((x_, y_), s_)
               for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def ellipses(x, y, w, h=None, rot=0.0, c='b', vmin=None, vmax=None, **kwargs):
    """Make a scatter plot of ellipses

    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Center of ellipses.
    w, h : scalar or array_like, shape (n, )
        Total length (diameter) of horizontal/vertical axis.
        `h` is set to be equal to `w` by default, ie. circle.
    rot : scalar or array_like, shape (n, )
        Rotation in degrees (anti-clockwise).
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    >>> a = np.arange(11)
    >>> ellipses(a, a, w=4, h=a, rot=a*30, c=a, alpha=0.5, ec='none')
    >>> plt.colorbar()

    References
    ----------
    With thanks to https://gist.github.com/syrte/592a062c562cd2a98a83
    """

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    if h is None:
        h = w

    zipped = np.broadcast(x, y, w, h, rot)
    patches = [Ellipse((x_, y_), w_, h_, rot_)
               for x_, y_, w_, h_, rot_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def cap_area(theta):
    """True area of a circle of a given radius drawn on the surface of a sphere

    Parameters
    ----------
    theta : array_like
        (angular) radius of a circle drawn on the surface of the unit sphere (in DEGREES)

    Returns
    -------
    area : array_like
       surface area on the sphere included within the passed angular radius

    Notes
    -----
        - The approximate formula pi*theta**2 is only accurate to ~0.0025% at 1o, ~0.25% at 10o,
          sufficient for bright source mask purposes. But the equation in this function is more general.
        - We recast the input array as float64 to circumvent precision issues with np.cos()
          when radii of only a few arcminutes are passed
        - Even for passed radii of 1 (0.1) arcsec, float64 is sufficiently precise to give the correct
          area to ~0.00043 (~0.043%) using np.cos()
    """

    # ADM recast input array as float64
    theta = theta.astype('<f8')

    # ADM factor to convert steradians to sq.deg.
    st2sq = 180.*180./np.pi/np.pi

    # ADM return area
    return st2sq*2*np.pi*(1-(np.cos(np.radians(theta))))


def sphere_circle_ra_off(theta, centdec, declocs):
    """Offsets in RA needed for given declinations in order to draw a (small) circle on the sphere

    Parameters
    ----------
    theta : :class:`float`
        (angular) radius of a circle drawn on the surface of the unit sphere (in DEGREES)

    centdec : :class:`float`
        declination of the center of the circle to be drawn on the sphere (in DEGREES)

    declocs : array_like
        declinations of positions on the boundary of the circle at which to calculate RA offsets (in DEGREES)

    Returns
    -------
    raoff : array_like
        offsets in RA that correspond to the passed dec locations for the given dec circle center (IN DEGREES)

    Notes
    -----
        - This function is ambivalent to the SIGN of the offset. In other words, it can only draw the semi-circle
          in theta from -90o->90o, which corresponds to offsets in the POSITIVE RA direction. The user must determine
          which offsets are to the negative side of the circle, or call this function twice.
    """

    # ADM convert the input angles from degrees to radians
    # ADM cast theta as float 64 to help deal with the cosine of very small angles
    thetar = np.radians(theta).astype('<f8')
    centdecr = np.radians(centdec)
    declocsr = np.radians(declocs)

    # ADM determine the offsets in RA from the small circle equation (easy to derive from, e.g. converting
    # ADM to Cartesian coordinates and using dot products). The answer is the arccos of the following:
    cosoffrar = (np.cos(thetar) - (np.sin(centdecr)*np.sin(declocsr))) / (np.cos(centdecr)*np.cos(declocsr))

    # ADM catch cases where the offset angle is very close to 0
    offrar = np.arccos(np.clip(cosoffrar, -1, 1))

    # ADM return the angular offsets in degrees
    return np.degrees(offrar)


def circle_boundaries(RAcens, DECcens, r, nloc):
    """Return RAs/Decs of a set of circular boundaries on the sky

    Parameters
    ----------
    RAcens : :class:`~numpy.ndarray`
        Right Ascension of the centers of the circles (DEGREES)
    DECcens : :class:`~numpy.ndarray`
        Declination of the centers of the circles (DEGREES)
    r : :class:`~numpy.ndarray`
        radius of the circles (ARCSECONDS)
    nloc : :class:`~numpy.ndarray`
        the number of locations to generate, equally spaced around the
        periphery of *each* circle

    Returns
    -------
    ras : :class:`~numpy.ndarray`
        The Right Ascensions of nloc equally spaced locations on the
            periphery of the mask
    dec : array_like.
        The Declinations of nloc equally space locations on the periphery
            of the mask
    """

    # ADM the radius of each mask in degrees with a 0.1% kick to get things beyond the mask edges
    radius = 1.001*r/3600.

    # ADM determine nloc Dec offsets equally spaced around the perimeter for each mask
    offdec = [rad*np.sin(np.arange(ns)*2*np.pi/ns) for ns, rad in zip(nloc, radius)]

    # ADM use offsets to determine DEC positions
    decs = DECcens + offdec

    # ADM determine the offsets in RA at these Decs given the mask center Dec
    offrapos = [sphere_circle_ra_off(th, cen, declocs) for th, cen, declocs in zip(radius, DECcens, decs)]

    # ADM determine which of the RA offsets are in the positive direction
    sign = [np.sign(np.cos(np.arange(ns)*2*np.pi/ns)) for ns in nloc]

    # ADM determine the RA offsets with the appropriate sign and add them to the RA of each mask
    offra = [o*s for o, s in zip(offrapos, sign)]
    ras = RAcens + offra

    # ADM have to turn the generated locations into 1-D arrays before returning them
    return np.hstack(ras), np.hstack(decs)
