# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==================
desitarget.geomask
==================

Utility functions for geometry on the sky, masking, etc.

.. _`this post on Stack Overflow`: https://stackoverflow.com/questions/7392143/python-implementations-of-packing-algorithm
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import os
import fitsio
from time import time

from astropy.coordinates import SkyCoord
from astropy import units as u

from desiutil import depend, brick
from desitarget.targetmask import desi_mask, targetid_mask
from desitarget.targets import encode_targetid, finalize
from desitarget.internal import sharedmem

import numpy.lib.recfunctions as rfn

import healpy as hp

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()


def get_imaging_maskbits(bitnamelist=None):
    """Return MASKBITS names and bits from the Legacy Surveys.

    Parameters
    ----------
    bitnamelist : :class:`list`, optional, defaults to ``None``
        If not ``None``, return the bit values corresponding to the
        passed names. Otherwise, return the full MASKBITS dictionary.

    Returns
    -------
    :class:`list` or `dict`
        A list of the MASKBITS values if `bitnamelist` is passed,
        otherwise the full MASKBITS dictionary of names-to-values.

    Notes
    -----
        - For the definitions of the mask bits, see, e.g.,
             https://www.legacysurvey.org/dr8/bitmasks/#maskbits
    """
    bitdict = {"BRIGHT": 1, "ALLMASK_G": 5, "ALLMASK_R": 6, "ALLMASK_Z": 7,
               "BAILOUT": 10, "MEDIUM": 11, "GALAXY": 12, "CLUSTER": 13}

    # ADM look up the bit value for each passed bit name.
    if bitnamelist is not None:
        return [bitdict[bitname] for bitname in bitnamelist]

    return bitdict


def get_default_maskbits(bgs=False):
    """Return the names of the default MASKBITS for targets.

    Parameters
    ----------
    bgs : :class:`bool`, defaults to ``False``.
        If ``True`` load the "default" scheme for Bright Galaxy Survey
        targets. Otherwise, load the default for other target classes.

    Returns
    -------
    :class:`list`
        A list of the default MASKBITS names for targets.
    """
    if bgs:
        return ["BRIGHT", "CLUSTER"]
    return ["BRIGHT", "GALAXY", "CLUSTER"]


def imaging_mask(maskbits, bitnamelist=get_default_maskbits(), bgsmask=False):
    """Apply the 'geometric' masks from the Legacy Surveys imaging.

    Parameters
    ----------
    maskbits : :class:`~numpy.ndarray` or ``None``
        General array of `Legacy Surveys mask`_ bits.
    bitnamelist : :class:`list`, defaults to func:`get_default_maskbits()`
        List of Legacy Surveys mask bits to set to ``False``.
    bgsmask : :class:`bool`, defaults to ``False``.
        Load the "default" scheme for Bright Galaxy Survey targets.
        Overrides `bitnamelist`.

    Returns
    -------
    :class:`~numpy.ndarray`
        A boolean array that is the same length as `maskbits` that
        contains ``False`` where any bits in `bitnamelist` are set.
    """
    # ADM default for the BGS.
    if bgsmask:
        bitnamelist = get_default_maskbits(bgs=True)

    # ADM get the bit values for the passed (or default) bit names.
    bits = get_imaging_maskbits(bitnamelist)

    # ADM Create array of True and set to False where a mask bit is set.
    mb = np.ones_like(maskbits, dtype='?')
    for bit in bits:
        mb &= ((maskbits & 2**bit) == 0)

    return mb


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
        An array that is the same length as RA/Dec that is ``True``
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
        An array that is the same length as ras/decs that is ``True``
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
    """Whether a set of points is in a set of circular masks on the sky.

    Parameters
    ----------
    ras : :class:`~numpy.ndarray`
        Array of Right Ascensions to test.
    decs : :class:`~numpy.ndarray`
        Array of Declinations to test.
    RAcen : :class:`~numpy.ndarray`
        Right Ascension of the centers of the circles (DEGREES).
    DECcen : :class:`~numpy.ndarray`
        Declination of the centers of the circles (DEGREES).
    r : :class:`~numpy.ndarray`
        Radius of the circles (ARCSECONDS).

    Returns
    -------
    :class:`boolean`
        An array that is the same length as RA/Dec that is ``True``
        for points that are in any of the masks and False for points
        that are not in any of the masks.
    """

    # ADM all matches start as False (nothing is yet in a circular mask).
    in_mask = np.zeros(len(ras), dtype=bool)

    # ADM coordinates of masks and targets into SkyCoord objects.
    ctargs = SkyCoord(ras*u.degree, decs*u.degree)
    cstars = SkyCoord(RAcens*u.degree, DECcens*u.degree)

    # ADM this is the largest search radius we should need to consider
    # ADM in the future an obvious speed up is to split on radius
    # ADM as large radii are rarer but take longer.
    maxrad = max(r)*u.arcsec

    # ADM coordinate match the star masks and the targets.
    idtargs, idstars, d2d, d3d = cstars.search_around_sky(ctargs, maxrad)

    # ADM catch the case where nothing fell in a mask.
    if len(idstars) == 0:
        return in_mask

    # ADM for a match, find separations less than the mask radius.
    w_in = np.where(d2d.arcsec < r[idstars])

    # ADM matches at less than the radius are in a mask (at least one).
    in_mask[idtargs[w_in]] = True

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
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

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
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.collections import PatchCollection

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


def box_area(radecbox):
    """Determines the area of an RA, Dec box in square degrees.

    Parameters
    ----------
    radecbox : :class:`list`
        4-entry list of coordinates [ramin, ramax, decmin, decmax] forming the
        edges of a box in RA/Dec (degrees).

    Returns
    -------
    :class:`list`
        The area of the passed box in square degrees.
    """
    ramin, ramax, decmin, decmax = radecbox

    # ADM check for some common mistakes.
    if decmin < -90. or decmax > 90. or decmax <= decmin or ramax <= ramin:
        msg = "Strange input: [ramin, ramax, decmin, decmax] = {}".format(radecbox)
        log.critical(msg)
        raise ValueError(msg)

    sindecmin, sindecmax = np.sin(np.radians(decmin)), np.sin(np.radians(decmax))
    spharea = (ramax-ramin)*np.degrees(sindecmax-sindecmin)

    return spharea


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
    # ADM radius in degrees with a 0.1% kick to push beyond the edge.
    radius = 1.001*r/3600.

    # ADM nloc Dec offsets equally spaced around the circle perimeter.
    offdec = np.array([rad*np.sin(np.arange(ns)*2*np.pi/ns)
                       for ns, rad in zip(nloc, radius)]).transpose()

    # ADM use offsets to determine DEC positions.
    decs = DECcens + offdec

    # ADM offsets in RA at these Decs given the mask center Dec.
    offrapos = [sphere_circle_ra_off(th, cen, declocs)
                for th, cen, declocs in zip(radius, DECcens, decs.transpose())]

    # ADM determine which RA offsets are in the positive direction.
    sign = [np.sign(np.cos(np.arange(ns)*2*np.pi/ns)) for ns in nloc]

    # ADM add RA offsets with the right sign to the the circle center.
    offra = np.array([o*s for o, s in zip(offrapos, sign)]).transpose()
    ras = RAcens + offra

    # ADM return the results as 1-D arrays.
    return np.hstack(ras), np.hstack(decs)


def bundle_bricks(pixnum, maxpernode, nside, brickspersec=1., prefix='targets',
                  gather=False, surveydirs=None, extra=None, seed=None,
                  nchunks=10):
    """Determine the optimal packing for bricks collected by HEALpixel integer.

    Parameters
    ----------
    pixnum : :class:`np.array`
        List of integers, e.g., HEALPixel numbers occupied by a set of
        bricks (e.g. array([16, 16, 16...12 , 13, 19]) ).
    maxpernode : :class:`int`
        The maximum number of pixels to bundle (e.g., if you were trying
        to pass `maxpernode` bricks, delineated by the HEALPixels they
        occupy, parallelized across a set of nodes).
    nside : :class:`int`
        The HEALPixel nside number thaat was used to generate `pixnum`
        (NESTED scheme).
    brickspersec : :class:`float`, optional, defaults to 1.
        The rough number of bricks processed per second by the code
        (parallelized across a chosen number of nodes)
    prefix : :class:`str`, optional, defaults to 'targets'
        Corresponds to the executable "X" that is run as select_X for a
        target type. This could be 'randoms', 'skies', 'targets', 'gfas'.
        Also, 'supp-skies' can be passed to cover supplemental skies.
    gather : :class:`bool`, optional, defaults to ``False``
        If ``True`` add a command to combine all the HEALPix-split files
        into one large file. If ``False``, do not provide that command.
        ONLY creates correct file names to gather RANDOMS files!
    surveydirs : :class:`list`
        Root directories for a Legacy Surveys Data Release. Item 1 is
        used as the main directory. IF the list is length-2 then the
        second directory is added as "-s2" in the output script (e.g.
        ["/global/project/projectdirs/cosmo/data/legacysurvey/dr6"]).
    extra : :class:`str`, optional
        Extra command line flags to be passed to the executable lines in
        the output slurm script.
    seed : :class:`int`, optional, defaults to 1
        Random seed for file name. Only relevant for `prefix='randoms'`.
    nchunks : :class:`int`, optional, defaults to 10
        Number of smaller catalogs to split the random catalog into. Only
        relevant for `prefix='randoms'`.

    Returns
    -------
    Nothing, but prints commands that would facilitate running a set of
    bricks by HEALPixel integer with the total number of bricks not to
    exceed `maxpernode`. Also prints total bricks on each node.

    Notes
    -----
        - For the packing algorithm see `this post on Stack Overflow`_.
    """
    # ADM interpret the passed directories.
    surveydir = os.path.normpath(surveydirs[0])
    surveydir2 = None
    if len(surveydirs) == 2:
        surveydir2 = os.path.normpath(surveydirs[1])

    # ADM the number of pixels (numpix) in each pixel (pix).
    pix, numpix = np.unique(pixnum, return_counts=True)

    # ADM the indices needed to reverse-sort the array on number of pixels,
    reverse_order = np.flipud(np.argsort(numpix))
    numpix = numpix[reverse_order]
    pix = pix[reverse_order]

    # ADM iteratively populate lists of the numbers of pixels
    # ADM and the corrsponding pixel numbers,
    # ADM only allow true bundling for skies and randoms.
    if prefix in ['skies', 'randoms']:
        bins = []
        for index, num in enumerate(numpix):
            # Try to fit this sized number into a bin
            for bin in bins:
                if np.sum(np.array(bin)[:, 0]) + num <= maxpernode:
                    # print 'Adding', item, 'to', bin
                    bin.append([num, pix[index]])
                    break
            else:
                # item didn't fit into any bin, start a new bin.
                bin = []
                bin.append([num, pix[index]])
                bins.append(bin)
        # ADM print to screen in the form of a slurm bash script, and
        # ADM other useful information.
        print("#######################################################")
        print("Numbers of bricks or files in each set of HEALPixels:")
        print("")

        # ADM the estimated margin for writing to disk in minutes.
        margin = 30
        if prefix == 'skies':
            margin = 5
        if prefix == 'randoms':
            margin = 90
        margin /= 60.

        maxeta = 0
        for bin in bins:
            num = np.array(bin)[:, 0]
            pix = np.array(bin)[:, 1]
            wpix = np.where(num > 0)[0]
            if len(wpix) > 0:
                goodpix, goodnum = pix[wpix], num[wpix]
                sorter = goodpix.argsort()
                goodpix, goodnum = goodpix[sorter], goodnum[sorter]
                outnote = ['{}: {}'.format(pix, num) for pix, num in zip(goodpix, goodnum)]
                # ADM add the total across all of the pixels
                outnote.append('Total: {}'.format(np.sum(goodnum)))
                # ADM a crude estimate of how long the script will take to run
                # ADM brickspersec is bricks/sec. Extra delta is minutes to write to disk.
                delta = 3./60.
                eta = delta + np.sum(goodnum)/brickspersec/3600
                outnote.append('Estimated time to run in hours (for 32 processors per node): {:.2f}h'
                               .format(eta))
                # ADM track the maximum estimated time for shell scripts, etc.
                if (eta+margin).astype(int) + 1 > maxeta:
                    maxeta = (eta+margin).astype(int) + 1
                print(outnote)

        print("")
        if gather:
            print('Estimated additional margin for writing to disk in hours: {:.2f}h'
                  .format(margin))
            print("")
        nnodes = len(bins)
    else:
        nbins = hp.nside2npix(nside)
        bins = [[[i, j]] for i, j in
                zip(np.ones(nbins, dtype='int'), np.arange(nbins))]
        maxeta = 1
        nnodes = min(16, len(bins))
        if prefix == 'supp-skies':
            nnodes = 4

    # ADM more than 48 nodes is a mistake!
    if nnodes > 48:
        nnodes = 48

    print("#######################################################")
    print("Possible salloc command if you want to run on the Cori interactive queue:")
    print("")
    print("salloc -N {} -C haswell -t 0{}:00:00 --qos interactive -L SCRATCH,project"
          .format(nnodes, maxeta))

    print("")
    print("#######################################################")
    print('Example shell script for slurm:')
    print('')
    print('#!/bin/bash -l')
    print('#SBATCH -q regular')
    print('#SBATCH -N {}'.format(nnodes))
    print('#SBATCH -t 0{}:00:00'.format(maxeta))
    print('#SBATCH -L SCRATCH,project')
    print('#SBATCH -C haswell')
    print('')

    # ADM extract the Data Release number from the survey directory
    dr = surveydir.split('dr')[-1][0]
    # ADM if an integer can't be extracted, use X instead.
    try:
        drstr = "-dr{}".format(int(dr))
    except ValueError:
        drstr = ""

    # ADM to handle inputs that look like "svX_targets".
    prefix2 = prefix
    if prefix[0:2] == "sv":
        prefix2 = "sv_targets"

    s2 = ""
    if surveydir2 is not None:
        s2 = "-s2 {}".format(surveydir2)

    cmd = "select"
    if prefix == "supp-skies":
        cmd = "supplement"
        prefix2 = "skies"

    from desitarget.io import _check_hpx_length, find_target_files

    pixtracker = []
    for bin in bins:
        num = np.array(bin)[:, 0]
        pix = np.array(bin)[:, 1]
        wpix = np.where(num > 0)[0]
        if len(wpix) > 0:
            goodpix = pix[wpix]
            goodpix.sort()
            # ADM check that we won't overwhelm the pixel scheme.
            _check_hpx_length(goodpix)
            strgoodpix = ",".join([str(pix) for pix in goodpix])
            pixtracker.append(strgoodpix)
            if extra is not None:
                strgoodpix += extra
            print("srun -N 1 {}_{} {} $CSCRATCH {} --nside {} --healpixels {} &"
                  .format(cmd, prefix2, surveydir, s2, nside, strgoodpix))
    print("wait")
    print("")
    if gather:
        ddrr = drstr.replace("-", "")
        for resolve, region, skip in zip([True, False, False],
                                         [None, "north", "south"],
                                         ["", "--skip", "--skip"]):
            outfiles = []
            for pix in pixtracker:
                outfn = find_target_files(
                    "$CSCRATCH", dr=ddrr, flavor=prefix, seed=seed, hp=pix,
                    resolve=resolve, region=region)
                outfiles.append(outfn)
            outfn = find_target_files(
                "$CSCRATCH", dr=ddrr, flavor=prefix, seed=seed, nohp=True,
                resolve=resolve, region=region)
            print("")
            # ADM split each pixel-file into 10 smaller catalogs.
            for fn in outfiles:
                adder = ""
                # ADM we'll need to add the MTL columns if they aren't
                # ADM added when the randoms are initially constructed.
                if "addmtl" not in extra:
                    adder = "--addmtl"
                    print("srun -N 1 split_randoms {} -n {} {} {} &".format(
                        fn, nchunks, adder, skip))
            print("")
            print("wait")
            print("")
            for nchunk in range(nchunks):
                ofs = [fn.replace(".fits", "-{}.fits".format(nchunk))
                       for fn in outfiles]
                ofn = outfn.replace(".fits", "-{}.fits".format(nchunk))
                print("srun -N 1 gather_targets '{}' {} {} {} &".format(
                    ";".join(ofs), ofn, prefix2.split("_")[-1], skip))
        print("")
        print("wait")
        print("")

    return


def add_hp_neighbors(nside, pixnum):
    """Add all neighbors in the NESTED scheme to a set of HEALPixels.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    pixnum : :class:`list` or `int` or `~numpy.ndarray`
        HEALPixel numbers (or a single HEALPixel number).

    Returns
    -------
    :class:`list`
        The passed list of pixels with all neighbors added to the list.
        Only unique pixels are returned, so any duplicate integers in
        the passed `pixnum` are removed.

    Notes
    -----
        - Syntactic sugar around `healpy.pixelfunc.get_all_neighbours()`.
    """
    # ADM convert pixels to theta/phi space and retrieve neighbors.
    theta, phi = hp.pix2ang(nside, pixnum, nest=True)
    # ADM remember to retain the original pixel numbers, too.
    pixnum = np.hstack(
        [pixnum, np.hstack(
            hp.pixelfunc.get_all_neighbours(nside, theta, phi, nest=True)
        )]
    )

    # ADM retrieve only the UNIQUE pixel numbers. It's possible that only
    # ADM one pixel was produced, so guard against pixnum being non-iterable.
    if not isinstance(pixnum, np.integer):
        pixnum = list(set(pixnum))
    else:
        pixnum = [pixnum]

    # ADM there are pixels with no neighbors, which returns -1. Remove these:
    if -1 in pixnum:
        pixnum.remove(-1)

    return pixnum


def brick_names_touch_hp(nside, numproc=1, fact=2**20):
    """Determine which of a set of brick names touch a set of HEALPixels.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    numproc : :class:`int`, optional, defaults to 1
        The number of parallel processes to use.
    fact : :class:`int`, optional defaults to 2**20
        see documentation for `healpy.query_polygon()`.

    Returns
    -------
    :class:`list`
        A list of lists of input brick names that touch each HEALPixel
        at `nside`. So, e.g. for `nside=2` the returned list will have
        48 entries, and, for example, output[0] will be a list of names
        of bricks that touch HEALPixel 0.

    Notes
    -----
        - Runs in ~65 (10) secs for numproc=1 (32) at nside=2 (fact=4).
        - Runs in ~325 (20) secs for numproc=1 (32) at nside=64 (fact=4).
        - Takes ~2x as long at the default fact=2**20 compared to fact=4,
          but fact=2**20 returns far fewer bricks for small `nside`.
    """
    t0 = time()
    # ADM grab the standard table of bricks.
    bricktable = brick.Bricks(bricksize=0.25).to_table()

    def _make_lookupdict(indexes):
        """for a set of indexes that correspond to bricktable rows, make
        a look-up dictionary of which pixels touch each brick"""

        lookupdict = {bt["BRICKNAME"]: hp_in_box(
            nside, [bt["RA1"], bt["RA2"], bt["DEC1"], bt["DEC2"]], fact=fact
        ) for bt in bricktable[indexes]}

        return lookupdict

    # ADM split the length of the bricktable into arrays of indexes.
    indexes = np.array_split(np.arange(len(bricktable)), numproc)

    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            lookupdict = pool.map(_make_lookupdict, indexes)
        lookupdict = {key: val for lud in lookupdict for key, val in lud.items()}
    else:
        lookupdict = _make_lookupdict(indexes[0])

    # ADM change the pixels-in-brick look-up table to a
    # ADM bricks-in-pixel look-up table.
    bricksperpixel = [[] for pix in range(hp.nside2npix(nside))]
    for brickname in lookupdict:
        for pixel in lookupdict[brickname]:
            bricksperpixel[pixel].append(brickname)

    log.info("Done...t = {:.1f}s".format(time()-t0))

    return bricksperpixel


def sweep_files_touch_hp(nside, pixlist, infiles):
    """Determine which of a set of sweep files touch a set of HEALPixels.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    pixlist : :class:`list` or `int`
        A set of HEALPixels at `nside`.
    infiles : :class:`list` or `str`
        A list of input (sweep filenames) OR a single filename.

    Returns
    -------
    :class:`list`
        A list of lists of input sweep files that touch each HEALPixel
        at `nside`. So, e.g. for `nside=2` the returned list will have
        48 entries, and, for example, output[0] will be a list of files
        that touch HEALPixel 0.
    :class:`list`
        The input `pixlist` reduced to just those pixels that touch
        the area covered by the input `infiles`.
    :class:`~numpy.ndarray`
        A flattened array of all HEALPixels touched by the input
        `infiles`. Each HEALPixel will appear multiple times if it's
        touched by multiple input sweep files.
    """
    # ADM convert a single filename to list of filenames.
    if isinstance(infiles, str):
        infiles = [infiles, ]

    # ADM work with pixlist as an array.
    pixlist = np.atleast_1d(pixlist)

    # ADM sanity check that nside is OK.
    check_nside(nside)

    # ADM a list of HEALPixels that touch each file.
    from desitarget.io import decode_sweep_name
    pixelsperfile = [decode_sweep_name(fn, nside=nside) for fn in infiles]

    # ADM a flattened array of all HEALPixels touched by the input
    # ADM files. Each HEALPixel will appear multiple times if it's
    # ADM touched by multiple input sweep files.
    pixnum = np.hstack(pixelsperfile)

    # ADM restrict input pixels to only those that touch an input file.
    ii = [pix in pixnum for pix in pixlist]
    pixlist = pixlist[ii]

    # ADM create a list of files that touch each HEALPixel.
    filesperpixel = [[] for pix in range(hp.nside2npix(nside))]
    for ifile, pixels in enumerate(pixelsperfile):
        for pix in pixels:
            filesperpixel[pix].append(infiles[ifile])

    return filesperpixel, pixlist, pixnum


def hp_in_box(nside, radecbox, inclusive=True, fact=4):
    """Determine which HEALPixels touch an RA, Dec box.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    radecbox : :class:`list`
        4-entry list of coordinates [ramin, ramax, decmin, decmax]
        forming the edges of a box in RA/Dec (degrees).
    inclusive : :class:`bool`, optional, defaults to ``True``
        see documentation for `healpy.query_polygon()`.
    fact : :class:`int`, optional defaults to 4
        see documentation for `healpy.query_polygon()`.

    Returns
    -------
    :class:`list`
        HEALPixels at the passed `nside` that touch the RA/Dec box.

    Notes
    -----
        - Uses `healpy.query_polygon()` to retrieve the RA geodesics
          and then :func:`hp_in_dec_range()` to limit by Dec.
        - When the RA range exceeds 180o, `healpy.query_polygon()`
          defines the range as that with the smallest area (i.e the box
          can wrap-around in RA). To avoid any ambiguity, this function
          will only limit by the passed Decs in such cases.
        - Only strictly correct for Decs from -90+1e-3(o) to 90-1e3(o).
    """
    ramin, ramax, decmin, decmax = radecbox

    # ADM area enclosed isn't well-defined if RA covers more than 180o.
    if np.abs(ramax-ramin) <= 180.:
        # ADM retrieve RA range. The 1e-3 prevents edge effects near poles.
        npole, spole = 90-1e-3, -90+1e-3
        # ADM convert RA/Dec to co-latitude and longitude in radians.
        rapairs = np.array([ramin, ramin, ramax, ramax])
        decpairs = np.array([spole, npole, npole, spole])
        thetapairs, phipairs = np.radians(90.-decpairs), np.radians(rapairs)

        # ADM convert to Cartesian vectors remembering to transpose
        # ADM to pass the array to query_polygon in the correct order.
        vecs = hp.dir2vec(thetapairs, phipairs).T

        # ADM determine the pixels that touch the RA range.
        pixra = hp.query_polygon(nside, vecs,
                                 inclusive=inclusive, fact=fact, nest=True)
    else:
        log.warning('Max RA ({}) and Min RA ({}) separated by > 180o...'
                    .format(ramax, ramin))
        log.warning('...will only limit to passed Declinations'
                    .format(nside))
        pixra = np.arange(hp.nside2npix(nside))

    # ADM determine the pixels that touch the Dec range.
    pixdec = hp_in_dec_range(nside, decmin, decmax, inclusive=inclusive)

    # ADM return the pixels in the box.
    pixnum = list(set(pixra).intersection(set(pixdec)))

    return pixnum


def hp_in_dec_range(nside, decmin, decmax, inclusive=True):
    """HEALPixels in a specified range of Declination.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    decmin, decmax : :class:`float`
        Declination range (degrees).
    inclusive : :class:`bool`, optional, defaults to ``True``
        see documentation for `healpy.query_strip()`.

    Returns
    -------
    :class:`list`
        (Nested) HEALPixels at `nside` in the specified Dec range.

    Notes
    -----
        - Just syntactic sugar around `healpy.query_strip()`.
        - `healpy.query_strip()` isn't implemented for the NESTED scheme
          in early healpy versions, so this queries in the RING scheme
          and then converts to the NESTED scheme.
    """
    # ADM convert Dec to co-latitude in radians.
    # ADM remember that, min/max swap because of the -ve sign.
    thetamin = np.radians(90.-decmax)
    thetamax = np.radians(90.-decmin)

    # ADM determine the pixels that touch the box.
    pixring = hp.query_strip(nside, thetamin, thetamax,
                             inclusive=inclusive, nest=False)
    pixnest = hp.ring2nest(nside, pixring)

    return pixnest


def hp_beyond_gal_b(nside, mingalb, neighbors=True):
    """Find HEALPixels with centers and neighbors beyond a Galactic b.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    mingalb : :class:`float`
        Closest latitude to Galactic plane to return HEALPixels
        (e.g. send 10 to limit to pixels beyond -10o <= b < 10o).
    neighbors : :class:`bool`, optional, defaults to ``True``
        If ``False``, return all HEALPixels with *centers* beyond the
        passed `mingalb`. If ``True`` also return the neighbors of these
        pixels (to avoid edge effects).

    Returns
    -------
    :class:`list`
        HEALPixels at the passed `nside` that lie north and south of the
        passed `mingalb`.

    Notes
    -----
        - Pixels are in the NESTED scheme.
    """
    # ADM retrieve all pixels at this nside.
    allpix = np.arange(hp.nside2npix(nside))

    # ADM convert pixels to RA/Dec centers.
    theta, phi = hp.pix2ang(nside, allpix, nest=True)
    ra, dec = np.degrees(phi), 90-np.degrees(theta)

    # ADM combine pixels north and south of passed mingalb into one set.
    isnpix = is_in_gal_box([ra, dec], [0., 360., mingalb, 90.], radec=True)
    isspix = is_in_gal_box([ra, dec], [0., 360., -90, -mingalb], radec=True)
    pix = list(allpix[isnpix | isspix])

    # ADM add neighbors, if requested.
    if neighbors:
        pix = add_hp_neighbors(nside, pix)

    return pix


def is_in_box(objs, radecbox):
    """Determine which of an array of objects are inside an RA, Dec box.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        An array of objects. Must include at least the columns "RA" and "DEC".
    radecbox : :class:`list`
        4-entry list of coordinates [ramin, ramax, decmin, decmax] forming the
        edges of a box in RA/Dec (degrees).

    Returns
    -------
    :class:`~numpy.ndarray`
        ``True`` for objects in the box, ``False`` for objects outside of the box.

    Notes
    -----
        - Tests the minimum RA/Dec with >= and the maximum with <
    """
    ramin, ramax, decmin, decmax = radecbox

    # ADM check for some common mistakes.
    if decmin < -90. or decmax > 90. or decmax <= decmin or ramax <= ramin:
        msg = "Strange input: [ramin, ramax, decmin, decmax] = {}".format(radecbox)
        log.critical(msg)
        raise ValueError(msg)

    ii = ((objs["RA"] >= ramin) & (objs["RA"] < ramax)
          & (objs["DEC"] >= decmin) & (objs["DEC"] < decmax))

    return ii


def hp_in_cap(nside, radecrad, inclusive=True, fact=4):
    """Determine which HEALPixels touch an RA, Dec, radius cap.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    radecrad : :class:`list`, defaults to `None`
        3-entry list of coordinates [ra, dec, radius] forming a cap or
        "circle" on the sky. ra, dec and radius are all in degrees.
    inclusive : :class:`bool`, optional, defaults to ``True``
        see documentation for `healpy.query_disc()`.
    fact : :class:`int`, optional defaults to 4
        see documentation for `healpy.query_disc()`.

    Returns
    -------
    :class:`list`
        A list of HEALPixels at the passed `nside` that touch the cap.

    Notes
    -----
        - Just syntactic sugar around `healpy.query_disc()`.
    """
    ra, dec, radius = radecrad

    # ADM RA/Dec to co-latitude/longitude, everything to radians.
    theta, phi, rad = np.radians(90.-dec), np.radians(ra), np.radians(radius)

    # ADM convert the colatitudes to Cartesian vectors remembering to
    # ADM transpose to pass the array to query_disc in the correct order.
    vec = hp.dir2vec(theta, phi).T

    # ADM determine the pixels that touch the box.
    pixnum = hp.query_disc(nside, vec, rad,
                           inclusive=inclusive, fact=fact, nest=True)

    return pixnum


def is_in_cap(objs, radecrad):
    """Determine which of an array of objects lie inside an RA, Dec, radius cap.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        An array of objects. Must include at least the columns "RA" and "DEC".
    radecrad : :class:`list`, defaults to `None`
        3-entry list of coordinates [ra, dec, radius] forming a cap or
        "circle" on the sky. ra, dec and radius are all in degrees.

    Returns
    -------
    :class:`~numpy.ndarray`
        ``True`` for objects in the cap, ``False`` for objects outside of the cap.

    Notes
    -----
        - Tests the separation with <=, so include objects on the cap boundary.
        - See also is_in_circle() which handles multiple caps.
    """
    ra, dec, radius = radecrad

    cobjs = SkyCoord(objs["RA"]*u.degree, objs["DEC"]*u.degree)
    center = SkyCoord(ra*u.degree, dec*u.degree)

    ii = center.separation(cobjs) <= radius*u.degree

    return ii


def is_in_hp(objs, nside, pixlist, radec=False):
    """Determine which of an array of objects lie inside a set of HEALPixels.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Array of objects. Must include at columns "RA" and "DEC".
    nside : :class:`int`
        The HEALPixel nside number (NESTED scheme).
    pixlist : :class:`list` or `int` or `~numpy.ndarray`
        The list of HEALPixels in which to find objects.
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` `objs` is an [RA, Dec] list instead of a rec array.

    Returns
    -------
    :class:`~numpy.ndarray`
        ``True`` for objects in pixlist, ``False`` for other objects.
    """
    # ADM if an integer is passed, convert it to an array.
    pixlist = np.atleast_1d(pixlist)

    if radec:
        ra, dec = objs
    else:
        ra, dec = objs["RA"], objs["DEC"]

    # ADM check whether ra, dec are in the pixel list
    theta, phi = np.radians(90-dec), np.radians(ra)
    pixnums = hp.ang2pix(nside, theta, phi, nest=True)
    w = np.hstack([np.where(pixnums == pix)[0] for pix in pixlist])
    ii = np.zeros(len(pixnums), dtype='bool')
    ii[w] = True

    return ii


def pixarea2nside(area):
    """Closest HEALPix nside for a given area.

    Parameters
    ----------
    area : :class:`float`
        area in square degrees.

    Returns
    -------
    :class:`int`
        HEALPix nside that corresponds to passed area.

    Notes
    -----
        - Only considers 2**x nside values (1, 2, 4, 8 etc.)
    """
    # ADM loop through nsides until we cross the passed area.
    nside = 1
    while (hp.nside2pixarea(nside, degrees=True) > area):
        nside *= 2

    # ADM there is no nside of 0 so bail if nside is still 1.
    if nside > 1:
        # ADM is the nside with the area that is smaller or larger
        # ADM than the passed area "closest" to the passed area?
        smaller = hp.nside2pixarea(nside, degrees=True)
        larger = hp.nside2pixarea(nside//2, degrees=True)
        if larger/area < area/smaller:
            return nside//2

    return nside


def check_nside(nside):
    """Flag an error if nside is not OK for NESTED HEALPixels.

    Parameters
    ----------
    nside : :class:`int` or `~numpy.ndarray`
        The HEALPixel nside number (NESTED scheme) or an
        array of such numbers.

    Returns
    -------
    Nothing, but raises a ValueRrror for a bad `nside`.
    """
    if nside is None:
        msg = "need to pass the NSIDE parameter?"
        log.critical(msg)
        raise ValueError(msg)
    nside = np.atleast_1d(nside)
    good = hp.isnsideok(nside, nest=True)
    if not np.all(good):
        msg = "NSIDE = {} not valid in the NESTED scheme"  \
            .format(np.array(nside)[~good])
        log.critical(msg)
        raise ValueError(msg)


def nside2nside(nside, nsidenew, pixlist):
    """Change a list of HEALPixel numbers to a different NSIDE.

    Parameters
    ----------
    nside : :class:`int`
        The current HEALPixel nside number (NESTED scheme).
    nsidenew : :class:`int`
        The new HEALPixel nside number (NESTED scheme).
    pixlist : :class:`list` or `~numpy.ndarray`
        The list of HEALPixels to be changed.

    Returns
    -------
    :class:`~numpy.ndarray`
        The altered list of HEALPixels.

    Notes
    -----
        - The size of the input list will be altered. For instance,
          nside=2, pixlist=[0,1] is covered by only pixel [0] at
          nside=1 but by pixels [0, 1, 2, 3, 4, 5, 6, 7] at nside=4.
        - Doesn't check that the passed pixels are valid at `nside`.
    """
    # ADM sanity check that we're in the nested scheme.
    check_nside([nside, nsidenew])

    pixlist = np.atleast_1d(pixlist)

    # ADM remember to use integer division throughout.
    # ADM if nsidenew is smaller (at a lower resolution), then
    # ADM downgrade the passed pixel numbers.
    if nsidenew <= nside:
        fac = (nside//nsidenew)**2
        pixlistnew = np.array(list(set(pixlist//fac)))
    else:
        # ADM if nsidenew is larger (at a higher resolution), then
        # ADM upgrade the passed pixel numbers.
        fac = (nsidenew//nside)**2
        pixlistnew = []
        for pix in pixlist:
            pixlistnew.append(np.arange(pix*fac, pix*fac+fac))
        pixlistnew = np.hstack(pixlistnew)

    return pixlistnew


def is_in_gal_box(objs, lbbox, radec=False):
    """Determine which of an array of objects are in a Galactic l, b box.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray` or `list`
        An array of objects. Must include at least the columns "RA" and "DEC".
    radecbox : :class:`list`
        4-entry list of coordinates [lmin, lmax, bmin, bmax] forming the
        edges of a box in Galactic l, b (degrees).
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` then the passed `objs` is an [RA, Dec] list instead of
        a rec array.

    Returns
    -------
    :class:`~numpy.ndarray`
        ``True`` for objects in the box, ``False`` for objects outside of the box.

    Notes
    -----
        - Tests the minimum l/b with >= and the maximum with <
    """
    lmin, lmax, bmin, bmax = lbbox

    # ADM check for some common mistakes.
    if bmin < -90. or bmax > 90. or bmax <= bmin or lmax <= lmin:
        msg = "Strange input: [lmin, lmax, bmin, bmax] = {}".format(lbbox)
        log.critical(msg)
        raise ValueError(msg)

    # ADM convert input RA/Dec to Galactic coordinates.
    if radec:
        ra, dec = objs
    else:
        ra, dec = objs["RA"], objs["DEC"]

    c = SkyCoord(ra*u.degree, dec*u.degree)
    gal = c.galactic

    # ADM and limit to (l, b) ranges.
    ii = ((gal.l.value >= lmin) & (gal.l.value < lmax)
          & (gal.b.value >= bmin) & (gal.b.value < bmax))

    return ii


def radec_match_to(matchto, objs, sep=1., radec=False, return_sep=False):
    """Match objects to a catalog list on RA/Dec.

    Parameters
    ----------
    matchto : :class:`~numpy.ndarray` or `list`
        Coordinates to match TO. Must include columns "RA" and "DEC".
    objs : :class:`~numpy.ndarray` or `list`
        Objects matched to `matchto`. Must include "RA" and "DEC".
    sep : :class:`float`, defaults to 1 arcsecond
        Separation at which to match `objs` to `matchto` in ARCSECONDS.
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` then `objs` and `matchto` are [RA, Dec] lists
        instead of rec arrays.
    return_sep : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return the separation between each object, not
        just the indexes of the match.

    Returns
    -------
    :class:`~numpy.ndarray` (of integers)
        Indexes in `matchto` where `objs` matches `matchto` at < `sep`.
    :class:`~numpy.ndarray` (of integers)
        Indexes in `objs` where `objs` matches `matchto` at < `sep`.
    :class:`~numpy.ndarray` (of floats)
        The distances in ARCSECONDS of the matches.
        Only returned if `return_sep` is ``True``.

    Notes
    -----
        - Sense is important. Every coordinate pair in `objs` is matched
          to `matchto`, but NOT every coordinate pair in `matchto` is
          matched to `objs`. `matchto` is the "parent" catalog being
          matched to, i.e. we're looking for the instances where `objs`
          has a match in `matchto`. The array of returned indexes thus
          can't be longer than `objs`. Consider this example:

          >>> mainra, maindec = [100], [30]
          >>> ras, decs = [100, 100, 100], [30, 30, 30]
          >>>
          >>> radec_match_to([mainra, maindec], [ras, decs], radec=True)
          >>> Out: (array([0, 0, 0]), array([0, 1, 2]))
          >>>
          >>> radec_match_to([ras, decs], [mainra, maindec], radec=True)
          >>> Out: (array([0]), array([0]))

        - Only returns the CLOSEST match within `sep` arcseconds.
    """
    if radec:
        ram, decm = matchto
        ra, dec = objs
    else:
        ram, decm = matchto["RA"], matchto["DEC"]
        ra, dec = objs["RA"], objs["DEC"]

    cmatchto = SkyCoord(ram*u.degree, decm*u.degree)
    cobjs = SkyCoord(ra*u.degree, dec*u.degree)

    idmatchto, d2d, _ = cobjs.match_to_catalog_sky(cmatchto)
    idobjs = np.arange(len(cobjs))

    ii = d2d < sep*u.arcsec

    if return_sep:
        return idmatchto[ii], idobjs[ii], d2d[ii].arcsec

    return idmatchto[ii], idobjs[ii]


def rewind_coords(ranow, decnow, pmra, pmdec,
                  epochnow=2015.5, epochnowdec=None,
                  epochpast=1991.5, epochpastdec=None):
    """Shift coordinates into the past based on proper motions.

    Parameters
    ----------
    ranow : :class:`flt` or `~numpy.ndarray`
        Right Ascension (degrees) at "current" epoch.
    decnow : :class:`flt` or `~numpy.ndarray`
        Declination (degrees) at "current" epoch.
    pmra : :class:`flt` or `~numpy.ndarray`
        Proper motion in RA (mas/yr).
    pmdec : :class:`flt` or `~numpy.ndarray`
        Proper motion in Dec (mas/yr).
    epochnow : :class:`flt` or `~numpy.ndarray`, optional
        The "current" epoch (years). Defaults to Gaia DR2 (2015.5).
    epochnowdec : :class:`flt` or `~numpy.ndarray`, optional
        If passed and not ``None`` then epochnow is interpreted as the
        epoch of the RA and this is interpreted as the epoch of the Dec.
    epochpast : :class:`flt` or `~numpy.ndarray`, optional
        Epoch in the past (years). Defaults to Tycho DR2 mean (1991.5).
    epochpastdec : :class:`flt` or `~numpy.ndarray`, optional
        If passed and not ``None`` then epochpast is interpreted as the
        epoch of the RA and this is interpreted as the epoch of the Dec.

    Returns
    -------
    :class:`~numpy.ndarray`
        Right Ascension in the past (degrees).
    :class:`~numpy.ndarray`
        Declination in the past (degrees).

    Notes
    -----
        - All output RAs will be in the range 0 < RA < 360o.
        - Only called "rewind_coords" to correspond to the default
          `epochnow` > `epochpast`. "fast forwarding" works fine, too,
          i.e., you can pass `epochpast` > `epochnow` to move coordinates
          to a future epoch.
        - Inaccurate to ~0.1" for motions as high as ~10"/yr (Barnard's
          Star) after ~200 years because of the simplified cosdec term.
    """
    # ADM allow for different RA/Dec coordinates.
    if epochnowdec is None:
        epochnowdec = epochnow
    if epochpastdec is None:
        epochpastdec = epochpast

    # ADM enforce "double-type" precision for RA/Dec floats.
    if isinstance(ranow, float):
        ranow = np.array([ranow], dtype='f8')
    if isinstance(decnow, float):
        decnow = np.array([decnow], dtype='f8')

    # ADM "rewind" coordinates.
    cosdec = np.cos(np.deg2rad(decnow))
    ra = ranow - ((epochnow-epochpast) * pmra / 3600. / 1000. / cosdec)
    dec = decnow - ((epochnowdec-epochpastdec) * pmdec / 3600. / 1000.)

    # ADM % 360. is to deal with wraparound bugs.
    return ra % 360., dec


def shares_hp(nside, objs1, objs2, radec=False):
    """Check if arrays of objects occupy the same HEALPixels.

    Parameters
    ----------
    nside : :class:`int`
        HEALPixel nside integer (NESTED scheme).
    objs1 : :class:`~numpy.ndarray` or `list`
        First set of objects. Must include columns "RA" and "DEC".
    objs : :class:`~numpy.ndarray` or `list`
        Second set of objects. Must include "RA" and "DEC".
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` then `objs1` and `objs2` are [RA, Dec] lists
        instead of rec arrays.

    Returns
    -------
    :class:`~numpy.ndarray` (of booleans)
        ``True`` for objects in `objs1` that share a HEALPixel with
         objects in ``objs2`` at `nside`. Same length as `objs1`
    :class:`~numpy.ndarray` (of booleans)
        ``True`` for objects in `objs2` that share a HEALPixel with
         objects in ``objs1`` at `nside`. Same length as `objs2`
    """
    if radec:
        ra1, dec1 = objs1
        ra2, dec2 = objs2
    else:
        ra1, dec1 = objs1["RA"], objs1["DEC"]
        ra2, dec2 = objs2["RA"], objs2["DEC"]

    theta1, phi1 = np.radians(90-dec1), np.radians(ra1)
    pix1 = hp.ang2pix(nside, theta1, phi1, nest=True)
    spix1 = set(pix1)

    theta2, phi2 = np.radians(90-dec2), np.radians(ra2)
    pix2 = hp.ang2pix(nside, theta2, phi2, nest=True)
    spix2 = set(pix2)

    one = np.array([pix in spix2 for pix in pix1])
    two = np.array([pix in spix1 for pix in pix2])

    return one, two
