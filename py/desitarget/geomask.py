# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==================
desitarget.geomask
==================

Utility functions for restricting targets to regions on the sky

"""
from __future__ import (absolute_import, division)
#
import numpy as np
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

# ADM fake the matplotlib display so it doesn't die on allocated nodes.
import matplotlib   # noqa: E402
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


def bundle_bricks(pixnum, maxpernode, nside, brickspersec=1., prefix='targets', gather=True,
                  surveydir="/global/project/projectdirs/cosmo/data/legacysurvey/dr6"):
    """Determine the optimal packing for bricks collected by HEALpixel integer.

    Parameters
    ----------
    pixnum : :class:`np.array`
        List of integers, e.g., HEALPixel numbers occupied by a set of bricks
        (e.g. array([16, 16, 16...12 , 13, 19]) ).
    maxpernode : :class:`int`
        The maximum number of pixels to bundle together (e.g., if you were
        trying to pass maxpernode bricks, delineated by the HEALPixels they
        occupy, parallelized across a set of nodes).
    nside : :class:`int`
        The HEALPixel nside number that was used to generate `pixnum` (NESTED scheme).
    brickspersec : :class:`float`, optional, defaults to 1.
        The rough number of bricks processed per second by the code (parallelized across
        a chosen number of nodes)
    prefix : :class:`str`, optional, defaults to 'targets'
        Should correspond to the binary executable "X" that is run as select_X for a
        target type. Depending on the type of target file that is being packed for
        parallelization, this could be 'randoms', 'skies', 'targets', etc.
    gather : :class:`bool`, optional, defaults to ``True``
        If ``True`` then provide a final command for combining all of the HEALPix-split
        files into one large file. If ``False``, comment out that command.
    surveydir : :class:`str`, optional, defaults to the DR6 directory at NERSC
        The root directory pointing to a Data Release from the Legacy Surveys,
        (e.g. "/global/project/projectdirs/cosmo/data/legacysurvey/dr6").

    Returns
    -------
    Nothing, but prints commands to screen that would facilitate running a
    set of bricks by HEALPixel integer with the total number of bricks not
    to exceed maxpernode. Also prints how many bricks would be on each node.

    Notes
    -----
    h/t https://stackoverflow.com/questions/7392143/python-implementations-of-packing-algorithm
    """
    # ADM the number of pixels (numpix) in each pixel (pix)
    pix, numpix = np.unique(pixnum, return_counts=True)

    # ADM the indices needed to reverse-sort the array on number of pixels
    reverse_order = np.flipud(np.argsort(numpix))
    numpix = numpix[reverse_order]
    pix = pix[reverse_order]

    # ADM iteratively populate lists of the numbers of pixels
    # ADM and the corrsponding pixel numbers
    bins = []

    for index, num in enumerate(numpix):
        # Try to fit this sized number into a bin
        for bin in bins:
            if np.sum(np.array(bin)[:, 0]) + num <= maxpernode:
                # print 'Adding', item, 'to', bin
                bin.append([num, pix[index]])
                break
        else:
            # item didn't fit into any bin, start a new bin
            bin = []
            bin.append([num, pix[index]])
            bins.append(bin)

    # ADM print to screen in the form of a slurm bash script, and
    # ADM other useful information
    print("#######################################################")
    print("Numbers of bricks or files in each set of HEALPixels:")
    print("")

    # ADM the estimated margin for writing to disk in minutes.
    margin = 30
    if prefix == 'skies':
        margin = 5
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
            # ADM brickspersec is bricks/sec. Extra delta is minutes to write to disk
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
    print("#######################################################")
    print("Possible salloc command if you want to run on the Cori interactive queue:")
    print("")
    print("salloc -N {} -C haswell -t 0{}:00:00 --qos interactive -L SCRATCH,project"
          .format(len(bins), maxeta))

    print("")
    print("#######################################################")
    print('Example shell script for slurm:')
    print('')
    print('#!/bin/bash -l')
    print('#SBATCH -q regular')
    print('#SBATCH -N {}'.format(len(bins)))
    print('#SBATCH -t 0{}:00:00'.format(maxeta))
    print('#SBATCH -L SCRATCH,project')
    print('#SBATCH -C haswell')
    print('')

    # ADM extract the Data Release number from the survey directory
    dr = surveydir.split('dr')[-1][0]
    comment = '#'
    if gather:
        comment = ''

    # ADM to handle inputs that look like "sv1_targets".
    prefix2 = prefix
    if prefix[0:2] == "sv":
        prefix2 = "sv_targets"

    outfiles = []
    for bin in bins:
        num = np.array(bin)[:, 0]
        pix = np.array(bin)[:, 1]
        wpix = np.where(num > 0)[0]
        if len(wpix) > 0:
            goodpix = pix[wpix]
            goodpix.sort()
            strgoodpix = ",".join([str(pix) for pix in goodpix])
            # ADM the replace is to handle inputs that look like "sv1_targets".
            outfile = "$CSCRATCH/{}-dr{}-hp-{}.fits".format(prefix.replace("_", "-"), dr, strgoodpix)
            outfiles.append(outfile)
            print("srun -N 1 select_{} {} {} --numproc 32 --nside {} --healpixels {} &"
                  .format(prefix2, surveydir, outfile, nside, strgoodpix))
    print("wait")
    print("")
    print("{}gather_targets '{}' $CSCRATCH/{}-dr{}.fits {}"
          # ADM the prefix2 manipulation is to handle inputs that look like "sv1_targets".
          .format(comment, ";".join(outfiles), prefix, dr, prefix2.split("_")[-1]))
    print("")

    return


def add_hp_neighbors(nside, pixnum):
    """Add all neighbors in the NESTED scheme to a set of HEALPixels.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    pixnum : :class:`list` or `int`
        list of HEALPixel numbers (or a single HEALPixel number).

    Returns
    -------
    :class:`list`
        The passed list of pixels with all neighbors added to the list.

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


def hp_in_box(nside, radecbox, inclusive=True, fact=4):
    """Determine which HEALPixels touch an RA, Dec box.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    radecbox : :class:`list`
        4-entry list of coordinates [ramin, ramax, decmin, decmax] forming the
        edges of a box in RA/Dec (degrees).
    inclusive : :class:`bool`, optional, defaults to ``True``
        see documentation for `healpy.query_polygon()`.
    fact : :class:`int`, optional defaults to 4
        see documentation for `healpy.query_polygon()`.

    Returns
    -------
    :class:`list`
        A list of HEALPixels at the passed `nside` that touch the passed RA/Dec box.

    Notes
    -----
        - Just syntactic sugar around `healpy.query_polygon()`.
        - If -90o or 90o are passed, then decmin and decmax are moved slightly away
          from the poles.
        - When the RA range exceeds 180o, `healpy.query_polygon()` defines the box as
          that with the smallest area (i.e the box can wrap-around in RA). To avoid
          any ambiguity, this function will return ALL HEALPixels in such cases.
    """
    ramin, ramax, decmin, decmax = radecbox

    # ADM handle some edge cases. Don't be too close to the poles.
    leeway = 1e-5
    if decmax > 90-leeway:
        decmax = 90-leeway
        log.warning('Max Dec too close to pole; set to {}o'.format(decmax))
    if decmin < -90+leeway:
        decmin = -90+leeway
        log.warning('Min Dec too close to pole; set to {}o'.format(decmin))

    # ADM area enclosed isn't well-defined if RA covers more than 180o.
    if np.abs(ramax-ramin) > 180.:
        log.warning('Max RA ({}) and Min RA ({}) separated by > 180o...'
                    .format(ramax, ramin))
        log.warning('...returning full set of HEALPixels at nside={}'
                    .format(nside))
        return np.arange(hp.nside2npix(nside))

    # ADM convert RA/Dec to co-latitude and longitude in radians.
    rapairs = np.array([ramin, ramin, ramax, ramax])
    decpairs = np.array([decmin, decmax, decmax, decmin])
    thetapairs, phipairs = np.radians(90.-decpairs), np.radians(rapairs)

    # ADM convert the colatitudes to Cartesian vectors remembering to
    # ADM transpose to pass the array to query_polygon in the correct order.
    vecs = hp.dir2vec(thetapairs, phipairs).T

    # ADM determine the pixels that touch the box.
    pixnum = hp.query_polygon(nside, vecs,
                              inclusive=inclusive, fact=fact, nest=True)

    return pixnum


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

    # ADM convert RA/Dec to co-latitude/longitude and everything to radians.
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


def is_in_hp(objs, nside, pixlist):
    """Determine which of an array of objects lie inside a set of HEALPixels.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        An array of objects. Must include at least the columns "RA" and "DEC".
    nside : :class:`int`
        The HEALPixel nside number (NESTED scheme).
    pixlist : :class:`list` or `~numpy.ndarray`
        The list of HEALPixels in which to find objects.

    Returns
    -------
    :class:`~numpy.ndarray`
        ``True`` for objects in pixlist, ``False`` for objects outside of pixlist.
    """
    theta, phi = np.radians(90-objs["DEC"]), np.radians(objs["RA"])
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
