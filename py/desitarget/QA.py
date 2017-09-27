# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==================
desitarget.QA
==================

Module dealing with Quality Assurance tests for Target Selection
"""
from __future__ import (absolute_import, division)
#
from time import time
import numpy as np
import fitsio
import os, re
from collections import defaultdict
from glob import glob
from scipy.optimize import leastsq
from scipy.spatial import ConvexHull
import random
import textwrap

from astropy import units as u
from astropy.coordinates import SkyCoord

from . import __version__ as desitarget_version

from desiutil import brick
from desiutil.log import get_logger, DEBUG
from desiutil.plots import init_sky, plot_sky_binned
from desitarget import desi_mask

import warnings, itertools
import matplotlib.pyplot as plt

import healpy as hp

import numpy.lib.recfunctions as rfn


def sph2car(ra, dec):
    """Convert RA and Dec to a Cartesian vector
    """

    phi = np.radians(np.asarray(ra))
    theta = np.radians(90.0 - np.asarray(dec))

    r = np.sin(theta)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.cos(theta)

    #ADM treat vectors smaller than our tolerance as zero
    tol = 1e-15
    x = np.where(np.abs(x) > tol,x,0)
    y = np.where(np.abs(y) > tol,y,0)
    z = np.where(np.abs(z) > tol,z,0)

    return np.array((x, y, z)).T


def area_of_hull(ras,decs,nhulls):
    """Determine the area of a convex hull from an array of RA, Dec points that define that hull
    
    Parameters
    ----------
    ras : :class:`~numpy.array`
        Right Ascensions of the points in the convex hull (boundary) in DEGREES
    decs : :class:`~numpy.array`
        Declinations of the points in the convex hull (boundary) in DEGREES
    nhulls : :class:`int`
        The number of hulls the user thinks they passed, to check each hull occupies a row
        not a column

    Returns
    -------
    :class:`float`
        The area of the convex hull (drawn on the surface of the sphere) in `square degrees`

    Notes
    -----
        - If a 2-D array of RAs and Decs representing multiple convex hulls is passed, than the 
          output will be a 1-D array of areas. Each hull must occupy the rows, e.g. for a
          set of RAs for 2 identical hulls with 15 points on the boundary, this would be the
          correct array ordering:

          >>> array([[ 7.14639156,  7.02689545,  7.01554989,  7.01027328,  7.01276138,
                       7.01444518,  7.0173733 ,  7.03278736,  7.22537629,  7.24887231,
                       7.25749704,  7.25629861,  7.25075961,  7.24776393,  7.2375672 ],
                     [ 7.14639156,  7.02689545,  7.01554989,  7.01027328,  7.01276138,
                     7.01444518,  7.0173733 ,  7.03278736,  7.22537629,  7.24887231,
                     7.25749704,  7.25629861,  7.25075961,  7.24776393,  7.2375672 ]])
        - This is only an approximation, because it uses the average latitude. See, e.g.:
          https://trs.jpl.nasa.gov/bitstream/handle/2014/41271/07-0286.pdf
          but it's accurate (to millionths of a per cent) for areas of a few sq. deg.
        - This routine will fail at the poles. So, decs should never be passed as -90. or 90.
    """

    if ras.ndim > 1 and ras.shape[0] != nhulls:
        raise IOError('Your array contains {} hulls. Perhaps you meant to pass its tranposition?'.format(ras.shape[0]))

    #ADM try to catch polar cases (this won't catch areas of larger than a few degrees,
    #ADM but this routine is inaccurate for large hulls anyway)
    if np.max(np.abs(decs)) >= 89.:
        raise IOError('You passed a declination of > 89o or < -89o. This routine does not work at or over the poles')
    
    #ADM ensure RAs run from 0 -> 360
    ras%=360

    #ADM we'll loop over pairs of vertices around the hull
    #ADM the axis command means that both 1-d and n-dimensional arrays can be passed
    startras = np.roll(ras,+1,axis=ras.ndim-1)
    endras = np.roll(ras,-1,axis=ras.ndim-1)
    
    rawidth = startras-endras
    #ADM To deal with wraparound issues, assume that any "large" RA intervals cross RA=0
    w = np.where(rawidth < -180.)
    rawidth[w] -= -360.
    w = np.where(rawidth > 180.)
    rawidth[w] -= 360.

    spharea = np.abs(0.5*np.sum(rawidth*np.degrees(np.sin(np.radians(decs))),axis=ras.ndim-1))

    return spharea


def targets_on_hull(ras,decs):
    """Create the convex hull (boundary) of an area from input arrays of locations
    
    Parameters
    ----------
    ras : :class:`~numpy.array`
        Right Ascensions of the points in the convex hull (boundary) in DEGREES
    decs : :class:`~numpy.array`
        Declinations of the points in the convex hull (boundary) in DEGREES

    Returns
    -------
    :class:`integer`
        The INDICES of the input RA/Dec that form the vertices of the boundary in counter-clockwise order

    Notes
    -----
    It takes about 15 seconds to run this on Edison for ~1 million objects
    """

    #ADM the 2 restricts to 2D (i.e. the RA/Dec plane)
    hull = ConvexHull(np.vstack(zip(ras,decs)))

    return hull.vertices


def remove_fractional_bricks(targs,frac=0.9,bricksize=0.25):
    """For targets grouped by their ``BRICKID`` remove bricks with incomplete areal coverage

    Parameters
    ----------
    targs : :class:`~numpy.ndarray`
        File of targets generated by `desitarget.bin.select_targets`
    frac : :class:`float`, optional, defaults to 90%
        The areal threshold. Bricks that have areal coverage below this number will be removed
    bricksize : :class:`float`, optional, defaults to 0.25
        The side-size such that all bricks have longest size < bricksize

    Returns
    -------
    :class:`~numpy.ndarray`
        The input array of targets (in the input format) with targets that are in a brick that has
        an areal coverage of less than frac removed

    Notes
    -----
    Works by calculating the convex hull (boundary) of the targets in each brick, calculating the area
    within that convex hull, and comparing to the expected area for that brick

    """

    #ADM first sort on the BRICKID, and retain the sorted values and the ordering of the sort
    indsort = np.argsort(targs['BRICKID'])

    #ADM pull out the RAs and Decs, which are the only columns we need other than the grouping column
    ras = targs["RA"][indsort]
    decs = targs["DEC"][indsort]
    bricks = targs['BRICKID'][indsort]

    #ADM find the break points where the array-sorted-by-BRICKID changes to a different brick
    wsplit = np.where(np.diff(bricks))[0]
    raperbrick = np.split(ras, wsplit+1)
    decperbrick = np.split(decs, wsplit+1)
    indexperbrick = np.split(indsort, wsplit+1)
    
    #ADM the number of points in each brick
    npts = np.array([ len(pts) for pts in raperbrick ])

    #ADM remove poorly populated bricks (less than 3 points in the brick) or it's impossible to build
    #ADM a closed hull. Remember, also, to track the indexes on the original targets array.
    w = np.where(npts >= 3)
    indexperbrick = [ indexperbrick[i] for i in w[0] ]
    raperbrick = [ raperbrick[i] for i in w[0] ]
    decperbrick = [ decperbrick[i] for i in w[0] ]

    #ADM derive the hull representations for the RAs, Decs in each brick (with more than 3 members)
    hulls = [ targets_on_hull(ras, decs) for ras, decs in zip(raperbrick,decperbrick) ]
    hareas = np.array([ area_of_hull(ras[hull],decs[hull],nhulls=1) for 
                        ras, decs, hull in zip(raperbrick, decperbrick, hulls) ])
    
    #ADM initialize the bricks class and grab the areas of the full brick
    b = brick.Bricks(bricksize=bricksize)
    bareas = np.array([ b.brickarea(ras[0],decs[0]) for ras, decs in zip(raperbrick, decperbrick) ])
    
    #ADM find the bricks where the areas suggest the hulls are more than "frac" complete
    w = np.where(hareas/bareas > frac)
    outindexes = [ indexperbrick[i] for i in w[0] ]

    #ADM unroll the outindexes array and return the targets in the "complete" bricks
    unroll = np.array(list(itertools.chain.from_iterable(outindexes)))
    
    return targs[unroll]


def remove_fractional_pixels(targs,frac=0.9,nside=256):
    """For targets grouped by their ``BRICKID`` remove bricks with incomplete areal coverage
    
    Parameters
    ----------
    targs : :class:`~numpy.ndarray` 
        File of targets generated by `desitarget.bin.select_targets` must contain a column
        of HEALPix pixels at the passed nside, called ``HPXFORQA``
    frac : :class:`float`, optional, defaults to 90%
        The areal threshold. Bricks that have areal coverage below this number will be removed
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The HEALPix pixel nside number

    Returns
    -------
    :class:`~numpy.ndarray`
        The input array of targets (in the input format) with targets that are in a pixel that has
        an areal coverage of less than frac removed

    Notes
    -----
    Works by calculating the convex hull (boundary) of the targets in each pixel, calculating the area
    within that convex hull, and comparing to the expected area for that pixel
    """

    #ADM first sort on the HEALPix pixel number, and retain the sorted values and the ordering of the sort
    indsort = np.argsort(targs['HPXFORQA'])

    #ADM pull out the RAs and Decs, which are the only columns we need other than the grouping column
    ras = targs["RA"][indsort]
    decs = targs["DEC"][indsort]
    pixels = targs['HPXFORQA'][indsort]

    #ADM find the break points where the array-sorted-by-HPXFORQA changes to a different pixel
    wsplit = np.where(np.diff(pixels))[0]
    raperpixel = np.split(ras, wsplit+1)
    decperpixel = np.split(decs, wsplit+1)
    indexperpixel = np.split(indsort, wsplit+1)
    
    #ADM the number of points in each pixel
    npts = np.array([ len(pts) for pts in raperpixel ])

    #ADM remove poorly populated pixels (less than 3 points in the pixel) or it's impossible to build
    #ADM a closed hull. Remember, also, to track the indexes on the original targets array.
    w = np.where(npts >= 3)
    indexperpixel = [ indexperpixel[i] for i in w[0] ]
    raperpixel = [ raperpixel[i] for i in w[0] ]
    decperpixel = [ decperpixel[i] for i in w[0] ]

    #ADM derive the hull representations for the RAs, Decs in each pixel (with more than 3 members)
    hulls = [ targets_on_hull(ras, decs) for ras, decs in zip(raperpixel,decperpixel) ]
    hareas = np.array([ area_of_hull(ras[hull],decs[hull],nhulls=1) for 
                        ras, decs, hull in zip(raperpixel, decperpixel, hulls) ])
    
    #ADM create an array of the areas of the full pixels at nside
    bareas = np.zeros(len(hareas))+hp.nside2pixarea(nside, degrees = True)
    
    #ADM find the pixels where the areas suggest the hulls are more than "frac" complete
    w = np.where(hareas/bareas > frac)
    outindexes = [ indexperpixel[i] for i in w[0] ]

    #ADM unroll the outindexes array and return the targets in the "complete" pixels
    unroll = np.array(list(itertools.chain.from_iterable(outindexes)))
    
    return targs[unroll]


def is_polygon_within_boundary(boundverts,polyverts):
    """Check whether a list of polygons are within the boundary of a convex hull

    Parameters
    ----------
    boundverts : :class:`float array`
       A `counter-clockwise` ordered array of vectors representing the Cartesian coordinates of 
       the vertices of the complex hull (boundary) of a survey, e.g.
       
       >>> array([[x1,  y1,  z1],      Vertex 1 on survey boundary
                  [x2,  y2,  z2],      Vertex 2 of survey boundary
                  [x3,  y3,  z3],      Vertex 3 of survey boundary
                  [x4,  y4,  z4],      Vertex 4 of survey boundary
                  [x5,  y5,  z5],      Vertex 5 of survey boundary
                  ..............
                  ..............
                  ..............
                  [xN,  yN,  zN]])     Vertex N of survey boundary
    polyverts : :class:`float array`
       An array of N M-vectors representing the Cartesian coordinates of the vertices of each
       polygon. For instance, M=4 would represent "rectangular" vertices drawn on the sphere:

       >>> array([[[x1,  y1,  z1],       Vertex 1 of first Rectangle
                   [x2,  y2,  z2],       Vertex 2 of first Rectangle
                   [x3,  y3,  z3],       Vertex 3 of first Rectangle
                   [x4,  y4,  z4]],      Vertex 4 of first Rectangle
                   ..............
                   ..............
                   ..............
                  [[Nx1, Ny1,  Nz1],     Vertex 1 of Nth Rectangle
                   [Nx2, Ny2,  Nz2],     Vertex 2 of Nth Rectangle
                   [Nx3, Ny3,  Nz3],     Vertex 3 of Nth Rectangle
                   [Nx4, Ny4,  Nz4]]])   Vertex 4 of Nth Rectangle

    Returns
    -------
    :class:`boolean array`
       An array the same length as polyvertices (i.e. the "number of rectangles") that is ``True`` 
       for polygons that are `fully` within the boundary and ``False`` for polygons that are 
       not within the boundary

    Notes
    -----
    Strictly, as we only test whether the vertices of the polygons are all within the boundary, 
    there can be corner cases where a small amount of a polygon edge intersects the boundary. 
    Testing the vertices should be good enough for desitarget QA, though.
    """

    #ADM recast inputs as float64 as the inner1d ufunc does not support higher-bit floats
    boundverts = boundverts.astype('f8')
    polyverts = polyverts.astype('f8')

    #ADM an array of Trues and Falses for the output. Default to True.
    boolwithin = np.ones(len(polyverts),dtype='bool')

    #ADM loop through each of the boundary vertices, starting with the final vertex to ensure that
    #ADM we traverse the entire boundary. Could make this faster by dropping polygons as soon as we
    #ADM find a negative vertex, but it's fairly quick as is
    for i in range(-1,len(boundverts)-1):
        #ADM The algorithm is to check the direction of the projection (dot product) of each vertex of each polygon
        #ADM onto each vector normal (cross product) to the geodesics (planes) that map out the survey boundary
        #ADM If this projection is positive, then we "turned left" to get to the polygon vertex, assuming the convex
        #ADM hull is ordered counter-clockwise
        test = inner1d(np.cross(boundverts[i],boundverts[i+1]),polyverts)
        #ADM if any of the vertices are not a left-turn from (within the) boundary points, set to False
        boolwithin[np.where(np.any(test < 0, axis=1))] = False

    return boolwithin


def generate_fluctuations(brickfilename, targettype, depthtype, depthorebvarray, random_state=None):
    """Based on depth or E(B-V) values for a brick, generate target fluctuations

    Parameters
    ----------
    brickfilename : :class:`str`
        File name of a list of a brick info file made by :func:`QA.brick_info`.
    targettype : :class: `str`
        Name of the target type for which to generate fluctuations
        options are ``ALL``, ``LYA``, ``MWS``, ``BGS``, ``QSO``, ``ELG``, ``LRG``
    depthtype : :class: `str`
        Name of the type of depth-and-band (or E(B-V)) for which to generate fluctuations
        options are ``DEPTH_G``, ``GALDEPTH_G``, ``DEPTH_R``, ``GALDEPTH_R``, ``DEPTH_Z``, ``GALDEPTH_Z``
        or pass ``EBV`` to generate fluctuations based off E(B-V) values
    depthorebvarray : :class:`float array`
        An array of brick depths (i.e. for N bricks, N realistic values of depth) or
        an array of E(B-V) values if ``EBV`` is passed

    Returns
    -------
    :class:`float`
        An array of the same length as ``depthorebvarray`` with per-brick fluctuations
        generated from actual DECaLS data

    """
    if random_state is None:
        random_state = np.random.RandomState()
        
    log = get_logger()

    #ADM check some impacts are as expected
    dts = ["PSFDEPTH_G","GALDEPTH_G","PSFDEPTH_R","GALDEPTH_R","PSFDEPTH_Z","GALDEPTH_Z","EBV"]
    if not depthtype in dts:
        raise ValueError("depthtype must be one of {}".format(" ".join(dts)))

    #if depthtype == "EBV":
        #print("generating per-brick fluctuations for E(B-V) values")
    #else:
        #print("generating per-brick fluctuations for depth values")

    if not type(depthorebvarray) == np.ndarray:
        raise ValueError("depthorebvarray must be a numpy array not type {}".format(type(depthorebvarray)))

    #ADM the number of bricks
    nbricks = len(depthorebvarray)
    tts = ["ALL","LYA","MWS","BGS","QSO","ELG","LRG"]
    if not targettype in tts:
        fluc = np.ones(nbricks)
        mess = "fluctuations for targettype {} are set to one".format(targettype)
        log.warning(mess)
        return fluc

    #ADM the target fluctuations are actually called FLUC_* in the model dictionary
    targettype = "FLUC_"+targettype

    #ADM generate/retrieve the model map dictionary
    modelmap = model_map(brickfilename)

    #ADM retrive the quadratic model
    means, sigmas = modelmap[depthtype][targettype]

    #ADM sample the distribution for each parameter in the quadratic
    #ADM fit for each of the total number of bricks
    asamp = random_state.normal(means[0], sigmas[0], nbricks)
    bsamp = random_state.normal(means[1], sigmas[1], nbricks)
    csamp = random_state.normal(means[2], sigmas[2], nbricks)

    #ADM grab the fluctuation in each brick
    fluc = asamp*depthorebvarray**2. + bsamp*depthorebvarray + csamp

    return fluc


def model_map(brickfilename,plot=False):
    """Make a model map of how 16,50,84 percentiles of brick depths and how targets fluctuate with brick depth and E(B-V)

    Parameters
    ----------
    brickfilename : :class:`str`
        File name of a list of a brick info file made by :func:`QA.brick_info`.
    plot : :class:`boolean`, optional
        generate a plot of the data and the model if ``True``

    Returns
    -------
    :class:`dictionary`
        model of brick fluctuations, median, 16%, 84% for overall
        brick depth variations and variation of target density with
        depth and EBV (per band). The first level of the nested
        dictionary are the keys PSFDEPTH_G, PSFDEPTH_R, PSFDEPTH_Z, GALDEPTH_G,
        GALDEPTH_R, GALDEPTH_Z, EBV. The second level are the keys PERC,
        which contains a list of values corresponding to the 16,50,84%
        fluctuations in that value of DEPTH or EBV per brick and the
        keys corresponding to each target_class, which contain a list
        of values [a,b,c,da,db,dc] which correspond to a quadratic model
        for the fluctuation of that target class in the form ``y = ax^2 +bx + c``
        together with the errors on a, b and c. Here, y would be the
        target density fluctuations and x would be the DEPTH or EBV value.

    """
    flucmap = fluc_map(brickfilename)

    #ADM the percentiles to consider for "mean" and "sigma
    percs = [16,50,84]

    firstcol = 1
    cols = flucmap.dtype.names
    for col in cols:
        #ADM loop through each of the depth columns (PSF and GAL) and EBV
        if re.search("PSFDEPTH",col) or re.search("EBV",col):
            #ADM the percentiles for this DEPTH/EBV across the brick
            coldict = {"PERCS": np.percentile(flucmap[col],percs)}
            #ADM fit quadratic models to target density fluctuation vs. EBV and
            #ADM target density fluctuation vs. depth
            for fcol in cols:
                if re.search("FLUC",fcol):
                    if plot:
                        print("doing",col,fcol)
                    quadparams = fit_quad(flucmap[col],flucmap[fcol],plot=plot)
                    #ADD this to the dictionary
                    coldict = dict({fcol:quadparams},**coldict)
            #ADM nest the information in an overall dictionary corresponding to
            #ADM this column name
            flucdict = {col:coldict}
            #ADM first time through set up the dictionary and
            #ADM on subsequent loops add to it
            if firstcol:
                outdict = flucdict
                firstcol = 0
            else:
                outdict = dict(flucdict,**outdict)

    return outdict


def fit_quad(x,y,plot=False):
    """Fit a quadratic model to (x,y) ordered data.

    Parameters
    ----------
    x : :class:`float`
        x-values of data (for typical x/y plot definition)
    y : :class:`float`
        y-values of data (for typical x/y plot definition)
    plot : :class:`boolean`, optional
        generate a plot of the data and the model if ``True``

    Returns
    -------
    params :class:`3-float`
        The values of a, b, c in the typical quadratic equation: ``y = ax^2 + bx + c``
    errs :class:`3-float`
        The error on the fit of each parameter
    """

    #ADM standard equation for a quadratic
    funcQuad = lambda params,x : params[0]*x**2+params[1]*x+params[2]
    #ADM difference between model and data
    errfunc = lambda params,x,y: funcQuad(params,x)-y
    #ADM initial guesses at params
    initparams = (1.,1.,1.)
    #ADM loop to get least squares fit
    with warnings.catch_warnings(): # added by Moustakas
        warnings.simplefilter('ignore')

        params,ok = leastsq(errfunc,initparams[:],args=(x,y))
        params, cov, infodict, errmsg, ok = leastsq(errfunc, initparams[:], args=(x, y),
                                                    full_output=1, epsfcn=0.0001)

    #ADM turn the covariance matrix into something chi-sq like
    #ADM via degrees of freedom
    if (len(y) > len(initparams)) and cov is not None:
        s_sq = (errfunc(params, x, y)**2).sum()/(len(y)-len(initparams))
        cov = cov * s_sq
    else:
        cov = np.inf

    #ADM estimate the error on the fit from the diagonal of the covariance matrix
    err = []
    for i in range(len(params)):
        try:
          err.append(np.absolute(cov[i][i])**0.5)
        except:
          err.append(0.)
    err = np.array(err)

    if plot:
        #ADM generate a model
        step = 0.01*(max(x)-min(x))
        xmod = step*np.arange(100)+min(x)
        ymod = xmod*xmod*params[0] + xmod*params[1] + params[2]
        #ADM rough upper and lower bounds from the errors
#        ymodhi = xmod*xmod*(params[0]+err[0]) + xmod*(params[1]+err[1]) + (params[2]+err[2])
#        ymodlo = xmod*xmod*(params[0]-err[0]) + xmod*(params[1]-err[1]) + (params[2]-err[2])
        ymodhi = xmod*xmod*params[0] + xmod*params[1] + (params[2]+err[2])
        ymodlo = xmod*xmod*params[0] + xmod*params[1] + (params[2]-err[2])
        #ADM axes that clip extreme outliers
        plt.axis([np.percentile(x,0.1),np.percentile(x,99.9),
                  np.percentile(y,0.1),np.percentile(y,99.9)])
        plt.plot(x,y,'k.',xmod,ymod,'b-',xmod,ymodhi,'b.',xmod,ymodlo,'b.')
        plt.show()

    return params, err


def fluc_map(brickfilename):
    """From a brick info file (as in construct_QA_file) create a file of target fluctuations

    Parameters
    ----------
    brickfilename : :class:`str`
        File name of a list of a brick info file made by :func:`QA.brick_info`.

    Returns
    -------
    :class:`~numpy.ndarray` 
        numpy structured array of number of times median target density
        for each brick for which NEXP is at least 3 in all of g/r/z bands.
        Contains EBV and pixel-weighted mean depth for building models
        of how target density fluctuates.
    """

    #ADM reading in brick_info file
    fx = fitsio.FITS(brickfilename, upper=True)
    alldata = fx[1].read()

    #ADM limit to just things with NEXP=3 in every band
    #ADM and that have reasonable depth values from the depth maps
    try:
        w = np.where( (alldata['NEXP_G'] > 2) & (alldata['NEXP_R'] > 2) & (alldata['NEXP_Z'] > 2) &
                      (alldata['PSFDEPTH_G'] > -90) & (alldata['PSFDEPTH_R'] > -90) & (alldata['PSFDEPTH_Z'] > -90))
    except:
        w = np.where( (alldata['NEXP_G'] > 2) & (alldata['NEXP_R'] > 2) & (alldata['NEXP_Z'] > 2) &
                      (alldata['DEPTH_G'] > -90) & (alldata['DEPTH_R'] > -90) & (alldata['DEPTH_Z'] > -90))
    alldata = alldata[w]

    #ADM choose some necessary columns and rename density columns,
    #ADM which we'll now base on fluctuations around the median

    #JM -- This will only work for DR3!
    cols = [
            'BRICKID','BRICKNAME','BRICKAREA','RA','DEC','EBV',
            'DEPTH_G','DEPTH_R','DEPTH_Z',
            'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
            'DENSITY_ALL', 'DENSITY_ELG', 'DENSITY_LRG',
            'DENSITY_QSO', 'DENSITY_LYA', 'DENSITY_BGS', 'DENSITY_MWS'
            ]
    data = alldata[cols]
    #newcols = [col.replace('DENSITY', 'FLUC') for col in cols]
    newcols = [
            'BRICKID','BRICKNAME','BRICKAREA','RA','DEC','EBV',
            'PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z',
            'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
            'FLUC_ALL', 'FLUC_ELG', 'FLUC_LRG',
            'FLUC_QSO', 'FLUC_LYA', 'FLUC_BGS', 'FLUC_MWS'
            ]
    data.dtype.names = newcols

    #ADM for each of the density columns loop through and replace
    #ADM density by value relative to median
    outdata = data.copy()
    for col in newcols:
        if re.search("FLUC",col):
            med = np.median(data[col])
            if med > 0:
                outdata[col] = data[col]/med
            else:
                outdata[col] = 1.

    return outdata


def mag_histogram(targetfilename,binsize,outfile):
    """Detemine the magnitude distribution of targets

    Parameters
    ----------
    targetfilename : :class:`str`
        File name of a list of targets created by select_targets
    binsize : :class:`float`
        bin size of the output histogram
    outfilename: :class:`str`
        Output file name for the magnitude histograms, which will be written as ASCII

    Returns
    -------
    :class:`Nonetype`
        No return...but prints a raw N(m) to screen for each target type
    """

    #ADM read in target file
    print('Reading in targets file')
    fx = fitsio.FITS(targetfilename, upper=True)
    targetdata = fx[1].read(columns=['BRICKID','DESI_TARGET','BGS_TARGET','MWS_TARGET','DECAM_FLUX'])

    #ADM open output file for writing
    file = open(outfile, "w")

    #ADM calculate the magnitudes of interest
    print('Calculating magnitudes')
    gfluxes = targetdata["DECAM_FLUX"][...,1]
    gmags = 22.5-2.5*np.log10(gfluxes*(gfluxes  > 1e-5) + 1e-5*(gfluxes < 1e-5))
    rfluxes = targetdata["DECAM_FLUX"][...,2]
    rmags = 22.5-2.5*np.log10(rfluxes*(rfluxes  > 1e-5) + 1e-5*(rfluxes < 1e-5))
    zfluxes = targetdata["DECAM_FLUX"][...,4]
    zmags = 22.5-2.5*np.log10(zfluxes*(zfluxes  > 1e-5) + 1e-5*(zfluxes < 1e-5))

    bitnames = ["ALL","LRG","ELG","QSO","BGS","MWS"]
    bitvals = [-1]+list(2**np.array([0,1,2,60,61]))

    #ADM set up bin edges in magnitude from 15 to 25 at resolution of binsize
    binedges = np.arange(((25.-15.)/binsize)+1)*binsize + 15

    #ADM loop through bits and print histogram of raw target numbers per magnitude
    for i, bitval in enumerate(bitvals):
        print('Doing',bitnames[i])
        w = np.where(targetdata["DESI_TARGET"] & bitval)
        if len(w[0]):
            ghist,dum = np.histogram(gmags[w],bins=binedges)
            rhist,dum = np.histogram(rmags[w],bins=binedges)
            zhist,dum = np.histogram(zmags[w],bins=binedges)
            file.write('{}    {}     {}     {}\n'.format(bitnames[i],'g','r','z'))
            for i in range(len(binedges)-1):
                outs = '{:.1f} {} {} {}\n'.format(0.5*(binedges[i]+binedges[i+1]),ghist[i],rhist[i],zhist[i])
                print(outs)
                file.write(outs)

    file.close()

    return None


def construct_QA_file(nrows):
    """Create a recarray to be populated with QA information

    Parameters
    ----------
    nrows : :class:`int`
        Number of rows in the recarray (size, in rows, of expected fits output)

    Returns
    -------
    :class:`~numpy.ndarray` 
         numpy structured array of brick information with nrows as specified
         and columns as below
    """

    data = np.zeros(nrows, dtype=[
            ('BRICKID','>i4'),('BRICKNAME','S8'),('BRICKAREA','>f4'),
            ('RA','>f4'),('DEC','>f4'),
            ('RA1','>f4'),('RA2','>f4'),
            ('DEC1','>f4'),('DEC2','>f4'),
            ('EBV','>f4'),
            ('PSFDEPTH_G','>f4'),('PSFDEPTH_R','>f4'),('PSFDEPTH_Z','>f4'),
            ('GALDEPTH_G','>f4'),('GALDEPTH_R','>f4'),('GALDEPTH_Z','>f4'),
            ('PSFDEPTH_G_PERCENTILES','f4',(5)), ('PSFDEPTH_R_PERCENTILES','f4',(5)),
            ('PSFDEPTH_Z_PERCENTILES','f4',(5)), ('GALDEPTH_G_PERCENTILES','f4',(5)),
            ('GALDEPTH_R_PERCENTILES','f4',(5)), ('GALDEPTH_Z_PERCENTILES','f4',(5)),
            ('NEXP_G','i2'),('NEXP_R','i2'),('NEXP_Z','i2'),
            ('DENSITY_ALL','>f4'),
            ('DENSITY_ELG','>f4'),('DENSITY_LRG','>f4'),
            ('DENSITY_QSO','>f4'),('DENSITY_LYA','>f4'),
            ('DENSITY_BGS','>f4'),('DENSITY_MWS','>f4'),
            ('DENSITY_BAD_ELG','>f4'),('DENSITY_BAD_LRG','>f4'),
            ('DENSITY_BAD_QSO','>f4'),('DENSITY_BAD_LYA','>f4'),
            ('DENSITY_BAD_BGS','>f4'),('DENSITY_BAD_MWS','>f4'),
            ])
    return data

def construct_HPX_file(nrows):
    """Create a recarray to be populated with HEALPixel information

    Parameters
    ----------
    nrows : :class:`int`
        Number of rows in the recarray (size, in rows, of expected fits output)

    Returns
    -------
    :class:`~numpy.ndarray` 
         numpy structured array to be populated with HEALPixel information with 
         nrows as specified and columns as below
    """

    data = np.zeros(nrows, dtype=[
            ('HPXID','>i4'),('HPXAREA','>f4'),
            ('RA','>f4'),('DEC','>f4'),
            ('EBV','>f4'),
            ('PSFDEPTH_G','>f4'),('PSFDEPTH_R','>f4'),('PSFDEPTH_Z','>f4'),
            ('GALDEPTH_G','>f4'),('GALDEPTH_R','>f4'),('GALDEPTH_Z','>f4'),
            ('PSFDEPTH_G_PERCENTILES','f4',(5)), ('PSFDEPTH_R_PERCENTILES','f4',(5)),
            ('PSFDEPTH_Z_PERCENTILES','f4',(5)), ('GALDEPTH_G_PERCENTILES','f4',(5)),
            ('GALDEPTH_R_PERCENTILES','f4',(5)), ('GALDEPTH_Z_PERCENTILES','f4',(5)),
            ('NEXP_G','i2'),('NEXP_R','i2'),('NEXP_Z','i2'),
            ('DENSITY_ALL','>f4'),
            ('DENSITY_ELG','>f4'),('DENSITY_LRG','>f4'),
            ('DENSITY_QSO','>f4'),('DENSITY_LYA','>f4'),
            ('DENSITY_BGS','>f4'),('DENSITY_MWS','>f4'),
            ])
    return data


def populate_brick_info(instruc,brickids,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3/'):
    """Add brick-related information to a numpy array of brickids

    Parameters
    ----------
    instruc : :class:`~numpy.ndarray` 
        numpy structured array containing at least
        ``BRICKNAME``,``BRICKID``,``RA``,``DEC``,``RA1``,``RA2``,``DEC1``,
        ``DEC2``,``NEXP_G``,``NEXP_R``,``NEXP_Z``,``EBV``]) to populate
    brickids : :class:`~numpy.ndarray` 
        numpy structured array (single list) of ``BRICKID`` integers
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory for a data release...e.g. for DR3 this would be
        ``/global/project/projectdirs/cosmo/data/legacysurvey/dr3/``

    Returns
    -------
    :class:`~numpy.ndarray` 
         instruc with the brick information columns now populated
    """

    #ADM columns to be read in from brick file
    cols = ['BRICKNAME','BRICKID','RA','DEC','RA1','RA2','DEC1','DEC2']

    #ADM read in the brick information file
    fx = fitsio.FITS(rootdirname+'/survey-bricks.fits.gz', upper=True)
    brickdata = fx[1].read(columns=cols)
    #ADM populate the coordinate/name/ID columns
    for col in cols:
        instruc[col] = brickdata[brickids-1][col]

    #ADM read in the data-release specific
    #ADM read in the brick information file
    fx = fitsio.FITS(glob(rootdirname+'/survey-bricks-dr*.fits.gz')[0], upper=True)
    ebvdata = fx[1].read(columns=['BRICKNAME','NEXP_G','NEXP_R','NEXP_Z','EBV'])

    #ADM as the BRICKID isn't in the dr-specific file, create
    #ADM a look-up dictionary to match indices via a list comprehension
    orderedbricknames = instruc["BRICKNAME"]
    dd = defaultdict(list)
    for index, item in enumerate(ebvdata["BRICKNAME"]):
        dd[item].append(index)
    matches = [index for item in orderedbricknames for index in dd[item] if item in dd]

    #ADM populate E(B-V) and NEXP
    instruc['NEXP_G'] = ebvdata[matches]['NEXP_G']
    instruc['NEXP_R'] = ebvdata[matches]['NEXP_R']
    instruc['NEXP_Z'] = ebvdata[matches]['NEXP_Z']
    instruc['EBV'] = ebvdata[matches]['EBV']

    return instruc


def populate_HPX_info(instruc,hpxids,nside=256):
    """Add HEALPixel-related information to a numpy array of HEALPixels

    Parameters
    ----------
    instruc : :class:`~numpy.ndarray` 
        numpy structured array containing at least
        ``HPXID``,``RA``,``DEC``, ``HPXAREA`` to populate
    hpxids : :class:`~numpy.ndarray` 
        numpy structured array (single list) of HEALPixel integers
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The HEALPix pixel nside number that was used to calculated the hpxids

    Returns
    -------
    :class:`~numpy.ndarray` 
         instruc with the HEALPixel information columns now populated
    """
    
    from desimodel import footprint
    import healpy

    #ADM determine the general HEALPixel area at this nside
    instruc["HPXAREA"] = hp.nside2pixarea(nside,degrees=True)
    #ADM multiply by the fraction of the HEALPixel that is outside of the DESI footprint
    pixweight = footprint.io.load_pixweight(nside)
    instruc["HPXAREA"] *= pixweight[instruc['HPXID']]

    #ADM populate the RA/Dec of the pixel
    theta, phi = hp.pix2ang(nside,instruc['HPXID'],nest=True)
    instruc["DEC"], instruc["RA"] = 90.-np.degrees(theta), np.degrees(phi)

    return instruc


def populate_depths(instruc,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3/'):
    """Add depth-related information to a numpy array

    Parameters
    ----------
    instruc : :class:`~numpy.ndarray` 
        numpy structured array containing at least
        ['BRICKNAME','BRICKAREA','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z',
        'GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','PSFDEPTH_G_PERCENTILES',
        'PSFDEPTH_R_PERCENTILES','PSFDEPTH_Z_PERCENTILES','GALDEPTH_G_PERCENTILES',
        'GALDEPTH_R_PERCENTILES','GALDEPTH_Z_PERCENTILES']
        to populate with depths and areas
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory for a data release...e.g. for DR3 this would be
        ``/global/project/projectdirs/cosmo/data/legacysurvey/dr3/``

    Returns
    -------
    :class:`~numpy.ndarray` 
         instruc with the per-brick depth and area columns now populated
    """
    #ADM the pixel scale area for a brick (in sq. deg.)
    pixtodeg = 0.262/3600./3600.

    #ADM read in the brick depth file
    fx = fitsio.FITS(glob(rootdirname+'*depth.fits.gz')[0], upper=True)
    depthdata = fx[1].read()

    #ADM construct the magnitude bin centers for the per-brick depth
    #ADM file, which is expressed as a histogram of 50 bins of 0.1mag
    magbins = np.arange(50)*0.1+20.05
    magbins[0] = 0

    #ADM percentiles at which to assess the depth
    percs = np.array([10,25,50,75,90])/100.

    #ADM lists to contain the brick names and for the depths, areas, percentiles
    names, areas = [], []
    depth_g, depth_r, depth_z = [], [], []
    galdepth_g, galdepth_r, galdepth_z= [], [], []
    perc_g, perc_r, perc_z = [], [], []
    galperc_g, galperc_r, galperc_z = [], [], []

    #ADM build a per-brick weighted depth. Also determine pixel-based area of brick.
    #ADM the per-brick depth file is histogram of 50 bins
    #ADM this grew organically, could make it more compact
    for i in range(0,len(depthdata),50):
        #ADM there must be measurements for all of the pixels in one band
        d = depthdata[i:i+50]
        totpix = sum(d['COUNTS_GAL_G']),sum(d['COUNTS_GAL_R']),sum(d['COUNTS_GAL_Z'])
        maxpix = max(totpix)
        #ADM percentiles in terms of pixel counts
        pixpercs = np.array(percs)*maxpix
        #ADM add pixel-weighted mean depth
        depth_g.append(np.sum(d['COUNTS_PTSRC_G']*magbins)/maxpix)
        depth_r.append(np.sum(d['COUNTS_PTSRC_R']*magbins)/maxpix)
        depth_z.append(np.sum(d['COUNTS_PTSRC_Z']*magbins)/maxpix)
        galdepth_g.append(np.sum(d['COUNTS_GAL_G']*magbins)/maxpix)
        galdepth_r.append(np.sum(d['COUNTS_GAL_R']*magbins)/maxpix)
        galdepth_z.append(np.sum(d['COUNTS_GAL_Z']*magbins)/maxpix)
        #ADM add name and pixel based area of the brick
        names.append(depthdata['BRICKNAME'][i])
        areas.append(maxpix*pixtodeg)
        #ADM add percentiles for depth...using a
        #ADM list comprehension, which is fast because the pixel numbers are ordered and
        #ADM says "give me the first magbin where we exceed a certain pixel percentile"
        if totpix[0]:
            perc_g.append([ magbins[np.where(np.cumsum(d['COUNTS_PTSRC_G']) > p )[0][0]] for p in pixpercs ])
            galperc_g.append([ magbins[np.where(np.cumsum(d['COUNTS_GAL_G']) > p )[0][0]] for p in pixpercs ])
        else:
            perc_g.append([0]*5)
            galperc_g.append([0]*5)

        if totpix[1]:
            perc_r.append([ magbins[np.where(np.cumsum(d['COUNTS_PTSRC_R']) > p )[0][0]] for p in pixpercs ])

            galperc_r.append([ magbins[np.where(np.cumsum(d['COUNTS_GAL_R']) > p )[0][0]] for p in pixpercs ])
        else:
            perc_r.append([0]*5)
            galperc_r.append([0]*5)

        if totpix[2]:
            perc_z.append([ magbins[np.where(np.cumsum(d['COUNTS_PTSRC_Z']) > p )[0][0]] for p in pixpercs ])
            galperc_z.append([ magbins[np.where(np.cumsum(d['COUNTS_GAL_Z']) > p )[0][0]] for p in pixpercs ])
        else:
            perc_z.append([0]*5)
            galperc_z.append([0]*5)

    #ADM HACK HACK HACK
    #ADM first find bricks that are not in the depth file and populate them
    #ADM with nonsense. This is a hack as I'm not sure why such bricks exist
    #ADM HACK HACK HACK
    orderedbricknames = instruc["BRICKNAME"]
    badbricks = np.ones(len(orderedbricknames))
    dd = defaultdict(list)
    for index, item in enumerate(orderedbricknames):
        dd[item].append(index)
    matches = [index for item in names for index in dd[item] if item in dd]
    badbricks[matches] = 0
    w = np.where(badbricks)
    badbricknames = orderedbricknames[w]
    for i, badbrickname in enumerate(badbricknames):
        names.append(badbrickname)
        areas.append(-99.)
        depth_g.append(-99.)
        depth_r.append(-99.)
        depth_z.append(-99.)
        galdepth_g.append(-99.)
        galdepth_r.append(-99.)
        galdepth_z.append(-99.)
        perc_g.append([-99.]*5)
        perc_r.append([-99.]*5)
        perc_z.append([-99.]*5)
        galperc_g.append([-99.]*5)
        galperc_r.append([-99.]*5)
        galperc_z.append([-99.]*5)

    #ADM, now order the brickname to match the input structure using
    #ADM a look-up dictionary to match indices via a list comprehension
    orderedbricknames = instruc["BRICKNAME"]
    dd = defaultdict(list)
    for index, item in enumerate(names):
        dd[item].append(index)
    matches = [index for item in orderedbricknames for index in dd[item] if item in dd]

    #ADM populate the depths and area
    instruc['BRICKAREA'] = np.array(areas)[matches]
    instruc['PSFDEPTH_G'] = np.array(depth_g)[matches]
    instruc['PSFDEPTH_R'] = np.array(depth_r)[matches]
    instruc['PSFDEPTH_Z'] = np.array(depth_z)[matches]
    instruc['GALDEPTH_G'] = np.array(galdepth_g)[matches]
    instruc['GALDEPTH_R'] = np.array(galdepth_r)[matches]
    instruc['GALDEPTH_Z'] = np.array(galdepth_z)[matches]
    instruc['PSFDEPTH_G_PERCENTILES'] = np.array(perc_g)[matches]
    instruc['PSFDEPTH_R_PERCENTILES'] = np.array(perc_r)[matches]
    instruc['PSFDEPTH_Z_PERCENTILES'] = np.array(perc_z)[matches]
    instruc['GALDEPTH_G_PERCENTILES'] = np.array(galperc_g)[matches]
    instruc['GALDEPTH_R_PERCENTILES'] = np.array(galperc_r)[matches]
    instruc['GALDEPTH_Z_PERCENTILES'] = np.array(galperc_z)[matches]

    return instruc


def populate_HPX_depths(instruc,targetdata):
    """Add depth-related information to a numpy array

    Parameters
    ----------
    instruc : :class:`~numpy.ndarray` 
        numpy structured array containing at least
        ['HPXID','HPXAREA','EBV','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z',
        'GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','PSFDEPTH_G_PERCENTILES',
        'PSFDEPTH_R_PERCENTILES','PSFDEPTH_Z_PERCENTILES','GALDEPTH_G_PERCENTILES',
        'GALDEPTH_R_PERCENTILES','GALDEPTH_Z_PERCENTILES']
        to populate with depths and areas
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory for a data release...e.g. for DR3 this would be
        ``/global/project/projectdirs/cosmo/data/legacysurvey/dr3/``

    Returns
    -------
    :class:`~numpy.ndarray` 
         instruc with the per-brick depth and area columns now populated
    """
    #ADM sort the input targetdata array on HEALPixel number
    targsorted = targetdata[targetdata['HPXID'].argsort()]
    #ADM find the edges of the sorted array corresponding to blocks of HEALPixels
    binedges = np.insert(instruc['HPXID'],len(instruc),instruc['HPXID'][-1]+1)
    h, dum = np.histogram(targsorted['HPXID'],bins=binedges)
    h = np.cumsum(h)
    #ADM insert a zero at the beginning to delineate the first block
    h = np.insert(h,0,0)

    #ADM percentiles at which to assess the depth
    percs = np.array([10,25,50,75,90])

    #ADM the column names that contain depth but not depth percentiles
    depthcolnames = [ s for s in instruc.dtype.names 
                      if "DEPTH" in s and "PERCENTILES" not in s ]

    #ADM loop through the blocks of HEALPixels and populate HEALPixel-level info
    for i in range(len(h)-1):

        #ADM check that some sorting bug hasn't caused a mismatch on HEALPixel
        if instruc[i]['HPXID'] != targsorted[h[i]]['HPXID']:
            log.error('HEALPixel mismatch between targets and HPX file!')

        #ADM populate the depth-based information
        for colname in depthcolnames:
            #ADM populate the mean depth fluxes
            meandepth = np.mean(targsorted[h[i]:h[i+1]][colname])
            #ADM guard against exposure with no observations in this band
            sig5fluxes = 5./np.sqrt(np.clip(meandepth,2.5e-17,2.5e17))
            instruc[colname][i] = 22.5-2.5*np.log10(sig5fluxes)
            #ADM populate the percentiles remember to flip as fluxes->mags
            pcolname = colname+'_PERCENTILES'
            fluxpercs = np.percentile(sig5fluxes,percs)
            instruc[pcolname][i] = np.flipud(22.5-2.5*np.log10(fluxpercs))

        #ADM populate the information on the number of exposures
        for band in 'GRZ':
            nobs = targsorted[h[i]:h[i+1]]['NOBS_'+band]
            #ADM round to the nearest integer
            instruc['NEXP_'+band][i] = np.rint(np.mean(nobs))
            
        #ADM populate the Galactic extinction 
        colname = 'MW_TRANSMISSION_G'
        #ADM cull to non-zero entries for the transmission information as
        #ADM there was a DR3 bug where some transmission values were zero
        w = np.where(targsorted[h[i]:h[i+1]][colname] > 0)
        #ADM calculate mean in linear transmission units, guarding against
        #ADM the consequences of the DR3 transmission bug
        if len(w[0]) == 0:
            mwt = 1.4e-13
        else:
            mwt = np.mean(targsorted[h[i]:h[i+1]][colname][w])
        #ADM convert to E(B-V) using 3.214 from, e.g.
        #ADM http://legacysurvey.org/dr4/catalogs/
        #ADM THE 3.214 A/E(B_V) COEFFICIENT COULD BE UPDATED IN FUTURE DRs!!!
        instruc['EBV'][i] = -2.5*np.log10(mwt)/3.214

    return instruc


def convert_target_data_model_for_QA(instruc):
    """Convert a subset of columns in a pre-DR4 targets file to the DR4 data model

    Parameters
    ----------
    instruc : :class:`~numpy.ndarray` 
        numpy structured array that contains at least
        ``DECAM_FLUX, DECAM_MW_TRANSMISSION, 
        DECAM_NOBS, DECAM_DEPTH, DECAM_GALDEPTH``
        to convert to the new data model.

    Returns
    -------
    :class:`~numpy.ndarray` 
        input structure with the ``DECAM_`` columns converted to the DR4+ data model
    """
    #ADM the old DECAM_ columns that need to be updated
    decamcols = ['FLUX','MW_TRANSMISSION','NOBS','GALDEPTH']
    decambands = 'UGRIZ'

    #ADM determine the data structure of the input array
    dt = instruc.dtype.descr
    names = list(instruc.dtype.names)
    #ADM remove the old column names
    for colstring in decamcols:
        loc = names.index('DECAM_'+colstring)
        names.pop(loc)
        dt.pop(loc)
        for bandnum in [1,2,4]:
            dt.append((colstring+"_"+decambands[bandnum],'>f4'))
    #ADM treat DECAM_DEPTH separately as the syntax is slightly different
    loc = names.index('DECAM_DEPTH')
    dt.pop(loc)
    for bandnum in [1,2,4]:
        dt.append(('PSFDEPTH_'+decambands[bandnum],'>f4'))

    #ADM create a new numpy array with the fields from the new data model...
    nrows = len(instruc)
    outstruc = np.empty(nrows, dtype=dt)

    #ADM change the DECAM columns from the old (2-D array) to new (named 1-D array) data model
    for bandnum in [1,2,4]:
        for colstring in decamcols:
            outstruc[colstring+"_"+decambands[bandnum]] = instruc["DECAM_"+colstring][:,bandnum]
        #ADM treat DECAM_DEPTH separately as the syntax is slightly different
        outstruc["PSFDEPTH_"+decambands[bandnum]] = instruc["DECAM_DEPTH"][:,bandnum]

    #ADM finally, populate the columns that haven't changed
    newcols = list(outstruc.dtype.names)
    oldcols = list(instruc.dtype.names)
    sharedcols = list(set(newcols).intersection(oldcols))
    for col in sharedcols:
        outstruc[col] = instruc[col]

    return outstruc


def HPX_info(targetfilename,outfilename='hp-info-dr3.fits',nside=256):
    """Create a file containing information in HEALPixels (depth, ebv, etc. as in construct_HPX_file)

    Parameters
    ----------
    targetfilename : :class:`str`
        File name of a list of targets created by select_targets
    outfilename: :class:`str`, defaults to 'hp-info-dr3.fits'
        Output file name for the hp_info file, which will be written as FITS
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The HEALPix pixel nside number

    Returns
    -------
    :class:`~numpy.ndarray` 
         numpy structured array of HEALPix information with columns as in construct_HPX_file
    """

    import healpy as hp
    start = time()

    #ADM set up log tracker
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    #ADM read in target file
    log.info('Reading in target file...t = {:.1f}s'.format(time()-start))
    indata = fitsio.read(targetfilename, upper=True)
    log.info("Working with {} targets".format(len(indata)))

    log.info('Removing targets that are outside DESI...t = {:.1f}s'.format(time()-start))
    #ADM remove targets that are outside the official DESI footprint
    from desimodel import footprint, io
    indesi = footprint.is_point_in_desi(io.load_tiles(),indata["RA"],indata["DEC"])
    w = np.where(indesi)
    if len(w[0]) > 0:
        indata = indata[w]
        log.info("{} targets in official DESI footprint".format(len(indata)))
    else:
        log.error("ZERO input targets are within the official DESI footprint!!!")

    #ADM if this is an old-style, pre-DR4 file, convert it to the new data model
    if 'DECAM_FLUX' in indata.dtype.names:
        log.info('Converting from old (pre-DR4) to new DR4 data model...t = {:.1f}s'
                 .format(time()-start))
        indata = convert_target_data_model_for_QA(indata)

    #ADM add a column "HPXID" containing the HEALPix number
    log.info('Gathering HEALPixel information...t = {:.1f}s'.format(time()-start))
    nrows = len(indata)
    theta, phi = np.radians(90-indata["DEC"]), np.radians(indata["RA"])
    hppix = hp.ang2pix(nside, theta, phi, nest=True)
    dt = indata.dtype.descr
    dt.append(('HPXID','>i8'))
    #ADM create a new array with all of the read-in column names and the new HPXID
    targetdata = np.empty(nrows, dtype=dt)
    for colname in indata.dtype.names:
        targetdata[colname] = indata[colname]
    targetdata["HPXID"] = hppix

    log.info('Determining unique HEALPixels...t = {:.1f}s'.format(time()-start))
    #ADM determine number of unique bricks and their integer IDs
    hpxids = np.array(list(set(targetdata['HPXID'])))
    hpxids.sort()

    log.info('Creating output HEALPixel structure...t = {:.1f}s'.format(time()-start))
    #ADM set up an output structure of size of the number of unique bricks
    npix = len(hpxids)
    outstruc = construct_HPX_file(npix)
    outstruc['HPXID'] = hpxids

    log.info('Adding HEALPixel information...t = {:.1f}s'.format(time()-start))
    #ADM add HEALPixel-specific information based on the HEALPixel IDs
    outstruc = populate_HPX_info(outstruc,hpxids,nside)

    log.info('Adding depth information...t = {:.1f}s'.format(time()-start))
    #ADM add per-brick depth and area information
    outstruc = populate_HPX_depths(outstruc,targetdata)

    log.info('Adding target density information...t = {:.1f}s'.format(time()-start))
    #ADM bits and names of interest for desitarget
    bitnames = ["DENSITY_ALL","DENSITY_LRG","DENSITY_ELG",
                "DENSITY_QSO","DENSITY_BGS","DENSITY_MWS"]
    #ADM -1 as a bit will return all values
    bitvals = [-1]
    for bitname in ['LRG','ELG','QSO','BGS_ANY','MWS_ANY']:
        bitvals.append(desi_mask[bitname])

    #ADM loop through bits and populate target densities for each class
    for i, bitval in enumerate(bitvals):
        w = np.where(targetdata["DESI_TARGET"] & bitval)
        if len(w[0]) > 0:
            targsperhpx = np.bincount(targetdata[w]['HPXID'],minlength=max(outstruc['HPXID'])+1)
            outstruc[bitnames[i]] = targsperhpx[outstruc['HPXID']]/outstruc['HPXAREA']

    log.info('Writing output file...t = {:.1f}s'.format(time()-start))
    #ADM everything should be populated, just write it out,
    #ADM also populate header info noting the HEALPixel scheme
    hdr = fitsio.FITSHDR()
    hdr['HPXNSIDE'] = nside
    hdr['HPXNEST'] = True

    fitsio.write(outfilename, outstruc, extname='HPXINFO', header=hdr, clobber=True)

    log.info('Done...t = {:.1f}s'.format(time()-start))
    return outstruc


def brick_info(targetfilename,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3/',outfilename='brick-info-dr3.fits'):
    """Create a file containing brick information (depth, ebv, etc. as in construct_QA_file)

    Parameters
    ----------
    targetfilename : :class:`str`
        File name of a list of targets created by select_targets
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory for a data release...e.g. for DR3 this would be
        ``/global/project/projectdirs/cosmo/data/legacysurvey/dr3/``
    outfilename: :class:`str`
        Output file name for the brick_info file, which will be written as FITS

    Returns
    -------
    :class:`~numpy.ndarray` 
         numpy structured array of brick information with columns as in construct_QA_file
    """

    start = time()

    #ADM read in target file
    print('Reading in target file...t = {:.1f}s'.format(time()-start))
    fx = fitsio.FITS(targetfilename, upper=True)
    targetdata = fx[1].read(columns=['BRICKID','DESI_TARGET','BGS_TARGET','MWS_TARGET'])

    #ADM add col

    print('Determining unique bricks...t = {:.1f}s'.format(time()-start))
    #ADM determine number of unique bricks and their integer IDs
    brickids = np.array(list(set(targetdata['BRICKID'])))
    brickids.sort()

    print('Creating output brick structure...t = {:.1f}s'.format(time()-start))
    #ADM set up an output structure of size of the number of unique bricks
    nbricks = len(brickids)
    outstruc = construct_QA_file(nbricks)

    print('Adding brick information...t = {:.1f}s'.format(time()-start))
    #ADM add brick-specific information based on the brickids
    outstruc = populate_brick_info(outstruc,brickids,rootdirname)

    print('Adding depth information...t = {:.1f}s'.format(time()-start))
    #ADM add per-brick depth and area information
    outstruc = populate_depths(outstruc,rootdirname)

    print('Adding target density information...t = {:.1f}s'.format(time()-start))
    #ADM bits and names of interest for desitarget
    #ADM -1 as a bit will return all values
    bitnames = ["DENSITY_ALL","DENSITY_LRG","DENSITY_ELG",
                "DENSITY_QSO","DENSITY_BGS","DENSITY_MWS"]
    bitvals = [-1]+list(2**np.array([0,1,2,60,61]))

    #ADM loop through bits and populate target densities for each class
    for i, bitval in enumerate(bitvals):
        w = np.where(targetdata["DESI_TARGET"] & bitval)
        if len(w[0]):
            targsperbrick = np.bincount(targetdata[w]['BRICKID'])
            outstruc[bitnames[i]] = targsperbrick[outstruc['BRICKID']]/outstruc['BRICKAREA']

    print('Writing output file...t = {:.1f}s'.format(time()-start))
    #ADM everything should be populated, just write it out
    fitsio.write(outfilename, outstruc, extname='BRICKINFO', clobber=True)

    print('Done...t = {:.1f}s'.format(time()-start))
    return outstruc


def _load_targdens():
    """Loads the target info dictionary as in :func:`desimodel.io.load_target_info()` and
    extracts the target density information in a format useful for targeting QA plots
    """

    from desimodel import io
    targdict = io.load_target_info()

    targdens = {}
    targdens['ELG'] = targdict['ntarget_elg']
    targdens['LRG'] = targdict['ntarget_lrg']
    targdens['QSO'] = targdict['ntarget_qso'] + targdict['ntarget_lya']
    targdens['BGS_ANY'] = targdict['ntarget_bgs_bright'] + targdict['ntarget_bgs_faint']
    targdens['STD_FSTAR'] = 0
    targdens['STD_BRIGHT'] = 0
    #ADM set "ALL" to be the sum over all the target classes
    targdens['ALL'] = sum(list(targdens.values()))

    return targdens


def _javastring():
    """Return a string that embeds a date in a webpage
    """

    js = textwrap.dedent("""
    <SCRIPT LANGUAGE="JavaScript">
    var months = new Array(13);
    months[1] = "January";
    months[2] = "February";
    months[3] = "March";
    months[4] = "April";
    months[5] = "May";
    months[6] = "June";
    months[7] = "July";
    months[8] = "August";
    months[9] = "September";
    months[10] = "October";
    months[11] = "November";
    months[12] = "December";
    var dateObj = new Date(document.lastModified)
    var lmonth = months[dateObj.getMonth() + 1]
    var date = dateObj.getDate()
    var fyear = dateObj.getYear()
    if (fyear < 2000)
    fyear = fyear + 1900
    if (date == 1 || date == 21 || date == 31)
    document.write(" " + lmonth + " " + date + "st, " + fyear)
    else if (date == 2 || date == 22)
    document.write(" " + lmonth + " " + date + "nd, " + fyear)
    else if (date == 3 || date == 23)
    document.write(" " + lmonth + " " + date + "rd, " + fyear)
    else
    document.write(" " + lmonth + " " + date + "th, " + fyear)    
    </SCRIPT>
    """)

    return js


def qaskymap(cat, objtype, qadir='.', upclip=None, weights=None, max_bin_area=1.0, fileprefix="skymap"):
    """Visualize the target density with a skymap. First version lifted 
    shamelessly from :mod:`desitarget.mock.QA` (which was originally written by `J. Moustakas`)

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``RA`` and ``DEC`` columns for coordinate
        information
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    upclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the targets at the "high density" end to make plots 
        conform to similar density scales
    weights : :class:`~numpy.array`, optional, defaults to None
        A weight for each of the passed targets (e.g., to upweight each target in a
        partial pixel at the edge of the DESI footprint)
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in the passed coordinates is chosen automatically to be as close as
        possible to this value without exceeding it
    fileprefix : :class:`str`, optional, defaults to ``"radec"`` for (RA/Dec)
        string to be added to the front of the output file name

    Returns
    -------
    Nothing
        But a .png plot of target densities is written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{objtype}.png``
    """

    label = '{} (targets/deg$^2$)'.format(objtype)
    fig, ax = plt.subplots(1)
    ax = np.atleast_1d(ax)
       
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        basemap = init_sky(galactic_plane_color='k', ax=ax[0]);
        plot_sky_binned(cat['RA'], cat['DEC'], weights=weights, max_bin_area=max_bin_area,
                        clip_lo='!1', clip_hi=upclip, cmap='jet', plot_type='healpix', 
                        label=label, basemap=basemap)

    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefix,objtype))
    fig.savefig(pngfile,bbox_inches='tight')

    plt.close()

    return


def qahisto(cat, objtype, qadir='.', targdens=None, upclip=None, weights=None, max_bin_area=1.0, 
            fileprefix="histo", catispix=False):
    """Visualize the target density with a histogram of densities. First version taken 
    shamelessly from :mod:`desitarget.mock.QA` (which was originally written by `J. Moustakas`)

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``RA`` and ``DEC`` columns for coordinate
        information
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    targdens : :class:`dictionary`, optional, defaults to None
        A dictionary of DESI target classes and the goal density for that class. Used, if
        passed, to label the goal density on the histogram plot        
    upclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the targets at the "high density" end to make plots 
        conform to similar density scales
    weights : :class:`~numpy.array`, optional, defaults to None
        A weight for each of the passed targets (e.g., to upweight each target in a
        partial pixel at the edge of the DESI footprint)
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in the passed coordinates is chosen automatically to be as close as
        possible to this value without exceeding it
    fileprefix : :class:`str`, optional, defaults to ``"histo"``
        string to be added to the front of the output file name
    catispix : :class:`boolean`, optional, defaults to ``False``
        If this is ``True``, then ``cat`` corresponds to the HEALpixel numbers already
        precomputed using ``pixels = footprint.radec2pix(nside, cat["RA"], cat["DEC"])``
        from the RAs and Decs ordered as for ``weights``, rather than the catalog itself.
        If this is True, then max_bin_area must correspond to the `nside` used to
        precompute the pixel numbers

    Returns
    -------
    Nothing
        But a .png histogram of target densities is written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{objtype}.png``
    """

    import healpy as hp

    #ADM determine the nside for the passed max_bin_area
    for n in range(1, 25):
        nside = 2 ** n
        bin_area = hp.nside2pixarea(nside, degrees=True)
        if bin_area <= max_bin_area:
            break
        
    #ADM the number of HEALPixels and their area at this nside
    npix = hp.nside2npix(nside)
    bin_area = hp.nside2pixarea(nside, degrees=True)

    #ADM the HEALPixel number for each RA/Dec (this call to desimodel
    #ADM assumes nest=True, so "weights" should assume nest=True, too)
    if catispix:
        pixels = cat.copy()
    else:
        from desimodel import footprint
        pixels = footprint.radec2pix(nside, cat["RA"], cat["DEC"])
    counts = np.bincount(pixels, weights=weights, minlength=npix)
    dens = counts[np.flatnonzero(counts)] / bin_area

    label = '{} (targets/deg$^2$)'.format(objtype)

    #ADM clip the targets to avoid high densities, if requested
    if upclip:
        dens = np.clip(dens,1,upclip)

    #ADM set the number of bins for the histogram (determined from trial and error)
    nbins = 80
    #ADM low density objects (QSOs and standard stars) look better with fewer bins
    if np.max(dens) < 500:
        nbins = 40
    #ADM the density value of the peak histogram bin
    h, b = np.histogram(dens,bins=nbins)
    peak = np.mean(b[np.argmax(h):np.argmax(h)+2])
    ypeak = np.max(h)

    #ADM set up and make the plot
    plt.clf()
    #ADM only plot to just less than upclip, to prevent displaying pile-ups in that bin
    plt.xlim((0,0.95*upclip))
    #ADM give a little space for labels on the y-axis
    plt.ylim((0,ypeak*1.2))
    plt.xlabel(label)
    plt.ylabel('Number of HEALPixels')
    plt.hist(dens, bins=nbins, histtype='stepfilled', alpha=0.6, 
             label='Observed {} Density (Peak={:.0f} per sq. deg.)'.format(objtype,peak))
    if objtype in targdens.keys():
        plt.axvline(targdens[objtype], ymax=0.8, ls='--', color='k', 
                    label='Goal {} Density (Goal={:.0f} per sq. deg.)'.format(objtype,targdens[objtype]))
    plt.legend(loc='upper left', frameon=False)

    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefix,objtype))
    plt.savefig(pngfile,bbox_inches='tight')

    plt.close()

    return


def qacolor(cat, objtype, qadir='.', fileprefix="color"):
    """Make color-based DESI targeting QA plots given a passed set of targets

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``FLUX_G``, ``FLUX_R``, ``FLUX_Z`` and 
        ``FLUX_W1``, ``FLUX_W2`` columns for color information
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    fileprefix : :class:`str`, optional, defaults to ``"color"`` for
        string to be added to the front of the output file name

    Returns
    -------
    Nothing
        But .png plots of target colors are written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{bands}-{objtype}.png``
        where bands might be, e.g., ``grz``
    """

    from matplotlib.colors import LogNorm

    #ADM convenience function to retrieve and unextinct DESI fluxes
    from desitarget.cuts import unextinct_fluxes
    flux = unextinct_fluxes(cat)

    #ADM convert to magnitudes (fluxes are in nanomaggies)
    #ADM should be fine to clip for plotting purposes
    loclip = 1e-16
    g = 22.5-2.5*np.log10(flux['GFLUX'].clip(loclip))
    r = 22.5-2.5*np.log10(flux['RFLUX'].clip(loclip))
    z = 22.5-2.5*np.log10(flux['ZFLUX'].clip(loclip))
    W1 = 22.5-2.5*np.log10(flux['W1FLUX'].clip(loclip))
    W2 = 22.5-2.5*np.log10(flux['W2FLUX'].clip(loclip))

    #ADM set up the r-z, g-r plot
    plt.clf()
    plt.xlabel('r - z')
    plt.ylabel('g - r')
    plt.set_cmap('inferno')
    plt.hist2d(r-z,g-r,bins=100,range=[[-1,3],[-1,3]],norm=LogNorm())
    plt.colorbar()
    #ADM make the plot
    pngfile = os.path.join(qadir, '{}-grz-{}.png'.format(fileprefix,objtype))
    plt.savefig(pngfile,bbox_inches='tight')
    plt.close()

    #ADM set up the r-z, r-W1 plot
    plt.clf()
    plt.xlabel('r - z')
    plt.ylabel('r - W1')
    plt.set_cmap('inferno')
    plt.hist2d(r-z,r-W1,bins=100,range=[[-1,3],[-1,3]],norm=LogNorm())
    plt.colorbar()
    #ADM make the plot
    pngfile = os.path.join(qadir, '{}-rzW1-{}.png'.format(fileprefix,objtype))
    plt.savefig(pngfile,bbox_inches='tight')
    plt.close()


def make_qa_plots(targs, qadir='.', targdens=None, max_bin_area=1.0, weight=True):
    """Make DESI targeting QA plots given a passed set of targets

    Parameters
    ----------
    targs : :class:`~numpy.array` or `str`
        An array of targets in the DESI data model format. If a string is passed then the
        targets are read fron the file with the passed name (supply the full directory path)
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    targdens : :class:`dictionary`, optional, set automatically by the code if not passed
        A dictionary of DESI target classes and the goal density for that class. Used to
        label the goal density on histogram plots
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in the passed coordinates is chosen automatically to be as close as
        possible to this value without exceeding it
    weight : :class:`boolean`, optional, defaults to True
        If this is set, weight pixels using the ``DESIMODEL`` HEALPix footprint file to
        ameliorate under dense pixels at the footprint edges

    Returns
    -------
    Nothing
        But a set of .png plots for target QA are written to qadir

    Notes
    -----
    The ``DESIMODEL`` environment variable must be set to find the file of HEALPixels 
    that overlap the DESI footprint
    """

    #ADM set up the default logger from desiutil
    from desimodel import io, footprint
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()
    log.info('Start making targeting QA density plots...t = {:.1f}s'.format(time()-start))

    #ADM if a filename was passed, read in the targets from that file
    if isinstance(targs, str):
        targs = fitsio.read(targs)
        log.info('Read in targets...t = {:.1f}s'.format(time()-start))

    #ADM restrict targets to just the DESI footprint
    indesi = footprint.is_point_in_desi(io.load_tiles(),targs["RA"],targs["DEC"])
    targs = targs[indesi]
    log.info('Restricted targets to DESI footprint...t = {:.1f}s'.format(time()-start))

    #ADM determine the nside for the passed max_bin_area
    for n in range(1, 25):
        nside = 2 ** n
        bin_area = hp.nside2pixarea(nside, degrees=True)
        if bin_area <= max_bin_area:
            break

    #ADM calculate HEALPixel numbers once, here, to avoid repeat calculations
    #ADM downstream
    pix = footprint.radec2pix(nside, targs["RA"], targs["DEC"])
    log.info('Calculated HEALPixel for each target...t = {:.1f}s'
             .format(time()-start))

    #ADM set up the weight of each HEALPixel, if requested.
    weights = np.ones(len(targs))
    if weight:
        #ADM retrieve the map of what HEALPixels are actually in the DESI footprint
        pixweight = io.load_pixweight(nside)
        #ADM determine what HEALPixels each target is in, to set the weights
        fracarea = pixweight[pix]
        #ADM weight by 1/(the fraction of each pixel that is in the DESI footprint)
        #ADM except for zero pixels, which are all outside of the footprint
        w = np.where(fracarea == 0)
        fracarea[w] = 1 #ADM to guard against division by zero warnings
        weights = 1./fracarea
        weights[w] = 0

        log.info('Assigned weights to pixels based on DESI footprint...t = {:.1f}s'
                 .format(time()-start))

    #ADM Current goal target densities for DESI (read from the DESIMODEL defaults)
    if targdens is None:
        targdens = _load_targdens()

    #ADM clip the target densities at an upper density to improve plot edges
    #ADM by rejecting highly dense outliers
    upclipdict = {'ELG': 4000, 'LRG': 1200, 'QSO': 400, 'ALL': 8000,
                  'STD_FSTAR': 200, 'STD_BRIGHT': 50, 'BGS_ANY': 4500}

    for objtype in targdens:
        if 'ALL' in objtype:
            w = np.arange(len(targs))
        else:
            w = np.where(targs["DESI_TARGET"] & desi_mask[objtype])

        #ADM make RA/Dec skymaps
        qaskymap(targs[w], objtype, qadir=qadir, upclip=upclipdict[objtype], 
                 weights=weights[w], max_bin_area=max_bin_area)

        log.info('Made sky map for {}...t = {:.1f}s'.format(objtype,time()-start))

        #ADM make histograms of densities. We already calculated the correctly 
        #ADM ordered HEALPixels and so don't need to repeat that calculation
        qahisto(pix[w], objtype, qadir=qadir, targdens=targdens, upclip=upclipdict[objtype], 
                weights=weights[w], max_bin_area = max_bin_area, catispix=True)

        log.info('Made histogram for {}...t = {:.1f}s'.format(objtype,time()-start))

        #ADM make color-color plots
        qacolor(targs[w], objtype, qadir=qadir, fileprefix="color")

        log.info('Made color-color plot for {}...t = {:.1f}s'.format(objtype,time()-start))

    log.info('Made QA density plots...t = {:.1f}s'.format(time()-start))


def make_qa_page(targs, makeplots=True, max_bin_area=1.0, qadir='.', weight=True):
    """Create a directory containing a webpage structure in which to embed QA plots

    Parameters
    ----------
    targs : :class:`~numpy.array` or `str`
        An array of targets in the DESI data model format. If a string is passed then the
        targets are read fron the file with the passed name (supply the full directory path)
    makeplots : :class:`boolean`, optional, default=True
        If ``True``, then create the plots as well as the webpage
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in the passed coordinates is chosen automatically to be as close as
        possible to this value without exceeding it
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    weight : :class:`boolean`, optional, defaults to True
        If this is set, weight pixels using the ``DESIMODEL`` HEALPix footprint file to
        ameliorate under dense pixels at the footprint edges
    Returns
    -------
    Nothing
        But the page `index.html` and associated pages and plots are written to ``qadir``

    Notes
    -----
    If making plots, then the ``DESIMODEL`` environment variable must be set to find 
    the file of HEALPixels that overlap the DESI footprint
    """
    
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()
    log.info('Start making targeting QA page...t = {:.1f}s'.format(time()-start))

    #ADM if a filename was passed, read in the targets from that file
    if isinstance(targs, str):
        targs = fitsio.read(targs)
        log.info('Read in targets...t = {:.1f}s'.format(time()-start))

    #ADM make a DR string based on the RELEASE column
    #ADM potentially there are multiple DRs in a file
    DRs = ", ".join([ "DR{}".format(release) for release in np.unique(targs["RELEASE"])//1000 ])

    #ADM Set up the names of the target classes and their goal densities using
    #ADM the goal target densities for DESI (read from the DESIMODEL defaults)
    targdens = _load_targdens()
    
    #ADM set up the html file and write preamble to it
    htmlfile = os.path.join(qadir, 'index.html')

    #ADM grab the magic string that writes the last-updated date to a webpage
    js = _javastring()

    #ADM html preamble
    html = open(htmlfile, 'w')
    html.write('<html><body>\n')
    html.write('<h1>DESI Targeting QA pages ({})</h1>\n'.format(DRs))

    #ADM links to each collection of plots for each object type
    html.write('<b><i>Jump to a target class:</i></b>\n')
    html.write('<ul>\n')
    for objtype in targdens.keys():
        html.write('<li><A HREF="{}.html">{}</A>\n'.format(objtype,objtype))
    html.write('</ul>\n')

    #ADM html postamble
    html.write('<b><i>Last updated {}</b></i>\n'.format(js))
    html.write('</html></body>\n')
    html.close()

    #ADM for each object type, make a separate page
    for objtype in targdens.keys():        
        #ADM call each page by the target class name, stick it in the requested directory
        htmlfile = os.path.join(qadir,'{}.html'.format(objtype))
        html = open(htmlfile, 'w')

        #ADM html preamble
        html.write('<html><body>\n')
        html.write('<h1>DESI Targeting QA pages - {} ({})</h1>\n'.format(objtype,DRs))

        #ADM Target Densities
        html.write('<h2>Target density plots</h2>\n')
        html.write('<table COLS=2 WIDTH="100%">\n')
        html.write('<tr>\n')
        #ADM add the plots...
        html.write('<td WIDTH="25%" align=left><A HREF="skymap-{}.png"><img SRC="skymap-{}.png" height=450 width=700></A></left></td>\n'
                   .format(objtype,objtype))
        html.write('<td WIDTH="25%" align=left><A HREF="histo-{}.png"><img SRC="histo-{}.png" height=430 width=510></A></left></td>\n'
                   .format(objtype,objtype))
        html.write('</tr>\n')
        html.write('</table>\n')

        #ADM color-color plots
        html.write('<h2>Color-color plots (corrected for Galactic extinction)</h2>\n')
        html.write('<table COLS=2 WIDTH="100%">\n')
        html.write('<tr>\n')
        #ADM add the plots...
        html.write('<td WIDTH="25%" align=left><A HREF="color-grz-{}.png"><img SRC="color-rzW1-{}.png" height=500 width=600></A></left></td>\n'
                   .format(objtype,objtype))
        html.write('<td WIDTH="25%" align=left><A HREF="color-rzW1-{}.png"><img SRC="color-grz-{}.png" height=500 width=600></A></left></td>\n'
                   .format(objtype,objtype))
        html.write('</tr>\n')
        html.write('</table>\n')

        #ADM html postamble
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    #ADM make the QA plots, if requested:
    if makeplots:
        make_qa_plots(targs, qadir=qadir, targdens=targdens, max_bin_area=max_bin_area, weight=weight)
