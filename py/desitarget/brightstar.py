# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.brightstar
=====================

Module for studying and masking bright stars in the sweeps
"""
from __future__ import (absolute_import, division)

from time import time
import numpy as np
import numpy.lib.recfunctions as rfn
import fitsio
from glob import glob
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.collections import PatchCollection

from . import __version__ as desitarget_version

from desitarget import io
from desitarget.internal import sharedmem
from desitarget import desi_mask, targetid_mask
from desitarget.targets import encode_targetid

from desiutil import depend, brick

import healpy as hp

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
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    plt.colorbar()

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
    a = np.arange(11)
    ellipses(a, a, w=4, h=a, rot=a*30, c=a, alpha=0.5, ec='none')
    plt.colorbar()

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


def max_objid_bricks(targs):
    """For a set of targets, return the maximum value of BRICK_OBJID in each BRICK_ID

    Parameters
    ----------
    targs : :class:`recarray`
        A recarray of targets as made by :mod:`desitarget.cuts.select_targets`

    Returns
    -------
    maxobjid : :class:`dictionary`
        A dictionary with keys for each unique BRICKID and values of the maximum OBJID in that brick
    """
    
    #ADM the maximum BRICKID in the passed target set
    brickmax = np.max(targs["BRICKID"])

    #ADM how many OBJIDs are in each unique brick, starting from 0 and ordered on BRICKID
    h = np.histogram(targs["BRICKID"],range=[0,brickmax],bins=brickmax)[0]
    #ADM remove zero entries from the histogram
    h = h[np.where(h > 0)]
    #ADM the index of the maximum OBJID in eacn brick if the bricks are ordered on BRICKID and OBJID
    maxind = np.cumsum(h)-1

    #ADM an array of BRICKID, OBJID sorted first on BRICKID and then on OBJID within each BRICKID
    ordered = np.array(sorted(zip(targs["BRICKID"],targs["BRICK_OBJID"]), key=lambda x: (x[0], x[1])))

    #ADM return a dictionary of the maximum OBJID (values) for each BRICKID (keys)
    return dict(ordered[maxind])


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
          sufficient for bright star mask purposes. But the equation in this function is more general.
        - We recast the input array as float64 to circumvent precision issues with np.cos()
          when radii of only a few arcminutes are passed
        - Even for passed radii of 1 (0.1) arcsec, float64 is sufficiently precise to give the correct
          area to ~0.00043 (~0.043%) using np.cos()
    """

    #ADM recast input array as float64
    theta = theta.astype('<f8')
    
    #ADM factor to convert steradians to sq.deg.
    st2sq = 180.*180./np.pi/np.pi

    #ADM return area
    return st2sq*2*np.pi*(1-(np.cos(np.radians(theta))))


def sphere_circle_ra_off(theta,centdec,declocs):
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

    #ADM convert the input angles from degrees to radians
    thetar = np.radians(theta)
    centdecr = np.radians(centdec)
    declocsr = np.radians(declocs)

    #ADM determine the offsets in RA from the small circle equation (easy to derive from, e.g. converting
    #ADM to Cartesian coordinates and using dot products). The answer is the arccos of the following:
    cosoffrar = (np.cos(thetar) - (np.sin(centdecr)*np.sin(declocsr))) / (np.cos(centdecr)*np.cos(declocsr))

    #ADM catch cases where the offset angle is very close to 0 
    offrar = np.arccos(np.clip(cosoffrar,-1,1))

    #ADM return the angular offsets in degrees
    return  np.degrees(offrar)


def collect_bright_stars(bands,maglim,numproc=4,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1',outfilename=None,verbose=False):
    """Extract a structure from the sweeps containing only bright stars in a given band to a given magnitude limit

    Parameters
    ----------
    bands : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z".
        Can pass multiple bands as string, e.g. "GRZ", in which case maglim has to be a
        list of the same length as the string
    maglim : :class:`float`
        The upper limit in that magnitude band for which to assemble a list of bright stars.
        Can pass a list of magnitude limits, in which case bands has to be a string of the
        same length (e.g., "GRZ" for [12.3,12.7,12.6]
    numproc : :class:`int`, optional
        Number of processes over which to parallelize
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory containing either sweeps or tractor files...e.g. for dr3 this might be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3/sweeps/dr3.1
    outfilename : :class:`str`, optional, defaults to not writing anything to file
        (FITS) File name to which to write the output structure of bright stars
    verbose : :class:`bool`, optional
        Send to write progress to screen

    Returns
    -------
    :class:`recarray`
        The structure of bright stars from the sweeps limited in the passed band(s) to the
        passed maglim(s).
    """

    #ADM use io.py to retrieve list of sweeps or tractor files
    infiles = io.list_sweepfiles(rootdirname)
    if len(infiles) == 0:
        infiles = io.list_tractorfiles(rootdirname)
    if len(infiles) == 0:
        raise IOError('No sweep or tractor files found in {}'.format(rootdirname))

    #ADM force the input maglim to be a list (in case a single value was passed)
    if type(maglim) == type(16) or type(maglim) == type(16.):
        maglim = [maglim]

    #ADM set bands to uppercase if passed as lower case
    bands = bands.upper()
    #ADM the band names as a flux array instead of a string
    bandnames = np.array([ "FLUX_"+band for band in bands ])

    if len(bandnames) != len(maglim):
        raise IOError('bands has to be the same length as maglim and {} does not equal {}'.format(len(bands),len(maglim)))

    #ADM change input magnitude(s) to a flux to test against
    fluxlim = 10.**((22.5-np.array(maglim))/2.5)

    #ADM parallel formalism from this step forward is stolen from cuts.select_targets

    #ADM function to grab the bright stars from a given file
    def _get_bright_stars(filename):
        '''Retrieves bright stars from a sweeps/Tractor file'''
        objs = io.read_tractor(filename)
        #ADM write the fluxes as an array instead of as named columns
        fluxes = objs[bandnames].view(objs[bandnames].dtype[0]).reshape(objs[bandnames].shape + (-1,))
        #ADM Retain rows for which ANY band is brighter than maglim
        w = np.where(np.any(fluxes > fluxlim,axis=1))
        if len(w[0]) > 0:
            return objs[w]

    #ADM counter for how many files have been processed
    #ADM critical to use np.ones because a numpy scalar allows in place modifications
    # c.f https://www.python.org/dev/peps/pep-3104/
    totfiles = np.ones((),dtype='i8')*len(infiles)
    nfiles = np.ones((), dtype='i8')
    t0 = time()
    if verbose:
        print('Collecting bright stars from sweeps...')

    def _update_status(result):
        '''wrapper function for the critical reduction operation,
        that occurs on the main parallel process'''
        if verbose and nfiles%25 == 0:
            elapsed = time() - t0
            rate = nfiles / elapsed
            print('{}/{} files; {:.1f} files/sec; {:.1f} total mins elapsed'.format(nfiles, totfiles, rate, elapsed/60.))
        nfiles[...] += 1  #this is an in-place modification
        return result

    #ADM did we ask to parallelize, or not?
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            starstruc = pool.map(_get_bright_stars, infiles, reduce=_update_status)
    else:
        starstruc = []
        for file in infiles:
            starstruc.append(_update_status(_get_bright_stars(file)))

    #ADM note that if there were no bright stars in a file then
    #ADM the _get_bright_stars function will have returned NoneTypes
    #ADM so we need to filter those out
    starstruc = [x for x in starstruc if x is not None]
    if len(starstruc) == 0:
        raise IOError('There are no stars brighter than {} in {} in files in {} with which to make a mask'.format(str(maglim),bands,rootdirname))
    #ADM concatenate all of the output recarrays
    starstruc = np.hstack(starstruc)

    #ADM if the name of a file for output is passed, then write to it
    if outfilename is not None:
        fitsio.write(outfilename, starstruc, clobber=True)

    return starstruc


def model_bright_stars(band,instarfile,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/'):

    """Build a dictionary of the fraction of bricks containing a star of a given
    magnitude in a given band as function of Galactic l and b

    Parameters
    ----------
    band : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z"
    instarfile : :class:`str`
        File of bright objects in (e.g.) sweeps, created by collect_bright_stars
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory for a data release...e.g. for dr3 this would be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/

    Returns
    -------
    :class:`dictionary`
        dictionary of the fraction of bricks containing a star of a given
        magnitude in a given band as function of Galactic l Keys are mag
        bin CENTERS, values are arrays running from 0->1 to 359->360
    :class:`dictionary`
        dictionary of the fraction of bricks containing a star of a given
        magnitude in a given band as function of Galactic b. Keys are mag
        bin CENTERS, values are arrays running from -90->-89 to 89->90

    Notes
    -----
        - converts using coordinates of the brick center, so is an approximation

    """
    #ADM histogram bin edges in Galactic coordinates at resolution of 1 degree
    lbinedges = np.arange(361)
    bbinedges = np.arange(-90,91)

    #ADM set band to uppercase if passed as lower case
    band = band.upper()

    #ADM read in the bright object file
    fx = fitsio.FITS(instarfile)
    objs = fx[1].read()
    #ADM convert fluxes in band of interest for each object to magnitudes
    mags = 22.5-2.5*np.log10(objs["FLUX_"+band])
    #ADM Galactic l and b for each object of interest
    c = SkyCoord(objs["RA"]*u.degree, objs["DEC"]*u.degree, frame='icrs')
    lobjs = c.galactic.l.degree
    bobjs = c.galactic.b.degree

    #ADM construct histogram bin edges in magnitude in passed band
    magstep = 0.1
    magmin = -1.5 #ADM magnitude of Sirius to 1 d.p.
    magmax = np.max(mags)
    magbinedges = np.arange(np.rint((magmax-magmin)/magstep))*magstep+magmin

    #ADM read in the data-release specific brick information file
    fx = fitsio.FITS(glob(rootdirname+'/survey-bricks-dr*.fits.gz')[0], upper=True)
    bricks = fx[1].read(columns=['RA','DEC'])

    #ADM convert RA/Dec of the brick center to Galatic coordinates and
    #ADM build a histogram of the number of bins at each coordinate...
    #ADM using the center is imperfect, so this is approximate at best
    c = SkyCoord(bricks["RA"]*u.degree, bricks["DEC"]*u.degree, frame='icrs')
    lbrick = c.galactic.l.degree
    bbrick = c.galactic.b.degree
    lhistobrick = (np.histogram(lbrick,bins=lbinedges))[0]
    bhistobrick = (np.histogram(bbrick,bins=bbinedges))[0]

    #ADM loop through the magnitude bins and populate a dictionary
    #ADM of the number of stars in this magnitude range per brick
    ldict, bdict = {}, {}
    for mag in magbinedges:
        key = "{:.2f}".format(mag+(0.5*magstep))
        #ADM range in magnitude
        w = np.where( (mags >= mag) & (mags < mag+magstep) )
        if len(w[0]):
            #ADM histograms of numbers of objects in l, b
            lhisto = (np.histogram(lobjs[w],bins=lbinedges))[0]
            bhisto = (np.histogram(bobjs[w],bins=bbinedges))[0]
            #ADM fractions of objects in l, b per brick
            #ADM use a sneaky where so that 0/0 results in 0
            lfrac = np.where(lhistobrick > 0, lhisto/lhistobrick, 0)
            bfrac = np.where(bhistobrick > 0, bhisto/bhistobrick, 0)
            #ADM populate the dictionaries
            ldict[key], bdict[key] = lfrac, bfrac

    return ldict, bdict


def make_bright_star_mask(bands,maglim,numproc=4,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1',infilename=None,outfilename=None,verbose=False):
    """Make a bright star mask from a structure of bright stars drawn from the sweeps

    Parameters
    ----------
    bands : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z".
        Can pass multiple bands as string, e.g. "GRZ", in which case maglim has to be a
        list of the same length as the string
    maglim : :class:`float`
        The upper limit in that magnitude band for which to assemble a list of bright stars.
        Can pass a list of magnitude limits, in which case bands has to be a string of the
        same length (e.g., "GRZ" for [12.3,12.7,12.6]
    numproc : :class:`int`, optional
        Number of processes over which to parallelize
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory containing either sweeps or tractor files...e.g. for dr3 this might be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3/sweeps/dr3.1
    infilename : :class:`str`, optional,
        if this exists, then the list of bright stars is read in from the file of this name
        if this is not passed, then code defaults to deriving the recarray of bright stars
        via a call to collect_bright_stars
    outfilename : :class:`str`, optional, defaults to not writing anything to file
        (FITS) File name to which to write the output bright star mask
    verbose : :class:`bool`, optional
        Send to write progress to screen

    Returns
    -------
    :class:`recarray`
        The bright star mask in the form RA, DEC, TARGETID, IN_RADIUS, NEAR_RADIUS (may also be written to file
        if "outfilename" is passed)
        The radii are in ARCMINUTES
        TARGETID is as calculated in :mod:`desitarget.targets.encode_targetid`

    Notes
    -----
        - IN_RADIUS is a smaller radius that corresponds to the IN_BRIGHT_OBJECT bit in data/targetmask.yaml
        - NEAR_RADIUS is a radius that corresponds to the NEAR_BRIGHT_OBJECT bit in data/targetmask.yaml
        - Currently uses the radius-as-a-function-of-B-mag for Tycho stars from the BOSS mask (in every band) to set
          the NEAR_RADIUS:
          R = (0.0802B*B - 1.860B + 11.625) (see Eqn. 9 of https://arxiv.org/pdf/1203.6594.pdf)
          and half that radius to set the IN_RADIUS.
        - It's an open question as to what the correct radii are for DESI observations

    """

    #ADM set bands to uppercase if passed as lower case
    bands = bands.upper()
    #ADM the band names and nobs columns as arrays instead of strings
    bandnames = np.array([ "FLUX_"+band for band in bands ])
    nobsnames = np.array([ "NOBS_"+band for band in bands ])

    #ADM force the input maglim to be a list (in case a single value was passed)
    if type(maglim) == type(16) or type(maglim) == type(16.):
        maglim = [maglim]

    if len(bandnames) != len(maglim):
        raise IOError('bands has to be the same length as maglim and {} does not equal {}'.format(len(bandnames),len(maglim)))

    #ADM change input magnitude(s) to a flux to test against
    fluxlim = 10.**((22.5-np.array(maglim))/2.5)

    if infilename is not None:
        objs = io.read_tractor(infilename)
    else:
        objs = collect_bright_stars(bands,maglim,numproc,rootdirname,outfilename,verbose)

    #ADM write the fluxes and bands as arrays instead of named columns
    fluxes = objs[bandnames].view(objs[bandnames].dtype[0]).reshape(objs[bandnames].shape + (-1,))
    nobs = objs[nobsnames].view(objs[nobsnames].dtype[0]).reshape(objs[nobsnames].shape + (-1,))

    #ADM set any observations with NOBS = 0 to have small flux so glitches don't end up as bright star masks. 
    w = np.where(nobs == 0)
    if len(w[0]) > 0:
        fluxes[w] = 0.

    #ADM limit to the passed faint limit
    w = np.where(np.any(fluxes > fluxlim,axis=1))
    fluxes = fluxes[w]
    objs = objs[w]

    #ADM grab the (GRZ) magnitudes for observations
    #ADM and record only the largest flux (smallest magnitude)
    fluxmax = np.max(fluxes,axis=1)
    mags = 22.5-2.5*np.log10(fluxmax)

    #ADM convert the largest magnitude into radii for "in" and "near" bright objects. This will require 
    #ADM more consideration to determine the truly correct numbers for DESI
    near_radius = (0.0802*mags*mags - 1.860*mags + 11.625)
    in_radius = 0.5*(0.0802*mags*mags - 1.860*mags + 11.625)

    #ADM calculate the TARGETID
    targetid = encode_targetid(objid=objs['OBJID'], brickid=objs['BRICKID'], release=objs['RELEASE'])

    #ADM create an output recarray that is just RA, Dec, TARGETID and the radius
    done = objs[['RA','DEC']].copy()
    done = rfn.append_fields(done,["TARGETID","IN_RADIUS","NEAR_RADIUS"],[targetid,in_radius,near_radius],
                             usemask=False,dtypes=['>i8','<f8','<f8'])

    if outfilename is not None:
        fitsio.write(outfilename, done, clobber=True)

    return done


def plot_mask(mask,limits=None,radius="IN_RADIUS",over=False,show=True):
    """Make a plot of a mask and either display it or retain the plot object for over-plotting

    Parameters
    ----------
    mask : :class:`recarray`
        A mask constructed by make_bright_star_mask (or read in from file in the make_bright_star_mask format)
    limits : :class:`list`, optional
        A list defining the RA/Dec limits of the plot as would be passed to matplotlib.pyplot.axis
    radius : :class: `str`, optional
        Which of the mask radii to plot ("IN_RADIUS" or "NEAR_RADIUS"). Both can be plotted by calling
        this function twice with show=False the first time and over=True the second time                    
    over : :class:`boolean`
        If True, then don't set-up the plot commands. Just issue the command to plot the mask so that the 
        mask will be over-plotted on any existing plot (of targets etc.)
    show : :class:`boolean`
        If True, then display the plot, Otherwise, just execute the plot commands so it can be shown or
        saved to file later              

    Returns
    -------
        Nothing
    """

    #ADM set up the plot
    if not over:
        plt.figure(figsize=(8,8))
        ax = plt.subplot(aspect='equal')
        plt.xlabel('RA (o)')
        plt.ylabel('Dec (o)')

        if limits is not None:
            plt.axis(limits)

    #ADM draw ellipse patches from the mask information converting radius to degrees
    #ADM include the cos(dec) term to expand the RA semi-major axis at higher declination
    #ADM note the "ellipses" takes the diameter, not the radius
    minoraxis = mask[radius]/60.
    majoraxis = minoraxis/np.cos(np.radians(mask["DEC"]))
    out = ellipses(mask["RA"], mask["DEC"], 2*majoraxis, 2*minoraxis, alpha=0.2, edgecolor='none')

    if show:
        plt.show()

    return

def is_in_bright_star(targs,starmask):
    """Determine whether a set of targets is in a bright star mask

    Parameters
    ----------
    targs : :class:`recarray`
        A recarray of targets as made by desitarget.cuts.select_targets
    starmask : :class:`recarray`
        A recarray containing a bright star mask as made by desitarget.brightstar.make_bright_star_mask

    Returns
    -------
    in_mask : array_like. 
        True for array entries that correspond to a target that is IN a bright star mask
    near_mask : array_like. 
        True for array entries that correspond to a target that is NEAR a bright star mask

    """

    #ADM initialize an array of all False (nothing is yet in a star mask)
    in_mask = np.zeros(len(targs), dtype=bool)
    near_mask = np.zeros(len(targs), dtype=bool)

    #ADM turn the coordinates of the masks and the targets into SkyCoord objects
    ctargs = SkyCoord(targs["RA"]*u.degree, targs["DEC"]*u.degree)
    cstars = SkyCoord(starmask["RA"]*u.degree, starmask["DEC"]*u.degree)

    #ADM this is the largest search radius we should need to consider
    #ADM in the future an obvious speed up is to split on radius
    #ADM as large radii are rarer but take longer
    maxrad = max(starmask["NEAR_RADIUS"])*u.arcmin

    #ADM coordinate match the star masks and the targets
    idtargs, idstars, d2d, d3d = cstars.search_around_sky(ctargs,maxrad)

    #ADM catch the case where nothing fell in a mask
    if len(idstars) == 0:
        return in_mask, near_mask

    #ADM for a matching star mask, find the angular separations that are less than the mask radius
    w_in = np.where(d2d.arcmin < starmask[idstars]["IN_RADIUS"])
    w_near = np.where(d2d.arcmin < starmask[idstars]["NEAR_RADIUS"])

    #ADM any matching idtargs that meet this separation criterion are in a mask (at least one)
    in_mask[idtargs[w_in]] = 'True'
    near_mask[idtargs[w_near]] = 'True'

    return in_mask, near_mask

def is_bright_star(targs,starmask):
    """Determine whether any of a set of targets are, themselves, a bright star mask

    Parameters
    ----------
    targs : :class:`recarray`
        A recarray of targets as made by desitarget.cuts.select_targets
    starmask : :class:`recarray`
        A recarray containing a bright star mask as made by desitarget.brightstar.make_bright_star_mask

    Returns
    -------
    is_mask : array_like. 
        True for array entries that correspond to targets that are, themselves, a bright star mask

    """

    #ADM initialize an array of all False (nothing yet has been shown to correspond to a star mask)
    is_mask = np.zeros(len(targs), dtype=bool)

    #ADM calculate the TARGETID for the targets
    targetid = encode_targetid(objid=targs['BRICK_OBJID'], 
                               brickid=targs['BRICKID'], 
                               release=targs['RELEASE'])

    #ADM super-fast set-based look-up of which TARGETIDs are matches between the masks and the targets
    matches = set(starmask["TARGETID"]).intersection(set(targetid))
    #ADM determine the indexes of the targets that have a TARGETID in matches
    w_mask = [ index for index, item in enumerate(targetid) if item in matches ]

    #ADM w_mask now contains the target indices that match to a bright star mask on TARGETID
    is_mask[w_mask] = 'True'

    return is_mask

def generate_safe_locations(starmask,Npersqdeg):
    """Given a bright star mask, generate SAFE (BADSKY) locations at its periphery

    Parameters
    ----------
    starmask : :class:`recarray`
        A recarray containing a bright star mask as made by :mod:`desitarget.brightstar.make_bright_star_mask`
    npersqdeg : :class:`int`
        The number of safe locations to generate per square degree of each mask

    Returns
    -------
    ra : array_like. 
        The Right Ascensions of the SAFE (BADSKY) locations
    dec : array_like. 
        The Declinations of the SAFE (BADSKY) locations

    Notes
    -----
        - See the Tech Note at https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=2346 for more details
    """
    
    #ADM the radius of each mask in degrees with a 0.1% kick to get things beyond the mask edges
    radius = 1.001*starmask["IN_RADIUS"]/60.

    #ADM determine the area of each mask
    area = cap_area(radius)

    #ADM determine the number of SAFE locations to assign to each
    #ADM mask given the passed number of locations per sq. deg.
    Nsafe = np.ceil(area*Npersqdeg).astype('i')

    #ADM determine Nsafe Dec offsets equally spaced around the perimeter for each mask
    offdec = [ rad*np.sin(np.arange(ns)*2*np.pi/ns) for ns, rad in zip(Nsafe,radius) ]

    #ADM use offsets to determine DEC positions
    dec = starmask["DEC"] + offdec

    #ADM determine the offsets in RA at these Decs given the mask center Dec
    offrapos =  [ sphere_circle_ra_off(th,cen,declocs) for th,cen,declocs in zip(radius,starmask["DEC"],dec) ]

    #ADM determine which of the RA offsets are in the positive direction
    sign = [ np.sign(np.cos(np.arange(ns)*2*np.pi/ns)) for ns in Nsafe ]

    #ADM determine the RA offsets with the appropriate sign and add them to the RA of each mask
    offra = [ o*s for o,s in zip(offrapos,sign) ]
    ra = starmask["RA"] + offra

    #ADM have to turn the generated locations into 1-D arrays before returning them
    return np.hstack(ra), np.hstack(dec)


def append_safe_targets(targs,starmask,nside=None,drbricks=None):
    """Append targets at SAFE (BADSKY) locations to target list, set bits in TARGETID and DESI_TARGET

    Parameters
    ----------
    targs : :class:`~numpy.ndarray`
        A recarray of targets as made by desitarget.cuts.select_targets
    nside : :class:`integer`
        The HEALPix nside used throughout the DESI data model
    starmask : :class:`~numpy.ndarray`
        A recarray containing a bright star mask as made by desitarget.brightstar.make_bright_star_mask
    drbricks : :class:`~numpy.ndarray`, optional
        A rec array containing at least the "release", "ra", "dec" and "nobjs" columns from a survey bricks file. 
        This is typically used for testing only.

    Returns
    -------
        The original recarray of targets (targs) is returned with additional SAFE (BADSKY) targets appended to it

    Notes
    -----
        - See the Tech Note at https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=2346 for more details
          on the SAFE (BADSKY) locations
        - See the Tech Note at https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=2348 for more details
          on setting the SKY bit in TARGETID
        - Currently hard-coded to create an additional 10,000 safe locations per sq. deg. of mask. What is the 
          correct number per sq. deg. (Npersqdeg) for DESI is an open question.
        - Perhaps we should move the default nside to a config file, somewhere?
    """

    #ADM Number of safe locations per sq. deg. of each mask in starmask
    Npersqdeg = 10000

    #ADM generate SAFE locations at the periphery of the star masks appropriate to a density of Npersqdeg
    ra, dec = generate_safe_locations(starmask,Npersqdeg)

    #ADM duplicate the targs rec array with a number of rows equal to the generated safe locations
    nrows = len(ra)
    safes = np.zeros(nrows, dtype=targs.dtype)

    #ADM populate the safes recarray with the RA/Dec of the SAFE locations
    safes["RA"] = ra
    safes["DEC"] = dec

    #ADM set the bit for SAFE locations in DESITARGET
    safes["DESI_TARGET"] |= desi_mask.BADSKY

    #ADM add the brick information for the SAFE/BADSKY targets
    b = brick.Bricks(bricksize=0.25)
    safes["BRICKID"] = b.brickid(safes["RA"],safes["DEC"])
    safes["BRICKNAME"] = b.brickname(safes["RA"],safes["DEC"])

    #ADM get the string version of the data release (to find directories for brick information)
    drint = np.max(targs['RELEASE']//1000)
    #ADM check the targets all have the same release
    checker = np.min(targs['RELEASE']//1000)
    if drint != checker:
        raise IOError('Objects from multiple data releases in same input numpy array?!')
    drstring = 'dr'+str(drint)

    #ADM now add the OBJIDs, ensuring they start higher than any other OBJID in the DR
    #ADM read in the Data Release bricks file
    if drbricks is None:
        rootdir = "/project/projectdirs/cosmo/data/legacysurvey/"+drstring.strip()+"/"
        drbricks = fitsio.read(rootdir+"survey-bricks-"+drstring.strip()+".fits.gz")
    #ADM the BRICK IDs that are populated for this DR
    drbrickids = b.brickid(drbricks["ra"],drbricks["dec"])
    #ADM the maximum possible BRICKID at bricksize=0.25
    brickmax = 662174
    #ADM create a histogram of how many SAFE/BADSKY objects are in each brick
    hsafes = np.histogram(safes["BRICKID"],range=[0,brickmax+1],bins=brickmax+1)[0]
    #ADM create a histogram of how many objects are in each brick in this DR
    hnobjs = np.zeros(len(hsafes),dtype=int)
    hnobjs[drbrickids] = drbricks["nobjs"]
    #ADM make each OBJID for a SAFE/BADSKY +1 higher than any other OBJID in the DR
    safes["BRICK_OBJID"] = hnobjs[safes["BRICKID"]] + 1
    #ADM sort the safes array on BRICKID
    safes = safes[safes["BRICKID"].argsort()]
    #ADM remove zero entries from the histogram of BRICKIDs in safes, for speed
    hsafes = hsafes[np.where(hsafes > 0)]
    #ADM the count by which to augment each OBJID to make unique OBJIDs for safes
    objsadd = np.hstack([ np.arange(i) for i in hsafes ])
    #ADM finalize the OBJID for each SAFE target
    safes["BRICK_OBJID"] += objsadd

    #ADM finally, update the TARGETID with the OBJID, the BRICKID, and the fact these are skies
    safes["TARGETID"] = encode_targetid(objid=safes['BRICK_OBJID'], 
                                        brickid=safes['BRICKID'],
                                        sky=1)
        
    #ADM return the input targs with the SAFE targets appended
    return np.hstack([targs,safes])


def set_target_bits(targs,starmask):
    """Apply bright star mask to targets, return desi_target array

    Parameters
    ----------
    targs : :class:`recarray`
        A recarray of targets as made by desitarget.cuts.select_targets
    starmask : :class:`recarray`
        A recarray containing a bright star mask as made by desitarget.brightstar.make_bright_star_mask

    Returns
    -------
        an ndarray of the updated desi_target bit that includes bright star information

    Notes
    -----
        - Sets IN_BRIGHT_OBJECT and NEAR_BRIGHT_OBJECT via coordinate matches to the mask centers and radii
        - Sets BRIGHT_OBJECT via an index match on TARGETID (defined as in :mod:`desitarget.targets.encode_targetid`)

    See :mod:`desitarget.targetmask` for the definition of each bit
    """

    bright_object = is_bright_star(targs,starmask)
    in_bright_object, near_bright_object = is_in_bright_star(targs,starmask)

    desi_target = targs["DESI_TARGET"].copy()

    desi_target |= bright_object * desi_mask.BRIGHT_OBJECT
    desi_target |= in_bright_object * desi_mask.IN_BRIGHT_OBJECT
    desi_target |= near_bright_object * desi_mask.NEAR_BRIGHT_OBJECT

    return desi_target


def mask_targets(targs,instarmaskfile=None,nside=None,bands="GRZ",maglim=[10,10,10],numproc=4,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1',outfilename=None,verbose=False,drbricks=None):
    """Add bits for whether objects are in a bright star mask, and SAFE (BADSKY) sky locations, to a list of targets

    Parameters
    ----------
    targs : :class:`str` or `~numpy.ndarray`
        A recarray of targets created by desitarget.cuts.select_targets OR a filename of
        a file that contains such a set of targets
    instarmaskfile : :class:`str`, optional
        An input bright star mask created by desitarget.brightstar.make_bright_star_mask
        If None, defaults to making the bright star mask from scratch
        The next 5 parameters are only relevant to making the bright star mask from scratch
    nside : :class:`integer`
        The HEALPix nside used throughout the DESI data model
    bands : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z".
        Can pass multiple bands as string, e.g. "GRZ", in which case maglim has to be a
        list of the same length as the string
    maglim : :class:`float`
        The upper limit in that magnitude band for which to assemble a list of bright stars.
        Can pass a list of magnitude limits, in which case bands has to be a string of the
        same length (e.g., "GRZ" for [12.3,12.7,12.6]
    numproc : :class:`int`, optional
        Number of processes over which to parallelize
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory containing either sweeps or tractor files...e.g. for dr3 this might be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3/sweeps/dr3.1
    outfilename : :class:`str`, optional, defaults to not writing anything to file
        (FITS) File name to which to write the output bright star mask ONE OF outfilename or
        instarmaskfile MUST BE PASSED
    verbose : :class:`bool`, optional
        Send to write progress to screen
    drbricks : :class:`~numpy.ndarray`, optional
        A rec array containing at least the "release", "ra", "dec" and "nobjs" columns from a survey bricks file
        This is typically used for testing only.

    Returns
    -------
    :class:`~numpy.ndarray`
        the input targets with the DESI_TARGET column updated to reflect the BRIGHT_OBJECT bits
        and SAFE (BADSKY) sky locations added around the perimeter of the bright star mask.

    Notes
    -----
        - See the Tech Note at https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=2346 for more details
          about SAFE (BADSKY) locations
        - Runs in about 10 minutes for 20M targets and 50k masks (roughly maglim=10)
        - (not including 5-10 minutes to build the star mask from scratch)
    """

    t0 = time()

    if instarmaskfile is None and outfilename is None:
        raise IOError('One of instarmaskfile or outfilename must be passed')

    #ADM Check if targs is a filename or the structure itself
    if isinstance(targs, str):
        if not os.path.exists(targs):
            raise ValueError("{} doesn't exist".format(targs))
        targs = fitsio.read(targs)

    #ADM check if a file for the bright star mask was passed, if not then create it
    if instarmaskfile is None:
        starmask = make_bright_star_mask(bands,maglim,numproc=numproc,
                                         rootdirname=rootdirname,outfilename=outfilename,verbose=verbose)
    else:
        starmask = fitsio.read(instarmaskfile)

    if verbose:
        ntargsin = len(targs)
        print('Number of targets {}...t={:.1f}s'.format(ntargsin, time()-t0))
        print('Number of star masks {}...t={:.1f}s'.format(len(starmask), time()-t0))

    #ADM generate SAFE locations and add them to the target list
    targs = append_safe_targets(targs,starmask,nside=nside,drbricks=drbricks)
    
    if verbose:
        print('Generated {} SAFE (BADSKY) locations...t={:.1f}s'.format(len(targs)-ntargsin, time()-t0))

    #ADM update the bits depending on whether targets are in a mask
    dt = set_target_bits(targs,starmask)
    done = targs.copy()
    done["DESI_TARGET"] = dt

    #ADM remove any SAFE locations that are in bright masks (because they aren't really safe)
    w = np.where(  ((done["DESI_TARGET"] & desi_mask.BADSKY) == 0)  | 
                   ((done["DESI_TARGET"] & desi_mask.IN_BRIGHT_OBJECT) == 0)  )
    if len(w[0]) > 0:
        done = done[w]

    if verbose:
        print("...of these, {} SAFE (BADSKY) locations aren't in masks...t={:.1f}s".format(len(done)-ntargsin, time()-t0))

    if verbose:
        print('Finishing up...t={:.1f}s'.format(time()-t0))

    return done
 
