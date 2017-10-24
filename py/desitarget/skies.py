# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==================
desitarget.skies
==================

Module dealing with the assignation of sky fibers for target selection
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import fitsio
from time import time

from astropy.coordinates import SkyCoord
from astropy import units as u

from desiutil import brick
from desitarget import io, desi_mask, targetid_mask
from desitarget.targets import encode_targetid, finalize
from desitarget.internal import sharedmem


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
    
    #ADM derive the eccentricity from the ellipticity
    #ADM guarding against the option that floats were passed
    e = np.atleast_1d(np.hypot(e1, e2))

    #ADM the position angle in radians and its cos/sin
    theta = np.atleast_1d(np.arctan2(e2, e1) / 2.)
    ct = np.cos(theta)
    st = np.sin(theta)

    #ADM ensure there's a maximum ratio of the semi-major
    #ADM to semi-minor axis, and calculate that ratio
    maxab = 1000.
    ab = np.zeros(len(e))+maxab
    w = np.where(e < 1)
    ab[w] = (1.+e[w])/(1.-e[w])
    w = np.where(ab > maxab)
    ab[w] = maxab

    #ADM convert the half-light radius to degrees
    r_deg = r / 3600.

    #ADM the 2x2 matrix to transform points measured in 
    #ADM effective-half-light-radius to RA/Dec offsets
    T = r_deg * np.array([[ct / ab, st], [-st / ab, ct]])
    
    return T


def ellipse_boundary(RAcen, DECcen, r, e1, e2):
    """Return RA/Dec of an elliptical boundary on the sky

    Parameters
    ----------
    RAcen : :class:`float` or `~numpy.ndarray` 
        Right Ascension of the center of the ellipse (DEGREES)
    DECcen : :class:`float` or `~numpy.ndarray` 
        Declination of the center of the ellipse (DEGREES)
    r : :class:`float` or `~numpy.ndarray` 
        Half-light radius of the ellipse (ARCSECONDS)
    e1 : :class:`float` or `~numpy.ndarray` 
        First ellipticity component of the ellipse
    e2 : :class:`float` or `~numpy.ndarray` 
        Second ellipticity component of the ellipse

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
    
    #ADM Retrieve the 2x2 matrix to transform points measured in 
    #ADM effective-half-light-radius to RA/Dec offsets
    G = ellipse_matrix(r, e1, e2)

    #ADM create a circle in effective-half-light-radius
    angle = np.linspace(0, 2.*np.pi, 50)
    vv = np.vstack([np.sin(angle),np.cos(angle)])

    #ADM transform circle to elliptical boundary
    dra = []
    ddec = []
    for i in range(np.shape(T)[-1]):
        dradec = np.dot(T[...,i],vv)
        dra.append(dradec[0])
        ddec.append(dradec[0])
        ras = RAcen[i] + dra
        decs = DECcen[i] + ddec 
    
    #ADM return the RA, Dec of the boundary or boundaries
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
    
    #ADM Retrieve the 2x2 matrix to transform points measured in 
    #ADM effective-half-light-radius to RA/Dec offsets...
    G = ellipse_matrix(r, e1, e2)
    #ADM ...and invert it
    Ginv = np.linalg.inv(G[...,0])

    dra = ras - RAcen
    ddec = decs - DECcen

    #ADM test whether points are larger than the effective
    #ADM circle of radius 1 generated in half-light-radius coordinates
    dx, dy = np.dot(Ginv,[dra,ddec])

    return np.hypot(dx,dy) < 1


def density_of_sky_fibers(margin=2.):
    """Use desihub products to find required density of sky fibers for DESI

    Parameters
    ----------
    margin : :class:`float`, optional, defaults to 2.
        Factor of extra sky positions to generate. So, for margin=2, twice as
        many sky positions as the default requirements will be generated

    Returns
    -------
    :class:`float`
        The density of sky fibers to generate
    """

    from desimodel.io import load_fiberpos, load_target_info
    fracsky = load_target_info()["frac_sky"]
    nfibers = len(load_fiberpos())
    nskymin = margin*fracsky*nfibers

    return nskymin


def calculate_separations(objs,navoid=2.):
    """Generate an array of separations (in arcseconds) for a set of objects

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        numpy structured array with UPPERCASE columns, OR a string 
        tractor/sweep filename. Must contain at least the columns
        "RA", "DEC", "SHAPEDEV_R", "SHAPEEXP_R"
    navoid : :class:`float`, optional, defaults to 2.
        the number of times the galaxy half-light radius (or seeing) to avoid
        objects out to when placing sky fibers

    Returns
    -------
    :class:`float`
        an array of maximum separations (in arcseconds) based on 
        de Vaucouleurs, Exponential or point-source half-light radii
    """

    #ADM check if input objs is a filename or the actual data
    if isinstance(objs, str):
        objs = io.read_tractor(objs)
    nobjs = len(objs)

    #ADM possible choices for separation based on de Vaucouleurs and Exponential profiles
    #ADM or a minimum of 2 arcseconds for point sources ("the seeing")
    sepchoices = np.vstack([objs["SHAPEDEV_R"], objs["SHAPEEXP_R"], np.ones(nobjs)*2]).T

    #ADM the maximum separation from de Vaucoulers/exponential/PSF choices
    sep = navoid*np.max(sepchoices,axis=1)

    return sep


def generate_sky_positions(objs,navoid=2.,nskymin=None):
    """Use a basic avoidance-of-other-objects approach to generate sky positions

    Parameters
    ----------
    objs : :class:`~numpy.ndarray` 
        numpy structured array with UPPERCASE columns, OR a string 
        tractor/sweep filename. Must contain at least the columns
        "RA", "DEC", "SHAPEDEV_R", "SHAPEEXP_R"
    navoid : :class:`float`, optional, defaults to 2.
        the number of times the galaxy half-light radius (or seeing) to avoid
        objects out to when placing sky fibers
    nskymin : :class:`float`, optional, defaults to reading from desimodel.io
        the minimum DENSITY of sky fibers to generate

    Returns
    -------
    ragood : :class:`~numpy.array`
        array of RA coordinates for good sky positions
    decgood : :class:`~numpy.array`
        array of Dec coordinates for good sky positions
    rabad : :class:`~numpy.array`
        array of RA coordinates for bad sky positions, i.e. positions that
        ARE within navoid half-light radii of a galaxy (or navoid*2 arcseconds
        for a PSF object)
    decbad : :class:`~numpy.array`
        array of Dec coordinates for bad sky positions, i.e. positions that
        ARE within navoid half-light radii of a galaxy (or navoid*2 arcseconds
        for a PSF object)
    """
    #ADM set up the default log
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()

    log.info('Generating sky positions...t = {:.1f}s'.format(time()-start))

    #ADM check if input objs is a filename or the actual data
    if isinstance(objs, str):
        objs = io.read_tractor(objs)
    nobjs = len(objs)

    #ADM an avoidance separation (in arcseconds) for each
    #ADM object based on its half-light radius/profile
    log.info('Calculating avoidance zones...t = {:.1f}s'.format(time()-start))
    sep = calculate_separations(objs,navoid)
    #ADM the maximum such separation for any object in the passed set in arcsec
    maxrad = max(sep)

    #ADM if needed, determine the minimum density of sky fibers to generate
    if nskymin is None:
        nskymin = density_of_sky_fibers()

    #ADM the coordinate limits and corresponding area of the passed objs
    ramin, ramax = np.min(objs["RA"]), np.max(objs["RA"])
    #ADM guard against the wraparound bug (should never be an issue for the sweeps, anyway)
    if ramax - ramin > 180.:
        ramax -= 360.
    decmin, decmax = np.min(objs["DEC"]),np.max(objs["DEC"])
    sindecmin, sindecmax = np.sin(np.radians(decmin)), np.sin(np.radians(decmax))
    spharea = (ramax-ramin)*np.degrees(sindecmax-sindecmin)

    #ADM how many sky positions we need to generate to meet the minimum density requirements
    nskies = int(spharea*nskymin)
    #ADM how many sky positions to generate, given that we'll reject objects close to bad
    #ADM sources. The factor of 10 was derived by trial-and-error...but this doesn't need
    #ADM to be optimal as this algorithm should become more sophisticated
    nchunk = nskies*10

    #ADM arrays of GOOD sky positions to populate with coordinates
    ragood, decgood = np.empty(nskies), np.empty(nskies)
    #ADM lists of BAD sky positions to populate with coordinates
    rabad, decbad = [], []

    #ADM ngenerate will become zero when we generate enough GOOD sky positions
    ngenerate = nskies

    while ngenerate:

        #ADM generate random points in RA and Dec (correctly distributed on the sphere)
        log.info('Generated {} test positions...t = {:.1f}s'.format(nchunk,time()-start))
        ra = np.random.uniform(ramin,ramax,nchunk)
        dec = np.degrees(np.arcsin(1.-np.random.uniform(1-sindecmax,1-sindecmin,nchunk)))

        #ADM set up the coordinate objects
        cskies = SkyCoord(ra*u.degree, dec*u.degree)
        cobjs = SkyCoord(objs["RA"]*u.degree, objs["DEC"]*u.degree)

        #ADM split the objects up using a separation of 4*navoid arcseconds in order to
        #ADM speed up the coordinate matching when we have some objects with large radii
        sepsplit = 4*navoid
        bigsepw = np.where(sep > sepsplit)[0]
        smallsepw = np.where(sep <= sepsplit)[0]

        #ADM set up a list of skies that don't match an object
        goodskies = np.ones(len(cskies),dtype=bool)

        #ADM guard against the case where there are no objects with small radii
        if len(smallsepw) > 0:
            #ADM match the small-separation objects and flag any skies that match such an object
            log.info('Matching positions out to {:.1f} arcsec...t = {:.1f}s'
                     .format(sepsplit,time()-start))
            idskies, idobjs, d2d, _ = cobjs[smallsepw].search_around_sky(cskies,sepsplit*u.arcsec)
            w = np.where(d2d.arcsec < sep[smallsepw[idobjs]])
            #ADM remember to guard against the case with no bad positions
            if len(w[0]) > 0:
                goodskies[idskies[w]] = False

        #ADM guard against the case where there are no objects with large radii
        if len(bigsepw) > 0:
            #ADM match the large-separation objects and flag any skies that match such an object
            log.info('Matching additional positions out to {:.1f} arcsec...t = {:.1f}s'
                     .format(maxrad,time()-start))
            idskies, idobjs, d2d, _ = cobjs[bigsepw].search_around_sky(cskies,maxrad*u.arcsec)
            w = np.where(d2d.arcsec < sep[bigsepw[idobjs]])
            #ADM remember to guard against the case with no bad positions
            if len(w[0]) > 0:
                goodskies[idskies[w]] = False

        #ADM good sky positions we found
        wgood = np.where(goodskies)[0]
        n1 = nskies - ngenerate
        ngenerate = max(0, ngenerate - len(wgood))
        n2 = nskies - ngenerate
        ragood[n1:n2] = ra[wgood[:n2 - n1]]
        decgood[n1:n2] = dec[wgood[:n2 - n1]]
        log.info('Need to generate a further {} positions...t = {:.1f}s'.format(ngenerate,time()-start))

        #ADM bad sky positions we found
        wbad = np.where(~goodskies)[0]
        rabad.append(list(ra[wbad]))
        decbad.append(list(dec[wbad]))

    #ADM we potentially created nested lists for the bad skies, so need to flatten
    #ADM also we can't need more bad sky positions than total sky positions
    rabad = np.array([item for sublist in rabad for item in sublist])[:nskies]
    decbad = np.array([item for sublist in decbad for item in sublist])[:nskies]

    log.info('Done...t = {:.1f}s'.format(time()-start))

    return ragood, decgood, rabad, decbad


def plot_sky_positions(ragood,decgood,rabad,decbad,objs,navoid=2.,limits=None,plotname=None):
    """plot an example set of sky positions to check if they avoid real objects

    Parameters
    ----------
    ragood : :class:`~numpy.array`
        array of RA coordinates for good sky positions
    decgood : :class:`~numpy.array`
        array of Dec coordinates for good sky positions
    rabad : :class:`~numpy.array`
        array of RA coordinates for bad sky positions, i.e. positions that
        ARE within the avoidance zones of the "objs"
    decbad : :class:`~numpy.array`
        array of Dec coordinates for bad sky positions, i.e. positions that
        ARE within the avoidance zones of the "objs"
    objs : :class:`~numpy.ndarray` 
        numpy structured array with UPPERCASE columns, OR a string 
        tractor/sweep filename. Must contain at least the columns
        "RA", "DEC", "SHAPEDEV_R", "SHAPEEXP_R"
    navoid : :class:`float`, optional, defaults to 2.
        the number of times the galaxy half-light radius (or seeing) that
        objects (objs) were avoided out to when generating sky positions
    limits : :class:`~numpy.array`, optional, defaults to None
        plot limits in the form [ramin, ramax, decmin, decmax] if None
        is passed, then a small subsection of the passed area is plotted
    plotname : :class:`str`, defaults to None    
        If a name is passed use matplotlib's savefig command to save the
        plot to that file name. Otherwise, display the plot

    Returns
    -------
    Nothing
    """

    import matplotlib.pyplot as plt
    from desitarget.brightstar import ellipses

    #ADM initialize the default logger
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()

    #ADM set up the figure and the axis labels
    plt.figure(figsize=(8,8))
    plt.xlabel('RA (o)')
    plt.ylabel('Dec (o)')

    #ADM check if input objs is a filename or the actual data
    if isinstance(objs, str):
        objs = io.read_tractor(objs)

    #ADM coordinate limits and corresponding area of the passed objs
    ramin, ramax = np.min(objs["RA"]), np.max(objs["RA"])
    decmin, decmax = np.min(objs["DEC"]), np.max(objs["DEC"])
    #ADM guard against wraparound bug (which should never be an issue for the sweeps, anyway)
    if ramax - ramin > 180.:
        ramax -= 360.

    #ADM the avoidance separation (in arcseconds) for each object based on 
    #ADM its half-light radius/profile
    sep = calculate_separations(objs,navoid)
    #ADM the maximum such separation for any object in the passed set IN DEGREES
    maxrad = max(sep)/3600.

    #ADM limit the figure range based on the passed objs
    if limits is None:
        rarange, decrange = ramax - ramin, decmax - decmin
        rastep, decstep = rarange*0.47, decrange*0.47
        ralo, rahi = ramin+rastep, ramax-rastep
        declo, dechi = decmin+decstep, decmax-decstep
    else:
        ralo, rahi, declo, dechi = limits

    plt.axis([ralo,rahi,declo,dechi])

    #ADM plot good and bad sky positions
    plt.scatter(ragood,decgood,marker='d',facecolors='none',edgecolors='k')
    plt.scatter(rabad,decbad,marker='s',facecolors='none',edgecolors='r')

    #ADM restrict the passed avoidance zones based on the passed limits
    #ADM remembering that we need to plot things at least the maximum/cos(maxdec)
    #ADM times the possible avoidance zone beyond the plot limits
    fac = 1./np.cos(np.radians(max(abs(decmin),decmax)))
    w = np.where( (objs["RA"] > ralo-fac*maxrad) & (objs["RA"] < rahi+fac*maxrad) & 
                  (objs["DEC"] > declo-fac*maxrad) & (objs["DEC"] < dechi+fac*maxrad))
    
    log.info('Number of avoidance zones in plot area {}...t = {:.1f}s'.format(len(w[0]),time()-start))

    #ADM set up the ellipse shapes based on sizes of the past avoidance zones
    #ADM remembering to stretch by the COS term to de-project the sky
    minoraxis = sep/3600.
    majoraxis = minoraxis/np.cos(np.radians(objs["DEC"]))

    log.info('Plotting avoidance zones...t = {:.1f}s'.format(time()-start))
    #ADM plot the avoidance zones as ellipses
    out = ellipses(objs[w]["RA"], objs[w]["DEC"], 2*majoraxis[w], 2*minoraxis[w], alpha=0.2, edgecolor='none')

    #ADM display the plot, if requested
    if plotname is None:
        log.info('Displaying plot...t = {:.1f}s'.format(time()-start))
        plt.show()
    else:
        plt.savefig(plotname)    

    log.info('Done...t = {:.1f}s'.format(time()-start))

    return


def make_sky_targets(objs,navoid=2.,nskymin=None):
    """Generate sky targets and translate them into the typical format for DESI targets

    Parameters
    ----------
    objs : :class:`~numpy.ndarray` 
        numpy structured array with UPPERCASE columns, OR a string 
        tractor/sweep filename. Must contain at least the columns
        "RA", "DEC", "SHAPEDEV_R", "SHAPEEXP_R"
    navoid : :class:`float`, optional, defaults to 2.
        the number of times the galaxy half-light radius (or seeing) to avoid
        objects out to when placing sky fibers
    nskymin : :class:`float`, optional, defaults to reading from desimodel.io
        the minimum DENSITY of sky fibers to generate

    Returns
    -------
    :class:`~numpy.ndarray`
        a structured array of good and bad sky positions in the DESI target format
    """

    #ADM initialize the default logger
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()

    #ADM check if input objs is a filename or the actual data
    if isinstance(objs, str):
        objs = io.read_tractor(objs)

    log.info('Generating sky positions...t = {:.1f}s'.format(time()-start))
    #ADM generate arrays of good and bad objects for this sweeps file (or set)
    ragood, decgood, rabad, decbad = generate_sky_positions(objs,navoid=navoid,nskymin=nskymin)
    ngood = len(ragood)
    nbad = len(rabad)
    nskies = ngood + nbad

    #ADM retrieve the standard DESI target array
    dt = io.tsdatamodel.dtype
    skies = np.zeros(nskies, dtype=dt)

    #ADM populate the output recarray with the RA/Dec of the good and bad sky locations
    skies["RA"][0:ngood], skies["DEC"][0:ngood] = ragood, decgood
    skies["RA"][ngood:nskies], skies["DEC"][ngood:nskies] = rabad, decbad

    #ADM create an array of target bits with the SKY information set
    desi_target = np.zeros(nskies,dtype='>i8')
    desi_target[0:ngood] |= desi_mask.SKY
    desi_target[ngood:nskies] |= desi_mask.BADSKY

    log.info('Looking up brick information...t = {:.1f}s'.format(time()-start))
    #ADM add the brick information for the sky targets
    b = brick.Bricks(bricksize=0.25)
    skies["BRICKID"] = b.brickid(skies["RA"],skies["DEC"])
    skies["BRICKNAME"] = b.brickname(skies["RA"],skies["DEC"])

    #ADM set the data release from the passed sweeps objects
    dr = np.max(objs['RELEASE'])
    #ADM check the passed sweeps objects have the same release
    checker = np.min(objs['RELEASE'])
    if dr != checker:
        raise IOError('Multiple data releases present in same input sweeps objects?!')
    skies["RELEASE"] = dr

    #ADM set the objid (just use a sequential number as setting skies
    #ADM to 1 in the TARGETID will make these unique
    #ADM *MAKE SURE TO SET THE BRIGHT STAR SAFE LOCATIONS OF THE MAXIMUM SKY OBJID*!!!
    skies["OBJID"] = np.arange(nskies)

    log.info('Finalizing target bits...t = {:.1f}s'.format(time()-start))
    #ADM add target bit columns to the output array, note that mws_target
    #ADM and bgs_target should be zeros for all sky objects
    dum = np.zeros_like(desi_target)
    skies = finalize(skies, desi_target, dum, dum, sky=1)

    log.info('Done...t = {:.1f}s'.format(time()-start))

    return skies


def select_skies(infiles, numproc=4):
    """Process input files in parallel to select blank sky positions

    Parameters
    ----------
    infiles : :class:`list` or `str`
        list of input filenames (tractor or sweep files),
            OR a single filename
    numproc : :clsss:`int`, optional, defaults to 4 
        number of parallel processes to use. Pass numproc=1 to run in serial

    Returns
    -------
    :class:`~numpy.ndarray`
        a structured array of good and bad sky positions in the DESI target format

    Notes
    -----
    Much of this is taken verbatim from `desitarget.cuts`

    """

    #ADM set up the default log
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    #ADM if a single file was passed, convert it to a list of files
    if isinstance(infiles,str):
        infiles = [infiles,]

    #ADM check that the files to be processed actually exist
    import os
    for filename in infiles:
        if not os.path.exists(filename):
            log.fatal('File {} not found!'.format(filename))


    #ADM function to run file-by-file for each sweeps file
    def _select_skies_file(filename):
        '''Returns targets in filename that pass the cuts'''
        return make_sky_targets(filename,navoid=2.,nskymin=None)

    #ADM to count the number of files that have been processed
    #ADM use of a numpy scalar is for backwards-compatability with Python 2
    #c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    start = time()
    
    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
        that occurs on the main parallel process'''
        if nbrick%50 == 0 and nbrick>0:
            rate = nbrick / (time() - start)
            log.info('{} files; {:.1f} files/sec'.format(nbrick, rate))

        nbrick[...] += 1    # this is an in-place modification
        return result
    
    if numproc > 1:
        #ADM process the input files in parallel
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            skies = pool.map(_select_skies_file, infiles, reduce=_update_status)
    else:
        #ADM process the input files in serial
        skies = list()
        for file in infiles:
            skies.append(_update_status(_select_skies_file(file)))

    skies = np.concatenate(skies)

    return skies


