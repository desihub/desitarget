# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
===================
desitarget.oldskies
===================

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
from desitarget import io
from desitarget.targetmask import desi_mask, targetid_mask
from desitarget.targets import encode_targetid, finalize
from desitarget.internal import sharedmem
from desitarget.geomask import ellipse_matrix, ellipse_boundary, is_in_ellipse_matrix
from desitarget.brightmask import is_in_bright_mask
from desitarget.cuts import _psflike

from collections import defaultdict

#ADM the default PSF SIZE to adopt, i.e., seeing will be
#ADM NO WORSE than this for the DESI survey at the Mayall
#ADM this can typically be scaled using the navoid parameter
psfsize = 2.

def find_duplicates_and_indexes(targetid):
    """For a list of integers, return any duplicates and indexes of duplicates

    Parameters
    ----------
    targetid : :class:`list` or `~numpy.ndarray`
        A list or array of integers

    Returns
    -------
    :class:`dictionary`
        A dictionary of the duplicates where the keys are the duplicated
        targetids and the values are the indexes of the duplicates

    Notes
    -----
    h/t to https://stackoverflow.com/questions/11236006/identify-duplicate-values-in-a-list-in-python#11236042
    """

    dd = defaultdict(list)
    for index, item in enumerate(targetid):
        dd[item].append(index)
    dd = { key:value for key,value in dd.items() if len(value)>1 }

    return dd

def density_of_sky_fibers(margin=1.5):
    """Use positioner patrol size to determine sky fiber density for DESI

    Parameters
    ----------
    margin : :class:`float`, optional, defaults to 1.5
        Factor of extra sky positions to generate. So, for margin=10, 10x as
        many sky positions as the default requirements will be generated

    Returns
    -------
    :class:`float`
        The density of sky fibers to generate in per sq. deg.
    """

    #ADM the patrol radius of a DESI positioner (in sq. deg.)
    patrol_radius = 6.4/60./60.

    #ADM hardcode the number of options per positioner
    options = 2.

    nskymin = margin*options/patrol_radius

    return nskymin


def model_density_of_sky_fibers(margin=1.5):
    """Use desihub products to find required density of sky fibers for DESI

    Parameters
    ----------
    margin : :class:`float`, optional, defaults to 1.5
        Factor of extra sky positions to generate. So, for margin=10, 10x as
        many sky positions as the default requirements will be generated

    Returns
    -------
    :class:`float`
        The density of sky fibers to generate in per sq. deg.
    """

    from desimodel.io import load_fiberpos, load_target_info
    fracsky = load_target_info()["frac_sky"]
    nfibers = len(load_fiberpos())
    nskymin = margin*fracsky*nfibers

    return nskymin


def calculate_separations(objs,navoid=1.):
    """Generate an array of separations (in arcseconds) for a set of objects

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        numpy structured array with UPPERCASE columns, OR a string
        tractor/sweep filename. Must contain at least the columns
        "RA", "DEC", "SHAPEDEV_R", "SHAPEEXP_R"
    navoid : :class:`float`, optional, defaults to 1.
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
    #ADM or a minimum of psfsize arcseconds for point sources ("the seeing")
    #ADM the default psfsize is supplied at the top of this code
    sepchoices = np.vstack([objs["SHAPEDEV_R"], objs["SHAPEEXP_R"], np.ones(nobjs)*psfsize]).T

    #ADM the maximum separation from de Vaucoulers/exponential/PSF choices
    sep = navoid*np.max(sepchoices,axis=1)

    return sep


def format_as_mask(objs,navoid=1.):
    """format a set of objects as a mask to use with desitarget.brightmask

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        numpy structured array with UPPERCASE columns, OR a string
        tractor/sweep filename. Must contain at least the columns
        "RA", "DEC", "SHAPEDEV_R", "SHAPEDEV_E1", "SHAPEDEV_E2"
        "SHAPEEXP_R", "SHAPEEXP_E1", "SHAPEDEV_E2", "TYPE"

    navoid : :class:`float`, optional, defaults to 1.
        the number of times the galaxy half-light radius (or seeing) to avoid
        objects out to when placing sky fibers

    Returns
    -------
    :class:`recarray`
        - The bright source mask in the form `RA`,`DEC`,`TARGETID`,`IN_RADIUS`,`E1`,`E2`,`TYPE`
        - TARGETID is as calculated in :mod:`desitarget.targets.encode_targetid`
        - The radius is in ARCSECONDS (and defaults to the equivalent of half-light radius for ellipses)
        - `E1` and `E2` are ellipticity components, which are 0 for unresolved objects
        - `TYPE` is always `PSF` for star-like objects. This is taken from the sweeps files, see, e.g.:
          http://legacysurvey.org/dr5/files/#sweep-catalogs                                                     
    """

    #ADM Create an output recarray that's just RA, Dec, TARGETID and the radii
    nrows = len(objs)
    dnames = ['RA','DEC','TARGETID','IN_RADIUS','E1','E2','TYPE']
    dtypes = ['>f8','>f8','>i8','>f4','>f4','>f4','|S4']
    dt = list(zip(dnames,dtypes))
    done = np.empty(nrows, dtype=dt)

    #ADM each mask's coordinates and type
    done['RA'] = objs['RA']
    done['DEC'] = objs['DEC']
    done['TYPE'] = objs['TYPE']

    #ADM calculate the TARGETID
    targetid = encode_targetid(objid=objs['OBJID'], brickid=objs['BRICKID'], release=objs['RELEASE'])
    done['TARGETID'] = targetid

    #ADM first set the shape parameters assuming everything is an exponential
    #ADM this will correctly assign e1, e2 of 0 to things with zero shape
    in_radius = objs['SHAPEEXP_R']*navoid
    e1 = objs['SHAPEEXP_E1']
    e2 = objs['SHAPEEXP_E2']

    #ADM now to account for deVaucouleurs objects, or things that are dominated by
    #ADM deVaucouleurs profiles, update objects with a larger "DEV" than "EXP" radius
    wdev = np.where(objs['SHAPEDEV_R'] > objs['SHAPEEXP_R'])
    if len(wdev[0]) > 0:
        in_radius[wdev] = objs[wdev]['SHAPEDEV_R']*navoid
        e1[wdev] = objs[wdev]['SHAPEDEV_E1']
        e2[wdev] = objs[wdev]['SHAPEDEV_E2']

    #ADM set anything that is type PSF to navoid times the psfsize
    wpsf = np.where(_psflike(objs['TYPE']))
    in_radius[wpsf] = psfsize*navoid

    #ADM set the remaining columns
    done['IN_RADIUS'] = in_radius
    done['E1'] = e1
    done['E2'] = e2

    return done


def generate_sky_positions(objs,navoid=1.,nskymin=None,maglim=[20,20,20]):
    """Use a basic avoidance-of-other-objects approach to generate sky positions

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        numpy structured array with UPPERCASE columns, OR a string
        tractor/sweep filename. Must contain at least the columns
        "RA", "DEC", "SHAPEDEV_R", "SHAPEDEV_E1", "SHAPEDEV_E2"
        "SHAPEEXP_R", "SHAPEEXP_E1", "SHAPEDEV_E2", "TYPE"
    navoid : :class:`float`, optional, defaults to 1.
        the number of times the galaxy half-light radius (or seeing) to avoid
        objects out to when placing sky fibers
    nskymin : :class:`float`, optional, defaults to reading from desimodel.io
        the minimum DENSITY of sky fibers to generate
    maglim : :class:`list`, optional, defaults to [20,20,20]
        The "upper limit" in the three optical DESI selection bands. Objects 
        fainter than these limits in ALL of g, r, z will NOT be used as masks.

    Returns
    -------
    ragood : :class:`~numpy.array`
        array of RA coordinates for good sky positions
    decgood : :class:`~numpy.array`
        array of Dec coordinates for good sky positions
    rabad : :class:`~numpy.array`
        array of RA coordinates for bad sky positions, i.e. positions that
        ARE within navoid half-light radii of a galaxy (or navoid*psfsize
        arcseconds for a PSF object)
    decbad : :class:`~numpy.array`
        array of Dec coordinates for bad sky positions, i.e. positions that
        ARE within navoid half-light radii of a galaxy (or navoid*psfsize
        arcseconds for a PSF object)
    """
    #ADM set up the default log
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()

    #ADM if needed, determine the minimum density of sky fibers to generate
    if nskymin is None:
        nskymin = density_of_sky_fibers()

    log.info('Generating sky positions at a density of {} per sq. deg....t = {:.1f}s'
             .format(nskymin,time()-start))

    #ADM check if input objs is a filename or the actual data
    if isinstance(objs, str):
        objs = io.read_tractor(objs)

    #ADM the coordinate limits and corresponding area of the passed objs
    ramin, ramax = np.min(objs["RA"]), np.max(objs["RA"])
    #ADM guard against the wraparound bug (should never be an issue for the sweeps, anyway)
    if ramax - ramin > 180.:
        ramax -= 360.
    decmin, decmax = np.min(objs["DEC"]),np.max(objs["DEC"])
    sindecmin, sindecmax = np.sin(np.radians(decmin)), np.sin(np.radians(decmax))
    spharea = (ramax-ramin)*np.degrees(sindecmax-sindecmin)
    log.info('Area covered by passed objects is {:.3f} sq. deg....t = {:.1f}s'
             .format(spharea,time()-start))

    #ADM change input magnitude limits to a flux to test against
    fluxlim = 10.**((22.5-np.array(maglim))/2.5)
    #ADM retrieve only objects at requested flux limits
    w = np.where(  (objs["FLUX_G"] > fluxlim[0]) 
                 | (objs["FLUX_R"] > fluxlim[1])  
                 | (objs["FLUX_Z"] > fluxlim[2]) )
    nobjs = len(w[0])
    if nobjs > 0:
        objs = objs[w]
    else:
        log.error('No objects in [G,R,Z] brighter than {}'.format(maglim))

    #ADM format the passed objects as a mask to facilitate working with the 
    #ADM masking software in the brightmask module
    mask = format_as_mask(objs,navoid=navoid)
    
    #ADM split the mask on roughly arcminute scales, which seems to greatly
    #ADM speed up the masking when there are a few objects with large radii
    biggerthansep = mask["IN_RADIUS"] > 60.
    wbig = np.where(biggerthansep)
    wsmall = np.where(~biggerthansep)
    nbig = len(wbig[0])
    nsmall = len(wsmall[0])
    log.info('{} out of {} masks are "big"'.format(nbig,nsmall+nbig))
    if nbig > 0:
        bigmask = mask[wbig]
        smallmask = mask[wsmall]
    else:
        smallmask = mask

    #ADM how many sky positions we need to generate to meet the minimum density requirements
    nskies = int(spharea*nskymin)
    #ADM how many sky positions to generate, given that we'll reject objects close to bad
    #ADM sources. The factor of 1.2 was derived by trial-and-error...but this doesn't need
    #ADM to be optimal as this algorithm should become more sophisticated
    nchunk = int(nskies*1.2)

    #ADM arrays of GOOD sky positions to populate with coordinates
    ragood, decgood = np.empty(nskies), np.empty(nskies)
    #ADM lists of BAD sky positions to populate with coordinates
    rabad, decbad = [], []

    #ADM ngenerate will become zero when we generate enough GOOD sky positions
    ngenerate = nskies

    while ngenerate:

        #ADM generate random points in RA and Dec (correctly distributed on the sphere)
        log.info('Generated {} test positions...t = {:.1f}s'.format(nchunk,time()-start))
        skies = np.empty(nchunk, dtype=[('RA', '>f8'), ('DEC', '>f8')])
        skies["RA"] = np.random.uniform(ramin,ramax,nchunk)
        skies["DEC"] = np.degrees(np.arcsin(1.-np.random.uniform(1-sindecmax,1-sindecmin,nchunk)))        

        #ADM set up a list of skies that don't match an object
        goodskies = np.ones(len(skies),dtype=bool)

        #ADM determine which of the sky positions are in an object (mask)
        #ADM first for the big objects (if there are any)
        if nbig > 0:
            isin = is_in_bright_mask(skies,bigmask,inonly=True)
            wbad = np.where(isin)
            wgood = np.where(~isin)
            #ADM set every sky that is in a big mask to False
            if len(wbad[0]) > 0:
                goodskies[wbad] = False

            if not len(wgood[0]) > 0:
                #ADM if everything intersected a "big" mask on the first pass
                #ADM then we may as well return everything as a "bad" sky
                log.warning("No possible good skies in {:.2f},{:.2f},{:.2f},{:.2f} box".format(
                    ramin,ramax,decmin,decmax))
                return np.array([]), np.array([]), skies["RA"], skies["DEC"]

        #ADM now for the small objects
        isin = is_in_bright_mask(skies,smallmask,inonly=True)
        wbad = np.where(isin)
        #ADM set every sky that is in a small mask to False
        if len(wbad[0]) > 0:
            goodskies[wbad] = False

        #ADM record the good sky positions we found
        ra, dec = skies["RA"], skies["DEC"]
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


def plot_sky_positions(ragood,decgood,rabad,decbad,objs,navoid=1.,maglim=[20,20,20],
                       limits=None,plotname=None):
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
    navoid : :class:`float`, optional, defaults to 1.
        the number of times the galaxy half-light radius (or seeing) that
        objects (objs) were avoided out to when generating sky positions
    maglim : :class:`list`, optional, defaults to [20,20,20]
        The "upper limit" in the three optical DESI selection bands. Objects 
        fainter than these limits in ALL of g, r, z will NOT be used as masks.
    limits : :class:`~numpy.array`, optional, defaults to None
        plot limits in the form [ramin, ramax, decmin, decmax] if None
        is passed, then the entire area is plotted
    plotname : :class:`str`, defaults to None
        If a name is passed use matplotlib's savefig command to save the
        plot to that file name. Otherwise, display the plot

    Returns
    -------
    Nothing
    """

    import matplotlib.pyplot as plt
    from desitarget.geomask import ellipses
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    #ADM initialize the default logger
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()

    #ADM set up the figure and the axis labels
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlabel('RA (o)')
    ax.set_ylabel('Dec (o)')

    #ADM check if input objs is a filename or the actual data
    if isinstance(objs, str):
        objs = io.read_tractor(objs)

    #ADM change input magnitude limits to a flux to test against
    fluxlim = 10.**((22.5-np.array(maglim))/2.5)
    #ADM retrieve only objects at requested flux limits
    w = np.where(  (objs["FLUX_G"] > fluxlim[0]) 
                 | (objs["FLUX_R"] > fluxlim[1])  
                 | (objs["FLUX_Z"] > fluxlim[2]) )
    nobjs = len(w[0])
    if nobjs > 0:
        objs = objs[w]
    else:
        log.error('No objects in [G,R,Z] brighter than {}'.format(maglim))

    #ADM coordinate limits for the passed objs
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
        ralo, rahi = ramin, ramax
        declo, dechi = decmin, decmax
    else:
        ralo, rahi, declo, dechi = limits

    dum = plt.axis([ralo,rahi,declo,dechi])

    #ADM plot good and bad sky positions
    ax.scatter(ragood,decgood,marker='.',facecolors='none',edgecolors='k')
    ax.scatter(rabad,decbad,marker='.',facecolors='r')

    #ADM the size that defines a PSF versus an elliptical avoidance zone
    sepsplit = (psfsize*navoid)+1e-8
    smallsepw = np.where(sep <= sepsplit)[0]
    bigsepw = np.where(sep > sepsplit)[0]

    #ADM first the PSF or "small separation objects"...
    #ADM set up the ellipse shapes based on sizes of the past avoidance zones
    #ADM remembering to stretch by the COS term to de-project the sky
    minoraxis = sep/3600.
    majoraxis = minoraxis/np.cos(np.radians(objs["DEC"]))
    log.info('Plotting avoidance zones...t = {:.1f}s'.format(time()-start))
    #ADM plot the avoidance zones as circles, stretched by their DEC position
    #ADM note that "ellipses" takes the diameter, not the radius
    out = ellipses(objs[smallsepw]["RA"], objs[smallsepw]["DEC"],
                   2*majoraxis[smallsepw], 2*minoraxis[smallsepw], alpha=0.4, edgecolor='none')

    #ADM now the elliptical or "large separation objects"...
    #ADM loop through the DEV and EXP shapes, and create polygons
    #ADM of them to plot (where they're defined)
    patches = []
    for i, valobj in enumerate(bigsepw):
        if objs[valobj]["SHAPEEXP_R"] > 0:
            #ADM points on the ellipse boundary for EXP objects
            ras, decs = ellipse_boundary(objs[valobj]["RA"], objs[valobj]["DEC"],
                                       objs[valobj]["SHAPEEXP_R"]*navoid,
                                       objs[valobj]["SHAPEEXP_E1"], objs[valobj]["SHAPEEXP_E2"])
            polygon = Polygon(np.array(list(zip(ras,decs))), True)
            patches.append(polygon)
        if objs[valobj]["SHAPEDEV_R"] > 0:
            #ADM points on the ellipse boundary for DEV objects
            ras, decs = ellipse_boundary(objs[valobj]["RA"], objs[valobj]["DEC"],
                                       objs[valobj]["SHAPEDEV_R"]*navoid,
                                       objs[valobj]["SHAPEDEV_E1"], objs[valobj]["SHAPEDEV_E2"])
            polygon = Polygon(np.array(list(zip(ras,decs))), True)
            patches.append(polygon)

    p = PatchCollection(patches, alpha=0.4, facecolors='b', edgecolors='b')
    ax.add_collection(p)

    #ADM display the plot, if requested
    if plotname is None:
        log.info('Displaying plot...t = {:.1f}s'.format(time()-start))
        plt.show()
    else:
        plt.savefig(plotname)

    log.info('Done...t = {:.1f}s'.format(time()-start))

    return


def make_sky_targets(objs,navoid=1.,nskymin=None,maglim=[20,20,20]):
    """Generate sky targets and translate them into the typical format for DESI targets

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        numpy structured array with UPPERCASE columns, OR a string
        tractor/sweep filename. Must contain at least the columns
        "RA", "DEC", "SHAPEDEV_R", "SHAPEEXP_R"
    navoid : :class:`float`, optional, defaults to 1.
        the number of times the galaxy half-light radius (or seeing) to avoid
        objects out to when placing sky fibers
    nskymin : :class:`float`, optional, defaults to reading from desimodel.io
        the minimum DENSITY of sky fibers to generate
    maglim : :class:`list`, optional, defaults to [20,20,20]
        The "upper limit" in each of the three optical DESI selection bands. 
        Masks fainter than these limits in g, r, z will NOT be plotted

    Returns
    -------
    :class:`~numpy.ndarray`
        a structured array of good and bad sky positions in the DESI target format

    Notes
    -----
    The code generates unique OBJIDs based on an integer counter for the numbers of
    objects (objs) passed. It will therefore fail if the length of objs is longer
    than the number of bits reserved for OBJID in `desitarget.targetmask`
    """

    #ADM initialize the default logger
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()

    nobjs = len(objs)
    #ADM ensure that objs isn't longer than the largest possible OBJID
    if nobjs > 2**targetid_mask.OBJID.nbits:
        log.error('{} objects passed, but OBJID cannot exceed {}'
                  .format(nobjs,2**targetid_mask.OBJID.nbits))

    #ADM check if input objs is a filename or the actual data
    if isinstance(objs, str):
        objs = io.read_tractor(objs)

    log.info('Generating sky positions...t = {:.1f}s'.format(time()-start))
    #ADM generate arrays of good and bad objects for this sweeps file (or set)
    ragood, decgood, rabad, decbad = generate_sky_positions(objs,navoid=navoid,
                                                            nskymin=nskymin,maglim=maglim)
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
    desi_target[ngood:nskies] |= desi_mask.BAD_SKY

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
    skies["OBJID"] = np.arange(nskies)

    log.info('Finalizing target bits...t = {:.1f}s'.format(time()-start))
    #ADM add target bit columns to the output array, note that mws_target
    #ADM and bgs_target should be zeros for all sky objects
    dum = np.zeros_like(desi_target)
    skies = finalize(skies, desi_target, dum, dum, sky=1)

    log.info('Done...t = {:.1f}s'.format(time()-start))

    return skies


def select_skies(infiles, numproc=4, maglim=[20,20,20]):
    """Process input files in parallel to select blank sky positions

    Parameters
    ----------
    infiles : :class:`list` or `str`
        list of input filenames (tractor or sweep files),
            OR a single filename
    numproc : :class:`int`, optional, defaults to 4 
        number of parallel processes to use. Pass numproc=1 to run in serial
    maglim : :class:`list`, optional, defaults to [20,20,20]
        The "upper limit" in the three optical DESI selection bands. Objects 
        fainter than these limits in ALL of g, r, z will NOT be used as masks.

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
            raise ValueError

    #ADM function to run file-by-file for each sweeps file
    def _select_skies_file(filename):
        '''Returns targets in filename that pass the cuts'''
        return make_sky_targets(filename,navoid=1.,nskymin=None,maglim=maglim)

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

    #ADM because sweeps may not contain unique bricks (a brick can span multiple
    #ADM sweeps files), it may be necessary to update the OBJIDs to be unique
    #ADM this is typically only the case for a small number of objects so its
    #ADM quickest just to update those objects...
    #ADM ...only proceed if duplicates exist, as finding duplicates is slow
    if len(skies) - len(set(skies["TARGETID"])) > 0:
        log.info('Reassigning unique TARGETIDs to sky objects...t = {:.1f}s'
                 .format(time()-start))
        dd = find_duplicates_and_indexes(skies["TARGETID"])
        #ADM the maximum current OBJID (anything above this will be unique)
        maxobjid = np.max(skies["BRICK_OBJID"])
        #ADM the indexes of all of the duplicates
        dups = np.hstack(dd.values())
        ndups = len(dups)
        #ADM assign all duplicates a new OBJID that is higher than any other
        newobjid = np.arange(maxobjid,maxobjid+ndups)+1
        newskies = skies.copy()
        skies["BRICK_OBJID"][dups] = newobjid
        #ADM recalculate the TARGETID for these newly assigned OBJIDs
        targetid = encode_targetid(objid=skies['BRICK_OBJID'][dups], 
                                   brickid=skies['BRICKID'][dups], 
                                   release=skies['RELEASE'][dups],
                                   sky=1)    
        skies["TARGETID"][dups] = targetid

    return skies
