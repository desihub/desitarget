# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=====================
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

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.collections import PatchCollection

from . import __version__ as desitarget_version
from . import gitversion

from desitarget import io
from desitarget.internal import sharedmem
from desitarget import desi_mask

def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of circles. 
    Similar to plt.scatter, but the size of circles are in data scale.

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

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)

    Attribution
    -----------
    With thanks to https://gist.github.com/synnick/5088216
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


def collect_bright_stars(bands,maglim,numproc=4,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1',outfilename=None,verbose=True):
    """Extract a structure from the sweeps containing only bright stars in a given band to a given magnitude limit

    Parameters
    ----------
    bands : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z"
        Can pass multiple bands as string, e.g. "GRZ", in which case maglim has to be a 
           list of the same length as the string
    maglim : :class:`float`
        The upper limit in that magnitude band for which to assemble a list of bright stars
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
    #ADM the band as an integer location
    bandint = np.array([ "UGRIZY".find(band) for band in bands ])

    if len(bandint) != len(maglim):
        raise IOError('bands has to be the same length as maglim and {} does not equal {}'.format(len(bandint),len(maglim)))

    #ADM change input magnitude(s) to a flux to test against
    fluxlim = 10.**((22.5-np.array(maglim))/2.5)

    #ADM parallel formalism from this step forward is stolen from cuts.select_targets

    #ADM function to grab the bright stars from a given file
    def _get_bright_stars(filename):
        '''Retrieves bright stars from a sweeps/Tractor file'''
        objs = io.read_tractor(filename)
        #ADM Retain rows for which ANY band is brighter than maglim
        w = np.where(np.any(objs["DECAM_FLUX"][...,bandint] > fluxlim,axis=1))
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
    :class:`recarray`
        dictionary of the fraction of bricks containing a star of a given
        magnitude in a given band as function of Galactic l Keys are mag
        bin CENTERS, values are arrays running from 0->1 to 359->360
    :class:`recarray`
        dictionary of the fraction of bricks containing a star of a given
        magnitude in a given band as function of Galactic b. Keys are mag
        bin CENTERS, values are arrays running from -90->-89 to 89->90

    Notes
    -----
    converts using coordinates of the brick center, so is an approximation

    """
    #ADM histogram bin edges in Galactic coordinates at resolution of 1 degree
    lbinedges = np.arange(361)
    bbinedges = np.arange(-90,91)

    #ADM set band to uppercase if passed as lower case
    band = band.upper()
    #ADM the band as an integer location
    bandint = "UGRIZY".find(band)

    #ADM read in the bright object file
    fx = fitsio.FITS(instarfile)
    objs = fx[1].read()
    #ADM convert fluxes in band of interest for each object to magnitudes
    mags = 22.5-2.5*np.log10(objs["DECAM_FLUX"][...,bandint])
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

 
def make_bright_star_mask(bands,maglim,numproc=4,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1',infilename=None,outfilename=None,verbose=True):
    """Make a bright star mask from a structure of bright stars drawn from the sweeps

    Parameters
    ----------
    bands : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z"
        Can pass multiple bands as string, e.g. "GRZ", in which case maglim has to be a 
           list of the same length as the string
    maglim : :class:`float`
        The upper limit in that magnitude band for which to assemble a list of bright stars
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
        The bright star mask in the form RA,DEC,TARGETID,RADIUS (may also be written to file if outfilename is passed)
        radius is in ARCMINUTES
        TARGETID is as calculated in desitarget.targets.py

    Notes
    -----
    Currently uses the radius-as-a-function-of-B-mag for Tycho stars from the BOSS mask (in every band):

    R = (0.0802B*B - 1.860B + 11.625) (see Eqn. 9 of https://arxiv.org/pdf/1203.6594.pdf)

    It's an open question as to what the correct radius is for DESI observations

    """

    #ADM set bands to uppercase if passed as lower case
    bands = bands.upper()
    #ADM the band as an integer location
    bandint = np.array([ "UGRIZY".find(band) for band in bands ])

    #ADM force the input maglim to be a list (in case a single value was passed)
    if type(maglim) == type(16) or type(maglim) == type(16.):
        maglim = [maglim]

    if len(bandint) != len(maglim):
        raise IOError('bands has to be the same length as maglim and {} does not equal {}'.format(len(bandint),len(maglim)))

    #ADM change input magnitude(s) to a flux to test against
    fluxlim = 10.**((22.5-np.array(maglim))/2.5)

    if infilename is not None:
        objs = io.read_tractor(infilename)
    else:
        objs = collect_bright_stars(bands,maglim,numproc,rootdirname,outfilename,verbose)
   
    #ADM set any observations with NOBS = 0 to have zero flux so glitches don't end up as bright star masks
    w = np.where(objs["DECAM_NOBS"] == 0)
    if len(w[0]) > 0:
        objs["DECAM_FLUX"][w] = 0.

    #ADM limit to the passed faint limit
    w = np.where(np.any(objs["DECAM_FLUX"][...,bandint] > fluxlim,axis=1))
    objs = objs[w]

    #ADM grab the (GRZ) magnitudes for observations
    #ADM and record only the largest flux (smallest magnitude)
    fluxmax =  np.max(objs["DECAM_FLUX"][...,bandint],axis=1)
    mags = 22.5-2.5*np.log10(fluxmax)

    #ADM convert the largest magnitude into a radius. This will require more consideration
    #ADM to determine the truly correct numbers for DESI
    radius = (0.0802*mags*mags - 1.860*mags + 11.625)

    #ADM calculate the TARGETID
    targetid = objs['BRICKID'].astype(np.int64)*1000000 + objs['OBJID']

    #ADM create an output recarray that is just RA, Dec, TARGETID and the radius
    done = objs[['RA','DEC']].copy()
    done = rfn.append_fields(done,["TARGETID","RADIUS"],[targetid,radius],usemask=False,dtypes=['>i8','<f4'])

    if outfilename is not None:
        fitsio.write(outfilename, done, clobber=True)

    return done


def plot_mask(mask,limits=None,over=False):
    """Make a plot of a mask and either display it or retain the plot object for over-plotting
    
    Parameters
    ----------
    mask : :class:`recarray`
        A mask constructed by make_bright_star_mask (or read in from file in the make_bright_star_mask format)
    limits : :class:`list`, optional
        A list defining the RA/Dec limits of the plot as would be passed to matplotlib.pyplot.axis
    over : :class:`boolean`
        If False, then don't "show" the plot, so that the plotting commands can be combined with
        other matplotlib commands to save the figure, over-plot targets etc.

    Returns
    -------
        Nothing
    """

    #ADM set up the plot
    plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect='equal')
    plt.xlabel('RA (o)')
    plt.ylabel('Dec (o)')

    if limits is not None:
        plt.axis(limits)

    #ADM draw circle patches from the mask information converting radius to degrees
    out = circles(mask["RA"], mask["DEC"], mask["RADIUS"]/60., alpha=0.2, edgecolor='none')

    if not over:
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
        mask : array_like. True if target is in a bright star mask  
    """

    #ADM initialize an array of all False (nothing is yet in a star mask)
    done = np.zeros(len(targs), dtype=bool)

    #ADM turn the coordinates of the masks and the targets into SkyCoord objects
    ctargs = SkyCoord(targs["RA"]*u.degree, targs["DEC"]*u.degree)
    cstars = SkyCoord(starmask["RA"]*u.degree, starmask["DEC"]*u.degree)
    
    #ADM this is the largest search radius we should need to consider
    #ADM in the future an obvious speed up is to split on radius 
    #ADM as large radii are rarer but take longer
    maxrad = max(starmask["RADIUS"])*u.arcmin

    #ADM coordinate match the star masks and the targets
    idtargs, idstars, d2d, d3d = cstars.search_around_sky(ctargs,maxrad)

    #ADM catch the case where nothing fell in a mask
    if len(idstars) == 0:
        return done

    #ADM for a matching star mask, find the angular separations that are less than the mask radius
    w = np.where(d2d.arcmin < starmask[idstars]["RADIUS"])

    #ADM any matching idtargs that meet this separation criterion are in a mask (at least one)
    done[idtargs[w]] = 'True'

    return done


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

    To Do
    -----
        - Currently sets IN_BRIGHT_OBJECT but should also match on the TARGETID to set BRIGHT_OBJECT bit
        - Should also set NEAR_BRIGHT_OBJECT at an appropriate radius in is_in_bright_star

    See desitarget.targetmask for the definition of each bit
    """

    in_bright_object = is_in_bright_star(targs,starmask)

    desi_target = targs["DESI_TARGET"].copy()
    desi_target |= in_bright_object * desi_mask.IN_BRIGHT_OBJECT
    
    return desi_target


def mask_targets(targs,instarmaskfile=None,bands="GRZ",maglim=[10,10,10],numproc=4,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1',outfilename=None,verbose=True):
    """Add bits for whether objects are in a bright star mask to list of targets

    Parameters
    ----------
    targs : :class:`str` or recarray
        A recarray of targets created by desitarget.cuts.select_targets OR a filename of
        a file that contains such a set of targets
    instarmaskfile : :class=`str`, optional
        An input bright star mask created by desitarget.brightstar.make_bright_star_mask
        If None, defaults to making the bright star mask from scratch
        The next 5 parameters are only relevant to making the bright star mask from scratch
    bands : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z"
        Can pass multiple bands as string, e.g. "GRZ", in which case maglim has to be a 
           list of the same length as the string
    maglim : :class:`float`
        The upper limit in that magnitude band for which to assemble a list of bright stars
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

    Returns:
    --------
        targets numpy structured array: the input targets with the DESI_TARGET column 
        updated to reflect the BRIGHT_OBJECT bits.

    Notes: 
    ------
        Runs in about 10 minutes for 20M targets and 50k masks (roughly maglim=10) 
        (not including 5-10 minutes to build the star mask from scratch)
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
        print('Number of targets {}...t={:.1f}s'.format(len(targs), time()-t0))
        print('Number of star masks {}...t={:.1f}s'.format(len(starmask), time()-t0))

    dt = set_target_bits(targs,starmask)
    done = targs.copy()
    done["DESI_TARGET"] = dt
    
    if verbose:
        print('Finishing up...t={:.1f}s'.format(time()-t0))

    return done
