# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==========================
desitarget.imagefootprint
==========================

Monte Carlo Legacy Surveys imaging at the pixel level to model the imaging footprint
"""
import os
import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS
from time import time
import healpy as hp

#ADM fake the matplotlib display so it doesn't die on allocated nodes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#ADM the parallelization script
from desitarget.internal import sharedmem

#ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

#ADM start the clock
start = time()


def dr_extension(drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """Determine the extension information for files in a legacy survey coadd directory
    
    Parameters
    ----------
    drdir : :class:`str`, optional, defaults to dr4 root directory on NERSC
       The root directory pointing to a Data Release from the Legacy Surveys

    Returns
    -------
    :class:`str`
        Whether the file extension is 'gz' or 'fz'
    :class:`int`
        The corresponding FITS extension number that needs to be read (0 or 1)
    """

    from glob import iglob
    
    #ADM for speed, create a generator of all of the nexp files in the coadd directory
    gen = iglob(drdir+"/coadd/*/*/*nexp*")
    #ADM and pop the first one
    anexpfile = next(gen)
    extn = anexpfile[-2:]

    if extn == 'gz':
        return 'gz', 0

    return 'fz', 1


def randoms_in_a_brick_from_edges(ramin,ramax,decmin,decmax,density=10000):
    """For given brick edges, return random (RA/Dec) positions in the brick

    Parameters
    ----------
    ramin : :class:`float`
        The minimum "edge" of the brick in Right Ascension
    ramax : :class:`float`
        The maximum "edge" of the brick in Right Ascension
    decmin : :class:`float`
        The minimum "edge" of the brick in Declination
    decmax : :class:`float`
        The maximum "edge" of the brick in Declination
    density : :class:`int`
        The number of random points to return per sq. deg. As a typical brick is 
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned

    Returns
    -------
    :class:`~numpy.array`
        Right Ascensions of random points in brick
    :class:`~numpy.array`
        Declinations of random points in brick
    """
    #ADM generate random points within the brick at the requested density
    #ADM guard against potential wraparound bugs (assuming bricks are typical
    #ADM sizes of 0.25 x 0.25 sq. deg., or not much larger than that
    if ramax - ramin > 350.:
        ramax -= 360.
    sindecmin, sindecmax = np.sin(np.radians(decmin)), np.sin(np.radians(decmax))
    spharea = (ramax-ramin)*np.degrees(sindecmax-sindecmin)
    nrand = int(spharea*density)
#    log.info('Full area covered by brick is {:.5f} sq. deg....t = {:.1f}s'
#              .format(spharea,time()-start))
    ras = np.random.uniform(ramin,ramax,nrand)
    decs = np.degrees(np.arcsin(1.-np.random.uniform(1-sindecmax,1-sindecmin,nrand)))

    nrand= len(ras)

#    log.info('Generated {} randoms in brick with bounds [{:.3f},{:.3f},{:.3f},{:.3f}]...t = {:.1f}s'
#                 .format(nrand,ramin,ramax,decmin,decmax,time()-start))

    return ras, decs


def randoms_in_a_brick_from_name(brickname,density=10000,
                       drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """For a given brick name, return random (RA/Dec) positions in the brick

    Parameters
    ----------
    brickname : :class:`str`
        Name of brick in which to generate random points
    density : :class:`int`
        The number of random points to return per sq. deg. As a typical brick is 
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned
    drdir : :class:`str`, optional, defaults to dr4 root directory on NERSC
       The root directory pointing to a Data Release from the Legacy Surveys

    Returns
    -------
    :class:`~numpy.array`
        Right Ascensions of random points in brick
    :class:`~numpy.array`
        Declinations of random points in brick

    Notes
    -----
        - First version copied shamelessly from Anand Raichoor
    """
    #ADM read in the survey bricks file to determine the brick boundaries
    hdu = fits.open(drdir+'survey-bricks.fits.gz')

    brickinfo = hdu[1].data
    wbrick = np.where(brickinfo['brickname']==brickname)[0]
    if len(wbrick)==0:
        log.error('Brick {} does not exist'.format(brickname))
#    else:
#        log.info('Working on brick {}...t = {:.1f}s'.format(brickname,time()-start))

    brick = brickinfo[wbrick][0]
    ramin, ramax, decmin, decmax = brick['ra1'], brick['ra2'], brick['dec1'], brick['dec2']

    #ADM generate random points within the brick at the requested density
    #ADM guard against potential wraparound bugs
    if ramax - ramin > 350.:
        ramax -= 360.
    sindecmin, sindecmax = np.sin(np.radians(decmin)), np.sin(np.radians(decmax))
    spharea = (ramax-ramin)*np.degrees(sindecmax-sindecmin)
    nrand = int(spharea*density)
#    log.info('Full area covered by brick {} is {:.5f} sq. deg....t = {:.1f}s'
#              .format(brickname,spharea,time()-start))
    ras = np.random.uniform(ramin,ramax,nrand)
    decs = np.degrees(np.arcsin(1.-np.random.uniform(1-sindecmax,1-sindecmin,nrand)))

    nrand= len(ras)

#    log.info('Generated {} randoms in brick {} with bounds [{:.3f},{:.3f},{:.3f},{:.3f}]...t = {:.1f}s'
#                 .format(nrand,brickname,ramin,ramax,decmin,decmax,time()-start))

    return ras, decs


def quantities_at_positions_in_a_brick(ras,decs,brickname,
                               drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """Return NOBS, GALDEPTH, PSFDEPTH (per-band) at positions in one brick of the Legacy Surveys

    Parameters
    ----------
    ras : :class:`~numpy.array`
        Right Ascensions of interest (degrees).
    decs : :class:`~numpy.array`
        Declinations of interest (degrees).
    brickname : :class:`str`
        Name of brick which contains RA/Dec positions, e.g., '1351p320'.
    drdir : :class:`str`, optional, defaults to dr4 root directory on NERSC
       The root directory pointing to a Data Release from the Legacy Surveys.

    Returns
    -------
    :class:`dictionary`
       The number of observations (NOBS_X), PSF depth (PSFDEPTH_X) and Galaxy depth (GALDEPTH_X) 
       at each passed position in the Legacy Surveys in each band X.

    Notes
    -----
        - First version copied shamelessly from Anand Raichoor 
    """
    npts = len(ras)

    #ADM determine whether the coadd files have extension .gz or .fz based on the DR directory
    extn, extn_nb = dr_extension(drdir)

    #ADM the output dictionary
    qdict = {}

    # as a speed up, we assume all images in different filters for the brick have the same WCS
    # -> if we have read it once (iswcs=True), we use this info
    iswcs = False

    #ADM loop through each of the filters and store the number of observations at the
    #ADM RA and Dec positions of the passed points
    for filt in ['g','r','z']:
        #ADM the input file labels, and output column names and output formats
        #ADM for each of the quantities of interest
        qnames = zip(['nexp','depth','galdepth'],
                     ['nobs','psfdepth','galdepth'],
                     ['i2','f4','f4'])
        for qin, qout, qform in qnames:
            fn = (drdir+'/coadd/'+brickname[:3]+'/'+brickname+'/'+
                  'legacysurvey-'+brickname+'-'+qin+'-'+filt+'.fits.'+extn)
            #ADM only process the WCS if there is a file corresponding to this filter
            if os.path.exists(fn):
                img = fits.open(fn)
                if (iswcs==False):
                    w = WCS(img[extn_nb].header)
                    x, y = w.all_world2pix(ras, decs, 0)
                    iswcs = True
                #ADM determine the quantity of interest at each passed location
                #ADM and store in a dictionary with the filter and quantity name.
                qdict[qout+'_'+filt] = img[extn_nb].data[y.astype("int"),x.astype("int")]
#                log.info('Determined {} using WCS for {}...t = {:.1f}s'
#                             .format(qout+'_'+filt,fn,time()-start))
            else:
#                log.info('no {} file at {}...t = {:.1f}s'
#                         .format(qin+'_'+filt,fn,time()-start))
                #ADM if the file doesn't exist, set the relevant quantities to zero
                #ADM for all of the passed
                qdict[qout+'_'+filt] = np.zeros(npts,dtype=qform)

#    log.info('Recorded quantities for each point in brick {}...t = {:.1f}s'
#                  .format(brickname,time()-start))

    return qdict


def hp_with_nobs_in_a_brick(ramin,ramax,decmin,decmax,brickname,density=10000,nside=256,
                            drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """Given a brick's edges/name, count randoms with NOBS > 1 in HEALPixels touching that brick

    Parameters
    ----------
    ramin : :class:`float`
        The minimum "edge" of the brick in Right Ascension
    ramax : :class:`float`
        The maximum "edge" of the brick in Right Ascension
    decmin : :class:`float`
        The minimum "edge" of the brick in Declination
    decmax : :class:`float`
        The maximum "edge" of the brick in Declination
    brickname : :class:`~numpy.array`
        Brick names that corresponnds to the brick edges, e.g., '1351p320'
    density : :class:`int`, optional, defaults to 10000
        The number of random points to return per sq. deg. As a typical brick is 
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel nside number) at which to build the map
    drdir : :class:`str`, optional, defaults to the the DR4 root directory at NERSC
        The root directory pointing to a Data Release of the Legacy Surveys, e.g.:
        "/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"

    Returns
    -------
    :class:`~numpy.ndarray`
        a numpy structured array with the following columns:
            HPXPIXEL: Integer numbers of (only) those HEALPixels that overlap the passed brick
            HPXCOUNT: Numbers of random points with one or more observations (NOBS > 0) in the 
                passed Data Release of the Legacy Surveys for each returned HPXPIXEL

    Notes
    -----
        - The HEALPixel numbering uses the NESTED scheme
        - In the event that there are no pixels with one or more observations in the passed
          brick, and empty structured array will be returned
    """
    #ADM this is only intended to work on one brick, so die if a larger array is passed
    if not isinstance(brickname,str):
        log.fatal("Only one brick can be passed at a time!")
        raise ValueError

    #ADM generate an empty structured array to return in the event that no pixels with
    #ADM counts were found
    hpxinfo = np.zeros(0, dtype=[('HPXPIXEL','>i4'),('HPXCOUNT','>i4')])

    #ADM generate random points within the brick at the requested density
    ras, decs = randoms_in_a_brick_from_edges(ramin,ramax,decmin,decmax,density=density)

    #ADM retrieve the number of observations for each random point
    nobs_g, nobs_r, nobs_z = nobs_at_positions_in_a_brick(ras,decs,brickname,drdir=drdir)

    #ADM only retain points with one or more observations in all bands
    w = np.where( (nobs_g > 0) & (nobs_r > 0) & (nobs_z > 0) )

    #ADM if there were some non-zero observations, populate the pixel numbers and counts
    if len(w[0]) > 0:
        pixnums = hp.ang2pix(nside,np.radians(90.-decs[w]),np.radians(ras[w]),nest=True)
        pixnum, pixcnt = np.unique(pixnums,return_counts=True)
        hpxinfo = np.zeros(len(pixnum), dtype=[('HPXPIXEL','>i4'),('HPXCOUNT','>i4')])
        hpxinfo['HPXPIXEL'] = pixnum
        hpxinfo['HPXCOUNT'] = pixcnt

    return hpxinfo


def get_quantities_in_a_brick(ramin,ramax,decmin,decmax,brickname,density=10000,
                            drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """NOBS, GALDEPTH, PSFDEPTH (per-band) for random points in a brick of the Legacy Surveys

    Parameters
    ----------
    ramin : :class:`float`
        The minimum "edge" of the brick in Right Ascension
    ramax : :class:`float`
        The maximum "edge" of the brick in Right Ascension
    decmin : :class:`float`
        The minimum "edge" of the brick in Declination
    decmax : :class:`float`
        The maximum "edge" of the brick in Declination
    brickname : :class:`~numpy.array`
        Brick names that corresponnds to the brick edges, e.g., '1351p320'
    density : :class:`int`, optional, defaults to 10000
        The number of random points to return per sq. deg. As a typical brick is 
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned
    drdir : :class:`str`, optional, defaults to the the DR4 root directory at NERSC
        The root directory pointing to a Data Release of the Legacy Surveys, e.g.:
        "/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"

    Returns
    -------
    :class:`~numpy.ndarray`
        a numpy structured array with the following columns:
            RA: Right Ascension of a random point
            DEC: Declination of a random point
            BRICKNAME: Passed brick name
            NOBS_G: Number of observations at this location in the g-band
            NOBS_R: Number of observations at this location in the r-band
            NOBS_Z: Number of observations at this location in the z-band
            PSFDEPTH_G: PSF depth at this location in the g-band
            PSFDEPTH_R: PSF depth at this location in the r-band
            PSFDEPTH_Z: PSF depth at this location in the z-band
            GALDEPTH_G: Galaxy depth at this location in the g-band
            GALDEPTH_R: Galaxy depth at this location in the r-band
            GALDEPTH_Z: Galaxy depth at this location in the z-band
    """
    #ADM this is only intended to work on one brick, so die if a larger array is passed
    if not isinstance(brickname,str):
        log.fatal("Only one brick can be passed at a time!")
        raise ValueError

    #ADM generate random points within the brick at the requested density
    ras, decs = randoms_in_a_brick_from_edges(ramin,ramax,decmin,decmax,density=density)

    #ADM retrieve the dictionary of quantities for each random point
    qdict = quantities_at_positions_in_a_brick(ras,decs,brickname,drdir=drdir)

    #ADM convert the dictionary to a structured array
    qinfo = np.zeros(len(ras), 
                     dtype=[('RA','f8'),('DEC','f8'),('BRICKNAME','S8'),
                            ('NOBS_G','i2'),('NOBS_R','i2'),('NOBS_Z','i2'),
                            ('PSFDEPTH_G','f4'),('PSFDEPTH_R','f4'),('PSFDEPTH_Z','f4'),
                            ('GALDEPTH_G','f4'),('GALDEPTH_R','f4'),('GALDEPTH_Z','f4')
                           ])
    #ADM store each quantity of interest in the structured array
    #ADM remembering that the dictionary keys are in lower case text
    cols = qdict.keys()
    for col in cols:
        qinfo[col.upper()] = qdict[col]
    #ADM add the RAs/Decs and brick name
    qinfo["RA"], qinfo["DEC"], qinfo["BRICKNAME"] = ras, decs, brickname

    return qinfo


def pixweight(randoms, nside=256, outplot=None):
    """NOBS, GALDEPTH, PSFDEPTH (per-band) for random points in a DR of the Legacy Surveys

    Parameters
    ----------
    randoms : :class:`~numpy.ndarray`
        A random catalog as made by, e.g., :func:`select_randoms()` or 
        :func:`quantities_at_positions_in_a_brick()` 
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel nside number) at which to build the map
    outplot : :class:`str`, optional, defaults to not making a plot
        Create a plot and write it to a file named `outplot` (this is passed to
        the `savefig` routine from `matplotlib.pyplot`

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of the weight for EACH pixel in the sky at the passed nside.

    Notes
    -----
        - The returned array contains the fraction of each pixel that overlaps areas that contain
          one or more observations in the passed Legacy Surveys Data Release (LS DR). 
        - `WEIGHT=1` means that this LS DR has one or more pointings across the entire pixel.
        - `WEIGHT=0` means that this pixel has no LS DR observations within it (e.g., perhaps 
          it is completely outside of the LS DR footprint).
        - `0 < WEIGHT < 1` for pixels that partially cover LS DR area with one or more observations.
        - The index of the array is the HEALPixel integer.
    """

    #ADM only retain points with one or more observations in all bands
    w = np.where( (nobs_g > 0) & (nobs_r > 0) & (nobs_z > 0) )

    npix = hp.nside2npix(nside)
    pix_cnt = np.bincount(hpxinfo['HPXPIXEL'], weights=hpxinfo['HPXCOUNT'], minlength=npix)
    
    #ADM we know the area of HEALPixels at this nside, so we know what the count SHOULD be
    expected_cnt = hp.nside2pixarea(nside,degrees=True)*density
    #ADM create a weight map based on the actual counts divided by the expected counts
    pix_weight = pix_cnt/expected_cnt

    #ADM if outplot was passed, make a plot of the weights in Mollweide projection
    if outplot is not None:
        log.info('Plotting pixel map and writing to {}'.format(outplot))
        hp.mollview(pix_weight, nest=True)
        plt.savefig(outplot)

    return pix_weight


def select_randoms(density=10000, numproc=16,
                   drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """NOBS, GALDEPTH, PSFDEPTH (per-band) for random points in a DR of the Legacy Surveys

    Parameters
    ----------
    density : :class:`int`, optional, defaults to 10000
        The number of random points to return per sq. deg. As a typical brick is 
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned
    numproc : :class:`int`, optional, defaults to 16
        The number of processes over which to parallelize
    drdir : :class:`str`, optional, defaults to dr4 root directory on NERSC
       The root directory pointing to a Data Release from the Legacy Surveys

    Returns
    -------
    :class:`~numpy.ndarray`
        a numpy structured array with the following columns:
            RA: Right Ascension of a random point
            DEC: Declination of a random point
            BRICKNAME: Passed brick name
            NOBS_G: Number of observations at this location in the g-band
            NOBS_R: Number of observations at this location in the r-band
            NOBS_Z: Number of observations at this location in the z-band
            PSFDEPTH_G: PSF depth at this location in the g-band
            PSFDEPTH_R: PSF depth at this location in the r-band
            PSFDEPTH_Z: PSF depth at this location in the z-band
            GALDEPTH_G: Galaxy depth at this location in the g-band
            GALDEPTH_R: Galaxy depth at this location in the r-band
            GALDEPTH_Z: Galaxy depth at this location in the z-band
    """

    #ADM read in the survey bricks file, which lists the bricks of interest for this DR
    from glob import glob
    sbfile = glob(drdir+'/*bricks-dr*')[0]
    hdu = fits.open(sbfile)
    brickinfo = hdu[1].data
    bricknames = brickinfo['brickname'][0:999]
    nbricks = len(bricknames)
    log.info('Processing {} bricks that have one or more observations...t = {:.1f}s'
             .format(nbricks,time()-start))

    #ADM initialize the bricks class, and retrieve the brick information look-up table
    #ADM so it can be used in a common fashion
    from desiutil import brick
    bricktable = brick.Bricks(bricksize=0.25).to_table()

    #ADM the critical function to run on every brick
    def _get_quantities(brickname):
        '''wrapper on nobs_positions_in_a_brick_from_edges() given a brick name'''
        #ADM retrieve the edges for the brick that we're working on
        wbrick = np.where(bricktable["BRICKNAME"] == brickname)[0]
        ramin, ramax, decmin, decmax = np.array(bricktable[wbrick]["RA1","RA2","DEC1","DEC2"])[0]

        #ADM populate the brick with random points, and retrieve the quantities
        #ADM of interest at those points
        return get_quantities_in_a_brick(ramin, ramax, decmin, decmax, brickname, 
                                         density=density, drdir=drdir)

    #ADM this is just to count bricks in _update_status
    nbrick = np.zeros((), dtype='i8')

    t0 = time()
    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if nbrick%50 == 0 and nbrick>0:
            rate = nbrick / (time() - t0)
            log.info('{}/{} bricks; {:.1f} bricks/sec'.format(nbrick, nbricks, rate))

        nbrick[...] += 1    # this is an in-place modification
        return result

    #- Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            qinfo = pool.map(_get_quantities, bricknames, reduce=_update_status)
    else:
        qinfo = list()
        for brickname in bricknames:
            qinfo.append(_update_status(_get_quantities(brickname)))

    qinfo = np.concatenate(qinfo)

    return qinfo
