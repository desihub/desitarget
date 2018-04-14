#!/usr/bin/env python

import os
import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS
from time import time

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


def randoms_in_a_brick(brickname,density=10000,
                       drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """For a given brick name, return random (RA/Dec) positions in the brick

    Parameters
    ----------
    brickname : :class:`str`
        Name of brick in which to generate random points
    density : :class:`float`
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
        - This version requires a survey bricks file to generate RAs and Decs in
          a brick (rather than, e.g., using :mod:`desiutil.brick)`
    """
    #GENERATE RANDOMS IN THE PASSED BRICK ============================================= 
    #ADM read in the survey bricks file to determine the brick boundaries
    hdu = fits.open(drdir+'survey-bricks.fits.gz')

    brickinfo = hdu[1].data
    wbrick = np.where(brickinfo['brickname']==brickname)[0]
    if len(wbrick)==0:
        log.error('Brick {} does not exist'.format(brickname))
    else:
        log.info('Working on brick {}...t = {:.1f}s'.format(brickname,time()-start))

    brick = brickinfo[wbrick][0]
    ramin, ramax, decmin, decmax = brick['ra1'], brick['ra2'], brick['dec1'], brick['dec2']

    #ADM generate random points within the brick at the requested density
    #ADM guard against potential wraparound bugs
    if ramax - ramin > 350.:
        ramax -= 360.
    sindecmin, sindecmax = np.sin(np.radians(decmin)), np.sin(np.radians(decmax))
    spharea = (ramax-ramin)*np.degrees(sindecmax-sindecmin)
    nrand = int(spharea*density)
    print('Full area covered by brick {} is {:.5f} sq. deg....t = {:.1f}s'
              .format(brickname,spharea,time()-start))
    ras = np.random.uniform(ramin,ramax,nrand)
    decs = np.degrees(np.arcsin(1.-np.random.uniform(1-sindecmax,1-sindecmin,nrand)))

    nrand= len(ras)

    log.info('Generated {} randoms in brick {} with bounds [{:.3f},{:.3f},{:.3f},{:.3f}]...t = {:.1f}s'
                 .format(nrand,brickname,ramin,ramax,decmin,decmax,time()-start))

    return ras, decs


def nobs_at_positions_in_brick(ras,decs,brickname,
                               drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """Return the number of observations at positions in one brick of the Legacy Surveys

    Parameters
    ----------
    ras : :class:`~numpy.array`
        Right Ascensions of interest (degrees)
    decs : :class:`~numpy.array`
        Declinations of interest (degrees)
    brickname : :class:`str`
        Name of brick which contains RA/Dec positions
    drdir : :class:`str`, optional, defaults to dr4 root directory on NERSC
       The root directory pointing to a Data Release from the Legacy Surveys

    Returns
    -------
    :class:`~numpy.array`
        The number of observations in the Legacy Surveys at each position in g-band
    :class:`~numpy.array`
        The number of observations in the Legacy Surveys at each position in r-band
    :class:`~numpy.array`
        The number of observations in the Legacy Surveys at each position in z-band

    Notes
    -----
        - First version copied shamelessly from Anand Raichoor 
    """
    npts = len(ras)

    #ADM determine whether the coadd files have extension .gz or .fz based on the DR directory
    extn, extn_nb = dr_extension(drdir)

    # as a speed up, we assume all images in different filters for the brick have the same WCS
    # -> if we have read it once (iswcs=True), we use this info
    iswcs = False

    #ADM loop through each of the filters and store the number of observations at the
    #ADM RA and Dec positions of th passed points
    nobsdict = {} 
    for filt in ['g','r','z']:
        nexpfile = (drdir+'/coadd/'+brickname[:3]+'/'+brickname+'/'+
                        'legacysurvey-'+brickname+'-'+'nexp'+'-'+filt+'.fits.'+extn)
        #ADM only process the WCS if there is a file corresponding to this filter
        if (os.path.exists(nexpfile)):
            img = fits.open(nexpfile)
            if (iswcs==False):
                w = WCS(img[extn_nb].header)
                x, y = w.all_world2pix(ras, decs, 0)
                iswcs = True
            #ADM determine the number of observations (NOBS) at each of the passed
            #ADM locations and store in arrays called nobs_g, nobs_r, nobs_z etc.
            nobsdict[filt] = img[extn_nb].data[y.astype("int"),x.astype("int")]
            log.info('Determined NOBS using WCS for {}...t = {:.1f}s'
                         .format(nexpfile,time()-start))
        else:
            log.info('no NEXP file at {}...t = {:.1f}s'.format(nexpfile,time()-start))
            #ADM if the file doesn't exist, set NOBS = 0 for all of the passed
            #ADM locations and store in arrays called nobs_g, nobs_r, nobs_z etc.
            nobsdict[filt] = np.zeros(npts,dtype="uint8")

    log.info('Recorded number of observations for each point in brick {}...t = {:.1f}s'
                 .format(brickname,time()-start))

    return nobsdict['g'], nobsdict['r'], nobsdict['z']


def nobs_at_positions(rasarray,decsarray,
                      drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """Return the number of observations at any positions in a Data Release of the Legacy Surveys

    Parameters
    ----------
    rasarray : :class:`~numpy.array`
        Right Ascensions of interest (degrees)
    decsarray : :class:`~numpy.array`
        Declinations of interest (degrees)
    drdir : :class:`str`, optional, defaults to dr4 root directory on NERSC
       The root directory pointing to a Data Release of the Legacy Surveys

    Returns
    -------
    :class:`~numpy.array`
        The number of observations in the Legacy Surveys at each position in g-band
    :class:`~numpy.array`
        The number of observations in the Legacy Surveys at each position in r-band
    :class:`~numpy.array`
        The number of observations in the Legacy Surveys at each position in z-band

    Notes
    -----
        - See also :func:`nobs_at_positions_in_brick`, which achieves the smae thing for
          a single brick and presupposes that all of the RAs/Decs are in that one brick
    """
    #ADM determine whether the coadd files have extension .gz or .fz based on the DR directory
    extn, extn_nb = dr_extension(drdir)

    #ADM assign brick names to each of the RAs/Decs
    from desiutil import brick
    b = brick.Bricks(bricksize=0.25)
    bricknames = b.brickname(rasarray,decsarray)
    log.info('loaded brick class and assigned brick names to positions...t = {:.1f}s'
                 .format(time()-start))

    #ADM set up output arrays of the number of observations in g, r, z
    #ADM default to -1 observations, so it's easier to test for bugs
    nras = len(rasarray)
    nobs_g = np.zeros(nras,dtype='int8')-1
    nobs_r = np.zeros(nras,dtype='int8')-1
    nobs_z = np.zeros(nras,dtype='int8')-1

    #ADM loop through the bricks, based on name and assign numbers of observations
    #ADM where we have them (otherwise, default to NOBS = 0 for a missing band)
    for brickname in set(bricknames):
        wbrick = np.where(bricknames == brickname)
        ras = rasarray[wbrick]
        decs = decsarray[wbrick]
        npts = len(ras)

        # as a speed up, we assume all images in different filters for the brick have the same WCS
        # -> if we have read it once (iswcs=True), we use this info
        iswcs = False

        #ADM loop through each of the filters and store the number of observations at the
        #ADM RA and Dec positions of the passed points
        nobsdict = {} 
        for filt in ['g','r','z']:
            nexpfile = (drdir+'/coadd/'+brickname[:3]+'/'+brickname+'/'+
                            'legacysurvey-'+brickname+'-'+'nexp'+'-'+filt+'.fits.'+extn)
            #ADM only process the WCS if there is a file corresponding to this filter
            if (os.path.exists(nexpfile)):
                img = fits.open(nexpfile)
                if (iswcs==False):
                    w = WCS(img[extn_nb].header)
                    x, y = w.all_world2pix(ras, decs, 0)
                    iswcs = True
                    #ADM determine the number of observations (NOBS) at each of the
                    #ADM passed locations and store in arrays called nobs_g, nobs_r, nobs_z etc.
                nobsdict[filt] = img[extn_nb].data[y.astype("int"),x.astype("int")]
                log.info('Determined NOBS using WCS for {}...t = {:.1f}s'
                         .format(nexpfile,time()-start))
            else:
                #ADM if the file doesn't exist, set NOBS = 0 for passed points that are in
                #ADM this brick and filter
                log.info('no NEXP file at {}...t = {:.1f}s'.format(nexpfile,time()-start))
                nobsdict[filt] = np.zeros(npts,dtype="uint8")

        log.info('Recorded number of observations for each point in brick {}...t = {:.1f}s'
                 .format(brickname,time()-start))

        #ADM populate the output final arrays based on which points are in this brick
        nobs_g[wbrick] =  nobsdict['g']
        nobs_r[wbrick] =  nobsdict['r']
        nobs_z[wbrick] =  nobsdict['z']

    return nobs_g, nobs_r, nobs_z

