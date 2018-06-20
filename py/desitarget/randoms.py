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
import fitsio

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


def randoms_in_a_brick_from_edges(ramin,ramax,decmin,decmax,density=100000):
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
    density : :class:`int`, optional, defaults to 100,000
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


def randoms_in_a_brick_from_name(brickname,density=100000,
                       drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """For a given brick name, return random (RA/Dec) positions in the brick

    Parameters
    ----------
    brickname : :class:`str`
        Name of brick in which to generate random points
    density : :class:`int`, optional, defaults to 100,000
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


def hp_with_nobs_in_a_brick(ramin,ramax,decmin,decmax,brickname,density=100000,nside=256,
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
    density : :class:`int`, optional, defaults to 100,000
        The number of random points to return per sq. deg. As a typical brick is 
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel nside number) at which to build the map
    drdir : :class:`str`, optional, defaults to the DR4 root directory at NERSC
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


def get_dust(ras,decs, 
             dust_dir="/project/projectdirs/desi/software/edison/dust/v0_1/maps"):
    """Get SFD E(B-V) values at a set of RA/Dec locations

    Parameters
    ----------
    ra : :class:`numpy.array`
        Right Ascension in degrees
    dec : :class:`numpy.array`
        Declination in degrees
    dust_dir : :class:`str`, optional, defaults to the NERSC dust map location
        The root directory pointing to SFD dust maps

    Returns
    -------
    :class:`numpy.array`
        E(B-V) values from the SFD dust maps at the passed locations
    """
    from desitarget.mock import sfdmap
    return sfdmap.ebv(ras, decs, mapdir=dust_dir)
    

def get_quantities_in_a_brick(ramin,ramax,decmin,decmax,brickname,density=100000,
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
    density : :class:`int`, optional, defaults to 100,000
        The number of random points to return per sq. deg. As a typical brick is 
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned
    drdir : :class:`str`, optional, defaults to the DR4 root directory at NERSC
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


def pixweight(randoms, density, nside=256, outplot=None):
    """Fraction of area covered in HEALPixels by a random catalog

    Parameters
    ----------
    randoms : :class:`~numpy.ndarray` or `str`
        A random catalog as made by, e.g., :func:`select_randoms()` or 
        :func:`quantities_at_positions_in_a_brick()`, or a file that contains such a catalog.
        Must contain the columns RA, DEC, NOBS_G, NOBS_R, NOBS_Z.
    density : :class:`int`
        The number of random points per sq. deg. At which the random catalog was
        generated (see also :func:`select_randoms()`).
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel nside number) at which to build the map.
    outplot : :class:`str`, optional, defaults to not making a plot
        Create a plot and write it to a file named `outplot` (this is passed to
        the `savefig` routine from `matplotlib.pyplot`.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of the weight for EACH pixel in the sky at the passed nside.

    Notes
    -----
        - The returned array contains the fraction of each pixel that overlaps areas that contain
          one or more observations in the passed random catalog.
        - `WEIGHT=1` means that this LS DR has one or more pointings across the entire pixel.
        - `WEIGHT=0` means that this pixel has no LS DR observations within it (e.g., perhaps 
          it is completely outside of the LS DR footprint).
        - `0 < WEIGHT < 1` for pixels that partially cover LS DR area with one or more observations.
        - The index of the array is the HEALPixel integer.
    """
    #ADM if a file name was passed for the random catalog, read it in
    if isinstance(randoms, str):
        randoms = fitsio.read(randoms)

    #ADM extract the columns of interest
    ras, decs = randoms["RA"], randoms["DEC"]
    nobs_g, nobs_r, nobs_z = randoms["NOBS_G"], randoms["NOBS_R"], randoms["NOBS_Z"]    

    #ADM only retain points with one or more observations in all bands
    w = np.where( (nobs_g > 0) & (nobs_r > 0) & (nobs_z > 0) )

    #ADM the counts in each HEALPixel in the survey
    if len(w[0]) > 0:
        pixnums = hp.ang2pix(nside,np.radians(90.-decs[w]),np.radians(ras[w]),nest=True)
        pixnum, pixcnt = np.unique(pixnums,return_counts=True)
    else:
        log.error("Empty array passed")

    #ADM generate the counts for the whole sky to retain zeros where there is no survey coverage
    npix = hp.nside2npix(nside)
    pix_cnt = np.bincount(pixnum, weights=pixcnt, minlength=npix)
    
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


def stellar_density(nside=256,
            gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Make a HEALPixel map of stellar density based on Gaia

    Parameters
    ----------
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel nside number) at which to build the map.
    gaiadir : :class:`str`, optional, defaults to Gaia DR2 path at NERSC
        Root directory of a Gaia Data Release as used by the Legacy Surveys.
    """
    #ADM the number of pixels and the pixel area at the passed nside
    npix = hp.nside2npix(nside)
    pixarea = hp.nside2pixarea(nside,degrees=True)

    #ADM an output array to populate containing all possible HEALPixels at the passed nside
    pixout = np.zeros(npix,dtype='int32')

    #ADM find all of the Gaia files
    from glob import glob
    filenames = glob(gaiadir+'/*fits')
    
    #ADM read in each file, restricting to the criteria for point sources
    #ADM and storing in a HEALPixel map at resolution nside
    for filename in filenames:
        #ADM save memory and speed up by only reading in a subset of columns
        gobjs = fitsio.read(filename,
                            columns=['ra','dec','phot_g_mean_mag','astrometric_excess_noise'])

        #ADM restrict to subset of sources using point source definition
        ra, dec = gobjs["ra"], gobjs["dec"]
        gmag, excess = gobjs["phot_g_mean_mag"], gobjs["astrometric_excess_noise"]
        point = (excess==0.) | (np.log10(excess) < 0.3*gmag-5.3)
        grange = (gmag >= 12) & (gmag < 17)
        w = np.where( point & grange )

        #ADM calculate the HEALPixels for the point sources
        theta, phi = np.radians(90-dec[w]), np.radians(ra[w])
        pixnums = hp.ang2pix(nside, theta, phi, nest=True)

        #ADM return the counts in each pixelnumber...
        pixnum, pixcnt = np.unique(pixnums,return_counts=True)
        #ADM...and populate the output array with the counts
        pixout[pixnum] += pixcnt

    #ADM return the density
    return pixout/pixarea
        

def pixmap(randoms, rand_density, nside=256,
           gaialoc='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """A HEALPixel map of useful quantities for analyzing a Legacy Surveys Data Release

    Parameters
    ----------
    randoms : :class:`~numpy.ndarray` or `str`
        A random catalog as made by, e.g., :func:`select_randoms()` or 
        :func:`quantities_at_positions_in_a_brick()`, or the name of such a file.
    rand_density : :class:`int`
        The number of random points per sq. deg. At which the random catalog was
        generated (see also :func:`select_randoms()`).
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel nside number) at which to build the map.
    gaialoc : :class:`str`, optional, defaults to Gaia DR2 path at NERSC
        If this is a directory, it is assumed to be the root directory of a Gaia Data 
        Release as used by the Legacy Surveys. If it is a FILE it is assumed to be a FITST 
        file that already contains the column "STARDENS", which is simply read in.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of useful information that includes
            - HPXPIXEL: HEALPixel integers at the passed `nside`
            - FRACAREA: The fraction of the pixel with at least one observation in any
                        band according to `randoms`. Made with :func:`pixweight()`
            - STARDENS: The stellar density in a pixel from Gaia. Made with
                        :func:`stellar_density()`
    """
    #ADM if a file name was passed for the random catalog, read it in
    if isinstance(randoms, str):
        log.info('Reading in random catalog...t = {:.1f}s'.format(time()-start))
        randoms = fitsio.read(randoms)

    #ADM determine the areal coverage at of the randoms at this nside
    log.info('Determining footprint...t = {:.1f}s'.format(time()-start))    
    pw = pixweight(randoms, rand_density, nside=nside)
    npix = len(pw)

    #ADM set up the output array
    hpxinfo = np.zeros(npix, dtype=[('HPXPIXEL','>i4'),('FRACAREA','>f4'),
                                    ('STARDENS','>f4')])

    #ADM add the areal coverage and pixel information to the outpu
    hpxinfo['HPXPIXEL'] = np.arange(npix)
    hpxinfo['FRACAREA'] = pw

    #ADM build the stellar density, or if gaialoc was passed as a file, just read it in
    if os.path.isdir(gaialoc):
        log.info('Calculating stellar density using Gaia files at {}...t = {:.1f}s'
                 .format(gaialoc,time()-start))    
        sd = stellar_density(nside=nside,gaiadir=gaialoc)
    else:
        sd = fitsio.read(gaialoc,columns=["STARDENS"])
        if len(sd) != len(hpxinfo):
            log.critical('Stellar density map in {} was not calculated at NSIDE={}'
                         .format(gaialoc,nside))
    hpxinfo["STARDENS"] = sd

    log.info('Done...t = {:.1f}s'.format(time()-start))    

    return hpxinfo


def bundle_bricks(pixnum, maxpernode, nside,
                  surveydir="/global/project/projectdirs/cosmo/data/legacysurvey/dr6"):
    """Determine the optimal packing for bricks collected by HEALpixel integer

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
        The HEALPixel nside number that was used to generate `pixnum`.
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
    #ADM the number of pixels (numpix) in each pixel (pix)
    pix, numpix = np.unique(pixnum,return_counts=True)

    #ADM the indices needed to reverse-sort the array on number of pixels
    reverse_order = np.flipud(np.argsort(numpix))
    numpix = numpix[reverse_order]
    pix = pix[reverse_order]

    #ADM iteratively populate lists of the numbers of pixels
    #ADM and the corrsponding pixel numbers
    bins = []

    for index, num in enumerate(numpix):
        # Try to fit this sized number into a bin
        for bin in bins:
            if np.sum(np.array(bin)[:,0]) + num <= maxpernode:
                #print 'Adding', item, 'to', bin
                bin.append([num,pix[index]])
                break
        else:
            # item didn't fit into any bin, start a new bin
            bin = []
            bin.append([num,pix[index]])
            bins.append(bin)

    #ADM print to screen in the form of a slurm bash script, and
    #ADM other useful information
    print("#######################################################")
    print("Numbers of bricks in each set of healpixels:")
    print("")
    maxeta = 0
    for bin in bins:
        num = np.array(bin)[:,0]
        pix = np.array(bin)[:,1]
        wpix = np.where(num > 0)[0]
        if len(wpix) > 0:
            goodpix, goodnum = pix[wpix], num[wpix]
            sorter = goodpix.argsort()
            goodpix, goodnum = goodpix[sorter], goodnum[sorter]
            outnote = ['{}: {}'.format(pix,num) for pix,num in zip(goodpix,goodnum)]
            #ADM add the total across all of the pixels
            outnote.append('Total: {}'.format(np.sum(goodnum)))
            #ADM a crude estimate of how long the script will take to run
            #ADM the float number is bricks/sec with some margin for writing to disk
            eta = np.sum(goodnum)/2.3/3600 
            outnote.append('Estimated time to run in hours (for 32 processors per node): {:.2f}h'
                           .format(eta))
            #ADM track the maximum estimated time for shell scripts, etc.
            if eta.astype(int) + 1 > maxeta:
                maxeta = eta.astype(int) + 1
            print(outnote)

    print("")
    print("#######################################################")
    print("Possible salloc command if you want to run on the interactive queue:")
    print("")
    print("salloc -N {} -C haswell -t 0{}:00:00 --qos interactive -L SCRATCH,project"
          .format(len(bins),maxeta))

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

    #ADM extract the Data Release number from the survey directory
    dr = surveydir.split('dr')[-1][0]

    outfiles = []
    for bin in bins:
        num = np.array(bin)[:,0]
        pix = np.array(bin)[:,1]
        wpix = np.where(num > 0)[0]
        if len(wpix) > 0:
            goodpix = pix[wpix]
            goodpix.sort()
            strgoodpix = ",".join([str(pix) for pix in goodpix])
            outfile = "$CSCRATCH/randoms-dr{}-hp-{}.fits".format(dr,strgoodpix)
            outfiles.append(outfile)
            print("srun -N 1 select_randoms {} {} --numproc 32 --nside {} --healpixels {} &"
                  .format(surveydir,outfile,nside,strgoodpix))
    print("wait")
    print("")
    print("gather_targets '{}' $CSCRATCH/randoms-dr{}.fits randoms".format(";".join(outfiles),dr))
    print("")

    return


def select_randoms(density=100000, numproc=32, nside=4, pixlist=None, bundlebricks=None,
                   drdir="/global/project/projectdirs/cosmo/data/legacysurvey/dr4/"):
    """NOBS, GALDEPTH, PSFDEPTH (per-band) for random points in a DR of the Legacy Surveys

    Parameters
    ----------
    density : :class:`int`, optional, defaults to 100,000
        The number of random points to return per sq. deg. As a typical brick is 
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned
    numproc : :class:`int`, optional, defaults to 32
        The number of processes over which to parallelize
    nside : :class:`int`, optional, defaults to nside=4 (214.86 sq. deg.)
        The HEALPixel nside number to be used with the `pixlist` and `bundlebricks` input.
    pixlist : :class:`list` or `int`, optional, defaults to None
        Bricks will only be processed if the CENTER of the brick lies within the bounds of
        pixels that are in this list of integers, at the supplied HEALPixel `nside`.
        Uses the HEALPix NESTED scheme. Useful for parallelizing. If pixlist is None
        then all bricks in the passed `survey` will be processed.
    bundlebricks : :class:`int`, defaults to None
        If not None, then instead of selecting the skies, print, to screen, the slurm
        script that will approximately balance the brick distribution at `bundlebricks`
        bricks per node. So, for instance, if bundlebricks is 14000 (which as of
        the latest git push works well to fit on the interactive nodes on Cori and run
        in about an hour), then commands would be returned with the correct pixlist values
        to pass to the code to pack at about 14000 bricks per node across all of the bricks
        in `survey`.
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
    bricknames = brickinfo['brickname']

    #ADM if the pixlist or bundlebricks option was sent, we'll need the HEALPixel
    #ADM information for each brick
    if pixlist is not None or bundlebricks is not None:
        theta, phi = np.radians(90-brickinfo["dec"]), np.radians(brickinfo["ra"])
        pixnum = hp.ang2pix(nside, theta, phi, nest=True)

    #ADM if the bundlebricks option was sent, call the packing code
    if bundlebricks is not None:
        bundle_bricks(pixnum, bundlebricks, nside, surveydir=drdir)
        return

    #ADM restrict to only bricks in a set of HEALPixels, if requested
    if pixlist is not None:
        #ADM if an integer was passed, turn it into a list
        if isinstance(pixlist,int):
            pixlist = [pixlist]
        wbricks = np.where([ pix in pixlist for pix in pixnum ])[0]
        bricknames = bricknames[wbricks]
        if len(wbricks) == 0:
            log.warning('ZERO bricks in passed pixel list!!!')
        log.info("Processing bricks in (nside={}, pixel numbers={}) HEALPixels"
                 .format(nside,pixlist))

    nbricks = len(bricknames)
    log.info('Processing {} bricks from DR at {} at density {:.1e} per sq. deg...t = {:.1f}s'
             .format(nbricks,drdir,density,time()-start))

    #ADM a little more information if we're slurming across nodes
    if os.getenv('SLURMD_NODENAME') is not None:
        print('Running on Node {}'.format(os.getenv('SLURMD_NODENAME')))

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
            #ADM if we're going to exceed 4 hours, warn the user
            if nbricks/rate > 4*3600.:
                log.error("May take > 4 hours to run. Try running with bundlebricks instead.")

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
