# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=====================
desitarget.skyfibers
=====================

Module dealing with the assignation of sky fibers at the pixel-level for target selection
"""
import os
import numpy as np
import fitsio
from astropy.wcs import WCS
from time import time
import photutils

#ADM some utility code taken from legacypipe and astrometry.net
from desitarget.skyutilities.astrometry.fits import fits_table
from desitarget.skyutilities.legacypipe.util import find_unique_pixels

from desitarget.targetmask import desi_mask, targetid_mask
from desitarget.targets import encode_targetid, finalize

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

#ADM this is an empty array of the full TS data model columns and dtypes for the skies
skydatamodel = np.array([], dtype=[
    ('RELEASE', '>i4'), ('BRICKID', '>i4'), ('BRICKNAME', 'S8'),
    ('OBJID', '<i4'), ('RA', '>f8'), ('DEC', '>f8'), 
    ('APFLUX_G', '>f4'), ('APFLUX_R', '>f4'), ('APFLUX_Z', '>f4'),
    ('APFLUX_IVAR_G', '>f4'), ('APFLUX_IVAR_R', '>f4'), ('APFLUX_IVAR_Z', '>f4'),
    ])

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
    nskies = margin*options/patrol_radius

    return nskies


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
    nskies = margin*fracsky*nfibers

    return nskies


def make_sky_targets_for_brick(survey, brickname, nskiespersqdeg=None, bands=['g','r','z'],
                               apertures_arcsec=[0.75,1.0],badskyflux=[1000.,1000.]):
    """Generate sky targets and record them in the typical format for DESI sky targets

    Parameters
    ----------
    survey : :class:`object`
        LegacySurveyData object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    brickname : :class:`str`
        Name of the brick in which to generate sky locations.
    nskiespersqdeg : :class:`float`, optional, defaults to reading from desimodel.io
        The minimum DENSITY of sky fibers to generate
    bands : :class:`list`, optional, defaults to ['g','r','z']
        List of bands to be used to define good sky locations.
    apertures_arcsec : :class:`list`, optional, defaults to [0.75,1.0]
        Radii in arcsec of apertures to sink and derive flux at a sky location.
    badskyflux : :class:`list` or `~numpy.array`, optional, defaults to [1000.,1000.]
        The flux level used to classify a sky position as "BAD" in nanomaggies in
        ANY band for each aperture size. The default corresponds to a magnitude of 15.
        Must have the same length as `apertures_arcsec`.

    Returns
    -------
    :class:`~numpy.ndarray`
        a structured array of good and bad sky positions in the DESI sky target format

    Notes
    -----
    The code generates unique OBJIDs based on an integer counter for the numbers of
    objects (objs) passed. It will therefore fail if the length of objs is longer
    than the number of bits reserved for OBJID in `desitarget.targetmask`
    """
    #ADM if needed, determine the minimum density of sky fibers to generate
    if nskiespersqdeg is None:
        nskiespersqdeg = density_of_sky_fibers(margin=2)

    #ADM the hard-coded size of a DESI brick expressed as an area
    #ADM this is actually slightly larger than the largest brick size
    #ADM which would be 0.25x0.25 at the equator
    area = 0.25*0.25

    #ADM the number of sky fibers to be generated. Must be a square number
    nskiesfloat = area*nskiespersqdeg
    nskies = (np.sqrt(nskiesfloat).astype('int16') + 1)**2
    log.info('Generating {} sky positions iin brick {}...t = {:.1f}s'
             .format(nskies,brickname,time()-start))

    #ADM ensure the number of sky positions to be generated doesn't exceed 
    #ADM the largest possible OBJID (which is unlikely)
    if nskies > 2**targetid_mask.OBJID.nbits:
        log.fatal('{} sky locations requested in brick {}, but OBJID cannot exceed {}'
            .format(nskies,brickname,2**targetid_mask.OBJID.nbits))

    #ADM generate sky fiber information for this brick name
    skytable = sky_fibers_for_brick(survey,brickname,nskies=nskies,bands=bands,
                                     apertures_arcsec=apertures_arcsec)

    #ADM retrieve the standard sky targets data model
    dt = skydatamodel.dtype
    #ADM and update it according to how many apertures were requested
    naps = len(apertures_arcsec)
    apcolindices = np.where(['APFLUX' in colname for colname in dt.names])[0]
    desc = dt.descr
    for i in apcolindices:
        desc[i] += (naps,)
            
    #ADM set up a rec array to hold all of the output information
    skies = np.zeros(nskies, dtype=desc)

    #ADM populate the output recarray with the RA/Dec of the sky locations
    skies["RA"], skies["DEC"] = skytable.ra, skytable.dec

    #ADM create an array of target bits with the SKY information set
    desi_target = np.zeros(nskies,dtype='>i8')
    desi_target |= desi_mask.SKY
    #ADM find where the badskyflux limit is exceeded in any band
    #ADM first convert badskyflux to an array in case it wasn't passed as such
    badskyflux = np.array(badskyflux)
    #ADM now check for things that exceed the passed badskyflux limits in any band.
    #ADM Also check for things that have infinite flux errors in any bands (these 
    #ADM were typically outside of the imaging footprint, were in CCD gaps, etc.)
    wbad = np.where( np.any( (skytable.apflux_g > badskyflux) | 
                             (skytable.apflux_r > badskyflux) | 
                             (skytable.apflux_z > badskyflux) |
                             (skytable.apflux_ivar_g == float('Inf')) |
                             (skytable.apflux_ivar_r == float('Inf')) |
                             (skytable.apflux_ivar_z == float('Inf')), axis=1) )
    #ADM if these criteria were met, this is a bad sky
    if len(wbad) > 0:
        desi_target[wbad] = desi_mask.BAD_SKY

    #ADM add the aperture flux measurements
    if naps == 1:
        skies["APFLUX_G"] = np.hstack(skytable.apflux_g)
        skies["APFLUX_IVAR_G"] = np.hstack(skytable.apflux_ivar_g)
        skies["APFLUX_R"] = np.hstack(skytable.apflux_r)
        skies["APFLUX_IVAR_R"] = np.hstack(skytable.apflux_ivar_r)
        skies["APFLUX_Z"] = np.hstack(skytable.apflux_z)
        skies["APFLUX_IVAR_Z"] = np.hstack(skytable.apflux_ivar_z)
    else:
        skies["APFLUX_G"] = skytable.apflux_g
        skies["APFLUX_IVAR_G"] = skytable.apflux_ivar_g
        skies["APFLUX_R"] = skytable.apflux_r
        skies["APFLUX_IVAR_R"] = skytable.apflux_ivar_r
        skies["APFLUX_Z"] = skytable.apflux_z
        skies["APFLUX_IVAR_Z"] = skytable.apflux_ivar_z

    #ADM add the brick information for the sky targets
    skies["BRICKID"] = skytable.brickid
    skies["BRICKNAME"] = skytable.brickname

    #ADM set the data release from the Legacy Surveys DR directory
    dr = int(survey.survey_dir[-2])*1000
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


def sky_fibers_for_brick(survey, brickname, nskies=144, bands=['g','r','z'],
                         apertures_arcsec=[0.5, 0.75, 1., 1.5, 2., 3.5, 5., 7.]):
    '''Produce DESI sky fiber locations in a brick, derived at the pixel-level

    Parameters
    ----------
    survey : :class:`object`
        LegacySurveyData object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    brickname : :class:`str`
        Name of the brick in which to generate sky locations.
    nskies : :class:`float`, optional, defaults to 144 (12 x 12)
        The minimum DENSITY of sky fibers to generate
    bands : :class:`list`, optional, defaults to ['g','r','z']
        List of bands to be used to define good sky locations.
    apertures_arcsec : :class:`list`, optional, defaults to [0.5,0.75,1.,1.5,2.,3.5,5.,7.]
        Radii in arcsec of apertures to sink and derive flux at a sky location.

    Returns
    ------- 
    :class:`object`
        A FITS table that includes: 
        - the brickid 
        - the brickname
        - the x and y pixel positions of the fiber location from the blobs file
        - the distance from the nearest blob of this fiber location
        - the RA and Dec positions of the fiber location
        - the aperture flux and ivar at the passed `apertures_arcsec`

    Notes 
    -----   
        - Initial version written by Dustin Lang (@dstndstn).
    '''

    fn = survey.find_file('blobmap', brick=brickname)
    blobs = fitsio.read(fn)
    log.info('Blobs: {} {}'.format(blobs.min(), blobs.max()))
    header = fitsio.read_header(fn)
    wcs = WCS(header)

    goodpix = (blobs == -1)
    for band in bands:
        fn = survey.find_file('nexp', brick=brickname, band=band)
        if not os.path.exists(fn):
            # Skip
            continue
        nexp = fitsio.read(fn)
        goodpix[nexp == 0] = False

    # Cut to unique brick area... required since the blob map drops
    # blobs that are completely outside the brick's unique area, thus
    # those locations are not masked.
    brick = survey.get_brick_by_name(brickname)
    #ADM the width and height of the image in pixels is just the
    #ADM shape of the input blobs file
    H, W = blobs.shape
    U = find_unique_pixels(wcs, W, H, None, brick.ra1, brick.ra2,
                           brick.dec1, brick.dec2)
    goodpix[U == 0] = False
    del U

    #ADM the minimum safe grid size is the number of pixels along an
    #ADM axis divided by the number of sky locations along any axis
    gridsize = np.min(blobs.shape/np.sqrt(nskies)).astype('int16')
    log.info('Gridding at {} pixels...t = {:.1f}s'
             .format(gridsize,time()-start))

    x,y,blobdist = sky_fiber_locations(goodpix, gridsize=gridsize)

    skyfibers = fits_table()
    skyfibers.brickid = np.zeros(len(x), np.int32) + brick.brickid
    skyfibers.brickname = np.array([brickname] * len(x))
    skyfibers.x = x.astype(np.int16)
    skyfibers.y = y.astype(np.int16)
    skyfibers.blobdist = blobdist
    #ADM start at pixel 0,0 in the top-left (the numpy standard)
    skyfibers.ra,skyfibers.dec = wcs.all_pix2world(x, y, 0)

    #ADM find the pixel scale using the square root of the determinant
    #ADM of the CD matrix (and convert from degrees to arcseconds)
    pixscale = np.sqrt(np.abs(np.linalg.det(wcs.wcs.cd)))*3600.
    apertures = np.array(apertures_arcsec) / pixscale
    naps = len(apertures)

    # Now, do aperture photometry at these points in the coadd images
    for band in bands:
        imfn = survey.find_file('image',  brick=brickname, band=band)
        ivfn = survey.find_file('invvar', brick=brickname, band=band)

        #ADM set the apertures for every band regardless of whether
        #ADM the file exists, so that we get zeros for missing bands
        apflux = np.zeros((len(skyfibers), naps), np.float32)
        #ADM set any zero flux to have an infinite inverse variance
        apiv   = np.zeros((len(skyfibers), naps), np.float32) + np.float('Inf')
        skyfibers.set('apflux_%s' % band, apflux)
        skyfibers.set('apflux_ivar_%s' % band, apiv)

        if not (os.path.exists(imfn) and os.path.exists(ivfn)):
            continue

        coimg = fitsio.read(imfn)
        coiv = fitsio.read(ivfn)

        with np.errstate(divide='ignore', invalid='ignore'):
            imsigma = 1./np.sqrt(coiv)
            imsigma[coiv == 0] = 0
        apxy = np.vstack((skyfibers.x, skyfibers.y)).T
        for irad,rad in enumerate(apertures):
            aper = photutils.CircularAperture(apxy, rad)
            p = photutils.aperture_photometry(coimg, aper, error=imsigma)
            apflux[:,irad] = p.field('aperture_sum')
            err = p.field('aperture_sum_err')
            apiv[:,irad] = 1. / err**2

    header = fitsio.FITSHDR()
    for i,ap in enumerate(apertures_arcsec):
        header.add_record(dict(name='AP%i' % i, value=ap, comment='Aperture radius (arcsec)'))
    skyfibers._header = header

    return skyfibers
    

def sky_fiber_locations(skypix, gridsize=300):
    '''The core worker function for `sky_fibers_for_brick` 

    Parameters
    ----------
    skypix : :class:`~numpy.array`
        NxN boolean array of pixels.
    gridsize : :class:`int`, optional, defaults to 300
        Resolution (in pixels) at which to split the `skypix` array in order to
        find sky locations. For example, if skypix is a 3600x3600 array of pixels, 
        gridsize=300 will return (3600/300) x (3600/300) = 12x12 = 144 locations.

    Notes
    -----
        - Implements the core trick of iteratively eroding the map of good sky
          locations to produce a distance-from-blobs map, and then return the max 
          values in that map in each cell of a grid.
        - Initial version written by Dustin Lang (@dstndstn).
    '''
    # Select possible locations for sky fibers
    from scipy.ndimage.morphology import binary_dilation, binary_erosion
    from scipy.ndimage.measurements import label, find_objects, center_of_mass
    from scipy.ndimage.filters import gaussian_filter

    nerosions = np.zeros(skypix.shape, np.int16)
    nerosions += skypix
    element = np.ones((3,3), bool)
    while True:
        skypix = binary_erosion(skypix, structure=element)
        nerosions += skypix
        log.info('After erosion: {} sky pixels'.format(np.sum(skypix)))
        if not np.any(skypix.ravel()):
            break

    # This is a hack to break ties in the integer 'nerosions' map.
    nerosions = gaussian_filter(nerosions.astype(np.float32), 1.0)
    peaks = (nerosions > 1)
    H,W = skypix.shape

    # find pixels that are larger than their 8 neighbors
    peaks[1:-1, 1:-1] &= (nerosions[1:-1,1:-1] >= nerosions[0:-2,1:-1])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1,1:-1] >= nerosions[2:  ,1:-1])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1,1:-1] >= nerosions[1:-1,0:-2])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1,1:-1] >= nerosions[1:-1,2:  ])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1,1:-1] >= nerosions[0:-2,0:-2])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1,1:-1] >= nerosions[0:-2,2:  ])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1,1:-1] >= nerosions[2:  ,0:-2])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1,1:-1] >= nerosions[2:  ,2:  ])

    # Split the image into 300 x 300-pixel cells, choose the highest peak in each one
    # (note, this is ignoring the brick-to-brick margin in laying down the grid)
    sx,sy = [],[]
    xx = np.round(np.linspace(0, W, 1+np.ceil(W / gridsize))).astype(int)
    yy = np.round(np.linspace(0, H, 1+np.ceil(H / gridsize))).astype(int)
    for ylo,yhi in zip(yy, yy[1:]):
        for xlo,xhi in zip(xx, xx[1:]):
            # Find max pixel in box
            subne = nerosions[ylo:yhi, xlo:xhi]
            I = np.argmax(subne)
            # Find all pixels equal to the max and take the one closest to the center of mass.
            maxval = subne.flat[I]
            cy,cx = center_of_mass(subne == maxval)
            xg = np.arange(xhi-xlo)
            yg = np.arange(yhi-ylo)
            dd = np.exp(-((yg[:,np.newaxis] - cy)**2 + (xg[np.newaxis,:] - cx)**2))
            dd[subne != maxval] = 0
            I = np.argmax(dd.flat)
            iy,ix = np.unravel_index(I, subne.shape)
            sx.append(ix + xlo)
            sy.append(iy + ylo)

    sx = np.array(sx)
    sy = np.array(sy)
    return sx, sy, nerosions[sy,sx]


def sky_fiber_plots(survey, brickname, skyfibers, basefn, bands=['g','r','z']):
    '''The core worker function for `sky_fibers_for_brick` 

    Parameters
    ----------
    survey : :class:`object`
        LegacySurveyData object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    brickname : :class:`str`
        Name of the brick from this DR of the Legacy Surveys to plot as an image.
    skyfibers : :class:`object`
        `skyfibers` object returned by :func:`sky_fibers_for_brick()`
    basefn : :class:`str`
        Base name for the output plot files.
    bands : :class:`list`, optional, defaults to ['g','r','z']
        List of bands to plot in the image (i.e. default is to plot a 3-color grz composite).
        This is particularly useful when a Legacy Surveys image-BAND.fits file is not found,
        in which case that particular band can be redacted from the bands list.

    Returns
    -------
    Nothing, but plots are written to:
        - basefn + '-1.png' : Sky Fiber Positions on the full image
        - basefn + '-2.png' : Postage stamps around each sky fiber position
        - basefn + '-3.png' : Aperture flux at each radius for each sky fiber

    Notes
    -----
        - Initial version written by Dustin Lang (@dstndstn).
    '''    
    from desitarget.skyutilities.legacypipe.util import get_rgb
    import pylab as plt

    rgbkwargs = dict(mnmx=(-1,100.), arcsinh=1.)

    imgs = []
    for band in bands:
        fn = survey.find_file('image',  brick=brickname, band=band)
        imgs.append(fitsio.read(fn))
    rgb = get_rgb(imgs, bands, **rgbkwargs)

    ima = dict(interpolation='nearest', origin='lower')
    plt.clf()
    plt.imshow(rgb, **ima)
    plt.plot(skyfibers.x, skyfibers.y, 'o', mfc='none', mec='r',
             mew=2, ms=10)
    plt.title('Sky fiber positions')
    plt.savefig(basefn + '-1.png')

    plt.clf()
    plt.subplots_adjust(hspace=0, wspace=0)
    SZ = 25
    fig = plt.gcf()
    fh,fw = fig.get_figheight(), fig.get_figwidth()
    C = int(np.ceil(np.sqrt(len(skyfibers) * fw / fh)))
    R = int(np.ceil(len(skyfibers) / float(C)))
    k = 1
    H,W = imgs[0].shape
    for x,y in zip(skyfibers.x, skyfibers.y):
        if x < SZ or y < SZ or x >= W-SZ or y >= H-SZ:
            continue
        plt.subplot(R,C,k)
        k += 1
        plt.imshow(rgb[y-SZ:y+SZ+1, x-SZ:x+SZ+1, :], **ima)
        plt.xticks([]); plt.yticks([])
    plt.suptitle('Sky fiber locations')
    plt.savefig(basefn + '-2.png')

    plt.clf()
    ccmap = dict(z='m')
    for band in bands:
        flux = skyfibers.get('apflux_%s' % band)
        plt.plot(flux.T, color=ccmap.get(band,band), alpha=0.1)
    plt.ylim(-10,10)
    #plt.xticks(np.arange(len(apertures_arcsec)),
    #           ['%g' % ap for ap in apertures_arcsec])
    plt.xlabel('Aperture')# (arcsec radius)')
    plt.ylabel('Aperture flux (nanomaggies)')
    plt.title('Sky fiber: aperture flux')
    plt.savefig(basefn + '-3.png')


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Generates possible DESI sky fiber locations in Legacy Survey bricks')
    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('--out', '-o', default='skyfibers.fits',
                        help='Output filename')
    parser.add_argument('--plots', '-p', default=None,
                        help='Plots base filename')
    parser.add_argument('--brick', default=None,
                        help='Brick name')

    opt = parser.parse_args()
    if not opt.brick:
        parser.print_help()
        sys.exit(-1)

    from desitarget.skyutilities.legacypipe.util import LegacySurveyData
    
    survey = LegacySurveyData(survey_dir=opt.survey_dir)

    skyfibers = sky_fibers_for_brick(survey, opt.brick)
    skyfibers.writeto(opt.out, header=skyfibers._header)
    print('Wrote', opt.out)

    if opt.plots:
        import matplotlib
        matplotlib.use('Agg')
        sky_fiber_plots(survey, opt.brick, skyfibers, opt.plots)
