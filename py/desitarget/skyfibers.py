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

def sky_fibers_for_brick(survey, brickname, bands=['g','r','z'],
                         apertures_arcsec=[0.5, 0.75, 1., 1.5, 2., 3.5, 5., 7.]):
    '''Produce DESI sky fiber locations in a brick, derived at the pixel-level

    Parameters
    ----------
    survey : :class:`object`
        LegacySurveyData object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    brickname : :class:`str`
        Name of the brick in which to generate sky locations
    bands : :class:`list`, optional, defaults to ['g','r','z']
        List of bands to be used to define good sky locations
    apertures_arcsec : :class:`list`
        Radii in arcsec of apertures to sink and derive flux at a sky location

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
        - Initial version written by Dustin Lang (@dstndstn)

    '''

    fn = survey.find_file('blobmap', brick=brickname)
    blobs = fitsio.read(fn)
    print('Blobs:', blobs.min(), blobs.max())
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

    x,y,blobdist = sky_fiber_locations(goodpix)

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

    # Now, do aperture photometry at these points in the coadd images
    for band in bands:
        imfn = survey.find_file('image',  brick=brickname, band=band)
        ivfn = survey.find_file('invvar', brick=brickname, band=band)
        if not (os.path.exists(imfn) and os.path.exists(ivfn)):
            continue
        coimg = fitsio.read(imfn)
        coiv = fitsio.read(ivfn)

        apflux = np.zeros((len(skyfibers), len(apertures)), np.float32)
        apiv   = np.zeros((len(skyfibers), len(apertures)), np.float32)
        skyfibers.set('apflux_%s' % band, apflux)
        skyfibers.set('apflux_ivar_%s' % band, apiv)
        with np.errstate(divide='ignore', invalid='ignore'):
            imsigma = 1./np.sqrt(coiv)
            imsigma[coiv == 0] = 0
        apxy = np.vstack((skyfibers.x, skyfibers.y)).T
        for irad,rad in enumerate(apertures):
            aper = photutils.CircularAperture(apxy, rad)
            p = photutils.aperture_photometry(coimg, aper, error=imsigma)
            apflux[:,irad] = p.field('aperture_sum')
            err = p.field('aperture_sum_err')
            apiv  [:,irad] = 1. / err**2

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
        NxN boolean array of pixels
    gridsize : :class:`int`, optional, defaults to 300
        Resolution (in pixels) at which to split the `skypix` array in order to
        find sky locations. For example, if skypix is a 3600x3600 array of pixels, 
        gridsize=300 will return (3600/300) x (3600/300) = 12x12 = 144 locations

    Notes
    -----
    Implements the core trick of iteratively eroding the map of good sky locations to
    produce a distance-from-blobs map, and then return the max values in that map in each 
    cell of a grid.
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
        print('After erosion,', np.sum(skypix), 'sky pixels')
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
        Name of the brick from this DR of the Legacy Surveys to plot as an image
    skyfibers : :class:`object`
        `skyfibers` object returned by :func:`sky_fibers_for_brick()`
    basefn : :class:`str`
        Base name for the output plot files    
    bands : :class:`list`, optional, defaults to ['g','r','z']
        List of bands to plot in the image (i.e. default is to plot a 3-color grz composite)
    
    Returns
    -------
    Nothing, but plots are written to:
        - basefn + '-1.png' : Sky Fiber Positions on the full image
        - basefn + '-2.png' : Postage stamps around each sky fiber position
        - basefn + '-3.png' : Aperture flux at each radius for each sky fiber

    Notes
    -----
    Implements the core trick of iteratively eroding the map of good sky locations to
    produce a distance-from-blobs map, and then return the max values in that map in each 
    cell of a grid.
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

