import os
import numpy as np

def sky_fibers_for_brick(survey, brickname, bands=['g','r','z'],
                         apertures_arcsec=[0.5, 0.75, 1., 1.5, 2., 3.5, 5., 7.]):
    '''
    Produces a table of possible DESI sky fiber locations for a given
    "brickname" (eg, "0001p000") read from the given LegacySurveyData object *survey*.
    '''
    import fitsio
    from astrometry.util.fits import fits_table
    from astrometry.util.util import Tan
    import photutils
    from legacypipe.utils import find_unique_pixels

    fn = survey.find_file('blobmap', brick=brickname)
    blobs = fitsio.read(fn)
    print('Blobs:', blobs.min(), blobs.max())
    header = fitsio.read_header(fn)
    wcs = Tan(header)

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
    H,W = wcs.shape
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
    skyfibers.ra,skyfibers.dec = wcs.pixelxy2radec(x+1, y+1)

    pixscale = wcs.pixel_scale()
    apertures = np.array(apertures_arcsec) / pixscale

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
    '''Worker function for sky_fibers_for_brick--- implements the core
    trick of iteratively eroding the map of good sky locations to
    produce a distance-from-blobs map, and then return the max values
    in that map in each cell of a grid.
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
    from legacypipe.survey import get_rgb
    import fitsio
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

    from legacypipe.survey import LegacySurveyData
    
    survey = LegacySurveyData(survey_dir=opt.survey_dir)

    skyfibers = sky_fibers_for_brick(survey, opt.brick)
    skyfibers.writeto(opt.out, header=skyfibers._header)
    print('Wrote', opt.out)

    if opt.plots:
        import matplotlib
        matplotlib.use('Agg')
        sky_fiber_plots(survey, opt.brick, skyfibers, opt.plots)

