# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
====================
desitarget.skyfibers
====================

Module dealing with the assignation of sky fibers at the pixel-level for target selection
"""
import os
import sys
import numpy as np
import fitsio
from astropy.wcs import WCS
from time import time
import photutils
import healpy as hp
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from scipy.ndimage.filters import gaussian_filter

# ADM some utility code taken from legacypipe and astrometry.net
from desitarget.skyutilities.astrometry.fits import fits_table
from desitarget.skyutilities.legacypipe.util import find_unique_pixels

from desitarget.targetmask import desi_mask, targetid_mask
from desitarget.targets import encode_targetid, finalize

# ADM the parallelization script
from desitarget.internal import sharedmem

# ADM set up the DESI default logger
from desiutil.log import get_logger

# ADM fake the matplotlib display so it doesn't die on allocated nodes.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

# ADM initialize the logger
log = get_logger()

# ADM start the clock
start = time()

# ADM this is an empty array of the full TS data model columns and dtypes for the skies
skydatamodel = np.array([], dtype=[
    ('RELEASE', '>i4'), ('BRICKID', '>i4'), ('BRICKNAME', 'S8'),
    ('OBJID', '<i4'), ('RA', '>f8'), ('DEC', '>f8'),
    ('APFLUX_G', '>f4'), ('APFLUX_R', '>f4'), ('APFLUX_Z', '>f4'),
    ('APFLUX_IVAR_G', '>f4'), ('APFLUX_IVAR_R', '>f4'), ('APFLUX_IVAR_Z', '>f4'),
    ('OBSCONDITIONS', '>i4')])


def density_of_sky_fibers(margin=1.5):
    """Use positioner patrol size to determine sky fiber density for DESI.

    Parameters
    ----------
    margin : :class:`float`, optional, defaults to 1.5
        Factor of extra sky positions to generate. So, for margin=10, 10x as
        many sky positions as the default requirements will be generated.

    Returns
    -------
    :class:`float`
        The density of sky fibers to generate in per sq. deg.
    """
    # ADM the patrol radius of a DESI positioner (in sq. deg.)
    patrol_radius = 6.4/60./60.

    # ADM hardcode the number of options per positioner
    options = 2.
    nskies = margin*options/patrol_radius

    return nskies


def model_density_of_sky_fibers(margin=1.5):
    """Use desihub products to find required density of sky fibers for DESI.

    Parameters
    ----------
    margin : :class:`float`, optional, defaults to 1.5
        Factor of extra sky positions to generate. So, for margin=10, 10x as
        many sky positions as the default requirements will be generated.

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


def make_skies_for_a_brick(survey, brickname, nskiespersqdeg=None, bands=['g', 'r', 'z'],
                           apertures_arcsec=[0.75, 1.0], badskyflux=[1000., 1000.],
                           write=False):
    """Generate skies for one brick in the typical format for DESI sky targets.

    Parameters
    ----------
    survey : :class:`object`
        `LegacySurveyData` object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    brickname : :class:`str`
        Name of the brick in which to generate sky locations.
    nskiespersqdeg : :class:`float`, optional, defaults to reading from desimodel.io
        The minimum DENSITY of sky fibers to generate.
    bands : :class:`list`, optional, defaults to ['g', 'r', 'z']
        List of bands to be used to define good sky locations.
    apertures_arcsec : :class:`list`, optional, defaults to [0.75, 1.0]
        Radii in arcsec of apertures to sink and derive flux at a sky location.
    badskyflux : :class:`list` or `~numpy.array`, optional, defaults to [1000., 1000.]
        The flux level used to classify a sky position as "BAD" in nanomaggies in
        ANY band for each aperture size. The default corresponds to a magnitude of 15.
        Must have the same length as `apertures_arcsec`.
    write : :class:`boolean`, defaults to False
        If `True`, write the skyfibers object (which is in the format of the output
        from :func:`sky_fibers_for_brick()`) to file. The file name is derived from
        the input `survey` object and is in the form:
        `%(survey.survey_dir)/metrics/%(brick).3s/skies-%(brick)s.fits.gz`
        which is returned by `survey.find_file('skies')`.

    Returns
    -------
    :class:`~numpy.ndarray`
        a structured array of sky positions in the DESI sky target format for a brick.

    Notes
    -----
    The code generates unique OBJIDs based on an integer counter for the numbers of
    objects (objs) passed. It will therefore fail if the length of objs is longer
    than the number of bits reserved for OBJID in `desitarget.targetmask`.
    """
    # ADM this is only intended to work on one brick, so die if a larger array is passed
    # ADM needs a hack on string type as Python 2 only considered bytes to be type str
    stringy = str
    if sys.version_info[0] == 2:
        # ADM is this is Python 2, redefine the string type
        stringy = basestring
    if not isinstance(brickname, stringy):
        log.fatal("Only one brick can be passed at a time!")
        raise ValueError

    # ADM if needed, determine the minimum density of sky fibers to generate
    if nskiespersqdeg is None:
        nskiespersqdeg = density_of_sky_fibers(margin=2)

    # ADM the hard-coded size of a DESI brick expressed as an area
    # ADM this is actually slightly larger than the largest brick size
    # ADM which would be 0.25x0.25 at the equator
    area = 0.25*0.25

    # ADM the number of sky fibers to be generated. Must be a square number
    nskiesfloat = area*nskiespersqdeg
    nskies = (np.sqrt(nskiesfloat).astype('int16') + 1)**2
    # log.info('Generating {} sky positions in brick {}...t = {:.1f}s'
    #         .format(nskies,brickname,time()-start))

    # ADM generate sky fiber information for this brick name
    skytable = sky_fibers_for_brick(survey, brickname, nskies=nskies, bands=bands,
                                    apertures_arcsec=apertures_arcsec)

    # ADM it's possible that a gridding could generate an unexpected
    # ADM number of sky fibers, so reset nskies based on the output
    nskies = len(skytable)

    # ADM ensure the number of sky positions that were generated doesn't exceed
    # ADM the largest possible OBJID (which is unlikely)
    if nskies > 2**targetid_mask.OBJID.nbits:
        log.fatal('{} sky locations requested in brick {}, but OBJID cannot exceed {}'
                  .format(nskies, brickname, 2**targetid_mask.OBJID.nbits))
        raise ValueError

    # ADM retrieve the standard sky targets data model
    dt = skydatamodel.dtype
    # ADM and update it according to how many apertures were requested
    naps = len(apertures_arcsec)
    apcolindices = np.where(['APFLUX' in colname for colname in dt.names])[0]
    desc = dt.descr
    for i in apcolindices:
        desc[i] += (naps,)

    # ADM set up a rec array to hold all of the output information
    skies = np.zeros(nskies, dtype=desc)

    # ADM populate the output recarray with the RA/Dec of the sky locations
    skies["RA"], skies["DEC"] = skytable.ra, skytable.dec

    # ADM create an array of target bits with the SKY information set
    desi_target = np.zeros(nskies, dtype='>i8')
    desi_target |= desi_mask.SKY

    # ADM Find locations where the fluxes are bad. First check if locations
    # ADM have infinite errors (zero ivars) or zero fluxes in BOTH of g and r
    # ADM (these are typically outside the imaging footprint, in CCD gaps, etc.).
    # ADM checking on z, too, is probably overkill, e.g.:
    # ADM https://github.com/desihub/desitarget/issues/348
    # ADM Remember that we need to test per-band as not all bands may have
    # ADM been requested as an input...
    bstracker = np.ones((nskies, naps), dtype=bool)
    if hasattr(skytable, 'apflux_g'):
        bstracker &= (skytable.apflux_g == 0) | (skytable.apflux_ivar_g == 0)
    if hasattr(skytable, 'apflux_r'):
        bstracker &= (skytable.apflux_r == 0) | (skytable.apflux_ivar_r == 0)

    # ADM ...now check for locations that exceed badskyflux limits in any band.
    # ADM Remember to make badskyflux an array in case it wasn't passed as such.
    badskyflux = np.array(badskyflux)
    if hasattr(skytable, 'apflux_g'):
        bstracker |= (skytable.apflux_g > badskyflux)
    if hasattr(skytable, 'apflux_r'):
        bstracker |= (skytable.apflux_r > badskyflux)
    if hasattr(skytable, 'apflux_z'):
        bstracker |= (skytable.apflux_z > badskyflux)

    # ADM check if this is a bad sky in any aperture, if so then set it to bad
    wbad = np.where(np.any(bstracker, axis=1))
    if len(wbad) > 0:
        desi_target[wbad] = desi_mask.BAD_SKY

    # ADM add the aperture flux measurements
    if naps == 1:
        if hasattr(skytable, 'apflux_g'):
            skies["APFLUX_G"] = np.hstack(skytable.apflux_g)
            skies["APFLUX_IVAR_G"] = np.hstack(skytable.apflux_ivar_g)
        if hasattr(skytable, 'apflux_r'):
            skies["APFLUX_R"] = np.hstack(skytable.apflux_r)
            skies["APFLUX_IVAR_R"] = np.hstack(skytable.apflux_ivar_r)
        if hasattr(skytable, 'apflux_z'):
            skies["APFLUX_Z"] = np.hstack(skytable.apflux_z)
            skies["APFLUX_IVAR_Z"] = np.hstack(skytable.apflux_ivar_z)
    else:
        if hasattr(skytable, 'apflux_g'):
            skies["APFLUX_G"] = skytable.apflux_g
            skies["APFLUX_IVAR_G"] = skytable.apflux_ivar_g
        if hasattr(skytable, 'apflux_r'):
            skies["APFLUX_R"] = skytable.apflux_r
            skies["APFLUX_IVAR_R"] = skytable.apflux_ivar_r
        if hasattr(skytable, 'apflux_z'):
            skies["APFLUX_Z"] = skytable.apflux_z
            skies["APFLUX_IVAR_Z"] = skytable.apflux_ivar_z

    # ADM add the brick information for the sky targets
    skies["BRICKID"] = skytable.brickid
    skies["BRICKNAME"] = skytable.brickname

    # ADM set the data release from the Legacy Surveys DR directory
    dr = int(survey.survey_dir.split('dr')[-1][0])*1000
    skies["RELEASE"] = dr

    # ADM set the objid (just use a sequential number as setting skies
    # ADM to 1 in the TARGETID will make these unique
    skies["OBJID"] = np.arange(nskies)

    # log.info('Finalizing target bits...t = {:.1f}s'.format(time()-start))
    # ADM add target bit columns to the output array, note that mws_target
    # ADM and bgs_target should be zeros for all sky objects
    dum = np.zeros_like(desi_target)
    skies = finalize(skies, desi_target, dum, dum, sky=1)

    if write:
        outfile = survey.find_file('skies', brick=brickname)
        log.info('Writing sky information to {}...t = {:.1f}s'
                 .format(outfile, time()-start))
        skytable.writeto(outfile, header=skytable._header)

    # log.info('Done...t = {:.1f}s'.format(time()-start))

    return skies


def sky_fibers_for_brick(survey, brickname, nskies=144, bands=['g', 'r', 'z'],
                         apertures_arcsec=[0.5, 0.75, 1., 1.5, 2., 3.5, 5., 7.]):
    """Produce DESI sky fiber locations in a brick, derived at the pixel-level

    Parameters
    ----------
    survey : :class:`object`
        `LegacySurveyData` object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    brickname : :class:`str`
        Name of the brick in which to generate sky locations.
    nskies : :class:`float`, optional, defaults to 144 (12 x 12)
        The minimum DENSITY of sky fibers to generate
    bands : :class:`list`, optional, defaults to ['g', 'r', 'z']
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
    """

    fn = survey.find_file('blobmap', brick=brickname)
    blobs = fitsio.read(fn)
    # log.info('Blob maximum value and minimum value in brick {}: {} {}'
    #         .format(brickname,blobs.min(),blobs.max()))
    header = fitsio.read_header(fn)
    wcs = WCS(header)

    goodpix = (blobs == -1)
    # ADM while looping through bands, check there's an image in
    # ADM at least one band, otherwise the blob map has no meaning
    # ADM for these bands, and aperture photometry is not possible
    onegoodband = False
    for band in bands:
        fn = survey.find_file('nexp', brick=brickname, band=band)
        if not os.path.exists(fn):
            # Skip
            continue
        nexp = fitsio.read(fn)
        goodpix[nexp == 0] = False
        onegoodband = True
    # ADM if there were no images in the passed bands, fail
    if not onegoodband:
        log.fatal('No images for passed bands: {}'.format(bands))
        raise ValueError

    # Cut to unique brick area... required since the blob map drops
    # blobs that are completely outside the brick's unique area, thus
    # those locations are not masked.
    brick = survey.get_brick_by_name(brickname)
    # ADM the width and height of the image in pixels is just the
    # ADM shape of the input blobs file
    H, W = blobs.shape
    U = find_unique_pixels(wcs, W, H, None, brick.ra1, brick.ra2,
                           brick.dec1, brick.dec2)
    goodpix[U == 0] = False
    del U

    # ADM the minimum safe grid size is the number of pixels along an
    # ADM axis divided by the number of sky locations along any axis
    gridsize = np.min(blobs.shape/np.sqrt(nskies)).astype('int16')
    # log.info('Gridding at {} pixels in brick {}...t = {:.1f}s'
    #         .format(gridsize,brickname,time()-start))
    x, y, blobdist = sky_fiber_locations(goodpix, gridsize=gridsize)

    skyfibers = fits_table()
    skyfibers.brickid = np.zeros(len(x), np.int32) + brick.brickid
    skyfibers.brickname = np.array([brickname] * len(x))
    skyfibers.x = x.astype(np.int16)
    skyfibers.y = y.astype(np.int16)
    skyfibers.blobdist = blobdist
    # ADM start at pixel 0,0 in the top-left (the numpy standard)
    skyfibers.ra, skyfibers.dec = wcs.all_pix2world(x, y, 0)

    # ADM find the pixel scale using the square root of the determinant
    # ADM of the CD matrix (and convert from degrees to arcseconds)
    pixscale = np.sqrt(np.abs(np.linalg.det(wcs.wcs.cd)))*3600.
    apertures = np.array(apertures_arcsec) / pixscale
    naps = len(apertures)

    # Now, do aperture photometry at these points in the coadd images
    for band in bands:
        imfn = survey.find_file('image',  brick=brickname, band=band)
        ivfn = survey.find_file('invvar', brick=brickname, band=band)

        # ADM set the apertures for every band regardless of whether
        # ADM the file exists, so that we get zeros for missing bands
        apflux = np.zeros((len(skyfibers), naps), np.float32)
        # ADM set any zero flux to have an infinite error (zero ivar)
        apiv = np.zeros((len(skyfibers), naps), np.float32)
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
        for irad, rad in enumerate(apertures):
            aper = photutils.CircularAperture(apxy, rad)
            p = photutils.aperture_photometry(coimg, aper, error=imsigma)
            apflux[:, irad] = p.field('aperture_sum')
            err = p.field('aperture_sum_err')
            # ADM where the error is 0, that actually means infinite error
            # ADM so, in reality, set the ivar to 0 for those cases and
            # ADM retain the true ivars where the error is non-zero
            wzero = np.where(err == 0)
            wnonzero = np.where(err > 0)
            apiv[:, irad][wnonzero] = 1./err[wnonzero]**2
            apiv[:, irad][wzero] = 0.

    header = fitsio.FITSHDR()
    for i, ap in enumerate(apertures_arcsec):
        header.add_record(dict(name='AP%i' % i, value=ap, comment='Aperture radius (arcsec)'))
    skyfibers._header = header

    return skyfibers


def sky_fiber_locations(skypix, gridsize=300):
    """The core worker function for `sky_fibers_for_brick`

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
    """
    nerosions = np.zeros(skypix.shape, np.int16)
    nerosions += skypix
    element = np.ones((3, 3), bool)
    while True:
        skypix = binary_erosion(skypix, structure=element)
        nerosions += skypix
#        log.info('After erosion: {} sky pixels'.format(np.sum(skypix)))
        if not np.any(skypix.ravel()):
            break

    # This is a hack to break ties in the integer 'nerosions' map.
    nerosions = gaussian_filter(nerosions.astype(np.float32), 1.0)
    peaks = (nerosions > 1)
    H, W = skypix.shape

    # find pixels that are larger than their 8 neighbors
    peaks[1:-1, 1:-1] &= (nerosions[1:-1, 1:-1] >= nerosions[0:-2, 1:-1])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1, 1:-1] >= nerosions[2:, 1:-1])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1, 1:-1] >= nerosions[1:-1, 0:-2])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1, 1:-1] >= nerosions[1:-1, 2:])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1, 1:-1] >= nerosions[0:-2, 0:-2])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1, 1:-1] >= nerosions[0:-2, 2:])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1, 1:-1] >= nerosions[2:, 0:-2])
    peaks[1:-1, 1:-1] &= (nerosions[1:-1, 1:-1] >= nerosions[2:, 2:])

    # Split the image into 300 x 300-pixel cells, choose the highest peak in each one
    # (note, this is ignoring the brick-to-brick margin in laying down the grid)
    sx, sy = [], []
    xx = np.round(np.linspace(0, W, 1+np.ceil(W / gridsize).astype(int))).astype(int)
    yy = np.round(np.linspace(0, H, 1+np.ceil(H / gridsize).astype(int))).astype(int)
    for ylo, yhi in zip(yy, yy[1:]):
        for xlo, xhi in zip(xx, xx[1:]):
            # Find max pixel in box
            subne = nerosions[ylo:yhi, xlo:xhi]
            I = np.argmax(subne)
            # Find all pixels equal to the max and take the one closest to the center of mass.
            maxval = subne.flat[I]
            cy, cx = center_of_mass(subne == maxval)
            xg = np.arange(xhi-xlo)
            yg = np.arange(yhi-ylo)
            dd = np.exp(-((yg[:, np.newaxis] - cy)**2 + (xg[np.newaxis, :] - cx)**2))
            dd[subne != maxval] = 0
            I = np.argmax(dd.flat)
            iy, ix = np.unravel_index(I, subne.shape)
            sx.append(ix + xlo)
            sy.append(iy + ylo)

    sx = np.array(sx)
    sy = np.array(sy)
    return sx, sy, nerosions[sy, sx]


def sky_fiber_plots(survey, brickname, skyfibers, basefn, bands=['g', 'r', 'z']):
    """Make QA plots for sky locations produced by `sky_fibers_for_brick`

    Parameters
    ----------
    survey : :class:`object`
        `LegacySurveyData` object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    brickname : :class:`str`
        Name of the brick from this DR of the Legacy Surveys to plot as an image.
    skyfibers : :class:`object`
        `skyfibers` object returned by :func:`sky_fibers_for_brick()`
    basefn : :class:`str`
        Base name for the output plot files.
    bands : :class:`list`, optional, defaults to ['g', 'r', 'z']
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
    """
    from desitarget.skyutilities.legacypipe.util import get_rgb
    import pylab as plt

    rgbkwargs = dict(mnmx=(-1, 100.), arcsinh=1.)

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
    fh, fw = fig.get_figheight(), fig.get_figwidth()
    C = int(np.ceil(np.sqrt(len(skyfibers) * fw / fh)))
    R = int(np.ceil(len(skyfibers) / float(C)))
    k = 1
    H, W = imgs[0].shape
    for x, y in zip(skyfibers.x, skyfibers.y):
        if x < SZ or y < SZ or x >= W-SZ or y >= H-SZ:
            continue
        plt.subplot(R, C, k)
        k += 1
        plt.imshow(rgb[y-SZ:y+SZ+1, x-SZ:x+SZ+1, :], **ima)
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('Sky fiber locations')
    plt.savefig(basefn + '-2.png')

    plt.clf()
    ccmap = dict(z='m')
    for band in bands:
        flux = skyfibers.get('apflux_%s' % band)
        plt.plot(flux.T, color=ccmap.get(band, band), alpha=0.1)
    plt.ylim(-10, 10)
    # plt.xticks(np.arange(len(apertures_arcsec)),
    #           ['%g' % ap for ap in apertures_arcsec])
    plt.xlabel('Aperture')  # (arcsec radius)')
    plt.ylabel('Aperture flux (nanomaggies)')
    plt.title('Sky fiber: aperture flux')
    plt.savefig(basefn + '-3.png')


def plot_good_bad_skies(survey, brickname, skies,
                        outplotdir='.', bands=['g', 'r', 'z']):
    """Plot good/bad sky locations against the background of a Legacy Surveys image

    Parameters
    ----------
    survey : :class:`object`
        `LegacySurveyData` object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    brickname : :class:`str`
        Name of the brick from this DR of the Legacy Surveys to plot as an image.
    skies : :class:`~numpy.ndarray`
        Array of sky locations and aperture fluxes, as, e.g., returned by
        :func:`make_skies_for_a_brick()` or :func:`select_skies()`
    outplotdir : :class:`str`, optional, defaults to '.'
        Output directory name to which to save the plot, passed to matplotlib's savefig
        routine. The actual plot is name outplotdir/skies-brickname-bands.png
    bands : :class:`list`, optional, defaults to ['g', 'r', 'z']
        List of bands to plot in the image (i.e. default is to plot a 3-color grz composite).
        This is particularly useful when the code fails because a Legacy Surveys
        image-BAND.fits file is not found, in which case that particular band can be
        redacted from the bands list.

    Returns
    -------
        Nothing, but a plot of the Legacy Surveys image for the Data Release corresponding
        to the `survey` object and the brick corresponding to `brickname` is written to
        `outplotname`. The plot contains the Legacy Surveys imaging with good sky locations
        plotted in green and bad sky locations in red.

    Notes
    -----
        - The array `skies` must contain at least the columns 'BRICKNAME', 'RA', 'DEC',
          and 'DESI_TARGET', but can contain multiple different values of 'BRICKNAME',
          provided that one of them corresponds to the passed `brickname`.

        - If the passed `survey` object doesn't correspond to the Data Release from which
          the passed `skies` array was derived, then the sky locations could be plotted
          at slightly incorrect positions. If the `skies` array was read from file, this
          can be checked by making survey that "DEPVER02" in the file header corresponds
          to the directory `survey.survey_dir`.
    """
    from desitarget.skyutilities.legacypipe.util import get_rgb
    import pylab as plt

    # ADM remember that fitsio reads things in as bytes, so convert to unicode.
    bricknames = skies['BRICKNAME'].astype('U')

    wbrick = np.where(bricknames == brickname)[0]
    if len(wbrick) == 0:
        log.fatal("No information for brick {} in passed skies array".format(brickname))
        raise ValueError
    else:
        log.info("Plotting sky locations on brick {}".format(brickname))

    # ADM derive the x and y pixel information for the sky fiber locations
    # ADM from the WCS of the survey blobs image
    fn = survey.find_file('blobmap', brick=brickname)
    header = fitsio.read_header(fn)
    wcs = WCS(header)
    xxx, yyy = wcs.all_world2pix(skies["RA"], skies["DEC"], 0)

    # ADM derive which of the sky fibers are BAD_SKY. The others are good.
    wbad = np.where((skies["DESI_TARGET"] & desi_mask.BAD_SKY) != 0)

    rgbkwargs = dict(mnmx=(-1, 100.), arcsinh=1.)

    # ADM find the images from the survey object and plot them
    imgs = []
    for band in bands:
        fn = survey.find_file('image',  brick=brickname, band=band)
        imgs.append(fitsio.read(fn))
    rgb = get_rgb(imgs, bands, **rgbkwargs)

    ima = dict(interpolation='nearest', origin='lower')
    plt.clf()
    plt.imshow(rgb, **ima)
    # ADM plot the good skies in green and the bad in red
    plt.plot(xxx, yyy, 'o', mfc='none', mec='g', mew=2, ms=10)
    plt.plot(xxx[wbad], yyy[wbad], 'o', mfc='none', mec='r', mew=2, ms=10)

    # ADM determine the plot title and name, and write it out
    bandstr = "".join(bands)
    plt.title('Skies for brick {} (BAD_SKY in red); bands = {}'
              .format(brickname, bandstr))
    outplotname = '{}/skies-{}-{}.png'.format(outplotdir, brickname, bandstr)
    log.info("Writing plot to {}".format(outplotname))
    plt.savefig(outplotname)


def bundle_bricks(pixnum, maxpernode, nside,
                  surveydir="/global/project/projectdirs/cosmo/data/legacysurvey/dr6"):
    """Determine the optimal packing for bricks collected by HEALpixel integer.

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
    # ADM the number of pixels (numpix) in each pixel (pix)
    pix, numpix = np.unique(pixnum, return_counts=True)

    # ADM convert the pixel numbers back to integers
    pix = pix.astype(int)

    # ADM the indices needed to reverse-sort the array on number of pixels
    reverse_order = np.flipud(np.argsort(numpix))
    numpix = numpix[reverse_order]
    pix = pix[reverse_order]

    # ADM iteratively populate lists of the numbers of pixels
    # ADM and the corrsponding pixel numbers
    bins = []

    for index, num in enumerate(numpix):
        # Try to fit this sized number into a bin
        for bin in bins:
            if np.sum(np.array(bin)[:, 0]) + num <= maxpernode:
                # print 'Adding', item, 'to', bin
                bin.append([num, pix[index]])
                break
        else:
            # item didn't fit into any bin, start a new bin
            bin = []
            bin.append([num, pix[index]])
            bins.append(bin)

    # ADM print to screen in the form of a slurm bash script, and
    # ADM other useful information

    print("#######################################################")
    print("Numbers of bricks in each set of healpixels:")
    print("")
    maxeta = 0
    for bin in bins:
        num = np.array(bin)[:, 0]
        pix = np.array(bin)[:, 1]
        wpix = np.where(num > 0)[0]
        if len(wpix) > 0:
            goodpix, goodnum = pix[wpix], num[wpix]
            sorter = goodpix.argsort()
            goodpix, goodnum = goodpix[sorter], goodnum[sorter]
            outnote = ['{}: {}'.format(pix, num) for pix, num in zip(goodpix, goodnum)]
            # ADM add the total across all of the pixels
            outnote.append('Total: {}'.format(np.sum(goodnum)))
            # ADM a crude estimate of how long the script will take to run
            # ADM brickspersec is bricks/sec
            brickspersec = 1.
            eta = np.sum(goodnum)/brickspersec/3600.
            outnote.append('Estimated time to run in hours (for 32 processors per node): {:.2f}h'
                           .format(eta))
            # ADM track the maximum estimated time for shell scripts, etc.
            if eta.astype(int) + 1 > maxeta:
                maxeta = eta.astype(int) + 1
            print(outnote)

    print("")
    print("#######################################################")
    print("Possible salloc command if you want to run on the interactive queue:")
    print("")
    print("salloc -N {} -C haswell -t 0{}:00:00 --qos interactive -L SCRATCH,project"
          .format(len(bins), maxeta))

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

    # ADM extract the Data Release number from the survey directory
    dr = surveydir.split('dr')[-1][0]

    outfiles = []
    for bin in bins:
        num = np.array(bin)[:, 0]
        pix = np.array(bin)[:, 1]
        wpix = np.where(num > 0)[0]
        if len(wpix) > 0:
            goodpix = pix[wpix]
            goodpix.sort()
            strgoodpix = ",".join([str(pix) for pix in goodpix])
            outfile = "$CSCRATCH/skies-dr{}-hp-{}.fits".format(dr, strgoodpix)
            outfiles.append(outfile)
            print("srun -N 1 select_skies {} {} --numproc 32 --nside {} --healpixels {} &"
                  .format(surveydir, outfile, nside, strgoodpix))
    print("wait")
    print("")
    print("gather_targets '{}' $CSCRATCH/skies-dr{}.fits skies".format(";".join(outfiles), dr))
    print("")

    return


def select_skies(survey, numproc=16, nskiespersqdeg=None, bands=['g', 'r', 'z'],
                 apertures_arcsec=[0.75, 1.0], badskyflux=[1000., 1000.],
                 nside=2, pixlist=None, writebricks=False, bundlebricks=None):
    """Generate skies in parallel for all bricks in a Legacy Surveys Data Release.

    Parameters
    ----------
    survey : :class:`object`
        `LegacySurveyData` object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    numproc : :class:`int`, optional, defaults to 16
        The number of processes over which to parallelize.
    nskiespersqdeg : :class:`float`, optional, defaults to reading from desimodel.io
        The minimum DENSITY of sky fibers to generate.
    bands : :class:`list`, optional, defaults to ['g', 'r', 'z']
        List of bands to be used to define good sky locations.
    apertures_arcsec : :class:`list`, optional, defaults to [0.75, 1.0]
        Radii in arcsec of apertures to sink and derive flux at a sky location.
    badskyflux : :class:`list` or `~numpy.array`, optional, defaults to [1000., 1000.]
        The flux level used to classify a sky position as "BAD" in nanomaggies in
        ANY band for each aperture size. The default corresponds to a magnitude of 15.
        Must have the same length as `apertures_arcsec`.
    nside : :class:`int`, optional, defaults to nside=2 (859.4 sq. deg.)
        The HEALPixel nside number to be used with the `pixlist` input.
    pixlist : :class:`list` or `int`, optional, defaults to None
        Bricks will only be processed if the CENTER of the brick lies within the bounds of
        pixels that are in this list of integers, at the supplied HEALPixel `nside`.
        Uses the HEALPix NESTED scheme. Useful for parallelizing. If pixlist is ``None``
        then all bricks in the passed `survey` will be processed.
    writebricks : :class:`boolean`, defaults to False
        If `True`, write the skyfibers object for EACH brick (in the format of the
        output from :func:`sky_fibers_for_brick()`) to file. The file name is derived
        from the input `survey` object and is in the form:
        `%(survey.survey_dir)/metrics/%(brick).3s/skies-%(brick)s.fits.gz`
        which is returned by `survey.find_file('skies')`.
    bundlebricks : :class:`int`, defaults to None
        If not None, then instead of selecting the skies, print, to screen, the slurm
        script that will approximately balance the brick distribution at `bundlebricks`
        bricks per node. So, for instance, if bundlebricks is 14000 (which as of
        the latest git push works well to fit on the interactive nodes on Cori), then
        commands would be returned with the correct pixlist values to pass to the code
        to pack at about 14000 bricks per node across all of the bricks in `survey`.

    Returns
    -------
    :class:`~numpy.ndarray`
        a structured array of sky positions in the DESI sky target format for all
        bricks in a Legacy Surveys Data Release.

    Notes
    -----
        - Some core code in this module was initially written by Dustin Lang (@dstndstn).
        - Returns nothing if bundlebricks is passed (and is not ``None``).
    """
    # ADM these comments were for debugging photutils/astropy dependencies
    # ADM and they can be removed at any time
#    import astropy
#    print(astropy.version)
#    print(astropy.version.version)
#    print(photutils.version)
#    print(photutils.version.version)

    # ADM read in the survey bricks file, which lists the bricks of interest for this DR
    from glob import glob
    sbfile = glob(survey.survey_dir+'/*bricks-dr*')[0]
    brickinfo = fitsio.read(sbfile)
    # ADM remember that fitsio reads things in as bytes, so convert to unicode
    bricknames = brickinfo['brickname'].astype('U')

    # ADM if the pixlist or bundlebricks option was sent, we'll need the HEALPpixel
    # ADM information for each brick
    if pixlist is not None or bundlebricks is not None:
        theta, phi = np.radians(90-brickinfo["dec"]), np.radians(brickinfo["ra"])
        pixnum = hp.ang2pix(nside, theta, phi, nest=True)

    # ADM if the bundlebricks option was sent, call the packing code
    if bundlebricks is not None:
        bundle_bricks(pixnum, bundlebricks, nside, surveydir=survey.survey_dir)
        return

    # ADM restrict to only bricks in a set of HEALPixels, if requested
    if pixlist is not None:
        # ADM if an integer was passed, turn it into a list
        if isinstance(pixlist, int):
            pixlist = [pixlist]
        wbricks = np.where([pix in pixlist for pix in pixnum])[0]
        bricknames = bricknames[wbricks]
        if len(wbricks) == 0:
            log.warning('ZERO bricks in passed pixel list!!!')
        log.info("Processing bricks in (nside={}, pixel numbers={}) HEALPixels"
                 .format(nside, pixlist))

    nbricks = len(bricknames)
    log.info('Processing {} bricks that have observations from DR at {}...t = {:.1f}s'
             .format(nbricks, survey.survey_dir, time()-start))

    # ADM a little more information if we're slurming across nodes
    if os.getenv('SLURMD_NODENAME') is not None:
        log.info('Running on Node {}'.format(os.getenv('SLURMD_NODENAME')))

    # ADM the critical function to run on every brick
    def _get_skies(brickname):
        '''wrapper on make_skies_for_a_brick() given a brick name'''

        return make_skies_for_a_brick(survey, brickname,
                                      nskiespersqdeg=nskiespersqdeg, bands=bands,
                                      apertures_arcsec=apertures_arcsec,
                                      badskyflux=badskyflux, write=writebricks)

    # ADM this is just in order to count bricks in _update_status
    nbrick = np.zeros((), dtype='i8')

    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if nbrick % 500 == 0 and nbrick > 0:
            rate = nbrick / (time() - t0)
            log.info('{}/{} bricks; {:.1f} bricks/sec'.format(nbrick, nbricks, rate))

        nbrick[...] += 1    # this is an in-place modification
        return result

    # - Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            skies = pool.map(_get_skies, bricknames, reduce=_update_status)
    else:
        skies = list()
        for brickname in bricknames:
            skies.append(_update_status(_get_skies(brickname)))

    # ADM Concatenate the parallelized results into one rec array of sky information
    skies = np.concatenate(skies)

    log.info('Done...t={:.1f}s'.format(time()-start))

    return skies
