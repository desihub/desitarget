# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
====================
desitarget.skyfibers
====================

Module to assign sky fibers at the pixel-level for target selection
"""
import os
import sys
import numpy as np
import fitsio
from astropy.wcs import WCS
from time import time
import photutils
import healpy as hp
from glob import glob
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage.measurements import label, find_objects, center_of_mass
from scipy.ndimage.filters import gaussian_filter

# ADM some utility code taken from legacypipe and astrometry.net.
from desitarget.skyutilities.astrometry.fits import fits_table
from desitarget.skyutilities.legacypipe.util import find_unique_pixels

from desitarget.targetmask import desi_mask, targetid_mask
from desitarget.targets import finalize
from desitarget import io
from desitarget.gaiamatch import find_gaia_files, gaia_dr_from_ref_cat
from desitarget.gaiamatch import get_gaia_nside_brick
from desitarget.geomask import is_in_gal_box, is_in_circle, is_in_hp

# ADM the parallelization script.
from desitarget.internal import sharedmem

from desiutil import brick
from desiutil.log import get_logger

# ADM set up the Legacy Surveys bricks object.
bricks = brick.Bricks(bricksize=0.25)

# ADM initialize the DESI logger.
log = get_logger()

# ADM start the clock
start = time()

# ADM this is an empty array of the full TS data model columns and dtypes for the skies
skydatamodel = np.array([], dtype=[
    ('RELEASE', '>i4'), ('BRICKID', '>i4'), ('BRICKNAME', 'S8'),
    ('OBJID', '<i4'), ('RA', '>f8'), ('DEC', '>f8'), ('BLOBDIST', '>f4'),
    ('FIBERFLUX_G', '>f4'), ('FIBERFLUX_R', '>f4'), ('FIBERFLUX_Z', '>f4'),
    ('FIBERFLUX_IVAR_G', '>f4'), ('FIBERFLUX_IVAR_R', '>f4'), ('FIBERFLUX_IVAR_Z', '>f4')
    ])


def get_brick_info(drdirs, counts=False, allbricks=False):
    """Retrieve brick names and coordinates from Legacy Surveys directories.

    Parameters
    ----------
    drdirs : :class:`list` or `str`
        A list of strings, each of which corresponds to a directory pointing
        to a Data Release from the Legacy Surveys. Can be of length one.
        e.g. ['/global/project/projectdirs/cosmo/data/legacysurvey/dr7'].
        or '/global/project/projectdirs/cosmo/data/legacysurvey/dr7'
        Can be None if `allbricks` is passed.
    counts : :class:`bool`, optional, defaults to ``False``
        If ``True`` also return a count of the number of times each brick
        appears ([RAcen, DECcen, RAmin, RAmax, DECmin, DECmax, CNT]).
    allbricks : :class:`bool`, optional, defaults to ``False``
        If ``True`` ignore `drdirs` and simply return a dictionary of ALL
        of the bricks.

    Returns
    -------
    :class:`dict`
        UNIQUE bricks covered by the Data Release(s). Keys are brick names
        and values are a list of the brick center and the brick corners
        ([RAcen, DECcen, RAmin, RAmax, DECmin, DECmax]).

    Notes
    -----
        - Tries a few different ways in case the survey bricks files have
          not yet been created.
    """
    # ADM convert a single input string to a list.
    if isinstance(drdirs, str):
        drdirs = [drdirs, ]

    # ADM turn the brick info table into a fast look-up dictionary.
    # ADM (note the bricks class is instantiated at the top of the code.)
    bricktable = bricks.to_table()
    brickdict = {}
    for b in bricktable:
        brickdict[b["BRICKNAME"]] = [b["RA"], b["DEC"],
                                     b["RA1"], b["RA2"],
                                     b["DEC1"], b["DEC2"]]

    # ADM if requested, return the dictionary of ALL bricks.
    if allbricks:
        return brickdict

    bricknames = []
    for dd in drdirs:
        # ADM in the simplest case, read in the survey bricks file, which lists
        # ADM the bricks of interest for this DR.
        sbfile = glob(dd+'/*bricks-dr*')
        if len(sbfile) > 0:
            brickinfo = fitsio.read(sbfile[0], upper=True)
            # ADM fitsio reads things in as bytes, so convert to unicode.
            bricknames.append(brickinfo['BRICKNAME'].astype('U'))
        else:
            # ADM hack for test bricks where we don't generate the bricks file.
            fns = glob(os.path.join(dd, 'tractor', '*', '*fits'))
            bricknames.append([io.brickname_from_filename(fn)
                               for fn in fns])

    # ADM don't count bricks twice, but record number of duplicate bricks.
    bricknames, cnts = np.unique(np.concatenate(bricknames), return_counts=True)

    # ADM only return the subset of the dictionary with bricks in the DR.
    if counts:
        return {bn: brickdict[bn] + [cnt] for bn, cnt in zip(bricknames, cnts)}
    return {bn: brickdict[bn] for bn in bricknames}


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
                           apertures_arcsec=[0.75], write=False):
    """Generate skies for one brick in the typical format for DESI sky targets.

    Parameters
    ----------
    survey : :class:`object`
        `LegacySurveyData` object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    brickname : :class:`str`
        Name of the brick in which to generate sky locations.
    nskiespersqdeg : :class:`float`, optional
        The minimum DENSITY of sky fibers to generate. Defaults to reading from
        :func:`~desimodel.io` with a margin of 4x.
    bands : :class:`list`, optional, defaults to ['g', 'r', 'z']
        List of bands to be used to define good sky locations.
    apertures_arcsec : :class:`list`, optional, defaults to [0.75]
        Radii in arcsec of apertures for which to derive flux at a sky location.
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
        - The code generates unique OBJIDs based on an integer counter
          for the numbers of objects (objs) passed. So, it will fail if
          the length of objs is longer than the number of bits reserved
          for OBJID in `desitarget.targetmask`.
        - The generated sky fiber locations will cover the pixel-based brick
          grid, which extends beyond the "true" geometric brick boundaries.
    """
    # ADM this is only intended to work on one brick, so die if a larger array is passed
    # ADM needs a hack on string type as Python 2 only considered bytes to be type str.
    stringy = str
    if sys.version_info[0] == 2:
        # ADM is this is Python 2, redefine the string type.
        stringy = basestring
    if not isinstance(brickname, stringy):
        log.fatal("Only one brick can be passed at a time!")
        raise ValueError

    # ADM if needed, determine the minimum density of sky fibers to generate.
    if nskiespersqdeg is None:
        nskiespersqdeg = density_of_sky_fibers(margin=4)

    # ADM the hard-coded size of a DESI brick expressed as an area
    # ADM this is actually slightly larger than the largest brick size
    # ADM which would be 0.25x0.25 at the equator.
    area = 0.25*0.25

    # ADM the number of sky fibers to be generated. Must be a square number.
    nskiesfloat = area*nskiespersqdeg
    nskies = (np.sqrt(nskiesfloat).astype('int16') + 1)**2
    # log.info('Generating {} sky positions in brick {}...t = {:.1f}s'
    #         .format(nskies,brickname,time()-start))

    # ADM generate sky fiber information for this brick name.
    skytable = sky_fibers_for_brick(survey, brickname, nskies=nskies, bands=bands,
                                    apertures_arcsec=apertures_arcsec)
    # ADM if the blob file doesn't exist, skip it.
    if skytable is None:
        return None

    # ADM it's possible that a gridding could generate an unexpected
    # ADM number of sky fibers, so reset nskies based on the output.
    nskies = len(skytable)

    # ADM ensure the number of sky positions that were generated doesn't exceed
    # ADM the largest possible OBJID (which is unlikely).
    if nskies > 2**targetid_mask.OBJID.nbits:
        log.fatal('{} sky locations requested in brick {}, but OBJID cannot exceed {}'
                  .format(nskies, brickname, 2**targetid_mask.OBJID.nbits))
        raise ValueError

    # ADM retrieve the standard sky targets data model.
    dt = skydatamodel.dtype
    # ADM and update it according to how many apertures were requested.
    naps = len(apertures_arcsec)
    apcolindices = np.where(['FIBERFLUX' in colname for colname in dt.names])[0]
    desc = dt.descr
    if naps > 1:
        for i in apcolindices:
            desc[i] += (naps,)

    # ADM set up a rec array to hold all of the output information.
    skies = np.zeros(nskies, dtype=desc)

    # ADM populate the output recarray with the RA/Dec of the sky locations.
    skies["RA"], skies["DEC"] = skytable.ra, skytable.dec

    # ADM create an array of target bits with the SKY information set.
    desi_target = np.zeros(nskies, dtype='>i8')
    desi_target |= desi_mask.SKY

    # ADM Find where the fluxes are potentially bad. First check if locations
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

    # ADM as BLOBDIST doesn't depend on the aperture, collapse across apertures.
    bstracker = np.any(bstracker, axis=1)

    # ADM ...now check for BADSKY locations that are in a blob.
    if hasattr(skytable, 'blobdist'):
        bstracker |= (skytable.blobdist == 0.)

    # ADM set any bad skies to BADSKY.
    desi_target[bstracker] = desi_mask.BAD_SKY

    # ADM add the aperture flux measurements.
    if naps == 1:
        if hasattr(skytable, 'apflux_g'):
            skies["FIBERFLUX_G"] = np.hstack(skytable.apflux_g)
            skies["FIBERFLUX_IVAR_G"] = np.hstack(skytable.apflux_ivar_g)
        if hasattr(skytable, 'apflux_r'):
            skies["FIBERFLUX_R"] = np.hstack(skytable.apflux_r)
            skies["FIBERFLUX_IVAR_R"] = np.hstack(skytable.apflux_ivar_r)
        if hasattr(skytable, 'apflux_z'):
            skies["FIBERFLUX_Z"] = np.hstack(skytable.apflux_z)
            skies["FIBERFLUX_IVAR_Z"] = np.hstack(skytable.apflux_ivar_z)
    else:
        if hasattr(skytable, 'apflux_g'):
            skies["FIBERFLUX_G"] = skytable.apflux_g
            skies["FIBERFLUX_IVAR_G"] = skytable.apflux_ivar_g
        if hasattr(skytable, 'apflux_r'):
            skies["FIBERFLUX_R"] = skytable.apflux_r
            skies["FIBERFLUX_IVAR_R"] = skytable.apflux_ivar_r
        if hasattr(skytable, 'apflux_z'):
            skies["FIBERFLUX_Z"] = skytable.apflux_z
            skies["FIBERFLUX_IVAR_Z"] = skytable.apflux_ivar_z

    # ADM add the brick info and blob distance for the sky targets.
    skies["BRICKID"] = skytable.brickid
    skies["BRICKNAME"] = skytable.brickname
    skies["BLOBDIST"] = skytable.blobdist

    # ADM set the data release from an object in a Tractor file.
    tfn = survey.find_file("tractor", brick=brickname)
    # ADM this file should be guaranteed to exist, except for unit tests.
    if os.path.exists(tfn):
        skies["RELEASE"] = fitsio.read(tfn, rows=0, columns='RELEASE')[0]

    # ADM set the objid (just use a sequential number as setting skies
    # ADM to 1 in the TARGETID will make these unique.
    skies["OBJID"] = np.arange(nskies)

    # log.info('Finalizing target bits...t = {:.1f}s'.format(time()-start))
    # ADM add target bit columns to the output array, note that mws_target
    # ADM and bgs_target should be zeros for all sky objects.
    dum = np.zeros_like(desi_target)
    skies = finalize(skies, desi_target, dum, dum, sky=True)

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
        Radii in arcsec of apertures for which to derive flux at a sky location.

    Returns
    -------
    :class:`object`
        A FITS table that includes:
        - the brickid
        - the brickname
        - the x and y pixel positions of the fiber location from the blobs file
        - the distance to the nearest blob at this fiber location
        - the RA and Dec positions of the fiber location
        - the aperture flux and ivar at the passed `apertures_arcsec`

    Notes
    -----
        - Initial version written by Dustin Lang (@dstndstn).
        - The generated sky fiber locations will cover the pixel-based brick
          grid, which extends beyond the "true" geometric brick boundaries.
    """

    fn = survey.find_file('blobmap', brick=brickname)
    # ADM if the file doesn't exist, warn and return immediately.
    if not os.path.exists(fn):
        log.warning('blobmap {} does not exist!!!'.format(fn))
        return None
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
    # ADM axis divided by the number of sky locations along any axis.
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
    # ADM start at pixel 0,0 in the top-left (the numpy standard).
    skyfibers.ra, skyfibers.dec = wcs.all_pix2world(x, y, 0)

    # ADM find the pixel scale using the square root of the determinant
    # ADM of the CD matrix (and convert from degrees to arcseconds).
    pixscale = np.sqrt(np.abs(np.linalg.det(wcs.wcs.cd)))*3600.
    apertures = np.array(apertures_arcsec) / pixscale
    naps = len(apertures)

    # Now, do aperture photometry at these points in the coadd images.
    for band in bands:
        imfn = survey.find_file('image',  brick=brickname, band=band)
        ivfn = survey.find_file('invvar', brick=brickname, band=band)

        # ADM set the apertures for every band regardless of whether
        # ADM the file exists, so that we get zeros for missing bands.
        apflux = np.zeros((len(skyfibers), naps), np.float32)
        # ADM set any zero flux to have an infinite error (zero ivar).
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
            # ADM retain the true ivars where the error is non-zero.
            # ADM also catch the occasional NaN (which are very rare).
            ii = np.isnan(err)
            err[ii] = 0.0
            wzero = np.where(err == 0)
            wnonzero = np.where(err > 0)
            apiv[:, irad][wnonzero] = 1./err[wnonzero]**2
            apiv[:, irad][wzero] = 0.

    header = fitsio.FITSHDR()
    for i, ap in enumerate(apertures_arcsec):
        header.add_record(dict(name='AP%i' % i, value=ap,
                               comment='Aperture radius (arcsec)'))
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
    # ADM from the WCS of the survey blobs image.
    fn = survey.find_file('blobmap', brick=brickname)
    header = fitsio.read_header(fn)
    wcs = WCS(header)
    xxx, yyy = wcs.all_world2pix(skies["RA"], skies["DEC"], 0)

    # ADM derive which of the sky fibers are BAD_SKY. The others are good.
    wbad = np.where((skies["DESI_TARGET"] & desi_mask.BAD_SKY) != 0)

    # ADM find the images from the survey object and plot them.
    imgs = []
    for band in bands:
        fn = survey.find_file('image',  brick=brickname, band=band)
        imgs.append(fitsio.read(fn))

    rgbkwargs = dict(mnmx=(-1, 100.), arcsinh=1.)
    rgb = get_rgb(imgs, bands, **rgbkwargs)
    # ADM hack to make sure rgb is never negative.
    rgb[rgb < 0] = 0

    ima = dict(interpolation='nearest', origin='lower')
    plt.clf()
    plt.imshow(rgb, **ima)
    # ADM plot the good skies in green and the bad in red.
    plt.plot(xxx, yyy, 'o', mfc='none', mec='g', mew=2, ms=10)
    plt.plot(xxx[wbad], yyy[wbad], 'o', mfc='none', mec='r', mew=2, ms=10)

    # ADM determine the plot title and name, and write it out.
    bandstr = "".join(bands)
    plt.title('Skies for brick {} (BAD_SKY in red); bands = {}'
              .format(brickname, bandstr))
    outplotname = '{}/skies-{}-{}.png'.format(outplotdir, brickname, bandstr)
    log.info("Writing plot to {}".format(outplotname))
    plt.savefig(outplotname)


def repartition_skies(skydirname, numproc=1):
    """Rewrite a skies directory so each file actually only contains sky
    locations in the HEALPixels that are listed in the file header.

    Parameters
    ----------
    skydirname : :class:`str`
        Full path to a directory containing files of skies that have been
        partitioned by HEALPixel (i.e. as made by `select_skies` with the
        `bundle_files` option).
    numproc : :class:`int`, optional, defaults to 1
        The number of processes over which to parallelize writing files.

    Returns
    -------
    Nothing, but rewrites the input directory such that each file only
    contains the HEALPixels listed in the file header.

    Notes
    -----
        - Necessary as although the targets and GFAs are parallelized
          to run in exact HEALPixel boundaries, skies are parallelized
          across bricks that have CENTERS in a given HEALPixel.
        - The original files, before the rewrite, are retained in a
          directory called "unpartitioned" in the original directory. The
          file names are appended by "-unpartitioned".
        - Takes about 25 (6.5, 5, 3.5) minutes for numproc=1 (8, 16, 32).
    """
    log.info("running on {} processors".format(numproc))

    # ADM grab the typical file header in the passed directory.
    hdr = io.read_targets_header(skydirname)
    # ADM remove the "standard" FITS header keys.
    protected = ["TFORM", "TTYPE", "EXTNAME", "XTENSION", "BITPIX", "NAXIS",
                 "PCOUNT", "GCOUNT", "TFIELDS"]
    hdrdict = {key: hdr[key] for key in hdr.keys()
               if not np.any([prot in key for prot in protected])}

    # ADM grab the typical nside for files in the passed directory.
    nside = hdr["FILENSID"]
    hps = np.arange(hp.nside2npix(nside))

    # ADM grab the Data Release number for files in the passed directory.
    depdict = {k: v for k, v in zip(
        [hdr[key].rstrip() for key in hdr if 'DEPNAM' in key],
        [hdr[key].rstrip() for key in hdr if 'DEPVER' in key])}
    drint = depdict['photcat'].lstrip("dr")

    # ADM each element of this array will be a HEALPixel, each HEALPixel
    # ADM will contain a dictionary with file names as keys and arrays of
    # ADM which rows of the file are in the HEALPixel as values.
    pixorderdict = [{} for pix in hps]

    # ADM make the "unpartioned" directory if it doesn't exist.
    os.makedirs(os.path.join(skydirname, "unpartitioned"), exist_ok=True)
    # ADM loop over the files in the sky directory and build the info.
    skyfiles = glob(os.path.join(skydirname, '*fits'))
    for skyfile in skyfiles:
        # ADM rename the sky file so as not to overwrite.
        sfnewname = os.path.join(os.path.dirname(skyfile), "unpartitioned",
                                 os.path.basename(skyfile) + "-unpartitioned")
        os.rename(skyfile, sfnewname)
        data, hdr = io.read_target_files(sfnewname, columns=["RA", "DEC"],
                                         header=True, verbose=False)
        theta, phi = np.radians(90-data["DEC"]), np.radians(data["RA"])
        pixinfile = hp.ang2pix(nside, theta, phi, nest=hdr["FILENEST"])
        spixinfile = set(pixinfile)
        for pix in spixinfile:
            pixorderdict[pix][sfnewname] = np.where(pixinfile == pix)[0]
        log.info("Read from (file NOW called) {}...t={:.1f}s".format(
            sfnewname, time()-start))

    def _write_hp_skies(healpixels):
        """Use pixorderdict to write files partitioned by HEALPixel.
        """
        for pix in healpixels:
            # ADM copy the header to avoid race conditions.
            newhdr = hdrdict.copy()
            skies = []
            if len(pixorderdict[pix]) > 0:
                for fn in pixorderdict[pix]:
                    skies.append(fitsio.read(fn, rows=pixorderdict[pix][fn]))
                skies = np.concatenate(skies)
                # ADM the header entry corresponding to the pixel number
                # ADM needs to be updated.
                newhdr['FILEHPX'] = pix

                # ADM get the appropriate full file name.
                outfile = io.find_target_files(skydirname, drint, flavor="skies",
                                               hp=pix, nside=nside)
                # ADM only need the file (we know the right directory).
                outfile = os.path.join(skydirname, os.path.basename(outfile))
                fitsio.write(outfile+'.tmp', skies, extname='SKY_TARGETS',
                             header=newhdr, clobber=True)
                os.rename(outfile+'.tmp', outfile)
                log.info('{} skies written to {}...t={:.1f}s'.format(
                    len(skies), outfile, time()-start))
        return

    # ADM split the pixels up into blocks of arrays to parallelize.
    hpsplit = np.array_split(hps, numproc)

    # ADM Parallel process writing of HEALPixel-partitioned files.
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            skies = pool.map(_write_hp_skies, hpsplit)
    else:
        _write_hp_skies(hpsplit[0])

    return


def get_supp_skies(ras, decs, radius=2.):
    """Random locations, avoid Gaia, format, return supplemental skies.

    Parameters
    ----------
    ras : :class:`~numpy.ndarray`
        Right Ascensions of sky locations (degrees).
    decs : :class:`~numpy.ndarray`
        Declinations of sky locations (degrees).
    radius : :class:`float`, optional, defaults to 2
        Radius at which to avoid (all) Gaia sources (arcseconds).

    Returns
    -------
    :class:`~numpy.ndarray`
        A structured array of supplemental sky positions in the DESI sky
        target format that avoid Gaia sources by `radius`.

    Notes
    -----
        - Written to be used when `ras` and `decs` are within a single
          Gaia-file HEALPixel, but should work for all cases.
    """
    # ADM determine Gaia files of interest and read the RAs/Decs.
    fns = find_gaia_files([ras, decs], neighbors=True, radec=True)
    gobjs = np.concatenate(
        [fitsio.read(fn, columns=["RA", "DEC"]) for fn in fns])

    # ADM convert radius to an array.
    r = np.zeros(len(gobjs))+radius

    # ADM determine matches between Gaia and the passed RAs/Decs.
    isin = is_in_circle(ras, decs, gobjs["RA"], gobjs["DEC"], r)
    good = ~isin

    # ADM build the output array from the sky targets data model.
    nskies = np.sum(good)
    supsky = np.zeros(nskies, dtype=skydatamodel.dtype)
    # ADM populate output array with the RA/Dec of the sky locations.
    supsky["RA"], supsky["DEC"] = ras[good], decs[good]
    # ADM add the brickid and name. Use Gaia at NSIDE=256, which roughly
    # ADM corresponds to the area of a brick. Using Gaia pixel numbers
    # ADM instead of bricks is necessary to avoid duplicate TARGETIDs as
    # ADM we parallelize supp_skies across pixels, not bricks.
    nside = get_gaia_nside_brick()
    theta, phi = np.radians(90-decs[good]), np.radians(ras[good])
    supsky["BRICKID"] = hp.ang2pix(nside, theta, phi, nest=True)
    supsky["BRICKNAME"] = 'hpxat{}'.format(nside)
    # ADM BLOBDIST is in ~Legacy Surveys pixels, with scale 0.262 "/pix.
    supsky["BLOBDIST"] = radius/0.262
    # ADM set all fluxes and IVARs to -1, so they're ill-defined.
    for name in skydatamodel.dtype.names:
        if "FLUX" in name:
            supsky[name] = -1.

    return supsky


def supplement_skies(nskiespersqdeg=None, numproc=16, gaiadir=None,
                     nside=None, pixlist=None, mindec=-30., mingalb=10.,
                     radius=2.):
    """Generate supplemental sky locations using Gaia-G-band avoidance.

    Parameters
    ----------
    nskiespersqdeg : :class:`float`, optional
        The minimum DENSITY of sky fibers to generate. Defaults to
        reading from :func:`~desimodel.io` with a margin of 4x.
    numproc : :class:`int`, optional, defaults to 16
        The number of processes over which to parallelize.
    gaiadir : :class:`str`, optional, defaults to $GAIA_DIR
        The GAIA_DIR environment variable is set to this directory.
        If None is passed, then it's assumed to already exist.
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPix `nside` to use with `pixlist`.
    pixlist : :class:`list` or `int`, optional, defaults to `None`
        Only return targets in a set of (NESTED) HEALpixels at the
        supplied `nside`. Useful for parallelizing across nodes.
        The first entry sets RELEASE for TARGETIDs, and must be < 1000
        (to prevent confusion with DR1 and above).
    mindec : :class:`float`, optional, defaults to -30
        Minimum declination (o) to include for output sky locations.
    mingalb : :class:`float`, optional, defaults to 10
        Closest latitude to Galactic plane for output sky locations
        (e.g. send 10 to limit to areas beyond -10o <= b < 10o).
    radius : :class:`float`, optional, defaults to 2
        Radius at which to avoid (all) Gaia sources (arcseconds).

    Returns
    -------
    :class:`~numpy.ndarray`
        a structured array of supplemental sky positions in the DESI sky
        target format within the passed `mindec` and `mingalb` limits.

    Notes
    -----
        - The environment variable $GAIA_DIR must be set, or `gaiadir`
          must be passed.
    """
    log.info("running on {} processors".format(numproc))

    # ADM if the GAIA directory was passed, set it.
    if gaiadir is not None:
        os.environ["GAIA_DIR"] = gaiadir

    # ADM if needed, determine the density of sky fibers to generate.
    if nskiespersqdeg is None:
        nskiespersqdeg = density_of_sky_fibers(margin=4)

    # ADM determine the HEALPixel nside of the standard Gaia files.
    anyfiles = find_gaia_files([0, 0], radec=True)
    hdr = fitsio.read_header(anyfiles[0], "GAIAHPX")
    nsidegaia = hdr["HPXNSIDE"]

    # ADM determine the Gaia Data Release.
    ref_cat = fitsio.read(anyfiles[0], rows=0, columns="REF_CAT")
    gdr = gaia_dr_from_ref_cat(ref_cat)[0]

    # ADM create a set of random locations accounting for mindec.
    log.info("Generating supplemental sky locations at Dec > {}o...t={:.1f}s"
             .format(mindec, time()-start))
    from desitarget.randoms import randoms_in_a_brick_from_edges
    ras, decs = randoms_in_a_brick_from_edges(
        0., 360., mindec, 90., density=nskiespersqdeg, wrap=False, seed=414)

    # ADM limit random locations by HEALPixel, if requested.
    if pixlist is not None:
        inhp = is_in_hp([ras, decs], nside, pixlist, radec=True)
        ras, decs = ras[inhp], decs[inhp]

    # ADM limit random locations by mingalb.
    log.info("Generated {} sky locations. Limiting to |b| > {}o...t={:.1f}s"
             .format(len(ras), mingalb, time()-start))
    bnorth = is_in_gal_box([ras, decs], [0, 360, mingalb, 90], radec=True)
    bsouth = is_in_gal_box([ras, decs], [0, 360, -90, -mingalb], radec=True)
    ras, decs = ras[bnorth | bsouth], decs[bnorth | bsouth]

    # ADM find which Gaia HEALPixels are occupied by the random points.
    log.info("Cut to {} sky locations. Finding their Gaia HEALPixels...t={:.1f}s"
             .format(len(ras), time()-start))
    theta, phi = np.radians(90-decs), np.radians(ras)
    pixels = hp.ang2pix(nsidegaia, theta, phi, nest=True)
    upixels = np.unique(pixels)
    npixels = len(upixels)
    log.info("Running across {} Gaia HEALPixels.".format(npixels))

    # ADM parallelize across pixels. The function to run on every pixel.
    def _get_supp(pix):
        """wrapper on get_supp_skies() given a HEALPixel"""
        ii = (pixels == pix)
        return get_supp_skies(ras[ii], decs[ii], radius=radius)

    # ADM this is just to count pixels in _update_status.
    npix = np.zeros((), dtype='i8')
    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if npix % 500 == 0 and npix > 0:
            rate = npix / (time() - t0)
            log.info('{}/{} HEALPixels; {:.1f} pixels/sec'.
                     format(npix, npixels, rate))
        npix[...] += 1    # this is an in-place modification.
        return result

    # - Parallel process across the unique pixels.
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            supp = pool.map(_get_supp, upixels, reduce=_update_status)
    else:
        supp = []
        for upix in upixels:
            supp.append(_update_status(_get_supp(upix)))

    # ADM Concatenate the parallelized results into one rec array.
    supp = np.concatenate(supp)

    # ADM build the OBJIDs from the number of sources per brick.
    # ADM the for loop doesn't seem the smartest way, but it is O(n).
    log.info("Begin assigning OBJIDs to bricks...t={:.1f}s".format(time()-start))
    brxid = supp["BRICKID"]
    # ADM start each brick counting from zero.
    cntr = np.zeros(np.max(brxid)+1, dtype=int)
    objid = []
    for ibrx in brxid:
        cntr[ibrx] += 1
        objid.append(cntr[ibrx])
    # ADM ensure the number of sky positions that were generated doesn't exceed
    # ADM the largest possible OBJID (which is unlikely).
    if np.any(cntr > 2**targetid_mask.OBJID.nbits):
        log.fatal('{} sky locations requested in brick {}, but OBJID cannot exceed {}'
                  .format(nskies, brickname, 2**targetid_mask.OBJID.nbits))
        raise ValueError
    supp["OBJID"] = np.array(objid)
    log.info("Assigned OBJIDs to bricks...t={:.1f}s".format(time()-start))

    # ADM add the TARGETID, DESITARGET bits etc.
    nskies = len(supp)
    desi_target = np.zeros(nskies, dtype='>i8')
    desi_target |= desi_mask.SUPP_SKY
    dum = np.zeros_like(desi_target)
    # ADM Use the Gaia Data Release for RELEASE, too.
    supp["RELEASE"] = gdr
    supp = finalize(supp, desi_target, dum, dum, sky=True, gdr=gdr)

    log.info('Done...t={:.1f}s'.format(time()-start))

    return supp


def select_skies(survey, numproc=16, nskiespersqdeg=None, bands=['g', 'r', 'z'],
                 apertures_arcsec=[0.75], nside=None, pixlist=None, writebricks=False):
    """Generate skies in parallel for bricks in a Legacy Surveys DR.

    Parameters
    ----------
    survey : :class:`object`
        `LegacySurveyData` object for a given Data Release of the Legacy Surveys; see
        :func:`~desitarget.skyutilities.legacypipe.util.LegacySurveyData` for details.
    numproc : :class:`int`, optional, defaults to 16
        The number of processes over which to parallelize.
    nskiespersqdeg : :class:`float`, optional
        The minimum DENSITY of sky fibers to generate. Defaults to reading from
        :func:`~desimodel.io` with a margin of 4x.
    bands : :class:`list`, optional, defaults to ['g', 'r', 'z']
        List of bands to be used to define good sky locations.
    apertures_arcsec : :class:`list`, optional, defaults to [0.75]
        Radii in arcsec of apertures for which to derive flux at a sky location.
    nside : :class:`int`, optional, defaults to ``None``
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

    Returns
    -------
    :class:`~numpy.ndarray`
        a structured array of sky positions in the DESI sky target format for all
        bricks in a Legacy Surveys Data Release.

    Notes
    -----
        - Some core code in this module was initially written by Dustin Lang (@dstndstn).
    """
    # ADM retrieve the bricks of interest for this DR.
    brickdict = get_brick_info([survey.survey_dir])
    bricknames = np.array(list(brickdict.keys()))

    # ADM restrict to only bricks in a set of HEALPixels, if requested.
    if pixlist is not None:
        bra, bdec, _, _, _, _ = np.vstack(list(brickdict.values())).T
        theta, phi = np.radians(90-bdec), np.radians(bra)
        pixnum = hp.ang2pix(nside, theta, phi, nest=True)
        # ADM if an integer was passed, turn it into a list.
        if isinstance(pixlist, int):
            pixlist = [pixlist]
        ii = [pix in pixlist for pix in pixnum]
        bricknames = bricknames[ii]
        # ADM if there are no bricks to process, then die immediately.
        if len(bricknames) == 0:
            log.warning('NO bricks found (nside={}, HEALPixels={}, DRdir={})!'
                        .format(nside, pixlist, survey.survey_dir))
            return
        log.info("Processing bricks (nside={}, HEALPixels={}, DRdir={})"
                 .format(nside, pixlist, survey.survey_dir))
    nbricks = len(bricknames)
    log.info('Processing {} bricks that have observations from DR at {}...t = {:.1f}s'
             .format(nbricks, survey.survey_dir, time()-start))

    # ADM a little more information if we're slurming across nodes.
    if os.getenv('SLURMD_NODENAME') is not None:
        log.info('Running on Node {}'.format(os.getenv('SLURMD_NODENAME')))

    # ADM the critical function to run on every brick.
    def _get_skies(brickname):
        '''wrapper on make_skies_for_a_brick() given a brick name'''

        return make_skies_for_a_brick(survey, brickname,
                                      nskiespersqdeg=nskiespersqdeg, bands=bands,
                                      apertures_arcsec=apertures_arcsec,
                                      write=writebricks)

    # ADM this is just in order to count bricks in _update_status.
    nbrick = np.zeros((), dtype='i8')

    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if nbrick % 500 == 0 and nbrick > 0:
            elapsed = time() - t0
            rate = nbrick / elapsed
            log.info('{}/{} bricks; {:.1f} bricks/sec; {:.1f} total mins elapsed'
                     .format(nbrick, nbricks, rate, elapsed/60.))

        nbrick[...] += 1    # this is an in-place modification.
        return result

    # - Parallel process input files.
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            skies = pool.map(_get_skies, bricknames, reduce=_update_status)
    else:
        skies = list()
        for brickname in bricknames:
            skies.append(_update_status(_get_skies(brickname)))

    # ADM some missing blobs may have contaminated the array.
    skies = [sk for sk in skies if sk is not None]
    # ADM Concatenate the parallelized results into one rec array.
    skies = np.concatenate(skies)

    # ADM make_skies_for_a_brick is pixel-based, so the locations can
    # ADM extend beyond the "true" geometric brick boundaries. Use the
    # ADM brick look-up table to remove these cases.
    brickid = bricks.brickid(skies["RA"], skies["DEC"])
    inbrick = skies["BRICKID"] == brickid
    skies = skies[inbrick]

    log.info('Done with (nside={}, HEALPixels={}, DRdir={})...t={:.1f}s'
             .format(nside, pixlist, survey.survey_dir, time()-start))

    return skies
