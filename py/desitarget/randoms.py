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
from glob import glob
from desitarget.gaiamatch import _get_gaia_dir
from desitarget.geomask import bundle_bricks, box_area
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask
from desitarget.targets import resolve

# ADM the parallelization script
from desitarget.internal import sharedmem

# ADM set up the DESI default logger
from desiutil.log import get_logger

# ADM fake the matplotlib display so it doesn't die on allocated nodes.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

# ADM set up the default logger from desiutil
log = get_logger()

# ADM start the clock
start = time()


def dr_extension(drdir):
    """Determine the extension information for files in a legacy survey coadd directory

    Parameters
    ----------
    drdir : :class:`str`
       The root directory pointing to a Data Release from the Legacy Surveys
       e.g. /global/project/projectdirs/cosmo/data/legacysurvey/dr7.

    Returns
    -------
    :class:`str`
        Whether the file extension is 'gz' or 'fz'.
    :class:`int`
        The corresponding FITS extension number that needs to be read (0 or 1).
    """

    from glob import iglob

    # ADM for speed, create a generator of all of the nexp files in the coadd directory.
    gen = iglob(drdir+"/coadd/*/*/*nexp*")
    # ADM and pop the first one.
    anexpfile = next(gen)
    extn = anexpfile[-2:]

    if extn == 'gz':
        return 'gz', 0

    return 'fz', 1


def randoms_in_a_brick_from_edges(ramin, ramax, decmin, decmax,
                                  density=100000, poisson=True):
    """For given brick edges, return random (RA/Dec) positions in the brick

    Parameters
    ----------
    ramin : :class:`float`
        The minimum "edge" of the brick in Right Ascension.
    ramax : :class:`float`
        The maximum "edge" of the brick in Right Ascension.
    decmin : :class:`float`
        The minimum "edge" of the brick in Declination.
    decmax : :class:`float`
        The maximum "edge" of the brick in Declination.
    density : :class:`int`, optional, defaults to 100,000
        The number of random points to return per sq. deg. As a typical brick is
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned.
    poisson : :class:`boolean`, optional, defaults to True
        Modify the number of random points in the brick so that instead of simply
        being the brick area x the density, it is a number drawn from a Poisson
        distribution with the expectation being the brick area x the density.

    Returns
    -------
    :class:`~numpy.array`
        Right Ascensions of random points in brick
    :class:`~numpy.array`
        Declinations of random points in brick
    """
    # ADM create a unique random seed on the basis of the brick.
    # ADM note this is only unique for bricksize=0.25 for bricks
    # ADM that are more than 0.25 degrees from the poles.
    uniqseed = int(4*ramin)*1000+int(4*(decmin+90))
    np.random.seed(uniqseed)

    # ADM generate random points within the brick at the requested density
    # ADM guard against potential wraparound bugs (assuming bricks are typical
    # ADM sizes of 0.25 x 0.25 sq. deg., or not much larger than that
    if ramax - ramin > 350.:
        ramax -= 360.
    spharea = box_area([ramin, ramax, decmin, decmax])

    if poisson:
        nrand = int(np.random.poisson(spharea*density))
    else:
        nrand = int(spharea*density)
#    log.info('Full area covered by brick is {:.5f} sq. deg....t = {:.1f}s'
#              .format(spharea,time()-start))
    ras = np.random.uniform(ramin, ramax, nrand)
    sindecmin, sindecmax = np.sin(np.radians(decmin)), np.sin(np.radians(decmax))
    decs = np.degrees(np.arcsin(1.-np.random.uniform(1-sindecmax, 1-sindecmin, nrand)))

    nrand = len(ras)

#    log.info('Generated {} randoms in brick with bounds [{:.3f},{:.3f},{:.3f},{:.3f}]...t = {:.1f}s'
#                 .format(nrand,ramin,ramax,decmin,decmax,time()-start))

    return ras, decs


def randoms_in_a_brick_from_name(brickname, drdir, density=100000):
    """For a given brick name, return random (RA/Dec) positions in the brick.

    Parameters
    ----------
    brickname : :class:`str`
        Name of brick in which to generate random points.
    drdir : :class:`str`
       The root directory pointing to a Data Release from the Legacy Surveys
       e.g. /global/project/projectdirs/cosmo/data/legacysurvey/dr7.
    density : :class:`int`, optional, defaults to 100,000
        The number of random points to return per sq. deg. As a typical brick is
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned.

    Returns
    -------
    :class:`~numpy.array`
        Right Ascensions of random points in brick.
    :class:`~numpy.array`
        Declinations of random points in brick.

    Notes
    -----
        - First version copied shamelessly from Anand Raichoor.
    """
    # ADM read in the survey bricks file to determine the brick boundaries
    hdu = fits.open(drdir+'survey-bricks.fits.gz')

    brickinfo = hdu[1].data
    wbrick = np.where(brickinfo['brickname'] == brickname)[0]
    if len(wbrick) == 0:
        log.error('Brick {} does not exist'.format(brickname))
    # else:
    #    log.info('Working on brick {}...t = {:.1f}s'.format(brickname,time()-start))

    brick = brickinfo[wbrick][0]
    ramin, ramax, decmin, decmax = brick['ra1'], brick['ra2'], brick['dec1'], brick['dec2']

    # ADM create a unique random seed on the basis of the brick.
    # ADM note this is only unique for bricksize=0.25 for bricks
    # ADM that are more than 0.25 degrees from the poles.
    uniqseed = int(4*ramin)*1000+int(4*(decmin+90))
    np.random.seed(uniqseed)

    # ADM generate random points within the brick at the requested density
    # ADM guard against potential wraparound bugs
    if ramax - ramin > 350.:
        ramax -= 360.
    spharea = box_area([ramin, ramax, decmin, decmax])

    nrand = int(spharea*density)
    # log.info('Full area covered by brick {} is {:.5f} sq. deg....t = {:.1f}s'
    #          .format(brickname,spharea,time()-start))
    ras = np.random.uniform(ramin, ramax, nrand)
    sindecmin, sindecmax = np.sin(np.radians(decmin)), np.sin(np.radians(decmax))
    decs = np.degrees(np.arcsin(1.-np.random.uniform(1-sindecmax, 1-sindecmin, nrand)))

    nrand = len(ras)

#    log.info('Generated {} randoms in brick {} with bounds [{:.3f},{:.3f},{:.3f},{:.3f}]...t = {:.1f}s'
#                 .format(nrand,brickname,ramin,ramax,decmin,decmax,time()-start))

    return ras, decs


def _pre_or_post_dr8(drdir):
    """Whether the imaging surveys directory structure is before or after DR8

    Parameters
    ----------
    drdir : :class:`str`
       The root directory pointing to a Data Release from the Legacy Surveys
       e.g. /global/project/projectdirs/cosmo/data/legacysurvey/dr7.

    Returns
    -------
    :class:`list`
       For DR8, this just returns the original directory as a list. For DR8
       this returns a list of two directories, one corresponding to DECaLS
       and one corresponding to BASS/MzLS.
    """
    if os.path.exists(os.path.join(drdir, "coadd")):
        drdirs = [drdir]
    else:
        wcoadd = glob(os.path.join(drdir, '*', "coadd"))
        drdirs = [os.path.dirname(dd) for dd in wcoadd]

    return drdirs


def dr8_quantities_at_positions_in_a_brick(ras, decs, brickname, drdir):
    """Wrapper on `quantities_at_positions_in_a_brick` for DR8 imaging and beyond.

    Notes
    -----
        - See :func:`~desitarget.randoms.quantities_at_positions_in_a_brick`
          for details. This function detects whether we have TWO coadd directories
          in the `drdir` (e.g. one for DECaLS and one for MzLS/BASS) and, if so,
          creates randoms for both surveys within the the passed brick. If not, it
          defaults to the behavior for only having one survey.
    """
    # ADM determine whether we have to traverse two sets of brick directories.
    drdirs = _pre_or_post_dr8(drdir)

    # ADM determine the dictionary of quantities for one or two directories.
    qall = []
    for dd in drdirs:
        qall.append(quantities_at_positions_in_a_brick(ras, decs, brickname, dd))

    # ADM concatenate everything in qall into one dictionary.
    qcombine = {}
    for k in qall[0].keys():
        qcombine[k] = np.concatenate([q[k] for q in qall])

    return qcombine


def quantities_at_positions_in_a_brick(ras, decs, brickname, drdir):
    """Return NOBS, GALDEPTH, PSFDEPTH (per-band) at positions in one brick of the Legacy Surveys

    Parameters
    ----------
    ras : :class:`~numpy.array`
        Right Ascensions of interest (degrees).
    decs : :class:`~numpy.array`
        Declinations of interest (degrees).
    brickname : :class:`str`
        Name of brick which contains RA/Dec positions, e.g., '1351p320'.
    drdir : :class:`str`
       The root directory pointing to a Data Release from the Legacy Surveys
       e.g. /global/project/projectdirs/cosmo/data/legacysurvey/dr7.

    Returns
    -------
    :class:`dictionary`
       The number of observations (NOBS_X), PSF depth (PSFDEPTH_X) and Galaxy depth (GALDEPTH_X)
       at each passed position in the Legacy Surveys in each band X. In addition, the MASKBITS
       information at each passed position for the brick.

    Notes
    -----
        - First version copied shamelessly from Anand Raichoor.
    """
    npts = len(ras)

    # ADM determine whether the coadd files have extension .gz or .fz based on the DR directory.
    extn, extn_nb = dr_extension(drdir)

    # ADM the output dictionary.
    qdict = {}

    # as a speed up, we assume all images in different filters for the brick have the same WCS
    # -> if we have read it once (iswcs=True), we use this info
    iswcs = False
    # ADM this will store the instrument name the first time we touch the wcs
    instrum = None

    rootdir = os.path.join(drdir, 'coadd', brickname[:3], brickname)
    # ADM loop through each of the filters and store the number of observations at the
    # ADM RA and Dec positions of the passed points.
    for filt in ['g', 'r', 'z']:
        # ADM the input file labels, and output column names and output formats
        # ADM for each of the quantities of interest.
        qnames = zip(['nexp', 'depth', 'galdepth'],
                     ['nobs', 'psfdepth', 'galdepth'],
                     ['i2', 'f4', 'f4'])
        for qin, qout, qform in qnames:
            fn = os.path.join(
                rootdir, 'legacysurvey-{}-{}-{}.fits.{}'.format(brickname, qin, filt, extn)
                )
            # ADM only process the WCS if there is a file corresponding to this filter.
            if os.path.exists(fn):
                img = fits.open(fn)
                if not iswcs:
                    # ADM also store the instrument name, if it isn't yet stored.
                    instrum = img[extn_nb].header["INSTRUME"].lower().strip()
                    w = WCS(img[extn_nb].header)
                    x, y = w.all_world2pix(ras, decs, 0)
                    iswcs = True
                # ADM determine the quantity of interest at each passed location
                # ADM and store in a dictionary with the filter and quantity name.
                qdict[qout+'_'+filt] = img[extn_nb].data[y.astype("int"), x.astype("int")]
                # log.info('Determined {} using WCS for {}...t = {:.1f}s'
                #          .format(qout+'_'+filt,fn,time()-start))
            else:
                # log.info('no {} file at {}...t = {:.1f}s'
                #          .format(qin+'_'+filt,fn,time()-start))
                # ADM if the file doesn't exist, set the relevant quantities to zero.
                qdict[qout+'_'+filt] = np.zeros(npts, dtype=qform)

    # ADM add the mask bits information.
    fn = os.path.join(rootdir,
                      'legacysurvey-{}-maskbits.fits.{}'.format(brickname, extn))
    # ADM only process the WCS if there is a file corresponding to this filter.
    if os.path.exists(fn):
        img = fits.open(fn)
        # ADM use the WCS calculated for the per-filter quantities above, if it exists.
        if not iswcs:
            # ADM also store the instrument name, if it isn't yet stored.
            instrum = img[extn_nb].header["INSTRUME"].lower().strip()
            w = WCS(img[extn_nb].header)
            x, y = w.all_world2pix(ras, decs, 0)
            iswcs = True
        # ADM add the maskbits to the dictionary.
        qdict['maskbits'] = img[extn_nb].data[y.astype("int"), x.astype("int")]
    else:
        # ADM if there is no maskbits file, populate with zeros.
        qdict['maskbits'] = np.zeros(npts, dtype='i2')

    # ADM finally, populate the photometric system in the quantity dictionary.
    if instrum == 'decam':
        qdict['photsys'] = np.array([b"S" for x in range(npts)], dtype='|S1')
    else:
        qdict['photsys'] = np.array([b"N" for x in range(npts)], dtype='|S1')
#    log.info('Recorded quantities for each point in brick {}...t = {:.1f}s'
#                  .format(brickname,time()-start))

    return qdict


def hp_with_nobs_in_a_brick(ramin, ramax, decmin, decmax, brickname, drdir,
                            density=100000, nside=256):
    """Given a brick's edges/name, count randoms with NOBS > 1 in HEALPixels touching that brick.

    Parameters
    ----------
    ramin : :class:`float`
        The minimum "edge" of the brick in Right Ascension.
    ramax : :class:`float`
        The maximum "edge" of the brick in Right Ascension
    decmin : :class:`float`
        The minimum "edge" of the brick in Declination.
    decmax : :class:`float`
        The maximum "edge" of the brick in Declination.
    brickname : :class:`~numpy.array`
        Brick names that corresponnds to the brick edges, e.g., '1351p320'.
    drdir : :class:`str`
       The root directory pointing to a Data Release from the Legacy Surveys
       e.g. /global/project/projectdirs/cosmo/data/legacysurvey/dr7.
    density : :class:`int`, optional, defaults to 100,000
        The number of random points to return per sq. deg. As a typical brick is
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned.
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel NESTED nside number) at which to build the map.

    Returns
    -------
    :class:`~numpy.ndarray`
        a numpy structured array with the following columns:
            HPXPIXEL: Integer numbers of (only) those HEALPixels that overlap the passed brick
            HPXCOUNT: Numbers of random points with one or more observations (NOBS > 0) in the
                passed Data Release of the Legacy Surveys for each returned HPXPIXEL.

    Notes
    -----
        - The HEALPixel numbering uses the NESTED scheme.
        - In the event that there are no pixels with one or more observations in the passed
          brick, and empty structured array will be returned.
    """
    # ADM this is only intended to work on one brick, so die if a larger array is passed.
    if not isinstance(brickname, str):
        log.fatal("Only one brick can be passed at a time!")
        raise ValueError

    # ADM generate an empty structured array to return in the event that no pixels with
    # ADM counts were found.
    hpxinfo = np.zeros(0, dtype=[('HPXPIXEL', '>i4'), ('HPXCOUNT', '>i4')])

    # ADM generate random points within the brick at the requested density.
    ras, decs = randoms_in_a_brick_from_edges(ramin, ramax, decmin, decmax, density=density)

    # ADM retrieve the number of observations for each random point.
    nobs_g, nobs_r, nobs_z = nobs_at_positions_in_a_brick(ras, decs, brickname, drdir=drdir)

    # ADM only retain points with one or more observations in all bands.
    w = np.where((nobs_g > 0) & (nobs_r > 0) & (nobs_z > 0))

    # ADM if there were some non-zero observations, populate the pixel numbers and counts.
    if len(w[0]) > 0:
        pixnums = hp.ang2pix(nside, np.radians(90.-decs[w]), np.radians(ras[w]), nest=True)
        pixnum, pixcnt = np.unique(pixnums, return_counts=True)
        hpxinfo = np.zeros(len(pixnum), dtype=[('HPXPIXEL', '>i4'), ('HPXCOUNT', '>i4')])
        hpxinfo['HPXPIXEL'] = pixnum
        hpxinfo['HPXCOUNT'] = pixcnt

    return hpxinfo


def get_dust(ras, decs, scaling=1, dustdir=None):
    """Get SFD E(B-V) values at a set of RA/Dec locations

    Parameters
    ----------
    ra : :class:`numpy.array`
        Right Ascension in degrees
    dec : :class:`numpy.array`
        Declination in degrees
    scaling : :class:`float`
        Pass 1 for the SFD98 dust maps. A scaling of 0.86 corresponds
        to the recalibration from Schlafly & Finkbeiner (2011).
    dustdir : :class:`str`, optional, defaults to $DUST_DIR+'/maps'
        The root directory pointing to SFD dust maps. If not
        sent the code will try to use $DUST_DIR+'/maps' before failing.

    Returns
    -------
    :class:`numpy.array`
        E(B-V) values from the SFD dust maps at the passed locations
    """
    from desiutil.dust import SFDMap
    return SFDMap(mapdir=dustdir).ebv(ras, decs, scaling=scaling)


def get_quantities_in_a_brick(ramin, ramax, decmin, decmax, brickname, drdir,
                              density=100000, dustdir=None):
    """NOBS, DEPTHS etc. (per-band) for random points in a brick of the Legacy Surveys

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
    drdir : :class:`str`
       The root directory pointing to a Data Release from the Legacy Surveys
       e.g. /global/project/projectdirs/cosmo/data/legacysurvey/dr7.
    density : :class:`int`, optional, defaults to 100,000
        The number of random points to return per sq. deg. As a typical brick is
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned
    dustdir : :class:`str`, optional, defaults to $DUST_DIR+'/maps'
        The root directory pointing to SFD dust maps. If not
        sent the code will try to use $DUST_DIR+'/maps' before failing.

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
            MASKBITS: Extra mask bits info as stored in the header of e.g.,
              dr7dir + 'coadd/111/1116p210/legacysurvey-1116p210-maskbits.fits.gz'
            EBV: E(B-V) at this location from the SFD dust maps
    """
    # ADM this is only intended to work on one brick, so die if a larger array is passed.
    if not isinstance(brickname, str):
        log.fatal("Only one brick can be passed at a time!")
        raise ValueError

    # ADM generate random points within the brick at the requested density.
    ras, decs = randoms_in_a_brick_from_edges(ramin, ramax, decmin, decmax, density=density)

    # ADM retrieve the dictionary of quantities for each random point.
    qdict = dr8_quantities_at_positions_in_a_brick(ras, decs, brickname, drdir)

    # ADM retrieve the E(B-V) values for each random point.
    ebv = get_dust(ras, decs, dustdir=dustdir)

    # ADM convert the dictionary to a structured array.
    qinfo = np.zeros(len(ras),
                     dtype=[('RA', 'f8'), ('DEC', 'f8'), ('BRICKNAME', 'S8'),
                            ('NOBS_G', 'i2'), ('NOBS_R', 'i2'), ('NOBS_Z', 'i2'),
                            ('PSFDEPTH_G', 'f4'), ('PSFDEPTH_R', 'f4'), ('PSFDEPTH_Z', 'f4'),
                            ('GALDEPTH_G', 'f4'), ('GALDEPTH_R', 'f4'), ('GALDEPTH_Z', 'f4'),
                            ('MASKBITS', 'i2'), ('EBV', 'f4'), ('PHOTSYS', '|S1')])
    # ADM store each quantity of interest in the structured array
    # ADM remembering that the dictionary keys are in lower case text.
    cols = qdict.keys()
    for col in cols:
        qinfo[col.upper()] = qdict[col]

    # ADM add the RAs/Decs and brick name.
    qinfo["RA"], qinfo["DEC"], qinfo["BRICKNAME"] = ras, decs, brickname

    # ADM add the dust values.
    qinfo["EBV"] = ebv

    return qinfo


def pixweight(randoms, density, nobsgrz=[0, 0, 0], nside=256, outplot=None, outarea=True):
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
    nobsgrz : :class:`list`, optional, defaults to [0,0,0]
        The number of observations in each of g AND r AND z that have to be EXCEEDED to include
        a random point in the count. The default is to include areas that have at least one
        observation in each band ([0,0,0]). `nobsgrz = [0,-1,-1]` would count areas with at
        least one (more than zero) observations in g-band but any number of observations (more
        than -1) in r-band and z-band.
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel NESTED nside number) at which to build the map.
    outplot : :class:`str`, optional, defaults to not making a plot
        Create a plot and write it to a file named `outplot` (this is passed to
        the `savefig` routine from `matplotlib.pyplot`.
    outarea : :class:`boolean`, optional, defaults to True
        Print the total area of the survey for these values of `nobsgrz` to screen.

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
    # ADM if a file name was passed for the random catalog, read it in
    if isinstance(randoms, str):
        randoms = fitsio.read(randoms)

    # ADM extract the columns of interest
    ras, decs = randoms["RA"], randoms["DEC"]
    nobs_g, nobs_r, nobs_z = randoms["NOBS_G"], randoms["NOBS_R"], randoms["NOBS_Z"]

    # ADM only retain points with one or more observations in all bands
    w = np.where((nobs_g > nobsgrz[0]) & (nobs_r > nobsgrz[1]) & (nobs_z > nobsgrz[2]))

    # ADM the counts in each HEALPixel in the survey
    if len(w[0]) > 0:
        pixnums = hp.ang2pix(nside, np.radians(90.-decs[w]), np.radians(ras[w]), nest=True)
        pixnum, pixcnt = np.unique(pixnums, return_counts=True)
    else:
        log.error("No area for which nobs exceed passed values of nobsgrz, or empty randoms array")

    # ADM generate the counts for the whole sky to retain zeros where there is no survey coverage
    npix = hp.nside2npix(nside)
    pix_cnt = np.bincount(pixnum, weights=pixcnt, minlength=npix)

    # ADM we know the area of HEALPixels at this nside, so we know what the count SHOULD be
    expected_cnt = hp.nside2pixarea(nside, degrees=True)*density
    # ADM create a weight map based on the actual counts divided by the expected counts
    pix_weight = pix_cnt/expected_cnt

    # ADM if outplot was passed, make a plot of the weights in Mollweide projection
    if outplot is not None:
        log.info('Plotting pixel map and writing to {}'.format(outplot))
        hp.mollview(pix_weight, nest=True)
        plt.savefig(outplot)

    # ADM if requested, print the total area of the survey to screen
    if outarea:
        area = np.sum(pix_weight*hp.nside2pixarea(nside, degrees=True))
        log.info('Area of survey with NOBS exceeding {} in [g,r,z] = {:.2f} sq. deg.'
                 .format(nobsgrz, area))

    log.info('Done...t = {:.1f}s'.format(time()-start))

    return pix_weight


def stellar_density(nside=256):
    """Make a HEALPixel map of stellar density based on Gaia.

    Parameters
    ----------
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel NESTED nside number) at which to build the map.

    Notes
    -----
        - The environment variable $GAIA_DIR must be set.
    """
    # ADM check that the GAIA_DIR is set and retrieve it.
    gaiadir = _get_gaia_dir()
    fitsdir = os.path.join(gaiadir, 'fits')

    # ADM the number of pixels and the pixel area at the passed nside.
    npix = hp.nside2npix(nside)
    pixarea = hp.nside2pixarea(nside, degrees=True)

    # ADM an output array to populate containing all possible HEALPixels at the passed nside.
    pixout = np.zeros(npix, dtype='int32')

    # ADM find all of the Gaia files.
    filenames = glob(fitsdir+'/*fits')

    # ADM read in each file, restricting to the criteria for point sources
    # ADM and storing in a HEALPixel map at resolution nside.
    for filename in filenames:
        # ADM save memory and speed up by only reading in a subset of columns.
        gobjs = fitsio.read(filename,
                            columns=['RA', 'DEC', 'PHOT_G_MEAN_MAG', 'ASTROMETRIC_EXCESS_NOISE'])

        # ADM restrict to subset of sources using point source definition.
        ra, dec = gobjs["RA"], gobjs["DEC"]
        gmag, excess = gobjs["PHOT_G_MEAN_MAG"], gobjs["ASTROMETRIC_EXCESS_NOISE"]
        point = (excess == 0.) | (np.log10(excess) < 0.3*gmag-5.3)
        grange = (gmag >= 12) & (gmag < 17)
        w = np.where(point & grange)

        # ADM calculate the HEALPixels for the point sources.
        theta, phi = np.radians(90-dec[w]), np.radians(ra[w])
        pixnums = hp.ang2pix(nside, theta, phi, nest=True)

        # ADM return the counts in each pixel number...
        pixnum, pixcnt = np.unique(pixnums, return_counts=True)
        # ADM...and populate the output array with the counts.
        pixout[pixnum] += pixcnt

    # ADM return the density
    return pixout/pixarea


def get_targ_dens(targets, nside=256):
    """The density of targets in HEALPixels

    Parameters
    ----------
    targets : :class:`~numpy.ndarray` or `str`
        A corresponding (same Legacy Surveys Data Release) target catalog as made by,
        e.g., :func:`desitarget.cuts.select_targets()`, or the name of such a file.
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel nside number) at which to build the map (NESTED scheme).

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of target densities with one column for every bit returned by
        :func:`desitarget.QA._load_targdens()`. The array contains the density of
        those targets in pixels at the passed `nside`
    """
    # ADM if a file name was passed for the targets catalog, read it in
    if isinstance(targets, str):
        log.info('Reading in target catalog...t = {:.1f}s'.format(time()-start))
        targets = fitsio.read(targets)

    # ADM the number of pixels and the pixel area at the passed nside
    npix = hp.nside2npix(nside)
    pixarea = hp.nside2pixarea(nside, degrees=True)

    # ADM retrieve the pixel numbers for every target RA/Dec
    ras, decs = targets["RA"], targets["DEC"]
    pixnums = hp.ang2pix(nside, np.radians(90.-decs), np.radians(ras), nest=True)

    # ADM retrieve the bit names of interest
    from desitarget.QA import _load_targdens
    bitnames = np.array(list(_load_targdens().keys()))

    # ADM and set up an array to hold the output target densities
    targdens = np.zeros(npix, dtype=[(bitname, 'f4') for bitname in bitnames])

    for bitname in bitnames:
        if 'ALL' in bitname:
            wbit = np.arange(len(targets))
        else:
            if ('BGS' in bitname) & ~('ANY' in bitname):
                wbit = np.where(targets["BGS_TARGET"] & bgs_mask[bitname])[0]
            elif ('MWS' in bitname) & ~('ANY' in bitname):
                wbit = np.where(targets["MWS_TARGET"] & mws_mask[bitname])[0]
            else:
                wbit = np.where(targets["DESI_TARGET"] & desi_mask[bitname])[0]

        if len(wbit) > 0:
            # ADM calculate the number of objects in each pixel for the
            # ADM targets of interest
            pixnum, pixcnt = np.unique(pixnums[wbit], return_counts=True)
            targdens[bitname][pixnum] = pixcnt/pixarea

    return targdens


def pixmap(randoms, targets, rand_density, nside=256, gaialoc=None):
    """A HEALPixel map of useful quantities for analyzing a Legacy Surveys Data Release

    Parameters
    ----------
    randoms : :class:`~numpy.ndarray` or `str`
        A random catalog as made by, e.g., :func:`select_randoms()` or
        :func:`quantities_at_positions_in_a_brick()`, or the name of such a file.
    targets : :class:`~numpy.ndarray` or `str`
        A corresponding (same Legacy Surveys Data Release) target catalog as made by,
        e.g., :func:`desitarget.cuts.select_targets()`, or the name of such a file.
    rand_density : :class:`int`
        The number of random points per sq. deg. At which the random catalog was
        generated (see also :func:`select_randoms()`).
    nside : :class:`int`, optional, defaults to nside=256 (~0.0525 sq. deg. or "brick-sized")
        The resolution (HEALPixel nside number) at which to build the map (NESTED scheme).
    gaialoc : :class:`str`, optional, defaults to ``None``
        If a file is passed it is assumed to be a FITS file that already contains the
        column "STARDENS", which is simply read in. Otherwise, the stellar density is
        constructed from the files stored in the default location indicated by the
        $GAIA_DIR environment variable.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of useful information that includes
            - HPXPIXEL: HEALPixel integers at the passed `nside`.
            - FRACAREA: The fraction of the pixel with at least one observation in any
                        band according to `randoms`. Made with :func:`pixweight()`.
            - STARDENS: The stellar density in a pixel from Gaia. Made with
                        :func:`stellar_density()`.
            - EBV: The E(B-V) in the pixel from the SFD dust map, derived from the
                   median of EBV values in the passed random catalog.
            - PSFDEPTH_G, R, Z: The PSF depth in g, r, z-band in the pixel, derived from
                                the median of PSFDEPTH values in the passed random catalog.
            - GALDEPTH_G, R, Z: The galaxy depth in g, r, z-band in the pixel, derived from
                                the median of GALDEPTH values in the passed random catalog.
            - One column for every bit returned by :func:`desitarget.QA._load_targdens()`.
              Each column contains the density of targets in pixels at the passed `nside`
    Notes
    -----
        - If `gaialoc` is ``None`` then the environment variable $GAIA_DIR must be set.
    """
    # ADM if a file name was passed for the random catalog, read it in
    if isinstance(randoms, str):
        log.info('Reading in random catalog...t = {:.1f}s'.format(time()-start))
        randoms = fitsio.read(randoms)

    # ADM if a file name was passed for the targets catalog, read it in
    if isinstance(targets, str):
        log.info('Reading in target catalog...t = {:.1f}s'.format(time()-start))
        targets = fitsio.read(targets)

    # ADM determine the areal coverage at of the randoms at this nside
    log.info('Determining footprint...t = {:.1f}s'.format(time()-start))
    pw = pixweight(randoms, rand_density, nside=nside)
    npix = len(pw)

    # ADM get the target densities
    log.info('Calculating target densities...t = {:.1f}s'.format(time()-start))
    targdens = get_targ_dens(targets, nside=nside)

    # ADM set up the output array
    datamodel = [('HPXPIXEL', '>i4'), ('FRACAREA', '>f4'), ('STARDENS', '>f4'), ('EBV', '>f4'),
                 ('PSFDEPTH_G', '>f4'), ('PSFDEPTH_R', '>f4'), ('PSFDEPTH_Z', '>f4'),
                 ('GALDEPTH_G', '>f4'), ('GALDEPTH_R', '>f4'), ('GALDEPTH_Z', '>f4')]
    datamodel += targdens.dtype.descr
    hpxinfo = np.zeros(npix, dtype=datamodel)
    # ADM set initial values to -1 so that they can easily be clipped
    hpxinfo[...] = -1

    # ADM add the areal coverage, pixel information and target densities
    hpxinfo['HPXPIXEL'] = np.arange(npix)
    hpxinfo['FRACAREA'] = pw
    for col in targdens.dtype.names:
        hpxinfo[col] = targdens[col]

    # ADM build the stellar density, or if gaialoc was passed as a file, just read it in.
    if gaialoc is None:
        log.info('Calculating stellar density using Gaia files in $GAIA_DIR...t = {:.1f}s'
                 .format(time()-start))
        sd = stellar_density(nside=nside)
    else:
        sd = fitsio.read(gaialoc, columns=["STARDENS"])
        if len(sd) != len(hpxinfo):
            log.critical('Stellar density map in {} was not calculated at NSIDE={}'
                         .format(gaialoc, nside))
    hpxinfo["STARDENS"] = sd

    # ADM add the median values of all of the other systematics
    log.info('Calculating medians of systematics from random catalog...t = {:.1f}s'
             .format(time()-start))
    ras, decs = randoms["RA"], randoms["DEC"]
    pixnums = hp.ang2pix(nside, np.radians(90.-decs), np.radians(ras), nest=True)

    # ADM some sorting to order the values to extract the medians
    pixorder = np.argsort(pixnums)
    pixels, pixcnts = np.unique(pixnums, return_counts=True)
    pixcnts = np.insert(pixcnts, 0, 0)
    pixcnts = np.cumsum(pixcnts)

    # ADM work through the ordered pixels to populate the median for
    # ADM each quantity of interest
    cols = ['EBV', 'PSFDEPTH_G', 'GALDEPTH_G', 'PSFDEPTH_R', 'GALDEPTH_R',
            'PSFDEPTH_Z', 'GALDEPTH_Z']
    for i in range(len(pixcnts)-1):
        inds = pixorder[pixcnts[i]:pixcnts[i+1]]
        pix = pixnums[inds][0]
        for col in cols:
            hpxinfo[col][pix] = np.median(randoms[col][inds])

    log.info('Done...t = {:.1f}s'.format(time()-start))

    return hpxinfo


def select_randoms(drdir, density=100000, numproc=32, nside=4, pixlist=None,
                   bundlebricks=None, brickspersec=2.5,
                   dustdir=None):
    """NOBS, GALDEPTH, PSFDEPTH (per-band) for random points in a DR of the Legacy Surveys

    Parameters
    ----------
    drdir : :class:`str`
       The root directory pointing to a Data Release from the Legacy Surveys
       e.g. /global/project/projectdirs/cosmo/data/legacysurvey/dr7.
    density : :class:`int`, optional, defaults to 100,000
        The number of random points to return per sq. deg. As a typical brick is
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned
    numproc : :class:`int`, optional, defaults to 32
        The number of processes over which to parallelize
    nside : :class:`int`, optional, defaults to nside=4 (214.86 sq. deg.)
        The (NESTED) HEALPixel nside to be used with the `pixlist` and `bundlebricks` input.
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
    brickspersec : :class:`float`, optional, defaults to 2.5
        The rough number of bricks processed per second by the code (parallelized across
        a chosen number of nodes). Used in conjunction with `bundlebricks` for the code
        to estimate time to completion when parallelizing across pixels.
    dustdir : :class:`str`, optional, defaults to $DUST_DIR+'maps'
        The root directory pointing to SFD dust maps. If not
        sent the code will try to use $DUST_DIR+'maps')
        before failing.

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
            MASKBITS: Extra mask bits info as stored in the header of e.g.,
              dr7dir + 'coadd/111/1116p210/legacysurvey-1116p210-maskbits.fits.gz'
            EBV: E(B-V) at this location from the SFD dust maps
    """
    # ADM read in the survey bricks file, which lists the bricks of interest for this DR.
    # ADM if this is pre-or-post-DR8 we need to find the correct directory or directories.
    drdirs = _pre_or_post_dr8(drdir)
    bricknames = []
    brickinfo = []
    for dd in drdirs:
        sbfile = glob(dd+'/*bricks-dr*')
        if len(sbfile) > 0:
            sbfile = sbfile[0]
            hdu = fits.open(sbfile)
            brickinfo.append(hdu[1].data)
            bricknames.append(hdu[1].data['BRICKNAME'])
        else:
            # ADM this is a hack for test bricks where we didn't always generate the
            # ADM bricks file. It's probably safe to remove it at some point.
            from desitarget.io import brickname_from_filename
            fns = glob(os.path.join(dd, 'tractor', '*', '*fits'))
            bricknames.append([brickname_from_filename(fn) for fn in fns])
            brickinfo.append([])
            if pixlist is not None or bundlebricks is not None:
                msg = 'DR-specific bricks file not found'
                msg += 'and pixlist or bundlebricks passed!!!'
                log.critical(msg)
                raise ValueError(msg)
    bricknames = np.concatenate(bricknames)
    brickinfo = np.concatenate(brickinfo)

    # ADM if the pixlist or bundlebricks option was sent, we'll need the HEALPixel
    # ADM information for each brick.
    if pixlist is not None or bundlebricks is not None:
        theta, phi = np.radians(90-brickinfo["dec"]), np.radians(brickinfo["ra"])
        pixnum = hp.ang2pix(nside, theta, phi, nest=True)

    # ADM if the bundlebricks option was sent, call the packing code.
    if bundlebricks is not None:
        bundle_bricks(pixnum, bundlebricks, nside, brickspersec=brickspersec,
                      prefix='randoms', surveydir=drdir)
        return

    # ADM restrict to only bricks in a set of HEALPixels, if requested.
    if pixlist is not None:
        # ADM if an integer was passed, turn it into a list.
        if isinstance(pixlist, int):
            pixlist = [pixlist]
        wbricks = np.where([pix in pixlist for pix in pixnum])[0]
        bricknames = bricknames[wbricks]
        if len(wbricks) == 0:
            log.warning('ZERO bricks in passed pixel list!!!')
        log.info("Processing bricks in (nside={}, pixel numbers={}) HEALPixels"
                 .format(nside, pixlist))

    nbricks = len(bricknames)
    log.info('Processing {} bricks from DR at {} at density {:.1e} per sq. deg...t = {:.1f}s'
             .format(nbricks, drdir, density, time()-start))

    # ADM a little more information if we're slurming across nodes.
    if os.getenv('SLURMD_NODENAME') is not None:
        log.info('Running on Node {}'.format(os.getenv('SLURMD_NODENAME')))

    # ADM initialize the bricks class, and retrieve the brick information look-up table
    # ADM so it can be used in a common fashion.
    from desiutil import brick
    bricktable = brick.Bricks(bricksize=0.25).to_table()

    # ADM the critical function to run on every brick.
    def _get_quantities(brickname):
        '''wrapper on nobs_positions_in_a_brick_from_edges() given a brick name'''
        # ADM retrieve the edges for the brick that we're working on
        wbrick = np.where(bricktable["BRICKNAME"] == brickname)[0]
        ramin, ramax, decmin, decmax = np.array(bricktable[wbrick]["RA1", "RA2", "DEC1", "DEC2"])[0]

        # ADM populate the brick with random points, and retrieve the quantities
        # ADM of interest at those points.
        return get_quantities_in_a_brick(ramin, ramax, decmin, decmax, brickname, drdir,
                                         density=density, dustdir=dustdir)

    # ADM this is just to count bricks in _update_status
    nbrick = np.zeros((), dtype='i8')

    t0 = time()

    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if nbrick % 50 == 0 and nbrick > 0:
            rate = nbrick / (time() - t0)
            log.info('{}/{} bricks; {:.1f} bricks/sec'.format(nbrick, nbricks, rate))
            # ADM if we're going to exceed 4 hours, warn the user
            if nbricks/rate > 4*3600.:
                log.error("May take > 4 hours to run. Try running with bundlebricks instead.")

        nbrick[...] += 1    # this is an in-place modification
        return result

    # - Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            qinfo = pool.map(_get_quantities, bricknames, reduce=_update_status)
    else:
        qinfo = list()
        for brickname in bricknames:
            qinfo.append(_update_status(_get_quantities(brickname)))

    # ADM concatenate the randoms into a single long list and resolve whether
    # ADM they are officially in the north or the south.
    qinfo = np.concatenate(qinfo)
    qinfo = resolve(qinfo)

    # ADM one last shuffle to randomize across brick boundaries.
    np.random.seed(616)
    np.random.shuffle(qinfo)

    return qinfo
