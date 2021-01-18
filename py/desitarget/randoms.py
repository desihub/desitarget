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
import photutils
from glob import glob, iglob

from desitarget.gaiamatch import get_gaia_dir
from desitarget.geomask import bundle_bricks, box_area
from desitarget.geomask import get_imaging_maskbits, get_default_maskbits
from desitarget.targets import resolve, main_cmx_or_sv, finalize
from desitarget.skyfibers import get_brick_info
from desitarget.io import read_targets_in_box, target_columns_from_header
from desitarget.targetmask import desi_mask as dMx

# ADM the parallelization script.
from desitarget.internal import sharedmem

# ADM set up the DESI default logger.
from desiutil import brick
from desiutil.log import get_logger

# ADM a look-up dictionary that converts priorities to bit-names.
bitperprio = {dMx[bn].priorities["UNOBS"]: dMx[bn] for bn in dMx.names()
              if len(dMx[bn].priorities) > 0}

# ADM set up the Legacy Surveys bricks objects.
bricks = brick.Bricks(bricksize=0.25)
# ADM make a BRICKNAME->BRICKID look-up table for speed.
bricktable = bricks.to_table()
bricklookup = {bt["BRICKNAME"]: bt["BRICKID"] for bt in bricktable}

# ADM set up the default logger from desiutil.
log = get_logger()

# ADM start the clock.
start = time()


def dr_extension(drdir):
    """Extension information for files in a Legacy Survey coadd directory

    Parameters
    ----------
    drdir : :class:`str`
        The root directory for a Data Release from the Legacy Surveys
        e.g. /global/project/projectdirs/cosmo/data/legacysurvey/dr7.

    Returns
    -------
    :class:`str`
        Whether the file extension is 'gz' or 'fz'.
    :class:`int`
        Corresponding FITS extension number to be read (0 or 1).

    Notes
    -----
        - If the directory structure seems wrong or can't be found then
          the post-DR4 convention (.fz files) is returned.
    """
    try:
        # ADM a generator of all of the nexp files in the coadd directory.
        gen = iglob(drdir+"/coadd/*/*/*nexp*")
        # ADM pop the first file in the generator.
        anexpfile = next(gen)
        extn = anexpfile[-2:]
        if extn == 'gz':
            return 'gz', 0
    # ADM this is triggered if the generator is empty.
    except StopIteration:
        msg = "couldn't find any nexp files in {}...".format(
            os.path.join(drdir, "coadd"))
        msg += "Defaulting to '.fz' extensions for Legacy Surveys coadd files"
        log.info(msg)

    return 'fz', 1


def finalize_randoms(randoms):
    """Add the standard "final" columns that are also added in targeting.

    Parameters
    ----------
    randoms : :class:`~numpy.ndarray`
        A random catalog as made by, e.g., :func:`select_randoms()`
        with `nomtl=True` or `select_randoms_bricks()` with
        `nomtl=False`. This function adds the default MTL information.

    Returns
    -------
    :class:`~numpy.array`
        The random catalog after the "final" targeting columns (such as
        "DESI_TARGET", etc.) have been added.

    Notes
    -----
        - Typically used in conjunction with :func:`add_default_mtl()`
    """
    # ADM make every random the highest-priority target.
    dt = np.zeros_like(randoms["RA"]) + bitperprio[np.max(list(bitperprio))]

    return finalize(randoms, dt, dt*0, dt*0, randoms=True)


def add_default_mtl(randoms, seed):
    """Add default columns that are added by MTL.

    Parameters
    ----------
    randoms : :class:`~numpy.ndarray`
        A random catalog as made by, e.g., :func:`select_randoms()`
        with `nomtl=True` or `select_randoms_bricks()` with
        `nomtl=False`. This function adds the default MTL information.
    seed : :class:`int`
        A seed for the random generator that sets the `SUBPRIORITY`.

    Returns
    -------
    :class:`~numpy.array`
        The random catalog after being passed through MTL.

    Notes
    -----
        - Typically you will need to run :func:`finalize_randoms()`
          first, to populate the columns for the target bits.
    """
    from desitarget.mtl import make_mtl
    randoms = np.array(make_mtl(randoms, obscon="DARK"))

    # ADM add OBCONDITIONS that will work for any obscon.
    from desitarget.targetmask import obsconditions as obscon
    randoms["OBSCONDITIONS"] = obscon.mask("|".join(obscon.names()))

    # ADM add a random SUBPRIORITY.
    np.random.seed(616+seed)
    nrands = len(randoms)
    randoms["SUBPRIORITY"] = np.random.random(nrands)

    return randoms


def randoms_in_a_brick_from_edges(ramin, ramax, decmin, decmax, density=100000,
                                  poisson=True, wrap=True, seed=1):
    """For brick edges, return random (RA/Dec) positions in the brick.

    Parameters
    ----------
    ramin : :class:`float`
        The minimum "edge" of the brick in Right Ascension (degrees).
    ramax : :class:`float`
        The maximum "edge" of the brick in Right Ascension (degrees).
    decmin : :class:`float`
        The minimum "edge" of the brick in Declination (degrees).
    decmax : :class:`float`
        The maximum "edge" of the brick in Declination (degrees).
    density : :class:`int`, optional, defaults to 100,000
        The number of random points to return per sq. deg.
    poisson : :class:`boolean`, optional, defaults to ``True``
        Modify the number of random points so that instead of simply
        being brick area x density, the number is drawn from a Poisson
        distribution with an expectation of brick area x density.
    wrap : :class:`boolean`, optional, defaults to ``True``
        If ``True``, bricks with `ramax`-`ramin` > 350o are assumed to
        wrap, which is corrected by subtracting 360o from `ramax`, as is
        reasonable for small bricks. ``False`` turns of this correction.
    seed : :class:`int`, optional, defaults to 1
        Random seed to use when shuffling across brick boundaries.
        The actual np.random.seed defaults to:
            seed*int(1e7)+int(4*ramin)*1000+int(4*(decmin+90))
    Returns
    -------
    :class:`~numpy.array`
        Right Ascensions of random points in brick (degrees).
    :class:`~numpy.array`
        Declinations of random points in brick (degrees).
    """
    # ADM create a unique random seed on the basis of the brick.
    # ADM note this is only unique for bricksize=0.25 for bricks
    # ADM that are more than 0.25 degrees from the poles.
    uniqseed = seed*int(1e7)+int(4*ramin)*1000+int(4*(decmin+90))
    # ADM np.random only allows seeds < 2**32...
    maxseed = (2**32-int(4*ramin)*1000-int(4*(decmin+90)))/1e7
    if seed > maxseed:
        msg = 'seed must be < {} but you passed {}!!!'.format(maxseed, seed)
        log.critical(msg)
        raise ValueError(msg)
    np.random.seed(uniqseed)

    # ADM generate random points within the brick at the requested density
    # ADM guard against potential wraparound bugs (assuming bricks are typical
    # ADM sizes of 0.25 x 0.25 sq. deg., or not much larger than that.
    if wrap:
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
    decs = np.degrees(
        np.arcsin(1.-np.random.uniform(1-sindecmax, 1-sindecmin, nrand)))

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


def pre_or_post_dr8(drdir):
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

    Notes
    -----
        - If the directory structure seems wrong or missing then the DR8
          (and after) convention of a north/south split is assumed.
    """
    if os.path.exists(os.path.join(drdir, "coadd")):
        drdirs = [drdir]
    else:
        drdirs = [os.path.join(drdir, region) for region in ["north", "south"]]

    return drdirs


def dr8_quantities_at_positions_in_a_brick(ras, decs, brickname, drdir,
                                           aprad=0.75):
    """Wrap `quantities_at_positions_in_a_brick` for DR8 and beyond.

    Notes
    -----
    - See :func:`~desitarget.randoms.quantities_at_positions_in_a_brick`
      for details. This wrapper looks for TWO coadd directories in
      `drdir` (one for DECaLS, one for MzLS/BASS) and, if it finds two,
      creates randoms for both surveys within the the passed brick. The
      wrapper also defaults to the behavior for only having one survey.
    """
    # ADM determine if we must traverse two sets of brick directories.
    drdirs = pre_or_post_dr8(drdir)

    # ADM make the dictionary of quantities for one or two directories.
    qall = []
    for dd in drdirs:
        q = quantities_at_positions_in_a_brick(ras, decs, brickname, dd,
                                               aprad=aprad)
        # ADM don't count bricks where we never read a file header.
        if q is not None:
            qall.append(q)

    # ADM concatenate everything in qall into one dictionary.
    qcombine = {}
    # ADM catch the case where a coadd directory is completely missing.
    if len(qall) == 0:
        log.warning("missing brick: {}".format(brickname))
    else:
        for k in qall[0].keys():
            qcombine[k] = np.concatenate([q[k] for q in qall])

    return qcombine


def quantities_at_positions_in_a_brick(ras, decs, brickname, drdir,
                                       aprad=0.75, justlist=False):
    """Observational quantities (per-band) at positions in a Legacy Surveys brick.

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
    aprad : :class:`float`, optional, defaults to 0.75
        Radii in arcsec of aperture for which to derive sky/fiber fluxes.
        Defaults to the DESI fiber radius. If aprad < 1e-8 is passed,
        the code to produce these values is skipped, as a speed-up, and
        `apflux_` output values are set to zero.
    justlist : :class:`bool`, optional, defaults to ``False``
        If ``True``, return a MAXIMAL list of all POSSIBLE files needed
        to run for `brickname` and `drdir`. Overrides other inputs, but
        ra/dec still have to be passed as *something* (e.g., [1], [1]).

    Returns
    -------
    :class:`dictionary`
       The number of observations (`nobs_x`), PSF depth (`psfdepth_x`)
       galaxy depth (`galdepth_x`), PSF size (`psfsize_x`), sky
       background (`apflux_x`) and inverse variance (`apflux_ivar_x`)
       at each passed position in each band x=g,r,z. Plus, the
       `psfdepth_w1` and `_w2` depths and the `maskbits`, `wisemask_w1`
       and `_w2` information at each passed position for the brick.
       Also adds a unique `objid` for each random, a `release` if
       a release number can be determined from the input `drdir`, and
       the photometric system `photsys` ("N" or "S" for north or south).

    Notes
    -----
        - First version copied shamelessly from Anand Raichoor.
    """
    # ADM guard against too low a density of random locations.
    npts = len(ras)
    if npts == 0:
        msg = 'brick {} is empty. Increase the density of random points!'.format(brickname)
        log.critical(msg)
        raise ValueError(msg)

    # ADM a list to populate with the files required to run the code.
    fnlist = []

    # ADM determine whether the coadd files have extension .gz or .fz
    # based on the DR directory.
    extn, extn_nb = dr_extension(drdir)

    # ADM the output dictionary.
    qdict = {}

    # as a speed up, we assume all images in different filters for the brick have the same WCS
    # -> if we have read it once (iswcs=True), we use this info
    iswcs = False
    # ADM this will store the instrument name the first time we touch the wcs
    instrum = None

    rootdir = os.path.join(drdir, 'coadd', brickname[:3], brickname)
    fileform = os.path.join(rootdir, 'legacysurvey-{}-{}-{}.fits.{}')
    # ADM loop through the filters and store the number of observations
    # ADM etc. at the RA and Dec positions of the passed points.
    for filt in ['g', 'r', 'z']:
        # ADM the input file labels, and output column names and output
        # ADM formats for each of the quantities of interest.
        qnames = zip(['nexp', 'depth', 'galdepth', 'psfsize', 'image'],
                     ['nobs', 'psfdepth', 'galdepth', 'psfsize', 'apflux'],
                     ['i2', 'f4', 'f4', 'f4', 'f4'])
        for qin, qout, qform in qnames:
            fn = fileform.format(brickname, qin, filt, extn)
            if justlist:
                fnlist.append(fn)
            else:
                # ADM only process the WCS if there's a file for this filter.
                # ADM also skip calculating aperture fluxes if aprad ~ 0.
                if os.path.exists(fn) and not (qout == 'apflux' and aprad < 1e-8):
                    img = fits.open(fn)[extn_nb]
                    if not iswcs:
                        # ADM store the instrument name, if it isn't stored.
                        instrum = img.header["INSTRUME"].lower().strip()
                        w = WCS(img.header)
                        x, y = w.all_world2pix(ras, decs, 0)
                        iswcs = True
                    # ADM get the quantity of interest at each location and
                    # ADM store in a dictionary with the filter and quantity.
                    if qout == 'apflux':
                        # ADM special treatment to photometer sky.
                        # ADM Read in the ivar image.
                        fnivar = fileform.format(brickname, 'invvar', filt, extn)
                        ivar = fits.open(fnivar)[extn_nb].data
                        with np.errstate(divide='ignore', invalid='ignore'):
                            # ADM ivars->errors, guard against 1/0.
                            imsigma = 1./np.sqrt(ivar)
                            imsigma[ivar == 0] = 0
                        # ADM aperture photometry at requested radius (aprad).
                        apxy = np.vstack((x, y)).T
                        aper = photutils.CircularAperture(apxy, aprad)
                        p = photutils.aperture_photometry(img.data, aper, error=imsigma)
                        # ADM store the results.
                        qdict[qout+'_'+filt] = np.array(p.field('aperture_sum'))
                        err = p.field('aperture_sum_err')
                        with np.errstate(divide='ignore', invalid='ignore'):
                            # ADM errors->ivars, guard against 1/0.
                            ivar = 1./err**2.
                            ivar[err == 0] = 0.
                        qdict[qout+'_ivar_'+filt] = np.array(ivar)
                    else:
                        qdict[qout+'_'+filt] = img.data[y.astype("int"), x.astype("int")]
                # ADM if the file doesn't exist, set quantities to zero.
                else:
                    if qout == 'apflux':
                        qdict['apflux_ivar_'+filt] = np.zeros(npts, dtype=qform)
                    qdict[qout+'_'+filt] = np.zeros(npts, dtype=qform)

    # ADM add the MASKBITS and WISEMASK information.
    fn = os.path.join(rootdir,
                      'legacysurvey-{}-maskbits.fits.{}'.format(brickname, extn))
    # ADM only process the WCS if there's a file for this filter.
    mnames = zip([extn_nb, extn_nb+1, extn_nb+2],
                 ['maskbits', 'wisemask_w1', 'wisemask_w2'],
                 ['>i2', '|u1', '|u1'])
    if justlist:
        fnlist.append(fn)
    else:
        for mextn, mout, mform in mnames:
            if os.path.exists(fn):
                img = fits.open(fn)[mextn]
                # ADM use the WCS for the per-filter quantities if it exists.
                if not iswcs:
                    # ADM store the instrument name, if it isn't yet stored.
                    instrum = img.header["INSTRUME"].lower().strip()
                    w = WCS(img.header)
                    x, y = w.all_world2pix(ras, decs, 0)
                    iswcs = True
                # ADM add the maskbits to the dictionary.
                qdict[mout] = img.data[y.astype("int"), x.astype("int")]
            else:
                # ADM if no files are found, populate with zeros.
                qdict[mout] = np.zeros(npts, dtype=mform)
                # ADM if there was no maskbits file, populate with BAILOUT.
                if mout == 'maskbits':
                    qdict[mout] |= 2**10

    # ADM populate the photometric system in the quantity dictionary.
    if not justlist:
        if instrum is None:
            # ADM don't count bricks where we never read a file header.
            return
        elif instrum == 'decam':
            qdict['photsys'] = np.array([b"S" for x in range(npts)], dtype='|S1')
        else:
            qdict['photsys'] = np.array([b"N" for x in range(npts)], dtype='|S1')
#        log.info('Recorded quantities for each point in brick {}...t = {:.1f}s'
#                      .format(brickname,time()-start))

    # ADM calculate and add WISE depths. The WCS is different for WISE.
    iswcs = False
    # ADM a dictionary of scalings from invvar to depth:
    norm = {'W1': 0.240, 'W2': 0.255}
    # ADM a dictionary of Vega-to-AB conversions:
    vega_to_ab = {'W1': 2.699, 'W2': 3.339}
    for band in ['W1', 'W2']:
        # ADM the input file labels, and output column names and output
        # ADM formats for each of the quantities of interest.
        qnames = zip(['invvar'], ['psfdepth'], ['f4'])
        for qin, qout, qform in qnames:
            fn = fileform.format(brickname, qin, band, extn)
            if justlist:
                fnlist.append(fn)
            else:
                # ADM only process the WCS if there's a file for this band.
                if os.path.exists(fn):
                    img = fits.open(fn)[extn_nb]
                    # ADM calculate the WCS if it wasn't, already.
                    if not iswcs:
                        w = WCS(img.header)
                        x, y = w.all_world2pix(ras, decs, 0)
                        iswcs = True
                    # ADM get the inverse variance at each location.
                    ivar = img.data[y.astype("int"), x.astype("int")]
                    # ADM convert to WISE depth in AB. From Dustin Lang on the
                    # decam-chatter mailing list on 06/20/19, 1:59PM MST:
                    # psfdepth_Wx_AB = invvar_Wx * norm_Wx**2 / fluxfactor_Wx**2
                    # where fluxfactor = 10.** (dm / -2.5), dm = vega_to_ab[band]
                    ff = 10.**(vega_to_ab[band] / -2.5)
                    # ADM store in a dictionary with the band and quantity.
                    qdict[qout+'_'+band] = ivar * norm[band]**2 / ff**2
                # ADM if the file doesn't exist, set quantities to zero.
                else:
                    qdict[qout+'_'+band] = np.zeros(npts, dtype=qform)

    # ADM look up the RELEASE based on "standard" DR directory structure.
    if justlist:
        # ADM we need a tractor file. Then we have a list of all needed
        # ADM files. So, return if justlist was passed as True.
        tracdir = os.path.join(drdir, 'tractor', brickname[:3])
        tracfile = os.path.join(tracdir, 'tractor-{}.fits'.format(brickname))
        fnlist.append(tracfile)
        return fnlist

    # ADM populate the release number using a header from an nexp file.
    fn = fileform.format(brickname, "nexp", '*', extn)
    gen = iglob(fn)
    try:
        release = fitsio.read_header(next(gen), extn_nb)["DRVERSIO"]
    # ADM if this isn't a standard DR structure, default to release=0.
    except StopIteration:
        release = 0
    qdict["release"] = np.zeros_like((qdict['nobs_g'])) + release

    # ADM assign OBJID based on ordering by RA. The ordering ensures that
    # ADM northern and southern objects get the same OBJID.
    qdict["objid"] = np.argsort(ras)

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
        Brick names that corresponds to the brick edges, e.g., '1351p320'.
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
    # ADM this is only intended to work on one brick, so die if a larger
    # ADM array is passed.
    if not isinstance(brickname, str):
        log.fatal("Only one brick can be passed at a time!")
        raise ValueError

    # ADM generate an empty structured array to return in the event that
    # ADM no pixels with counts were found.
    hpxinfo = np.zeros(0, dtype=[('HPXPIXEL', '>i4'), ('HPXCOUNT', '>i4')])

    # ADM generate random points in the brick at the requested density.
    ras, decs = randoms_in_a_brick_from_edges(ramin, ramax, decmin, decmax,
                                              density=density, wrap=False)

    # ADM retrieve the number of observations for each random point.
    nobs_g, nobs_r, nobs_z = nobs_at_positions_in_a_brick(ras, decs, brickname,
                                                          drdir=drdir)

    # ADM only retain points with one or more observations in all bands.
    w = np.where((nobs_g > 0) & (nobs_r > 0) & (nobs_z > 0))

    # ADM for non-zero observations, populate pixel numbers and counts.
    if len(w[0]) > 0:
        pixnums = hp.ang2pix(nside, np.radians(90.-decs[w]), np.radians(ras[w]),
                             nest=True)
        pixnum, pixcnt = np.unique(pixnums, return_counts=True)
        hpxinfo = np.zeros(len(pixnum),
                           dtype=[('HPXPIXEL', '>i4'), ('HPXCOUNT', '>i4')])
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


def get_quantities_in_a_brick(ramin, ramax, decmin, decmax, brickname,
                              density=100000, dustdir=None, aprad=0.75,
                              zeros=False, drdir=None, seed=1):
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
        Brick names that corresponds to the brick edges, e.g., '1351p320'
    density : :class:`int`, optional, defaults to 100,000
        The number of random points to return per sq. deg. As a typical brick is
        ~0.25 x 0.25 sq. deg. about (0.0625*density) points will be returned
    dustdir : :class:`str`, optional, defaults to $DUST_DIR+'/maps'
        The root directory pointing to SFD dust maps. If not
        sent the code will try to use $DUST_DIR+'/maps' before failing.
    aprad : :class:`float`, optional, defaults to 0.75
        Radii in arcsec of aperture for which to derive sky/fiber fluxes.
        Defaults to the DESI fiber radius. If aprad < 1e-8 is passed,
        the code to produce these values is skipped, as a speed-up, and
        `apflux_` output values are set to zero.
    zeros : :class:`bool`, optional, defaults to ``False``
        If ``True`` don't look up pixel-level info for the brick, just
        return zeros. The only quantities populated are those that don't
        need pixels (`RA`, `DEC`, `BRICKID`, `BRICKNAME`, `EBV`) and the
        `NOBS_` quantities (which are set to zero).
    drdir : :class:`str`, optional, defaults to None
        The root directory pointing to a DR from the Legacy Surveys
        e.g. /global/project/projectdirs/cosmo/data/legacysurvey/dr7.
        Only necessary to pass if zeros is ``False``.
    seed : :class:`int`, optional, defaults to 1
        See :func:`~desitarget.randoms.randoms_in_a_brick_from_edges`.

    Returns
    -------
    :class:`~numpy.ndarray`
        a numpy structured array with the following columns:
            RELEASE: The Legacy Surveys release number.
            OBJID: A unique (to each brick) source identifier.
            BRICKID: ID that corresponds to the passed brick name.
            BRICKNAME: Passed brick name.
            RA, DEC: Right Ascension, Declination of a random location.
            NOBS_G, R, Z: Number of observations in g, r, z-band.
            PSFDEPTH_G, R, Z: PSF depth at this location in g, r, z.
            GALDEPTH_G, R, Z: Galaxy depth in g, r, z.
            PSFDEPTH_W1, W2: (PSF) depth in W1, W2 (AB mag system).
            PSFSIZE_G, R, Z: Weighted average PSF FWHM (arcsec).
            APFLUX_G, R, Z: Sky background extracted in `aprad`.
                Will be zero if `aprad` < 1e-8 is passed.
            APFLUX_IVAR_G, R, Z: Inverse variance of sky background.
                Will be zero if `aprad` < 1e-8 is passed.
            MASKBITS: mask information. See header of extension 1 of e.g.
              'coadd/132/1320p317/legacysurvey-1320p317-maskbits.fits.fz'
            WISEMASK_W1: mask info. See header of extension 2 of e.g.
              'coadd/132/1320p317/legacysurvey-1320p317-maskbits.fits.fz'
            WISEMASK_W2: mask info. See header of extension 3 of e.g.
              'coadd/132/1320p317/legacysurvey-1320p317-maskbits.fits.fz'
            EBV: E(B-V) at this location from the SFD dust maps.
            PHOTSYS: resolved north/south ('N' for an MzLS/BASS location,
              'S' for a DECaLS location).
    """
    # ADM only intended to work on one brick, so die for larger arrays.
    if not isinstance(brickname, str):
        log.fatal("Only one brick can be passed at a time!")
        raise ValueError

    # ADM generate random points in the brick at the requested density.
    ras, decs = randoms_in_a_brick_from_edges(ramin, ramax, decmin, decmax,
                                              density=density, wrap=False,
                                              seed=seed)

    # ADM only look up pixel-level quantities if zeros was not sent.
    if not zeros:
        # ADM retrieve the dictionary of quantities at each location.
        qdict = dr8_quantities_at_positions_in_a_brick(ras, decs, brickname,
                                                       drdir, aprad=aprad)

        # ADM catch where a coadd directory is completely missing.
        if len(qdict) > 0:
            # ADM if 2 different camera combinations overlapped a brick
            # ADM then we need to duplicate the ras, decs as well.
            if len(qdict['photsys']) == 2*len(ras):
                ras = np.concatenate([ras, ras])
                decs = np.concatenate([decs, decs])

        # ADM the structured array to output.
        qinfo = np.zeros(
            len(ras),
            dtype=[('RELEASE', '>i2'), ('BRICKID', '>i4'), ('BRICKNAME', 'S8'),
                   ('OBJID', '>i4'), ('RA', '>f8'), ('DEC', 'f8'),
                   ('NOBS_G', 'i2'), ('NOBS_R', 'i2'), ('NOBS_Z', 'i2'),
                   ('PSFDEPTH_G', 'f4'), ('PSFDEPTH_R', 'f4'), ('PSFDEPTH_Z', 'f4'),
                   ('GALDEPTH_G', 'f4'), ('GALDEPTH_R', 'f4'), ('GALDEPTH_Z', 'f4'),
                   ('PSFDEPTH_W1', 'f4'), ('PSFDEPTH_W2', 'f4'),
                   ('PSFSIZE_G', 'f4'), ('PSFSIZE_R', 'f4'), ('PSFSIZE_Z', 'f4'),
                   ('APFLUX_G', 'f4'), ('APFLUX_R', 'f4'), ('APFLUX_Z', 'f4'),
                   ('APFLUX_IVAR_G', 'f4'), ('APFLUX_IVAR_R', 'f4'), ('APFLUX_IVAR_Z', 'f4'),
                   ('MASKBITS', 'i2'), ('WISEMASK_W1', '|u1'), ('WISEMASK_W2', '|u1'),
                   ('EBV', 'f4'), ('PHOTSYS', '|S1')]
        )
    else:
        qinfo = np.zeros(
            len(ras),
            dtype=[('BRICKID', '>i4'), ('BRICKNAME', 'S8'), ('RA', 'f8'), ('DEC', 'f8'),
                   ('NOBS_G', 'i2'), ('NOBS_R', 'i2'), ('NOBS_Z', 'i2'),
                   ('EBV', 'f4')]
        )

    # ADM retrieve the E(B-V) values for each random point.
    ebv = get_dust(ras, decs, dustdir=dustdir)

    # ADM we only looked up pixel-level quantities if zeros wasn't sent.
    if not zeros:
        # ADM catch the case where a coadd directory was missing.
        if len(qdict) > 0:
            # ADM store each quantity of interest in the structured array
            # ADM remembering that the dictionary keys are lower-case text.
            cols = qdict.keys()
            for col in cols:
                qinfo[col.upper()] = qdict[col]

    # ADM add the RAs/Decs, brick id and brick name.
    brickid = bricklookup[brickname]
    qinfo["RA"], qinfo["DEC"] = ras, decs
    qinfo["BRICKNAME"], qinfo["BRICKID"] = brickname, brickid

    # ADM add the dust values.
    qinfo["EBV"] = ebv

    return qinfo


def pixweight(randoms, density, nobsgrz=[0, 0, 0], nside=256,
              outarea=True, maskbits=None):
    """Fraction of area covered in HEALPixels by a random catalog.

    Parameters
    ----------
    randoms : :class:`~numpy.ndarray` or `str`
        A random catalog as made by, e.g., :func:`select_randoms()` or
        :func:`quantities_at_positions_in_a_brick()`, or a file that
        contains such a catalog. Must contain the columns RA, DEC,
        NOBS_G, NOBS_R, NOBS_Z, MASKBITS.
    density : :class:`int`
        The number of random points per sq. deg. At which the random
        catalog was generated (see also :func:`select_randoms()`).
    nobsgrz : :class:`list`, optional, defaults to [0,0,0]
        The number of observations in each of g AND r AND z that must
        be EXCEEDED to include a random point in the count. The default
        is to include areas that have at least one observation in each
        band ([0,0,0]). `nobsgrz = [0,-1,-1]` would count areas with at
        least one (more than zero) observations in g-band but any number
        of observations (more than -1) in r-band and z-band.
    nside : :class:`int`, optional, defaults to nside=256
        The resolution (HEALPixel NESTED nside number) at which to build
        the map (default nside=256 is ~0.0525 sq. deg. or "brick-sized")
    outarea : :class:`boolean`, optional, defaults to True
        Print the total area of the survey for passed values to screen.
    maskbits : :class:`int`, optional, defaults to ``None``
        If not ``None`` then restrict to only locations with these
        values of maskbits NOT set (bit inclusive, so for, e.g., 7,
        restrict to random points with none of 2**0, 2**1 or 2**2 set).

    Returns
    -------
    :class:`~numpy.ndarray`
        The weight for EACH pixel in the sky at the passed nside.

    Notes
    -----
        - `WEIGHT=1` means >=1 pointings across the entire pixel.
        - `WEIGHT=0` means zero observations within it (e.g., perhaps
          the pixel is completely outside of the LS DR footprint).
        - `0 < WEIGHT < 1` for pixels that partially cover the LS DR
          area with one or more observations.
        - The index of the returned array is the HEALPixel integer.
    """
    # ADM if a file name was passed for the random catalog, read it in.
    if isinstance(randoms, str):
        randoms = fitsio.read(randoms)

    # ADM extract the columns of interest
    ras, decs = randoms["RA"], randoms["DEC"]
    nobs_g = randoms["NOBS_G"]
    nobs_r = randoms["NOBS_R"]
    nobs_z = randoms["NOBS_Z"]

    # ADM only retain points with one or more observations in all bands
    # ADM and appropriate maskbits values.
    ii = (nobs_g > nobsgrz[0])
    ii &= (nobs_r > nobsgrz[1])
    ii &= (nobs_z > nobsgrz[2])

    # ADM also restrict to appropriate maskbits values, if passed.
    if maskbits is not None:
        mb = randoms["MASKBITS"]
        ii &= (mb & maskbits) == 0

    # ADM the counts in each HEALPixel in the survey.
    if np.sum(ii) > 0:
        pixnums = hp.ang2pix(nside, np.radians(90.-decs[ii]),
                             np.radians(ras[ii]), nest=True)
        pixnum, pixcnt = np.unique(pixnums, return_counts=True)
    else:
        msg = "zero area based on randoms with passed constraints"
        log.error(msg)
        raise ValueError

    # ADM whole-sky-counts to retain zeros for zero survey coverage.
    npix = hp.nside2npix(nside)
    pix_cnt = np.bincount(pixnum, weights=pixcnt, minlength=npix)

    # ADM expected area based on the HEALPixels at this nside.
    expected_cnt = hp.nside2pixarea(nside, degrees=True)*density
    # ADM weight map based on (actual counts)/(expected counts).
    pix_weight = pix_cnt/expected_cnt

    # ADM if requested, print the total area of the survey to screen.
    if outarea:
        area = np.sum(pix_weight*hp.nside2pixarea(nside, degrees=True))
        if maskbits is None:
            log.info('Area of survey with NOBS > {} in [g,r,z] = {:.2f} sq. deg.'
                     .format(nobsgrz, area))
        else:
            log.info(
                'Area, NOBS > {} in [g,r,z], maskbits of {} = {:.2f} sq. deg.'
                .format(nobsgrz, maskbits, area))

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
    gaiadir = get_gaia_dir()
    hpdir = os.path.join(gaiadir, 'healpix')

    # ADM the number of pixels and the pixel area at nside.
    npix = hp.nside2npix(nside)
    pixarea = hp.nside2pixarea(nside, degrees=True)

    # ADM an output array of all possible HEALPixels at nside.
    pixout = np.zeros(npix, dtype='int32')

    # ADM find all of the Gaia files.
    filenames = glob(os.path.join(hpdir, '*fits'))

    # ADM read in each file, restricting to the criteria for point
    # ADM sources and storing in a HEALPixel map at resolution nside.
    nfiles = len(filenames)
    t0 = time()
    for nfile, filename in enumerate(filenames):
        if nfile % 1000 == 0 and nfile > 0:
            elapsed = time() - t0
            rate = nfile / elapsed
            log.info('{}/{} files; {:.1f} files/sec; {:.1f} total mins elapsed'
                     .format(nfile, nfiles, rate, elapsed/60.))

        # ADM save memory, speed up by only reading a subset of columns.
        gobjs = fitsio.read(
            filename,
            columns=['RA', 'DEC', 'PHOT_G_MEAN_MAG', 'ASTROMETRIC_EXCESS_NOISE']
        )

        # ADM restrict to subset of point sources.
        ra, dec = gobjs["RA"], gobjs["DEC"]
        gmag = gobjs["PHOT_G_MEAN_MAG"]
        excess = gobjs["ASTROMETRIC_EXCESS_NOISE"]
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


def get_targ_dens(targets, Mx, nside=256):
    """The density of targets in HEALPixels.

    Parameters
    ----------
    targets : :class:`~numpy.ndarray` or `str`
        A corresponding (same Legacy Surveys Data Release) target catalog as made by,
        e.g., :func:`desitarget.cuts.select_targets()`, or the name of such a file.
    Mx : :class:`list` or `~numpy.array`
        The targeting bitmasks associated with the passed targets, assumed to be
        a desi, bgs and mws mask in that order (for either SV or the main survey).
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

    # ADM retrieve the bitmasks.
    if Mx[0]._name == 'cmx_mask':
        msg = 'generating target densities does NOT work for CMX files!!!'
        log.critical(msg)
        raise ValueError(msg)
    else:
        desi_mask, bgs_mask, mws_mask = Mx

    # ADM the number of pixels and the pixel area at the passed nside
    npix = hp.nside2npix(nside)
    pixarea = hp.nside2pixarea(nside, degrees=True)

    # ADM retrieve the pixel numbers for every target RA/Dec
    ras, decs = targets["RA"], targets["DEC"]
    pixnums = hp.ang2pix(nside, np.radians(90.-decs), np.radians(ras), nest=True)

    # ADM retrieve the bit names of interest
    from desitarget.QA import _load_targdens
    bitnames = np.array(list(_load_targdens(bit_mask=Mx).keys()))

    # ADM and set up an array to hold the output target densities
    targdens = np.zeros(npix, dtype=[(bitname, 'f4') for bitname in bitnames])

    for bitname in bitnames:
        if 'ALL' in bitname:
            ii = np.ones(len(targets)).astype('bool')
        else:
            if ('BGS' in bitname) and not('S_ANY' in bitname):
                ii = targets["BGS_TARGET"] & bgs_mask[bitname] != 0
            elif (('MWS' in bitname or 'BACKUP' in bitname) and
                  not('S_ANY' in bitname)):
                ii = targets["MWS_TARGET"] & mws_mask[bitname] != 0
            else:
                ii = targets["DESI_TARGET"] & desi_mask[bitname] != 0

        if np.any(ii):
            # ADM calculate the number of objects in each pixel for the
            # ADM targets of interest
            pixnum, pixcnt = np.unique(pixnums[ii], return_counts=True)
            targdens[bitname][pixnum] = pixcnt/pixarea

    return targdens


def pixmap(randoms, targets, rand_density, nside=256, gaialoc=None):
    """HEALPix map of useful quantities for a Legacy Surveys Data Release

    Parameters
    ----------
    randoms : :class:`~numpy.ndarray` or `str`
        Catalog or file of randoms as made by :func:`select_randoms()` or
        :func:`quantities_at_positions_in_a_brick()`. Must contain the
        columns 'RA', 'DEC', 'EBV', 'PSFDEPTH_W1/W2/G/R/Z', 'NOBS_G/R/Z'
        'GALDEPTH_G/R/Z', 'PSFSIZE_G/R/Z', 'MASKBITS'.
    targets : :class:`~numpy.ndarray` or `str`
        Corresponding (i.e. same Data Release) catalog or file of targets
        as made by, e.g., :func:`desitarget.cuts.select_targets()`, or
        the the name of a directory containing HEALPix-split targets that
        can be read by :func:`desitarget.io.read_targets_in_box()`.
    rand_density : :class:`int`
        Number of random points per sq. deg. at which the random catalog
        was generated (see also :func:`select_randoms()`).
    nside : :class:`int`, optional, defaults to nside=256
        Resolution (HEALPix nside) at which to build the (NESTED) map.
        The default corresponds to ~0.0525 sq. deg. (or "brick-sized")
    gaialoc : :class:`str`, optional, defaults to ``None``
        Name of a FITS file that already contains a column "STARDENS",
        which is simply read in. If ``None``, the stellar density is
        constructed from files in $GAIA_DIR.

    Returns
    -------
    :class:`~numpy.ndarray`
        An array of useful information that includes
            - HPXPIXEL: HEALPixel integers at the passed `nside`.
            - FRACAREA: Fraction of pixel with at least one observation
                        in any band. Made with :func:`pixweight()`.
            - STARDENS: The stellar density in a pixel from Gaia. Made
                        with :func:`stellar_density()`.
            - EBV: E(B-V) in pixel from the SFD dust map, from the
                   median of EBV values in the passed `randoms`.
            - PSFDEPTH_G, R, Z: PSF depth in the pixel, from the median
                                of PSFDEPTH values in `randoms`.
            - GALDEPTH_G, R, Z: Galaxy depth in the pixel, from the
                                median of GALDEPTH values in `randoms`.
            - PSFDEPTH_W1, W2: (AB PSF) depth in the pixel, from the
                               median of values in the passed `randoms`.
            - PSFSIZE_G, R, Z: Weighted average PSF FWHM, in arcsec, in
                               the pixel, from the median of PSFSIZE
                               values in the passed random catalog.
            - FRACAREA_X: Fraction of pixel with at least one observation
                          in any band with MASKBITS==X (bitwise OR, so,
                          e.g. if X=7 then fraction for 2^0 | 2^1 | 2^2).
            - One column for every bit that is returned by
              :func:`desitarget.QA._load_targdens()`. Each column
              contains the target density in the pixel.
    :class:`str`
        Survey to which `targets` corresponds, e.g., 'main', 'svX', etc.

    Notes
    -----
        - If `gaialoc` is ``None`` then $GAIA_DIR must be set.
    """
    # ADM if a file name was passed for the random catalog, read it in
    if isinstance(randoms, str):
        log.info('Reading in random catalog...t = {:.1f}s'.format(time()-start))
        randoms = fitsio.read(randoms)

    # ADM if a file name was passed for the targets catalog, read it in
    if isinstance(targets, str):
        log.info('Reading in target catalog...t = {:.1f}s'.format(time()-start))
        # ADM grab appropriate columns for an SV/cmx/main survey file.
        targcols = target_columns_from_header(targets)
        cols = np.concatenate([["RA", "DEC"], targcols])
        targets = read_targets_in_box(targets, columns=cols)
    log.info('Read targets and randoms...t = {:.1f}s'.format(time()-start))

    # ADM change target column names, and retrieve associated survey information.
    _, Mx, survey, targets = main_cmx_or_sv(targets, rename=True)

    # ADM determine the areal coverage of the randoms at this nside.
    log.info('Determining footprint...t = {:.1f}s'.format(time()-start))
    pw = pixweight(randoms, rand_density, nside=nside)
    npix = len(pw)

    # ADM areal coverage for some combinations of MASKBITS.
    mbcomb = []
    mbstore = []
    for mb in [get_imaging_maskbits(get_default_maskbits()),
               get_imaging_maskbits(get_default_maskbits(bgs=True))]:
        bitint = np.sum(2**np.array(mb))
        mbcomb.append(bitint)
        log.info('Determining footprint for maskbits not in {}...t = {:.1f}s'
                 .format(bitint, time()-start))
        mbstore.append(pixweight(randoms, rand_density,
                                 nside=nside, maskbits=bitint))

    log.info('Determining footprint...t = {:.1f}s'.format(time()-start))
    pw = pixweight(randoms, rand_density, nside=nside)
    npix = len(pw)

    # ADM get the target densities.
    log.info('Calculating target densities...t = {:.1f}s'.format(time()-start))
    targdens = get_targ_dens(targets, Mx, nside=nside)

    # ADM set up the output array.
    datamodel = [('HPXPIXEL', '>i4'), ('FRACAREA', '>f4'), ('STARDENS', '>f4'), ('EBV', '>f4'),
                 ('PSFDEPTH_G', '>f4'), ('PSFDEPTH_R', '>f4'), ('PSFDEPTH_Z', '>f4'),
                 ('GALDEPTH_G', '>f4'), ('GALDEPTH_R', '>f4'), ('GALDEPTH_Z', '>f4'),
                 ('PSFDEPTH_W1', '>f4'), ('PSFDEPTH_W2', '>f4'),
                 ('PSFSIZE_G', '>f4'), ('PSFSIZE_R', '>f4'), ('PSFSIZE_Z', '>f4')]
    # ADM the maskbits-dependent areas.
    datamodel += [("FRACAREA_{}".format(bitint), '>f4') for bitint in mbcomb]
    # ADM the density of each target class.
    datamodel += targdens.dtype.descr
    hpxinfo = np.zeros(npix, dtype=datamodel)
    # ADM set initial values to -1 so that they can easily be clipped.
    hpxinfo[...] = -1

    # ADM add the areal coverage, pixel information and target densities.
    hpxinfo['HPXPIXEL'] = np.arange(npix)
    hpxinfo['FRACAREA'] = pw
    for bitint, fracarea in zip(mbcomb, mbstore):
        hpxinfo['FRACAREA_{}'.format(bitint)] = fracarea
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

    # ADM add the median values of all of the other systematics.
    log.info('Calculating medians of systematics from random catalog...t = {:.1f}s'
             .format(time()-start))
    ras, decs = randoms["RA"], randoms["DEC"]
    pixnums = hp.ang2pix(nside, np.radians(90.-decs), np.radians(ras), nest=True)

    # ADM some sorting to order the values to extract the medians.
    pixorder = np.argsort(pixnums)
    pixels, pixcnts = np.unique(pixnums, return_counts=True)
    pixcnts = np.insert(pixcnts, 0, 0)
    pixcnts = np.cumsum(pixcnts)
    log.info('Done sorting...t = {:.1f}s'.format(time()-start))
    # ADM work through the ordered pixels to populate the median for
    # ADM each quantity of interest.
    cols = ['EBV', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
            'PSFDEPTH_G', 'GALDEPTH_G', 'PSFSIZE_G',
            'PSFDEPTH_R', 'GALDEPTH_R', 'PSFSIZE_R',
            'PSFDEPTH_Z', 'GALDEPTH_Z', 'PSFSIZE_Z']
    t0 = time()
    npix = len(pixcnts)
    stepper = npix//50
    for i in range(npix-1):
        inds = pixorder[pixcnts[i]:pixcnts[i+1]]
        pix = pixnums[inds][0]
        for col in cols:
            hpxinfo[col][pix] = np.median(randoms[col][inds])
        if i % stepper == 0 and i > 0:
            elapsed = time() - t0
            rate = i / elapsed
            log.info('{}/{} pixels; {:.1f} pix/sec; {:.1f} total mins elapsed'
                     .format(i, npix, rate, elapsed/60.))

    log.info('Done...t = {:.1f}s'.format(time()-start))

    return hpxinfo, survey


def select_randoms_bricks(brickdict, bricknames, numproc=32, drdir=None,
                          zeros=False, nomtl=True, cnts=True, density=None,
                          dustdir=None, aprad=None, seed=1):

    """Parallel-process a random catalog for a set of brick names.

    Parameters
    ----------
    brickdict : :class:`dict`
        Look-up dictionary for a set of bricks, as made by, e.g.
        :func:`~desitarget.skyfibers.get_brick_info`.
    bricknames : :class:`~numpy.array`
        The names of the bricks in `brickdict` to process.
    drdir : :class:`str`, optional, defaults to None
        See :func:`~desitarget.randoms.get_quantities_in_a_brick`.
    zeros : :class:`bool`, optional, defaults to ``False``
        See :func:`~desitarget.randoms.get_quantities_in_a_brick`.
    nomtl : :class:`bool`, optional, defaults to ``True``
        If ``True`` then do NOT add MTL quantities to the output array.
    cnts : :class:`bool`, optional, defaults to ``True``
        See :func:`~desitarget.skyfibers.get_brick_info`.
    seed : :class:`int`, optional, defaults to 1
        See :func:`~desitarget.randoms.randoms_in_a_brick_from_edges`.

    Returns
    -------
    :class:`~numpy.ndarray`
        a numpy structured array with the same columns as returned by
        :func:`~desitarget.randoms.get_quantities_in_a_brick`. If
        `zeros` and `nomtl` are both ``False`` additional columns are
        returned, as added by :func:`~desitarget.targets.finalize`.

    Notes
    -----
    - See :func:`~desitarget.randoms.select_randoms` for definitions of
      `numproc`, `density`, `dustdir`, `aprad`.
    """
    nbricks = len(bricknames)
    log.info('Run {} bricks from {} at density {:.1e} per sq. deg...t = {:.1f}s'
             .format(nbricks, drdir, density, time()-start))

    # ADM the critical function to run on every brick.
    def _get_quantities(brickname):
        """wrapper on get_quantities_in_a_brick() given a brick name"""
        # ADM retrieve the edges for the brick that we're working on.
        if cnts:
            bra, bdec, bramin, bramax, bdecmin, bdecmax, _ = brickdict[brickname]
        else:
            bra, bdec, bramin, bramax, bdecmin, bdecmax = brickdict[brickname]

        # ADM populate the brick with random points, and retrieve the quantities
        # ADM of interest at those points.
        randoms = get_quantities_in_a_brick(
            bramin, bramax, bdecmin, bdecmax, brickname, drdir=drdir,
            density=density, dustdir=dustdir, aprad=aprad, zeros=zeros,
            seed=seed)

        if zeros or nomtl:
            return randoms
        return finalize_randoms(randoms)

    # ADM this is just to count bricks in _update_status.
    nbrick = np.zeros((), dtype='i8')
    t0 = time()
    # ADM write a total of 25 output messages during processing.
    interval = nbricks // 25

    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if nbrick % interval == 0 and nbrick > 0:
            elapsed = time() - t0
            rate = nbrick / elapsed
            log.info('{}/{} bricks; {:.1f} bricks/sec; {:.1f} total mins elapsed'
                     .format(nbrick, nbricks, rate, elapsed/60.))
            # ADM if we're going to exceed 4 hours, warn the user.
            if nbricks/rate > 4*3600.:
                msg = 'May take > 4 hours to run. May fail on interactive nodes.'
                log.warning(msg)

        nbrick[...] += 1    # this is an in-place modification.
        return result

    # - Parallel process input files.
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


def supplement_randoms(donebns, density=10000, numproc=32, dustdir=None,
                       seed=1):
    """Random catalogs of "zeros" for missing bricks.

    Parameters
    ----------
    donebns : :class:`~numpy.ndarray`
        Names of bricks that have been "completed". Bricks NOT in
        `donebns` will be returned without any pixel-level quantities.
        Need not be a unique set (bricks can be repeated in `donebns`).
    density : :class:`int`, optional, defaults to 10,000
        Number of random points per sq. deg. A typical brick is ~0.25 x
        0.25 sq. deg. so ~(0.0625*density) points will be returned.
    seed : :class:`int`, optional, defaults to 1
        See :func:`~desitarget.randoms.randoms_in_a_brick_from_edges`.
        A seed of 615 + `seed` is also used to shuffle randoms across
        brick boundaries.

    Returns
    -------
    :class:`~numpy.ndarray`
        a numpy structured array with the same columns returned by
        :func:`~desitarget.randoms.get_quantities_in_a_brick`
        when that function is passed zeros=True.

    Notes
    -----
    - See :func:`~desitarget.randoms.select_randoms` for definitions of
      `numproc`, `dustdir`.
    """
    # ADM find the missing bricks.
    brickdict = get_brick_info(None, allbricks=True)
    allbns = np.array(list(brickdict.keys()), dtype=donebns.dtype)
    bricknames = np.array(list(set(allbns) - set(donebns)), dtype='U')
    brickdict = {bn: brickdict[bn] for bn in bricknames}

    qzeros = select_randoms_bricks(brickdict, bricknames, numproc=numproc,
                                   zeros=True, cnts=False, density=density,
                                   dustdir=dustdir, seed=seed)

    # ADM one last shuffle to randomize across brick boundaries.
    np.random.seed(615+seed)
    np.random.shuffle(qzeros)

    return qzeros


def select_randoms(drdir, density=100000, numproc=32, nside=None, pixlist=None,
                   bundlebricks=None, nchunks=10, brickspersec=2.5, extra=None,
                   nomtl=True, dustdir=None, aprad=0.75, seed=1):
    """NOBS, DEPTHs (per-band), MASKs for random points in a Legacy Surveys DR.

    Parameters
    ----------
    drdir : :class:`str`
        Root directory for a Data Release from the Legacy Surveys
        e.g. /global/project/projectdirs/cosmo/data/legacysurvey/dr7.
    density : :class:`int`, optional, defaults to 100,000
        Number of random points to return per sq. deg. As a brick is
        ~0.25 x 0.25 sq. deg. ~0.0625*density points will be returned.
    numproc : :class:`int`, optional, defaults to 32
        The number of processes over which to parallelize.
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPixel nside to be used with the `pixlist` and
        `bundlebricks` input.
    pixlist : :class:`list` or `int`, optional, defaults to ``None``
        Bricks will only be processed if the brick CENTER is within the
        HEALpixels in this list, at the input `nside`. Uses the HEALPix
        NESTED scheme. Useful for parallelizing. If pixlist is ``None``
        then all bricks in the input `survey` will be processed.
    bundlebricks : :class:`int`, defaults to ``None``
        If not ``None``, then instead of selecting randoms, print a slurm
        script to balance the bricks at `bundlebricks` bricks per node.
    nchunks : :class:`int`, optional, defaults to 10
        Number of smaller catalogs to split the random catalog into
        inside the `bundlebricks` slurm script.
    brickspersec : :class:`float`, optional, defaults to 2.5
        The rough number of bricks processed per second (parallelized
        across a chosen number of nodes). Used with `bundlebricks` to
        estimate time to completion when parallelizing across pixels.
    extra : :class:`str`, optional
        Extra command line flags to be passed to the executable lines in
        the output slurm script. Used in conjunction with `bundlefiles`.
    nomtl : :class:`bool`, optional, defaults to ``True``
        If ``True`` then do NOT add MTL quantities to the output array.
    dustdir : :class:`str`, optional, defaults to $DUST_DIR+'maps'
        The root directory pointing to SFD dust maps. If None the code
        will try to use $DUST_DIR+'maps') before failing.
    aprad : :class:`float`, optional, defaults to 0.75
        Radii in arcsec of aperture for which to derive sky/fiber fluxes.
        Defaults to the DESI fiber radius. If aprad < 1e-8 is passed,
        the code to produce these values is skipped, as a speed-up, and
        `apflux_` output values are set to zero.
    seed : :class:`int`, optional, defaults to 1
        Random seed to use when shuffling across brick boundaries.
        The actual np.random.seed defaults to 615+`seed`. See also use
        in :func:`~desitarget.randoms.randoms_in_a_brick_from_edges`.

    Returns
    -------
    :class:`~numpy.ndarray`
        a numpy structured array with the same columns as returned by
        :func:`~desitarget.randoms.get_quantities_in_a_brick` that
        includes all of the randoms resolved by the north/south divide.
    :class:`~numpy.ndarray`
        as above but just for randoms in northern bricks.
    :class:`~numpy.ndarray`
        as above but just for randoms in southern bricks.
    """
    # ADM grab brick information for this data release. Depending on whether this
    # ADM is pre-or-post-DR8 we need to find the correct directory or directories.
    drdirs = pre_or_post_dr8(drdir)
    brickdict = get_brick_info(drdirs, counts=True)
    # ADM this is just the UNIQUE brick names across all surveys.
    bricknames = np.array(list(brickdict.keys()))

    # ADM if the pixlist or bundlebricks option was sent, we'll need the HEALPixel
    # ADM information for each brick.
    if pixlist is not None or bundlebricks is not None:
        bra, bdec, _, _, _, _, cnts = np.vstack(list(brickdict.values())).T
        theta, phi = np.radians(90-bdec), np.radians(bra)
        pixnum = hp.ang2pix(nside, theta, phi, nest=True)

    # ADM if the bundlebricks option was sent, call the packing code.
    if bundlebricks is not None:
        # ADM pixnum only contains unique bricks, need to add duplicates.
        allpixnum = np.concatenate([np.zeros(cnt, dtype=int)+pix for
                                    cnt, pix in zip(cnts.astype(int), pixnum)])
        bundle_bricks(allpixnum, bundlebricks, nside, gather=True, seed=seed,
                      brickspersec=brickspersec, prefix='randoms',
                      surveydirs=[drdir], extra=extra, nchunks=nchunks)
        # ADM because the broader function returns three outputs.
        return None, None, None

    # ADM restrict to only bricks in a set of HEALPixels, if requested.
    if pixlist is not None:
        # ADM if an integer was passed, turn it into a list.
        if isinstance(pixlist, int):
            pixlist = [pixlist]
        ii = [pix in pixlist for pix in pixnum]
        bricknames = bricknames[ii]
        if len(bricknames) == 0:
            log.warning('ZERO bricks in passed pixel list!!!')
        log.info("Processing bricks in (nside={}, pixel numbers={}) HEALPixels"
                 .format(nside, pixlist))

    # ADM a little more information if we're slurming across nodes.
    if os.getenv('SLURMD_NODENAME') is not None:
        log.info('Running on Node {}'.format(os.getenv('SLURMD_NODENAME')))

    # ADM recover the pixel-level quantities in the DR bricks.
    randoms = select_randoms_bricks(brickdict, bricknames, numproc=numproc,
                                    drdir=drdir, density=density, nomtl=nomtl,
                                    dustdir=dustdir, aprad=aprad, seed=seed)

    # ADM add columns that are added by MTL.
    if nomtl is False:
        randoms = add_default_mtl(randoms, seed)

    # ADM one last shuffle to randomize across brick boundaries.
    np.random.seed(615+seed)
    np.random.shuffle(randoms)

    # ADM remove bricks that overlap between two surveys.
    randomsres = resolve(randoms)

    # ADM a flag for which targets are from the 'N' photometry.
    from desitarget.cuts import _isonnorthphotsys
    isn = _isonnorthphotsys(randoms["PHOTSYS"])

    return randomsres, randoms[isn], randoms[~isn]
