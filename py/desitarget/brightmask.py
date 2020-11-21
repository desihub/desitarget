# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.brightmask
=====================

Module for studying and masking bright sources in the sweeps

.. _`Tech Note 2346`: https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=2346
.. _`Tech Note 2348`: https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=2348
"""
from time import time
import fitsio
import healpy as hp
import os
import re
from glob import glob

import numpy as np
import numpy.lib.recfunctions as rfn

from astropy.coordinates import SkyCoord
from astropy import units as u

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from desitarget import io
from desitarget.internal import sharedmem
from desitarget.targetmask import desi_mask, targetid_mask
from desitarget.targets import encode_targetid, decode_targetid
from desitarget.gaiamatch import find_gaia_files, get_gaia_nside_brick
from desitarget.geomask import circles, cap_area, circle_boundaries, is_in_hp
from desitarget.geomask import ellipses, ellipse_boundary, is_in_ellipse
from desitarget.geomask import radec_match_to, rewind_coords, add_hp_neighbors
from desitarget.cuts import _psflike
from desitarget.tychomatch import get_tycho_dir, get_tycho_nside
from desitarget.tychomatch import find_tycho_files_hp
from desitarget.gfa import add_urat_pms

from desiutil import depend, brick
# ADM set up default logger
from desiutil.log import get_logger

# ADM fake the matplotlib display so it doesn't die on allocated nodes.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt    # noqa: E402

log = get_logger()

maskdatamodel = np.array([], dtype=[
    ('RA', '>f8'), ('DEC', '>f8'), ('PMRA', '>f4'), ('PMDEC', '>f4'),
    ('REF_CAT', '|S2'), ('REF_ID', '>i8'), ('REF_MAG', '>f4'),
    ('URAT_ID', '>i8'), ('IN_RADIUS', '>f4'), ('NEAR_RADIUS', '>f4'),
    ('E1', '>f4'), ('E2', '>f4'), ('TYPE', '|S3')
])


def get_mask_dir():
    """Convenience function to grab the MASK_DIR environment variable.

    Returns
    -------
    :class:`str`
        The directory stored in the $MASK_DIR environment variable.
    """
    # ADM check that the $MASK_DIR environment variable is set.
    maskdir = os.environ.get('MASK_DIR')
    if maskdir is None:
        msg = "Set $MASK_DIR environment variable!"
        log.critical(msg)
        raise ValueError(msg)

    return maskdir


def get_recent_mask_dir(input_dir=None):
    """Grab the most recent sub-directory of masks in MASK_DIR.

    Parameters
    ----------
    input_dir : :class:`str`, optional, defaults to ``None``
        If passed and not ``None``, then this is returned as the output.

    Returns
    -------
    :class:`str`
        If `input_dir` is not ``None``, then the most recently created
        sub-directory (with the appropriate format for a mask directory)
        in $MASK_DIR is returned.
    """
    if input_dir is not None:
        return input_dir
    else:
        # ADM glob the most recent mask directory.
        try:
            md = os.environ["MASK_DIR"]
        except KeyError:
            msg = "pass a mask directory, turn off masking, or set $MASK_DIR!"
            log.error(msg)
            raise IOError(msg)
        # ADM a fairly exhaustive list of possible mask directories.
        mds = glob(os.path.join(md, "*maglim*")) + \
            glob(os.path.join(md, "*/*maglim*")) + \
            glob(os.path.join(md, "*/*/*maglim*"))
        if len(mds) == 0:
            msg = "no mask sub-directories found in {}".format(md)
            log.error(msg)
            raise IOError(msg)
        return max(mds, key=os.path.getctime)


def radii(mag):
    """The relation used to set the radius of bright star masks.

    Parameters
    ----------
    mag : :class:`flt` or :class:`recarray`
        Magnitude. Typically, in order of preference, G-band for Gaia
        or VT then HP then BT for Tycho.

    Returns
    -------
    :class:`recarray`
        The `IN_RADIUS`, corresponding to the `IN_BRIGHT_OBJECT` bit
        in `data/targetmask.yaml`.
    :class:`recarray`
        The `NEAR_RADIUS`, corresponding to the `NEAR_BRIGHT_OBJECT` bit
        in data/targetmask.yaml`.
    """
    # ADM mask all sources with mag < 12 at 5 arcsecs.
    inrad = (mag < 12.) * 5.
    # ADM the NEAR_RADIUS is twice the IN_RADIUS.
    nearrad = inrad*2.

    return inrad, nearrad


def _rexlike(rextype):
    """If the object is REX (a round exponential galaxy)"""

    # ADM explicitly checking for an empty input.
    if rextype is None:
        log.error("NoneType submitted to _rexlike function")

    rextype = np.asarray(rextype)
    # ADM in Python3 these string literals become byte-like
    # ADM so to retain Python2 compatibility we need to check
    # ADM against both bytes and unicode.
    # ADM also 'REX' for astropy.io.fits; 'REX ' for fitsio (sigh).
    rexlike = ((rextype == 'REX') | (rextype == b'REX') |
               (rextype == 'REX ') | (rextype == b'REX '))
    return rexlike


def max_objid_bricks(targs):
    """For a set of targets, return the maximum value of BRICK_OBJID in each BRICK_ID

    Parameters
    ----------
    targs : :class:`recarray`
        A recarray of targets as made by :mod:`desitarget.cuts.select_targets`

    Returns
    -------
    maxobjid : :class:`dictionary`
        A dictionary with keys for each unique BRICKID and values of the maximum OBJID in that brick
    """

    # ADM the maximum BRICKID in the passed target set.
    brickmax = np.max(targs["BRICKID"])

    # ADM how many OBJIDs are in each unique brick, starting from 0 and ordered on BRICKID.
    h = np.histogram(targs["BRICKID"], range=[0, brickmax], bins=brickmax)[0]
    # ADM remove zero entries from the histogram.
    h = h[np.where(h > 0)]
    # ADM the index of the maximum OBJID in eacn brick if the bricks are ordered on BRICKID and OBJID.
    maxind = np.cumsum(h)-1

    # ADM an array of BRICKID, OBJID sorted first on BRICKID and then on OBJID within each BRICKID.
    ordered = np.array(sorted(zip(targs["BRICKID"], targs["BRICK_OBJID"]), key=lambda x: (x[0], x[1])))

    # ADM return a dictionary of the maximum OBJID (values) for each BRICKID (keys).
    return dict(ordered[maxind])


def make_bright_star_mask_in_hp(nside, pixnum, verbose=True, gaiaepoch=2015.5,
                                maglim=12., matchrad=1., maskepoch=2023.0):
    """Make a bright star mask in a HEALPixel using Tycho, Gaia and URAT.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    pixnum : :class:`int`
        A single HEALPixel number.
    verbose : :class:`bool`
        If ``True`` then log informational messages.

    Returns
    -------
    :class:`recarray`
        The bright star mask in the form of `maskdatamodel.dtype`.

    Notes
    -----
        - Runs in a a minute or so for a typical nside=4 pixel.
        - See :func:`~desitarget.brightmask.make_bright_star_mask` for
          descriptions of the output mask and the other input parameters.
    """
    # ADM start the clock.
    t0 = time()

    # ADM read in the Tycho files.
    tychofns = find_tycho_files_hp(nside, pixnum, neighbors=False)
    tychoobjs = []
    for fn in tychofns:
        tychoobjs.append(fitsio.read(fn, ext='TYCHOHPX'))
    tychoobjs = np.concatenate(tychoobjs)
    # ADM create the Tycho reference magnitude, which is VT then HP
    # ADM then BT in order of preference.
    tychomag = tychoobjs["MAG_VT"].copy()
    tychomag[tychomag == 0] = tychoobjs["MAG_HP"][tychomag == 0]
    tychomag[tychomag == 0] = tychoobjs["MAG_BT"][tychomag == 0]
    # ADM discard any Tycho objects below the input magnitude limit
    # ADM and outside of the HEALPixels of interest.
    theta, phi = np.radians(90-tychoobjs["DEC"]), np.radians(tychoobjs["RA"])
    tychohpx = hp.ang2pix(nside, theta, phi, nest=True)
    ii = (tychohpx == pixnum) & (tychomag < maglim)
    tychomag, tychoobjs = tychomag[ii], tychoobjs[ii]
    if verbose:
        log.info('Read {} (mag < {}) Tycho objects (pix={})...t={:.1f} mins'.
                 format(np.sum(ii), maglim, pixnum, (time()-t0)/60))

    # ADM read in the associated Gaia files. Also grab
    # ADM neighboring pixels to prevent edge effects.
    gaiafns = find_gaia_files(tychoobjs, neighbors=True)
    gaiaobjs = []
    cols = 'SOURCE_ID', 'RA', 'DEC', 'PHOT_G_MEAN_MAG', 'PMRA', 'PMDEC'
    for fn in gaiafns:
        if os.path.exists(fn):
            gaiaobjs.append(fitsio.read(fn, ext='GAIAHPX', columns=cols))

    gaiaobjs = np.concatenate(gaiaobjs)
    gaiaobjs = rfn.rename_fields(gaiaobjs, {"SOURCE_ID": "REF_ID"})
    # ADM limit Gaia objects to 3 magnitudes fainter than the passed
    # ADM limit. This leaves some (!) leeway when matching to Tycho.
    gaiaobjs = gaiaobjs[gaiaobjs['PHOT_G_MEAN_MAG'] < maglim + 3]
    if verbose:
        log.info('Read {} (G < {}) Gaia sources (pix={})...t={:.1f} mins'.format(
            len(gaiaobjs), maglim+3, pixnum, (time()-t0)/60))

    # ADM substitute URAT where Gaia proper motions don't exist.
    ii = ((np.isnan(gaiaobjs["PMRA"]) | (gaiaobjs["PMRA"] == 0)) &
          (np.isnan(gaiaobjs["PMDEC"]) | (gaiaobjs["PMDEC"] == 0)))
    if verbose:
        log.info('Add URAT for {} Gaia objs with no PMs (pix={})...t={:.1f} mins'
                 .format(np.sum(ii), pixnum, (time()-t0)/60))

    urat = add_urat_pms(gaiaobjs[ii], numproc=1)
    if verbose:
        log.info('Found an additional {} URAT objects (pix={})...t={:.1f} mins'
                 .format(np.sum(urat["URAT_ID"] != -1), pixnum, (time()-t0)/60))
    for col in "PMRA", "PMDEC":
        gaiaobjs[col][ii] = urat[col]
    # ADM need to track the URATID to track which objects have
    # ADM substituted proper motions.
    uratid = np.zeros_like(gaiaobjs["REF_ID"])-1
    uratid[ii] = urat["URAT_ID"]

    # ADM match to remove Tycho objects already in Gaia. Prefer the more
    # ADM accurate Gaia proper motions. Note, however, that Tycho epochs
    # ADM can differ from the mean (1991.5) by as as much as 0.86 years,
    # ADM so a star with a proper motion as large as Barnard's Star
    # ADM (10.3 arcsec) can be off by a significant margin (~10").
    margin = 10.
    ra, dec = rewind_coords(gaiaobjs["RA"], gaiaobjs["DEC"],
                            gaiaobjs["PMRA"], gaiaobjs["PMDEC"],
                            epochnow=gaiaepoch)
    # ADM match Gaia to Tycho with a suitable margin.
    if verbose:
        log.info('Match Gaia to Tycho with margin={}" (pix={})...t={:.1f} mins'
                 .format(margin, pixnum, (time()-t0)/60))
    igaia, itycho = radec_match_to([ra, dec],
                                   [tychoobjs["RA"], tychoobjs["DEC"]],
                                   sep=margin, radec=True)
    if verbose:
        log.info('{} matches. Refining at 1" (pix={})...t={:.1f} mins'.format(
            len(itycho), pixnum, (time()-t0)/60))

    # ADM match Gaia to Tycho at the more exact reference epoch.
    epoch_ra = tychoobjs[itycho]["EPOCH_RA"]
    epoch_dec = tychoobjs[itycho]["EPOCH_DEC"]
    # ADM some of the Tycho epochs aren't populated.
    epoch_ra[epoch_ra == 0], epoch_dec[epoch_dec == 0] = 1991.5, 1991.5
    ra, dec = rewind_coords(gaiaobjs["RA"][igaia], gaiaobjs["DEC"][igaia],
                            gaiaobjs["PMRA"][igaia], gaiaobjs["PMDEC"][igaia],
                            epochnow=gaiaepoch,
                            epochpast=epoch_ra, epochpastdec=epoch_dec)
    # ADM catch the corner case where there are no initial matches.
    if ra.size > 0:
        _, refined = radec_match_to([ra, dec], [tychoobjs["RA"][itycho],
                                    tychoobjs["DEC"][itycho]], radec=True)
    else:
        refined = np.array([], dtype='int')
    # ADM retain Tycho objects that DON'T match Gaia.
    keep = np.ones(len(tychoobjs), dtype='bool')
    keep[itycho[refined]] = False
    tychokeep, tychomag = tychoobjs[keep], tychomag[keep]
    if verbose:
        log.info('Kept {} Tychos with no Gaia match (pix={})...t={:.1f} mins'
                 .format(len(tychokeep), pixnum, (time()-t0)/60))

    # ADM now we're done matching to Gaia, limit Gaia to the passed
    # ADM magnitude limit and to the HEALPixel boundary of interest.
    theta, phi = np.radians(90-gaiaobjs["DEC"]), np.radians(gaiaobjs["RA"])
    gaiahpx = hp.ang2pix(nside, theta, phi, nest=True)
    ii = (gaiahpx == pixnum) & (gaiaobjs['PHOT_G_MEAN_MAG'] < maglim)
    gaiakeep, uratid = gaiaobjs[ii], uratid[ii]
    if verbose:
        log.info('Mask also comprises {} Gaia sources (pix={})...t={:.1f} mins'
                 .format(len(gaiakeep), pixnum, (time()-t0)/60))

    # ADM move the coordinates forwards to the input mask epoch.
    epoch_ra, epoch_dec = tychokeep["EPOCH_RA"], tychokeep["EPOCH_DEC"]
    # ADM some of the Tycho epochs aren't populated.
    epoch_ra[epoch_ra == 0], epoch_dec[epoch_dec == 0] = 1991.5, 1991.5
    ra, dec = rewind_coords(
        tychokeep["RA"], tychokeep["DEC"], tychokeep["PM_RA"], tychokeep["PM_DEC"],
        epochnow=epoch_ra, epochnowdec=epoch_dec, epochpast=maskepoch)
    tychokeep["RA"], tychokeep["DEC"] = ra, dec
    ra, dec = rewind_coords(
        gaiakeep["RA"], gaiakeep["DEC"], gaiakeep["PMRA"], gaiakeep["PMDEC"],
        epochnow=gaiaepoch, epochpast=maskepoch)
    gaiakeep["RA"], gaiakeep["DEC"] = ra, dec

    # ADM finally, format according to the mask data model...
    gaiamask = np.zeros(len(gaiakeep), dtype=maskdatamodel.dtype)
    tychomask = np.zeros(len(tychokeep), dtype=maskdatamodel.dtype)
    for col in "RA", "DEC":
        gaiamask[col] = gaiakeep[col]
        gaiamask["PM"+col] = gaiakeep["PM"+col]
        tychomask[col] = tychokeep[col]
        tychomask["PM"+col] = tychokeep["PM_"+col]
    gaiamask["REF_ID"] = gaiakeep["REF_ID"]
    # ADM take care to rigorously convert to int64 for Tycho.
    tychomask["REF_ID"] = tychokeep["TYC1"].astype('int64')*int(1e6) + \
        tychokeep["TYC2"].astype('int64')*10 + tychokeep["TYC3"]
    gaiamask["REF_CAT"], tychomask["REF_CAT"] = 'G2', 'T2'
    gaiamask["REF_MAG"] = gaiakeep['PHOT_G_MEAN_MAG']
    tychomask["REF_MAG"] = tychomag
    gaiamask["URAT_ID"], tychomask["URAT_ID"] = uratid, -1
    gaiamask["TYPE"], tychomask["TYPE"] = 'PSF', 'PSF'
    mask = np.concatenate([gaiamask, tychomask])
    # ADM ...and add the mask radii.
    mask["IN_RADIUS"], mask["NEAR_RADIUS"] = radii(mask["REF_MAG"])

    if verbose:
        log.info("Done making mask...(pix={})...t={:.1f} mins".format(
            pixnum, (time()-t0)/60.))

    return mask


def make_bright_star_mask(maglim=12., matchrad=1., numproc=32,
                          maskepoch=2023.0, gaiaepoch=2015.5,
                          nside=None, pixels=None):
    """Make an all-sky bright star mask using Tycho, Gaia and URAT.

    Parameters
    ----------
    maglim : :class:`float`, optional, defaults to 12.
        Faintest magnitude at which to make the mask. This magnitude is
        interpreted as G-band for Gaia and, in order of preference, VT
        then HP then BT for Tycho (not every Tycho source has each band).
    matchrad : :class:`int`, optional, defaults to 1.
        Tycho sources that match a Gaia source at this separation in
        ARCSECONDS are NOT included in the output mask. The matching is
        performed rigorously, accounting for Gaia proper motions.
    numproc : :class:`int`, optional, defaults to 16.
        Number of processes over which to parallelize
    maskepoch : :class:`float`
        The mask is built at this epoch. Not all sources have proper
        motions from every survey, so proper motions are used, in order
        of preference, from Gaia, URAT, then Tycho.
    gaiaepoch : :class:`float`, optional, defaults to Gaia DR2 (2015.5)
        The epoch of the Gaia observations. Should be 2015.5 unless we
        move beyond Gaia DR2.
    nside : :class:`int`, optional, defaults to ``None``
        If passed, create a mask only in nested HEALPixels in `pixels`
        at this `nside`. Otherwise, run for the whole sky. If `nside`
        is passed then `pixels` must be passed too.
    pixels : :class:`list`, optional, defaults to ``None``
        If passed, create a mask only in nested HEALPixels at `nside` for
        pixel integers in `pixels`. Otherwise, run for the whole sky. If
        `pixels` is passed then `nside` must be passed too.

    Returns
    -------
    :class:`recarray`
        - The bright star mask in the form of `maskdatamodel.dtype`:
        - `REF_CAT` is `"T2"` for Tycho and `"G2"` for Gaia.
        - `REF_ID` is `Tyc1`*1,000,000+`Tyc2`*10+`Tyc3` for Tycho2;
          `"sourceid"` for Gaia-DR2 and Gaia-DR2 with URAT.
        - `REF_MAG` is, in order of preference, G-band for Gaia, VT
          then HP then BT for Tycho.
        - `URAT_ID` contains the URAT reference number for Gaia objects
          that use the URAT proper motion, or -1 otherwise.
        - The radii are in ARCSECONDS.
        - `E1` and `E2` are placeholders for ellipticity components, and
          are set to 0 for Gaia and Tycho sources.
        - `TYPE` is always `PSF` for star-like objects.
        - Note that the mask is based on objects in the pixel AT THEIR
          NATIVE EPOCH *NOT* AT THE INPUT `maskepoch`. It is therefore
          possible for locations in the output mask to be just beyond
          the boundaries of the input pixel.

    Notes
    -----
        - Runs (all-sky) in ~20 minutes for `numproc=32` and `maglim=12`.
        - `IN_RADIUS` (`NEAR_RADIUS`) corresponds to `IN_BRIGHT_OBJECT`
          (`NEAR_BRIGHT_OBJECT`) in `data/targetmask.yaml`. These radii
          are set in the function `desitarget.brightmask.radius()`.
        - The correct mask size for DESI is an open question.
        - The `GAIA_DIR`, `URAT_DIR` and `TYCHO_DIR` environment
          variables must be set.
    """
    log.info("running on {} processors".format(numproc))

    # ADM check if HEALPixel parameters have been correctly sent.
    io.check_both_set(pixels, nside)

    # ADM grab the nside of the Tycho files, which is a reasonable
    # ADM resolution for bright stars.
    if nside is None:
        nside = get_tycho_nside()
        npixels = hp.nside2npix(nside)
        # ADM array of HEALPixels over which to parallelize...
        pixels = np.arange(npixels)
        # ADM ...shuffle for better balance across nodes (as there are
        # ADM more stars in regions of the sky where pixels adjoin).
    np.random.shuffle(pixels)

    # ADM the common function that is actually parallelized across.
    def _make_bright_star_mx(pixnum):
        """returns bright star mask in one HEALPixel"""
        return make_bright_star_mask_in_hp(
            nside, pixnum, maglim=maglim, matchrad=matchrad,
            gaiaepoch=gaiaepoch, maskepoch=maskepoch, verbose=False)

    # ADM this is just to count pixels in _update_status.
    npix = np.zeros((), dtype='i8')
    t0 = time()

    def _update_status(result):
        """wrap key reduction operation on the main parallel process"""
        if npix % 10 == 0 and npix > 0:
            rate = (time() - t0) / npix
            log.info('{}/{} HEALPixels; {:.1f} secs/pixel...t = {:.1f} mins'.
                     format(npix, npixels, rate, (time()-t0)/60.))
        npix[...] += 1
        return result

    # ADM Parallel process across HEALPixels.
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            mask = pool.map(_make_bright_star_mx, pixels, reduce=_update_status)
    else:
        mask = list()
        for pixel in pixels:
            mask.append(_update_status(_make_bright_star_mx(pixel)))

    mask = np.concatenate(mask)

    log.info("Done making mask...t = {:.1f} mins".format((time()-t0)/60.))

    return mask


def plot_mask(mask, limits=None, radius="IN_RADIUS", show=True):
    """Plot a mask or masks.

    Parameters
    ----------
    mask : :class:`recarray`
        A mask, as constructed by, e.g. :func:`make_bright_star_mask()`.
    limits : :class:`list`, optional
        RA/Dec plot limits in the form [ramin, ramax, decmin, decmax].
    radius : :class: `str`, optional
        Which mask radius to plot (``IN_RADIUS`` or ``NEAR_RADIUS``).
    show : :class:`boolean`
        If ``True``, then display the plot, Otherwise, just execute the
        plot commands so it can be added to or saved to file later.

    Returns
    -------
    Nothing
    """
    # ADM make this work even for a single mask.
    mask = np.atleast_1d(mask)

    # ADM set up the plot.
    fig, ax = plt.subplots(1, figsize=(8, 8))

    plt.xlabel('RA (o)')
    plt.ylabel('Dec (o)')

    # ADM set up some default plot limits if they weren't passed.
    if limits is None:
        maskra, maskdec, tol = mask["RA"], mask["DEC"], mask[radius]/3600.
        limits = [np.max(maskra-tol), np.min(maskra+tol),
                  np.min(maskdec-tol), np.max(maskdec+tol)]
    ax.axis(limits)

    # ADM only consider a limited mask range corresponding to a few
    # ADM times the largest mask radius beyond the requested limits.
    # ADM remember that the passed mask sizes are in arcseconds.
    tol = 3.*np.max(mask[radius])/3600.
    # ADM the np.min/np.max combinations are to guard against people
    # ADM passing flipped RAs (so RA increases to the east).
    ii = ((mask["RA"] > np.min(limits[:2])-tol) &
          (mask["RA"] < np.max(limits[:2])+tol) &
          (mask["DEC"] > np.min(limits[-2:])-tol) &
          (mask["DEC"] < np.max(limits[-2:])+tol))
    if np.sum(ii) == 0:
        msg = 'No mask entries within specified limits ({})'.format(limits)
        log.error(msg)
        raise ValueError(msg)
    else:
        mask = mask[ii]

    # ADM create ellipse polygons for each entry in the mask and
    # ADM make a list of matplotlib patches for them.
    patches = []
    for i, ellipse in enumerate(mask):
        # ADM create points on the ellipse boundary.
        ras, decs = ellipse_boundary(
            ellipse["RA"], ellipse["DEC"],
            ellipse[radius], ellipse["E1"], ellipse["E2"])
        polygon = Polygon(np.array(list(zip(ras, decs))), True)
        patches.append(polygon)

    p = PatchCollection(patches, alpha=0.4, facecolors='b', edgecolors='b')
    ax.add_collection(p)

    if show:
        plt.show()

    return


def is_in_bright_mask(targs, sourcemask, inonly=False):
    """Determine whether a set of targets is in a bright star mask.

    Parameters
    ----------
    targs : :class:`recarray`
        A recarray of targets, skies etc., as made by, e.g.,
        :func:`desitarget.cuts.select_targets()`.
    sourcemask : :class:`recarray`
        A recarray containing a mask as made by, e.g.,
        :func:`desitarget.brightmask.make_bright_star_mask()`
    inonly : :class:`boolean`, optional, defaults to False
        If ``True``, then only calculate the `in_mask` return but not
        the `near_mask` return, which is about a factor of 2 faster.

    Returns
    -------
    :class:`list`
        [`in_mask`, `near_mask`] where `in_mask` (`near_mask`) is a
        boolean array that is ``True`` for `targs` that are IN (NEAR) a
        mask. If `inonly` is ``True`` then this is just [`in_mask`].
    :class: `list`
        [`used_in_mask`, `used_near_mask`] where `used_in_mask`
        (`used_near_mask`) is a boolean array that is ``True`` for masks
        in `sourcemask` that contain a target at the IN (NEAR) radius.
        If `inonly` is ``True`` then this is just [`used_in_mask`].
    """
    t0 = time()

    # ADM initialize arrays of all False (nothing is yet in a mask).
    in_mask = np.zeros(len(targs), dtype=bool)
    near_mask = np.zeros(len(targs), dtype=bool)
    used_in_mask = np.zeros(len(sourcemask), dtype=bool)
    used_near_mask = np.zeros(len(sourcemask), dtype=bool)

    # ADM turn the mask and target coordinates into SkyCoord objects.
    ctargs = SkyCoord(targs["RA"]*u.degree, targs["DEC"]*u.degree)
    cmask = SkyCoord(sourcemask["RA"]*u.degree, sourcemask["DEC"]*u.degree)

    # ADM this is the largest search radius we should need to consider.
    # ADM In the future an obvious speed up is to split on radius
    # ADM as large radii are rarer but take longer.
    maxrad = max(sourcemask["IN_RADIUS"])*u.arcsec
    if not inonly:
        maxrad = max(sourcemask["NEAR_RADIUS"])*u.arcsec

    # ADM coordinate match the masks and the targets.
    # ADM assuming all of the masks are circles-on-the-sky.
    idtargs, idmask, d2d, d3d = cmask.search_around_sky(ctargs, maxrad)

    # ADM catch the case where nothing fell in a mask.
    if len(idmask) == 0:
        if inonly:
            return [in_mask], [used_in_mask]
        return [in_mask, near_mask], [used_in_mask, used_near_mask]

    # ADM need to differentiate targets that are in ellipse-on-the-sky
    # ADM masks from targets that are in circle-on-the-sky masks.
    rex_or_psf = _rexlike(sourcemask[idmask]["TYPE"]) | _psflike(
        sourcemask[idmask]["TYPE"])
    w_ellipse = np.where(~rex_or_psf)

    # ADM only continue if there are any elliptical masks.
    if len(w_ellipse[0]) > 0:
        idelltargs = idtargs[w_ellipse]
        idellmask = idmask[w_ellipse]

        log.info('Testing {} targets against {} elliptical masks...t={:.1f}s'
                 .format(len(set(idelltargs)), len(set(idellmask)), time()-t0))

        # ADM to speed the calculation, make a dictionary of which
        # ADM targets (the values) associate with each mask (the keys).
        targidineachmask = {}
        # ADM first initiate a list for each relevant key (mask ID).
        for maskid in set(idellmask):
            targidineachmask[maskid] = []
        # ADM then append those lists until they contain the IDs of each
        # ADM relevant target as the values.
        for index, targid in enumerate(idelltargs):
            targidineachmask[idellmask[index]].append(targid)

        # ADM loop through the masks and determine which relevant points
        # ADM occupy them for both the IN_RADIUS and the NEAR_RADIUS.
        for maskid in targidineachmask:
            targids = targidineachmask[maskid]
            ellras, elldecs = targs[targids]["RA"], targs[targids]["DEC"]
            mask = sourcemask[maskid]
            # ADM Refine being in a mask based on the elliptical masks.
            in_ell = is_in_ellipse(
                ellras, elldecs, mask["RA"], mask["DEC"],
                mask["IN_RADIUS"], mask["E1"], mask["E2"])
            in_mask[targids] |= in_ell
            used_in_mask[maskid] |= np.any(in_ell)
            if not inonly:
                in_ell = is_in_ellipse(ellras, elldecs,
                                       mask["RA"], mask["DEC"],
                                       mask["NEAR_RADIUS"],
                                       mask["E1"], mask["E2"])
                near_mask[targids] |= in_ell
                used_near_mask[maskid] |= np.any(in_ell)

        log.info('Done with elliptical masking...t={:1f}s'.format(time()-t0))

    # ADM Finally, record targets in a circles-on-the-sky mask, which
    # ADM trumps any information about just being in an elliptical mask.
    # ADM Find separations less than the mask radius for circle masks
    # ADM matches meeting these criteria are in at least one circle mask.
    w_in = (d2d.arcsec < sourcemask[idmask]["IN_RADIUS"]) & (rex_or_psf)
    in_mask[idtargs[w_in]] = True
    used_in_mask[idmask[w_in]] = True

    if not inonly:
        w_near = (d2d.arcsec < sourcemask[idmask]["NEAR_RADIUS"]) & (rex_or_psf)
        near_mask[idtargs[w_near]] = True
        used_near_mask[idmask[w_near]] = True
        return [in_mask, near_mask], [used_in_mask, used_near_mask]

    return [in_mask], [used_in_mask]


def is_bright_source(targs, sourcemask):
    """Determine whether targets are, themselves, a bright source mask.

    Parameters
    ----------
    targs : :class:`recarray`
        Targets as made by, e.g., :func:`desitarget.cuts.select_targets()`.
    sourcemask : :class:`recarray`
        A recarray containing a bright source mask as made by, e.g.,
        :func:`desitarget.brightmask.make_bright_star_mask()`

    Returns
    -------
    is_mask : array_like
        ``True`` for `targs` that are, themselves, a mask.
    """

    # ADM initialize an array of all False (nothing yet has been shown
    # ADM to correspond to a mask).
    is_mask = np.zeros(len(targs), dtype=bool)

    # ADM calculate the TARGETID for the targets.
    targetid = encode_targetid(objid=targs['BRICK_OBJID'],
                               brickid=targs['BRICKID'],
                               release=targs['RELEASE'])

    # ADM super-fast set-based look-up of which TARGETIDs are match
    # ADM between the masks and the targets.
    matches = set(sourcemask["TARGETID"]).intersection(set(targetid))
    # ADM indexes of the targets that have a TARGETID in matches.
    w_mask = [index for index, item in enumerate(targetid) if item in matches]

    # ADM w_mask now holds target indices that match a mask on TARGETID.
    is_mask[w_mask] = True

    return is_mask


def generate_safe_locations(sourcemask, Nperradius=1):
    """Given a mask, generate SAFE (BADSKY) locations at its periphery.

    Parameters
    ----------
    sourcemask : :class:`recarray`
        A recarray containing a bright mask as made by, e.g.,
        :func:`desitarget.brightmask.make_bright_star_mask()`
    Nperradius : :class:`int`, optional, defaults to 1.
        Number of safe locations to make per arcsec radius of each mask.

    Returns
    -------
    ra : array_like.
        The Right Ascensions of the SAFE (BADSKY) locations.
    dec : array_like.
        The Declinations of the SAFE (BADSKY) locations.

    Notes
    -----
        - See `Tech Note 2346`_ for details.
    """

    # ADM the radius of each mask in arcseconds with a 0.1% kick to
    # ADM ensure that positions are beyond the mask edges.
    radius = sourcemask["IN_RADIUS"]*1.001

    # ADM determine the number of SAFE locations to assign to each
    # ADM mask given the passed number of locations per unit radius.
    Nsafe = np.ceil(radius*Nperradius).astype('i')

    # ADM need to differentiate targets that are in ellipse-on-the-sky masks
    # ADM from targets that are in circle-on-the-sky masks.
    rex_or_psf = _rexlike(sourcemask["TYPE"]) | _psflike(sourcemask["TYPE"])
    w_ellipse = np.where(~rex_or_psf)
    w_circle = np.where(rex_or_psf)

    # ADM set up an array to hold coordinates around the mask peripheries.
    ras, decs = np.array([]), np.array([])

    # ADM generate the safe location for circular masks (which is quicker).
    if len(w_circle[0]) > 0:
        circras, circdecs = circle_boundaries(sourcemask[w_circle]["RA"],
                                              sourcemask[w_circle]["DEC"],
                                              radius[w_circle], Nsafe[w_circle])
        ras, decs = np.concatenate((ras, circras)), np.concatenate((decs, circdecs))

    # ADM generate the safe location for elliptical masks
    # ADM (which is slower as it requires a loop).
    if len(w_ellipse[0]) > 0:
        for w in w_ellipse[0]:
            ellras, elldecs = ellipse_boundary(sourcemask[w]["RA"],
                                               sourcemask[w]["DEC"], radius[w],
                                               sourcemask[w]["E1"],
                                               sourcemask[w]["E2"], Nsafe[w])
            ras = np.concatenate((ras, ellras))
            decs = np.concatenate((decs, elldecs))

    return ras, decs


def get_safe_targets(targs, sourcemask, bricks_are_hpx=False):
    """Get SAFE (BADSKY) locations for targs, set TARGETID/DESI_TARGET.

    Parameters
    ----------
    targs : :class:`~numpy.ndarray`
        Targets made by, e.g. :func:`desitarget.cuts.select_targets()`.
    sourcemask : :class:`~numpy.ndarray`
        A bright source mask as made by, e.g.
        :func:`desitarget.brightmask.make_bright_star_mask()`.
    bricks_are_hpx : :class:`bool`, optional, defaults to ``False``
        Instead of using bricks to calculate BRICKIDs, use HEALPixels at
        the "standard" size from :func:`gaiamatch.get_gaia_nside_brick()`.

    Returns
    -------
    :class:`~numpy.ndarray`
        SAFE (BADSKY) locations for `targs` with the same data model as
        for `targs`.

    Notes
    -----
        - `Tech Note 2346`_ details SAFE (BADSKY) locations.
        - `Tech Note 2348`_ details setting the SKY bit in TARGETID.
        - Hard-coded to create 1 safe location per arcsec of mask radius.
          The correct number (Nperradius) for DESI is an open question.
    """
    # ADM number of safe locations per radial arcsec of each mask.
    Nperradius = 1

    # ADM grab SAFE locations around masks at a density of Nperradius.
    ra, dec = generate_safe_locations(sourcemask, Nperradius)

    # ADM duplicate targs data model for safe locations.
    nrows = len(ra)
    safes = np.zeros(nrows, dtype=targs.dtype)
    # ADM return early if there are no safe locations.
    if nrows == 0:
        return safes

    # ADM populate the safes with the RA/Dec of the SAFE locations.
    safes["RA"] = ra
    safes["DEC"] = dec

    # ADM set the bit for SAFE locations in DESITARGET.
    safes["DESI_TARGET"] |= desi_mask.BAD_SKY

    # ADM add the brick information for the SAFE/BADSKY targets.
    if bricks_are_hpx:
        nside = get_gaia_nside_brick()
        theta, phi = np.radians(90-safes["DEC"]), np.radians(safes["RA"])
        safes["BRICKID"] = hp.ang2pix(nside, theta, phi, nest=True)
        safes["BRICKNAME"] = 'hpxat{}'.format(nside)
    else:
        b = brick.Bricks(bricksize=0.25)
        safes["BRICKID"] = b.brickid(safes["RA"], safes["DEC"])
        safes["BRICKNAME"] = b.brickname(safes["RA"], safes["DEC"])

    # ADM now add OBJIDs, counting backwards from the maximum possible
    # ADM OBJID to ensure no duplicateion of TARGETIDs for supplemental
    # ADM skies, which build their OBJIDs by counting forwards from 0.
    maxobjid = 2**targetid_mask.OBJID.nbits - 1
    sortid = np.argsort(safes["BRICKID"])
    _, cnts = np.unique(safes["BRICKID"], return_counts=True)
    brickids = np.concatenate([maxobjid-np.arange(i) for i in cnts])
    safes["BRICK_OBJID"][sortid] = brickids

    # ADM finally, update the TARGETID.
    # ADM first, check the GAIA DR number for these skies.
    _, _, _, _, _, gdr = decode_targetid(targs["TARGETID"])
    if len(set(gdr)) != 1:
        msg = "Skies are based on multiple Gaia Data Releases:".format(set(gdr))
        log.critical(msg)
        raise ValueError(msg)

    safes["TARGETID"] = encode_targetid(objid=safes['BRICK_OBJID'],
                                        brickid=safes['BRICKID'],
                                        sky=1,
                                        gaiadr=gdr[0])

    # ADM return the input targs with the SAFE targets appended.
    return safes


def set_target_bits(targs, sourcemask, return_masks=False):
    """Apply bright source mask to targets, return desi_target array.

    Parameters
    ----------
    targs : :class:`recarray`
        Targets as made by, e.g., :func:`desitarget.cuts.select_targets()`.
    sourcemask : :class:`recarray`
        A recarray containing a bright source mask as made by, e.g.
        :mod:`desitarget.brightmask.make_bright_star_mask` or
        :mod:`desitarget.brightmask.make_bright_source_mask`.
    return_masks : :class:`bool`
        If ``True`` also return boolean arrays of which of the
        masks in `sourcemask` contain a target.


    Returns
    -------
    :class:`recarray`
        `DESI_TARGET` column updates with bright source information bits.
    :class:`list`, only returned if `return_masks` is ``True``
        [`used_in_mask`, `used_near_mask`] where `used_in_mask`
        (`used_near_mask`) is a boolean array that is ``True`` for masks
        in `sourcemask` that contain a target at the IN (NEAR) radius.

    Notes
    -----
        - Sets ``IN_BRIGHT_OBJECT`` and ``NEAR_BRIGHT_OBJECT`` via
          matches to circular and/or elliptical masks.
        - Sets ``BRIGHT_OBJECT`` via an index match on ``TARGETID``
          (defined as in :func:`desitarget.targets.encode_targetid()`).

    See :mod:`desitarget.targetmask` for the definition of each bit.
    """
    if "TARGETID" in sourcemask.dtype.names:
        bright_object = is_bright_source(targs, sourcemask)
    else:
        bright_object = 0

    intargs, inmasks = is_in_bright_mask(targs, sourcemask)
    in_bright_object, near_bright_object = intargs

    desi_target = targs["DESI_TARGET"].copy()

    desi_target |= bright_object * desi_mask.BRIGHT_OBJECT
    desi_target |= in_bright_object * desi_mask.IN_BRIGHT_OBJECT
    desi_target |= near_bright_object * desi_mask.NEAR_BRIGHT_OBJECT

    if return_masks:
        return desi_target, inmasks
    return desi_target


def mask_targets(targs, inmaskdir, nside=2, pixlist=None, bricks_are_hpx=False):
    """Add bits for if objects occupy masks, and SAFE (BADSKY) locations.

    Parameters
    ----------
    targs : :class:`str` or `~numpy.ndarray`
        An array of targets/skies etc. created by, e.g.,
        :func:`desitarget.cuts.select_targets()` OR the filename of a
        file that contains such a set of targets/skies, etc.
    inmaskdir : :class:`str`, optional
        An input bright star mask file or HEALPixel-split directory as
        made by :func:`desitarget.brightmask.make_bright_star_mask()`
    nside : :class:`int`, optional, defaults to 2
        The nside at which the targets were generated. If the mask is
        a HEALPixel-split directory, then this helps to perform more
        efficient masking as only the subset of masks that are in
        pixels containing `targs` at this `nside` will be considered
        (together with neighboring pixels to account for edge effects).
    pixlist : :class:`list` or `int`, optional
        A set of HEALPixels corresponding to the `targs`. Only the subset
        of masks in HEALPixels in `pixlist` at `nside` will be considered
        (together with neighboring pixels to account for edge effects).
        If ``None``, then the pixels touched by `targs` is derived from
        from `targs` itself.
    bricks_are_hpx : :class:`bool`, optional, defaults to ``False``
        Instead of using bricks to calculate BRICKIDs, use HEALPixels at
        the "standard" size from :func:`gaiamatch.get_gaia_nside_brick()`.

    Returns
    -------
    :class:`~numpy.ndarray`
        Input targets with the `DESI_TARGET` column updated to reflect
        the `BRIGHT_OBJECT` bits and SAFE (`BADSKY`) sky locations added
        around the perimeter of the mask.

    Notes
    -----
        - `Tech Note 2346`_ details SAFE (BADSKY) locations.
    """
    t0 = time()

    # ADM Check if targs is a file name or the structure itself.
    if isinstance(targs, str):
        if not os.path.exists(targs):
            raise ValueError("{} doesn't exist".format(targs))
        targs = fitsio.read(targs)

    # ADM determine which pixels are occupied by targets.
    if pixlist is None:
        theta, phi = np.radians(90-targs["DEC"]), np.radians(targs["RA"])
        pixlist = list(set(hp.ang2pix(nside, theta, phi, nest=True)))
    else:
        # ADM in case an integer was passed.
        pixlist = np.atleast_1d(pixlist)
    log.info("Masking using masks in {} at nside={} in HEALPixels={}".format(
        inmaskdir, nside, pixlist))
    pixlistwneigh = add_hp_neighbors(nside, pixlist)

    # ADM read in the (potentially HEALPixel-split) mask.
    sourcemask = io.read_targets_in_hp(inmaskdir, nside, pixlistwneigh)

    ntargs = len(targs)
    log.info('Total number of masks {}'.format(len(sourcemask)))
    log.info('Total number of targets {}...t={:.1f}s'.format(ntargs, time()-t0))

    # ADM update the bits depending on whether targets are in a mask.
    # ADM also grab masks that contain or are near a target.
    dt, mx = set_target_bits(targs, sourcemask, return_masks=True)
    targs["DESI_TARGET"] = dt
    inmasks, nearmasks = mx

    # ADM generate SAFE locations for masks that contain a target.
    safes = get_safe_targets(targs, sourcemask[inmasks],
                             bricks_are_hpx=bricks_are_hpx)

    # ADM update the bits for the safe locations depending on whether
    # ADM they're in a mask.
    safes["DESI_TARGET"] = set_target_bits(safes, sourcemask)
    # ADM it's possible that a safe location was generated outside of
    # ADM the requested HEALPixels.
    inhp = is_in_hp(safes, nside, pixlist)
    safes = safes[inhp]

    # ADM combine the targets and safe locations.
    done = np.concatenate([targs, safes])

    # ADM assert uniqueness of TARGETIDs.
    stargs, ssafes = set(targs["TARGETID"]), set(safes["TARGETID"])
    msg = "TARGETIDs for targets not unique"
    assert len(stargs) == len(targs), msg
    msg = "TARGETIDs for safes not unique"
    assert len(ssafes) == len(safes), msg
    msg = "TARGETIDs for safes duplicated in targets. Generating TARGETIDs"
    msg += " backwards from maxobjid in get_safe_targets() has likely failed"
    msg += " due to somehow generating a large number of safe locations."
    assert len(stargs.intersection(ssafes)) == 0, msg

    log.info('Generated {} SAFE (BADSKY) locations...t={:.1f}s'.format(
        len(done)-ntargs, time()-t0))

    # ADM remove any SAFE locations that are in bright masks (because they aren't really safe).
    ii = (((done["DESI_TARGET"] & desi_mask.BAD_SKY) == 0) |
          ((done["DESI_TARGET"] & desi_mask.IN_BRIGHT_OBJECT) == 0))
    done = done[ii]

    log.info("...of these, {} SAFE (BADSKY) locations aren't in masks...t={:.1f}s"
             .format(len(done)-ntargs, time()-t0))

    log.info('Finishing up...t={:.1f}s'.format(time()-t0))

    return done
