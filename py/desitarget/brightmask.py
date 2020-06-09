# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.brightmask
=====================

Module for studying and masking bright sources in the sweeps

.. _`Tech Note 2346`: https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=2346
.. _`Tech Note 2348`: https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=2348
.. _`the DR5 sweeps`: http://legacysurvey.org/dr5/files/#sweep-catalogs
.. _`Legacy Surveys catalogs`: http://legacysurvey.org/dr5/catalogs/
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
from desitarget.targets import encode_targetid
from desitarget.gaiamatch import find_gaia_files
from desitarget.geomask import circles, cap_area, circle_boundaries
from desitarget.geomask import ellipses, ellipse_boundary, is_in_ellipse
from desitarget.geomask import radec_match_to, rewind_coords
from desitarget.cuts import _psflike
from desitarget.tychomatch import get_tycho_dir, get_tycho_nside
from desitarget.tychomatch import find_tycho_files_hp
from desitarget.gfa import add_urat_pms

from desiutil import depend, brick
# ADM set up default logger
from desiutil.log import get_logger
log = get_logger()

# ADM fake the matplotlib display so it doesn't die on allocated nodes.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt    # noqa: E402

# ADM factor by which the "in" radius is smaller than the "near" radius
infac = 0.5
# ADM and vice-versa.
nearfac = 1./infac
# ADM size of the bright star mask (NEAR_RADIUS) in arcseconds.
masksize = 5.

maskdatamodel = np.array([], dtype=[
    ('RA', '>f8'), ('DEC', '>f8'),
    ('REF_CAT', '|S2'), ('REF_ID', '>i8'), ('REF_MAG', '>f4'),
    ('URAT_ID', '>i8'), ('IN_RADIUS', '>f4'), ('NEAR_RADIUS', '>f4'),
    ('E1', '>f4'), ('E2', '>f4'), ('TYPE', '|S3')
])


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


def make_bright_star_mask_in_hp(nside, pixnum,
                                maglim=12., matchrad=1., epoch=2023.0):
    """Make a bright star mask in a HEALPixel using Tycho, Gaia and URAT.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    pixnum : :class:`int`
        A single HEALPixel number.
    maglim : :class:`float`, optional, defaults to 12.
        Faintest magnitude at which to make the mask. This magnitude is
        interpreted as G-band for Gaia and, in order of preference, VT
        then HP then BT for Tycho (not every Tycho source has each band).
    matchrad : :class:`int`, optional, defaults to 1.
        Tycho sources that match a Gaia source at this separation in
        ARCSECONDS are NOT included in the output mask. The matching is
        performed rigorously, accounting for Gaia proper motions.
    epoch : :class:`float`
        The mask is built at this epoch. Not all sources have proper
        motions from every survey, so proper motions are used, in order
        of preference, from Gaia, URAT, then Tycho.

    Returns
    -------
    :class:`recarray`
        - The bright source mask in the form of `maskdatamodel.dtype`:
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

    Notes
    -----
        - `IN_RADIUS` (`NEAR_RADIUS`) corresponds to `IN_BRIGHT_OBJECT`
          (`NEAR_BRIGHT_OBJECT`) in `data/targetmask.yaml`. These radii
          are set in the function `desitarget.brightmask.radius()`.
        - The correct mask size for DESI is an open question.
        - The `GAIA_DIR`, `URAT_DIR` and `TYCHO_DIR` environment
          variables must be set.
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
    # ADM discard any Tycho objects below the input magnitude limit.
    ii = tychomag < maglim
    tychomag, tychoobjs = tychomag[ii], tychoobjs[ii]
    log.info('Read {} (mag < {}) Tycho objects...t = {:.1f} mins'.format(
        np.sum(ii), maglim, (time()-t0)/60))

    # ADM read in the associated Gaia files. Also grab
    # ADM neighboring pixels to prevent edge effects.
    gaiafns = find_gaia_files(tychoobjs, neighbors=True)
    gaiaobjs = []
    cols = 'SOURCE_ID', 'RA', 'DEC', 'PHOT_G_MEAN_MAG', 'PMRA', 'PMDEC'
    for fn in gaiafns:
        gaiaobjs.append(fitsio.read(fn, ext='GAIAHPX', columns=cols))
    gaiaobjs = np.concatenate(gaiaobjs)
    gaiaobjs = rfn.rename_fields(gaiaobjs, {"SOURCE_ID":"REF_ID"})
    # ADM limit Gaia objects to 3 magnitudes fainter than the passed
    # ADM limit. This leaves some (!) leeway when matching to Tycho.
    gaiaobjs = gaiaobjs[gaiaobjs['PHOT_G_MEAN_MAG'] < maglim + 3]
    log.info('Read {} (G < {}) Gaia sources...t = {:.1f} mins'.format(
        len(gaiaobjs), maglim + 3, (time()-t0)/60))

    # ADM substitute URAT where Gaia proper motions don't exist.
    ii = ((np.isnan(gaiaobjs["PMRA"]) | (gaiaobjs["PMRA"] == 0)) &
          (np.isnan(gaiaobjs["PMDEC"]) | (gaiaobjs["PMDEC"] == 0)))
    log.info('Add URAT for {} Gaia objects with no PMs...t = {:.1f} mins'.format(
        np.sum(ii), (time()-t0)/60))
    urat = add_urat_pms(gaiaobjs[ii], numproc=1)
    log.info('Found an additional {} URAT objects...t = {:.1f} mins'.format(
        np.sum(urat["URAT_ID"] != -1), (time()-t0)/60))
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
                            gaiaobjs["PMRA"], gaiaobjs["PMDEC"])
    # ADM match Gaia to Tycho with a suitable margin.
    log.info('Match Gaia to Tycho with margin={}"...t = {:.1f} mins'.format(
        margin, (time()-t0)/60))
    igaia, itycho = radec_match_to([ra, dec],
                                   [tychoobjs["RA"], tychoobjs["DEC"]],
                                   sep=margin, radec=True)
    log.info('{} matches. Refining at 1"...t = {:.1f} mins'.format(
        len(itycho), (time()-t0)/60))
    # ADM match Gaia to Tycho at the more exact reference epoch.
    epoch_ra = tychoobjs[itycho]["EPOCH_RA"]
    epoch_dec = tychoobjs[itycho]["EPOCH_DEC"]
    # ADM some of the Tycho epochs aren't populated.
    epoch_ra[epoch_ra == 0], epoch_dec[epoch_dec == 0] = 1991.5, 1991.5
    ra, dec = rewind_coords(gaiaobjs["RA"][igaia], gaiaobjs["DEC"][igaia],
                            gaiaobjs["PMRA"][igaia], gaiaobjs["PMDEC"][igaia],
                            epochpast=epoch_ra, epochpastdec=epoch_dec)
    _, refined = radec_match_to([ra, dec], [tychoobjs["RA"][itycho],
                                tychoobjs["DEC"][itycho]], radec=True)
    # ADM retain Tycho objects that DON'T match Gaia.
    keep = np.ones(len(tychoobjs), dtype='bool')
    keep[itycho[refined]] = False
    tychokeep, tychomag = tychoobjs[keep], tychomag[keep]
    log.info('Kept {} Tycho sources with no Gaia match...t = {:.1f} mins'.format(
        len(tychokeep), (time()-t0)/60))

    # ADM now we're done matching to Gaia, limit Gaia to the passed
    # ADM magnitude limit and to the HEALPixel boundary of interest.
    theta, phi = np.radians(90-gaiaobjs["DEC"]), np.radians(gaiaobjs["RA"])
    gaiahpx = hp.ang2pix(nside, theta, phi, nest=True)
    ii = (gaiahpx == pixnum) & (gaiaobjs['PHOT_G_MEAN_MAG'] < maglim)
    gaiakeep, uratid = gaiaobjs[ii], uratid[ii]

    # ADM finally, format according to the mask data model...
    gaiamask = np.zeros(len(gaiakeep), dtype=maskdatamodel.dtype)
    tychomask = np.zeros(len(tychokeep), dtype=maskdatamodel.dtype)
    for col in "RA", "DEC":
        gaiamask[col] = gaiakeep[col]
        tychomask[col] = tychokeep[col]
    gaiamask["REF_ID"] = gaiakeep["REF_ID"]
    # ADM take care to rigorously convert to int64.
    tychomask["REF_ID"] = tychokeep["TYC1"]
    tychomask["REF_ID"] *= int(1e6)
    tychomask["REF_ID"] += tychokeep["TYC2"]*10 + tychokeep["TYC3"]
    gaiamask["REF_CAT"], tychomask["REF_CAT"] = 'G2', 'T2'
    gaiamask["REF_MAG"] = gaiakeep['PHOT_G_MEAN_MAG']
    tychomask["REF_MAG"] = tychomag
    gaiamask["URAT_ID"], tychomask["URAT_ID"] = uratid, -1
    gaiamask["TYPE"], tychomask["TYPE"] = 'PSF', 'PSF'
    mask = np.concatenate([gaiamask, tychomask])

    return mask


def make_bright_star_mask_old(bands, maglim, numproc=4,
                          rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1',
                          infilename=None, outfilename=None):
    """Make a bright star mask from a structure of bright stars drawn from the sweeps.

    Parameters
    ----------
    bands : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z".
        Can pass multiple bands as string, e.g. ``"GRZ"``, in which case maglim has to be a
        list of the same length as the string.
    maglim : :class:`float`
        The upper limit in that magnitude band for which to assemble a list of bright stars.
        Can pass a list of magnitude limits, in which case bands has to be a string of the
        same length (e.g., ``"GRZ"`` for [12.3,12.7,12.6]).
    numproc : :class:`int`, optional
        Number of processes over which to parallelize.
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory containing either sweeps or tractor files...e.g. for dr3 this might be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3/sweep/dr3.1. This is only
        used if ``infilename`` is not passed.
    infilename : :class:`str`, optional,
        if this exists, then the list of bright stars is read in from the file of this name
        if this is not passed, then code defaults to deriving the recarray of bright stars
        from ``rootdirname`` via a call to ``collect_bright_stars``.
    outfilename : :class:`str`, optional, defaults to not writing anything to file
        (FITS) File name to which to write the output bright star mask.

    Returns
    -------
    :class:`recarray`
        - The bright source mask in the form ``RA``, ``DEC``, ``TARGETID``,
          ``IN_RADIUS``, ``NEAR_RADIUS``, ``E1``, ``E2``, ``TYPE``
          (may also be written to file if "outfilename" is passed).
        - TARGETID is as calculated in :mod:`desitarget.targets.encode_targetid`.
        - The radii are in ARCSECONDS (they default to equivalents of half-light radii for ellipses).
        - `E1` and `E2` are ellipticity components, which are 0 for unresolved objects.
        - `TYPE` is always `PSF` for star-like objects. This is taken from the sweeps files, see, e.g.:
          http://legacysurvey.org/dr5/files/#sweep-catalogs.

    Notes
    -----
        - ``IN_RADIUS`` is a smaller radius that corresponds to the ``IN_BRIGHT_OBJECT`` bit in
          ``data/targetmask.yaml`` (and is in ARCSECONDS).
        - ``NEAR_RADIUS`` is a radius that corresponds to the ``NEAR_BRIGHT_OBJECT`` bit in
          ``data/targetmask.yaml`` (and is in ARCSECONDS).
        - Currently uses the radius-as-a-function-of-B-mag for Tycho stars from the BOSS mask
          (in every band) to set the ``NEAR_RADIUS``:
          R = (0.0802B*B - 1.860B + 11.625) (see Eqn. 9 of https://arxiv.org/pdf/1203.6594.pdf)
          and half that radius to set the ``IN_RADIUS``. We convert this from arcminutes to arcseconds.
        - It's an open question as to what the correct radii are for DESI observations.
    """
    # ADM this is just a special case of make_bright_source_mask
    sourcemask = make_bright_source_mask(bands, maglim,
                                         numproc=numproc, rootdirname=rootdirname,
                                         infilename=infilename, outfilename=None)
    # ADM check if a source is unresolved.
    psflike = _psflike(sourcemask["TYPE"])
    wstar = np.where(psflike)
    if len(wstar[0]) > 0:
        done = sourcemask[wstar]
        if outfilename is not None:
            fitsio.write(outfilename, done, clobber=True)
        return done
    else:
        log.error('No PSF-like objects brighter than {} in {}'
                  .format(str(maglim), bands))
        return -1


def make_bright_source_mask(bands, maglim, numproc=4,
                            rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr5/sweep/5.0',
                            infilename=None, outfilename=None):
    """Make a mask of bright sources from a structure of bright sources drawn from the sweeps.

    Parameters
    ----------
    bands : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z".
        Can pass multiple bands as string, e.g. ``"GRZ"``, in which case maglim has to be a
        list of the same length as the string.
    maglim : :class:`float`
        The upper limit in that magnitude band for which to assemble a list of bright sources.
        Can pass a list of magnitude limits, in which case bands has to be a string of the
        same length (e.g., ``"GRZ"`` for [12.3,12.7,12.6]).
    numproc : :class:`int`, optional
        Number of processes over which to parallelize.
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory containing either sweeps or tractor files...e.g. for dr5 this might be
        ``/global/project/projectdirs/cosmo/data/legacysurvey/dr5/sweep/dr5.0``. This is only
        used if ``infilename`` is not passed.
    infilename : :class:`str`, optional,
        if this exists, then the list of bright sources is read in from the file of this name.
        if this is not passed, then code defaults to deriving the recarray of bright sources
        from ``rootdirname`` via a call to ``collect_bright_sources``.
    outfilename : :class:`str`, optional, defaults to not writing anything to file
        (FITS) File name to which to write the output bright source mask.

    Returns
    -------
    :class:`recarray`
        - The bright source mask in the form ``RA`, ``DEC``, ``TARGETID``, ``IN_RADIUS``,
          ``NEAR_RADIUS``, ``E1``, ``E2``, ``TYPE``
          (may also be written to file if ``outfilename`` is passed).
        - ``TARGETID`` is as calculated in :mod:`desitarget.targets.encode_targetid`.
        - The radii are in ARCSECONDS (they default to equivalents of half-light radii for ellipses).
        - ``E1`` and ``E2`` are the ellipticity components as defined at the bottom of, e.g.:
          http://legacysurvey.org/dr5/catalogs/.
        - ``TYPE`` is the ``TYPE`` from the sweeps files, see, e.g.:
          http://legacysurvey.org/dr5/files/#sweep-catalogs.

    Notes
    -----
        - ``IN_RADIUS`` is a smaller radius that corresponds to the ``IN_BRIGHT_OBJECT`` bit in
          ``data/targetmask.yaml`` (and is in ARCSECONDS).
        - ``NEAR_RADIUS`` is a radius that corresponds to the ``NEAR_BRIGHT_OBJECT`` bit in
          ``data/targetmask.yaml`` (and is in ARCSECONDS).
        - Currently uses the radius-as-a-function-of-B-mag for Tycho stars from the BOSS mask
          (in every band) to set the ``NEAR_RADIUS``:
          R = (0.0802B*B - 1.860B + 11.625) (see Eqn. 9 of https://arxiv.org/pdf/1203.6594.pdf)
          and half that radius to set the ``IN_RADIUS``. We convert this from arcminutes to arcseconds.
        - It's an open question as to what the correct radii are for DESI observations.
    """

    # ADM set bands to uppercase if passed as lower case.
    bands = bands.upper()
    # ADM the band names and nobs columns as arrays instead of strings.
    bandnames = np.array(["FLUX_"+band for band in bands])
    nobsnames = np.array(["NOBS_"+band for band in bands])

    # ADM force the input maglim to be a list (in case a single value was passed).
    if isinstance(maglim, int) or isinstance(maglim, float):
        maglim = [maglim]

    if len(bandnames) != len(maglim):
        msg = "bands has to be the same length as maglim and {} does not equal {}".format(
            len(bandnames), len(maglim))
        raise IOError(msg)

    # ADM change input magnitude(s) to a flux to test against.
    fluxlim = 10.**((22.5-np.array(maglim))/2.5)

    if infilename is not None:
        objs = io.read_tractor(infilename)
    else:
        objs = collect_bright_sources(bands, maglim, numproc, rootdirname, outfilename)

    # ADM write the fluxes and bands as arrays instead of named columns

    # ADM limit to the passed faint limit.
    ok = np.zeros(objs[bandnames[0]].shape, dtype=bool)
    fluxes = np.zeros((len(ok), len(bandnames)), dtype=objs[bandnames[0]].dtype)
    for i, (bandname, nobsname) in enumerate(zip(bandnames, nobsnames)):
        fluxes[:, i] = objs[bandname].copy()
        # ADM set any observations with NOBS = 0 to have small flux
        # so glitches don't end up as bright object masks.
        fluxes[objs[nobsname] == 0, i] = 0.0
        ok |= (fluxes[:, i] > fluxlim[i])

    w = np.where(ok)

    fluxes = fluxes[w]
    objs = objs[w]

    # ADM grab the (GRZ) magnitudes for observations
    # ADM and record only the largest flux (smallest magnitude).
    fluxmax = np.max(fluxes, axis=1)
    mags = 22.5-2.5*np.log10(fluxmax)

    # ADM each object's TYPE.
    objtype = objs["TYPE"]

    # ADM calculate the TARGETID.
    targetid = encode_targetid(objid=objs['OBJID'], brickid=objs['BRICKID'], release=objs['RELEASE'])

    # ADM first set the shape parameters assuming everything is an exponential
    # ADM this will correctly assign e1, e2 of 0 to things with zero shape.
    in_radius = objs['SHAPEEXP_R']
    e1 = objs['SHAPEEXP_E1']
    e2 = objs['SHAPEEXP_E2']
    # ADM now to account for deVaucouleurs objects, or things that are dominated by
    # ADM deVaucouleurs profiles, update objects with a larger "DEV" than "EXP" radius.
    wdev = np.where(objs['SHAPEDEV_R'] > objs['SHAPEEXP_R'])
    if len(wdev[0]) > 0:
        in_radius[wdev] = objs[wdev]['SHAPEDEV_R']
        e1[wdev] = objs[wdev]['SHAPEDEV_E1']
        e2[wdev] = objs[wdev]['SHAPEDEV_E2']
    # ADM finally use the Tycho radius (see the notes above) for PSF or star-like objects.
    # ADM More consideration will be needed to derive correct numbers for this for DESI!!!
    # ADM this calculation was for "near" Tycho objects and was in arcmin, so we convert
    # ADM it to arcsec and multiply it by infac (see the top of the module).
    tycho_in_radius = infac*(0.0802*mags*mags - 1.860*mags + 11.625)*60.
    wpsf = np.where(_psflike(objtype))
    in_radius[wpsf] = tycho_in_radius[wpsf]

    # ADM set "near" as a multiple of "in" radius using the factor at the top of the code.
    near_radius = in_radius*nearfac

    # ADM create an output recarray that is just RA, Dec, TARGETID and the radius.
    done = objs[['RA', 'DEC']].copy()
    done = rfn.append_fields(done, ["TARGETID", "IN_RADIUS", "NEAR_RADIUS", "E1", "E2", "TYPE"],
                             [targetid, in_radius, near_radius, e1, e2, objtype],
                             usemask=False,
                             dtypes=['>i8', '>f4', '>f4', '>f4', '>f4', '|S4'])

    if outfilename is not None:
        fitsio.write(outfilename, done, clobber=True)

    return done


def plot_mask(mask, limits=None, radius="IN_RADIUS", show=True):
    """Make a plot of a mask and either display it or retain the plot object for over-plotting.

    Parameters
    ----------
    mask : :class:`recarray`
        A mask constructed by ``make_bright_source_mask``
        (or read in from file in the ``make_bright_source_mask`` format).
    limits : :class:`list`, optional
        The RA/Dec limits of the plot in the form [ramin, ramax, decmin, decmax].
    radius : :class: `str`, optional
        Which mask radius to plot (``IN_RADIUS`` or ``NEAR_RADIUS``). Both can be plotted
        by calling this function twice with show=False and then with ``over=True``.
    show : :class:`boolean`
        If ``True``, then display the plot, Otherwise, just execute the plot commands
        so it can be added to, shown or saved to file later.

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
    w = np.where((mask["RA"] > np.min(limits[:2])-tol) & (mask["RA"] < np.max(limits[:2])+tol) &
                 (mask["DEC"] > np.min(limits[-2:])-tol) & (mask["DEC"] < np.max(limits[-2:])+tol))
    if len(w[0]) == 0:
        log.error('No mask entries within specified limits ({})'.format(limits))
    else:
        mask = mask[w]

    # ADM create ellipse polygons for each entry in the mask and
    # ADM make a list of matplotlib patches for them.
    patches = []
    for i, ellipse in enumerate(mask):
        # ADM create points on the ellipse boundary.
        ras, decs = ellipse_boundary(ellipse["RA"], ellipse["DEC"], ellipse[radius],
                                     ellipse["E1"], ellipse["E2"])
        polygon = Polygon(np.array(list(zip(ras, decs))), True)
        patches.append(polygon)

    p = PatchCollection(patches, alpha=0.4, facecolors='b', edgecolors='b')
    ax.add_collection(p)

    if show:
        plt.show()

    return


def is_in_bright_mask(targs, sourcemask, inonly=False):
    """Determine whether a set of targets is in a bright source mask.

    Parameters
    ----------
    targs : :class:`recarray`
        A recarray of targets as made by, e.g., :mod:`desitarget.cuts.select_targets`.
    sourcemask : :class:`recarray`
        A recarray containing a mask as made by, e.g.,
        :mod:`desitarget.brightmask.make_bright_star_mask` or
        :mod:`desitarget.brightmask.make_bright_source_mask`.
    inonly : :class:`boolean`, optional, defaults to False
        If True, then only calculate the in_mask return but not the near_mask return,
        which is about a factor of 2 faster.

    Returns
    -------
    in_mask : array_like.
        ``True`` for array entries that correspond to a target that is IN a mask.
    near_mask : array_like.
        ``True`` for array entries that correspond to a target that is NEAR a mask.
    """
    t0 = time()

    # ADM initialize an array of all False (nothing is yet in a mask).
    in_mask = np.zeros(len(targs), dtype=bool)
    near_mask = np.zeros(len(targs), dtype=bool)

    # ADM turn the coordinates of the masks and the targets into SkyCoord objects.
    ctargs = SkyCoord(targs["RA"]*u.degree, targs["DEC"]*u.degree)
    cmask = SkyCoord(sourcemask["RA"]*u.degree, sourcemask["DEC"]*u.degree)

    # ADM this is the largest search radius we should need to consider
    # ADM in the future an obvious speed up is to split on radius
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
            return in_mask
        return in_mask, near_mask

    # ADM need to differentiate targets that are in ellipse-on-the-sky masks
    # ADM from targets that are in circle-on-the-sky masks.
    rex_or_psf = _rexlike(sourcemask[idmask]["TYPE"]) | _psflike(sourcemask[idmask]["TYPE"])
    w_ellipse = np.where(~rex_or_psf)

    # ADM only continue if there are any elliptical masks.
    if len(w_ellipse[0]) > 0:
        idelltargs = idtargs[w_ellipse]
        idellmask = idmask[w_ellipse]

        log.info('Testing {} total targets against {} total elliptical masks...t={:.1f}s'
                 .format(len(set(idelltargs)), len(set(idellmask)), time()-t0))

        # ADM to speed the calculation, make a dictionary of which targets (the
        # ADM values) are associated with each mask (the keys).
        targidineachmask = {}
        # ADM first initiate a list for each relevant key (mask ID).
        for maskid in set(idellmask):
            targidineachmask[maskid] = []
        # ADM then append those lists until they contain the IDs of each
        # ADM relevant target as the values.
        for index, targid in enumerate(idelltargs):
            targidineachmask[idellmask[index]].append(targid)

        # ADM loop through the masks and determine which relevant points occupy
        # ADM them for both the IN_RADIUS and the NEAR_RADIUS.
        for maskid in targidineachmask:
            targids = targidineachmask[maskid]
            ellras, elldecs = targs[targids]["RA"], targs[targids]["DEC"]
            mask = sourcemask[maskid]
            # ADM Refine True/False for being in a mask based on the elliptical masks.
            in_mask[targids] |= is_in_ellipse(ellras, elldecs, mask["RA"], mask["DEC"],
                                              mask["IN_RADIUS"], mask["E1"], mask["E2"])
            if not inonly:
                near_mask[targids] |= is_in_ellipse(ellras, elldecs,
                                                    mask["RA"], mask["DEC"],
                                                    mask["NEAR_RADIUS"],
                                                    mask["E1"], mask["E2"])

        log.info('Done with elliptical masking...t={:1f}s'.format(time()-t0))

    # ADM finally, record targets that were in a circles-on-the-sky mask, which
    # ADM trumps any information about just being in an elliptical mask.
    # ADM find angular separations less than the mask radius for circle masks
    # ADM matches that meet these criteria are in a circle mask (at least one).
    w_in = np.where((d2d.arcsec < sourcemask[idmask]["IN_RADIUS"]) & rex_or_psf)
    in_mask[idtargs[w_in]] = True

    if not inonly:
        w_near = np.where((d2d.arcsec < sourcemask[idmask]["NEAR_RADIUS"]) & rex_or_psf)
        near_mask[idtargs[w_near]] = True
        return in_mask, near_mask

    return in_mask


def is_bright_source(targs, sourcemask):
    """Determine whether any of a set of targets are, themselves, a bright source mask.

    Parameters
    ----------
    targs : :class:`recarray`
        A recarray of targets as made by, e.g., :mod:`desitarget.cuts.select_targets`.
    sourcemask : :class:`recarray`
        A recarray containing a bright source mask as made by, e.g.,
        :mod:`desitarget.brightmask.make_bright_star_mask` or
        :mod:`desitarget.brightmask.make_bright_source_mask`.

    Returns
    -------
    is_mask : array_like
        True for array entries that correspond to targets that are, themselves, a mask.

    """

    # ADM initialize an array of all False (nothing yet has been shown to correspond to a mask).
    is_mask = np.zeros(len(targs), dtype=bool)

    # ADM calculate the TARGETID for the targets.
    targetid = encode_targetid(objid=targs['BRICK_OBJID'],
                               brickid=targs['BRICKID'],
                               release=targs['RELEASE'])

    # ADM super-fast set-based look-up of which TARGETIDs are matches between the masks and the targets.
    matches = set(sourcemask["TARGETID"]).intersection(set(targetid))
    # ADM determine the indexes of the targets that have a TARGETID in matches.
    w_mask = [index for index, item in enumerate(targetid) if item in matches]

    # ADM w_mask now contains the target indices that match to a bright mask on TARGETID.
    is_mask[w_mask] = True

    return is_mask


def generate_safe_locations(sourcemask, Nperradius=1):
    """Given a bright source mask, generate SAFE (BADSKY) locations at its periphery.

    Parameters
    ----------
    sourcemask : :class:`recarray`
        A recarray containing a bright mask as made by, e.g.,
        :mod:`desitarget.brightmask.make_bright_star_mask` or
        :mod:`desitarget.brightmask.make_bright_source_mask`.
    Nperradius : :class:`int`, optional, defaults to 1 per arcsec of radius
        The number of safe locations to generate scaled by the radius of each mask
        in ARCSECONDS (i.e. the number of positions per arcsec of radius).

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
            ras, decs = np.concatenate((ras, ellras)), np.concatenate((decs, elldecs))

    return ras, decs


def append_safe_targets(targs, sourcemask, nside=None, drbricks=None):
    """Append targets at SAFE (BADSKY) locations to target list, set bits in TARGETID and DESI_TARGET.

    Parameters
    ----------
    targs : :class:`~numpy.ndarray`
        A recarray of targets as made by, e.g. :mod:`desitarget.cuts.select_targets`.
    nside : :class:`integer`
        The HEALPix nside used throughout the DESI data model.
    sourcemask : :class:`~numpy.ndarray`
        A recarray containing a bright source mask as made by, e.g.
        :mod:`desitarget.brightmask.make_bright_star_mask` or
        :mod:`desitarget.brightmask.make_bright_source_mask`.
    drbricks : :class:`~numpy.ndarray`, optional
        A rec array containing at least the "release", "ra", "dec" and "nobjs" columns from a survey bricks file.
        This is typically used for testing only.

    Returns
    -------
        The original recarray of targets (targs) is returned with additional SAFE (BADSKY) targets appended to it.

    Notes
    -----
        - See `Tech Note 2346`_ for more on the SAFE (BADSKY) locations.
        - See `Tech Note 2348`_ for more on setting the SKY bit in TARGETID.
        - Currently hard-coded to create an additional 1 safe location per arcsec of mask radius.
          The correct number per radial element (Nperradius) for DESI is an open question.
    """

    # ADM Number of safe locations per radial arcsec of each mask.
    Nperradius = 1

    # ADM generate SAFE locations at the periphery of the masks appropriate to a density of Nperradius.
    ra, dec = generate_safe_locations(sourcemask, Nperradius)

    # ADM duplicate the targs rec array with a number of rows equal to the generated safe locations.
    nrows = len(ra)
    safes = np.zeros(nrows, dtype=targs.dtype)

    # ADM populate the safes recarray with the RA/Dec of the SAFE locations.
    safes["RA"] = ra
    safes["DEC"] = dec

    # ADM set the bit for SAFE locations in DESITARGET.
    safes["DESI_TARGET"] |= desi_mask.BAD_SKY

    # ADM add the brick information for the SAFE/BADSKY targets.
    b = brick.Bricks(bricksize=0.25)
    safes["BRICKID"] = b.brickid(safes["RA"], safes["DEC"])
    safes["BRICKNAME"] = b.brickname(safes["RA"], safes["DEC"])

    # ADM get the string version of the data release (to find directories for brick information).
    drint = np.max(targs['RELEASE']//1000)
    # ADM check the targets all have the same release.
    checker = np.min(targs['RELEASE']//1000)
    if drint != checker:
        raise IOError('Objects from multiple data releases in same input numpy array?!')
    drstring = 'dr'+str(drint)

    # ADM now add the OBJIDs, ensuring they start higher than any other OBJID in the DR
    # ADM read in the Data Release bricks file.
    if drbricks is None:
        rootdir = "/project/projectdirs/cosmo/data/legacysurvey/"+drstring.strip()+"/"
        drbricks = fitsio.read(rootdir+"survey-bricks-"+drstring.strip()+".fits.gz")
    # ADM the BRICK IDs that are populated for this DR.
    drbrickids = b.brickid(drbricks["ra"], drbricks["dec"])
    # ADM the maximum possible BRICKID at bricksize=0.25.
    brickmax = 662174
    # ADM create a histogram of how many SAFE/BADSKY objects are in each brick.
    hsafes = np.histogram(safes["BRICKID"], range=[0, brickmax+1], bins=brickmax+1)[0]
    # ADM create a histogram of how many objects are in each brick in this DR.
    hnobjs = np.zeros(len(hsafes), dtype=int)
    hnobjs[drbrickids] = drbricks["nobjs"]
    # ADM make each OBJID for a SAFE/BADSKY +1 higher than any other OBJID in the DR.
    safes["BRICK_OBJID"] = hnobjs[safes["BRICKID"]] + 1
    # ADM sort the safes array on BRICKID.
    safes = safes[safes["BRICKID"].argsort()]
    # ADM remove zero entries from the histogram of BRICKIDs in safes, for speed.
    hsafes = hsafes[np.where(hsafes > 0)]
    # ADM the count by which to augment each OBJID to make unique OBJIDs for safes.
    objsadd = np.hstack([np.arange(i) for i in hsafes])
    # ADM finalize the OBJID for each SAFE target.
    safes["BRICK_OBJID"] += objsadd

    # ADM finally, update the TARGETID with the OBJID, the BRICKID, and the fact these are skies.
    safes["TARGETID"] = encode_targetid(objid=safes['BRICK_OBJID'],
                                        brickid=safes['BRICKID'],
                                        sky=1)

    # ADM return the input targs with the SAFE targets appended.
    return np.hstack([targs, safes])


def set_target_bits(targs, sourcemask):
    """Apply bright source mask to targets, return desi_target array.

    Parameters
    ----------
    targs : :class:`recarray`
        A recarray of targets as made by, e.g., :mod:`desitarget.cuts.select_targets`.
    sourcemask : :class:`recarray`
        A recarray containing a bright source mask as made by, e.g.
        :mod:`desitarget.brightmask.make_bright_star_mask` or
        :mod:`desitarget.brightmask.make_bright_source_mask`.

    Returns
    -------
        an ndarray of the updated desi_target bit that includes bright source information.

    Notes
    -----
        - Sets ``IN_BRIGHT_OBJECT`` and ``NEAR_BRIGHT_OBJECT`` via matches to
          circular and/or elliptical masks.
        - Sets BRIGHT_OBJECT via an index match on TARGETID
          (defined as in :mod:`desitarget.targets.encode_targetid`).

    See :mod:`desitarget.targetmask` for the definition of each bit.
    """

    bright_object = is_bright_source(targs, sourcemask)
    in_bright_object, near_bright_object = is_in_bright_mask(targs, sourcemask)

    desi_target = targs["DESI_TARGET"].copy()

    desi_target |= bright_object * desi_mask.BRIGHT_OBJECT
    desi_target |= in_bright_object * desi_mask.IN_BRIGHT_OBJECT
    desi_target |= near_bright_object * desi_mask.NEAR_BRIGHT_OBJECT

    return desi_target


def mask_targets(targs, inmaskfile=None, nside=None, bands="GRZ", maglim=[10, 10, 10], numproc=4,
                 rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1',
                 outfilename=None, drbricks=None):
    """Add bits for if objects are in a bright mask, and SAFE (BADSKY) locations, to a target set.

    Parameters
    ----------
    targs : :class:`str` or `~numpy.ndarray`
        A recarray of targets created by :mod:`desitarget.cuts.select_targets` OR a filename of
        a file that contains such a set of targets
    inmaskfile : :class:`str`, optional
        An input bright source mask created by, e.g.
        :mod:`desitarget.brightmask.make_bright_star_mask` or
        :mod:`desitarget.brightmask.make_bright_source_mask`
        If None, defaults to making the bright mask from scratch
        The next 5 parameters are only relevant to making the bright mask from scratch
    nside : :class:`integer`
        The HEALPix nside used throughout the DESI data model
    bands : :class:`str`
        A magnitude band from the sweeps, e.g., "G", "R", "Z".
        Can pass multiple bands as string, e.g. "GRZ", in which case maglim has to be a
        list of the same length as the string
    maglim : :class:`float`
        The upper limit in that magnitude band for which to assemble a list of bright sources.
        Can pass a list of magnitude limits, in which case bands has to be a string of the
        same length (e.g., "GRZ" for [12.3,12.7,12.6])
    numproc : :class:`int`, optional
        Number of processes over which to parallelize
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory containing either sweeps or tractor files...e.g. for dr3 this might be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3/sweep/dr3.1
    outfilename : :class:`str`, optional, defaults to not writing anything to file
        (FITS) File name to which to write the output mask ONE OF outfilename or
        inmaskfile MUST BE PASSED
    drbricks : :class:`~numpy.ndarray`, optional
        A rec array containing at least the "release", "ra", "dec" and "nobjs" columns from a survey bricks file
        This is typically used for testing only.

    Returns
    -------
    :class:`~numpy.ndarray`
        the input targets with the DESI_TARGET column updated to reflect the BRIGHT_OBJECT bits
        and SAFE (BADSKY) sky locations added around the perimeter of the bright source mask.

    Notes
    -----
        - See `Tech Note 2346`_ for more details about SAFE (BADSKY) locations.
        - Runs in about 10 minutes for 20M targets and 50k masks (roughly maglim=10).
    """

    t0 = time()

    if inmaskfile is None and outfilename is None:
        raise IOError('One of inmaskfile or outfilename must be passed')

    # ADM Check if targs is a filename or the structure itself.
    if isinstance(targs, str):
        if not os.path.exists(targs):
            raise ValueError("{} doesn't exist".format(targs))
        targs = fitsio.read(targs)

    # ADM check if a file for the bright source mask was passed, if not then create it.
    if inmaskfile is None:
        sourcemask = make_bright_source_mask(bands, maglim, numproc=numproc,
                                             rootdirname=rootdirname, outfilename=outfilename)
    else:
        sourcemask = fitsio.read(inmaskfile)

    ntargsin = len(targs)
    log.info('Number of targets {}...t={:.1f}s'.format(ntargsin, time()-t0))
    log.info('Number of masks {}...t={:.1f}s'.format(len(sourcemask), time()-t0))

    # ADM generate SAFE locations and add them to the target list.
    targs = append_safe_targets(targs, sourcemask, nside=nside, drbricks=drbricks)

    log.info('Generated {} SAFE (BADSKY) locations...t={:.1f}s'.format(len(targs)-ntargsin, time()-t0))

    # ADM update the bits depending on whether targets are in a mask.
    dt = set_target_bits(targs, sourcemask)
    done = targs.copy()
    done["DESI_TARGET"] = dt

    # ADM remove any SAFE locations that are in bright masks (because they aren't really safe).
    w = np.where(((done["DESI_TARGET"] & desi_mask.BAD_SKY) == 0) |
                 ((done["DESI_TARGET"] & desi_mask.IN_BRIGHT_OBJECT) == 0))
    if len(w[0]) > 0:
        done = done[w]

    log.info("...of these, {} SAFE (BADSKY) locations aren't in masks...t={:.1f}s"
             .format(len(done)-ntargsin, time()-t0))

    log.info('Finishing up...t={:.1f}s'.format(time()-t0))

    return done
