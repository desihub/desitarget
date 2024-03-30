"""
desitarget.streams.cuts
=======================

Target selection cuts for the DESI MWS Stellar Stream programs.

Borrows heavily from Sergey Koposov.
"""
from time import time

import healpy
import datetime
import os
import scipy.interpolate
import numpy as np
import astropy.table as atpy

from desitarget.cuts import _psflike
from desitarget.streams.utilities import sphere_rotate, correct_pm, rotate_pm, \
    betw, pm12_sel_func, plx_sel_func, get_CMD_interpolator, stream_distance,  \
    get_stream_parameters
from desitarget.streams.io import read_data_per_stream
from desitarget.targets import resolve
from desitarget.streams.targets import finalize

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()


def is_in_GD1(objs):
    """Whether a target lies within the GD1 stellar stream.

    Parameters
    ----------
    objs : :class:`array_like`
        Numpy rec array with at least the Legacy Surveys/Gaia columns:
        RA, DEC, PARALLAX, PMRA, PMDEC, PARALLAX_IVAR, PMRA_IVAR,
        PMDEC_IVAR, EBV, FLUX_G, FLUX_R, FLUX_Z, PSEUDOCOLOUR, TYPE,
        ASTROMETRIC_PARAMS_SOLVED, NU_EFF_USED_IN_ASTROMETRY,
        ECL_LAT, PHOT_G_MEAN_MAG.

    Returns
    -------
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM" target.
    :class:`array_like`
        ``True`` if the object is a faint "FAINT_NO_PM" target.
    :class:`array_like`
        ``True`` if the object is a white dwarf "FILLER" target.
    """
    # ADM start the clock.
    start = time()

    # ADM the name of the stream.
    stream_name = "GD1"

    log.info(f"Starting selection for {stream_name}...t={time()-start:.1f}s")

    # ADM look up the defining parameters of the stream.
    stream = get_stream_parameters(stream_name)
    # ADM the parameters that define the coordinates of the stream.
    rapol, decpol, ra_ref = stream["RAPOL"], stream["DECPOL"], stream["RA_REF"]
    # ADM the parameters that define the extent of the stream.
    mind, maxd = stream["MIND"], stream["MAXD"]

    # ADM rotate the data into the coordinate system of the stream.
    fi1, fi2 = sphere_rotate(objs['RA'], objs['DEC'], rapol, decpol, ra_ref)

    # ADM distance of the stream (similar to Koposov et al. 2010 paper).
    dist = stream_distance(fi1, stream_name)

    # ADM heliocentric correction to proper motion.
    xpmra, xpmdec = correct_pm(objs['RA'], objs['DEC'],
                               objs['PMRA'], objs['PMDEC'], dist)

    # ADM reflex correction for proper motions.
    pmfi1, pmfi2 = rotate_pm(objs['RA'], objs['DEC'], xpmra, xpmdec,
                             rapol, decpol, ra_ref)

    # ADM derive the combined proper motion error.
    # ADM guard against dividing by zero.
    pmra_error = np.zeros_like(objs["PMRA_IVAR"]) + 1e8
    ii = objs['PMRA_IVAR'] != 0
    pmra_error[ii] = 1./np.sqrt(objs[ii]['PMRA_IVAR'])
    pmdec_error = np.zeros_like(objs["PMDEC_IVAR"]) + 1e8
    ii = objs['PMDEC_IVAR'] != 0
    pmdec_error[ii] = 1./np.sqrt(objs[ii]['PMDEC_IVAR'])
    pm_err = np.sqrt(0.5 * (pmra_error**2 + pmdec_error**2))

    # ADM dust correction.
    ext_coeff = dict(g=3.237, r=2.176, z=1.217)
    eg, er, ez = [ext_coeff[_] * objs['EBV'] for _ in 'grz']
    ext = {}
    ext['G'] = eg
    ext['R'] = er
    ext['Z'] = ez

    g, r, z = [22.5 - 2.5 * np.log10(objs['FLUX_' + _]) - ext[_] for _ in 'GRZ']

    # ADM some spline function over which to interpolate.
    TRACK = scipy.interpolate.CubicSpline([-90, -70, -50, -40, -20, 0, 20],
                                          [-3, -1.5, -.2, -0., -.0, -1.2, -3])
    PM1TRACK = scipy.interpolate.UnivariateSpline(
        [-90, -70, -50, -30, -15, 0, 20],
        [-5.5, -6.3, -8, -6.5, -5.7, -5, -3.5])
    PM2TRACK = scipy.interpolate.UnivariateSpline(
        [-90, -60, -45, -30, 0, 20],
        [-.7, -.7, -0.35, -0.1, 0.2, .5], s=0)

    # ADM create an interpolated set of phi2 coords (in stream coords).
    dfi2 = fi2 - TRACK(fi1)

    # ADM derive the isochrone track for the stream.
    CMD_II = get_CMD_interpolator(stream_name)

    # ADM how far the data lies from the isochrone.
    delta_cmd = g - r - CMD_II(r - 5 * np.log10(dist * 1e3) + 5)

    # ADM necessary parameters are set up; perform the actual selection.
    bright_limit, faint_limit = 16, 21

    # ADM lies in the stream.
    field_sel = betw(dfi2, -10, 10) & betw(fi1, -90, 21)

    # ADM Gaia-based selection (proper motion and parallax).
    pm_pad = 2  # mas/yr padding in pm selection
    gaia_astrom_sel = pm12_sel_func(PM1TRACK(fi1), PM2TRACK(fi1), pmfi1, pmfi2,
                                    pm_err, pm_pad, 2.5)
    gaia_astrom_sel &= plx_sel_func(fi1, objs, 2.5)
    gaia_astrom_sel &= r > bright_limit

    log.info(f"Objects in the field: {field_sel.sum()}...t={time()-start:.1f}s")
    log.info(f"With correct astrometry: {(gaia_astrom_sel & field_sel).sum()}")

    # ADM padding around the isochrone.
    bright_cmd_sel = betw(delta_cmd, -.2, .2)

    # ADM isochrone selection.
    stellar_locus_blue_sel = ((betw(r - z - (-.17 + .67 * (g - r)), -0.2, 0.2)
                               & ((g - r) <= 1.1)))
    stellar_locus_red_sel = (((g - r > 1.1)
                              & betw(g - r - (1.05 + .25 * (r - z)), -.2, .2)))
    stellar_locus_sel = stellar_locus_blue_sel | stellar_locus_red_sel

    tot = np.sum(field_sel & gaia_astrom_sel & bright_cmd_sel)
    print(f"With correct astrometry AND cmd: {tot}...t={time()-start:.1f}s")

    # ADM selection for objects that lack Gaia astrometry.
    # ADM has type PSF and in a reasonable isochrone window.
    startyp = _psflike(objs["TYPE"])
    cmd_win = 0.1 + 10**(-2 + (r - 20) / 2.5)

    # ADM overall faint selection.
    faint_sel = objs['PMRA'] == 0
    faint_sel &= betw(r, 20, faint_limit)
    faint_sel &= betw(np.abs(delta_cmd), 0, cmd_win)
    faint_sel &= startyp
    faint_sel &= stellar_locus_sel
    tot = np.sum(faint_sel & field_sel)
    log.info(f"Objects that meet faint selection: {tot}...t={time()-start:.1f}s")

    # ADM "filler" selections.
    # (PSF type + blue in colour and not previously selected)
    common_filler_sel = betw(r, 19, faint_limit)
    common_filler_sel &= startyp
    common_filler_sel &= ~faint_sel
    common_filler_sel &= ~gaia_astrom_sel
    common_filler_sel &= stellar_locus_sel

    filler_sel = common_filler_sel & betw(g - r, -.3, 1.2)

    filler_red_sel = common_filler_sel & betw(g - r, 1.2, 2.2)
    tot = np.sum(filler_sel & field_sel)
    log.info(f"Objects meeting filler selection: {tot}...t={time()-start:.1f}s")

    log.info(f"Finished selection for {stream_name}...t={time()-start:.1f}s")

    bright_pm = bright_cmd_sel & gaia_astrom_sel & field_sel
    faint_no_pm = faint_sel & field_sel
    filler = filler_sel & field_sel

    # ADM sanity check that selections do not overlap.
    check = bright_pm.astype(int) + faint_no_pm.astype(int) + filler.astype(int)
    if np.max(check) > 1:
        msg = "Selections should be unique but they overlap!"
        log.error(msg)

    return bright_pm, faint_no_pm, filler


def set_target_bits(objs, stream_names=["GD1"]):
    """Select stream targets, returning target mask arrays.

    Parameters
    ----------
    objects : :class:`~numpy.ndarray`
        numpy structured array with UPPERCASE columns needed for
        stream target selection. See, e.g.,
        :func:`~desitarget.stream.cuts.is_in_GD1` for column names.
    stream_names : :class:`list`
        A list of stream names to process. Defaults to all streams.

    Returns
    -------
    :class:`~numpy.ndarray`
        (desi_target, bgs_target, mws_target, scnd_target) where each
        element is an array of target selection bitmasks for each object.

    Notes
    -----
    - See ../data/targetmask.yaml for the definition of each bit.
    """
    from desitarget.targetmask import desi_mask, scnd_mask

    # ADM set up a zerod scnd_target array to |= with later.
    scnd_target = np.zeros_like(objs["RA"], dtype='int64')

    # ADM might be able to make this more general by putting the
    # ADM bit names in the data/yaml file and using globals()
    # ADM to recover the is_in() functions.

    if "GD1" in stream_names:
        gd1_bright_pm, gd1_faint_no_pm, gd1_filler = is_in_GD1(objs)

        scnd_target |= gd1_bright_pm * scnd_mask.GD1_BRIGHT_PM
        scnd_target |= gd1_faint_no_pm * scnd_mask.GD1_FAINT_NO_PM
        scnd_target |= gd1_filler * scnd_mask.GD1_FILLER

    # ADM tell DESI_TARGET where SCND_ANY was updated.
    desi_target = (scnd_target != 0) * desi_mask.SCND_ANY

    # ADM set BGS_TARGET and MWS_TARGET to zeros.
    bgs_target = np.zeros_like(scnd_target)
    mws_target = np.zeros_like(scnd_target)

    return desi_target, bgs_target, mws_target, scnd_target


def select_targets(swdir, stream_names=["GD1"], readperstream=True,
                   addnors=True, readcache=True):
    """Process files from an input directory to select targets.

    Parameters
    ----------
    swdir : :class:`str`
        Root directory of Legacy Surveys sweep files for a given data
        release for ONE of EITHER north or south, e.g.
        "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0".
    stream_names : :class:`list`
        A list of stream names to process. Defaults to all streams.
    readperstream : :class:`bool`, optional, defaults to ``True``
        When set, read each stream's data individually instead of looping
        through all possible sweeps files. This is likely quickest and
        most useful when working with a single stream. For multiple
        streams it may cause issues when duplicate targets are selected.
    addnors : :class:`bool`
        If ``True`` then if `swdir` contains "north" add sweep files from
        the south by substituting "south" in place of "north" (and vice
        versa, i.e. if `swdir` contains "south" add sweep files from the
        north by substituting "north" in place of "south").
    readcache : :class:`bool`, optional, defaults to ``True``
        If ``True`` read all data from previously made cache files,
        in cases where such files exist. If ``False`` don't read
        from caches AND OVERWRITE any cached files, if they exist. Cache
        files are named $TARG_DIR/streamcache/streamname-drX-cache.fits,
        where streamname is the lower-case name from `stream_names` and
        drX is the Legacy Surveys Data Release (parsed from `swdir`).

    Returns
    -------
    :class:`~numpy.ndarray`
        Targets in the input `swdir` which pass the cuts with added
        targeting columns such as ``TARGETID``, and ``DESI_TARGET``,
        ``BGS_TARGET``, ``MWS_TARGET``, ``SCND_TARGET`` (i.e. target
        selection bitmasks).
    """
    if readperstream:
        # ADM loop over streams and read in the data per-stream.
        # ADM eventually, for multiple streams, we would likely switch
        # ADM to read in each sweep file and parallelizing across files.
        allobjs = []
        for stream_name in stream_names:
            # ADM read in the data.
            strm = get_stream_parameters(stream_name)
            # ADM the parameters that define the coordinates of the stream.
            rapol, decpol, ra_ref = strm["RAPOL"], strm["DECPOL"], strm["RA_REF"]
            # ADM the parameters that define the extent of the stream.
            mind, maxd = strm["MIND"], strm["MAXD"]
            # ADM read in the data.
            objs = read_data_per_stream(swdir, rapol, decpol, ra_ref, mind, maxd,
                                        stream_name,
                                        addnors=addnors, readcache=readcache)
        allobjs.append(objs)
        objects = np.concatenate(allobjs)
    else:
        # ADM --TODO-- write loop across sweeps instead of streams.
        msg = ("readperstream must be True until we implement looping "
               "over sweeps instead of streams")
        log.error(msg)

    # ADM process the targets.
    desi_target, bgs_target, mws_target, scnd_target = set_target_bits(
        objs, stream_names=[stream_name])

    # ADM finalize the targets.
    # ADM anything with DESI_TARGET !=0 is truly a target.
    ii = (desi_target != 0)
    objects = objects[ii]
    desi_target = desi_target[ii]
    bgs_target = bgs_target[ii]
    mws_target = mws_target[ii]
    scnd_target = scnd_target[ii]

    # ADM add TARGETID and targeting bitmask columns.
    targets = finalize(objects, desi_target, bgs_target, mws_target, scnd_target)

    # ADM resolve any duplicates between imaging data releases.
    targets = resolve(targets)

    # ADM we'll definitely need to update the read_data loop if we ever
    # ADM have overlapping targets in overlapping streams.
    if len(np.unique(targets["TARGETID"])) != len(targets):
        msg = ("Targets are not unique. The code needs updated to read in the "
               "sweep files one-by-one (as in desitarget.cuts.select_targets()) "
               "rather than caching each individual stream")
        log.error(msg)

    return targets
