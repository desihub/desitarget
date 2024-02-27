"""
desitarget.streamcuts
=====================

Target selection cuts for the DESI MWS Stellar Stream programs.

Borrows heavily from Sergey Koposov.
"""
from time import time

# ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

# ADM start the clock
start = time()

import healpy
import datetime
import os
import scipy.interpolate
import numpy as np
import astropy.table as atpy

from desitarget.cuts import _psflike
from desitarget.streams.utilities import read_data, sphere_rotate
from desitarget.streams.utilities import correct_pm, rotate_pm, betw
from desitarget.streams.utilities import pm12_sel_func, plx_sel_func


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
        ``True`` if the object is a BGS target of type ``targtype``.

    Notes
    -----

    """
    # ADM the parameters that define the coordinates of the stream.
    rapol, decpol, ra_ref = 34.5987, 29.7331, 200

    # ADM the parameters that define the extent of the stream.
    mind, maxd = 80, 100

    # ADM the name of the stream.
    stream_name = "GD1"

    # ADM collect the data, including Gaia-matching to DR3.
    D = read_data(swdir, rapol, decpol, ra_ref, mind, maxd, stream_name,
                  readcache=readcache)

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
    pmra_error = 1./np.sqrt(objs['PMRA_IVAR'])
    pmdec_error = 1./np.sqrt(objs['PMDEC_IVAR'])
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
    PM2TRACK = scipy.interpolate.UnivariateSpline([-90, -60, -45, -30, 0, 20],
                                              [-.7, -.7, -0.35, -0.1, 0.2, .5],
                                              s=0)

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
    gaia_astrom_sel = pm12_sel_func(fi1, pmfi1, pmfi2, pm_err, pm_pad, 2.5)
    gaia_astrom_sel &= plx_sel_func(fi1, D, 2.5)
    gaia_astrom_sel &= r > bright_limit

    log.info(f"Objects in the field of the stream: {field_sel.sum()}")
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
    print(f"With correct astrometry AND correct cmd: {tot}")

    # ADM selection for objects that lack Gaia astrometry.
    # ADM has type PSF and in a reasonable isochrone window.
    startyp = _psflike(objs["TYPE"])
    cmd_win = 0.1 + 10**(-2 + (r - 20) / 2.5)

    # ADM overall faint selection.
    faint_sel = ~np.isfinite(objs['PMRA'])
    faint_sel &= betw(r, 20, faint_limit)
    faint_sel &= betw(np.abs(delta_cmd), 0, cmd_win))
    faint_sel &= startyp
    faint_sel &= stellar_locus_sel
    tot = np.sum(faint_sel & field_sel)
    log.info(f"Objects in the field that meet the faint selection: {tot}")
    
    # ADM "filler" selections.
    # (PSF type + blue in colour and not previously selected)
    common_filler_sel &= betw(r, 19, faint_limit)
    common_filler_sel &= startyp
    common_filler_sel &= ~faint_sel
    common_filler_set &= ~gaia_astrom_sel
    common_filler_sel &= stellar_locus_sel

    filler_sel = common_filler_sel & betw(g - r, -.3, 1.2)

    filler_red_sel = common_filler_sel & betw(g - r, 1.2, 2.2)
    tot = np.sum(filler_sel & field_sel)
    log.info(f"Objects in the field that meet the filler selection: {tot}")
    
    return
