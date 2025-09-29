"""
desitarget.streams.cuts
=======================

Target selection cuts for the DESI MWS Stellar Stream programs.

Borrows heavily from Sergey Koposov.
"""
from time import time

import healpy
import datetime
import yaml
import os
import scipy.interpolate
from importlib import resources
import numpy as np
import astropy.table as atpy
import astropy.coordinates as acoo
import astropy.units as auni

from desitarget.cuts import _psflike
from desitarget.streams.utilities import sphere_rotate, correct_pm, rotate_pm, \
    betw, pm12_sel_func, plx_sel_func, get_CMD_interpolator, stream_distance,  \
    get_targmwext_parameters, simple_plx_sel, oldpop_bhb_sel, pm12_distdep_sel_func, \
    spatial_sel_func, dwarf_plx_sel_func, pm0_sel_func, cmd_sel_func, targmwext_resolve,\
    sort_targmwext_by_rank
from desitarget.streams.io import read_data_per_stream
from desitarget.targets import resolve
from desitarget.streams.targets import finalize

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()


def is_in_GD1(objs, streamname):
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
        ``True`` if the object is a bright "BRIGHT_PM1" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM2" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM3" target.
    :class:`array_like`
        ``True`` if the object is a faint "FAINT_CMD" target.
    :class:`array_like`
        ``True`` if the object is a white dwarf "FILLER" target.
    """
    # CWR modified for DESI extension.
    # ADM start the clock.
    start = time()

    # ADM the name of the stream.
    stream_name = "GD1"

    log.info(f"Starting selection for {stream_name}...t={time()-start:.1f}s")

    # ADM look up the defining parameters of the stream.
    stream = get_targmwext_parameters(stream_name)
    # ADM the parameters that define the coordinates of the stream.
    rapol, decpol, ra_ref = stream["RAPOL"], stream["DECPOL"], stream["RA_REF"]
    # ADM the parameters that define the extent of the stream.
    mind, maxd = stream["MIND"], stream["MAXD"]

    # ADM limit input coordinates to region of the stream.
    cstream = acoo.SkyCoord(rapol*auni.degree, decpol*auni.degree)
    cobjs = acoo.SkyCoord(objs["RA"]*auni.degree, objs["DEC"]*auni.degree)

    # ADM separation between the objects of interest and the stream.
    sep = cobjs.separation(cstream)    # ADM only retain objects in the stream based on their indexes.
    in_stream = np.where(betw(sep.value, mind, maxd))[0]
    isobjs = objs[in_stream]
    log.info(f"Objects near stream: {len(in_stream)}...t={time()-start:.1f}s")
    del cobjs

    # ADM rotate the position data into the coordinate system of the stream.
    fi1, fi2 = sphere_rotate(isobjs['RA'], isobjs['DEC'], rapol, decpol, ra_ref)
    log.info(f"done sphere_rotate...t={time()-start:.1f}s")

    # ADM distance of the stream (similar to Koposov et al. 2010 paper).
    dist = stream_distance(fi1, stream_name, stream)
    log.info(f"done stream_distance...t={time()-start:.1f}s")

    # ADM/CMR REFLEX CORRECTION to proper motion.
    xpmra, xpmdec = correct_pm(isobjs['RA'], isobjs['DEC'],
                               isobjs['PMRA'], isobjs['PMDEC'], dist)
    log.info(f"done correct_pm...t={time()-start:.1f}s")
    # ADM/CMR rotate the REFLEX-CORRECTED proper motions into the coordinate system of the stream.
    pmfi1, pmfi2 = rotate_pm(isobjs['RA'], isobjs['DEC'],
                             xpmra, xpmdec, rapol, decpol, ra_ref)
    log.info(f"done rotate_pm...t={time()-start:.1f}s")
    # ADM derive the combined proper motion error.
    # CMR: RMS error, appropriate for PM ~< PM_err. See Lindegren GAIA-C3-TN-LU-LL-129-01
    pm_err = np.sqrt(0.5 * (isobjs["PMRA_ERROR"]**2 + isobjs["PMDEC_ERROR"]**2))
    log.info(f"done pm_error...t={time()-start:.1f}s")

    # ADM dust correction.
    ext_coeff = dict(g=3.237, r=2.176, z=1.217)
    eg, er, ez = [ext_coeff[_] * isobjs['EBV'] for _ in 'grz']
    ext = {}
    ext['G'] = eg
    ext['R'] = er
    ext['Z'] = ez

    g, r, z = [22.5 - 2.5 * np.log10(isobjs['FLUX_' + _]) - ext[_] for _ in 'GRZ']

    # ADM some spline functions over which to interpolate.
    # CMR stream track in stream coordinates, phi2(phi1)
    TRACK = scipy.interpolate.CubicSpline(stream['PHI1T'], stream['PHI2T'])
    # CMR phi1_cosphi2 proper motion trace, pm_phi1(phi1)
    PM1TRACK = scipy.interpolate.UnivariateSpline(
        stream['PMPHI1_PHI1T'], stream['PMPHI1T'])
    # CMR phi2 proper motion trace, pm_phi2(phi1)
    PM2TRACK = scipy.interpolate.UnivariateSpline(
        stream['PMPHI2_PHI1T'], stream['PMPHI2T'], s=0)

    # ADM create an interpolated set of phi2 coords (in stream coords).
    # CMR this is the distance, in phi2, of each star from the track phi2(phi1)
    dfi2 = fi2 - TRACK(fi1)

    # ADM derive the isochrone track for the stream.
    CMD_II = get_CMD_interpolator(stream_name)

    # ADM how far the data lies from the isochrone.
    delta_cmd = g - r - CMD_II(r - 5 * np.log10(dist * 1e3) + 5)

    # ADM necessary parameters are set up; perform the actual selection.
    # CNR bright, faint and intermediate magnitude limits now in yaml file

    # ADM lies in the stream.
    # CMR modified to use limits from yaml file.
    field_sel = betw(dfi2, stream['DPHI2_MINUS'], stream['DPHI2_PLUS'])
    field_sel &= betw(fi1, stream['PHI1_MINUS'], stream['PHI1_PLUS'])

    # ADM Gaia-based selection (proper motion and parallax).
    # CMR modified to use PM_PAD and PM_NSIG from yaml file
    gaia_astrom_sel = pm12_sel_func(PM1TRACK(fi1), PM2TRACK(fi1), pmfi1, pmfi2,
                                    pm_err, stream['PM_PAD'], stream['PM_NSIG'])
    # CMR modified to use PLX_NSIG from yaml file
    gaia_astrom_sel &= plx_sel_func(dist, isobjs, stream['PLX_NSIG'])

    # CMR magnitude ranges
    brightpm1_magsel = (r > stream['BRIGHT_LIMIT']) & (z <= stream['BRIGHTPM1_LIMIT'])
    brightpm2_magsel = (z > stream['BRIGHTPM1_LIMIT']) & (z <= stream['BRIGHTPM2_LIMIT'])
    brightpm3_magsel = (z > stream['BRIGHTPM2_LIMIT']) & (z <= stream['BRIGHTPM3_LIMIT'])

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

    # ADM selection for objects that lack Gaia astrometry.
    # ADM has type PSF and in a reasonable isochrone window.
    startyp = _psflike(isobjs["TYPE"])
    cmd_win = 0.1 + 10**(-2 + (r - 20) / 2.5)

    # ADM overall faint selection.
    # CMR modified to use mag limts from yaml file
    faint_cmag_sel = betw(z, stream['FAINT_CMD_LIMIT'], stream['FAINT_LIMIT'])
    faint_cmag_sel &= betw(np.abs(delta_cmd), 0, cmd_win)

    # ADM "filler" selections.
    # (PSF type + blue in colour and not previously selected)
    common_filler_sel = betw(z, stream['BRIGHTPM2_LIMIT'], stream['FAINT_LIMIT'])
    common_filler_sel &= startyp
    common_filler_sel &= stellar_locus_sel

    filler_sel = common_filler_sel & betw(g - r, -.3, 1.2)
    filler_red_sel = common_filler_sel & betw(g - r, 1.2, 2.2)

    bright_pm1 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm1_magsel
    bright_pm2 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm2_magsel
    bright_pm3 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm3_magsel
    bright_pm = bright_pm1 | bright_pm2 | bright_pm3

    # adapted to streams from NRS FAINT_CMD selection
    # NRS passes CMD selection
    # NRS does NOT have Gaia astrometry OR is fainter than BRIGHTPM3_LIMIT
    # NRS passes phi2 stream selection
    # NRS passes Magnitude selection
    # NRS NOT in BRIGHT_PM
    faint_cmd = (
        faint_cmag_sel & field_sel & startyp
        & (~np.isfinite(isobjs["PMRA"]) | (z > stream['BRIGHTPM3_LIMIT']))
        & ~bright_pm
    )

    filler = filler_sel & field_sel & ~bright_pm1 & ~bright_pm2 & ~bright_pm3 & ~faint_cmd

    # CMR moved here to write numbers of final selections, but less useful for timing.
    log.info(f"Objects meeting bright selection: {np.sum(bright_pm)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm1 selection: {np.sum(bright_pm1)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm2 selection: {np.sum(bright_pm2)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm3 selection: {np.sum(bright_pm3)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting faint_cmd selection: {np.sum(faint_cmd)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting filler selection: {np.sum(filler)}...t={time()-start:.1f}s")

    log.info(f"Finished selection for {stream_name}...t={time()-start:.1f}s")

    # ADM sanity check that selections do not overlap.
    check = bright_pm1.astype(int) + bright_pm2.astype(int) + bright_pm3.astype(int) + faint_cmd.astype(int) + filler.astype(int)
    # ADM guard against check being an empty list if there are no targets.
    if len(check) > 0:
        if np.max(check) > 1:
            msg = "Selections should be unique but they overlap!"
            log.error(msg)

    # ADM we sub-selected objects to just those in the stream, so we need
    # ADM to expand back to all of the passed objects. Objects that are
    # ADM not in the stream should be retained as False.
    nobjs = len(objs)
    f_bright_pm1 = np.zeros(nobjs, dtype=bool)
    f_bright_pm2 = np.zeros(nobjs, dtype=bool)
    f_bright_pm3 = np.zeros(nobjs, dtype=bool)
    f_faint_cmd = np.zeros(nobjs, dtype=bool)
    f_filler = np.zeros(nobjs, dtype=bool)
    # return arrys for pm_only and for consistency with dSph and UFD targeting
    f_pm_only = np.zeros(nobjs, dtype=bool)

    f_bright_pm1[in_stream] = bright_pm1
    f_bright_pm2[in_stream] = bright_pm2
    f_bright_pm3[in_stream] = bright_pm3
    f_faint_cmd[in_stream] = faint_cmd
    f_filler[in_stream] = filler

    return f_bright_pm1, f_bright_pm2, f_bright_pm3, f_pm_only, f_faint_cmd, f_filler


def is_in_ORPHAN(objs, streamname):
    """Whether a target lies within the Orphan stellar stream.

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
        ``True`` if the object is a faint "FAINT_CMD" target.
    :class:`array_like`
        ``True`` if the object is a white dwarf "FILLER" target.

    Notes
    -----
    Based on Sergey Koposov's is_in_GD1() code.
    """
    # ADM start the clock.
    start = time()

    stream_name = "ORPHAN"

    log.info(f"Starting selection for {stream_name}...t={time()-start:.1f}s")

    # CMR get the defining parameters of the stream
    stream = get_targmwext_parameters(stream_name)
    # parameters that define the coordinates of the stream.
    rapol, decpol, ra_ref = stream["RAPOL"], stream["DECPOL"], stream["RA_REF"]
    # parameters that define the extent of the stream, angular distance from the poles of the stream coord system
    mind, maxd = stream["MIND"], stream["MAXD"]

    # ADM limit input coordinates to region of the stream.
    cstream = acoo.SkyCoord(rapol*auni.degree, decpol*auni.degree)
    cobjs = acoo.SkyCoord(objs["RA"]*auni.degree, objs["DEC"]*auni.degree)

    # ADM separation between the objects of interest and the stream.
    sep = cobjs.separation(cstream)    # ADM only retain objects in the stream based on their indexes.
    in_stream = np.where(betw(sep.value, mind, maxd))[0]
    isobjs = objs[in_stream]
    log.info(f"Objects near stream: {len(in_stream)}...t={time()-start:.1f}s")

    # ADM rotate the data into the coordinate system of the stream.
    fi1, fi2 = sphere_rotate(isobjs['RA'], isobjs['DEC'], rapol, decpol, ra_ref)

    # CMR use interpolated stream tracks for distance estimate
    # CMR CHANGED ARGS *******
    dist = stream_distance(fi1, stream_name, stream)

    # CMR rotate PMs to stream frame, NO reflex correction
    pmfi1, pmfi2 = rotate_pm(isobjs['RA'], isobjs['DEC'], isobjs['PMRA'], isobjs['PMDEC'], rapol, decpol, ra_ref)

    # ADM derive the combined proper motion error.
    # CMR: RMS error, appropriate for PM ~< PM_err. See Lindegren GAIA-C3-TN-LU-LL-129-01
    pm_err = np.sqrt(0.5*(isobjs["PMRA_ERROR"]**2 + isobjs["PMDEC_ERROR"]**2))

    # ADM dust correction.
    ext_coeff = dict(g=3.237, r=2.176, z=1.217)
    eg, er, ez = [ext_coeff[_] * isobjs['EBV'] for _ in 'grz']
    ext = {}
    ext['G'] = eg
    ext['R'] = er
    ext['Z'] = ez

    g, r, z = [22.5 - 2.5 * np.log10(isobjs['FLUX_' + _]) - ext[_] for _ in 'GRZ']

    # ADM some spline functions over which to interpolate.
    # CMR stream track in stream coordinates, phi2(phi1)
    TRACK = scipy.interpolate.CubicSpline(stream['PHI1T'], stream['PHI2T'])
    # CMR phi1_cosphi2 proper motion trace, pm_phi1(phi1)
    PM1TRACK = scipy.interpolate.UnivariateSpline(stream['PMPHI1_PHI1T'],
                                                  stream['PMPHI1T'], s=0.02, ext=3)
    # CMR phi2 proper motion trace, pm_phi2(phi1)
    PM2TRACK = scipy.interpolate.UnivariateSpline(stream['PMPHI2_PHI1T'],
                                                  stream['PMPHI2T'], s=0.02, ext=3)

    # ADM create an interpolated set of phi2 coords (in stream coords).
    # CMR this is delta phi2, distance from the stream track in phi2
    dfi2 = fi2 - TRACK(fi1)

    # ADM derive the isochrone track for the stream.
    CMD_II = get_CMD_interpolator(stream_name)

    # ADM how far the data lies from the isochrone.
    delta_cmd = g - r - CMD_II(r - 5 * np.log10(dist * 1e3) + 5)

    # ADM necessary parameters are set up; perform the actual selection.
    bright_limit, faint_limit = 16, 21

    # CMR check if a star is in the stream track region, using stream extent in yaml file
    field_sel = betw(dfi2, stream['DPHI2_MINUS'], stream['DPHI2_PLUS'])
    field_sel &= betw(fi1, stream['PHI1_MINUS'], stream['PHI1_PLUS'])

    # ADM Gaia-based selection (proper motion and parallax).
    gaia_astrom_sel = pm12_distdep_sel_func(PM1TRACK(fi1), PM2TRACK(fi1), pmfi1, pmfi2,
                                            pm_err, dist, stream['VEL_PAD'], stream['PM_NSIG'])
    plx_sel = simple_plx_sel(dist, isobjs, stream['PLX_NSIG'], 0.1)
    gaia_astrom_sel &= plx_sel
    # gaia_astrom_sel &= r > bright_limit

    log.info(f"Objects in the field: {field_sel.sum()}...t={time()-start:.1f}s")
    log.info(f"With correct astrometry: {(gaia_astrom_sel & field_sel).sum()}")

    # ADM select if within padding range of isochrone.
    # CMR Note: no magnitude limits imposed here despite the name
    bright_iso_sel = betw(delta_cmd, -.2, .2)

    # CMR select BHBs by color. Note: no magnitude limits imposed here despite the name
    bright_bhb_sel = oldpop_bhb_sel(g, r, dist)

    # joint CMD selection
    bright_cmd_sel = bright_iso_sel | bright_bhb_sel

    # CMR magnitude ranges
    brightpm1_magsel = (r > stream['BRIGHT_LIMIT']) & (z <= stream['BRIGHTPM1_LIMIT'])
    brightpm2_magsel = (z > stream['BRIGHTPM1_LIMIT']) & (z <= stream['BRIGHTPM2_LIMIT'])
    brightpm3_magsel = (z > stream['BRIGHTPM2_LIMIT']) & (z <= stream['BRIGHTPM3_LIMIT'])

    # ADM isochrone selection.
    stellar_locus_blue_sel = ((betw(r - z - (-.17 + .67 * (g - r)), -0.2, 0.2)
                               & ((g - r) <= 1.1)))
    stellar_locus_red_sel = (((g - r > 1.1)
                              & betw(g - r - (1.05 + .25 * (r - z)), -.2, .2)))
    stellar_locus_sel = stellar_locus_blue_sel | stellar_locus_red_sel

    # tot = np.sum(field_sel & gaia_astrom_sel & bright_cmd_sel)
    # print(f"Obj meeting bright selection: {tot}...t={time()-start:.1f}s")

    # ADM selection for objects that lack Gaia astrometry.
    # ADM has type PSF and in a reasonable isochrone window.
    startyp = _psflike(isobjs["TYPE"])
    cmd_win = 0.1 + 10**(-2 + (r - 20) / 2.5)

    # CMR overall faint selection, using limits from yaml file
    faint_cmag_sel = betw(z, stream['FAINT_CMD_LIMIT'], stream['FAINT_LIMIT'])
    faint_cmag_sel &= betw(np.abs(delta_cmd), 0, cmd_win)

    # ADM "filler" selections.
    # (PSF type + blue in colour and not previously selected)
    common_filler_sel = betw(z, stream['BRIGHTPM2_LIMIT'], stream['FAINT_LIMIT'])
    common_filler_sel &= startyp
    common_filler_sel &= stellar_locus_sel

    filler_sel = common_filler_sel & betw(g - r, -.3, 1.2)

    filler_red_sel = common_filler_sel & betw(g - r, 1.2, 2.2)

    bright_pm1 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm1_magsel
    bright_pm2 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm2_magsel
    bright_pm3 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm3_magsel
    bright_pm = bright_pm1 | bright_pm2 | bright_pm3

    # adapted to streams from NRS FAINT_CMD selection
    # NRS passes CMD selection
    # NRS does NOT have Gaia astrometry OR is fainter than BRIGHTPM3_LIMIT
    # NRS passes phi2 stream selection
    # NRS passes Magnitude selection
    # NRS NOT in BRIGHT_PM
    faint_cmd = (
        faint_cmag_sel & field_sel & startyp
        & (~np.isfinite(isobjs["PMRA"]) | (z > stream['BRIGHTPM3_LIMIT']))
        & ~bright_pm
    )

    filler = filler_sel & field_sel & ~bright_pm1 & ~bright_pm2 & ~bright_pm3 & ~faint_cmd

    log.info(f"Objects meeting bright selection: {np.sum(bright_pm)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm1 selection: {np.sum(bright_pm1)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm2 selection: {np.sum(bright_pm2)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm3 selection: {np.sum(bright_pm3)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting faint_cmd selection: {np.sum(faint_cmd)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting filler selection: {np.sum(filler)}...t={time()-start:.1f}s")

    log.info(f"Finished selection for {stream_name}...t={time()-start:.1f}s")

    # ADM sanity check that selections do not overlap.
    check = bright_pm1.astype(int) + bright_pm2.astype(int) + bright_pm3.astype(int) + faint_cmd.astype(int) + filler.astype(int)

    # ADM guard against check being an empty list if there are no targets.
    if len(check) > 0:
        if np.max(check) > 1:
            msg = "Selections should be unique but they overlap!"
            log.error(msg)

    # ADM we sub-selected objects to just those in the stream, so we need
    # ADM to expand back to all of the passed objects. Objects that are
    # ADM not in the stream should be retained as False.
    nobjs = len(objs)
    f_bright_pm1 = np.zeros(nobjs, dtype=bool)
    f_bright_pm2 = np.zeros(nobjs, dtype=bool)
    f_bright_pm3 = np.zeros(nobjs, dtype=bool)
    f_faint_cmd = np.zeros(nobjs, dtype=bool)
    f_filler = np.zeros(nobjs, dtype=bool)
    # return arrays for pm_only for consistency with dwarf targeting
    f_pm_only = np.zeros(nobjs, dtype=bool)

    f_bright_pm1[in_stream] = bright_pm1
    f_bright_pm2[in_stream] = bright_pm2
    f_bright_pm3[in_stream] = bright_pm3
    f_faint_cmd[in_stream] = faint_cmd
    f_filler[in_stream] = filler

    return f_bright_pm1, f_bright_pm2, f_bright_pm3, f_pm_only, f_faint_cmd, f_filler


def is_in_dwarf(objs, dwarf_name):
    """Performs target selection on a source catalog for a given dwarf galaxy.

    Parameters
    ----------
    objs : :class:`array_like`
        Numpy rec array with at least the Legacy Surveys/Gaia columns:
        RA, DEC, PARALLAX, PMRA, PMDEC, PARALLAX_IVAR, PMRA_IVAR,
        PMDEC_IVAR, EBV, FLUX_G, FLUX_R, FLUX_Z, PSEUDOCOLOUR, TYPE,
        ASTROMETRIC_PARAMS_SOLVED, NU_EFF_USED_IN_ASTROMETRY,
        ECL_LAT, PHOT_G_MEAN_MAG.
    dwarf_name : :class:`str`
        Name of a dwarf galaxy that appears in the ../data/dwarfs.yaml
        file. Possibilities include 'BOOTES_1', 'CANES_VENATICI_1',
        'DRACO_1', 'SEXTANS_1', and 'URSA_MINOR_1'.

    Returns
    -------
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM1" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM2" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM3" target.
    :class:`array_like`
        ``True`` if the object is a faint "PM_ONLY" target.
    :class:`array_like`
        ``True`` if the object is a faint "FAINT_CMD" target.
    :class:`array_like`
        ``True`` if the object is a white dwarf "FILLER" target.
    """
    # NRS modified from DESI extesion stream selection.
    # ADM start the clock.
    start = time()
    log.info(f"Starting selection for {dwarf_name}...t={time()-start:.1f}s")

    # NRS look up the defining parameters of the dwarf.
    dwarf = get_targmwext_parameters(dwarf_name)
    # NRS galaxy coordinates in degrees.
    ra0, dec0 = dwarf["RA"], dwarf["DEC"]
    # NRS galaxy proper motions in mas/yr.
    pmra0, pmdec0 = dwarf["PMRA"], dwarf["PMDEC"]
    # NRS spatial extent in degrees for initial data read.
    maxd = dwarf["MAXD"]
    mind = 0.0
    # NRS galaxy distance in kpc.
    dist = dwarf["DIST"]

    # CMR limit input coordinates to region of the dwarf
    cdwarf = acoo.SkyCoord(ra0*auni.degree, dec0*auni.degree)
    cobjs = acoo.SkyCoord(objs["RA"]*auni.degree, objs["DEC"]*auni.degree)

    # ADM separation between the objects of interest and the dwarf.
    sep = cobjs.separation(cdwarf)    # ADM only retain objects in the dwarf based on their indexes.
    in_dwarf = np.where(betw(sep.value, mind, maxd))[0]
    idobjs = objs[in_dwarf]
    log.info(f"Objects near dwarf: {len(in_dwarf)}...t={time()-start:.1f}s")

    # ADM dust correction.
    ext_coeff = dict(g=3.237, r=2.176, z=1.217)
    g, r, z = [22.5 - 2.5 * np.log10(idobjs['FLUX_' + _]) for _ in 'GRZ']
    eg, er, ez = [ext_coeff[_] * idobjs['EBV'] for _ in 'grz']
    g0 = g - eg
    r0 = r - er
    z0 = z - ez
    g0_r0 = g0 - r0
    r0_z0 = r0 - z0

    # NRS spatial selection; currently redundant with catalog creation.
    field_sel = spatial_sel_func(ra0, dec0, maxd, idobjs)
    log.info(f"Objects in the field: {field_sel.sum()}...t={time()-start:.1f}s")
    # NRS Gaia-based selection (proper motion, parallax, bright limit).
    gaia_pm_sel = pm0_sel_func(
        pmra0, pmdec0, idobjs, pad=dwarf['PM_PAD'], mult=dwarf['PM_NSIG']
    )
    gaia_plx_sel = dwarf_plx_sel_func(
        dist, idobjs, plx_sys=dwarf['PLX_SYS'], mult=dwarf['PLX_NSIG'],
        keep_all_neg=True, min_plx_plxerr=-5
    )
    gaia_astrom_sel = gaia_pm_sel & gaia_plx_sel
    log.info(f"With correct astrometry: {(gaia_astrom_sel & field_sel).sum()}")

    # NRS CMD selection
    cmd_sel = cmd_sel_func(dwarf_name, idobjs)

    # NRS magnitude ranges
    brightpm1_magsel = (r > dwarf['BRIGHT_LIMIT']) & (z <= dwarf['BRIGHTPM1_LIMIT'])
    brightpm2_magsel = betw(z, dwarf['BRIGHTPM1_LIMIT'], dwarf['BRIGHTPM2_LIMIT'])
    brightpm3_magsel = betw(z, dwarf['BRIGHTPM2_LIMIT'], dwarf['BRIGHTPM3_LIMIT'])
    pm_only_magsel = (r > dwarf['BRIGHT_LIMIT']) & (z <= dwarf['PM_ONLY_LIMIT'])
    faint_cmd_magsel = betw(z, dwarf['FAINT_CMD_LIMIT'], dwarf['FAINT_LIMIT'])
    filler_magsel = betw(z, dwarf['FILLER_LIMIT'], dwarf['FAINT_LIMIT'])

    # NRS FILLER stellar locus selection.
    stellar_locus_blue_sel = (
        betw(r0_z0 - (-0.17 + 0.67 * g0_r0), -0.2, 0.2)
        & (g0_r0 <= 1.1)
    )
    stellar_locus_red_sel = (
        betw(g0_r0 - (1.05 + 0.25 * r0_z0), -0.2, 0.2)
        & (g0_r0 > 1.1)
    )
    stellar_locus_sel = stellar_locus_blue_sel | stellar_locus_red_sel

    # NRS BRIGHT_PM targets
    # NRS passes CMD selection
    # NRS passes PM + Parallax selection
    # NRS passes Spatial selection
    # NRS passes Magnitude selection
    bright_pm1 = cmd_sel & gaia_astrom_sel & field_sel & brightpm1_magsel
    bright_pm2 = cmd_sel & gaia_astrom_sel & field_sel & brightpm2_magsel
    bright_pm3 = cmd_sel & gaia_astrom_sel & field_sel & brightpm3_magsel
    bright_pm = bright_pm1 | bright_pm2 | bright_pm3

    # NRS PM_ONLY targets
    # NRS does NOT pass CMD selection
    # NRS passes PM + Parallax selection
    # NRS passes Spatial selection
    # NRS passes Magnitude selection
    # NRS passes Color selection
    pm_only = ~cmd_sel & gaia_astrom_sel & field_sel & pm_only_magsel & betw(g0_r0, -0.3, 1.3)

    # NRS FAINT_CMD targets
    # NRS passes CMD selection
    # NRS does NOT have Gaia astrometry OR is fainter than BRIGHTPM3_LIMIT
    # NRS passes Spatial selection
    # NRS passes Magnitude selection
    # NRS NOT in BRIGHT_PM
    faint_cmd = (
        cmd_sel & field_sel & faint_cmd_magsel & _psflike(idobjs["TYPE"])
        & (~np.isfinite(idobjs["PMRA"]) | (z > dwarf['BRIGHTPM3_LIMIT']))
        & ~bright_pm
    )

    # NRS FILLER targets
    # NRS passes Spatial selection
    # NRS passes Magnitude selection
    # NRS passes Stellar Locus selection
    # NRS passes Color selection
    # NRS passes PSF Type selection
    # NRS NOT in BRIGHT_PM, PM_ONLY, or FAINT_CMD
    filler = (
        field_sel & filler_magsel & stellar_locus_sel
        & betw(g0_r0, -0.3, 1.2) & _psflike(idobjs["TYPE"])
        & ~bright_pm & ~pm_only & ~faint_cmd
    )

    # CMR moved these here so we write numbers of the final selections, but less useful for timing
    log.info(f"Objects meeting BRIGHTPM selection: {np.sum(bright_pm)}")
    log.info(f"Objects meeting BRIGHTPM1 selection: {np.sum(bright_pm1)}")
    log.info(f"Objects meeting BRIGHTPM2 selection: {np.sum(bright_pm2)}")
    log.info(f"Objects meeting BRIGHTPM3 selection: {np.sum(bright_pm3)}")
    log.info(f"Objects meeting FAINT_CMD selection: {np.sum(faint_cmd)}")
    log.info(f"Objects meeting PM_ONLY selection: {np.sum(pm_only)}")
    log.info(f"Objects meeting FILLER selection: {np.sum(filler)}")
    log.info(f"Finished selection for {dwarf_name}...t={time()-start:.1f}s")

    # ADM sanity check that selections do not overlap.
    check = bright_pm1.astype(int) + bright_pm2.astype(int) + bright_pm3.astype(int) + pm_only.astype(int) \
        + faint_cmd.astype(int) + filler.astype(int)
    if len(check) > 0:
        if np.max(check) > 1:
            msg = "Selections should be unique but they overlap!"
            log.error(msg)

    # ADM we sub-selected objects to just those in the stream, so we need
    # ADM to expand back to all of the passed objects. Objects that are
    # ADM not in the stream should be retained as False.
    nobjs = len(objs)
    f_bright_pm1 = np.zeros(nobjs, dtype=bool)
    f_bright_pm2 = np.zeros(nobjs, dtype=bool)
    f_bright_pm3 = np.zeros(nobjs, dtype=bool)
    f_faint_cmd = np.zeros(nobjs, dtype=bool)
    f_filler = np.zeros(nobjs, dtype=bool)
    f_pm_only = np.zeros(nobjs, dtype=bool)

    f_bright_pm1[in_dwarf] = bright_pm1
    f_bright_pm2[in_dwarf] = bright_pm2
    f_bright_pm3[in_dwarf] = bright_pm3
    f_pm_only[in_dwarf] = pm_only
    f_faint_cmd[in_dwarf] = faint_cmd
    f_filler[in_dwarf] = filler

    return f_bright_pm1, f_bright_pm2, f_bright_pm3, f_pm_only, f_faint_cmd, f_filler


def is_in_PAL5(objs, streamname):
    """Whether a target lies within the Palomar 5 stellar stream.

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
        ``True`` if the object is a bright "BRIGHT_PM1" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM2" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM3" target.
    :class:`array_like`
        ``True`` if the object is a faint "FAINT_CMD" target.
    :class:`array_like`
        ``True`` if the object is a white dwarf "FILLER" target.
    """
    # CWR modified for DESI extension.
    # ADM start the clock.
    start = time()

    # ADM the name of the stream.
    stream_name = "PAL5"

    log.info(f"Starting selection for {stream_name}...t={time()-start:.1f}s")

    # ADM look up the defining parameters of the stream.
    stream = get_targmwext_parameters(stream_name)
    # ADM the parameters that define the coordinates of the stream.
    rapol, decpol, ra_ref = stream["RAPOL"], stream["DECPOL"], stream["RA_REF"]
    # ADM the parameters that define the extent of the stream.
    mind, maxd = stream["MIND"], stream["MAXD"]

    # ADM limit input coordinates to region of the stream.
    cstream = acoo.SkyCoord(rapol*auni.degree, decpol*auni.degree)
    cobjs = acoo.SkyCoord(objs["RA"]*auni.degree, objs["DEC"]*auni.degree)

    # ADM separation between the objects of interest and the stream.
    sep = cobjs.separation(cstream)    # ADM only retain objects in the stream based on their indexes.
    in_stream = np.where(betw(sep.value, mind, maxd))[0]
    isobjs = objs[in_stream]
    log.info(f"Objects near stream: {len(in_stream)}...t={time()-start:.1f}s")
    del cobjs

    # ADM rotate the position data into the coordinate system of the stream.
    fi1, fi2 = sphere_rotate(isobjs['RA'], isobjs['DEC'], rapol, decpol, ra_ref)

    # ADM distance of the stream.
    dist = stream_distance(fi1, stream_name, stream)

    # ADM/CMR rotate the proper motions into the coordinate system of the stream.
    pmfi1, pmfi2 = rotate_pm(isobjs['RA'], isobjs['DEC'],
                             isobjs['PMRA'], isobjs['PMDEC'], rapol, decpol, ra_ref)
    # ADM derive the combined proper motion error.
    # CMR: RMS error, appropriate for PM ~< PM_err. See Lindegren GAIA-C3-TN-LU-LL-129-01
    pm_err = np.sqrt(0.5 * (isobjs["PMRA_ERROR"]**2 + isobjs["PMDEC_ERROR"]**2))

    # ADM dust correction.
    ext_coeff = dict(g=3.237, r=2.176, z=1.217)
    eg, er, ez = [ext_coeff[_] * isobjs['EBV'] for _ in 'grz']
    ext = {}
    ext['G'] = eg
    ext['R'] = er
    ext['Z'] = ez

    g, r, z = [22.5 - 2.5 * np.log10(isobjs['FLUX_' + _]) - ext[_] for _ in 'GRZ']

    # ADM some spline functions over which to interpolate.
    # CMR stream track in stream coordinates, phi2(phi1).
    # CMR for Pal 5, smooth the literature track in Phi2 vs. Phi1 so we can extrapolate
    TRACK = scipy.interpolate.UnivariateSpline(stream['PHI1T'], stream['PHI2T'], s=0)
    # CMR phi1_cosphi2 proper motion trace, pm_phi1(phi1)
    PM1TRACK = scipy.interpolate.UnivariateSpline(
        stream['PMPHI1_PHI1T'], stream['PMPHI1T'])
    # CMR phi2 proper motion trace, pm_phi2(phi1)
    PM2TRACK = scipy.interpolate.UnivariateSpline(
        stream['PMPHI2_PHI1T'], stream['PMPHI2T'])

    # ADM create an interpolated set of phi2 coords (in stream coords).
    # CMR this is the distance, in phi2, of each star from the track phi2(phi1)
    dfi2 = fi2 - TRACK(fi1)

    # ADM derive the isochrone track for the stream.
    CMD_II = get_CMD_interpolator(stream_name)

    # ADM how far the data lies from the isochrone.
    delta_cmd = g - r - CMD_II(r - 5 * np.log10(dist * 1e3) + 5)

    # ADM necessary parameters are set up; perform the actual selection.
    # CNR bright, faint and intermediate magnitude limits now in yaml file

    # ADM lies in the stream.
    # CMR modified to use limits from yaml file.
    field_sel = betw(dfi2, stream['DPHI2_MINUS'], stream['DPHI2_PLUS'])
    field_sel &= betw(fi1, stream['PHI1_MINUS'], stream['PHI1_PLUS'])

    # ADM Gaia-based selection (proper motion and parallax).
    # CMR modified to use PM_PAD and PM_NSIG from yaml file
    gaia_astrom_sel = pm12_sel_func(PM1TRACK(fi1), PM2TRACK(fi1), pmfi1, pmfi2,
                                    pm_err, stream['PM_PAD'], stream['PM_NSIG'])
    # CMR modified to use PLX_NSIG from yaml file. Pal 5 is at mostly the same distance
    # so make a wide plx selection rather than try to do something distance dependent
    gaia_astrom_sel &= plx_sel_func(dist, isobjs, stream['PLX_NSIG'])

    # CMR magnitude ranges
    brightpm1_magsel = (r > stream['BRIGHT_LIMIT']) & (z <= stream['BRIGHTPM1_LIMIT'])
    brightpm2_magsel = (z > stream['BRIGHTPM1_LIMIT']) & (z <= stream['BRIGHTPM2_LIMIT'])
    brightpm3_magsel = (z > stream['BRIGHTPM2_LIMIT']) & (z <= stream['BRIGHTPM3_LIMIT'])

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

    # ADM selection for objects that lack Gaia astrometry.
    # ADM has type PSF and in a reasonable isochrone window.
    startyp = _psflike(isobjs["TYPE"])
    cmd_win = 0.1 + 10**(-2 + (r - 20) / 2.5)

    # ADM overall faint selection.
    # CMR modified to use mag limts from yaml file
    faint_cmag_sel = betw(z, stream['FAINT_CMD_LIMIT'], stream['FAINT_LIMIT'])
    faint_cmag_sel &= betw(np.abs(delta_cmd), 0, cmd_win)

    # ADM "filler" selections.
    # (PSF type + blue in colour and not previously selected)
    common_filler_sel = betw(z, stream['BRIGHTPM2_LIMIT'], stream['FAINT_LIMIT'])
    common_filler_sel &= startyp
    common_filler_sel &= stellar_locus_sel

    filler_sel = common_filler_sel & betw(g - r, -.3, 1.2)
    filler_red_sel = common_filler_sel & betw(g - r, 1.2, 2.2)

    bright_pm1 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm1_magsel
    bright_pm2 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm2_magsel
    bright_pm3 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm3_magsel
    bright_pm = bright_pm1 | bright_pm2 | bright_pm3

    # adapted to streams from NRS FAINT_CMD selection
    # NRS passes CMD selection
    # NRS does NOT have Gaia astrometry OR is fainter than BRIGHTPM3_LIMIT
    # NRS passes phi2 stream selection
    # NRS passes Magnitude selection
    # NRS NOT in BRIGHT_PM
    faint_cmd = (
        faint_cmag_sel & field_sel & startyp
        & (~np.isfinite(isobjs["PMRA"]) | (z > stream['BRIGHTPM3_LIMIT']))
        & ~bright_pm
    )

    filler = filler_sel & field_sel & ~bright_pm1 & ~bright_pm2 & ~bright_pm3 & ~faint_cmd

    # CMR moved here to write numbers of final selections, but less useful for timing.
    log.info(f"Objects meeting bright selection: {np.sum(bright_pm)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm1 selection: {np.sum(bright_pm1)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm2 selection: {np.sum(bright_pm2)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm3 selection: {np.sum(bright_pm3)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting faint_cmd selection: {np.sum(faint_cmd)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting filler selection: {np.sum(filler)}...t={time()-start:.1f}s")

    log.info(f"Finished selection for {stream_name}...t={time()-start:.1f}s")

    # ADM sanity check that selections do not overlap.
    check = bright_pm1.astype(int) + bright_pm2.astype(int) + bright_pm3.astype(int) + faint_cmd.astype(int) + filler.astype(int)
    # ADM guard against check being an empty list if there are no targets.
    if len(check) > 0:
        if np.max(check) > 1:
            msg = "Selections should be unique but they overlap!"
            log.error(msg)

    # ADM we sub-selected objects to just those in the stream, so we need
    # ADM to expand back to all of the passed objects. Objects that are
    # ADM not in the stream should be retained as False.
    nobjs = len(objs)
    f_bright_pm1 = np.zeros(nobjs, dtype=bool)
    f_bright_pm2 = np.zeros(nobjs, dtype=bool)
    f_bright_pm3 = np.zeros(nobjs, dtype=bool)
    f_faint_cmd = np.zeros(nobjs, dtype=bool)
    f_filler = np.zeros(nobjs, dtype=bool)
    # return arrys for pm_only and for consistency with dSph and UFD targeting
    f_pm_only = np.zeros(nobjs, dtype=bool)

    f_bright_pm1[in_stream] = bright_pm1
    f_bright_pm2[in_stream] = bright_pm2
    f_bright_pm3[in_stream] = bright_pm3
    f_faint_cmd[in_stream] = faint_cmd
    f_filler[in_stream] = filler

    return f_bright_pm1, f_bright_pm2, f_bright_pm3, f_pm_only, f_faint_cmd, f_filler


def is_in_C19(objs, streamname):
    """Whether a target lies within the C-19 stellar stream.

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
        ``True`` if the object is a bright "BRIGHT_PM1" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM2" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM3" target.
    :class:`array_like`
        ``True`` if the object is a faint "FAINT_CMD" target.
    :class:`array_like`
        ``True`` if the object is a white dwarf "FILLER" target.
    """
    # ADM start the clock.
    start = time()

    # ADM the name of the stream.
    stream_name = "C19"

    log.info(f"Starting selection for {stream_name}...t={time()-start:.1f}s")

    # ADM look up the defining parameters of the stream.
    stream = get_targmwext_parameters(stream_name)
    # ADM the parameters that define the coordinates of the stream.
    rapol, decpol, ra_ref = stream["RAPOL"], stream["DECPOL"], stream["RA_REF"]
    # ADM the parameters that define the extent of the stream.
    mind, maxd = stream["MIND"], stream["MAXD"]

    # ADM limit input coordinates to region of the stream.
    cstream = acoo.SkyCoord(rapol*auni.degree, decpol*auni.degree)
    cobjs = acoo.SkyCoord(objs["RA"]*auni.degree, objs["DEC"]*auni.degree)

    # ADM separation between the objects of interest and the stream.
    sep = cobjs.separation(cstream)    # ADM only retain objects in the stream based on their indexes.
    in_stream = np.where(betw(sep.value, mind, maxd))[0]
    isobjs = objs[in_stream]
    log.info(f"Objects near stream: {len(in_stream)}...t={time()-start:.1f}s")
    del cobjs

    # ADM rotate the position data into the coordinate system of the stream.
    fi1, fi2 = sphere_rotate(isobjs['RA'], isobjs['DEC'], rapol, decpol, ra_ref)

    # ADM distance of the stream.
    dist = stream_distance(fi1, stream_name, stream)

    # CMR note proper motion selection for C19 is in PMRA,PMDec, not stream coords

    # ADM derive the combined proper motion error.
    # CMR: RMS error, appropriate for PM ~< PM_err. See Lindegren GAIA-C3-TN-LU-LL-129-01
    pm_err = np.sqrt(0.5 * (isobjs["PMRA_ERROR"]**2 + isobjs["PMDEC_ERROR"]**2))

    # ADM dust correction.
    ext_coeff = dict(g=3.237, r=2.176, z=1.217)
    eg, er, ez = [ext_coeff[_] * isobjs['EBV'] for _ in 'grz']
    ext = {}
    ext['G'] = eg
    ext['R'] = er
    ext['Z'] = ez

    g, r, z = [22.5 - 2.5 * np.log10(isobjs['FLUX_' + _]) - ext[_] for _ in 'GRZ']

    # ADM some spline functions over which to interpolate.
    # CMR stream track in stream coordinates, phi2(phi1).
    # CMR s=0 puts the spline through all the data points
    TRACK = scipy.interpolate.UnivariateSpline(stream['PHI1T'], stream['PHI2T'], s=0)
    # CMR pmra_cosdec proper motion trace, pm_ra(phi1)
    PMRATRACK = scipy.interpolate.UnivariateSpline(
        stream['PMRA_PHI1T'], stream['PMRAT'], s=0)
    # CMR Dec proper motion trace, pm_dec(phi1)
    PMDECTRACK = scipy.interpolate.UnivariateSpline(
        stream['PMDEC_PHI1T'], stream['PMDECT'], s=0)

    # ADM create an interpolated set of phi2 coords (in stream coords).
    # CMR this is the distance, in phi2, of each star from the track phi2(phi1)
    dfi2 = fi2 - TRACK(fi1)

    # ADM derive the isochrone track for the stream.
    CMD_II = get_CMD_interpolator(stream_name)

    # ADM how far the data lies from the isochrone.
    delta_cmd = g - r - CMD_II(r - 5 * np.log10(dist * 1e3) + 5)

    # ADM necessary parameters are set up; perform the actual selection.
    # CNR bright, faint and intermediate magnitude limits now in yaml file

    # ADM lies in the stream.
    # CMR modified to use limits from yaml file.
    field_sel = betw(dfi2, stream['DPHI2_MINUS'], stream['DPHI2_PLUS'])
    field_sel &= betw(fi1, stream['PHI1_MINUS'], stream['PHI1_PLUS'])

    # ADM Gaia-based selection (proper motion and parallax).
    # CMR pm12_sel_func works with PMRA,PMDec, too
    gaia_astrom_sel = pm12_sel_func(PMRATRACK(fi1), PMDECTRACK(fi1),
                                    isobjs['PMRA'], isobjs['PMDEC'], pm_err,
                                    stream['PM_PAD'], stream['PM_NSIG'])
    # CMR modified to use PLX_NSIG from yaml file. C19 is at mostly the same distance
    # so make a wide plx selection rather than try to do something distance dependent
    gaia_astrom_sel &= plx_sel_func(dist, isobjs, stream['PLX_NSIG'])

    # CMR magnitude ranges
    brightpm1_magsel = (r > stream['BRIGHT_LIMIT']) & (z <= stream['BRIGHTPM1_LIMIT'])
    brightpm2_magsel = (z > stream['BRIGHTPM1_LIMIT']) & (z <= stream['BRIGHTPM2_LIMIT'])
    brightpm3_magsel = (z > stream['BRIGHTPM2_LIMIT']) & (z <= stream['BRIGHTPM3_LIMIT'])

    log.info(f"Objects in the field: {field_sel.sum()}...t={time()-start:.1f}s")
    log.info(f"With correct astrometry: {(gaia_astrom_sel & field_sel).sum()}")

    # ADM padding around the isochrone.
    bright_iso_sel = betw(delta_cmd, -.2, .2)

    # CMR select BHBs by color. Note: no magnitude limits imposed here despite the name
    bright_bhb_sel = oldpop_bhb_sel(g, r, dist)

    # joint CMD selection
    bright_cmd_sel = bright_iso_sel | bright_bhb_sel
    
    # ADM isochrone selection.
    stellar_locus_blue_sel = ((betw(r - z - (-.17 + .67 * (g - r)), -0.2, 0.2)
                               & ((g - r) <= 1.1)))
    stellar_locus_red_sel = (((g - r > 1.1)
                              & betw(g - r - (1.05 + .25 * (r - z)), -.2, .2)))
    stellar_locus_sel = stellar_locus_blue_sel | stellar_locus_red_sel

    # ADM selection for objects that lack Gaia astrometry.
    # ADM has type PSF and in a reasonable isochrone window.
    startyp = _psflike(isobjs["TYPE"])
    cmd_win = 0.1 + 10**(-2 + (r - 20) / 2.5)

    # ADM overall faint selection.
    # CMR modified to use mag limts from yaml file
    faint_cmag_sel = betw(z, stream['FAINT_CMD_LIMIT'], stream['FAINT_LIMIT'])
    faint_cmag_sel &= betw(np.abs(delta_cmd), 0, cmd_win)

    # ADM "filler" selections.
    # (PSF type + blue in colour and not previously selected)
    common_filler_sel = betw(z, stream['BRIGHTPM2_LIMIT'], stream['FAINT_LIMIT'])
    common_filler_sel &= startyp
    common_filler_sel &= stellar_locus_sel

    filler_sel = common_filler_sel & betw(g - r, -.3, 1.2)
    filler_red_sel = common_filler_sel & betw(g - r, 1.2, 2.2)

    bright_pm1 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm1_magsel
    bright_pm2 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm2_magsel
    bright_pm3 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm3_magsel
    bright_pm = bright_pm1 | bright_pm2 | bright_pm3

    # adapted to streams from NRS FAINT_CMD selection
    # NRS passes CMD selection
    # NRS does NOT have Gaia astrometry OR is fainter than BRIGHTPM3_LIMIT
    # NRS passes phi2 stream selection
    # NRS passes Magnitude selection
    # NRS NOT in BRIGHT_PM
    faint_cmd = (
        faint_cmag_sel & field_sel & startyp
        & (~np.isfinite(isobjs["PMRA"]) | (z > stream['BRIGHTPM3_LIMIT']))
        & ~bright_pm
    )

    filler = filler_sel & field_sel & ~bright_pm1 & ~bright_pm2 & ~bright_pm3 & ~faint_cmd

    # CMR moved here to write numbers of final selections, but less useful for timing.
    log.info(f"Objects meeting bright selection: {np.sum(bright_pm)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm1 selection: {np.sum(bright_pm1)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm2 selection: {np.sum(bright_pm2)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm3 selection: {np.sum(bright_pm3)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting faint_cmd selection: {np.sum(faint_cmd)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting filler selection: {np.sum(filler)}...t={time()-start:.1f}s")

    log.info(f"Finished selection for {stream_name}...t={time()-start:.1f}s")

    # ADM sanity check that selections do not overlap.
    check = bright_pm1.astype(int) + bright_pm2.astype(int) + bright_pm3.astype(int) + faint_cmd.astype(int) + filler.astype(int)
    # ADM guard against check being an empty list if there are no targets.
    if len(check) > 0:
        if np.max(check) > 1:
            msg = "Selections should be unique but they overlap!"
            log.error(msg)

    # ADM we sub-selected objects to just those in the stream, so we need
    # ADM to expand back to all of the passed objects. Objects that are
    # ADM not in the stream should be retained as False.
    nobjs = len(objs)
    f_bright_pm1 = np.zeros(nobjs, dtype=bool)
    f_bright_pm2 = np.zeros(nobjs, dtype=bool)
    f_bright_pm3 = np.zeros(nobjs, dtype=bool)
    f_faint_cmd = np.zeros(nobjs, dtype=bool)
    f_filler = np.zeros(nobjs, dtype=bool)
    # return arrys for pm_only and for consistency with dSph and UFD targeting
    f_pm_only = np.zeros(nobjs, dtype=bool)

    f_bright_pm1[in_stream] = bright_pm1
    f_bright_pm2[in_stream] = bright_pm2
    f_bright_pm3[in_stream] = bright_pm3
    f_faint_cmd[in_stream] = faint_cmd
    f_filler[in_stream] = filler

    return f_bright_pm1, f_bright_pm2, f_bright_pm3, f_pm_only, f_faint_cmd, f_filler


def set_target_bits(objs, targmwext_names=["GD1", "BOOTES_1"]):
    """Select stream and dwarf targets, returning target mask arrays.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        numpy structured array with UPPERCASE columns needed for
        stream target selection. See, e.g.,
        :func:`~desitarget.stream.cuts.is_in_GD1` for column names.
    targmwext_names : :class:`list`
        A list of stream names to process. Default is available streams.

    Returns
    -------
    :class:`~numpy.ndarray`
        (desi_target, bgs_target, mws_target, scnd_target) where each
        element is an array of target selection bitmasks for each object.

    Notes
    -----
    - See ../data/targetmask.yaml for the definition of each bit.
    """
    from desitarget.targetmask import desi_mask, mws_mask

    # CMR get a list of all the streams and dwarfs we know about
    fn = resources.files('desitarget').joinpath('data/streams.yaml')
    with open(fn) as f:
        streaminfo = yaml.safe_load(f)
    all_stream_names = list(streaminfo.keys())
    fn2 = resources.files('desitarget').joinpath('data/dwarfs.yaml')
    with open(fn2) as f:
        dwarfinfo = yaml.safe_load(f)
    all_dwarf_names = list(dwarfinfo.keys())
    # ADM set up a zerod mws_target array to |= with later.
    # CMR changed to mws
    mws_target = np.zeros_like(objs["RA"], dtype='int64')

    # ADM might be able to make this more general by putting the
    # ADM bit names in the data/yaml file and using globals()
    # ADM to recover the is_in() functions.

    # CMR updated for extension
    for targmwext in targmwext_names:
        if targmwext in all_dwarf_names:
            bit_name = f"MWS_{targmwext}"
            func_name = "is_in_dwarf"
            func_call = globals()[func_name]
        elif targmwext in all_stream_names:
            bit_name = f"MWS_{targmwext}"
            func_name = f"is_in_{targmwext}"
            func_call = globals()[func_name]

        ibright_pm1, ibright_pm2, ibright_pm3, ipm_only, ifaint_cmd, ifiller = func_call(
            objs, targmwext)

        bright_pm1, bright_pm2, bright_pm3, pm_only, faint_cmd, filler = targmwext_resolve(
            targmwext, mws_target, ibright_pm1, ibright_pm2, ibright_pm3, ipm_only,
            ifaint_cmd, ifiller)

        # ADM/CMR set mws desi extension bit
        any_set = bright_pm1 | bright_pm2 | bright_pm3 | pm_only | faint_cmd | filler
        mws_target |= any_set * mws_mask.MWS_EXT
        # CMR set stream name bit
        mws_target |= any_set * mws_mask[bit_name]
        # CMR now set target subclass bit masks
        mws_target |= faint_cmd * mws_mask.MWS_FAINT_CMD
        mws_target |= filler * mws_mask.MWS_FILLER
        mws_target |= pm_only * mws_mask.MWS_PM_ONLY
        targmwextpar = get_targmwext_parameters(targmwext)
        target_bit_set = targmwextpar["TARGET_BIT_SET"]
        if target_bit_set == "STREAM":
            mws_target |= bright_pm1 * mws_mask.MWS_STREAM_PM1
            mws_target |= bright_pm2 * mws_mask.MWS_STREAM_PM2
            mws_target |= bright_pm3 * mws_mask.MWS_STREAM_PM3
        elif target_bit_set == "DSPH":
            mws_target |= bright_pm1 * mws_mask.MWS_DSPH_PM1
            mws_target |= bright_pm2 * mws_mask.MWS_DSPH_PM2
            mws_target |= bright_pm3 * mws_mask.MWS_DSPH_PM3
        elif target_bit_set == "UFD":
            mws_target |= bright_pm1 * mws_mask.MWS_UFD_PM1
            mws_target |= bright_pm2 * mws_mask.MWS_UFD_PM2
            mws_target |= bright_pm3 * mws_mask.MWS_UFD_PM3

    # ADM tell DESI_TARGET where MWS_ANY was updated.
    # CMR updated to MWS.
    desi_target = (mws_target != 0) * desi_mask.MWS_ANY

    # ADM/CMR set BGS_TARGET and SCND_TARGET to zeros.
    bgs_target = np.zeros_like(mws_target)
    scnd_target = np.zeros_like(mws_target)

    return desi_target, bgs_target, mws_target, scnd_target


def select_targets(swdir, targnames_in=None, readpertarg=False,
                   addnors=True, readcache=True, numproc=1, mindec=-20,
                   nside=None, pixint=None):
    """Process files from an input directory to select targets.

    Parameters
    ----------
    swdir : :class:`str`
        Root directory of Legacy Surveys sweep files for a given data
        release for ONE of EITHER north or south, e.g.
        "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0".
    targnames_in : :class:`list`
        A list of MWS extension target classes to process. Defaults to
        all MWS extension target classes. This will get sorted by rank.
    readpertarg : :class:`bool`, optional, defaults to ``False``
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
        from caches AND OVERWRITE any cached files, if they exist. Caches
        are stored in the $TARG_DIR/streamcache/drX/ directory, where drX
        is the Legacy Surveys Data Release (parsed from `swdir`).
    numproc : :class:`int`, optional, defaults to 1 for serial
        The number of parallel processes to use. `numproc` of 16 is a
        good balance between speed and file I/O.
    mindec : :class:`float` or `int`, optional, defaults to -20 (20oS)
        Hard limit on data (objects south of this are not returned).
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPixel nside used with `pixint`. Only used if
        `readpertarg` is ``False``.
    pixint : :class:`int`, optional, defaults to `None`
        Only read and cache targets in (NESTED) HEALpixels at `nside`.
        For parallelizing. Only used if `readpertarg` is ``False``.

    Returns
    -------
    :class:`~numpy.ndarray`
        Targets in the input `swdir` which pass the cuts with added
        targeting columns such as ``TARGETID``, and ``DESI_TARGET``,
        ``BGS_TARGET``, ``MWS_TARGET`` (i.e. target selection bitmasks).
    """
    # ADM ensure the default is to run everything.
    if targnames_in is None:
        fn = resources.files('desitarget').joinpath('data/streams.yaml')
        with open(fn) as f:
            streams = list(yaml.safe_load(f).keys())
        fn = resources.files('desitarget').joinpath('data/dwarfs.yaml')
        with open(fn) as f:
            dwarfs = list(yaml.safe_load(f).keys())
        targnames_in = streams + dwarfs
    log.info(f"Running these target classes: {targnames_in}")

    # CMR rank order the streams and dwarfs for which we are selecting
    # targets, using TARGMWEXT_RANK in the yaml files. This is for
    # disambiguating possible duplicate targets if we have streams
    # or dwarfs that overlap on the sky
    targnames = sort_targmwext_by_rank(targnames_in)

    # CMR get a list of all the streams and dwarfs we know about
    fn = resources.files('desitarget').joinpath('data/streams.yaml')
    with open(fn) as f:
        streaminfo = yaml.safe_load(f)
    all_stream_names = list(streaminfo.keys())
    fn2 = resources.files('desitarget').joinpath('data/dwarfs.yaml')
    with open(fn2) as f:
        dwarfinfo = yaml.safe_load(f)
    all_dwarf_names = list(dwarfinfo.keys())

    if readpertarg:
        # ADM loop over streams and read in the data per stream or dwarf
        allobjs = []
        for targ in targnames:
            if targ in all_dwarf_names:
                # NRS look up the defining parameters of the dwarf.
                dwarf = get_targmwext_parameters(targ)
                # NRS galaxy coordinates in degrees.
                ra0, dec0 = dwarf["RA"], dwarf["DEC"]
                # NRS spatial extent in degrees for initial data read.
                maxd = dwarf["MAXD"]
                mind = 0
                # NRS read in the data. CMR reuse read_data_per_stream
                objs = read_data_per_stream(
                    swdir, ra0, dec0, mind, maxd, targ, numproc=numproc,
                    mindec=mindec, addnors=addnors, readcache=readcache, readall=False
                )
            elif targ in all_stream_names:
                # ADM read in the data.
                strm = get_targmwext_parameters(targ)
                # ADM the parameters that define the coordinates of the stream.
                rapol, decpol, ra_ref = strm["RAPOL"], strm["DECPOL"], strm["RA_REF"]
                # ADM the parameters that define the extent of the stream.
                mind, maxd = strm["MIND"], strm["MAXD"]
                # ADM read in the data.
                objs = read_data_per_stream(
                    swdir, rapol, decpol, mind, maxd, targ, numproc=numproc,
                    mindec=mindec, addnors=addnors, readcache=readcache, readall=False
                )
            else:
                log.info(f"{targ} not found in list of known streams and dwarfs")
            allobjs.append(objs)
        objects = np.concatenate(allobjs)
    else:
        # ADM otherwise, read in all of the sweeps. This requires
        # ADM some dummy inputs.
        objects = read_data_per_stream(
            swdir, 0, 0, 0, 0, "", numproc=numproc, mindec=mindec,
            addnors=addnors, readcache=readcache, nside=nside, pixint=pixint,
            readall=True)

    # ADM process the targets.
    desi_target, bgs_target, mws_target, scnd_target = set_target_bits(
        objects, targmwext_names=targnames)

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

    # ADM prudent to check we don't have duplicate TARGETIDs as we're
    # ADM potentially processing overlapping streams.
    if len(np.unique(targets["TARGETID"])) != len(targets):
        msg = ("Targets must be unique but there are some duplicated TARGETIDs!")
        log.error(msg)

    # ADM a final sort on RA to mitigate reproducibility issues.
    # ADM for instance, we've had conflicting SUBPRIORITY in the past.
    ii = np.argsort(targets)
    targets = targets[ii]

    return targets
