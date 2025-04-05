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
import astropy.coordinates as acoo
import astropy.units as auni

from desitarget.cuts import _psflike
from desitarget.streams.utilities import sphere_rotate, correct_pm, rotate_pm, \
    betw, pm12_sel_func, plx_sel_func, get_CMD_interpolator, stream_distance,  \
    get_stream_parameters, simple_plx_sel, oldpop_bhb_sel, pm12_distdep_sel_func
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
        ``True`` if the object is a bright "BRIGHT_PM1" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM2" target.
    :class:`array_like`
        ``True`` if the object is a bright "BRIGHT_PM3" target.
    :class:`array_like`
        ``True`` if the object is a faint "FAINT_NO_PM" target.
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
    stream = get_stream_parameters(stream_name)
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
    # gaia_astrom_sel &= r > stream['BRIGHT_LIMIT'] CWR don't need this.

    # CMR magnitude ranges
    brightpm1_magsel = (r > stream['BRIGHT_LIMIT']) & (r <= stream['BRIGHTPM1_LIMIT'])
    brightpm2_magsel = (r > stream['BRIGHTPM1_LIMIT']) & (r <= stream['BRIGHTPM2_LIMIT'])
    brightpm3_magsel = (r > stream['BRIGHTPM2_LIMIT']) & (r <= stream['BRIGHTPM3_LIMIT'])

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
    faint_sel = ~np.isfinite(isobjs['PMRA'])  # CWR no PM information.
    faint_sel &= betw(r, stream['FAINT_NO_PM_LIMIT'], stream['FAINT_LIMIT'])
    faint_sel &= betw(np.abs(delta_cmd), 0, cmd_win)
    faint_sel &= startyp
    faint_sel &= stellar_locus_sel

    # ADM "filler" selections.
    # (PSF type + blue in colour and not previously selected)
    # CMR modified to use limits from yaml file
    common_filler_sel = betw(r, stream['BRIGHTPM2_LIMIT'], stream['FAINT_LIMIT'])
    common_filler_sel &= startyp
    common_filler_sel &= ~faint_sel
    # CMR no: want objs with gaia astrom fainter than bright_pm3 faint lim.
    # common_filler_sel &= ~gaia_astrom_sel
    common_filler_sel &= stellar_locus_sel

    filler_sel = common_filler_sel & betw(g - r, -.3, 1.2)
    filler_red_sel = common_filler_sel & betw(g - r, 1.2, 2.2)

    bright_pm = bright_cmd_sel & gaia_astrom_sel & field_sel
    bright_pm1 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm1_magsel
    bright_pm2 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm2_magsel
    bright_pm3 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm3_magsel
    faint_no_pm = faint_sel & field_sel
    filler = filler_sel & field_sel & ~bright_pm1 & ~bright_pm2 & ~bright_pm3

    # CMR moved here to write numbers of final selections, but less useful for timing.
    log.info(f"Objects meeting bright selection: {np.sum(bright_pm)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm1 selection: {np.sum(bright_pm1)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm2 selection: {np.sum(bright_pm2)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm3 selection: {np.sum(bright_pm3)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting faint_no_pm selection: {np.sum(faint_no_pm)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting filler selection: {np.sum(filler)}...t={time()-start:.1f}s")

    log.info(f"Finished selection for {stream_name}...t={time()-start:.1f}s")

    # ADM sanity check that selections do not overlap.
    check = bright_pm1.astype(int) + bright_pm2.astype(int) + bright_pm3.astype(int) + faint_no_pm.astype(int) + filler.astype(int)
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
    f_faint_no_pm = np.zeros(nobjs, dtype=bool)
    f_filler = np.zeros(nobjs, dtype=bool)

    f_bright_pm1[in_stream] = bright_pm1
    f_bright_pm2[in_stream] = bright_pm2
    f_bright_pm3[in_stream] = bright_pm3
    f_faint_no_pm[in_stream] = faint_no_pm
    f_filler[in_stream] = filler

    return f_bright_pm1, f_bright_pm2, f_bright_pm3, f_faint_no_pm, f_filler

# based on Sergey's is_in_GD1
def is_in_ORPHAN(objs):
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
        ``True`` if the object is a faint "FAINT_NO_PM" target.
    :class:`array_like`
        ``True`` if the object is a white dwarf "FILLER" target.
    """
    # ADM start the clock.
    start = time()

    stream_name = "ORPHAN"
    
    log.info(f"Starting selection for {stream_name}...t={time()-start:.1f}s")
    
    # CMR get the defining parameters of the stream
    stream = get_stream_parameters(stream_name)
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
    eg, er, ez = [ext_coeff[_] * objs['EBV'] for _ in 'grz']
    ext = {}
    ext['G'] = eg
    ext['R'] = er
    ext['Z'] = ez

    g, r, z = [22.5 - 2.5 * np.log10(objs['FLUX_' + _]) - ext[_] for _ in 'GRZ']

    # ADM some spline functions over which to interpolate.
    # CMR stream track in stream coordinates, phi2(phi1)
    TRACK = scipy.interpolate.CubicSpline(stream['PHI1T'], stream['PHI2T'])
    # CMR phi1_cosphi2 proper motion trace, pm_phi1(phi1)
    PM1TRACK = scipy.interpolate.UnivariateSpline(stream['PMPHI1_PHI1T'], stream['PMPHI1T'], s=0.02,ext=3)
    # CMR phi2 proper motion trace, pm_phi2(phi1)
    PM2TRACK = scipy.interpolate.UnivariateSpline(stream['PMPHI2_PHI1T'], stream['PMPHI2T'], s=0.02,ext=3)
    
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
    #pm_pad = 0.15
    #gaia_astrom_sel = pm12_sel_func(PM1TRACK(fi1), PM2TRACK(fi1), pmfi1, pmfi2,
    #                                pm_err, stream['PM_PAD'], stream['PM_NSIG'])
    gaia_astrom_sel = pm12_distdep_sel_func(PM1TRACK(fi1), PM2TRACK(fi1), pmfi1, pmfi2,
                                            pm_err, dist, stream['VEL_PAD'], stream['PM_NSIG'])
    plx_sel = simple_plx_sel(dist, isobjs, stream['PLX_NSIG'], 0.1)
    gaia_astrom_sel &= plx_sel
    #gaia_astrom_sel &= r > bright_limit

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
    brightpm1_magsel = (r > stream['BRIGHT_LIMIT']) & (r <= stream['BRIGHTPM1_LIMIT'])
    brightpm2_magsel = (r > stream['BRIGHTPM1_LIMIT']) & (r <= stream['BRIGHTPM2_LIMIT'])
    brightpm3_magsel = (r > stream['BRIGHTPM2_LIMIT']) & (r <= stream['BRIGHTPM3_LIMIT'])

    # ADM isochrone selection.
    stellar_locus_blue_sel = ((betw(r - z - (-.17 + .67 * (g - r)), -0.2, 0.2)
                               & ((g - r) <= 1.1)))
    stellar_locus_red_sel = (((g - r > 1.1)
                              & betw(g - r - (1.05 + .25 * (r - z)), -.2, .2)))
    stellar_locus_sel = stellar_locus_blue_sel | stellar_locus_red_sel

    #tot = np.sum(field_sel & gaia_astrom_sel & bright_cmd_sel)
    #print(f"Obj meeting bright selection: {tot}...t={time()-start:.1f}s")

    # ADM selection for objects that lack Gaia astrometry.
    # ADM has type PSF and in a reasonable isochrone window.
    startyp = _psflike(objs["TYPE"])
    cmd_win = 0.1 + 10**(-2 + (r - 20) / 2.5)

    # CMR overall faint selection, using limits from yaml file
    faint_sel = ~np.isfinite(isobjs['PMRA'])
    faint_sel &= betw(r, stream['FAINT_NO_PM_LIMIT'], stream['FAINT_LIMIT'])
    faint_sel &= betw(np.abs(delta_cmd), 0, cmd_win)
    faint_sel &= startyp
    faint_sel &= stellar_locus_sel
    #tot = np.sum(faint_sel & field_sel)
    #log.info(f"Objects meeting faint selection: {tot}...t={time()-start:.1f}s")

    # ADM "filler" selections.
    # (PSF type + blue in colour and not previously selected)
    common_filler_sel = betw(r, stream['BRIGHTPM2_LIMIT'], stream['FAINT_LIMIT'])
    common_filler_sel &= startyp
    common_filler_sel &= ~faint_sel
    #common_filler_sel &= ~gaia_astrom_sel
    common_filler_sel &= stellar_locus_sel

    filler_sel = common_filler_sel & betw(g - r, -.3, 1.2)

    filler_red_sel = common_filler_sel & betw(g - r, 1.2, 2.2)
    #tot = np.sum(filler_sel & field_sel)
    #log.info(f"Objects meeting filler selection: {tot}...t={time()-start:.1f}s")

    bright_pm = bright_cmd_sel & gaia_astrom_sel & field_sel
    bright_pm1 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm1_magsel
    bright_pm2 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm2_magsel
    bright_pm3 = bright_cmd_sel & gaia_astrom_sel & field_sel & brightpm3_magsel
    faint_no_pm = faint_sel & field_sel
    filler = filler_sel & field_sel & ~bright_pm1 & ~bright_pm2 & ~bright_pm3

    log.info(f"Objects meeting bright selection: {np.sum(bright_pm)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm1 selection: {np.sum(bright_pm1)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm2 selection: {np.sum(bright_pm2)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting bright pm3 selection: {np.sum(bright_pm3)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting faint_no_pm selection: {np.sum(faint_no_pm)}...t={time()-start:.1f}s")
    log.info(f"Objects meeting filler selection: {np.sum(filler)}...t={time()-start:.1f}s")

    log.info(f"Finished selection for {stream_name}...t={time()-start:.1f}s")
    
    # ADM sanity check that selections do not overlap.
    check = bright_pm1.astype(int) + bright_pm2.astype(int) + bright_pm3.astype(int) + faint_no_pm.astype(int) + filler.astype(int)
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
    f_faint_no_pm = np.zeros(nobjs, dtype=bool)
    f_filler = np.zeros(nobjs, dtype=bool)

    f_bright_pm1[in_stream] = bright_pm1
    f_bright_pm2[in_stream] = bright_pm2
    f_bright_pm3[in_stream] = bright_pm3
    f_faint_no_pm[in_stream] = faint_no_pm
    f_filler[in_stream] = filler

    return f_bright_pm1, f_bright_pm2, f_bright_pm3, f_faint_no_pm, f_filler
    

def set_target_bits(objs, stream_names=["GD1"]):
    """Select stream targets, returning target mask arrays.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        numpy structured array with UPPERCASE columns needed for
        stream target selection. See, e.g.,
        :func:`~desitarget.stream.cuts.is_in_GD1` for column names.
    stream_names : :class:`list`
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

    # ADM set up a zerod mws_target array to |= with later.
    # CMR changed to mws
    mws_target = np.zeros_like(objs["RA"], dtype='int64')

    # ADM might be able to make this more general by putting the
    # ADM bit names in the data/yaml file and using globals()
    # ADM to recover the is_in() functions.

    # CMR updated for extension
    for stream in stream_names:
        bit_name = f"MWS_{stream}"
        func_name = f"is_in_{stream}"
        func_call = globals()[func_name]

        bright_pm1, bright_pm2, bright_pm3, faint_no_pm, filler = func_call(objs)

        # ADM/CMR set mws desi extension bit
        any_set = bright_pm1 | bright_pm2 | bright_pm3 | faint_no_pm | filler
        mws_target |= any_set * mws_mask.MWS_EXT
        # CMR set stream name bit
        mws_target |= any_set * mws_mask[bit_name]
        # CMR now set target subclass bit masks
        mws_target |= bright_pm1 * mws_mask.MWS_BRIGHT_PM1
        mws_target |= bright_pm2 * mws_mask.MWS_BRIGHT_PM2
        mws_target |= bright_pm3 * mws_mask.MWS_BRIGHT_PM3
        mws_target |= faint_no_pm * mws_mask.MWS_FAINT_NO_PM
        mws_target |= filler * mws_mask.MWS_FILLER

    # ADM tell DESI_TARGET where MWS_ANY was updated.
    # CMR updated to MWS.
    desi_target = (mws_target != 0) * desi_mask.MWS_ANY

    # ADM/CMR set BGS_TARGET and SCND_TARGET to zeros.
    bgs_target = np.zeros_like(mws_target)
    scnd_target = np.zeros_like(mws_target)

    return desi_target, bgs_target, mws_target, scnd_target


def select_targets(swdir, stream_names=["GD1"], readperstream=False,
                   addnors=True, readcache=True, numproc=1, mindec=-20):
    """Process files from an input directory to select targets.

    Parameters
    ----------
    swdir : :class:`str`
        Root directory of Legacy Surveys sweep files for a given data
        release for ONE of EITHER north or south, e.g.
        "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0".
    stream_names : :class:`list`
        A list of stream names to process. Defaults to all streams.
    readperstream : :class:`bool`, optional, defaults to ``False``
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
    numproc : :class:`int`, optional, defaults to 1 for serial
        The number of parallel processes to use. `numproc` of 16 is a
        good balance between speed and file I/O.
    mindec : :class:`float` or `int`, optional, defaults to -20 (20oS)
        Hard limit on data (objects south of this are not returned).

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
        allobjs = []
        for stream_name in stream_names:
            # ADM read in the data.
            strm = get_stream_parameters(stream_name)
            # ADM the parameters that define the coordinates of the stream.
            rapol, decpol, ra_ref = strm["RAPOL"], strm["DECPOL"], strm["RA_REF"]
            # ADM the parameters that define the extent of the stream.
            mind, maxd = strm["MIND"], strm["MAXD"]
            # ADM read in the data.
            objs = read_data_per_stream(
                swdir, rapol, decpol, mind, maxd, stream_name, numproc=numproc,
                mindec=mindec, addnors=addnors, readcache=readcache, readall=False
            )
            allobjs.append(objs)
        objects = np.concatenate(allobjs)
    else:
        # ADM otherwise, read in all of the sweeps. This requires
        # ADM some dummy inputs.
        objects = read_data_per_stream(
            swdir, 0, 0, 0, 0, "", numproc=numproc, mindec=mindec,
            addnors=addnors, readcache=readcache, readall=True)

    # ADM process the targets.
    desi_target, bgs_target, mws_target, scnd_target = set_target_bits(
        objects, stream_names=stream_names)

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

    # ADM a final sort on RA to mitigate reproducibility issues.
    # ADM for instance, we've had conflicting SUBPRIORITY in the past.
    ii = np.argsort(targets)
    targets = targets[ii]

    return targets
