import astropy.units as auni
import astropy.coordinates as acoo
import astropy.table as atpy
import numpy as np
import sys
from matplotlib.path import Path

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()

# Define the center points of M31 and M33
m31_ra, m31_dec = 10.68470833, 41.26875
m33_ra, m33_dec = 23.461667, 30.6602


def cosd(x):
    return np.cos(np.deg2rad(x))


def sind(x):
    return np.sin(np.deg2rad(x))


def betw(x, xmin, xmax):
    imask = ((x > xmin) & (x < xmax))
    return imask


plx_pad, pm_pad = 0.05, 0.05


def downsampler(ra, dec, gal_b, subset, p_15, p_35, m31_rad=2.5, m33_rad=.5):
    """
    Compute the probability for downsampling to select with p_15 prob
    at b=-15 and p_35 at b=-35
    """
    prob_sel = np.ones(len(gal_b))
    p_m31_central = 1
    p_m33_central = 1
    m31_dist = acoo.SkyCoord(ra=ra * auni.deg, dec=dec * auni.deg).separation(
        acoo.SkyCoord(ra=m31_ra * auni.deg, dec=m31_dec * auni.deg)).deg
    m33_dist = acoo.SkyCoord(ra=ra * auni.deg, dec=dec * auni.deg).separation(
        acoo.SkyCoord(ra=m31_ra * auni.deg, dec=m31_dec * auni.deg)).deg
    in_m31 = betw(m31_dist, 0, m31_rad)
    in_m33 = betw(m33_dist, 0, m33_rad)
    prob_sel[subset] = np.clip((20 * p_15 * p_35 / (20 * p_15 + (p_35 - p_15) *
                                                    (gal_b[subset] + 35))), 0,
                               1)
    # Set the probability in regions around M31 and M33 to be unity
    prob_sel[subset & in_m31] = p_m31_central
    prob_sel[subset & in_m33] = p_m33_central
    return prob_sel


def insideM31disk(ra, dec):  # Added by Arjun
    """
    Check whether the ra,dec points lie within the ellipse defining the M31
    disk
    """
    # Define the center of M31
    m31ra, m31dec = 10.68470833, 41.26875

    # Define the ellipse for M31
    a_e_m31 = 1.5
    b_e_m31 = 0.337
    PA0_m31 = 45.0  # major axis position angle in degrees
    PA_m31 = 90 - PA0_m31  # redefine the position angle so that it is
    # measured counterclockwise from the EW axis

    # Rotate coordinates and identify the sources within the M31 disk
    ra_rot = (ra - m31ra) * cosd(PA_m31) + (dec - m31dec) * sind(PA_m31)
    dec_rot = -(ra - m31ra) * sind(PA_m31) + (dec - m31dec) * cosd(PA_m31)
    inside_m31 = (ra_rot / a_e_m31)**2 + (dec_rot / b_e_m31)**2 <= 1

    return inside_m31


def insideM33disk(ra, dec):
    """
    Check whether the ra,dec points lie within the ellipse defining the M33
    disk
    """

    # Define the center of M33
    m33ra, m33dec = 23.461667, 30.6602

    # Define the ellipse for M33
    a_e_m33 = 0.5
    b_e_m33 = 0.337
    PA0_m33 = 35.0  # major axis position angle in degrees
    PA_m33 = 90 - PA0_m33  # redefine the position angle so that it is
    # measured counterclockwise from the EW axis

    # Rotate coordinates and identify the sources within the M31 disk
    ra_rot = (ra - m33ra) * cosd(PA_m33) + (dec - m33dec) * sind(PA_m33)
    dec_rot = -(ra - m33ra) * sind(PA_m33) + (dec - m33dec) * cosd(PA_m33)
    inside_m33 = (ra_rot / a_e_m33)**2 + (dec_rot / b_e_m33)**2 <= 1

    return inside_m33


def select_m31_all(objs, remove_observed=True):
    """
    Return boolean masks for each target class from the object catalogue.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Object catalog. Prepared by Sergey Koposov. File containing
        typical `objs` is available at $TARG_DIR/../sergey_m31/, which
        is also /global/cfs/cdirs/desi/target/sergey_m31/.
    remove_observed : :class:`bool`, optional, defaults to ``True``
        If ``True`` then remove existing targets observed as part of the
        M31/DARK tertiary program.

    Returns
    -------
    :class:`tuple`
        A seven-entry tuple of Boolean masks for the following target
        classes in order. The three red-giant branch (RGB) classes are
        selected from PAndAS and differ on the basis of metallicity:

        RGBLOW: Candidate RGB star; _GIANT target class.
        RGBHIGH: Candidate RGB star; _GIANT target class.
        AGB: (very red) Candidate RGB star; _GIANT target class.
        QSO: QSO candidates selected from Gaia/WISE; _QSO target class.
        BRIGHT: M31, or MW outer halo, stars from Gaia; _BRIGHT class.
        FILLER: Filler sources selected from Gaia; _FILLER target class. 
        CLUSTER: Special sources selected from a range of smaller
        catalogs (GCs, OCs, PNs, HII regions); _SPECIAL target class.
    """
    pandas_g = objs['PANDAS_G']
    pandas_i = objs['PANDAS_I']
    gaia_bp, gaia_rp, gaia_G = objs['GAIA_PHOT_BP_MEAN_MAG'], objs[
        'GAIA_PHOT_RP_MEAN_MAG'], objs['GAIA_PHOT_G_MEAN_MAG']
    wise_w1 = objs['W1MPRO']
    wise_w2 = objs['W2MPRO']
    ra, dec = objs['RA'], objs['DEC']
    pmra, pmdec, parallax = objs['PMRA'], objs['PMDEC'], objs['PARALLAX']
    aen = objs['ASTROMETRIC_EXCESS_NOISE']
    pmra_error = np.zeros_like(ra) + np.nan
    pmdec_error = np.zeros_like(ra) + np.nan
    parallax_error = np.zeros_like(ra) + np.nan
    gal_b = acoo.SkyCoord(ra=ra * auni.deg,
                          dec=dec * auni.deg).transform_to(acoo.Galactic).b.deg
    for col in ['PMRA_IVAR', 'PMDEC_IVAR', 'PARALLAX_IVAR']:
        curarr = {
            'PMRA_IVAR': pmra_error,
            'PMDEC_IVAR': pmdec_error,
            'PARALLAX_IVAR': parallax_error
        }[col]
        curarr[objs[col] > 0] = 1. / objs[col][objs[col] > 0]**.5

    Tgz = (np.maximum(
        ((pandas_g - pandas_i) - 1.8) * 0.15, -(pandas_g - pandas_i) - 1.8) *
           0.03) - 0.13 + 1.3 * (pandas_g - pandas_i)
    zmag = pandas_g - Tgz
    # For the Gaia-only sources we use a different relation
    zmag[~np.isfinite(zmag)] = (
        0.36 + gaia_G - 0.7 * (gaia_bp - gaia_rp) + 0.06 *
        (gaia_bp - gaia_rp - 1.5)**2)[~np.isfinite(zmag)]

    # Correct the PAndAS photometry for the MW dust extinction using the
    # Schlegel-Finkbeiner-Davis dust maps
    ebv = objs["EBV"]
    # ebv1 = np.clip(ebv, 0, .1 + ((XP_dx**2 + XP_dy**2) > 1.5**2).astype(int))
    XP_dx, XP_dy = (((ra + 180) % 360 - 180 - m31_ra) *
                    np.cos(np.deg2rad(m31_dec)), (dec - m31_dec))
    ebv1 = np.where((XP_dx**2 + XP_dy**2) > 1.5**2, ebv, np.clip(ebv, 0, .09))
    ext_g = 3.8  # rounded from ibata 2014
    ext_i = 2.1
    ext_z = 1.211  # Legacy Surveys website:
    ext_bp = 3.02  # using extinction_coefficient
    ext_rp = 1.81  #
    ext_G = 2.32  #
    ext_w1 = 0.18
    ext_w2 = 0.11
    # https://www.legacysurvey.org/dr10/catalogs/#galactic-extinction-coefficients
    g0 = pandas_g - ext_g * ebv1
    i0 = pandas_i - ext_i * ebv1
    z0 = zmag - ext_z * ebv1
    gaia_G0 = gaia_G - ext_G * ebv1
    gaia_bp0 = gaia_bp - ext_bp * ebv1
    gaia_rp0 = gaia_rp - ext_rp * ebv1
    wise_w1_0 = wise_w1 - ext_w1 * ebv1
    wise_w2_0 = wise_w2 - ext_w2 * ebv1

    astrom_sel = ((np.abs(pmra) < pm_pad + 2 * pmra_error) &
                  (np.abs(pmdec) < pm_pad + 2 * pmdec_error) &
                  (parallax < plx_pad + 2 * parallax_error) |
                  (~np.isfinite(parallax_error)))

    # Define the polygons for the three regions
    vertices1 = [(1, 21.7), (1.9, 21.7), (2.0, 20.2), (1.3, 20.4),
                 (1, 21.7)]  # RGB Low metallicity
    vertices2 = [(1.9, 21.7), (2.7, 21.7), (2.7, 20.2), (2.0, 20.2),
                 (1.9, 21.7)]  # RGB Int/High metallicity
    vertices3 = [(2.7, 21.7), (4.5, 21.7), (4.5, 19.9), (2.7, 20.2),
                 (2.7, 21.7)]  # RGB High metallicity + AGB

    # make a Path object
    path_rgblo = Path(vertices1)
    path_rgbhi = Path(vertices2)
    path_agb = Path(vertices3)

    # Define the space
    x = g0 - i0
    y = z0
    points = np.column_stack((x, y))

    # Test which points are inside the polygon
    inrgblo = path_rgblo.contains_points(points)
    inrgbhi = path_rgbhi.contains_points(points)
    inagb = path_agb.contains_points(points)

    # Define the good photometry mask
    goodphot = (g0 > 0) & (i0 > 0) & (objs['TYPE'] == 'PPSF')

    # Final masks for target selection
    rgblo_sel = astrom_sel & goodphot & inrgblo
    rgbhi_sel = astrom_sel & goodphot & inrgbhi
    agb_sel = astrom_sel & goodphot & inagb

    # Define a function for scaling the probability as a function of GALB but
    # only for the rgbhi selection
    p_15 = 0.25
    p_35 = 1
    rgbhi_prob_sel = downsampler(ra, dec, gal_b, rgbhi_sel, p_15, p_35)

    sel = objs['RANDOM'] < rgbhi_prob_sel
    rgbhi_sel = rgbhi_sel & sel
    # NOTE THIS IS A DIFFERENT AEN RELATION
    gaia_aen_star_sel = aen < 10.**(0.4 + 0.25 * (gaia_G - 19.0))
    # THE PREVIOUS WAS NOT CORRECT
    # this was validated by looking at *dr3* aen of type=PSF vs not PSF

    qso_Gminmag = 14.0
    qso_Gmaxmag = 21.0
    qso_sel = (gaia_G > 0) & (wise_w1 > 0) & (wise_w2 > 0)
    qso_sel &= (gaia_G >= qso_Gminmag) & (gaia_G <= qso_Gmaxmag)
    qso_sel &= astrom_sel
    qso_sel &= gaia_aen_star_sel
    qso_sel &= (((wise_w1_0 - wise_w2_0) > (1.0 - 0.125 *
                                            (gaia_G0 - wise_w1_0)))
                & ((wise_w1_0 - wise_w2_0) > 0.5))
    Tbprp = gaia_bp0 - gaia_rp0
    qso_sel &= (gaia_G0 <= (26.46 - 5.991 * Tbprp + 1.313 * Tbprp * Tbprp -
                            0.07856 * Tbprp**3))
    qso_sel &= (gaia_bp < 21)  # it's broken beyond
    desi1_sgc_declim = 32.5  # Northern Declination limit of DESI-1 targeting
    # in the SGC
    qso_sel &= (dec >= desi1_sgc_declim)

    # ## Select the Gaia Bright Source Targets for M31

    bright_Gminmag = 16.0
    bright_Gmaxmag = 21.0

    bright_sel = (gaia_G >= bright_Gminmag) & (gaia_G <= bright_Gmaxmag)
    bright_sel &= astrom_sel
    bright_sel &= gaia_aen_star_sel

    inside_m31 = insideM31disk(ra, dec)
    inside_m33 = insideM33disk(ra, dec)

    bright_sel = bright_sel & (~qso_sel)
    bright_sel |= (
        (inside_m31 | inside_m33)  # Added by Arjun- this adds ~ 30k sources
        & (gaia_G >= bright_Gminmag) & (gaia_G <= bright_Gmaxmag))

    filler_Gminmag = 16.0
    filler_Gmaxmag = 21.

    filler_sel = (gaia_G >= filler_Gminmag) & (gaia_G <= filler_Gmaxmag)
    filler_sel &= ((np.abs(pmra) - 5.0 * pmra_error <= 0.1) &
                   (np.abs(pmdec) - 5.0 * pmdec_error <= 0.1))
    filler_sel &= gaia_aen_star_sel
    filler_sel = filler_sel & (~bright_sel) & (~qso_sel)

    cluster_sel = objs['IS_SPECIAL_OBJECT']
    if remove_observed:
        no = ~objs['RVS_FLAG']
    else:
        no = np.ones(len(objs), dtype=bool)
    return (
        rgblo_sel & no,  # RGBLOW
        rgbhi_sel & no,  # RGBHIGH
        agb_sel & no,  # AGB(very red)
        qso_sel & no,  # QSO
        bright_sel & no,  # BRIGHT
        filler_sel & no,  # FILLER
        cluster_sel & no  # CLUSTER
    )


def select_targets(filename):
    """Process an input file to select M31/M33 1B targets.

    Parameters
    ----------
    filename : :class:`str`
    objs : :class:`~numpy.ndarray`
        Name of file containing input catalog. Made by Sergey Koposov.
        Typical file is available at $TARG_DIR/../sergey_m31/, which
        is also /global/cfs/cdirs/desi/target/sergey_m31/.

    Returns
    -------
    :class:`~numpy.ndarray`
        Targets in the input `swdir` which pass the cuts with added
        targeting columns such as ``TARGETID``, and ``DESI_TARGET``
        ``BGS_TARGET``, ``MWS_TARGET`` (i.e. target selection bitmasks).
    """
    # ADM read in the targets.
    objs = atpy.Table().read(filename, mask_invalid=False)

    # ADM check the data model.
    from desitarget.streams.io import streamcolsLS, streamcolsGaia
    Mdescr = streamcolsLS.dtype.descr + streamcolsGaia.dtype.descr
    # ADM need to switch ERRORs to iVARs in the data model.
    Mdescr = [(i[0], i[1]) if not "ERROR" in i[0] else
              (i[0].replace("ERROR", "IVAR"), i[1]) for i in Mdescr]
    mismatchdm = ~np.array([i in objs.dtype.descr for i in Mdescr])
    if np.sum(mismatchdm) > 0:
        badcols = np.array(Mdescr)[mismatchdm]
        msg = f"Required data model wrong for:\n {badcols}\n"
        msg += f"Compare to:\n {objs.dtype.descr}"
        log.critical(msg)

    # ADM deterine the target classes.
    (rgblo_sel, rgbhi_sel, agb_sel, qso_sel, bright_sel, filler_sel,
     cluster_sel) = select_m31_all(objs)

    # ADM initialize an empty array for the target bits.
    from desitarget.targetmask import desi_mask, mws_mask
    mws_target = np.zeros_like(objs["RA"], dtype='int64')

    # ADM set up the target bits.
    mws_target |= rgblo_sel * mws_mask.M31_GIANT
    mws_target |= rgbhi_sel * mws_mask.M31_GIANT
    mws_target |= agb_sel * mws_mask.M31_GIANT
    mws_target |= qso_sel * mws_mask.M31_QSO
    mws_target |= bright_sel * mws_mask.M31_BRIGHT
    mws_target |= filler_sel * mws_mask.M31_FILLER
    mws_target |= cluster_sel * mws_mask.M31_SPECIAL

    # ADM tell DESI_TARGET where MWS_ANY was updated.
    desi_target = (mws_target != 0) * desi_mask.MWS_ANY

    # ADM set BGS_TARGET and SCND_TARGET to zeros.
    bgs_target = np.zeros_like(mws_target)
    scnd_target = np.zeros_like(mws_target)

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
    # ADM shouldn't be necessary for M31/M33 targets but just in case.
    targets = resolve(targets)

    # ADM always prudent to check we don't have duplicate TARGETIDs
    # ADM even though this should be impossible for M31/M33 targets.
    if len(np.unique(targets["TARGETID"])) != len(targets):
        msg = ("Targets must be unique but there are some duplicated TARGETIDs!")
        log.error(msg)

    # ADM a final sort on RA to mitigate reproducibility issues.
    # ADM for instance, we've had conflicting SUBPRIORITY in the past.
    ii = np.argsort(targets)
    targets = targets[ii]

    return targets
