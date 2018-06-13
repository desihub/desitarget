# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
====================
desitarget.gaiamatch
====================

Useful Gaia matching routines, in case Gaia isn't absorbed into the Legacy Surveys
"""
import os, sys
import numpy as np
import fitsio
from time import time
import healpy as hp
from desitarget.io import check_fitsio_version
from astropy.coordinates import SkyCoord
from astropy import units as u

#ADM set up the DESI default logger
from desiutil.log import get_logger
log = get_logger()

#ADM start the clock
start = time()

#ADM the current data model for Gaia columns for READING from Gaia files
ingaiadatamodel = np.array([], dtype=[
            ('SOURCE_ID', '>i8'), ('RA', '>f8'), ('DEC', '>f8'),
            ('PHOT_G_MEAN_MAG', '>f4'), ('PHOT_G_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('PHOT_BP_MEAN_MAG', '>f4'), ('PHOT_BP_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('PHOT_RP_MEAN_MAG', '>f4'), ('PHOT_RP_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('ASTROMETRIC_EXCESS_NOISE', '>f4'), ('PARALLAX', '>f4'), 
            ('PMRA', '>f4'), ('PMRA_ERROR', '>f4'),
            ('PMDEC', '>f4'), ('PMDEC_ERROR', '>f4'),
                                   ])

#ADM the current data model for Gaia columns for WRITING to target files
gaiadatamodel = np.array([], dtype=[
            ('REF_ID', '>i8'), ('GAIA_RA', '>f8'), ('GAIA_DEC', '>f8'),
            ('GAIA_PHOT_G_MEAN_MAG', '>f4'), ('GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('GAIA_PHOT_BP_MEAN_MAG', '>f4'), ('GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('GAIA_PHOT_RP_MEAN_MAG', '>f4'), ('GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', '>f4'),
            ('GAIA_ASTROMETRIC_EXCESS_NOISE', '>f4'), ('PARALLAX', '>f4'), 
            ('PMRA', '>f4'), ('PMRA_IVAR', '>f4'),
            ('PMDEC', '>f4'), ('PMDEC_IVAR', '>f4'),
                                   ])

def read_gaia_file(filename, header=False):
    """Read in a Gaia "chunks" file in the appropriate format for desitarget

    Parameters
    ----------
    filename : :class:`str`
        File name of a single Gaia "chunks" file.

    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return (data, header) instead of just data.

    Returns
    -------
    :class:`list`
        Gaia data translated to targeting format (upper-case etc.) with the
        columns corresponding to `desitarget.secondary.gaiadatamodel`

    Notes
    -----
        - A better location for this might be in `desitarget.io`?
    """
    #ADM check we aren't going to have an epic fail on the the version of fitsio
    check_fitsio_version()

    #ADM prepare to read in the gaia data by reading in columns
    fx = fitsio.FITS(filename, upper=True)
    fxcolnames = fx[1].get_colnames()
    hdr = fx[1].read_header()

    #ADM the default list of columns
    readcolumns = list(ingaiadatamodel.dtype.names)
    #ADM read 'em in
    outdata = fx[1].read(columns=readcolumns)
    #ADM change the data model to what we want for each column
    outdata.dtype = gaiadatamodel.dtype

    #ADM the proper motion ERRORS need to be converted to IVARs
    #ADM remember to leave 0 entries as 0
    for col in ['PMRA_IVAR', 'PMDEC_IVAR']:
        w = np.where(outdata[col] != 0)[0]
        outdata[col][w] = 1./(outdata[col][w]**2.)

    #ADM return data read in from the gaia file, with the header if requested
    if header:
        fx.close()
        return outdata, hdr
    else:
        fx.close()
        return outdata


def find_gaia_files(objs, neighbors=True,
            gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Find full paths to all relevant gaia "chunks" files for an object array

    Parameters
    ----------
    objs : :class:`numpy.ndarray`
        Array of objects. Must contain at least the columns "RA" and "DEC".
    neighbors : :class:`bool`, optional, defaults to ``True``
        Return all of the pixels that touch the Gaia files of interest
        in order to prevent edge effects (e.g. if a Gaia source is 1 arcsec
        away from a primary source and so in an adjacent pixel)
    gaiadir : :class:`str`, optional, defaults to Gaia DR2 path at NERSC
        Root directory of a Gaia Data Release as used by the Legacy Surveys.

    Returns
    -------
    :class:`list`
        A list of all Gaia files that need to be read in to account for objects
        at the passed locations.
    """
    #ADM the resolution at which the chunks files are stored
    nside = 32

    #ADM convert RA/Dec to co-latitude and longitude in radians; note that the
    #ADM Legacy Surveys do NOT use the NESTED scheme for storing Gaia files
    theta, phi = np.radians(90-objs["DEC"]), np.radians(objs["RA"])

    #ADM if neighbors was sent, then retrieve all pixels that touch each
    #ADM pixel covered by the provided locations, to prevent edge effects...
    if neighbors:
        pixnum = np.hstack(hp.pixelfunc.get_all_neighbours(nside, theta, phi))
    #ADM ...otherwise just retrieve the pixels in which the locations lie
    else:
        pixnum = hp.ang2pix(nside, theta, phi)

    #ADM retrieve only the UNIQUE pixel numbers. It's possible that only
    #ADM one pixel was produced, so guard against pixnum being non-iterable
    if not isinstance(pixnum,np.integer):
        pixnum = list(set(pixnum))
    else:
        pixnum = [pixnum]

    #ADM there are pixels with no neighbors, which returns -1. Remove these:
    if -1 in pixnum:
        pixnum.remove(-1)

    #ADM format in the gaia chunked format used by the Legacy Surveys
    gaiafiles = ['{}/chunk-{:05d}.fits'.format(gaiadir,pn) for pn in pixnum]

    return gaiafiles


def find_gaia_files_box(gaiabounds, neighbors=True,
            gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Find full paths to all relevant gaia "chunks" files for an object array

    Parameters
    ----------
    gaiabounds : :class:`list`
        A region of the sky bounded by RA/Dec. Pass as a 4-entry list to 
        represent an area bounded by [RAmin, RAmax, DECmin, DECmax]
    neighbors : :class:`bool`, optional, defaults to ``True``
        Return all of the pixels that touch the pixels in the box in
        order to guard against edge effects
    gaiadir : :class:`str`, optional, defaults to Gaia DR2 path at NERSC
        Root directory of a Gaia Data Release as used by the Legacy Surveys.

    Returns
    -------
    :class:`list`
        A list of all Gaia files that need to be read in to account for objects
        in the passed box.

    Notes
    -----
        - Uses the `healpy` routines that rely on `fact`, so the usual
          warnings about returning different pixel sets at different values
          of `fact` apply. See:
          https://healpy.readthedocs.io/en/latest/generated/healpy.query_polygon.html
    
    """
    #ADM the resolution at which the chunks files are stored
    nside = 32

    #ADM retrive the RA/Dec bounds from the passed list
    ramin, ramax, decmin, decmax = gaiabounds

    #ADM convert RA/Dec to co-latitude and longitude in radians
    rapairs = np.array([ramin,ramin,ramax,ramax])
    decpairs = np.array([decmin,decmax,decmax,decmin])
    thetapairs, phipairs = np.radians(90.-decpairs), np.radians(rapairs)

    #ADM convert the colatitudes to Cartesian vectors remembering to
    #ADM transpose to pass the array to query_polygon in the correct order
    vecs = hp.dir2vec(thetapairs,phipairs).T

    #ADM determine the pixels that touch the box; note that the Legacy
    #ADM Surveys do NOT use the NESTED scheme for storing Gaia files
    pixnum = hp.query_polygon(nside,vecs,inclusive=True,fact=4)

    #ADM if neighbors was sent, then retrieve all pixels that touch each
    #ADM pixel covered by the provided locations, to prevent edge effects...
    if neighbors:
        #ADM first convert back to theta/phi to retrieve neighbors
        theta, phi = hp.pix2ang(nside,pixnum)
        pixnum = np.hstack(hp.pixelfunc.get_all_neighbours(nside, theta, phi))

    #ADM retrieve only the UNIQUE pixel numbers. It's possible that only
    #ADM one pixel was produced, so guard against pixnum being non-iterable
    if not isinstance(pixnum,np.integer):
        pixnum = list(set(pixnum))
    else:
        pixnum = [pixnum]

    #ADM there are pixels with no neighbors, which returns -1. Remove these:
    if -1 in pixnum:
        pixnum.remove(-1)

    #ADM format in the gaia chunked format used by the Legacy Surveys
    gaiafiles = ['{}/chunk-{:05d}.fits'.format(gaiadir,pn) for pn in pixnum]

    return gaiafiles


def match_gaia_to_primary(objs, matchrad=1., 
                    retaingaia=False, gaiabounds=[0.,360.,-90.,90.], 
            gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Match a set of objects to Gaia "chunks" files and return the Gaia information

    Parameters
    ----------
    objs : :class:`numpy.ndarray`
        Must contain at least "RA" and "DEC".
    matchrad : :class:`float`, optional, defaults to 1 arcsec
        The matching radius in arcseconds.
    retaingaia : :class:`float`, optional, defaults to False
        If set, return all of the Gaia information in the "area" occupied by
        the passed objects (whether a Gaia object matches a passed RA/Dec
        or not.) THIS ASSUMES THAT THE PASSED OBJECTS ARE FROM A SWEEPS file 
        and that the integer values nearest the maximum and minimum passed RAs 
        and Decs fairly represent the areal "edges" of that file.
    gaiabounds : :class:`list`, optional, defaults to the whole sky
        Used in conjunction with `retaingaia` to determine over what area to
        retrieve Gaia objects that don't match a sweeps object. Pass a 4-entry
        list to represent an area bounded by [RAmin, RAmax, DECmin, DECmax]
    gaiadir : :class:`str`, optional, defaults to Gaia DR2 path at NERSC
        Root directory of a Gaia Data Release as used by the Legacy Surveys.

    Returns
    -------
    :class:`numpy.ndarray`
        The matching Gaia information for each object, where the returned format and
        columns correspond to `desitarget.secondary.gaiadatamodel`

    Notes
    -----
        - The first len(objs) objects correspond row-by-row to the passed objects.
        - For objects that do NOT have a match in the Gaia files, the "REF_ID"
          column is set to -1, and all other columns are zero.
        - If `retaingaia` is True then objects after the first len(objs) objects are 
          Gaia objects that do not have a sweeps match but that are in the area
          bounded by `gaiabounds`
    """
    #ADM I'm getting this old Cython RuntimeWarning on search_around_sky ****:
    # RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
    #ADM but it doesn't seem malicious, so I'm filtering. I think its caused
    #ADM by importing a scipy compiled against an older numpy than is installed
    #ADM e.g. https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
    import warnings

    #ADM if retaingaia is True, retain all Gaia objects in a sweeps-like box
    if retaingaia:
         ramin, ramax, decmin, decmax = gaiabounds

    #ADM convert the coordinates of the input objects to a SkyCoord object
    cobjs = SkyCoord(objs["RA"]*u.degree, objs["DEC"]*u.degree)
    nobjs = cobjs.size

    #ADM deal with the special case that only a single object was passed
    if nobjs == 1:
        return match_gaia_to_primary_single(objs,matchrad=matchrad,gaiadir=gaiadir)

    #ADM set up a zerod array of Gaia information for the passed objects
    gaiainfo = np.zeros(nobjs, dtype=gaiadatamodel.dtype)

    #ADM a supplemental (zero-length) array to hold Gaia objects that don't 
    #ADM match a sweeps object, in case retaingaia was set
    suppgaiainfo = np.zeros(0, dtype=gaiadatamodel.dtype)

    #ADM objects without matches should have REF_ID of -1
    gaiainfo['REF_ID'] = -1

    #ADM determine which Gaia files need to be considered
    if retaingaia:
        gaiafiles = find_gaia_files_box(gaiabounds, gaiadir=gaiadir)
    else:
        gaiafiles = find_gaia_files(objs, gaiadir=gaiadir)

    #ADM loop through the Gaia files and match to the passed objects
    for file in gaiafiles:
        gaia = read_gaia_file(file)
        cgaia = SkyCoord(gaia["GAIA_RA"]*u.degree, gaia["GAIA_DEC"]*u.degree)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #ADM ****here's where the warning occurs...
            idobjs, idgaia, _, _ = cgaia.search_around_sky(cobjs,matchrad*u.arcsec)
        #ADM assign the Gaia info to the array that corresponds to the passed objects
        gaiainfo[idobjs] = gaia[idgaia]

        #ADM if retaingaia was set, also build an array of Gaia objects that
        #ADM don't have sweeps matches, but are within the RA/Dec bounds
        if retaingaia:
            #ADM find the Gaia IDs that didn't match the passed objects
            nomatch = set(np.arange(len(gaia)))-set(idgaia) 
            noidgaia = np.array(list(nomatch))
            #ADM which Gaia objects with these IDs are within the bounds
            if len(noidgaia) > 0:
                suppg = gaia[noidgaia]
                winbounds = np.where((suppg["GAIA_RA"] >= ramin) & (suppg["GAIA_RA"] < ramax) 
                        & (suppg["GAIA_DEC"] >= decmin) & (suppg["GAIA_DEC"] < decmax) )[0]
                #ADM Append those Gaia objects to the suppgaiainfo array
                if len(winbounds) > 0:
                    suppgaiainfo = np.hstack([suppgaiainfo,suppg[winbounds]])

    if retaingaia:
        gaiainfo = np.hstack([gaiainfo,suppgaiainfo])

    return gaiainfo


def match_gaia_to_primary_single(objs, matchrad=1.,
            gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Match ONE object to Gaia "chunks" files and return the Gaia information

    Parameters
    ----------
    objs : :class:`numpy.ndarray`
        Must contain at least "RA" and "DEC". MUST BE A SINGLE ROW.
    matchrad : :class:`float`, optional, defaults to 1 arcsec
        The matching radius in arcseconds.
    gaiadir : :class:`str`, optional, defaults to Gaia DR2 path at NERSC
        Root directory of a Gaia Data Release as used by the Legacy Surveys.

    Returns
    -------
    :class:`numpy.ndarray`
        The matching Gaia information for the object, where the returned format and
        columns correspond to `desitarget.secondary.gaiadatamodel`

    Notes
    -----
        - If the object does NOT have a match in the Gaia files, the "REF_ID"
          column is set to -1, and all other columns are zero
    """
    #ADM I'm getting this old Cython RuntimeWarning on search_around_sky ****:
    # RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
    #ADM but it doesn't seem malicious, so I'm filtering. I think its caused
    #ADM by importing a scipy compiled against an older numpy than is installed
    #ADM e.g. https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
    import warnings
    
    #ADM convert the coordinates of the input objects to a SkyCoord object
    cobjs = SkyCoord(objs["RA"]*u.degree, objs["DEC"]*u.degree)
    nobjs = cobjs.size
    if nobjs > 1:
        log.error("Only matches one row but {} rows were sent".format(nobjs))

    #ADM set up a zerod array of Gaia information for the passed object
    gaiainfo = np.zeros(nobjs, dtype=gaiadatamodel.dtype)

    #ADM an object without matches should have REF_ID of -1
    gaiainfo['REF_ID'] = -1

    #ADM determine which Gaia files need to be considered
    gaiafiles = find_gaia_files(objs, gaiadir=gaiadir)

    #ADM loop through the Gaia files and match to the passed object
    for file in gaiafiles:
        gaia = read_gaia_file(file)
        cgaia = SkyCoord(gaia["GAIA_RA"]*u.degree, gaia["GAIA_DEC"]*u.degree)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #ADM ****here's where the warning occurs...
            sep = cobjs.separation(cgaia)
            idgaia = np.where(sep < matchrad*u.arcsec)[0]
        #ADM assign the Gaia info to the array that corresponds to the passed object
        if len(idgaia) > 0:
            gaiainfo = gaia[idgaia]
        
    return gaiainfo
