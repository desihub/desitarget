# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
====================
desitarget.secondary
====================

Module with useful matching, bit assigning, etc. functions for secondary targets
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

#ADM the current data model for Gaia files
gaiadatamodel = np.array([], dtype=[
                            ('SOURCE_ID', '>i8'), ('RA', '>f8'), ('DEC', '>f8'),
                            ('PHOT_G_MEAN_MAG', '>f4'), ('PHOT_RP_MEAN_MAG', '>f4'),
                            ('PARALLAX', '>f4'), ('PMRA', '>f4'), ('PMDEC', '>f4')
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
    readcolumns = list(gaiadatamodel.dtype.names)
    #ADM read 'em in
    outdata = fx[1].read(columns=readcolumns)

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
        pixnum = set(pixnum)
    else:
        pixnum = [pixnum]

    #ADM format in the gaia chunked format used by the Legacy Surveys
    gaiafiles = ['{}/chunk-{:05d}.fits'.format(gaiadir,pn) for pn in pixnum]

    return gaiafiles


def match_gaia_to_primary(objs, matchrad=1.,
            gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Match a set of objects to Gaia "chunks" files and return the Gaia information

    Parameters
    ----------
    objs : :class:`numpy.ndarray`
        Must contain at least "RA" and "DEC".
    matchrad : :class:`float`, optional, defaults to 1 arcsec
        The matching radius in arcseconds.
    gaiadir : :class:`str`, optional, defaults to Gaia DR2 path at NERSC
        Root directory of a Gaia Data Release as used by the Legacy Surveys.

    Returns
    -------
    :class:`numpy.ndarray`
        The matching Gaia information for each object, where the returned format and
        columns correspond to `desitarget.secondary.gaiadatamodel

    Notes
    -----
        - For objects that do NOT have a match in the Gaia files, the "SOURCE_ID"
          column is set to -1, and all other columns are zero
    """
    #ADM set up a zerod array of Gaia information for the passed objects
    nobjs = len(objs)
    gaiainfo = np.zeros(nobjs, dtype=gaiadatamodel.dtype)

    #ADM objects without matches should have SOURCE_ID of -1
    gaiainfo['SOURCE_ID'] = -1

    #ADM convert the coordinates of the input objects to a SkyCoord object
    cobjs = SkyCoord(objs["RA"]*u.degree, objs["DEC"]*u.degree)

    #ADM determine which Gaia files need to be considered
    gaiafiles = find_gaia_files(objs, gaiadir=gaiadir)

    #ADM loop through the Gaia files and match to the passed objects
    for file in gaiafiles:
        gaia = read_gaia_files(file)
        cgaia = SkyCoord(gaia["RA"]*u.degree, gaia["DEC"]*u.degree)
        idobjs, idgaia, _, _ = cgaia.search_around_sky(cobjs,matchrad*u.arcsec)

        #ADM assign the Gaia info to the array that corresponds to the passed objects
        gaiainfo[idobjs] = gaia[idgaia]

    return gaiainfo


