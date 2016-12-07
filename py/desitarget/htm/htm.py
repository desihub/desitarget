# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==============
desitarget.htm
==============

All-Python module for performin HTM look-ups
See here for HTM: http://www.skyserver.org/htm/
"""
from __future__ import (absolute_import, division)
#
from time import time
import numpy as np
import fitsio
from glob import glob
from astropy.coordinates import SkyCoord
from astropy import units as u

from . import __version__ as desitarget_version
from . import gitversion

def lookup(ra,dec,level=20,charpix=True):
    """Return the HTM pixel for a given RA/Dec

    Parameters
    ----------
    ra : :class:`float`
        A Right Ascension in degrees (can be a numpy array with multiple values)
    dec : :class:`float`
        A Declination in degrees (can be a numpy array with multiple values)
    level : :class:`int`, optional
        Which level of the HTM tree to pixelize down to
    charpix : :class:`bool`, optional, defaults to True
        If True, return pixels in character format, otherwise return them in integer format

    Returns
    -------
    :class:`char` or `int``
        The HTM pixels corresponding to the passed RA/Dec at the requisite level. Will be the same
        length as length of ra and dec
    """

    #ADM convert input spherical coordinates to Cartesian
    v = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    v.representation = 'cartesian'

    #ADM we begin to hit 64-bit floating point issues at level 25 but this is small enough for
    #ADM most applications (at level 25 a spherical triangle's longest side is ~1/100 arcsec)
    if level > 25:
        print "WARNING: Module htm.htm.py: pixels too small for 64-bit floats"
        print "LEVEL WILL BE SET TO 25"
        level = 25



