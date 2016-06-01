# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
========
mocks.io
========

Makes mock target catalogs
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import fitsio
import os, re
from . import __version__ as desitarget_version
from . import gitversion

def estimate_density(ra, dec):
    """Estimate the number density from a small patch
    
    Args:
        ra: array_like
            An array with RA positions.
        dec: array_like
            An array with Dec positions.

    Returns:
        density: float
           Object number density computed over a small patch.
    """
    density = 0.0 

    footprint_area = 20. * 45.* np.sin(45. * np.pi/180.)/(45. * np.pi/180.)
    smalldata = ra[(ra>170.) & (dec<190.) & (dec>0.) & (dec<45.)]
    n_in = len(smalldata)
    density = n_in/footprint_area

    return density
