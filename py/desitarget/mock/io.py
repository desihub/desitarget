# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.io
==================

Code to find the location of the mock data.

"""
from __future__ import absolute_import, division, print_function

import os

def get_healpix_dir(nside, pixnum, basedir='.'):
    '''
    Returns standardized path

    Args:
        nside: (int) healpix nside 2**k with 0<k<30
        pixnum: (int) healpix NESTED pixel number for this nside

    Optional:
        basedir: (str) base directory

    Note: may standardize with functions in desispec.io, but separate for now
    '''
    subdir = str(pixnum // 100)
    return os.path.abspath(os.path.join(basedir, subdir, str(pixnum)))

def findfile(filetype, nside, pixnum, basedir='.', ext='fits', obscon=None):
    '''
    Returns standardized filepath

    Args:
        filetype: (str) file prefix, e.g. 'sky' or 'targets'
        nside: (int) healpix nside 2**k with 0<k<30
        pixnum: (int) healpix NESTED pixel number for this nside

    Optional:
        basedir: (str) base directory
        ext: (str) file extension
        obscon: (str) e.g. 'dark', 'bright' to add extra dir grouping
    '''
    path = get_healpix_dir(nside, pixnum, basedir=basedir)

    if obscon is not None:
        path = os.path.join(path, obscon.lower())

        filename = '{filetype}-{obscon}-{nside}-{pixnum}.{ext}'.format(
            filetype=filetype, obscon=obscon.lower(), nside=nside, pixnum=pixnum, ext=ext)
        
    else:
        filename = '{filetype}-{nside}-{pixnum}.{ext}'.format(
            filetype=filetype, nside=nside, pixnum=pixnum, ext=ext)

    return os.path.join(path, filename)
