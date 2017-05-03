#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desitarget
==========

Tools for DESI_ target selection.

.. _DESI: http://desi.lbl.gov
"""

# help with 2to3 support.
from __future__ import absolute_import, division, print_function

from ._version import __version__

def gitversion():
    """Returns `git describe --tags --dirty --always`,
    or 'unknown' if not a git repo"""
    import os
    from subprocess import Popen, PIPE, STDOUT
    origdir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    try:
        p = Popen(['git', "describe", "--tags", "--dirty", "--always"], stdout=PIPE, stderr=STDOUT)
    except EnvironmentError:
        return 'unknown'

    os.chdir(origdir)
    out = p.communicate()[0]
    if p.returncode == 0:
        #- avoid py3 bytes and py3 unicode; get native str in both cases
        return str(out.rstrip().decode('ascii'))
    else:
        return 'unknown'

# desitarget.targetmask makes more sense?
from .targetmask import obsconditions, obsmask
from .targetmask import desi_mask, mws_mask, bgs_mask, targetid_mask
from .contammask import contam_mask
