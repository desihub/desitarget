# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=======================
desitarget.mock.spectra
=======================

Functions dealing with assigning template spectra to mock targets.

"""
from __future__ import (absolute_import, division, print_function)

import numpy as np
from desisim.io import read_basis_templates

class TemplateKDTree(object):
    """Build a KD Tree for each object type.

    """
    def __init__(self):
        from scipy.spatial import KDTree

        self.bgs_tree = KDTree(self.bgs)

        def bgs():
            """Quantities we care about: redshift (z), M_0.1r, and 0.1(g-r)."""
            meta = read_basis_templates(objtype='bgs', onlymeta=True)
            zobj = meta['Z'].data
            mabs = meta['SDSS_UGRIZ_ABSMAG_Z01'].data
            rmabs = mabs[:, 2]
            gr = mabs[:, 1] - mabs[:, 2]
            return np.vstack((zobj, rmabs, gr)).T

        def mws():
            """Quantities we care about: Teff, logg, and [Fe/H]."""
            meta = read_basis_templates(objtype='star', onlymeta=True)
            teff = meta['TEFF'].data
            logg = meta['LOGG'].data
            feh = meta['FEH'].data
            return np.vstack((teff, logg, feh)).T

        def wd():
            """Quantities we care about: Teff and logg.

            TODO (@moustakas): deal with DA vs DB types!
            
            """
            meta = read_basis_templates(objtype='wd', onlymeta=True)
            teff = meta['TEFF'].data
            logg = meta['LOGG'].data
            return np.vstack((teff, logg)).T

        def query(self, objtype, matrix):
            """Return the nearest template number based on the KD Tree.

            Args:
              objtype (str): object type
              matrix (numpy.ndarray): (M,N) array (M=number of properties,
                N=number of objects) in the same format as the corresponding
                function for each object type (e.g., self.bgs).

            Returns:
              dist: distance to nearest template
              indx: index of nearest template
            
            """
            if objtype.lower() == 'bgs':
                dist, indx = self.bgs_tree.query(matrix)
                
            elif objtype.lower() == 'mws':
                dist, indx = self.mws_tree.query(matrix)
                
            elif objtype.lower() == 'wd':
                dist, indx = self.wd_tree.query(matrix)
                
            elif objtype.lower() == 'elg':
                dist, indx = self.elg_tree.query(matrix)
                
            elif objtype.lower() == 'lrg':
                dist, indx = self.lrg_tree.query(matrix)
                
            return dist, indx

def empty_truth_table(nobj=1):
    """Initialize the truth table for each mock object."""
    from astropy.table import Table, Column

    truth = Table()
    truth.add_column(Column(name='TARGETID', length=nobj, dtype='K'))
    truth.add_column(Column(name='MOCKID', length=nobj, dtype='K'))
    truth.add_column(Column(name='RA', length=nobj, dtype='D'))
    truth.add_column(Column(name='DEC', length=nobj, dtype='D'))

    truth.add_column(Column(name='BRICKNAME', length=nobj, dtype='8A'))
    truth.add_column(Column(name='SOURCETYPE', length=nobj, dtype=(str, 10)))

    truth.add_column(Column(name='TRUEZ', length=nobj, dtype='f4', data=np.zeros(nobj)))
    truth.add_column(Column(name='TRUETYPE', length=nobj, dtype=(str, 10)))
    truth.add_column(Column(name='TRUESUBTYPE', length=nobj, dtype=(str, 10)))

    truth.add_column(Column(name='TEMPLATEID', length=nobj, dtype='i4', data=np.zeros(nobj)-1))
    truth.add_column(Column(name='SEED', length=nobj, dtype='int64', data=np.zeros(nobj)-1))
    truth.add_column(Column(name='MAG', length=nobj, dtype='f4',data=np.zeros(nobj)-1))
    truth.add_column(Column(name='DECAM_FLUX', shape=(6,), length=nobj, dtype='f4'))
    truth.add_column(Column(name='WISE_FLUX', shape=(2,), length=nobj, dtype='f4'))

    truth.add_column(Column(name='OIIFLUX', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='erg/(s*cm2)'))
    truth.add_column(Column(name='HBETAFLUX', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='erg/(s*cm2)'))

    truth.add_column(Column(name='TEFF', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='K'))
    truth.add_column(Column(name='LOGG', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='m/(s**2)'))
    truth.add_column(Column(name='FEH', length=nobj, dtype='f4', data=np.zeros(nobj)-1))

    return truth
