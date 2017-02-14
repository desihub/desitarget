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
from time import time

from desisim.io import read_basis_templates, empty_metatable

class TemplateKDTree(object):
    """Build a KD Tree for each object type.

    """
    def __init__(self):
        from scipy.spatial import KDTree

        self.bgs_meta = read_basis_templates(objtype='BGS', onlymeta=True)
        self.elg_meta = read_basis_templates(objtype='ELG', onlymeta=True)
        self.lrg_meta = read_basis_templates(objtype='LRG', onlymeta=True)
        self.mws_meta = read_basis_templates(objtype='STAR', onlymeta=True)
        self.qso_meta = read_basis_templates(objtype='QSO', onlymeta=True)
        self.wd_meta = read_basis_templates(objtype='WD', onlymeta=True)

        self.bgs_tree = KDTree(self.bgs())
        self.elg_tree = KDTree(self.elg())
        #self.lrg_tree = KDTree(self.lrg())
        self.mws_tree = KDTree(self.mws())
        #self.qso_tree = KDTree(self.qso())
        self.wd_tree = KDTree(self.wd())

    def bgs(self):
        """Quantities we care about: redshift (z), M_0.1r, and 0.1(g-r)."""
        zobj = self.bgs_meta['Z'].data
        mabs = self.bgs_meta['SDSS_UGRIZ_ABSMAG_Z01'].data
        rmabs = mabs[:, 2]
        gr = mabs[:, 1] - mabs[:, 2]
        return np.vstack((zobj, rmabs, gr)).T

    def elg(self):
        """Quantities we care about: redshift, g-r, r-z."""
        
        zobj = self.elg_meta['Z'].data
        gr = self.elg_meta['DECAM_G'].data - self.elg_meta['DECAM_R'].data
        rz = self.elg_meta['DECAM_R'].data - self.elg_meta['DECAM_Z'].data
        #W1W2 = self.elg_meta['W1'].data - self.elg_meta['W2'].data
        return np.vstack((zobj, gr, rz)).T

    #def lrg(self):
    #    """Quantities we care about: redshift, XXX"""
    #    pass 

    def mws(self):
        """Quantities we care about: Teff, logg, and [Fe/H].

        TODO (@moustakas): need to deal with standard stars and other selections. 

        """
        teff = self.mws_meta['TEFF'].data
        logg = self.mws_meta['LOGG'].data
        feh = self.mws_meta['FEH'].data
        return np.vstack((teff, logg, feh)).T

    #def qso(self):
    #    """Quantities we care about: redshift, XXX"""
    #    pass 

    def wd(self):
        """Quantities we care about: Teff and logg.

        TODO (@moustakas): deal with DA vs DB types!
        
        """
        teff = self.wd_meta['TEFF'].data
        logg = self.wd_meta['LOGG'].data
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
        if objtype.upper() == 'BGS':
            dist, indx = self.bgs_tree.query(matrix)
            
        elif objtype.upper() == 'ELG':
            dist, indx = self.elg_tree.query(matrix)
            
        elif objtype.upper() == 'LRG':
            dist, indx = self.lrg_tree.query(matrix)
            
        elif objtype.upper() == 'MWS':
            dist, indx = self.mws_tree.query(matrix)
            
        elif objtype.upper() == 'QSO':
            dist, indx = self.qso_tree.query(matrix)
            
        elif objtype.upper() == 'WD':
            dist, indx = self.wd_tree.query(matrix)
            
        return dist, indx

class MockSpectra(object):
    """Generate spectra for each type of mock.

    Currently just choose the closest template; we can get fancier later.

    """
    def __init__(self, wavemin=None, wavemax=None, dw=0.2):

        from desimodel.io import load_throughput
        
        self.tree = TemplateKDTree()

        # Build a default wavelength vector.
        if wavemin is None:
            wavemin = load_throughput('b').wavemin - 10.0
        if wavemax is None:
            wavemax = load_throughput('z').wavemax + 10.0
            
        self.wavemin = wavemin
        self.wavemax = wavemax
        self.dw = dw
        self.wave = np.arange(round(wavemin, 1), wavemax, dw)

        #self.__normfilter = 'decam2014-r' # default normalization filter

        # Initialize the templates once:
        from desisim.templates import BGS, ELG
        self.bgs = BGS(wave=self.wave, normfilter='sdss2010-r') # Need to generalize this!
        self.elg = ELG(wave=self.wave)

    def getspectra_bgs_durham_mxxl_hdf5(self, data, index=None):
        """Generate spectra for the BGS/MXXL mock sample.

        DATA needs to have Z, SDSS_absmag_r01, SDSS_01gr, VDISP, and SEED, which
        are assigned in mock.io.read_durham_mxxl_hdf5.  See also
        TemplateKDTree.bgs().

        """
        objtype = 'BGS'

        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        # Get the nearest template.
        alldata = np.vstack((data['Z'][index],
                             data['SDSS_absmag_r01'][index],
                             data['SDSS_01gr'][index])).T
        dist, templateid = self.tree.query(objtype, alldata)

        input_meta = empty_metatable(nmodel=nobj, objtype=objtype)
        input_meta['TEMPLATEID'] = templateid
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        #print('Building spectra for {}'.format(objtype))
        #bgs = BGS(wave=self.wave, normfilter=data['FILTERNAME'])
        #self.bgs.normfilter = data['FILTERNAME']

        # ToDo (@moustakas): apply Galactic extinction.
        t0 = time()
        flux, _, meta = self.bgs.make_templates(input_meta=input_meta, nocolorcuts=True, novdisp=True)
        print('Time in getspectra', time() - t0)

        return flux, meta

    def getspectra_elg_gaussianfield(self, data, index=None):
        """Generate spectra for the ELG/GaussianField mock sample.

        DATA needs to have Z, DECAM_GR, DECAM_RZ, VDISP, and SEED, which are
        assigned in mock.io.read_gaussianfield.  See also TemplateKDTree.elg().

        """
        objtype = 'ELG'

        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        # Get the nearest template.
        alldata = np.vstack((data['Z'][index],
                             data['DECAM_GR'][index],
                             data['DECAM_RZ'][index])).T
        dist, templateid = self.tree.query(objtype, alldata)

        input_meta = empty_metatable(nmodel=nobj, objtype=objtype)
        input_meta['TEMPLATEID'] = templateid
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]
            
        flux, _, meta = self.elg.make_templates(input_meta=input_meta, nocolorcuts=True, novdisp=True)

        return flux, meta

