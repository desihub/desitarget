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
import multiprocessing

from desisim.io import read_basis_templates, empty_metatable

def _get_colors_onez(args):
    """Filler function to synthesize photometry at a given redshift"""
    return get_colors_onez(*args)

def get_colors_onez(z, flux, wave, filt):
    print(z)
    zwave = wave.astype('float') * (1 + z)
    phot = filt.get_ab_maggies(flux, zwave, mask_invalid=False)
    gr = -2.5 * np.log10( phot['decam2014-g'] / phot['decam2014-r'] )
    rz = -2.5 * np.log10( phot['decam2014-r'] / phot['decam2014-z'] )
    return [gr, rz]

class TemplateKDTree(object):
    """Build a KD Tree for each object type.

    """
    def __init__(self, nproc=1):
        from speclite import filters
        from scipy.spatial import KDTree

        self.nproc = nproc

        self.bgs_meta = read_basis_templates(objtype='BGS', onlymeta=True)
        self.elg_meta = read_basis_templates(objtype='ELG', onlymeta=True)
        self.lrg_meta = read_basis_templates(objtype='LRG', onlymeta=True)
        self.qso_meta = read_basis_templates(objtype='QSO', onlymeta=True)
        self.wd_da_meta = read_basis_templates(objtype='WD', subtype='DA', onlymeta=True)
        self.wd_db_meta = read_basis_templates(objtype='WD', subtype='DB', onlymeta=True)

        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')

        # Read all the stellar spectra and synthesize DECaLS/WISE fluxes.
        self.star_phot()

        #self.elg_phot()
        #self.elg_kcorr = read_basis_templates(objtype='ELG', onlykcorr=True)

        self.bgs_tree = KDTree(self._bgs())
        self.elg_tree = KDTree(self._elg())
        #self.lrg_tree = KDTree(self._lrg())
        #self.qso_tree = KDTree(self._qso())
        self.star_tree = KDTree(self._star())
        self.wd_da_tree = KDTree(self._wd_da())
        self.wd_db_tree = KDTree(self._wd_db())

    def star_phot(self):
        """Synthesize photometry for the full set of stellar templates."""
        star_normfilter = 'decam2014-r'

        star_flux, star_wave, star_meta = read_basis_templates(objtype='STAR')
        star_maggies_table = self.decamwise.get_ab_maggies(star_flux, star_wave, mask_invalid=True)

        star_maggies = np.zeros( (len(star_meta), len(self.decamwise)) )
        for ff, key in enumerate(star_maggies_table.columns):
            star_maggies[:, ff] = star_maggies_table[key] / star_maggies_table[star_normfilter] # maggies
        self.star_flux_g = star_maggies[:, 0]
        self.star_flux_r = star_maggies[:, 1]
        self.star_flux_z = star_maggies[:, 2]
        self.star_flux_w1 = star_maggies[:, 3]
        self.star_flux_w2 = star_maggies[:, 4]
        
        self.star_meta = star_meta

    def elg_kcorr(self):
        """Compute K-corrections for the ELG templates on a grid of redshift."""
        
        flux, wave, meta = read_basis_templates(objtype='ELG')
        nt = len(meta)

        zmin, zmax, dz = 0.0, 2.0, 0.1
        nz = np.round( (zmax - zmin) / dz ).astype('i2')

        colors = dict(
            redshift = np.linspace(0.0, 2.0, nz),
            gr = np.zeros( (nt, nz) ),
            rz = np.zeros( (nt, nz) )
            )

        #from time import time
        #t0 = time()
        #for iz, red in enumerate(colors['redshift']):
        #    print(iz)
        #    zwave = wave.astype('float') * (1 + red)
        #    phot = self.grz.get_ab_maggies(flux, zwave, mask_invalid=False)
        #    colors['gr'][:, iz] = -2.5 * np.log10( phot['decam2014-g'] / phot['decam2014-r'] )
        #    colors['rz'][:, iz] = -2.5 * np.log10( phot['decam2014-r'] / phot['decam2014-z'] )
        #print( (time() - t0) / 60 )

        from time import time
        t0 = time()
        zargs = list()
        for red in colors['redshift']:
            zargs.append( (red, flux, wave, self.grz) )

        if self.nproc > 1:
            pool = multiprocessing.Pool(self.nproc)
            result = pool.map( _get_colors_onez, zargs )
            pool.close()
        else:
            result = list()
            for onearg in zargs:
                result.append( _get_colors_onez(onearg) )
        print( (time() - t0) / 60 )

        import matplotlib.pyplot as plt
        for tt in range(10):
            plt.scatter(colors['rz'][tt, :], colors['gr'][tt, :])
        plt.xlim(-0.5, 2.0) ;  plt.ylim(-0.5, 2.0)
        plt.show()
        import pdb ; pdb.set_trace()
        
    def _bgs(self):
        """Quantities we care about: redshift (z), M_0.1r, and 0.1(g-r).  This needs to
        be generalized to accommodate other mocks!

        """
        zobj = self.bgs_meta['Z'].data
        mabs = self.bgs_meta['SDSS_UGRIZ_ABSMAG_Z01'].data
        rmabs = mabs[:, 2]
        gr = mabs[:, 1] - mabs[:, 2]
        return np.vstack((zobj, rmabs, gr)).T

    def _elg(self):
        """Quantities we care about: redshift, g-r, r-z."""
        
        zobj = self.elg_meta['Z'].data
        gr = self.elg_meta['DECAM_G'].data - self.elg_meta['DECAM_R'].data
        rz = self.elg_meta['DECAM_R'].data - self.elg_meta['DECAM_Z'].data
        #W1W2 = self.elg_meta['W1'].data - self.elg_meta['W2'].data
        return np.vstack((zobj, gr, rz)).T

    #def _elg(self):
    #    """Quantities we care about: redshift, g-r, r-z."""
    #    
    #    zobj = self.elg_meta['Z'].data
    #    gr = self.elg_meta['DECAM_G'].data - self.elg_meta['DECAM_R'].data
    #    rz = self.elg_meta['DECAM_R'].data - self.elg_meta['DECAM_Z'].data
    #    #W1W2 = self.elg_meta['W1'].data - self.elg_meta['W2'].data
    #    return np.vstack((zobj, gr, rz)).T

    def _lrg(self):
        """Quantities we care about: r-z, r-W1."""
        
        zobj = self.elg_meta['Z'].data
        gr = self.elg_meta['DECAM_G'].data - self.elg_meta['DECAM_R'].data
        rz = self.elg_meta['DECAM_R'].data - self.elg_meta['DECAM_Z'].data
        #W1W2 = self.elg_meta['W1'].data - self.elg_meta['W2'].data
        return np.vstack((zobj, gr, rz)).T

    def _star(self):
        """Quantities we care about: Teff, logg, and [Fe/H].

        """
        teff = self.star_meta['TEFF'].data
        logg = self.star_meta['LOGG'].data
        feh = self.star_meta['FEH'].data
        return np.vstack((teff, logg, feh)).T

    #def qso(self):
    #    """Quantities we care about: redshift, XXX"""
    #    pass 

    def _wd_da(self):
        """DA white dwarf.  Quantities we care about: Teff and logg.

        """
        teff = self.wd_da_meta['TEFF'].data
        logg = self.wd_da_meta['LOGG'].data
        return np.vstack((teff, logg)).T

    def _wd_db(self):
        """DB white dwarf.  Quantities we care about: Teff and logg.

        """
        teff = self.wd_db_meta['TEFF'].data
        logg = self.wd_db_meta['LOGG'].data
        return np.vstack((teff, logg)).T

    def query(self, objtype, matrix, subtype=''):
        """Return the nearest template number based on the KD Tree.

        Args:
          objtype (str): object type
          matrix (numpy.ndarray): (M,N) array (M=number of properties,
            N=number of objects) in the same format as the corresponding
            function for each object type (e.g., self.bgs).
          subtype (str, optional): subtype (only for white dwarfs)

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
            
        elif objtype.upper() == 'STAR':
            dist, indx = self.star_tree.query(matrix)
            
        elif objtype.upper() == 'QSO':
            dist, indx = self.qso_tree.query(matrix)
            
        elif objtype.upper() == 'WD':
            if subtype.upper() == 'DA':
                dist, indx = self.wd_da_tree.query(matrix)
            elif subtype.upper() == 'DB':
                dist, indx = self.wd_db_tree.query(matrix)
            else:
                log.warning('Unrecognized SUBTYPE {}!'.format(subtype))
                raise ValueError
                
        return dist, indx

class MockSpectra(object):
    """Generate spectra for each type of mock.  Currently just choose the closest
    template; we can get fancier later.

    ToDo (@moustakas): apply Galactic extinction.

    """
    def __init__(self, wavemin=None, wavemax=None, dw=0.2, nproc=1,
                 rand=None, verbose=False):

        from desimodel.io import load_throughput
        
        self.tree = TemplateKDTree(nproc=nproc)

        # Build a default (buffered) wavelength vector.
        if wavemin is None:
            wavemin = load_throughput('b').wavemin - 10.0
        if wavemax is None:
            wavemax = load_throughput('z').wavemax + 10.0
            
        self.wavemin = wavemin
        self.wavemax = wavemax
        self.dw = dw
        self.wave = np.arange(round(wavemin, 1), wavemax, dw)

        self.rand = rand
        self.verbose = verbose

        #self.__normfilter = 'decam2014-r' # default normalization filter

        # Initialize the templates once:
        from desisim.templates import BGS, ELG, LRG, QSO, STAR, WD
        self.bgs_templates = BGS(wave=self.wave, normfilter='sdss2010-r') # Need to generalize this!
        self.elg_templates = ELG(wave=self.wave, normfilter='decam2014-r')
        self.lrg_templates = LRG(wave=self.wave, normfilter='decam2014-z')
        self.qso_templates = QSO(wave=self.wave, normfilter='decam2014-g')
        self.lya_templates = QSO(wave=self.wave, normfilter='decam2014-g')
        self.star_templates = STAR(wave=self.wave, normfilter='decam2014-r')
        self.wd_da_templates = WD(wave=self.wave, normfilter='decam2014-g', subtype='DA')
        self.wd_db_templates = WD(wave=self.wave, normfilter='decam2014-g', subtype='DB')
        
    def bgs(self, data, index=None, mockformat='durham_mxxl_hdf5'):
        """Generate spectra for BGS.

        Currently only the MXXL (durham_mxxl_hdf5) mock is supported.  DATA
        needs to have Z, SDSS_absmag_r01, SDSS_01gr, VDISP, and SEED, which are
        assigned in mock.io.read_durham_mxxl_hdf5.  See also
        TemplateKDTree.bgs().

        """
        objtype = 'BGS'
        if index is None:
            index = np.arange(len(data['Z']))
            
        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'durham_mxxl_hdf5':
            alldata = np.vstack((data['Z'][index],
                                 data['SDSS_absmag_r01'][index],
                                 data['SDSS_01gr'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.bgs_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, novdisp=False,
                                                          verbose=self.verbose)

        return flux, meta

    def elg(self, data, index=None, mockformat='gaussianfield'):
        """Generate spectra for the ELG sample.

        Currently only the GaussianField mock sample is supported.  DATA needs
        to have Z, GR, RZ, VDISP, and SEED, which are assigned in
        mock.io.read_gaussianfield.  See also TemplateKDTree.elg().

        """
        objtype = 'ELG'
        if index is None:
            index = np.arange(len(data['Z']))

        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'gaussianfield':
            alldata = np.vstack((data['Z'][index],
                                 data['GR'][index],
                                 data['RZ'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        if False:
            import matplotlib.pyplot as plt
            def elg_colorbox(ax):
                """Draw the ELG selection box."""
                from matplotlib.patches import Polygon
                grlim = ax.get_ylim()
                coeff0, coeff1 = (1.15, -0.15), (-1.2, 1.6)
                rzmin, rzpivot = 0.3, (coeff1[1] - coeff0[1]) / (coeff0[0] - coeff1[0])
                verts = [(rzmin, grlim[0]),
                         (rzmin, np.polyval(coeff0, rzmin)),
                         (rzpivot, np.polyval(coeff1, rzpivot)),
                         ((grlim[0] - 0.1 - coeff1[1]) / coeff1[0], grlim[0] - 0.1)
                         ]
                ax.add_patch(Polygon(verts, fill=False, ls='--', color='k'))

            fig, ax = plt.subplots()
            ax.scatter(data['RZ'][index], data['GR'][index])
            ax.set_xlim(-0.5, 2) ; plt.ylim(-0.5, 2)
            elg_colorbox(ax)
            plt.show()
            import pdb ; pdb.set_trace()

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.elg_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, novdisp=False,
                                                          verbose=self.verbose)

        return flux, meta

    def elg_test(self, data, index=None, mockformat='gaussianfield'):
        """Test script -- generate spectra for the ELG sample.

        Currently only the GaussianField mock sample is supported.  DATA needs
        to have Z, GR, RZ, VDISP, and SEED, which are assigned in
        mock.io.read_gaussianfield.  See also TemplateKDTree.elg().

        """
        objtype = 'ELG'
        if index is None:
            index = np.arange(len(data['Z']))

        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'gaussianfield':
            alldata = np.vstack((data['Z'][index],
                                 data['GR'][index],
                                 data['RZ'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d

        f1 = interp1d(np.squeeze(self.tree.elg_kcorr['REDSHIFT']), np.squeeze(self.tree.elg_kcorr['GR']), axis=0)
        gr = f1(data['Z'][index])
        plt.plot(np.squeeze(self.tree.elg_kcorr['REDSHIFT']), np.squeeze(self.tree.elg_kcorr['GR'])[:, 500])
        plt.scatter(data['Z'][index], gr[:, 500], marker='x', color='red', s=15)
        plt.show()

        def elg_colorbox(ax):
            """Draw the ELG selection box."""
            from matplotlib.patches import Polygon
            grlim = ax.get_ylim()
            coeff0, coeff1 = (1.15, -0.15), (-1.2, 1.6)
            rzmin, rzpivot = 0.3, (coeff1[1] - coeff0[1]) / (coeff0[0] - coeff1[0])
            verts = [(rzmin, grlim[0]),
                     (rzmin, np.polyval(coeff0, rzmin)),
                     (rzpivot, np.polyval(coeff1, rzpivot)),
                     ((grlim[0] - 0.1 - coeff1[1]) / coeff1[0], grlim[0] - 0.1)
                     ]
            ax.add_patch(Polygon(verts, fill=False, ls='--', color='k'))

        fig, ax = plt.subplots()
        ax.scatter(data['RZ'][index], data['GR'][index])
        ax.set_xlim(-0.5, 2) ; plt.ylim(-0.5, 2)
        elg_colorbox(ax)
        plt.show()
        import pdb ; pdb.set_trace()

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.elg_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, novdisp=False,
                                                          verbose=self.verbose)

        return flux, meta

    def lrg(self, data, index=None, mockformat='gaussianfield'):
        """Generate spectra for the LRG sample.

        Currently only the GaussianField mock sample is supported.  DATA needs
        to have Z, GR, RZ, VDISP, and SEED, which are assigned in
        mock.io.read_gaussianfield.  See also TemplateKDTree.lrg().

        """
        objtype = 'LRG'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'gaussianfield':
            # This is wrong: choose a template with equal probability.
            templateid = self.rand.choice(self.tree.lrg_meta['TEMPLATEID'], len(index))
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.lrg_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, novdisp=False,
                                                          verbose=self.verbose)

        return flux, meta

    def mws(self, data, index=None, mockformat='galaxia'):
        """Generate spectra for the MWS_NEARBY and MWS_MAIN samples.

        """
        objtype = 'STAR'
        if index is None:
            index = np.arange(len(data['Z']))

        input_meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'FEH'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'FEH')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == '100pc':
            alldata = np.vstack((data['TEFF'][index],
                                 data['LOGG'][index],
                                 data['FEH'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
            
        elif mockformat.lower() == 'galaxia':
            alldata = np.vstack((data['TEFF'][index],
                                 data['LOGG'][index],
                                 data['FEH'][index])).T
            _, templateid = self.tree.query(objtype, alldata)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        input_meta['TEMPLATEID'] = templateid
        flux, _, meta = self.star_templates.make_templates(input_meta=input_meta,
                                                          verbose=self.verbose) # Note! No colorcuts.

        return flux, meta

    def mws_nearby(self, data, index=None, mockformat='100pc'):
        """Generate spectra for the MWS_NEARBY sample.

        """
        flux, meta = self.mws(data, index=index, mockformat=mockformat)
        return flux, meta

    def mws_main(self, data, index=None, mockformat='galaxia'):
        """Generate spectra for the MWS_MAIN sample.

        """
        flux, meta = self.mws(data, index=index, mockformat=mockformat)
        return flux, meta

    def faintstar(self, data, index=None, mockformat='galaxia'):
        """Generate spectra for the FAINTSTAR (faint stellar) sample.

        """
        flux, meta = self.mws(data, index=index, mockformat=mockformat)
        return flux, meta

    def mws_wd(self, data, index=None, mockformat='wd'):
        """Generate spectra for the MWS_WD sample.  Deal with DA vs DB white dwarfs
        separately.

        """
        objtype = 'WD'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        input_meta = empty_metatable(nmodel=nobj, objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'SUBTYPE'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'TEMPLATESUBTYPE')):
            input_meta[inkey] = data[datakey][index]

        if mockformat.lower() == 'wd':
            meta = empty_metatable(nmodel=nobj, objtype=objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')
            
            for subtype in ('DA', 'DB'):
                these = np.where(input_meta['SUBTYPE'] == subtype)[0]
                if len(these) > 0:
                    alldata = np.vstack((data['TEFF'][index][these],
                                         data['LOGG'][index][these])).T
                    _, templateid = self.tree.query(objtype, alldata, subtype=subtype)

                    input_meta['TEMPLATEID'][these] = templateid
                    
                    template_function = 'wd_{}_templates'.format(subtype.lower())
                    flux1, _, meta1 = getattr(self, template_function).make_templates(input_meta=input_meta[these],
                                                          verbose=self.verbose)
                    
                    meta[these] = meta1
                    flux[these, :] = flux1
            
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        return flux, meta

    def qso(self, data, index=None, mockformat='gaussianfield'):
        """Generate spectra for the QSO or QSO/LYA samples.

        Note: We need to make sure NORMFILTER matches!

        """
        from desisim.lya_spectra import get_spectra
        
        objtype = 'QSO'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        if mockformat.lower() == 'gaussianfield':
            input_meta = empty_metatable(nmodel=nobj, objtype=objtype)
            for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT'),
                                      ('SEED', 'MAG', 'Z')):
                input_meta[inkey] = data[datakey][index]

            # Build the tracer and Lya forest QSO spectra separately.
            meta = empty_metatable(nmodel=nobj, objtype=objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')

            lya = np.where( data['TEMPLATESUBTYPE'][index] == 'LYA' )[0]
            tracer = np.where( data['TEMPLATESUBTYPE'][index] == '' )[0]

            if len(tracer) > 0:
                flux1, _, meta1 = self.qso_templates.make_templates(input_meta=input_meta[tracer],
                                                                    lyaforest=False,
                                                                    nocolorcuts=True,
                                                                    verbose=self.verbose)
                meta[tracer] = meta1
                flux[tracer, :] = flux1

            if len(lya) > 0:
                alllyafile = data['LYAFILES'][index][lya]
                alllyahdu = data['LYAHDU'][index][lya]
                
                for lyafile in sorted(set(alllyafile)):
                    these = np.where( lyafile == alllyafile )[0]

                    templateid = alllyahdu[these] - 1 # templateid is 0-indexed
                    flux1, _, meta1 = get_spectra(lyafile, templateid=templateid, normfilter=data['FILTERNAME'],
                                                  rand=self.rand, qso=self.lya_templates, nocolorcuts=True)
                    meta1['SUBTYPE'] = 'LYA'
                    meta[lya[these]] = meta1
                    flux[lya[these], :] = flux1

        elif mockformat.lower() == 'lya':
            # Build spectra for Lyman-alpha QSOs. Deprecated!
            from desisim.lya_spectra import get_spectra
            from desitarget.mock.io import decode_rownum_filenum

            meta = empty_metatable(nmodel=nobj, objtype=objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')
            
            rowindx, fileindx = decode_rownum_filenum(data['MOCKID'][index])
            for indx1 in set(fileindx):
                lyafile = data['FILES'][indx1]
                these = np.where(indx1 == fileindx)[0]
                templateid = rowindx[these].astype('int')
            
                flux1, _, meta1 = get_spectra(lyafile, templateid=templateid,
                                              normfilter=data['FILTERNAME'],
                                              rand=self.rand, qso=self.lya_templates)
                meta[these] = meta1
                flux[these, :] = flux1
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        return flux, meta

    def sky(self, data, index=None, mockformat=None):
        """Generate spectra for SKY.

        """
        objtype = 'SKY'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)
            
        meta = empty_metatable(nmodel=nobj, objtype=objtype)
        for inkey, datakey in zip(('SEED', 'REDSHIFT'),
                                  ('SEED', 'Z')):
            meta[inkey] = data[datakey][index]
        flux = np.zeros((nobj, len(self.wave)), dtype='i1')

        return flux, meta

class MockMagnitudes(object):
    """
    Generate mock magnitudes for each mock.
    """
    def __init__(self, nproc=1, rand=None, verbose=False):
        self.rand = rand
        self.verbose = verbose
    def bgs(self, data, index=None, mockformat='durham_mxxl_hdf5'):
        """Generate magnitudes for BGS.
        Currently only the MXXL (durham_mxxl_hdf5) mock is supported.  DATA
        needs to have Z, SDSS_absmag_r01, SDSS_01gr, VDISP, and SEED, which are
        assigned in mock.io.read_durham_mxxl_hdf5.  
        """
        objtype = 'BGS'
        if index is None:
            index = np.arange(len(data['Z']))
        meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            meta[inkey] = data[datakey][index]

        if mockformat.lower() != 'durham_mxxl_hdf5':
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta
        meta['FLUX_R'][:] = 10**((22.5 - data['MAG'][index])/2.5) # r-band flux
        meta['FLUX_G'][:] = 10**((22.5-(data['SDSS_01gr'][index] + data['MAG'][index]))/2.5) # g-band flux
        return meta
    
    def mws(self, data, index=None, mockformat='galaxia'):
        """Generate magnitudes for the MWS_NEARBY and MWS_MAIN samples.
        """
        objtype = 'STAR'
        if index is None:
            index = np.arange(len(data['Z']))

        meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'FEH'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'FEH')):
            meta[inkey] = data[datakey][index]

        if not (mockformat.lower() in ['100pc','galaxia']):
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta

        meta['FLUX_R'][:] = 10**((22.5 - data['MAG'][index])/2.5) # r-band flux
        return meta
                
    def mws_nearby(self, data, index=None, mockformat='100pc'):
        """Generate magnitudes for the MWS_NEARBY sample.
        """
        meta = self.mws(data, index=index, mockformat=mockformat)
        return meta

    def mws_main(self, data, index=None, mockformat='galaxia'):
        """Generate magnitudes for the MWS_MAIN sample.
        """
        meta = self.mws(data, index=index, mockformat=mockformat)
        return meta

    def faintstar(self, data, index=None, mockformat='galaxia'):
        """Generate magnitudes for the FAINTSTAR (faint stellar) sample.
        """
        meta = self.mws(data, index=index, mockformat=mockformat)
        return meta
        
    def elg(self, data, index=None, mockformat='gaussianfield'):
        """Generate magnitudes for the ELG sample.
        Currently only the GaussianField mock sample is supported.  DATA needs
        to have Z, GR, RZ, VDISP, and SEED, which are assigned in
        mock.io.read_gaussianfield.  
        """
        objtype = 'ELG'
        if index is None:
            index = np.arange(len(data['Z']))

        meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            meta[inkey] = data[datakey][index]

        if mockformat.lower() != 'gaussianfield':
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta
        
        meta['FLUX_G'][:] = 10**((22.5 - (data['GR'][index] + data['MAG'][index]))/2.5) # g-band flux
        meta['FLUX_R'][:] = 10**((22.5 - data['MAG'][index])/2.5) # r-band flux
        meta['FLUX_Z'][:] = 10**((22.5 - (data['MAG'][index] - data['RZ'][index]))/2.5) # z-band flux
        return meta
    
    def lrg(self, data, index=None, mockformat='gaussianfield'):
        """Generate magnitudes for the LRG sample.
        Currently only the GaussianField mock sample is supported.  DATA needs
        to have Z, GR, RZ, VDISP, and SEED, which are assigned in
        mock.io.read_gaussianfield.  
        """
        objtype = 'LRG'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        meta = empty_metatable(nmodel=len(index), objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            meta[inkey] = data[datakey][index]

        if mockformat.lower() != 'gaussianfield':
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta
        
        meta['FLUX_G'][:] = 10**((22.5 - (data['GR'][index] + data['RZ'][index] + data['MAG'][index]))/2.5) # g-band flux  
        meta['FLUX_R'][:] =  10**((22.5 - (data['MAG'][index] + data['RZ'][index]))/2.5) # r-band flux
        meta['FLUX_Z'][:] = 10**((22.5 - data['MAG'][index])/2.5) # z-band flux
        meta['FLUX_W1'][:] = 10**((22.5 - (data['RZ'][index] + data['MAG'][index] - data['RW1'][index]))/2.5)# wise flux 1
        meta['FLUX_W2'][:] = 10**((22.5 - (data['RZ'][index] + data['MAG'][index] - data['RW1'][index] - data['W1W2'][index]))/2.5)# wise flux 2
        
        return meta
    
    def qso(self, data, index=None, mockformat='gaussianfield'):
        """Generate magnitudes for the QSO or QSO/LYA samples.
        """
        from desisim.lya_spectra import get_spectra
        
        objtype = 'QSO'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)
        meta = empty_metatable(nmodel=nobj, objtype=objtype)

        if mockformat.lower() != 'gaussianfield':
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta
        
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT'),
                                      ('SEED', 'MAG', 'Z')):
            meta[inkey] = data[datakey][index]
        
        meta['FLUX_G'][:] = 10**((22.5 - data['MAG'][index])/2.5) # g-band flux
        meta['FLUX_R'][:] = 10**((22.5 - (data['MAG'][index] - data['GR'][index]))/2.5) # r-band flux
        meta['FLUX_Z'][:] = 10**((22.5 - (data['MAG'][index] - data['GR'][index] - data['RZ'][index]))/2.5) # z-band flux
        meta['FLUX_W1'][:] = 10**((22.5 - (data['MAG'][index] - data['GR'][index] - data['RW1'][index]))/2.5)# wise flux 1
        meta['FLUX_W2'][:] = 10**((22.5 - (data['MAG'][index] - data['GR'][index] - data['RW1'][index] - data['W1W2'][index]))/2.5)# wise flux 2
        
        lya = np.where( data['TEMPLATESUBTYPE'][index] == 'LYA' )[0]                 
        if len(lya) > 0:
            meta['SUBTYPE'][lya] = 'LYA'
                                       
        return meta
    def mws_wd(self, data, index=None, mockformat='wd'):
        """Generate magnitudes for the MWS_WD sample.  Deal with DA vs DB white dwarfs
        separately.
        """
        objtype = 'WD'
        if index is None:
            index = np.arange(len(data['Z']))
        nobj = len(index)

        meta = empty_metatable(nmodel=nobj, objtype=objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'SUBTYPE'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'TEMPLATESUBTYPE')):
            meta[inkey] = data[datakey][index]

        if mockformat.lower() != 'wd':
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))
            return meta
            
        for subtype in ('DA', 'DB'):
            these = np.where(meta['SUBTYPE'] == subtype)[0]
            if len(these) > 0:
                meta['SUBTYPE'][these] = subtype    
                meta['TEMPLATEID'][:] = -1
    
        meta['FLUX_G'][:] = 10**((22.5 - data['MAG'][index])/2.5) # g-band flux
        return meta