# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=======================
desitarget.mock.spectra
=======================

Functions dealing with assigning template spectra to mock targets.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
from desisim.io import read_basis_templates, empty_metatable

class TemplateKDTree(object):
    """Build a KD Tree for each object type.

    """
    def __init__(self, nproc=1, verbose=False):
        from speclite import filters
        from scipy.spatial import cKDTree as KDTree
        from desiutil.log import get_logger, DEBUG

        if verbose:
            self.log = get_logger(DEBUG)
        else:
            self.log = get_logger()
            
        self.nproc = nproc
        self.verbose = verbose

        self.bgs_meta = read_basis_templates(objtype='BGS', onlymeta=True)#, verbose=False)
        self.elg_meta = read_basis_templates(objtype='ELG', onlymeta=True)#, verbose=False)
        self.lrg_meta = read_basis_templates(objtype='LRG', onlymeta=True)#, verbose=False)
        self.qso_meta = read_basis_templates(objtype='QSO', onlymeta=True)#, verbose=False)
        self.wd_da_meta = read_basis_templates(objtype='WD', subtype='DA', onlymeta=True)#, verbose=False)
        self.wd_db_meta = read_basis_templates(objtype='WD', subtype='DB', onlymeta=True)#, verbose=False)

        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')

        # Read all the stellar spectra and synthesize DECaLS/WISE fluxes.
        self.star_phot()

        self.bgs_tree = KDTree(self._bgs())
        self.elg_tree = KDTree(self._elg())
        #self.lrg_tree = KDTree(self._lrg())
        #self.qso_tree = KDTree(self._qso())
        self.star_tree = KDTree(self._star())
        self.wd_da_tree = KDTree(self._wd_da())
        self.wd_db_tree = KDTree(self._wd_db())

    def star_phot(self, normfilter='decam2014-r'):
        """Synthesize photometry for the full set of stellar templates."""

        star_flux, star_wave, star_meta = read_basis_templates(objtype='STAR')#, verbose=False)
        star_maggies_table = self.decamwise.get_ab_maggies(star_flux, star_wave, mask_invalid=True)

        star_maggies = dict()
        for key in star_maggies_table.columns:
            star_maggies[key] = star_maggies_table[key] / star_maggies_table[normfilter] # normalized maggies
        self.star_flux_g = star_maggies['decam2014-g']
        self.star_flux_r = star_maggies['decam2014-r']
        self.star_flux_z = star_maggies['decam2014-z']
        self.star_flux_w1 = star_maggies['wise2010-W1']
        self.star_flux_w2 = star_maggies['wise2010-W1']
        
        self.star_meta = star_meta

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
                self.log.warning('Unrecognized SUBTYPE {}!'.format(subtype))
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

        self.tree = TemplateKDTree(nproc=nproc, verbose=verbose)

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
        from desisim.templates import BGS, ELG, LRG, QSO, SIMQSO, STAR, WD
        self.bgs_templates = BGS(wave=self.wave, normfilter='sdss2010-r') # Need to generalize this!
        self.elg_templates = ELG(wave=self.wave, normfilter='decam2014-r')
        self.lrg_templates = LRG(wave=self.wave, normfilter='decam2014-z')
        self.qso_templates = QSO(wave=self.wave, normfilter='decam2014-g')
        self.simqso_templates = SIMQSO(wave=self.wave, normfilter='decam2014-g')
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
        from desisim.lya_spectra import read_lya_skewers,apply_lya_transmission
        import fitsio

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
                ilya = index[lya].astype(int)
                nqso = ilya.size
                                
                if 'LYAHDU' in data : 
                    # this is the old format with one HDU per spectrum
                    alllyafile = data['LYAFILES'][ilya]
                    alllyahdu = data['LYAHDU'][ilya]

                    for lyafile in sorted(set(alllyafile)):
                        these = np.where( lyafile == alllyafile )[0]

                        templateid = alllyahdu[these] - 1 # templateid is 0-indexed
                        flux1, _, meta1 = get_spectra(lyafile, templateid=templateid, normfilter=data['FILTERNAME'],
                                                      rand=self.rand, qso=self.lya_templates, nocolorcuts=True)
                        meta1['SUBTYPE'] = 'LYA'
                        meta[lya[these]] = meta1
                        flux[lya[these], :] = flux1
                else : # new format
                    # Read skewers.
                    skewer_wave = None
                    skewer_trans = None
                    skewer_meta = None
                    
                    # All the files that contain at least one QSO skewer.
                    alllyafile = data['LYAFILES'][ilya]
                    uniquelyafiles = sorted(set(alllyafile))
                                        
                    for lyafile in uniquelyafiles:
                        these = np.where( alllyafile == lyafile )[0]
                        objid_in_data = data['OBJID'][ilya][these]
                        objid_in_mock = (fitsio.read(lyafile, columns=['MOCKID'], upper=True,
                                                     ext=1).astype(float)).astype(int)
                        o2i = dict()
                        for i, o in enumerate(objid_in_mock):
                            o2i[o] = i
                        indices_in_mock_healpix = np.zeros(objid_in_data.size).astype(int)
                        for i, o in enumerate(objid_in_data):
                            if not o in o2i:
                                self.log.error("No MOCKID={} in {}. It's a bug, should never happen".format(o,lyafile))
                                raise(KeyError("No MOCKID={} in {}. It's a bug, should never happen".format(o,lyafile)))
                            indices_in_mock_healpix[i] = o2i[o]
                        
                        tmp_wave, tmp_trans, tmp_meta = read_lya_skewers(lyafile, indices=indices_in_mock_healpix) 
                                                
                        if skewer_wave is None:
                            skewer_wave = tmp_wave
                            dw = skewer_wave[1]-skewer_wave[0] # this is just to check same wavelength
                            skewer_trans = np.zeros((nqso,skewer_wave.size)) # allocate skewer_array
                            skewer_meta = dict()
                            for k in tmp_meta.dtype.names:
                                skewer_meta[k] = np.zeros(nqso).astype(tmp_meta[k].dtype)
                        else :
                            # check wavelength is the same for all skewers
                            assert(np.max(np.abs(wave-tmp_wave))<0.001*dw)
                        
                        skewer_trans[these] = tmp_trans
                        for k in skewer_meta.keys():
                            skewer_meta[k][these] = tmp_meta[k]
                    
                    # Check we matched things correctly.
                    assert(np.max(np.abs(skewer_meta["Z"]-data['Z'][ilya]))<0.000001)
                    assert(np.max(np.abs(skewer_meta["RA"]-data['RA'][ilya]))<0.000001)
                    assert(np.max(np.abs(skewer_meta["DEC"]-data['DEC'][ilya]))<0.000001)
                    
                    # Now we create a series of QSO spectra all at once, which
                    # is faster than calling each one at a time.
                    
                    #seed = self.rand.randint(2**32)
                    #qso  = self.lya_templates
                    qso_flux, qso_wave, qso_meta = self.simqso_templates.make_templates(
                        nmodel=nqso, redshift=data['Z'][ilya], #seed=seed,
                        lyaforest=False, nocolorcuts=True)
                    
                    # apply transmission to QSOs
                    qso_flux = apply_lya_transmission(qso_wave, qso_flux, skewer_wave, skewer_trans)
                    
                    qso_meta['SUBTYPE'] = 'LYA'
                    meta[lya] = qso_meta
                    flux[lya, :] = qso_flux
                    
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
            
        return meta
