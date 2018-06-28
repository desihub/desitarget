# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=========================
desitarget.mock.mockmaker
=========================

Read mock catalogs and assign spectra.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
from pkg_resources import resource_filename

import fitsio
import healpy as hp
from astropy.table import Table, Column

from desimodel.io import load_pixweight
from desimodel import footprint
from desiutil.brick import brickname as get_brickname_from_radec

from desiutil.log import get_logger, DEBUG
log = get_logger()

try:
    from scipy import constants
    C_LIGHT = constants.c/1000.0
except TypeError: # This can happen during documentation builds.
    C_LIGHT = 299792458.0/1000.0

def mw_transmission(data, dust_dir=None):
    """Compute the grzW1W2 Galactic transmission for every object.
    
    Parameters
    ----------
    data : :class:`dict`
        Input dictionary of sources with RA, Dec coordinates, modified on output
        to contain reddening and the MW transmission in various bands.
    params : :class:`dict`
        Dictionary summary of the input configuration file, restricted to a
        particular source_name (e.g., 'QSO').
    dust_dir : :class:`str`
        Full path to the dust maps.

    Raises
    ------
    ValueError
        If dust_dir is not defined.
    
    """
    from desitarget.mock import sfdmap

    if dust_dir is None:
        log.warning('DUST_DIR input required.')
        raise ValueError

    extcoeff = dict(G = 3.214, R = 2.165, Z = 1.221, W1 = 0.184, W2 = 0.113)
    data['EBV'] = sfdmap.ebv(data['RA'], data['DEC'], mapdir=dust_dir)

    for band in ('G', 'R', 'Z', 'W1', 'W2'):
        data['MW_TRANSMISSION_{}'.format(band)] = 10**(-0.4 * extcoeff[band] * data['EBV'])

def imaging_depth(source_data):
    """Add the imaging depth to the source_data dictionary.

    Note: In future, this should be a much more sophisticated model based on the
    actual imaging data releases (e.g., it should depend on healpixel).

    Parameters
    ----------
    source_data : :class:`dict`
        Input dictionary of sources with RA, Dec coordinates, modified on output
        to contain the PSF and galaxy depth in various bands.
            
    """
    nobj = len(source_data['RA'])

    psfdepth_mag = np.array((24.65, 23.61, 22.84)) # 5-sigma, mag
    galdepth_mag = np.array((24.7, 23.9, 23.0))    # 5-sigma, mag

    psfdepth_ivar = (1 / 10**(-0.4 * (psfdepth_mag - 22.5)))**2 # 5-sigma, 1/nanomaggies**2
    galdepth_ivar = (1 / 10**(-0.4 * (galdepth_mag - 22.5)))**2 # 5-sigma, 1/nanomaggies**2

    for ii, band in enumerate(('G', 'R', 'Z')):
        source_data['PSFDEPTH_{}'.format(band)] = np.repeat(psfdepth_ivar[ii], nobj)
        source_data['GALDEPTH_{}'.format(band)] = np.repeat(galdepth_ivar[ii], nobj)

    wisedepth_mag = np.array((22.3, 23.8)) # 1-sigma, mag
    wisedepth_ivar = 1 / (5 * 10**(-0.4 * (wisedepth_mag - 22.5)))**2 # 5-sigma, 1/nanomaggies**2

    for ii, band in enumerate(('W1', 'W2')):
        source_data['PSFDEPTH_{}'.format(band)] = np.repeat(wisedepth_ivar[ii], nobj)
    
def empty_targets_table(nobj=1):
    """Initialize an empty 'targets' table.

    Parameters
    ----------
    nobj : :class:`int`
        Number of objects.

    Returns
    -------
    targets : :class:`astropy.table.Table`
        Targets table.
    
    """
    targets = Table()

    # RELEASE
    targets.add_column(Column(name='BRICKID', length=nobj, dtype='i4'))
    targets.add_column(Column(name='BRICKNAME', length=nobj, dtype='U8'))
    targets.add_column(Column(name='BRICK_OBJID', length=nobj, dtype='i4'))
    # TYPE
    targets.add_column(Column(name='RA', length=nobj, dtype='f8', unit='degree'))
    targets.add_column(Column(name='DEC', length=nobj, dtype='f8', unit='degree'))
    # RA_IVAR
    # DEC_IVAR
    # DCHISQ
    targets.add_column(Column(name='FLUX_G', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_R', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_Z', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_W1', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_W2', length=nobj, dtype='f4', unit='nanomaggies'))
    # FLUX_W3
    # FLUX_W4
    # FLUX_IVAR_G
    # FLUX_IVAR_R
    # FLUX_IVAR_Z
    # FLUX_IVAR_W1
    # FLUX_IVAR_W2
    # FLUX_IVAR_W3
    # FLUX_IVAR_W4
    targets.add_column(Column(name='MW_TRANSMISSION_G', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_Z', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_W1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_W2', length=nobj, dtype='f4'))
    # MW_TRANSMISSION_W3
    # MW_TRANSMISSION_W4
    # NOBS_G
    # NOBS_R
    # NOBS_Z
    # FRACFLUX_G
    # FRACFLUX_R
    # FRACFLUX_Z
    targets.add_column(Column(name='PSFDEPTH_G', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='PSFDEPTH_R', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='PSFDEPTH_Z', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='GALDEPTH_G', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='GALDEPTH_R', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='GALDEPTH_Z', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    # The following two columns do not appear in the data targets catalog.
    targets.add_column(Column(name='PSFDEPTH_W1', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='PSFDEPTH_W2', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='SHAPEDEV_R', length=nobj, dtype='f4', unit='arcsec'))
    # SHAPEDEV_R_IVAR
    targets.add_column(Column(name='SHAPEDEV_E1', length=nobj, dtype='f4'))
    # SHAPEDEV_E1_IVAR
    targets.add_column(Column(name='SHAPEDEV_E2', length=nobj, dtype='f4'))
    # SHAPEDEV_E2_IVAR    
    targets.add_column(Column(name='SHAPEEXP_R', length=nobj, dtype='f4', unit='arcsec'))
    # SHAPEEXP_R_IVAR
    targets.add_column(Column(name='SHAPEEXP_E1', length=nobj, dtype='f4'))
    # SHAPEEXP_E1_IVAR
    targets.add_column(Column(name='SHAPEEXP_E2', length=nobj, dtype='f4'))
    # SHAPEEXP_E2_IVAR
    targets.add_column(Column(name='SUBPRIORITY', length=nobj, dtype='f8'))
    targets.add_column(Column(name='TARGETID', length=nobj, dtype='int64'))
    targets.add_column(Column(name='DESI_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='BGS_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='MWS_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='HPXPIXEL', length=nobj, dtype='i8'))
    # PHOTSYS

    return targets

def empty_truth_table(nobj=1):
    """Initialize an empty 'truth' table.

    Parameters
    ----------
    nobj : :class:`int`
        Number of objects.

    Returns
    -------
    truth : :class:`astropy.table.Table`
        Truth table.
    
    """
    truth = Table()
    truth.add_column(Column(name='TARGETID', length=nobj, dtype='int64'))
    truth.add_column(Column(name='MOCKID', length=nobj, dtype='int64'))
    truth.add_column(Column(name='CONTAM_TARGET', length=nobj, dtype='i8'))

    truth.add_column(Column(name='TRUEZ', length=nobj, dtype='f4', data=np.zeros(nobj)))
    truth.add_column(Column(name='TRUESPECTYPE', length=nobj, dtype='U10')) # GALAXY, QSO, STAR, etc.
    truth.add_column(Column(name='TEMPLATETYPE', length=nobj, dtype='U10')) # ELG, BGS, STAR, WD, etc.
    truth.add_column(Column(name='TEMPLATESUBTYPE', length=nobj, dtype='U10')) # DA, DB, etc.

    truth.add_column(Column(name='TEMPLATEID', length=nobj, dtype='i4', data=np.zeros(nobj)-1))
    truth.add_column(Column(name='SEED', length=nobj, dtype='int64', data=np.zeros(nobj)-1))
    truth.add_column(Column(name='MAG', length=nobj, dtype='f4', data=np.zeros(nobj)+30, unit='mag'))
    truth.add_column(Column(name='VDISP', length=nobj, dtype='f4', data=np.zeros(nobj), unit='km/s'))
    
    truth.add_column(Column(name='FLUX_G', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_R', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_Z', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_W1', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_W2', length=nobj, dtype='f4', unit='nanomaggies'))

    truth.add_column(Column(name='OIIFLUX', length=nobj, dtype='f4',
                            data=np.zeros(nobj)-1, unit='erg/(s*cm2)'))
    truth.add_column(Column(name='HBETAFLUX', length=nobj, dtype='f4',
                            data=np.zeros(nobj)-1, unit='erg/(s*cm2)'))

    truth.add_column(Column(name='TEFF', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='K'))
    truth.add_column(Column(name='LOGG', length=nobj, dtype='f4', data=np.zeros(nobj)-1, unit='m/(s**2)'))
    truth.add_column(Column(name='FEH', length=nobj, dtype='f4', data=np.zeros(nobj)-1))

    return truth

def _get_radec(mockfile, nside, pixmap, mxxl=False):

    log.info('Reading {}'.format(mockfile))
    radec = fitsio.read(mockfile, columns=['RA', 'DEC'], upper=True, ext=1)
    ra = radec['RA'].astype('f8') % 360.0 # enforce 0 < ra < 360
    dec = radec['DEC'].astype('f8')

    log.info('Assigning healpix pixels with nside = {}.'.format(nside))
    allpix = footprint.radec2pix(nside, ra, dec)

    pixweight = load_pixweight(nside, pixmap=pixmap)

    return ra, dec, allpix, pixweight
        
def _default_wave(wavemin=None, wavemax=None, dw=0.2):
    """Generate a default wavelength vector for the output spectra."""
    from desimodel.io import load_throughput
    
    if wavemin is None:
        wavemin = load_throughput('b').wavemin - 10.0
    if wavemax is None:
        wavemax = load_throughput('z').wavemax + 10.0
            
    return np.arange(round(wavemin, 1), wavemax, dw)

class SelectTargets(object):
    """Methods to help select various target types.

    """
    def __init__(self):
        from astropy.io import fits
        from ..targetmask import (desi_mask, bgs_mask, mws_mask)
        from ..contammask import contam_mask
        
        self.desi_mask = desi_mask
        self.bgs_mask = bgs_mask
        self.mws_mask = mws_mask
        self.contam_mask = contam_mask

        # Read and cache the default pixel weight map.
        pixfile = os.path.join(os.environ['DESIMODEL'],'data','footprint','desi-healpix-weights.fits')
        with fits.open(pixfile) as hdulist:
            self.pixmap = hdulist[0].data

    def scatter_photometry(self, data, truth, targets, indx=None, psf=True,
                           seed=None, qaplot=False):
        """Add noise to the input (noiseless) photometry based on the depth.  The input
        targets table is modified in place.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        indx : :class:`numpy.ndarray`, optional
            Scatter the photometry of a subset of the objects in the data
            dictionary, as specified using their zero-indexed indices.
        psf : :class:`bool`, optional
            For point sources (e.g., QSO, STAR) use the PSFDEPTH values,
            otherwise use GALDEPTH.  Defaults to True.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        qaplot : :class:`bool`, optional
            Generate a QA plot for debugging.

        """
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)
            
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if psf:
            depthprefix = 'PSF'
        else:
            depthprefix = 'GAL'

        factor = 5 # -- should this be 1 or 5???

        for band in ('G', 'R', 'Z'):
            fluxkey = 'FLUX_{}'.format(band)
            depthkey = '{}DEPTH_{}'.format(depthprefix, band)

            sigma = 1 / np.sqrt(data[depthkey][indx]) / 5 # nanomaggies, 1-sigma
            targets[fluxkey][:] = truth[fluxkey] + rand.normal(scale=sigma)

        for band in ('W1', 'W2'):
            fluxkey = 'FLUX_{}'.format(band)
            depthkey = 'PSFDEPTH_{}'.format(band)

            sigma = 1 / np.sqrt(data[depthkey][indx]) / 5 # nanomaggies, 1-sigma
            targets[fluxkey][:] = truth[fluxkey] + rand.normal(scale=sigma)

        if qaplot:
            self._qaplot_scatter_photometry(targets, truth)

    def _qaplot_scatter_photometry(self, targets, truth):
        """Build a simple QAplot, useful for debugging """
        import matplotlib.pyplot as plt

        gr1 = -2.5 * np.log10( truth['FLUX_G'] / truth['FLUX_R'] )
        rz1 = -2.5 * np.log10( truth['FLUX_R'] / truth['FLUX_Z'] )
        gr = -2.5 * np.log10( targets['FLUX_G'] / targets['FLUX_R'] )
        rz = -2.5 * np.log10( targets['FLUX_R'] / targets['FLUX_Z'] )
        plt.scatter(rz1, gr1, color='red', alpha=0.5, edgecolor='none', 
                    label='Noiseless Photometry')
        plt.scatter(rz, gr, alpha=0.5, color='green', edgecolor='none',
                    label='Noisy Photometry')
        plt.xlim(-0.5, 2) ; plt.ylim(-0.5, 2)
        plt.legend(loc='upper left')
        plt.show()

    def _update_normfilter(self, normfilter, objtype=None):
        """Update the normalization filter."""
        from speclite import filters
        if normfilter is not None:
            if objtype == 'WD':
                self.da_template_maker.normfilter = normfilter
                self.db_template_maker.normfilter = normfilter
                self.da_template_maker.normfilt = filters.load_filters(normfilter)
                self.db_template_maker.normfilt = filters.load_filters(normfilter)
            else:
                self.template_maker.normfilter = normfilter
                self.template_maker.normfilt = filters.load_filters(normfilter)

    def _sample_vdisp(self, ra, dec, mean=1.9, sigma=0.15, fracvdisp=(0.1, 1),
                      seed=None, nside=128):
        """Assign velocity dispersions to a subset of objects."""
        rand = np.random.RandomState(seed)

        def _sample(nmodel=1):
            nvdisp = int(np.max( ( np.min( ( np.round(nmodel * fracvdisp[0]), fracvdisp[1] ) ), 1 ) ))
            vvdisp = 10**rand.normal(loc=mean, scale=sigma, size=nvdisp)
            return rand.choice(vvdisp, nmodel)

        # Hack! Assign the same velocity dispersion to galaxies in the same
        # healpixel.
        nobj = len(ra)
        vdisp = np.zeros(nobj)

        healpix = footprint.radec2pix(nside, ra, dec)
        for pix in set(healpix):
            these = np.in1d(healpix, pix)
            vdisp[these] = _sample(nmodel=np.count_nonzero(these))

        return vdisp

    def sample_gmm_nospectra(self, meta, rand=None):
        """Sample from one of the Gaussian mixture models generated by
        desitarget/doc/nb/gmm-quicksurvey.ipynb.

        Note: Any changes to the quantities fitted in that notebook need to be
        updated here!
        
        """
        if rand is None:
            rand = np.random.RandomState()

        nsample = len(meta)
        target = meta['OBJTYPE'][0].upper()
        
        data = self.GMM_nospectra.sample(nsample, random_state=rand)

        alltags = dict()
        alltags['ELG'] = ('r', 'g - r', 'r - z', 'z - W1', 'W1 - W2', 'oii')
        alltags['LRG'] = ('z', 'g - r', 'r - z', 'z - W1', 'W1 - W2')
        alltags['BGS'] = ('r', 'g - r', 'r - z', 'z - W1', 'W1 - W2')
        alltags['QSO'] = ('g', 'g - r', 'r - z', 'z - W1', 'W1 - W2')
        alltags['LYA'] = ('g', 'g - r', 'r - z', 'z - W1', 'W1 - W2')

        tags = alltags[target]
        sample = np.empty( nsample, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            sample[tt] = data[:, ii]

        if target == 'ELG' or target == 'BGS':
            rmag = sample['r']
            zmag = rmag - sample['r - z']
            gmag = sample['g - r'] + rmag
            normmag = rmag
        elif target == 'LRG':
            zmag = sample['z']
            rmag = sample['r - z'] + zmag
            gmag = sample['g - r'] + rmag
            normmag = zmag
        elif target == 'QSO' or target == 'LYA':
            gmag = sample['g']
            rmag = gmag - sample['g - r']
            zmag = rmag - sample['r - z']
            normmag = gmag

        W1mag = zmag - sample['z - W1'] 
        W2mag = W1mag - sample['W1 - W2'] 
        
        meta['MAG'] = normmag
        meta['FLUX_G'] = 1e9 * 10**(-0.4 * gmag)
        meta['FLUX_R'] = 1e9 * 10**(-0.4 * rmag)
        meta['FLUX_Z'] = 1e9 * 10**(-0.4 * zmag)
        meta['FLUX_W1'] = 1e9 * 10**(-0.4 * W1mag)
        meta['FLUX_W2'] = 1e9 * 10**(-0.4 * W2mag)

        return meta

    def deredden(self, targets):
        """Correct photometry for Galactic extinction."""

        unredflux = list()
        for band in ('G', 'R', 'Z', 'W1', 'W2'):
            unredflux.append(targets['FLUX_{}'.format(band)] /
                             targets['MW_TRANSMISSION_{}'.format(band)])
        gflux, rflux, zflux, w1flux, w2flux = unredflux

        return gflux, rflux, zflux, w1flux, w2flux

    def populate_targets_truth(self, data, meta, indx=None, seed=None, psf=True,
                               gmm=None,  truespectype='', templatetype='',
                               templatesubtype=''):
        """Initialize and populate the targets and truth tables given a dictionary of
        source properties and a spectral metadata table.  

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        indx : :class:`numpy.ndarray`, optional
            Populate the tables of a subset of the objects in the data
            dictionary, as specified using their zero-indexed indices.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        psf : :class:`bool`, optional
            For point sources (e.g., QSO, STAR) use the PSFDEPTH values,
            otherwise use GALDEPTH.  Defaults to True.
        gmm : :class:`numpy.ndarray`, optional
            Sample properties drawn from
            desiutil.sklearn.GaussianMixtureModel.sample.
        truespectype : :class:`str` or :class:`numpy.array`, optional
            True spectral type.  Defaults to ''.
        templatetype : :class:`str` or :class:`numpy.array`, optional
            True template type.  Defaults to ''.
        templatesubtype : :class:`str` or :class:`numpy.array`, optional
            True template subtype.  Defaults to ''.
        
        Returns
        -------
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        if seed is None:
            seed = self.seed
            
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        # Initialize the tables.
        targets = empty_targets_table(nobj)
        truth = empty_truth_table(nobj)

        # Add some basic info.
        for key in ('RA', 'DEC', 'BRICKNAME'):
            targets[key][:] = data[key][indx]
            
        truth['MOCKID'][:] = data['MOCKID'][indx]

        # Add dust and depth.
        for band in ('G', 'R', 'Z', 'W1', 'W2'):
            key = 'MW_TRANSMISSION_{}'.format(band)
            targets[key][:] = data[key][indx]

        for band in ('G', 'R', 'Z'):
            for prefix in ('PSF', 'GAL'):
                key = '{}DEPTH_{}'.format(prefix, band)
                targets[key][:] = data[key][indx]

        for band in ('W1', 'W2'):
            key = 'PSFDEPTH_{}'.format(band)
            targets[key][:] = data[key][indx]

        # Add spectral / template type and subtype.
        for value, key in zip( (truespectype, templatetype, templatesubtype),
                               ('TRUESPECTYPE', 'TEMPLATETYPE', 'TEMPLATESUBTYPE') ):
            if isinstance(value, np.ndarray):
                truth[key][:] = value
            else:
                truth[key][:] = np.repeat(value, nobj)

        if gmm is not None:
            for gmmkey, key in zip( ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2'),
                                    ('SHAPEEXP_R', 'SHAPEEXP_E1', 'SHAPEEXP_E2',
                                     'SHAPEDEV_R', 'SHAPEDEV_E1', 'SHAPEDEV_E2') ):
                targets[key][:] = gmm[gmmkey]

        # Copy various quantities from the metadata table.
        for key in ('TEMPLATEID', 'SEED', 'REDSHIFT', 'MAG', 'VDISP', 'FLUX_G', 'FLUX_R', 'FLUX_Z',
                    'FLUX_W1', 'FLUX_W2', 'OIIFLUX', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'):
            truth[key.replace('REDSHIFT', 'TRUEZ')][:] = meta[key]

        # Scatter the photometry based on the depth.
        self.scatter_photometry(data, truth, targets, indx=indx, psf=psf, seed=seed)

        # Finally, attenuate the observed photometry for Galactic extinction.
        for band, key in zip( ('G', 'R', 'Z', 'W1', 'W2'),
                              ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2') ):
            targets[key][:] = targets[key] * data['MW_TRANSMISSION_{}'.format(band)][indx]
        
        return targets, truth

    def mock_density(self, mockfile=None, nside=16, density_per_pixel=False):
        """Compute the median density of targets in the full mock. 

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog.
        nside : :class:`int`
            Healpixel nside for the calculation.
        density_per_pixel : :class:`bool`, optional
            Return the density per healpixel rather than just the median
            density, which may be useful for statistical purposes.

        Returns
        -------
        mock_density : :class:`int` or :class:`numpy.ndarray`
            Median density of targets per deg2 or target density in all
            healpixels (if density_per_pixel=True).  

        Raises
        ------
        ValueError
            If mockfile is not defined.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError

        try:
            mockfile = mockfile.format(**os.environ)
        except KeyError as e:
            log.warning('Environment variable not set for mockfile: {}'.format(e))
            raise ValueError
        
        areaperpix = hp.nside2pixarea(nside, degrees=True)

        radec = fitsio.read(mockfile, columns=['RA', 'DEC'], upper=True, ext=1)
        healpix = footprint.radec2pix(nside, radec['RA'], radec['DEC'])

        # Get the weight per pixel, protecting against divide-by-zero.
        pixweight = load_pixweight(nside, pixmap=self.pixmap)
        weight = np.zeros_like(radec['RA'])
        good = np.nonzero(pixweight[healpix])
        weight[good] = 1 / pixweight[healpix[good]]

        mock_density = np.bincount(healpix, weights=weight) / areaperpix # [targets/deg]
        mock_density = mock_density[np.flatnonzero(mock_density)]

        if density_per_pixel:
            return mock_density
        else:
            return np.median(mock_density)

    def qamock_sky(self, data, xlim=(0, 4), nozhist=False, png=None):
        """Generate a QAplot showing the sky and redshift distribution of the objects in
        the mock.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.

        """
        import warnings
        import matplotlib.pyplot as plt
        from desiutil.plots import init_sky, plot_sky_binned
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            basemap = init_sky(galactic_plane_color='k', ax=ax[0]);
            plot_sky_binned(data['RA'], data['DEC'], weights=data['WEIGHT'],
                            max_bin_area=hp.nside2pixarea(data['NSIDE'], degrees=True),
                            verbose=False, clip_lo='!1', cmap='viridis',
                            plot_type='healpix', basemap=basemap,
                            label=r'{} (targets/deg$^2$)'.format(self.objtype))
            
        if not nozhist:
            ax[1].hist(data['Z'], bins=100, histtype='stepfilled',
                       alpha=0.6, label=self.objtype, weights=data['WEIGHT'])
            ax[1].set_xlabel('Redshift')
            ax[1].set_xlim( xlim )
            ax[1].yaxis.set_major_formatter(plt.NullFormatter())
            ax[1].legend(loc='upper right', frameon=False)
        else:
            ax[1].axis('off')
        fig.subplots_adjust(wspace=0.2)

        if png:
            print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()

class ReadGaussianField(SelectTargets):
    """Read a Gaussian random field style mock catalog.

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    cached_radec = None
    
    def __init__(self, dust_dir=None, bricksize=0.25):
        super(ReadGaussianField, self).__init__()
        
        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 target_name='', mock_density=False):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        target_name : :class:`str`
            Name of the target being read (e.g., ELG, LRG).
        mock_density : :class:`bool`, optional
            Compute and return the median target density in the mock.  Defaults
            to False.

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data file is not found.
        ValueError
            If mockfile is not defined or if nside is not a scalar.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError

        try:
            mockfile = mockfile.format(**os.environ)
        except KeyError as e:
            log.warning('Environment variable not set for mockfile: {}'.format(e))
            raise ValueError
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        # Default set of healpixels is the whole DESI footprint.
        if healpixels is None:
            if nside is None:
                nside = 16
            log.info('Reading the whole DESI footprint with nside = {}.'.format(nside))
            healpixels = footprint.tiles2pix(nside)

        if nside is None:
            log.warning('Nside must be a scalar input.')
            raise ValueError

        # Read the ra,dec coordinates, pixel weight map, generate mockid, and
        # then restrict to the desired healpixels.
        if self.cached_radec is None:
            ra, dec, allpix, pixweight = _get_radec(mockfile, nside, self.pixmap)
            ReadGaussianField.cached_radec = (mockfile, nside, ra, dec, allpix, pixweight)
        else:
            cached_mockfile, cached_nside, ra, dec, allpix, pixweight = ReadGaussianField.cached_radec
            if cached_mockfile != mockfile or cached_nside != nside:
                ra, dec, allpix, pixweight = _get_radec(mockfile, nside, self.pixmap)
                ReadGaussianField.cached_radec = (mockfile, nside, ra, dec, allpix, pixweight)
            else:
                log.info('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, allpix, pixweight = ReadGaussianField.cached_radec

        mockid = np.arange(len(ra)) # unique ID/row number
        
        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s).'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        mockid = mockid[cut]
        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = ra[cut]
        dec = dec[cut]

        # Assign bricknames.
        brickname = get_brickname_from_radec(ra, dec, bricksize=self.bricksize)

        # Add redshifts.
        if target_name.upper() == 'SKY':
            zz = np.zeros(len(ra))
        else:
            data = fitsio.read(mockfile, columns=['Z_COSMO', 'DZ_RSD'], upper=True, ext=1, rows=cut)
            zz = (data['Z_COSMO'].astype('f8') + data['DZ_RSD'].astype('f8')).astype('f4')
            
        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'gaussianfield',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz}

        # Add MW transmission and the imaging depth.
        if self.dust_dir:
            mw_transmission(out, dust_dir=self.dust_dir)
            imaging_depth(out)

        # Optionally compute the mean mock density.
        if mock_density:
            out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile)

        return out

class ReadUniformSky(SelectTargets):
    """Read a uniform sky style mock catalog.

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    cached_radec = None
    
    def __init__(self, dust_dir=None, bricksize=0.25):
        super(ReadUniformSky, self).__init__()

        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 target_name='', mock_density=False):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        target_name : :class:`str`
            Name of the target being read (e.g., ELG, LRG).
        mock_density : :class:`bool`, optional
            Compute and return the median target density in the mock.  Defaults
            to False.

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data file is not found.
        ValueError
            If mockfile is not defined or if nside is not a scalar.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError

        try:
            mockfile = mockfile.format(**os.environ)
        except KeyError as e:
            log.warning('Environment variable not set for mockfile: {}'.format(e))
            raise ValueError
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        # Default set of healpixels is the whole DESI footprint.
        if healpixels is None:
            if nside is None:
                nside = 16
            log.info('Reading the whole DESI footprint with nside = {}.'.format(nside))
            healpixels = footprint.tiles2pix(nside)

        if nside is None:
            log.warning('Nside must be a scalar input.')
            raise ValueError

        # Read the ra,dec coordinates, pixel weight map, generate mockid, and
        # then restrict to the desired healpixels.
        if self.cached_radec is None:
            ra, dec, allpix, pixweight = _get_radec(mockfile, nside, self.pixmap)
            ReadUniformSky.cached_radec = (mockfile, nside, ra, dec, allpix, pixweight)
        else:
            cached_mockfile, cached_nside, ra, dec, allpix, pixweight = ReadUniformSky.cached_radec
            if cached_mockfile != mockfile or cached_nside != nside:
                ra, dec, allpix, pixweight = _get_radec(mockfile, nside, self.pixmap)
                ReadUniformSky.cached_radec = (mockfile, nside, ra, dec, allpix, pixweight)
            else:
                log.info('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, allpix, pixweight = ReadUniformSky.cached_radec

        mockid = np.arange(len(ra)) # unique ID/row number

        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s).'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        mockid = mockid[cut]
        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = ra[cut]
        dec = dec[cut]

        # Assign bricknames.
        brickname = get_brickname_from_radec(ra, dec, bricksize=self.bricksize)

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'uniformsky',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': np.zeros(len(ra))}

        # Add MW transmission and the imaging depth.
        if self.dust_dir:
            mw_transmission(out, dust_dir=self.dust_dir)
            imaging_depth(out)

        # Optionally compute the mean mock density.
        if mock_density:
            out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile)

        return out

class ReadGalaxia(SelectTargets):
    """Read a Galaxia style mock catalog.

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    cached_pixweight = None

    def __init__(self, bricksize=0.25, dust_dir=None):
        super(ReadGalaxia, self).__init__()

        self.bricksize = bricksize
        self.dust_dir = dust_dir

    def readmock(self, mockfile=None, healpixels=[], nside=[], nside_galaxia=8, 
                 target_name='MWS_MAIN', magcut=None):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the top-level directory of the Galaxia mock catalog.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        nside_galaxia : :class:`int`
            Healpixel nside indicating how the mock on-disk has been organized.
            Defaults to 8.
        target_name : :class:`str`
            Name of the target being read (e.g., MWS_MAIN).
        magcut : :class:`float`
            Magnitude cut (hard-coded to SDSS r-band) to subselect targets
            brighter than magcut. 

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the top-level Galaxia directory is not found.
        ValueError
            (1) If either mockfile or nside_galaxia are not defined; (2) if
            healpixels or nside are not scalar inputs; or (3) if the input
            target_name is not recognized.

        """
        from desitarget.targets import encode_targetid
        from desitarget.mock.io import get_healpix_dir, findfile

        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError

        try:
            mockfile = mockfile.format(**os.environ)
        except KeyError as e:
            log.warning('Environment variable not set for mockfile: {}'.format(e))
            raise ValueError
        
        if nside_galaxia is None:
            log.warning('Nside_galaxia input is required.')
            raise ValueError
        
        mockfile_nside = os.path.join(mockfile, str(nside_galaxia))
        if not os.path.isdir(mockfile_nside):
            log.warning('Galaxia top-level directory {} not found!'.format(mockfile_nside))
            raise IOError

        # Because of the size of the Galaxia mock, healpixels (and nside) must
        # be scalars.
        if len(np.atleast_1d(healpixels)) != 1 and len(np.atleast_1d(nside)) != 1:
            log.warning('Healpixels and nside must be scalar inputs.')
            raise ValueError

        if self.cached_pixweight is None:
            pixweight = load_pixweight(nside, pixmap=self.pixmap)
            ReadGalaxia.cached_pixweight = (pixweight, nside)
        else:
            pixweight, cached_nside = ReadGalaxia.cached_pixweight
            if cached_nside != nside:
                pixweight = load_pixweight(nside, pixmap=self.pixmap)
                ReadGalaxia.cached_pixweight = (pixweight, nside)
            else:
                log.info('Using cached pixel weight map.')
                pixweight, _ = ReadGalaxia.cached_pixweight

        # Get the set of nside_galaxia pixels that belong to the desired
        # healpixels (which have nside).  This will break if healpixels is a
        # vector.
        theta, phi = hp.pix2ang(nside, healpixels, nest=True)
        pixnum = hp.ang2pix(nside_galaxia, theta, phi, nest=True)

        if target_name.upper() == 'MWS_MAIN':
            filetype = 'mock_allsky_galaxia_desi'
        elif target_name.upper() == 'FAINTSTAR':
            filetype = ('mock_superfaint_allsky_galaxia_desi_b10_cap_north',
                        'mock_superfaint_allsky_galaxia_desi_b10_cap_south')
        else:
            log.warning('Unrecognized target name {}!'.format(target_name))
            raise ValueError

        for ff in np.atleast_1d(filetype):
            galaxiafile = findfile(filetype=ff, nside=nside_galaxia, pixnum=pixnum,
                                   basedir=mockfile_nside, ext='fits')
            if os.path.isfile(galaxiafile):
                break

        if len(galaxiafile) == 0:
            log.warning('File {} not found!'.format(galaxiafile))
            raise IOError

        log.info('Reading {}'.format(galaxiafile))
        radec = fitsio.read(galaxiafile, columns=['RA', 'DEC'], upper=True, ext=1)
        nobj = len(radec)

        objid = np.arange(nobj)
        mockid = encode_targetid(objid=objid, brickid=pixnum, mock=1)

        allpix = footprint.radec2pix(nside, radec['RA'], radec['DEC'])

        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        mockid = mockid[cut]
        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = radec['RA'][cut].astype('f8') % 360.0 # enforce 0 < ra < 360
        dec = radec['DEC'][cut].astype('f8')
        del radec

        cols = ['V_HELIO', 'SDSSU_TRUE_NODUST', 'SDSSG_TRUE_NODUST',
                'SDSSR_TRUE_NODUST', 'SDSSI_TRUE_NODUST', 'SDSSZ_TRUE_NODUST',
                'SDSSR_OBS', 'TEFF', 'LOGG', 'FEH']
        data = fitsio.read(galaxiafile, columns=cols, upper=True, ext=1, rows=cut)
        zz = (data['V_HELIO'].astype('f4') / C_LIGHT).astype('f4')
        mag = data['SDSSR_TRUE_NODUST'].astype('f4') # SDSS r-band, extinction-corrected
        mag_obs = data['SDSSR_OBS'].astype('f4')     # SDSS r-band, observed
        teff = 10**data['TEFF'].astype('f4')         # log10!
        logg = data['LOGG'].astype('f4')
        feh = data['FEH'].astype('f4')

        # Temporary hack to select SDSS standards using extinction-corrected
        # SDSS mags.
        boss_std = self.select_sdss_std(data['SDSSU_TRUE_NODUST'],
                                        data['SDSSG_TRUE_NODUST'],
                                        data['SDSSR_TRUE_NODUST'],
                                        data['SDSSI_TRUE_NODUST'],
                                        data['SDSSZ_TRUE_NODUST'],
                                        obs_rmag=None)
        
        if magcut:
            cut = mag < magcut
            if np.count_nonzero(cut) == 0:
                log.warning('No objects with r < {}!'.format(magcut))
                return dict()
            else:
                mockid = mockid[cut]
                allpix = allpix[cut]
                weight = weight[cut]
                ra = ra[cut]
                dec = dec[cut]
                boss_std = boss_std[cut]
                zz = zz[cut]
                mag = mag[cut]
                mag_obs = mag_obs[cut]
                teff = teff[cut]
                logg = logg[cut]
                feh = feh[cut]
                nobj = len(ra)
                log.info('Trimmed to {} {}s with r < {}.'.format(nobj, target_name, magcut))

        # Assign bricknames.
        brickname = get_brickname_from_radec(ra, dec, bricksize=self.bricksize)
        
        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'galaxia',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': mag, 'MAG_OBS': mag_obs,
               'TEFF': teff, 'LOGG': logg, 'FEH': feh,
               'NORMFILTER': 'sdss2010-r', 'BOSS_STD': boss_std}

        # Add MW transmission and the imaging depth.
        if self.dust_dir:
            mw_transmission(out, dust_dir=self.dust_dir)
            imaging_depth(out)

        return out

    def select_sdss_std(self, umag, gmag, rmag, imag, zmag, obs_rmag=None):
        """Select standard stars using SDSS photometry and the BOSS algorithm.
        
        According to http://www.sdss.org/dr12/algorithms/boss_std_ts the r-band
        magnitude for the magnitude cuts is the extinction corrected magnitude.

        Parameters
        ----------
        umag : :class:`float`
            SDSS u-band extinction-corrected magnitude.
        gmag : :class:`float`
            SDSS g-band extinction-corrected magnitude.
        rmag : :class:`float`
            SDSS r-band extinction-corrected magnitude.
        imag : :class:`float`
            SDSS i-band extinction-corrected magnitude.
        zmag : :class:`float`
            SDSS z-band extinction-corrected magnitude.
        obs_rmag : :class:`float`
            SDSS r-band observed (not extinction-corrected) magnitude.

        """
        umg_cut = ((umag - gmag) - 0.82)**2
        gmr_cut = ((gmag - rmag) - 0.30)**2
        rmi_cut = ((rmag - imag) - 0.09)**2
        imz_cut = ((imag - zmag) - 0.02)**2
    
        is_std = np.sqrt((umg_cut + gmr_cut + rmi_cut + imz_cut)) < 0.08
    
        if obs_rmag is not None:
            is_std &= (15.0 < obs_rmag) & (obs_rmag < 19)
        
        return is_std

class ReadLyaCoLoRe(SelectTargets):
    """Read a CoLoRe mock catalog of Lya skewers.

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    def __init__(self, dust_dir=None, bricksize=0.25):
        super(ReadLyaCoLoRe, self).__init__()

        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 target_name='LYA', nside_lya=16, mock_density=False):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the top-level directory of the CoLoRe mock catalog.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        target_name : :class:`str`
            Name of the target being read (if not LYA).
        nside_lya : :class:`int`
            Healpixel nside indicating how the mock on-disk has been organized.
            Defaults to 16.
        mock_density : :class:`bool`, optional
            Compute and return the median target density in the mock.  Defaults
            to False.

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the top-level mock data file is not found.
        ValueError
            If mockfile, nside, or nside_lya are not defined.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError
        
        try:
            mockfile = mockfile.format(**os.environ)
        except KeyError as e:
            log.warning('Environment variable not set for mockfile: {}'.format(e))
            raise ValueError
        
        if nside_lya is None:
            log.warning('Nside_lya input is required.')
            raise ValueError
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        mockdir = os.path.dirname(mockfile)
    
        # Default set of healpixels is the whole DESI footprint.
        if healpixels is None:
            if nside is None:
                nside = 16
            log.info('Reading the whole DESI footprint with nside = {}.'.format(nside))
            healpixels = footprint.tiles2pix(nside)

        if nside is None:
            log.warning('Nside must be a scalar input.')
            raise ValueError

        pixweight = load_pixweight(nside, pixmap=self.pixmap)

        # Read the ra,dec coordinates and then restrict to the desired
        # healpixels.
        log.info('Reading {}'.format(mockfile))
        tmp = fitsio.read(mockfile, columns=['RA', 'DEC', 'MOCKID' ,'Z', 'PIXNUM'],
                          upper=True, ext=1)
        
        ra = tmp['RA'].astype('f8') % 360.0 # enforce 0 < ra < 360
        dec = tmp['DEC'].astype('f8')            
        zz = tmp['Z'].astype('f4')
        mockpix = tmp['PIXNUM']
        mockid = (tmp['MOCKID'].astype(float)).astype(int)
        #objid = (tmp['MOCKID'].astype(float)).astype(int) # will change
        #mockid = objid.copy()
        del tmp

        log.info('Assigning healpix pixels with nside = {}'.format(nside))
        allpix = footprint.radec2pix(nside, ra, dec)

        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s).'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = ra[cut]
        dec = dec[cut]
        zz = zz[cut]
        #objid = objid[cut]
        mockpix = mockpix[cut]
        mockid = mockid[cut]

        # Build the full filenames.
        lyafiles = []
        for mpix in mockpix:
            lyafiles.append("%s/%d/%d/transmission-%d-%d.fits"%(
                mockdir, mpix//100, mpix, nside_lya, mpix))
            
        # Assign bricknames.
        brickname = get_brickname_from_radec(ra, dec, bricksize=self.bricksize)

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'CoLoRe',               
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               #'OBJID': objid,
               'MOCKID': mockid, 'LYAFILES': np.array(lyafiles),
               'BRICKNAME': brickname, 'RA': ra, 'DEC': dec, 'Z': zz}

        # Add MW transmission and the imaging depth.
        if self.dust_dir:
            mw_transmission(out, dust_dir=self.dust_dir)
            imaging_depth(out)

        # Optionally compute the mean mock density.
        if mock_density:
            out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile)

        return out

class ReadMXXL(SelectTargets):
    """Read a MXXL mock catalog of BGS targets.

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    cached_radec = None

    def __init__(self, dust_dir=None, bricksize=0.25):
        super(ReadMXXL, self).__init__()

        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 target_name='BGS', magcut=None, only_coords=False):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the top-level directory of the CoLoRe mock catalog.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        target_name : :class:`str`
            Name of the target being read (if not BGS).
        magcut : :class:`float`
            Magnitude cut (hard-coded to SDSS r-band) to subselect targets
            brighter than magcut. 
        only_coords : :class:`bool`, optional
            To get some improvement in speed, only read the target coordinates
            and some other basic info.

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data file is not found.
        ValueError
            If mockfile is not defined or if nside is not a scalar.

        """
        import h5py
        
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError
        
        try:
            mockfile = mockfile.format(**os.environ)
        except KeyError as e:
            log.warning('Environment variable not set for mockfile: {}'.format(e))
            raise ValueError
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        # Default set of healpixels is the whole DESI footprint.
        if healpixels is None:
            if nside is None:
                nside = 16
            log.info('Reading the whole DESI footprint with nside = {}.'.format(nside))
            healpixels = footprint.tiles2pix(nside)

        if nside is None:
            log.warning('Nside must be a scalar input.')
            raise ValueError

        # Read the data, generate mockid, and then restrict to the input
        # healpixel.
        def _read_mockfile(mockfile, nside, pixmap):
            # Work around hdf5 <1.10 bug on /project; see
            # http://www.nersc.gov/users/data-analytics/data-management/i-o-libraries/hdf5-2/h5py/
            hdf5_flock = os.getenv('HDF5_USE_FILE_LOCKING')
            os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
            with h5py.File(mockfile, mode='r') as f:
                ra  = f['Data/ra'][:].astype('f8') % 360.0 # enforce 0 < ra < 360
                dec = f['Data/dec'][:].astype('f8')
                zz = f['Data/z_obs'][:].astype('f4')
                rmag = f['Data/app_mag'][:].astype('f4')
                absmag = f['Data/abs_mag'][:].astype('f4')
                gr = f['Data/g_r'][:].astype('f4')

            if hdf5_flock is not None:
                os.environ['HDF5_USE_FILE_LOCKING'] = hdf5_flock
            else:
                del os.environ['HDF5_USE_FILE_LOCKING']

            log.info('Assigning healpix pixels with nside = {}'.format(nside))
            allpix = footprint.radec2pix(nside, ra, dec)

            pixweight = load_pixweight(nside, pixmap=pixmap)
        
            return ra, dec, zz, rmag, absmag, gr, allpix, pixweight

        # Read the ra,dec coordinates, pixel weight map, generate mockid, and
        # then restrict to the desired healpixels.
        if self.cached_radec is None:
            ra, dec, zz, rmag, absmag, gr, allpix, pixweight = _read_mockfile(mockfile, nside, self.pixmap)
            ReadMXXL.cached_radec = (mockfile, nside, ra, dec, zz, rmag, absmag, gr, allpix, pixweight)
        else:
            cached_mockfile, cached_nside, ra, dec, zz, rmag, absmag, gr, allpix, pixweight = ReadMXXL.cached_radec
            if cached_mockfile != mockfile or cached_nside != nside:
                ra, dec, zz, rmag, absmag, gr, allpix, pixweight = _read_mockfile(mockfile, nside, self.pixmap)
                ReadMXXL.cached_radec = (mockfile, nside, ra, dec, zz, rmag, absmag, gr, allpix, pixweight)
            else:
                log.info('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, zz, rmag, absmag, gr, allpix, pixweight = ReadMXXL.cached_radec

        mockid = np.arange(len(ra)) # unique ID/row number
        
        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s).'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        mockid = mockid[cut]
        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = ra[cut]
        dec = dec[cut]
        zz = zz[cut]
        rmag = rmag[cut]
        absmag = absmag[cut]
        gr = gr[cut]

        if magcut:
            cut = rmag < magcut
            if np.count_nonzero(cut) == 0:
                log.warning('No objects with r < {}!'.format(magcut))
                return dict()
            else:
                mockid = mockid[cut]
                allpix = allpix[cut]
                weight = weight[cut]
                ra = ra[cut]
                dec = dec[cut]
                zz = zz[cut]
                rmag = rmag[cut]
                absmag = absmag[cut]
                gr = gr[cut]
                nobj = len(ra)
                log.info('Trimmed to {} {}s with r < {}.'.format(nobj, target_name, magcut))

        # Optionally (for a little more speed) only return some basic info. 
        if only_coords:
            return {'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'Z': zz,
                    'MAG': rmag, 'WEIGHT': weight, 'NSIDE': nside}

        # Assign bricknames.
        brickname = get_brickname_from_radec(ra, dec, bricksize=self.bricksize)

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'durham_mxxl_hdf5',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': rmag, 'SDSS_absmag_r01': absmag,
               'SDSS_01gr': gr, 'NORMFILTER': 'sdss2010-r'}

        # Add MW transmission and the imaging depth.
        if self.dust_dir:
            mw_transmission(out, dust_dir=self.dust_dir)
            imaging_depth(out)

        return out

class ReadGAMA(SelectTargets):
    """Read a GAMA catalog of BGS targets.  This reader will only generally be used
    for the Survey Validation Data Challenge.

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    cached_radec = None
    
    def __init__(self, dust_dir=None, bricksize=0.25):
        super(ReadGAMA, self).__init__()
        
        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 target_name='', magcut=None, only_coords=False):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        target_name : :class:`str`
            Name of the target being read (e.g., ELG, LRG).
        magcut : :class:`float`
            Magnitude cut (hard-coded to SDSS r-band) to subselect targets
            brighter than magcut. 
        only_coords : :class:`bool`, optional
            To get some improvement in speed, only read the target coordinates
            and some other basic info.

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data file is not found.
        ValueError
            If mockfile or healpixels are not defined, or if nside is not a
            scalar.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError

        try:
            mockfile = mockfile.format(**os.environ)
        except KeyError as e:
            log.warning('Environment variable not set for mockfile: {}'.format(e))
            raise ValueError
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        # Require healpixels, or could pass the set of tiles and use
        # footprint.tiles2pix() to convert to healpixels given nside.
        if healpixels is None:
            log.warning('Healpixels input is required.') 
            raise ValueError
        
        if nside is None:
            log.warning('Nside must be a scalar input.')
            raise ValueError

        # Read the ra,dec coordinates, pixel weight map, generate mockid, and
        # then restrict to the desired healpixels.
        if self.cached_radec is None:
            ra, dec, allpix, pixweight = _get_radec(mockfile, nside, self.pixmap)
            ReadGAMA.cached_radec = (mockfile, nside, ra, dec, allpix, pixweight)
        else:
            cached_mockfile, cached_nside, ra, dec, allpix, pixweight = ReadGAMA.cached_radec
            if cached_mockfile != mockfile or cached_nside != nside:
                ra, dec, allpix, pixweight = _get_radec(mockfile, nside, self.pixmap)
                ReadGAMA.cached_radec = (mockfile, nside, ra, dec, allpix, pixweight)
            else:
                log.info('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, allpix, pixweight = ReadGAMA.cached_radec

        mockid = np.arange(len(ra)) # unique ID/row number
        
        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s).'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        mockid = mockid[cut]
        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = ra[cut]
        dec = dec[cut]

        # Assign bricknames.
        brickname = get_brickname_from_radec(ra, dec, bricksize=self.bricksize)

        # Add photometry, absolute magnitudes, and redshifts.
        columns = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'Z', 'UGRIZ_ABSMAG_01']
        data = fitsio.read(mockfile, columns=columns, upper=True, ext=1, rows=cut)
        zz = data['Z'].astype('f4')
        rmag = 22.5 - 2.5 * np.log10(data['FLUX_R']).astype('f4')

        # Pack into a basic dictionary.  Could include shapes and other spectral
        # properties here.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'bgs-gama',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz, 'RMABS_01': data['UGRIZ_ABSMAG_01'][:, 2],
               'UG_01': data['UGRIZ_ABSMAG_01'][:, 0]-data['UGRIZ_ABSMAG_01'][:, 1],
               'GR_01': data['UGRIZ_ABSMAG_01'][:, 1]-data['UGRIZ_ABSMAG_01'][:, 2],
               'RI_01': data['UGRIZ_ABSMAG_01'][:, 2]-data['UGRIZ_ABSMAG_01'][:, 3],
               'IZ_01': data['UGRIZ_ABSMAG_01'][:, 3]-data['UGRIZ_ABSMAG_01'][:, 4],
               'NORMFILTER': 'decam2014-r', 'MAG': rmag}

        # Add MW transmission and the imaging depth.
        if self.dust_dir:
            mw_transmission(out, dust_dir=self.dust_dir)
            imaging_depth(out)

        return out

class ReadMWS_WD(SelectTargets):
    """Read a mock catalog of Milky Way Survey white dwarf targets (MWS_WD). 

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    cached_radec = None

    def __init__(self, dust_dir=None, bricksize=0.25):
        super(ReadMWS_WD, self).__init__()

        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 target_name='WD', mock_density=False):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        target_name : :class:`str`
            Name of the target being read (if not WD).
        mock_density : :class:`bool`, optional
            Compute and return the median target density in the mock.  Defaults
            to False.

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data file is not found.
        ValueError
            If mockfile is not defined or if nside is not a scalar.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError
        
        try:
            mockfile = mockfile.format(**os.environ)
        except KeyError as e:
            log.warning('Environment variable not set for mockfile: {}'.format(e))
            raise ValueError
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        # Default set of healpixels is the whole DESI footprint.
        if healpixels is None:
            if nside is None:
                nside = 16
            log.info('Reading the whole DESI footprint with nside = {}.'.format(nside))
            healpixels = footprint.tiles2pix(nside)

        if nside is None:
            log.warning('Nside must be a scalar input.')
            raise ValueError

        # Read the ra,dec coordinates, pixel weight map, generate mockid, and
        # then restrict to the desired healpixels.
        if self.cached_radec is None:
            ra, dec, allpix, pixweight = _get_radec(mockfile, nside, self.pixmap)
            ReadMWS_WD.cached_radec = (mockfile, nside, ra, dec, allpix, pixweight)
        else:
            cached_mockfile, cached_nside, ra, dec, allpix, pixweight = ReadMWS_WD.cached_radec
            if cached_mockfile != mockfile or cached_nside != nside:
                ra, dec, allpix, pixweight = _get_radec(mockfile, nside, self.pixmap)
                ReadMWS_WD.cached_radec = (mockfile, nside, ra, dec, allpix, pixweight)
            else:
                log.info('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, allpix, pixweight = ReadMWS_WD.cached_radec

        mockid = np.arange(len(ra)) # unique ID/row number

        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s).'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        mockid = mockid[cut]
        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = ra[cut]
        dec = dec[cut]

        cols = ['RADIALVELOCITY', 'G_SDSS', 'TEFF', 'LOGG', 'SPECTRALTYPE']
        data = fitsio.read(mockfile, columns=cols, upper=True, ext=1, rows=cut)
        zz = (data['RADIALVELOCITY'] / C_LIGHT).astype('f4')
        mag = data['G_SDSS'].astype('f4') # SDSS g-band
        teff = data['TEFF'].astype('f4')
        logg = data['LOGG'].astype('f4')
        templatesubtype = np.char.upper(data['SPECTRALTYPE'].astype('<U'))

        # Assign bricknames.
        brickname = get_brickname_from_radec(ra, dec, bricksize=self.bricksize)

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'mws_wd',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': mag, 'TEFF': teff, 'LOGG': logg,
               'NORMFILTER': 'sdss2010-g', 'TEMPLATESUBTYPE': templatesubtype}

        # Add MW transmission and the imaging depth.
        if self.dust_dir:
            mw_transmission(out, dust_dir=self.dust_dir)
            imaging_depth(out)

        # Optionally compute the mean mock density.
        if mock_density:
            out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile)

        return out
    
class ReadMWS_NEARBY(SelectTargets):
    """Read a mock catalog of Milky Way Survey nearby targets (MWS_NEARBY). 

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    cached_radec = None
    
    def __init__(self, dust_dir=None, bricksize=0.25):
        super(ReadMWS_NEARBY, self).__init__()

        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 target_name='MWS_NEARBY', mock_density=False):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        target_name : :class:`str`
            Name of the target being read (if not MWS_NEARBY).
        mock_density : :class:`bool`, optional
            Compute and return the median target density in the mock.  Defaults
            to False.

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data file is not found.
        ValueError
            If mockfile is not defined or if nside is not a scalar.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError

        try:
            mockfile = mockfile.format(**os.environ)
        except KeyError as e:
            log.warning('Environment variable not set for mockfile: {}'.format(e))
            raise ValueError
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        # Default set of healpixels is the whole DESI footprint.
        if healpixels is None:
            if nside is None:
                nside = 16
            log.info('Reading the whole DESI footprint with nside = {}.'.format(nside))
            healpixels = footprint.tiles2pix(nside)

        if nside is None:
            log.warning('Nside must be a scalar input.')
            raise ValueError

        # Read the ra,dec coordinates, pixel weight map, generate mockid, and
        # then restrict to the desired healpixels.
        if self.cached_radec is None:
            ra, dec, allpix, pixweight = _get_radec(mockfile, nside, self.pixmap)
            ReadMWS_NEARBY.cached_radec = (mockfile, nside, ra, dec, allpix, pixweight)
        else:
            cached_mockfile, cached_nside, ra, dec, allpix, pixweight = ReadMWS_NEARBY.cached_radec
            if cached_mockfile != mockfile or cached_nside != nside:
                ra, dec, allpix, pixweight = _get_radec(mockfile, nside, self.pixmap)
                ReadMWS_NEARBY.cached_radec = (mockfile, nside, ra, dec, allpix, pixweight)
            else:
                log.info('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, allpix, pixweight = ReadMWS_NEARBY.cached_radec
        
        mockid = np.arange(len(ra)) # unique ID/row number

        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s).'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        mockid = mockid[cut]
        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = ra[cut]
        dec = dec[cut]

        cols = ['RADIALVELOCITY', 'MAGG', 'TEFF', 'LOGG', 'FEH', 'SPECTRALTYPE']
        data = fitsio.read(mockfile, columns=cols, upper=True, ext=1, rows=cut)
        zz = (data['RADIALVELOCITY'] / C_LIGHT).astype('f4')
        mag = data['MAGG'].astype('f4') # SDSS g-band
        teff = data['TEFF'].astype('f4')
        logg = data['LOGG'].astype('f4')
        feh = data['FEH'].astype('f4')
        templatesubtype = data['SPECTRALTYPE']

        # Assign bricknames.
        brickname = get_brickname_from_radec(ra, dec, bricksize=self.bricksize)

        # Pack into a basic dictionary.  Is the normalization filter g-band???
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'mws_100pc',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': mag, 'TEFF': teff, 'LOGG': logg, 'FEH': feh,
               'NORMFILTER': 'sdss2010-g', 'TEMPLATESUBTYPE': templatesubtype}

        # Add MW transmission and the imaging depth.
        if self.dust_dir:
            mw_transmission(out, dust_dir=self.dust_dir)
            imaging_depth(out)

        # Optionally compute the mean mock density.
        if mock_density:
            out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile)
            
        return out

class QSOMaker(SelectTargets):
    """Read QSO mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        each target.  Defaults to `decam2014-g`.

    """
    wave, template_maker, GMM_nospectra = None, None, None
    
    def __init__(self, seed=None, normfilter='decam2014-g', **kwargs):
        from desisim.templates import SIMQSO
        from desiutil.sklearn import GaussianMixtureModel

        super(QSOMaker, self).__init__()

        self.seed = seed
        self.objtype = 'QSO'

        if self.wave is None:
            QSOMaker.wave = _default_wave()

        if self.template_maker is None:
            QSOMaker.template_maker = SIMQSO(wave=self.wave, normfilter=normfilter)

        if self.GMM_nospectra is None:
            gmmfile = resource_filename('desitarget', 'mock/data/quicksurvey_gmm_qso.fits')
            QSOMaker.GMM_nospectra = GaussianMixtureModel.load(gmmfile)
            
        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'GaussianRandomField',
                                             'v0.0.7_2LPT', 'QSO.fits')

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=None, nside=None, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'gaussianfield'.
        dust_dir : :class:`str`
            Full path to the dust maps.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        mock_density : :class:`bool`, optional
            Compute the median target density in the mock.  Defaults to False.

        Returns
        -------
        data : :class:`dict`
            Dictionary of target properties with various keys (to be documented). 

        Raises
        ------
        ValueError
            If mockformat is not recognized.

        """
        self.mockformat = mockformat.lower()
        
        if self.mockformat == 'gaussianfield':
            MockReader = ReadGaussianField(dust_dir=dust_dir)
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile
            
        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   mock_density=mock_density)
        self._update_normfilter(data.get('NORMFILTER'))

        return data

    def make_spectra(self, data=None, indx=None, seed=None, no_spectra=False):
        """Generate tracer QSO spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        no_spectra : :class:`bool`, optional
            Do not generate spectra.  Defaults to False.

        Returns
        -------
        flux : :class:`numpy.ndarray`
            Target spectra.
        wave : :class:`numpy.ndarray`
            Corresponding wavelength array.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        
        """
        if seed is None:
            seed = self.seed

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)
            
        if no_spectra:
            from desisim.io import empty_metatable
            rand = np.random.RandomState(seed)
            
            flux = []
            meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            meta['SEED'] = rand.randint(2**31, size=nobj)
            meta['REDSHIFT'] = data['Z'][indx]
            self.sample_gmm_nospectra(meta, rand=rand) # noiseless photometry from pre-computed GMMs
        else:
            flux, wave, meta = self.template_maker.make_templates(
                nmodel=nobj, redshift=data['Z'][indx], seed=seed,
                lyaforest=False, nocolorcuts=True)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx,
                                                     psf=True, seed=seed,
                                                     truespectype='QSO',
                                                     templatetype='QSO')

        return flux, self.wave, meta, targets, truth

    def select_targets(self, targets, truth, **kwargs):
        """Select QSO targets.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        from desitarget.cuts import isQSO_colors
          
        gflux, rflux, zflux, w1flux, w2flux = self.deredden(targets)
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux)

        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH

class LYAMaker(SelectTargets):
    """Read LYA mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        each target.  Defaults to `decam2014-g`.

    """
    wave, template_maker, GMM_nospectra = None, None, None

    def __init__(self, seed=None, normfilter='decam2014-g', **kwargs):
        from desisim.templates import SIMQSO
        from desiutil.sklearn import GaussianMixtureModel

        super(LYAMaker, self).__init__()

        self.seed = seed
        self.objtype = 'LYA'

        if self.wave is None:
            LYAMaker.wave = _default_wave()
            
        if self.template_maker is None:
            LYAMaker.template_maker = SIMQSO(wave=self.wave, normfilter=normfilter)

        if self.GMM_nospectra is None:
            gmmfile = resource_filename('desitarget', 'mock/data/quicksurvey_gmm_lya.fits')
            LYAMaker.GMM_nospectra = GaussianMixtureModel.load(gmmfile)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'lya_forest', 'v2.0.2', 'master.fits')

    def read(self, mockfile=None, mockformat='CoLoRe', dust_dir=None,
             healpixels=None, nside=None, nside_lya=16, mock_density=False,
             **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'CoLoRe'.
        dust_dir : :class:`str`
            Full path to the dust maps.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        nside_lya : :class:`int`
            Healpixel nside indicating how the mock on-disk has been organized.
            Defaults to 16.
        mock_density : :class:`bool`, optional
            Compute the median target density in the mock.  Defaults to False.

        Returns
        -------
        :class:`dict`
            Dictionary of target properties with various keys (to be documented). 

        Raises
        ------
        ValueError
            If mockformat is not recognized.

        """
        self.mockformat = mockformat.lower()
        
        if self.mockformat == 'colore':
            MockReader = ReadLyaCoLoRe(dust_dir=dust_dir)
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   nside_lya=nside_lya, mock_density=mock_density)
        self._update_normfilter(data.get('NORMFILTER'))

        return data

    def make_spectra(self, data=None, indx=None, seed=None, no_spectra=False):
        """Generate QSO spectra with the 3D Lya forest skewers included. 

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        no_spectra : :class:`bool`, optional
            Do not generate spectra.  Defaults to False.

        Returns
        -------
        flux : :class:`numpy.ndarray`
            Target spectra.
        wave : :class:`numpy.ndarray`
            Corresponding wavelength array.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        Raises
        ------
        KeyError
            If there is a mismatch between MOCKID in the data dictionary and the
            skewer files on-disk.

        """
        import numpy.ma as ma
        from desispec.interpolation import resample_flux
        from desisim.lya_spectra import read_lya_skewers, apply_lya_transmission
        
        if seed is None:
            seed = self.seed
            
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if no_spectra:
            from desisim.io import empty_metatable
            rand = np.random.RandomState(seed)
            
            flux = []
            meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            meta['SEED'] = rand.randint(2**31, size=nobj)
            meta['REDSHIFT'] = data['Z'][indx]
            self.sample_gmm_nospectra(meta, rand=rand) # noiseless photometry from pre-computed GMMs
        else:
            # Read skewers.
            skewer_wave = None
            skewer_trans = None
            skewer_meta = None

            # All the files that contain at least one QSO skewer.
            alllyafile = data['LYAFILES'][indx]
            uniquelyafiles = sorted(set(alllyafile))

            for lyafile in uniquelyafiles:
                these = np.where( alllyafile == lyafile )[0]

                mockid_in_data = data['MOCKID'][indx][these]
                mockid_in_mock = (fitsio.read(lyafile, columns=['MOCKID'], upper=True,
                                             ext=1).astype(float)).astype(int)
                o2i = dict()
                for i, o in enumerate(mockid_in_mock):
                    o2i[o] = i
                indices_in_mock_healpix = np.zeros(mockid_in_data.size).astype(int)
                for i, o in enumerate(mockid_in_data):
                    if not o in o2i:
                        log.warning("No MOCKID={} in {}. It's a bug, should never happen".format(o, lyafile))
                        raise KeyError

                    indices_in_mock_healpix[i] = o2i[o]

                tmp_wave, tmp_trans, tmp_meta = read_lya_skewers(
                    lyafile, indices=indices_in_mock_healpix) 

                if skewer_wave is None:
                    skewer_wave = tmp_wave
                    dw = skewer_wave[1] - skewer_wave[0] # this is just to check same wavelength
                    skewer_trans = np.zeros((nobj, skewer_wave.size)) # allocate skewer_array
                    skewer_meta = dict()
                    for k in tmp_meta.dtype.names:
                        skewer_meta[k] = np.zeros(nobj).astype(tmp_meta[k].dtype)
                else :
                    # check wavelength is the same for all skewers
                    assert( np.max(np.abs(wave-tmp_wave)) < 0.001*dw )

                skewer_trans[these] = tmp_trans
                for k in skewer_meta.keys():
                    skewer_meta[k][these] = tmp_meta[k]

            # Check we matched things correctly.
            assert(np.max(np.abs(skewer_meta['Z']-data['Z'][indx]))<0.000001)
            assert(np.max(np.abs(skewer_meta['RA']-data['RA'][indx]))<0.000001)
            assert(np.max(np.abs(skewer_meta['DEC']-data['DEC'][indx]))<0.000001)

            # Now generate the QSO spectra simultaneously **at full wavelength
            # resolution**.  We do this because the Lya forest (and DLAs) will have
            # changed the colors, so we need to re-synthesize the photometry.
            qso_flux, qso_wave, meta = self.template_maker.make_templates(
                nmodel=nobj, redshift=data['Z'][indx], seed=seed,
                lyaforest=False, nocolorcuts=True, noresample=True)
            meta['SUBTYPE'] = 'LYA'

            # Apply the Lya forest transmission.
            _flux = apply_lya_transmission(qso_wave, qso_flux, skewer_wave, skewer_trans)

            # Add DLAs (ToDo).
            # ...

            # Update the photometry
            maggies = self.template_maker.decamwise.get_ab_maggies(
                1e-17 * _flux, qso_wave.copy(), mask_invalid=True)
            for band, filt in zip( ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'),
                                   ('decam2014-g', 'decam2014-r', 'decam2014-z',
                                    'wise2010-W1', 'wise2010-W2') ):
                meta[band] = ma.getdata(1e9 * maggies[filt]) # nanomaggies

            # Unfortunately, to resample to the desired output wavelength vector we
            # need to loop.
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')
            for ii in range(nobj):
                flux[ii, :] = resample_flux(self.wave, qso_wave, _flux[ii, :],
                                            extrapolate=True)
                                     
        targets, truth = self.populate_targets_truth(data, meta, indx=indx,
                                                     psf=True,seed=seed,
                                                     truespectype='QSO',
                                                     templatetype='QSO',
                                                     templatesubtype='LYA')

        return flux, self.wave, meta, targets, truth

    def select_targets(self, targets, truth, **kwargs):
        """Select Lya/QSO targets.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        from desitarget.cuts import isQSO_colors

        gflux, rflux, zflux, w1flux, w2flux = self.deredden(targets)
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux)

        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH

class LRGMaker(SelectTargets):
    """Read LRG mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    nside_chunk : :class:`int`, optional
        Healpixel nside for further subdividing the sample when assigning
        velocity dispersion to targets.  Defaults to 128.
    normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        each target.  Defaults to `decam2014-z`.

    """
    wave, tree, template_maker = None, None, None
    GMM, GMM_nospectra = None, None
    
    def __init__(self, seed=None, nside_chunk=128, normfilter='decam2014-z', **kwargs):
        from scipy.spatial import cKDTree as KDTree
        from desisim.templates import LRG
        from desiutil.sklearn import GaussianMixtureModel

        super(LRGMaker, self).__init__()

        self.seed = seed
        self.nside_chunk = nside_chunk
        self.objtype = 'LRG'

        if self.wave is None:
            LRGMaker.wave = _default_wave()
        if self.template_maker is None:
            LRGMaker.template_maker = LRG(wave=self.wave, normfilter=normfilter)
            
        self.meta = self.template_maker.basemeta

        if self.tree is None:
            zobj = self.meta['Z'].data
            LRGMaker.tree = KDTree(np.vstack((zobj)).T)

        if self.GMM is None:
            gmmfile = resource_filename('desitarget', 'mock/data/dr2/lrg_gmm.fits')
            LRGMaker.GMM = GaussianMixtureModel.load(gmmfile)

        if self.GMM_nospectra is None:
            gmmfile = resource_filename('desitarget', 'mock/data/quicksurvey_gmm_lrg.fits')
            LRGMaker.GMM_nospectra = GaussianMixtureModel.load(gmmfile)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'GaussianRandomField',
                                             'v0.0.7_2LPT', 'LRG.fits')

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=None, nside=None, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'gaussianfield'.
        dust_dir : :class:`str`
            Full path to the dust maps.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        mock_density : :class:`bool`, optional
            Compute the median target density in the mock.  Defaults to False.

        Returns
        -------
        :class:`dict`
            Dictionary of target properties with various keys (to be documented). 

        Raises
        ------
        ValueError
            If mockformat is not recognized.

        """
        self.mockformat = mockformat.lower()
        if self.mockformat == 'gaussianfield':
            MockReader = ReadGaussianField(dust_dir=dust_dir)
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   mock_density=mock_density)
        self._update_normfilter(data.get('NORMFILTER'))

        return data

    def _GMMsample(self, nsample=1, seed=None):
        """Sample from the Gaussian mixture model (GMM) for LRGs."""

        rand = np.random.RandomState(seed)
        params = self.GMM.sample(nsample, rand).astype('f4')

        tags = ('g', 'r', 'z', 'w1', 'w2', 'w3', 'w4')
        tags = tags + ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2')

        samp = np.empty( nsample, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]
            
        return samp

    def _query(self, matrix):
        """Return the nearest template number based on the KD Tree."""
        dist, indx = self.tree.query(matrix)
        return dist, indx
    
    def make_spectra(self, data=None, indx=None, seed=None, no_spectra=False):
        """Generate LRG spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        no_spectra : :class:`bool`, optional
            Do not generate spectra.  Defaults to False.

        Returns
        -------
        flux : :class:`numpy.ndarray`
            Target spectra.
        wave : :class:`numpy.ndarray`
            Corresponding wavelength array.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        
        """
        from desisim.io import empty_metatable

        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)
        
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        gmm = self._GMMsample(nobj, seed=seed)

        if no_spectra:
            flux = []
            meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            meta['SEED'] = rand.randint(2**31, size=nobj)
            meta['REDSHIFT'] = data['Z'][indx]
            self.sample_gmm_nospectra(meta, rand=rand) # noiseless photometry from pre-computed GMMs
        else:
            input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            input_meta['SEED'] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'] = data['Z'][indx]
            input_meta['VDISP'] = self._sample_vdisp(data['RA'][indx], data['DEC'][indx],
                                                     mean=2.3, sigma=0.1, seed=seed,
                                                     nside=self.nside_chunk)

            input_meta['MAG'] = gmm['z']
            if self.template_maker.normfilter != 'decam2014-z':
                log.warning('Mismatching normalization filter!  Expecting {} but have {}'.format(
                    'decam2014-z', self.template_maker.normfilter))
                raise ValueError

            if self.mockformat == 'gaussianfield':
                # This is not quite right, but choose a template with equal probability.
                templateid = rand.choice(self.meta['TEMPLATEID'], nobj)
                input_meta['TEMPLATEID'] = templateid

            flux, _, meta = self.template_maker.make_templates(input_meta=input_meta,
                                                               nocolorcuts=True, novdisp=False)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=False,
                                                     seed=seed, gmm=gmm,
                                                     truespectype='GALAXY',
                                                     templatetype='LRG')

        return flux, self.wave, meta, targets, truth

    def select_targets(self, targets, truth, **kwargs):
        """Select LRG targets.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        from desitarget.cuts import isLRG_colors

        gflux, rflux, zflux, w1flux, w2flux = self.deredden(targets)
        lrg = isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux)

        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH

class ELGMaker(SelectTargets):
    """Read ELG mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    nside_chunk : :class:`int`, optional
        Healpixel nside for further subdividing the sample when assigning
        velocity dispersion to targets.  Defaults to 128.
    normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        each target.  Defaults to `decam2014-z`.

    """
    wave, tree, template_maker = None, None, None
    GMM, GMM_nospectra = None, None
    
    def __init__(self, seed=None, nside_chunk=128, normfilter='decam2014-r', **kwargs):
        from scipy.spatial import cKDTree as KDTree
        from desisim.templates import ELG
        from desiutil.sklearn import GaussianMixtureModel

        super(ELGMaker, self).__init__()

        self.seed = seed
        self.nside_chunk = nside_chunk
        self.objtype = 'ELG'

        if self.wave is None:
            ELGMaker.wave = _default_wave()
        if self.template_maker is None:
            ELGMaker.template_maker = ELG(wave=self.wave, normfilter=normfilter)
            
        self.meta = self.template_maker.basemeta

        if self.tree is None:
            zobj = self.meta['Z'].data
            gr = self.meta['DECAM_G'].data - self.meta['DECAM_R'].data
            rz = self.meta['DECAM_R'].data - self.meta['DECAM_Z'].data
            ELGMaker.tree = KDTree(np.vstack((zobj, gr, rz)).T)

        if self.GMM is None:
            gmmfile = resource_filename('desitarget', 'mock/data/dr2/elg_gmm.fits')
            ELGMaker.GMM = GaussianMixtureModel.load(gmmfile)

        if self.GMM_nospectra is None:
            gmmfile = resource_filename('desitarget', 'mock/data/quicksurvey_gmm_elg.fits')
            ELGMaker.GMM_nospectra = GaussianMixtureModel.load(gmmfile)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'GaussianRandomField',
                                             'v0.0.7_2LPT', 'ELG.fits')

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=None, nside=None, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'gaussianfield'.
        dust_dir : :class:`str`
            Full path to the dust maps.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        mock_density : :class:`bool`, optional
            Compute the median target density in the mock.  Defaults to False.

        Returns
        -------
        :class:`dict`
            Dictionary of target properties with various keys (to be documented). 

        Raises
        ------
        ValueError
            If mockformat is not recognized.

        """
        self.mockformat = mockformat.lower()
        if self.mockformat == 'gaussianfield':
            MockReader = ReadGaussianField(dust_dir=dust_dir)
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   mock_density=mock_density)
        self._update_normfilter(data.get('NORMFILTER'))

        return data
            
    def _GMMsample(self, nsample=1, seed=None):
        """Sample from the Gaussian mixture model (GMM) for ELGs."""

        rand = np.random.RandomState(seed)
        params = self.GMM.sample(nsample, rand).astype('f4')

        tags = ('g', 'r', 'z', 'w1', 'w2', 'w3', 'w4')
        tags = tags + ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2')

        samp = np.empty( nsample, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]
            
        return samp

    def _query(self, matrix):
        """Return the nearest template number based on the KD Tree."""
        dist, indx = self.tree.query(matrix)
        return dist, indx
    
    def make_spectra(self, data=None, indx=None, seed=None, no_spectra=False):
        """Generate ELG spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        no_spectra : :class:`bool`, optional
            Do not generate spectra.  Defaults to False.

        Returns
        -------
        flux : :class:`numpy.ndarray`
            Target spectra.
        wave : :class:`numpy.ndarray`
            Corresponding wavelength array.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        
        """
        from desisim.io import empty_metatable
        
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        gmm = self._GMMsample(nobj, seed=seed)

        if no_spectra:
            flux = []
            meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            meta['SEED'] = rand.randint(2**31, size=nobj)
            meta['REDSHIFT'] = data['Z'][indx]
            self.sample_gmm_nospectra(meta, rand=rand) # noiseless photometry from pre-computed GMMs
        else:
            input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            input_meta['SEED'] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'] = data['Z'][indx]
            input_meta['VDISP'] = self._sample_vdisp(data['RA'][indx], data['DEC'][indx],
                                                     mean=1.9, sigma=0.15, seed=seed,
                                                     nside=self.nside_chunk)
            input_meta['MAG'] = gmm['r']
            if self.template_maker.normfilter != 'decam2014-r':
                log.warning('Mismatching normalization filter!  Expecting {} but have {}'.format(
                    'decam2014-z', self.template_maker.normfilter))
                raise ValueError

            if self.mockformat == 'gaussianfield':
                alldata = np.vstack((data['Z'][indx],
                                     gmm['g']-gmm['r'],
                                     gmm['r']-gmm['z'])).T
                _, templateid = self._query(alldata)
                input_meta['TEMPLATEID'] = templateid

            flux, _, meta = self.template_maker.make_templates(input_meta=input_meta,
                                                               nocolorcuts=True, novdisp=False)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=False,
                                                     seed=seed, gmm=gmm,
                                                     truespectype='GALAXY',
                                                     templatetype='ELG')

        return flux, self.wave, meta, targets, truth

    def select_targets(self, targets, truth, **kwargs):
        """Select ELG targets.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        from desitarget.cuts import isELG

        gflux, rflux, zflux, w1flux, w2flux = self.deredden(targets)
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)

        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH

class BGSMaker(SelectTargets):
    """Read BGS mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    nside_chunk : :class:`int`, optional
        Healpixel nside for further subdividing the sample when assigning
        velocity dispersion to targets.  Defaults to 128.
    normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        each target.  Defaults to `decam2014-r`.

    """
    wave, tree, template_maker = None, None, None
    GMM, GMM_nospectra = None, None
    
    def __init__(self, seed=None, nside_chunk=128, normfilter='decam2014-r', **kwargs):
        from scipy.spatial import cKDTree as KDTree
        from desisim.templates import BGS
        from desiutil.sklearn import GaussianMixtureModel

        super(BGSMaker, self).__init__()

        self.seed = seed
        self.nside_chunk = nside_chunk
        self.objtype = 'BGS'

        if self.wave is None:
            BGSMaker.wave = _default_wave()
        if self.template_maker is None:
            BGSMaker.template_maker = BGS(wave=self.wave, normfilter=normfilter)
            
        self.meta = self.template_maker.basemeta

        if self.tree is None:
            zobj = self.meta['Z'].data
            mabs = self.meta['SDSS_UGRIZ_ABSMAG_Z01'].data
            rmabs = mabs[:, 2]
            gr = mabs[:, 1] - mabs[:, 2]
            BGSMaker.tree = KDTree(np.vstack((zobj, rmabs, gr)).T)

        if self.GMM is None:
            gmmfile = resource_filename('desitarget', 'mock/data/dr2/bgs_gmm.fits')
            BGSMaker.GMM = GaussianMixtureModel.load(gmmfile)

        if self.GMM_nospectra is None:
            gmmfile = resource_filename('desitarget', 'mock/data/quicksurvey_gmm_bgs.fits')
            BGSMaker.GMM_nospectra = GaussianMixtureModel.load(gmmfile)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'bgs', 'MXXL', 'desi_footprint',
                                             'v0.0.4', 'BGS.hdf5')

    def read(self, mockfile=None, mockformat='durham_mxxl_hdf5', dust_dir=None,
             healpixels=None, nside=None, magcut=None, only_coords=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'durham_mxxl_hdf5'.
        dust_dir : :class:`str`
            Full path to the dust maps.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        magcut : :class:`float`
            Magnitude cut (hard-coded to SDSS r-band) to subselect targets
            brighter than magcut. 
        only_coords : :class:`bool`, optional
            For various applications, only read the target coordinates.

        Returns
        -------
        :class:`dict`
            Dictionary of target properties with various keys (to be documented). 

        Raises
        ------
        ValueError
            If mockformat is not recognized.

        """
        self.mockformat = mockformat.lower()
        if self.mockformat == 'durham_mxxl_hdf5':
            MockReader = ReadMXXL(dust_dir=dust_dir)
        elif self.mockformat == 'bgs-gama':
            MockReader = ReadGAMA(dust_dir=dust_dir)
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile
            
        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   magcut=magcut, only_coords=only_coords)
        self._update_normfilter(data.get('NORMFILTER'))

        return data

    def _GMMsample(self, nsample=1, seed=None):
        """Sample from the Gaussian mixture model (GMM) for BGS."""

        rand = np.random.RandomState(seed)
        params = self.GMM.sample(nsample, rand).astype('f4')

        tags = ('g', 'r', 'z', 'w1', 'w2', 'w3', 'w4')
        tags = tags + ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2')

        samp = np.empty( nsample, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]
            
        return samp

    def _query(self, matrix):
        """Return the nearest template number based on the KD Tree."""
        dist, indx = self.tree.query(matrix)
        return dist, indx
    
    def make_spectra(self, data=None, indx=None, seed=None, no_spectra=False):
        """Generate BGS spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        no_spectra : :class:`bool`, optional
            Do not generate spectra.  Defaults to False.

        Returns
        -------
        flux : :class:`numpy.ndarray`
            Target spectra.
        wave : :class:`numpy.ndarray`
            Corresponding wavelength array.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        
        """
        from desisim.io import empty_metatable
        
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        gmm = self._GMMsample(nobj, seed=seed)

        if no_spectra:
            flux = []
            meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            meta['SEED'] = rand.randint(2**31, size=nobj)
            meta['REDSHIFT'] = data['Z'][indx]
            self.sample_gmm_nospectra(meta, rand=rand) # noiseless photometry from pre-computed GMMs
        else:
            input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            input_meta['SEED'] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'] = data['Z'][indx]
            input_meta['MAG'] = data['MAG'][indx]
            input_meta['VDISP'] = self._sample_vdisp(data['RA'][indx], data['DEC'][indx],
                                                     mean=1.9, sigma=0.15, seed=seed,
                                                     nside=self.nside_chunk)

            if self.mockformat == 'durham_mxxl_hdf5':
                alldata = np.vstack((data['Z'][indx],
                                     data['SDSS_absmag_r01'][indx],
                                     data['SDSS_01gr'][indx])).T
                _, templateid = self._query(alldata)
                input_meta['TEMPLATEID'] = templateid
            elif self.mockformat == 'bgs-gama':
                # Could conceivably use other colors here--
                alldata = np.vstack((data['Z'][indx],
                                     data['RMABS_01'][indx],
                                     data['GR_01'][indx])).T
                _, templateid = self._query(alldata)
                input_meta['TEMPLATEID'] = templateid
            else:
                pass

            flux, _, meta = self.template_maker.make_templates(input_meta=input_meta, 
                                                               nocolorcuts=True, novdisp=False)
            

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=False,
                                                     seed=seed, gmm=gmm,
                                                     truespectype='GALAXY',
                                                     templatetype='BGS')
        
        return flux, self.wave, meta, targets, truth

    def select_targets(self, targets, truth, **kwargs):
        """Select BGS targets.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        from desitarget.cuts import isBGS_bright, isBGS_faint

        gflux, rflux, zflux, w1flux, w2flux = self.deredden(targets)

        # Select BGS_BRIGHT targets.
        bgs_bright = isBGS_bright(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT_SOUTH
        targets['DESI_TARGET'] |= (bgs_bright != 0) * self.desi_mask.BGS_ANY

        # Select BGS_FAINT targets.
        bgs_faint = isBGS_faint(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT_SOUTH
        targets['DESI_TARGET'] |= (bgs_faint != 0) * self.desi_mask.BGS_ANY
        
class STARMaker(SelectTargets):
    """Lower-level Class for preparing for stellar spectra to be generated,
    selecting standard stars, and selecting stars as contaminants for
    extragalactic targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        each target.  Defaults to `decam2014-r`.
    star_normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        the faint stars.  Defaults to `sdss2010-r`.

    """
    wave, template_maker, tree = None, None, None
    star_maggies_g, star_maggies_r = None, None
    
    def __init__(self, seed=None, normfilter='decam2014-r', **kwargs):
        from scipy.spatial import cKDTree as KDTree
        from speclite import filters
        from desisim.templates import STAR

        super(STARMaker, self).__init__()

        self.seed = seed
        self.objtype = 'STAR'

        if self.wave is None:
            STARMaker.wave = _default_wave()
        if self.template_maker is None:
            STARMaker.template_maker = STAR(wave=self.wave, normfilter=normfilter)

        self.meta = self.template_maker.basemeta
        #self.star_normfilter = star_normfilter

        # Pre-compute normalized synthetic photometry for the full set of
        # stellar templates.
        if self.star_maggies_g is None or self.star_maggies_r is None:
            flux, wave = self.template_maker.baseflux, self.template_maker.basewave

            decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                             'wise2010-W1', 'wise2010-W2')
            maggies = decamwise.get_ab_maggies(flux, wave, mask_invalid=True)

            # Normalize to both sdss-g and sdss-r
            sdssg = filters.load_filters('sdss2010-g')
            sdssr = filters.load_filters('sdss2010-r')

            def _get_maggies(flux, wave, maggies, normfilter):
                normmaggies = normfilter.get_ab_maggies(flux, wave, mask_invalid=True)

                for filt, flux in zip( maggies.colnames, ('FLUX_G', 'FLUX_R', 'FLUX_Z',
                                                          'FLUX_W1', 'FLUX_W2') ):

                    maggies[filt] /= normmaggies[normfilter.names[0]]
                    maggies.rename_column(filt, flux)
                    
                return maggies

            STARMaker.star_maggies_g = _get_maggies(flux, wave, maggies.copy(), sdssg)
            STARMaker.star_maggies_r = _get_maggies(flux, wave, maggies.copy(), sdssr)

        # Build the KD Tree.
        if self.tree is None:
            STARMaker.tree = KDTree(np.vstack((self.meta['TEFF'].data,
                                               self.meta['LOGG'].data,
                                               self.meta['FEH'].data)).T)
        
    def _query(self, matrix):
        """Return the nearest template number based on the KD Tree."""
        dist, indx = self.tree.query(matrix)
        return dist, indx

    def template_photometry(self, data=None, indx=None, rand=None):
        """Get stellar photometry from the templates themselves, by-passing the
        generation of spectra.

        """
        from desisim.io import empty_metatable

        if rand is None:
            rand = np.random.RandomState()

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)
        
        meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        meta['SEED'] = rand.randint(2**31, size=nobj)
        meta['REDSHIFT'] = data['Z'][indx]
        meta['MAG'] = data['MAG'][indx]
        meta['TEFF'] = data['TEFF'][indx]
        meta['LOGG'] = data['LOGG'][indx]
        meta['FEH'] = data['FEH'][indx]

        if self.mockformat == 'galaxia':
            alldata = np.vstack((data['TEFF'][indx],
                                 data['LOGG'][indx],
                                 data['FEH'][indx])).T
            _, templateid = self._query(alldata)
        elif self.mockformat == 'mws_100pc':
            alldata = np.vstack((data['TEFF'][indx],
                                 data['LOGG'][indx],
                                 data['FEH'][indx])).T
            _, templateid = self._query(alldata)

        normmag = 1e9 * 10**(-0.4 * data['MAG'][indx]) # nanomaggies
        
        if data['NORMFILTER'] == 'sdss2010-g':
            star_maggies = self.star_maggies_g
        elif data['NORMFILTER'] == 'sdss2010-r':
            star_maggies = self.star_maggies_r
        else:
            log.warning('Unrecognized normalization filter!')
            raise ValueError
            
        for key in ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'):
            meta[key][:] = star_maggies[key][templateid] * normmag

        return meta
 
    def select_standards(self, targets, truth, boss_std=None):
        """Select bright- and dark-time standard stars.  Input tables are modified in
        place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        boss_std : :class:`numpy.ndarray`, optional
            Boolean array generated by ReadGalaxia.select_sdss_std indicating
            whether a star satisfies the SDSS/BOSS standard-star selection
            criteria.  Defaults to None.

        """
        from desitarget.cuts import isFSTD

        gflux, rflux, zflux, w1flux, w2flux = self.deredden(targets)
        obs_rflux = targets['FLUX_R'] # observed (attenuated) flux

        gsnr, rsnr, zsnr = gflux*0+100, rflux*0+100, zflux*0+100    # Hack -- fixed S/N
        gfracflux, rfracflux, zfracflux = gflux*0, rflux*0, zflux*0 # # No contamination from neighbors.
        objtype = np.repeat('PSF', len(targets)).astype('U3') # Right data type?!?

        # Select dark-time FSTD targets.  Temporary hack to use the BOSS
        # standard-star selection algorith.
        if boss_std is not None:
            rbright, rfaint = 16, 19
            fstd = boss_std * ( obs_rflux < 10**((22.5 - rbright)/2.5) ) * (
                obs_rflux > 10**((22.5 - rfaint)/2.5) )
        else:
            fstd = isFSTD(gflux=gflux, rflux=rflux, zflux=zflux, objtype=objtype,
                          gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, 
                          gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux, 
                          obs_rflux=obs_rflux)

        targets['DESI_TARGET'] |= (fstd != 0) * self.desi_mask.STD_FSTAR

        # Select bright-time FSTD targets.  Temporary hack to use the BOSS
        # standard-star selection algorith.
        if boss_std is not None:
            rbright, rfaint = 14, 17
            fstd_bright = boss_std * ( obs_rflux < 10**((22.5 - rbright)/2.5) ) * (
                obs_rflux > 10**((22.5 - rfaint)/2.5) )
        else:
            fstd_bright = isFSTD(gflux=gflux, rflux=rflux, zflux=zflux, objtype=objtype,
                                 gsnr=gsnr, rsnr=rsnr, zsnr=zsnr, 
                                 gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux, 
                                 obs_rflux=obs_rflux, bright=True)

        targets['DESI_TARGET'] |= (fstd_bright != 0) * self.desi_mask.STD_BRIGHT

    def select_contaminants(self, targets, truth):
        """Select stellar (faint and bright) contaminants for the extragalactic targets.
        Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """ 
        from desitarget.cuts import isBGS_faint, isELG, isLRG_colors, isQSO_colors

        gflux, rflux, zflux, w1flux, w2flux = self.deredden(targets)

        # Select stellar contaminants for BGS_FAINT targets.
        bgs_faint = isBGS_faint(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT_SOUTH
        targets['DESI_TARGET'] |= (bgs_faint != 0) * self.desi_mask.BGS_ANY
        
        truth['CONTAM_TARGET'] |= (bgs_faint != 0) * self.contam_mask.BGS_IS_STAR
        truth['CONTAM_TARGET'] |= (bgs_faint != 0) * self.contam_mask.BGS_CONTAM

        # Select stellar contaminants for ELG targets.
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        
        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_IS_STAR
        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_CONTAM

        # Select stellar contaminants for LRG targets.
        lrg = isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux)
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH

        truth['CONTAM_TARGET'] |= (lrg != 0) * self.contam_mask.LRG_IS_STAR
        truth['CONTAM_TARGET'] |= (lrg != 0) * self.contam_mask.LRG_CONTAM

        # Select stellar contaminants for QSO targets.
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                           w2flux=w2flux)
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH

        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_IS_STAR
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_CONTAM

class MWS_MAINMaker(STARMaker):
    """Read MWS_MAIN mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        each target.  Defaults to `decam2014-r`.
    calib_only : :class:`bool`, optional
        Use MWS_MAIN stars as calibration (standard star) targets, only.
        Defaults to False.

    """
    def __init__(self, seed=None, normfilter='decam2014-r', calib_only=False, **kwargs):
        super(MWS_MAINMaker, self).__init__()

        self.calib_only = calib_only

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'mws', 'galaxia', 'alpha',
                                             'v0.0.5', 'healpix')

    def read(self, mockfile=None, mockformat='galaxia', dust_dir=None,
             healpixels=None, nside=None, nside_galaxia=8, magcut=None,
             **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'galaxia'.
        dust_dir : :class:`str`
            Full path to the dust maps.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        nside_galaxia : :class:`int`
            Healpixel nside indicating how the mock on-disk has been organized.
            Defaults to 8.

        Returns
        -------
        :class:`dict`
            Dictionary of target properties with various keys (to be documented). 

        Raises
        ------
        ValueError
            If mockformat is not recognized.

        """
        self.mockformat = mockformat.lower()
        if self.mockformat == 'galaxia':
            MockReader = ReadGalaxia(dust_dir=dust_dir)
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile
            
        data = MockReader.readmock(mockfile, target_name='MWS_MAIN',
                                   healpixels=healpixels, nside=nside,
                                   nside_galaxia=nside_galaxia, magcut=magcut)
        self._update_normfilter(data.get('NORMFILTER'))

        return data
    
    def make_spectra(self, data=None, indx=None, seed=None, no_spectra=False):
        """Generate MWS_MAIN stellar spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        no_spectra : :class:`bool`, optional
            Do not generate spectra.  Defaults to False.

        Returns
        -------
        flux : :class:`numpy.ndarray`
            Target spectra.
        wave : :class:`numpy.ndarray`
            Corresponding wavelength array.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        
        """
        from desisim.io import empty_metatable
        
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)
        
        if no_spectra:
            flux = []
            meta = self.template_photometry(data, indx, rand)
        else:
            input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            input_meta['SEED'] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'] = data['Z'][indx]
            input_meta['MAG'] = data['MAG'][indx]
            input_meta['TEFF'] = data['TEFF'][indx]
            input_meta['LOGG'] = data['LOGG'][indx]
            input_meta['FEH'] = data['FEH'][indx]

            if self.mockformat == 'galaxia':
                alldata = np.vstack((data['TEFF'][indx],
                                     data['LOGG'][indx],
                                     data['FEH'][indx])).T
                _, templateid = self._query(alldata)
                input_meta['TEMPLATEID'] = templateid

            # Note! No colorcuts.
            flux, _, meta = self.template_maker.make_templates(input_meta=input_meta)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=True,
                                                     seed=seed, truespectype='STAR',
                                                     templatetype='STAR')
                                                           
        return flux, self.wave, meta, targets, truth

    def select_targets(self, targets, truth, boss_std=None):
        """Select various MWS stars, standard stars, and (bright) contaminants for
        extragalactic targets.  Input tables are modified in place.

        Note: The selection here eventually will be done with Gaia (I think).

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        boss_std : :class:`numpy.ndarray`, optional
            Boolean array generated by ReadGalaxia.select_sdss_std indicating
            whether a star satisfies the SDSS/BOSS standard-star selection
            criteria.  Defaults to None.

        """
        def _isMWS_MAIN(rflux):
            """A function like this should be in desitarget.cuts. Select 15<r<19 stars."""
            main = rflux > 10**( (22.5 - 19.0) / 2.5 )
            main &= rflux <= 10**( (22.5 - 15.0) / 2.5 )
            return main

        gflux, rflux, zflux, w1flux, w2flux = self.deredden(targets)

        # Select MWS_MAIN targets.
        mws_main = _isMWS_MAIN(rflux=rflux)

        if not self.calib_only:
            targets['MWS_TARGET'] |= (mws_main != 0) * self.mws_mask.mask('MWS_MAIN')
            targets['DESI_TARGET'] |= (mws_main != 0) * self.desi_mask.MWS_ANY
        
            # Select bright stellar contaminants for the extragalactic targets.
            self.select_contaminants(targets, truth)

        # Select standard stars.
        self.select_standards(targets, truth, boss_std=boss_std)

class FAINTSTARMaker(STARMaker):
    """Read FAINTSTAR mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        each target.  Defaults to `decam2014-r`.

    """
    def __init__(self, seed=None, normfilter='decam2014-r', **kwargs):
        super(FAINTSTARMaker, self).__init__()

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'mws', 'galaxia', 'alpha',
                                             '0.0.5_superfaint', 'healpix')

    def read(self, mockfile=None, mockformat='galaxia', dust_dir=None,
             healpixels=None, nside=None, nside_galaxia=8, magcut=None,
             **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'galaxia'.
        dust_dir : :class:`str`
            Full path to the dust maps.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        nside_galaxia : :class:`int`
            Healpixel nside indicating how the mock on-disk has been organized.
            Defaults to 8.

        Returns
        -------
        :class:`dict`
            Dictionary of target properties with various keys (to be documented). 

        Raises
        ------
        ValueError
            If mockformat is not recognized.

        """
        self.mockformat = mockformat.lower()
        if self.mockformat == 'galaxia':
            MockReader = ReadGalaxia(dust_dir=dust_dir)
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile
            
        data = MockReader.readmock(mockfile, target_name='FAINTSTAR',
                                   healpixels=healpixels, nside=nside,
                                   nside_galaxia=nside_galaxia, magcut=magcut)
        self._update_normfilter(data.get('NORMFILTER'))

        return data
    
    def make_spectra(self, data=None, indx=None, boss_std=None, seed=None, no_spectra=False):
        """Generate FAINTSTAR stellar spectra.

        Note: These (numerous!) objects are only used as contaminants, so we use
        the templates themselves for the spectra rather than generating them
        on-the-fly.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.
        boss_std : :class:`numpy.ndarray`, optional
            Boolean array generated by ReadGalaxia.select_sdss_std indicating
            whether a star satisfies the SDSS/BOSS standard-star selection
            criteria.  Defaults to None.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        no_spectra : :class:`bool`, optional
            Do not generate spectra.  Defaults to False.

        Returns
        -------
        flux : :class:`numpy.ndarray`
            Target spectra.
        wave : :class:`numpy.ndarray`
            Corresponding wavelength array.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        from desisim.io import empty_metatable
        
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)

        objseeds = rand.randint(2**31, size=nobj)

        if self.mockformat == 'galaxia':
            alldata = np.vstack((data['TEFF'][indx],
                                 data['LOGG'][indx],
                                 data['FEH'][indx])).T
            _, templateid = self._query(alldata)

        # Initialize dummy targets and truth tables.
        _targets = empty_targets_table(nobj)
        _truth = empty_truth_table(nobj)
        
        # Pack the noiseless stellar photometry in the truth table, generate
        # noisy photometry, and then select targets.
        normmag = 1e9 * 10**(-0.4 * data['MAG'][indx]) # nanomaggies

        if data['NORMFILTER'] == 'sdss2010-r':
            star_maggies = self.star_maggies_r
        elif data['NORMFILTER'] == 'sdss2010-g':
            star_maggies = self.star_maggies_g
        else:
            log.warning('Unrecognized normalization filter!')
            raise ValueError
        
        for key in ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'):
            _truth[key][:] = star_maggies[key][templateid] * normmag

        for band in ('G', 'R', 'Z', 'W1', 'W2'):
            for prefix in ('MW_TRANSMISSION', 'PSFDEPTH'):
                key = '{}_{}'.format(prefix, band)
                _targets[key][:] = data[key][indx]

        self.scatter_photometry(data, _truth, _targets, indx=indx, psf=True, qaplot=False)

        self.select_targets(_targets, _truth, boss_std=boss_std)

        keep = np.where(_targets['DESI_TARGET'] != 0)[0]
        log.debug('Pre-selected {} FAINTSTAR targets.'.format(len(keep)))

        if len(keep) > 0:
            input_meta = empty_metatable(nmodel=len(keep), objtype=self.objtype)
            input_meta['SEED'] = objseeds[keep]
            input_meta['REDSHIFT'] = data['Z'][indx][keep]
            input_meta['MAG'] = data['MAG'][indx][keep]
            input_meta['TEFF'] = data['TEFF'][indx][keep]
            input_meta['LOGG'] = data['LOGG'][indx][keep]
            input_meta['FEH'] = data['FEH'][indx][keep]

            if no_spectra:
                flux = []
                meta = input_meta
                targets, truth = self.populate_targets_truth(data, meta, indx=indx[keep],
                                                             psf=True, seed=seed,
                                                             truespectype='STAR',
                                                             templatetype='STAR')
            else:
                input_meta['TEMPLATEID'] = templateid[keep]

                # Note! No colorcuts.
                flux, _, meta = self.template_maker.make_templates(input_meta=input_meta)

                # Force consistency in the noisy photometry so we select the same targets. 
                targets, truth = self.populate_targets_truth(data, meta, indx=indx[keep],
                                                             psf=True, seed=seed,
                                                             truespectype='STAR',
                                                             templatetype='STAR')
                for filt in ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'):
                    targets[filt][:] = _targets[filt][keep]

            if boss_std is None:
                self.select_targets(targets, truth)
            else:
                self.select_targets(targets, truth, boss_std=boss_std[keep])

            return flux, self.wave, meta, targets, truth

        else:
            return [], self.wave, None, [], []
                                                           
    def select_targets(self, targets, truth, boss_std=None):
        """Select faint stellar contaminants for the extragalactic targets.  Input
        tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        boss_std : :class:`numpy.ndarray`, optional
            Boolean array generated by ReadGalaxia.select_sdss_std indicating
            whether a star satisfies the SDSS/BOSS standard-star selection
            criteria.  Defaults to None.

        """
        self.select_contaminants(targets, truth)

class MWS_NEARBYMaker(STARMaker):
    """Read MWS_NEARBY mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        each target.  Defaults to `decam2014-g`.

    """
    def __init__(self, seed=None, normfilter='decam2014-g', **kwargs):
        super(MWS_NEARBYMaker, self).__init__()

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'mws', '100pc', 'v0.0.3',
                                             'mock_100pc.fits')

    def read(self, mockfile=None, mockformat='mws_100pc', dust_dir=None,
             healpixels=None, nside=None, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'mws_100pc'.
        dust_dir : :class:`str`
            Full path to the dust maps.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        mock_density : :class:`bool`, optional
            Compute the median target density in the mock.  Defaults to False.

        Returns
        -------
        :class:`dict`
            Dictionary of target properties with various keys (to be documented). 

        Raises
        ------
        ValueError
            If mockformat is not recognized.

        """
        self.mockformat = mockformat.lower()
        if self.mockformat == 'mws_100pc':
            MockReader = ReadMWS_NEARBY(dust_dir=dust_dir)
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name='MWS_NEARBY',
                                   healpixels=healpixels, nside=nside,
                                   mock_density=mock_density)
        self._update_normfilter(data.get('NORMFILTER'))

        return data
    
    def make_spectra(self, data=None, indx=None, seed=None, no_spectra=False):
        """Generate MWS_NEARBY stellar spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        no_spectra : :class:`bool`, optional
            Do not generate spectra.  Defaults to False.

        Returns
        -------
        flux : :class:`numpy.ndarray`
            Target spectra.
        wave : :class:`numpy.ndarray`
            Corresponding wavelength array.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        
        """
        from desisim.io import empty_metatable
        
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)
        
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if no_spectra:
            flux = []
            meta = self.template_photometry(data, indx, rand)
        else:
            input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            input_meta['SEED'] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'] = data['Z'][indx]
            input_meta['MAG'] = data['MAG'][indx]
            input_meta['TEFF'] = data['TEFF'][indx]
            input_meta['LOGG'] = data['LOGG'][indx]
            input_meta['FEH'] = data['FEH'][indx]

            if self.mockformat == 'mws_100pc':
                alldata = np.vstack((data['TEFF'][indx],
                                     data['LOGG'][indx],
                                     data['FEH'][indx])).T
                _, templateid = self._query(alldata)
                input_meta['TEMPLATEID'] = templateid

            # Note! No colorcuts.
            flux, _, meta = self.template_maker.make_templates(input_meta=input_meta)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=True,
                                                     seed=seed, truespectype='STAR',
                                                     templatetype='STAR',
                                                     templatesubtype=data['TEMPLATESUBTYPE'][indx])
                                                           
        return flux, self.wave, meta, targets, truth

    def select_targets(self, targets, truth, **kwargs):
        """Select MWS_NEARBY targets.  Input tables are modified in place.

        Note: The selection here eventually will be done with Gaia (I think) so
        for now just do a "perfect" selection.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        mws_nearby = np.ones(len(targets)) # select everything!
        #mws_nearby = (truth['MAG'] <= 20.0) * 1 # SDSS g-band!

        targets['MWS_TARGET'] |= (mws_nearby != 0) * self.mws_mask.mask('MWS_NEARBY')
        targets['DESI_TARGET'] |= (mws_nearby != 0) * self.desi_mask.MWS_ANY

class WDMaker(SelectTargets):
    """Read WD mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    normfilter : :class:`str`, optional
        Normalization filter for defining normalization (apparent) magnitude of
        each target.  Defaults to `decam2014-r`.
    calib_only : :class:`bool`, optional
        Use WDs as calibration (standard star) targets, only.  Defaults to False. 

    """
    wave, da_template_maker, db_template_maker = None, None, None
    wd_maggies_da, wd_maggies_db, tree_da, tree_db = None, None, None, None

    def __init__(self, seed=None, normfilter='decam2014-g', calib_only=False, **kwargs):
        from scipy.spatial import cKDTree as KDTree
        from speclite import filters 
        from desisim.templates import WD
        
        super(WDMaker, self).__init__()

        self.seed = seed
        self.objtype = 'WD'
        self.calib_only = calib_only

        if self.wave is None:
            WDMaker.wave = _default_wave()
            
        if self.da_template_maker is None:
            WDMaker.da_template_maker = WD(wave=self.wave, subtype='DA', normfilter=normfilter)
            
        if self.db_template_maker is None:
            WDMaker.db_template_maker = WD(wave=self.wave, subtype='DB', normfilter=normfilter)
        
        self.meta_da = self.da_template_maker.basemeta
        self.meta_db = self.db_template_maker.basemeta

        # Pre-compute normalized synthetic photometry for the full set of DA and
        # DB templates.
        if self.wd_maggies_da is None or self.wd_maggies_db is None:

            wave = self.da_template_maker.basewave
            flux_da, flux_db = self.da_template_maker.baseflux, self.db_template_maker.baseflux

            decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                             'wise2010-W1', 'wise2010-W2')
            maggies_da = decamwise.get_ab_maggies(flux_da, wave, mask_invalid=True)
            maggies_db = decamwise.get_ab_maggies(flux_db, wave, mask_invalid=True)

            # Normalize to sdss-g
            normfilter = filters.load_filters('sdss2010-g')

            def _get_maggies(flux, wave, maggies, normfilter):
                normmaggies = normfilter.get_ab_maggies(flux, wave, mask_invalid=True)

                for filt, flux in zip( maggies.colnames, ('FLUX_G', 'FLUX_R', 'FLUX_Z',
                                                          'FLUX_W1', 'FLUX_W2') ):

                    maggies[filt] /= normmaggies[normfilter.names[0]]
                    maggies.rename_column(filt, flux)
                    
                return maggies

            WDMaker.wd_maggies_da = _get_maggies(flux_da, wave, maggies_da.copy(), normfilter)
            WDMaker.wd_maggies_db = _get_maggies(flux_db, wave, maggies_db.copy(), normfilter)

        # Build the KD Trees
        if self.tree_da is None:
            WDMaker.tree_da = KDTree(np.vstack((self.meta_da['TEFF'].data,
                                             self.meta_da['LOGG'].data)).T)
        if self.tree_db is None:
            WDMaker.tree_db = KDTree(np.vstack((self.meta_db['TEFF'].data,
                                         self.meta_db['LOGG'].data)).T)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'mws', 'wd', 'v0.0.2',
                                             'mock_wd.fits')

    def read(self, mockfile=None, mockformat='mws_wd', dust_dir=None,
             healpixels=None, nside=None, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'mws_wd'.
        dust_dir : :class:`str`
            Full path to the dust maps.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        mock_density : :class:`bool`, optional
            Compute the median target density in the mock.  Defaults to False.

        Returns
        -------
        :class:`dict`
            Dictionary of target properties with various keys (to be documented). 

        Raises
        ------
        ValueError
            If mockformat is not recognized.

        """
        self.mockformat = mockformat.lower()
        if self.mockformat == 'mws_wd':
            MockReader = ReadMWS_WD(dust_dir=dust_dir)
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   mock_density=mock_density)
        self._update_normfilter(data.get('NORMFILTER'), objtype=self.objtype)

        return data

    def _query(self, matrix, subtype='DA'):
        """Return the nearest template number based on the KD Tree."""
        if subtype.upper() == 'DA':
            dist, indx = self.tree_da.query(matrix)
        elif subtype.upper() == 'DB':
            dist, indx = self.tree_db.query(matrix)
        else:
            log.warning('Unrecognized SUBTYPE {}!'.format(subtype))
            raise ValueError

        return dist, indx
    
    def template_photometry(self, data=None, indx=None, rand=None, subtype='DA'):
        """Get stellar photometry from the templates themselves, by-passing the
        generation of spectra.

        """
        from desisim.io import empty_metatable

        if rand is None:
            rand = np.random.RandomState()

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)
        
        meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        meta['SEED'] = rand.randint(2**31, size=nobj)
        meta['REDSHIFT'] = data['Z'][indx]
        meta['MAG'] = data['MAG'][indx]
        meta['TEFF'] = data['TEFF'][indx]
        meta['LOGG'] = data['LOGG'][indx]
        meta['SUBTYPE'] = data['TEMPLATESUBTYPE'][indx]

        if self.mockformat == 'mws_wd':
            alldata = np.vstack((data['TEFF'][indx],
                                 data['LOGG'][indx])).T
            _, templateid = self._query(alldata)

        normmag = 1e9 * 10**(-0.4 * data['MAG'][indx]) # nanomaggies
        
        if data['NORMFILTER'] == 'sdss2010-g':
            if subtype == 'DA':
                wd_maggies = self.wd_maggies_da
            elif subtype == 'DB':
                wd_maggies = self.wd_maggies_da
            else:
                log.warning('Unrecognized subtype {}!'.format(subtype))
                raise ValueError
        else:
            log.warning('Unrecognized normalization filter!')
            raise ValueError
            
        for key in ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'):
            meta[key][:] = wd_maggies[key][templateid] * normmag

        return meta

    def make_spectra(self, data=None, indx=None, seed=None, no_spectra=False):
        """Generate WD spectra, dealing with DA vs DB white dwarfs separately.
        
        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.
        no_spectra : :class:`bool`, optional
            Do not generate spectra.  Defaults to False.

        Returns
        -------
        flux : :class:`numpy.ndarray`
            Target spectra.
        wave : :class:`numpy.ndarray`
            Corresponding wavelength array.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        from desisim.io import empty_metatable
        
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)
        
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        #if no_spectra:
        #    flux = []
        #    meta = self.template_photometry(data, indx, rand)
        #else:
        if self.mockformat == 'mws_wd':
            allsubtype = data['TEMPLATESUBTYPE'][indx]
            if no_spectra:
                meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
                flux = []
            else:
                meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
                flux = np.zeros([nobj, len(self.wave)], dtype='f4')

                input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
                input_meta['SEED'] = rand.randint(2**31, size=nobj)
                input_meta['REDSHIFT'] = data['Z'][indx]
                input_meta['MAG'] = data['MAG'][indx]
                input_meta['TEFF'] = data['TEFF'][indx]
                input_meta['LOGG'] = data['LOGG'][indx]
                input_meta['SUBTYPE'] = data['TEMPLATESUBTYPE'][indx]

            for subtype in ('DA', 'DB'):
                these = np.where(allsubtype == subtype)[0]
                if len(these) > 0:
                    alldata = np.vstack((data['TEFF'][indx][these],
                                         data['LOGG'][indx][these])).T
                    _, templateid = self._query(alldata, subtype=subtype)

                    if no_spectra:
                        meta1 = self.template_photometry(data, indx[these],
                                                         rand, subtype)
                        meta[these] = meta1
                    else:
                        input_meta['TEMPLATEID'][these] = templateid

                        template_maker = getattr(self, '{}_template_maker'.format(subtype.lower()))
                        flux1, _, meta1 = template_maker.make_templates(input_meta=input_meta[these])

                        meta[these] = meta1
                        flux[these, :] = flux1

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=True,
                                                     seed=seed, truespectype='WD',
                                                     templatetype='WD',
                                                     templatesubtype=allsubtype)

        return flux, self.wave, meta, targets, truth

    def select_targets(self, targets, truth, **kwargs):
        """Select MWS_WD targets and STD_WD standard stars.  Input tables are modified
        in place.

        Note: The selection here eventually will be done with Gaia (I think) so
        for now just do a "perfect" selection.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        #mws_wd = np.ones(len(targets)) # select everything!
        mws_wd = ((truth['MAG'] >= 15.0) * (truth['MAG'] <= 20.0)) * 1 # SDSS g-band!

        if not self.calib_only:
            targets['MWS_TARGET'] |= (mws_wd != 0) * self.mws_mask.mask('MWS_WD')
            targets['DESI_TARGET'] |= (mws_wd != 0) * self.desi_mask.MWS_ANY

        # Select STD_WD; cut just on g-band magnitude (not TEMPLATESUBTYPE!)
        std_wd = (truth['MAG'] <= 19.0) * 1 # SDSS g-band!
        targets['DESI_TARGET'] |= (std_wd !=0) * self.desi_mask.mask('STD_WD')

class SKYMaker(SelectTargets):
    """Read SKY mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.

    """
    wave = None
    
    def __init__(self, seed=None, **kwargs):
        super(SKYMaker, self).__init__()

        self.seed = seed
        self.objtype = 'SKY'

        if self.wave is None:
            SKYMaker.wave = _default_wave()
        
        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'uniformsky', '0.1',
                                             'uniformsky-2048-0.1.fits')

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=None, nside=None, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'gaussianfield'.
        dust_dir : :class:`str`
            Full path to the dust maps.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        mock_density : :class:`bool`, optional
            Compute the median target density in the mock.  Defaults to False.

        Returns
        -------
        :class:`dict`
            Dictionary of target properties with various keys (to be documented). 

        Raises
        ------
        ValueError
            If mockformat is not recognized.

        """
        self.mockformat = mockformat.lower()
        if self.mockformat == 'uniformsky':
            MockReader = ReadUniformSky(dust_dir=dust_dir)
        elif self.mockformat == 'gaussianfield':
            MockReader = ReadGaussianField(dust_dir=dust_dir)
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   mock_density=mock_density)
        self._update_normfilter(data.get('NORMFILTER'))

        return data

    def make_spectra(self, data=None, indx=None, seed=None, **kwargs):
        """Generate SKY spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.

        Returns
        -------
        flux : :class:`numpy.ndarray`
            Target spectra.
        wave : :class:`numpy.ndarray`
            Corresponding wavelength array.
        meta : :class:`astropy.table.Table`
            Spectral metadata table.
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        
        """
        from desisim.io import empty_metatable
        
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        meta['SEED'] = rand.randint(2**31, size=nobj)
        meta['REDSHIFT'] = data['Z'][indx]
        
        flux = np.zeros((nobj, len(self.wave)), dtype='i1')

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=False,
                                                     seed=seed, truespectype='SKY',
                                                     templatetype='SKY')

        return flux, self.wave, meta, targets, truth

    def select_targets(self, targets, truth, **kwargs):
        """Select SKY targets (i.e., everything).  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        targets['DESI_TARGET'] |= self.desi_mask.mask('SKY')
