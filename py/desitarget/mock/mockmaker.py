# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
=========================
desitarget.mock.mockmaker
=========================

Read mock catalogs and assign spectra.
"""
import os
import numpy as np
from glob import glob
from pkg_resources import resource_filename

import fitsio
import healpy as hp

from desimodel.io import load_pixweight
from desimodel import footprint
from desitarget import cuts
from desisim.io import empty_metatable

from desiutil.log import get_logger, DEBUG
log = get_logger()

try:
    from scipy import constants
    C_LIGHT = constants.c/1000.0
except TypeError: # This can happen during documentation builds.
    C_LIGHT = 299792458.0/1000.0

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
    from astropy.table import Table, Column
    
    targets = Table()

    targets.add_column(Column(name='RELEASE', length=nobj, dtype='i2'))
    targets.add_column(Column(name='BRICKID', length=nobj, dtype='i4'))
    targets.add_column(Column(name='BRICKNAME', length=nobj, dtype='S8'))
    targets.add_column(Column(name='OBJID', length=nobj, dtype='<i4'))
    targets.add_column(Column(name='TYPE', length=nobj, dtype='S4'))
    targets.add_column(Column(name='RA', length=nobj, dtype='f8', unit='degree'))
    targets.add_column(Column(name='DEC', length=nobj, dtype='f8', unit='degree'))
    targets.add_column(Column(name='RA_IVAR', length=nobj, dtype='f4', unit='1/degree**2'))
    targets.add_column(Column(name='DEC_IVAR', length=nobj, dtype='f4', unit='1/degree**2'))
    targets.add_column(Column(name='DCHISQ', length=nobj, dtype='f4', data=np.zeros( (nobj, 5) )))
    targets['DCHISQ'][:, 0] = 1.0 # initialize
    
    targets.add_column(Column(name='FLUX_G', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_R', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_Z', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_W1', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_W2', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_W3', length=nobj, dtype='f4', unit='nanomaggies'))
    targets.add_column(Column(name='FLUX_W4', length=nobj, dtype='f4', unit='nanomaggies'))
    
    targets.add_column(Column(name='FLUX_IVAR_G', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='FLUX_IVAR_R', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='FLUX_IVAR_Z', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='FLUX_IVAR_W1', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='FLUX_IVAR_W2', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='FLUX_IVAR_W3', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='FLUX_IVAR_W4', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    
    targets.add_column(Column(name='MW_TRANSMISSION_G', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_Z', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_W1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_W2', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_W3', length=nobj, dtype='f4'))
    targets.add_column(Column(name='MW_TRANSMISSION_W4', length=nobj, dtype='f4'))

    targets.add_column(Column(name='NOBS_G', length=nobj, dtype='i2'))
    targets.add_column(Column(name='NOBS_R', length=nobj, dtype='i2'))
    targets.add_column(Column(name='NOBS_Z', length=nobj, dtype='i2'))
    targets.add_column(Column(name='FRACFLUX_G', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FRACFLUX_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FRACFLUX_Z', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FRACMASKED_G', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FRACMASKED_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FRACMASKED_Z', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FRACIN_G', data=np.ones(nobj).astype('f4')))
    targets.add_column(Column(name='FRACIN_R', data=np.ones(nobj).astype('f4')))
    targets.add_column(Column(name='FRACIN_Z', data=np.ones(nobj).astype('f4')))
    targets.add_column(Column(name='ALLMASK_G', length=nobj, dtype='f4'))
    targets.add_column(Column(name='ALLMASK_R', length=nobj, dtype='f4'))
    targets.add_column(Column(name='ALLMASK_Z', length=nobj, dtype='f4'))
    targets.add_column(Column(name='WISEMASK_W1', length=nobj, dtype='|u1'))
    targets.add_column(Column(name='WISEMASK_W2', length=nobj, dtype='|u1'))
    
    targets.add_column(Column(name='PSFDEPTH_G', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='PSFDEPTH_R', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='PSFDEPTH_Z', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='GALDEPTH_G', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='GALDEPTH_R', length=nobj, dtype='f4', unit='1/nanomaggies**2'))
    targets.add_column(Column(name='GALDEPTH_Z', length=nobj, dtype='f4', unit='1/nanomaggies**2'))

    targets.add_column(Column(name='FRACDEV', length=nobj, dtype='f4'))
    targets.add_column(Column(name='FRACDEV_IVAR', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_R', length=nobj, dtype='f4', unit='arcsec'))
    targets.add_column(Column(name='SHAPEDEV_R_IVAR', length=nobj, dtype='f4', unit='1/arcsec**2'))
    targets.add_column(Column(name='SHAPEDEV_E1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_E1_IVAR', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_E2', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEDEV_E2_IVAR', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEEXP_R', length=nobj, dtype='f4', unit='arcsec'))
    targets.add_column(Column(name='SHAPEEXP_R_IVAR', length=nobj, dtype='f4', unit='1/arcsec**2'))
    targets.add_column(Column(name='SHAPEEXP_E1', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEEXP_E1_IVAR', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEEXP_E2', length=nobj, dtype='f4'))
    targets.add_column(Column(name='SHAPEEXP_E2_IVAR', length=nobj, dtype='f4'))

    targets.add_column(Column(name='FIBERFLUX_G', length=nobj, dtype='>f4'))
    targets.add_column(Column(name='FIBERFLUX_R', length=nobj, dtype='>f4'))
    targets.add_column(Column(name='FIBERFLUX_Z', length=nobj, dtype='>f4'))
    targets.add_column(Column(name='FIBERTOTFLUX_G', length=nobj, dtype='>f4'))
    targets.add_column(Column(name='FIBERTOTFLUX_R', length=nobj, dtype='>f4'))
    targets.add_column(Column(name='FIBERTOTFLUX_Z', length=nobj, dtype='>f4'))

    # Gaia columns
    targets.add_column(Column(name='REF_CAT', length=nobj, dtype='S2'))
    targets.add_column(Column(name='REF_ID', data=np.repeat(-1, nobj).astype('int64'))) # default is -1
    targets.add_column(Column(name='GAIA_PHOT_G_MEAN_MAG', length=nobj, dtype='f4'))
    targets.add_column(Column(name='GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', length=nobj, dtype='f4'))
    targets.add_column(Column(name='GAIA_PHOT_BP_MEAN_MAG', length=nobj, dtype='f4'))
    targets.add_column(Column(name='GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', length=nobj, dtype='f4'))
    targets.add_column(Column(name='GAIA_PHOT_RP_MEAN_MAG', length=nobj, dtype='f4'))    
    targets.add_column(Column(name='GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', length=nobj, dtype='f4'))

    targets.add_column(Column(name='GAIA_PHOT_BP_RP_EXCESS_FACTOR', length=nobj, dtype='f4'))  # not sure what this should be
    targets.add_column(Column(name='GAIA_ASTROMETRIC_SIGMA5D_MAX', length=nobj, dtype='f4'))   # not sure what this should be
    targets.add_column(Column(name='GAIA_ASTROMETRIC_PARAMS_SOLVED', length=nobj, dtype='i8')) # default is 0, "good" is 31
    
    targets.add_column(Column(name='GAIA_ASTROMETRIC_EXCESS_NOISE', length=nobj, dtype='f4'))
    targets.add_column(Column(name='GAIA_DUPLICATED_SOURCE', length=nobj, dtype=bool)) # default is False
    targets.add_column(Column(name='PARALLAX', length=nobj, dtype='f4'))
    targets.add_column(Column(name='PARALLAX_IVAR', data=np.ones(nobj, dtype='f4'))) # default is unity
    targets.add_column(Column(name='PMRA', length=nobj, dtype='f4'))
    targets.add_column(Column(name='PMRA_IVAR', data=np.ones(nobj, dtype='f4'))) # default is unity
    targets.add_column(Column(name='PMDEC', length=nobj, dtype='f4'))
    targets.add_column(Column(name='PMDEC_IVAR', data=np.ones(nobj, dtype='f4'))) # default is unity

    targets.add_column(Column(name='MASKBITS', length=nobj, dtype='>i2'))

    targets.add_column(Column(name='EBV', length=nobj, dtype='f4'))
    targets.add_column(Column(name='PHOTSYS', length=nobj, dtype='|S1'))
    targets.add_column(Column(name='DESI_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='BGS_TARGET', length=nobj, dtype='i8'))
    targets.add_column(Column(name='MWS_TARGET', length=nobj, dtype='i8'))

    # All these columns are added in mock.build.targets_truth
    #targets.add_column(Column(name='TARGETID', length=nobj, dtype='int64'))
    #targets.add_column(Column(name='PRIORITY_INIT', length=nobj, dtype='i8'))
    #targets.add_column(Column(name='SUBPRIORITY', length=nobj, dtype='f8'))
    #targets.add_column(Column(name='NUMOBS_INIT', length=nobj, dtype='i8'))
    #targets.add_column(Column(name='HPXPIXEL', length=nobj, dtype='i8'))

    return targets

def empty_truth_table(nobj=1, templatetype='', use_simqso=True):
    """Initialize an empty 'truth' table.

    Parameters
    ----------
    nobj : :class:`int`
        Number of objects.
    use_simqso : :class:`bool`, optional
        Initialize a SIMQSO-style objtruth table. Defaults to True.

    Returns
    -------
    truth : :class:`astropy.table.Table`
        Truth table.
    objtruth : :class:`astropy.table.Table`
        Objtype-specific truth table (if applicable).
    
    """
    from astropy.table import Table, Column
    
    truth = Table()
    truth.add_column(Column(name='TARGETID', length=nobj, dtype='int64'))
    truth.add_column(Column(name='MOCKID', length=nobj, dtype='int64'))

    truth.add_column(Column(name='TRUEZ', length=nobj, dtype='f4', data=np.zeros(nobj)))
    truth.add_column(Column(name='TRUESPECTYPE', length=nobj, dtype='S10')) # GALAXY, QSO, STAR, etc.
    truth.add_column(Column(name='TEMPLATETYPE', length=nobj, dtype='S10')) # ELG, BGS, STAR, WD, etc.
    truth.add_column(Column(name='TEMPLATESUBTYPE', length=nobj, dtype='S10')) # DA, DB, etc.

    truth.add_column(Column(name='TEMPLATEID', length=nobj, dtype='i4', data=np.zeros(nobj)-1))
    truth.add_column(Column(name='SEED', length=nobj, dtype='int64', data=np.zeros(nobj)-1))
    truth.add_column(Column(name='MAG', length=nobj, dtype='f4', data=np.zeros(nobj), unit='mag'))
    truth.add_column(Column(name='MAGFILTER', length=nobj, dtype='S15')) # normalization filter

    truth.add_column(Column(name='FLUX_G', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_R', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_Z', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_W1', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_W2', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_W3', length=nobj, dtype='f4', unit='nanomaggies'))
    truth.add_column(Column(name='FLUX_W4', length=nobj, dtype='f4', unit='nanomaggies'))

    _, objtruth = empty_metatable(nmodel=nobj, objtype=templatetype, simqso=use_simqso)
    if len(objtruth) == 0:
        objtruth = [] # need an empty list for the multiprocessing in build.select_targets
    else:
        if (templatetype == 'QSO' or templatetype == 'ELG' or
            templatetype == 'LRG' or templatetype == 'BGS'):
            objtruth.add_column(Column(name='TRUEZ_NORSD', length=nobj, dtype='f4'))

    return truth, objtruth

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

    Parameters
    ----------
    bricksize : :class:`float`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        and brickid to each object.  Defaults to 0.25 deg.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    GMM_LRG, GMM_ELG, GMM_BGS, GMM_QSO, FFA = None, None, None, None, None

    def __init__(self, bricksize=0.25, survey='main', **kwargs):
        from astropy.io import fits

        from speclite import filters
        from desiutil.dust import SFDMap
        from desiutil.brick import Bricks
        from desimodel.fastfiberacceptance import FastFiberAcceptance

        self.survey = survey
        if survey == 'main':
            from desitarget.targetmask import desi_mask, bgs_mask, mws_mask
        elif survey == 'sv1':
            from desitarget.sv1.sv1_targetmask import desi_mask, bgs_mask, mws_mask
        else:
            log.warning('Survey {} not recognized!'.format(survey))
            raise ValueError
            
        self.desi_mask = desi_mask
        self.bgs_mask = bgs_mask
        self.mws_mask = mws_mask

        self.Bricks = Bricks(bricksize=bricksize)
        self.SFDMap = SFDMap()

        # Cache the plate scale (which is approximate; see
        # $DESIMODEL/data/desi.yaml), and the FastFiberAcceptance class for the
        # fiberflux calculation, below.
        self.plate_scale_arcsec2um = 107.0 / 1.52 # [um/arcsec]
        if self.FFA is None:
            SelectTargets.FFA = FastFiberAcceptance(filename=os.path.join(
                os.getenv('DESIMODEL'), 'data', 'throughput', 'galsim-fiber-acceptance.fits'))

        self.bassmzlswise = filters.load_filters('BASS-g', 'BASS-r', 'MzLS-z', 'wise2010-W1', 'wise2010-W2')
        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z', 'wise2010-W1', 'wise2010-W2')

        # Read and cache the default pixel weight map.
        pixfile = os.path.join(os.environ['DESIMODEL'],'data','footprint','desi-healpix-weights.fits')
        with fits.open(pixfile) as hdulist:
            self.pixmap = hdulist[0].data

        # from desisim.templates.GALAXY
        self.fiberflux_fraction = {'ELG': 0.6, 'LRG': 0.4, 'BGS': 0.3}

    def mw_transmission(self, data):
        """Compute the grzW1W2 Galactic transmission for every object.

        Parameters
        ----------
        data : :class:`dict`
            Input dictionary of sources with RA, Dec coordinates, modified on output
            to contain reddening and the MW transmission in various bands.

        Raises
        ------

        """
        extcoeff = dict(G = 3.214, R = 2.165, Z = 1.211, W1 = 0.184, W2 = 0.113, W3 = 0.0241, W4 = 0.00910)
        data['EBV'] = self.SFDMap.ebv(data['RA'], data['DEC'], scaling=1.0)

        for band in ('G', 'R', 'Z', 'W1', 'W2', 'W3', 'W4'):
            data['MW_TRANSMISSION_{}'.format(band)] = 10**(-0.4 * extcoeff[band] * data['EBV'])

    def mw_dust_extinction(self, Rv=3.1):
        """Cache the spectroscopic Galactic extinction curve for later use.

        Parameters
        ----------
        Rv : :class:`float`
            Total-to-selective extinction factor.  Defaults to 3.1.

        Raises
        ------

        """
        from desiutil.dust import ext_odonnell
        extinction = Rv * ext_odonnell(self.wave, Rv=Rv)
        return extinction

    def imaging_depth(self, data):
        """Add the imaging depth to the data dictionary.

        Note: In future, this should be a much more sophisticated model based on the
        actual imaging data releases (e.g., it should depend on healpixel).

        Parameters
        ----------
        data : :class:`dict`
            Input dictionary of sources with RA, Dec coordinates, modified on output
            to contain the PSF and galaxy depth in various bands.

        """
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        nobj = len(data['RA'])

        psfdepth_mag = np.array((24.65, 23.61, 22.84)) # 5-sigma, mag
        galdepth_mag = np.array((24.7, 23.9, 23.0))    # 5-sigma, mag

        psfdepth_ivar = (1 / 10**(-0.4 * (psfdepth_mag - 22.5)))**2 # 5-sigma, 1/nanomaggies**2
        galdepth_ivar = (1 / 10**(-0.4 * (galdepth_mag - 22.5)))**2 # 5-sigma, 1/nanomaggies**2

        for ii, band in enumerate(('G', 'R', 'Z')):
            data['PSFDEPTH_{}'.format(band)] = np.repeat(psfdepth_ivar[ii], nobj)
            data['GALDEPTH_{}'.format(band)] = np.repeat(galdepth_ivar[ii], nobj)

        # compute the WISE depth, which is largely a function of ecliptic latitude 
        coord = SkyCoord(data['RA']*u.deg, data['DEC']*u.deg)
        ecoord = coord.transform_to('barycentrictrueecliptic')
        beta = ecoord.lat.value
        if np.count_nonzero(beta > 89) > 0: # don't explode at the pole!
            beta[beta > 89] = 89
        if np.count_nonzero(beta < -89) > 0: # don't explode at the pole!
            beta[beta < -89] = -89
        beta = np.radians(ecoord.lat.value) # [radians]

        sig_syst = [0.5, 2.0]                   # systematic uncertainty due to low-level 
                                                # background structure e.g. striping
        neff = [15.7832, 18.5233]               # effective number of pixels in PSF
        vega2ab = [2.699, 3.339]
        sig_stat_beta0 = [3.5127802, 9.1581879] # random uncertainty [AB nanomaggies]

        for ii, band in enumerate(('W1', 'W2')):
            sig_stat = sig_stat_beta0[ii] / np.sqrt( 1.0 / np.cos(beta) )
            sig = np.sqrt( sig_stat**2 + sig_syst[ii]**2 )

            wisedepth_mag = 22.5 - 2.5 * np.log10( sig * np.sqrt(neff[ii]) ) + vega2ab[ii] # 1-sigma, AB mag
            wisedepth_ivar = 1 / (5 * 10**(-0.4 * (wisedepth_mag - 22.5)))**2 # 5-sigma, 1/nanomaggies**2
            data['PSFDEPTH_{}'.format(band)] = wisedepth_ivar

    def scatter_photometry(self, data, truth, targets, indx=None, 
                           seed=None, qaplot=False):
        """Add noise to the input (noiseless) photometry based on the depth (as well as
        the inverse variance fluxes in GRZW1W2).

        The input targets table is modified in place.

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

        if 'TYPE' in data.keys():
            for band in ('G', 'R', 'Z'):
                fluxkey = 'FLUX_{}'.format(band)
                ivarkey = 'FLUX_IVAR_{}'.format(band)

                for depthprefix in ('PSF', 'GAL'):
                    if depthprefix == 'PSF':
                        these = np.where( data['TYPE'][indx] == 'PSF' )[0] # point sources
                    else:
                        these = np.where( data['TYPE'][indx] != 'PSF' )[0] # galaxies

                    if len(these) > 0:
                        depthkey = '{}DEPTH_{}'.format(depthprefix, band)

                        sigma = 1 / np.sqrt(data[depthkey][indx][these]) / 5 # nanomaggies, 1-sigma
                        targets[fluxkey][these] = truth[fluxkey][these] + rand.normal(scale=sigma)

                        targets[ivarkey][these] = 1 / sigma**2

        # WISE sources are all point sources
        for band in ('W1', 'W2'):
            fluxkey = 'FLUX_{}'.format(band)
            ivarkey = 'FLUX_IVAR_{}'.format(band)
            depthkey = 'PSFDEPTH_{}'.format(band)

            sigma = 1 / np.sqrt(data[depthkey][indx]) / 5 # nanomaggies, 1-sigma
            targets[fluxkey][:] = truth[fluxkey] + rand.normal(scale=sigma)

            targets[ivarkey][:] = 1 / sigma**2

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

    def read_GMM(self, target=None):
        """Read the GMM for the full range of morphological types of a given target
        type, as well as the magnitude-dependent morphological fraction.

        See desitarget/doc/nb/gmm-dr9.ipynb for details.

        """
        from astropy.io import fits
        from astropy.table import Table
        from desiutil.sklearn import GaussianMixtureModel

        if target is not None:
            try:
                if getattr(self, 'GMM_{}'.format(target.upper())) is not None:
                    return
            except:
                pass
                #return

            gmmdir = resource_filename('desitarget', 'mock/data/dr9')
            if not os.path.isdir:
                log.warning('DR9 GMM directory {} not found!'.format(gmmdir))
                raise IOError
            
            fracfile = os.path.join(gmmdir, 'fractype_{}.fits'.format(target.lower()))
            fractype = Table.read(fracfile)

            gmm = []
            for morph in ('PSF', 'REX', 'EXP', 'DEV', 'SER'):
                gmmfile = os.path.join(gmmdir, 'gmm_{}_{}.fits'.format(target.lower(), morph.lower()))
                if os.path.isfile(gmmfile): # not all targets have all morphologies
                    # Get the GMM properties modeled.
                    cols = []
                    with fits.open(gmmfile, 'readonly') as ff:
                        ncol = ff[0].header['NCOL']
                        for ii in range(ncol):
                            cols.append(ff[0].header['COL{:02d}'.format(ii)])
                    gmm.append( (morph, cols, GaussianMixtureModel.load(gmmfile)) )

            # Now unpack the list of tuples into a more convenient set of
            # variables and then repack and return.
            morph = [info[0] for info in gmm]
            gmmcols = [info[1] for info in gmm]
            GMM = [info[2] for info in gmm]

            setattr(self, 'GMM_{}'.format(target.upper()), (morph, fractype, gmmcols, GMM))

    def sample_GMM(self, nobj, isouth=None, target=None, seed=None, morph=None,
                   prior_mag=None, prior_redshift=None):
        """Sample from the GMMs read by self.read_GMM.

        See desitarget/doc/nb/gmm-dr9.ipynb for details.

        """
        rand = np.random.RandomState(seed)
        
        try:
            GMM = getattr(self, 'GMM_{}'.format(target.upper()))
            if GMM is None:
                self.read_GMM(target=target)
                GMM = getattr(self, 'GMM_{}'.format(target.upper()))
        except:
            return None # no GMM for this target

        if target == 'LRG':
            colorcuts_function = cuts.isLRG_colors
        elif target == 'ELG':
            colorcuts_function = cuts.isELG_colors
        elif target == 'QSO':
            colorcuts_function = cuts.isQSO_colors
        else:
            colorcuts_function = None

        # Allow an input morphological type, e.g., to simulate contaminants.
        if morph is None:
            morph = GMM[0]
        if isouth is None:
            isouth = np.ones(nobj).astype(bool)

        south = np.where( isouth )[0]
        north = np.where( ~isouth )[0]

        # Marginalize the morphological fractions over magnitude.
        magbins = GMM[1]['MAG'].data
        deltam = np.diff(magbins)[0]
        minmag, maxmag = magbins.min()-deltam / 2, magbins.max()+deltam / 2

        # Get the total number of each morphological type, accounting for
        # rounding.
        frac2d_magbins = np.vstack( [GMM[1][mm].data for mm in np.atleast_1d(morph)] )
        norm = np.sum(frac2d_magbins, axis=1)
        frac1d_morph = norm / np.sum(norm)
        nobj_morph = np.round(frac1d_morph * nobj).astype(int)
        dn = np.sum(nobj_morph) - nobj
        if dn > 0:
            nobj_morph[np.argmax(nobj_morph)] -= dn
        elif dn < 0:
            nobj_morph[np.argmax(nobj_morph)] += dn

        # Next, sample from the GMM for each morphological type.  For
        # simplicity we ignore the north-south split here.
        gmmout = {'MAGFILTER': np.zeros(nobj).astype('U15'), 'TYPE': np.zeros(nobj).astype('U4')}
        for key in ('MAG', 'SHAPE_R', 'SHAPE_R_IVAR', 
                    'SHAPE_E1', 'SHAPE_E1_IVAR',
                    'SHAPE_E2', 'SHAPE_E2_IVAR',
                    'GR', 'RZ', 'ZW1', 'W1W2'):
            gmmout[key] = np.zeros(nobj).astype('f4')

        def _samp_iterate(samp, target='', south=True, rand=None, maxiter=5,
                          colorcuts_function=None, fiberflux_fraction=1.0):
            """Sample from the given GMM iteratively and only keep objects that pass our
            color-cuts."""
            nneed = len(samp)
            need = np.arange(nneed)

            makemore, itercount = True, 1
            while makemore:
                #print(itercount, nneed)
                # This algorithm is not quite right because the GMMs are drawn
                # from DR9/south, but we're using them to simulate "north"
                # photometry as well.
                _samp = GMM[3][ii].sample(nneed, random_state=rand)
                for jj, tt in enumerate(cols):
                    samp[tt][need] = _samp[:, jj]

                if colorcuts_function is None:
                    nneed = 0
                    makemore = False
                else:
                    if 'z' in samp.dtype.names:
                        zmag = samp['z'][need]
                        rmag = samp['rz'][need] + zmag
                        gmag = samp['gr'][need] + rmag
                    elif 'g' in samp.dtype.names:
                        gmag = samp['g'][need]
                        rmag = gmag - samp['gr'][need]
                        zmag = rmag - samp['rz'][need]
                    else:
                        rmag = samp['r'][need]
                        zmag = rmag - samp['rz'][need]
                        gmag = samp['gr'][need] + rmag

                    if 'zw1' in samp.dtype.names:
                        w1mag = zmag - samp['zw1'][need]
                    else:
                        w1mag = np.zeros_like(rmag)
                    if 'w1w2' in samp.dtype.names:
                        w2mag = w1mag - samp['w1w2'][need]
                    else:
                        w2mag = np.zeros_like(rmag)

                    gflux, rflux, zflux, w1flux, w2flux = [1e9 * 10**(-0.4*mg) for mg in
                                                           (gmag, rmag, zmag, w1mag, w2mag)]
                    gfiberflux = fiberflux_fraction * gflux
                    rfiberflux = fiberflux_fraction * rflux
                    zfiberflux = fiberflux_fraction * zflux

                    if target == 'QSO':
                        itarg = colorcuts_function(gflux=gflux, rflux=rflux, zflux=zflux,
                                                   w1flux=w1flux, w2flux=w2flux, 
                                                   south=south)
                    elif target == 'BGS':
                        itarg = colorcuts_function(gflux=gflux, rflux=rflux, zflux=zflux,
                                                   w1flux=w1flux, w2flux=w2flux, 
                                                   gfiberflux=gfiberflux,
                                                   rfibertotflux=rfiberflux,
                                                   zfiberflux=zfiberflux,
                                                   south=south)
                    elif target == 'ELG':
                        itarg_vlo, _itarg = colorcuts_function(gflux=gflux, rflux=rflux, zflux=zflux,
                                                               w1flux=w1flux, w2flux=w2flux, 
                                                               gfiberflux=gfiberflux,
                                                               rfiberflux=rfiberflux,
                                                               zfiberflux=zfiberflux,
                                                               south=south)
                        itarg = np.logical_or(itarg_vlo, _itarg)
                    else:
                        itarg = colorcuts_function(gflux=gflux, rflux=rflux, zflux=zflux,
                                                   w1flux=w1flux, w2flux=w2flux, 
                                                   gfiberflux=gfiberflux,
                                                   rfiberflux=rfiberflux,
                                                   zfiberflux=zfiberflux,
                                                   south=south)

                    need = np.where( itarg == False )[0]
                    nneed = len(need)

                if nneed == 0 or itercount == maxiter:
                    makemore = False
                itercount += 1

            return samp

        if target in self.fiberflux_fraction.keys():
            fiberflux_fraction = self.fiberflux_fraction[target]
        else:
            fiberflux_fraction = 1.0

        for ii, mm in enumerate(np.atleast_1d(morph)):
            if nobj_morph[ii] > 0:
                # Should really be using north/south GMMs.
                cols = GMM[2][ii]
                samp = np.zeros( nobj, dtype=np.dtype( [(tt, 'f4') for tt in cols] ) )

                # Iterate to make sure the sampled objects pass color-cuts! 
                if len(north) > 0:
                    samp[north] = _samp_iterate(samp[north], target=target, south=False, rand=rand,
                                                colorcuts_function=colorcuts_function,
                                                fiberflux_fraction=fiberflux_fraction)
                if len(south) > 0:
                    samp[south] = _samp_iterate(samp[south], target=target, south=True, rand=rand,
                                                colorcuts_function=colorcuts_function)
                    
                # Choose samples with the appropriate magnitude-dependent
                # probability, for this morphological type.
                prob = np.interp(samp[cols[0]], magbins, frac2d_magbins[ii, :])
                prob /= np.sum(prob)
                if np.count_nonzero(prob) < nobj_morph[ii]:
                    these = rand.choice(nobj, size=nobj_morph[ii], replace=False)
                else:
                    these = rand.choice(nobj, size=nobj_morph[ii], p=prob, replace=False)

                gthese = np.arange(nobj_morph[ii]) + np.sum(nobj_morph[:ii])

                if 'z' in samp.dtype.names:
                    gmmout['MAG'][gthese] = samp['z'][these]
                elif 'g' in samp.dtype.names:
                    gmmout['MAG'][gthese] = samp['g'][these]
                else:
                    gmmout['MAG'][gthese] = samp['r'][these]

                if 'zw1' in samp.dtype.names:
                    gmmout['ZW1'][gthese] = samp['zw1'][these]
                if 'w1w2' in samp.dtype.names:
                    gmmout['W1W2'][gthese] = samp['w1w2'][these]
                
                gmmout['GR'][gthese] = samp['gr'][these]
                gmmout['RZ'][gthese] = samp['rz'][these]
                gmmout['TYPE'][gthese] = np.repeat(mm, nobj_morph[ii])

                for col in ('reff', 'e1', 'e2'):
                    sampcol = '{}_{}'.format(col, mm.lower()) # e.g., reff_dev
                    sampsnrcol = 'snr_{}'.format(sampcol)     # e.g., snr_reff_dev

                    outcol = 'shape{}_{}'.format(mm.lower().replace('rex', 'exp'), col.replace('reff', 'r')).upper()
                    outivarcol = '{}_ivar'.format(outcol).upper()
                    if sampcol in samp.dtype.names:
                        val = samp[sampcol][these]
                        if col == 'reff':
                            val = 10**val
                        gmmout[outcol][gthese] = val
                        gmmout[outivarcol][gthese] = (10**samp[sampsnrcol][these] / val)**2 # S/N-->ivar

        # Assign filter names.
        if np.sum(isouth) > 0:
            if target == 'LRG':
                gmmout['MAGFILTER'][isouth] = np.repeat('decam2014-z', np.sum(isouth))
            elif target == 'ELG':
                gmmout['MAGFILTER'][isouth] = np.repeat('decam2014-g', np.sum(isouth))
            else:
                gmmout['MAGFILTER'][isouth] = np.repeat('decam2014-r', np.sum(isouth))

        if np.sum(~isouth) > 0:
            if target == 'LRG':
                gmmout['MAGFILTER'][~isouth] = np.repeat('MzLS-z', np.sum(~isouth))
            elif target == 'ELG':
                gmmout['MAGFILTER'][~isouth] = np.repeat('BASS-g', np.sum(~isouth))
            else:
                gmmout['MAGFILTER'][~isouth] = np.repeat('BASS-r', np.sum(~isouth))

        # Sort based on the input/prior magnitude (e.g., for the BGS/MXXL
        # mocks), but note that we will very likely end up with duplicated
        # morphologies and colors.
        if prior_mag is not None:
            dmcut = 0.3
            srt = np.zeros(nobj).astype(int)
            for ii, mg in enumerate(prior_mag):
                dm = np.where( (np.abs(mg-gmmout['MAG']) < dmcut) )[0]
                if len(dm) == 0:
                    srt[ii] = np.argmin(np.abs(mg-gmmout['MAG']))
                else:
                    srt[ii] = rand.choice(dm)
            for key in gmmout.keys():
                gmmout[key][:] = gmmout[key][srt]
            # Remove these keys to preserve the values assigned in the reader
            # (e.g., ReadMXXL) class.
            [gmmout.pop(key) for key in ('MAG', 'MAGFILTER')]

        # Shuffle based on input/prior redshift, so we can get a broad
        # correlation between magnitude and redshift.
        if prior_redshift is not None:
            pass
            #dat = np.zeros(nobj, dtype=[('redshift', 'f4'), ('mag', 'f4')])
            #dat['redshift'] = prior_redshift
            #dat['mag'] = gmmout['MAG']
            #srt = np.argsort(dat, order=('redshift', 'mag'))

        return gmmout

    def KDTree_rescale(self, matrix, south=False, subtype=''):
        """Normalize input parameters to [0, 1]."""
        nobj, ndim = matrix.shape
        if subtype == '':
            try:
                # no north-south split (e.g., BGS/MXXL)
                param_min = self.param_min
                param_range = self.param_range
            except:
                if south:
                    param_min = self.param_min_south
                    param_range = self.param_range_south
                else:
                    param_min = self.param_min_north
                    param_range = self.param_range_north
        else:
            if subtype.upper() == 'DA':
                param_min = self.param_min_da
                param_range = self.param_range_da
            elif subtype.upper() == 'DB':
                param_min = self.param_min_db
                param_range = self.param_range_db
            else:
                log.warning('Unrecognized SUBTYPE {}!'.format(subtype))
                raise ValueError
                
        return ( (matrix - np.tile(param_min, nobj).reshape(nobj, ndim)) /
                 np.tile( param_range, nobj).reshape(nobj, ndim) )
        
    def KDTree_build(self, matrix, south=True, subtype=''):
        """Build a KD-tree."""
        from scipy.spatial import cKDTree as KDTree
        return KDTree( self.KDTree_rescale(matrix, south=south, subtype=subtype) )

    def KDTree_query(self, matrix, return_dist=False, south=True, subtype=''):
        """Return the nearest template number based on the KD Tree."""

        matrix_rescaled = self.KDTree_rescale(matrix, south=south, subtype=subtype)
        
        if subtype == '':
            try:
                # no north-south split (e.g., BGS/MXXL)
                dist, indx = self.KDTree.query( matrix_rescaled ) 
            except:
                if south:
                    dist, indx = self.KDTree_south.query( matrix_rescaled )
                else:
                    dist, indx = self.KDTree_north.query( matrix_rescaled )
        else:
            if subtype.upper() == 'DA':
                dist, indx = self.KDTree_da.query( matrix_rescaled )
            elif subtype.upper() == 'DB':
                dist, indx = self.KDTree_db.query( matrix_rescaled )
            else:
                log.warning('Unrecognized SUBTYPE {}!'.format(subtype))
                raise ValueError

        if return_dist:
            return dist, indx.astype('i4')
        else:
            return indx.astype('i4')
        
    def _nospectra_photometry(self, meta, rand, data, indx, target_name, contaminants=False):
        """Populate the photometry in meta in no-spectra mode."""

        nobj = len(indx)

        def _g_and_z(rmag, gr, rz):
            zmag = rmag - rz
            gmag = gr + rmag
            return gmag, zmag
        
        def _r_and_z(gmag, gr, rz):
            rmag = gmag - gr
            zmag = rmag - rz
            return rmag, zmag

        def _g_and_r(zmag, gr, rz):
            rmag = rz + zmag
            gmag = gr + rmag
            return gmag, rmag

        gr, rz = data['GR'][indx], data['RZ'][indx]

        #if target_name == 'QSO':
        #    if contaminants:
        #        gmag = normmag = data['GMAG'][indx]
        #        magfilter = data['MAGFILTER-G'][indx]
        #    else:
        #        gmag = normmag = data['MAG'][indx]
        #        magfilter = data['MAGFILTER'][indx]
        #    rmag, zmag = _r_and_z(gmag, gr, rz)
            
        if target_name == 'LRG':
            if contaminants:
                zmag = normmag = data['ZMAG'][indx]
                magfilter = data['MAGFILTER-Z'][indx]
            else:
                zmag = normmag = data['MAG'][indx]
                magfilter = data['MAGFILTER'][indx]
            gmag, rmag = _g_and_r(zmag, gr, rz)
        elif target_name == 'ELG':
            if contaminants:
                gmag = normmag = data['GMAG'][indx]
                magfilter = data['MAGFILTER-G'][indx]
            else:
                gmag = normmag = data['MAG'][indx]
                magfilter = data['MAGFILTER'][indx]
            rmag, zmag = _r_and_z(gmag, gr, rz)
        else:
            rmag = normmag = data['MAG'][indx]
            magfilter = data['MAGFILTER'][indx]
            gmag, zmag = _g_and_z(rmag, gr, rz)

        W1mag = zmag - data['ZW1'][indx]
        W2mag = W1mag - data['W1W2'][indx]
        
        meta['SEED'][:] = rand.randint(2**31, size=nobj)
        meta['REDSHIFT'][:] = data['Z'][indx]

        meta['MAG'][:] = normmag
        meta['MAGFILTER'][:] = magfilter
        meta['FLUX_G'][:] = 1e9 * 10**(-0.4 * gmag)
        meta['FLUX_R'][:] = 1e9 * 10**(-0.4 * rmag)
        meta['FLUX_Z'][:] = 1e9 * 10**(-0.4 * zmag)
        meta['FLUX_W1'][:] = 1e9 * 10**(-0.4 * W1mag)
        meta['FLUX_W2'][:] = 1e9 * 10**(-0.4 * W2mag)
        
    def get_fiberfraction(self, targets, south=True, ref_seeing=1.0, ref_lambda=5500.0):
        """Estimate the fraction of the integrated flux that enters the fiber.

        Assume a reference seeing value (seeingref) of 1.0 arcsec FWHM at
        a reference wavelength (lambdaref) of 5500 Angstrom.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        south : :class:`bool`
            True for sources with DECaLS photometry and False for sources with
            BASS+MzLS photometry.
        ref_seeing : :class:`float`
            Reference seeing FWHM in arcsec.  Defaults to 1.0.
        ref_lambda : :class:`float`
            Reference wavelength in Angstrom.  Defaults to 5500 A.

        Returns
        -------
        fiberfraction_g : :class:`numpy.ndarray`
            Fraction of the total g-band flux entering the fiber.
        fiberfraction_r : :class:`numpy.ndarray`
            Fraction of the total r-band flux entering the fiber.
        fiberfraction_z : :class:`numpy.ndarray`
            Fraction of the total z-band flux entering the fiber.
    
        Raises
        ------
        ValueError
            If fiberfraction is outside the bounds [0-1] (inclusive).

        """
        ntarg = len(targets)
        fiberfraction_g = np.zeros(ntarg).astype('f4')
        fiberfraction_r, fiberfraction_z = np.zeros_like(fiberfraction_g), np.zeros_like(fiberfraction_g)

        if south:
            lambdafilts = self.decamwise.effective_wavelengths[:4].value # [Angstrom]
        else:
            lambdafilts = self.bassmzlswise.effective_wavelengths[:4].value # [Angstrom]

        # Not quite right to use a bulge-like surface-brightness profile for SER.
        type2source = {'PSF': 'POINT', 'REX': 'DISK', 'EXP': 'DISK',
                       'DEV': 'BULGE', 'SER': 'BULGE'}

        for morphtype in ('PSF', 'REX', 'EXP', 'DEV', 'SER'):
            istype = targets['TYPE'] == morphtype
            if np.sum(istype) > 0:
                # Assume the radius is independent of wavelength.
                reff = targets['SHAPEEXP_R'][istype].data
                offset = np.zeros( np.sum(istype) ) # fiber offset [um]

                for band, lambdafilt, fiberfraction in zip( ('G', 'R', 'Z'), lambdafilts,
                                                        (fiberfraction_g, fiberfraction_r, fiberfraction_z) ):
                    sigma_um = np.repeat( ref_seeing * (lambdafilt / ref_lambda)**(-1.0 / 5.0) /
                                          2.35482 * self.plate_scale_arcsec2um, np.sum(istype) ) # [um]
                    fiberfraction[istype] = self.FFA.value(type2source[morphtype], sigma_um, offset, hlradii=reff)

        # Sanity check.
        if np.sum( (fiberfraction_r < 0) * (fiberfraction_r > 1) ) > 0:
            log.warning('FIBERFRACTION should be [0-1].')
            raise ValueError

        # Put a floor of 5% (otherwise would be zero for large objects).
        for fiberfraction in (fiberfraction_g, fiberfraction_r, fiberfraction_z):
            zero = fiberfraction == 0
            if np.sum(zero) > 0:
                fiberfraction[zero] = 0.05

        return fiberfraction_g, fiberfraction_r, fiberfraction_z

    def populate_targets_truth(self, flux, data, meta, objmeta, indx=None,
                               seed=None, use_simqso=True, truespectype='',
                               templatetype='', templatesubtype=''):
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
        use_simqso : :class:`bool`, optional
            Initialize a SIMQSO-style objtruth table. Defaults to True.
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
        objtruth : :class:`astropy.table.Table`
            Corresponding objtype-specific truth table (if applicable).

        """
        if seed is None:
            seed = self.seed
            
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        # Initialize the tables.
        targets = empty_targets_table(nobj)
        truth, objtruth = empty_truth_table(nobj, templatetype=templatetype,
                                            use_simqso=use_simqso)

        truth['MOCKID'][:] = data['MOCKID'][indx]
        if len(objtruth) > 0:
            if 'Z_NORSD' in data.keys() and 'TRUEZ_NORSD' in objtruth.colnames:
                objtruth['TRUEZ_NORSD'][:] = data['Z_NORSD'][indx]
            if 'OIIFLUX' in data.keys() and 'OIIFLUX' in objtruth.colnames: # ELGs
                objtruth['OIIFLUX'][:] = data['OIIFLUX'][indx]

        # Copy all information from DATA to TARGETS.
        for key in data.keys():
            if key in targets.colnames:
                if isinstance(data[key], np.ndarray):
                    targets[key][:] = data[key][indx]
                else:
                    targets[key][:] = np.repeat(data[key], nobj)

        # Assign RELEASE, PHOTSYS, [RA,DEC]_IVAR, and DCHISQ
        targets['RELEASE'][:] = 9999

        isouth = self.is_south(targets['DEC'])
        south = np.where( isouth )[0]
        north = np.where( ~isouth )[0]
        if len(south) > 0:
            targets['PHOTSYS'][south] = 'S'
        if len(north) > 0:
            targets['PHOTSYS'][north] = 'N'
            
        targets['RA_IVAR'][:], targets['DEC_IVAR'][:] = 1e8, 1e8
        targets['DCHISQ'][:] = np.tile( [0.0, 100, 200, 300, 400], (nobj, 1)) # for QSO selection

        # Add dust, depth, and nobs.
        for band in ('G', 'R', 'Z', 'W1', 'W2', 'W3', 'W4'):
            key = 'MW_TRANSMISSION_{}'.format(band)
            targets[key][:] = data[key][indx]

        for band in ('G', 'R', 'Z'):
            for prefix in ('PSF', 'GAL'):
                key = '{}DEPTH_{}'.format(prefix, band)
                targets[key][:] = data[key][indx]
            nobskey = 'NOBS_{}'.format(band)
            targets[nobskey][:] = 2 # assume constant!

        # Add spectral / template type and subtype.
        for value, key in zip( (truespectype, templatetype, templatesubtype),
                               ('TRUESPECTYPE', 'TEMPLATETYPE', 'TEMPLATESUBTYPE') ):
            if isinstance(value, np.ndarray):
                truth[key][:] = value
            else:
                truth[key][:] = np.repeat(value, nobj)

        # Copy various quantities from the metadata table.
        for key in meta.colnames:
            if key in truth.colnames:
                truth[key][:] = meta[key]
            elif key == 'REDSHIFT':
                truth['TRUEZ'][:] = meta['REDSHIFT']

        if len(objmeta) > 0 and len(objtruth) > 0: # some objects have no metadata...
            for key in objmeta.colnames:
                if key in objtruth.colnames:
                    #The modified emision line model don't have the same shape as the default.
                    if (key=='EMLINES') and (objtruth[key][:].shape != objmeta[key].shape) :
                        objtruth.replace_column('EMLINES',np.zeros((nobj,len(objmeta['EMLINES'][0, :, 0]), 3))-1)
                    objtruth[key][:] = objmeta[key]
            
        # Scatter the observed photometry based on the depth and then attenuate
        # for Galactic extinction.
        self.scatter_photometry(data, truth, targets, indx=indx, seed=seed)

        for band, key in zip( ('G', 'R', 'Z', 'W1', 'W2', 'W3', 'W4'),
                              ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1',
                               'FLUX_W2', 'FLUX_W3', 'FLUX_W4') ):
            targets[key][:] = targets[key] * data['MW_TRANSMISSION_{}'.format(band)][indx]

        # Attenuate the spectra for extinction, too.
        if len(flux) > 0 and 'EBV' in data.keys():
            flux *= 10**( -0.4 * data['EBV'][indx, np.newaxis] * self.extinction )

        # Finally compute the (simulated, observed) flux within the fiber.
        for these, issouth in zip( (north, south), (False, True) ):
            if len(these) > 0:
                fiberfraction = self.get_fiberfraction(targets[these], south=issouth)
                for band, fraction in zip( ('G', 'R', 'Z'), fiberfraction ):
                    fiberflux = targets['FLUX_{}'.format(band)][these] * fraction

                    targets['FIBERFLUX_{}'.format(band)][these] = fiberflux
                    targets['FIBERTOTFLUX_{}'.format(band)][these] = fiberflux

        return targets, truth, objtruth

    def mock_density(self, mockfile=None, nside=64, density_per_pixel=False,
                     zmax_qso=None, zmin_lya=None):
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
        zmax_qso : :class:`float`
            Maximum redshift of tracer QSOs to read, to ensure no
            double-counting with Lya mocks.  Defaults to None.
        zmin_lya : :class:`float`
            Minimum redshift of Lya skewers, to ensure no double-counting with
            QSO mocks.  Defaults to None.

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

        # This is a little fragile because it makes assumptions about the data
        # model.
        if zmax_qso is not None:
            radec = fitsio.read(mockfile, columns=['RA', 'DEC', 'Z_COSMO', 'DZ_RSD'],
                                upper=True, ext=1)
            zz = (radec['Z_COSMO'].astype('f8') + radec['DZ_RSD'].astype('f8')).astype('f4')
            cut = np.where( zz < zmax_qso )[0]
            radec = radec[cut]
        elif zmin_lya is not None:
            radec = fitsio.read(mockfile, columns=['RA', 'DEC', 'Z_QSO_RSD'],
                                upper=True, ext=1)
            zz = radec['Z_QSO_RSD'].astype('f4')
            cut = np.where( zz >= zmin_lya )[0]
            radec = radec[cut]
        else:
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
            basemap = init_sky(galactic_plane_color='k', ax=ax[0])
            plot_sky_binned(data['RA'], data['DEC'], weights=data['WEIGHT'],
                            max_bin_area=hp.nside2pixarea(data['NSIDE'], degrees=True),
                            verbose=False, clip_lo='!1', clip_hi='95%', 
                            cmap='viridis', plot_type='healpix', basemap=basemap,
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

    def is_south(self, dec):
        """Divide the "north" and "south" photometric systems based on a
        constant-declination cut.

        Parameters
        ----------
        dec : :class:`numpy.ndarray`
            Declination of candidate targets (decimal degrees). 

        """
        from desitarget.io import desitarget_resolve_dec
        return dec <= desitarget_resolve_dec()

    def remove_north_south_bits(self, desi_target, bgs_target, mws_target):
        """Remove all the "north" and "south" targeting bits.  See the discussion here
        for details: https://github.com/desihub/desitarget/pull/426

        Parameters
        ----------
        desi_target : :class:`int64`
            Dark-time targeting bit from targetmask.yaml.
        bgs_target : :class:`int64`
            BGS targeting bit from targetmask.yaml.
        mws_target : :class:`int64`
            MWS targeting bit from targetmask.yaml.

        """
        for category, target, mask in zip( ('MWS', 'BGS', 'DESI'),
                                           (mws_target, bgs_target, desi_target), 
                                           (self.mws_mask, self.bgs_mask, self.desi_mask) ):
            for bitname in mask.names():
                if 'SOUTH' in bitname or 'NORTH' in bitname:
                    these = target & mask.mask(bitname) != 0
                    if np.sum(these) > 0:
                        target[these] -= mask.mask(bitname) # subtract that bit
                        # For MWS and BGS, if we subtracted out the last bit
                        # then we also have to unset MWS_ANY and BGS_ANY, respectively.
                        if category == 'MWS' or category == 'BGS':
                            andthose = target[these] == 0
                            if np.sum(andthose) > 0:
                                desi_target[these][andthose] -= self.desi_mask.mask('{}_ANY'.format(category))

class ReadGaussianField(SelectTargets):
    """Read a Gaussian random field style mock catalog."""
    cached_radec = None
    
    def __init__(self, **kwargs):
        super(ReadGaussianField, self).__init__(**kwargs)
        
    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 zmax_qso=None, target_name='', mock_density=False,
                 only_coords=False, seed=None):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        zmax_qso : :class:`float`
            Maximum redshift of tracer QSOs to read, to ensure no
            double-counting with Lya mocks.  Defaults to None.
        target_name : :class:`str`
            Name of the target being read (e.g., ELG, LRG).
        mock_density : :class:`bool`, optional
            Compute and return the median target density in the mock.  Defaults
            to False.
        only_coords : :class:`bool`, optional
            To get some improvement in speed, only read the target coordinates
            and some other basic info.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.

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
                nside = 64
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
                log.debug('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, allpix, pixweight = ReadGaussianField.cached_radec

        mockid = np.arange(len(ra)) # unique ID/row number
        
        fracarea = pixweight[allpix]        
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint
        if np.all(cut[1:] >= cut[:-1]) is False:
            log.fatal('Index cut must be monotonically increasing, otherwise fitsio will resort it!')
            raise ValueError

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s)'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        mockid = mockid[cut]
        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = ra[cut]
        dec = dec[cut]

        # Add redshifts.
        if target_name.upper() == 'SKY':
            zz = np.zeros(len(ra))
        else:
            data = fitsio.read(mockfile, columns=['Z_COSMO', 'DZ_RSD'], upper=True, ext=1, rows=cut)
            zz = (data['Z_COSMO'].astype('f8') + data['DZ_RSD'].astype('f8')).astype('f4')
            zz_norsd = data['Z_COSMO'].astype('f4')

            # cut on maximum redshift
            if zmax_qso is not None:
                cut = np.where( zz < zmax_qso )[0]
                nobj = len(cut)
                log.info('Trimmed to {} objects with z<{:.3f}'.format(nobj, zmax_qso))
                if nobj == 0:
                    return dict()
                mockid = mockid[cut]
                allpix = allpix[cut]
                weight = weight[cut]
                ra = ra[cut]
                dec = dec[cut]
                zz = zz[cut]
                zz_norsd = zz_norsd[cut]

        # Optionally (for a little more speed) only return some basic info. 
        if only_coords:
            return {'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'Z': zz,
                    'WEIGHT': weight, 'NSIDE': nside}

        isouth = self.is_south(dec)

        # Get photometry and morphologies by sampling from the Gaussian
        # mixture models.
        log.info('Sampling from {} Gaussian mixture model.'.format(target_name))
        gmmout = self.sample_GMM(nobj, target=target_name, isouth=isouth,
                                 seed=seed, prior_redshift=zz)

        # For ELGs, also sample to get the [OII] flux for use with --no-spectra.
        # Note that these are the "true" values (before photometric scatter and
        # Galactic extinction).

        # Here's the code used to determine these correlations--
        # import fitsio ; import numpy as np ; import matplotlib.pyplot as plt ; from desisim.util import medxbin
        # ff = fitsio.FITS('/global/projecta/projectdirs/desi/datachallenge/reference_runs/18.11/targets/truth.fits')
        # alltarg = ff['TRUTH'].read(columns=['FLUX_R', 'TARGETID'])
        # elgtarg = ff['TRUTH_ELG'].read(columns=['OIIFLUX', 'TARGETID'])
        # ww = np.isin(alltarg['TARGETID'], elgtarg['TARGETID'])
        # oii = np.log10( 1e17 * elgtarg['OIIFLUX'] )
        # rmag = 22.5 - 2.5 * np.log10( alltarg['FLUX_R'][ww] )
        # 
        # def statsinbins(xx, yy, binsize=0.1, minpts=10):
        #     xmin = xx.min()
        #     xmax = xx.max()
        #     nbin = int( (np.nanmax(xx) - np.nanmin(xx) ) / binsize )
        #     stats = np.zeros(nbin, [
        #         ('xmean', 'f4'), ('xmedian', 'f4'), ('xbin', 'f4'), ('npts', 'i4'),
        #         ('ymedian', 'f4'), ('ymean', 'f4'), ('ystd', 'f4'), ('y25', 'f4'), ('y75', 'f4')])
        #     _xbin = np.linspace(xx.min(), xx.max(), nbin)
        #     idx  = np.digitize(xx, _xbin)
        #     for kk in range(nbin):
        #         these = idx == kk
        #         npts = np.count_nonzero( oii[these] )
        #         stats['xbin'][kk] = _xbin[kk]
        #         stats['npts'][kk] = npts
        #         if npts > 0:
        #             stats['xmedian'][kk] = np.nanmedian( xx[these] )
        #             stats['xmean'][kk] = np.nanmean( xx[these] )
        #             stats['ystd'][kk] = np.nanstd( yy[these] )
        #             stats['ymean'][kk] = np.nanmean( yy[these] )
        #             qq = np.nanpercentile( yy[these], [25, 50, 75] )
        #             stats['y25'][kk] = qq[0]
        #             stats['ymedian'][kk] = qq[1]
        #             stats['y75'][kk] = qq[2]
        #     keep = stats['npts'] > minpts
        #     return stats[keep]
        # stats = statsinbins(rmag, oii, binsize=0.05, minpts=10)
        # rmagpivot = 22
        # coeff_median = np.polyfit(stats['xbin'] - rmagpivot, stats['ymedian'], 2)
        # coeff_scatter = np.polyfit(stats['xbin'] - rmagpivot, stats['ystd'], 2)
        # 
        # fig, ax = plt.subplots()
        # ax.hexbin(rmag, oii, mincnt=1, cmap='viridis', bins='log', gridsize=60)
        # ax.set_xlabel('r (AB mag, no scatter or extinction)')
        # ax.set_ylabel(r'$\log_{10} [{\rm OII}]\ \ (10^{-17}\ {\rm erg}\ {\rm s}^{-1}\ {\rm cm}^{-2})$')
        # ax.plot(stats['xbin'], stats['ymedian'], color='k', ls='-')
        # ax.plot(stats['xbin'], stats['y75'], color='k', ls='--')
        # ax.plot(stats['xbin'], stats['y25'], color='k', ls='--')
        # ax.plot(stats['xbin'], np.polyval(coeff_median, stats['xbin'] - rmagpivot), color='red')
        # fig.savefig('/global/homes/i/ioannis/rmag-vs-oiiflux.png')

        if target_name.upper() == 'ELG' and gmmout is not None:
            rmagpivot = 22
            rand = np.random.RandomState(seed)

            oiicoeff_median = np.array([ -5.31384197e-04, -4.24876618e-01, 1.17861064e+00])
            oiicoeff_scatter = np.array([-0.00188197, -0.00077902, 0.30842156])
            oiiflux = np.polyval(oiicoeff_median, gmmout['MAG'] - rmagpivot)
            oiiflux_sigma = np.polyval(oiicoeff_scatter, gmmout['MAG'] - rmagpivot)
            for ii, sigma in enumerate(oiiflux_sigma):
                if sigma < 0: # edge case
                    sigma = np.polyval(oiicoeff_scatter, 0) 
                oiiflux[ii] += rand.normal(loc=0, scale=sigma)

            gmmout.update({'OIIFLUX': 1e-17 * oiiflux})

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'gaussianfield',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': self.Bricks.brickname(ra, dec).astype('S8'),
               'BRICKID': self.Bricks.brickid(ra, dec),
               'RA': ra, 'DEC': dec, 'Z': zz, 'Z_NORSD': zz_norsd,
               'SOUTH': isouth}
        if gmmout is not None:
            out.update(gmmout)

        # Add MW transmission and the imaging depth.
        self.mw_transmission(out)
        self.imaging_depth(out)

        # Optionally compute the mean mock density.
        if mock_density:
            out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile, zmax_qso=zmax_qso)

        return out

class ReadBuzzard(SelectTargets):
    """Read a Buzzard style mock catalog."""
    cached_pixweight = None
    
    def __init__(self, **kwargs):
        super(ReadBuzzard, self).__init__(**kwargs)
        
    def readmock(self, mockfile=None, healpixels=[], nside=[], nside_buzzard=8,
                 target_name='', magcut=None, only_coords=False, seed=None):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        nside_buzzard : :class:`int`
            Healpixel nside indicating how the mock on-disk has been organized.
            Defaults to 8.
        target_name : :class:`str`
            Name of the target being read (e.g., ELG, LRG).
        magcut : :class:`float`
            Magnitude cut (hard-coded to DECam r-band) to subselect targets
            brighter than magcut. 
        only_coords : :class:`bool`, optional
            To get some improvement in speed, only read the target coordinates
            and some other basic info.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.

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
        
        if nside_buzzard is None:
            log.warning('Nside_buzzard input is required.')
            raise ValueError
        
        mockfile_nside = os.path.join(mockfile, str(nside_buzzard))
        if not os.path.isdir(mockfile_nside):
            log.warning('Buzzard top-level directory {} not found!'.format(mockfile_nside))
            raise IOError

        # Because of the size of the Buzzard mock, healpixels (and nside) must
        # be scalars.
        if len(np.atleast_1d(healpixels)) != 1 and len(np.atleast_1d(nside)) != 1:
            log.warning('Healpixels and nside must be scalar inputs.')
            raise ValueError

        if self.cached_pixweight is None:
            pixweight = load_pixweight(nside, pixmap=self.pixmap)
            ReadBuzzard.cached_pixweight = (pixweight, nside)
        else:
            pixweight, cached_nside = ReadBuzzard.cached_pixweight
            if cached_nside != nside:
                pixweight = load_pixweight(nside, pixmap=self.pixmap)
                ReadBuzzard.cached_pixweight = (pixweight, nside)
            else:
                log.debug('Using cached pixel weight map.')
                pixweight, _ = ReadBuzzard.cached_pixweight

        # Get the set of nside_buzzard pixels that belong to the desired
        # healpixels (which have nside).  This will break if healpixels is a
        # vector.
        theta, phi = hp.pix2ang(nside, healpixels, nest=True)
        pixnum = hp.ang2pix(nside_buzzard, theta, phi, nest=True)

        buzzardfile = findfile(filetype='Buzzard_v1.6_lensed', nside=nside_buzzard, pixnum=pixnum,
                               basedir=mockfile_nside, ext='fits')
        if len(buzzardfile) == 0:
            log.warning('File {} not found!'.format(buzzardfile))
            raise IOError

        log.info('Reading {}'.format(buzzardfile))
        radec = fitsio.read(buzzardfile, columns=['RA', 'DEC'], upper=True, ext=1)
        nobj = len(radec)

        objid = np.arange(nobj)
        mockid = encode_targetid(objid=objid, brickid=pixnum, mock=1)

        allpix = footprint.radec2pix(nside, radec['RA'], radec['DEC'])

        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint
        if np.all(cut[1:] >= cut[:-1]) is False:
            log.fatal('Index cut must be monotonically increasing, otherwise fitsio will resort it!')
            raise ValueError

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s)'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        mockid = mockid[cut]
        objid = objid[cut]
        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = radec['RA'][cut].astype('f8') % 360.0 # enforce 0 < ra < 360
        dec = radec['DEC'][cut].astype('f8')
        del radec

        cols = ['Z', 'TMAG']
        #cols = ['Z', 'COEFFS', 'TMAG']
        data = fitsio.read(buzzardfile, columns=cols, upper=True, ext=1, rows=cut)
        zz = data['Z'].astype('f4')
        tmag = data['TMAG'].astype('f4')

        if magcut:
            cut = tmag[:, 2] < magcut # r-band
            if np.count_nonzero(cut) == 0:
                log.warning('No objects with r < {}!'.format(magcut))
                return dict()
            else:
                mockid = mockid[cut]
                objid = objid[cut]
                allpix = allpix[cut]
                weight = weight[cut]
                ra = ra[cut]
                dec = dec[cut]
                zz = zz[cut]
                tmag = tmag[np.where(cut)[0], :]

                nobj = len(ra)
                log.info('Trimmed to {} {}s with r < {}.'.format(nobj, target_name, magcut))
        
        # Optionally (for a little more speed) only return some basic info. 
        if only_coords:
            return {'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'Z': zz,
                    'WEIGHT': weight, 'NSIDE': nside}

        isouth = self.is_south(dec)

        ## Get photometry and morphologies by sampling from the Gaussian
        ## mixture models.
        #log.info('Sampling from {} Gaussian mixture model.'.format(target_name))
        #gmmout = self.sample_GMM(nobj, target=target_name, isouth=isouth,
        #                         seed=seed, prior_redshift=zz)
        #gmmout = None

        gmag = data['TMAG'][:, 1].astype('f4') # DES g-band, no MW extinction 
        rmag = data['TMAG'][:, 2].astype('f4') # DES r-band, no MW extinction 
        zmag = data['TMAG'][:, 4].astype('f4') # DES z-band, no MW extinction 

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'buzzard',
            'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
            'MOCKID': mockid, 'BRICKNAME': self.Bricks.brickname(ra, dec).astype('S8'),
            'BRICKID': self.Bricks.brickid(ra, dec),
            'RA': ra, 'DEC': dec, 'Z': zz,
            'MAG': rmag, 'MAGFILTER': np.repeat('decam2014-r', nobj),
            'GMAG': gmag, 'MAGFILTER-G': np.repeat('decam2014-g', nobj),
            'ZMAG': zmag, 'MAGFILTER-Z': np.repeat('decam2014-z', nobj),
            'SOUTH': isouth}
            
        #if gmmout is not None:
        #    out.update(gmmout)

        # Add MW transmission and the imaging depth.
        self.mw_transmission(out)
        self.imaging_depth(out)

        ## Optionally compute the mean mock density.
        #if mock_density:
        #    out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile)

        return out

class ReadUniformSky(SelectTargets):
    """Read a uniform sky style mock catalog."""
    cached_radec = None
    
    def __init__(self, **kwargs):
        super(ReadUniformSky, self).__init__(**kwargs)

    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 target_name='', mock_density=False, only_coords=False):
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
        only_coords : :class:`bool`, optional
            To get some improvement in speed, only read the target coordinates
            and some other basic info.  Defaults to False.

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
                nside = 64
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
                log.debug('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, allpix, pixweight = ReadUniformSky.cached_radec

        mockid = np.arange(len(ra)) # unique ID/row number

        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint
        if np.all(cut[1:] >= cut[:-1]) is False:
            log.fatal('Index cut must be monotonically increasing, otherwise fitsio will resort it!')
            raise ValueError

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

        # Optionally (for a little more speed) only return some basic info. 
        if only_coords:
            return {'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'Z': zz,
                    'WEIGHT': weight, 'NSIDE': nside}

        isouth = self.is_south(dec)

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'uniformsky',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': self.Bricks.brickname(ra, dec).astype('S8'),
               'BRICKID': self.Bricks.brickid(ra, dec),
               'RA': ra, 'DEC': dec, 'Z': np.zeros(len(ra)),
               'SOUTH': isouth}

        # Add MW transmission and the imaging depth.
        self.mw_transmission(out)
        self.imaging_depth(out)

        # Optionally compute the mean mock density.
        if mock_density:
            out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile)

        return out

class ReadGalaxia(SelectTargets):
    """Read a Galaxia style mock catalog."""
    cached_pixweight = None

    def __init__(self, **kwargs):
        super(ReadGalaxia, self).__init__(**kwargs)

    def readmock(self, mockfile=None, healpixels=[], nside=[], nside_galaxia=8, 
                 target_name='MWS_MAIN', magcut=None, faintstar_mockfile=None,
                 faintstar_magcut=None, seed=None):
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
        faintstar_mockfile : :class:`str`, optional
            Full path to the top-level directory of the Galaxia faint star mock
            catalog.
        faintstar_magcut : :class:`float`, optional
            Magnitude cut (hard-coded to SDSS r-band) to subselect faint star
            targets brighter than magcut.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.

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

        rand = np.random.RandomState(seed)
        
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
                log.debug('Using cached pixel weight map.')
                pixweight, _ = ReadGalaxia.cached_pixweight

        # Get the set of nside_galaxia pixels that belong to the desired
        # healpixels (which have nside).  This will break if healpixels is a
        # vector.
        theta, phi = hp.pix2ang(nside, healpixels, nest=True)
        pixnum = hp.ang2pix(nside_galaxia, theta, phi, nest=True)

        if target_name.upper() == 'MWS_MAIN' or target_name.upper() == 'CONTAM_STAR':
            filetype = 'mock_allsky_galaxia_desi'
        elif target_name.upper() == 'FAINTSTAR':
            filetype = 'mock_superfaint_allsky_galaxia_desi_b10'
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
        if np.all(cut[1:] >= cut[:-1]) is False:
            log.fatal('Index cut must be monotonically increasing, otherwise fitsio will resort it!')
            raise ValueError

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s)'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        mockid = mockid[cut]
        objid = objid[cut]
        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = radec['RA'][cut].astype('f8') % 360.0 # enforce 0 < ra < 360
        dec = radec['DEC'][cut].astype('f8')
        del radec

        # Only the MWS_MAIN mock has Gaia and TRUE_VHELIO.
        cols = ['TRUE_MAG_R_SDSS_NODUST', 'TRUE_MAG_G_SDSS_NODUST', 'TRUE_MAG_Z_SDSS_NODUST',
                'TRUE_TEFF', 'TRUE_LOGG', 'TRUE_FEH']
        if target_name.upper() == 'MWS_MAIN' or target_name.upper() == 'CONTAM_STAR':
            cols = cols + ['TRUE_VHELIO', 'GAIA_PHOT_G_MEAN_MAG', 'PMRA', 'PMDEC', 'PM_RA_IVAR',
                           'PM_DEC_IVAR', 'PARALLAX', 'PARALLAX_IVAR']
        data = fitsio.read(galaxiafile, columns=cols, upper=True, ext=1, rows=cut)
        mag = data['TRUE_MAG_R_SDSS_NODUST'].astype('f4')  # SDSS r-band, extinction-corrected
        gmag = data['TRUE_MAG_G_SDSS_NODUST'].astype('f4') # SDSS g-band, extinction-corrected
        zmag = data['TRUE_MAG_Z_SDSS_NODUST'].astype('f4') # SDSS z-band, extinction-corrected
        teff = 10**data['TRUE_TEFF'].astype('f4')          # log10!
        logg = data['TRUE_LOGG'].astype('f4')
        feh = data['TRUE_FEH'].astype('f4')

        if target_name.upper() == 'MWS_MAIN' or target_name.upper() == 'CONTAM_STAR':
            zz = (data['TRUE_VHELIO'].astype('f4') / C_LIGHT).astype('f4')
        else:
            zz = (rand.normal(loc=0.0, scale=200.0, size=len(data)) / C_LIGHT).astype('f4') # Hack!

        if target_name.upper() == 'MWS_MAIN' or target_name.upper() == 'CONTAM_STAR':
            ref_id = mockid
            ref_cat = np.repeat('G2', len(mockid))
            gaia_g = data['GAIA_PHOT_G_MEAN_MAG'].astype('f4')
            gaia_pmra = data['PMRA'].astype('f4')
            gaia_pmdec = data['PMDEC'].astype('f4')
            gaia_pmra_ivar = data['PM_RA_IVAR'].astype('f4')
            gaia_pmdec_ivar = data['PM_DEC_IVAR'].astype('f4')
            gaia_parallax = data['PARALLAX'].astype('f4')
            gaia_parallax_ivar = data['PARALLAX_IVAR'].astype('f4')
        else:
            ref_id = np.zeros(nobj).astype('f4')-1 # no data is -1
            ref_cat = np.repeat('', len(mockid))
            gaia_g = np.zeros(nobj).astype('f4')
            gaia_pmra = np.zeros(nobj).astype('f4')
            gaia_pmdec = np.zeros(nobj).astype('f4')
            gaia_pmra_ivar = np.ones(nobj).astype('f4')  # default is unity
            gaia_pmdec_ivar = np.ones(nobj).astype('f4') # default is unity
            gaia_parallax = np.zeros(nobj).astype('f4')
            gaia_parallax_ivar = np.ones(nobj).astype('f4') # default is unity

        if magcut:
            cut = mag < magcut
            if np.count_nonzero(cut) == 0:
                log.warning('No objects with r < {}!'.format(magcut))
                return dict()
            else:
                mockid = mockid[cut]
                objid = objid[cut]
                allpix = allpix[cut]
                weight = weight[cut]
                ra = ra[cut]
                dec = dec[cut]
                zz = zz[cut]
                mag = mag[cut]
                gmag = gmag[cut]
                zmag = zmag[cut]
                teff = teff[cut]
                logg = logg[cut]
                feh = feh[cut]

                ref_id = ref_id[cut]
                ref_cat = ref_cat[cut]
                gaia_g = gaia_g[cut]
                gaia_pmra = gaia_pmra[cut]
                gaia_pmdec = gaia_pmdec[cut]
                gaia_pmra_ivar = gaia_pmra_ivar[cut]
                gaia_pmdec_ivar = gaia_pmdec_ivar[cut]
                gaia_parallax = gaia_parallax[cut]
                gaia_parallax_ivar = gaia_parallax_ivar[cut]
                
                nobj = len(ra)
                log.info('Trimmed to {} {}s with r < {}.'.format(nobj, target_name, magcut))

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'galaxia',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': self.Bricks.brickname(ra, dec).astype('S8'),
               'BRICKID': self.Bricks.brickid(ra, dec),
               'RA': ra, 'DEC': dec, 'Z': zz,
               'TEFF': teff, 'LOGG': logg, 'FEH': feh,
               'MAG': mag, 'MAGFILTER': np.repeat('sdss2010-r', nobj),
               #'GMAG': gmag,
               'ZMAG': zmag,
               
               'SOUTH': self.is_south(dec), 'TYPE': np.repeat('PSF', nobj),

               'REF_ID': ref_id,
               'REF_CAT': ref_cat,
               'GAIA_PHOT_G_MEAN_MAG': gaia_g,
               #'GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR' - f4
               'GAIA_PHOT_BP_MEAN_MAG': np.zeros(nobj).astype('f4'), # placeholder
               #'GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR' - f4
               'GAIA_PHOT_RP_MEAN_MAG': np.zeros(nobj).astype('f4'), # placeholder
               #'GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR' - f4
               'GAIA_ASTROMETRIC_EXCESS_NOISE': np.zeros(nobj).astype('f4'), # placeholder
               #'GAIA_DUPLICATED_SOURCE' - b1 # default is False
               'GAIA_ASTROMETRIC_PARAMS_SOLVED': np.zeros(nobj).astype('i8')+31, # good astrometry
               'PARALLAX': gaia_parallax,
               'PARALLAX_IVAR': gaia_parallax_ivar,
               'PMRA': gaia_pmra,
               'PMRA_IVAR': gaia_pmra_ivar,
               'PMDEC': gaia_pmdec,
               'PMDEC_IVAR': gaia_pmdec_ivar}

        # Add MW transmission and the imaging depth.
        self.mw_transmission(out)
        self.imaging_depth(out)

        # Optionally include faint stars.
        if faintstar_mockfile is not None:
            log.debug('Supplementing with FAINTSTAR mock targets.')
            faintdata = ReadGalaxia().readmock(mockfile=faintstar_mockfile, target_name='FAINTSTAR',
                                               healpixels=healpixels, nside=nside,
                                               nside_galaxia=nside_galaxia, magcut=faintstar_magcut,
                                               seed=seed)
            
            # Stack and shuffle so we get a mix of bright and faint stars.
            rand = np.random.RandomState(seed)
            newnobj = nobj + len(faintdata['RA'])
            newindx = rand.choice(newnobj, size=newnobj, replace=False)

            for key in out.keys():
                if type(out[key]) == np.ndarray:
                    out[key] = np.hstack( (out[key], faintdata[key]) )[newindx]

            del faintdata

        return out

class ReadLyaCoLoRe(SelectTargets):
    """Read a CoLoRe mock catalog of Lya skewers."""
    def __init__(self, **kwargs):
        super(ReadLyaCoLoRe, self).__init__(**kwargs)

    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 target_name='LYA', nside_lya=16, zmin_lya=None,
                 mock_density=False, sqmodel='default',only_coords=False, seed=None):
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
        zmin_lya : :class:`float`
            Minimum redshift of Lya skewers, to ensure no double-counting with
            QSO mocks.  Defaults to None.
        mock_density : :class:`bool`, optional
            Compute and return the median target density in the mock.  Defaults
            to False.
        only_coords : :class:`bool`, optional
            Only read the target coordinates and some other basic info.
            Defaults to False.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.

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
        from astropy import cosmology
        try:
            from simqso.sqbase import ContinuumKCorr
            if sqmodel == 'default':
                from simqso.sqmodels import BOSS_DR9_PLEpivot as model_PLEpivot
            else:
                from desisim.scripts.lya_simqso_model import model_PLEpivot

            from simqso.sqgrids import generateQlfPoints
        except ImportError:
            message = 'Please install https://github.com/imcgreer/simqso'
            log.error(message)
            raise(ImportError(message))
            
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
                nside = 64
            log.info('Reading the whole DESI footprint with nside = {}.'.format(nside))
            healpixels = footprint.tiles2pix(nside)
        if nside is None:
            log.warning('Nside must be a scalar input.')
            raise ValueError

        pixweight = load_pixweight(nside, pixmap=self.pixmap)

        # Read the ra,dec coordinates and then restrict to the desired
        # healpixels.
        log.info('Reading {}'.format(mockfile))
        try: # new data model
            tmp = fitsio.read(mockfile, columns=['RA', 'DEC', 'MOCKID', 'Z_QSO_RSD',
                                                 'Z_QSO_NO_RSD', 'PIXNUM'],
                              upper=True, ext=1)
            zz = tmp['Z_QSO_RSD'].astype('f4')
            zz_norsd = tmp['Z_QSO_NO_RSD'].astype('f4')
        except: # old data model
            tmp = fitsio.read(mockfile, columns=['RA', 'DEC', 'MOCKID' ,'Z', 'PIXNUM'],
                              upper=True, ext=1)
            zz = tmp['Z'].astype('f4')
            zz_norsd = tmp['Z'].astype('f4')
    
        ra = tmp['RA'].astype('f8') % 360.0 # enforce 0 < ra < 360
        dec = tmp['DEC'].astype('f8')            
        mockpix = tmp['PIXNUM']
        mockid = (tmp['MOCKID'].astype(float)).astype(int)
            
        del tmp

        log.info('Assigning healpix pixels with nside = {}'.format(nside))
        allpix = footprint.radec2pix(nside, ra, dec)

        fracarea = pixweight[allpix]
        # force DESI footprint
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0]
        if np.all(cut[1:] >= cut[:-1]) is False:
            log.fatal('Index cut must be monotonically increasing, otherwise fitsio will resort it!')
            raise ValueError

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()

        log.info('Trimmed to {} {}s in {} healpixel(s)'.format(
            nobj, target_name, len(np.atleast_1d(healpixels))))

        allpix = allpix[cut]
        weight = 1 / fracarea[cut]
        ra = ra[cut]
        dec = dec[cut]
        zz = zz[cut]
        zz_norsd = zz_norsd[cut]
        #objid = objid[cut]
        mockpix = mockpix[cut]
        mockid = mockid[cut]

        # Cut on minimum redshift.
        if zmin_lya is not None:
            cut = np.where( zz >= zmin_lya )[0]
            nobj = len(cut)
            log.info('Trimmed to {} {}s with z>={:.3f}'.format(nobj, target_name, zmin_lya))
            if nobj == 0:
                return dict()
            allpix = allpix[cut]
            weight = weight[cut]
            ra = ra[cut]
            dec = dec[cut]
            zz = zz[cut]
            zz_norsd = zz_norsd[cut]
            #objid = objid[cut]
            mockpix = mockpix[cut]
            mockid = mockid[cut]

        # Optionally (for a little more speed) only return some basic info. 
        if only_coords:
            return {'MOCKID': mockid, 'RA': ra, 'DEC': dec, 'Z': zz,
                    'WEIGHT': weight, 'NSIDE': nside}

        # Build the full filenames.
        lyafiles = []
        for mpix in mockpix:
            lyafiles.append("%s/%d/%d/transmission-%d-%d.fits.gz"%(
                mockdir, mpix//100, mpix, nside_lya, mpix))
        isouth = self.is_south(dec)

        # Draw apparent magnitudes from an BOSS/DR9 QSO luminosity function
        # (code taken from also desisim.templates.SIMQSO).
        #
        #   from speclite import filters
        #   print(filters.load_filters('BASS-r').effective_wavelengths.value,
        #         filters.load_filters('decam2014-r').effective_wavelengths.value)

        magrange = (17, 23.5)
        zrange = (np.min(zz), np.max(zz))
        
        normfilter_north, normfilter_south = 'BASS-r', 'DECam-r'
        weff_normfilter_north, weff_normfilter_south = 6437.79282937, 6469.62203811
        
        kcorr_north = ContinuumKCorr(normfilter_north, 1450, effWaveBand=weff_normfilter_north)
        kcorr_south = ContinuumKCorr(normfilter_south, 1450, effWaveBand=weff_normfilter_south)
        qlf = model_PLEpivot(cosmo=cosmology.core.FlatLambdaCDM(70.0, 0.3))

        mag = np.zeros(nobj).astype('f4')
        magfilter = np.zeros(nobj).astype('S15')

        south = np.where(isouth)[0]
        north = np.where(~isouth)[0]
        for these, issouth in zip( (north, south), (False, True) ):
            if len(these) > 0:
                if issouth:
                    qsometa = generateQlfPoints(qlf, magrange, zrange, zin=zz[these],
                                                kcorr=kcorr_south, qlfseed=seed,
                                                gridseed=seed)
                    mag[these] = qsometa.data['appMag']
                    magfilter[these] = normfilter_north
                else:
                    qsometa = generateQlfPoints(qlf, magrange, zrange, zin=zz[these],
                                                kcorr=kcorr_north, qlfseed=seed,
                                                gridseed=seed)
                    mag[these] = qsometa.data['appMag']
                    magfilter[these] = normfilter_south
        # Get photometry and morphologies by sampling from the Gaussian
        # mixture models.
        log.info('Sampling from {} Gaussian mixture model.'.format(target_name))
        gmmout = self.sample_GMM(nobj, target='QSO', isouth=isouth,
                                 seed=seed, prior_redshift=zz, prior_mag=mag)

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'CoLoRe',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               #'OBJID': objid,
               'MOCKID': mockid, 'LYAFILES': np.array(lyafiles),
               'BRICKNAME': self.Bricks.brickname(ra, dec).astype('S8'),
               'BRICKID': self.Bricks.brickid(ra, dec),
               'RA': ra, 'DEC': dec, 'Z': zz, 'Z_NORSD': zz_norsd,
               'MAG': mag, 'MAGFILTER': magfilter,
               'SOUTH': isouth}
        if gmmout is not None:
            out.update(gmmout)

        # Add MW transmission and the imaging depth.
        self.mw_transmission(out)
        self.imaging_depth(out)

        # Optionally compute the mean mock density.
        if mock_density:
            out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile, zmin_lya=zmin_lya)

        return out

class ReadMXXL(SelectTargets):
    """Read a MXXL mock catalog of BGS targets."""
    cached_radec = None

    def __init__(self, **kwargs):
        super(ReadMXXL, self).__init__(**kwargs)

    def readmock(self, mockfile=None, healpixels=None, nside=None,
                 target_name='BGS', magcut=None, only_coords=False,
                 mock_density=False, seed=None):
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
        mock_density : :class:`bool`, optional
            Compute and return the median target density in the mock.  Defaults
            to False.
        seed : :class:`int`, optional
            Seed for reproducibility and random number generation.

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
                nside = 64
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
                log.debug('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, zz, rmag, absmag, gr, allpix, pixweight = ReadMXXL.cached_radec

        mockid = np.arange(len(ra)) # unique ID/row number
        
        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint
        if np.all(cut[1:] >= cut[:-1]) is False:
            log.fatal('Index cut must be monotonically increasing, otherwise fitsio will resort it!')
            raise ValueError

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

        isouth = self.is_south(dec)

        # Get photometry and morphologies by sampling from the Gaussian mixture
        # models.  This is a total hack because our apparent magnitudes (rmag)
        # will not be consistent with the Gaussian draws.  But as a hack just
        # sort the shapes and sizes on rmag.
        log.info('Sampling from {} Gaussian mixture model.'.format(target_name))
        gmmout = self.sample_GMM(nobj, target=target_name, isouth=isouth,
                                 seed=seed, prior_mag=rmag)

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'durham_mxxl_hdf5',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': self.Bricks.brickname(ra, dec).astype('S8'),
               'BRICKID': self.Bricks.brickid(ra, dec),
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': rmag, 'SDSS_absmag_r01': absmag,
               'SDSS_01gr': gr, 'MAGFILTER': np.repeat('sdss2010-r', nobj),
               'SOUTH': isouth}

        if gmmout is not None:
            out.update(gmmout)

        # Add MW transmission and the imaging depth.
        self.mw_transmission(out)
        self.imaging_depth(out)

        # Optionally compute the mean mock density.
        if mock_density:
            out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile)

        return out

class ReadGAMA(SelectTargets):
    """Read a GAMA catalog of BGS targets.  This reader will only generally be used
    for the Survey Validation Data Challenge."""
    cached_radec = None
    
    def __init__(self, **kwargs):
        super(ReadGAMA, self).__init__(**kwargs)
        
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
                log.debug('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, allpix, pixweight = ReadGAMA.cached_radec

        mockid = np.arange(len(ra)) # unique ID/row number
        
        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint
        if np.all(cut[1:] >= cut[:-1]) is False:
            log.fatal('Index cut must be monotonically increasing, otherwise fitsio will resort it!')
            raise ValueError

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

        # Add photometry, absolute magnitudes, and redshifts.
        columns = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'Z', 'UGRIZ_ABSMAG_01']
        data = fitsio.read(mockfile, columns=columns, upper=True, ext=1, rows=cut)
        zz = data['Z'].astype('f4')
        rmag = 22.5 - 2.5 * np.log10(data['FLUX_R']).astype('f4')

        # Pack into a basic dictionary.  Could include shapes and other spectral
        # properties here.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'bgs-gama',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': self.Bricks.brickname(ra, dec).astype('S8'),
               'BRICKID': self.Bricks.brickid(ra, dec),
               'RA': ra, 'DEC': dec, 'Z': zz, 'RMABS_01': data['UGRIZ_ABSMAG_01'][:, 2],
               'UG_01': data['UGRIZ_ABSMAG_01'][:, 0]-data['UGRIZ_ABSMAG_01'][:, 1],
               'GR_01': data['UGRIZ_ABSMAG_01'][:, 1]-data['UGRIZ_ABSMAG_01'][:, 2],
               'RI_01': data['UGRIZ_ABSMAG_01'][:, 2]-data['UGRIZ_ABSMAG_01'][:, 3],
               'IZ_01': data['UGRIZ_ABSMAG_01'][:, 3]-data['UGRIZ_ABSMAG_01'][:, 4],
               'MAGFILTER': np.repeat('decam2014-r', nobj),
               'MAG': rmag, 'SOUTH': self.is_south(dec)}

        # Add MW transmission and the imaging depth.
        self.mw_transmission(out)
        self.imaging_depth(out)

        return out

class ReadMWS_WD(SelectTargets):
    """Read a mock catalog of Milky Way Survey white dwarf targets (MWS_WD)."""
    cached_radec = None

    def __init__(self, **kwargs):
        super(ReadMWS_WD, self).__init__(**kwargs)

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
            If mockfile is not defined or if nside is not a scalar or if the
            selection index isn't monotonically increasing.

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
                nside = 64
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
                log.debug('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, allpix, pixweight = ReadMWS_WD.cached_radec

        mockid = np.arange(len(ra)) # unique ID/row number

        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint
        if np.all(cut[1:] >= cut[:-1]) is False:
            log.fatal('Index cut must be monotonically increasing, otherwise fitsio will resort it!')
            raise ValueError

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

        cols = ['RADIALVELOCITY', 'TEFF', 'LOGG', 'SPECTRALTYPE',
                'PHOT_G_MEAN_MAG', 'PHOT_BP_MEAN_MAG', 'PHOT_RP_MEAN_MAG',
                'PMRA', 'PMDEC', 'PARALLAX', 'PARALLAX_ERROR',
                'ASTROMETRIC_EXCESS_NOISE', 'RA']
        data = fitsio.read(mockfile, columns=cols, upper=True, ext=1, rows=cut)

        zz = (data['RADIALVELOCITY'] / C_LIGHT).astype('f4')
        teff = data['TEFF'].astype('f4')
        logg = data['LOGG'].astype('f4')
        mag = data['PHOT_G_MEAN_MAG'].astype('f4')
        templatesubtype = np.char.upper(data['SPECTRALTYPE'].astype('<U'))

        gaia_g = data['PHOT_G_MEAN_MAG'].astype('f4')
        gaia_bp = data['PHOT_BP_MEAN_MAG'].astype('f4')
        gaia_rp = data['PHOT_RP_MEAN_MAG'].astype('f4')
        gaia_pmra = data['PMRA'].astype('f4')
        gaia_pmdec = data['PMDEC'].astype('f4')
        gaia_parallax = data['PARALLAX'].astype('f4')
        gaia_parallax_ivar = (1 / data['PARALLAX_ERROR']**2).astype('f4')
        gaia_noise = data['ASTROMETRIC_EXCESS_NOISE'].astype('f4')

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'mws_wd',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': self.Bricks.brickname(ra, dec).astype('S8'),
               'BRICKID': self.Bricks.brickid(ra, dec),
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': mag, 'TEFF': teff, 'LOGG': logg,
               'MAGFILTER': np.repeat('sdss2010-g', nobj),
               'TEMPLATESUBTYPE': templatesubtype,

               'REF_ID': mockid,
               'REF_CAT': np.repeat('G2', len(mockid)),
               'GAIA_PHOT_G_MEAN_MAG': gaia_g,
               #'GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR' - f4
               'GAIA_PHOT_BP_MEAN_MAG': gaia_bp,
               #'GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR' - f4
               'GAIA_PHOT_RP_MEAN_MAG': gaia_rp,
               #'GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR' - f4
               'GAIA_ASTROMETRIC_EXCESS_NOISE': gaia_noise,
               #'GAIA_DUPLICATED_SOURCE' - b1 # default is False
               'GAIA_ASTROMETRIC_PARAMS_SOLVED': np.zeros(nobj).astype('i8')+31, # good astrometry
               'PARALLAX': gaia_parallax,
               'PARALLAX_IVAR': gaia_parallax_ivar,
               'PMRA': gaia_pmra,
               'PMRA_IVAR': np.ones(nobj).astype('f4'),  # placeholder!
               'PMDEC': gaia_pmdec,
               'PMDEC_IVAR': np.ones(nobj).astype('f4'), # placeholder!
               
               'SOUTH': self.is_south(dec), 'TYPE': np.repeat('PSF', nobj)}

        # Add MW transmission and the imaging depth.
        self.mw_transmission(out)
        self.imaging_depth(out)

        # Optionally compute the mean mock density.
        if mock_density:
            out['MOCK_DENSITY'] = self.mock_density(mockfile=mockfile)

        return out
    
class ReadMWS_NEARBY(SelectTargets):
    """Read a mock catalog of Milky Way Survey nearby targets (MWS_NEARBY)."""
    cached_radec = None
    
    def __init__(self, **kwargs):
        super(ReadMWS_NEARBY, self).__init__(**kwargs)

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
                nside = 64
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
                log.debug('Using cached coordinates, healpixels, and pixel weights from {}'.format(mockfile))
                _, _, ra, dec, allpix, pixweight = ReadMWS_NEARBY.cached_radec
        
        mockid = np.arange(len(ra)) # unique ID/row number

        fracarea = pixweight[allpix]
        cut = np.where( np.in1d(allpix, healpixels) * (fracarea > 0) )[0] # force DESI footprint
        if np.all(cut[1:] >= cut[:-1]) is False:
            log.fatal('Index cut must be monotonically increasing, otherwise fitsio will resort it!')
            raise ValueError

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

        cols = ['TRUE_RADIAL_VELOCITY', 'TRUE_TEFF', 'TRUE_LOGG', 'TRUE_FEH', 'TRUE_TYPE',
                'GAIA_PHOT_G_MEAN_MAG', 'GAIA_PHOT_BP_MEAN_MAG', 'GAIA_PHOT_RP_MEAN_MAG',
                'PMRA', 'PMDEC', 'PARALLAX']
        data = fitsio.read(mockfile, columns=cols, upper=True, ext=1, rows=cut)
        zz = (data['TRUE_RADIAL_VELOCITY'] / C_LIGHT).astype('f4')
        mag = data['GAIA_PHOT_G_MEAN_MAG'].astype('f4') # not quite SDSS g-band but very close 
        teff = data['TRUE_TEFF'].astype('f4')
        logg = data['TRUE_LOGG'].astype('f4')
        feh = data['TRUE_FEH'].astype('f4')
        templatesubtype = data['TRUE_TYPE']

        gaia_g = data['GAIA_PHOT_G_MEAN_MAG'].astype('f4')
        gaia_bp = data['GAIA_PHOT_BP_MEAN_MAG'].astype('f4')
        gaia_rp = data['GAIA_PHOT_RP_MEAN_MAG'].astype('f4')
        gaia_pmra = data['PMRA'].astype('f4')
        gaia_pmdec = data['PMDEC'].astype('f4')
        gaia_parallax = data['PARALLAX'].astype('f4')
        #gaia_parallax_ivar = (1 / data['PARALLAX_ERROR']**2).astype('f4')

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'mws_100pc',
               'HEALPIX': allpix, 'NSIDE': nside, 'WEIGHT': weight,
               'MOCKID': mockid, 'BRICKNAME': self.Bricks.brickname(ra, dec).astype('S8'),
               'BRICKID': self.Bricks.brickid(ra, dec),
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': mag, 'TEFF': teff, 'LOGG': logg, 'FEH': feh,
               'MAGFILTER': np.repeat('sdss2010-g', nobj), 'TEMPLATESUBTYPE': templatesubtype,

               'REF_ID': mockid,
               'REF_CAT': np.repeat('G2', len(mockid)),
               'GAIA_PHOT_G_MEAN_MAG': gaia_g,
               #'GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR' - f4
               'GAIA_PHOT_BP_MEAN_MAG': gaia_bp,
               #'GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR' - f4
               'GAIA_PHOT_RP_MEAN_MAG': gaia_rp,
               #'GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR' - f4
               'GAIA_ASTROMETRIC_PARAMS_SOLVED': np.zeros(nobj).astype('f4')+31, # good astrometry
               #'GAIA_ASTROMETRIC_EXCESS_NOISE': gaia_noise,
               #'GAIA_DUPLICATED_SOURCE' - b1 # default is False
               'PARALLAX': gaia_parallax,
               #'PARALLAX_IVAR': gaia_parallax_ivar,
               'PMRA': gaia_pmra,
               'PMRA_IVAR': np.ones(nobj).astype('f4'),  # placeholder!
               'PMDEC': gaia_pmdec,
               'PMDEC_IVAR': np.ones(nobj).astype('f4'), # placeholder!
               
               'SOUTH': self.is_south(dec), 'TYPE': np.repeat('PSF', nobj)}

        # Add MW transmission and the imaging depth.
        self.mw_transmission(out)
        self.imaging_depth(out)

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
    use_simqso : :class:`bool`, optional
        Use desisim.templates.SIMQSO to generated templates rather than
        desisim.templates.QSO.  Defaults to True.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    wave, template_maker, GMM_QSO = None, None, None
    
    def __init__(self, seed=None, use_simqso=True, survey='main', **kwargs):
        from desisim.templates import SIMQSO, QSO
        from desiutil.sklearn import GaussianMixtureModel

        super(QSOMaker, self).__init__(survey=survey)

        self.seed = seed
        self.objtype = 'QSO'
        self.use_simqso = use_simqso

        if self.wave is None:
            QSOMaker.wave = _default_wave()
        self.extinction = self.mw_dust_extinction()

        if self.template_maker is None:
            if self.use_simqso:
                QSOMaker.template_maker = SIMQSO(wave=self.wave)
            else:
                QSOMaker.template_maker = QSO(wave=self.wave)

        if self.GMM_QSO is None:
            self.read_GMM(target='QSO')

    def read(self, mockfile=None, mockformat='gaussianfield', healpixels=None,
             nside=None, zmax_qso=None, only_coords=False, mock_density=False,
             **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'gaussianfield'.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        zmax_qso : :class:`float`
            Maximum redshift of tracer QSOs to read, to ensure no
            double-counting with Lya mocks.  Defaults to None.
        only_coords : :class:`bool`, optional
            For various applications, only read the target coordinates.
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
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'DarkSky', 'v1.0.1', 'qso_0_inpt.fits')
            MockReader = ReadGaussianField()
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile
            
        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   only_coords=only_coords, seed=self.seed,
                                   zmax_qso=zmax_qso, mock_density=mock_density)

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
        objtruth : :class:`astropy.table.Table`
            Corresponding objtype-specific truth table (if applicable).
        
        """
        if seed is None:
            seed = self.seed

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)
            
        rand = np.random.RandomState(seed)
        if no_spectra:
            flux = []
            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            self._nospectra_photometry(meta, rand, data, indx, data['TARGET_NAME'])
        else:
            # Sample from the north/south GMMs
            south = np.where( data['SOUTH'][indx] == True )[0]
            north = np.where( data['SOUTH'][indx] == False )[0]

            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype, simqso=self.use_simqso)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')

            if self.use_simqso:
                for these, issouth in zip( (north, south), (False, True) ):
                    if len(these) > 0:
                        flux1, _, meta1, objmeta1 = self.template_maker.make_templates(
                            nmodel=len(these), redshift=np.atleast_1d(data['Z'][indx][these]),
                            seed=seed, lyaforest=False, nocolorcuts=True, south=issouth)

                        meta[these] = meta1
                        objmeta[these] = objmeta1
                        flux[these, :] = flux1
            else:
                input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype, input_meta=True)
                input_meta['SEED'][:] = rand.randint(2**31, size=nobj)
                input_meta['REDSHIFT'][:] = data['Z'][indx]
                
                if self.mockformat == 'gaussianfield':
                    input_meta['MAG'][:] = data['MAG'][indx]
                    input_meta['MAGFILTER'][:] = data['MAGFILTER'][indx]

                for these, issouth in zip( (north, south), (False, True) ):
                    if len(these) > 0:
                        flux1, _, meta1, objmeta1 = self.template_maker.make_templates(
                            input_meta=input_meta[these], lyaforest=False, nocolorcuts=True,
                            south=issouth)

                        meta[these] = meta1
                        objmeta[these] = objmeta1
                        flux[these, :] = flux1

        targets, truth, objtruth = self.populate_targets_truth(
            flux, data, meta, objmeta, indx=indx, use_simqso=self.use_simqso,
            seed=seed, truespectype='QSO', templatetype='QSO')

        return flux, self.wave, targets, truth, objtruth

    def select_targets(self, targets, truth, targetname='QSO'):
        """Select QSO targets.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        targetname : :class:`str`
            Target selection cuts to apply.

        """
        if self.survey == 'sv1':
            qso_selection = 'randomforest'
        else:
            qso_selection = 'colorcuts'
            
        if self.use_simqso:
            desi_target, bgs_target, mws_target = cuts.apply_cuts(targets, tcnames=targetname,
                                                                  qso_selection=qso_selection,
                                                                  survey=self.survey)
        else:
            desi_target, bgs_target, mws_target = cuts.apply_cuts(
                targets, tcnames=targetname, qso_selection='colorcuts',
                qso_optical_cuts=True, survey=self.survey)

        self.remove_north_south_bits(desi_target, bgs_target, mws_target)

        targets['DESI_TARGET'] |= desi_target
        targets['BGS_TARGET'] |= bgs_target
        targets['MWS_TARGET'] |= mws_target

class LYAMaker(SelectTargets):
    """Read LYA mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    balprob : :class:`float`, optional
        Probability of a including one or more BALs.  Defaults to 0.0. 
    add_dla : :class:`bool`, optional
        Statistically include DLAs along the line of sight.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    wave, template_maker = None, None

    def __init__(self, seed=None, use_simqso=True,sqmodel='default',\
                 balprob=0.0,add_dla=False,add_metals=False,add_lyb=False,\
                 survey='main', **kwargs):
        from desisim.templates import SIMQSO, QSO
        from desiutil.sklearn import GaussianMixtureModel

        super(LYAMaker, self).__init__(survey=survey)

        self.seed = seed
        self.objtype = 'LYA'
        self.use_simqso = use_simqso
        self.sqmodel=sqmodel
        self.balprob = balprob
        self.add_dla = add_dla
        self.add_metals=add_metals
        self.add_lyb=add_lyb
        if balprob > 0:
            from desisim.bal import BAL
            self.BAL = BAL()

        if self.wave is None:
            LYAMaker.wave = _default_wave()
        self.extinction = self.mw_dust_extinction()
            
        if self.template_maker is None:
            if self.use_simqso:
                LYAMaker.template_maker = SIMQSO(wave=self.wave,sqmodel=self.sqmodel)
            else:
                LYAMaker.template_maker = QSO(wave=self.wave)
    def read(self, mockfile=None, mockformat='CoLoRe', healpixels=None, nside=None,
             nside_lya=16, zmin_lya=None, mock_density=False, only_coords=False,
             **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'CoLoRe'.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        nside_lya : :class:`int`
            Healpixel nside indicating how the mock on-disk has been organized.
            Defaults to 16.
        zmin_lya : :class:`float`
            Minimum redshift of Lya skewers, to ensure no double-counting with
            QSO mocks.  Defaults to None.
        mock_density : :class:`bool`, optional
            Compute the median target density in the mock.  Defaults to False.
        only_coords : :class:`bool`, optional
            Only read the target coordinates and some other basic info.
            Defaults to False.

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
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'lya_forest', 'london', 'v9.0', 'v9.0.0', 'master.fits')
            MockReader = ReadLyaCoLoRe()
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile
        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   nside_lya=nside_lya, zmin_lya=zmin_lya,
                                   mock_density=mock_density,
                                   only_coords=only_coords, seed=self.seed)
        return data

    def make_spectra(self, data=None, indx=None, seed=None,no_spectra=False,add_dlas=None,add_metals=None,add_lyb=None):
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
        from astropy.table import vstack
        from desispec.interpolation import resample_flux
        
        from desisim.lya_spectra import read_lya_skewers, apply_lya_transmission,lambda_RF_LYA
        from desisim.dla import dla_spec

        if add_dlas is None:
            add_dlas=self.add_dla
        if add_metals is None:
            add_metals=self.add_metals
        if add_lyb is None:
            add_lyb=self.add_lyb
        if seed is None:
            seed = self.seed
            
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        rand = np.random.RandomState(seed)
        if no_spectra:
            flux = []
            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            self._nospectra_photometry(meta, rand, data, indx, data['TARGET_NAME'])
        else:
            # Handle north/south photometry.
            south = np.where( data['SOUTH'][indx] == True )[0]
            north = np.where( data['SOUTH'][indx] == False )[0]

            if not self.use_simqso:
                input_meta = empty_metatable(nmodel=nobj, objtype='QSO', subtype='LYA', input_meta=True)
                input_meta['SEED'][:] = rand.randint(2**31, size=nobj)
                input_meta['REDSHIFT'][:] = data['Z'][indx]
                input_meta['MAG'][:] = data['MAG'][indx]
                input_meta['MAGFILTER'][:] = data['MAGFILTER'][indx]
                                
            # Read skewers.
            skewer_wave = None
            skewer_trans = None
            skewer_meta = None

            # Gather all the files containing at least one QSO skewer.
            alllyafile = data['LYAFILES'][indx]
            uniquelyafiles = sorted(set(alllyafile))
            for lyafile in uniquelyafiles:
                these = np.where( alllyafile == lyafile )[0]

                mockid_in_data = data['MOCKID'][indx][these]
                if not os.path.isfile(lyafile):
                    lyafile=lyafile.replace(".gz","")
                    if not os.path.isfile(lyafile):
                        log.warning('transmisionfile {} not found!'.format(lyafile))
                        raise IOError
                mockid_in_mock = (fitsio.read(lyafile, columns=['MOCKID'], upper=True,
                                              ext=1).astype(float)).astype(int)
                o2i = dict()
                for i, o in enumerate(mockid_in_mock):
                    o2i[o] = i
                indices_in_mock_healpix = np.zeros(mockid_in_data.size).astype(int)
                for i, o in enumerate(mockid_in_data):
                    if not o in o2i:
                        log.warning("No MOCKID={} in {}, which should never happen".format(o, lyafile))
                        raise KeyError
                    indices_in_mock_healpix[i] = o2i[o]
               # Note: there are read_dlas=False and add_metals=False options. AXGM: This has been updated.
                tmp_wave, tmp_trans, tmp_meta, dla_info = read_lya_skewers(lyafile,indices=indices_in_mock_healpix,read_dlas=add_dlas,add_metals=add_metals,add_lyb=add_lyb)

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
            assert(np.max(np.abs(skewer_meta['Z']-data['Z'][indx])) < 0.000001)
            assert(np.max(np.abs(skewer_meta['RA']-data['RA'][indx])) < 0.000001)
            assert(np.max(np.abs(skewer_meta['DEC']-data['DEC'][indx])) < 0.000001)

            # Now generate the QSO spectra simultaneously **at full wavelength
            # resolution**.  We do this because the Lya forest will have changed
            # the colors, so we need to re-synthesize the photometry below.
            meta, objmeta = empty_metatable(nmodel=nobj, objtype='QSO', simqso=self.use_simqso)
            if self.use_simqso:
                qso_flux = np.zeros([nobj, len(self.template_maker.basewave)], dtype='f4')
            else:
                qso_flux = np.zeros([nobj, len(self.template_maker.eigenwave)], dtype='f4')
                qso_wave = np.zeros_like(qso_flux)
            
            for these, issouth in zip( (north, south), (False, True) ):
                if len(these) > 0:
                    if self.use_simqso:
                        qso_flux1, qso_wave, meta1, objmeta1 = self.template_maker.make_templates(
                            nmodel=len(these), redshift=data['Z'][indx][these], seed=seed,
                            lyaforest=False, nocolorcuts=True, noresample=True, south=issouth)
                    else:
                        qso_flux1, qso_wave1, meta1, objmeta1 = self.template_maker.make_templates(
                            input_meta=input_meta[these], lyaforest=False, nocolorcuts=True,
                            noresample=True, south=issouth)
                        qso_wave[these, :] = qso_wave1
                        
                    meta[these] = meta1
                    if(objmeta['EMLINES'].shape!=objmeta1['EMLINES'].shape):
                        objmeta.replace_column('EMLINES',np.zeros((nobj,len(objmeta1['EMLINES'][0, :, 0]), 3))-1)
                    objmeta[these] = objmeta1
                    qso_flux[these, :] = qso_flux1
            meta['SUBTYPE'][:] = 'LYA'
            ##Added DLAs before lya forest trasnmission.
            if add_dlas:
                min_lya_z = np.min(skewer_wave/lambda_RF_LYA - 1)
                for ii in range(len(skewer_meta['MOCKID'])):
                    # quasars with z < min_z will not have any DLA in spectrum
                    if min_lya_z>skewer_meta['Z'][ii]: continue
                    # skewer ID
                    idd=skewer_meta['MOCKID'][ii]
                    dlas=[]
                    for dla in dla_info[dla_info['MOCKID']==idd]:
                        # Adding only DLAs with z < zqso
                        if dla['Z_DLA_RSD']>=skewer_meta['Z'][ii]: continue
                        dlas.append(dict(z=dla['Z_DLA_RSD'],N=dla['N_HI_DLA'],dlaid=dla['DLAID']))
                    transmission_dla = dla_spec(skewer_wave,dlas)
                    if len(dlas)>0:
                        skewer_trans[ii] = transmission_dla * skewer_trans[ii]
                        objmeta['DLA'][ii]=True

            # Apply the Lya forest transmission.
            _flux = apply_lya_transmission(qso_wave, qso_flux, skewer_wave, skewer_trans)
            # Add BALs
            if self.balprob > 0:
                log.debug('Adding BAL(s) with probability {}'.format(self.balprob))
                _flux, balmeta = self.BAL.insert_bals(qso_wave, _flux, meta['REDSHIFT'],
                                                      seed=self.seed,
                                                      balprob=self.balprob)
                objmeta['BAL_TEMPLATEID'][:] = balmeta['TEMPLATEID']

            # Synthesize north/south photometry.
            for these, filters in zip( (north, south), (self.template_maker.bassmzlswise, self.template_maker.decamwise) ):
                if len(these) > 0:
                    if self.use_simqso:
                        maggies = filters.get_ab_maggies(1e-17 * _flux[these, :], qso_wave.copy(), mask_invalid=True)
                        for band, filt in zip( ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'), filters.names):
                            meta[band][these] = ma.getdata(1e9 * maggies[filt]) # nanomaggies
                    else:
                        # We have to loop (and pad) since each QSO has a different wavelength array.
                        maggies = []
                        for ii in range(len(these)):
                            padflux, padwave = filters.pad_spectrum(_flux[these[ii], :], qso_wave[these[ii], :], method='edge')
                            maggies.append(filters.get_ab_maggies(1e-17 * padflux, padwave.copy(), mask_invalid=True))
                            
                        maggies = vstack(maggies)
                        for band, filt in zip( ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'), filters.names):
                            meta[band][these] = ma.getdata(1e9 * maggies[filt]) # nanomaggies
            # Unfortunately, in order to resample to the desired output
            # wavelength vector we need to loop.
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')
            if qso_wave.ndim == 2:
                for ii in range(nobj):
                    flux[ii, :] = resample_flux(self.wave, qso_wave[ii, :], _flux[ii, :], extrapolate=True)
            else:
                for ii in range(nobj):
                    flux[ii, :] = resample_flux(self.wave, qso_wave, _flux[ii, :], extrapolate=True)

        targets, truth, objtruth = self.populate_targets_truth(
            flux, data, meta, objmeta, indx=indx, seed=seed,
            truespectype='QSO', templatetype='QSO', templatesubtype='LYA')

        return flux, self.wave, targets, truth, objtruth

    def select_targets(self, targets, truth, targetname='QSO'):
        """Select Lya/QSO targets.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        targetname : :class:`str`
            Target selection cuts to apply.

        """
        if targetname == 'LYA':
            tcnames = 'QSO'
        else:
            tcnames = targetname

        if self.survey == 'sv1':
            qso_selection = 'randomforest'
        else:
            qso_selection = 'colorcuts'
            
        desi_target, bgs_target, mws_target = cuts.apply_cuts(targets, tcnames=tcnames,
                                                              qso_selection=qso_selection,
                                                              survey=self.survey)
        
        self.remove_north_south_bits(desi_target, bgs_target, mws_target)
        
        targets['DESI_TARGET'] |= desi_target
        targets['BGS_TARGET'] |= bgs_target
        targets['MWS_TARGET'] |= mws_target

class LRGMaker(SelectTargets):
    """Read LRG mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    nside_chunk : :class:`int`, optional
        Healpixel nside for further subdividing the sample when assigning
        velocity dispersion to targets.  Defaults to 128.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    wave, template_maker = None, None
    GMM_LRG, KDTree_north, KDTree_south = None, None, None

    def __init__(self, seed=None, nside_chunk=128, survey='main', **kwargs):
        from desisim.templates import LRG
        from desiutil.sklearn import GaussianMixtureModel

        super(LRGMaker, self).__init__(survey=survey)

        self.seed = seed
        self.nside_chunk = nside_chunk
        self.objtype = 'LRG'

        if self.wave is None:
            LRGMaker.wave = _default_wave()
        self.extinction = self.mw_dust_extinction()

        if self.template_maker is None:
            LRGMaker.template_maker = LRG(wave=self.wave)
            
        self.meta = self.template_maker.basemeta

        # Build the KD Tree.
        zobj = self.meta['Z'].data
        gr_north = (self.meta['BASS_G'] - self.meta['BASS_R']).data
        rz_north = (self.meta['BASS_R'] - self.meta['MZLS_Z']).data
        zW1_north = (self.meta['MZLS_Z'] - self.meta['W1']).data
            
        gr_south = (self.meta['DECAM_G'] - self.meta['DECAM_R']).data
        rz_south = (self.meta['DECAM_R'] - self.meta['DECAM_Z']).data
        zW1_south = (self.meta['DECAM_Z'] - self.meta['W1']).data

        self.param_min_north = ( zobj.min(), gr_north.min(), rz_north.min(), zW1_north.min() )
        self.param_min_south = ( zobj.min(), gr_south.min(), rz_south.min(), zW1_south.min() )
        self.param_range_north = ( np.ptp(zobj), np.ptp(gr_north), np.ptp(rz_north), np.ptp(zW1_north) )
        self.param_range_south = ( np.ptp(zobj), np.ptp(gr_south), np.ptp(rz_south), np.ptp(zW1_south) )
        
        if self.KDTree_north is None:
            LRGMaker.KDTree_north = self.KDTree_build(
                np.vstack((
                    zobj,
                    gr_north,
                    rz_north,
                    zW1_north)).T, south=False )
        if self.KDTree_south is None:
            LRGMaker.KDTree_south = self.KDTree_build(
                np.vstack((
                    zobj,
                    gr_south,
                    rz_south,
                    zW1_south)).T, south=True )
            
        if self.GMM_LRG is None:
            self.read_GMM(target='LRG')

    def read(self, mockfile=None, mockformat='gaussianfield', healpixels=None,
             nside=None, only_coords=False, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'gaussianfield'.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        only_coords : :class:`bool`, optional
            For various applications, only read the target coordinates.
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
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'DarkSky', 'v1.0.1', 'lrg_0_inpt.fits')
            MockReader = ReadGaussianField()
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   only_coords=only_coords,
                                   mock_density=mock_density, seed=self.seed)

        return data

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
        objtruth : :class:`astropy.table.Table`
            Corresponding objtype-specific truth table (if applicable).
        
        """
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)
        
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if no_spectra:
            flux = []
            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            # noiseless photometry from pre-computed GMMs
            self._nospectra_photometry(meta, rand, data, indx, data['TARGET_NAME'])
        else:
            input_meta, _ = empty_metatable(nmodel=nobj, objtype=self.objtype)
            input_meta['SEED'][:] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'][:] = data['Z'][indx]
            vdisp = self._sample_vdisp(data['RA'][indx], data['DEC'][indx], mean=2.3,
                                       sigma=0.1, seed=seed, nside=self.nside_chunk)

            # Differentiate north/south photometry.
            south = np.where( data['SOUTH'][indx] == True )[0]
            north = np.where( data['SOUTH'][indx] == False )[0]

            if self.mockformat == 'gaussianfield':
                for these, issouth in zip( (north, south), (False, True) ):
                    if len(these) > 0:
                        input_meta['MAG'][these] = data['MAG'][indx][these]
                        input_meta['MAGFILTER'][these] = data['MAGFILTER'][indx][these]
                        input_meta['TEMPLATEID'][these] = self.KDTree_query(
                            np.vstack((
                                data['Z'][indx][these],
                                data['GR'][indx][these],
                                data['RZ'][indx][these],
                                data['ZW1'][indx][these])).T, south=issouth)
                        
            # Build north/south spectra separately.
            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')

            for these, issouth in zip( (north, south), (False, True) ):
                if len(these) > 0:
                    flux1, _, meta1, objmeta1 = self.template_maker.make_templates(
                        input_meta=input_meta[these], vdisp=vdisp[these], south=issouth,
                        nocolorcuts=True)

                    meta[these] = meta1
                    objmeta[these] = objmeta1
                    flux[these, :] = flux1
                    
        targets, truth, objtruth = self.populate_targets_truth(
            flux, data, meta, objmeta, indx=indx, seed=seed,
            truespectype='GALAXY', templatetype='LRG')

        return flux, self.wave, targets, truth, objtruth

    def select_targets(self, targets, truth, targetname='LRG'):
        """Select LRG targets.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        targetname : :class:`str`
            Target selection cuts to apply.

        """
        desi_target, bgs_target, mws_target = cuts.apply_cuts(targets, tcnames=targetname,
                                                              survey=self.survey)
        
        self.remove_north_south_bits(desi_target, bgs_target, mws_target)
        
        targets['DESI_TARGET'] |= desi_target
        targets['BGS_TARGET'] |= bgs_target
        targets['MWS_TARGET'] |= mws_target

class ELGMaker(SelectTargets):
    """Read ELG mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    nside_chunk : :class:`int`, optional
        Healpixel nside for further subdividing the sample when assigning
        velocity dispersion to targets.  Defaults to 128.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    wave, template_maker = None, None
    GMM_ELG, KDTree_north, KDTree_south = None, None, None
    
    def __init__(self, seed=None, nside_chunk=128, survey='main', **kwargs):
        from desisim.templates import ELG
        from desiutil.sklearn import GaussianMixtureModel

        super(ELGMaker, self).__init__(survey=survey)

        self.seed = seed
        self.nside_chunk = nside_chunk
        self.objtype = 'ELG'

        if self.wave is None:
            ELGMaker.wave = _default_wave()
        self.extinction = self.mw_dust_extinction()

        if self.template_maker is None:
            ELGMaker.template_maker = ELG(wave=self.wave)
            
        self.meta = self.template_maker.basemeta

        # Build the KD Trees
        zobj = self.meta['Z'].data
        gr_north = (self.meta['BASS_G'] - self.meta['BASS_R']).data
        rz_north = (self.meta['BASS_R'] - self.meta['MZLS_Z']).data
        gr_south = (self.meta['DECAM_G'] - self.meta['DECAM_R']).data
        rz_south = (self.meta['DECAM_R'] - self.meta['DECAM_Z']).data

        self.param_min_north = ( zobj.min(), gr_north.min(), rz_north.min() )
        self.param_min_south = ( zobj.min(), gr_south.min(), rz_south.min() )
        self.param_range_north = ( np.ptp(zobj), np.ptp(gr_north), np.ptp(rz_north) )
        self.param_range_south = ( np.ptp(zobj), np.ptp(gr_south), np.ptp(rz_south) )

        if self.KDTree_north is None:
            ELGMaker.KDTree_north = self.KDTree_build(
                np.vstack((
                    zobj,
                    gr_north,
                    rz_north)).T, south=False )
        if self.KDTree_south is None:
            ELGMaker.KDTree_south = self.KDTree_build(
                np.vstack((
                    zobj,
                    gr_south,
                    rz_south)).T, south=True )

        if self.GMM_ELG is None:
            self.read_GMM(target='ELG')

    def read(self, mockfile=None, mockformat='gaussianfield', healpixels=None,
             nside=None, only_coords=False, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'gaussianfield'.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        only_coords : :class:`bool`, optional
            For various applications, only read the target coordinates.
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
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'DarkSky', 'v1.0.1', 'elg_0_inpt.fits')
            MockReader = ReadGaussianField()
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   only_coords=only_coords,
                                   mock_density=mock_density, seed=self.seed)

        return data
            
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
        objtruth : :class:`astropy.table.Table`
            Corresponding objtype-specific truth table (if applicable).
        
        """
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if no_spectra:
            flux = []
            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            # noiseless photometry from pre-computed GMMs
            self._nospectra_photometry(meta, rand, data, indx, data['TARGET_NAME'])
        else:
            input_meta, _ = empty_metatable(nmodel=nobj, objtype=self.objtype)
            input_meta['SEED'][:] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'][:] = data['Z'][indx]
            vdisp = self._sample_vdisp(data['RA'][indx], data['DEC'][indx], mean=1.9,
                                       sigma=0.15, seed=seed, nside=self.nside_chunk)

            # Differentiate north/south photometry.
            south = np.where( data['SOUTH'][indx] == True )[0]
            north = np.where( data['SOUTH'][indx] == False )[0]

            if self.mockformat == 'gaussianfield':
                for these, issouth in zip( (north, south), (False, True) ):
                    if len(these) > 0:
                        input_meta['MAG'][these] = data['MAG'][indx][these]
                        input_meta['MAGFILTER'][these] = data['MAGFILTER'][indx][these]
                        input_meta['TEMPLATEID'][these] = self.KDTree_query(
                            np.vstack((
                                data['Z'][indx][these],
                                data['GR'][indx][these],
                                data['RZ'][indx][these])).T, south=issouth)

            # Build north/south spectra separately.
            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')

            for these, issouth in zip( (north, south), (False, True) ):
                if len(these) > 0:
                    flux1, _, meta1, objmeta1 = self.template_maker.make_templates(
                        input_meta=input_meta[these], vdisp=vdisp[these], south=issouth,
                        nocolorcuts=True)

                    meta[these] = meta1
                    objmeta[these] = objmeta1
                    flux[these, :] = flux1

        targets, truth, objtruth = self.populate_targets_truth(
            flux, data, meta, objmeta, indx=indx, seed=seed,
            truespectype='GALAXY', templatetype='ELG')

        return flux, self.wave, targets, truth, objtruth

    def select_targets(self, targets, truth, targetname='ELG'):
        """Select ELG targets.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        targetname : :class:`str`
            Target selection cuts to apply.

        """
        desi_target, bgs_target, mws_target = cuts.apply_cuts(targets, tcnames=targetname,
                                                              survey=self.survey)
        
        self.remove_north_south_bits(desi_target, bgs_target, mws_target)
        
        targets['DESI_TARGET'] |= desi_target
        targets['BGS_TARGET'] |= bgs_target
        targets['MWS_TARGET'] |= mws_target

class BGSMaker(SelectTargets):
    """Read BGS mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    nside_chunk : :class:`int`, optional
        Healpixel nside for further subdividing the sample when assigning
        velocity dispersion to targets.  Defaults to 128.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    wave, template_maker, GMM_BGS, KDTree = None, None, None, None
    
    def __init__(self, seed=None, nside_chunk=128, survey='main', **kwargs):
        super(BGSMaker, self).__init__(survey=survey)

        self.seed = seed
        self.nside_chunk = nside_chunk
        self.objtype = 'BGS'

        if self.wave is None:
            BGSMaker.wave = _default_wave()
        self.extinction = self.mw_dust_extinction()

        if self.template_maker is None:
            from desisim.templates import BGS
            BGSMaker.template_maker = BGS(wave=self.wave)
            
        self.meta = self.template_maker.basemeta

        zobj = self.meta['Z'].data
        mabs = self.meta['SDSS_UGRIZ_ABSMAG_Z01'].data
        rmabs = mabs[:, 2]
        gr = mabs[:, 1] - mabs[:, 2]

        self.param_min = ( zobj.min(), rmabs.min(), gr.min() )
        self.param_range = ( np.ptp(zobj), np.ptp(rmabs), np.ptp(gr) )
        if self.KDTree is None:
            BGSMaker.KDTree = self.KDTree_build(np.vstack((zobj, rmabs, gr)).T)

        if self.GMM_BGS is None:
            self.read_GMM(target='BGS')

    def read(self, mockfile=None, mockformat='durham_mxxl_hdf5', healpixels=None,
             nside=None, magcut=None, only_coords=False, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'durham_mxxl_hdf5'.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        magcut : :class:`float`
            Magnitude cut (hard-coded to SDSS r-band) to subselect targets
            brighter than magcut. 
        only_coords : :class:`bool`, optional
            For various applications, only read the target coordinates.
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
        if self.mockformat == 'durham_mxxl_hdf5':
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'bgs', 'MXXL', 'desi_footprint', 'v0.0.4', 'BGS.hdf5')            
            MockReader = ReadMXXL()
        elif self.mockformat == 'gaussianfield':
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'GaussianRandomField', 'v0.0.8_2LPT', 'BGS.fits')
            MockReader = ReadGaussianField()
        elif self.mockformat == 'bgs-gama':
            MockReader = ReadGAMA()
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile
            
        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   magcut=magcut, only_coords=only_coords,
                                   mock_density=mock_density, seed=self.seed)

        return data

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
        objtruth : :class:`astropy.table.Table`
            Corresponding objtype-specific truth table (if applicable).
        
        """
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if no_spectra:
            flux = []
            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            # noiseless photometry from pre-computed GMMs
            self._nospectra_photometry(meta, rand, data, indx, data['TARGET_NAME'])
        else:
            input_meta, _ = empty_metatable(nmodel=nobj, objtype=self.objtype)
            input_meta['SEED'][:] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'][:] = data['Z'][indx]
            
            vdisp = self._sample_vdisp(data['RA'][indx], data['DEC'][indx], mean=1.9,
                                       sigma=0.15, seed=seed, nside=self.nside_chunk)

            if self.mockformat == 'durham_mxxl_hdf5':
                input_meta['TEMPLATEID'][:] = self.KDTree_query( np.vstack((
                    data['Z'][indx],
                    data['SDSS_absmag_r01'][indx],
                    data['SDSS_01gr'][indx])).T )

            elif self.mockformat == 'bgs-gama':
                # Could conceivably use other colors here--
                input_meta['TEMPLATEID'][:] = self.KDTree_query( np.vstack((
                    data['Z'][indx],
                    data['RMABS_01'][indx],
                    data['GR_01'][indx])).T )
                
            elif self.mockformat == 'gaussianfield':
                # This is not quite right, but choose a template with equal probability.
                input_meta['TEMPLATEID'][:] = rand.choice(self.meta['TEMPLATEID'], nobj)
                
            input_meta['MAG'][:] = data['MAG'][indx]
            input_meta['MAGFILTER'][:] = data['MAGFILTER'][indx]
                
            # Build north/south spectra separately.
            south = np.where( data['SOUTH'][indx] == True )[0]
            north = np.where( data['SOUTH'][indx] == False )[0]

            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')

            for these, issouth in zip( (north, south), (False, True) ):
                if len(these) > 0:
                    flux1, _, meta1, objmeta1 = self.template_maker.make_templates(
                        input_meta=input_meta[these], vdisp=vdisp[these], south=issouth,
                        nocolorcuts=True)

                    meta[these] = meta1
                    objmeta[these] = objmeta1
                    flux[these, :] = flux1

        targets, truth, objtruth = self.populate_targets_truth(
            flux, data, meta, objmeta, indx=indx, seed=seed,
            truespectype='GALAXY', templatetype='BGS')

        return flux, self.wave, targets, truth, objtruth

    def select_targets(self, targets, truth, targetname='BGS'):
        """Select BGS targets.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
        desi_target, bgs_target, mws_target = cuts.apply_cuts(targets, tcnames=targetname,
                                                              survey=self.survey)
        
        self.remove_north_south_bits(desi_target, bgs_target, mws_target)
        
        targets['DESI_TARGET'] |= desi_target
        targets['BGS_TARGET'] |= bgs_target
        targets['MWS_TARGET'] |= mws_target
        
class STARMaker(SelectTargets):
    """Lower-level Class for preparing for stellar spectra to be generated,
    selecting standard stars, and selecting stars as contaminants for
    extragalactic targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    no_spectra : :class:`bool`, optional
        Initialize and cache template photometry.  Defaults to False.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    wave, template_maker, KDTree = None, None, None
    star_maggies_g_north, star_maggies_r_north = None, None
    star_maggies_g_south, star_maggies_r_south = None, None
    
    def __init__(self, seed=None, no_spectra=False, survey='main', **kwargs):
        from speclite import filters
        from desisim.templates import STAR

        super(STARMaker, self).__init__(survey=survey)

        self.seed = seed
        self.objtype = 'STAR'

        if self.wave is None:
            STARMaker.wave = _default_wave()
        self.extinction = self.mw_dust_extinction()

        if self.template_maker is None:
            STARMaker.template_maker = STAR(wave=self.wave)

        self.meta = self.template_maker.basemeta

        # Pre-compute normalized synthetic photometry for the full set of
        # stellar templates.
        if no_spectra and (self.star_maggies_g_north is None or self.star_maggies_r_north is None or
            self.star_maggies_g_south is None or self.star_maggies_r_south is None):
            log.info('Caching stellar template photometry.')

            if 'SYNTH_SDSS2010_R' in self.meta.colnames: # from DESI-COLORS HDU (basis templates >=v3.1)

                # Get the WISE colors from the SDSS r minus W1, W2 precomputed colors
                maggies_north = self.meta[['SYNTH_BASS_G', 'SYNTH_BASS_R', 'SYNTH_MZLS_Z']]
                maggies_south = self.meta[['SYNTH_DECAM2014_G', 'SYNTH_DECAM2014_R', 'SYNTH_DECAM2014_Z']]

                maggies_north['SYNTH_WISE2010_W1'] = self.meta['SYNTH_SDSS2010_R'] * 10**(-0.4 * self.meta['W1-R'].data)
                maggies_south['SYNTH_WISE2010_W1'] = self.meta['SYNTH_SDSS2010_R'] * 10**(-0.4 * self.meta['W1-R'].data)
                maggies_north['SYNTH_WISE2010_W2'] = self.meta['SYNTH_SDSS2010_R'] * 10**(-0.4 * self.meta['W2-R'].data)
                maggies_south['SYNTH_WISE2010_W2'] = self.meta['SYNTH_SDSS2010_R'] * 10**(-0.4 * self.meta['W2-R'].data)

                # Normalize to both sdss-g and sdss-r
                def _get_maggies(outmaggies, normmaggies):
                    for filt, flux in zip( outmaggies.colnames, ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2') ):
                        outmaggies[filt] /= normmaggies
                        outmaggies.rename_column(filt, flux)
                    return outmaggies

                STARMaker.star_maggies_g_north = _get_maggies(maggies_north.copy(), self.meta['SYNTH_SDSS2010_G'])
                STARMaker.star_maggies_r_north = _get_maggies(maggies_north.copy(), self.meta['SYNTH_SDSS2010_R'])
                STARMaker.star_maggies_g_south = _get_maggies(maggies_south.copy(), self.meta['SYNTH_SDSS2010_G'])
                STARMaker.star_maggies_r_south = _get_maggies(maggies_south.copy(), self.meta['SYNTH_SDSS2010_R'])
            else:
                sdssg = filters.load_filters('sdss2010-g')
                sdssr = filters.load_filters('sdss2010-r')

                flux, wave = self.template_maker.baseflux, self.template_maker.basewave
                padflux, padwave = sdssr.pad_spectrum(flux, wave, method='edge')

                maggies_north = self.bassmzlswise.get_ab_maggies(padflux, padwave, mask_invalid=True)
                maggies_south = self.decamwise.get_ab_maggies(padflux, padwave, mask_invalid=True)
                if 'W1-R' in self.meta.colnames: # >v3.0 templates
                    sdssrnorm = sdssr.get_ab_maggies(padflux, padwave)['sdss2010-r'].data
                    maggies_north['wise2010-W1'] = sdssrnorm * 10**(-0.4 * self.meta['W1-R'].data)
                    maggies_south['wise2010-W1'] = sdssrnorm * 10**(-0.4 * self.meta['W1-R'].data)
                    maggies_north['wise2010-W2'] = sdssrnorm * 10**(-0.4 * self.meta['W2-R'].data)
                    maggies_south['wise2010-W2'] = sdssrnorm * 10**(-0.4 * self.meta['W2-R'].data)

                # Normalize to both sdss-g and sdss-r
                def _get_maggies(flux, wave, outmaggies, normfilter):
                    normmaggies = normfilter.get_ab_maggies(flux, wave, mask_invalid=True)
                    for filt, flux in zip( outmaggies.colnames, ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2') ):
                        outmaggies[filt] /= normmaggies[normfilter.names[0]]
                        outmaggies.rename_column(filt, flux)
                    return outmaggies

                STARMaker.star_maggies_g_north = _get_maggies(flux, wave, maggies_north.copy(), sdssg)
                STARMaker.star_maggies_r_north = _get_maggies(flux, wave, maggies_north.copy(), sdssr)
                STARMaker.star_maggies_g_south = _get_maggies(flux, wave, maggies_south.copy(), sdssg)
                STARMaker.star_maggies_r_south = _get_maggies(flux, wave, maggies_south.copy(), sdssr)

        # Build the KD Tree.
        logteff = np.log10(self.meta['TEFF'].data)
        logg = self.meta['LOGG']
        feh = self.meta['FEH']

        self.param_min = ( logteff.min(), logg.min(), feh.min() )
        self.param_range = ( np.ptp(logteff), np.ptp(logg), np.ptp(feh) )

        if self.KDTree is None:
            STARMaker.KDTree = self.KDTree_build(np.vstack((logteff, logg, feh)).T)

    def template_photometry(self, data=None, indx=None, rand=None, south=True):
        """Get stellar photometry from the templates themselves, by-passing the
        generation of spectra.

        """
        if rand is None:
            rand = np.random.RandomState()

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)
        
        meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        meta['SEED'][:] = rand.randint(2**31, size=nobj)
        meta['REDSHIFT'][:] = data['Z'][indx]
        meta['MAG'][:] = data['MAG'][indx]
        meta['MAGFILTER'][:] = data['MAGFILTER'][indx]
        
        objmeta['TEFF'][:] = data['TEFF'][indx]
        objmeta['LOGG'][:] = data['LOGG'][indx]
        objmeta['FEH'][:] = data['FEH'][indx]

        if self.mockformat == 'galaxia':
            templateid = self.KDTree_query(np.vstack((
                np.log10(data['TEFF'][indx]).data,
                data['LOGG'][indx], data['FEH'][indx])).T)
            
        elif self.mockformat == 'mws_100pc':
            templateid = self.KDTree_query(np.vstack((
                np.log10(data['TEFF'][indx]),
                data['LOGG'][indx], data['FEH'][indx])).T)

        normmag = 1e9 * 10**(-0.4 * data['MAG'][indx]) # nanomaggies

        # A little fragile -- assume that MAGFILTER is the same for all objects...
        if south:
            if data['MAGFILTER'][0] == 'sdss2010-g':
                star_maggies = self.star_maggies_g_south
            elif data['MAGFILTER'][0] == 'sdss2010-r':
                star_maggies = self.star_maggies_r_south
            else:
                log.warning('Unrecognized normalization filter {}!'.format(data['MAGFILTER'][0]))
                raise ValueError
        else:
            if data['MAGFILTER'][0] == 'sdss2010-g':
                star_maggies = self.star_maggies_g_north
            elif data['MAGFILTER'][0] == 'sdss2010-r':
                star_maggies = self.star_maggies_r_north
            else:
                log.warning('Unrecognized normalization filter {}!'.format(data['MAGFILTER'][0]))
                raise ValueError

        for key in ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'):
            meta[key][:] = star_maggies[key][templateid] * normmag

        return meta, objmeta
 
class MWS_MAINMaker(STARMaker):
    """Read MWS_MAIN mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    calib_only : :class:`bool`, optional
        Use MWS_MAIN stars as calibration (standard star) targets, only.
        Defaults to False.
    no_spectra : :class:`bool`, optional
        Initialize and cache template photometry.  Defaults to False.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    def __init__(self, seed=None, calib_only=False, no_spectra=False,
                 survey='main', **kwargs):
        super(MWS_MAINMaker, self).__init__(seed=seed, no_spectra=no_spectra,
                                            survey=survey)

        self.seed = seed
        self.calib_only = calib_only

    def read(self, mockfile=None, mockformat='galaxia', healpixels=None,
             nside=None, nside_galaxia=8, target_name='MWS_MAIN', magcut=None,
             faintstar_mockfile=None, faintstar_magcut=None, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'galaxia'.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        nside_galaxia : :class:`int`
            Healpixel nside indicating how the mock on-disk has been organized.
            Defaults to 8.
        target_name : :class:`str`
            Name of the target being read.  Defaults to 'MWS_MAIN'.
        magcut : :class:`float`
            Magnitude cut (hard-coded to SDSS r-band) to subselect targets
            brighter than magcut.
        faintstar_mockfile : :class:`str`, optional
            Full path to the top-level directory of the Galaxia faint star mock
            catalog.
        faintstar_magcut : :class:`float`, optional
            Magnitude cut (hard-coded to SDSS r-band) to subselect faint star
            targets brighter than magcut.

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
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'mws', 'galaxia', 'alpha', 'v0.0.5', 'healpix')
            MockReader = ReadGalaxia()
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile
            
        data = MockReader.readmock(mockfile, target_name=target_name,
                                   healpixels=healpixels, nside=nside,
                                   nside_galaxia=nside_galaxia, magcut=magcut,
                                   faintstar_mockfile=faintstar_mockfile,
                                   faintstar_magcut=faintstar_magcut,
                                   seed=self.seed)

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
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)

        if no_spectra:
            flux = []
            meta, objmeta = self.template_photometry(data, indx, rand)
        else:
            input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype, input_meta=True)
            input_meta['SEED'][:] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'][:] = data['Z'][indx]
            input_meta['MAG'][:] = data['MAG'][indx]
            input_meta['MAGFILTER'][:] = data['MAGFILTER'][indx]

            if self.mockformat == 'galaxia':
                input_meta['TEMPLATEID'][:] = self.KDTree_query(
                    np.vstack((np.log10(data['TEFF'][indx]),
                               data['LOGG'][indx],
                               data['FEH'][indx])).T)

            # Build north/south spectra separately.
            south = np.where( data['SOUTH'][indx] == True )[0]
            north = np.where( data['SOUTH'][indx] == False )[0]
        
            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')

            for these, issouth in zip( (north, south), (False, True) ):
                if len(these) > 0:
                    # Note: no "nocolorcuts" argument!
                    flux1, _, meta1, objmeta1 = self.template_maker.make_templates(
                        input_meta=input_meta[these], south=issouth)

                    meta[these] = meta1
                    objmeta[these] = objmeta1
                    flux[these, :] = flux1

        targets, truth, objtruth = self.populate_targets_truth(
            flux, data, meta, objmeta, indx=indx, 
            seed=seed, truespectype='STAR', templatetype='STAR')
        
        return flux, self.wave, targets, truth, objtruth

    def select_targets(self, targets, truth, targetname='MWS_MAIN'):
        """Select various MWS stars and standard stars.  Input tables are modified in
        place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        targetname : :class:`str`
            Target selection cuts to apply.

        """
        if targetname == 'MWS_MAIN':
            if self.calib_only:
                tcnames = 'STD'
            else:
                tcnames = ['MWS', 'STD']
        else:
            tcnames = targetname

        if self.survey == 'sv1':
            qso_selection = 'randomforest'
        else:
            qso_selection = 'colorcuts'
            
        # Note: We pass qso_selection to cuts.apply_cuts because MWS_MAIN
        # targets can be used as QSO contaminants.
        desi_target, bgs_target, mws_target = cuts.apply_cuts(targets, tcnames=tcnames,
                                                              qso_selection=qso_selection,
                                                              survey=self.survey)

        # Subtract out the MWS_NEARBY and MWS_WD/STD_WD targeting bits, since
        # those are handled in the MWS_NEARBYMaker and WDMaker classes,
        # respectively.
        for mwsbit in self.mws_mask.names():
            if 'NEARBY' in mwsbit or 'WD' in mwsbit:
                these = mws_target & self.mws_mask.mask(mwsbit) != 0
                if np.sum(these) > 0:
                    mws_target[these] -= self.mws_mask.mask(mwsbit)
                    andthose = mws_target[these] == 0
                    if np.sum(andthose) > 0:
                        desi_target[these][andthose] -= self.desi_mask.mask('MWS_ANY')
        
        these = desi_target & self.desi_mask.mask('STD_WD') != 0
        if np.sum(these) > 0:
            desi_target[these] -= self.desi_mask.mask('STD_WD')

        self.remove_north_south_bits(desi_target, bgs_target, mws_target)
        
        targets['DESI_TARGET'] |= desi_target
        targets['BGS_TARGET'] |= bgs_target
        targets['MWS_TARGET'] |= mws_target

class MWS_NEARBYMaker(STARMaker):
    """Read MWS_NEARBY mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    no_spectra : :class:`bool`, optional
        Do not initialize template photometry.  Defaults to False.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    def __init__(self, seed=None, no_spectra=False, survey='main', **kwargs):
        super(MWS_NEARBYMaker, self).__init__(seed=seed, no_spectra=no_spectra,
                                              survey=survey)

    def read(self, mockfile=None, mockformat='mws_100pc', healpixels=None,
             nside=None, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'mws_100pc'.
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
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'mws', '100pc', 'v0.0.4', 'mock_100pc.fits')
            MockReader = ReadMWS_NEARBY()
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name='MWS_NEARBY',
                                   healpixels=healpixels, nside=nside,
                                   mock_density=mock_density)

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
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)
        
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if no_spectra:
            flux = []
            meta, objmeta = self.template_photometry(data, indx, rand)
        else:
            input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype, input_meta=True)
            input_meta['SEED'][:] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'][:] = data['Z'][indx]
            input_meta['MAG'][:] = data['MAG'][indx]
            input_meta['MAGFILTER'][:] = data['MAGFILTER'][indx]

            if self.mockformat == 'mws_100pc':
                input_meta['TEMPLATEID'][:] = self.KDTree_query(
                    np.vstack((np.log10(data['TEFF'][indx]),
                               data['LOGG'][indx],
                               data['FEH'][indx])).T)

            # Build north/south spectra separately.
            south = np.where( data['SOUTH'][indx] == True )[0]
            north = np.where( data['SOUTH'][indx] == False )[0]
        
            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')

            for these, issouth in zip( (north, south), (False, True) ):
                if len(these) > 0:
                    # Note: no "nocolorcuts" argument!
                    flux1, _, meta1, objmeta1 = self.template_maker.make_templates(
                        input_meta=input_meta[these], south=issouth)

                    meta[these] = meta1
                    objmeta[these] = objmeta1
                    flux[these, :] = flux1

        targets, truth, objtruth = self.populate_targets_truth(
            flux, data, meta, objmeta, indx=indx, 
            seed=seed, truespectype='STAR', templatetype='STAR',
            templatesubtype=data['TEMPLATESUBTYPE'][indx])

        return flux, self.wave, targets, truth, objtruth

    def select_targets(self, targets, truth, targetname='MWS'):
        """Select MWS_NEARBY targets.  Input tables are modified in place.

        Note: The selection here eventually will be done with Gaia (I think) so
        for now just do a "perfect" selection.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        targetname : :class:`str`
            Target selection cuts to apply.

        """
        desi_target, bgs_target, mws_target = cuts.apply_cuts(targets, tcnames=targetname,
                                                              survey=self.survey)

        # Subtract out *all* the MWS targeting bits except MWS_NEARBY since
        # those are separately handled in the MWS_MAINMaker and WDMaker classes.
        for mwsbit in self.mws_mask.names():
            if mwsbit == 'MWS_NEARBY':
                pass
            else:
                these = mws_target & self.mws_mask.mask(mwsbit) != 0
                if np.sum(these) > 0:
                    mws_target[these] -= self.mws_mask.mask(mwsbit)
                    andthose = mws_target[these] == 0
                    if np.sum(andthose) > 0:
                        desi_target[these][andthose] -= self.desi_mask.mask('MWS_ANY')
        
        self.remove_north_south_bits(desi_target, bgs_target, mws_target)
        
        targets['DESI_TARGET'] |= desi_target
        targets['BGS_TARGET'] |= bgs_target
        targets['MWS_TARGET'] |= mws_target

class WDMaker(SelectTargets):
    """Read WD mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    no_spectra : :class:`bool`, optional
        Do not initialize template photometry.  Defaults to False.
    calib_only : :class:`bool`, optional
        Use WDs as calibration (standard star) targets, only.  Defaults to False. 
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    wave, da_template_maker, db_template_maker = None, None, None
    KDTree_da, KDTree_db = None, None
    wd_maggies_da_north, wd_maggies_da_north = None, None
    wd_maggies_db_south, wd_maggies_db_south = None, None

    def __init__(self, seed=None, calib_only=False, no_spectra=False,
                 survey='main', **kwargs):
        from speclite import filters
        from desisim.templates import WD
        
        super(WDMaker, self).__init__(survey=survey)

        self.seed = seed
        self.objtype = 'WD'
        self.calib_only = calib_only

        if self.wave is None:
            WDMaker.wave = _default_wave()
        self.extinction = self.mw_dust_extinction()
            
        if self.da_template_maker is None:
            WDMaker.da_template_maker = WD(wave=self.wave, subtype='DA')
            
        if self.db_template_maker is None:
            WDMaker.db_template_maker = WD(wave=self.wave, subtype='DB')
        
        self.meta_da = self.da_template_maker.basemeta
        self.meta_db = self.db_template_maker.basemeta

        # Pre-compute normalized synthetic photometry for the full set of DA and
        # DB templates.
        if no_spectra and (self.wd_maggies_da_north is None or self.wd_maggies_da_south is None or
            self.wd_maggies_db_north is None or self.wd_maggies_db_south is None):
            log.info('Caching WD template photometry.')

            if 'SYNTH_SDSS2010_G' in self.meta_da.colnames: # from DESI-COLORS HDU (basis templates >=v3.1)
                maggies_da_north = self.meta_da[['SYNTH_BASS_G', 'SYNTH_BASS_R', 'SYNTH_MZLS_Z',
                                                 'SYNTH_WISE2010_W1', 'SYNTH_WISE2010_W2']]
                maggies_db_north = self.meta_db[['SYNTH_BASS_G', 'SYNTH_BASS_R', 'SYNTH_MZLS_Z',
                                                 'SYNTH_WISE2010_W1', 'SYNTH_WISE2010_W2']]
                maggies_da_south = self.meta_da[['SYNTH_DECAM2014_G', 'SYNTH_DECAM2014_R', 'SYNTH_DECAM2014_Z',
                                                 'SYNTH_WISE2010_W1', 'SYNTH_WISE2010_W2']]
                maggies_db_south = self.meta_db[['SYNTH_DECAM2014_G', 'SYNTH_DECAM2014_R', 'SYNTH_DECAM2014_Z',
                                                 'SYNTH_WISE2010_W1', 'SYNTH_WISE2010_W2']]

                # Normalize to sdss-g
                def _get_maggies(outmaggies, normmaggies):
                    for filt, flux in zip( outmaggies.colnames, ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2') ):
                        outmaggies[filt] /= normmaggies
                        outmaggies.rename_column(filt, flux)
                    return outmaggies
                    
                WDMaker.wd_maggies_da_north = _get_maggies(maggies_da_north.copy(), self.meta_da['SYNTH_SDSS2010_G'])
                WDMaker.wd_maggies_da_south = _get_maggies(maggies_da_south.copy(), self.meta_da['SYNTH_SDSS2010_G'])
                WDMaker.wd_maggies_db_north = _get_maggies(maggies_db_north.copy(), self.meta_db['SYNTH_SDSS2010_G'])
                WDMaker.wd_maggies_db_south = _get_maggies(maggies_db_south.copy(), self.meta_db['SYNTH_SDSS2010_G'])
            else:
                wave = self.da_template_maker.basewave
                flux_da, flux_db = self.da_template_maker.baseflux, self.db_template_maker.baseflux

                maggies_da_north = self.bassmzlswise.get_ab_maggies(flux_da, wave, mask_invalid=True)
                maggies_db_north = self.bassmzlswise.get_ab_maggies(flux_db, wave, mask_invalid=True)
                maggies_da_south = self.decamwise.get_ab_maggies(flux_da, wave, mask_invalid=True)
                maggies_db_south = self.decamwise.get_ab_maggies(flux_db, wave, mask_invalid=True)

                # Normalize to sdss-g
                normfilter = filters.load_filters('sdss2010-g')
                def _get_maggies(flux, wave, outmaggies, normfilter):
                    normmaggies = normfilter.get_ab_maggies(flux, wave, mask_invalid=True)
                    for filt, flux in zip( outmaggies.colnames, ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2') ):
                        outmaggies[filt] /= normmaggies[normfilter.names[0]]
                        outmaggies.rename_column(filt, flux)
                    return outmaggies

                WDMaker.wd_maggies_da_north = _get_maggies(flux_da, wave, maggies_da_north.copy(), normfilter)
                WDMaker.wd_maggies_da_south = _get_maggies(flux_da, wave, maggies_da_south.copy(), normfilter)
                WDMaker.wd_maggies_db_north = _get_maggies(flux_db, wave, maggies_db_north.copy(), normfilter)
                WDMaker.wd_maggies_db_south = _get_maggies(flux_db, wave, maggies_db_south.copy(), normfilter)

        # Build the KD Trees
        logteff_da = np.log10(self.meta_da['TEFF'].data)
        logteff_db = np.log10(self.meta_db['TEFF'].data)
        logg_da = self.meta_da['LOGG'].data
        logg_db = self.meta_db['LOGG'].data

        self.param_min_da = ( logteff_da.min(), logg_da.min() )
        self.param_range_da = ( np.ptp(logteff_da), np.ptp(logg_da) )
        self.param_min_db = ( logteff_db.min(), logg_db.min() )
        self.param_range_db = ( np.ptp(logteff_db), np.ptp(logg_db) )

        if self.KDTree_da is None:
            WDMaker.KDTree_da = self.KDTree_build(np.vstack((logteff_da, logg_da)).T, subtype='DA')
            
        if self.KDTree_db is None:
            WDMaker.KDTree_db = self.KDTree_build(np.vstack((logteff_db, logg_db)).T, subtype='DB')

    def read(self, mockfile=None, mockformat='mws_wd', healpixels=None,
             nside=None, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'mws_wd'.
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
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'mws', 'wd', 'v0.0.2', 'mock_wd.fits')
            MockReader = ReadMWS_WD()
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   mock_density=mock_density)

        return data

    def wd_template_photometry(self, data=None, indx=None, rand=None,
                               subtype='DA', south=True):
        """Get stellar photometry from the templates themselves, by-passing the
        generation of spectra.

        """
        if rand is None:
            rand = np.random.RandomState()

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)
        
        meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        meta['SEED'][:] = rand.randint(2**31, size=nobj)
        meta['REDSHIFT'][:] = data['Z'][indx]
        meta['MAG'][:] = data['MAG'][indx]
        meta['MAGFILTER'][:] = data['MAGFILTER'][indx]
        meta['SUBTYPE'][:] = data['TEMPLATESUBTYPE'][indx]

        objmeta['TEFF'][:] = data['TEFF'][indx]
        objmeta['LOGG'][:] = data['LOGG'][indx]

        if self.mockformat == 'mws_wd':
            templateid = self.KDTree_query(
                np.vstack((np.log10(data['TEFF'][indx].data),
                           data['LOGG'][indx])).T, subtype=subtype)
        normmag = 1e9 * 10**(-0.4 * data['MAG'][indx]) # nanomaggies

        if south:
            if subtype == 'DA':
                wd_maggies = self.wd_maggies_da_south
            elif subtype == 'DB':
                wd_maggies = self.wd_maggies_db_south
            else:
                log.warning('Unrecognized subtype {}!'.format(subtype))
                raise ValueError
        else:
            if subtype == 'DA':
                wd_maggies = self.wd_maggies_da_north
            elif subtype == 'DB':
                wd_maggies = self.wd_maggies_db_north
            else:
                log.warning('Unrecognized subtype {}!'.format(subtype))
                raise ValueError
            
        for key in ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'):
            meta[key][:] = wd_maggies[key][templateid] * normmag

        return meta, objmeta

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
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)
        
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if self.mockformat == 'mws_wd':
            if no_spectra:
                flux = []
                meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            else:
                input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype, input_meta=True)
                input_meta['SEED'][:] = rand.randint(2**31, size=nobj)
                input_meta['REDSHIFT'][:] = data['Z'][indx]
                input_meta['MAG'][:] = data['MAG'][indx]
                input_meta['MAGFILTER'][:] = data['MAGFILTER'][indx]

                meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
                flux = np.zeros([nobj, len(self.wave)], dtype='f4')

            allsubtype = data['TEMPLATESUBTYPE'][indx]
            for subtype in ('DA', 'DB'):
                match = np.where(allsubtype == subtype)[0]
                if len(match) > 0:
                    if not no_spectra:
                        input_meta['TEMPLATEID'][match] = self.KDTree_query(
                            np.vstack((np.log10(data['TEFF'][indx][match].data),
                                       data['LOGG'][indx][match])).T,
                            subtype=subtype)

                    # Build north/south spectra separately.
                    south = np.where( data['SOUTH'][indx][match] == True )[0]
                    north = np.where( data['SOUTH'][indx][match] == False )[0]

                    for these, issouth in zip( (north, south), (False, True) ):
                        if len(these) > 0:
                            if no_spectra:
                                meta1, objmeta1 = self.wd_template_photometry(
                                    data, indx[match][these], rand, subtype,
                                    south=issouth)
                                meta[match[these]] = meta1
                                objmeta[match[these]] = objmeta1
                            else:
                                # Note: no "nocolorcuts" argument!
                                template_maker = getattr(self, '{}_template_maker'.format(subtype.lower()))
                                flux1, _, meta1, objmeta1 = template_maker.make_templates(
                                    input_meta=input_meta[match[these]], south=issouth)

                                meta[match[these]] = meta1
                                objmeta[match[these]] = objmeta1
                                flux[match[these], :] = flux1

        targets, truth, objtruth = self.populate_targets_truth(
            flux, data, meta, objmeta, indx=indx, 
            seed=seed, truespectype='WD', templatetype='WD',
            templatesubtype=allsubtype)
        
        return flux, self.wave, targets, truth, objtruth

    def select_targets(self, targets, truth, targetname='WD'):
        """Select MWS_WD targets and STD_WD standard stars.  Input tables are modified
        in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        targetname : :class:`str`
            Target selection cuts to apply.

        """
        if targetname == 'WD':
            if self.calib_only:
                tcnames = 'STD'
            else:
                tcnames = ['MWS', 'STD']
        else:
            tcnames = targetname

        # Assume that MWS_MAIN and MWS_NEARBY objects are *never* selected from
        # the WD mock.
        
        desi_target, bgs_target, mws_target = cuts.apply_cuts(targets, tcnames=tcnames,
                                                              survey=self.survey)

        self.remove_north_south_bits(desi_target, bgs_target, mws_target)
        
        targets['DESI_TARGET'] |= desi_target
        targets['BGS_TARGET'] |= bgs_target
        targets['MWS_TARGET'] |= mws_target

class SKYMaker(SelectTargets):
    """Read SKY mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    wave = None
    
    def __init__(self, seed=None, survey='main', **kwargs):
        super(SKYMaker, self).__init__(survey=survey)

        self.seed = seed
        self.objtype = 'SKY'

        if self.wave is None:
            SKYMaker.wave = _default_wave()
        self.extinction = self.mw_dust_extinction()
        
    def read(self, mockfile=None, mockformat='uniformsky', healpixels=None,
             nside=None, only_coords=False, mock_density=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'gaussianfield'.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        only_coords : :class:`bool`, optional
            For various applications, only read the target coordinates.
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
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'uniformsky', '0.2', 'uniformsky-2048-0.2.fits')
            MockReader = ReadUniformSky()
        elif self.mockformat == 'gaussianfield':
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'GaussianRandomField', '0.0.1', '2048', 'random.fits')
            MockReader = ReadGaussianField()
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   only_coords=only_coords,
                                   mock_density=mock_density)

        return data

    def make_spectra(self, data=None, indx=None, seed=None, no_spectra=False):
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
        objtruth : :class:`astropy.table.Table`
            Corresponding objtype-specific truth table (if applicable).
        
        """
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if no_spectra:
            flux = []
        else:
            flux = np.zeros((nobj, len(self.wave)), dtype='f4')
            
        meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        meta['SEED'][:] = rand.randint(2**31, size=nobj)
        meta['REDSHIFT'][:] = data['Z'][indx]
        
        targets, truth, objtruth = self.populate_targets_truth(
            flux, data, meta, objmeta, indx=indx, seed=seed,
            truespectype='SKY', templatetype='SKY')

        return flux, self.wave, targets, truth, objtruth

    def select_targets(self, targets, truth, targetname='SKY'):
        """Select SKY targets (i.e., everything).  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        targetname : :class:`str`
            Target selection cuts to apply.

        """
        targets['DESI_TARGET'] |= self.desi_mask.mask(targetname)

class BuzzardMaker(SelectTargets):
    """Read Buzzard mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.
    no_spectra : :class:`bool`, optional
        Do not pre-select extragalactic contaminants.  Defaults to False.
    survey : :class:`str`, optional
        Specify which target masks yaml file to use.  The options are `main`
        (main survey) and `sv1` (first iteration of SV).  Defaults to `main`.

    """
    wave, template_maker = None, None
    
    def __init__(self, seed=None, nside_chunk=128, no_spectra=False,
                 survey='main', **kwargs):
        super(BuzzardMaker, self).__init__(survey=survey)

        self.seed = seed
        self.objtype = 'BGS'
        self.nside_chunk = nside_chunk

        if self.wave is None:
            BuzzardMaker.wave = _default_wave()
        self.extinction = self.mw_dust_extinction()

        if self.template_maker is None:
            from desisim.templates import BGS
            BuzzardMaker.template_maker = BGS(wave=self.wave)

        # Store the metadata table and then pre-select extragalactic
        # contaminants/templates, which will add columns to self.meta in place
        # (unless no_spectra=True, in which case skip it).
        self.meta = self.template_maker.basemeta
        if not no_spectra:
            self.extragalactic_contaminants(seed)

    def extragalactic_contaminants(self, seed, nmonte=100):
        """Pre-select Buzzard/BGS templates that could be photometric contaminants.

        """
        from astropy.table import Column
        from desisim.io import find_basis_template
        from desitarget.cuts import isELG_colors, isQSO_colors

        rand = np.random.RandomState(seed)

        # Average grzW1W2 imaging depth and scattered photometry
        psfdepth_sigma = 10**(-0.4 * (np.array((24.65, 23.61, 22.84)) - 22.5)) / 5 # 1-sigma, nanomaggies
        wisedepth_sigma = 10**(-0.4 * (np.array((22.3, 21.8)) - 22.5))             # 1-sigma, nanomaggies
        depth_sigma = np.hstack( (psfdepth_sigma, wisedepth_sigma) )

        gflux_err  = rand.normal(loc=0, scale=depth_sigma[0], size=nmonte)
        rflux_err  = rand.normal(loc=0, scale=depth_sigma[1], size=nmonte)
        zflux_err  = rand.normal(loc=0, scale=depth_sigma[2], size=nmonte)
        w1flux_err = rand.normal(loc=0, scale=depth_sigma[3], size=nmonte)
        w2flux_err = rand.normal(loc=0, scale=depth_sigma[4], size=nmonte)

        # Unpack the metadata table.
        templatefile = find_basis_template('BGS')
        hdr = fitsio.read_header(templatefile, ext='DESI-COLORS')
        zgrid = hdr['DZ'] * np.arange(hdr['NZ']) + hdr['ZMIN']
        nz, nt = len(zgrid), len(self.meta)

        bigzgrid = np.tile(zgrid, (nt, 1)).reshape(nt, nz).flatten()
        bigtemplateid = np.tile(self.meta['TEMPLATEID'].data, (1, nz)).reshape(nz, nt).T.flatten()

        # First ELGs--

        # ELG contaminants will be (most likely) lower-redshift (z<0.7)
        # interlopers, so focus on that redshift range, extract the fluxes, and
        # scale to g=22 (to satisfy the bright cut).
        elgzcut = bigzgrid < 0.75
        elgscale = (10**(-0.4 * (22 - 22.5) ) / self.meta['SYNTH_DECAM2014_G'].data).flatten()[elgzcut]

        gflux  = elgscale * self.meta['SYNTH_DECAM2014_G'].data.flatten()[elgzcut]
        rflux  = elgscale * self.meta['SYNTH_DECAM2014_R'].data.flatten()[elgzcut]
        zflux  = elgscale * self.meta['SYNTH_DECAM2014_Z'].data.flatten()[elgzcut]
        w1flux = elgscale * self.meta['SYNTH_WISE2010_W1'].data.flatten()[elgzcut]
        w2flux = elgscale * self.meta['SYNTH_WISE2010_W2'].data.flatten()[elgzcut]

        fiberflux_fraction = self.fiberflux_fraction['ELG']
        gfiberflux = fiberflux_fraction * gflux
        rfiberflux = fiberflux_fraction * rflux 
        zfiberflux = fiberflux_fraction * zflux

        # Monte Carlo the colors and take the union of all the templates that
        # scatter into the ELG color-box.
        iselg = np.zeros(len(gflux)).astype(bool)
        for ii in range(nmonte):
            _iselg_vlo, _iselg = isELG_colors(
                south = True,
                gflux = gflux + gflux_err[ii],
                rflux = rflux + rflux_err[ii],
                zflux = zflux + zflux_err[ii],
                gfiberflux = gfiberflux,
                rfiberflux = rfiberflux,
                zfiberflux = zfiberflux,
                w1flux= w1flux + w1flux_err[ii],
                w2flux= w2flux + w2flux_err[ii])
            iselg = np.logical_or(np.logical_or(iselg, _iselg_vlo), _iselg)

        # Determine which templates scatter into the ELG color-box **in each
        # redshift bin.***
        idelg  = np.digitize(bigzgrid[elgzcut][iselg], zgrid)
        contam_elg = []
        for ii in range(len(zgrid)):
            these = np.where(idelg == ii)[0]
            #print(bigzgrid[elgzcut][iselg][these], len(these))
            contam_elg.append( bigtemplateid[elgzcut][iselg][these] )
            
        #these = np.unique(bigtemplateid[elgzcut][iselg])
        #contam_elg = np.zeros(nt).astype(bool)
        #contam_elg[these] = True
        #self.meta.add_column(Column(name='CONTAM_ELG', data=contam_elg))
        
        #log.debug('Found {} / {} templates that scatter into the ELG color box with z < 0.75.'.format(len(these), nt))

        # Now QSOs--

        # I'm not sure what the redshift distribution of QSO contaminants will be...
        qsozcut = bigzgrid < 2.5

        gflux  = self.meta['SYNTH_DECAM2014_G'].data.flatten()[qsozcut]
        rflux  = self.meta['SYNTH_DECAM2014_R'].data.flatten()[qsozcut]
        zflux  = self.meta['SYNTH_DECAM2014_Z'].data.flatten()[qsozcut]
        w1flux = self.meta['SYNTH_WISE2010_W1'].data.flatten()[qsozcut]
        w2flux = self.meta['SYNTH_WISE2010_W2'].data.flatten()[qsozcut]

        # Monte Carlo the colors and take the union of all the templates that
        # scatter into the QSO color-box.
        isqso = np.zeros(len(gflux)).astype(bool)
        for ii in range(nmonte):
            isqso = np.logical_or( (isqso), (isQSO_colors(
                south = True, 
                gflux = gflux + gflux_err[ii],
                rflux = rflux + rflux_err[ii],
                zflux = zflux + zflux_err[ii],
                w1flux= w1flux + w1flux_err[ii],
                w2flux= w2flux + w2flux_err[ii])) )

        idqso  = np.digitize(bigzgrid[qsozcut][isqso], zgrid)
        contam_qso = []
        for ii in range(len(zgrid)):
            these = np.where(idqso == ii)[0]
            #print(bigzgrid[qsozcut][isqso][these], len(these))
            contam_qso.append( bigtemplateid[qsozcut][isqso][these] )

        # Keep the results.
        self.contam_zgrid = zgrid
        self.contam_elg = contam_elg
        self.contam_qso = contam_qso

    def read(self, mockfile=None, mockformat='buzzard', healpixels=None,
             nside=None, nside_buzzard=8, target_name='', magcut=None,
             only_coords=False, **kwargs):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the mock catalog to read.
        mockformat : :class:`str`
            Mock catalog format.  Defaults to 'buzzard'.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        nside_buzzard : :class:`int`
            Healpixel nside indicating how the mock on-disk has been organized.
            Defaults to 8.
        target_name : :class:`str`
            Name of the target being read.  Defaults to ''.
        magcut : :class:`float`
            Magnitude cut (hard-coded to DECam r-band) to subselect targets
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
        if self.mockformat == 'buzzard':
            self.default_mockfile = os.path.join(
                os.getenv('DESI_ROOT'), 'mocks', 'buzzard', 'buzzard_v1.6_desicut')
            MockReader = ReadBuzzard()
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile

        data = MockReader.readmock(mockfile, target_name=target_name, 
                                   healpixels=healpixels, nside=nside,
                                   nside_buzzard=nside_buzzard,
                                   only_coords=only_coords,
                                   magcut=magcut)

        return data

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
        objtruth : :class:`astropy.table.Table`
            Corresponding objtype-specific truth table (if applicable).
        
        """
        if seed is None:
            seed = self.seed
        rand = np.random.RandomState(seed)

        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        if no_spectra:
            flux = []
            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            self._nospectra_photometry(meta, rand, data, indx, data['TARGET_NAME'], contaminants=True)
        else:
            input_meta, _ = empty_metatable(nmodel=nobj, objtype=self.objtype)
            input_meta['SEED'][:] = rand.randint(2**31, size=nobj)
            input_meta['REDSHIFT'][:] = data['Z'][indx]
            
            vdisp = self._sample_vdisp(data['RA'][indx], data['DEC'][indx], mean=1.9,
                                       sigma=0.15, seed=seed, nside=self.nside_chunk)

            input_meta['MAG'][:] = data['MAG'][indx]
            input_meta['MAGFILTER'][:] = data['MAGFILTER'][indx]
            
            # Choose a template from the right redshift bin of contaminants
            # depending on what type of contaminant we're simulating.
            if data['TARGET_NAME'].upper() == 'QSO':
                contamid  = np.digitize(input_meta['REDSHIFT'], self.contam_zgrid)
                templateid = []
                for cid in np.unique(contamid):
                    if len(self.contam_qso[cid]) > 0:
                        templateid.append( rand.choice(self.contam_qso[cid], np.count_nonzero(cid == contamid)) )
                    else:
                        # throw-away templates
                        templateid.append( rand.choice(self.meta['TEMPLATEID'].data, np.count_nonzero(cid == contamid)) )
                templateid = np.hstack(templateid)
            elif data['TARGET_NAME'].upper() == 'ELG':
                contamid  = np.digitize(input_meta['REDSHIFT'], self.contam_zgrid)
                templateid = []
                for cid in np.unique(contamid):
                    if len(self.contam_elg[cid]) > 0:
                        templateid.append( rand.choice(self.contam_elg[cid], np.count_nonzero(cid == contamid)) )
                    else:
                        # throw-away templates
                        templateid.append( rand.choice(self.meta['TEMPLATEID'].data, np.count_nonzero(cid == contamid)) )
                templateid = np.hstack(templateid)
            else:
                log.warning('Need to pre-select more classes of contaminants!')
                templateid = rand.choice(self.meta['TEMPLATEID'], nobj)

            input_meta['TEMPLATEID'][:] = templateid
                
            # Build north/south spectra separately.
            south = np.where( data['SOUTH'][indx] == True )[0]
            north = np.where( data['SOUTH'][indx] == False )[0]

            meta, objmeta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')

            for these, issouth in zip( (north, south), (False, True) ):
                if len(these) > 0:
                    flux1, _, meta1, objmeta1 = self.template_maker.make_templates(
                        input_meta=input_meta[these], vdisp=vdisp[these], south=issouth,
                        nocolorcuts=True)

                    meta[these] = meta1
                    objmeta[these] = objmeta1
                    flux[these, :] = flux1

        targets, truth, objtruth = self.populate_targets_truth(
            flux, data, meta, objmeta, indx=indx, seed=seed,
            truespectype='GALAXY', templatetype='BGS')

        return flux, self.wave, targets, truth, objtruth

    def select_targets(self, targets, truth, targetname='BGS'):
        """Select extragalactic contaminants.  Input tables are modified in place.

        Parameters
        ----------
        targets : :class:`astropy.table.Table`
            Input target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.
        targetname : :class:`str`
            Target selection cuts to apply.

        """
        # There's an inconsistency here if use_simqso=False in the configuration
        # file...we should be setting optical=True.
        if self.survey == 'sv1':
            qso_selection = 'randomforest'
        else:
            qso_selection = 'colorcuts'
            
        desi_target, bgs_target, mws_target = cuts.apply_cuts(targets, tcnames=targetname,
                                                              qso_selection=qso_selection,
                                                              survey=self.survey)

        self.remove_north_south_bits(desi_target, bgs_target, mws_target)
        
        targets['DESI_TARGET'] |= desi_target
        targets['BGS_TARGET'] |= bgs_target
        targets['MWS_TARGET'] |= mws_target
