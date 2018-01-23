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
from astropy.table import Table, Column

from desisim.io import read_basis_templates
from desiutil.brick import brickname as get_brickname_from_radec
from desimodel.footprint import radec2pix

from desiutil.log import get_logger, DEBUG
log = get_logger()

"""How to distribute 52 user bits of targetid.

Used to generate target IDs as combination of input file and row in input file.
Sets the maximum number of rows per input file for mocks using this scheme to
generate target IDs

"""
# First 32 bits are row
ENCODE_ROW_END     = 32
ENCODE_ROW_MASK    = 2**ENCODE_ROW_END - 2**0
ENCODE_ROW_MAX     = ENCODE_ROW_MASK
# Next 20 bits are file
ENCODE_FILE_END    = 52
ENCODE_FILE_MASK   = 2**ENCODE_FILE_END - 2**ENCODE_ROW_END
ENCODE_FILE_MAX    = ENCODE_FILE_MASK >> ENCODE_ROW_END

try:
    from scipy import constants
    C_LIGHT = constants.c/1000.0
except TypeError: # This can happen during documentation builds.
    C_LIGHT = 299792458.0/1000.0

def encode_rownum_filenum(rownum, filenum):
    """Encodes row and file number in 52 packed bits.

    Parameters
    ----------
    rownum : :class:`int`
        Row in input file.
    filenum : :class:`int`
        File number in input file set.

    Returns
    -------
    :class:`numpy.ndarray`
        52 packed bits encoding row and file number.

    """
    assert(np.shape(rownum) == np.shape(filenum))
    assert(np.all(rownum  >= 0))
    assert(np.all(rownum  <= int(ENCODE_ROW_MAX)))
    assert(np.all(filenum >= 0))
    assert(np.all(filenum <= int(ENCODE_FILE_MAX)))

    # This should be a 64 bit integer.
    encoded_value = ( (np.asarray(filenum,dtype=np.uint64) << ENCODE_ROW_END) +
                      np.asarray(rownum, dtype=np.uint64) )

    # Note return signed
    return np.asarray(encoded_value, dtype=np.int64)

def decode_rownum_filenum(encoded_values):
    """Invert encode_rownum_filenum to obtain row number and file number.

    Parameters
    ----------
    encoded_values : :class:`int64 ndarray`
        Input encoded values.

    Returns
    -------
    filenum : :class:`str`
        File number.
    rownum : :class:`int`
        Row number.
    
    """
    filenum = (np.asarray(encoded_values,dtype=np.uint64) & ENCODE_FILE_MASK) >> ENCODE_ROW_END
    rownum  = (np.asarray(encoded_values,dtype=np.uint64) & ENCODE_ROW_MASK)
    return rownum, filenum

def make_mockid(objid, n_per_file):
    """
    Compute mockid from row and file IDs.

    Parameters
    ----------
    objid : :class:`numpy.ndarray`
        Row identification number.
    n_per_file : :class:`list`
        Number of items per file that went into objid.

    Returns
    -------
    mockid : :class:`numpy.ndarray`
        Encoded row and file ID.

    """
    n_files = len(n_per_file)
    n_obj = len(objid)

    n_p_file = np.array(n_per_file)
    n_per_file_cumsum = n_p_file.cumsum()

    filenum = np.zeros(n_obj, dtype='int64')
    for n in range(1, n_files):
        filenum[n_per_file_cumsum[n-1]:n_per_file_cumsum[n]] = n

    return encode_rownum_filenum(objid, filenum)

def mw_transmission(source_data, dust_dir=None):
    """Compute the grzW1W2 Galactic transmission for every object.
    
    Parameters
    ----------
    source_data : :class:`dict`
        Input dictionary of sources with RA, Dec coordinates, modified on output
        to contain reddening and the MW transmission in various bands.
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
    source_data['EBV'] = sfdmap.ebv(source_data['RA'],
                                    source_data['DEC'],
                                    mapdir=dust_dir)

    for band in ('G', 'R', 'Z', 'W1', 'W2'):
        source_data['MW_TRANSMISSION_{}'.format(band)] = 10**(-0.4 * extcoeff[band] * source_data['EBV'])

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
    # Do we need obsconditions or not?!?
    #targets.add_column(Column(name='OBSCONDITIONS', length=nobj, dtype='i8'))

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

def _indesi(data):
    """Demand that all objects lie within the DESI footprint."""
    import desimodel.io
    import desimodel.footprint

    n_obj = len(source_data['RA'])
    tiles = desimodel.io.load_tiles()
    if n_obj > 0:
        indesi = desimodel.footprint.is_point_in_desi(tiles, source_data['RA'], source_data['DEC'])
        for k in source_data.keys():
            if type(source_data[k]) is np.ndarray:
                if n_obj == len(source_data[k]):
                    source_data[k] = source_data[k][indesi]

def _sample_vdisp(data, mean=1.9, sigma=0.15, fracvdisp=(0.1, 1),
                  rand=None, nside=128):
    """Assign velocity dispersions to a subset of objects."""
    if rand is None:
        rand = np.random.RandomState()

    def _sample(nmodel=1):
        nvdisp = int(np.max( ( np.min( ( np.round(nmodel * fracvdisp[0]), fracvdisp[1] ) ), 1 ) ))
        vvdisp = 10**rand.normal(loc=mean, scale=sigma, size=nvdisp)
        return rand.choice(vvdisp, nmodel)

    # Hack! Assign the same velocity dispersion to galaxies in the same healpixel.
    nobj = len(data['RA'])
    vdisp = np.zeros(nobj)

    healpix = radec2pix(nside, data['RA'], data['DEC'])
    for pix in set(healpix):
        these = np.in1d(healpix, pix)
        vdisp[these] = _sample(nmodel=np.count_nonzero(these))

    return vdisp

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
        from desitarget import (desi_mask, bgs_mask, mws_mask,
                                contam_mask, obsconditions)
        
        self.desi_mask = desi_mask
        self.bgs_mask = bgs_mask
        self.mws_mask = mws_mask
        self.contam_mask = contam_mask
        self.obsconditions = obsconditions

    def scatter_photometry(self, data, truth, targets, indx=None, psf=True,
                           qaplot=False):
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
        qaplot : :class:`bool`, optional
            Generate a QA plot for debugging.

        """
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
            targets[fluxkey][:] = truth[fluxkey] + self.rand.normal(scale=sigma)

        for band in ('W1', 'W2'):
            fluxkey = 'FLUX_{}'.format(band)
            depthkey = 'PSFDEPTH_{}'.format(band)

            sigma = 1 / np.sqrt(data[depthkey][indx]) / 5 # nanomaggies, 1-sigma
            targets[fluxkey][:] = truth[fluxkey] + self.rand.normal(scale=sigma)

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

    def populate_targets_truth(self, data, meta, indx=None, psf=True):
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
        psf : :class:`bool`, optional
            For point sources (e.g., QSO, STAR) use the PSFDEPTH values,
            otherwise use GALDEPTH.  Defaults to True.

        Returns
        -------
        targets : :class:`astropy.table.Table`
            Target catalog.
        truth : :class:`astropy.table.Table`
            Corresponding truth table.

        """
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
        for key, source_key in zip( ['TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TRUESPECTYPE'],
                                    ['TEMPLATETYPE', 'TEMPLATESUBTYPE', 'TRUESPECTYPE'] ):
            if isinstance(data[source_key], np.ndarray):
                truth[key][:] = data[source_key][indx]
            else:
                truth[key][:] = np.repeat(data[source_key], nobj)

        # Add shapes and sizes (if available for this target class).
        if 'SHAPEEXP_R' in data.keys():
            for key in ('SHAPEEXP_R', 'SHAPEEXP_E1', 'SHAPEEXP_E2',
                        'SHAPEDEV_R', 'SHAPEDEV_E1', 'SHAPEDEV_E2'):
                targets[key][:] = data[key][indx]

        # Copy various quantities from the metadata table.
        for key in ('TEMPLATEID', 'SEED', 'REDSHIFT', 'MAG', 'VDISP', 'FLUX_G', 'FLUX_R', 'FLUX_Z',
                    'FLUX_W1', 'FLUX_W2', 'OIIFLUX', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'):
            truth[key.replace('REDSHIFT', 'TRUEZ')][:] = meta[key]

        # Scatter the photometry based on the depth.
        self.scatter_photometry(data, truth, targets, indx=indx, psf=psf)

        return targets, truth

class ReadGaussianField(object):
    """Read a Gaussian random field style mock catalog.

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    def __init__(self, dust_dir=None, bricksize=0.25):
        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=[], nside=[], target_name=''):
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

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data files are not found.
        ValueError
            If mockfile is not defined or if healpixels is not a scalar.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError
            
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        if len(np.atleast_1d(healpixels)) != 1 and len(np.atleast_1d(nside)) != 1:
            log.warning('Healpixels and nside must be scalar inputs.')
            raise ValueError
        
        # Read the ra,dec coordinates, generate mockid, and then restrict to the
        # input healpixel.
        log.info('Reading {}'.format(mockfile))
        radec = fitsio.read(mockfile, columns=['RA', 'DEC'], upper=True, ext=1)
        nobj = len(radec)

        files = list()
        n_per_file = list()
        files.append(mockfile)
        n_per_file.append(nobj)

        objid = np.arange(nobj, dtype='i8')
        mockid = make_mockid(objid, n_per_file)

        log.info('Assigning healpix pixels with nside = {}'.format(nside))
        allpix = radec2pix(nside, radec['RA'], radec['DEC'])
        cut = np.where( np.in1d(allpix, healpixels)*1 )[0]

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()
        else:
            log.info('Trimmed to {} {}s in healpixel(s) {}'.format(nobj, target_name, healpixels))

        objid = objid[cut]
        mockid = mockid[cut]
        ra = radec['RA'][cut].astype('f8') % 360.0 # enforce 0 < ra < 360
        dec = radec['DEC'][cut].astype('f8')
        del radec

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
               'OBJID': objid, 'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz,
               'FILES': files, 'N_PER_FILE': n_per_file}

        # Add MW transmission and the imaging depth.
        mw_transmission(out, dust_dir=self.dust_dir)
        imaging_depth(out)

        return out

class ReadGalaxia(object):
    """Read a Galaxia style mock catalog.

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    def __init__(self, bricksize=0.25, dust_dir=None):
        self.bricksize = bricksize
        self.dust_dir = dust_dir

    def readmock(self, mockfile=None, healpixels=[], nside=[], target_name='',
                 nside_galaxia=8, magcut=None):
        """Read the catalog.

        Parameters
        ----------
        mockfile : :class:`str`
            Full path to the top-level directory of the Galaxia mock catalog.
        healpixels : :class:`int`
            Healpixel number to read.
        nside : :class:`int`
            Healpixel nside corresponding to healpixels.
        target_name : :class:`str`
            Name of the target being read (e.g., MWS_MAIN).
        nside_galaxia : :class:`int`
            Healpixel nside indicating how the mock on-disk has been organized.
            Defaults to 8.
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
            If the mock data files are not found.
        ValueError
            If mockfile is not defined or if healpixels is not a scalar.

        """
        import healpy as hp
        from desitarget.mock.io import get_healpix_dir, findfile

        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError
        
        mockfile_nside = os.path.join(mockfile, str(nside_galaxia))
        if not os.path.isdir(mockfile_nside):
            log.warning('Galaxia top-level directory {} not found!'.format(mockfile_nside))
            raise IOError

        if len(np.atleast_1d(healpixels)) != 1 and len(np.atleast_1d(nside)) != 1:
            log.warning('Healpixels and nside must be scalar inputs.')
            raise ValueError

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

        files = list()
        n_per_file = list()
        files.append(galaxiafile)
        n_per_file.append(nobj)

        objid = np.arange(nobj, dtype='i8')
        mockid = make_mockid(objid, n_per_file)

        allpix = radec2pix(nside, radec['RA'], radec['DEC'])
        cut = np.where( np.in1d(allpix, healpixels)*1 )[0]

        nobj = len(cut)
        if nobj == 0:
            return dict()

        objid = objid[cut]
        mockid = mockid[cut]
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

        # Temporary hack to select SDSS standards se extinction-corrected SDSS mags.
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
                objid = objid[cut]
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
               'OBJID': objid, 'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': mag, 'MAG_OBS': mag_obs,
               'TEFF': teff, 'LOGG': logg, 'FEH': feh,
               'NORMFILTER': 'sdss2010-r', 'BOSS_STD': boss_std, 
               'FILES': files, 'N_PER_FILE': n_per_file}

        # Add MW transmission and the imaging depth.
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

class ReadLyaCoLoRe(object):
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
        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=[], nside=[], target_name='LYA',
                 nside_lya=16):
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

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data files are not found.
        ValueError
            If mockfile is not defined or if healpixels is not a scalar.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        mockdir = os.path.dirname(mockfile)
    
        if len(np.atleast_1d(healpixels)) != 1 and len(np.atleast_1d(nside)) != 1:
            log.warning('Healpixels and nside must be scalar inputs.')
            raise ValueError

        # Read the ra,dec coordinates and then restrict to the desired
        # healpixels.
        log.info('Reading {}'.format(mockfile))
        tmp = fitsio.read(mockfile, columns=['RA', 'DEC', 'MOCKID' ,'Z', 'PIXNUM'],
                          upper=True, ext=1)
        
        ra = tmp['RA'].astype('f8') % 360.0 # enforce 0 < ra < 360
        dec = tmp['DEC'].astype('f8')            
        zz = tmp['Z'].astype('f4')
        objid = (tmp['MOCKID'].astype(float)).astype(int) # will change
        mockpix = tmp['PIXNUM']
        mockid = objid.copy()
        del tmp

        log.info('Assigning healpix pixels with nside = {}'.format(nside))
        allpix = radec2pix(nside, ra, dec)
        cut = np.where( np.in1d(allpix, healpixels)*1 )[0]

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()
        else:
            log.info('Trimmed to {} {}s in healpixel(s) {}'.format(nobj, target_name, healpixels))

        ra = ra[cut]
        dec = dec[cut]
        zz = zz[cut]
        objid = objid[cut]
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
               'LYAFILES': np.array(lyafiles),
               'OBJID': objid, 'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz}

        # Add MW transmission and the imaging depth.
        mw_transmission(out, dust_dir=self.dust_dir)
        imaging_depth(out)

        return out

class ReadMXXL(object):
    """Read a MXXL mock catalog of BGS targets.

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    def __init__(self, dust_dir=None, bricksize=0.25):
        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=[], nside=[], target_name='BGS',
                 magcut=None):
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

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data files are not found.
        ValueError
            If mockfile is not defined or if healpixels is not a scalar.

        """
        import h5py
        
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        if len(np.atleast_1d(healpixels)) != 1 and len(np.atleast_1d(nside)) != 1:
            log.warning('Healpixels and nside must be scalar inputs.')
            raise ValueError

        # Read the ra,dec coordinates, generate mockid, and then restrict to the
        # desired healpixels.
        f = h5py.File(mockfile)
        ra  = f['Data/ra'][...].astype('f8') % 360.0 # enforce 0 < ra < 360
        dec = f['Data/dec'][...].astype('f8')
        nobj = len(ra)

        files = list()
        files.append(mockfile)
        n_per_file = list()
        n_per_file.append(nobj)

        objid = np.arange(nobj, dtype='i8')
        mockid = make_mockid(objid, n_per_file)

        log.info('Assigning healpix pixels with nside = {}'.format(nside))
        allpix = radec2pix(nside, ra, dec)
        these = np.in1d(allpix, healpixels)
        cut = np.where( these*1 )[0]

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()
        else:
            log.info('Trimmed to {} {}s in healpixels {}'.format(nobj, target_name, healpixels))

        objid = objid[cut]
        mockid = mockid[cut]
        ra = ra[cut]
        dec = dec[cut]

        zz = f['Data/z_obs'][these].astype('f4')
        rmag = f['Data/app_mag'][these].astype('f4')
        absmag = f['Data/abs_mag'][these].astype('f4')
        gr = f['Data/g_r'][these].astype('f4')
        f.close()

        if magcut:
            cut = rmag < magcut
            if np.count_nonzero(cut) == 0:
                log.warning('No objects with r < {}!'.format(magcut))
                return dict()
            else:
                objid = objid[cut]
                mockid = mockid[cut]
                ra = ra[cut]
                dec = dec[cut]
                zz = zz[cut]
                rmag = rmag[cut]
                absmag = absmag[cut]
                gr = gr[cut]
                nobj = len(ra)
                log.info('Trimmed to {} {}s with r < {}.'.format(nobj, target_name, magcut))

        # Assign bricknames.
        brickname = get_brickname_from_radec(ra, dec, bricksize=self.bricksize)

        # Pack into a basic dictionary.
        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'durham_mxxl_hdf5',
               'OBJID': objid, 'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': rmag, 'SDSS_absmag_r01': absmag,
               'SDSS_01gr': gr, 'NORMFILTER': 'sdss2010-r',
               'FILES': files, 'N_PER_FILE': n_per_file}

        # Add MW transmission and the imaging depth.
        mw_transmission(out, dust_dir=self.dust_dir)
        imaging_depth(out)

        return out

class ReadMWS_WD(object):
    """Read a mock catalog of Milky Way Survey white dwarf targets (MWS_WD). 

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    def __init__(self, dust_dir=None, bricksize=0.25):
        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=[], nside=[], target_name='WD'):
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

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data files are not found.
        ValueError
            If mockfile is not defined or if healpixels is not a scalar.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        if len(np.atleast_1d(healpixels)) != 1 and len(np.atleast_1d(nside)) != 1:
            log.warning('Healpixels and nside must be scalar inputs.')
            raise ValueError
            
        # Read the ra,dec coordinates, generate mockid, and then restrict to the
        # desired healpixels.
        log.info('Reading {}'.format(mockfile))
        radec = fitsio.read(mockfile, columns=['RA', 'DEC'], upper=True, ext=1)
        nobj = len(radec)

        files = list()
        n_per_file = list()
        files.append(mockfile)
        n_per_file.append(nobj)

        objid = np.arange(nobj, dtype='i8')
        mockid = make_mockid(objid, n_per_file)

        log.info('Assigning healpix pixels with nside = {}'.format(nside))
        allpix = radec2pix(nside, radec['RA'], radec['DEC'])
        cut = np.where( np.in1d(allpix, healpixels)*1 )[0]

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()
        else:
            log.info('Trimmed to {} {}s in healpixels {}'.format(nobj, target_name, healpixels))

        objid = objid[cut]
        mockid = mockid[cut]
        ra = radec['RA'][cut].astype('f8') % 360.0 # enforce 0 < ra < 360
        dec = radec['DEC'][cut].astype('f8')
        del radec

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
               'OBJID': objid, 'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': mag, 'TEFF': teff, 'LOGG': logg,
               'NORMFILTER': 'sdss2010-g', 'TEMPLATESUBTYPE': templatesubtype,
               'FILES': files, 'N_PER_FILE': n_per_file}

        # Add MW transmission and the imaging depth.
        mw_transmission(out, dust_dir=self.dust_dir)
        imaging_depth(out)

        return out
    
class ReadMWS_NEARBY(object):
    """Read a mock catalog of Milky Way Survey nearby targets (MWS_NEARBY). 

    Parameters
    ----------
    dust_dir : :class:`str`
        Full path to the dust maps.
    bricksize : :class:`int`, optional
        Brick diameter used in the imaging surveys; needed to assign a brickname
        to each object.  Defaults to 0.25 deg.

    """
    def __init__(self, dust_dir=None, bricksize=0.25):
        self.dust_dir = dust_dir
        self.bricksize = bricksize

    def readmock(self, mockfile=None, healpixels=[], nside=[], target_name='MWS_NEARBY'):
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

        Returns
        -------
        :class:`dict`
            Dictionary with various keys (to be documented).

        Raises
        ------
        IOError
            If the mock data files are not found.
        ValueError
            If mockfile is not defined or if healpixels is not a scalar.

        """
        if mockfile is None:
            log.warning('Mockfile input is required.')
            raise ValueError

        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        if len(np.atleast_1d(healpixels)) != 1 and len(np.atleast_1d(nside)) != 1:
            log.warning('Healpixels and nside must be scalar inputs.')
            raise ValueError

        # Read the ra,dec coordinates, generate mockid, and then restrict to the
        # desired healpixels.
        log.info('Reading {}'.format(mockfile))
        radec = fitsio.read(mockfile, columns=['RA', 'DEC'], upper=True, ext=1)
        nobj = len(radec)

        files = list()
        n_per_file = list()
        files.append(mockfile)
        n_per_file.append(nobj)

        objid = np.arange(nobj, dtype='i8')
        mockid = make_mockid(objid, n_per_file)

        log.info('Assigning healpix pixels with nside = {}'.format(nside))
        allpix = radec2pix(nside, radec['RA'], radec['DEC'])
        cut = np.where( np.in1d(allpix, healpixels)*1 )[0]

        nobj = len(cut)
        if nobj == 0:
            log.warning('No {}s in healpixels {}!'.format(target_name, healpixels))
            return dict()
        else:
            log.info('Trimmed to {} {}s in healpixels {}'.format(nobj, target_name, healpixels))

        objid = objid[cut]
        mockid = mockid[cut]
        ra = radec['RA'][cut].astype('f8') % 360.0 # enforce 0 < ra < 360
        dec = radec['DEC'][cut].astype('f8')
        del radec

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
               'OBJID': objid, 'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': mag, 'TEFF': teff, 'LOGG': logg, 'FEH': feh,
               'NORMFILTER': 'sdss2010-g', 'TEMPLATESUBTYPE': templatesubtype,
               'FILES': files, 'N_PER_FILE': n_per_file}

        # Add MW transmission and the imaging depth.
        mw_transmission(out, dust_dir=self.dust_dir)
        imaging_depth(out)

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
    def __init__(self, seed=None, normfilter='decam2014-g'):
        from desisim.templates import SIMQSO

        super(QSOMaker, self).__init__()

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'QSO'

        self.template_maker = SIMQSO(wave=self.wave, normfilter=normfilter)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'GaussianRandomField',
                                             'v0.0.5', 'QSO.fits')

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=[], nside=[], **kwargs):
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
                                   healpixels=healpixels, nside=nside)

        if bool(data):
            data = self._prepare_spectra(data)

        return data

    def _prepare_spectra(self, data):
        """Update the data dictionary with quantities needed to generate spectra.""" 
        data.update({
            'TRUESPECTYPE': 'QSO', 'TEMPLATETYPE': 'QSO', 'TEMPLATESUBTYPE': '',
            })

        return data

    def make_spectra(self, data=None, indx=None):
        """Generate tracer QSO spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.

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
            
        flux, wave, meta = self.template_maker.make_templates(
            nmodel=nobj, redshift=data['Z'][indx], seed=self.seed,
            lyaforest=False, nocolorcuts=True)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=True)

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
          
        qso = isQSO_colors(gflux=targets['FLUX_G'],
                           rflux=targets['FLUX_R'],
                           zflux=targets['FLUX_Z'],
                           w1flux=targets['FLUX_W1'],
                           w2flux=targets['FLUX_W2'])

        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH
        #targets['OBSCONDITIONS'] |= (qso != 0)  * self.obsconditions.mask(
        #    self.desi_mask.QSO.obsconditions)
        #targets['OBSCONDITIONS'] |= (qso != 0)  * self.obsconditions.mask(
        #    self.desi_mask.QSO_SOUTH.obsconditions)

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
    def __init__(self, seed=None, normfilter='decam2014-g'):
        from desisim.templates import SIMQSO

        super(LYAMaker, self).__init__()

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'LYA'

        self.template_maker = SIMQSO(wave=self.wave, normfilter=normfilter)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'lya_forest', 'v2.0.2', 'master.fits')

    def read(self, mockfile=None, mockformat='CoLoRe', dust_dir=None,
             healpixels=[], nside=[], nside_lya=16, **kwargs):
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
                                   nside_lya=nside_lya)

        if bool(data):
            data = self._prepare_spectra(data)

        return data

    def _prepare_spectra(self, data):
        """Update the data dictionary with quantities needed to generate spectra.""" 
        data.update({
            'TRUESPECTYPE': 'QSO', 'TEMPLATETYPE': 'QSO', 'TEMPLATESUBTYPE': 'LYA',
            })

        return data

    def make_spectra(self, data=None, indx=None):
        """Generate QSO spectra with the 3D Lya forest skewers included. 

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.

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
        from desisim.lya_spectra import read_lya_skewers, apply_lya_transmission
        
        if indx is None:
            indx = np.arange(len(data['RA']))
        nobj = len(indx)

        # Read skewers.
        skewer_wave = None
        skewer_trans = None
        skewer_meta = None

        # All the files that contain at least one QSO skewer.
        alllyafile = data['LYAFILES'][indx]
        uniquelyafiles = sorted(set(alllyafile))

        for lyafile in uniquelyafiles:
            these = np.where( alllyafile == lyafile )[0]
            objid_in_data = data['OBJID'][indx][these]
            objid_in_mock = (fitsio.read(lyafile, columns=['MOCKID'], upper=True,
                                         ext=1).astype(float)).astype(int)
            o2i = dict()
            for i, o in enumerate(objid_in_mock):
                o2i[o] = i
            indices_in_mock_healpix = np.zeros(objid_in_data.size).astype(int)
            for i, o in enumerate(objid_in_data):
                if not o in o2i:
                    log.warning("No MOCKID={} in {}. It's a bug, should never happen".format(o, lyafile))
                    raise KeyError
                
                indices_in_mock_healpix[i] = o2i[o]

            tmp_wave, tmp_trans, tmp_meta = read_lya_skewers(lyafile, indices=indices_in_mock_healpix) 

            if skewer_wave is None:
                skewer_wave = tmp_wave
                dw = skewer_wave[1]-skewer_wave[0] # this is just to check same wavelength
                skewer_trans = np.zeros((nobj, skewer_wave.size)) # allocate skewer_array
                skewer_meta = dict()
                for k in tmp_meta.dtype.names:
                    skewer_meta[k] = np.zeros(nobj).astype(tmp_meta[k].dtype)
            else :
                # check wavelength is the same for all skewers
                assert(np.max(np.abs(wave-tmp_wave))<0.001*dw)

            skewer_trans[these] = tmp_trans
            for k in skewer_meta.keys():
                skewer_meta[k][these] = tmp_meta[k]

        # Check we matched things correctly.
        assert(np.max(np.abs(skewer_meta['Z']-data['Z'][indx]))<0.000001)
        assert(np.max(np.abs(skewer_meta['RA']-data['RA'][indx]))<0.000001)
        assert(np.max(np.abs(skewer_meta['DEC']-data['DEC'][indx]))<0.000001)

        # Now generate the QSO spectra simultaneously.
        qso_flux, qso_wave, meta = self.template_maker.make_templates(
            nmodel=nobj, redshift=data['Z'][indx], seed=self.seed,
            lyaforest=False, nocolorcuts=True, noresample=False)
        meta['SUBTYPE'] = 'LYA'

        # Apply the Lya forest transmission.
        flux = apply_lya_transmission(qso_wave, qso_flux, skewer_wave, skewer_trans)

        # Add DLAa (ToDo).

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=True)

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

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']
          
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux)

        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH
        #targets['OBSCONDITIONS'] |= (qso != 0)  * self.obsconditions.mask(
        #    self.desi_mask.QSO.obsconditions)
        #targets['OBSCONDITIONS'] |= (qso != 0)  * self.obsconditions.mask(
        #    self.desi_mask.QSO_SOUTH.obsconditions)

class LRGMaker(SelectTargets):
    """Read LRG mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.

    """
    def __init__(self, seed=None):
        from scipy.spatial import cKDTree as KDTree
        from desiutil.sklearn import GaussianMixtureModel

        super(LRGMaker, self).__init__()

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'LRG'

        self.meta = read_basis_templates(objtype='LRG', onlymeta=True)

        zobj = self.meta['Z'].data
        self.tree = KDTree(np.vstack((zobj)).T)

        gmmfile = resource_filename('desitarget', 'mock/data/lrg_gmm.fits')
        self.GMM = GaussianMixtureModel.load(gmmfile)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'GaussianRandomField',
                                             'v0.0.5', 'LRG.fits')

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=[], nside=[], nside_chunk=128, **kwargs):
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
        nside_chunk : :class:`int`
            Healpixel nside for further subdividing the sample when assigning
            velocity dispersion to targets.

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
                                   healpixels=healpixels, nside=nside)

        if bool(data):
            data = self._prepare_spectra(data, nside_chunk=nside_chunk)
            
        return data

    def _GMMsample(self, nsample=1):
        """Sample from the Gaussian mixture model (GMM) for LRGs."""
        params = self.GMM.sample(nsample, self.rand).astype('f4')

        tags = ('g', 'r', 'z', 'w1', 'w2', 'w3', 'w4')
        tags = tags + ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2')

        samp = np.empty( nsample, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]
            
        return samp

    def _prepare_spectra(self, data, nside_chunk=128):
        """Update the data dictionary with quantities needed to generate spectra.""" 
        from desisim.templates import LRG

        gmm = self._GMMsample(len(data['RA']))
        normmag = gmm['z']
        normfilter = 'decam2014-z'
        self.template_maker = LRG(wave=self.wave, normfilter=normfilter)
        
        seed = self.rand.randint(2**32, size=len(data['RA']))
        vdisp = _sample_vdisp(data, mean=2.3, sigma=0.1, rand=self.rand, nside=nside_chunk)

        data.update({
            'TRUESPECTYPE': 'GALAXY', 'TEMPLATETYPE': 'LRG', 'TEMPLATESUBTYPE': '',
            'SEED': seed, 'VDISP': vdisp, 'MAG': normmag, 
            'SHAPEEXP_R': gmm['exp_r'], 'SHAPEEXP_E1': gmm['exp_e1'], 'SHAPEEXP_E2': gmm['exp_e2'], 
            'SHAPEDEV_R': gmm['dev_r'], 'SHAPEDEV_E1': gmm['dev_e1'], 'SHAPEDEV_E2': gmm['dev_e2'],
            })

        return data

    def _query(self, matrix):
        """Return the nearest template number based on the KD Tree."""
        dist, indx = self.tree.query(matrix)
        return dist, indx
    
    def make_spectra(self, data=None, indx=None):
        """Generate LRG spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.

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

        input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][indx]

        if self.mockformat == 'gaussianfield':
            # This is not quite right, but choose a template with equal probability.
            templateid = self.rand.choice(self.meta['TEMPLATEID'], nobj)
            input_meta['TEMPLATEID'] = templateid

        flux, _, meta = self.template_maker.make_templates(input_meta=input_meta,
                                                           nocolorcuts=True, novdisp=False)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=False)

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

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']

        lrg = isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux)

        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH
        #targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(
        #    self.desi_mask.LRG.obsconditions)
        #targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(
        #    self.desi_mask.LRG_SOUTH.obsconditions)

class ELGMaker(SelectTargets):
    """Read ELG mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.

    """
    def __init__(self, seed=None):
        from scipy.spatial import cKDTree as KDTree
        from desiutil.sklearn import GaussianMixtureModel

        super(ELGMaker, self).__init__()

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'ELG'

        self.meta = read_basis_templates(objtype='ELG', onlymeta=True)

        zobj = self.meta['Z'].data
        gr = self.meta['DECAM_G'].data - self.meta['DECAM_R'].data
        rz = self.meta['DECAM_R'].data - self.meta['DECAM_Z'].data
        self.tree = KDTree(np.vstack((zobj, gr, rz)).T)

        gmmfile = resource_filename('desitarget', 'mock/data/elg_gmm.fits')
        self.GMM = GaussianMixtureModel.load(gmmfile)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'GaussianRandomField',
                                             'v0.0.5', 'ELG.fits')

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=[], nside=[], nside_chunk=128, **kwargs):
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
        nside_chunk : :class:`int`
            Healpixel nside for further subdividing the sample when assigning
            velocity dispersion to targets.

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
                                   healpixels=healpixels, nside=nside)

        if bool(data):
            data = self._prepare_spectra(data, nside_chunk=nside_chunk)

        return data
            
    def _GMMsample(self, nsample=1):
        """Sample from the Gaussian mixture model (GMM) for ELGs."""
        params = self.GMM.sample(nsample, self.rand).astype('f4')

        tags = ('g', 'r', 'z', 'w1', 'w2', 'w3', 'w4')
        tags = tags + ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2')

        samp = np.empty( nsample, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]
            
        return samp

    def _prepare_spectra(self, data, nside_chunk=128):
        """Update the data dictionary with quantities needed to generate spectra.""" 
        from desisim.templates import ELG

        gmm = self._GMMsample(len(data['RA']))
        normmag = gmm['r']
        normfilter = 'decam2014-r'
        self.template_maker = ELG(wave=self.wave, normfilter=normfilter)
        
        seed = self.rand.randint(2**32, size=len(data['RA']))
        vdisp = _sample_vdisp(data, mean=1.9, sigma=0.15, rand=self.rand, nside=nside_chunk)

        data.update({
            'TRUESPECTYPE': 'GALAXY', 'TEMPLATETYPE': 'ELG', 'TEMPLATESUBTYPE': '',
            'SEED': seed, 'VDISP': vdisp, 'MAG': normmag,
            'GR': gmm['g']-gmm['r'], 'RZ': gmm['r']-gmm['z'],
            'RW1': gmm['r']-gmm['w1'], 'W1W2': gmm['w1']-gmm['w2'],
            'SHAPEEXP_R': gmm['exp_r'], 'SHAPEEXP_E1': gmm['exp_e1'], 'SHAPEEXP_E2': gmm['exp_e2'], 
            'SHAPEDEV_R': gmm['dev_r'], 'SHAPEDEV_E1': gmm['dev_e1'], 'SHAPEDEV_E2': gmm['dev_e2'],
            })

        return data

    def _query(self, matrix):
        """Return the nearest template number based on the KD Tree."""
        dist, indx = self.tree.query(matrix)
        return dist, indx
    
    def make_spectra(self, data=None, indx=None):
        """Generate ELG spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.

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

        input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][indx]

        if self.mockformat == 'gaussianfield':
            alldata = np.vstack((data['Z'][indx],
                                 data['GR'][indx],
                                 data['RZ'][indx])).T
            _, templateid = self._query(alldata)
            input_meta['TEMPLATEID'] = templateid

        flux, _, meta = self.template_maker.make_templates(input_meta=input_meta,
                                                           nocolorcuts=True, novdisp=False)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=False)

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

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']
        
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)

        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        #targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(
        #    self.desi_mask.ELG.obsconditions)
        #targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(
        #    self.desi_mask.ELG_SOUTH.obsconditions)

class BGSMaker(SelectTargets):
    """Read BGS mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.

    """
    def __init__(self, seed=None):
        from scipy.spatial import cKDTree as KDTree
        from desiutil.sklearn import GaussianMixtureModel

        super(BGSMaker, self).__init__()

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'BGS'

        self.meta = read_basis_templates(objtype='BGS', onlymeta=True)

        zobj = self.meta['Z'].data
        mabs = self.meta['SDSS_UGRIZ_ABSMAG_Z01'].data
        rmabs = mabs[:, 2]
        gr = mabs[:, 1] - mabs[:, 2]
        self.tree = KDTree(np.vstack((zobj, rmabs, gr)).T)

        gmmfile = resource_filename('desitarget', 'mock/data/bgs_gmm.fits')
        self.GMM = GaussianMixtureModel.load(gmmfile)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'bgs', 'mXXL', 'desi_footprint',
                                             'v0.0.4', 'BGS.hdf5')

    def read(self, mockfile=None, mockformat='durham_mxxl_hdf5', dust_dir=None,
             healpixels=[], nside=[], nside_chunk=128, magcut=None, **kwargs):
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
        nside_chunk : :class:`int`
            Healpixel nside for further subdividing the sample when assigning
            velocity dispersion to targets.

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
        else:
            log.warning('Unrecognized mockformat {}!'.format(mockformat))
            raise ValueError

        if mockfile is None:
            mockfile = self.default_mockfile
            
        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   magcut=magcut)

        if bool(data):
            data = self._prepare_spectra(data, nside_chunk=nside_chunk)

        return data

    def _GMMsample(self, nsample=1):
        """Sample from the Gaussian mixture model (GMM) for BGS."""
        params = self.GMM.sample(nsample, self.rand).astype('f4')

        tags = ('g', 'r', 'z', 'w1', 'w2', 'w3', 'w4')
        tags = tags + ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2')

        samp = np.empty( nsample, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]
            
        return samp

    def _prepare_spectra(self, data, nside_chunk=128):
        """Update the data dictionary with quantities needed to generate spectra.""" 
        from desisim.templates import BGS

        gmm = self._GMMsample(len(data['RA']))
        self.template_maker = BGS(wave=self.wave, normfilter=data['NORMFILTER'])

        seed = self.rand.randint(2**32, size=len(data['RA']))
        vdisp = _sample_vdisp(data, mean=1.9, sigma=0.15, rand=self.rand, nside=nside_chunk)

        data.update({
            'TRUESPECTYPE': 'GALAXY', 'TEMPLATETYPE': 'BGS', 'TEMPLATESUBTYPE': '',
            'SEED': seed, 'VDISP': vdisp, 
            'SHAPEEXP_R': gmm['exp_r'], 'SHAPEEXP_E1': gmm['exp_e1'], 'SHAPEEXP_E2': gmm['exp_e2'], 
            'SHAPEDEV_R': gmm['dev_r'], 'SHAPEDEV_E1': gmm['dev_e1'], 'SHAPEDEV_E2': gmm['dev_e2'],
            })

        return data

    def _query(self, matrix):
        """Return the nearest template number based on the KD Tree."""
        dist, indx = self.tree.query(matrix)
        return dist, indx
    
    def make_spectra(self, data=None, indx=None):
        """Generate BGS spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.

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

        input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][indx]

        if self.mockformat == 'durham_mxxl_hdf5':
            alldata = np.vstack((data['Z'][indx],
                                 data['SDSS_absmag_r01'][indx],
                                 data['SDSS_01gr'][indx])).T
            _, templateid = self._query(alldata)
            input_meta['TEMPLATEID'] = templateid

        flux, _, meta = self.template_maker.make_templates(input_meta=input_meta,
                                                           nocolorcuts=True, novdisp=False)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=False)
        
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

        rflux = targets['FLUX_R']

        # Select BGS_BRIGHT targets.
        bgs_bright = isBGS_bright(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT_SOUTH
        targets['DESI_TARGET'] |= (bgs_bright != 0) * self.desi_mask.BGS_ANY

        #targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(
        #    self.bgs_mask.BGS_BRIGHT.obsconditions)
        #targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(
        #    self.bgs_mask.BGS_BRIGHT_SOUTH.obsconditions)
        #targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(
        #    self.desi_mask.BGS_ANY.obsconditions)
        
        # Select BGS_FAINT targets.
        bgs_faint = isBGS_faint(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT_SOUTH
        targets['DESI_TARGET'] |= (bgs_faint != 0) * self.desi_mask.BGS_ANY
        
        #targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
        #    self.bgs_mask.BGS_FAINT.obsconditions)
        #targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
        #    self.bgs_mask.BGS_FAINT_SOUTH.obsconditions)
        #targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
        #    self.desi_mask.BGS_ANY.obsconditions)

class STARMaker(SelectTargets):
    """Lower-level Class for preparing for stellar spectra to be generated,
    selecting standard stars, and selecting stars as contaminants for
    extragalactic targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.

    """
    def __init__(self, seed=None):
        from scipy.spatial import cKDTree as KDTree
        from speclite import filters

        super(STARMaker, self).__init__()

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'STAR'

        # Pre-compute normalized synthetic photometry for the full set of stellar templates.
        flux, wave, meta = read_basis_templates(objtype='STAR')
        self.meta = meta

        # Assume a normalization filter of SDSS-r.
        self.star_normfilter = 'sdss2010-r'
        sdssr = filters.load_filters(self.star_normfilter)
        decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                         'wise2010-W1', 'wise2010-W2')
        
        maggies = decamwise.get_ab_maggies(flux, wave, mask_invalid=True)
        normmaggies = sdssr.get_ab_maggies(flux, wave, mask_invalid=True)
        
        for filt, flux in zip( maggies.colnames, ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2') ):
            maggies[filt] /= normmaggies[self.star_normfilter]
            maggies.rename_column(filt, flux)
            
        self.star_maggies = maggies

        # Build the KD Tree.
        self.tree = KDTree(np.vstack((self.meta['TEFF'].data,
                                      self.meta['LOGG'].data,
                                      self.meta['FEH'].data)).T)
        
    def _prepare_spectra(self, data):
        """Update the data dictionary with quantities needed to generate spectra.""" 
        from desisim.templates import STAR

        self.template_maker = STAR(wave=self.wave, normfilter=data['NORMFILTER'])
        seed = self.rand.randint(2**32, size=len(data['RA']))

        data.update({
            'TRUESPECTYPE': 'STAR', 'TEMPLATETYPE': 'STAR', 'TEMPLATESUBTYPE': '',
            'SEED': seed, 
            })

        return data

    def _query(self, matrix):
        """Return the nearest template number based on the KD Tree."""
        dist, indx = self.tree.query(matrix)
        return dist, indx

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

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']

        obs_rflux = rflux / targets['MW_TRANSMISSION_R'] # attenuate for Galactic dust

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
        #targets['OBSCONDITIONS'] |= (fstd != 0) * self.obsconditions.mask(
        #    self.desi_mask.STD_FSTAR.obsconditions)

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
        #targets['OBSCONDITIONS'] |= (fstd_bright != 0) * self.obsconditions.mask(
        #    self.desi_mask.STD_BRIGHT.obsconditions)

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

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']

        # Select stellar contaminants for BGS_FAINT targets.
        bgs_faint = isBGS_faint(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT_SOUTH
        targets['DESI_TARGET'] |= (bgs_faint != 0) * self.desi_mask.BGS_ANY
        
        #targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
        #    self.bgs_mask.BGS_FAINT.obsconditions)
        #targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
        #    self.bgs_mask.BGS_FAINT_SOUTH.obsconditions)
        #targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
        #    self.desi_mask.BGS_ANY.obsconditions)
        
        truth['CONTAM_TARGET'] |= (bgs_faint != 0) * self.contam_mask.BGS_IS_STAR
        truth['CONTAM_TARGET'] |= (bgs_faint != 0) * self.contam_mask.BGS_CONTAM

        # Select stellar contaminants for ELG targets.
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        
        #targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(
        #    self.desi_mask.ELG.obsconditions)
        #targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(
        #    self.desi_mask.ELG_SOUTH.obsconditions)
        
        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_IS_STAR
        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_CONTAM

        # Select stellar contaminants for LRG targets.
        lrg = isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux)
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH

        #targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(
        #    self.desi_mask.LRG.obsconditions)
        #targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(
        #    self.desi_mask.LRG_SOUTH.obsconditions)
        
        truth['CONTAM_TARGET'] |= (lrg != 0) * self.contam_mask.LRG_IS_STAR
        truth['CONTAM_TARGET'] |= (lrg != 0) * self.contam_mask.LRG_CONTAM

        # Select stellar contaminants for QSO targets.
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                           w2flux=w2flux)
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH

        #targets['OBSCONDITIONS'] |= (qso != 0) * self.obsconditions.mask(
        #    self.desi_mask.QSO.obsconditions)
        #targets['OBSCONDITIONS'] |= (qso != 0) * self.obsconditions.mask(
        #    self.desi_mask.QSO_SOUTH.obsconditions)
        
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_IS_STAR
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_CONTAM

class MWS_MAINMaker(STARMaker):
    """Read MWS_MAIN mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.

    """
    def __init__(self, seed=None):
        super(MWS_MAINMaker, self).__init__(seed=seed)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'mws', 'galaxia', 'alpha',
                                             'v0.0.5', 'healpix')

    def read(self, mockfile=None, mockformat='galaxia', dust_dir=None,
             healpixels=[], nside=[], nside_galaxia=8, magcut=None,
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

        if bool(data):
            data = self._prepare_spectra(data)

        return data
    
    def make_spectra(self, data=None, indx=None):
        """Generate MWS_MAIN stellar spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.

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

        input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'FEH'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'FEH')):
            input_meta[inkey] = data[datakey][indx]

        if self.mockformat == 'galaxia':
            alldata = np.vstack((data['TEFF'][indx],
                                 data['LOGG'][indx],
                                 data['FEH'][indx])).T
            _, templateid = self._query(alldata)
            input_meta['TEMPLATEID'] = templateid

        # Note! No colorcuts.
        flux, _, meta = self.template_maker.make_templates(input_meta=input_meta)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=True)
                                                           
        return flux, self.wave, meta, targets, truth

    def select_targets(self, targets, truth, boss_std=None):
        """Select MWS_MAIN, MWS_MAIN_VERY_FAINT, standard stars, and (bright)
        contaminants for extragalactic targets.  Input tables are modified in
        place.

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

        def _isMWS_MAIN_VERY_FAINT(rflux):
            """A function like this should be in desitarget.cuts. Select 19<r<20 filler stars."""
            faint = rflux > 10**( (22.5 - 20.0) / 2.5 )
            faint &= rflux <= 10**( (22.5 - 19.0) / 2.5 )
            return faint
        
        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']

        # Select MWS_MAIN targets.
        mws_main = _isMWS_MAIN(rflux=rflux)
        #mws_main = np.ones(len(targets)) # select everything!
        
        targets['MWS_TARGET'] |= (mws_main != 0) * self.mws_mask.mask('MWS_MAIN')
        targets['DESI_TARGET'] |= (mws_main != 0) * self.desi_mask.MWS_ANY
        #targets['OBSCONDITIONS'] |= (mws_main != 0)  * self.obsconditions.mask(
        #    self.mws_mask.MWS_MAIN.obsconditions)
        #targets['OBSCONDITIONS'] |= (mws_main != 0)  * self.obsconditions.mask(
        #    self.desi_mask.MWS_ANY.obsconditions)
        
        mws_main_very_faint = _isMWS_MAIN_VERY_FAINT(rflux=rflux)
        targets['MWS_TARGET'] |= (mws_main_very_faint != 0) * self.mws_mask.mask('MWS_MAIN_VERY_FAINT')
        targets['DESI_TARGET'] |= (mws_main_very_faint != 0) * self.desi_mask.MWS_ANY
        #targets['OBSCONDITIONS'] |= (mws_main_very_faint != 0)  * self.obsconditions.mask(
        #    self.mws_mask.MWS_MAIN_VERY_FAINT.obsconditions)
        #targets['OBSCONDITIONS'] |= (mws_main_very_faint != 0)  * self.obsconditions.mask(
        #    self.desi_mask.MWS_ANY.obsconditions)

        # Select standard stars.
        self.select_standards(targets, truth, boss_std=boss_std)
        
        # Select bright stellar contaminants for the extragalactic targets.
        self.select_contaminants(targets, truth)

class FAINTSTARMaker(STARMaker):
    """Read FAINTSTAR mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.

    """
    def __init__(self, seed=None):
        super(FAINTSTARMaker, self).__init__(seed=seed)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'mws', 'galaxia', 'alpha',
                                             '0.0.5_superfaint', 'healpix')

    def read(self, mockfile=None, mockformat='galaxia', dust_dir=None,
             healpixels=[], nside=[], nside_galaxia=8, magcut=None,
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

        if bool(data):
            data = self._prepare_spectra(data)

        return data
    
    def make_spectra(self, data=None, indx=None, boss_std=None):
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

        if self.mockformat == 'galaxia':
            alldata = np.vstack((data['TEFF'][indx],
                                 data['LOGG'][indx],
                                 data['FEH'][indx])).T
            _, templateid = self._query(alldata)

        # Initialize dummy the targets and truth tables.
        _targets = empty_targets_table(nobj)
        _truth = empty_truth_table(nobj)
        
        # Pack the noiseless stellar photometry in the truth table, generate
        # noisy photometry, and then select targets.
        if data['NORMFILTER'] != self.star_normfilter:
            log.warning('Mismatching normalization filters!')
            raise ValueError
        
        normmag = 1e9 * 10**(-0.4 * data['MAG'][indx]) # nanomaggies
        
        for key in ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'):
            _truth[key][:] = self.star_maggies[key][templateid] * normmag

        self.scatter_photometry(data, _truth, _targets, indx=indx, psf=True, qaplot=False)

        self.select_targets(_targets, _truth, boss_std=boss_std)

        keep = np.where(_targets['DESI_TARGET'] != 0)[0]
        log.debug('Pre-selected {} FAINTSTAR targets.'.format(len(keep)))

        if len(keep) > 0:
            input_meta = empty_metatable(nmodel=len(keep), objtype=self.objtype)
            for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'FEH'),
                                      ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'FEH')):
                input_meta[inkey] = data[datakey][indx][keep]
                
            input_meta['TEMPLATEID'] = templateid[keep]

            # Note! No colorcuts.
            flux, _, meta = self.template_maker.make_templates(input_meta=input_meta)

            # Force consistency in the noisy photometry so we select the same targets. 
            targets, truth = self.populate_targets_truth(data, meta, indx=indx[keep], psf=True)
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

    """
    def __init__(self, seed=None):
        super(MWS_NEARBYMaker, self).__init__(seed=seed)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'mws', '100pc', 'v0.0.3',
                                             'mock_100pc.fits')


    def read(self, mockfile=None, mockformat='mws_100pc', dust_dir=None,
             healpixels=[], nside=[], **kwargs):
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
                                   healpixels=healpixels, nside=nside)

        if bool(data):
            data = self._prepare_spectra(data)

        return data
    
    def make_spectra(self, data=None, indx=None):
        """Generate MWS_NEARBY stellar spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.

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

        input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'FEH'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'FEH')):
            input_meta[inkey] = data[datakey][indx]

        if self.mockformat == 'mws_100pc':
            alldata = np.vstack((data['TEFF'][indx],
                                 data['LOGG'][indx],
                                 data['FEH'][indx])).T
            _, templateid = self._query(alldata)
            input_meta['TEMPLATEID'] = templateid

        # Note! No colorcuts.
        flux, _, meta = self.template_maker.make_templates(input_meta=input_meta)

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=True)
                                                           
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
        #targets['OBSCONDITIONS'] |= (mws_nearby != 0)  * self.obsconditions.mask(
        #    self.mws_mask.MWS_NEARBY.obsconditions)
        #targets['OBSCONDITIONS'] |= (mws_nearby != 0)  * self.obsconditions.mask(
        #    self.desi_mask.MWS_ANY.obsconditions)

class WDMaker(SelectTargets):
    """Read WD mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.

    """
    def __init__(self, seed=None):
        from scipy.spatial import cKDTree as KDTree
        
        super(WDMaker, self).__init__()

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'WD'
        
        self.meta_da = read_basis_templates(objtype='WD', subtype='DA', onlymeta=True)
        self.meta_db = read_basis_templates(objtype='WD', subtype='DB', onlymeta=True)

        self.tree_da = KDTree(np.vstack((self.meta_da['TEFF'].data,
                                         self.meta_da['LOGG'].data)).T)
        self.tree_db = KDTree(np.vstack((self.meta_db['TEFF'].data,
                                         self.meta_db['LOGG'].data)).T)

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'mws', 'wd', 'v0.0.2',
                                             'mock_wd.fits')

    def read(self, mockfile=None, mockformat='mws_wd', dust_dir=None,
             healpixels=[], nside=[], **kwargs):
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
                                   healpixels=healpixels, nside=nside)

        if bool(data):
            data = self._prepare_spectra(data)

        return data

    def _prepare_spectra(self, data):
        """Update the data dictionary with quantities needed to generate spectra.""" 
        from desisim.templates import WD
        
        self.da_template_maker = WD(wave=self.wave, subtype='DA', normfilter=data['NORMFILTER'])
        self.db_template_maker = WD(wave=self.wave, subtype='DB', normfilter=data['NORMFILTER'])

        seed = self.rand.randint(2**32, size=len(data['RA']))

        data.update({
            'TRUESPECTYPE': 'WD', 'TEMPLATETYPE': 'WD', 'SEED': seed, # no subtype here
            })

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
    
    def make_spectra(self, data=None, indx=None):
        """Generate WD spectra, dealing with DA vs DB white dwarfs separately.
        
        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.

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

        input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'SUBTYPE'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'TEMPLATESUBTYPE')):
            input_meta[inkey] = data[datakey][indx]
            
        if self.mockformat == 'mws_wd':
            meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')
            
            for subtype in ('DA', 'DB'):
                these = np.where(input_meta['SUBTYPE'] == subtype)[0]
                if len(these) > 0:
                    alldata = np.vstack((data['TEFF'][indx][these],
                                         data['LOGG'][indx][these])).T
                    _, templateid = self._query(alldata, subtype=subtype)
                    
                    input_meta['TEMPLATEID'][these] = templateid
                    
                    template_maker = getattr(self, '{}_template_maker'.format(subtype.lower()))
                    flux1, _, meta1 = template_maker.make_templates(input_meta=input_meta[these])
                    
                    meta[these] = meta1
                    flux[these, :] = flux1
            

        targets, truth = self.populate_targets_truth(data, meta, indx=indx, psf=True)

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

        targets['MWS_TARGET'] |= (mws_wd != 0) * self.mws_mask.mask('MWS_WD')
        targets['DESI_TARGET'] |= (mws_wd != 0) * self.desi_mask.MWS_ANY
        #targets['OBSCONDITIONS'] |= (mws_wd != 0)  * self.obsconditions.mask(
        #    self.mws_mask.MWS_WD.obsconditions)
        #targets['OBSCONDITIONS'] |= (mws_wd != 0)  * self.obsconditions.mask(
        #    self.desi_mask.MWS_ANY.obsconditions)

        # Select STD_WD; cut just on g-band magnitude (not TEMPLATESUBTYPE!)
        std_wd = (truth['MAG'] <= 19.0) * 1 # SDSS g-band!
        targets['DESI_TARGET'] |= (std_wd !=0) * self.desi_mask.mask('STD_WD')
        #targets['OBSCONDITIONS'] |= (std_wd != 0)  * self.obsconditions.mask(
        #    self.desi_mask.STD_WD.obsconditions)

class SKYMaker(SelectTargets):
    """Read SKY mocks, generate spectra, and select targets.

    Parameters
    ----------
    seed : :class:`int`, optional
        Seed for reproducibility and random number generation.

    """
    def __init__(self, seed=None):
        super(SKYMaker, self).__init__()

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'SKY'

        # Default mock catalog.
        self.default_mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks',
                                             'GaussianRandomField',
                                             'v0.0.1', '2048', 'random.fits')

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=[], nside=[], **kwargs):
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
        nside_chunk : :class:`int`
            Healpixel nside for further subdividing the sample when assigning
            velocity dispersion to targets.

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
                                   healpixels=healpixels, nside=nside)

        if bool(data):
            data = self._prepare_spectra(data)

        return data

    def _prepare_spectra(self, data):
        """Update the data dictionary with quantities needed to generate spectra.""" 
        seed = self.rand.randint(2**32, size=len(data['RA']))
        data.update({
            'TRUESPECTYPE': 'SKY', 'TEMPLATETYPE': 'SKY', 'TEMPLATESUBTYPE': '',
            'SEED': seed,
            })

        return data

    def make_spectra(self, data=None, indx=None):
        """Generate SKY spectra.

        Parameters
        ----------
        data : :class:`dict`
            Dictionary of source properties.
        indx : :class:`numpy.ndarray`, optional
            Generate spectra for a subset of the objects in the data dictionary,
            as specified using their zero-indexed indices.

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
            
        meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'REDSHIFT'), ('SEED', 'Z')):
            meta[inkey] = data[datakey][indx]
        flux = np.zeros((nobj, len(self.wave)), dtype='i1')

        targets, truth = self.populate_targets_truth(data, meta, indx=indx)

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
        #targets['OBSCONDITIONS'] |= self.obsconditions.mask(
        #    self.desi_mask.SKY.obsconditions)
