# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.iospectra
=========================

Read mock catalogs and assign spectra.

"""
from __future__ import absolute_import, division, print_function

import os
import numpy as np
from pkg_resources import resource_filename

import fitsio

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
    rownum : int
        Row in input file.
    filenum : int
        File number in input file set.

    Returns
    -------
    encoded value(s) : int64 numpy.ndarray
        52 packed bits encoding row and file number.

    """
    assert(np.shape(rownum) == np.shape(filenum))
    assert(np.all(rownum  >= 0))
    assert(np.all(rownum  <= int(ENCODE_ROW_MAX)))
    assert(np.all(filenum >= 0))
    assert(np.all(filenum <= int(ENCODE_FILE_MAX)))

    # This should be a 64 bit integer.
    encoded_value = (np.asarray(filenum,dtype=np.uint64) << ENCODE_ROW_END) + np.asarray(rownum, dtype=np.uint64)

    # Note return signed
    return np.asarray(encoded_value, dtype=np.int64)

def decode_rownum_filenum(encoded_values):
    """Inverts encode_rownum_filenum to obtain row number and file number.

    Parameters
    ----------
    encoded_values(s) : int64 ndarray

    Returns
    -------
    filenum : str
        File number.
    rownum : int
        Row number.

    """
    filenum = (np.asarray(encoded_values,dtype=np.uint64) & ENCODE_FILE_MASK) >> ENCODE_ROW_END
    rownum  = (np.asarray(encoded_values,dtype=np.uint64) & ENCODE_ROW_MASK)
    return rownum, filenum

def make_mockid(objid, n_per_file):
    """
    Computes mockid from row and file IDs.

    Parameters
    ----------
    objid : int array
        Row identification number.
    n_per_file : int list
        Number of items per file that went into objid.

    Returns
    -------
    mockid : int array
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
    
    Args:
        source_data : dict
            Input dictionary (read by read_mock_catalog) with coordinates. 
            
    Returns:
        source_data : dict
            Modified input dictionary with MW transmission included. 

    """
    from desitarget.mock import sfdmap

    if dust_dir is None:
        log.warning('DUST_DIR is a required input!')
        raise ValueError

    extcoeff = dict(G = 3.214, R = 2.165, Z = 1.221, W1 = 0.184, W2 = 0.113)
    source_data['EBV'] = sfdmap.ebv(source_data['RA'],
                                    source_data['DEC'],
                                    mapdir=dust_dir)

    for band in ('G', 'R', 'Z', 'W1', 'W2'):
        source_data['MW_TRANSMISSION_{}'.format(band)] = 10**(-0.4 * extcoeff[band] * source_data['EBV'])

def _sample_vdisp(data, mean=1.9, sigma=0.15, fracvdisp=(0.1, 1), rand=None, nside=128):
    """Choose a subset of velocity dispersions."""

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
    """Generate a default wavelength vector for the output spectra.

    """
    from desimodel.io import load_throughput
    
    if wavemin is None:
        wavemin = load_throughput('b').wavemin - 10.0
    if wavemax is None:
        wavemax = load_throughput('z').wavemax + 10.0
            
    return np.arange(round(wavemin, 1), wavemax, dw)

class ReadGaussianField(object):

    def __init__(self, bricksize=0.25, dust_dir=None):

        self.bricksize = bricksize
        self.dust_dir = dust_dir

    def readmock(self, mockfile, target_name='', nside=8, healpixels=None, magcut=None):
        """Read the mock catalog.

        """
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        self.nside = nside
    
        # Read the whole DESI footprint.
        if healpixels is None:
            from desimodel.footprint import tiles2pix
            healpixels = tiles2pix(self.nside)

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

        log.info('Assigning healpix pixels with nside = {}'.format(self.nside))
        allpix = radec2pix(self.nside, radec['RA'], radec['DEC'])
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

        # Add MW transmission
        mw_transmission(out, dust_dir=self.dust_dir)

        return out

class ReadMXXL(object):

    def __init__(self, bricksize=0.25, dust_dir=None):

        self.bricksize = bricksize
        self.dust_dir = dust_dir

    def readmock(self, mockfile, target_name='BGS', nside=8, healpixels=None, magcut=None):
        """Read the mock catalog.

        """
        import h5py
        
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        self.nside = nside

        # Read the whole DESI footprint.
        if healpixels is None:
            from desimodel.footprint import tiles2pix
            healpixels = tiles2pix(self.nside)
            
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

        out = {'TARGET_NAME': target_name, 'MOCKFORMAT': 'gaussianfield',
               'OBJID': objid, 'MOCKID': mockid, 'BRICKNAME': brickname,
               'RA': ra, 'DEC': dec, 'Z': zz, 'MAG': rmag, 'SDSS_absmag_r01': absmag,
               'SDSS_01gr': gr, 'NORMFILTER': 'sdss2010-r',
               'FILES': files, 'N_PER_FILE': n_per_file}

        # Add MW transmission
        mw_transmission(out, dust_dir=self.dust_dir)

        return out

class ReadMWSWD(object):

    def __init__(self, bricksize=0.25, dust_dir=None):

        self.bricksize = bricksize
        self.dust_dir = dust_dir

    def readmock(self, mockfile, target_name='WD', nside=8, healpixels=None, magcut=None):
        """Read the mock catalog.

        """
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        self.nside = nside
    
        # Read the whole DESI footprint.
        if healpixels is None:
            from desimodel.footprint import tiles2pix
            healpixels = tiles2pix(self.nside)
            
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

        log.info('Assigning healpix pixels with nside = {}'.format(self.nside))
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

        # Add MW transmission
        mw_transmission(out, dust_dir=self.dust_dir)

        return out
    
class ReadMWSNearby(object):

    def __init__(self, bricksize=0.25, dust_dir=None):

        self.bricksize = bricksize
        self.dust_dir = dust_dir

    def readmock(self, mockfile, target_name='MWS_NEARBY', nside=8, healpixels=None, magcut=None):
        """Read the mock catalog.

        """
        if not os.path.isfile(mockfile):
            log.warning('Mock file {} not found!'.format(mockfile))
            raise IOError

        self.nside = nside
    
        # Read the whole DESI footprint.
        if healpixels is None:
            from desimodel.footprint import tiles2pix
            healpixels = tiles2pix(self.nside)
            
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

        log.info('Assigning healpix pixels with nside = {}'.format(self.nside))
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

        # Add MW transmission
        mw_transmission(out, dust_dir=self.dust_dir)

        return out

class SelectTargets(object):
    """Select various types of targets.

    """
    def __init__(self, **kwargs):
        
        #super(SelectTargets, self).__init__(**kwargs)

        from desitarget import (desi_mask, bgs_mask, mws_mask,
                                contam_mask, obsconditions)
        
        self.desi_mask = desi_mask
        self.bgs_mask = bgs_mask
        self.mws_mask = mws_mask
        self.contam_mask = contam_mask
        self.obsconditions = obsconditions

class QSOMaker(SelectTargets):

    def __init__(self, seed=None, verbose=False, normfilter='decam2014-g', **kwargs):

        from desisim.templates import SIMQSO

        super(QSOMaker, self).__init__(**kwargs)

        self.seed = seed
        self.verbose = verbose
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'QSO'

        self.template_maker = SIMQSO(wave=self.wave, normfilter=normfilter)

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=None, nside=8, magcut=None):
        """Read the mock file."""

        if mockformat == 'gaussianfield':
            MockReader = ReadGaussianField(dust_dir=dust_dir)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   magcut=magcut)

        if bool(data):
            data = self._prepare_spectra(data, nside_chunk=nside_chunk)

        return data

    def _prepare_spectra(self, data):

        data.update({
            'TRUESPECTYPE': 'QSO', 'TEMPLATETYPE': self.objtype, 'TEMPLATESUBTYPE': '',
            })

        return data

    def make_spectra(self, data=None, index=None):
        """Generate tracer QSO spectra."""
        
        if index is None:
            index = np.arange(len(data['RA']))
            
        flux, wave, meta = self.template_maker.make_templates(
            nmodel=nobj, redshift=data['Z'][index], seed=self.seed,
            lyaforest=False, nocolorcuts=True, verbose=self.verbose)

        return flux, self.wave, meta

    def select_targets(self, targets, truth):
        """Select QSO targets."""
        from desitarget.cuts import isQSO_colors

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']
          
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux, optical=True)

        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH
        targets['OBSCONDITIONS'] |= (qso != 0)  * self.obsconditions.mask(
            self.desi_mask.QSO.obsconditions)
        targets['OBSCONDITIONS'] |= (qso != 0)  * self.obsconditions.mask(
            self.desi_mask.QSO_SOUTH.obsconditions)

class LRGMaker(SelectTargets):
    """Read LRG mocks and generate spectra."""

    def __init__(self, seed=None, verbose=False, **kwargs):

        from scipy.spatial import cKDTree as KDTree
        from desiutil.sklearn import GaussianMixtureModel

        super(LRGMaker, self).__init__(**kwargs)

        self.seed = seed
        self.verbose = verbose
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'LRG'

        self.meta = read_basis_templates(objtype='LRG', onlymeta=True)

        zobj = self.meta['Z'].data
        self.tree = KDTree(np.vstack((zobj)).T)

        gmmfile = resource_filename('desitarget', 'mock/data/lrg_gmm.fits')
        self.GMM = GaussianMixtureModel.load(gmmfile)

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=None, nside=8, nside_chunk=128, magcut=None):
        """Read the mock file."""

        if mockformat == 'gaussianfield':
            MockReader = ReadGaussianField(dust_dir=dust_dir)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   magcut=magcut)

        if bool(data):
            data = self._prepare_spectra(data, nside_chunk=nside_chunk)
            
        return data

    def _prepare_spectra(self, data, nside_chunk=128):
        
        from desisim.templates import LRG

        gmm = self.GMMsample(len(data['RA']))
        normmag = gmm['z']
        normfilter = 'decam2014-z'
        self.template_maker = LRG(wave=self.wave, normfilter=normfilter)
        
        seed = self.rand.randint(2**32, size=len(data['RA']))
        vdisp = _sample_vdisp(data, mean=2.3, sigma=0.1, rand=self.rand, nside=nside_chunk)

        data.update({
            'TRUESPECTYPE': 'GALAXY', 'TEMPLATETYPE': self.objtype, 'TEMPLATESUBTYPE': '',
            'SEED': seed, 'VDISP': vdisp, 'MAG': normmag, 
            'SHAPEEXP_R': gmm['exp_r'], 'SHAPEEXP_E1': gmm['exp_e1'], 'SHAPEEXP_E2': gmm['exp_e2'], 
            'SHAPEDEV_R': gmm['dev_r'], 'SHAPEDEV_E1': gmm['dev_e1'], 'SHAPEDEV_E2': gmm['dev_e2'],
            })

        return data

    def GMMsample(self, nsample=1):
        """Sample from the GMM."""
        
        params = self.GMM.sample(nsample, self.rand).astype('f4')

        tags = ('g', 'r', 'z', 'w1', 'w2', 'w3', 'w4')
        tags = tags + ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2')

        samp = np.empty( nsample, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]
            
        return samp

    def query(self, matrix):
        """Return the nearest template number based on the KD Tree."""

        dist, indx = self.tree.query(matrix)
        return dist, indx
    
    def make_spectra(self, data=None, index=None):
        """Generate LRG spectra."""
        from desisim.io import empty_metatable
        
        if index is None:
            index = np.arange(len(data['RA']))
        nobj = len(index)

        input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if data['MOCKFORMAT'].lower() == 'gaussianfield':
            # This is not quite right, but choose a template with equal probability.
            templateid = self.rand.choice(self.meta['TEMPLATEID'], nobj)
            input_meta['TEMPLATEID'] = templateid
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        flux, _, meta = self.template_maker.make_templates(input_meta=input_meta,
                                                           nocolorcuts=True, novdisp=False,
                                                           verbose=self.verbose)
        
        return flux, self.wave, meta

    def select_targets(self, targets, truth):
        """Select LRG targets."""
        from desitarget.cuts import isLRG_colors

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']

        lrg = isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux)

        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH
        targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(
            self.desi_mask.LRG.obsconditions)
        targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(
            self.desi_mask.LRG_SOUTH.obsconditions)

class ELGMaker(SelectTargets):
    """Read ELG mocks and generate spectra."""

    def __init__(self, seed=None, verbose=False, **kwargs):

        from scipy.spatial import cKDTree as KDTree
        from desiutil.sklearn import GaussianMixtureModel

        super(ELGMaker, self).__init__(**kwargs)

        self.seed = seed
        self.verbose = verbose
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

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=None, nside=8, nside_chunk=128, magcut=None):
        """Read the mock file."""

        if mockformat == 'gaussianfield':
            MockReader = ReadGaussianField(dust_dir=dust_dir)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   magcut=magcut)

        if bool(data):
            data = self._prepare_spectra(data, nside_chunk=nside_chunk)

        return data
            
    def _prepare_spectra(self, data, nside_chunk=128):
        
        from desisim.templates import ELG

        gmm = self.GMMsample(len(data['RA']))
        normmag = gmm['r']
        normfilter = 'decam2014-r'
        self.template_maker = ELG(wave=self.wave, normfilter=normfilter)
        
        seed = self.rand.randint(2**32, size=len(data['RA']))
        vdisp = _sample_vdisp(data, mean=1.9, sigma=0.15, rand=self.rand, nside=nside_chunk)

        data.update({
            'TRUESPECTYPE': 'GALAXY', 'TEMPLATETYPE': self.objtype, 'TEMPLATESUBTYPE': '',
            'SEED': seed, 'VDISP': vdisp, 'MAG': normmag,
            'GR': gmm['g']-gmm['r'], 'RZ': gmm['r']-gmm['z'],
            'RW1': gmm['r']-gmm['w1'], 'W1W2': gmm['w1']-gmm['w2'],
            'SHAPEEXP_R': gmm['exp_r'], 'SHAPEEXP_E1': gmm['exp_e1'], 'SHAPEEXP_E2': gmm['exp_e2'], 
            'SHAPEDEV_R': gmm['dev_r'], 'SHAPEDEV_E1': gmm['dev_e1'], 'SHAPEDEV_E2': gmm['dev_e2'],
            })

        return data

    def GMMsample(self, nsample=1):
        """Sample from the GMM."""
        
        params = self.GMM.sample(nsample, self.rand).astype('f4')

        tags = ('g', 'r', 'z', 'w1', 'w2', 'w3', 'w4')
        tags = tags + ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2')

        samp = np.empty( nsample, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]
            
        return samp

    def query(self, matrix):
        """Return the nearest template number based on the KD Tree."""

        dist, indx = self.tree.query(matrix)
        return dist, indx
    
    def make_spectra(self, data=None, index=None):
        """Generate ELG spectra."""
        from desisim.io import empty_metatable
        
        if index is None:
            index = np.arange(len(data['RA']))
        nobj = len(index)

        input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if data['MOCKFORMAT'].lower() == 'gaussianfield':
            alldata = np.vstack((data['Z'][index],
                                 data['GR'][index],
                                 data['RZ'][index])).T
            _, templateid = self.query(alldata)
            input_meta['TEMPLATEID'] = templateid
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        flux, _, meta = self.template_maker.make_templates(input_meta=input_meta,
                                                           nocolorcuts=True, novdisp=False,
                                                           verbose=self.verbose)
        
        return flux, self.wave, meta

    def select_targets(self, targets, truth):
        """Select ELG targets."""
        from desitarget.cuts import isELG

        gflux, rflux, zflux, w1flux, w2flux = targets['FLUX_G'], targets['FLUX_R'], \
          targets['FLUX_Z'], targets['FLUX_W1'], targets['FLUX_W2']
        
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)

        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(
            self.desi_mask.ELG.obsconditions)
        targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(
            self.desi_mask.ELG_SOUTH.obsconditions)

class BGSMaker(SelectTargets):
    """Read BGS mocks and generate spectra."""

    def __init__(self, seed=None, verbose=False, **kwargs):

        from scipy.spatial import cKDTree as KDTree
        from desiutil.sklearn import GaussianMixtureModel

        super(BGSMaker, self).__init__(**kwargs)

        self.seed = seed
        self.verbose = verbose
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

    def read(self, mockfile=None, mockformat='durham_mxxl_hdf5', dust_dir=None,
             healpixels=None, nside=8, nside_chunk=128, magcut=None):
        """Read the mock file."""
        
        if mockformat == 'durham_mxxl_hdf5':
            MockReader = ReadMXXL(dust_dir=dust_dir)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   magcut=magcut)

        if bool(data):
            data = self._prepare_spectra(data, nside_chunk=nside_chunk)

        return data

    def _prepare_spectra(self, data, nside_chunk=128):
        
        from desisim.templates import BGS

        gmm = self.GMMsample(len(data['RA']))
        self.template_maker = BGS(wave=self.wave, normfilter=data['NORMFILTER'])

        seed = self.rand.randint(2**32, size=len(data['RA']))
        vdisp = _sample_vdisp(data, mean=1.9, sigma=0.15, rand=self.rand, nside=nside_chunk)

        data.update({
            'TRUESPECTYPE': 'GALAXY', 'TEMPLATETYPE': self.objtype, 'TEMPLATESUBTYPE': '',
            'SEED': seed, 'VDISP': vdisp, 
            'SHAPEEXP_R': gmm['exp_r'], 'SHAPEEXP_E1': gmm['exp_e1'], 'SHAPEEXP_E2': gmm['exp_e2'], 
            'SHAPEDEV_R': gmm['dev_r'], 'SHAPEDEV_E1': gmm['dev_e1'], 'SHAPEDEV_E2': gmm['dev_e2'],
            })

        return data

    def GMMsample(self, nsample=1):
        """Sample from the GMM."""
        
        params = self.GMM.sample(nsample, self.rand).astype('f4')

        tags = ('g', 'r', 'z', 'w1', 'w2', 'w3', 'w4')
        tags = tags + ('exp_r', 'exp_e1', 'exp_e2', 'dev_r', 'dev_e1', 'dev_e2')

        samp = np.empty( nsample, dtype=np.dtype( [(tt, 'f4') for tt in tags] ) )
        for ii, tt in enumerate(tags):
            samp[tt] = params[:, ii]
            
        return samp

    def query(self, matrix):
        """Return the nearest template number based on the KD Tree."""

        dist, indx = self.tree.query(matrix)
        return dist, indx
    
    def make_spectra(self, data=None, index=None):
        """Generate BGS spectra."""
        from desisim.io import empty_metatable
        
        if index is None:
            index = np.arange(len(data['RA']))
        nobj = len(index)

        input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'VDISP'),
                                  ('SEED', 'MAG', 'Z', 'VDISP')):
            input_meta[inkey] = data[datakey][index]

        if data['MOCKFORMAT'].lower() == 'durham_mxxl_hdf5':
            alldata = np.vstack((data['Z'][index],
                                 data['SDSS_absmag_r01'][index],
                                 data['SDSS_01gr'][index])).T
            _, templateid = self.query(alldata)
            input_meta['TEMPLATEID'] = templateid
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        flux, _, meta = self.template_maker.make_templates(input_meta=input_meta,
                                                           nocolorcuts=True, novdisp=False,
                                                           verbose=self.verbose)
        
        return flux, self.wave, meta

    def select_targets(self, targets, truth):
        """Select BGS targets."""
        from desitarget.cuts import isBGS_bright, isBGS_faint

        rflux = targets['FLUX_R']

        # Select BGS_BRIGHT targets.
        bgs_bright = isBGS_bright(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT_SOUTH
        targets['DESI_TARGET'] |= (bgs_bright != 0) * self.desi_mask.BGS_ANY

        targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(
            self.bgs_mask.BGS_BRIGHT.obsconditions)
        targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(
            self.bgs_mask.BGS_BRIGHT_SOUTH.obsconditions)
        targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(
            self.desi_mask.BGS_ANY.obsconditions)
        
        # Select BGS_FAINT targets.
        bgs_faint = isBGS_faint(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT_SOUTH
        targets['DESI_TARGET'] |= (bgs_faint != 0) * self.desi_mask.BGS_ANY
        
        targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
            self.bgs_mask.BGS_FAINT.obsconditions)
        targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
            self.bgs_mask.BGS_FAINT_SOUTH.obsconditions)
        targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(
            self.desi_mask.BGS_ANY.obsconditions)

class STARMaker(SelectTargets):
    """Lower-level Class for generating stellar spectra."""

    def __init__(self, seed=None, verbose=False, **kwargs):

        from scipy.spatial import cKDTree as KDTree
        from speclite import filters

        super(STARMaker, self).__init__(**kwargs)

        self.seed = seed
        self.verbose = verbose
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'STAR'

        # Pre-compute normalized synthetic photometry for the full set of stellar templates.
        flux, wave, meta = read_basis_templates(objtype='STAR')#, verbose=False)
        self.meta = meta

        decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                         'wise2010-W1', 'wise2010-W2')
        
        maggies = decamwise.get_ab_maggies(flux, wave, mask_invalid=True)
        for filt, flux in zip( maggies.colnames, ('flux_g', 'flux_r', 'flux_z', 'flux_W1', 'flux_W2') ):
            maggies.rename_column(filt, flux)

        # Build the KD Tree.
        self.tree = KDTree(np.vstack((self.meta['TEFF'].data,
                                      self.meta['LOGG'].data,
                                      self.meta['FEH'].data)).T)
        
    def _prepare_spectra(self, data):
        
        from desisim.templates import STAR

        self.template_maker = STAR(wave=self.wave, normfilter=data['NORMFILTER'])
        seed = self.rand.randint(2**32, size=len(data['RA']))

        data.update({
            'TRUESPECTYPE': 'STAR', 'TEMPLATETYPE': self.objtype, 'TEMPLATESUBTYPE': '',
            'SEED': seed, 
            })

        return data

    def query(self, matrix):
        """Return the nearest template number based on the KD Tree."""

        dist, indx = self.tree.query(matrix)
        return dist, indx

class MWS_NEARBYMaker(STARMaker):
    """Read various types of stellar mocks and generate spectra."""

    def __init__(self, **kwargs):

        super(MWS_NEARBYMaker, self).__init__(**kwargs)

    def read(self, mockfile=None, mockformat='mws_100pc', dust_dir=None,
             healpixels=None, nside=8, nside_chunk=128, magcut=None):
        """Read the mock file."""
        
        if mockformat == 'mws_100pc':
            MockReader = ReadMWSNearby(dust_dir=dust_dir)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   magcut=magcut)

        if bool(data):
            data = self._prepare_spectra(data)

        return data
    
    def make_spectra(self, data=None, index=None):
        """Generate MWS_100PC stellar spectra."""
        from desisim.io import empty_metatable
        
        if index is None:
            index = np.arange(len(data['RA']))
        nobj = len(index)

        input_meta = empty_metatable(nmodel=len(index), objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'FEH'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'FEH')):
            input_meta[inkey] = data[datakey][index]

        if data['MOCKFORMAT'].lower() == 'mws_100pc':
            alldata = np.vstack((data['TEFF'][index],
                                 data['LOGG'][index],
                                 data['FEH'][index])).T
            _, templateid = self.query(alldata)
            input_meta['TEMPLATEID'] = templateid
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        flux, _, meta = self.template_maker.make_templates(input_meta=input_meta,
                                                           verbose=self.verbose) # Note! No colorcuts.
                                                          
        return flux, self.wave, meta

    def select_targets(self, targets, truth):
        """Select MWS_NEARBY targets.  The selection eventually will be done with Gaia,
        so for now just do a "perfect" selection.

        """
        mws_nearby = np.ones(len(targets)) # select everything!
        #mws_nearby = (truth['MAG'] <= 20.0) * 1 # SDSS g-band!

        targets['MWS_TARGET'] |= (mws_nearby != 0) * self.mws_mask.mask('MWS_NEARBY')
        targets['DESI_TARGET'] |= (mws_nearby != 0) * self.desi_mask.MWS_ANY
        targets['OBSCONDITIONS'] |= (mws_nearby != 0)  * self.obsconditions.mask(
            self.mws_mask.MWS_NEARBY.obsconditions)
        targets['OBSCONDITIONS'] |= (mws_nearby != 0)  * self.obsconditions.mask(
            self.desi_mask.MWS_ANY.obsconditions)

class WDMaker(SelectTargets):
    """Read WD mocks and generate spectra."""

    def __init__(self, seed=None, verbose=False, **kwargs):

        from scipy.spatial import cKDTree as KDTree
        
        super(WDMaker, self).__init__(**kwargs)

        self.seed = seed
        self.verbose = verbose
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'WD'
        
        self.meta_da = read_basis_templates(objtype='WD', subtype='DA', onlymeta=True)
        self.meta_db = read_basis_templates(objtype='WD', subtype='DB', onlymeta=True)

        self.tree_da = KDTree(np.vstack((self.meta_da['TEFF'].data,
                                         self.meta_da['LOGG'].data)).T)
        self.tree_db = KDTree(np.vstack((self.meta_db['TEFF'].data,
                                         self.meta_db['LOGG'].data)).T)

    def read(self, mockfile=None, mockformat='mws_wd', dust_dir=None,
             healpixels=None, nside=8, nside_chunk=128, magcut=None):
        """Read the mock file."""
        if mockformat == 'mws_wd':
            MockReader = ReadMWSWD(dust_dir=dust_dir)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   magcut=magcut)

        if bool(data):
            data = self._prepare_spectra(data)

        return data

    def _prepare_spectra(self, data):
        
        from desisim.templates import WD
        self.da_template_maker = WD(wave=self.wave, subtype='DA', normfilter=data['NORMFILTER'])
        self.db_template_maker = WD(wave=self.wave, subtype='DB', normfilter=data['NORMFILTER'])

        seed = self.rand.randint(2**32, size=len(data['RA']))

        data.update({
            'TRUESPECTYPE': 'STAR', 'TEMPLATETYPE': self.objtype, 
            'SEED': seed, 
            })

        return data

    def query(self, matrix, subtype='DA'):
        """Return the nearest template number based on the KD Tree."""

        if subtype.upper() == 'DA':
            dist, indx = self.tree_da.query(matrix)
        elif subtype.upper() == 'DB':
            dist, indx = self.tree_db.query(matrix)
        else:
            log.warning('Unrecognized SUBTYPE {}!'.format(subtype))
            raise ValueError

        return dist, indx
    
    def make_spectra(self, data=None, index=None):
        """Generate WD spectra.  Deal with DA vs DB white dwarfs separately.

        """
        from desisim.io import empty_metatable
        
        if index is None:
            index = np.arange(len(data['RA']))
        nobj = len(index)

        input_meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'MAG', 'REDSHIFT', 'TEFF', 'LOGG', 'SUBTYPE'),
                                  ('SEED', 'MAG', 'Z', 'TEFF', 'LOGG', 'TEMPLATESUBTYPE')):
            input_meta[inkey] = data[datakey][index]
            
        if data['MOCKFORMAT'].lower() == 'mws_wd':
            meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
            flux = np.zeros([nobj, len(self.wave)], dtype='f4')
            
            for subtype in ('DA', 'DB'):
                these = np.where(input_meta['SUBTYPE'] == subtype)[0]
                if len(these) > 0:
                    alldata = np.vstack((data['TEFF'][index][these],
                                         data['LOGG'][index][these])).T
                    _, templateid = self.query(alldata, subtype=subtype)
                    
                    input_meta['TEMPLATEID'][these] = templateid
                    
                    template_maker = getattr(self, '{}_template_maker'.format(subtype.lower()))
                    flux1, _, meta1 = template_maker.make_templates(input_meta=input_meta[these],
                                                                    verbose=self.verbose)
                    meta[these] = meta1
                    flux[these, :] = flux1
            
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        return flux, self.wave, meta

    def select_targets(self, targets, truth):
        """Select MWS_WD and STD_WD targets.  The selection eventually will be done with
        Gaia, so for now just do a "perfect" selection here.

        """
        #mws_wd = np.ones(len(targets)) # select everything!
        mws_wd = ((truth['MAG'] >= 15.0) * (truth['MAG'] <= 20.0)) * 1 # SDSS g-band!

        targets['MWS_TARGET'] |= (mws_wd != 0) * self.mws_mask.mask('MWS_WD')
        targets['DESI_TARGET'] |= (mws_wd != 0) * self.desi_mask.MWS_ANY
        targets['OBSCONDITIONS'] |= (mws_wd != 0)  * self.obsconditions.mask(
            self.mws_mask.MWS_WD.obsconditions)
        targets['OBSCONDITIONS'] |= (mws_wd != 0)  * self.obsconditions.mask(
            self.desi_mask.MWS_ANY.obsconditions)

        # Select STD_WD; cut just on g-band magnitude (not TEMPLATESUBTYPE!)
        std_wd = (truth['MAG'] <= 19.0) * 1 # SDSS g-band!
        targets['DESI_TARGET'] |= (std_wd !=0) * self.desi_mask.mask('STD_WD')
        targets['OBSCONDITIONS'] |= (std_wd != 0)  * self.obsconditions.mask(
            self.desi_mask.STD_WD.obsconditions)

class SKYMaker(SelectTargets):

    def __init__(self, seed=None, **kwargs):

        super(SKYMaker, self).__init__(**kwargs)

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.wave = _default_wave()
        self.objtype = 'SKY'

    def read(self, mockfile=None, mockformat='gaussianfield', dust_dir=None,
             healpixels=None, nside=8, magcut=None):
        """Read the mock file."""

        if mockformat == 'gaussianfield':
            MockReader = ReadGaussianField(dust_dir=dust_dir)
        else:
            raise ValueError('Unrecognized mockformat {}!'.format(mockformat))

        data = MockReader.readmock(mockfile, target_name=self.objtype,
                                   healpixels=healpixels, nside=nside,
                                   magcut=magcut)

        if bool(data):
            data = self._prepare_spectra(data)

        return data

    def _prepare_spectra(self, data):

        seed = self.rand.randint(2**32, size=len(data['RA']))
        data.update({
            'TRUESPECTYPE': 'SKY', 'TEMPLATETYPE': self.objtype, 'TEMPLATESUBTYPE': '',
            'SEED': seed,
            })

        return data

    def make_spectra(self, data=None, index=None):
        """Generate SKY spectra."""
        from desisim.io import empty_metatable
        
        if index is None:
            index = np.arange(len(data['RA']))
        nobj = len(index)
            
        meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'REDSHIFT'), ('SEED', 'Z')):
            meta[inkey] = data[datakey][index]
        flux = np.zeros((nobj, len(self.wave)), dtype='i1')

        return flux, self.wave, meta

    def select_targets(self, targets, truth):
        """Select SKY targets."""

        targets['DESI_TARGET'] |= self.desi_mask.mask('SKY')
        targets['OBSCONDITIONS'] |= self.obsconditions.mask(
            self.desi_mask.SKY.obsconditions)

