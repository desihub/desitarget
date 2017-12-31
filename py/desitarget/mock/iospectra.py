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
from glob import glob

import fitsio
from scipy import constants

from desisim.io import read_basis_templates, empty_metatable
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

class GaussianField(object):

    def __init__(self, seed=None, bricksize=0.25, **kwargs):

        super(GaussianField, self).__init__(seed=seed, **kwargs)

        self.bricksize = bricksize
        self.mockdir_root = os.path.join( os.getenv('DESI_ROOT'), 'mocks', 'GaussianRandomField' )

        if not hasattr(self, 'seed'):
            self.seed = seed

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
        seed = self.rand.randint(2**32, size=nobj)

        # Add redshifts.
        if target_name.upper() == 'SKY':
            zz = np.zeros(len(ra))
            mag = np.zeros_like(zz) + 22 # placeholder
        else:
            data = fitsio.read(mockfile, columns=['Z_COSMO', 'DZ_RSD'], upper=True, ext=1, rows=cut)
            zz = (data['Z_COSMO'].astype('f8') + data['DZ_RSD'].astype('f8')).astype('f4')
            mag = np.zeros_like(zz) + 22 # placeholder
            
        # Return a basic dictionary.
        out = {'OBJID': objid, 'MOCKID': mockid, 'RA': ra, 'DEC': dec, 
               'BRICKNAME': brickname, 'SEED': seed, 'FILES': files,
               'N_PER_FILE': n_per_file, 'Z': zz, 'MAG': mag}

        return out

class MockSpectra(object):
    """Generate spectra for each type of mock target.

    ToDo (@moustakas): apply Galactic extinction.

    """
    def __init__(self, wavemin=None, wavemax=None, dw=0.2, 
                 seed=None, verbose=False):
        from desimodel.io import load_throughput

        if not hasattr(self, 'seed'):
            self.seed = seed

        # Build a default (buffered) wavelength vector.
        if wavemin is None:
            wavemin = load_throughput('b').wavemin - 10.0
        if wavemax is None:
            wavemax = load_throughput('z').wavemax + 10.0
            
        self.wavemin = wavemin
        self.wavemax = wavemax
        self.dw = dw
        self.wave = np.arange(round(wavemin, 1), wavemax, dw)

        #self.tree = TemplateKDTree(nproc=nproc, verbose=verbose)

class QSO(GaussianField, MockSpectra):

    def __init__(self, seed=None, **kwargs):
        from desisim.templates import SIMQSO

        self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        
        super(QSO, self).__init__(seed=self.seed, **kwargs)

        self.objtype = 'QSO'
        self.mockfile = os.path.join( self.mockdir_root, 'v0.0.5', '{}.fits'.format(self.objtype) ) # default 
        self.template_maker = SIMQSO(wave=self.wave, normfilter='decam2014-g')

    def read(self, mockfile=None, healpixels=None, nside=8):
        """Read the mock file."""

        if mockfile is None:
            mockfile = self.mockfile

        data = self.readmock(mockfile, target_name=self.objtype,
                             healpixels=healpixels, nside=nside)
        if bool(data):
            data.update({'TRUESPECTYPE': self.objtype, 'TEMPLATETYPE': self.objtype,
                         'TEMPLATESUBTYPE': ''})

        return data

    def spectra(self, data=None, index=None):
        """Generate tracer QSO spectra."""
        
        if index is None:
            index = np.arange(len(data['RA']))
        nobj = len(index)
            
        flux, wave, meta = self.template_maker.make_templates(
            nmodel=nobj, redshift=data['Z'][index], seed=self.seed,
            lyaforest=False, nocolorcuts=True)

        return flux, self.wave, meta

class Sky(GaussianField, MockSpectra):

    def __init__(self, seed=None):
        super(Sky, self).__init__()

        self.objtype = 'SKY'
        self.mockfile = os.path.join( self.mockdir_root, 'v0.0.1', '2048', 'random.fits' ) # default 

    def read(self, mockfile=None, healpixels=None, nside=8):
        """Read the mock file."""

        if mockfile is None:
            mockfile = self.mockfile

        data = self.readmock(mockfile, target_name=self.objtype,
                             healpixels=healpixels, nside=nside)
        if bool(data):
            data.update({'TRUESPECTYPE': self.objtype, 'TEMPLATETYPE': self.objtype,
                         'TEMPLATESUBTYPE': ''})

        return data

    def spectra(self, data=None, index=None):
        """Generate SKY spectra."""
        
        if index is None:
            index = np.arange(len(data['RA']))
        nobj = len(index)
            
        meta = empty_metatable(nmodel=nobj, objtype=self.objtype)
        for inkey, datakey in zip(('SEED', 'REDSHIFT'),
                                  ('SEED', 'Z')):
            meta[inkey] = data[datakey][index]
        flux = np.zeros((nobj, len(self.wave)), dtype='i1')

        return flux, self.wave, meta
