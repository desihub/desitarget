# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==================
desitarget.mock.io
==================

Handles mock data to build target catalogs.
"""
from __future__ import (absolute_import, division, print_function)
#
import numpy as np
import fitsio
import os, re
import desitarget.io
import h5py
import desitarget.targets

"""
How to distribute 52 user bits of targetid.

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


############################################################
def _load_mock_mws_file(filename):
    """
    Reads mock information for MWS bright time survey.

    Parameters:
    ----------
    filename: :class:`str`
        Name of a single MWS mock file.

    Returns:
    -------
    Dictionary with the following entries.

        'RA': :class: `numpy.ndarray`
            RA positions for the objects in the mock.
        'DEC': :class: `numpy.ndarray`
            DEC positions for the objects in the mock.
        'Z': :class: `numpy.ndarray`
            Heliocentric radial velocity divided by the speed of light.

        'd_helio': :class `numpy.ndarray'
            Heliocentric distance in kpc, required only to avoid overlap
            with 100pc sample.
        'SDSSr_true': :class: `numpy.ndarray`
            Apparent magnitude in SDSS bands r, before extinction.
        'SDSS[grz]_obs': :class: `numpy.ndarray`
             Apparent magnitudes in SDSS grz bands, including extinction.
    """
    print('Reading '+filename)
    C_LIGHT = 299792.458
    desitarget.io.check_fitsio_version()
    data = fitsio.read(filename,
                       columns= ['objid','RA','DEC','v_helio','d_helio', 'SDSSr_true',
                                 'SDSSr_obs', 'SDSSg_obs', 'SDSSz_obs'])


    objid       = data['objid'].astype('i8')
    ra          = data['RA'].astype('f8') % 360.0 #enforce 0 < ra < 360
    dec         = data['DEC'].astype('f8')
    v_helio     = data['v_helio'].astype('f8')
    d_helio     = data['d_helio'].astype('f8')
    SDSSr_true  = data['SDSSr_true'].astype('f8')
    SDSSg_obs   = data['SDSSg_obs'].astype('f8')
    SDSSr_obs   = data['SDSSr_obs'].astype('f8')
    SDSSz_obs   = data['SDSSz_obs'].astype('f8')

    return {'objid':objid,
            'RA':ra, 'DEC':dec, 'Z': v_helio/C_LIGHT,
            'd_helio': d_helio,
            'SDSSr_true': SDSSr_true, 'SDSSr_obs': SDSSr_obs,
            'SDSSg_obs' : SDSSg_obs, 'SDSSz_obs' : SDSSz_obs}



############################################################
def _load_mock_lyaqso_file(filename):
    """
    Reads mock information for 

    Parameters:
    ----------
    filename: :class:`str`
        Name of a single MWS mock file.

    Returns:
    -------
    Dictionary with the following entries.

        'RA': :class: `numpy.ndarray`
            RA positions for the objects in the mock.
        'DEC': :class: `numpy.ndarray`
            DEC positions for the objects in the mock.
        'Z': :class: `numpy.ndarray`
            Redshift
    """

    desitarget.io.check_fitsio_version()

    h = fitsio.FITS(filename)

    heads = [head.read_header() for head in h]

    n = len(heads) - 1 # the first item in heads is empty
    z = np.zeros(n)
    ra = np.zeros(n)
    dec = np.zeros(n)
    
    for i in range(n):
        z[i]  = heads[i+1]["ZQSO"]
        ra[i]  = heads[i+1]["RA"]
        dec[i]  = heads[i+1]["DEC"]

    n = len(ra)
    objid = np.arange(n, dtype='i8')
    ra = ra * 180.0 / np.pi
    dec = dec * 180.0 / np.pi
    ra          = ra % 360.0 #enforce 0 < ra < 360

    return {'objid':objid, 'RA':ra, 'DEC':dec, 'Z': z}


############################################################
def encode_rownum_filenum(rownum, filenum):
    """Encodes row and file number in 52 packed bits.

    Parameters:
        rownum (int): Row in input file
        filenum (int): File number in input file set

    Return:
        encoded value(s) (int64 ndarray):
            52 packed bits encoding row and file number
    """
    assert(np.shape(rownum) == np.shape(filenum))
    assert(np.all(rownum  >= 0))
    assert(np.all(rownum  <= int(ENCODE_ROW_MAX)))
    assert(np.all(filenum >= 0))
    assert(np.all(filenum <= int(ENCODE_FILE_MAX)))

    # This should be a 64 bit integer.
    encoded_value = (np.asarray(filenum,dtype=np.uint64) << ENCODE_ROW_END) + np.asarray(rownum,dtype=np.uint64)

    # Note return signed
    return np.asarray(encoded_value,dtype=np.int64)

############################################################
def decode_rownum_filenum(encoded_values):
    """Inverts encode_rownum_filenum to obtain row number and file number.

    Parameters:
        encoded_values(s) (int64 ndarray)

    Return:
        (filenum, rownum) (int)
    """
    filenum = (np.asarray(encoded_values,dtype=np.uint64) & ENCODE_FILE_MASK) >> ENCODE_ROW_END
    rownum  = (np.asarray(encoded_values,dtype=np.uint64) & ENCODE_ROW_MASK)
    return rownum,filenum

############################################################
def read_wd100pc(mock_dir, target_type, mock_name=None):
    """ Reads a single-file GUMS-based mock that includes 'big brick'
    bricknames as in the Galaxia and Galfast mocks.

    Parameters:
    ----------
    root_mock_dir: :class:`str`
        Path to the mock file.

    mock_name: :class:`str`
        Optional name of the mock file.
        default: 'mock_wd100pc.fits'

    brickname_list:
        Optional list of specific bricknames to read.

    Returns:
    -------
    Dictionary with the following entries.

        'RA': :class: `numpy.ndarray`
            RA positions for the objects in the mock.
        'DEC': :class: `numpy.ndarray`
            DEC positions for the objects in the mock.
        'Z': :class: `numpy.ndarray`
            Heliocentric radial velocity divided by the speed of light.
        'magg': :class: `numpy.ndarray`
            Apparent magnitudes in Gaia G band
    """
    desitarget.io.check_fitsio_version()
    C_LIGHT = 299792.458

    mock_name = 'mock_wd100pc.fits'
    filename  = os.path.join(mock_dir,mock_name)
    data = fitsio.read(filename,
                       columns= ['RA','DEC','radialvelocity','magg','WD','objid'])

    obijd       = data['objid'].astype('i8') 
    ra          = data['RA'].astype('f8') % 360.0 #enforce 0 < ra < 360
    dec         = data['DEC'].astype('f8')
    v_helio     = data['radialvelocity'].astype('f8')
    magg        = data['magg'].astype('f8')
    is_wd       = data['WD'].astype('i4')

    files = list()
    files.append(filename)
    n_per_file = list()
    n_per_file.append(len(ra))


    print('read {} objects'.format(n_per_file[0]))
    return {'RA':ra, 'DEC':dec, 'Z': v_helio/C_LIGHT,
            'magg': magg, 'WD':is_wd, 'FILES': files, 'N_PER_FILE': n_per_file}

############################################################
def read_galaxia(mock_dir, target_type, mock_name=None):
    """ Reads and concatenates MWS mock files stored below the root directory.

    Parameters:
    ----------
    root_mock_dir: :class:`str`
        Path to all the 'desi_galfast' files.

    mock_prefix: :class:`str`
        Start of individual file names.

    brickname_list:
        Optional list of specific bricknames to read.

    Returns:
    -------
    Dictionary concatenating all the 'desi_galfast' files with the following entries.

        'RA': :class: `numpy.ndarray`
            RA positions for the objects in the mock.
        'DEC': :class: `numpy.ndarray`
            DEC positions for the objects in the mock.
        'Z': :class: `numpy.ndarray`
            Heliocentric radial velocity divided by the speed of light.
        'SDSSr_true': :class: `numpy.ndarray`
            Apparent magnitudes in SDSS bands, including extinction.
        'SDSSr_obs': :class: `numpy.ndarray`
             Apparent magnitudes in SDSS bands, including extinction.
    """
    # Build iterator of all desi_galfast files
    iter_mock_files = desitarget.io.iter_files(mock_dir, '', ext="fits")

    # Read each file

    # Multiprocessing parallel I/O, but this fails for galaxia 0.0.2 mocks
    # due to python issue https://bugs.python.org/issue17560 where
    # Pool.map can't return objects with more then 2**32-1 bytes:
    # multiprocessing.pool.MaybeEncodingError: Error sending result:
    # Reason: 'error("'i' format requires -2147483648 <= number <= 2147483647",)'
    # Leaving this code here for the moment in case we fine a workaround

    # import multiprocessing
    # print('Reading individual mock files')
    # file_list = list(iter_mock_files)
    # nfiles = len(file_list)
    # ncpu = max(1, multiprocessing.cpu_count() // 2)
    # print('using {} parallel readers'.format(ncpu))
    # p = multiprocessing.Pool(ncpu)
    # target_list = p.map(_load_mock_mws_file, file_list)

    print('Reading individual mock files')
    target_list = list()
    file_list   = list()
    nfiles      = 0
    for mock_file in iter_mock_files:
        nfiles += 1
        data_this_file = _load_mock_mws_file(mock_file)
        target_list.append(data_this_file)
        file_list.append(mock_file)
        print('read file {} {}'.format(nfiles, mock_file))

    print('Read {} files'.format(nfiles))

    # Concatenate all the dictionaries into a single dictionary, in an order
    # determined by np.argsort applied to the base name of each path in
    # file_list.
    file_order = np.argsort([os.path.basename(x) for x in file_list])

    print('Combining mock files')
    ordered_file_list = list()
    n_per_file  = list()
    full_data   = dict()
    if len(target_list) > 0:
        for k in list(target_list[0]): #iterate over keys
            print(' -- {}'.format(k))
            data_list_this_key = list()
            for itarget in file_order: #append all the arrays corresponding to a given key
                data_list_this_key.append(target_list[itarget][k])

            full_data[k] = np.concatenate(data_list_this_key) #consolidate data dictionary

        # Count number of points per file
        k          = list(target_list[0])[0] # pick the first available column
        n_per_file = [len(target_list[itarget][k]) for itarget in file_order]
        odered_file_list = [file_list[itarget] for itarget in file_order]

    print('Read {} objects'.format(np.sum(n_per_file)))

    full_data['FILES']      = ordered_file_list
    full_data['N_PER_FILE'] = n_per_file
    return full_data


############################################################
def read_lyaqso(mock_dir, target_type, mock_name=None):
    """ Reads and concatenates MWS mock files stored below the root directory.

    Parameters:
    ----------
    root_mock_dir: :class:`str`
        Path to all the 'desi_galfast' files.

    mock_prefix: :class:`str`
        Start of individual file names.

    brickname_list:
        Optional list of specific bricknames to read.

    Returns:
    -------
    Dictionary concatenating all the 'desi_galfast' files with the following entries.

        'RA': :class: `numpy.ndarray`
            RA positions for the objects in the mock.
        'DEC': :class: `numpy.ndarray`
            DEC positions for the objects in the mock.
        'Z': :class: `numpy.ndarray`
            Heliocentric radial velocity divided by the speed of light.
    """
    # Build iterator of all desi_galfast files
    iter_mock_files = desitarget.io.iter_files(mock_dir, '', ext="fits.gz")

    # Read each file
    print('Reading individual mock files')
    file_list = list(iter_mock_files)
    nfiles = len(file_list)

    import multiprocessing
    ncpu = max(1, multiprocessing.cpu_count() // 2)
    print('using {} parallel readers'.format(ncpu))
    p = multiprocessing.Pool(ncpu)
    target_list = p.map(_load_mock_lyaqso_file, file_list)

    print('Read {} files'.format(nfiles))

    # Concatenate all the dictionaries into a single dictionary, in an order
    # determined by np.argsort applied to the base name of each path in
    # file_list.
    file_order = np.argsort([os.path.basename(x) for x in file_list])

    print('Combining mock files')
    ordered_file_list = list()
    n_per_file  = list()
    full_data   = dict()
    if len(target_list) > 0:
        for k in list(target_list[0]): #iterate over keys
            print(' -- {}'.format(k))
            data_list_this_key = list()
            for itarget in file_order: #append all the arrays corresponding to a given key
                data_list_this_key.append(target_list[itarget][k])

            full_data[k] = np.concatenate(data_list_this_key) #consolidate data dictionary

        # Count number of points per file
        k          = list(target_list[0])[0] # pick the first available column
        n_per_file = [len(target_list[itarget][k]) for itarget in file_order]
        odered_file_list = [file_list[itarget] for itarget in file_order]

    print('Read {} objects'.format(np.sum(n_per_file)))

    full_data['FILES']      = ordered_file_list
    full_data['N_PER_FILE'] = n_per_file
    return full_data

############################################################
def read_gaussianfield(mock_dir, target_type, mock_name=None):
    """Reads preliminary mocks (positions only) for the dark time survey.

    Parameters:
    ----------
    filename : :class:`str`
        File name of one mock dark time file.
    read_z =: :class:`boolean`
        Option to read the redshift column.

    Returns:
    --------
    ra: :class: `numpy.ndarray`
        Array with the RA positions for the objects in the mock.
    dec: :class: `numpy.ndarray`
        Array with the DEC positions for the objects in the mock.
    z: :class: `numpy.ndarray`
        Array with the redshiffts for the objects in the mock.
        Zeros if read_z = False
    """
    desitarget.io.check_fitsio_version()
    if mock_name is None:
        filename = os.path.join(mock_dir, target_type+'.fits')
    else:
        filename = os.path.join(mock_dir, mock_name+'.fits')

    try:
        data = fitsio.read(filename,columns=['RA','DEC','Z'], upper=True)
        ra   = data[ 'RA'].astype('f8') % 360.0 #enforce 0 < ra < 360
        dec  = data['DEC'].astype('f8')
        zz   = data[  'Z'].astype('f8')
    except:
        data = fitsio.read(filename,columns=['RA','DEC'], upper=True)
        ra   = data[ 'RA'].astype('f8') % 360.0 #enforce 0 < ra < 360
        dec  = data['DEC'].astype('f8')
        zz = np.random.uniform(0.0, 1.0, size=len(ra))

    print('read {} lines from {}'.format(len(data), filename))
    del data
    files = list()
    files.append(filename)
    n_per_file = list()
    n_per_file.append(len(ra))
    return {'RA':ra, 'DEC':dec, 'Z':zz, 'FILES': files, 'N_PER_FILE': n_per_file}

############################################################
def read_durham_mxxl_hdf5(mock_dir, target_type, mock_name=None):
    """ Reads mock information for MXXL bright time survey galaxies.

    Args:
        filename (str): Name of a single mock file.

    Returns:
        dict with the following entries (all ndarrays):

        RA          : RA positions for the objects in the mock.
        DEC         : DEC positions for the objects in the mock.
        Z           : Heliocentric radial velocity divided by the speed of light.
        SDSSr_true  : Apparent magnitudes in SDSS r band.
    """

    filename = os.path.join(mock_dir, target_type+'.hdf5')
    f = h5py.File(filename)
    ra  = f["Data/ra"][...].astype('f8') % 360.0
    dec = f["Data/dec"][...].astype('f8')
    SDSSr_true   = f["Data/app_mag"][...].astype('f8')
    zred   = f["Data/z_obs"][...].astype('f8')
    f.close()

    print('read {} lines from {}'.format(len(ra), filename))

    files = list()
    files.append(filename)
    n_per_file = list()
    n_per_file.append(len(ra))

    return {'RA':ra, 'DEC':dec, 'Z': zred ,
            'SDSSr_true':SDSSr_true, 'FILES': files, 'N_PER_FILE': n_per_file}

############################################################
def read_mock_durham(core_filename, photo_filename):
    """
    Args:
    -----
        core_filename: filename of the hdf5 file storing core lightconedata
        photo_filename: filename of the hdf5 storing photometric data

    Returns:
    -------
        objects: ndarray with the structure required to go through
        desitarget.cuts.select_targets()
    """

    import h5py

    fin_core = h5py.File(core_filename, "r")
    fin_mags = h5py.File(photo_filename, "r")

    core_data = fin_core.require_group('/Data')
    photo_data = fin_mags.require_group('/Data')


    gal_id_string = core_data['GalaxyID'].value # these are string values, not integers!

    n_gals = 0
    n_gals = core_data['ra'].size

    #the mock has to be converted in order to create the following columns
    columns = [
        'BRICKID', 'BRICKNAME', 'OBJID', 'BRICK_PRIMARY', 'TYPE',
        'RA', 'RA_IVAR', 'DEC', 'DEC_IVAR',
        'DECAM_FLUX', 'DECAM_MW_TRANSMISSION',
        'WISE_FLUX', 'WISE_MW_TRANSMISSION',
        'SHAPEDEV_R', 'SHAPEEXP_R',
        ]


    obj_id = np.arange(n_gals)
    brickid = np.ones(n_gals, dtype='int64')
    shapedev_r  = np.zeros(n_gals)
    shapeexp_r = np.zeros(n_gals)
    wise_mw_transmission = np.ones((n_gals,4))
    decam_mw_transmission = np.ones((n_gals,6))
    brick_primary = np.ones(n_gals, dtype=bool)
    morpho_type = np.chararray(n_gals, itemsize=3)
    morpho_type[:] = 'EXP'
    brick_name = np.chararray(n_gals, itemsize=8)
    brick_name[:] = '0durham0'

    ra = core_data['ra'].value
    dec = core_data['dec'].value
    dec_ivar = 1.0E10 * np.ones(n_gals)
    ra_ivar = 1.0E10 * np.ones(n_gals)

    wise_flux = np.zeros((n_gals,4))
    decam_flux = np.zeros((n_gals,6))

    g_mags = photo_data['appDgo_tot_ext'].value
    r_mags = photo_data['appDro_tot_ext'].value
    z_mags = photo_data['appDzo_tot_ext'].value

    decam_flux[:,1] = 10**((22.5 - g_mags)/2.5)
    decam_flux[:,2] = 10**((22.5 - r_mags)/2.5)
    decam_flux[:,4] = 10**((22.5 - z_mags)/2.5)

    #this corresponds to the return type of read_tractor() using DECaLS DR1 tractor data.
    type_table = [
        ('BRICKID', '>i4'),
        ('BRICKNAME', '|S8'),
        ('OBJID', '>i4'),
        ('BRICK_PRIMARY', '|b1'),
        ('TYPE', '|S4'),
        ('RA', '>f8'),
        ('RA_IVAR', '>f4'),
        ('DEC', '>f8'),
        ('DEC_IVAR', '>f4'),
        ('DECAM_FLUX', '>f4', (6,)),
        ('DECAM_MW_TRANSMISSION', '>f4', (6,)),
        ('WISE_FLUX', '>f4', (4,)),
        ('WISE_MW_TRANSMISSION', '>f4', (4,)),
        ('SHAPEEXP_R', '>f4'),
        ('SHAPEDEV_R', '>f4')
    ]
    data = np.ndarray(shape=(n_gals), dtype=type_table)
    data['BRICKID'] = brickid
    data['BRICKNAME'] = brick_name
    data['OBJID'] = obj_id
    data['BRICK_PRIMARY'] = brick_primary
    data['TYPE'] = morpho_type
    data['RA'] = ra
    data['RA_IVAR'] = ra_ivar
    data['DEC'] = dec
    data['DEC_IVAR'] = dec_ivar
    data['DECAM_FLUX'] = decam_flux
    data['DECAM_MW_TRANSMISSION'] = decam_mw_transmission
    data['WISE_FLUX'] = wise_flux
    data['WISE_MW_TRANSMISSION'] = wise_mw_transmission
    data['SHAPEEXP_R'] = shapeexp_r
    data['SHAPEDEV_R'] = shapedev_r

    return data
