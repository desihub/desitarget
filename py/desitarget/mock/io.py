# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==================
desitarget.mock.io
==================

Handles mock data to build target catalogs.
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import fitsio
import os, re
import desitarget.io

MOCK_ENCODE_FILENUM_OFFSET = int(1e9)
"""
Used to generate target IDs as combination of input file and row in input
ile. Sets the maximum number of rows per input file for mocks using this
scheme to generate target IDs
"""

############################################################
def _load_mock_bgs_mxxl_file(filename):
    """ Reads mock information for MXXL bright time survey galaxies.
    
    Args:
        filename (str): Name of a single mock file.
    
    Returns:
        dict with the following entries (all ndarrays):

        objid       : Mock object ID
        brickid     : Mock brick ID
        RA          : RA positions for the objects in the mock.
        DEC         : DEC positions for the objects in the mock.
        Z           : Heliocentric radial velocity divided by the speed of light.
        SDSSr_true  : Apparent magnitudes in SDSS r band.
    """
    desitarget.io.check_fitsio_version()
    data = fitsio.read(filename,
                       columns= ['objid','brickid',
                                 'RA','DEC','Z', 'R'])

    objid       = data['objid'].astype('i8')
    brickid     = data['brickid'].astype('i8')
    ra          = data['RA'].astype('f8') % 360.0 #enforce 0 < ra < 360
    dec         = data['DEC'].astype('f8')
    SDSSr_true  = data['R'].astype('f8')
    zred        = data['Z'].astype('f8')

    return {'objid':objid,'brickid':brickid,
            'RA':ra, 'DEC':dec, 'Z': zred ,
            'SDSSr_true':SDSSr_true}

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
        'SDSSr_true': :class: `numpy.ndarray`
            Apparent magnitudes in SDSS bands, including extinction.
        'SDSSr_obs': :class: `numpy.ndarray`
             Apparent magnitudes in SDSS bands, including extinction.
    """
    C_LIGHT = 300000.0
    desitarget.io.check_fitsio_version()
    data = fitsio.read(filename,
                       columns= ['objid','brickid',
                                 'RA','DEC','v_helio','SDSSr_true', 'SDSSr_obs'])

    objid       = data['objid'].astype('i8')
    brickid     = data['brickid'].astype('i8')
    ra          = data['RA'].astype('f8') % 360.0 #enforce 0 < ra < 360
    dec         = data['DEC'].astype('f8')
    v_helio     = data['v_helio'].astype('f8')
    SDSSr_true  = data['SDSSr_true'].astype('f8')
    SDSSr_obs   = data['SDSSr_obs'].astype('f8')

    return {'objid':objid,'brickid':brickid,
            'RA':ra, 'DEC':dec, 'Z': v_helio/C_LIGHT, 
            'SDSSr_true': SDSSr_true, 'SDSSr_obs': SDSSr_obs}

############################################################
def _load_mock_wd100pc_file(filename):
    """
    Reads mock information for MWS bright time survey.
   
    WD/100pc mock, all sky in one file.

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
        'magg': :class: `numpy.ndarray`
            Apparent magnitudes in Gaia G band.
        'WD': :class: `numpt.ndarray'
            1 == WD, 0 == Not a WD 
    """
    C_LIGHT = 300000.0
    desitarget.io.check_fitsio_version()
    data = fitsio.read(filename,
                       columns= ['objid','brickid',
                                 'RA','DEC','radialvelocity','magg','WD'])

    objid       = data['objid'].astype('i8')
    brickid     = data['brickid'].astype('i8')
    ra          = data['RA'].astype('f8') % 360.0 #enforce 0 < ra < 360
    dec         = data['DEC'].astype('f8')
    v_helio     = data['radialvelocity'].astype('f8')
    magg        = data['magg'].astype('f8')
    is_wd       = data['WD'].astype('i4')

    return {'objid':objid,'brickid':brickid,
            'RA':ra, 'DEC':dec, 'Z': v_helio/C_LIGHT, 
            'magg': magg, 'WD':is_wd}

############################################################
def _read_mock_add_file_and_row_number(target_list,full_data):
    """Adds row and file number to dict of properties. 
    
    Parameters:
        target_list (list of dicts):
            Each dict in the list contains data for one file, list is in same
            order as files are read.

        full_data (dict): 
            dict returned by any of the mock-reading routines.

    Side effects:
        Modifies full_data.
    """
    full_data['rownum']  = np.empty(0)
    full_data['filenum'] = np.empty(0)

    fiducial_key = target_list[0].keys()[0]
    for ifile,target_item in enumerate(target_list):
        nrows                = len(target_item[fiducial_key])
        full_data['rownum']  = np.append(full_data['rownum'],  np.arange(0,nrows))
        full_data['filenum'] = np.append(full_data['filenum'], np.repeat(ifile,nrows))
    return 

############################################################
def encode_rownum_filenum(rownum, filenum):
    """
    Parameters:
        rownum (int): Row in input file 
        filenum (int): File number in input file set

    Return:
        encoded value(s) (int64 ndarray):
            MOCK_ENCODE_FILENUM_OFFSET*filenum + rownum
    """
    assert(np.shape(rownum) == np.shape(filenum))
    assert(np.all(rownum < MOCK_ENCODE_FILENUM_OFFSET))
    assert(np.all(rownum > 0))
    assert(np.all(filenum > 0))

    encoded_value = MOCK_ENCODE_FILENUM_OFFSET*filenum + rownum
    return np.asarray(encoded_value,dtype=np.int64)

############################################################
def read_mock_wd100pc_brighttime(root_mock_wd100pc_dir='',mock_wd100pc_name=None):
    """ Reads a single-file GUMS-based mock that includes 'big brick'
    bricknames as in the Galaxia and Galfast mocks.

    Parameters:
    ----------    
    root_mock_wd100pc_dir: :class:`str`
        Path to the mock file.

    mock_wd100pc_name: :class:`str`
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
    if mock_wd100pc_name is None:
        mock_wd100pc_name = 'mock_wd100pc.fits'
    filename  = os.path.join(root_mock_wd100pc_dir,mock_wd100pc_name)
    full_data = _load_mock_wd100pc_file(filename)

    # Add file and row number
    fiducial_key         = full_data.keys()[0]
    nrows                = len(full_data[fiducial_key])
    full_data['rownum']  = np.arange(0,nrows)
    full_data['filenum'] = np.zeros(nrows,dtype=np.int)

    file_list = [filename]

    return full_data, file_list

############################################################
def read_mock_mws_brighttime(root_mock_mws_dir='',mock_mws_prefix='',brickname_list=None):
    """ Reads and concatenates the MWS mock files stored below the root directory.

    Parameters:
    ----------    
    mock_mws_dir: :class:`str`
        Path to all the 'desi_galfast' files.

    mock_mws_prefix: :class:`str`
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
    iter_mock_files = desitarget.io.iter_files(root_mock_mws_dir, mock_mws_prefix, ext="fits")
    
    # Read each file
    print('Reading individual mock files')
    target_list = list()
    file_list   = list()
    nfiles      = 0
    for mock_file in iter_mock_files:
        nfiles += 1

        # Filter on bricknames
        if brickname_list is not None:
            brickname_of_target = desitarget.io.brickname_from_filename_with_prefix(mock_file,prefix=mock_mws_prefix)
            if not brickname_of_target in brickname_list:
                continue
        
        # print(mock_file) # Don't necessarily want to do this
        target_list.append(_load_mock_mws_file(mock_file))
        file_list.append(mock_file)

    print('Found %d files, read %d after filtering'%(nfiles,len(target_list)))

    # Concatenate all the dictionaries into a single dictionary
    print('Combining mock files')
    full_data = dict()
    if len(target_list) > 0:
        for k in target_list[0].keys():
            full_data[k] = np.empty(0)
            for target_item in target_list:                
                full_data[k] = np.append(full_data[k] ,target_item[k])

        # Add file and row number
        _read_mock_add_file_and_row_number(target_list,full_data)
    
    return full_data,file_list

############################################################
def read_mock_bgs_mxxl_brighttime(root_mock_bgs_mxxl_dir='',mock_prefix='',brickname_list=None):
    """ Reads and concatenates the brick-style BGS MXXL mock files stored below the root directory.

    Parameters:
    ----------    
    mock_mws_dir: :class:`str`
        Path to all the 'mock_mxxl' files.

    mock_prefix: :class:`str`
        Start of individual file names.

    brickname_list:
        Optional list of specific bricknames to read.
        
    Returns:
    -------
    Dictionary concatenating all the 'mock_mxxl' files with the following entries.

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
    # Build iterator of all mock brick files
    iter_mock_files = desitarget.io.iter_files(root_mock_bgs_mxxl_dir, mock_prefix, ext="fits")
    
    # Read each file
    print('Reading individual mock files')
    target_list = list()
    file_list   = list()
    nfiles      = 0
    for mock_file in iter_mock_files:
        nfiles += 1

        # Filter on bricknames
        if brickname_list is not None:
            brickname_of_target = desitarget.io.brickname_from_filename_with_prefix(mock_file,prefix=mock_prefix)
            if not brickname_of_target in brickname_list:
                continue
        
        # print(mock_file) # Don't necessarily want to do this
        target_list.append(_load_mock_bgs_mxxl_file(mock_file))
        file_list.append(mock_file)

    print('Found %d files, read %d after filtering'%(nfiles,len(target_list)))

    # Concatenate all the dictionaries into a single dictionary
    print('Combining mock files')
    full_data = dict()
    if len(target_list) > 0:
        for k in target_list[0].keys():
            full_data[k] = np.empty(0)
            for target_item in target_list:
                full_data[k] = np.append(full_data[k] ,target_item[k])

        # Add file and row number
        _read_mock_add_file_and_row_number(target_list,full_data)
    
    return full_data,file_list


############################################################
def read_mock_dark_time(filename, read_z=True):
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

    if read_z :
        data = fitsio.read(filename,columns=['RA','DEC','Z'])
        ra   = data[ 'RA'].astype('f8') % 360.0 #enforce 0 < ra < 360
        dec  = data['DEC'].astype('f8')
        zz   = data[  'Z'].astype('f8')
    else:
        data = fitsio.read(filename,columns=['RA','DEC'])
        ra   = data[ 'RA'].astype('f8') % 360.0 #enforce 0 < ra < 360
        dec  = data['DEC'].astype('f8')
        zz   = np.zeros(len(ra))

    del data

    return ( (ra,dec,zz))

    
                                                                                                                             

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




