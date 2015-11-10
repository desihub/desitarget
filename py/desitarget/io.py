"""
    This file knows how to write a TS catalogue.

"""

# everybody likes np
import numpy as np 
import numpy.lib.recfunctions
from astropy.io import fits
import fitsio
import os, re
from distutils.version import LooseVersion

import desitarget
from desitarget.internal import sharedmem

def read_mock_durham(core_filename, photo_filename):
    """
    Args:
    core_filename: filename of the hdf5 file storing core lightconedata
    photo_filename: filename of the hdf5 storing photometric data

    Returns:
    objects: ndarray with the structure required to go through desitarget.cuts.select_targets()   
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

def read_tractor(filename, header=False):
    """ 
        Read a tractor catalogue. Always the latest DR. 
        
        Args:
            filename: a file name of one tractor file
            
        Optional:
            header: if true, return (data, header) instead of just data

        Returns:
            ndarray with the tractor schema, uppercase field names.
    """
    ### return fits.getdata(filename, 1)

    #- fitsio prior to 0.9.8rc1 has a bug parsing boolean columns.
    #- LooseVersion doesn't handle rc1 as we want, so also check for 0.9.8xxx
    if LooseVersion(fitsio.__version__) < LooseVersion('0.9.8') and \
        not fitsio.__version__.startswith('0.9.8'):
            print('ERROR: fitsio >0.9.8rc1 required (not {})'.format(\
                    fitsio.__version__))
            raise ImportError

    #- Columns needed for target selection and/or passing forward
    columns = [
        'BRICKID', 'BRICKNAME', 'OBJID', 'BRICK_PRIMARY', 'TYPE',
        'RA', 'RA_IVAR', 'DEC', 'DEC_IVAR',
        'DECAM_FLUX', 'DECAM_MW_TRANSMISSION',
        'DECAM_FRACFLUX', 'DECAM_FLUX_IVAR',
        'WISE_FLUX', 'WISE_MW_TRANSMISSION',
        'SHAPEDEV_R', 'SHAPEEXP_R',
        ]

    #- if header is True, data will be tuple of (data, header) but that is
    #- actually what we want to return in that case anyway
    data = fitsio.read(filename, 1, upper=True, columns=columns, header=header)
    return data

def fix_tractor_dr1_dtype(objects):
    """DR1 tractor files have inconsitent dtype for the TYPE field.  Fix this.
    
    Args:
        objects : numpy structured array from target file
        
    Returns:
        structured array with TYPE.dtype = 'S4' if needed
        
    If the type was already correct, returns the original array
    """
    if objects['TYPE'].dtype == 'S4':
        return objects
    else:
        dt = objects.dtype.descr
        for i in range(len(dt)):
            if dt[i][0] == 'TYPE':
                dt[i] = ('TYPE', 'S4')
                break
        return objects.astype(np.dtype(dt))


def write_targets(filename, data, indir=None):
    """ 
        Write a target catalogue. 
        
        Args:
            filename : output target selection file
            data     : numpy structured array of targets to save

    """
    # FIXME: assert data and tsbits schema

    #- Create header to include versions, etc.
    hdr = fitsio.FITSHDR()
    hdr['DEPNAM00'] = 'desitarget'
    hdr.add_record(dict(name='DEPVER00', value=desitarget.__version__, comment='desitarget.__version__'))
    hdr['DEPNAM01'] = 'desitarget-git'
    hdr.add_record(dict(name='DEPVAL01', value=desitarget.gitversion(), comment='git revision'))
    
    if indir is not None:
        hdr['DEPNAM02'] = 'tractor-files'
        hdr['DEPVER02'] = indir

    fitsio.write(filename, data, extname='TARGETS', header=hdr, clobber=True)

def map_tractor(function, root, bricklist=None, numproc=4, reduce=None):
    """Apply `function` to tractor files in `root`
    
    Args:
        function: python function that takes a tractor filename as input
        root: root directory to scan to find tractor files
        
    Optional:
        bricklist: only process files with bricknames in this list
        numproc: number of parallel processes to use
        
    Returns tuple of 3 lists (bricknames, brickfiles, results):
        bricknames : brick names found in root
        brickfiles : brick files found in root
        results : return values of applying function on each brickfile

    e.g. results[i] = function(brickfiles[i])

    """
    bricknames = list()
    brickfiles = list()
    for bname, filepath in iter_tractor(root):
        if bricklist is None or bname in bricklist:
            bricknames.append(bname)
            brickfiles.append(filepath)
    
    pool = sharedmem.MapReduce(np=numproc)

    with pool:
        results = pool.map(function, brickfiles, reduce=reduce)
        
    return bricknames, brickfiles, results

def iter_tractor(root):
    """ Iterator over all tractor files in a directory.

        Parameters
        ----------
        root : string
            Path to start looking
        
        Returns
        -------
        An iterator of (brickname, filename).

        Examples
        --------
        >>> for brickname, filename in iter_tractor('./'):
        >>>     print(brickname, filename)
        
        Notes
        -----
        root can be a directory or a single file; both create an iterator
    """
    def parse_filename(filename):
        """parse filename to check if this is a tractor brick file;
        returns brickname if it is, otherwise raises ValueError"""
        if not filename.endswith('.fits'): raise ValueError
        #- match filename tractor-0003p027.fits -> brickname 0003p027
        match = re.search('tractor-(\d{4}[pm]\d{3})\.fits', 
                os.path.basename(filename))

        if not match: raise ValueError

        brickname = match.group(1)
        return brickname

    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            for filename in filenames:
                try:
                    brickname = parse_filename(filename)
                    yield brickname, os.path.join(dirpath, filename)
                except ValueError:
                    #- not a brick file but that's ok; keep going
                    pass
    else:
        try:
            brickname = parse_filename(os.path.basename(root))
            yield brickname, root
        except ValueError:
            pass
    
