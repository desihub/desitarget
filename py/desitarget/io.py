"""
    This file knows how to write a TS catalogue.

"""

from desitarget import cuts

# everybody likes np
import numpy as np 
import numpy.lib.recfunctions
from astropy.io import fits
import fitsio
import os, re
from distutils.version import LooseVersion

def read_mock_durham(filename):
    """
    Args:
    filename: filename of the hdf5 file storing lightconedata.
    
    Returns:
    target_id: 1D numpy array, array of unique target IDs associated to the magnitudes. 
    ra : 1D numpy array, Right Ascension
    dec: 1D numpy array, declination
    g_mags: 1D numpy array, magnitudes in g band.
    r_mags: 1D numpy array, magnitudes in g band.
    z_mags: 1D numpy array, magnitudes in g band.
    returns 1D numpy array of a subset from target_id.
    
    """
    raise NotImplementedError("Read and convert a Durham mock to DECALS schema")

    fin = h5py.File(filename, "r") 
    data = fin.require_group('/Data') 
    ra = data['ra'].value                                                                                    
    dec = data['dec'].value                                                                                  
    gal_id_string = data['GalaxyID'].value # these are string values, not integers!                               
    g_mags = data['appDgo_tot_ext'].value                                                                        
    r_mags = data['appDro_tot_ext'].value                                                                        
    z_mags = data['appDzo_tot_ext'].value  
    n_gals = 0
    n_gals = ra.size
    target_id = np.arange(n_gals)

def read_tractor(filename):
    """ 
        Read a tractor catalogue. Always the latest DR. 
        
        Args:
            filename: a file name of one tractor file

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

    columns = [
        'BRICKID', 'BRICKNAME', 'OBJID', 'BRICK_PRIMARY', 'TYPE',
        'RA', 'RA_IVAR', 'DEC', 'DEC_IVAR',
        'DECAM_FLUX', 'DECAM_MW_TRANSMISSION',
        'WISE_FLUX', 'WISE_MW_TRANSMISSION',
        'SHAPEDEV_R', 'SHAPEEXP_R',
        ]
    data = fitsio.read(filename, 1, upper=True, columns=columns)
    return data

def fix_tractor_dr1_dtype(objects):
    """DR1 tractor files have inconsitent dtype for the TYPE field.  Fix this.
    
    Args:
        objects : numpy structured array from target file
        
    Returns:
        structured array with TYPE.dtype = '|S4' if needed
        
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


def write_targets(filename, data):
    """ 
        Write a target catalogue. 
        
        Args:
            filename : a file name of one tractor file.
                 File contains the original tractor catalogue fields,
                 with target selection mask added as 'TSBITS'

            data     : numpy structured array of targets to save

        Notes:
            This function raises an exception if the file already exists.
            Depending on the use case, this function may need to be split
            into two functions: one appends the column, another writes the
            fits file.
        
    """
    # FIXME: assert data and tsbits schema

    fitsio.write(filename, data, extname='TARGETS', clobber=True)

def map_tractor(function, root, bricklist=None, nproc=4):
    import multiprocessing as mp
    brickfiles = list()
    for brickname, filepath in iter_tractor(root):
        if bricklist is None or brickname in bricklist:
            brickfiles.append(filepath)
                        
    pool = mp.Pool(nproc)
    results = pool.map(function, brickfiles)
    return results

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
    
