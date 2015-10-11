"""
    This file knjow how to write a TS catalogue.

"""

# everybody likes np
import numpy as np 
from astropy.io import fits

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

    try:
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
    except Exception, e:
        import traceback
        print 'ERROR in loadlightconedurham'
        traceback.print_exc()
        raise e
    return target_id, ra, dec, g_mags, r_mags, z_mags

       
def read_tractor(filename):
    """ Read a tractor catalogue. Always the latest DR. 

        Notes
        -----
        If other DR is needed. We shall add functions like:
            read_tractor_dr1

    """
    data = fits.open(filename)[1].data.copy()
    # FIXME: assert the schema is sufficient.
    return data

def write_targets(filename, data, tsbits):
    # FIXME: assert data and tsbits schema
    #
    # add a column name
    width = 1 if len(tsbits.shape) == 1 else tsbits.shape[-1]
    tsbits = tsbits.view(dtype=[('TSBITS', (tsbits.dtype, width))])

    with fits.open(filename, mode='ostream') as ff:
        # OK we will follow the rules at
        # https://pythonhosted.org/pyfits/users_guide/users_table.html
        # to merge two tables.
        #
        # This does look like a workaround of lacking in numpy api.
        hdu1 = fits.BinTableHDU(data)
        hdu2 = fits.BinTableHDU(tsbits)

        columns = hdu1.columns + hdu2.columns

        hdu = fits.BinTableHDU.from_columns(columns)
        ff.append(hdu)
        ff.verify() # for what?

