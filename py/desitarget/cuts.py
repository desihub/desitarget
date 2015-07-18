"""
desitarget.cuts
===================

Utility functions to perform target selection based on 
color cuts on a dataset
"""

from __future__ import absolute_import, division

import numpy as np
import h5py
import sys
import numpy as np
from astropy.io import fits
import os

def select_target(target_id, g_mags, r_mags, z_mags, target_type=""):
    """
    args:
    target_id : 1D integer array of unique target IDs associated to the magnitudes.
    g_mags : 1D magnitudes in the g-band
    r_mags : 1D magnitudes in the r-band
    z_mags : 1D magnitudes in the z-band
    
    returns 1D numpy array of a subset from target_id.
    
    We compute the conditions that correspond to LRG/ELG/QSO/BGS 
    target selection following the criteria described here:
    
    https://desi.lbl.gov/trac/wiki/TargetSelection
    
    NOTE: requirements on WISE bands are not included yet

    """

    target_types = ["ELG", "LRG", "QSO", "BGS"]


    target_true = np.empty(0, dtype='int')
    if(target_type in target_types):
        if(target_type=="LRG"):
            target_true =  np.where((r_mags < 23.0) & (z_mags < 20.56) & ((r_mags-z_mags)>1.6))

        elif (target_type=="ELG"):
            target_true = np.where((r_mags < 23.4) & 
                                   ((r_mags - z_mags)>0.3) & 
                                   ((r_mags - z_mags)<1.5) & 
                                   ((g_mags - r_mags)<(r_mags - z_mags - 0.2)) & 
                                   ((g_mags - r_mags)< 1.2 - (r_mags - z_mags)))

        elif (target_type=="QSO"):
            target_true = np.where((r_mags < 23.0) &
                                   ((g_mags - r_mags) < 1.0) &
                                   ((r_mags - z_mags) > -0.3) &
                                   ((r_mags - z_mags) < 1.1))
        elif (target_type=="BGS"):
                        target_true =  np.where((r_mags < 19.35))

    else:
         raise RuntimeError("Target type %s not recognized"%(target_type))
        


    return target_true

 

def load_light_cone_durham(filename):
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

       



def selection_to_fits(target_id, g_mags, r_mags, z_mags, ra, dec, output_dir="./", tile_id=0, tile_ra=0.0, tile_dec=0.0):
    """
    args:
    target_id : 1D integer array of unique target IDs associated to the magnitudes.
    g_mags : 1D magnitudes in the g-band
    r_mags : 1D magnitudes in the r-band
    z_mags : 1D magnitudes in the z-band
    ra: 1D Right Ascension for all galaxies
    dec: 1D declination for all galaxies
    output_dir (Optional[str]): output directory. Defaults to "./" 
    tile_id (Optional[int]): Tile identification number. Defaults to 0
    tile_ra (Optioanl[float]): Nominal RA center of the targets. Defaults to 0.0
    tile_dec (Optioanl[float]): Nominal dec center of the targets. Defaults to 0.0
        
    returns:
    Does not return any value.
    
    Notes:
    Makes the target selection with the limitations in desitarget.cuts.select_target
    """

    types_to_extract = ['ELG', 'LRG']
    priority = {'ELG': 4, 'LRG': 3, 'QSO': 1}
    nobs = {'ELG':1, 'LRG':2, 'QSO': 2}
    
    
    target_file_id = np.empty((0), dtype='int')
    target_db_id = np.empty((0), dtype='int')
    target_ra = np.empty((0))
    target_dec = np.empty((0))
    target_priority = np.empty((0), dtype='int')
    target_nobs = np.empty((0), dtype='int')
    target_types = np.empty((0))
        

    n_total = 0
    for target_type in types_to_extract:
        print target_type
        target_true = select_target(target_id, g_mags, r_mags, z_mags, target_type=target_type)
        n_target = np.size(target_true)
        n_total = n_total + n_target
        if(n_target>0):            
            print n_target
            target_file_id = np.append(target_file_id, np.int_(target_id[target_true]))
            target_db_id = np.append(target_db_id, np.int_(target_id[target_true]))
            target_ra = np.append(target_ra, ra[target_true])
            target_dec = np.append(target_dec, dec[target_true])
            target_priority = np.append(target_priority, np.int_(priority[target_type]*np.ones(n_target)))
            target_nobs = np.append(target_nobs, np.int_(nobs[target_type]*np.ones(n_target)))
            tmp_type = np.chararray(n_target, itemsize=8)
            tmp_type[:] = target_type
            target_types = np.append(target_types, tmp_type)    
    
    if(n_total>0):
        fits_filename = "%s/Targets_Tile_%06d.fits"%(output_dir, tile_id)
        if(os.path.isfile(fits_filename)):
            os.remove(fits_filename)

        c0=fits.Column(name='ID', format='K', array=target_file_id)
        c1=fits.Column(name='TARGETID', format='K', array=target_db_id)
        c2=fits.Column(name='RA', format='D', array=target_ra)
        c3=fits.Column(name='DEC', format='D', array=target_dec)
        c4=fits.Column(name='PRIORITY', format='D', array=target_priority)
        c5=fits.Column(name='NOBS', format='D', array=target_nobs)
        c6=fits.Column(name='OBJTYPE', format='8A', array=target_types)
        
        targetcat=fits.ColDefs([c0,c1,c2,c3,c4,c5,c6])
        
        table_targetcat_hdu=fits.TableHDU.from_columns(targetcat)
        table_targetcat_hdu.header['TILE_ID'] = tile_id
        table_targetcat_hdu.header['TILE_RA'] = tile_ra
        table_targetcat_hdu.header['TILE_DEC'] = tile_dec
        
        hdu=fits.PrimaryHDU()
        hdulist=fits.HDUList([hdu])
        hdulist.append(table_targetcat_hdu)
        hdulist.verify()
        hdulist.writeto(fits_filename)        

    return


    
