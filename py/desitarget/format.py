"""
desitarget.format
===================

Utility functions to write/read files in the suggested FITS format.

See DocDB-1029 https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=1049
"""
import numpy as np
import cuts 
import h5py
import sys
import numpy as np
from astropy.io import fits
import os


def selection_to_fits(target_id, g_mags, r_mags, z_mags, output_dir="./", tile_id=0, tile_ra=0.0, tile_dec=0.0):
    """
    args:
    target_id : 1D integer array of unique target IDs associated to the magnitudes.
    g_mags : 1D magnitudes in the g-band
    r_mags : 1D magnitudes in the r-band
    z_mags : 1D magnitudes in the z-band
    output_dir (Optional[str]): output directory. Defaults to "./" 
    tile_id (Optional[int]): Tile identification number. Defaults to 0
    tile_ra (Optioanl[float]): Nominal RA center of the targets. Defaults to 0.0
    tile_dec (Optioanl[float]): Nominal dec center of the targets. Defaults to 0.0
        
    returns:
    Does not return any value.
    
    Notes:
    Makes the target selection with the limitations in desitarget.cuts.select_target
    """

    possible_types = ['ELG', 'LRG', 'QSO']
    priority = {'ELG': 4, 'LRG': 3, 'QSO': 1}
    nobs = {'ELG':1, 'LRG':2, 'QSO': 2}
    
    
    target_id = np.empty((0), dtype='int')
    target_db_id = np.empty((0), dtype='int')
    target_ra = np.empty((0))
    target_dec = np.empty((0))
    target_priority = np.empty((0), dtype='int')
    target_nobs = np.empty((0), dtype='int')
    target_types = np.empty((0))
        

    n_total = 0
    for target_type in possible_types:
        target_true = cuts.select_target(target_id, g_mags, r_mags, z_mags, target_type=target_type)
        n_target = np.size(target_true)
        n_total = n_total + n_target
        if(n_target>0):            
            target_db_id = np.append(target_db_id, np.int_(gal_id[lrg_true]))
            target_ra = np.append(target_ra, ra_data[lrg_true])
            target_dec = np.append(target_dec, dec_data[lrg_true])
            target_priority = np.append(target_priority, np.int_(priority[target_type]*np.ones(n_lrg)))
            target_nobs = np.append(target_nobs, np.int_(nobs[target_type]*np.ones(n_lrg)))
            tmp_type = np.chararray(n_lrg, itemsize=8)
            tmp_type[:] = target_type
            target_types = np.append(target_types, tmp_type)    
    
    if(n_total>0):
        filename = "%s/Targets_Tile_%06d.fits"%(output_dir, tile_id)

        c0=fits.Column(name='ID', format='K', array=target_id)
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
