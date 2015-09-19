"""
    This file knjow how to write a TS catalogue.

"""

# everybody likes np
import numpy as np 
from astropy.io import fits

def write_ts_catalogue(
        ID, TARGETID, RA, DEC, PRIORITY, NOBS, OBJTYPE,
        output_dir="./", tile_id=0, tile_ra=0.0, tile_dec=0.0):

    dtype = np.dtype([
            ('ID', 'int32'),
            ('TARGETID', 'int32'),
            ('RA', 'float64'),
            ('DEC', 'float64'),
            ('PRIORITY', 'int32'),
            ('NOBS', 'int32'),
            ('OBJTYPE', 'S8')])

    data = np.empty(len(ID), dtype=dtype)
    data['ID'][:] = ID
    data['TARGETID'][:] = TARGETID
    data['RA'][:] = RA
    data['DEC'][:] = DEC
    data['PRIORITY'][:] = PRIORITY
    data['NOBS'][:] = NOBS
    data['OBJTYPE'][:] = OBJTYPE

    fits_filename = "%s/Targets_Tile_%06d.fits"%(output_dir, tile_id)
    with fits.open(fits_filename, mode='ostream') as ff:
        hdu = fits.BinTableHDU(data)
        hdu.header['TILE_ID'] = tile_id
        hdu.header['TILE_RA'] = tile_ra
        hdu.header['TILE_DEC'] = tile_dec
        ff.append(hdu)
        ff.verify() # for what?

def test():
    a = np.arange(10)
    write_ts_catalogue(a, a, a, a, a, a, 'nothing')

if __name__ == '__main__':
    test()

