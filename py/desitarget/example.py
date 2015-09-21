import cuts
import decals
import mock
import os
from astropy.io import fits
import numpy as np
    
def fits_to_bin_example():
    """
    Takes a FITS file for fiber assignment and puts it into Matin White binary format
    """
    type_id = {'ELG':4, 'LRG':3, 'QSO': 2}
    outputdir="/gpfs/data/jeforero/desidata/inputfiber/"
    inputfile = os.path.join(outputdir, 'Targets_Tile_000000.fits')
    fin = fits.open(inputfile)

    ra = fin[1].data['RA']
    dc = fin[1].data['DEC']
    zz  = np.ones(ra.size, dtype='f4')
    types  = fin[1].data['OBJTYPE']


    Nt =  np.array([ra.size],dtype='i4')
    id       = np.zeros(ra.size,dtype='i4')
    pp       = np.zeros(ra.size,dtype='f4')
    no       = np.zeros(ra.size,dtype='i4')
    ra       = np.zeros(ra.size,dtype='f4')
    dc       = np.zeros(ra.size,dtype='f4')

    pp[:] = fin[1].data['PRIORITY']
    no[:] = fin[1].data['NOBS']
    ra[:] = fin[1].data['RA']
    dc[:] = fin[1].data['DEC']

    print ra
    print dc 

    for t in type_id:
        index = np.where(types==t)
        if(np.size(index)):
            id[index] = id[index] + type_id[t]

    icat = 0
    fout = open("%s/Targets_Tile_%05d.rdzipn"%(outputdir, icat),"w")
    Nt.tofile(fout)
    ra.tofile(fout)
    dc.tofile(fout)
    zz.tofile(fout)
    id.tofile(fout)
    pp.tofile(fout)
    no.tofile(fout)
    fout.close()
    return 



def cut_example():
    """
    This example takes a Durham light cone, performs target selection 
    and writes the results in a FITS format for fiber allocation.
    """
    mockfile="/gpfs/data/Lightcone/lightcone_out/LC144/GAL437a/Generic.r25/Gonzalez13.DB.MillGas.field1.photometry.0.hdf5"
    target_id, ra, dec, g_mags, r_mags, z_mags = cuts.load_light_cone_durham(mockfile)
    outputdir="/gpfs/data/jeforero/desidata/inputfiber/"
    cuts.selection_to_fits(target_id, g_mags, r_mags, z_mags, ra, dec, 
                           output_dir=outputdir, 
                           tile_ra=ra.mean(), tile_dec=dec.mean())

def cut_decals_example():
    # find this file from DR1 catalogue on edison:/global/project/projectdirs/cosmo/data/legacysurvey/dr1/tractor/
    tractorfile="tractor-2437p082.fits"
    data = fits.open(tractorfile)[1].data
    with np.errstate(all='ignore'):
        where = np.where(decals.LRG(data))
    print("%d / %d objects are selected." %( len(where), len(data)))
    return where

def cut_mock_example():
    import numpy

    TOFLUX = lambda mag : 10 ** ((22.5 - mag) / 2.5)

    randoms = numpy.random.uniform(size=(4, 1000))
    r = randoms[0] * (24- 16) + 16
    g = randoms[1] * 2.5 - 0.5 + r
    W1 = r - (randoms[2] * 8 - 2)
    W2 = r - (randoms[2] * 8 - 2)
    z = r - (randoms[3] * 3.0 - 0.5)

    GFLUX = TOFLUX(g)
    RFLUX = TOFLUX(r)
    ZFLUX = TOFLUX(z)
    W1FLUX = TOFLUX(W1)
    W2FLUX = TOFLUX(W1)
    
    data = np.rec.fromarrays(
            [GFLUX, RFLUX, W1FLUX, W2FLUX, ZFLUX],
            names=['GFLUX', 'RFLUX', 'W1FLUX', 'W2FLUX', 'ZFLUX'],
            )

    with np.errstate(all='ignore'):
        where = np.where(mock.LRG(data))[0]

    print("%d / %d objects are selected." %( len(where), len(data)))
    return where

cut_decals_example()
cut_mock_example()

#cut_example()
#fits_to_bin_example()


