import cuts


def cut_example():
    mockfile="/gpfs/data/Lightcone/lightcone_out/LC144/GAL437a/Generic.r25/Gonzalez13.DB.MillGas.field1.photometry.0.hdf5"
    target_id, ra, dec, g_mags, r_mags, z_mags = cuts.load_light_cone_durham(mockfile)
    outputdir="/gpfs/data/jeforero/desidata/inputfiber/"
    cuts.selection_to_fits(target_id, g_mags, r_mags, z_mags, ra, dec, 
                           output_dir=outputdir, 
                           tile_ra=ra.mean(), tile_dec=dec.mean())
    
    
