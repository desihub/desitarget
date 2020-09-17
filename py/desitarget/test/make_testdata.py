# - For the record (and future updates):
# ADM This code generates tractor, sweep, targets, pixweight, mask
# ADM file subsets for testing.
# - The hardcoded paths are for NERSC, but you can swap out any
# - legacy survey data release path as needed.
# ADM Now (10/04/19) based off DR8 sweeps and Tractor files.

# SJB prevent import from running this code
if __name__ == "__main__":
    import fitsio
    import numpy as np
    import numpy.lib.recfunctions as rfn
    import healpy as hp
    from os.path import basename
    from time import time
    # from astropy.io import fits
    from desitarget.cuts import apply_cuts
    from desitarget.cmx import cmx_cuts
    from desitarget.io import read_tractor
    from desitarget.targets import finalize
    from desitarget.QA import _load_systematics
    # from desitarget.gaiamatch import find_gaia_files

    start = time()
    tractordir = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr8/south/tractor/330/'
    # tractordir = '/project/projectdirs/cosmo/data/legacysurvey/dr7/tractor/330/'
    # tractordir = '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/tractor/330'
    # tractordir = '/data/legacysurvey/dr3.1/tractor/330/'
    for brick in ['3301m002', '3301m007', '3303p000']:
        filepath = '{}/tractor-{}.fits'.format(tractordir, brick)
        desi_target, bgs_target, mws_target = apply_cuts(filepath)
        # ADM as nobody is testing the MWS in the sandbox, yet, we need to
        # ADM ensure we ignore MWS targets for testing the main algorithms.
        yes = np.where((desi_target != 0) & (mws_target == 0))[0]
        no = np.where(desi_target == 0)[0]
        keep = np.concatenate([yes[0:3], no[0:3]])
        data, hdr = read_tractor(filepath, header=True)

        # ADM the FRACDEV and FRACDEV_IVAR columns can
        # ADM contain some NaNs, which break testing.
        wnan = np.where(data["FRACDEV"] != data["FRACDEV"])
        if len(wnan[0]) > 0:
            data["FRACDEV"][wnan] = 0.
        wnan = np.where(data["FRACDEV_IVAR"] != data["FRACDEV_IVAR"])
        if len(wnan[0]) > 0:
            data["FRACDEV_IVAR"][wnan] = 0.

        # ADM the "CONTINUE" comment keyword is not yet implemented
        # ADM in fitsio, so delete it to prevent fitsio barfing on headers.
        hdr.delete("CONTINUE")
        fitsio.write('t/'+basename(filepath), data[keep], header=hdr, clobber=True)
        print('made Tractor file for brick {}...t={:.2f}s'.format(brick, time()-start))

    sweepdir = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr8/south/sweep/8.0/'
    # sweepdir = '/project/projectdirs/cosmo/data/legacysurvey/dr7/sweep/7.1/'
    # sweepdir = '/project/projectdirs/cosmo/data/legacysurvey/dr3.1/sweep/3.1'
    # sweepdir = '/data/legacysurvey/dr2p/sweep/'
    for radec in ['310m005-320p000', '320m005-330p000', '330m005-340p000']:
        filepath = '{}/sweep-{}.fits'.format(sweepdir, radec)
        desi_target, bgs_target, mws_target = apply_cuts(filepath)
        cmx_target = cmx_cuts.apply_cuts(filepath)

        # ADM as nobody is testing the MWS in the sandbox, yet, we need to.
        # ADM ensure we ignore MWS targets for testing the main algorithms.
        yes = np.where((desi_target != 0) & (mws_target == 0))[0]
        no = np.where(desi_target == 0)[0]
        keep = np.concatenate([yes[0:3], no[0:3]])
        data, hdr = read_tractor(filepath, header=True)

        # ADM the "CONTINUE" comment keyword is not yet implemented
        # ADM in fitsio, so delete it to prevent fitsio barfing on headers.
        hdr.delete("CONTINUE")

        fitsio.write('t/'+basename(filepath), data[keep], header=hdr, clobber=True)

        print('made sweeps file for range {}...t={:.2f}s'.format(radec, time()-start))

    # ADM only need to write out one set of targets. So fine outside of loop.
    # ADM create a targets file for testing QA (main survey and commissioning)
    # ADM we get more test coverage if one file has > 1000 targets.
    many = yes[:1001]
    targets = finalize(data[many], desi_target[many],
                       bgs_target[many], mws_target[many])
    cmx_targets = finalize(data[keep], desi_target[keep],
                           bgs_target[keep], mws_target[keep], survey='cmx')
    # ADM remove some columns from the target file that aren't needed for
    # ADM testing. It's a big file.
    needtargs = np.empty(
        len(many), dtype=[('RA', '>f8'), ('DEC', '>f8'), ('RELEASE', '>i2'),
                          ('FLUX_G', '>f4'), ('FLUX_R', '>f4'), ('FLUX_Z', '>f4'),
                          ('FLUX_W1', '>f4'), ('FLUX_W2', '>f4'), ('MW_TRANSMISSION_G', '>f4'),
                          ('MW_TRANSMISSION_R', '>f4'), ('MW_TRANSMISSION_Z', '>f4'),
                          ('MW_TRANSMISSION_W1', '>f4'), ('MW_TRANSMISSION_W2', '>f4'),
                          ('PARALLAX', '>f4'), ('PMRA', '>f4'), ('PMDEC', '>f4'),
                          ('DESI_TARGET', '<i8'), ('BGS_TARGET', '<i8'), ('MWS_TARGET', '<i8')]
    )
    for col in needtargs.dtype.names:
        needtargs[col] = targets[col]
    fitsio.write('t/targets.fits', needtargs, extname='TARGETS', header=hdr, clobber=True)
    fitsio.write('t/cmx-targets.fits', cmx_targets, extname='TARGETS', header=hdr, clobber=True)

    # ADM as of DR7, ignore the Gaia files
    # ADM adding Gaia files to which to match
    # for brick in ['3301m002', '3301m007', '3303p000']:
    #    filepath = '{}/tractor-{}.fits'.format(tractordir, brick)
    #    data = fitsio.read('t/'+basename(filepath))
    #    # ADM use find_gaia_files to determine which Gaia files potentially
    #    # ADM match the sweeps objects of interest
    #    for gaiafile in find_gaia_files(data):
    #        # ADM for each of the relevant Gaia files, read the first 5 rows
    #        gaiadata = fitsio.read(gaiafile, rows=range(5))
    #        # ADM and write them to a special Gaia directory
    #        fitsio.write('tgaia/'+basename(gaiafile), gaiadata, clobber=True)

    # for radec in ['310m005-320p000', '320m005-330p000', '330m005-340p000']:
    #    filepath = '{}/sweep-{}.fits'.format(sweepdir, radec)
    #    data = fitsio.read('t/'+basename(filepath))
    #    # ADM use find_gaia_files to determine which Gaia files potentially
    #    # ADM match the sweeps objects of interest
    #    for gaiafile in find_gaia_files(data):
    #        # ADM for each of the relevant Gaia files, read the first 5 rows
    #        gaiadata = fitsio.read(gaiafile, rows=range(5))
    #        # ADM and write them to a special Gaia directory
    #        fitsio.write('tgaia/'+basename(gaiafile), gaiadata, clobber=True)

    # ADM adding a file to make a mask for bright stars
    # ADM this should go in its own directory /t2 (others are in t1)
    # ADM post version 0.40.0 of desitarget, masking uses Gaia not
    # ADM the sweeps, so this has been supplanted by make_testgaia.py.
    # filepath = '{}/sweep-{}.fits'.format(sweepdir, '190m005-200p000')
    # data, hdr = read_tractor(filepath, header=True)
    # ADM the "CONTINUE" comment keyword is not yet implemented
    # ADM in fitsio, so delete it to prevent fitsio barfing on headers
    # hdr.delete("CONTINUE")
    # keep = np.where(data["FLUX_Z"] > 100000)
    # fitsio.write('t2/'+basename(filepath), data[keep], header=hdr, clobber=True)

    # ADM adding a fake pixel weight map
    sysdic = _load_systematics()
    npix = hp.nside2npix(2)
    pixmap = np.ones(npix, dtype=[(k, '>f4') for k in sysdic.keys()])
    pixmap = rfn.append_fields(pixmap, "ALL", np.ones(npix), dtypes='>f4')
    fitsio.write('t/pixweight.fits', pixmap, clobber=True)

    print('Done...t={:.2f}s'.format(time()-start))
