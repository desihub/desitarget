# ADM This code was used to generate Gaia files for testing.
# ADM All that is required is a link to the desitarget GAIA_DIR.

# SJB prevent import from running this code
if __name__ == "__main__":
    import os
    import fitsio
    import numpy as np
    from time import time
    from pkg_resources import resource_filename
    from desitarget.gaiamatch import find_gaia_files
    from desitarget.tychomatch import find_tycho_files
    from desitarget.uratmatch import find_urat_files
    from desitarget import io

    start = time()

    # ADM choose the Gaia files to cover the same object
    # ADM locations as the sweeps/tractor files.
    datadir = resource_filename('desitarget.test', 't')
    tractorfiles = sorted(io.list_tractorfiles(datadir))
    sweepfiles = sorted(io.list_sweepfiles(datadir))

    # ADM read in relevant Gaia files.
    gaiafiles = []
    for fn in sweepfiles + tractorfiles:
        objs = fitsio.read(fn, columns=["RA", "DEC"])
        gaiafiles.append(find_gaia_files(objs, neighbors=False))
    gaiafiles = np.unique(np.concatenate(gaiafiles))

    # ADM loop through the Gaia files and write out some rows
    # ADM to the "t4" unit test directory.
    tychofiles, uratfiles = [], []
    if not os.path.exists("t4"):
        os.makedirs(os.path.join("t4", "healpix"))
    for fn in gaiafiles:
        objs, hdr = fitsio.read(fn, 1, header=True)
        outfile = os.path.join("t4", "healpix", os.path.basename(fn))
        fitsio.write(outfile, objs[:25], header=hdr, clobber=True, extname="GAIAHPX")
        # ADM find some Tycho and URAT files that accompany the Gaia files.
        tychofiles.append(find_tycho_files(objs[:25], neighbors=False))
        uratfiles.append(find_urat_files(objs[:25], neighbors=False))
        print("writing {}".format(outfile))
    tychofiles = np.unique(np.concatenate(tychofiles))
    uratfiles = np.unique(np.concatenate(uratfiles))

    # ADM loop through the Gaia files and write out accompanying Tycho
    # ADM and URAT objects.
    for direc, fns, ext in zip(["tycho", "urat"],
                               [tychofiles, uratfiles],
                               ["TYCHOHPX", "URATHPX"]):
        outdir = os.path.join("t4", direc)
        if not os.path.exists(outdir):
            os.makedirs(os.path.join("t4", direc, "healpix"))
        for fn in fns:
            objs, hdr = fitsio.read(fn, 1, header=True)
            outfile = os.path.join("t4", direc, "healpix", os.path.basename(fn))
            s = set(gaiafiles)
            # ADM ensure a match with objects in the Gaia files.
            ii = np.array(
                [len(set(find_gaia_files(i, neighbors=False)).intersection(s))>0
                 for i in objs])
            fitsio.write(outfile, objs[ii][:25],
                         clobber=True, header=hdr, extname=ext)
            print("writing {}".format(outfile))

    print('Done...t={:.2f}s'.format(time()-start))
