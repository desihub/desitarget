# ADM This code was used to generate commissioning files for testing.
# ADM All that is required is a link to the CMX_DIR

# SJB prevent import from running this code
if __name__ == "__main__":
    import os
    from time import time
    from glob import glob
    import fitsio

    start = time()

    cmxdir = os.getenv("CMX_DIR")
    fns = glob(os.path.join(cmxdir, "*fits"))
    for fn in fns:
        print("reading {}".format(fn))
        objs = fitsio.read(fn)
        outfile = os.path.join("t3", os.path.basename(fn))
        print("writing {}".format(outfile))
        fitsio.write(outfile, objs[:5], clobber=True)

    print('Done...t={:.2f}s'.format(time()-start))
