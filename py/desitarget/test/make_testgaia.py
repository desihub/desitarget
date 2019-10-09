# ADM This code was used to generate Gaia files for testing.
# ADM All that is required is a link to the desitarget GAIA_DIR.

import os
import fitsio
import numpy as np
from time import time
from pkg_resources import resource_filename
from desitarget.gaiamatch import find_gaia_files
from desitarget import io

start = time()

# ADM choose the Gaia files to cover the same object
# ADM locations as the sweeps/tractor files.
datadir = resource_filename('desitarget.test', 't')
tractorfiles = sorted(io.list_tractorfiles(datadir))
sweepfiles = sorted(io.list_sweepfiles(datadir))

# ADM read in each of the relevant Gaia files.
gaiafiles = []
for fn in sweepfiles + tractorfiles:
    objs = fitsio.read(fn, columns=["RA", "DEC"])
    gaiafiles.append(find_gaia_files(objs, neighbors=False))
gaiafiles = np.unique(gaiafiles)

# ADM loop through the Gaia files and write out some rows
# ADM to the "t4" unit test directory.
if not os.path.exists("t4"):
    os.makedirs(os.path.join("t4", "healpix"))
for fn in gaiafiles:
    objs = fitsio.read(fn)
    outfile = os.path.join("t4", "healpix", os.path.basename(fn))
    fitsio.write(outfile, objs[:25], clobber=True)
    print("writing {}".format(outfile))

print('Done...t={:.2f}s'.format(time()-start))
