# Much of this code is from version dr6.rc1.2 of legacypipe:
# https://github.com/legacysurvey/legacypipe
# The files from which this code was derived
# (legacypipe/survey.py) and (legacypipe/utils.py)
# are licensed under the BSD-3 license as of
# version dr6.rc1.2 of legacypipe (git hash e8a188a).
"""
=======================================
desitarget.skyutilities.legacypipe.util
=======================================

Module so desitarget sky fiber code doesn't need explicit legacypipe dependencies
"""
import numpy as np
import os
from desitarget.skyutilities.astrometry.fits import fits_table

def get_rgb(imgs, bands, mnmx=None, arcsinh=None):
    '''Given a list of images in the given bands, returns a scaled RGB image.

    Parameters
    ----------
    imgs : :class:`list`
        List of numpy arrays, all the same size, in nanomaggies.
    bands : :class:`list`
        List of strings, *e.g.*, ``['g','r','z']``.
    mnmx : :func:`tuple`
        (min,max), values that will become black/white *after* scaling.
        Default is (-3,10).
    arcsinh : :class:`float`, optional
        Use nonlinear scaling as in SDSS.

    Returns
    -------
    array-like
        (H,W,3) numpy array with values between 0 and 1.
    '''
    bands = ''.join(bands)

    scales = dict(g = (2, 0.0066),
                  r = (1, 0.01),
                  z = (0, 0.025),
                      )

    h,w = imgs[0].shape
    rgb = np.zeros((h,w,3), np.float32)
    for im,band in zip(imgs, bands):
        if not band in scales:
            print('Warning: band', band, 'not used in creating RGB image')
            continue
        plane,scale = scales.get(band, (0,1.))
        # print('RGB: band', band, 'in plane', plane, 'scaled by', scale)
        rgb[:,:,plane] = (im / scale).astype(np.float32)

    if mnmx is None:
        mn,mx = -3, 10
    else:
        mn,mx = mnmx

    if arcsinh is not None:
        def nlmap(x):
            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)
        rgb = nlmap(rgb)
        mn = nlmap(mn)
        mx = nlmap(mx)

    rgb = (rgb - mn) / (mx - mn)

    return rgb


def _ring_unique(wcs, W, H, i, unique, ra1,ra2,dec1,dec2):
    lo, hix, hiy = i, W-i-1, H-i-1
    # one slice per side; we double-count the last pix of each side.
    sidex = slice(lo,hix+1)
    sidey = slice(lo,hiy+1)
    top = (lo, sidex)
    bot = (hiy, sidex)
    left  = (sidey, lo)
    right = (sidey, hix)
    xx = np.arange(W)
    yy = np.arange(H)
    nu,ntot = 0,0
    for slc in [top, bot, left, right]:
        #print('xx,yy', xx[slc], yy[slc])
        (yslc,xslc) = slc
        #ADM the one change to the legacypipe "borrowed" code, as in
        #ADM desitarget we use astropy in place of astrometry.net
        #ADM start at pixel 0,0 in the top-left (the numpy standard)
        rr, dd = wcs.all_pix2world(xx[xslc], yy[yslc], 0)
#        rr,dd = wcs.pixelxy2radec(xx[xslc]+1, yy[yslc]+1)
        U = (rr >= ra1 ) * (rr < ra2 ) * (dd >= dec1) * (dd < dec2)
        #print('Pixel', i, ':', np.sum(U), 'of', len(U), 'pixels are unique')
        unique[slc] = U
        nu += np.sum(U)
        ntot += len(U)
    #if allin:
    #    print('Scanned to pixel', i)
    #    break
    return nu,ntot

def find_unique_pixels(wcs, W, H, unique, ra1,ra2,dec1,dec2):
    if unique is None:
        unique = np.ones((H,W), bool)
    # scan the outer annulus of pixels, and shrink in until all pixels
    # are unique.
    step = 10
    for i in range(0, W//2, step):
        nu,ntot = _ring_unique(wcs, W, H, i, unique, ra1,ra2,dec1,dec2)
        #print('Pixel', i, ': nu/ntot', nu, ntot)
        if nu > 0:
            i -= step
            break
        unique[:i,:] = False
        unique[H-1-i:,:] = False
        unique[:,:i] = False
        unique[:,W-1-i:] = False

    for j in range(max(i+1, 0), W//2):
        nu,ntot = _ring_unique(wcs, W, H, j, unique, ra1,ra2,dec1,dec2)
        #print('Pixel', j, ': nu/ntot', nu, ntot)
        if nu == ntot:
            break
    return unique



class LegacySurveyData(object):
    '''
    A class describing the contents of a LEGACY_SURVEY_DIR directory --
    tables of CCDs and of bricks, and calibration data.  Methods for
    dealing with the CCDs and bricks tables.

    This class is also responsible for creating LegacySurveyImage
    objects (eg, DecamImage objects), which then allow data to be read
    from disk.
    '''

    def __init__(self, survey_dir=None):
        '''Create a LegacySurveyData object using data from the given
        *survey_dir* directory, or from the $LEGACY_SURVEY_DIR environment
        variable.

        Parameters
        ----------
        survey_dir : string
            Where to look for files including calibration files, tables of CCDs and bricks,
            image data, etc.
        '''
        from desiutil.log import get_logger
        from collections import OrderedDict

        if survey_dir is None:
            log.info('survey_dir not passed...using pwd as survey_dir, but this is likely to fail')
            survey_dir = os.getcwd()

        self.survey_dir = survey_dir
        self.cache_dir = None
        self.output_dir = '.'

        self.output_file_hashes = OrderedDict()
        self.ccds = None
        self.bricks = None

        # Create and cache a kd-tree for bricks_touching_radec_box ?
        self.cache_tree = False
        self.bricktree = None
        ### HACK! Hard-coded brick edge size, in degrees!
        self.bricksize = 0.25

        self.allbands = 'ugrizY'

        self.version = None

        # Filename prefix for coadd files
        self.file_prefix = 'legacysurvey'

    def __str__(self):
        return ('%s: dir %s, out %s' %
                (type(self).__name__, self.survey_dir, self.output_dir))

    def find_file(self, filetype, brick=None, brickpre=None, band='%(band)s',
                  output=False):
        '''
        Returns the filename of a Legacy Survey file.

        *filetype* : string, type of file to find, including:
             "tractor" -- Tractor catalogs
             "depth"   -- PSF depth maps
             "galdepth" -- Canonical galaxy depth maps
             "nexp" -- number-of-exposure maps

        *brick* : string, brick name such as "0001p000"

        *output*: True if we are about to write this file; will use self.outdir as
        the base directory rather than self.survey_dir.

        Returns: path to the specified file (whether or not it exists).
        '''
        from glob import glob

        if brick is None:
            brick = '%(brick)s'
            brickpre = '%(brick).3s'
        else:
            brickpre = brick[:3]

        if output:
            basedir = self.output_dir
        else:
            basedir = self.survey_dir
            # ADM if you can't find the file, it's one
            # ADM directory up. Assuming this is dr8+.
            basedir = basedir.rstrip('/')
            check = os.path.basename(os.path.dirname(basedir))
            if check[:2] == 'dr' and int(check[-1]) >= 8:
                truebasedir = os.path.dirname(basedir)
            else:
                truebasedir = basedir

        if brick is not None:
            codir = os.path.join(basedir, 'coadd', brickpre, brick)

        # Swap in files in the self.cache_dir, if they exist.
        def swap(fn):
            if output:
                return fn
            return self.check_cache(fn)
        def swaplist(fns):
            if output or self.cache_dir is None:
                return fns
            return [self.check_cache(fn) for fn in fns]

        sname = self.file_prefix

        if filetype == 'bricks':
            fn = 'survey-bricks.fits.gz'
            return swap(os.path.join(truebasedir, fn))

        elif filetype == 'ccds':
            if self.version in ['dr1','dr2']:
                return swaplist([os.path.join(basedir, 'decals-ccds.fits.gz')])
            else:
                return swaplist(
                    glob(os.path.join(truebasedir, 'survey-ccds-*.fits.gz')))

        elif filetype == 'ccd-kds':
            return swaplist(
                glob(os.path.join(truebasedir, 'survey-ccds-*.kd.fits')))

        elif filetype == 'tycho2':
            return swap(os.path.join(basedir, 'tycho2.fits.gz'))

        elif filetype == 'annotated-ccds':
            if self.version == 'dr2':
                return swaplist(
                    glob(os.path.join(basedir, 'decals-ccds-annotated.fits')))
            return swaplist(
                glob(os.path.join(truebasedir, 'ccds-annotated-*.fits.gz')))

        elif filetype == 'tractor':
            return swap(os.path.join(basedir, 'tractor', brickpre,
                                     'tractor-%s.fits' % brick))

        elif filetype == 'tractor-intermediate':
            return swap(os.path.join(basedir, 'tractor-i', brickpre,
                                     'tractor-%s.fits' % brick))

        elif filetype == 'galaxy-sims':
            return swap(os.path.join(basedir, 'tractor', brickpre,
                                     'galaxy-sims-%s.fits' % brick))

        elif filetype in ['ccds-table', 'depth-table']:
            ty = filetype.split('-')[0]
            return swap(
                os.path.join(codir, '%s-%s-%s.fits' % (sname, brick, ty)))

        elif filetype in ['image-jpeg', 'model-jpeg', 'resid-jpeg',
                          'imageblob-jpeg', 'simscoadd-jpeg','imagecoadd-jpeg']:
            ty = filetype.split('-')[0]
            return swap(
                os.path.join(codir, '%s-%s-%s.jpg' % (sname, brick, ty)))

        elif filetype in ['invvar', 'chi2', 'image', 'model', 'depth', 'galdepth', 'nexp']:
            return swap(os.path.join(codir, '%s-%s-%s-%s.fits.fz' %
                                     (sname, brick, filetype,band)))

        elif filetype in ['blobmap']:
            return swap(os.path.join(basedir, 'metrics', brickpre,
                                     'blobs-%s.fits.gz' % (brick)))

        elif filetype in ['skies']:
            return swap(os.path.join(basedir, 'metrics', brickpre,
                                     'skies-%s.fits.gz' % (brick)))

        elif filetype in ['all-models']:
            return swap(os.path.join(basedir, 'metrics', brickpre,
                                     'all-models-%s.fits' % (brick)))

        elif filetype == 'checksums':
            return swap(os.path.join(basedir, 'tractor', brickpre,
                                     'brick-%s.sha256sum' % brick))

        print('Unknown filetype "%s"' % filetype)
        assert(False)

    def check_cache(self, fn):
        if self.cache_dir is None:
            return fn
        cfn = fn.replace(self.survey_dir, self.cache_dir)
        if os.path.exists(cfn):
            return cfn
        return fn

    def get_bricks(self):
        '''
        Returns a table of bricks.  The caller owns the table.

        For read-only purposes, see *get_bricks_readonly()*, which
        uses a cached version.
        '''
        return fits_table(self.find_file('bricks'))

    def get_bricks_readonly(self):
        '''
        Returns a read-only (shared) copy of the table of bricks.
        '''
        if self.bricks is None:
            self.bricks = self.get_bricks()
            # Assert that bricks are the sizes we think they are.
            # ... except for the two poles, which are half-sized
            assert(np.all(np.abs((self.bricks.dec2 - self.bricks.dec1)[1:-1] -
                                 self.bricksize) < 1e-3))
        return self.bricks

    def get_brick_by_name(self, brickname):
        '''
        Returns a brick (as one row in a table) by name (string).
        '''
        B = self.get_bricks_readonly()
        I, = np.nonzero(np.array([n == brickname for n in B.brickname]))
        if len(I) == 0:
            return None
        return B[I[0]]
