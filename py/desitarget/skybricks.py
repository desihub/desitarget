# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.skybricks
====================

Dynamic lookup of whether a given RA,Dec location is a good place to
put a sky fiber.
"""
import os
import numpy as np

class Skybricks(object):
    '''
    This class handles dynamic lookup of whether a given (RA,Dec)
    should make a good location for a sky fiber.
    '''    
    def __init__(self, skybricks_dir=None):
        '''
        Create a Skybricks object, reading metadata.

        Parameters
        ----------
        skybricks_dir : :class:`str`
            The directory to find skybricks data files; if None, will read from
            SKYBRICKS_DIR environment variable.
        '''
        import fitsio
        if skybricks_dir is None:
            skybricks_dir = os.environ.get('SKYBRICKS_DIR', None)
        if skybricks_dir is None:
            raise RuntimeError('Environment variable SKYBRICKS_DIR is not set; '
                               'needed to look up dynamic sky fiber positions')
        self.skybricks_dir = skybricks_dir
        skybricks_fn = os.path.join(self.skybricks_dir, 'skybricks-exist.fits')
        self.skybricks = fitsio.read(skybricks_fn, upper=True)
        self.skykd = _radec2kd(self.skybricks['RA'], self.skybricks['DEC'])

    def lookup_tile(self, tilera, tiledec, tileradius, ras, decs):
        '''
        Looks up a set of RA,Dec positions that are all within a given
        tile (disk on the sky).

        Parameters
        ----------
        tilera : :class:`float`
            tile center RA (deg)
        tiledec : :class:`float`
            tile center Dec (deg)
        tileradius : :class:`float`
            tile radius (deg)
        ras : :class:`~numpy.ndarray`
            array of RA locations to look up
        decs : :class:`~numpy.ndarray`
            array of Dec locations to look up

        Returns
        -------
        :class:`~numpy.array`
            Boolean array, same length as *ras*/*decs* inputs, of `good_sky` values.
            (True = good place to put a sky fiber)
        '''
        import fitsio
        from astropy.wcs import WCS
        from desiutil.log import get_logger
        log = get_logger()

        # skybricks are 1 x 1 deg.
        brickrad = (1. * np.sqrt(2.) / 2.)
        searchrad = 1.01 * (tileradius + brickrad)
        # here, convert search radius to radians -- an overestimate vs
        # unit-sphere distance, but that's the safe direction.
        searchrad = np.deg2rad(searchrad)
        tilexyz = _radec2xyz([tilera], [tiledec])
        sky_inds = self.skykd.query_ball_point(tilexyz[0,:], searchrad)
        # handle non-array iterables (eg lists) as inputs
        ras = np.array(ras)
        decs = np.array(decs)
        good_sky = np.zeros(ras.shape, bool)
        # Check possibly-overlapping skybricks.
        for i in sky_inds:
            # Do any of the query points overlap in the brick's RA,DEC unique-area bounding-box?
            I = np.flatnonzero(
                (ras  >= self.skybricks['RA1'][i]) *
                (ras  <  self.skybricks['RA2'][i]) *
                (decs >= self.skybricks['DEC1'][i]) *
                (decs <  self.skybricks['DEC2'][i]))
            log.debug('Skybricks: %i locations overlap skybrick %s' % (len(I), self.skybricks['BRICKNAME'][i]))
            if len(I) == 0:
                continue

            # Read skybrick file
            fn = os.path.join(self.skybricks_dir,
                              'sky-%s.fits.gz' % self.skybricks['BRICKNAME'][i])
            if not os.path.exists(fn):
                log.warning('Missing "skybrick" file: %s' % fn)
                continue
            skymap,hdr = fitsio.read(fn, header=True)
            H,W = skymap.shape
            # create WCS object
            w = WCS(naxis=2)
            w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
            w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
            w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
            w.wcs.cd = [[hdr['CD1_1'], hdr['CD1_2']],
                        [hdr['CD2_1'], hdr['CD2_2']]]
            x,y = w.wcs_world2pix(ras.flat[I], decs.flat[I], 0)
            x = np.round(x).astype(int)
            y = np.round(y).astype(int)
            # we have margins that should ensure this...
            if not (np.all(x >= 0) and np.all(x <  W) and np.all(y >= 0) and np.all(y <  H)):
                raise RuntimeError('Skybrick %s: locations project outside the brick bounds' % (self.skybricks['BRICKNAME'][i]))

            # FIXME -- look at surrounding pixels too??
            good_sky.flat[I] = (skymap[y, x] == 0)
        return good_sky

def _radec2kd(ra, dec):
    """
    Creates a scipy KDTree from the given *ra*, *dec* arrays (in deg).
    """
    from scipy.spatial import KDTree
    xyz = _radec2xyz(ra, dec)
    return KDTree(xyz)

def _radec2xyz(ra, dec):
    """
    Converts arrays from *ra*, *dec* (in deg) to XYZ unit-sphere
    coordinates.
    """
    rr = np.deg2rad(ra)
    dd = np.deg2rad(dec)
    return np.vstack((np.cos(rr) * np.cos(dd),
                      np.sin(rr) * np.cos(dd),
                      np.sin(dd))).T
