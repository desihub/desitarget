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
        hdr = fitsio.read_header(skybricks_fn)
        self.skykd = _radec2kd(self.skybricks['RA'], self.skybricks['DEC'])
        # default skybricks are 1 x 1 deg.
        self.brick_radius = hdr.get('SKYBRAD', 1.*np.sqrt(2.)/2.)
        self.nside = hdr.get('SKYHPNS', None)
        self.nest = True

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

        searchrad = 1.01 * (tileradius + self.brick_radius)
        # here, convert search radius to radians -- an overestimate vs
        # unit-sphere distance, but that's the safe direction.
        searchrad = np.deg2rad(searchrad)
        tilexyz = _radec2xyz([tilera], [tiledec])
        sky_inds = self.skykd.query_ball_point(tilexyz[0, :], searchrad)
        # handle non-array iterables (eg lists) as inputs
        ras = np.array(ras)
        decs = np.array(decs)
        good_sky = np.zeros(ras.shape, bool)
        have_radecbox = 'RA1' in self.skybricks.dtype.fields
        have_nside = (self.nside is not None) and ('HEALPIX' in self.skybricks.dtype.fields)
        if have_nside:
            import healpy as hp
            hps = hp.ang2pix(self.nside, np.radians((90. - decs)), np.radians(ras),
                             nest=self.nest)
        for i in sky_inds:
            # Check possibly-overlapping skybricks in [RA1,RA2], [DEC1,DEC2] boxes
            # (if available)
            if have_radecbox:
                # Do any of the query points overlap in the brick's RA,DEC unique-area bounding-box?
                ra1 = self.skybricks['RA1'][i]
                ra2 = self.skybricks['RA2'][i]
                # normal case
                if ra1 < ra2:
                    raok = (ras >= ra1) * (ras <= ra2)
                else:
                    # RA wrap-around -- ra1 <= 360, ra2 >= 0
                    raok = np.logical_or(ras >= ra1, ras <= ra2)
                I = np.flatnonzero(raok *
                                   (decs >= self.skybricks['DEC1'][i]) *
                                   (decs <  self.skybricks['DEC2'][i]))
                log.debug('Skybricks: %i locations overlap skybrick %s' % (len(I), self.skybricks['BRICKNAME'][i]))
            elif have_nside:
                # Do any of the query points land in this "brick's" healpix?
                I = np.flatnonzero(hps == self.skybricks['HEALPIX'][i])
            else:
                I = np.arange(len(ras))
            if len(I) == 0:
                continue

            # Read skybrick file, looking for fits.fz then fits.gz
            for ext in ['fz', 'gz']:
                fn = os.path.join(self.skybricks_dir,
                                  'sky-{}.fits.{}'.format(
                                      self.skybricks['BRICKNAME'][i], ext))
                if os.path.exists(fn):
                    break
            else:
                log.warning('Missing "skybrick" file: %s/.fz' % fn)
                continue

            skymap, hdr = fitsio.read(fn, header=True)
            H, W = skymap.shape
            # create WCS object
            w = WCS(naxis=2)
            w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
            w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
            w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
            w.wcs.cd = [[hdr['CD1_1'], hdr['CD1_2']],
                        [hdr['CD2_1'], hdr['CD2_2']]]
            x, y = w.wcs_world2pix(ras.flat[I], decs.flat[I], 0)
            x = np.round(x).astype(int)
            y = np.round(y).astype(int)
            in_bounds = (x >= 0) * (x < W) * (y >= 0) * (y < H)
            # we have margins that should ensure this...
            if have_radecbox and not np.all(in_bounds):
                raise RuntimeError('Skybrick %s: RA,Decs project outside the brick bounds' % (self.skybricks['BRICKNAME'][i]))

            # Look at the nearest pixel in the skybrick map.
            good_sky.flat[I[in_bounds]] = (skymap[y[in_bounds], x[in_bounds]] == 0)
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
