# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.skyhealpixs
====================

Dynamic lookup of whether a given RA,Dec location is a good place to
put a sky fiber.
Scripts based on desitarget.skybricks, adapted for healpix split.
"""
import os
import numpy as np


class Skyhealpixs(object):
    '''
    This class handles dynamic lookup of whether a given (RA,Dec)
    should make a good location for a sky fiber.
    '''
    def __init__(self, skyhealpixs_dir=None, nside=64, nest=True):
        '''
        Create a Skyhealpixs object, reading metadata.

        Parameters
        ----------
        skyhealpixs_dir : :class:`str`
        nside (defaults to 64) : :class:`int`
        nest (defaults to True) : :clasee:`bool`
            The directory to find skyhealpixs data files; if None, will read from
            SKYHEALPIXS_DIR environment variable.
        '''
        import fitsio
        if skyhealpixs_dir is None:
            skyhealpixs_dir = os.environ.get('SKYHEALPIXS_DIR', None)
        if skyhealpixs_dir is None:
            raise RuntimeError('Environment variable SKYHEALPIXS_DIR is not set; '
                               'needed to look up dynamic sky fiber positions')
        self.skyhealpixs_dir = skyhealpixs_dir
        self.nside, self.nest = nside, nest

    def lookup_position(self, ras, decs):
        '''
        Looks up a set of RA,Dec positions that are all within a given
        tile (disk on the sky).

        Parameters
        ----------
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
        import healpy as hp
        log = get_logger()

        # AR pixel padding in file name, taking the largest pixel number
        npadpix = len(str(hp.nside2npix(self.nside)))

        # AR infos
        log.info(
            'settings : skyhealpixs_dir={}, nside={}, nest={}'.format(
                self.skyhealpixs_dir, self.nside, self.nest,
            )
        )
        log.info('the code will look for {} files'.format(
                os.path.join(self.skyhealpixs_dir, 'skymap-{}.fits.gz'.format(
                    "".join(np.repeat("?", npadpix)),
                    )
                ),
            )
        )

        # handle non-array iterables (eg lists) as inputs
        ras = np.array(ras)
        decs = np.array(decs)
        good_sky = np.zeros(ras.shape, bool)
        # AR healpix pixels
        pixs = hp.ang2pix(self.nside, np.radians((90. - decs)), np.radians(ras), nest=self.nest)
        # AR Check healpix pixels
        for pix in np.unique(pixs):
            I = pixs == pix
            log.debug('Skyhealpixs: %i locations overlap healpix pixel %s' % (len(I), pix))

            # Read healpix file
            padpix = "{number:0{width}d}".format(number=pix, width=npadpix)
            fn = os.path.join(self.skyhealpixs_dir, 'skymap-{}.fits.gz'.format(padpix))
            if not os.path.exists(fn):
                log.warning('Missing "skyhealpix" file: {}'.format(fn))
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
            # we have margins that should ensure this...
            if not (np.all(x >= 0) and np.all(x < W) and np.all(y >= 0) and np.all(y < H)):
                raise RuntimeError('Skyhealpix %s: locations project outside the brick bounds' % (fn))

            # FIXME -- look at surrounding pixels too??
            good_sky.flat[I] = (skymap[y, x] == 0)
        return good_sky
