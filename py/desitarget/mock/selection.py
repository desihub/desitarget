# Licensed under a 4-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.selection
=========================

Applies selection criteria on mock target catalogs.

"""
from __future__ import (absolute_import, division)

import numpy as np

class SelectTargets(object):
    """Select various types of targets.  Most of this functionality is taken from
    desitarget.cuts but that code has not been factored in a way that is
    convenient at this time.

    """
    def __init__(self, logger=None, rand=None, brick_info=None):
        from desitarget import desi_mask, bgs_mask, mws_mask, obsconditions
        self.desi_mask = desi_mask
        self.bgs_mask = bgs_mask
        self.mws_mask = mws_mask
        self.obsconditions = obsconditions

        self.log = logger
        self.rand = rand
        self.brick_info = brick_info

        self.decam_extcoeff = (3.995, 3.214, 2.165, 1.592, 1.211, 1.064) # extinction coefficients
        self.wise_extcoeff = (0.184, 0.113, 0.0241, 0.00910)
        self.sdss_extcoeff = (4.239, 3.303, 2.285, 1.698, 1.263)

    def bgs_select(self, targets, truth=None):
        """Select BGS targets.  Note that obsconditions for BGS_ANY are set to BRIGHT
        only. Is this what we want?

        """
        from desitarget.cuts import isBGS_bright, isBGS_faint

        rflux = targets['DECAM_FLUX'][..., 2]

        # Select BGS_BRIGHT targets.
        bgs_bright = isBGS_bright(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT
        targets['BGS_TARGET'] |= (bgs_bright != 0) * self.bgs_mask.BGS_BRIGHT_SOUTH
        targets['DESI_TARGET'] |= (bgs_bright != 0) * self.desi_mask.BGS_ANY
        for oo in self.bgs_mask.BGS_BRIGHT.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (bgs_bright != 0) * self.obsconditions.mask(oo)

        # Select BGS_FAINT targets.
        bgs_faint = isBGS_faint(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT_SOUTH
        targets['DESI_TARGET'] |= (bgs_faint != 0) * self.desi_mask.BGS_ANY
        for oo in self.bgs_mask.BGS_FAINT.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(oo)

        return targets

    def elg_select(self, targets, truth=None):
        """Select ELG targets."""
        from desitarget.cuts import isELG

        gflux = targets['DECAM_FLUX'][..., 1]
        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)

        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        for oo in self.desi_mask.ELG.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(oo)

        return targets

    def lrg_select(self, targets, truth=None):
        """Select LRG targets."""
        from desitarget.cuts import isLRG

        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        w1flux = targets['WISE_FLUX'][..., 0]
        lrg = isLRG(rflux=rflux, zflux=zflux, w1flux=w1flux)

        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH
        for oo in self.desi_mask.LRG.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(oo)

        return targets

    def mws_main_select(self, targets, truth=None):
        """Select MWS_MAIN, MWS_MAIN_VERY_FAINT, STD_FSTAR, and STD_BRIGHT targets.  The
        selection eventually will be done with Gaia (I think).

        """
        from desitarget.cuts import isFSTD

        def _isMWS_MAIN(rflux):
            """A function like this should be in desitarget.cuts. Select 15<r<19 stars."""
            main = rflux > 10**((22.5-19.0)/2.5)
            main &= rflux <= 10**((22.5-15.0)/2.5)
            return main

        def _isMWS_MAIN_VERY_FAINT(rflux):
            """A function like this should be in desitarget.cuts. Select 19<r<20 filler stars."""
            faint = rflux > 10**((22.5-20.0)/2.5)
            faint &= rflux <= 10**((22.5-19.0)/2.5)
            return faint

        gflux = targets['DECAM_FLUX'][..., 1]
        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        obs_rflux = rflux * 10**(-0.4 * targets['EBV'] * self.decam_extcoeff[2]) # attenuate for dust

        snr = np.zeros_like(targets['DECAM_FLUX']) + 100      # Hack -- fixed S/N
        fracflux = np.zeros_like(targets['DECAM_FLUX']).T     # No contamination from neighbors.
        objtype = np.repeat('PSF', len(targets)).astype('U3') # Right data type?!?

        # Select MWS_MAIN targets.
        mws_main = _isMWS_MAIN(rflux=rflux)
        targets['MWS_TARGET'] |= (mws_main != 0) * self.mws_mask.mask('MWS_MAIN')
        targets['DESI_TARGET'] |= (mws_main != 0) * self.desi_mask.MWS_ANY
        for oo in self.mws_mask.MWS_MAIN.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (mws_main != 0) * self.obsconditions.mask(oo)

        # Select MWS_MAIN_VERY_FAINT targets.
        mws_very_faint = _isMWS_MAIN_VERY_FAINT(rflux=rflux)
        targets['MWS_TARGET'] |= (mws_very_faint != 0) * self.mws_mask.mask('MWS_MAIN_VERY_FAINT')
        targets['DESI_TARGET'] |= (mws_very_faint != 0) * self.desi_mask.MWS_ANY
        for oo in self.mws_mask.MWS_MAIN_VERY_FAINT.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (mws_very_faint != 0) * self.obsconditions.mask(oo)

        # Select dark-time FSTD targets.
        fstd = isFSTD(gflux=gflux, rflux=rflux, zflux=zflux, objtype=objtype,
                      decam_fracflux=fracflux, decam_snr=snr, obs_rflux=obs_rflux)
        targets['DESI_TARGET'] |= (fstd != 0) * self.desi_mask.STD_FSTAR
        for oo in self.desi_mask.STD_FSTAR.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (fstd != 0) * self.obsconditions.mask(oo)

        # Select bright-time FSTD targets.
        fstd_bright = isFSTD(gflux=gflux, rflux=rflux, zflux=zflux, objtype=objtype,
                             decam_fracflux=fracflux, decam_snr=snr, obs_rflux=obs_rflux,
                             bright=True)
        targets['DESI_TARGET'] |= (fstd_bright != 0) * self.desi_mask.STD_BRIGHT
        for oo in self.desi_mask.STD_BRIGHT.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (fstd_bright != 0) * self.obsconditions.mask(oo)

        return targets

    def mws_nearby_select(self, targets, truth=None):
        """Select MWS_NEARBY targets.  The selection eventually will be done with Gaia,
        so for now just do a "perfect" selection.

        """
        mws_nearby = np.ones(len(targets)) # select everything!
        #mws_nearby = (truth['MAG'] <= 20.0) * 1 # SDSS g-band!

        targets['MWS_TARGET'] |= (mws_nearby != 0) * self.mws_mask.mask('MWS_NEARBY')
        targets['DESI_TARGET'] |= (mws_nearby != 0) * self.desi_mask.MWS_ANY
        for oo in self.mws_mask.MWS_NEARBY.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (mws_nearby != 0) * self.obsconditions.mask(oo)

        return targets

    def mws_wd_select(self, targets, truth=None):
        """Select MWS_WD and STD_WD targets.  The selection eventually will be done with
        Gaia, so for now just do a "perfect" selection here.

        """
        #mws_wd = np.ones(len(targets)) # select everything!
        mws_wd = ((truth['MAG'] >= 15.0) * (truth['MAG'] <= 20.0)) * 1 # SDSS g-band!

        targets['MWS_TARGET'] |= (mws_wd != 0) * self.mws_mask.mask('MWS_WD')
        targets['DESI_TARGET'] |= (mws_wd != 0) * self.desi_mask.MWS_ANY
        for oo in self.mws_mask.MWS_WD.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (mws_wd != 0) * self.obsconditions.mask(oo)

        # Select STD_WD; cut just on g-band magnitude (not TEMPLATESUBTYPE!)
        std_wd = (truth['MAG'] <= 19.0) * 1 # SDSS g-band!
        targets['DESI_TARGET'] |= (std_wd !=0) * self.desi_mask.mask('STD_WD')
        for oo in self.desi_mask.STD_WD.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (std_wd != 0) * self.obsconditions.mask(oo)

        return targets

    def qso_select(self, targets, truth=None):
        """Select QSO targets.  Unfortunately we can't apply the appropriate color-cuts
        because our spectra don't go red enough (i.e., into the WISE bands).  So
        all the QSOs pass for now.

        """
        if False:
            from desitarget.cuts import isQSO

            gflux = targets['DECAM_FLUX'][..., 1]
            rflux = targets['DECAM_FLUX'][..., 2]
            zflux = targets['DECAM_FLUX'][..., 4]
            w1flux = targets['WISE_FLUX'][..., 0]
            w2flux = targets['WISE_FLUX'][..., 1]
            qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                               w1flux=w1flux, w2flux=w2flux)
        else:
            qso = np.ones(len(targets)) # select everything!

        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH
        for oo in self.desi_mask.QSO.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (qso != 0) * self.obsconditions.mask(oo)

        return targets

    def sky_select(self, targets, truth=None):
        """Select SKY targets."""

        targets['DESI_TARGET'] |= self.desi_mask.mask('SKY')
        for oo in self.desi_mask.SKY.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= self.obsconditions.mask(oo)

        return targets

    def density_select(self, targets, density, sourcename):
        """Downsample a target sample to a desired number density in targets/deg2."""

        nobj = len(targets)

        unique_bricks = list(set(targets['BRICKNAME']))
        n_brick = len(unique_bricks)

        keep = []
        for thisbrick in unique_bricks:
            brickindx = np.where(self.brick_info['BRICKNAME'] == thisbrick)[0]
            if len(brickindx) == 0:
                log.warning('No matching brick {}!'.format(thisbrick))
                raise ValueError
            brick_area = self.brick_info['BRICKAREA'][brickindx]

            onbrick = np.where(targets['BRICKNAME'] == thisbrick)[0]
            n_in_brick = len(onbrick)
            if n_in_brick == 0:
                self.log.warning('No objects on brick {}, which should not happen!'.format(thisbrick))
                raise ValueError

            mock_density = n_in_brick / brick_area
            desired_density = self.brick_info['FLUC_EBV'][sourcename][brickindx] * density

            frac_keep = desired_density / mock_density
            self.log.debug('Downsampling {}s from {} to {} targets/deg2.'.format(sourcename,
                                                                                 mock_density,
                                                                                 desired_density))

            if (frac_keep > 1.0):
                self.log.warning('Brick {}: mock density {}/deg2 too low!.'.format(thisbrick, mock_density))
                frac_keep = 1.0

            #import pdb ; pdb.set_trace()
            keep.append(self.rand.choice(onbrick, int(n_in_brick * frac_keep), replace=False))

        return np.hstack(keep)
