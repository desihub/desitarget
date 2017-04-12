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
        from desitarget import desi_mask, bgs_mask, mws_mask, contam_mask, obsconditions
        
        self.desi_mask = desi_mask
        self.bgs_mask = bgs_mask
        self.mws_mask = mws_mask
        self.contam_mask = contam_mask
        self.obsconditions = obsconditions

        self.log = logger
        self.rand = rand
        self.brick_info = brick_info

        self.decam_extcoeff = (3.995, 3.214, 2.165, 1.592, 1.211, 1.064) # extinction coefficients
        self.wise_extcoeff = (0.184, 0.113, 0.0241, 0.00910)
        self.sdss_extcoeff = (4.239, 3.303, 2.285, 1.698, 1.263)

        #self.mws_scale = 10**(-0.4*(23.4-19.6))
        #self.log.info('Using a constant factor to scale the MWS to fainter magnitudes!')

    def _star_select(self, targets, truth):
        """Select stellar (faint and bright) contaminants for the extragalactic
        targets.

        """ 
        from desitarget.cuts import isBGS_faint, isELG, isLRG, isQSO_colors

        gflux = targets['DECAM_FLUX'][..., 1]
        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        w1flux = targets['WISE_FLUX'][..., 0]
        w2flux = targets['WISE_FLUX'][..., 1]

        # Select faint stellar contaminants for BGS_FAINT targets.
        bgs_faint = isBGS_faint(rflux=rflux)
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT
        targets['BGS_TARGET'] |= (bgs_faint != 0) * self.bgs_mask.BGS_FAINT_SOUTH
        targets['DESI_TARGET'] |= (bgs_faint != 0) * self.desi_mask.BGS_ANY
        for oo in self.bgs_mask.BGS_FAINT.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (bgs_faint != 0) * self.obsconditions.mask(oo)
            
        truth['CONTAM_TARGET'] |= (bgs_faint != 0) * self.contam_mask.BGS_IS_STAR
        truth['CONTAM_TARGET'] |= (bgs_faint != 0) * self.contam_mask.BGS_CONTAM

        # Select faint stellar contaminants for ELG targets.
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        for oo in self.desi_mask.ELG.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(oo)

        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_IS_STAR
        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_CONTAM

        # Select faint stellar contaminants for LRG targets.
        lrg = isLRG(gflux=gflux, rflux=rflux, zflux=zflux)
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH
        for oo in self.desi_mask.LRG.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(oo)

        truth['CONTAM_TARGET'] |= (lrg != 0) * self.contam_mask.LRG_IS_STAR
        truth['CONTAM_TARGET'] |= (lrg != 0) * self.contam_mask.LRG_CONTAM

        # Select faint stellar contaminants for QSO targets.
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                           w2flux=w2flux, optical=False) # Note optical=False!
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH
        for oo in self.desi_mask.QSO.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (qso != 0) * self.obsconditions.mask(oo)

        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_IS_STAR
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_CONTAM

    def _std_select(self, targets, truth):
        """Select bright- and dark-time standard stars."""
        from desitarget.cuts import isFSTD

        gflux = targets['DECAM_FLUX'][..., 1]
        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        w1flux = targets['WISE_FLUX'][..., 0]
        w2flux = targets['WISE_FLUX'][..., 1]

        obs_rflux = rflux * 10**(-0.4 * targets['EBV'] * self.decam_extcoeff[2]) # attenuate for dust

        snr = np.zeros_like(targets['DECAM_FLUX']) + 100      # Hack -- fixed S/N
        fracflux = np.zeros_like(targets['DECAM_FLUX']).T     # No contamination from neighbors.
        objtype = np.repeat('PSF', len(targets)).astype('U3') # Right data type?!?

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

        
    def bgs_select(self, targets, truth):
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

    def elg_select(self, targets, truth):
        """Select ELG targets and contaminants."""
        from desitarget.cuts import isELG, isQSO_colors

        gflux = targets['DECAM_FLUX'][..., 1]
        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        w1flux = targets['WISE_FLUX'][..., 0]
        w2flux = targets['WISE_FLUX'][..., 1]
        
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)

        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        for oo in self.desi_mask.ELG.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(oo)

        # Select ELG contaminants for QSO targets.  There should be a morphology
        # cut here, too, so we're going to overestimate the number of
        # contaminants.
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                           w2flux=w2flux, optical=False) # Note optical=False!
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH
        for oo in self.desi_mask.QSO.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (qso != 0) * self.obsconditions.mask(oo)
            
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_IS_ELG
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_IS_GALAXY
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_CONTAM

    def faintstar_select(self, targets, truth):
        """Select faint stellar contaminants for the extragalactic targets.""" 

        self._star_select(targets, truth)

    def lrg_select(self, targets, truth):
        """Select LRG targets."""
        from desitarget.cuts import isLRG

        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        w1flux = targets['WISE_FLUX'][..., 0]
        w1flux = targets['WISE_FLUX'][..., 0]
        w2flux = targets['WISE_FLUX'][..., 1]
        lrg = isLRG(rflux=rflux, zflux=zflux, w1flux=w1flux)

        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG
        targets['DESI_TARGET'] |= (lrg != 0) * self.desi_mask.LRG_SOUTH
        for oo in self.desi_mask.LRG.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (lrg != 0) * self.obsconditions.mask(oo)

        # Select LRG contaminants for QSO targets.  There should be a morphology
        # cut here, too, so we're going to overestimate the number of
        # contaminants.
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                           w2flux=w2flux, optical=False) # Note optical=False!
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH
        for oo in self.desi_mask.QSO.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (qso != 0) * self.obsconditions.mask(oo)

        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_IS_LRG
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_IS_GALAXY
        truth['CONTAM_TARGET'] |= (qso != 0) * self.contam_mask.QSO_CONTAM

    def mws_main_select(self, targets, truth):
        """Select MWS_MAIN, MWS_MAIN_VERY_FAINT, STD_FSTAR, and STD_BRIGHT targets.  The
        selection eventually will be done with Gaia (I think).  Also select
        contaminants for the other target classes.

        """
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
        w1flux = targets['WISE_FLUX'][..., 0]
        w2flux = targets['WISE_FLUX'][..., 1]

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

        # Select standard stars.
        self.std_select(targets, truth)
        
        # Select bright (MWS) stellar contaminants for the extragalactic
        # targets.
        self._star_select(targets, truth)

    def mws_nearby_select(self, targets, truth):
        """Select MWS_NEARBY targets.  The selection eventually will be done with Gaia,
        so for now just do a "perfect" selection.

        """
        mws_nearby = np.ones(len(targets)) # select everything!
        #mws_nearby = (truth['MAG'] <= 20.0) * 1 # SDSS g-band!

        targets['MWS_TARGET'] |= (mws_nearby != 0) * self.mws_mask.mask('MWS_NEARBY')
        targets['DESI_TARGET'] |= (mws_nearby != 0) * self.desi_mask.MWS_ANY
        for oo in self.mws_mask.MWS_NEARBY.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (mws_nearby != 0) * self.obsconditions.mask(oo)

    def mws_wd_select(self, targets, truth):
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

    def qso_select(self, targets, truth):
        """Select QSO targets and contaminants."""
        from desitarget.cuts import isQSO_colors, isELG

        gflux = targets['DECAM_FLUX'][..., 1]
        rflux = targets['DECAM_FLUX'][..., 2]
        zflux = targets['DECAM_FLUX'][..., 4]
        w1flux = targets['WISE_FLUX'][..., 0]
        w2flux = targets['WISE_FLUX'][..., 1]
        qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                           w1flux=w1flux, w2flux=w2flux, optical=True)

        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO
        targets['DESI_TARGET'] |= (qso != 0) * self.desi_mask.QSO_SOUTH
        for oo in self.desi_mask.QSO.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (qso != 0) * self.obsconditions.mask(oo)

        # Select QSO contaminants for ELG targets.  There'd be no morphology cut
        # and we're missing the WISE colors, so the density won't be right.
        elg = isELG(gflux=gflux, rflux=rflux, zflux=zflux)
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG
        targets['DESI_TARGET'] |= (elg != 0) * self.desi_mask.ELG_SOUTH
        for oo in self.desi_mask.ELG.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= (elg != 0) * self.obsconditions.mask(oo)

        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_IS_QSO
        truth['CONTAM_TARGET'] |= (elg != 0) * self.contam_mask.ELG_CONTAM

    def sky_select(self, targets, truth):
        """Select SKY targets."""

        targets['DESI_TARGET'] |= self.desi_mask.mask('SKY')
        for oo in self.desi_mask.SKY.obsconditions.split('|'):
            targets['OBSCONDITIONS'] |= self.obsconditions.mask(oo)

    def density_select(self, targets, truth, source_name, target_name, density):
        """Downsample a target sample to a desired number density in targets/deg2."""
        nobj = len(targets)

        unique_bricks = list(set(targets['BRICKNAME']))
        n_brick = len(unique_bricks)

        keep = []
        for thisbrick in unique_bricks:
            brickindx = np.where(self.brick_info['BRICKNAME'] == thisbrick)[0]
            if len(brickindx) == 0:
                self.log.warning('No matching brick {}!'.format(thisbrick))
                raise ValueError
            brick_area = float(self.brick_info['BRICKAREA'][brickindx])

            #onbrick = np.where(targets['BRICKNAME'] == thisbrick)[0]
            onbrick = np.where((targets['BRICKNAME'] == thisbrick) *
                               (truth['TEMPLATETYPE'] == target_name.upper()))[0]

            n_in_brick = len(onbrick)
            if n_in_brick == 0:
                self.log.warning('No objects on brick {}, which should not happen!'.format(thisbrick))
                raise ValueError

            # Downsample in density.
            if density:
                mock_density = n_in_brick / brick_area
                desired_density = float(self.brick_info['FLUC_EBV'][source_name][brickindx] * density)

                frac_keep = desired_density / mock_density
                self.log.debug('Downsampling {}s from {} to {} targets/deg2.'.format(source_name,
                                                                                     mock_density,
                                                                                     desired_density))

                if (frac_keep > 1.0):
                    self.log.warning('Brick {}: mock density {}/deg2 too low!'.format(thisbrick, mock_density))
                    frac_keep = 1.0

                keep.append(self.rand.choice(onbrick, np.ceil( int(n_in_brick * frac_keep) ), replace=False))

        return np.hstack(keep)

    def contaminants_select(self, targets, truth, source_name, target_name, contam):
        """Downsample contaminants to a desired number density in targets/deg2."""
        nobj = len(targets)

        unique_bricks = list(set(targets['BRICKNAME']))
        n_brick = len(unique_bricks)

        toss = []
        for thisbrick in unique_bricks:
            brickindx = np.where(self.brick_info['BRICKNAME'] == thisbrick)[0]
            if len(brickindx) == 0:
                self.log.warning('No matching brick {}!'.format(thisbrick))
                raise ValueError
            brick_area = float(self.brick_info['BRICKAREA'][brickindx])

            onbrick = np.where(targets['BRICKNAME'] == thisbrick)[0]
            n_in_brick = len(onbrick)
            if n_in_brick == 0:
                self.log.warning('No objects on brick {}, which should not happen!'.format(thisbrick))
                raise ValueError

            for contam_name in contam.keys():

                if contam_name == 'GALAXY':
                    onbrick_contam = np.where(
                        ( (targets['DESI_TARGET'][onbrick] & self.desi_mask.mask(target_name)) != 0 ) *
                        ( (targets['DESI_TARGET'][onbrick] & self.desi_mask.mask('ELG')) == 0 ) *
                        ( (targets['DESI_TARGET'][onbrick] & self.desi_mask.mask('LRG')) == 0 ) *
                        ( truth['TRUESPECTYPE'][onbrick] == contam_name )
                        )[0]
                else:
                    onbrick_contam = np.where(
                        ( (targets['DESI_TARGET'][onbrick] & self.desi_mask.mask(target_name)) != 0 ) *
                        ( truth['TRUESPECTYPE'][onbrick] == contam_name )
                        )[0]
                    
                n_in_brick_contam = len(onbrick_contam)

                contam_density = len(onbrick_contam) / brick_area
                desired_density = float(self.brick_info['FLUC_EBV'][source_name][brickindx] * contam[contam_name])

                frac_keep = desired_density / contam_density
                self.log.debug('Downsampling {}/{} contaminants from {} to {} targets/deg2.'.format(target_name,
                                                                                                    contam_name,
                                                                                                    contam_density,
                                                                                                    desired_density))

                if (frac_keep > 1.0):
                    self.log.warning('Brick {}: contaminant density {}/deg2 too low!'.format(thisbrick, contam_density))
                    frac_keep = 1.0

                toss_contam = self.rand.choice(onbrick_contam, int( np.ceil(
                    n_in_brick_contam * (1 - frac_keep) ) ), replace=False)
                toss.append(onbrick[toss_contam])
                
                #targets[onbrick[toss_contam]]['DESI_TARGET'] & self.desi_mask.mask(target_name)
                #truth[onbrick[toss_contam]]
                #import pdb ; pdb.set_trace()

        return np.hstack(toss)
