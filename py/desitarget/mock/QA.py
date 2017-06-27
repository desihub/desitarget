# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.QA
==================

Generate QA figures from the output of desitarget.mock.join_targets_truth.

"""
from __future__ import (absolute_import, division, print_function)

import os
import warnings

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from desimodel.footprint import radec2pix
from desitarget import desi_mask, bgs_mask, mws_mask, contam_mask

def target_density(cat, nside=128):
    """Determine the target density by grouping targets in healpix pixels.  The code
    below was code shamelessly taken from desiutil.plot.plot_sky_binned (by
    D. Kirkby).
    
    Args:
        cat: Table with columns RA and DEC
    
    Optional:
        nside: healpix nside, integer power of 2

    """
    npix = hp.nside2npix(nside)
    bin_area = hp.nside2pixarea(nside, degrees=True)

    pixels = radec2pix(nside, cat['RA'], cat['DEC'])
    #pixels = hp.ang2pix(nside, np.radians(90 - cat['DEC']), 
    #                    np.radians(cat['RA']), nest=False)
    counts = np.bincount(pixels, weights=None, minlength=npix)
    dens = counts[np.flatnonzero(counts)] / bin_area
            
    return dens

def qadensity(cat, objtype, targdens=None, max_bin_area=1.0, qadir='.'):
    """Visualize the target density with a skymap and histogram.
    
    """
    from desiutil.plots import init_sky, plot_sky_binned
    
    label = '{} (targets/deg$^2$)'.format(objtype)
    if targdens:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    else:
        fig, ax = plt.subplots(1)
    ax = np.atleast_1d(ax)
       
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        basemap = init_sky(galactic_plane_color='k', ax=ax[0]);
        plot_sky_binned(cat['RA'], cat['DEC'], max_bin_area=max_bin_area,
                        clip_lo='!1', cmap='jet', plot_type='healpix', 
                        label=label, basemap=basemap)
        
    if targdens:
        dens = target_density(cat)
        ax[1].hist(dens, bins=100, histtype='stepfilled', alpha=0.6, label='Observed {} Density'.format(objtype))
        if objtype in targdens.keys():
            ax[1].axvline(x=targdens[objtype], ls='--', color='k', label='Goal {} Density'.format(objtype))
        ax[1].set_xlabel(label)
        ax[1].set_ylabel('Number of Healpixels')
        ax[1].legend(loc='upper left', frameon=False)
        fig.subplots_adjust(wspace=0.2)

    pngfile = os.path.join(qadir, '{}_target_density.png'.format(objtype))
    fig.savefig(pngfile)

    return pngfile

def qa_qso(targets, truth, qadir='.'):
    """Detailed QA plots for QSOs."""
    
    dens = dict()
    
    these = np.where(targets['DESI_TARGET'] & desi_mask.mask('QSO') != 0)[0]
    dens['QSO_TARGETS'] = target_density(targets[these])

    these = np.where((targets['DESI_TARGET'] & desi_mask.mask('QSO') != 0) * (truth['TRUEZ'] < 2.1))[0]
    dens['QSO_TRACER'] = target_density(targets[these])

    these = np.where((targets['DESI_TARGET'] & desi_mask.mask('QSO') != 0) * (truth['TRUEZ'] >= 2.1))[0]
    dens['QSO_LYA'] = target_density(targets[these])
    
    these = np.where((targets['DESI_TARGET'] & desi_mask.mask('QSO') != 0) * 
                     (truth['CONTAM_TARGET'] & contam_mask.mask('QSO_IS_GALAXY')) != 0)[0]
    dens['QSO_IS_GALAXY'] = target_density(targets[these])

    these = np.where((targets['DESI_TARGET'] & desi_mask.mask('QSO') != 0) * 
                     (truth['CONTAM_TARGET'] & contam_mask.mask('QSO_IS_STAR')) != 0)[0]
    dens['QSO_IS_STAR'] = target_density(targets[these])

    bins = 50
    lim = (0, 280)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(dens['QSO_TARGETS'], bins=bins, range=lim, label='All Targets',
               color='k', lw=1, histtype='step')
    ax[0].hist(dens['QSO_TRACER'], bins=bins, range=lim, alpha=0.6, ls='--',
               lw=2, label='Tracer QSOs')#, histtype='step')
    ax[0].hist(dens['QSO_LYA'], bins=bins, range=lim, lw=2, label='Lya QSOs')
    ax[0].set_ylabel('Number of Healpixels')
    ax[0].set_xlabel('Targets / deg$^2$')
    ax[0].set_title('True QSOs')
    ax[0].legend(loc='upper right')
    
    lim = (0, 100)
    
    #ax[1].hist(dens['QSO_TARGETS'], bins=bins, range=lim, label='All Targets',
    #           color='k', lw=1, histtype='step')
    ax[1].hist(dens['QSO_IS_STAR'], bins=bins, range=lim, alpha=0.3, label='QSO_IS_STAR')
    ax[1].hist(dens['QSO_IS_GALAXY'], bins=bins, range=lim, alpha=0.5, label='QSO_IS_GALAXY')
    ax[1].set_ylabel('Number of Healpixels')
    ax[1].set_xlabel('Targets / deg$^2$')
    ax[1].set_title('QSO Contaminants')
    ax[1].legend(loc='upper right')

    pngfile = os.path.join(qadir, '{}_detail_density.png'.format('qso'))
    fig.savefig(pngfile)
    
    return pngfile

def qa_elg(targets, truth, qadir='.'):
    """Detailed QA plots for ELGs."""

    dens_all = 2400
    dens_loz = dens_all*0.05
    dens_hiz = dens_all*0.05
    dens_star = dens_all*0.1
    dens_rightz = dens_all - dens_loz - dens_hiz - dens_star
    
    dens = dict()

    these = np.where(targets['DESI_TARGET'] & desi_mask.mask('ELG') != 0)[0]
    dens['ELG_TARGETS'] = target_density(targets[these])

    these = np.where((targets['DESI_TARGET'] & desi_mask.mask('ELG') != 0) *
                     (truth['TRUEZ'] >= 0.6) * (truth['TRUEZ'] <= 1.6))[0]
    dens['ELG_IS_RIGHTZ'] = target_density(targets[these])

    these = np.where((targets['DESI_TARGET'] & desi_mask.mask('QSO') != 0) * (truth['TRUEZ'] < 0.6))[0]
    dens['ELG_IS_LOZ'] = target_density(targets[these])

    these = np.where((targets['DESI_TARGET'] & desi_mask.mask('QSO') != 0) * (truth['TRUEZ'] > 1.6))[0]
    dens['ELG_IS_HIZ'] = target_density(targets[these])

    these = np.where((targets['DESI_TARGET'] & desi_mask.mask('ELG') != 0) * 
                     (truth['CONTAM_TARGET'] & contam_mask.mask('ELG_IS_STAR')) != 0)[0]
    dens['ELG_IS_STAR'] = target_density(targets[these])

    bins = 50
    lim = (0, 3000)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    #line = ax[0].axvline(x=dens_all, ls='-')
    ax[0].hist(dens['ELG_TARGETS'], bins=bins, range=lim, #color=line.get_color(),
               color='k', lw=1, label='All Targets', histtype='step')
    
    #line = ax[0].axvline(x=dens_rightz, ls='--')
    ax[0].hist(dens['ELG_IS_RIGHTZ'], bins=bins, range=lim, alpha=0.6, # color=line.get_color(), 
               ls='--', lw=2, label='ELG (0.6<z<1.6)')#, histtype='step')
    ax[0].set_ylabel('Number of Healpixels')
    ax[0].set_xlabel('Targets / deg$^2$')
    ax[0].set_title('True ELGs')
    ax[0].legend(loc='upper left')

    lim = (0, 300)
    
    #ax[1].hist(dens['ELG_TARGETS'], bins=bins, range=lim, label='All Targets',
    #           color='k', lw=1, histtype='step')
    #line = ax[1].axvline(x=dens_star, ls='-')
    ax[1].hist(dens['ELG_IS_STAR'], bins=bins, range=lim, #color=line.get_color(), 
               alpha=0.5, label='ELG_IS_STAR')

    #line = ax[1].axvline(x=dens_loz, ls='-')
    ax[1].hist(dens['ELG_IS_LOZ'], bins=bins, range=lim, #color=line.get_color(),
               alpha=0.5, label='ELG_IS_LOZ (z<0.6)')

    #line = ax[1].axvline(x=dens_hiz, ls='-')
    ax[1].hist(dens['ELG_IS_HIZ'], bins=bins, range=lim, #color=line.get_color(),
               alpha=0.5, label='ELG_IS_HIZ (z>1.6)')

    ax[1].set_ylabel('Number of Healpixels')
    ax[1].set_xlabel('Targets / deg$^2$')
    ax[1].set_title('ELG Contaminants')
    ax[1].legend(loc='upper right')

    pngfile = os.path.join(qadir, '{}_detail_density.png'.format('elg'))
    fig.savefig(pngfile)
        
    return pngfile

def _img2html(html, pngfile, log):
    log.info('Writing {}'.format(pngfile))
    html.write('<a><img width=1024 src="{}" href="{}"></a>\n'.format(
        os.path.basename(pngfile), os.path.basename(pngfile) ))

def qa_targets_truth(output_dir, verbose=True):
    """Generate QA plots from the joined targets and truth catalogs.

    time select_mock_targets --output_dir debug --qa

    """
    import fitsio

    from desiutil.log import get_logger, DEBUG
    from desitarget import desi_mask, bgs_mask, mws_mask, contam_mask

    if verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    qadir = os.path.join(output_dir, 'qa')
    if os.path.exists(qadir):
        if os.listdir(qadir):
            log.warning('Output directory {} is not empty.'.format(qadir))
    else:
        log.info('Creating directory {}'.format(qadir))
        os.makedirs(qadir)
    log.info('Writing to output QA directory {}'.format(qadir))

    # Read the catalogs.
    targfile = os.path.join(output_dir, 'targets.fits')
    truthfile = os.path.join(output_dir, 'truth.fits')
    skyfile = os.path.join(output_dir, 'sky.fits')
    stddarkfile = os.path.join(output_dir, 'standards-dark.fits')
    stdbrightfile = os.path.join(output_dir, 'standards-bright.fits')

    cat = list()
    for ff in (targfile, truthfile, skyfile, stddarkfile, stdbrightfile):
        if os.path.exists(ff):
            log.info('Reading {}'.format(ff))
            cat.append( fitsio.read(ff, ext=1, upper=True) )
        else:
            log.warning('File {} not found.'.format(ff))
            cat.append( None )

    targets, truth, sky, stddark, stdbright = [cc for cc in cat]

    # Do some sanity checking of the catalogs.
    nobj, nsky, ndark, nbright = [len(cc) for cc in (targets, sky, stddark, stdbright)]
    if nobj != len(truth):
        log.fatal('Mismatch in the number of objects in targets.fits (N={}) and truth.fits (N={})!'.format(
            nobj, len(truth)))
        raise ValueError

    # Assign healpixels to estimate the area covered by the catalog.
    nside = 256
    npix = hp.nside2npix(nside)
    areaperpix = hp.nside2pixarea(nside, degrees=True)
    pix = radec2pix(nside, targets['RA'], targets['DEC'])
    counts = np.bincount(pix, weights=None, minlength=npix)
    area = np.sum(counts > 10) * areaperpix
    log.info('Approximate area spanned by catalog = {:.2f} deg2'.format(area))

    binarea = 1.0

    htmlfile = os.path.join(qadir, 'index.html')
    log.info('Building {}'.format(htmlfile))
    html = open(htmlfile, 'w')
    html.write('<html><body>\n')
    html.write('<h1>QA directory: {}</h1>\n'.format(qadir))

    html.write('<ul>\n')
    html.write('<li>Approximate total area = {:.1f} deg2</li>\n'.format(area))
    html.write('<li>Science targets = {}</li>\n'.format(nobj))
    html.write('<li>Sky targets = {}</li>\n'.format(nsky))
    html.write('<li>Dark standards = {}</li>\n'.format(ndark))
    html.write('<li>Bright standards = {}</li>\n'.format(nbright))
    html.write('</ul>\n')

    # Desired target densities, including contaminants.
    html.write('<hr>\n')
    html.write('<hr>\n')
    html.write('<h2>Raw target densities (including contaminants)</h2>\n')
    targdens = {'ELG': 2400, 'LRG': 350, 'QSO': 260, 'SKY': 1400} 

    if nobj > 0:
        html.write('<h3>Science targets - ELG, LRG, QSO, BGS_ANY, MWS_ANY</h3>\n')
        for obj in ('ELG', 'LRG', 'QSO', 'BGS_ANY', 'MWS_ANY'):
            these = np.where((targets['DESI_TARGET'] & desi_mask.mask(obj)) != 0)[0]
            if len(these) > 0:
                pngfile = qadensity(targets[these], obj, targdens, qadir=qadir, max_bin_area=binarea)
                _img2html(html, pngfile, log)
    
        html.write('<h3>Science targets - BGS_BRIGHT, BGS_FAINT</h3>\n')
        for obj in ('BGS_BRIGHT', 'BGS_FAINT'):
            these = np.where((targets['BGS_TARGET'] & bgs_mask.mask(obj)) != 0)[0]
            if len(these) > 0:
                pngfile = qadensity(targets[these], obj, targdens, qadir=qadir, max_bin_area=binarea)
                _img2html(html, pngfile, log)
    
        html.write('<h3>Science targets - MWS_MAIN, MWS_MAIN_VERY_FAINT, MWS_NEARBY, MWS_WD</h3>\n')
        for obj in ('MWS_MAIN', 'MWS_MAIN_VERY_FAINT', 'MWS_NEARBY', 'MWS_WD'):
            these = np.where((targets['MWS_TARGET'] & mws_mask.mask(obj)) != 0)[0]
            if len(these) > 0:
                pngfile = qadensity(targets[these], obj, targdens, qadir=qadir, max_bin_area=binarea)
                _img2html(html, pngfile, log)
        html.write('<hr>\n')

    if nsky > 0:
        html.write('<h3>Sky targets</h3>\n')
        obj = 'SKY'
        these = np.where((sky['DESI_TARGET'] & desi_mask.mask(obj)) != 0)[0]
        if len(these) > 0:
            pngfile = qadensity(sky[these], obj, targdens, qadir=qadir, max_bin_area=binarea)
            _img2html(html, pngfile, log)
        html.write('<hr>\n')

    if ndark > 0 or nbright > 0:
        html.write('<h3>Standard Stars</h3>\n')
        for cat, nn, obj in zip( (stddark, stdbright), (ndark, nbright),
                                 ('STD_FSTAR', 'STD_BRIGHT') ):
            if nn > 0:
                these = np.where((cat['DESI_TARGET'] & desi_mask.mask(obj)) != 0)[0]
                if len(these) > 0:
                    pngfile = qadensity(cat[these], obj, targdens, qadir=qadir, max_bin_area=binarea)
                    _img2html(html, pngfile, log)
    
    html.write('<hr>\n')
    html.write('<hr>\n')

    # Desired target densities, including contaminants.
    html.write('<h2>Detailed target densities</h2>\n')

    html.write('<h3>QSOs</h3>\n')
    pngfile = qa_qso(targets, truth, qadir=qadir)
    _img2html(html, pngfile, log)
    html.write('<hr>\n')

    html.write('<h3>ELGs</h3>\n')
    pngfile = qa_elg(targets, truth, qadir=qadir)
    _img2html(html, pngfile, log)

    html.write('<hr>\n')

    html.write('</html></body>\n')
    html.close()
