# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desitarget.mock.QA
==================

Generate QA figures from the output of desitarget.mock.join_targets_truth.

"""
from __future__ import (absolute_import, division, print_function)

import os
import numpy as np
import warnings

def target_density(cat):
    """Determine the target density by grouping targets in healpix pixels.  The code
    below was code shamelessly taken from desiutil.plot.plot_sky_binned (by
    D. Kirkby).

    nside = 64 corresponds to about 0.210 deg2, about a factor of 3 larger
    than the nominal imaging brick area (0.25x0.25=0.625 deg2), as determined 
    by this snippet of code:

      max_bin_area = 0.5
      for n in range(1, 10):
          nside = 2 ** n
          bin_area = hp.nside2pixarea(nside, degrees=True)
          print(nside, bin_area)
          if bin_area <= max_bin_area:
              break

    """
    import healpy as hp
        
    nside = 128
    npix = hp.nside2npix(nside)
    bin_area = hp.nside2pixarea(nside, degrees=True)

    pixels = hp.ang2pix(nside, np.radians(90 - cat['DEC']), 
                        np.radians(cat['RA']), nest=False)
    counts = np.bincount(pixels, weights=None, minlength=npix)
    dens = counts[np.flatnonzero(counts)] / bin_area
            
    return dens

def qadensity(cat, objtype, targdens=None, max_bin_area=1.0, qadir='.'):
    """Visualize the target density with a skymap and histogram.
    
    """
    import matplotlib.pyplot as plt
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

def _img2html(html, pngfile, log):
    log.info('Writing {}'.format(pngfile))
    html.write('<a><img width=1024 src="{}" href="{}"></a>\n'.format(
        os.path.basename(pngfile), os.path.basename(pngfile) ))

def qa_targets_truth(output_dir, verbose=True, clobber=False):
    """Generate QA plots from the joined targets and truth catalogs.

    time select_mock_targets --output_dir debug --qa

    """
    import shutil
    import fitsio

    from desiutil.log import get_logger, DEBUG
    from desitarget import desi_mask, bgs_mask, mws_mask, contam_mask

    if verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    qadir = os.path.join(output_dir, 'qa')

    try:
        os.stat(qadir)
        if os.listdir(qadir):
            if clobber:
                shutil.rmtree(qadir)
                os.makedirs(qadir)
            else:
                log.warning('Output QA directory {} is not empty; please set clobber=True.'.format(qadir))
                return
    except:
        log.info('Creating QA directory {}'.format(qadir))
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
    nobj, nsky, ndark, nbright = len(targets), len(sky), len(stddark), len(stdbright)
    if nobj != len(truth):
        log.fatal('Mismatch in the number of objects in targets.fits (N={}) and truth.fits (N={})!'.format(nobj, len(truth)))
        raise ValueError

    # Pick a reasonable healpix area.
    area = targets['RA'].max() - targets['RA'].min()
    area *=  ( np.sin( targets['DEC'].max()*np.pi/180.) -
               np.sin( targets['DEC'].min()*np.pi/180.) ) * 180 / np.pi
    log.info('Approximate (rectangular) area spanned by catalog = {:.2f} deg2'.format(area))
    binarea = area / 10
    
    htmlfile = os.path.join(qadir, 'index.html')
    log.info('Building {}'.format(htmlfile))
    html = open(htmlfile, 'w')
    html.write('<html><body>\n')
    html.write('<h1>QA directory: {}</h1>\n'.format(qadir))

    html.write('<ul>\n')
    html.write('<li>Approximate (rectangular) area = {:.2f} deg2</li>\n'.format(area))
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
                pngfile = qadensity(targets[these], obj, targdens, qadir=qadir)
                _img2html(html, pngfile, log)
    
        html.write('<h3>Science targets - BGS_BRIGHT, BGS_FAINT</h3>\n')
        for obj in ('BGS_BRIGHT', 'BGS_FAINT'):
            these = np.where((targets['BGS_TARGET'] & bgs_mask.mask(obj)) != 0)[0]
            if len(these) > 0:
                pngfile = qadensity(targets[these], obj, targdens, qadir=qadir)
                _img2html(html, pngfile, log)
    
        html.write('<h3>Science targets - MWS_MAIN, MWS_MAIN_VERY_FAINT, MWS_NEARBY, MWS_WD</h3>\n')
        for obj in ('MWS_MAIN', 'MWS_MAIN_VERY_FAINT', 'MWS_NEARBY', 'MWS_WD'):
            these = np.where((targets['MWS_TARGET'] & mws_mask.mask(obj)) != 0)[0]
            if len(these) > 0:
                pngfile = qadensity(targets[these], obj, targdens, qadir=qadir)
                _img2html(html, pngfile, log)
        html.write('<hr>\n')

    if nsky > 0:
        html.write('<h3>Sky targets</h3>\n')
        obj = 'SKY'
        these = np.where((sky['DESI_TARGET'] & desi_mask.mask(obj)) != 0)[0]
        if len(these) > 0:
            pngfile = qadensity(targets[these], obj, targdens, qadir=qadir)
            _img2html(html, pngfile, log)
        html.write('<hr>\n')

    if ndark > 0 or nbright > 0:
        html.write('<h3>Standard Stars</h3>\n')
        for cat, nn, obj in zip( (stddark, stdbright), (ndark, nbright),
                                 ('STD_FSTAR', 'STD_BRIGHT') ):
            if nn > 0:
                these = np.where((cat['DESI_TARGET'] & desi_mask.mask(obj)) != 0)[0]
                if len(these) > 0:
                    pngfile = qadensity(cat[these], obj, targdens, qadir=qadir)
                    _img2html(html, pngfile, log)
        html.write('<hr>\n')
    
    html.write('<hr>\n')
    
    html.write('</html></body>\n')
    html.close()

    #import pdb ; pdb.set_trace()
    #import sys ; sys.exit(1)
