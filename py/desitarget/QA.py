# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==================
desitarget.QA
==================

Module dealing with Quality Assurance tests for Target Selection
"""
from __future__ import (absolute_import, division)
#
from time import time
import numpy as np
import fitsio
import os
import re
import random
import textwrap
import warnings
import itertools
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rfn
import healpy as hp

from collections import defaultdict
from glob import glob
from scipy.optimize import leastsq
from scipy.spatial import ConvexHull

from astropy import units as u
from astropy.coordinates import SkyCoord

from desiutil import brick
from desiutil.log import get_logger, DEBUG
from desiutil.plots import init_sky, plot_sky_binned, plot_healpix_map, prepare_data
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask
from desitarget.cmx.cmx_targetmask import cmx_mask


def _parse_tcnames(tcstring=None, add_all=True):
    """Turn a comma-separated string of target class names into a list.

    Parameters
    ----------
    tcstring : :class:`str`, optional, defaults to `"ELG,QSO,LRG,MWS,BGS,STD,(ALL)"`
        Comma-separated names of target classes e.g. QSO,LRG.
        Options are `ELG`, `QSO`, `LRG`, `MWS`, `BGS`, `STD`.
    add_all : :class:`boolean`, optional, defaults to ``True``
        If ``True``, then include `ALL` in the default names.

    Returns
    -------
    :class:`list`
        The string of names is converted to a list.

    Notes
    -----
        - One use of this function is to check for valid target class
          strings. An IOError is raised if a string is invalid.
    """

    tcdefault = ["ELG", "QSO", "LRG", "MWS", "BGS", "STD"]
    if add_all:
        tcdefault = ["ELG", "QSO", "LRG", "MWS", "BGS", "STD", "ALL"]

    if tcstring is None:
        tcnames = tcdefault
    else:    
        tcnames = [ bn for bn in tcstring.split(',') ]
        if not np.all([ tcname in tcdefault for tcname in tcnames ]):
            log.critical("passed tcnames should be one of {}".format(tcdefault))
            raise IOError

    return tcnames

def _load_systematics():
    """Loads information for making systematics plots.

    Returns
    -------
    :class:`dictionary` 
        A dictionary where the keys are the names of the systematics
        and the values are arrays of where to clip these systematics in plots
    """

    sysdict = {}

    sysdict['FRACAREA']=[0.01,1.,'Fraction of pixel area covered']
    sysdict['STARDENS']=[150.,4000.,'log10(Stellar Density) per sq. deg.']
    sysdict['EBV']=[0.001,0.1,'E(B-V)']
    sysdict['PSFDEPTH_G']=[63.,6300.,'PSF Depth in g-band']
    sysdict['PSFDEPTH_R']=[25.,2500.,'PSF Depth in r-band']
    sysdict['PSFDEPTH_Z']=[4.,400.,'PSF Depth in z-band']
    sysdict['GALDEPTH_G']=[63.,6300.,'Galaxy Depth in g-band']
    sysdict['GALDEPTH_R']=[25.,2500.,'Galaxy Depth in r-band']
    sysdict['GALDEPTH_Z']=[4.,400.,'Galaxy Depth in z-band']

    return sysdict

def _prepare_systematics(data,colname):
    """Functionally convert systematics to more user-friendly numbers
    
    Parameters
    ----------
    data :class:`~numpy.array` 
        An array of the systematic
    colname : :class:`str`
        The column name of the passed systematic, e.g. ``STARDENS``

    Returns
    -------
    :class:`~numpy.array` 
        The systematics converted by the appropriate function
    """

    #ADM depth columns need converted to a magnitude-like number
    if "DEPTH" in colname:
        #ADM zero and negative values should be a very low number (0)
        wgood = np.where(data > 0)[0]
        outdata = np.zeros(len(data))
        if len(wgood) > 0:
            outdata[wgood] = 22.5-2.5*np.log10(5./np.sqrt(data[wgood]))
    #ADM the STARDENS columns needs to be expressed as a log
    elif "STARDENS" in colname:
        #ADM zero and negative values should be a very negative number (-99)
        wgood = np.where(data > 0)[0]
        outdata = np.zeros(len(data))-99.
        if len(wgood) > 0:
            outdata[wgood] = np.log10(data[wgood])
    else:
        #ADM other columns don't need converted
        outdata = data

    return outdata

def _load_targdens(tcnames=None, cmx=False):
    """Loads the target info dictionary as in :func:`desimodel.io.load_target_info()` and
    extracts the target density information in a format useful for targeting QA plots

    Parameters
    ----------
    tcnames : :class:`list`
        A list of strings, e.g. "['QSO','LRG','ALL'] If passed, return only a dictionary
        for those specific bits
    cmx : :class:`boolean`, optional, defaults to ``False``
        If passed, load the commissioning bits (with zero density constraints) instead
        of the main survey/SV bits.

    Returns
    -------
    :class:`dictionary` 
        A dictionary where the keys are the bit names and the values are the densities           
    """

    if cmx == False:
        from desimodel import io
        targdict = io.load_target_info()

        targdens = {}
        targdens['ELG'] = targdict['ntarget_elg']
        targdens['LRG'] = targdict['ntarget_lrg']
        targdens['QSO'] = targdict['ntarget_qso'] + targdict['ntarget_badqso']
        targdens['BGS_ANY'] = targdict['ntarget_bgs_bright'] + targdict['ntarget_bgs_faint']
        targdens['STD_FAINT'] = 0.
        targdens['STD_BRIGHT'] = 0.
        targdens['MWS_ANY'] = targdict['ntarget_mws']
        # ADM set "ALL" to be the sum over all the target classes
        targdens['ALL'] = sum(list(targdens.values()))

        # ADM add in some sub-classes, not that ALL has been calculated
        targdens['LRG_1PASS'] = 0.
        targdens['LRG_2PASS'] = 0.

        targdens['BGS_FAINT'] = targdict['ntarget_bgs_faint']
        targdens['BGS_BRIGHT'] = targdict['ntarget_bgs_bright']

        targdens['MWS_MAIN'] = 0.
        targdens['MWS_MAIN_RED'] = 0.
        targdens['MWS_MAIN_BLUE'] = 0.
        targdens['MWS_WD'] = 0.
        targdens['MWS_NEARBY'] = 0.
    else:
        targdens = {k:0. for k in cmx_mask.names()}
    
    if tcnames is None:
        return targdens
    else:
        # ADM this is a dictionary comprehension
        return {key: value for key, value in targdens.items() if key in tcnames}

def _javastring():
    """Return a string that embeds a date in a webpage
    """

    js = textwrap.dedent("""
    <SCRIPT LANGUAGE="JavaScript">
    var months = new Array(13);
    months[1] = "January";
    months[2] = "February";
    months[3] = "March";
    months[4] = "April";
    months[5] = "May";
    months[6] = "June";
    months[7] = "July";
    months[8] = "August";
    months[9] = "September";
    months[10] = "October";
    months[11] = "November";
    months[12] = "December";
    var dateObj = new Date(document.lastModified)
    var lmonth = months[dateObj.getMonth() + 1]
    var date = dateObj.getDate()
    var fyear = dateObj.getYear()
    if (fyear < 2000)
    fyear = fyear + 1900
    if (date == 1 || date == 21 || date == 31)
    document.write(" " + lmonth + " " + date + "st, " + fyear)
    else if (date == 2 || date == 22)
    document.write(" " + lmonth + " " + date + "nd, " + fyear)
    else if (date == 3 || date == 23)
    document.write(" " + lmonth + " " + date + "rd, " + fyear)
    else
    document.write(" " + lmonth + " " + date + "th, " + fyear)    
    </SCRIPT>
    """)

    return js

def collect_mock_data(targfile):
    """Given a full path to a mock target file, read in all relevant mock data

    Parameters
    ----------
    targs : :class:`str`
        The full path to a mock target file in the DESI X per cent survey directory structure
        e.g., /global/projecta/projectdirs/desi/datachallenge/dc17b/targets/

    Returns
    -------
    :class:`~numpy.array` 
        A rec array containing the mock targets
    :class:`~numpy.array` 
        A rec array containing the mock truth objects used to generate the mock targets

    Notes
    -----
        - Will return 0 if the file structure is incorrect (i.e. if the "truths" can't be
          found based on the "targs")
    """

    start = time()

    # ADM set up the default logger from desiutil
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    # ADM retrieve the directory that contains the targets
    targdir = os.path.dirname(targfile)
    if targdir == '':
        targdir = '.'
    
    # ADM retrieve the mock data release name
    dcdir = os.path.dirname(targdir)
    dc = os.path.basename(dcdir)

    # ADM the file containing truth
    truthfile = '{}/truth.fits'.format(targdir)

    # ADM check that the truth file exists
    if not os.path.exists(truthfile):
        log.warning("Directory structure to truth file is not as expected")
        return 0

    # ADM read in the relevant mock data and return it
    targs = fitsio.read(targfile)
    log.info('Read in mock targets...t = {:.1f}s'.format(time()-start))
    truths = fitsio.read(truthfile)
    log.info('Read in mock truth objects...t = {:.1f}s'.format(time()-start))

    return targs, truths

def qaskymap(cat, objtype, qadir='.', upclip=None, weights=None, max_bin_area=1.0, fileprefix="skymap"):
    """Visualize the target density with a skymap. First version lifted 
    shamelessly from :mod:`desitarget.mock.QA` (which was originally written by `J. Moustakas`)

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``RA`` and ``DEC`` columns for coordinate
        information
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    upclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the targets at the "high density" end to make plots 
        conform to similar density scales
    weights : :class:`~numpy.array`, optional, defaults to None
        A weight for each of the passed targets (e.g., to upweight each target in a
        partial pixel at the edge of the DESI footprint)
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in the passed coordinates is chosen automatically to be as close as
        possible to this value without exceeding it
    fileprefix : :class:`str`, optional, defaults to ``"radec"`` for (RA/Dec)
        String to be added to the front of the output file name

    Returns
    -------
    Nothing
        But a .png plot of target densities is written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{objtype}.png``
    """

    label = '{} (targets/deg$^2$)'.format(objtype)
    fig, ax = plt.subplots(1)
    ax = np.atleast_1d(ax)
       
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        basemap = init_sky(galactic_plane_color='k', ax=ax[0]);
        plot_sky_binned(cat['RA'], cat['DEC'], weights=weights, max_bin_area=max_bin_area,
                        clip_lo='!1', clip_hi=upclip, cmap='jet', plot_type='healpix', 
                        label=label, basemap=basemap)

    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefix,objtype))
    fig.savefig(pngfile,bbox_inches='tight')

    plt.close()

    return

def qasystematics_skyplot(pixmap, colname, qadir='.', downclip=None, upclip=None,
                  fileprefix="systematics", plottitle=""):
    """Visualize systematics with a sky map

    Parameters
    ----------
    pixmap : :class:`~numpy.array`
        An array of systematics binned in HEALPixels, made by, e.g. `make_imaging_weight_map`.
        Assumed to be in the NESTED scheme and ORDERED BY INCREASING HEALPixel.
    colname : :class:`str`
        The name of the passed systematic, e.g. ``STARDENS``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    downclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the systematics at the low end
    upclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the systematics at the high end
    fileprefix : :class:`str`, optional, defaults to ``"histo"``
        String to be added to the front of the output file name
    plottitle : :class:`str`, optional, defaults to empty string
        An informative title for the plot

    Returns
    -------
    Nothing
        But a .png histogram of target densities is written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{colname}.png``
    """

    label = '{}'.format(plottitle)
    fig, ax = plt.subplots(1)
    ax = np.atleast_1d(ax)

    # ADM if downclip was passed as a number, turn it to a string with
    # ADM an exclamation mark to mask the plot background completely 
    if downclip is not None:
        if type(downclip) != str:
            downclip = '!' + str(downclip)

    # ADM prepare the data to be plotted by matplotlib routines
    pixmap = prepare_data(pixmap, clip_lo=downclip, clip_hi=upclip)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        basemap = init_sky(galactic_plane_color='k', ax=ax[0]);
        plot_healpix_map(pixmap, nest=True,  cmap='jet', label=label, basemap=basemap)

    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefix,colname))
    fig.savefig(pngfile,bbox_inches='tight')

    plt.close()

    return

def qasystematics_scatterplot(pixmap, syscolname, targcolname, qadir='.', 
                              downclip=None, upclip=None, nbins=10, 
                              fileprefix="sysdens", xlabel=None):
    """Make a target density vs. systematic scatter plot

    Parameters
    ----------
    pixmap : :class:`~numpy.array`
        An array of systematics binned in HEALPixels, made by, e.g. `make_imaging_weight_map`
    syscolname : :class:`str`
        The name of the passed systematic, e.g. ``STARDENS``
    targcolname : :class:`str`
        The name of the passed column of target densities, e.g. ``QSO``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    downclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the systematics at the low end
    upclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the systematics at the high end
    nbins : :class:`int`, optional, defaults to 10
        The number of bins to produce in the scatter plot
    fileprefix : :class:`str`, optional, defaults to ``"histo"``
        String to be added to the front of the output file name
    xlabel : :class:`str`, optional, if None defaults to ``syscolname``
        An informative title for the x-axis of the plot

    Returns
    -------
    Nothing
        But a .png histogram of target densities is written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{syscolname}-{targcolname}.png``
        
    Notes
    -----
    The passed ``pixmap`` must contain a column ``FRACAREA`` which is used to filter out any
    pixel with less than 90% areal coverage
    """
    # ADM set up the logger.
    from desiutil.log import get_logger, DEBUG
    log = get_logger()

    # ADM exit if we have a target density column that isn't populated.
    try:
        if np.all(pixmap[targcolname]==0):
            log.info("Target densities not populated for {}".format(targcolname))
            return
    # ADM also exit gracefully if a column name doesn't exist.
    except ValueError:
        log.info("Target densities not populated for {}".format(targcolname))
        return

    # ADM if no xlabel was passed, default to syscolname
    if xlabel is None:
        xlabel = syscolname

    # ADM remove anything that is in areas with low coverage, or doesn't meet
    # ADM the clipping criteria
    if downclip is None:
        downclip = -1e30
    if upclip is None:
        upclip = 1e30
    w = np.where(   (pixmap['FRACAREA'] > 0.9) & 
                    (pixmap[syscolname] >= downclip) & (pixmap[syscolname] < upclip) )[0]
    if len(w) > 0:
        pixmapgood = pixmap[w]
    else:
        log.error("Pixel map has no areas with >90% coverage for passed up/downclip")
        log.info("Proceeding without clipping systematics for {}".format(syscolname))
        w = np.where(pixmap['FRACAREA'] > 0.9)
        pixmapgood = pixmap[w]

    # ADM set up the x-axis as the systematic of interest
    xx = pixmapgood[syscolname]
    # ADM let np.histogram choose a sensible binning
    _, bins = np.histogram(xx, nbins)
    # ADM the bin centers rather than the edges
    binmid = np.mean(np.vstack([bins,np.roll(bins,1)]),axis=0)[1:]

    # ADM set up the y-axis as the deviation of the target density from median density
    yy = pixmapgood[targcolname]/np.median(pixmapgood[targcolname])

    # ADM determine which bin each systematics value is in
    wbin  = np.digitize(xx,bins)
    # ADM np.digitize closes the end bin whereas np.histogram
    # ADM leaves it open, so shift the end bin value back by one
    wbin[np.argmax(wbin)] -= 1

    # ADM apply thr digitization to the target density values
    # ADM note that the first digitized bin is 1 not zero
    meds = [np.median(yy[wbin==bin]) for bin in range(1,nbins+1)]

    # ADM make the plot
    plt.scatter(xx,yy,marker='.',color='b', alpha=0.8, s=0.8)
    plt.plot(binmid,meds,'k--',lw=2)

    # ADM set the titles and y range
    plt.ylim([0.5,1.5])
    plt.xlabel(xlabel)
    plt.ylabel("Relative {} density".format(targcolname))

    pngfile = os.path.join(qadir, '{}-{}-{}.png'
                           .format(fileprefix,syscolname,targcolname))
    plt.savefig(pngfile,bbox_inches='tight')

    plt.close()

    return

def qahisto(cat, objtype, qadir='.', targdens=None, upclip=None, weights=None, max_bin_area=1.0, 
            fileprefix="histo", catispix=False):
    """Visualize the target density with a histogram of densities. First version taken 
    shamelessly from :mod:`desitarget.mock.QA` (which was originally written by `J. Moustakas`)

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``RA`` and ``DEC`` columns for coordinate
        information
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    targdens : :class:`dictionary`, optional, defaults to None
        A dictionary of DESI target classes and the goal density for that class. Used, if
        passed, to label the goal density on the histogram plot        
    upclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the targets at the "high density" end to make plots 
        conform to similar density scales
    weights : :class:`~numpy.array`, optional, defaults to None
        A weight for each of the passed targets (e.g., to upweight each target in a
        partial pixel at the edge of the DESI footprint)
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in the passed coordinates is chosen automatically to be as close as
        possible to this value without exceeding it
    fileprefix : :class:`str`, optional, defaults to ``"histo"``
        String to be added to the front of the output file name
    catispix : :class:`boolean`, optional, defaults to ``False``
        If this is ``True``, then ``cat`` corresponds to the HEALpixel numbers already
        precomputed using ``pixels = footprint.radec2pix(nside, cat["RA"], cat["DEC"])``
        from the RAs and Decs ordered as for ``weights``, rather than the catalog itself.
        If this is True, then max_bin_area must correspond to the `nside` used to
        precompute the pixel numbers

    Returns
    -------
    Nothing
        But a .png histogram of target densities is written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{objtype}.png``
    """

    import healpy as hp

    # ADM determine the nside for the passed max_bin_area
    for n in range(1, 25):
        nside = 2 ** n
        bin_area = hp.nside2pixarea(nside, degrees=True)
        if bin_area <= max_bin_area:
            break
        
    # ADM the number of HEALPixels and their area at this nside
    npix = hp.nside2npix(nside)
    bin_area = hp.nside2pixarea(nside, degrees=True)

    # ADM the HEALPixel number for each RA/Dec (this call to desimodel
    # ADM assumes nest=True, so "weights" should assume nest=True, too)
    if catispix:
        pixels = cat.copy()
    else:
        from desimodel import footprint
        pixels = footprint.radec2pix(nside, cat["RA"], cat["DEC"])
    counts = np.bincount(pixels, weights=weights, minlength=npix)
    dens = counts[np.flatnonzero(counts)] / bin_area

    label = '{} (targets/deg$^2$)'.format(objtype)

    # ADM clip the targets to avoid high densities, if requested
    if upclip:
        dens = np.clip(dens,1,upclip)

    # ADM set the number of bins for the histogram (determined from trial and error)
    nbins = 80
    # ADM low density objects (QSOs and standard stars) look better with fewer bins
    if np.max(dens) < 500:
        nbins = 40
    # ADM the density value of the peak histogram bin
    h, b = np.histogram(dens,bins=nbins)
    peak = np.mean(b[np.argmax(h):np.argmax(h)+2])
    ypeak = np.max(h)

    # ADM set up and make the plot
    plt.clf()
    # ADM only plot to just less than upclip, to prevent displaying pile-ups in that bin
    plt.xlim((0,0.95*upclip))
    # ADM give a little space for labels on the y-axis
    plt.ylim((0,ypeak*1.2))
    plt.xlabel(label)
    plt.ylabel('Number of HEALPixels')

    plt.hist(dens, bins=nbins, histtype='stepfilled', alpha=0.6, 
             label='Observed {} Density (Peak={:.0f} per sq. deg.)'.format(objtype,peak))
    if objtype in targdens.keys():
        plt.axvline(targdens[objtype], ymax=0.8, ls='--', color='k', 
                    label='Goal {} Density (Goal={:.0f} per sq. deg.)'.format(objtype,targdens[objtype]))
    plt.legend(loc='upper left', frameon=False)

    # ADM add some metric conditions which are considered a failure for this
    # ADM target class...only for classes that have an expected target density
    good = True
    if targdens[objtype] > 0.:
        # ADM determine the cumulative version of the histogram of densities
        cum = np.cumsum(h)/np.sum(h)
        # ADM extract which bins correspond to the "68%" of central values
        w = np.where( (cum > 0.15865) & (cum < 0.84135) )[0]
        if len(w) > 0:
            minbin, maxbin = b[w][0], b[w][-1]
            # ADM this is a good plot if the peak value is within the ~68% of central values
            good = (targdens[objtype] > minbin) & (targdens[objtype] < maxbin)
    
    # ADM write out the plot
    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefix,objtype))
    if good:
        plt.savefig(pngfile,bbox_inches='tight')
    # ADM write out a plot with a yellow warning box
    else:
        plt.savefig(pngfile,bbox_inches='tight',facecolor='yellow')

    plt.close()

    return

def qamag(cat, objtype, qadir='.', fileprefix="nmag"):
    """Make magnitude-based DESI targeting QA plots given a passed set of targets.

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``FLUX_G``, ``FLUX_R``, ``FLUX_Z`` and 
        ``FLUX_W1``, columns for magnitude information
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    fileprefix : :class:`str`, optional, defaults to ``"nmag"`` for
        String to be added to the front of the output file name

    Returns
    -------
    Nothing
        But .png plots of target colors are written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{filter}-{objtype}.png``
        where filter might be, e.g., ``g``. ASCII versions of those files are
        also written with columns of magnitude bin and target number density. The
        file is called ``{qadir}/{fileprefix}-{filter}-{objtype}.dat``
    """

    # ADM columns in the passed cat as an array 
    cols = np.array(list(cat.dtype.names))

    # ADM value of flux to clip at for plotting purposes
    loclip = 1e-16

    # ADM magnitudes for which to plot histograms
    filters = ['G','R','Z','W1']
    magnames = [ 'FLUX_' + filter for filter in filters ]
        
    for fluxname in magnames:

        # ADM convert to magnitudes (fluxes are in nanomaggies)
        # ADM should be fine to clip for plotting purposes
        mag = 22.5-2.5*np.log10(cat[fluxname].clip(loclip))

        # ADM the name of the filters
        filtername = fluxname[5:].lower()
        #ADM WISE bands have upper-case filter names
        if filtername[0] == 'w':
            filtername = filtername.upper()
        
        # ADM plot the magnitude histogram
        # ADM set the number of bins for the redshift histogram to run in 
        # ADM 0.5 intervals from 14 to 14 + 0.5*bins
        nbins, binsize, binstart = 24, 0.5, 14
        bins = np.arange(nbins)*binsize+binstart
        # ADM insert a 0 bin and a 100 bin to catch the edges
        bins = np.insert(bins,0,0.)
        bins = np.insert(bins,len(bins),100.)

        # ADM the density value of the peak redshift histogram bin
        h, b = np.histogram(mag,bins=bins)
        peak = np.mean(b[np.argmax(h):np.argmax(h)+2])
        ypeak = np.max(h)

        # ADM set up and make the plot
        plt.clf()
        # ADM restrict the magnitude limits
        plt.xlim(14, 25)
        # ADM give a little space for labels on the y-axis
        plt.ylim((0,ypeak*1.2))
        plt.xlabel(filtername)
        plt.ylabel('N('+filtername+')')
        plt.hist(mag, bins=bins, histtype='stepfilled', alpha=0.6, 
             label='Observed {} {}-mag Distribution (Peak {}={:.0f})'.format(objtype,filtername,filtername,peak))
        plt.legend(loc='upper left', frameon=False)

        pngfile = os.path.join(qadir, '{}-{}-{}.png'.format(fileprefix,filtername,objtype))
        plt.savefig(pngfile,bbox_inches='tight')
        plt.close()

        # ADM create an ASCII file binned 0.1 mags
        nbins, binmin, binmax = 100, 14, 24
        h, b = np.histogram(mag,bins=nbins, range=(binmin,binmax) )
        bincent =  ((np.roll(b,1)+b)/2)[1:]
        datfile = pngfile.replace("png","dat")
        np.savetxt(datfile,np.vstack((bincent,h)).T,
                   fmt='%.2f',header='{}   N({})'.format(filtername,filtername))

    return

def qagaia(cat, objtype, qadir='.', fileprefix="gaia"):
    """Make Gaia-based DESI targeting QA plots given a passed set of targets.

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least "RA", "PARALLAX", 
        "PMRA" and "PMDEC" 
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    fileprefix : :class:`str`, optional, defaults to ``"gaia"``
        String to be added to the front of the output file name

    Returns
    -------
    Nothing
        But .png plots of Gaia information are written to ``qadir``. Two plots are made:
           The file containing distances from parallax is called:
                 ``{qadir}/{fileprefix}-{parallax}-{objtype}.png``
           The file containing proper motion information is called:
                 ``{qadir}/{fileprefix}-{pm}-{objtype}.png``
    """

    # ADM change the parallaxes (which are in mas) to distances in parsecs
    # ADM clip at very small parallaxes to avoid divide-by-zero
    r = 1000./np.clip(cat["PARALLAX"],1e-16,1e16)
    # ADM set the angle element of the plot to RA
    theta = np.radians(cat["RA"])

    # ADM set up the plot in polar projection
    ax = plt.subplot(111, projection='polar')
    ax.scatter(theta, r, s=2, alpha=0.6)

    # ADM only plot out to 110 pc
    ax.set_rmax(125)
    # ADM add a grid of distances
    rticknum = np.arange(1,6)*25
    rticknames = ["{}".format(num) for num in rticknum]
    # ADM include the parsec unit for the outermost distance label
    rticknames[-1] +='pc'
    # ADM the most flexible set of rtick controllers is in the ytick attribute
    ax.set_yticks(rticknum)
    ax.set_yticklabels(rticknames)
    ax.grid(True)

    # ADM save the plot
    ax.set_title("Distances at each RA based on Gaia parallaxes", va='bottom')
    pngfile = os.path.join(qadir,'{}-{}-{}.png'.format(fileprefix,'parallax',objtype))
    plt.savefig(pngfile,bbox_inches='tight')
    plt.close()

    # ADM plot the proper motions in RA/Dec against each other
    plt.clf()
    plt.xlabel(r'$PM_{RA}\,(mas\,yr^{-1})$')
    plt.ylabel(r'$PM_{DEC}\,(mas\,yr^{-1})$')

    ralim = (-25, 25)
    declim = (-25, 25)
    nobjs = len(cat)

    # ADM make a contour plot if we have lots of points...
    if nobjs > 1000:
        hb = plt.hexbin(cat["PMRA"], cat["PMDEC"],
                        mincnt=1, cmap=plt.cm.get_cmap('RdYlBu'),
                        bins='log', extent=(*ralim, *declim), gridsize=60)
        cb = plt.colorbar(hb)
        cb.set_label(r'$\log_{10}$ (Number of Sources)')

    # ADM...otherwise make a scatter plot
    else:
        plt.scatter(cat["PMRA"], cat["PMDEC"], alpha=0.6)

    # ADM save the plot
    pngfile = os.path.join(qadir,'{}-{}-{}.png'.format(fileprefix,'pm',objtype))
    plt.savefig(pngfile,bbox_inches='tight')
    plt.close()

    return

def mock_qafractype(cat, objtype, qadir='.', fileprefix="mock-fractype"):
    """Targeting QA Bar plot of the fraction of each classification type assigned to (mock) targets

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``TRUESPECTYPE``
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    fileprefix : :class:`str`, optional, defaults to ``"mock-fractype"`` for
        String to be added to the front of the output file name

    Returns
    -------
    Nothing
        But .png plots of target colors are written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{objtype}.png``
    """

    # ADM for this type of object, create the names of the possible contaminants
    from desitarget import contam_mask
    types = np.array(contam_mask.names())
    # ADM this is something of a hack as it assumes we'll keept
    # ADM the first 3 letters of each object ("BGS", "ELG" etc.) as sacred
    wtypes = np.where(np.array([ objtype[:3] in type[:3] for type in types ]))

    # ADM only make a plot if we have contaminant information
    if len(wtypes[0]) > 0:
        # ADM the relevant contaminant types for this object class
        types = types[wtypes]

        # ADM count each type of object
        ntypes = len(types)
        typecnt = []
        for typ in types:
            w = np.where(  (cat["CONTAM_TARGET"] & contam_mask[typ]) != 0 )
            typecnt.append(len(w[0]))

        # ADM express each type as a fraction of all objects in the catalog
        frac = np.array(typecnt)/len(cat)

        # ADM set up and make the bar plot with the legend
        plt.clf()
        plt.ylabel('fraction')
        if np.max(frac) > 0:
            plt.ylim(0,1.2*np.max(frac))
            
        x = np.arange(ntypes)
        plt.bar(x,frac,alpha=0.6,
                label='Fraction of {} classified as'.format(objtype))
        plt.legend(loc='upper left', frameon=False)
    
        # ADM add the names of the types to the x-axis
        # ADM first converting the strings to unicode if they're byte-type
        if isinstance(types[0],bytes):
            types = [ type.decode() for type in types ]
        
        # ADM to save space, only plot the name of the contaminant on the x-axis 
        # ADM not the object type (e.g. label as LRG not "QSO_IS_LRG"
        labels = [typ.split('_')[-1] for typ in types]
        plt.xticks(x, labels)

    # ADM if there was no contaminant info for this objtype, 
    # ADM then make an empty plot with a message
    else:
        log = get_logger()
        log.warning('No contaminant information for objects of type {}'.format(objtype))
        plt.clf()
        plt.ylabel('fraction')
        plt.xlim(0.,1.)
        plt.ylim(0.,1.)
        plt.text(0.5,0.5,'NO DATA')

    # ADM write out the plot
    pngfile = os.path.join(qadir,'{}-{}.png'.format(fileprefix,objtype))
    plt.savefig(pngfile,bbox_inches='tight')
    plt.close()

    return

def mock_qanz(cat, objtype, qadir='.', fileprefixz="mock-nz", fileprefixzmag="mock-zvmag"):
    """Make N(z) and z vs. mag DESI QA plots given a passed set of MOCK TRUTH.

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``TRUEZ`` for redshift information 
        and ``MAG`` for magnitude information
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    fileprefixz : :class:`str`, optional, defaults to ``"color"`` for
        String to be added to the front of the output N(z) plot file name
    fileprefixzmag : :class:`str`, optional, defaults to ``"color"`` for
        String to be added to the front of the output z vs. mag plot file name

    Returns
    -------
    Nothing
        But .png plots of target colors are written to ``qadir``. Two plots are made:
           The file containing N(z) is called:
                 ``{qadir}/{fileprefixz}-{objtype}.png``
           The file containing z vs. zerr is called:
                 ``{qadir}/{fileprefixzmag}-{objtype}.png``
    """

    # ADM the number of passed objects
    nobjs = len(cat)

    # ADM plot the redshift histogram

    # Get the unique combination of template types and subtypes
    templatetypes = np.char.strip(np.char.decode(cat['TEMPLATETYPE']))
    templatesubtypes = np.char.strip(np.char.decode(cat['TEMPLATESUBTYPE']))

    truez = cat["TRUEZ"]
    binsz = 0.04
    
    # ADM set up and make the plot
    plt.clf()
    plt.xlabel('True Redshift z')
    plt.ylabel('N(z)')
    for templatetype in sorted(set(templatetypes)):
        for templatesubtype in set(templatesubtypes):
            these = np.where( (templatetype == templatetypes) * (templatesubtype == templatesubtypes) )[0]
            if len(these) > 0:
                if templatesubtype == '':
                    label = '{} is {}'.format(objtype, templatetype)
                else:
                    label = '{} is {}-{}'.format(objtype, templatetype, templatesubtype)
                    
                nbin = np.max( (np.rint( np.ptp(truez[these]) / binsz).astype(int), 1) )
                
                nn, bins = np.histogram(truez[these], bins=nbin, 
                                        range=(truez[these].min(), truez[these].max()))
                cbins = (bins[:-1] + bins[1:]) / 2.0
                plt.bar(cbins, nn, align='center', alpha=0.75, label=label,
                        width=binsz, linewidth=0)
                        
    plt.legend(loc='upper right', frameon=True)

    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefixz,objtype))
    plt.savefig(pngfile,bbox_inches='tight')
    plt.close()

    # ADM plot the z vs. mag scatter plot
    plt.clf()
    plt.ylabel('Normalization magnitude')
    plt.xlabel('True Redshift z')
    plt.set_cmap('inferno')

    zlim =  (-0.05, cat["TRUEZ"].max()*1.05)
    maglim =  (cat["MAG"].min(), cat["MAG"].max()+0.75)

    # ADM make a contour plot if we have lots of points...
    if nobjs > 1000:
        #plt.hist2d(cat["TRUEZ"], cat["MAG"], bins=100, norm=LogNorm())
        #plt.colorbar()
        hb = plt.hexbin(cat["TRUEZ"], cat["MAG"], mincnt=1, cmap=plt.cm.get_cmap('RdYlBu'),
                        bins='log', extent=(*zlim, *maglim), gridsize=60)
        cb = plt.colorbar(hb)
        cb.set_label(r'$\log_{10}$ (Number of Targets)')
        
    # ADM...otherwise make a scatter plot
    else:
        for templatetype in sorted(set(templatetypes)):
            for templatesubtype in set(templatesubtypes):
                these = np.where( (templatetype == templatetypes) * (templatesubtype == templatesubtypes) )[0]
                if len(these) > 0:
                    if templatesubtype == '':
                        label = '{} is {}'.format(objtype, templatetype)
                    else:
                        label = '{} is {}-{}'.format(objtype, templatetype, templatesubtype)
                    plt.scatter(cat["TRUEZ"][these], cat["MAG"][these], alpha=0.6, label=label)
                    
        #plt.plot(cat["TRUEZ"],cat["MAG"],'bo', alpha=0.6)
        plt.xlim(zlim)
        plt.ylim(maglim)
        plt.legend(loc='upper right', frameon=True, ncol=3)

    # ADM create the plot
    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefixzmag,objtype))
    plt.savefig(pngfile,bbox_inches='tight')
    plt.close()

    return

def qacolor(cat, objtype, extinction, qadir='.', fileprefix="color", nodustcorr=False):
    """Make color-based DESI targeting QA plots given a passed set of targets.

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``FLUX_G``, ``FLUX_R``, ``FLUX_Z`` and 
        ``FLUX_W1``, ``FLUX_W2`` columns for color information
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``
    extinction : :class:`~numpy.array`
        An array containing the extinction in each band of interest, must contain at least the columns
        MW_TRANSMISSION_G, MW_TRANSMISSION_R, MW_TRANSMISSION_Z, MW_TRANSMISSION_W1, MW_TRANSMISSION_W2
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots
    fileprefix : :class:`str`, optional, defaults to ``"color"`` for
        String to be added to the front of the output file name
    nodustcorr : :class:`boolean`, optional, defaults to False
        Do not correct for dust extinction.

    Returns
    -------
    Nothing
        But .png plots of target colors are written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{bands}-{objtype}.png``
        where bands might be, e.g., ``grz``
    """
    from matplotlib.patches import Polygon

    def elg_colorbox(ax, plottype='gr-rz', verts=None):
        """Draw the ELG selection box."""
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if plottype == 'gr-rz':
            coeff0, coeff1 = (1.15, -0.15), (-1.2, 1.6)
            rzmin, rzpivot = 0.3, (coeff1[1] - coeff0[1]) / (coeff0[0] - coeff1[0])
            verts = [(rzmin, ylim[0]),
                     (rzmin, np.polyval(coeff0, rzmin)),
                     (rzpivot, np.polyval(coeff1, rzpivot)),
                     ((ylim[0] - 0.1 - coeff1[1]) / coeff1[0], ylim[0] - 0.1)
                ]
        if verts:
            ax.add_patch(Polygon(verts, fill=False, ls='--', lw=3, color='k'))

    def qso_colorbox(ax, plottype='gr-rz', verts=None):
        """Draw the QSO selection boxes."""

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if plottype == 'gr-rz':
            verts = [(-0.3, 1.3),
                     (1.1, 1.3),
                     (1.1, ylim[0]-0.05),
                     (-0.3, ylim[0]-0.05)
                    ]

        if plottype == 'r-W1W2':
            verts = None
            ax.axvline(x=22.7, ls='--', lw=3, color='k')
            ax.axhline(y=-0.4, ls='--', lw=3, color='k')

        if plottype == 'gz-grzW':
            gzaxis = np.linspace(-0.5, 2.0, 50)
            ax.plot(gzaxis, np.polyval([1.0, -1.0], gzaxis), 
                     ls='--', lw=3, color='k')

        if verts:
            ax.add_patch(Polygon(verts, fill=False, ls='--', lw=3, color='k'))

    # ADM unextinct fluxes
    if nodustcorr:
        gflux = cat['FLUX_G']
        rflux = cat['FLUX_R']
        zflux = cat['FLUX_Z']
        w1flux = cat['FLUX_W1']
        w2flux = cat['FLUX_W2']
    else:
        gflux = cat['FLUX_G'] / extinction['MW_TRANSMISSION_G']
        rflux = cat['FLUX_R'] / extinction['MW_TRANSMISSION_R']
        zflux = cat['FLUX_Z'] / extinction['MW_TRANSMISSION_Z']
        w1flux = cat['FLUX_W1'] / extinction['MW_TRANSMISSION_W1']
        w2flux = cat['FLUX_W2'] / extinction['MW_TRANSMISSION_W2']

    # ADM the number of passed objects
    nobjs = len(cat)

    # ADM convert to magnitudes (fluxes are in nanomaggies)
    # ADM should be fine to clip for plotting purposes
    loclip = 1e-16
    g = 22.5-2.5*np.log10(gflux.clip(loclip))
    r = 22.5-2.5*np.log10(rflux.clip(loclip))
    z = 22.5-2.5*np.log10(zflux.clip(loclip))
    W1 = 22.5-2.5*np.log10(w1flux.clip(loclip))
    W2 = 22.5-2.5*np.log10(w2flux.clip(loclip))

    # Some color ranges -- need to be smarter here
    if objtype == 'LRG':
        grlim = (0.5, 3)
        rzlim = (0.5, 3)
        rW1lim = (1.5, 5.0)
    else:
        grlim = (-0.5, 1.6)
        rzlim = (-0.5, 1.5)
        rW1lim = (-1.0, 3.0)
        
    W1W2lim = (-1.0, 1.0)

    #-------------------------------------------------------
    # ADM set up the r-z, g-r plot
    plt.clf()
    plt.xlabel(r'$r - z$')
    plt.ylabel(r'$g - r$')
    # ADM make a contour plot if we have lots of points...
    if nobjs > 1000:    
        hb = plt.hexbin(r-z, g-r, mincnt=1, cmap=plt.cm.get_cmap('RdYlBu'),
                        bins='log', extent=(*grlim, *rzlim), gridsize=60)
        cb = plt.colorbar(hb)
        cb.set_label(r'$\log_{10}$ (Number of Galaxies)')

    # ADM...otherwise make a scatter plot
    else:
        plt.scatter(r-z, g-r, alpha=0.6)

    plt.xlim(rzlim)
    plt.ylim(grlim)
        
    if objtype == 'ELG':
        elg_colorbox(plt.gca(), plottype='gr-rz')
    if objtype == 'QSO':
        qso_colorbox(plt.gca(), plottype='gr-rz')

    # ADM make the plot
    pngfile = os.path.join(qadir, '{}-grz-{}.png'.format(fileprefix,objtype))
    plt.savefig(pngfile,bbox_inches='tight')
    plt.close()

    #-------------------------------------------------------
    # ADM set up the r-z, r-W1 plot
    plt.clf()
    plt.xlabel(r'$r - z$')
    plt.ylabel(r'$r - W_1$')
    # ADM make a contour plot if we have lots of points...
    if nobjs > 1000:
        hb = plt.hexbin(r-z, r-W1, mincnt=1, cmap=plt.cm.get_cmap('RdYlBu'),
                        bins='log', extent=(*rzlim, *rW1lim), gridsize=60)
        cb = plt.colorbar(hb)
        cb.set_label(r'$\log_{10}$ (Number of Galaxies)')
        
        #plt.set_cmap('inferno')
        #counts, xedges, yedges, image = \
        #    plt.hist2d(r-z,r-W1,bins=100,range=[[-1,3],[-1,3]],norm=LogNorm())
        #if np.sum(counts) > 0:
        #    plt.colorbar()
        #else:
        #    nobjs = 0
    # ADM...otherwise make a scatter plot
    else:
        plt.scatter(r-z, r-W1, alpha=0.6)

    plt.xlim(rzlim)
    plt.ylim(rW1lim)

    ## ADM...or we might not have any WISE data
    #if nobjs == 0:
    #    log = get_logger()
    #    log.warning('No data within r-W1 vs. r-z ranges')
    #    plt.clf()
    #    plt.xlabel('r - z')
    #    plt.ylabel('r - W1')
    #    plt.xlim(rzlim)
    #    plt.ylim(rW1lim)
    #    plt.text(1.,1.,'NO DATA')

    # ADM save the plot
    pngfile=os.path.join(qadir, '{}-rzW1-{}.png'.format(fileprefix,objtype))
    plt.savefig(pngfile,bbox_inches='tight')
    plt.close()

    #-------------------------------------------------------
    # ADM set up the r-z, W1-W2 plot
    plt.clf()
    plt.xlabel(r'$r - z$')
    plt.ylabel(r'$W_1 - W_2$')
    # ADM make a contour plot if we have lots of points...
    if nobjs > 1000:
        hb = plt.hexbin(r-z, W1-W2, mincnt=1, cmap=plt.cm.get_cmap('RdYlBu'),
                        bins='log', extent=(*rzlim, *W1W2lim), gridsize=60)
        cb = plt.colorbar(hb)
        cb.set_label(r'$\log_{10}$ (Number of Galaxies)')
        
        #plt.set_cmap('inferno')
        #counts, xedges, yedges, image = \
        #    plt.hist2d(r-z,W1-W2,bins=100,range=[[-1,3],[-1,3]],norm=LogNorm())
        #if np.sum(counts) > 0:
        #    plt.colorbar()
        #else:
        #    nobjs = 0
    # ADM...otherwise make a scatter plot
    else:
        plt.scatter(r-z, W1-W2, alpha=0.6)

    plt.xlim(rzlim)
    plt.ylim(W1W2lim)

    ## ADM...or we might not have any WISE data
    #if nobjs == 0:
    #    log = get_logger()
    #    log.warning('No data within r-W1 vs. r-z ranges')
    #    plt.clf()
    #    plt.xlabel('r - z')
    #    plt.ylabel('W1 - W2')
    #    plt.xlim([-1,3])
    #    plt.ylim([-1,3])
    #    plt.text(1.,1.,'NO DATA')

    # ADM save the plot
    pngfile=os.path.join(qadir, '{}-rzW1W2-{}.png'.format(fileprefix,objtype))
    plt.savefig(pngfile,bbox_inches='tight')
    plt.close()

def _in_desi_footprint(targs):
    """Convenience function for using is_point_in_desi to find which targets are in the footprint
    Parameters
    ----------
    targs : :class:`~numpy.array` or `str`
        Targets in the DESI data model format, or any array that contains "RA" and "DEC" columns

    Returns
    -------
    :class:`integer`
        The INDICES of the input targs that are in the DESI footprint
    """
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()
    log.info('Start restricting to DESI footprint...t = {:.1f}s'.format(time()-start))

    # ADM restrict targets to just the DESI footprint
    from desimodel import io, footprint
    indesi = footprint.is_point_in_desi(io.load_tiles(),targs["RA"],targs["DEC"])
    windesi = np.where(indesi)
    if len(windesi[0]) > 0:
        log.info("{:.3f}% of targets are in official DESI footprint".format(100.*len(windesi[0])/len(targs)))
    else:
        log.error("ZERO input targets are within the official DESI footprint!!!")

    log.info('Restricted targets to DESI footprint...t = {:.1f}s'.format(time()-start))

    return windesi

def make_qa_plots(targs, qadir='.', targdens=None, max_bin_area=1.0, weight=True,
                  imaging_map_file=None, truths=None, tcnames=None, cmx=False):
    """Make DESI targeting QA plots given a passed set of targets.

    Parameters
    ----------
    targs : :class:`~numpy.array` or `str`
        An array of targets in the DESI data model format. If a string is passed then the
        targets are read from the file with the passed name (supply the full directory path).
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    targdens : :class:`dictionary`, optional, set automatically by the code if not passed
        A dictionary of DESI target classes and the goal density for that class. Used to
        label the goal density on histogram plots.
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in the passed coordinates is chosen automatically to be as close as
        possible to this value without exceeding it.
    weight : :class:`boolean`, optional, defaults to True
        If this is set, weight pixels using the ``DESIMODEL`` HEALPix footprint file to
        ameliorate under dense pixels at the footprint edges.
    imaging_map_file : :class:`str`, optional, defaults to no weights
        If `weight` is set, then this file contains the location of the imaging HEALPixel
        map (e.g. made by :func:` desitarget.randoms.pixmap()` if this is not
        sent, then the weights default to 1 everywhere (i.e. no weighting).
    truths : :class:`~numpy.array` or `str`
        The truth objects from which the targs were derived in the DESI data model format. 
        If a string is passed then read from that file (supply the full directory path).
    tcnames : :class:`list`, defaults to None
        A list of strings, e.g. ['QSO','LRG','ALL'] If passed, return only the QA pages
        for those specific bits. A useful speed-up when testing.
    cmx : :class:`boolean`, optional, defaults to ``False``
        If passed, load the commissioning bits (with zero density constraints) instead
        of the main survey/SV bits.

    Returns
    -------
    :class:`float`
        The total area of the survey used to make the QA plots.

    Notes
    -----
        - The ``DESIMODEL`` environment variable must be set to find the default expected 
          target densities.
        - On execution, a set of .png plots for target QA are written to `qadir`.
    """

    # ADM set up the default logger from desiutil.
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()
    log.info('Start making targeting QA plots...t = {:.1f}s'.format(time()-start))

    # ADM if a filename was passed, read in the targets from that file.
    if isinstance(targs, str):
        targs = fitsio.read(targs)
        log.info('Read in targets...t = {:.1f}s'.format(time()-start))
    if truths is not None:
        if isinstance(truths, str):
            truths = fitsio.read(truths)
            log.info('Read in truth...t = {:.1f}s'.format(time()-start))

    # ADM determine the nside for the passed max_bin_area.
    for n in range(1, 25):
        nside = 2 ** n
        bin_area = hp.nside2pixarea(nside, degrees=True)
        if bin_area <= max_bin_area:
            break
        
    # ADM calculate HEALPixel numbers once, here, to avoid repeat calculations
    # ADM downstream.
    from desimodel import footprint
    pix = footprint.radec2pix(nside, targs["RA"], targs["DEC"])
    log.info('Calculated HEALPixel for each target...t = {:.1f}s'
             .format(time()-start))
    # ADM set up the weight of each HEALPixel, if requested.
    weights = np.ones(len(targs))
    # ADM a count of the uniq pixels that are covered, useful for area calculations.
    uniqpixset = np.array(list(set(pix)))
    # ADM the total pixel weight assuming none of the areas are fractional
    # ADM or need rewighted (i.e. each pixel's weight is 1).
    totalpixweight = len(uniqpixset)

    if weight:
        # ADM load the imaging weights file.
        if imaging_map_file is not None:
            from desitarget import io as dtio
            pixweight = dtio.load_pixweight_recarray(imaging_map_file,nside)["FRACAREA"]
            # ADM determine what HEALPixels each target is in, to set the weights.
            fracarea = pixweight[pix]
            # ADM weight by 1/(the fraction of each pixel that is in the DESI footprint)
            # ADM except for zero pixels, which are all outside of the footprint.
            w = np.where(fracarea == 0)
            fracarea[w] = 1 # ADM to guard against division by zero warnings.
            weights = 1./fracarea
            weights[w] = 0

            # ADM if we have weights, then redetermine the total pix weight.
            totalpixweight = np.sum(pixweight[uniqpixset])

            log.info('Assigned weights to pixels based on DESI footprint...t = {:.1f}s'
                     .format(time()-start))

    # ADM calculate the total area (useful for determining overall average densities
    # ADM from the total number of targets/the total area).
    pixarea = hp.nside2pixarea(nside,degrees=True)
    totarea = pixarea*totalpixweight

    # ADM Current goal target densities for DESI.
    if targdens is None:
        targdens = _load_targdens(tcnames=tcnames, cmx=cmx)

    # ADM clip the target densities at an upper density to improve plot edges
    # ADM by rejecting highly dense outliers
    upclipdict = {k:5000. for k in targdens}
    if cmx:
        main_mask = cmx_mask
    else:
        main_mask = desi_mask
        upclipdict = {'ELG': 4000, 'LRG': 1200, 'QSO': 400, 'ALL': 8000,
                      'STD_FAINT': 200, 'STD_BRIGHT': 50,
                      'LRG_1PASS': 1000, 'LRG_2PASS': 500,
                      'BGS_FAINT': 2500, 'BGS_BRIGHT': 2500, 'BGS_ANY': 5000,
                      'MWS_ANY': 2000, 'MWS_MAIN': 10000, 'MWS_WD': 50, 'MWS_NEARBY': 50,
                      'MWS_MAIN_RED': 4000, 'MWS_MAIN_BLUE': 4000}

    for objtype in targdens:
        if 'ALL' in objtype:
            w = np.arange(len(targs))
        else:
            if ('BGS' in objtype) and not('ANY' in objtype) and not(cmx):
                w = np.where(targs["BGS_TARGET"] & bgs_mask[objtype])[0]
            elif ('MWS' in objtype) and not('ANY' in objtype) and not(cmx):
                w = np.where(targs["MWS_TARGET"] & mws_mask[objtype])[0]
            else:
                w = np.where(targs["DESI_TARGET"] & main_mask[objtype])[0]

        if len(w) > 0:
            # ADM make RA/Dec skymaps
            qaskymap(targs[w], objtype, qadir=qadir, upclip=upclipdict[objtype], 
                     weights=weights[w], max_bin_area=max_bin_area)
            log.info('Made sky map for {}...t = {:.1f}s'.format(objtype,time()-start))

            # ADM make histograms of densities. We already calculated the correctly 
            # ADM ordered HEALPixels and so don't need to repeat that calculation
            qahisto(pix[w], objtype, qadir=qadir, targdens=targdens, upclip=upclipdict[objtype], 
                    weights=weights[w], max_bin_area = max_bin_area, catispix=True)
            log.info('Made histogram for {}...t = {:.1f}s'.format(objtype,time()-start))

            # ADM make color-color plots
            qacolor(targs[w], objtype, targs[w], qadir=qadir, fileprefix="color")
            log.info('Made color-color plot for {}...t = {:.1f}s'.format(objtype,time()-start))

            # ADM make magnitude histograms
            qamag(targs[w], objtype, qadir=qadir, fileprefix="nmag")
            log.info('Made magnitude histogram plot for {}...t = {:.1f}s'.format(objtype,time()-start))

            if truths is not None:
                # ADM make noiseless color-color plots
                qacolor(truths[w], objtype, targs[w], qadir=qadir,
                        fileprefix="mock-color", nodustcorr=True)
                log.info('Made (mock) color-color plot for {}...t = {:.1f}s'.format(objtype,time()-start))

                # ADM make N(z) plots
                mock_qanz(truths[w], objtype, qadir=qadir, fileprefixz="mock-nz",
                          fileprefixzmag="mock-zvmag")
                log.info('Made (mock) redshift plots for {}...t = {:.1f}s'.format(objtype,time()-start))

                ## ADM plot what fraction of each selected object is actually a contaminant
                #mock_qafractype(truths[w], objtype, qadir=qadir, fileprefix="mock-fractype")
                #log.info('Made (mock) classification fraction plots for {}...t = {:.1f}s'.format(objtype,time()-start))
                
            # ADM make Gaia-based plots if we have Gaia columns
            if "PARALLAX" in targs.dtype.names:
                qagaia(targs[w], objtype, qadir=qadir, fileprefix="gaia")
                log.info('Made Gaia-based plots for {}...t = {:.1f}s'.format(objtype,time()-start))

    log.info('Made QA plots...t = {:.1f}s'.format(time()-start))
    return totarea

def make_qa_page(targs, mocks=False, makeplots=True, max_bin_area=1.0, qadir='.', 
                 clip2foot=False, weight=True, imaging_map_file=None, 
                 tcnames=None, systematics=True):
    """Create a directory containing a webpage structure in which to embed QA plots.

    Parameters
    ----------
    targs : :class:`~numpy.array` or `str`
        An array of targets in the DESI data model format. If a string is passed then the
        targets are read from the file with the passed name (supply the full directory path).
    mocks : :class:`boolean`, optional, default=False
        If ``True``, add plots that are only relevant to mocks at the bottom of the webpage.
    makeplots : :class:`boolean`, optional, default=True
        If ``True``, then create the plots as well as the webpage.
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in the passed coordinates is chosen automatically to be as close as
        possible to this value without exceeding it.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    clip2foot : :class:`boolean`, optional, defaults to False
        use :mod:`desimodel.footprint.is_point_in_desi` to restrict the passed targets to
        only those that lie within the DESI spectroscopic footprint.
    weight : :class:`boolean`, optional, defaults to True
        If this is set, weight pixels using to ameliorate under dense pixels at the footprint 
        edges. This uses the `imaging_map_file` HEALPix file for real targets and the default 
        ``DESIMODEL`` HEALPix footprint file for mock targets.
    imaging_map_file : :class:`str`, optional, defaults to no weights
        If `weight` is set, then this file contains the location of the imaging HEALPixel
        map (e.g. made by :func:`desitarget.randoms.pixmap()`. If this is not sent, 
        then the weights default to 1 everywhere (i.e. no weighting) for the real targets.
        If this is not set, then systematics plots cannot be made.
    tcnames : :class:`list`
        A list of strings, e.g. ['QSO','LRG','ALL'] If passed, return only the QA pages
        for those specific bits. A useful speed-up when testing
    systematics : :class:`boolean`, optional, defaults to ``True``
        If sent, then add plots of systematics to the front page.

    Returns
    -------
    Nothing
        But the page `index.html` and associated pages and plots are written to ``qadir``

    Notes
    -----
    If making plots, then the ``DESIMODEL`` environment variable must be set to find 
    the file of HEALPixels that overlap the DESI footprint
    """

    from desispec.io.util import makepath
    # ADM set up the default logger from desiutil.
    from desiutil.log import get_logger, DEBUG
    log = get_logger(DEBUG)

    start = time()
    log.info('Start making targeting QA page...t = {:.1f}s'.format(time()-start))

    # ADM if mocks was passed, build the full set of relevant data.
    if mocks:
        if isinstance(targs, str):
            mockdata = collect_mock_data(targs)
            if mockdata == 0:
                mocks = False
            else:
                targs, truths = mockdata
        else:
            log.warning('To make mock-related plots, targs must be a directory+file-location string...')
            log.warning('...will proceed by only producing the non-mock plots...')
            mocks = False
    else:
        truths = None

    # ADM if a filename was passed, read in the targets from that file.
    if isinstance(targs, str):
        targs = fitsio.read(targs)
        log.info('Read in targets...t = {:.1f}s'.format(time()-start))

    # ADM automatically detect whether we're running this for the main survey 
    # ADM or SV, etc. and change the column names accordingly.
    svs = "DESI"  # ADM this is to store sv iteration or cmx as a string.
    colnames = np.array(targs.dtype.names)
    svcolnames = colnames[ ['SV' in name or 'CMX' in name for name in colnames] ]
    # ADM set cmx flag to True if 'CMX_TARGET' is a column and rename that column.
    cmx = 'CMX_TARGET' in svcolnames
    # ADM use the commissioning mask bits/names if we have a CMX file.
    if cmx:
        main_mask = cmx_mask
    else:
        main_mask = desi_mask
    targs = rfn.rename_fields(targs, {'CMX_TARGET':'DESI_TARGET'})
    # ADM strip "SVX" off any columns (rfn.rename_fields forgives missing fields).
    for field in svcolnames:
        svs = field.split('_')[0] 
        targs = rfn.rename_fields(targs, {field:"_".join(field.split('_')[1:])})

    # ADM determine the working nside for the passed max_bin_area.
    for n in range(1, 25):
        nside = 2 ** n
        bin_area = hp.nside2pixarea(nside, degrees=True)
        if bin_area <= max_bin_area:
            break

    # ADM if requested, restrict the targets (and mock files) to the DESI footprint.
    if clip2foot:
        w = _in_desi_footprint(targs)
        targs = targs[w]
        # ADM ASSUMES THAT the targets and truth objects are row-by-row parallel.
        if mocks:
            truths = truths[w]

    # ADM make a DR string based on the RELEASE column.
    # ADM potentially there are multiple DRs in a file.
    if mocks:
        DRs = 'DR Mock'
    else:
        if 'RELEASE' in targs.dtype.names:
            DRs = ", ".join([ "DR{}".format(release) for release in np.unique(targs["RELEASE"])//1000 ])
        else:
            DRs = "DR Unknown"

    # ADM Set up the names of the target classes and their goal densities using
    # ADM the goal target densities for DESI (read from the DESIMODEL defaults).
    targdens = _load_targdens(tcnames=tcnames, cmx=cmx)

    # ADM set up the html file and write preamble to it.
    htmlfile = makepath(os.path.join(qadir, 'index.html'))

    # ADM grab the magic string that writes the last-updated date to a webpage.
    js = _javastring()

    # ADM html preamble.
    htmlmain = open(htmlfile, 'w')
    htmlmain.write('<html><body>\n')
    htmlmain.write('<h1>{} Targeting QA pages ({})</h1>\n'.format(svs,DRs))

    # ADM links to each collection of plots for each object type.
    htmlmain.write('<b><h2>Jump to a target class:</h2></b>\n')
    htmlmain.write('<ul>\n')
    for objtype in targdens.keys():
        htmlmain.write('<li><A HREF="{}.html"><b>{}</b></A>\n'.format(objtype,objtype))
    htmlmain.write('</ul>\n')

    # ADM for each object type, make a separate page
    for objtype in targdens.keys():        
        # ADM call each page by the target class name, stick it in the requested directory
        htmlfile = os.path.join(qadir,'{}.html'.format(objtype))
        html = open(htmlfile, 'w')

        # ADM html preamble
        html.write('<html><body>\n')
        html.write('<h1>DESI Targeting QA pages - {} ({})</h1>\n'.format(objtype,DRs))

        # ADM Target Densities
        html.write('<h2>Target density plots</h2>\n')
        html.write('<table COLS=2 WIDTH="100%">\n')
        html.write('<tr>\n')
        # ADM add the plots...
        html.write('<td align=center><A HREF="skymap-{}.png"><img SRC="skymap-{}.png" width=100% height=auto></A></td>\n'
                   .format(objtype,objtype))
        html.write('<td align=center><A HREF="histo-{}.png"><img SRC="histo-{}.png" width=75% height=auto></A></td>\n'
                   .format(objtype,objtype))
        html.write('</tr>\n')
        html.write('</table>\n')

        # ADM color-color plots
        html.write('<h2>Target color-color plots (corrected for Galactic extinction)</h2>\n')
        html.write('<table COLS=3 WIDTH="100%">\n')
        html.write('<tr>\n')
        # ADM add the plots...
        for colors in ["grz","rzW1","rzW1W2"]:
            html.write('<td align=center><A HREF="color-{}-{}.png"><img SRC="color-{}-{}.png" width=95% height=auto></A></td>\n'
                       .format(colors,objtype,colors,objtype))
        html.write('</tr>\n')
        html.write('</table>\n')

        # ADM magnitude plots
        html.write('<h2>Magnitude histograms (NOT corrected for Galactic extinction)</h2>\n')
        html.write('<table COLS=4 WIDTH="100%">\n')
        html.write('<tr>\n')
        # ADM add the plots 
        for band in ["g","r","z","W1"]:
            html.write('<td align=center><A HREF="nmag-{}-{}.png"><img SRC="nmag-{}-{}.png" width=95% height=auto></A></td>\n'
                       .format(band,objtype,band,objtype))
        html.write('</tr>\n')
        html.write('</table>\n')
        # ADM add the ASCII files to the images
        for band in ["g","r","z","W1"]:
            html.write('<td align=center><A HREF="nmag-{}-{}.dat">nmag-{}-{}.dat</A></td>\n'
                       .format(band,objtype,band,objtype))
        html.write('</tr>\n')
        html.write('</table>\n')

        # ADM parallax and proper motion plots, if we have that information
        if "PARALLAX" in targs.dtype.names:
            html.write('<h2>Gaia based plots</h2>\n')
            html.write('<table COLS=2 WIDTH="100%">\n')
            html.write('<tr>\n')
            # ADM add the plots...
            html.write('<td align=center><A HREF="gaia-pm-{}.png"><img SRC="gaia-pm-{}.png" width=75% height=auto></A></td>\n'
                       .format(objtype,objtype))
            html.write('<td align=center><A HREF="gaia-parallax-{}.png"><img SRC="gaia-parallax-{}.png" width=71% height=auto></A></td>\n'
                       .format(objtype,objtype))
            html.write('</tr>\n')
            html.write('</table>\n')

        # ADM add special plots if we have mock data
        if mocks:
            html.write('<hr>\n')
            html.write('<h1>DESI Mock QA\n')

            # ADM redshift plots
            html.write('<h2>True Redshift plots</h2>\n')
            html.write('<table COLS=2 WIDTH="100%">\n')
            html.write('<tr>\n')
            # ADM add the plots...
            html.write('<td align=center><A HREF="mock-nz-{}.png"><img SRC="mock-nz-{}.png" height=auto width=95%></A></td>\n'
                       .format(objtype,objtype))
            html.write('<td align=center><A HREF="mock-zvmag-{}.png"><img SRC="mock-zvmag-{}.png" height=auto width=95%></A></td>\n'
                       .format(objtype,objtype))
            html.write('</tr>\n')
            html.write('</table>\n')

            # ADM color-color plots
            html.write('<h2>(Truth) color-color plots (corrected for Galactic extinction)</h2>\n')
            html.write('<table COLS=3 WIDTH="100%">\n')
            html.write('<tr>\n')
            # ADM add the plots...
            for colors in ["grz","rzW1","rzW1W2"]:
                html.write('<td align=center><A HREF="mock-color-{}-{}.png"><img SRC="mock-color-{}-{}.png" height=auto width=95%></A></td>\n'
                       .format(colors,objtype,colors,objtype))
            html.write('</tr>\n')
            html.write('</table>\n')

            ## ADM classification fraction plots
            #html.write('<h2>Fraction of each spectral type plots</h2>\n')
            #html.write('<table COLS=1 WIDTH="40%">\n')
            #html.write('<tr>\n')
            ## ADM add the plots...
            #html.write('<td align=center><A HREF="{}-{}.png"><img SRC="{}-{}.png" height=auto width=95%></A></td>\n'
            #           .format("mock-fractype",objtype,"mock-fractype",objtype))
            #html.write('</tr>\n')
            #html.write('</table>\n')

        # ADM add target density vs. systematics plots, if systematics plots were requested.
        # ADM these plots aren't useful if we're looking at commissioning data.
        if systematics and not(cmx):
            # ADM fail if the pixel systematics weights file was not passed.
            if imaging_map_file is None:
                log.error("imaging_map_file was not passed so systematics cannot be tracked. Try again passing systematics=False.")
                raise IOError
            sysdic = _load_systematics()
            sysnames = list(sysdic.keys())
            # ADM html text to embed the systematics plots
            html.write('<h2>Target Density variation vs. Systematics plots</h2>\n')
            html.write('<table COLS=3 WIDTH="100%">\n')
            html.write('<tr>\n')
            # ADM add the plots...
            while(len(sysnames) > 2):
                for sys in sysnames[:3]:
                    html.write('<td align=center><A HREF="sysdens-{}-{}.png"><img SRC="sysdens-{}-{}.png" height=auto width=95%></A></td>\n'
                               .format(sys,objtype,sys,objtype))
                # ADM pop off the 3 columns of systematics that have already been written
                sysnames = sysnames[3:]
                html.write('</tr>\n')
            # ADM we popped three systematics at a time, there could be a remaining one or two
            if len(sysnames) == 2:
                for sys in sysnames:
                    html.write('<td align=center><A HREF="sysdens-{}-{}.png"><img SRC="sysdens-{}-{}.png" height=auto width=95%></A></td>\n'
                               .format(sys,objtype,sys,objtype))
                html.write('</tr>\n')
            if len(sysnames) == 1:
                html.write('<td align=center><A HREF="sysdens-{}-{}.png"><img SRC="sysdens-{}-{}.png" height=auto width=95%></A></td>\n'
                               .format(sysnames[0],objtype,sysnames[0],objtype))
                html.write('</tr>\n')
            html.write('</table>\n')

        # ADM html postamble
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    # ADM make the QA plots, if requested:
    if makeplots:
        totarea = make_qa_plots(targs, truths=truths, 
                                qadir=qadir, targdens=targdens, max_bin_area=max_bin_area,
                                weight=weight, imaging_map_file=imaging_map_file, cmx=cmx)

        # ADM add a correlation matrix recording the overlaps between different target
        # ADM classes as a density
        log.info('Making correlation matrix...t = {:.1f}s'.format(time()-start))
        htmlmain.write('<br><h2>Overlaps in target densities (per sq. deg.)</h2>\n')
        htmlmain.write('<PRE><span class="inner-pre" style="font-size: 16px">\n')
        # ADM only retain classes that are actually in the DESI target bit list
        settargdens = set(main_mask.names()).intersection(set(targdens))
        # ADM write out a list of the target categories
        headerlist = list(settargdens)
        headerlist.insert(0," ")
        header = " ".join(['{:>11s}'.format(i) for i in headerlist])+'\n\n'
        htmlmain.write(header)
        # ADM for each pair of target classes, determine how many targets per unit area
        # ADM have the relevant target bit set for both target classes in the pair
        for i, objtype1 in enumerate(settargdens):
            overlaps = [objtype1]
            for j, objtype2 in enumerate(settargdens):
                if j < i:
                    overlaps.append(" ")
                else:
                    dt = targs["DESI_TARGET"]
                    overlap = np.sum(((dt & main_mask[objtype1]) != 0) & ((dt & main_mask[objtype2]) != 0))/totarea
                    overlaps.append("{:.1f}".format(overlap))
            htmlmain.write(" ".join(['{:>11s}'.format(i) for i in overlaps])+'\n\n')
        # ADM close the matrix text output
        htmlmain.write('</span></PRE>\n\n\n')
        log.info('Done with correlation matrix...t = {:.1f}s'.format(time()-start))

    # ADM if requested, add systematics plots
    if systematics:
        from desitarget import io as dtio
        pixmap = dtio.load_pixweight_recarray(imaging_map_file,nside)
        sysdic = _load_systematics()
        sysnames = list(sysdic.keys())
        # ADM html text to embed the systematics plots
        htmlmain.write('<h2>Systematics plots</h2>\n')
        htmlmain.write('<table COLS=2 WIDTH="100%">\n')
        htmlmain.write('<tr>\n')
        # ADM add the plots...
        while(len(sysnames) > 1):
            for sys in sysnames[:2]:
                htmlmain.write('<td align=center><A HREF="systematics-{}.png"><img SRC="systematics-{}.png" height=auto width=95%></A></td>\n'
                               .format(sys,sys))
                # ADM pop off the 2 columns of systematics that have already been written
            sysnames = sysnames[2:]
            htmlmain.write('</tr>\n')
        # ADM we popped two systematics at a time, there could be a remaining one
        if len(sysnames)==1:
            htmlmain.write('<td align=center><A HREF="systematics-{}.png"><img SRC="systematics-{}.png" height=auto width=95%></A></td>\n'
                           .format(sysnames[0],sysnames[0]))
            htmlmain.write('</tr>\n')
        htmlmain.write('</table>\n')
        # ADM add the plots
        if makeplots:
            sysnames = list(sysdic.keys())
            for sysname in sysnames:
                # ADM convert the data and the systematics ranges to more human-readable quantities
                d, u , plotlabel = sysdic[sysname]
                down, up = _prepare_systematics(np.array([d,u]),sysname)
                pixmap[sysname] = _prepare_systematics(pixmap[sysname],sysname)
                # ADM make the systematics sky plots
                qasystematics_skyplot(pixmap[sysname],sysname,
                              qadir=qadir,downclip=down,upclip=up,plottitle=plotlabel)
                # ADM make the systematics vs. target density scatter plots
                # ADM for each target type. These plots aren't useful for commissioning.
                if not(cmx):
                    for objtype in targdens.keys():
                       # ADM hack to have different FRACAREA quantities for the sky maps and
                        # ADM the scatter plots
                        if sysname=="FRACAREA":
                            down = 0.9
                        qasystematics_scatterplot(pixmap,sysname,objtype,qadir=qadir,
                                        downclip=down,upclip=up,nbins=10,xlabel=plotlabel)

        log.info('Done with systematics...t = {:.1f}s'.format(time()-start))

    # ADM html postamble for main page
    htmlmain.write('<b><i>Last updated {}</b></i>\n'.format(js))
    htmlmain.write('</html></body>\n')
    htmlmain.close()
    print(htmlmain.closed)

    # ADM make sure all of the relevant directories and plots can be read by a web-browser
    cmd = 'chmod 644 {}/*'.format(qadir)
    ok = os.system(cmd)
    cmd = 'chmod 775 {}'.format(qadir)
    ok = os.system(cmd)
