# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==================
desitarget.QA
==================

Module dealing with Quality Assurance tests for Target Selection
"""
from __future__ import (absolute_import, division)
from time import time
import numpy as np
import fitsio
import os
import re
import random
import textwrap
import warnings
import itertools
import numpy.lib.recfunctions as rfn
import healpy as hp
from collections import defaultdict
from glob import glob, iglob
from scipy.optimize import leastsq
from scipy.spatial import ConvexHull
from astropy import units as u
from astropy.coordinates import SkyCoord
from desiutil import brick
from desiutil.log import get_logger
from desitarget.internal import sharedmem
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask
from desitarget.targets import main_cmx_or_sv
from desitarget.io import read_targets_in_box, target_columns_from_header
from desitarget.geomask import pixarea2nside
# ADM fake the matplotlib display so it doesn't die on allocated nodes.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

try:
    from scipy import constants
    C_LIGHT = constants.c/1000.0
except TypeError:  # This can happen during documentation builds.
    C_LIGHT = 299792458.0/1000.0

# ADM set up the default logger from desiutil
log = get_logger()

_type2color = {'STAR': 'orange', 'GALAXY': 'red', 'QSO-LYA': 'green',
               'WD': 'purple', 'NOT ELG': 'gray'}   # , 'QSO': }


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
        tcnames = [bn for bn in tcstring.split(',')]
        if not np.all([tcname in tcdefault for tcname in tcnames]):
            msg = "passed tcnames should be one of {}".format(tcdefault)
            log.critical(msg)
            raise ValueError(msg)

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

    sysdict['FRACAREA'] = [0.01, 1., 'Fraction of pixel area covered']
    sysdict['STARDENS'] = [150., 4000., 'log10(Stellar Density) per sq. deg.']
    sysdict['EBV'] = [0.001, 0.1, 'E(B-V)']
    sysdict['PSFDEPTH_G'] = [63., 6300., 'PSF Depth in g-band']
    sysdict['PSFDEPTH_R'] = [25., 2500., 'PSF Depth in r-band']
    sysdict['PSFDEPTH_Z'] = [4., 400., 'PSF Depth in z-band']
    sysdict['GALDEPTH_G'] = [63., 6300., 'Galaxy Depth in g-band']
    sysdict['GALDEPTH_R'] = [25., 2500., 'Galaxy Depth in r-band']
    sysdict['GALDEPTH_Z'] = [4., 400., 'Galaxy Depth in z-band']
    sysdict['PSFSIZE_G'] = [0., 3., 'PSF Size in g-band']
    sysdict['PSFSIZE_R'] = [0., 3., 'PSF Size in r-band']
    sysdict['PSFSIZE_Z'] = [0., 3., 'PSF Size in z-band']

    return sysdict


def _prepare_systematics(data, colname):
    """Functionally convert systematics to more user-friendly numbers.

    Parameters
    ----------
    data :class:`~numpy.array`
        An array of the systematic.
    colname : :class:`str`
        The column name of the passed systematic, e.g. ``STARDENS``.

    Returns
    -------
    :class:`~numpy.array`
        The systematics converted by the appropriate function
    """

    # ADM depth columns need converted to a magnitude-like number.
    if "DEPTH" in colname:
        # ADM zero and negative values should be a very low number (0).
        wgood = np.where(data > 0)[0]
        outdata = np.zeros(len(data))
        if len(wgood) > 0:
            outdata[wgood] = 22.5-2.5*np.log10(5./np.sqrt(data[wgood]))
    # ADM the STARDENS columns needs to be expressed as a log.
    elif "STARDENS" in colname:
        # ADM zero and negative values should be a very negative number (-99).
        wgood = np.where(data > 0)[0]
        outdata = np.zeros(len(data))-99.
        if len(wgood) > 0:
            outdata[wgood] = np.log10(data[wgood])
    else:
        # ADM other columns don't need converted.
        outdata = data

    return outdata


def _load_targdens(tcnames=None, bit_mask=None):
    """Loads the target info dictionary as in :func:`desimodel.io.load_target_info()` and
    extracts the target density information in a format useful for targeting QA plots.

    Parameters
    ----------
    tcnames : :class:`list`
        A list of strings, e.g. "['QSO','LRG','ALL'] If passed, return only a dictionary
        for those specific bits.
    bit_mask : :class:`list` or `~numpy.array`, optional, defaults to ``None``
        If passed, load the bit names from this mask (with no associated expected
        densities) rather than loading the main survey bits and densities. Must be a
        desi mask object, e.g., loaded as `from desitarget.targetmask import desi_mask`.
        Any bit names that contain "NORTH" or "SOUTH" or calibration bits will be
        removed. A list of several masks can be passed rather than a single mask.

    Returns
    -------
    :class:`dictionary`
        A dictionary where the keys are the bit names and the values are the densities.

    Notes
    -----
        If `bit_mask` happens to correpond to the main survey masks, then the default
        behavior is triggered (as if `bit_mask=None`).
    """
    bit_masks = np.atleast_1d(bit_mask)

    if bit_mask is None or bit_masks[0]._name == 'desi_mask':
        from desimodel import io
        targdict = io.load_target_info()

        targdens = {}
        targdens['ELG'] = targdict['ntarget_elg']
        targdens['LRG'] = targdict['ntarget_lrg']
        targdens['QSO'] = targdict['ntarget_qso'] + targdict['ntarget_badqso']
        targdens['BGS_ANY'] = targdict['ntarget_bgs_bright'] + targdict['ntarget_bgs_faint']   # add BGS_WISE bit 'targdict['ntarget_bgs_wise'] to BGS_ANY
        targdens['MWS_ANY'] = targdict['ntarget_mws']
        # ADM set "ALL" to be the sum over all the target classes
        targdens['ALL'] = sum(list(targdens.values()))

        # ADM add in some sub-classes, now that ALL has been calculated
        targdens['STD_FAINT'] = 0.
        targdens['STD_BRIGHT'] = 0.

        targdens['LRG'] = 0.
#        targdens['LRG_1PASS'] = 0.
#        targdens['LRG_2PASS'] = 0.

        targdens['BGS_FAINT'] = targdict['ntarget_bgs_faint']
        targdens['BGS_BRIGHT'] = targdict['ntarget_bgs_bright']
        targdens['BGS_WISE'] = 0.
        # targdens['BGS_WISE'] = targdict['ntarget_bgs_wise']	#uncomment and modify for BGS_WISE bit

        targdens['MWS_BROAD'] = 0.
        targdens['MWS_MAIN_RED'] = 0.
        targdens['MWS_MAIN_BLUE'] = 0.
        targdens['MWS_WD'] = 0.
        targdens['MWS_NEARBY'] = 0.
    else:
        names = []
        for bit_mask in bit_masks:
            # ADM this is the list of words contained in bits that we don't want to consider for QA.
            badnames = ["NORTH", "SOUTH", "NO_TARGET", "SECONDARY",
                        "BRIGHT_OBJECT", "SKY", "KNOWN", "BACKUP", "SCND"]
            names.append([name for name in bit_mask.names()
                          if not any(badname in name for badname in badnames)])
        targdens = {k: 0. for k in np.concatenate(names)}

    if tcnames is None:
        return targdens
    else:
        out = {}
        for key, value in targdens.items():
            if key in tcnames:
                out.update({key: value})
            elif 'LRG' in key and 'LRG' in tcnames:
                out.update({key: value})
            elif 'MWS' in key and 'MWS' in tcnames:
                out.update({key: value})
            elif 'STD' in key and 'STD' in tcnames:
                out.update({key: value})
            elif 'BGS' in key and 'BGS' in tcnames:
                out.update({key: value})
        return out


def _load_dndz(tcnames=None):
    """Load the predicted redshift distributions for each target class.

    Parameters
    ----------
    tcnames : :class:`list`
        A list of strings, e.g. "['QSO','LRG','ALL'] If passed, return only a dictionary
        for those specific bits.

    Returns
    -------
    :class:`dictionary`
        A dictionary where the keys are the bit names and the values are the
        dndz as a function of redshift (as another dictionary with keys 'z', and
        'dndz', respectively.

    """
    import astropy.io
    import desimodel.io

    log = get_logger()

    densities = desimodel.io.load_target_info()

    alltarg = ('ELG', 'LRG', 'QSO', 'BGS_ANY', 'BGS_BRIGHT', 'BGS_FAINT')
    if tcnames is None:
        tcnames = alltarg

    out = dict()
    for targ, suffix in zip(alltarg, ('elg', 'lrg', 'qso', 'bgs', 'bgs_bright', 'bgs_faint')):
        dndzfile = os.path.join(desimodel.io.datadir(), 'targets', 'nz_{}.dat'.format(suffix))
        if not os.path.isfile(dndzfile):
            log.warning('Redshift distribution file {} not found!'.format(dndzfile))
        else:
            if targ in tcnames or 'ALL' in tcnames:
                if targ == 'LRG':
                    names = ('zmin', 'zmax', 'dndz', 'dndz_boss')
                else:
                    names = ('zmin', 'zmax', 'dndz')
                dat = astropy.io.ascii.read(dndzfile, names=names, format='basic',
                                            comment='#', delimiter=' ', guess=False)

                # Renormalize dn/dz to match the expected target densities.
                denskey = 'ntarget_{}'.format(targ.lower())
                if denskey in densities.keys():
                    norm = np.sum(dat['dndz'].data) / densities[denskey]
                else:
                    norm = 1.0

                dz = (dat['zmax'] - dat['zmin']).data
                zz = dat['zmin'].data + dz / 2
                out[targ] = {'z': zz, 'dndz': dat['dndz'].data / norm, 'dz': dz}

    return out


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


def read_data(targfile, mocks=False, downsample=None, header=False):
    """Read in the data, including any mock data (if present).

    Parameters
    ----------
    targfile : :class:`str`
        The full path to a mock target file in the DESI X per cent survey
        directory structure, e.g.,
        /global/projecta/projectdirs/desi/datachallenge/dc17b/targets/,
        or to a data file, or to a directory of HEALPixel-split target
        files which will be read in with
        :func:`desitarget.io.read_targets_in_box`.
    mocks : :class:`boolean`, optional, defaults to ``False``
        If ``True``, read in mock data.
    downsample : :class:`int`, optional, defaults to `None`
        If not `None`, downsample targets by (roughly) this value, e.g.
        for `downsample=10` a set of 900 targets would have ~90 random
        targets returned. A speed-up for experimenting with large files.
    header : :class:`bool`, optional, defaults to ``False``
        If ``True`` then return the header of the file as an additional
        output (targs, truths, objtruths, header) instead of (targs,
        truths, objtruths).

    Returns
    -------
    targs : :class:`~numpy.array`
        A rec array containing the targets catalog.
    truths : :class:`~numpy.array`
        A rec array containing the truths catalog (if present and `mocks=True`).
    objtruths : :class:`dict`
        Object type-specific truth metadata (if present and `mocks=True`).
    """
    start = time()

    # ADM set up the default logger from desiutil.
    log = get_logger()

    # ADM retrieve the directory that contains the targets.
    targdir = os.path.dirname(targfile)
    if targdir == '':
        targdir = '.'

    # ADM from the header of the input files, retrieve the appropriate
    # ADM names for the SV, main, or cmx _TARGET columns.
    targcols = target_columns_from_header(targfile)

    # ADM limit to the data columns used by the QA code to save memory.
    colnames = ["RA", "DEC", "RELEASE", "PARALLAX", "PMRA", "PMDEC"]
    for band in "G", "R", "Z", "W1", "W2":
        colnames.append("{}_{}".format("FLUX", band))
        colnames.append("{}_{}".format("MW_TRANSMISSION", band))
    cols = np.concatenate([colnames, targcols])

    # ADM read in the targets catalog and return it.
    if header:
        targs, hdr = read_targets_in_box(targfile, columns=cols,
                                         downsample=downsample, header=True)
    else:
        targs = read_targets_in_box(targfile, columns=cols, header=False)
    log.info('Read in targets...t = {:.1f}s'.format(time()-start))
    truths, objtruths = None, None

    if mocks:
        truthfile = targfile.replace('targets-', 'truth-')  # fragile!

        # ADM check that the truth file exists.
        if not os.path.exists(truthfile):
            log.warning("Directory structure to truth file is not as expected")
            return targs, None, None

        truthinfo = fitsio.FITS(truthfile)
        truths = truthinfo['TRUTH'].read()
        log.info('Read in truth catalog...t = {:.1f}s'.format(time()-start))

        objtruths = dict()
        for objtype in set(truths['TEMPLATETYPE']):
            try:
                oo = objtype.decode('utf-8').strip().upper()
            except AttributeError:
                oo = objtype

            extname = 'TRUTH_{}'.format(oo)
            if extname in truthinfo:
                objtruths[oo] = truthinfo[extname].read()

        return targs, truths, objtruths
    else:
        if header:
            return targs, None, None, hdr
        else:
            return targs, None, None


def qaskymap(cat, objtype, qadir='.', upclip=None, weights=None,
             max_bin_area=1.0, fileprefix="skymap"):
    """Visualize the target density with a skymap. First version lifted
    shamelessly from :mod:`desitarget.mock.QA` (which was originally
    written by `J. Moustakas`).

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``RA`` and ``DEC``
        columns for coordinate information.
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that
        corresponds to the passed ``cat``.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    upclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the targets at the "high density"
        end to make plots conform to similar density scales.
    weights : :class:`~numpy.array`, optional, defaults to None
        A weight for each of the passed targets (e.g., to upweight each
        target in a partial pixel at the edge of the DESI footprint).
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in RA/Dec in `targs` is chosen to be as close as
        possible to this value.
    fileprefix : :class:`str`, optional, defaults to ``"radec"`` for (RA/Dec)
        String to be added to the front of the output file name.

    Returns
    -------
    :class:`dict`
        Dictionary of the 10 densest pixels in the DESI tiling. Includes
        RA, DEC, DENSITY (per sq. deg.) and NSIDE for each HEALpixel.

    Notes
    -----
    In addition to the returned dictionary, a .png sky map is written to
    `qadir`. The file is called: `{qadir}/{fileprefix}-{objtype}.png`.
    """
    label = r'{} (targs deg$^{{-2}}$'.format(objtype)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # ADM grab the data needed to make the plot, masking values < 1
        # ADM to ensure that areas outside of the footprint are empty.
        from desiutil.plots import plot_sky_binned
        _, data = plot_sky_binned(cat['RA'], cat['DEC'], weights=weights,
                                  max_bin_area=max_bin_area, clip_lo='!1',
                                  plot_type='healpix', colorbar=False,
                                  return_grid_data=True)

        # ADM use the plot data to find the densest pixels...
        # ADM order indexes of unmasked HEALpixels by density.
        denspix = np.arange(len(data))[~data.mask][np.argsort(data[~data.mask].data)]
        # ADM find the RA/Dec of these ordered HEALPixels.
        nside = hp.npix2nside(len(data))
        theta, phi = hp.pix2ang(nside, denspix, nest=False)
        ra, dec = np.degrees(phi), 90-np.degrees(theta)
        # ADM clip to just Decs at which DESI will observe.
        infoot = _in_desi_footprint([ra, dec], radec=True)
        # ADM set up an output dictionary of the 10 densest pixels.
        outdict = {"RA": ra[infoot][-10:], "DEC": dec[infoot][-10:],
                   "DENSITY": data[denspix[infoot][-10:]].data, "NSIDE": nside}

        # ADM if upclip was passed, just clip at that raw value.
        if upclip is not None:
            data.data[data.data > upclip] = upclip
        # ADM otherwise, clip at percentiles and note them in the label.
        else:
            lo, hi = 5, 95
            # ADM catch the corner case of no unmasked data.
            if np.any(~data.mask):
                plo, phi = np.percentile(data[~data.mask], [lo, hi])
                label += r'; {}% ({:.0f} deg$^{{-2}}$) < dens '.format(lo, plo)
                label += r'< {}% ({:.0f} deg$^{{-2}}$)'.format(hi, phi)
                data.data[data.data > phi] = phi
                data.data[data.data < plo] = plo
        label += ')'

        # ADM Make the plot.
        from desiutil.plots import init_sky, plot_healpix_map
        ax = init_sky(galactic_plane_color='k')
        ax = plot_healpix_map(data, nest=False, cmap="jet", label=label, ax=ax)

    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefix, objtype))
    plt.savefig(pngfile, bbox_inches='tight')

    plt.close()

    return outdict


def qasystematics_skyplot(pixmap, colname, qadir='.', downclip=None, upclip=None,
                          fileprefix="systematics", plottitle=""):
    """Visualize systematics with a sky map.

    Parameters
    ----------
    pixmap : :class:`~numpy.array`
        An array of systematics binned in HEALPixels, made by, e.g. `make_imaging_weight_map`.
        Assumed to be in the NESTED scheme and ORDERED BY INCREASING HEALPixel.
    colname : :class:`str`
        The name of the passed systematic, e.g. ``STARDENS``.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    downclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the systematics at the low end.
    upclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the systematics at the high end.
    fileprefix : :class:`str`, optional, defaults to ``"histo"``
        String to be added to the front of the output file name.
    plottitle : :class:`str`, optional, defaults to empty string
        An informative title for the plot.

    Returns
    -------
    Nothing
        But a .png histogram of target densities is written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{colname}.png``.
    """

    label = '{}'.format(plottitle)

    # ADM if downclip was passed as a number, turn it to a string with
    # ADM an exclamation mark to mask the plot background completely.
    if downclip is not None:
        if type(downclip) != str:
            downclip = '!' + str(downclip)

    # ADM prepare the data to be plotted by matplotlib routines.
    from desiutil.plots import prepare_data
    pixmap = prepare_data(pixmap, clip_lo=downclip, clip_hi=upclip)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from desiutil.plots import init_sky, plot_healpix_map
        ax = init_sky(galactic_plane_color='k')
        ax = plot_healpix_map(pixmap, nest=True,  cmap='jet', label=label, ax=ax)

    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefix, colname))
    plt.savefig(pngfile, bbox_inches='tight')

    plt.close()

    return


def qasystematics_scatterplot(pixmap, syscolname, targcolname, qadir='.',
                              downclip=None, upclip=None, nbins=10,
                              fileprefix="sysdens", xlabel=None):
    """Make a target density vs. systematic scatter plot.

    Parameters
    ----------
    pixmap : :class:`~numpy.array`
        An array of systematics binned in HEALPixels, made by, e.g. `make_imaging_weight_map`.
    syscolname : :class:`str`
        The name of the passed systematic, e.g. ``STARDENS``.
    targcolname : :class:`str`
        The name of the passed column of target densities, e.g. ``QSO``.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    downclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the systematics at the low end.
    upclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the systematics at the high end.
    nbins : :class:`int`, optional, defaults to 10
        The number of bins to produce in the scatter plot.
    fileprefix : :class:`str`, optional, defaults to ``"histo"``
        String to be added to the front of the output file name.
    xlabel : :class:`str`, optional, if None defaults to ``syscolname``
        An informative title for the x-axis of the plot.

    Returns
    -------
    Nothing
        But a .png histogram of target densities is written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{syscolname}-{targcolname}.png``.

    Notes
    -----
    The passed ``pixmap`` must contain a column ``FRACAREA`` which is used to filter out any
    pixel with less than 90% areal coverage.
    """
    # ADM set up the logger.
    log = get_logger()

    # ADM exit if we have a target density column that isn't populated.
    try:
        if np.all(pixmap[targcolname] == 0):
            log.info("Target densities not populated for {}".format(targcolname))
            return
    # ADM also exit gracefully if a column name doesn't exist.
    except ValueError:
        log.info("Target densities not populated for {}".format(targcolname))
        return

    # ADM if no xlabel was passed, default to syscolname.
    if xlabel is None:
        xlabel = syscolname

    # ADM remove anything that is in areas with low coverage, or doesn't meet
    # ADM the clipping criteria.
    if downclip is None:
        downclip = -1e30
    if upclip is None:
        upclip = 1e30
    ii = ((pixmap['FRACAREA'] > 0.9) &
          (pixmap[syscolname] >= downclip) & (pixmap[syscolname] < upclip))
    if np.any(ii):
        pixmapgood = pixmap[ii]
    else:
        log.error("Pixel map has no areas with >90% coverage for passed up/downclip")
        log.info("Proceeding without clipping systematics for {}".format(syscolname))
        ii = pixmap['FRACAREA'] > 0.9
        pixmapgood = pixmap[ii]

    # ADM set up the x-axis as the systematic of interest.
    xx = pixmapgood[syscolname]
    # ADM let np.histogram choose a sensible binning.
    _, bins = np.histogram(xx, nbins)
    # ADM the bin centers rather than the edges.
    binmid = np.mean(np.vstack([bins, np.roll(bins, 1)]), axis=0)[1:]

    # ADM set up the y-axis as the deviation of the target density from median density.
    yy = pixmapgood[targcolname]/np.median(pixmapgood[targcolname])

    # ADM determine which bin each systematics value is in.
    wbin = np.digitize(xx, bins)
    # ADM np.digitize closes the end bin whereas np.histogram
    # ADM leaves it open, so shift the end bin value back by one.
    wbin[np.argmax(wbin)] -= 1

    # ADM apply the digitization to the target density values
    # ADM note that the first digitized bin is 1 not zero.
    # meds = [np.median(yy[wbin == bin]) for bin in range(1, nbins+1)]
    meds = list()
    for bin in range(1, nbins+1):
        ii = (wbin == bin)
        if np.any(ii):
            meds.append(np.median(yy[ii]))
        else:
            meds.append(np.NaN)

    # ADM make the plot.
    plt.scatter(xx, yy, marker='.', color='b', alpha=0.8, s=0.8)
    plt.plot(binmid, meds, 'k--', lw=2)

    # ADM set the titles and y range.
    plt.ylim([0.5, 1.5])
    plt.xlabel(xlabel)
    plt.ylabel("Relative {} density".format(targcolname))

    pngfile = os.path.join(qadir, '{}-{}-{}.png'
                           .format(fileprefix, syscolname, targcolname))
    plt.savefig(pngfile, bbox_inches='tight')

    plt.close()


def qahisto(cat, objtype, qadir='.', targdens=None, upclip=None, weights=None, max_bin_area=1.0,
            fileprefix="histo", catispix=False):
    """Visualize the target density with a histogram of densities. First version taken
    shamelessly from :mod:`desitarget.mock.QA` (which was originally written by `J. Moustakas`).

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``RA`` and ``DEC`` columns for coordinate
        information.
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    targdens : :class:`dictionary`, optional, defaults to None
        A dictionary of DESI target classes and the goal density for that class. Used, if
        passed, to label the goal density on the histogram plot.
    upclip : :class:`float`, optional, defaults to None
        A cutoff at which to clip the targets at the "high density" end to make plots
        conform to similar density scales.
    weights : :class:`~numpy.array`, optional, defaults to None
        A weight for each of the passed targets (e.g., to upweight each target in a
        partial pixel at the edge of the DESI footprint).
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size in RA/Dec in `targs` is chosen to be as close as possible to this value.
    fileprefix : :class:`str`, optional, defaults to ``"histo"``
        String to be added to the front of the output file name.
    catispix : :class:`boolean`, optional, defaults to ``False``
        If this is ``True``, then ``cat`` corresponds to the HEALpixel numbers already
        precomputed using ``pixels = footprint.radec2pix(nside, cat["RA"], cat["DEC"])``
        from the RAs and Decs ordered as for ``weights``, rather than the catalog itself.
        If this is True, then max_bin_area must correspond to the `nside` used to
        precompute the pixel numbers.

    Returns
    -------
    Nothing
        But a .png histogram of target densities is written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{objtype}.png``.
    """

    import healpy as hp

    # ADM determine the nside for the passed max_bin_area.
    nside = pixarea2nside(max_bin_area)

    # ADM the number of HEALPixels and their area at this nside.
    npix = hp.nside2npix(nside)
    bin_area = hp.nside2pixarea(nside, degrees=True)

    # ADM the HEALPixel number for each RA/Dec (this call to desimodel
    # ADM assumes nest=True, so "weights" should assume nest=True, too).
    if catispix:
        pixels = cat.copy()
    else:
        from desimodel import footprint
        pixels = footprint.radec2pix(nside, cat["RA"], cat["DEC"])
    counts = np.bincount(pixels, weights=weights, minlength=npix)
    dens = counts[np.flatnonzero(counts)] / bin_area

    dhi, dlo = dens.min(), dens.max()

    if upclip is not None:
        densmin, densmax = dhi, dlo
        xmin, xmax = 1, upclip
        xplotmin, xplotmax = 0, upclip
        label = r'{} (targets deg$^{{-2}}$)'.format(objtype)
    else:
        # ADM restrict density range to 0.5->99.5% to clip outliers.
        densmin, densmax = np.percentile(dens, [0.5, 99.5])
        xmin, xmax = densmin, densmax
        xplotmin, xplotmax = densmin*0.9, densmax*1.1
        label = r'{} (targets deg$^{{-2}}$; central 99% of densities)'.format(objtype)
    ddens = 0.05 * (densmax - densmin)

    if ddens == 0:  # small number of pixels
        nbins = 5
        ddens = 0.05 * (xmax - xmin)
    else:
        dbins = np.arange(densmin, densmax, ddens)  # bin left edges
        if len(dbins) < 10:
            ddens = (densmax - densmin) / 10
            dbins = np.arange(densmin, densmax, ddens)
        nbins = len(dbins)

    # ADM the density value of the peak histogram bin.
    h, b = np.histogram(dens, bins=nbins, range=(densmin, densmax))
    peak = np.mean(b[np.argmax(h):np.argmax(h)+2])
    ypeak = np.max(h)

    # ADM set up and make the plot.
    plt.clf()
    # ADM only plot to just less than upclip, to prevent displaying pile-ups in that bin.
    plt.xlim((xplotmin, xplotmax))
    # ADM give a little space for labels on the y-axis.
    plt.ylim((0, ypeak*1.2))
    plt.xlabel(label)
    plt.ylabel('Number of HEALPixels')

    label = r'Observed {} Density (Peak={:.0f} deg$^{{-2}}$)'.format(objtype, peak)
    nn, bins = np.histogram(dens, bins=nbins, range=(densmin, densmax))
    cbins = (bins[:-1] + bins[1:]) / 2.0
    plt.bar(cbins, nn, align='center', alpha=0.6, label=label, width=ddens)
    # plt.hist(dens, bins=nbins, histtype='stepfilled', alpha=0.6, label=label)

    if objtype in targdens.keys():
        plt.axvline(targdens[objtype], ymax=0.8, ls='--', color='k',
                    label=r'Goal {} Density (Goal={:.0f} deg$^{{-2}}$)'.format(
                        objtype, targdens[objtype]))
    plt.legend(loc='upper left', frameon=False, fontsize=10)

    # ADM add some metric conditions which are considered a failure for this
    # ADM target class...only for classes that have an expected target density.
    good = True
    if targdens[objtype] > 0.:
        # ADM determine the cumulative version of the histogram of densities.
        cum = np.cumsum(h)/np.sum(h)
        # ADM extract which bins correspond to the "68%" of central values.
        w = np.where((cum > 0.15865) & (cum < 0.84135))[0]
        if len(w) > 0:
            minbin, maxbin = b[w][0], b[w][-1]
            # ADM this is a good plot if the peak value is within the ~68% of central values.
            good = (targdens[objtype] > minbin) & (targdens[objtype] < maxbin)

    # ADM write out the plot.
    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefix, objtype))
    if good:
        plt.savefig(pngfile, bbox_inches='tight')
    # ADM write out a plot with a yellow warning box.
    else:
        plt.savefig(pngfile, bbox_inches='tight', facecolor='yellow')

    plt.close()


def qamag(cat, objtype, qadir='.', fileprefix="nmag", area=1.0):
    """Make magnitude-based DESI targeting QA plots given a passed set of targets.

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``FLUX_G``, ``FLUX_R``, ``FLUX_Z`` and
        ``FLUX_W1``, columns for magnitude information.
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    fileprefix : :class:`str`, optional, defaults to ``"nmag"`` for
        String to be added to the front of the output file name.
    area : :class:`float`
        Total area in deg2.

    Returns
    -------
    Nothing
        But .png plots of target colors are written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{filter}-{objtype}.png``
        where filter might be, e.g., ``g``. ASCII versions of those files are
        also written with columns of magnitude bin and target number density. The
        file is called ``{qadir}/{fileprefix}-{filter}-{objtype}.dat``.
    """

    # ADM columns in the passed cat as an array.
    cols = np.array(list(cat.dtype.names))

    # ADM value of flux to clip at for plotting purposes.
    loclip = 1e-16

    if 'NEARBY' in objtype:
        brightmag, faintmag, deltam = 6, 18, 0.5
    else:
        brightmag, faintmag, deltam = 14, 24, 0.5

    magbins = np.arange(brightmag, faintmag, deltam)  # bin left edges

    deltam_out = 0.1  # for output file
    magbins_out = np.arange(brightmag, faintmag, deltam_out)  # bin left edges

    # ADM magnitudes for which to plot histograms.
    filters = ['G', 'R', 'Z', 'W1']
    magnames = ['FLUX_' + filter for filter in filters]

    for fluxname in magnames:

        # ADM convert to magnitudes (fluxes are in nanomaggies).
        # ADM should be fine to clip for plotting purposes.
        mag = 22.5-2.5*np.log10(cat[fluxname].clip(loclip))
        mag = mag[np.isfinite(mag)]

        # ADM the name of the filters.
        filtername = fluxname[5:].lower()
        # ADM WISE bands have upper-case filter names.
        if filtername[0] == 'w':
            filtername = filtername.upper()

        dndm, dndmbins = np.histogram(mag, bins=magbins, range=(brightmag, faintmag))
        cdndmbins = (dndmbins[:-1] + dndmbins[1:]) / 2.0

        # ADM the density value of the peak.
        peak = np.mean(dndmbins[np.argmax(dndm):np.argmax(dndm)+2])
        ypeak = np.max(dndm) / area

        # ADM set up and make the plot.
        plt.clf()
        # ADM restrict the magnitude limits.
        plt.xlim(brightmag, faintmag+0.5)
        # ADM give a little space for labels on the y-axis.
        if ypeak != 0.0:
            plt.ylim((0, ypeak * 1.2))
        plt.xlabel(filtername)
        plt.ylabel('dn / dm (targets deg$^{{-2}}$)'.format(filtername))

        label = 'Observed {} {}-mag dn/dm (Peak {}={:.0f})'.format(
            objtype, filtername, filtername, peak)

        plt.bar(cdndmbins, dndm / area, align='center', alpha=0.9, label=label, width=deltam)
        plt.legend(loc='upper left', frameon=False)

        pngfile = os.path.join(qadir, '{}-{}-{}.png'
                               .format(fileprefix, filtername, objtype))
        plt.savefig(pngfile, bbox_inches='tight')
        plt.close()

        # ADM create an ASCII file binned 0.1 mags.
        dndm_out, dndmbins_out = np.histogram(mag, bins=magbins_out, range=(brightmag, faintmag))
        cdndmbins_out = (dndmbins_out[:-1] + dndmbins_out[1:]) / 2.0
        datfile = pngfile.replace("png", "dat")
        np.savetxt(datfile, np.vstack((cdndmbins_out, dndm_out / area)).T,
                   fmt=('%.2f', '%.3f'), header='{} dn/dm (number per deg2 per 0.1 mag)'.format(filtername))


def qagaia(cat, objtype, qadir='.', fileprefix="gaia", nobjscut=1000, seed=None):
    """Make Gaia-based DESI targeting QA plots given a passed set of targets.

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least "RA", "PARALLAX",
        "PMRA" and "PMDEC".
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    fileprefix : :class:`str`, optional, defaults to ``"gaia"``
        String to be added to the front of the output file name.
    nobjscut : :class:`int`, optional, defaults to ``1000``
        Make a hexbin plot when the number of objects is greater than
        ``nobjscut``, otherwise make a scatterplot.
    seed : :class:`int`, optional
        Seed to reproduce random points plotted on hexbin plots.

    Returns
    -------
    Nothing
        But .png plots of Gaia information are written to ``qadir``. Two plots are made:
           The file containing distances from parallax is called:
                 ``{qadir}/{fileprefix}-{parallax}-{objtype}.png``.
           The file containing proper motion information is called:
                 ``{qadir}/{fileprefix}-{pm}-{objtype}.png``.

    """
    rand = np.random.RandomState(seed)

    # ADM change the parallaxes (which are in mas) to distances in parsecs.
    # ADM clip at very small parallaxes to avoid divide-by-zero.
    r = 1000./np.clip(cat["PARALLAX"], 1e-16, 1e16)
    # ADM set the angle element of the plot to RA.
    theta = np.radians(cat["RA"])

    objcolor = {'ALL': 'black', objtype: 'blue'}

    # ADM set up the plot in polar projection.
    ax = plt.subplot(111, projection='polar')
    ax.scatter(theta, r, s=2, alpha=0.6)

    # ADM only plot out to 110 pc.
    ax.set_rmax(125)
    # ADM add a grid of distances.
    rticknum = np.arange(1, 6)*25
    rticknames = ["{}".format(num) for num in rticknum]
    # ADM include the parsec unit for the outermost distance label.
    rticknames[-1] += 'pc'
    # ADM the most flexible set of rtick controllers is in the ytick attribute.
    ax.set_yticks(rticknum)
    ax.set_yticklabels(rticknames)
    ax.grid(True)

    # ADM save the plot.
    ax.set_title("Distances at each RA based on Gaia parallaxes", va='bottom')
    pngfile = os.path.join(qadir, '{}-{}-{}.png'.format(fileprefix, 'parallax', objtype))
    plt.savefig(pngfile, bbox_inches='tight')
    plt.close()

    # ADM plot the proper motions in RA/Dec against each other.
    plt.clf()
    plt.xlabel(r'$PM_{RA}\,(mas\,yr^{-1})$')
    plt.ylabel(r'$PM_{DEC}\,(mas\,yr^{-1})$')

    cmap = plt.cm.get_cmap('RdYlBu')

    ralim = (-25, 25)
    declim = (-25, 25)
    nobjs = len(cat)

    # ADM make a contour plot if we have lots of points...
    if nobjs > nobjscut:
        hb = plt.hexbin(cat["PMRA"], cat["PMDEC"], mincnt=1, cmap=cmap,
                        bins='log', extent=(*ralim, *declim), gridsize=60)
        cb = plt.colorbar(hb)
        irand = rand.choice(nobjs, size=nobjscut, replace=False)
        plt.scatter(cat["PMRA"][irand], cat["PMDEC"][irand], alpha=0.6, color=objcolor[objtype], s=5)
        cb.set_label(r'$\log_{{10}}$ (Number of {})'.format(objtype))

    # ADM...otherwise make a scatter plot.
    else:
        plt.scatter(cat["PMRA"], cat["PMDEC"], alpha=0.6, color='blue')

    # ADM save the plot.
    pngfile = os.path.join(qadir, '{}-{}-{}.png'.format(fileprefix, 'pm', objtype))
    plt.savefig(pngfile, bbox_inches='tight')
    plt.close()


def mock_qafractype(cat, objtype, qadir='.', fileprefix="mock-fractype"):
    """Targeting QA Bar plot of the fraction of each classification type assigned to (mock) targets.

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``TRUESPECTYPE``.
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    fileprefix : :class:`str`, optional, defaults to ``"mock-fractype"`` for
        String to be added to the front of the output file name.

    Returns
    -------
    Nothing
        But .png plots of target colors are written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{objtype}.png``.
    """

    # ADM for this type of object, create the names of the possible contaminants.
    from desitarget import contam_mask
    types = np.array(contam_mask.names())
    # ADM this is something of a hack as it assumes we'll keep
    # ADM the first 3 letters of each object ("BGS", "ELG" etc.) as sacred.
    wtypes = np.where(np.array([objtype[:3] in type[:3] for type in types]))

    # ADM only make a plot if we have contaminant information.
    if len(wtypes[0]) > 0:
        # ADM the relevant contaminant types for this object class.
        types = types[wtypes]

        # ADM count each type of object.
        ntypes = len(types)
        typecnt = []
        for typ in types:
            w = np.where((cat["CONTAM_TARGET"] & contam_mask[typ]) != 0)
            typecnt.append(len(w[0]))

        # ADM express each type as a fraction of all objects in the catalog.
        frac = np.array(typecnt)/len(cat)

        # ADM set up and make the bar plot with the legend.
        plt.clf()
        plt.ylabel('fraction')
        if np.max(frac) > 0:
            plt.ylim(0, 1.2*np.max(frac))

        x = np.arange(ntypes)
        plt.bar(x, frac, alpha=0.6,
                label='Fraction of {} classified as'.format(objtype))
        plt.legend(loc='upper left', frameon=False)

        # ADM add the names of the types to the x-axis.
        # ADM first converting the strings to unicode if they're byte-type.
        if isinstance(types[0], bytes):
            types = [type.decode() for type in types]

        # ADM to save space, only plot the name of the contaminant on the x-axis
        # ADM not the object type (e.g. label as LRG not "QSO_IS_LRG").
        labels = [typ.split('_')[-1] for typ in types]
        plt.xticks(x, labels)

    # ADM if there was no contaminant info for this objtype,
    # ADM then make an empty plot with a message.
    else:
        log = get_logger()
        log.warning('No contaminant information for objects of type {}'.format(objtype))
        plt.clf()
        plt.ylabel('fraction')
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.text(0.5, 0.5, 'NO DATA')

    # ADM write out the plot.
    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefix, objtype))
    plt.savefig(pngfile, bbox_inches='tight')
    plt.close()

    return


def mock_qanz(cat, objtype, qadir='.', area=1.0, dndz=None, nobjscut=1000,
              fileprefixz="mock-nz", fileprefixzmag="mock-zvmag"):
    """Make N(z) and z vs. mag DESI QA plots given a passed set of MOCK TRUTH.

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``TRUEZ`` for redshift information
        and ``MAG`` for magnitude information.
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    area : :class:`float`
        Total area in deg2.
    dndz : :class:`dict`
        Dictionary output of `_load_dndz`
    nobjscut : :class:`int`, optional, defaults to ``1000``
        Make a hexbin plot when the number of objects is greater than
        ``nobjscut``, otherwise make a scatterplot.
    fileprefixz : :class:`str`, optional, defaults to ``"color"`` for
        String to be added to the front of the output N(z) plot file name.
    fileprefixzmag : :class:`str`, optional, defaults to ``"color"`` for
        String to be added to the front of the output z vs. mag plot file name.

    Returns
    -------
    Nothing
        But .png plots of target colors are written to ``qadir``. Two plots are made:
           The file containing N(z) is called:
                 ``{qadir}/{fileprefixz}-{objtype}.png``.
           The file containing z vs. zerr is called:
                 ``{qadir}/{fileprefixzmag}-{objtype}.png``.
    """

    # ADM the number of passed objects.
    nobjs = len(cat)

    truez = cat["TRUEZ"]
    zmax = truez.max()*1.1

    if 'STD' in objtype or 'MWS' in objtype or 'WD' in objtype:
        truez *= C_LIGHT  # [km/s]
        zlabel = 'True Radial Velocity (km/s)'
    else:
        zlabel = r'True Redshift $z$'

    if 'QSO' in objtype:
        dz = 0.05
        zmin = -0.05
    elif 'STD' in objtype or 'MWS' in objtype or 'WD' in objtype:
        if 'WD' in objtype or 'NEARBY' in objtype:
            dz = 5
            zmin, zmax = -100, 100
        else:
            dz = 10
            zmin, zmax = -300, 300
    else:
        dz = 0.02
        zmin = -0.15

    zbins = np.arange(zmin, zmax, dz)  # bin left edges
    if len(zbins) < 10:
        dz = (zmax - zmin) / 10
        zbins = np.arange(zmin, zmax, dz)
    nzbins = len(zbins)

    objcolor = {'ALL': 'black', objtype: 'blue'}
    type2color = {**_type2color, **objcolor}

    # Get the unique combination of template types, subtypes, and true spectral
    # types.  Lya QSOs and galaxies contaminating ELGs are a special case.
    if isinstance(cat['TEMPLATETYPE'][0], bytes):
        templatetypes = np.char.strip(np.char.decode(cat['TEMPLATETYPE']))
        templatesubtypes = np.char.strip(np.char.decode(cat['TEMPLATESUBTYPE']))
        truespectypes = np.char.strip(np.char.decode(cat['TRUESPECTYPE']))
    else:
        # Already a string; just strip whitespace.
        templatetypes = np.char.strip(cat['TEMPLATETYPE'])
        templatesubtypes = np.char.strip(cat['TEMPLATESUBTYPE'])
        truespectypes = np.char.strip(cat['TRUESPECTYPE'])

    islya = np.where(['LYA' in tt for tt in templatesubtypes])[0]
    if len(islya) > 0:
        truespectypes[islya] = np.array(['{}-LYA'.format(tt) for tt in truespectypes[islya]])

    if objtype == 'ELG':
        elgcontam = np.where(templatetypes == 'BGS')[0]
        if len(elgcontam) > 0:
            truespectypes[elgcontam] = 'NOT ELG'

    # Plot the histogram in the reverse order of the number of objects.
    nthese = np.zeros(len(np.unique(truespectypes)))
    for ii, truespectype in enumerate(np.unique(truespectypes)):
        nthese[ii] = np.sum(truespectype == truespectypes)
    srt = np.argsort(nthese)[::-1]

    if len(np.unique(truespectypes)) > 2:
        ncol = 2
    else:
        ncol = 1

    # ADM set up and make the plot.
    plt.clf()
    plt.xlabel(zlabel)
    plt.ylabel(r'dn / dz (targets deg$^{-2}$)')

    for truespectype in np.unique(truespectypes)[srt]:
        these = np.where(truespectype == truespectypes)[0]
        if len(these) > 0:
            if 'QSO' in objtype:
                label = r'{} is {} ({:.0f} deg$^{{-2}}$)'.format(
                    objtype, truespectype, len(these) / area)
                # label = r'{} is {} (N={}, {:.0f} deg$^{{-2}}$)'.format(
                #    objtype, truespectype, len(these), len(these) / area)
            else:
                label = r'{} is {} ({:.0f} deg$^{{-2}}$))'.format(
                    objtype, truespectype, len(these) / area)
                # label = r'{} is {} (N={}, {:.0f} deg$^{{-2}}$))'.format(
                #    objtype, truespectype, len(these), len(these) / area)
            nn, bins = np.histogram(truez[these], bins=nzbins, range=(zmin, zmax))
            cbins = (bins[:-1] + bins[1:]) / 2.0

            if truespectype not in type2color.keys():
                log.warning('Missing color for spectral type {}!'.format(truespectype))
                col = 'black'
            else:
                col = type2color[truespectype]
            plt.bar(cbins, nn / area, align='center', alpha=0.6, label=label, width=dz, color=col)

    if dndz is not None and objtype in dndz.keys():
        newdndz = np.interp(zbins, dndz[objtype]['z'], dndz[objtype]['dndz'], left=0, right=0)
        newdndz *= np.sum(dndz[objtype]['dndz']) / np.sum(newdndz)
        plt.step(zbins, newdndz, alpha=0.5, color='k', lw=2,
                 label=r'Expected {} dn/dz ({:.0f} deg$^{{-2}}$)'.format(
                     objtype, np.sum(dndz[objtype]['dndz'])))
        plt.ylim(0, np.max(newdndz) * 1.5)

    if 'LRG' in objtype:
        loc = 'upper left'
    else:
        loc = 'upper right'
    plt.legend(loc=loc, frameon=True, fontsize=10,
               handletextpad=0.5, labelspacing=0)

    # plt.xlim(zmin, zmax)

    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefixz, objtype))
    plt.savefig(pngfile, bbox_inches='tight')
    plt.close()

    # ADM plot the z vs. mag scatter plot.
    plt.clf()
    plt.xlabel(zlabel)

    cmap = plt.cm.get_cmap('RdYlBu')

    if 'LRG' in objtype:
        band = 'z'
        flux = cat['FLUX_Z'].clip(1e-16)
    else:
        band = 'r'
        flux = cat['FLUX_R'].clip(1e-16)

    plt.ylabel('{} (AB mag)'.format(band))

    mag = 22.5 - 2.5 * np.log10(flux)

    if 'BGS' in objtype or 'MWS' in objtype or 'STD' in objtype:
        if 'NEARBY' in objtype:
            magbright, magfaint = 6, 18
        else:
            magbright, magfaint = 14, 21
    else:
        if 'LRG' in objtype:
            magbright, magfaint = 17.5, 21
        elif 'ELG' in objtype:
            magbright, magfaint = 19.5, 24
        else:
            magbright, magfaint = 17, 24

    dolegend = False
    for ii, truespectype in enumerate(np.unique(truespectypes)[srt]):
        these = np.where(truespectype == truespectypes)[0]
        if truespectype not in type2color.keys():
            log.warning('Missing color for spectral type {}!'.format(truespectype))
            col = 'black'
        else:
            col = type2color[truespectype]
        if len(these) > 0:
            label = '{} is {}'.format(objtype, truespectype)
            # ADM make a contour plot if we have lots of points...
            if len(these) > nobjscut:
                hb = plt.hexbin(truez, mag, mincnt=1, bins='log', cmap=cmap,
                                extent=(zmin, zmax, magbright, magfaint),
                                gridsize=60)
                if ii == 0:
                    cb = plt.colorbar(hb)
                    cb.set_label(r'$\log_{{10}}$ (Number of {})'.format(label))
            else:
                # ADM...otherwise make a scatter plot.
                dolegend = True
                plt.scatter(truez[these], mag[these], alpha=0.6, label=label, color=col)

    if dolegend:
        plt.legend(loc='upper right', frameon=True, ncol=ncol, fontsize=10,
                   handletextpad=0.5, labelspacing=0)

    plt.xlim((zmin, zmax))
    plt.ylim((magbright, magfaint))

    # ADM create the plot
    pngfile = os.path.join(qadir, '{}-{}.png'.format(fileprefixzmag, objtype))
    plt.savefig(pngfile, bbox_inches='tight')
    plt.close()


def qacolor(cat, objtype, extinction, qadir='.', fileprefix="color",
            nodustcorr=False, mocks=False, nobjscut=1000, seed=None):
    """Make color-based DESI targeting QA plots given a passed set of targets.

    Parameters
    ----------
    cat : :class:`~numpy.array`
        An array of targets that contains at least ``FLUX_G``, ``FLUX_R``, ``FLUX_Z`` and
        ``FLUX_W1``, ``FLUX_W2`` columns for color information.
    objtype : :class:`str`
        The name of a DESI target class (e.g., ``"ELG"``) that corresponds to the passed ``cat``.
    extinction : :class:`~numpy.array`
        An array containing the extinction in each band of interest, must contain at least the columns
        ``MW_TRANSMISSION_G, _R, _Z, _W1, _W2``.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    fileprefix : :class:`str`, optional, defaults to ``"color"``
        String to be added to the front of the output file name.
    nodustcorr : :class:`boolean`, optional, defaults to False
        Do not correct for dust extinction.
    mocks : :class:`boolean`, optional, default=False
        If ``True``, input catalog is a "truths" catalog.
    nobjscut : :class:`int`, optional, defaults to ``1000``
        Make a hexbin plot when the number of objects is greater than
        ``nobjscut``, otherwise make a scatterplot.
    seed : :class:`int`, optional
        Seed to reproduce random points plotted on hexbin plots.

    Returns
    -------
    Nothing
        But .png plots of target colors are written to ``qadir``. The file is called:
        ``{qadir}/{fileprefix}-{bands}-{objtype}.png``
        where bands might be, e.g., ``grz``.
    """
    from matplotlib.ticker import MultipleLocator, MaxNLocator
    from matplotlib.patches import Polygon

    rand = np.random.RandomState(seed)

    def elg_colorbox(ax, plottype='grz', verts=None):
        """Draw the ELG selection box."""

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if plottype == 'grz':
            coeff0, coeff1 = (1.15, -0.15), (-1.2, 1.6)
            rzmin, rzpivot = 0.3, (coeff1[1] - coeff0[1]) / (coeff0[0] - coeff1[0])
            verts = [(rzmin, ylim[0]),
                     (rzmin, np.polyval(coeff0, rzmin)),
                     (rzpivot, np.polyval(coeff1, rzpivot)),
                     ((ylim[0] - 0.1 - coeff1[1]) / coeff1[0], ylim[0] - 0.1)]

        if verts:
            ax.add_patch(Polygon(verts, fill=False, ls='--', lw=3, color='k'))

    def qso_colorbox(ax, plottype='grz', verts=None):
        """Draw the QSO selection boxes."""

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if plottype == 'grz':
            verts = [(-0.3, 1.3),
                     (1.1, 1.3),
                     (1.1, ylim[0]-0.05),
                     (-0.3, ylim[0]-0.05)]

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

    # ADM unextinct fluxes.
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

    # ADM the number of passed objects.
    nobjs = len(cat)

    # ADM convert to magnitudes (fluxes are in nanomaggies)
    # ADM should be fine to clip for plotting purposes.
    loclip = 1e-16
    g = 22.5-2.5*np.log10(gflux.clip(loclip))
    r = 22.5-2.5*np.log10(rflux.clip(loclip))
    z = 22.5-2.5*np.log10(zflux.clip(loclip))
    W1 = 22.5-2.5*np.log10(w1flux.clip(loclip))
    # ADM Modify W2 slightly so the W1-W2 color doesn't pile up at 0.
    W2 = 22.5-2.5*np.log10(w2flux.clip(loclip*100))

    # For QSOs only--
    W = 0.75 * W1 + 0.25 * W2
    grz = (g + 0.8*r + 0.5*z) / 2.3

    # Some color ranges -- need to be smarter here.
    if 'LRG' in objtype:
        grlim = (0.5, 2.5)
        rzlim = (0.5, 2.5)
    elif 'MWS' in objtype:
        if 'WD' in objtype:
            grlim = (-1.5, 1.0)
            rzlim = (-1.5, 1.0)
        else:
            grlim = (-0.5, 2.0)
            rzlim = (-0.5, 3.0)
    elif 'BGS' in objtype:
        grlim = (-0.5, 2.5)
        rzlim = (-0.5, 2.5)
    else:
        grlim = (-0.5, 1.5)
        rzlim = (-0.5, 1.5)

    # if 'ELG' in objtype:
    #     zW1lim = (-3, 2.5)
    #     W1W2lim = (-3, 2.5)
    # else:
    zW1lim = (-1.5, 3.5)
    W1W2lim = (-1.5, 1.5)

    cmap = plt.cm.get_cmap('RdYlBu')

    objcolor = {'ALL': 'black', objtype: 'black'}
    type2color = {**_type2color, **objcolor}

    #  Make the plots!
    for plotnumber in range(3):
        if plotnumber == 0:
            plottype = 'grz'
            xlabel, ylabel = r'$r - z$', r'$g - r$'
            xlim, ylim = rzlim, grlim
            xlocator, ylocator = 0.5, 0.5
            xdata, ydata = r - z, g - r
        elif plotnumber == 1:
            plottype = 'rzW1'
            xlabel, ylabel = r'$z - W_{1}$', r'$r - z$'
            xlim, ylim = zW1lim, rzlim
            xlocator, ylocator = 1.0, 0.5
            xdata, ydata = z - W1, r - z
        elif plotnumber == 2:
            plottype = 'rzW1W2'
            xlabel, ylabel = r'$W_{1} - W_{2}$', r'$r - z$'
            xlim, ylim = W1W2lim, rzlim
            xlocator, ylocator = 0.5, 0.5
            xdata, ydata = W1 - W2, r - z
        else:
            log.error('Unrecognized plot number {}!'.format(plotnumber))
            raise ValueError

        if mocks:
            # Get the unique combination of template types, subtypes, and true
            # spectral types.  Lya QSOs are a special case.
            if isinstance(cat['TEMPLATETYPE'][0], bytes):
                templatetypes = np.char.strip(np.char.decode(cat['TEMPLATETYPE']))
                templatesubtypes = np.char.strip(np.char.decode(cat['TEMPLATESUBTYPE']))
                truespectypes = np.char.strip(np.char.decode(cat['TRUESPECTYPE']))
            else:
                # Already a string, just strip whitespace.
                templatetypes = np.char.strip(cat['TEMPLATETYPE'])
                templatesubtypes = np.char.strip(cat['TEMPLATESUBTYPE'])
                truespectypes = np.char.strip(cat['TRUESPECTYPE'])

            islya = np.where(['LYA' in tt for tt in templatesubtypes])[0]
            if len(islya) > 0:
                truespectypes[islya] = np.array(['{}-LYA'.format(tt) for tt in truespectypes[islya]])

            # Plot the histogram in the reverse order of the number of objects.
            nthese = np.zeros(len(np.unique(truespectypes)))
            for ii, truespectype in enumerate(np.unique(truespectypes)):
                nthese[ii] = np.sum(truespectype == truespectypes)
            srt = np.argsort(nthese)[::-1]

            if len(np.unique(truespectypes)) > 2:
                ncol = 2
            else:
                ncol = 1

        plt.clf()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if mocks:
            dolegend = False
            for ii, truespectype in enumerate(np.unique(truespectypes)[srt]):
                these = np.where(truespectype == truespectypes)[0]

                if truespectype not in type2color.keys():
                    log.warning('Missing color for spectral type {}!'.format(truespectype))
                    col = 'black'
                else:
                    col = type2color[truespectype]

                if len(these) > 0:
                    label = '{} is {}'.format(objtype, truespectype)
                    if len(these) > nobjscut:
                        hb = plt.hexbin(xdata[these], ydata[these], mincnt=1,
                                        cmap=cmap, bins='log',
                                        extent=(*xlim, *ylim), gridsize=60)
                        if ii == 0:
                            cb = plt.colorbar(hb)
                            # cb.locator = MaxNLocator(nbins=5)
                            # cb.update_ticks()
                            cb.set_label(r'$\log_{{10}}$ (Number of {})'.format(label))
                    else:
                        dolegend = True
                        plt.scatter(xdata[these], ydata[these], alpha=0.6, label=label, color=col)

            if dolegend:
                plt.legend(loc='upper right', frameon=True, ncol=ncol, fontsize=10,
                           handletextpad=0.5, labelspacing=0)
        else:
            if nobjs > nobjscut:
                hb = plt.hexbin(xdata, ydata, mincnt=1, cmap=cmap, bins='log',
                                extent=(*xlim, *ylim), gridsize=60)
                cb = plt.colorbar(hb)
                irand = rand.choice(nobjs, size=nobjscut, replace=False)
                plt.scatter(xdata[irand], ydata[irand], alpha=0.6, color=objcolor[objtype], s=5)
                # cb.locator = MaxNLocator(nbins=5)
                # cb.update_ticks()
                cb.set_label(r'$\log_{{10}}$ (Number of {})'.format(objtype))
            else:
                plt.scatter(xdata, ydata, alpha=0.6, color=objcolor[objtype])

        plt.xlim(xlim)
        plt.ylim(ylim)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(xlocator))
        ax.yaxis.set_major_locator(MultipleLocator(ylocator))

        if objtype == 'ELG':
            elg_colorbox(plt.gca(), plottype=plottype)

        if objtype == 'QSO':
            qso_colorbox(plt.gca(), plottype=plottype)
            if plottype == 'rzW1W2':
                plt.axvline(x=-0.4, lw=3, ls='--', color='k')

        # ADM make the plot.
        pngfile = os.path.join(qadir, '{}-{}-{}.png'.format(fileprefix, plottype, objtype))
        plt.savefig(pngfile, bbox_inches='tight')
        plt.close()


def _in_desi_footprint(targs, radec=False):
    """Convenience function for using is_point_in_desi to find which targets are in the footprint.

    Parameters
    ----------
    targs : :class:`~numpy.array` or `str`
        Targets in the DESI data model format, or any array that
        contains ``RA`` and ``DEC`` columns.
    radec : :class:`bool`, optional, defaults to ``False``
        If ``True`` then the passed `objs` is an [RA, Dec] list instead of
        a rec array.
    Returns
    -------
    :class:`integer`
        The INDICES of the input targs that are in the DESI footprint.
    """
    log = get_logger()

    start = time()
    log.info('Start restricting to DESI footprint...t = {:.1f}s'
             .format(time()-start))

    if radec:
        ra, dec = targs
    else:
        ra, dec = targs["RA"], targs["DEC"]

    # ADM restrict targets to just the DESI footprint.
    from desimodel import io, footprint
    indesi = footprint.is_point_in_desi(io.load_tiles(), ra, dec)
    windesi = np.where(indesi)
    if len(windesi[0]) > 0:
        log.info("{:.3f}% of targets are in official DESI footprint"
                 .format(100.*len(windesi[0])/len(ra)))
    else:
        log.error("ZERO input targets are within the official DESI footprint!!!")

    log.info('Restricted targets to DESI footprint...t = {:.1f}s'
             .format(time()-start))

    return windesi


def make_qa_plots(targs, qadir='.', targdens=None, max_bin_area=1.0, weight=True,
                  imaging_map_file=None, truths=None, objtruths=None, tcnames=None,
                  cmx=False, bit_mask=None, mocks=False, numproc=8):
    """Make DESI targeting QA plots given a passed set of targets.

    Parameters
    ----------
    targs : :class:`~numpy.array` or `str`
        Array of targets in the DESI data model format. If a string is
        passed then the targets are read from the file with the passed
        name (supply the full directory path).
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    targdens : :class:`dictionary`, optional
        A dictionary of DESI target classes and the goal density for
        that class. Used to label the goal density on histogram plots.
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        The bin size for sky maps in RA/Dec correspond to this value.
    weight : :class:`boolean`, optional, defaults to True
        If set, weight pixels using the ``DESIMODEL`` HEALPix footprint
        file to offset under-dense pixels at the footprint edges.
    imaging_map_file : :class:`str`, optional, defaults to no weights
        If `weight` is set, this file is the location of the imaging
        HEALPixel map (e.g. made by :func:` desitarget.randoms.pixmap()`.
        If not sent, then weights default to 1 (i.e. no weighting).
    truths : :class:`~numpy.array` or `str`
        The truth objects from which the targs were derived in the DESI
        data model format. If a string is passed then read from that file
        (supply the full directory path).
    objtruths : :class:`dict`
        Object type-specific truth metadata.
    tcnames : :class:`list`, defaults to None
        Strings, e.g. ['QSO','LRG','ALL'] If passed, return only the QA
        pages for those specific bits. A useful speed-up when testing.
    cmx : :class:`boolean`, defaults to ``False``
        Pass as ``True`` to use commissioning bits instead of SV or main
        survey bits. Commissioning files have no MWS or BGS columns.
    bit_mask : :class:`list` or `~numpy.array`, optional
        Load bit names from this passed mask or list of masks instead of
        from the (default) main survey mask.
    mocks : :class:`boolean`, optional, default=False
        If ``True``, also make plots that are only relevant to mocks.
    numproc : :class:`int`, optional, defaults to 8
        The number of parallel processes to use to generate plots.

    Returns
    -------
    :class:`float`
        The total area of the survey used to make the QA plots.
    :class:`dict`
        A nested dictionary of each of the bit-names. Each bit-key has a
        dictionary of the 10 densest pixels in the DESI tiling. Includes
        RA, DEC, DENSITY (per sq. deg.) and NSIDE for each HEALpixel.

    Notes
    -----
        - The ``DESIMODEL`` environment variable must be set to find the
          default expected target densities.
        - When run, a set of targeting .png plots are written to `qadir`.
    """
    # ADM set up the default logger from desiutil.
    log = get_logger()

    start = time()
    log.info('Start making targeting QA plots...t = {:.1f}s'.format(time()-start))

    if mocks and targs is None and truths is None and objtruths is None:
        if isinstance(targs, str):
            targs, truths, objtruths = collect_mock_data(targs)
            if mockdata == 0:
                mocks = False
            else:
                pass  # = mockdata
        else:
            log.warning('To make mock-related plots, targs must be a directory+file-location string...')
            log.warning('...will proceed by only producing the non-mock plots...')

    else:
        # ADM if a filename was passed, read in the targets from that file.
        if isinstance(targs, str):
            targs = read_targets_in_box(targs)
            log.info('Read in targets...t = {:.1f}s'.format(time()-start))
            truths, objtruths = None, None

    # ADM determine the nside for the passed max_bin_area.
    nside = pixarea2nside(max_bin_area)

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
            pixweight = dtio.load_pixweight_recarray(imaging_map_file, nside)["FRACAREA"]
            # ADM determine what HEALPixels each target is in, to set the weights.
            fracarea = pixweight[pix]
            # ADM weight by 1/(the fraction of each pixel that is in the DESI footprint)
            # ADM except for zero pixels, which are all outside of the footprint.
            ii = fracarea == 0
            fracarea[ii] = 1  # ADM to guard against division by zero warnings.
            weights = 1./fracarea
            weights[ii] = 0

            # ADM if we have weights, then redetermine the total pix weight.
            totalpixweight = np.sum(pixweight[uniqpixset])

            log.info('Assigned weights to pixels based on DESI footprint...t = {:.1f}s'
                     .format(time()-start))

    # ADM calculate the total area (useful for determining overall average densities
    # ADM from the total number of targets/the total area).
    pixarea = hp.nside2pixarea(nside, degrees=True)
    totarea = pixarea*totalpixweight

    # ADM Current goal target densities for DESI.
    if targdens is None:
        targdens = _load_targdens(tcnames=tcnames, bit_mask=bit_mask)

    if mocks:
        dndz = _load_dndz()

    # ADM clip the target densities at an upper density to improve plot edges
    # ADM by rejecting highly dense outliers.
    upclipdict = {k: None for k in targdens}
    if bit_mask is not None:
        if cmx:
            d_mask = bit_mask[0]
        else:
            d_mask, b_mask, m_mask = bit_mask
    else:
        d_mask, b_mask, m_mask = desi_mask, bgs_mask, mws_mask
        upclipdict = {'ELG': 4000, 'LRG': 1200, 'QSO': 400, 'ALL': 8000,
                      'STD_FAINT': 300, 'STD_BRIGHT': 300,
                      # 'STD_FAINT': 200, 'STD_BRIGHT': 50,
                      # 'LRG_1PASS': 1000, 'LRG_2PASS': 500,
                      'BGS_FAINT': 2500, 'BGS_BRIGHT': 2500, 'BGS_WISE': 2500, 'BGS_ANY': 5000,
                      'MWS_ANY': 2000, 'MWS_BROAD': 2000, 'MWS_WD': 50, 'MWS_NEARBY': 50,
                      'MWS_MAIN_RED': 2000, 'MWS_MAIN_BLUE': 2000}

    nbits = len(targdens)
    nbit = np.ones((), dtype='i8')
    t0 = time()

    # ADM this will hold dictionaries of the densest pixels for each bit.
    densdict = {}

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        log.info('Done {}/{} bit names...t = {:.1f}s'.format(nbit, nbits, time()-t0))
        nbit[...] += 1    # this is an in-place modification.
        return result

    def _generate_plots(objtype):
        """Make relevant plots for each bit name in objtype"""

        if 'ALL' in objtype:
            ii = np.ones(len(targs)).astype('bool')
        else:
            if ('BGS' in objtype) and not('S_ANY' in objtype) and not(cmx):
                ii = targs["BGS_TARGET"] & b_mask[objtype] != 0
            elif (('MWS' in objtype or 'BACKUP' in objtype) and
                  not('S_ANY' in objtype) and not(cmx)):
                ii = targs["MWS_TARGET"] & m_mask[objtype] != 0
            else:
                ii = targs["DESI_TARGET"] & d_mask[objtype] != 0

        # ADM set up a dummy output in case there are no targets.
        dd = {"RA": [], "DEC": [], "DENSITY": [], "NSIDE": []}
        if np.any(ii):
            # ADM make RA/Dec skymaps.
            dd = qaskymap(targs[ii], objtype, weights=weights[ii], qadir=qadir,
                          upclip=upclipdict[objtype], max_bin_area=max_bin_area)
            log.info('Made sky map for {}...t = {:.1f}s'
                     .format(objtype, time()-start))

            # ADM make histograms of densities. We already calculated the correctly
            # ADM ordered HEALPixels and so don't need to repeat that calculation.
            qahisto(pix[ii], objtype, qadir=qadir, targdens=targdens, upclip=upclipdict[objtype],
                    weights=weights[ii], max_bin_area=max_bin_area, catispix=True)
            log.info('Made histogram for {}...t = {:.1f}s'
                     .format(objtype, time()-start))

            # ADM make color-color plots
            qacolor(targs[ii], objtype, targs[ii], qadir=qadir, fileprefix="color")
            log.info('Made color-color plot for {}...t = {:.1f}s'
                     .format(objtype, time()-start))

            # ADM make magnitude histograms
            qamag(targs[ii], objtype, qadir=qadir, fileprefix="nmag", area=totarea)
            log.info('Made magnitude histogram plot for {}...t = {:.1f}s'
                     .format(objtype, time()-start))

            if truths is not None:
                # ADM make noiseless color-color plots
                qacolor(truths[ii], objtype, targs[ii], qadir=qadir, mocks=True,
                        fileprefix="mock-color", nodustcorr=True)
                log.info('Made (mock) color-color plot for {}...t = {:.1f}s'
                         .format(objtype, time()-start))

                # ADM make N(z) plots
                mock_qanz(truths[ii], objtype, qadir=qadir, area=totarea, dndz=dndz,
                          fileprefixz="mock-nz", fileprefixzmag="mock-zvmag")
                log.info('Made (mock) redshift plots for {}...t = {:.1f}s'
                         .format(objtype, time()-start))

                # # ADM plot what fraction of each selected object is actually a contaminant
                # mock_qafractype(truths[ii], objtype, qadir=qadir, fileprefix="mock-fractype")
                # log.info('Made (mock) classification fraction plots for {}...t = {:.1f}s'.format(objtype, time()-start))

            # ADM make Gaia-based plots if we have Gaia columns
            if "PARALLAX" in targs.dtype.names and np.any(targs['PARALLAX'] != 0):
                qagaia(targs[ii], objtype, qadir=qadir, fileprefix="gaia")
                log.info('Made Gaia-based plots for {}...t = {:.1f}s'
                         .format(objtype, time()-start))

        return {objtype: dd}

    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            dens = pool.map(_generate_plots, list(targdens.keys()), reduce=_update_status)
    else:
        dens = []
        for objtype in targdens:
            dens.append(_update_status(_generate_plots(objtype)))

    # ADM manipulate the output array to make it a nested dictionary.
    densdict = {list(i.keys())[0]: list(i.values())[0] for i in dens}

    log.info('Made QA plots...t = {:.1f}s'.format(time()-start))
    return totarea, densdict


def make_qa_page(targs, mocks=False, makeplots=True, max_bin_area=1.0, qadir='.',
                 clip2foot=False, weight=True, imaging_map_file=None,
                 tcnames=None, systematics=True, numproc=8, downsample=None):
    """Make a directory containing a webpage in which to embed QA plots.

    Parameters
    ----------
    targs : :class:`~numpy.array` or `str`
        An array of targets in the DESI data model format. If a string is
        passed then the targets are read from the file with the passed
        name (supply the full directory path). The string can also be a
        directory of HEALPixel-split target files which will be read in
        using :func:`desitarget.io.read_targets_in_box`.
    mocks : :class:`boolean`, optional, default=False
        If ``True``, add plots only relevant to mocks to the webpage.
    makeplots : :class:`boolean`, optional, default=True
        If ``True``, then create the plots as well as the webpage.
    max_bin_area : :class:`float`, optional, defaults to 1 degree
        Bin size in RA/Dec is set as close as possible to this value.
    qadir : :class:`str`, optional, defaults to the current directory
        The output directory to which to write produced plots.
    clip2foot : :class:`boolean`, optional, defaults to False
        Use :mod:`desimodel.footprint.is_point_in_desi` to restrict
        `targs` to the DESI spectroscopic footprint.
    weight : :class:`boolean`, optional, defaults to ``True``
        If set, weight pixels to offset under-dense pixels at footprint
        edges. Uses the `imaging_map_file` HEALPix file for real targets
        and the ``DESIMODEL`` HEALPix footprint file for mock targets.
    imaging_map_file : :class:`str`, optional, defaults to no weights
        If `weight` is set, then this is the location of the imaging
        HEALPix map (e.g. made by :func:`desitarget.randoms.pixmap`).
        Defaults to 1 everywhere (i.e. no weights) for the real targets.
        If this is not set, then systematics plots cannot be made.
    tcnames : :class:`list`
        String-list, e.g. ['QSO','LRG','ALL'] If passed, return only QA
        pages for those specific bits. A useful speed-up when testing.
    systematics : :class:`boolean`, optional, defaults to ``True``
        If sent, then add plots of systematics to the front page.
    numproc : :class:`int`, optional, defaults to 8
        The number of parallel processes to use to generate plots.
    downsample : :class:`int`, optional, defaults to `None`
        If not `None`, downsample targets by (roughly) this value, e.g.
        for `downsample=10` a set of 900 targets would have ~90 random
        targets returned. A speed-up for experimenting with large files.

    Returns
    -------
    Nothing
        But the page `index.html` and associated pages and plots are written to ``qadir``.

    Notes
    -----
    If making plots, then the ``DESIMODEL`` environment variable must be set to find
    the file of HEALPixels that overlap the DESI footprint.
    """
# ADM the following parameter assignments die on my NERSC build
# ADM for some reason, so I'm turning them off, for now.
#    import matplotlib as mpl
#    mpl.rcParams['xtick.major.width'] = 2
#    mpl.rcParams['ytick.major.width'] = 2
#    mpl.rcParams['xtictark.minor.width'] = 2
#    mpl.rcParams['ytick.minor.width'] = 2
#    mpl.rcParams['font.size'] = 13

    from desispec.io.util import makepath
    # ADM set up the default logger from desiutil.
    log = get_logger()

    start = time()
    log.info('Start making targeting QA page...t = {:.1f}s'.format(time()-start))

    if downsample is not None:
        log.info('Downsampling by a factor of {}'.format(downsample))

    if isinstance(targs, str):
        targs, truths, objtruths = read_data(targs, mocks=mocks,
                                             downsample=downsample)
    else:
        if mocks:
            log.warning('Please pass the filename to the targeting catalog so the "mock" QA plots can be generated.')
        truths, objtruths = None, None

    # ADM automatically detect whether we're running this for the main survey
    # ADM or SV, etc. and load the appropriate mask.
    colnames, masks, svs, targs = main_cmx_or_sv(targs, rename=True)
    svs = svs.upper()
    cmx = svs == 'CMX'

    # ADM determine the working nside for the passed max_bin_area.
    nside = pixarea2nside(max_bin_area)

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
        DRs = 'Mock Targets'
    else:
        if 'RELEASE' in targs.dtype.names:
            DRs = ", ".join(["DR{}".format(release)
                             for release in np.unique(targs["RELEASE"]//1000)])
        else:
            DRs = "DR Unknown"

    # ADM Set up the names of the target classes and their goal densities using
    # ADM the goal target densities for DESI (read from the DESIMODEL defaults).
    targdens = _load_targdens(tcnames=tcnames, bit_mask=masks)

    # ADM set up the html file and write preamble to it.
    htmlfile = makepath(os.path.join(qadir, 'index.html'))

    # ADM grab the magic string that writes the last-updated date to a webpage.
    js = _javastring()

    # ADM make the plots for the page, if requested.
    if makeplots:
        if svs == "MAIN":
            totarea, densdict = make_qa_plots(
                targs, truths=truths, objtruths=objtruths, numproc=numproc,
                qadir=qadir, targdens=targdens, max_bin_area=max_bin_area,
                weight=weight, imaging_map_file=imaging_map_file, mocks=mocks
            )
        else:
            totarea, densdict = make_qa_plots(
                targs, truths=truths, objtruths=objtruths, numproc=numproc,
                qadir=qadir, targdens=targdens, max_bin_area=max_bin_area,
                weight=weight, imaging_map_file=imaging_map_file,
                cmx=cmx, bit_mask=masks, mocks=mocks
            )

    # ADM html preamble.
    htmlmain = open(htmlfile, 'w')
    htmlmain.write('<html><body>\n')
    htmlmain.write('<h1>{} Targeting QA ({})</h1>\n'.format(svs, DRs))

    # ADM links to each collection of plots for each object type.
    htmlmain.write('<b><h2>Jump to a target class:</h2></b>\n')
    htmlmain.write('<ul>\n')
    for objtype in sorted(targdens.keys()):
        htmlmain.write('<li><A HREF="{}.html"><b>{}</b></A>\n'.format(objtype, objtype))
    htmlmain.write('</ul>\n')

    # ADM for each object type, make a separate page.
    for objtype in targdens.keys():
        # ADM call each page by the target class name, out it in the requested directory.
        htmlfile = os.path.join(qadir, '{}.html'.format(objtype))
        html = open(htmlfile, 'w')

        # ADM html preamble.
        html.write('<html><body>\n')
        html.write('<h1>{} Targeting QA pages - {} ({})</h1>\n'.format(svs, objtype, DRs))

        # ADM Target Densities.
        html.write('<h2>Target density</h2>\n')
        # ADM Write out the densest pixels.
        if makeplots:
            html.write('<pre>')
            resol = 0
            if isinstance(densdict[objtype]["NSIDE"], int):
                resol = hp.nside2resol(densdict[objtype]["NSIDE"], arcmin=True)
            html.write('Densest HEALPixels (NSIDE={}; ~{:.0f} arcmin across; links are to LS viewer):\n'.format(
                densdict[objtype]["NSIDE"], resol))
            ras, decs, dens = densdict[objtype]["RA"], densdict[objtype]["DEC"], densdict[objtype]["DENSITY"]
            for i in range(len(ras)):
                ender = "   "
                if i % 2 == 1:
                    ender = "\n"
                link = '<A HREF="http://legacysurvey.org/viewer?'
                link += 'ra={}&dec={}&layer={}&zoom=10" target="external">'.format(
                    ras[i], decs[i], DRs.split(",")[0].lower())
                label = "RA: {:.3f}&deg; DEC: {:.3f}&deg;</A> DENSITY: {:.0f} deg<sup>-2</sup>".format(
                    ras[i], decs[i], dens[i])
                html.write(link+label+ender)
            html.write('</pre>')
        html.write('<table COLS=2 WIDTH="100%">\n')
        html.write('<tr>\n')
        # ADM add the plots...
        html.write('<td align=center><A HREF="skymap-{}.png"><img SRC="skymap-{}.png" width=100% height=auto></A></td>\n'
                   .format(objtype, objtype))
        html.write('<td align=center><A HREF="histo-{}.png"><img SRC="histo-{}.png" width=75% height=auto></A></td>\n'
                   .format(objtype, objtype))
        html.write('</tr>\n')
        html.write('</table>\n')

        # ADM color-color plots.
        html.write('<h2>Color-color diagrams (corrected for Galactic extinction)</h2>\n')
        html.write('<table COLS=3 WIDTH="100%">\n')
        html.write('<tr>\n')
        # ADM add the plots...
        for colors in ["grz", "rzW1", "rzW1W2"]:
            html.write('<td align=center><A HREF="color-{}-{}.png"><img SRC="color-{}-{}.png" width=95% height=auto></A></td>\n'
                       .format(colors, objtype, colors, objtype))
        html.write('</tr>\n')
        html.write('</table>\n')

        # ADM magnitude plots.
        html.write('<h2>Magnitude distributions (uncorrected for Galactic extinction)</h2>\n')
        html.write('<table COLS=4 WIDTH="100%">\n')
        html.write('<tr>\n')
        # ADM add the plots .
        for band in ["g", "r", "z", "W1"]:
            html.write('<td align=center><A HREF="nmag-{}-{}.png"><img SRC="nmag-{}-{}.png" width=95% height=auto></A></td>\n'
                       .format(band, objtype, band, objtype))
        html.write('</tr>\n')
        html.write('</table>\n')
        # ADM add the ASCII files to the images.
        for band in ["g", "r", "z", "W1"]:
            html.write('<td align=center><A HREF="nmag-{}-{}.dat">nmag-{}-{}.dat</A></td>\n'
                       .format(band, objtype, band, objtype))
        html.write('</tr>\n')
        html.write('</table>\n')

        # ADM parallax and proper motion plots, if we have that information.
        if "PARALLAX" in targs.dtype.names and np.sum(targs['PARALLAX'] != 0) > 0:
            html.write('<h2>Gaia diagnostics</h2>\n')
            html.write('<table COLS=2 WIDTH="100%">\n')
            html.write('<tr>\n')
            # ADM add the plots...
            html.write('<td align=center><A HREF="gaia-pm-{}.png"><img SRC="gaia-pm-{}.png" width=75% height=auto></A></td>\n'
                       .format(objtype, objtype))
            html.write('<td align=center><A HREF="gaia-parallax-{}.png"><img SRC="gaia-parallax-{}.png" width=71% height=auto></A></td>\n'
                       .format(objtype, objtype))
            html.write('</tr>\n')
            html.write('</table>\n')

        # ADM add special plots if we have mock data.
        if mocks:
            html.write('<hr>\n')
            html.write('<h1>Mock QA</h1>\n')
            html.write('<h4>No Galactic extinction or photometric scatter.</h4>\n')

            # ADM redshift plots.
            html.write('<h2>True redshift distributions</h2>\n')
            html.write('<table COLS=2 WIDTH="100%">\n')
            html.write('<tr>\n')
            # ADM add the plots...
            html.write('<td align=center><A HREF="mock-nz-{}.png"><img SRC="mock-nz-{}.png" height=auto width=95%></A></td>\n'
                       .format(objtype, objtype))
            html.write('<td align=center><A HREF="mock-zvmag-{}.png"><img SRC="mock-zvmag-{}.png" height=auto width=95%></A></td>\n'
                       .format(objtype, objtype))
            html.write('</tr>\n')
            html.write('</table>\n')

            # ADM color-color plots.
            html.write('<h2>True color-color diagrams</h2>\n')
            html.write('<table COLS=3 WIDTH="100%">\n')
            html.write('<tr>\n')
            # ADM add the plots...
            for colors in ["grz", "rzW1", "rzW1W2"]:
                html.write('<td align=center><A HREF="mock-color-{}-{}.png"><img SRC="mock-color-{}-{}.png" height=auto width=95%></A></td>\n'
                           .format(colors, objtype, colors, objtype))
            html.write('</tr>\n')
            html.write('</table>\n')

            # # ADM classification fraction plots.
            # html.write('<h2>Fraction of each spectral type plots</h2>\n')
            # html.write('<table COLS=1 WIDTH="40%">\n')
            # html.write('<tr>\n')
            # # ADM add the plots...
            # html.write('<td align=center><A HREF="{}-{}.png"><img SRC="{}-{}.png" height=auto width=95%></A></td>\n'
            #           .format("mock-fractype", objtype, "mock-fractype", objtype))
            # html.write('</tr>\n')
            # html.write('</table>\n')

        # ADM add target density vs. systematics plots, if systematics plots were requested.
        # ADM these plots aren't useful if we're looking at commissioning data.
        if systematics and not(cmx):
            # ADM fail if the pixel systematics weights file was not passed.
            if imaging_map_file is None:
                log.error("imaging_map_file was not passed so systematics cannot be tracked. Try again passing systematics=False.")
                raise IOError
            sysdic = _load_systematics()
            sysnames = list(sysdic.keys())
            # ADM html text to embed the systematics plots.
            html.write('<h2>Target Density variation vs. Systematics plots</h2>\n')
            html.write('<table COLS=3 WIDTH="100%">\n')
            html.write('<tr>\n')
            # ADM add the plots...
            while(len(sysnames) > 2):
                for sys in sysnames[:3]:
                    html.write('<td align=center><A HREF="sysdens-{}-{}.png"><img SRC="sysdens-{}-{}.png" height=auto width=95%></A></td>\n'
                               .format(sys, objtype, sys, objtype))
                # ADM pop off the 3 columns of systematics that have already been written.
                sysnames = sysnames[3:]
                html.write('</tr>\n')
            # ADM we popped three systematics at a time, there could be a remaining one or two.
            if len(sysnames) == 2:
                for sys in sysnames:
                    html.write('<td align=center><A HREF="sysdens-{}-{}.png"><img SRC="sysdens-{}-{}.png" height=auto width=95%></A></td>\n'
                               .format(sys, objtype, sys, objtype))
                html.write('</tr>\n')
            if len(sysnames) == 1:
                html.write('<td align=center><A HREF="sysdens-{}-{}.png"><img SRC="sysdens-{}-{}.png" height=auto width=95%></A></td>\n'
                           .format(sysnames[0], objtype, sysnames[0], objtype))
                html.write('</tr>\n')
            html.write('</table>\n')

        # ADM html postamble.
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    if makeplots:
        # ADM add a correlation matrix recording the overlaps between different target
        # ADM classes as a density.
        log.info('Making correlation matrix...t = {:.1f}s'.format(time()-start))
        htmlmain.write('<br><h2>Overlaps in target densities (per sq. deg.)</h2>\n')
        htmlmain.write('<PRE><span class="inner-pre" style="font-size: 14px">\n')
        # ADM only retain classes that are actually in the DESI target bit list.
        settargdens = set(masks[0].names()).intersection(set(targdens))
        # ADM write out a list of the target categories.
        headerlist = list(settargdens)
        headerlist.sort()
        # ADM edit SV target categories to their initial letters to squeeze space.
        hl = headerlist.copy()
        if svs[0:2] == 'SV':
            for tc in 'QSO', 'ELG', 'LRG', 'BGS', 'MWS':
                hl = [h.replace(tc, tc[0]) for h in hl]
                # ADM also change SUPER->SUP, COLOR->COL to squeeze space
                hl = [h.replace('SUPER', 'SUP') for h in hl]
                hl = [h.replace('COLOR', 'COL') for h in hl]
        # ADM truncate the bit names at "trunc" characters to pack them more easily.
        trunc = 8
        truncform = '{:>'+str(trunc)+'s}'
        headerwrite = [bitname[:trunc] for bitname in hl]
        headerwrite.insert(0, " ")
        header = " ".join([truncform.format(i) for i in headerwrite])+'\n\n'
        htmlmain.write(header)
        # ADM for each pair of target classes, determine how many targets per unit area
        # ADM have the relevant target bit set for both target classes in the pair.
        for i, objtype1 in enumerate(headerlist):
            overlaps = [hl[i][:trunc]]
            for j, objtype2 in enumerate(headerlist):
                if j < i:
                    overlaps.append(" ")
                else:
                    dt = targs["DESI_TARGET"]
                    overlap = np.sum(((dt & masks[0][objtype1]) != 0) & ((dt & masks[0][objtype2]) != 0))/totarea
                    overlaps.append("{:.1f}".format(overlap))
            htmlmain.write(" ".join([truncform.format(i) for i in overlaps])+'\n\n')
        # ADM close the matrix text output.
        htmlmain.write('</span></PRE>\n\n\n')
        log.info('Done with correlation matrix...t = {:.1f}s'.format(time()-start))

    # ADM if requested, add systematics plots.
    if systematics:
        from desitarget import io as dtio
        pixmap = dtio.load_pixweight_recarray(imaging_map_file, nside)
        sysdic = _load_systematics()
        sysnames = list(sysdic.keys())
        # ADM html text to embed the systematics plots.
        htmlmain.write('<h2>Systematics plots</h2>\n')
        htmlmain.write('<table COLS=2 WIDTH="100%">\n')
        htmlmain.write('<tr>\n')
        # ADM add the plots...
        while(len(sysnames) > 1):
            for sys in sysnames[:2]:
                htmlmain.write('<td align=center><A HREF="systematics-{}.png"><img SRC="systematics-{}.png" height=auto width=95%></A></td>\n'
                               .format(sys, sys))
                # ADM pop off the 2 columns of systematics that have already been written.
            sysnames = sysnames[2:]
            htmlmain.write('</tr>\n')
        # ADM we popped two systematics at a time, there could be a remaining one.
        if len(sysnames) == 1:
            htmlmain.write('<td align=center><A HREF="systematics-{}.png"><img SRC="systematics-{}.png" height=auto width=95%></A></td>\n'
                           .format(sysnames[0], sysnames[0]))
            htmlmain.write('</tr>\n')
        htmlmain.write('</table>\n')
        # ADM add the plots.
        if makeplots:
            sysnames = list(sysdic.keys())
            for sysname in sysnames:
                # ADM convert the data and the systematics ranges to more human-readable quantities.
                d, u, plotlabel = sysdic[sysname]
                down, up = _prepare_systematics(np.array([d, u]), sysname)
                pixmap[sysname] = _prepare_systematics(pixmap[sysname], sysname)
                # ADM make the systematics sky plots.
                qasystematics_skyplot(pixmap[sysname], sysname,
                                      qadir=qadir, downclip=down, upclip=up, plottitle=plotlabel)
                # ADM make the systematics vs. target density scatter plots
                # ADM for each target type. These plots aren't useful for commissioning.
                if not(cmx):
                    for objtype in targdens.keys():
                        # ADM hack to have different FRACAREA quantities for the sky maps and
                        # ADM the scatter plots.
                        if sysname == "FRACAREA":
                            down = 0.9
                        qasystematics_scatterplot(pixmap, sysname, objtype, qadir=qadir,
                                                  downclip=down, upclip=up, nbins=10, xlabel=plotlabel)
                log.info('Made systematics plots for {}...t ={:.1f}s'.format(
                    sysname, time()-start))
        log.info('Done with systematics...t = {:.1f}s'.format(time()-start))

    # ADM html postamble for main page.
    htmlmain.write('<b><i>Last updated {}</b></i>\n'.format(js))
    htmlmain.write('</html></body>\n')
    htmlmain.close()

    # ADM make sure all of the relevant directories and plots can be read by a web-browser.
    cmd = 'chmod 644 {}/*'.format(qadir)
    ok = os.system(cmd)
    cmd = 'chmod 775 {}'.format(qadir)
    ok = os.system(cmd)
