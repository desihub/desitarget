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
import os, re
from collections import defaultdict
from glob import glob

from . import __version__ as desitarget_version
from . import gitversion

from desiutil import depend

def construct_QA_file(nrows):
    """Create a recarray to be populated with QA information

    Parameters
    ----------
    nrows : :class:`int`
        Number of rows in the recarray (size, in rows, of expected fits output)    

    Returns
    -------
    :class:`recarray`
         numpy structured array of brick information with nrows as specified
         and columns as below
    """

    data = np.zeros(nrows, dtype=[
            ('BRICKID','>i4'),('BRICKNAME','S8'),('BRICKAREA','>f4'),
            ('RA','>f4'),('DEC','>f4'),
            ('RA1','>f4'),('RA2','>f4'),
            ('DEC1','>f4'),('DEC2','>f4'),
            ('EBV','>f4'),
            ('DEPTH_G','>f4'),('DEPTH_R','>f4'),('DEPTH_Z','>f4'),
            ('GALDEPTH_G','>f4'),('GALDEPTH_R','>f4'),('GALDEPTH_Z','>f4'),
            ('DENSITY_ALL','>f4'),
            ('DENSITY_ELG','>f4'),('DENSITY_LRG','>f4'),
            ('DENSITY_QSO','>f4'),('DENSITY_LYA','>f4'), 
            ('DENSITY_BGS','>f4'),('DENSITY_MWS','>f4'),
            ('DENSITY_BAD_ELG','>f4'),('DENSITY_BAD_LRG','>f4'),
            ('DENSITY_BAD_QSO','>f4'),('DENSITY_BAD_LYA','>f4'),
            ('DENSITY_BAD_BGS','>f4'),('DENSITY_BAD_MWS','>f4'),
            ])
    return data

def populate_brick_info(instruc,brickids,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3/'):
    """Add brick-related information to a numpy array of brickids

    Parameters
    ----------
    instruc : :class:`recarray`
        numpy structured array containing at least
        ['BRICKNAME','BRICKID','RA','DEC','RA1','RA2','DEC1','DEC2','EBV']) to populate
    brickids : :class:`recarray`
        numpy structured array (single list) of BRICKID integers
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory for a data release...e.g. for dr3 this would be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3/

    Returns
    -------
    :class:`recarray`
         instruc with the brick information columns now populated
    """

    #ADM columns to be populated (could change to an input for more sophistication)
    cols = ['BRICKNAME','BRICKID','RA','DEC','RA1','RA2','DEC1','DEC2']

    #ADM read in the brick information file
    fx = fitsio.FITS(rootdirname+'/survey-bricks.fits.gz', upper=True)
    brickdata = fx[1].read(columns=cols)
    #ADM populate the coordinate/name/ID columns
    for col in cols:
        instruc[col] = brickdata[brickids-1][col]

    #ADM read in the data-release specific
    #ADM read in the brick information file
    fx = fitsio.FITS(glob(rootdirname+'/survey-bricks-dr*.fits.gz')[0], upper=True)
    ebvdata = fx[1].read(columns=['BRICKNAME','EBV'])

    #ADM as the BRICKID isn't in the dr-specific file, create
    #ADM a look-up dictionary to match indices via a list comprehension
    orderedbricknames = instruc["BRICKNAME"]
    dd = defaultdict(list)
    for index, item in enumerate(ebvdata["BRICKNAME"]):
        dd[item].append(index)
    matches = [index for item in orderedbricknames for index in dd[item] if item in dd]

    #ADM populate E(B-V)
    instruc['EBV'] = ebvdata[matches]['EBV']

    return instruc
    

def populate_depths(instruc,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3/'):
    """Add depth-related information to a numpy array

    Parameters
    ----------
    instruc : :class:`recarray`
        numpy structured array containing at least
        ['BRICKNAME','BRICKAREA''DEPTH_G','DEPTH_R','DEPTH_Z',
        'GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z']
        to populate with depths and areas
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory for a data release...e.g. for dr3 this would be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3/

    Returns
    -------
    :class:`recarray`
         instruc with the per-brick depth and area columns now populated
    """
    #ADM the pixel scale area for a brick (in sq. deg.)
    pixtodeg = 0.262/3600./3600.

    #ADM read in the brick depth file
    fx = fitsio.FITS(glob(rootdirname+'*depth.fits.gz')[0], upper=True)
    depthdata = fx[1].read()

    #ADM construct the magnitude bin centers for the per-brick depth
    #ADM file, which is expressed as a histogram of 50 bins of 0.1mag
    magbins = np.arange(50)*0.1+20.05

    #ADM a list to contain the brick names and for the depths and areas
    names = []
    depth_g = []
    depth_r = []
    depth_z = []
    galdepth_g = []
    galdepth_r = []
    galdepth_z = []
    areas = []

    #ADM build a per-brick weighted depth. Also determine pixel-based area of brick.
    #ADM the per-brick depth file is histogram of 50 bins
    for i in range(0,len(depthdata),50):
        #ADM there must be pixels in one band
        totpix = max(sum(depthdata[i:i+50]['COUNTS_GAL_G']),sum(depthdata[i:i+50]['COUNTS_GAL_R']),sum(depthdata[i:i+50]['COUNTS_GAL_Z']))
        #ADM pixel-weighted depth
        depth_g.append(np.sum(depthdata['COUNTS_PTSRC_G'][i:i+50]*magbins)/totpix)
        depth_r.append(np.sum(depthdata['COUNTS_PTSRC_R'][i:i+50]*magbins)/totpix)
        depth_z.append(np.sum(depthdata['COUNTS_PTSRC_Z'][i:i+50]*magbins)/totpix)
        galdepth_g.append(np.sum(depthdata['COUNTS_GAL_G'][i:i+50]*magbins)/totpix)
        galdepth_r.append(np.sum(depthdata['COUNTS_GAL_R'][i:i+50]*magbins)/totpix)
        galdepth_z.append(np.sum(depthdata['COUNTS_GAL_Z'][i:i+50]*magbins)/totpix)
        #ADM name and pixel based area of the brick
        names.append(depthdata['BRICKNAME'][i])
        areas.append(totpix*pixtodeg)

    #ADM HACK HACK HACK
    #ADM first find bricks that are not in the depth file and populate them
    #ADM with nonsense. This is a hack as I'm not sure why such bricks exist
    #ADM HACK HACK HACK
    orderedbricknames = instruc["BRICKNAME"]
    badbricks = np.ones(len(orderedbricknames))
    dd = defaultdict(list)
    for index, item in enumerate(orderedbricknames):
        dd[item].append(index)
    matches = [index for item in names for index in dd[item] if item in dd]
    badbricks[matches] = 0
    w = np.where(badbricks)
    badbricknames = orderedbricknames[w]
    for i, badbrickname in enumerate(badbricknames):
        names.append(badbrickname)
        areas.append(-99.)
        depth_g.append(-99.)
        depth_r.append(-99.)
        depth_z.append(-99.)
        galdepth_g.append(-99.)
        galdepth_r.append(-99.)
        galdepth_z.append(-99.)

    #ADM, now order the brickname to match the input structure using
    #ADM a look-up dictionary to match indices via a list comprehension
    orderedbricknames = instruc["BRICKNAME"]
    dd = defaultdict(list)
    for index, item in enumerate(names):
        dd[item].append(index)
    matches = [index for item in orderedbricknames for index in dd[item] if item in dd]

    #ADM populate the depths
    instruc['BRICKAREA'] = np.array(areas)[matches]
    instruc['DEPTH_G'] = np.array(depth_g)[matches]
    instruc['DEPTH_R'] = np.array(depth_r)[matches]
    instruc['DEPTH_Z'] = np.array(depth_z)[matches]
    instruc['GALDEPTH_G'] = np.array(galdepth_g)[matches]
    instruc['GALDEPTH_R'] = np.array(galdepth_r)[matches]
    instruc['GALDEPTH_Z'] = np.array(galdepth_z)[matches]

    return instruc


def brick_info(targetfilename,rootdirname='/global/project/projectdirs/cosmo/data/legacysurvey/dr3/',outfilename='brick-info-dr3.fits'):
    """

    Parameters
    ----------
    targetfilename : :class:`str`
        File name of a list of targets created by select_targets
    rootdirname : :class:`str`, optional, defaults to dr3
        Root directory for a data release...e.g. for dr3 this would be
        /global/project/projectdirs/cosmo/data/legacysurvey/dr3/
    outfilename: :class:`str`
        Output file name for the brick_info file, which will be written as FITS

    Returns
    -------
    :class:`recarray`
         numpy structured array of brick information with columns as in construct_QA_file
    """

    start = time()

    #ADM read in target file
    print('Reading in target file...t = {:.1f}s'.format(time()-start))
    fx = fitsio.FITS(targetfilename, upper=True)
    targetdata = fx[1].read(columns=['BRICKID','DESI_TARGET','BGS_TARGET','MWS_TARGET'])

    print('Determining unique bricks...t = {:.1f}s'.format(time()-start))
    #ADM determine number of unique bricks and their integer IDs
    brickids = np.array(list(set(targetdata['BRICKID'])))
    brickids.sort()

    print('Creating output brick structure...t = {:.1f}s'.format(time()-start))
    #ADM set up an output structure of size of the number of unique bricks
    nbricks = len(brickids)
    outstruc = construct_QA_file(nbricks)

    print('Adding brick information...t = {:.1f}s'.format(time()-start))
    #ADM add brick-specific information based on the brickids
    outstruc = populate_brick_info(outstruc,brickids,rootdirname)

    print('Adding depth information...t = {:.1f}s'.format(time()-start))
    #ADM add per-brick depth and area information
    outstruc = populate_depths(outstruc,rootdirname)
   
    print('Adding target density information...t = {:.1f}s'.format(time()-start))
    #ADM bits and names of interest for desitarget
    #ADM -1 as a bit will return all values
    bitnames = ["DENSITY_ALL","DENSITY_LRG","DENSITY_ELG",
                "DENSITY_QSO","DENSITY_BGS","DENSITY_MWS"]
    bitvals = [-1]+list(2**np.array([0,1,2,60,61]))

    #ADM loop through bits and populate target densities for each class
    for i, bitval in enumerate(bitvals):
        w = np.where(targetdata["DESI_TARGET"] & bitval)
        if len(w[0]):
            targsperbrick = np.bincount(targetdata[w]['BRICKID'])
            outstruc[bitnames[i]] = targsperbrick[outstruc['BRICKID']]/outstruc['BRICKAREA']

    print('Writing output file...t = {:.1f}s'.format(time()-start))
    #ADM everything should be populated, just write it out
    fitsio.write(outfilename, outstruc, extname='BRICKINFO', clobber=True)

    print('Done...t = {:.1f}s'.format(time()-start))
    return outstruc



