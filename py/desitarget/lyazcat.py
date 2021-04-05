#This is where the script for making the LyA retargeting ZCAT is going to go, separate from the mtl.py file.
import os
import numpy as np
import healpy as hp
import numpy.lib.recfunctions as rfn
import sys
from astropy.table import Table
from astropy.io import ascii
import fitsio
from time import time
from datetime import datetime
from glob import glob

from . import __version__ as dt_version
from desitarget.targetmask import obsmask, obsconditions, zwarn_mask
from desitarget.targets import calc_priority, calc_numobs_more
from desitarget.targets import main_cmx_or_sv, switch_main_cmx_or_sv
from desitarget.targets import set_obsconditions, decode_targetid
from desitarget.geomask import match, match_to
from desitarget.internal import sharedmem
from desitarget import io

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()

zcatdatamodel = np.array([], dtype=[
    ('RA', '>f8'), ('DEC', '>f8'), ('TARGETID', '>i8'),
    ('NUMOBS', '>i4'), ('Z', '>f8'), ('ZWARN', '>i8'), ('ZTILEID', '>i4'),
    ('Z_COMB','>f8'),('Z_COMB_PROB','>f8'),('Z_QN','>f8'),('Z_QN_CONF','>f8'),
    ('Z_SQ','>f8'),('Z_SQ_CONF','>f8')
    ])

def make_blank_zcat(zcatdir, tiles):
    """Make the simplest possible zcat using SV1-era redrock outputs.
    Parameters
    ----------
    zcatdir : :class:`str`
        Full path to the "daily" directory that hosts redshift catalogs.
    tiles : :class:`~numpy.array`
        Numpy array of tiles to be processed. Must contain at least:
        * TILEID - unique tile identifier.
        * ZDATE - final night processed to complete the tile (YYYYMMDD).
    Returns
    -------
    :class:`~astropy.table.Table`
        A zcat in the official format (`zcatdatamodel`) compiled from
        the `tiles` in `zcatdir`.
    Notes
    -----
    - How the `zcat` is constructed could certainly change once we have
      the final schema in place.
    """
    # ADM the root directory in the data model.
    rootdir = os.path.join(zcatdir, "tiles", "cumulative")

    # ADM for each tile, read in the spectroscopic and targeting info.
    allzs = []
    allfms = []
    for tile in tiles:
        # ADM build the correct directory structure.
        tiledir = os.path.join(rootdir, str(tile["TILEID"]))
        ymdir = os.path.join(tiledir, tile["ZDATE"])
        # ADM and retrieve the redshifts.
        zbestfns = glob(os.path.join(ymdir, "zbest*"))
        for zbestfn in zbestfns:
            zz = fitsio.read(zbestfn, "ZBEST")
            allzs.append(zz)
            # ADM only read in the first set of exposures.
            fm = fitsio.read(zbestfn, "FIBERMAP", rows=np.arange(len(zz)))
            allfms.append(fm)
            # ADM check the correct TILEID was written in the fibermap.
            if set(fm["TILEID"]) != set([tile["TILEID"]]):
                msg = "Directory and fibermap don't match for tile".format(tile)
                log.critical(msg)
                raise ValueError(msg)
    zs = np.concatenate(allzs)
    fms = np.concatenate(allfms)

    # ADM remove -ve TARGETIDs which should correspond to bad fibers.
    zs = zs[zs["TARGETID"] >= 0]
    fms = fms[fms["TARGETID"] >= 0]

    # ADM check the TARGETIDs are unique. If they aren't the likely
    # ADM explanation is that overlapping tiles (which could include
    # ADM duplicate targets) are being processed.
    if len(zs) != len(set(zs["TARGETID"])):
        msg = "a target is duplicated!!! You are likely trying to process "
        msg += "overlapping tiles when one of these tiles should already have "
        msg += "been processed and locked in mtl-done-tiles.ecsv"
        log.critical(msg)
        raise ValueError(msg)

    # ADM currently, the spectroscopic files aren't coadds, so aren't
    # ADM unique. We therefore need to look up (any) coordinates for
    # ADM each z in the fibermap.
    zid = match_to(fms["TARGETID"], zs["TARGETID"])

    # ADM write out the zcat as a file with the correct data model.
    zcat = Table(np.zeros(len(zs), dtype=zcatdatamodel.dtype))

    zcat["RA"] = fms[zid]["TARGET_RA"]
    zcat["DEC"] = fms[zid]["TARGET_DEC"]
    zcat["ZTILEID"] = fms[zid]["TILEID"]
    zcat["NUMOBS"] = zs["NUMTILE"]
    for col in set(zcat.dtype.names) - set(['RA', 'DEC', 'NUMOBS', 'ZTILEID']):
        zcat[col] = zs[col]

    return zcat

def add_qn_data(zcat):
    #This is where the QuasarNET stuff will go.
    print('Adding QuasarNET data is not yet supported.')
    zcat['Z_QN'][:] = -1
    zcat['Z_QN_CONF'][:] = -1
    
    return zcat

def add_sq_data(zcat):
    #This is where the SQUEzE stuff will go.
    print('Adding SQUEzE data is not yet supported.')
    zcat['Z_SQ'][:] = -1
    zcat['Z_SQ_CONF'][:] = -1

    return zcat

def zcomb_selector(zcat,proc_flag=False)
    #This is where the final Z_COMB and Z_COMB_CONF will be selected and 
    # the columns are populated.

    #proc_flag is in case I want to add a procedure later that weighs
    #model fits like eBOSS, or doing more complex comparisons between 
    #QuasarNET and SQUEzE

    zcat['Z_COMB'][:] = zcat['Z']
    zcat['Z_COMB_PROB'][:] = 0.95

    return(zcat)

def zcat_writer(outputdir,zcat,qn_check,sq_check):
    file_dtag = time.strftime('%Y%m%dT%H%M%S')
    outputname = os.path.join(outputdir, 'zbest_lya_', file_dtag, '.fits')

    prim_hdu = fits.PrimaryHDU()
    prim_hdu.header['QN_ADDED'] = str(qn_check)
    prim_hdu.header['SQ_ADDED'] = str(sq_check)

    data_hdu = fits.BinTableHDU.from_columns(zcat,name='ZCATALOG')

    data_out = fits.HDUList([prim_hdu,data_hdu])
    data_out.writeto(outputname)

    return outputname    


if __name__=='__main__':
    zcat_dir = sys.argv[1]
    tile_list = sys.argv[2]
    qn_check = False
    sq_check = False

    rr_zcat = make_empty_zcat(zcat_dir,tile_list)
    if qn_check and sq_check:
        qn_rr_zcat = add_qn_data(rr_zcat)
        sq_qn_rr_zcat = add_sq_data(qn_rr_zcat)
        fzcat = zcomb_selector(sq_qn_rr_zcat)

    elif qn_check and not sq_check:
        qn_rr_zcat = add_qn_data(rr_zcat)
        fzcat = zcomb_selector(qn_rr_zcat)

    elif sq_check and not qn_check:
        sq_rr_zcat = add_sq_data(rr_zcat)
        fzcat = zcomb_selector(sq_rr_zcat)

    else:
        fzcat = zcomb_selector(rr_zcat)

    outputdir = os.path.join(os.getenv('CSCRATCH'), 'lya_test')
    outputname = zcat_writer(outputdir,fzcat)
    
    print('  {} written out correctly.'.format(outputname))


