"""
desitarget.gfa
==============

Guide/Focus/Aligment targets
"""

import desimodel.focalplane
import fitsio
import numpy as np
import os.path
import desitarget.io
import desimodel.io
import glob
import os


def near_tile(data, tilera, tiledec, window_ra=4.0, window_dec=4.0):
    """Trims the input data to a rectangular windonw in RA,DEC.

    Parameters
    ----------
    data : :class:`np.ndarray`
        Array with target data. Includes at least 'RA' and 'DEC' columns.
    tilera: :class:`float`
        Scalar with the central RA coordinate.
    tiledec: :class:`float`
        Scalar with the central DEC coordinate
    window_ra: :class:`float`
        Value of the window in RA to trim the data.
    window_dec: :class:`float`
        Value of the window in DEC to trim the data.

    Returns
    -------
    :class:`bool`
        Boolean array. True if the target falls inside the window. False otherwise.
    """
    delta_RA = data['RA'] - tilera
    delta_dec = data['DEC'] - tiledec
    jj = np.fabs(delta_RA) < window_ra
    jj = jj | ((delta_RA + 360.0) < window_ra)
    jj = jj | ((360.0 - delta_RA) < window_ra)
    jj = jj & (np.fabs(delta_dec) < window_dec)
    return jj


def write_gfa_targets(sweep_dir="./", desi_tiles=None, output_path="./", log=None):
    """Computes and writes to disk GFA targets for every tile

    Parameters
    ----------
    sweep_dir : :class:`string`
        Path to the sweep files.

    desi_tiles: :class:`np.ndarray`
        Set of desitiles to compute the GFA targets.

    output_path : :class:`string`
        Path where the "gfa_targets_tile" files will be written.

    log : :class: `desiutil.log`
        Desiutil logger
    """

    if log is None:
        from desiutil.log import get_logger
        log = get_logger()

    if desi_tiles is None:
        desi_tiles = desimodel.io.load_tiles()

    # list sweep files to be used
    sweep_files = desitarget.io.list_sweepfiles(sweep_dir)
    n_sweep = len(sweep_files)
    log.info('{} sweep files'.format(len(sweep_files)))

    #load all sweep data
    sweep_data = []
    #n_sweep = 10

    for i in range(n_sweep):
        sweep_file = sweep_files[i]
        data = fitsio.read(sweep_file, columns=['RA', 'DEC', 'FLUX_R'])

        #- Keep just mag>18
        rfluxlim = 10**(0.4*(22.5-18))
        ii = data['FLUX_R'] > rfluxlim
        data = data[ii]

        #- Faster for a small number of test tiles, but slower if using all tiles
        # keep = np.zeros(len(data), dtype=bool)
        # for tile in desi_tiles:
        #     keep |= near_tile(data, tile['RA'], tile['DEC'])
        # if np.any(keep):
        #     sweep_data.append(data[keep])

        sweep_data.append(data)

        log.info('Loaded file {} out of {}'.format(i, n_sweep))

    all_sweep = np.concatenate(sweep_data, axis=0)

    log.info('There are {:.2f}M targets in the sweeps'.format(len(all_sweep)/1E6))

    #find IDs of targets on every individual tile
    for i in range(len(desi_tiles)):
        tile_id = desi_tiles['TILEID'][i]
        log.info('computing TILEID {:05d} on RA {:6.2f} DEC {:6.2f}'.format(tile_id, desi_tiles['RA'][i], desi_tiles['DEC'][i]))

        # select targets in a smaller window centered on tile
        jj = near_tile(all_sweep, desi_tiles['RA'][i], desi_tiles['DEC'][i])

        #find GFA targets in the smaller input window
        if np.count_nonzero(jj):
            mini_sweep = all_sweep[jj]
            log.info('Inside mini_sweep: {:.2f}M targets'.format(len(mini_sweep)/1E6))

            targetindices, gfaindices = desimodel.focalplane.on_tile_gfa(tile_id, mini_sweep)
            log.info('Found {:d} targets on TILEID {:05d}'.format(len(targetindices), tile_id))

            if len(targetindices):
                gfa_targets = np.lib.recfunctions.append_fields(
                    mini_sweep[targetindices], 'GFA_LOC', gfaindices,
                    usemask=False)

                filename = os.path.join(output_path, "gfa_targets_tile_{:05d}.fits".format(tile_id))
                log.info("writing to {}".format(filename))
                a = fitsio.write(filename, gfa_targets, extname='GFA', clobber=True)


def add_gfa_info_to_fa_tiles(gfa_file_path="./", fa_file_path=None, output_path=None, log=None):
    """Adds GFA info into fiberassign tiles.

    Parameters
    ----------
    gfa_file_path : :class:`string`
        Path to the "gfa_targets_tile" files.

    fa_file_path : :class:`string`
        Path to the results of fiberassign.

    output_path : :class:`string`
        Path where the "tile_*" files will be rewritten including the GFA info

    log : :class: `desiutil.log`
        Desiutil logger
    """
    if log is None:
        from desiutil.log import get_logger
        log = get_logger()
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    #rewrite a new tilefile with all the info in three HDUs
    gfa_files = glob.glob(os.path.join(gfa_file_path, "gfa_targets_*.fits"))
    gfa_tile_id = {}
    for gfa_file in gfa_files:
        f = gfa_file.split('/')[-1]
        fileid = f.split("_")[-1]
        fileid = fileid[0:5]
        gfa_tile_id[fileid] = gfa_file

    if fa_file_path:
        fiberassign_tilefiles = glob.glob(os.path.join(fa_file_path,"tile*.fits"))
        log.info('{} fiberassign tile files'.format(len(fiberassign_tilefiles)))
    else:
        fiberassign_tilefiles = []
        log.info('Empty fiberassign path')

    fa_tile_id = {}
    for fa_file in fiberassign_tilefiles:
        f = fa_file.split('/')[-1]
        fileid = f.split("_")[-1]
        fileid = fileid[0:5]
        fa_tile_id[fileid] = fa_file

    for gfa_id in gfa_tile_id.keys():
        if gfa_id in fa_tile_id.keys():
            log.info('rewriting tilefile for tileid {}'.format(gfa_id))
            gfa_data = fitsio.read(gfa_tile_id[gfa_id])
            fiber_data = fitsio.read(fa_tile_id[gfa_id], ext=1)
            potential_data = fitsio.read(fa_tile_id[gfa_id], ext=2)

            tileout = os.path.join(output_path, 'tile_{}.fits'.format(gfa_id))
            fitsio.write(tileout, fiber_data, extname='FIBERASSIGN', clobber=True)
            fitsio.write(tileout, potential_data, extname='POTENTIAL')
            fitsio.write(tileout, gfa_data, extname='GFA')


def gaia_gfa_from_sweep(objects, 
            gaiadir='/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom'):
    """Create a set of GFAs from Gaia-matching for one sweep file or sweep objects

    Parameters
    ----------
    objects: :class:`numpy.ndarray` or `str`
        Numpy structured array with UPPERCASE columns needed for target selection, OR 
        a string tractor/sweep filename.
    gaiadir : :class:`str`, optional, defaults to Gaia DR2 path at NERSC
        Root directory of a Gaia Data Release as used by the Legacy Surveys.

    Returns
    -------
    :class:`numpy.ndarray`
        The corresponding Gaia objects, with sweeps information added where available
    """
    #ADM read in objects if a filename was passed instead of the actual data
    if isinstance(objects, str):
        objects = io.read_tractor(objects)

    
