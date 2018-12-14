"""
desitarget.gfa
==============

Guide/Focus/Alignment targets
"""

import fitsio
import numpy as np
import os.path
import glob
import os

import desimodel.focalplane
import desimodel.io

import desitarget.io
from desitarget.internal import sharedmem
from desitarget.gaiamatch import match_gaia_to_primary
from desitarget.targets import encode_targetid

from time import time

# ADM set up default DESI logger
from desiutil.log import get_logger
log = get_logger()
start = time()

# ADM the current data model for columns in the GFA files
gfadatamodel = np.array([], dtype=[
    ('TARGETID', 'i8'),  ('BRICKID', 'i4'), ('BRICK_OBJID', 'i4'),
    ('RA', 'f8'), ('DEC', 'f8'), ('RA_IVAR', 'f4'), ('DEC_IVAR', 'f4'),
    ('TYPE', 'S4'),
    ('FLUX_G', 'f4'), ('FLUX_R', 'f4'), ('FLUX_Z', 'f4'),
    ('FLUX_IVAR_G', 'f4'), ('FLUX_IVAR_R', 'f4'), ('FLUX_IVAR_Z', 'f4'),
    ('REF_ID', 'i8'),
    ('PMRA', 'f4'), ('PMDEC', 'f4'), ('PMRA_IVAR', 'f4'), ('PMDEC_IVAR', 'f4'),
    ('GAIA_PHOT_G_MEAN_MAG', '>f4'), ('GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', '>f4'),
    ('GAIA_ASTROMETRIC_EXCESS_NOISE', '>f4')
])


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

    # load all sweep data
    sweep_data = []
    # n_sweep = 10

    for i in range(n_sweep):
        sweep_file = sweep_files[i]
        data = fitsio.read(sweep_file, columns=['RA', 'DEC', 'FLUX_R'])

        # - Keep just mag>18
        rfluxlim = 10**(0.4*(22.5-18))
        ii = data['FLUX_R'] > rfluxlim
        data = data[ii]

        # - Faster for a small number of test tiles, but slower if using all tiles
        # keep = np.zeros(len(data), dtype=bool)
        # for tile in desi_tiles:
        #     keep |= near_tile(data, tile['RA'], tile['DEC'])
        # if np.any(keep):
        #     sweep_data.append(data[keep])

        sweep_data.append(data)

        log.info('Loaded file {} out of {}'.format(i, n_sweep))

    all_sweep = np.concatenate(sweep_data, axis=0)

    log.info('There are {:.2f}M targets in the sweeps'.format(len(all_sweep)/1E6))

    # find IDs of targets on every individual tile
    for i in range(len(desi_tiles)):
        tile_id = desi_tiles['TILEID'][i]
        log.info('computing TILEID {:05d} on RA {:6.2f} DEC {:6.2f}'.format(tile_id, desi_tiles['RA'][i], desi_tiles['DEC'][i]))

        # select targets in a smaller window centered on tile
        jj = near_tile(all_sweep, desi_tiles['RA'][i], desi_tiles['DEC'][i])

        # find GFA targets in the smaller input window
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

    # rewrite a new tilefile with all the info in three HDUs
    gfa_files = glob.glob(os.path.join(gfa_file_path, "gfa_targets_*.fits"))
    gfa_tile_id = {}
    for gfa_file in gfa_files:
        f = gfa_file.split('/')[-1]
        fileid = f.split("_")[-1]
        fileid = fileid[0:5]
        gfa_tile_id[fileid] = gfa_file

    if fa_file_path:
        fiberassign_tilefiles = glob.glob(os.path.join(fa_file_path, "tile*.fits"))
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


def gaia_gfas_from_sweep(objects, maglim=18.,
                         gaiamatch=False, gaiabounds=[0., 360., -90., 90.]):
    """Create a set of GFAs from Gaia-matching for one sweep file or sweep objects

    Parameters
    ----------
    objects: :class:`numpy.ndarray` or `str`
        Numpy structured array with UPPERCASE columns needed for target selection, OR
        a string corresponding to a sweep filename.
    maglim : :class:`float`, optional, defaults to 18
        Magnitude limit for GFAs in Gaia G-band.
    gaiamatch : defaults to ``False``
        If ``True``, match to Gaia DR2 chunks files and populate
        Gaia columns, otherwise assume those columns already exist
    gaiabounds : :class:`list`, optional, defaults to the whole sky
        The area over which to retrieve Gaia objects that don't match a sweeps object.
        Pass a 4-entry list to form a box bounded by [RAmin, RAmax, DECmin, DECmax].

    Returns
    -------
    :class:`numpy.ndarray`
        GFA objects from Gaia for the region bounded by `gaiabounds`, formatted
        according to `desitarget.gfa.gfadatamodel`.
    """
    # ADM read in objects if a filename was passed instead of the actual data
    if isinstance(objects, str):
        objects = desitarget.io.read_tractor(objects)

    # ADM issue a warning if gaiamatch was not sent but there's no Gaia information
    if np.max(objects['PARALLAX']) == 0. and not gaiamatch:
        log.warning("Zero objects have a parallax. Did you mean to send gaiamatch?")

    # ADM add the Gaia coordinate columns if they don't exist and
    # ADM if Gaia-matching was requested
    if gaiamatch and "GAIA_RA" not in objects.dtype.names:
        gc = np.array([], dtype=[('GAIA_RA', '>f8'), ('GAIA_DEC', '>f8')])
        dt = objects.dtype.descr + gc.dtype.descr
        nrows = len(objects)
        objectswgc = np.zeros(nrows, dtype=dt)
        for col in objects.dtype.names:
            objectswgc[col] = objects[col]
        objects = objectswgc

    # ADM As a mild speed up, only consider sweeps objects brighter than 3 mags
    # ADM fainter than the passed Gaia magnitude limit. Note that Gaia G-band
    # ADM approximates SDSS r-band.
    w = np.where((objects["FLUX_G"] > 10**((22.5-(maglim+3))/2.5)) |
                 (objects["FLUX_R"] > 10**((22.5-(maglim+3))/2.5)) |
                 (objects["FLUX_Z"] > 10**((22.5-(maglim+3))/2.5)))[0]
    objects = objects[w]

    nobjs = len(objects)

    # ADM match the sweeps objects to Gaia retaining Gaia objects that do not
    # ADM have a match in the sweeps, if Gaia matching was requested
#    log.info('Starting Gaia match for {} objects...t = {:.1f}s'
#             .format(nobjs,time()-start))
    if gaiamatch:
        # ADM match with a fairly discriminating radius (0.1 arcsec) to just
        # ADM get the best sweeps-Gaia correspondence
        gaiainfo = match_gaia_to_primary(objects, matchrad=0.1,
                                         retaingaia=True, gaiabounds=gaiabounds)
        log.info('Done with Gaia match...t = {:.1f}s'.format(time()-start))
        # ADM add the Gaia column information to the primary array
        for col in gaiainfo.dtype.names:
            objects[col] = gaiainfo[col][:nobjs]

        # ADM an additional array to hold the Gaia objects that have no sweeps match
        supg = np.zeros(len(gaiainfo) - nobjs, dtype=objects.dtype)
        # ADM make sure all of these additional columns have "ridiculous" numbers
        supg[...] = -1
        # ADM but default the IVARs that would appear in the sweeps (g/r/z) to 0
        for col in ["FLUX_IVAR_G", "FLUX_IVAR_R", "FLUX_IVAR_Z"]:
            supg[col] = 0.
        # ADM and then TYPE to PSF
        supg["TYPE"] = 'PSF'
        # ADM populate these additional objects
        for col in gaiainfo.dtype.names:
            supg[col] = gaiainfo[col][nobjs:]

        # ADM combine the primary and supplemental arrays
        objects = np.hstack([objects, supg])

        # ADM store the Gaia RA/DEC as the default for matched objects
        # ADM as it's really the Gaia astrometry we want
        for col in ["RA", "DEC"]:
            objects[col] = objects["GAIA_"+col]

    # ADM only retain objects with Gaia matches
    # ADM it's fine to propagate an empty array if there are no matches
    # ADM note that the sweeps use 0 for objects with no REF_ID
    # ADM and desitarget.gaiamatch uses -1. So, > 0 checks both.
    w = np.where(objects["REF_ID"] > 0)[0]
    objects = objects[w]

    # ADM it's possible that a Gaia object matches two sweeps objects, so
    # ADM only record unique Gaia IDs
    _, ind = np.unique(objects["REF_ID"], return_index=True)
#    log.info('Removed {} duplicated Gaia objects...t = {:.1f}s'
#             .format(len(objects)-len(ind),time()-start))
    objects = objects[ind]

    # ADM determine a TARGETID for any objects on a brick (this should
    # ADM end up as -1 for anything that is Gaia-only (from Gaia-matching)
    # ADM as all of objid, brickid and release should be -1
    targetid = encode_targetid(objid=objects['OBJID'],
                               brickid=objects['BRICKID'],
                               release=objects['RELEASE'])

    # ADM format everything according to the data model
    gfas = np.zeros(len(objects), dtype=gfadatamodel.dtype)
    # ADM make sure all columns initially have "ridiculous" numbers
    gfas[...] = -99.
    # ADM remove the TARGETID and BRICK_OBJID columns and populate them later
    # ADM as they require special treatment
    cols = list(gfadatamodel.dtype.names)
    for col in ["TARGETID", "BRICK_OBJID"]:
        cols.remove(col)
    for col in cols:
        gfas[col] = objects[col]
    # ADM populate the TARGETID column
    gfas["TARGETID"] = targetid
    # ADM populate the BRICK_OBJID column
    gfas["BRICK_OBJID"] = objects["OBJID"]

    # ADM cut the GFAs by a hard limit on magnitude
    w = np.where(gfas['GAIA_PHOT_G_MEAN_MAG'] < maglim)[0]
    gfas = gfas[w]

    # ADM a final clean-up to remove columns that are Nan (from
    # ADM Gaia-matching) or are 0 (in the sweeps)
    for col in ["PMRA", "PMDEC"]:
        w = np.where(~np.isnan(gfas[col]) & (gfas[col] != 0))[0]
        gfas = gfas[w]
#    log.info('Removed {} Gaia objects with NaN columns...t = {:.1f}s'
#             .format(len(objects)-len(gfas),time()-start))

    return gfas


def decode_sweep_name(sweepname):
    """Retrieve RA/Dec edges from a full directory path to a sweep file

    Parameters
    ----------
    sweepname : :class:`str`
        Full path to a sweep file, e.g., /a/b/c/sweep-350m005-360p005.fits

    Returns
    -------
    :class:`list`
        A 4-entry list of the edges of the region covered by the sweeps file
        in the form [RAmin, RAmax, DECmin, DECmax]
        For the above example this would be [350., 360., -5., 5.]
    """
    # ADM extract just the file part of the name
    sweepname = os.path.basename(sweepname)

    # ADM the RA/Dec edges
    ramin, ramax = float(sweepname[6:9]), float(sweepname[14:17])
    decmin, decmax = float(sweepname[10:13]), float(sweepname[18:21])

    # ADM flip the signs on the DECs, if needed
    if sweepname[9] == 'm':
        decmin *= -1
    if sweepname[17] == 'm':
        decmax *= -1

    return [ramin, ramax, decmin, decmax]


def select_gfas(infiles, maglim=18, numproc=4, gaiamatch=False):
    """Create a set of GFA locations using Gaia

    Parameters
    ----------
    infiles : :class:`list` or `str`
        A list of input filenames (sweep files) OR a single filename.
    maglim : :class:`float`, optional, defaults to 18
        Magnitude limit for GFAs in Gaia G-band.
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    gaiamatch : defaults to ``False``
        If ``True``, match to Gaia DR2 chunks files and populate
        Gaia columns, otherwise assume those columns already exist

    Returns
    -------
    :class:`numpy.ndarray`
        GFA objects from Gaia across all of the passed input files, formatted
        according to `desitarget.gfa.gfadatamodel`.

    Notes
    -----
        - if numproc==1, use the serial code instead of the parallel code.
    """

    # ADM convert a single file, if passed to a list of files
    if isinstance(infiles, str):
        infiles = [infiles, ]

    # ADM check that files exist before proceeding
    for filename in infiles:
        if not os.path.exists(filename):
            raise ValueError("{} doesn't exist".format(filename))

    nfiles = len(infiles)

    # ADM the critical function to run on every file
    def _get_gfas(fn):
        '''wrapper on gaia_gfas_from_sweep() given a file name'''
        # ADM we need to pass the boundaries of the sweeps file, too
        bounds = decode_sweep_name(fn)

        return gaia_gfas_from_sweep(
            fn, maglim=maglim, gaiamatch=gaiamatch, gaiabounds=bounds
        )

    # ADM this is just to count sweeps files in _update_status
    nfile = np.zeros((), dtype='i8')

    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if nfile % 50 == 0 and nfile > 0:
            rate = nfile / (time() - t0)
            log.info('{}/{} files; {:.1f} files/sec'.format(nfile, nfiles, rate))
        nfile[...] += 1    # this is an in-place modification
        return result

    # - Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            gfas = pool.map(_get_gfas, infiles, reduce=_update_status)
    else:
        gfas = list()
        for file in infiles:
            gfas.append(_update_status(_get_gfas(file)))

    gfas = np.concatenate(gfas)

    return gfas
