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
from time import time
import healpy as hp

import desimodel.focalplane
import desimodel.io
from desimodel.footprint import is_point_in_desi

import desitarget.io
from desitarget.internal import sharedmem
from desitarget.gaiamatch import read_gaia_file, find_gaia_files_beyond_gal_b
from desitarget.gaiamatch import find_gaia_files_tiles, find_gaia_files_box
from desitarget.gaiamatch import find_gaia_files_hp, _get_gaia_nside, gaia_psflike
from desitarget.uratmatch import match_to_urat
from desitarget.targets import encode_targetid, resolve
from desitarget.geomask import is_in_gal_box, is_in_box, is_in_hp
from desitarget.geomask import bundle_bricks, sweep_files_touch_hp

from desiutil import brick
from desiutil.log import get_logger

# ADM set up the Legacy Surveys bricks object.
bricks = brick.Bricks(bricksize=0.25)
# ADM set up the default DESI logger.
log = get_logger()

# ADM the current data model for columns in the GFA files.
gfadatamodel = np.array([], dtype=[
    ('RELEASE', '>i4'), ('TARGETID', 'i8'),
    ('BRICKID', 'i4'), ('BRICK_OBJID', 'i4'),
    ('RA', 'f8'), ('DEC', 'f8'), ('RA_IVAR', 'f4'), ('DEC_IVAR', 'f4'),
    ('TYPE', 'S4'), ('MASKBITS', '>i2'),
    ('FLUX_G', 'f4'), ('FLUX_R', 'f4'), ('FLUX_Z', 'f4'),
    ('FLUX_IVAR_G', 'f4'), ('FLUX_IVAR_R', 'f4'), ('FLUX_IVAR_Z', 'f4'),
    ('REF_ID', 'i8'), ('REF_CAT', 'S2'), ('REF_EPOCH', 'f4'),
    ('PARALLAX', 'f4'), ('PARALLAX_IVAR', 'f4'),
    ('PMRA', 'f4'), ('PMDEC', 'f4'), ('PMRA_IVAR', 'f4'), ('PMDEC_IVAR', 'f4'),
    ('GAIA_PHOT_G_MEAN_MAG', '>f4'), ('GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', '>f4'),
    ('GAIA_PHOT_BP_MEAN_MAG', '>f4'), ('GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', '>f4'),
    ('GAIA_PHOT_RP_MEAN_MAG', '>f4'), ('GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', '>f4'),
    ('GAIA_ASTROMETRIC_EXCESS_NOISE', '>f4'), ('URAT_ID', '>i8'), ('URAT_SEP', '>f4')
])


def gaia_morph(gaia):
    """Retrieve morphological type for Gaia sources.

    Parameters
    ----------
    gaia: :class:`~numpy.ndarray`
        Numpy structured array containing at least the columns,
        `GAIA_PHOT_G_MEAN_MAG` and `GAIA_ASTROMETRIC_EXCESS_NOISE`.

    Returns
    -------
    :class:`~numpy.array`
        An array of strings that is the same length as the input array
        and is set to either "GPSF" or "GGAL" based on a
        morphological cut with Gaia.
    """
    # ADM determine which objects are Gaia point sources.
    g = gaia['GAIA_PHOT_G_MEAN_MAG']
    aen = gaia['GAIA_ASTROMETRIC_EXCESS_NOISE']
    psf = gaia_psflike(aen, g)

    # ADM populate morphological information.
    morph = np.zeros(len(gaia), dtype=gfadatamodel["TYPE"].dtype)
    morph[psf] = b'GPSF'
    morph[~psf] = b'GGAL'

    return morph


def gaia_gfas_from_sweep(filename, maglim=18.):
    """Create a set of GFAs for one sweep file.

    Parameters
    ----------
    filename: :class:`str`
        A string corresponding to the full path to a sweep file name.
    maglim : :class:`float`, optional, defaults to 18
        Magnitude limit for GFAs in Gaia G-band.

    Returns
    -------
    :class:`~numpy.ndarray`
        GFA objects from Gaia, formatted according to `desitarget.gfa.gfadatamodel`.
    """
    # ADM read in the objects.
    objects = fitsio.read(filename)

    # ADM As a mild speed up, only consider sweeps objects brighter than 3 mags
    # ADM fainter than the passed Gaia magnitude limit. Note that Gaia G-band
    # ADM approximates SDSS r-band.
    ii = ((objects["FLUX_G"] > 10**((22.5-(maglim+3))/2.5)) |
          (objects["FLUX_R"] > 10**((22.5-(maglim+3))/2.5)) |
          (objects["FLUX_Z"] > 10**((22.5-(maglim+3))/2.5)))
    objects = objects[ii]
    nobjs = len(objects)

    # ADM only retain objects with Gaia matches.
    # ADM It's fine to propagate an empty array if there are no matches
    # ADM The sweeps use 0 for objects with no REF_ID.
    objects = objects[objects["REF_ID"] > 0]

    # ADM determine a TARGETID for any objects on a brick.
    targetid = encode_targetid(objid=objects['OBJID'],
                               brickid=objects['BRICKID'],
                               release=objects['RELEASE'])

    # ADM format everything according to the data model.
    gfas = np.zeros(len(objects), dtype=gfadatamodel.dtype)
    # ADM make sure all columns initially have "ridiculous" numbers.
    gfas[...] = -99.
    gfas["REF_CAT"] = ""
    gfas["REF_EPOCH"] = 2015.5
    # ADM remove the TARGETID, BRICK_OBJID, REF_CAT, REF_EPOCH columns
    # ADM and populate them later as they require special treatment.
    cols = list(gfadatamodel.dtype.names)
    for col in ["TARGETID", "BRICK_OBJID", "REF_CAT", "REF_EPOCH",
                "URAT_ID", "URAT_SEP"]:
        cols.remove(col)
    for col in cols:
        gfas[col] = objects[col]
    # ADM populate the TARGETID column.
    gfas["TARGETID"] = targetid
    # ADM populate the BRICK_OBJID column.
    gfas["BRICK_OBJID"] = objects["OBJID"]
    # ADM REF_CAT and REF_EPOCH didn't exist before DR8.
    for refcol in ["REF_CAT", "REF_EPOCH"]:
        if refcol in objects.dtype.names:
            gfas[refcol] = objects[refcol]

    # ADM cut the GFAs by a hard limit on magnitude.
    ii = gfas['GAIA_PHOT_G_MEAN_MAG'] < maglim
    gfas = gfas[ii]

    # ADM remove any sources based on LSLGA (retain Tycho/T2 sources).
    # ADM the try/except/decode catches both bytes and unicode strings.
    try:
        ii = np.array([rc.decode()[0] == "L" for rc in gfas["REF_CAT"]],
                      dtype=bool)
    except AttributeError:
        ii = np.array([i[0] == "L" for rc in gfas["REF_CAT"]], dtype=bool)
    gfas = gfas[~ii]

    return gfas


def gaia_in_file(infile, maglim=18, mindec=-30., mingalb=10.,
                 nside=None, pixlist=None, addobjid=False, addparams=False):
    """Retrieve the Gaia objects from a HEALPixel-split Gaia file.

    Parameters
    ----------
    infile : :class:`str`
        File name of a single Gaia "healpix" file.
    maglim : :class:`float`, optional, defaults to 18
        Magnitude limit for GFAs in Gaia G-band.
    mindec : :class:`float`, optional, defaults to -30
        Minimum declination (o) to include for output Gaia objects.
    mingalb : :class:`float`, optional, defaults to 10
        Closest latitude to Galactic plane for output Gaia objects
        (e.g. send 10 to limit to areas beyond -10o <= b < 10o)"
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPix `nside` to use with `pixlist`.
    pixlist : :class:`list` or `int`, optional, defaults to `None`
        Only return sources in a set of (NESTED) HEALpixels at the
        supplied `nside`.
    addobjid : :class:`bool`, optional, defaults to ``False``
        If ``True``, include, in the output, a column "GAIA_OBJID"
        that is the integer number of each row read from file.
    addparams : :class:`bool`, optional, defaults to ``False``
        If ``True``, include some additional Gaia columns:
        "GAIA_ASTROMETRIC_EXCESS_NOISE", "GAIA_DUPLICATED_SOURCE"
        and "GAIA_ASTROMETRIC_PARAMS_SOLVED'.

    Returns
    -------
    :class:`~numpy.ndarray`
        Gaia objects in the passed Gaia file brighter than `maglim`,
        formatted according to `desitarget.gfa.gfadatamodel`.

    Notes
    -----
       - A "Gaia healpix file" here is as made by, e.g.
         :func:`~desitarget.gaiamatch.gaia_fits_to_healpix()`
    """
    # ADM read in the Gaia file and limit to the passed magnitude.
    objs = read_gaia_file(infile, addobjid=addobjid)
    ii = objs['GAIA_PHOT_G_MEAN_MAG'] < maglim
    objs = objs[ii]

    # ADM rename GAIA_RA/DEC to RA/DEC, as that's what's used for GFAs.
    for radec in ["RA", "DEC"]:
        objs.dtype.names = [radec if col == "GAIA_"+radec else col
                            for col in objs.dtype.names]

    # ADM initiate the GFA data model.
    dt = gfadatamodel.dtype.descr
    if addobjid:
        for tup in ('GAIA_BRICKID', '>i4'), ('GAIA_OBJID', '>i4'):
            dt.append(tup)
    if addparams:
        for tup in [('GAIA_DUPLICATED_SOURCE', '?'),
                    ('GAIA_ASTROMETRIC_PARAMS_SOLVED', '>i1')]:
            dt.append(tup)

    gfas = np.zeros(len(objs), dtype=dt)
    # ADM make sure all columns initially have "ridiculous" numbers
    gfas[...] = -99.
    for col in gfas.dtype.names:
        if isinstance(gfas[col][0].item(), (bytes, str)):
            gfas[col] = 'U'
        if isinstance(gfas[col][0].item(), int):
            gfas[col] = -1
    # ADM some default special cases. Default to REF_EPOCH of Gaia DR2,
    # ADM make RA/Dec very precise for Gaia measurements.
    # ADM MASKBITS should default to zero indicating no flags are set.
    gfas["REF_EPOCH"] = 2015.5
    gfas["RA_IVAR"], gfas["DEC_IVAR"] = 1e16, 1e16
    gfas["MASKBITS"] = 0

    # ADM populate the common columns in the Gaia/GFA data models.
    cols = set(gfas.dtype.names).intersection(set(objs.dtype.names))
    for col in cols:
        gfas[col] = objs[col]

    # ADM update the Gaia morphological type.
    gfas["TYPE"] = gaia_morph(gfas)

    # ADM populate the BRICKID columns.
    gfas["BRICKID"] = bricks.brickid(gfas["RA"], gfas["DEC"])

    # ADM limit by HEALPixel first as that's the fastest.
    if pixlist is not None:
        inhp = is_in_hp(gfas, nside, pixlist)
        gfas = gfas[inhp]
    # ADM limit by Dec first to speed transform to Galactic coordinates.
    decgood = is_in_box(gfas, [0., 360., mindec, 90.])
    gfas = gfas[decgood]
    # ADM now limit to requesed Galactic latitude range.
    if mingalb > 1e-9:
        bbad = is_in_gal_box(gfas, [0., 360., -mingalb, mingalb])
        gfas = gfas[~bbad]

    return gfas


def all_gaia_in_tiles(maglim=18, numproc=4, allsky=False,
                      tiles=None, mindec=-30, mingalb=10, nside=None,
                      pixlist=None, addobjid=False, addparams=False):
    """An array of all Gaia objects in the DESI tiling footprint

    Parameters
    ----------
    maglim : :class:`float`, optional, defaults to 18
        Magnitude limit for GFAs in Gaia G-band.
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    allsky : :class:`bool`,  defaults to ``False``
        If ``True``, assume that the DESI tiling footprint is the
        entire sky regardless of the value of `tiles`.
    tiles : :class:`~numpy.ndarray`, optional, defaults to ``None``
        Array of DESI tiles. If None, then load the entire footprint.
    mindec : :class:`float`, optional, defaults to -30
        Minimum declination (o) to include for output Gaia objects.
    mingalb : :class:`float`, optional, defaults to 10
        Closest latitude to Galactic plane for output Gaia objects
        (e.g. send 10 to limit to areas beyond -10o <= b < 10o).
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPix `nside` to use with `pixlist`.
    pixlist : :class:`list` or `int`, optional, defaults to `None`
        Only return sources in a set of (NESTED) HEALpixels at the
        supplied `nside`.
    addobjid : :class:`bool`, optional, defaults to ``False``
        If ``True``, include, in the output, a column "GAIA_OBJID"
        that is the integer number of each row read from each Gaia file.
    addparams : :class:`bool`, optional, defaults to ``False``
        If ``True``, include some additional Gaia columns:
        "GAIA_DUPLICATED_SOURCE" and "GAIA_ASTROMETRIC_PARAMS_SOLVED'.

    Returns
    -------
    :class:`~numpy.ndarray`
        Gaia objects within the passed geometric constraints brighter
        than `maglim`, formatted like `desitarget.gfa.gfadatamodel`.

    Notes
    -----
       - The environment variables $GAIA_DIR and $DESIMODEL must be set.
    """
    # ADM to guard against no files being found.
    if pixlist is None:
        dummyfile = find_gaia_files_hp(_get_gaia_nside(), [0],
                                       neighbors=False)[0]
    else:
        # ADM this is critical for, e.g., unit tests for which the
        # ADM Gaia "00000" pixel file might not exist.
        dummyfile = find_gaia_files_hp(nside, pixlist[0],
                                       neighbors=False)[0]
    dummygfas = np.array([], gaia_in_file(dummyfile, addparams=addparams).dtype)

    # ADM grab paths to Gaia files in the sky or the DESI footprint.
    if allsky:
        infilesbox = find_gaia_files_box([0, 360, mindec, 90])
        infilesgalb = find_gaia_files_beyond_gal_b(mingalb)
        infiles = list(set(infilesbox).intersection(set(infilesgalb)))
        if pixlist is not None:
            infileshp = find_gaia_files_hp(nside, pixlist, neighbors=False)
            infiles = list(set(infiles).intersection(set(infileshp)))
    else:
        infiles = find_gaia_files_tiles(tiles=tiles, neighbors=False)
    nfiles = len(infiles)

    # ADM the critical function to run on every file.
    def _get_gaia_gfas(fn):
        '''wrapper on gaia_in_file() given a file name'''
        return gaia_in_file(fn, maglim=maglim, mindec=mindec, mingalb=mingalb,
                            nside=nside, pixlist=pixlist,
                            addobjid=addobjid, addparams=addparams)

    # ADM this is just to count sweeps files in _update_status.
    nfile = np.zeros((), dtype='i8')
    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if nfile % 100 == 0 and nfile > 0:
            elapsed = (time()-t0)/60.
            rate = nfile/elapsed/60.
            log.info('{}/{} files; {:.1f} files/sec...t = {:.1f} mins'
                     .format(nfile, nfiles, rate, elapsed))
        nfile[...] += 1    # this is an in-place modification.
        return result

    # - Parallel process Gaia files.
    if numproc > 1 and nfiles > 0:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            gfas = pool.map(_get_gaia_gfas, infiles, reduce=_update_status)
    else:
        gfas = list()
        for file in infiles:
            gfas.append(_update_status(_get_gaia_gfas(file)))

    if len(gfas) > 0:
        gfas = np.concatenate(gfas)
    else:
        # ADM if nothing was found, return an empty np array.
        gfas = dummygfas

    log.info('Retrieved {} Gaia objects...t = {:.1f} mins'
             .format(len(gfas), (time()-t0)/60.))

    return gfas


def add_urat_pms(objs, numproc=4):
    """Add proper motions from URAT to a set of objects.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        Array of objects to update. Must include the columns "PMRA",
        "PMDEC", "REF_ID" (unique per object) "URAT_ID" and "URAT_SEP".
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.

    Returns
    -------
    :class:`~numpy.ndarray`
        The input array with the "PMRA", PMDEC", "URAT_ID" and "URAT_SEP"
        columns updated to include URAT information.

    Notes
    -----
       - Order is retained using "REF_ID": The input and output
         arrays should have the same order.
    """
    # ADM check REF_ID is indeed unique for each object.
    assert len(objs["REF_ID"]) == len(np.unique(objs["REF_ID"]))

    # ADM record the original REF_IDs so we can match back to them.
    origids = objs["REF_ID"]

    # ADM loosely group the input objects on the sky. NSIDE=16 seems
    # ADM to nicely balance sample sizes for matching, with the code
    # ADM being quicker for clumped objects because of file I/O.
    theta, phi = np.radians(90-objs["DEC"]), np.radians(objs["RA"])
    pixels = hp.ang2pix(16, theta, phi, nest=True)

    # ADM reorder objects (and pixels themselves) based on pixel number.
    ii = np.argsort(pixels)
    objs, pixels = objs[ii], pixels[ii]

    # ADM create pixel-split sub-lists of the objects.
    # ADM here, np.diff marks the transition to the next pixel number.
    splitobjs = np.split(objs, np.where(np.diff(pixels))[0]+1)
    nallpix = len(splitobjs)

    # ADM function to run on each of the HEALPix-split input objs.
    def _get_urat_matches(splitobj):
        '''wrapper on match_to_urat() for rec array (matchrad=0.5")'''
        # ADM also return the REF_ID to track the objects.
        return [match_to_urat(splitobj, matchrad=0.5), splitobj["REF_ID"]]

    # ADM this is just to count pixels in _update_status.
    npix = np.zeros((), dtype='i8')
    t0 = time()

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if npix % 200 == 0 and npix > 0:
            elapsed = (time()-t0)/60.
            rate = npix/elapsed/60.
            log.info('{}/{} pixels; {:.1f} pix/sec...t = {:.1f} mins'
                     .format(npix, nallpix, rate, elapsed))
        npix[...] += 1    # this is an in-place modification.
        return result

    # - Parallel process pixels.
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            urats = pool.map(_get_urat_matches, splitobjs, reduce=_update_status)
    else:
        urats = []
        for splitobj in splitobjs:
            urats.append(_update_status(_get_urat_matches(splitobj)))

    # ADM remember to grab the REFIDs as well as the URAT matches...and
    # ADM to catch the corner case where objects occupy only one pixel.
    if len(urats) == 1:
        refids = urats[0][1]
        urats = urats[0][0]
    else:
        refids = np.concatenate(np.array(urats, dtype=object)[:, 1])
        urats = np.concatenate(np.array(urats, dtype=object)[:, 0])

    # ADM sort the output to match the input, on REF_ID.
    ii = np.zeros_like(refids)
    ii[np.argsort(origids)] = np.argsort(refids)
    assert np.all(refids[ii] == origids)

    return urats[ii]


def select_gfas(infiles, maglim=18, numproc=4, nside=None,
                pixlist=None, bundlefiles=None, extra=None,
                mindec=-30, mingalb=10, addurat=True):
    """Create a set of GFA locations using Gaia and matching to sweeps.

    Parameters
    ----------
    infiles : :class:`list` or `str`
        A list of input filenames (sweep files) OR a single filename.
    maglim : :class:`float`, optional, defaults to 18
        Magnitude limit for GFAs in Gaia G-band.
    numproc : :class:`int`, optional, defaults to 4
        The number of parallel processes to use.
    nside : :class:`int`, optional, defaults to `None`
        (NESTED) HEALPix `nside` to use with `pixlist` and `bundlefiles`.
    pixlist : :class:`list` or `int`, optional, defaults to `None`
        Only return targets in a set of (NESTED) HEALpixels at the
        supplied `nside`. Useful for parallelizing.
    bundlefiles : :class:`int`, defaults to `None`
        If not `None`, then, instead of selecting gfas, print the slurm
        script to run in pixels at `nside`. Is an integer rather than
        a boolean for historical reasons.
    extra : :class:`str`, optional
        Extra command line flags to be passed to the executable lines in
        the output slurm script. Used in conjunction with `bundlefiles`.
    mindec : :class:`float`, optional, defaults to -30
        Minimum declination (o) for output sources that do NOT match
        an object in the passed `infiles`.
    mingalb : :class:`float`, optional, defaults to 10
        Closest latitude to Galactic plane for output sources that
        do NOT match an object in the passed `infiles` (e.g. send
        10 to limit to regions beyond -10o <= b < 10o)".
    addurat : :class:`bool`, optional, defaults to ``True``
        If ``True`` then substitute proper motions from the URAT
        catalog where Gaia is missing proper motions. Requires that
        the :envvar:`URAT_DIR` is set and points to data downloaded and
        formatted by, e.g., :func:`~desitarget.uratmatch.make_urat_files`.

    Returns
    -------
    :class:`~numpy.ndarray`
        GFA objects from Gaia with the passed geometric constraints
        limited to the passed maglim and matched to the passed input
        files, formatted according to `desitarget.gfa.gfadatamodel`.

    Notes
    -----
        - If numproc==1, use the serial code instead of parallel code.
        - If numproc > 4, then numproc=4 is enforced for (just those)
          parts of the code that are I/O limited.
    """
    # ADM the code can have memory issues for nside=2 with large numproc.
    if nside is not None and nside < 4 and numproc > 8:
        msg = 'Memory may be an issue near Plane for nside < 4 and numproc > 8'
        log.warning(msg)

    # ADM force to no more than numproc=4 for I/O limited processes.
    numproc4 = numproc
    if numproc4 > 4:
        log.info('Forcing numproc to 4 for I/O limited parts of code')
        numproc4 = 4

    # ADM convert a single file, if passed to a list of files.
    if isinstance(infiles, str):
        infiles = [infiles, ]

    # ADM check that files exist before proceeding.
    for filename in infiles:
        if not os.path.exists(filename):
            msg = "{} doesn't exist".format(filename)
            log.critical(msg)
            raise ValueError(msg)

    # ADM if the pixlist option was sent, we'll need to
    # ADM know which HEALPixels touch each file.
    if pixlist is not None:
        filesperpixel, _, _ = sweep_files_touch_hp(
            nside, pixlist, infiles)

    # ADM if the bundlefiles option was sent, call the packing code.
    if bundlefiles is not None:
        # ADM were files from one or two input directories passed?
        surveydirs = list(set([os.path.dirname(fn) for fn in infiles]))
        bundle_bricks([0], bundlefiles, nside, gather=False,
                      prefix='gfas', surveydirs=surveydirs, extra=extra)
        return

    # ADM restrict to input files in a set of HEALPixels, if requested.
    if pixlist is not None:
        infiles = list(set(np.hstack([filesperpixel[pix] for pix in pixlist])))
        if len(infiles) == 0:
            log.info('ZERO sweep files in passed pixel list!!!')
        log.info("Processing files in (nside={}, pixel numbers={}) HEALPixels"
                 .format(nside, pixlist))
    nfiles = len(infiles)

    # ADM a little more information if we're slurming across nodes.
    if os.getenv('SLURMD_NODENAME') is not None:
        log.info('Running on Node {}'.format(os.getenv('SLURMD_NODENAME')))

    # ADM the critical function to run on every file.
    def _get_gfas(fn):
        '''wrapper on gaia_gfas_from_sweep() given a file name'''
        return gaia_gfas_from_sweep(fn, maglim=maglim)

    # ADM this is just to count sweeps files in _update_status.
    t0 = time()
    nfile = np.zeros((), dtype='i8')

    def _update_status(result):
        """wrapper function for the critical reduction operation,
        that occurs on the main parallel process"""
        if nfile % 20 == 0 and nfile > 0:
            elapsed = (time()-t0)/60.
            rate = nfile/elapsed/60.
            log.info('{}/{} files; {:.1f} files/sec...t = {:.1f} mins'
                     .format(nfile, nfiles, rate, elapsed))
        nfile[...] += 1    # this is an in-place modification.
        return result

    # - Parallel process input files.
    if len(infiles) > 0:
        if numproc4 > 1:
            pool = sharedmem.MapReduce(np=numproc4)
            with pool:
                gfas = pool.map(_get_gfas, infiles, reduce=_update_status)
        else:
            gfas = list()
            for file in infiles:
                gfas.append(_update_status(_get_gfas(file)))
        gfas = np.concatenate(gfas)
        # ADM resolve any duplicates between imaging data releases.
        gfas = resolve(gfas)

    # ADM retrieve Gaia objects in the DESI footprint or passed tiles.
    log.info('Retrieving additional Gaia objects...t = {:.1f} mins'
             .format((time()-t0)/60))
    gaia = all_gaia_in_tiles(maglim=maglim, numproc=numproc4, allsky=True,
                             mindec=mindec, mingalb=mingalb,
                             nside=nside, pixlist=pixlist)

    # ADM remove any duplicates. Order is important here, as np.unique
    # ADM keeps the first occurence, and we want to retain sweeps
    # ADM information as much as possible.
    if len(infiles) > 0:
        gfas = np.concatenate([gfas, gaia])
        _, ind = np.unique(gfas["REF_ID"], return_index=True)
        gfas = gfas[ind]
    else:
        gfas = gaia

    # ADM for zero/NaN proper motion objects, add URAT proper motions.
    if addurat:
        ii = ((np.isnan(gfas["PMRA"]) | (gfas["PMRA"] == 0)) &
              (np.isnan(gfas["PMDEC"]) | (gfas["PMDEC"] == 0)))
        log.info('Adding URAT for {} objects with no PMs...t = {:.1f} mins'
                 .format(np.sum(ii), (time()-t0)/60))
        urat = add_urat_pms(gfas[ii], numproc=numproc)
        log.info('Found an additional {} URAT objects...t = {:.1f} mins'
                 .format(np.sum(urat["URAT_ID"] != -1), (time()-t0)/60))
        for col in "PMRA", "PMDEC", "URAT_ID", "URAT_SEP":
            gfas[col][ii] = urat[col]

    # ADM restrict to only GFAs in a set of HEALPixels, if requested.
    if pixlist is not None:
        ii = is_in_hp(gfas, nside, pixlist)
        gfas = gfas[ii]

    return gfas
