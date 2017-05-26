def target_density(targets):
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

    pixels = hp.ang2pix(nside, np.radians(90 - targets['DEC']), 
                        np.radians(targets['RA']), nest=True)
    counts = np.bincount(pixels, weights=None, minlength=npix)
    dens = counts[np.flatnonzero(counts)] / bin_area
            
    return dens

def qa_targets_truth(output_dir, verbose=True, clobber=False):
    """Generate QA plots from the joined targets and truth catalogs.

    time select_mock_targets --output_dir debug --qa

    """
    import fitsio
    from desitarget import desi_mask, bgs_mask, mws_mask, contam_mask

    if verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

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

    # Do some sanity checking.
    nobj, nsky, ndark, nbright = len(targets), len(sky), len(stddark), len(stdbright)
    if nobj != len(truth):
        log.fatal('Mismatch in the number of objects in targets.fits (N={}) and truth.fits (N={})!'.format(nobj, len(truth))
        raise ValueError



        
    


    import pdb ; pdb.set_trace()
