import numpy as np

from desitarget.cuts import unextinct_fluxes


def isLRG_2016v3(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None): 

    """
    Daniel Eisenstein -- Jan 9, 2017
    This is version 3 of the Eisenstein/Dawson Summer 2016 work.
    But anymask has been changed to allmask, which probably means that
    the flux cuts need to be re-tuned.  That is, mlrg2<19.6 may need
    to change to 19.5 or 19.4.

    # Inputs: decam_flux, decam_flux_ivar, decam_allmask, decam_mw_transmission
    # wise_flux, wise_flux_ivar, wise_mw_transmission
    # Using g, r, z, and W1 information.

    # Applying the reddening
    # Also clip r, z, and W1 at 0 to avoid warnings from negative numbers raised to fractional powers 
    # We do not require gflux>0!  Objects can be very red.
    """

    gflux = decam_flux[1]/decam_mw_transmission[1]
    rflux = decam_flux[2].clip(0)/decam_mw_transmission[2]
    zflux = decam_flux[4].clip(0)/decam_mw_transmission[4]
    W1flux =  wise_flux[0].clip(0)/wise_mw_transmission[0]

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
        lrg = primary.copy()

    # Some basic quality in r, z, and W1
    # r, z, and W will all be >0 from this.
    lrg &= (decam_flux_ivar[2]>0) && (decam_allmask[2]==0) && rflux>0
    lrg &= (decam_flux_ivar[4]>0) && (decam_allmask[4]==0) && zflux>0
    lrg &= (wise_flux[0]*sqrt(wise_flux_ivar[0])>4)

    # Basic flux and color cuts
    lrg &= (zflux > 10**(0.4*(22.5-20.4))) # z<20.4
    lrg &= (zflux < 10**(0.4*(22.5-18))) # z>18
    lrg &= (zflux < 10**(0.4*2.5)*rflux) # r-z<2.5
    lrg &= (zflux > 10**(0.4*0.8)*rflux) # r-z>0.8

    # This is the star-galaxy separation cut
    # Wlrg = (z-W)-(r-z)/3 + 0.3 >0 , which is equiv to r+3*W < 4*z+0.9
    lrg &= (rflux*W1flux**3 > (zflux**4)*10**(-0.4*0.9))

    # Now for the work-horse sliding flux-color cut:
    # mlrg2 = z-2*(r-z-1.2) < 19.6 -> 3*z < 19.6-2.4-2*r
    lrg &= (zflux**3 > 10**(0.4*(22.5+2.4-19.6))*rflux**2)

    # Another guard against bright & red outliers
    # mlrg2 = z-2*(r-z-1.2) > 17.4 -> 3*z > 17.4-2.4-2*r
    lrg &= (zflux**3 < 10**(0.4*(22.5+2.4-17.4))*rflux**2)

    # Finally, a cut to exclude the z<0.4 objects while retaining the elbow at z=0.4-0.5. 

    # r-z>1.2 || (good_data_in_g && g-r>1.7)
    # Note that we do not require gflux>0
    lrg &= ( (zflux > 10**(0.4*1.2)*rflux) ||
             ( (decam_flux_ivar[1]>0) && (decam_allmask[1]==0) &&
               (rflux>10**(0.4*1.7)*gflux) ) ) 


def apply_sandbox_cuts(objects,qso_selection='randomforest'):
    """Perform target selection on objects, returning target mask arrays

    Args:
        objects: numpy structured array with UPPERCASE columns needed for
            target selection, OR a string tractor/sweep filename

    Options:
        qso_selection : algorithm to use for QSO selection; valid options
            are 'colorcuts' and 'randomforest'
            
    Returns:
        (desi_target, bgs_target, mws_target) where each element is
        an ndarray of target selection bitmask flags for each object
        
    Bugs:
        If objects is a astropy Table with lowercase column names, this
        converts them to UPPERCASE in-place, thus modifying the input table.
        To avoid this, pass in objects.copy() instead. 

    See desitarget.targetmask for the definition of each bit
    """
    #- Check if objects is a filename instead of the actual data
    if isinstance(objects, str):
        from desitarget import io
        objects = io.read_tractor(objects)
    
    #- ensure uppercase column names if astropy Table
    if isinstance(objects, (Table, Row)):
        for col in list(objects.columns.values()):
            if not col.name.isupper():
                col.name = col.name.upper()

    #- undo Milky Way extinction
    flux = unextinct_fluxes(objects)
    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']
    
    wise_snr = objects['WISE_FLUX'] * np.sqrt(objects['WISE_FLUX_IVAR'])


    # Delta chi2 between PSF and SIMP morphologies; note the sign....
    dchisq = objects['DCHISQ'] 
    deltaChi2 = dchisq[...,0] - dchisq[...,1]

    #- DR1 has targets off the edge of the brick; trim to just this brick
    try:
        primary = objects['BRICK_PRIMARY']
    except (KeyError, ValueError):
        if _is_row(objects):
            primary = True
        else:
            primary = np.ones_like(objects, dtype=bool)
        
    lrg = isLRG(primary=primary, zflux=zflux, rflux=rflux, w1flux=w1flux)

    elg = isELG(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux)

    bgs_bright = isBGS_bright(primary=primary, rflux=rflux, objtype=objtype)
    bgs_faint  = isBGS_faint(primary=primary, rflux=rflux, objtype=objtype)
    
    if qso_selection=='colorcuts' :
        qso = isQSO_cuts(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2, objtype=objtype,
                wise_snr=wise_snr)
    elif qso_selection == 'randomforest':
        qso = isQSO_randomforest(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2, objtype=objtype)
    else:
        raise ValueError('Unknown qso_selection {}; valid options are {}'.format(qso_selection, qso_selection_options))

    #----- Standard stars
    fstd = isFSTD_colors(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux)

    fstd &= psflike(objtype)
    fracflux = objects['DECAM_FRACFLUX'].T        
    signal2noise = objects['DECAM_FLUX'] * np.sqrt(objects['DECAM_FLUX_IVAR'])
    with warnings.catch_warnings():
        # FIXME: what warnings are we ignoring?
        warnings.simplefilter('ignore')
        for j in (1,2,4):  #- g, r, z
            fstd &= fracflux[j] < 0.04
            fstd &= signal2noise[..., j] > 10

    #- observed flux; no Milky Way extinction
    obs_rflux = objects['DECAM_FLUX'][..., 2]
    fstd &= obs_rflux < 10**((22.5-16.0)/2.5)
    fstd &= obs_rflux > 10**((22.5-19.0)/2.5)

    #-----
    #- construct the targetflag bits
    #- Currently our only cuts are DECam based (i.e. South)
    desi_target  = lrg * desi_mask.LRG_SOUTH
    desi_target |= elg * desi_mask.ELG_SOUTH
    desi_target |= qso * desi_mask.QSO_SOUTH

    desi_target |= lrg * desi_mask.LRG
    desi_target |= elg * desi_mask.ELG
    desi_target |= qso * desi_mask.QSO

    desi_target |= fstd * desi_mask.STD_FSTAR
    
    bgs_target = bgs_bright * bgs_mask.BGS_BRIGHT
    bgs_target |= bgs_bright * bgs_mask.BGS_BRIGHT_SOUTH
    bgs_target |= bgs_faint * bgs_mask.BGS_FAINT
    bgs_target |= bgs_faint * bgs_mask.BGS_FAINT_SOUTH

    #- nothing for MWS yet; will be GAIA-based
    if isinstance(bgs_target, numbers.Integral):
        mws_target = 0
    else:
        mws_target = np.zeros_like(bgs_target)

    #- Are any BGS or MWS bit set?  Tell desi_target too.
    desi_target |= (bgs_target != 0) * desi_mask.BGS_ANY
    desi_target |= (mws_target != 0) * desi_mask.MWS_ANY

    return desi_target, bgs_target, mws_target

def select_sandbox_targets(infiles, numproc=4, verbose=False):
    """
    Process input files in parallel to select targets.
    

    Args:
        infiles: list of input filenames (tractor or sweep files),
            OR a single filename
        
    Optional:
        numproc: number of parallel processes to use
        verbose: if True, print progress messages
        qso_selection : algorithm to use for QSO selection; valid options
            are 'colorcuts' and 'randomforest'
        
    Returns:
        targets numpy structured array: the subset of input targets which
            pass the cuts, including extra columns for DESI_TARGET,
            BGS_TARGET, and MWS_TARGET target selection bitmasks. 
            
    Notes:
        if numproc==1, use serial code instead of parallel
    """
    #- Convert single file to list of files
    if isinstance(infiles,str):
        infiles = [infiles,]

    #- Sanity check that files exist before going further
    for filename in infiles:
        if not os.path.exists(filename):
            raise ValueError("{} doesn't exist".format(filename))

    #- function to run on every brick/sweep file
    def _select_targets_file(filename):
        '''Returns targets in filename that pass the cuts'''
        from desitarget import io
        objects = io.read_tractor(filename)
        desi_target, bgs_target, mws_target = apply_sandbox_cuts(objects)

        #- desi_target includes BGS_ANY and MWS_ANY, so we can filter just
        #- on desi_target != 0
        keep = (desi_target != 0)
        objects = objects[keep]
        desi_target = desi_target[keep]
        bgs_target = bgs_target[keep]
        mws_target = mws_target[keep]

        #- Add *_target mask columns
        targets = desitarget.targets.finalize(
            objects, desi_target, bgs_target, mws_target)

        return io.fix_tractor_dr1_dtype(targets)

    # Counter for number of bricks processed;
    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    t0 = time()
    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if verbose and nbrick%50 == 0 and nbrick>0:
            rate = nbrick / (time() - t0)
            print('{} files; {:.1f} files/sec'.format(nbrick, rate))

        nbrick[...] += 1    # this is an in-place modification
        return result

    #- Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            targets = pool.map(_select_targets_file, infiles, reduce=_update_status)
    else:
        targets = list()
        for x in infiles:
            targets.append(_update_status(_select_targets_file(x)))
        
    targets = np.concatenate(targets)
    
    return targets
