from __future__ import absolute_import, division, print_function
import warnings
from time import time
import os.path
import numpy as np

from desitarget import io
from desitarget.internal import sharedmem
import desitarget.targets
from desitarget import desi_mask, bgs_mask, mws_mask

"""
Target Selection for DECALS catalogue data

https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (e.g. LRG, ELG or QSO).
"""

def apply_cuts(objects):
    """Perform target selection on objects, returning target mask arrays

    Args:
        objects: numpy structured array with UPPERCASE columns needed for
            target selection, OR a string tractor/sweep filename
            
    Returns:
        (desi_target, bgs_target, mws_target) where each element is
        an ndarray of target selection bitmask flags for each object
        
    See desitarget.targetmask for the definition of each bit
    """
    #- Check if objects is a filename instead of the actual data
    if isinstance(objects, (str, unicode)):
        objects = io.read_tractor(objects)
    
    #- construct milky way extinction corrected fluxes
    dered_decam_flux = objects['DECAM_FLUX'] / objects['DECAM_MW_TRANSMISSION']
    gflux = dered_decam_flux[:, 1]
    rflux = dered_decam_flux[:, 2]
    zflux = dered_decam_flux[:, 4]

    dered_wise_flux = objects['WISE_FLUX'] / objects['WISE_MW_TRANSMISSION']
    w1flux = dered_wise_flux[:, 0]
    wflux = 0.75* w1flux + 0.25*dered_wise_flux[:, 1]

    #- DR1 has targets off the edge of the brick; trim to just this brick
    if 'BRICK_PRIMARY' in objects.dtype.names:
        primary = objects['BRICK_PRIMARY']
    else:
        primary = np.ones(len(objects), dtype=bool)
        
    #----- LRG
    lrg = primary.copy()
    lrg &= rflux > 10**((22.5-23.0)/2.5)
    lrg &= zflux > 10**((22.5-22.56)/2.5)
    lrg &= w1flux > 10**((22.5-19.35)/2.5)
    lrg &= zflux > rflux * 10**(1.6/2.5)
    #- clip to avoid warnings from negative numbers raised to fractional powers
    lrg &= w1flux * rflux.clip(0)**(1.33-1) > zflux.clip(0)**1.33 * 10**(-0.33/2.5)

    #----- ELG
    elg = primary.copy()
    elg &= rflux > 10**((22.5-23.4)/2.5)
    elg &= zflux > rflux * 10**(0.3/2.5)
    elg &= zflux < rflux * 10**(1.5/2.5)
    elg &= rflux**2 < gflux * zflux * 10**(-0.2/2.5)
    elg &= zflux < gflux * 10**(1.2/2.5)

    #----- Quasars
    psflike = ((objects['TYPE'] == 'PSF') | (objects['TYPE'] == 'PSF '))    
    qso = primary.copy()
    qso &= psflike
    qso &= rflux > 10**((22.5-23.0)/2.5)
    qso &= rflux < gflux * 10**(1.0/2.5)
    qso &= zflux > rflux * 10**(-0.3/2.5)
    qso &= zflux < rflux * 10**(1.1/2.5)
    #- clip to avoid warnings from negative numbers raised to fractional powers
    qso &= wflux * gflux.clip(0)**1.2 > rflux.clip(0)**(1+1.2) * 10**(-0.4/2.5)
    ### qso &= wflux * gflux**1.2 > rflux**(1+1.2) * 10**(2/2.5)

    #------ Bright Galaxy Survey
    #- 'PSF' for astropy.io.fits; 'PSF ' for fitsio (sigh)
    bgs = primary.copy()
    bgs &= ~psflike
    bgs &= rflux > 10**((22.5-19.35)/2.5)

    #----- Standard stars
    fstd = primary.copy()
    fstd &= psflike
    fracflux = objects['DECAM_FRACFLUX'].T        
    signal2noise = objects['DECAM_FLUX'] * np.sqrt(objects['DECAM_FLUX_IVAR'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for j in (1,2,4):  #- g, r, z
            fstd &= fracflux[j] < 0.04
            fstd &= signal2noise[:, j] > 10

    #- observed flux; no Milky Way extinction
    obs_rflux = objects['DECAM_FLUX'][:, 2]
    fstd &= obs_rflux < 10**((22.5-16.0)/2.5)
    fstd &= obs_rflux > 10**((22.5-19.0)/2.5)
    #- colors near BD+17; ignore warnings about flux<=0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grcolor = 2.5 * np.log10(rflux / gflux)
        rzcolor = 2.5 * np.log10(zflux / rflux)
        fstd &= (grcolor - 0.32)**2 + (rzcolor - 0.13)**2 < 0.06**2

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
    
    bgs_target = bgs * bgs_mask.BGS_BRIGHT
    bgs_target |= bgs * bgs_mask.BGS_BRIGHT_SOUTH

    #- nothing for MWS yet; will be GAIA-based
    mws_target = np.zeros_like(bgs_target)

    #- Are any BGS or MWS bit set?  Tell desi_target too.
    desi_target |= (bgs_target != 0) * desi_mask.BGS_ANY
    desi_target |= (mws_target != 0) * desi_mask.MWS_ANY

    return desi_target, bgs_target, mws_target

def select_targets(infiles, numproc=4, verbose=False):
    """
    Process input files in parallel to select targets
    
    Args:
        infiles: list of input filenames (tractor or sweep files),
            OR a single filename
        
    Optional:
        numproc: number of parallel processes to use
        verbose: if True, print progress messages
        
    Returns:
        targets numpy structured array: the subset of input targets which
            pass the cuts, including extra columns for DESI_TARGET,
            BGS_TARGET, and MWS_TARGET target selection bitmasks. 
            
    Notes:
        if numproc==1, use serial code instead of parallel
    """
    #- Convert single file to list of files
    if isinstance(infiles, (str, unicode)):
        infiles = [infiles,]

    #- Sanity check that files exist before going further
    for filename in infiles:
        if not os.path.exists(filename):
            raise ValueError("{} doesn't exist".format(filename))
    
    #- function to run on every brick/sweep file
    def _select_targets_file(filename):
        '''Returns targets in filename that pass the cuts'''
        objects = io.read_tractor(filename)
        desi_target, bgs_target, mws_target = apply_cuts(objects)
        
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

# def calc_numobs(targets, targetflags):
#     """
#     Return array of number of observations needed for each target.
#     
#     Args:
#         targets: numpy structured array with tractor inputs
#         targetflags: array of target selection bit flags 
#     
#     Returns:
#         array of integers of number of observations needed
#     """
#     #- Default is one observation
#     nobs = np.ones(len(targets), dtype='i4')
#     
#     #- If it wasn't selected by any target class, it gets 0 observations
#     #- Normally these would have already been removed, but just in case...
#     nobs[targetflags == 0] = 0
#     
#     #- LRGs get 1, 2, or 3 observations depending upon magnitude
#     zflux = targets['DECAM_FLUX'][:,4] / targets['DECAM_MW_TRANSMISSION'][:,4]    
#     islrg = (targetflags & targetmask.LRG) != 0
#     lrg2 = islrg & (zflux < 10**((22.5-20.36)/2.5))
#     lrg3 = islrg & (zflux < 10**((22.5-20.56)/2.5))
#     nobs[lrg2] = 2
#     nobs[lrg3] = 3
#     
#     #- TBD: flag QSOs for 4-5 obs ahead of time, or only after confirming
#     #- that they are redshift>2.15 (i.e. good for Lyman-alpha)?
#     
#     return nobs
