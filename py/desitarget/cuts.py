import warnings
from time import time
import os.path
import numpy as np
from astropy.table import Table, Row
from pkg_resources import resource_filename

from desitarget.internal import sharedmem
import desitarget.targets
from desitarget import desi_mask, bgs_mask, mws_mask

"""
Target Selection for DECALS catalogue data

https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (e.g. LRG, ELG or QSO).
"""

def isLRG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None):
    """Target Definition of LRG. Returning a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only the object is an LRG
            target.

    """
    #----- Luminous Red Galaxies
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')

    lrg = primary.copy()
    lrg &= zflux > 10**((22.5-20.46)/2.5)  # z<20.46
    lrg &= zflux > rflux * 10**(1.5/2.5)   # (r-z)>1.5
    lrg &= w1flux > 0                      # W1flux>0
    #- clip to avoid warnings from negative numbers raised to fractional powers
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    lrg &= w1flux * rflux**(1.8-1.0) > zflux**1.8 * 10**(-1.0/2.5)

    return lrg

def isELG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None):
    """Target Definition of ELG. Returning a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only the object is an ELG
            target.

    """
    #----- Emission Line Galaxies
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    elg = primary.copy()
    elg &= rflux > 10**((22.5-23.4)/2.5)                       # r<23.4
    elg &= zflux > rflux * 10**(0.3/2.5)                       # (r-z)>0.3
    elg &= zflux < rflux * 10**(1.6/2.5)                       # (r-z)<1.6

    # Clip to avoid warnings from negative numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    elg &= rflux**2.15 < gflux * zflux**1.15 * 10**(-0.15/2.5) # (g-r)<1.15(r-z)-0.15
    elg &= zflux**1.2 < gflux * rflux**0.2 * 10**(1.6/2.5)     # (g-r)<1.6-1.2(r-z)

    return elg

def isFSTD_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None):
    """Select FSTD targets just based on color cuts. Returns a boolean array. 

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : boolean array, True if the object has colors like an FSTD

    Notes:
        The full FSTD target selection also includes PSF-like and fracflux
        cuts; this function is only cuts on the colors.

    """
    #----- F-type standard stars
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    fstd = primary.copy()

    #- colors near BD+17; ignore warnings about flux<=0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grcolor = 2.5 * np.log10(rflux / gflux)
        rzcolor = 2.5 * np.log10(zflux / rflux)
        fstd &= (grcolor - 0.32)**2 + (rzcolor - 0.13)**2 < 0.06**2

    return fstd

def isMWSSTAR_colors(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None):
    """Select a reasonable range of g-r colors for MWS targets. Returns a boolean array. 

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : boolean array, True if the object has colors like an old stellar population,
        which is what we expect for the main MWS sample

    Notes:
        The full MWS target selection also includes PSF-like and fracflux
        cuts and will include Gaia information; this function is only to enforce
        a reasonable range of color/TEFF when simulating data. 

    """
    #----- Old stars, g-r > 0
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    mwsstar = primary.copy()

    #- colors g-r > 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grcolor = 2.5 * np.log10(rflux / gflux)
        mwsstar &= (grcolor > 0.0)

    return mwsstar

def psflike(psftype):
    """ If the object is PSF """
    #- 'PSF' for astropy.io.fits; 'PSF ' for fitsio (sigh)
    #- this could be fixed in the IO routine too.
    psftype = np.asarray(psftype)
    #ADM in Python3 these string literals become byte-like
    #ADM still ultimately better to fix in IO, I'd think
    #psflike = ((psftype == 'PSF') | (psftype == 'PSF '))
    psflike = ((psftype == 'PSF') | (psftype == 'PSF ') |(psftype == b'PSF') | (psftype == b'PSF '))
    return psflike

def isBGS(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, objtype=None, primary=None):
    """Target Definition of BGS. Returning a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, w1, and w2 bands.
        objtype: array_like or None
            If given, The TYPE column of the catalogue.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only the object is a BGS
            target.

    """
    #------ Bright Galaxy Survey
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()
    bgs &= rflux > 10**((22.5-19.5)/2.5)
    if objtype is not None:
        bgs &= ~psflike(objtype)
    return bgs

def isQSO(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, objtype=None,
          wise_snr=None, deltaChi2=None, primary=None):
    """Target Definition of QSO. Returning a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        objtype: array_like or None
            If given, the TYPE column of the Tractor catalogue.
        deltaChi2: array_like or None
            If given, chi2 difference between PSF and SIMP models,  dchisq_PSF - dchisq_SIMP
        wise_snr: array_like or None
            If given, the S/N in the W1 and W2 bands.
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only the object is a QSO
            target.

    """
    #----- Quasars
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')

    # Create some composite fluxes.
    wflux = 0.75* w1flux + 0.25*w2flux
    grzflux = (gflux + 0.8*rflux + 0.5*zflux) / 2.3

    qso = primary.copy()
    qso &= rflux > 10**((22.5-22.7)/2.5)    # r<22.7
    qso &= grzflux < 10**((22.5-17)/2.5)    # grz>17
    qso &= rflux < gflux * 10**(1.3/2.5)    # (g-r)<1.3
    qso &= zflux > rflux * 10**(-0.3/2.5)   # (r-z)>-0.3
    qso &= zflux < rflux * 10**(1.1/2.5)    # (r-z)<1.1
    qso &= w2flux > w1flux * 10**(-0.4/2.5) # (W1-W2)>-0.4
    qso &= wflux * gflux > zflux * grzflux * 10**(-1.0/2.5) # (grz-W)>(g-z)-1.0

    # Harder cut on stellar contamination
    mainseq = rflux > gflux * 10**(0.20/2.5)

    # Clip to avoid warnings from negative numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    mainseq &= rflux**(1+1.5) > gflux * zflux**1.5 * 10**((-0.100+0.175)/2.5)
    mainseq &= rflux**(1+1.5) < gflux * zflux**1.5 * 10**((+0.100+0.175)/2.5)
    mainseq &= w2flux < w1flux * 10**(0.3/2.5)
    qso &= ~mainseq

    if wise_snr is not None:
        qso &= wise_snr[..., 0] > 4
        qso &= wise_snr[..., 1] > 2

    if objtype is not None:
        qso &= psflike(objtype)

    if deltaChi2 is not None:
        qso &= deltaChi2>40.

    return qso


def isQSO_BDT(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, objtype=None,
         deltaChi2=None, primary=None):
    """Target Definition of QSO using a BDT. Returning a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        objtype: array_like or None
            If given, the TYPE column of the Tractor catalogue.
        deltaChi2: array_like or None
             If given, difference of chi2 bteween PSF and SIMP morphology
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only the object is a QSO
            target.

    """
    #----- Quasars
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')

    # build variables for BDT    
    nfeatures=11 # number of variables in BDT
#    nbEntries=len(rflux)
    nbEntries=rflux.size
#    print 'length =',len(rflux),' v2 =',rflux.size
    colors, r, DECaLSOK = getColors(nbEntries,nfeatures,gflux,rflux,zflux,w1flux,w2flux)

    #Preselection to speed up the process, store the indexes
    rMax = 22.7  # r<22.7
    preSelection = (r<rMax) &  psflike(objtype) & DECaLSOK 
    colorsCopy = colors.copy()
    colorsReduced = colorsCopy[preSelection]
    colorsIndex =  np.arange(0,nbEntries,dtype=np.int64)
    colorsReducedIndex =  colorsIndex[preSelection]

    #Load BDT
    pathToBDT = resource_filename('desitarget', "data/bdt_model_dr3/bdt.pkl") 
    # Compute BDT probability
    from sklearn.externals import joblib
    bdt = joblib.load(pathToBDT) 
    prob = np.zeros(nbEntries)     

    if (colorsReducedIndex.any()) :
        objects_bdt = bdt.predict_proba(colorsReduced)
        # add BDT probability to preselected objects
        j=0
        for i in colorsReducedIndex :
            prob[i]=objects_bdt[j,1]
            j += 1


    #define pcut, relaxed cut for faint objects
    pcut = np.where(r>20.0,0.95 - (r-20.0)*0.08,0.95)

    qso = primary.copy()
    qso &= r<rMax  
    qso &= DECaLSOK  
      
    if objtype is not None:
        qso &= psflike(objtype)

    if deltaChi2 is not None:
        qso &= deltaChi2>30.

    if nbEntries==1 : #to pass one test...
        qso &= prob[0]>pcut
    else :
        qso &= prob>pcut        
        
    return qso


def getColors(nbEntries,nfeatures,gflux,rflux,zflux,w1flux,w2flux):
 
    limitInf=1.e-04
    gflux = gflux.clip(limitInf)
    rflux = rflux.clip(limitInf)
    zflux = zflux.clip(limitInf)
    w1flux = w1flux.clip(limitInf)
    w2flux = w2flux.clip(limitInf)

    g=np.where( gflux>limitInf,22.5-2.5*np.log10(gflux), 0.)
    r=np.where( rflux>limitInf,22.5-2.5*np.log10(rflux), 0.)
    z=np.where( zflux>limitInf,22.5-2.5*np.log10(zflux), 0.)
    W1=np.where( w1flux>limitInf, 22.5-2.5*np.log10(w1flux), 0.)
    W2=np.where( w2flux>limitInf, 22.5-2.5*np.log10(w2flux), 0.)

    DECaLSOK = (g>0.) & (r>0.) & (z>0.) & (W1>0.) & (W2>0.)

    colors  = np.zeros((nbEntries,nfeatures))
    colors[:,0]=g-r
    colors[:,1]=r-z
    colors[:,2]=g-z
    colors[:,3]=g-W1
    colors[:,4]=r-W1
    colors[:,5]=z-W1
    colors[:,6]=g-W2
    colors[:,7]=r-W2
    colors[:,8]=z-W2
    colors[:,9]=W1-W2
    colors[:,10]=r

    return colors, r, DECaLSOK

def _is_row(table):
    '''Return True/False if this is a row of a table instead of a full table
    
    supports numpy.ndarray, astropy.io.fits.FITS_rec, and astropy.table.Table
    '''
    import astropy.io.fits.fitsrec
    import astropy.table.row
    if isinstance(table, (astropy.io.fits.fitsrec.FITS_record, astropy.table.row.Row)) or \
        np.isscalar(table):
        return True
    else:
        return False

def unextinct_fluxes(objects):
    """
    Calculate unextincted DECam and WISE fluxes
    
    Args:
        objects: array or Table with columns DECAM_FLUX, DECAM_MW_TRANSMISSION,
            WISE_FLUX, and WISE_MW_TRANSMISSION
            
    Returns:
        array or Table with columns GFLUX, RFLUX, ZFLUX, W1FLUX, W2FLUX
        
    Output type is Table if input is Table, otherwise numpy structured array
    """
    dtype = [('GFLUX', 'f4'), ('RFLUX', 'f4'), ('ZFLUX', 'f4'),
             ('W1FLUX', 'f4'), ('W2FLUX', 'f4')]
    if _is_row(objects):
        result = np.zeros(1, dtype=dtype)[0]
    else:
        result = np.zeros(len(objects), dtype=dtype)

#    dered_decam_flux = objects['DECAM_FLUX'] / objects['DECAM_MW_TRANSMISSION']
    dered_decam_flux = np.divide(objects['DECAM_FLUX'] , objects['DECAM_MW_TRANSMISSION'], 
                                 where=objects['DECAM_MW_TRANSMISSION']!=0)
    result['GFLUX'] = dered_decam_flux[..., 1]
    result['RFLUX'] = dered_decam_flux[..., 2]
    result['ZFLUX'] = dered_decam_flux[..., 4]

#    dered_wise_flux = objects['WISE_FLUX'] / objects['WISE_MW_TRANSMISSION']
    dered_wise_flux =  np.divide(objects['WISE_FLUX'] , objects['WISE_MW_TRANSMISSION'],
                                 where=objects['WISE_MW_TRANSMISSION']!=0)
    result['W1FLUX'] = dered_wise_flux[..., 0]
    result['W2FLUX'] = dered_wise_flux[..., 1]

    if isinstance(objects, Table):
        return Table(result)
    else:
        return result

def apply_cuts(objects,qso_selection='bdt'):
    """Perform target selection on objects, returning target mask arrays

    Args:
        objects: numpy structured array with UPPERCASE columns needed for
            target selection, OR a string tractor/sweep filename
            
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

    bgs = isBGS(primary=primary, rflux=rflux, objtype=objtype)
    
    if qso_selection=='colorcuts' :
        qso = isQSO(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2, objtype=objtype,
                wise_snr=wise_snr)
    else :    
        qso = isQSO_BDT(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2, objtype=objtype)

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
    
    bgs_target = bgs * bgs_mask.BGS_BRIGHT
    bgs_target |= bgs * bgs_mask.BGS_BRIGHT_SOUTH

    #- nothing for MWS yet; will be GAIA-based
    if isinstance(bgs_target, int):
        mws_target = 0
    else:
        mws_target = np.zeros_like(bgs_target)

    #- Are any BGS or MWS bit set?  Tell desi_target too.
    desi_target |= (bgs_target != 0) * desi_mask.BGS_ANY
    desi_target |= (mws_target != 0) * desi_mask.MWS_ANY

    return desi_target, bgs_target, mws_target

def select_targets(infiles, numproc=4, verbose=False, qso_selection='bdt'):
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
        desi_target, bgs_target, mws_target = apply_cuts(objects,qso_selection)
        
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
