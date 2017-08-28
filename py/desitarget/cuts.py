"""
desitarget.cuts
===============

Target Selection for DECALS catalogue data

https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (*e.g.* LRG, ELG or QSO).
"""
import warnings
from time import time
import os.path

import numbers
import sys

import numpy as np
from astropy.table import Table, Row
from pkg_resources import resource_filename

from desitarget import io
from desitarget.internal import sharedmem
import desitarget.targets
from desitarget import desi_mask, bgs_mask, mws_mask

def isLRG_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                        w2flux=None, ggood=None, primary=None):
    """See :func:`~desitarget.cuts.isLRG` for details.
    This function applies just the flux and color cuts.
    """

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
        lrg = primary.copy()

    if ggood is None:
        ggood = np.ones_like(gflux, dtype='?')

    # Basic flux and color cuts
    lrg = primary.copy()
    lrg &= (zflux > 10**(0.4*(22.5-20.4))) # z<20.4
    lrg &= (zflux < 10**(0.4*(22.5-18))) # z>18
    lrg &= (zflux < 10**(0.4*2.5)*rflux) # r-z<2.5
    lrg &= (zflux > 10**(0.4*0.8)*rflux) # r-z>0.8

    # The code below can overflow, since the fluxes are float32 arrays
    # which have a maximum value of 3e38. Therefore, if eg. zflux~1.0e10
    # this will overflow, and crash the code.
    with np.errstate(over='ignore'):
        # This is the star-galaxy separation cut
        # Wlrg = (z-W)-(r-z)/3 + 0.3 >0 , which is equiv to r+3*W < 4*z+0.9
        lrg &= (rflux*w1flux**3 > (zflux**4)*10**(-0.4*0.9))

        # Now for the work-horse sliding flux-color cut:
        # mlrg2 = z-2*(r-z-1.2) < 19.6 -> 3*z < 19.6-2.4-2*r
        lrg &= (zflux**3 > 10**(0.4*(22.5+2.4-19.6))*rflux**2)

        # Another guard against bright & red outliers
        # mlrg2 = z-2*(r-z-1.2) > 17.4 -> 3*z > 17.4-2.4-2*r
        lrg &= (zflux**3 < 10**(0.4*(22.5+2.4-17.4))*rflux**2)

        # Finally, a cut to exclude the z<0.4 objects while retaining the elbow at
        # z=0.4-0.5.  r-z>1.2 || (good_data_in_g and g-r>1.7).  Note that we do not
        # require gflux>0.
        lrg &= np.logical_or((zflux > 10**(0.4*1.2)*rflux), (ggood & (rflux>10**(0.4*1.7)*gflux)))

    return lrg


def isLRG(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, 
          rflux_snr=None, zflux_snr=None, w1flux_snr=None,
          gflux_ivar=None, primary=None):
    """Target Definition of LRG. Returning a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1 and W2 bands (if needed).
        gflux, rflux_snr, zflux_snr, w1flux_snr: array_like
            The signal-to-noise in the r, z and W1 bands defined as the flux
            per band divided by sigma (flux x the sqrt of the inverse variance)
        gflux_ivar: array_like
            The inverse variance of the flux in g-band
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only the object is an LRG
            target.

    Notes:
        This is version 3 of the Eisenstein/Dawson Summer 2016 work on LRG target
        selection, but anymask has been changed to allmask, which probably means
        that the flux cuts need to be re-tuned.  That is, mlrg2<19.6 may need to
        change to 19.5 or 19.4. --Daniel Eisenstein -- Jan 9, 2017
    """
    #----- Luminous Red Galaxies
    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
        lrg = primary.copy()

    # Some basic quality in r, z, and W1.  Note by @moustakas: no allmask cuts
    # used!).  Also note: We do not require gflux>0!  Objects can be very red.
    lrg = primary.copy()
    lrg &= (rflux_snr > 0) # and rallmask == 0
    lrg &= (zflux_snr > 0) # and zallmask == 0
    lrg &= (w1flux_snr > 4)
    lrg &= (rflux > 0)
    lrg &= (zflux > 0)
    ggood = (gflux_ivar > 0) # and gallmask == 0

    # Apply color, flux, and star-galaxy separation cuts
    lrg &= isLRG_colors(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, 
                               w2flux=w2flux, ggood=ggood, primary=primary)

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
    """

    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    fstd = primary.copy()

    # Clip to avoid warnings from negative numbers.
    gflux = gflux.clip(0)
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)

    # colors near BD+17
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grcolor = 2.5 * np.log10(rflux / gflux)
        rzcolor = 2.5 * np.log10(zflux / rflux)
        fstd &= (grcolor - 0.26)**2 + (rzcolor - 0.13)**2 < 0.06**2

    return fstd


def isFSTD(gflux=None, rflux=None, zflux=None, primary=None, 
           gfracflux=None, rfracflux=None, zfracflux=None,
           gsnr=None, rsnr=None, zsnr=None,
           objtype=None, obs_rflux=None, bright=False):
    """Select FSTD targets using color cuts and photometric quality cuts (PSF-like
    and fracflux).  See isFSTD_colors() for additional info.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
          The flux in nano-maggies of g, r, z, w1, and w2 bands.
        gfracflux, rfracflux, zfracflux: array_like
          Profile-weight fraction of the flux from other sources divided by the 
          total flux in g, r and z bands.
        gsnr, rsnr, zsnr: array_like
          The signal-to-noise ratio in g, r, and z bands.
        primary: array_like or None
          If given, the BRICK_PRIMARY column of the catalogue.
        bright: apply magnitude cuts for "bright" conditions; otherwise, choose
          "normal" brightness standards.

    Returns:
        mask : boolean array, True if the object has colors like an FSTD
    """
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')
    fstd = primary.copy()

    # Apply the magnitude and color cuts.
    fstd &= isFSTD_colors(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux)

    # Apply type=PSF, fracflux, and S/N cuts.
    fstd &= _psflike(objtype)

    #ADM probably a more elegant way to do this, coded it like this for
    #ADM data model transition from 2-D to 1-D arrays
    fracflux = [gfracflux, rfracflux, zfracflux]
    snr = [gsnr, rsnr, zsnr]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # fracflux can be Inf/NaN
        for j in (0, 1, 2):  # g, r, z
            fstd &= fracflux[j] < 0.04
            fstd &= snr[j] > 10

    # Observed flux; no Milky Way extinction
    if obs_rflux is None:
        obs_rflux = rflux

    if bright:
        rbright = 14.0
        rfaint = 17.0
    else:
        rbright = 16.0
        rfaint = 19.0

    fstd &= obs_rflux < 10**((22.5 - rbright)/2.5)
    fstd &= obs_rflux > 10**((22.5 - rfaint)/2.5)

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

def isBGS_faint(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, objtype=None, primary=None):
    """Target Definition of BGS faint targets, returning a boolean array.

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
    bgs &= rflux > 10**((22.5-20.0)/2.5)
    bgs &= rflux <= 10**((22.5-19.5)/2.5)
    if objtype is not None:
        bgs &= ~_psflike(objtype)

    return bgs

def isBGS_bright(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, objtype=None, primary=None):
    """Target Definition of BGS bright targets, returning a boolean array.

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
        bgs &= ~_psflike(objtype)
    return bgs

def isQSO_colors(gflux, rflux, zflux, w1flux, w2flux, optical=False):
    """Tests if objects have QSO-like colors, i.e. a subset of the QSO cuts.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        optical : Just apply optical color-cuts (default False)

    Returns:
        mask : array_like. True if the object has QSO-like colors.
    """
    #----- Quasars
    # Create some composite fluxes.
    wflux = 0.75* w1flux + 0.25*w2flux
    grzflux = (gflux + 0.8*rflux + 0.5*zflux) / 2.3

    qso = np.ones(len(gflux), dtype='?')
    qso &= rflux > 10**((22.5-22.7)/2.5)    # r<22.7
    qso &= grzflux < 10**((22.5-17)/2.5)    # grz>17
    qso &= rflux < gflux * 10**(1.3/2.5)    # (g-r)<1.3
    qso &= zflux > rflux * 10**(-0.3/2.5)   # (r-z)>-0.3
    qso &= zflux < rflux * 10**(1.1/2.5)    # (r-z)<1.1

    if not optical:
        qso &= w2flux > w1flux * 10**(-0.4/2.5) # (W1-W2)>-0.4
        qso &= wflux * gflux > zflux * grzflux * 10**(-1.0/2.5) # (grz-W)>(g-z)-1.0

    # Harder cut on stellar contamination
    mainseq = rflux > gflux * 10**(0.20/2.5)

    # Clip to avoid warnings from negative numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    mainseq &= rflux**(1+1.5) > gflux * zflux**1.5 * 10**((-0.100+0.175)/2.5)
    mainseq &= rflux**(1+1.5) < gflux * zflux**1.5 * 10**((+0.100+0.175)/2.5)
    if not optical:
        mainseq &= w2flux < w1flux * 10**(0.3/2.5)
    qso &= ~mainseq

    return qso

def isQSO_cuts(gflux, rflux, zflux, w1flux, w2flux, w1snr, w2snr, deltaChi2,
               objtype=None, primary=None):
    """Cuts based QSO target selection

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
        deltaChi2: array_like
            chi2 difference between PSF and SIMP models,  dchisq_PSF - dchisq_SIMP
        w1snr: array_like[ntargets]
            S/N in the W1 band.
        w2snr: array_like[ntargets]
            S/N in the W2 band.
        objtype (optional): array_like or None
            If given, the TYPE column of the Tractor catalogue.
        primary (optional): array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only the object is a QSO
            target.

    Notes:
        Uses isQSO_colors() to make color cuts first, then applies
            w1snr, w2snr, deltaChi2, and optionally primary and objtype cuts

    """
    qso = isQSO_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                       w1flux=w1flux, w2flux=w2flux)

    qso &= w1snr > 4
    qso &= w2snr > 2

    qso &= deltaChi2>40.

    if primary is not None:
        qso &= primary

    if objtype is not None:
        qso &= _psflike(objtype)

    return qso

def isQSO_randomforest(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, objtype=None,
         deltaChi2=None, primary=None):
    """Target Definition of QSO using a random forest returning a boolean array.

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

    # build variables for random forest
    nfeatures=11 # number of variables in random forest
    nbEntries=rflux.size
    colors, r, DECaLSOK = _getColors(nbEntries, nfeatures, gflux, rflux, zflux, w1flux, w2flux)

    #Preselection to speed up the process, store the indexes
    rMax = 22.7  # r<22.7
    #ADM this previous had no np.where but was flagging DeprecationWarnings on
    #ADM indexing a Boolean, so I switched the Boolean to an integer via np.where
    preSelection = np.where( (r<rMax) & _psflike(objtype) & DECaLSOK )
    colorsCopy = colors.copy()
    colorsReduced = colorsCopy[preSelection]
    colorsIndex =  np.arange(0,nbEntries,dtype=np.int64)
    colorsReducedIndex =  colorsIndex[preSelection]

    #Path to random forest files
    pathToRF = resource_filename('desitarget', "data")

    # Compute random forest probability
    from desitarget.myRF import myRF
    prob = np.zeros(nbEntries)

    if (colorsReducedIndex.any()) :
        rf = myRF(colorsReduced,pathToRF,numberOfTrees=200,version=1)
        fileName = pathToRF + '/rf_model_dr3.npz'
        rf.loadForest(fileName)
        objects_rf = rf.predict_proba()
        # add random forest probability to preselected objects
        j=0
        for i in colorsReducedIndex :
            prob[i]=objects_rf[j]
            j += 1

    #define pcut, relaxed cut for faint objects
    pcut = np.where(r>20.0,0.95 - (r-20.0)*0.08,0.95)

    qso = primary.copy()
    qso &= r<rMax
    qso &= DECaLSOK

    if objtype is not None:
        qso &= _psflike(objtype)

    if deltaChi2 is not None:
        qso &= deltaChi2>30.

    if nbEntries==1 : # for call of a single object
        qso &= prob[0]>pcut
    else :
        qso &= prob>pcut

    return qso

def _psflike(psftype):
    """ If the object is PSF """
    #ADM explicitly checking for NoneType. I can't see why we'd ever want to
    #ADM run this test on empty information. In the past we have had bugs where
    #ADM we forgot to pass objtype=objtype in, e.g., isFSTD
    if psftype is None:
        raise ValueError("NoneType submitted to _psfflike function")

    #- 'PSF' for astropy.io.fits; 'PSF ' for fitsio (sigh)
    #ADM fixed this in I/O.
    psftype = np.asarray(psftype)
    #ADM in Python3 these string literals become byte-like
    #ADM so to retain Python2 compatibility we need to check
    #ADM against both bytes and unicode
    psflike = ((psftype == 'PSF') | (psftype == b'PSF'))
    return psflike

def _getColors(nbEntries, nfeatures, gflux, rflux, zflux, w1flux, w2flux):

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
        objects: array or Table with columns FLUX_G, FLUX_R, FLUX_Z, 
            MW_TRANSMISSION_G, MW_TRANSMISSION_R, MW_TRANSMISSION_Z,
            FLUX_W1, FLUX_W2, MW_TRANSMISSION_W1, MW_TRANSMISSION_W2

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

#ADM This was a hack for DR3 because of some corrupt sweeps/Tractor files,
#ADM the comment can be removed if DR4/DR5 run OK. It's just here as a reminder.
#    dered_decam_flux = np.divide(objects['DECAM_FLUX'] , objects['DECAM_MW_TRANSMISSION'],
#                                 where=objects['DECAM_MW_TRANSMISSION']!=0)
    result['GFLUX'] = objects['FLUX_G'] / objects['MW_TRANSMISSION_G']
    result['RFLUX'] = objects['FLUX_R'] / objects['MW_TRANSMISSION_R']
    result['ZFLUX'] = objects['FLUX_Z'] / objects['MW_TRANSMISSION_Z']

#ADM This was a hack for DR3 because of some corrupt sweeps/Tractor files,
#ADM the comment can be removed if DR4/DR5 run OK. It's just here as a reminder.
    result['W1FLUX'] = objects['FLUX_W1'] / objects['MW_TRANSMISSION_W1']
    result['W2FLUX'] = objects['FLUX_W2'] / objects['MW_TRANSMISSION_W2']

    if isinstance(objects, Table):
        return Table(result)
    else:
        return result

def apply_cuts(objects, qso_selection='randomforest'):
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
        objects = io.read_tractor(objects)

    #- ensure uppercase column names if astropy Table
    if isinstance(objects, (Table, Row)):
        for col in list(objects.columns.values()):
            if not col.name.isupper():
                col.name = col.name.upper()

    obs_rflux = objects['FLUX_R'] # observed r-band flux (used for F standards, below)

    #- undo Milky Way extinction
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']

    gfluxivar = objects['FLUX_IVAR_G']

    gfracflux = objects['FRACFLUX_G'].T # note transpose
    rfracflux = objects['FRACFLUX_R'].T # note transpose
    zfracflux = objects['FRACFLUX_Z'].T # note transpose

    gsnr = objects['FLUX_G'] * np.sqrt(objects['FLUX_IVAR_G'])
    rsnr = objects['FLUX_R'] * np.sqrt(objects['FLUX_IVAR_R'])
    zsnr = objects['FLUX_Z'] * np.sqrt(objects['FLUX_IVAR_Z'])
    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    w2snr = objects['FLUX_W2'] * np.sqrt(objects['FLUX_IVAR_W2'])

    # Delta chi2 between PSF and SIMP morphologies; note the sign....
    dchisq = objects['DCHISQ']
    deltaChi2 = dchisq[...,0] - dchisq[...,1]

    #ADM remove handful of NaN values from DCHISQ values and make them unselectable
    w = np.where(deltaChi2 != deltaChi2)
    #ADM this is to catch the single-object case for unit tests
    if len(w[0]) > 0:
        deltaChi2[w] = -1e6

    #- DR1 has targets off the edge of the brick; trim to just this brick
    try:
        primary = objects['BRICK_PRIMARY']
    except (KeyError, ValueError):
        if _is_row(objects):
            primary = True
        else:
            primary = np.ones_like(objects, dtype=bool)

    lrg = isLRG(primary=primary, gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                   gflux_ivar=gfluxivar, rflux_snr=rsnr, zflux_snr=zsnr, w1flux_snr=w1snr)
    
    elg = isELG(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux)

    bgs_bright = isBGS_bright(primary=primary, rflux=rflux, objtype=objtype)
    bgs_faint  = isBGS_faint(primary=primary, rflux=rflux, objtype=objtype)

    if qso_selection=='colorcuts' :
        qso = isQSO_cuts(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                         w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2, objtype=objtype,
                         w1snr=w1snr, w2snr=w2snr)
    elif qso_selection == 'randomforest':
        qso = isQSO_randomforest(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                                 w1flux=w1flux, w2flux=w2flux, deltaChi2=deltaChi2, objtype=objtype)
    else:
        raise ValueError('Unknown qso_selection {}; valid options are {}'.format(qso_selection,
                                                                                 qso_selection_options))
    #ADM Make sure to pass all of the needed columns! At one point we stopped
    #ADM passing objtype, which meant no standards were being returned.
    fstd = isFSTD(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                  gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                  gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                  obs_rflux=obs_rflux, objtype=objtype)
    fstd_bright = isFSTD(primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                  gfracflux=gfracflux, rfracflux=rfracflux, zfracflux=zfracflux,
                  gsnr=gsnr, rsnr=rsnr, zsnr=zsnr,
                  obs_rflux=obs_rflux, objtype=objtype, bright=True)

    # Construct the targetflag bits; currently our only cuts are DECam based
    # (i.e. South).  This should really be refactored into a dedicated function.
    desi_target  = lrg * desi_mask.LRG_SOUTH
    desi_target |= elg * desi_mask.ELG_SOUTH
    desi_target |= qso * desi_mask.QSO_SOUTH

    desi_target |= lrg * desi_mask.LRG
    desi_target |= elg * desi_mask.ELG
    desi_target |= qso * desi_mask.QSO

    # Standards; still need to set STD_WD
    desi_target |= fstd * desi_mask.STD_FSTAR
    desi_target |= fstd_bright * desi_mask.STD_BRIGHT

    # BGS, bright and faint
    bgs_target = bgs_bright * bgs_mask.BGS_BRIGHT
    bgs_target |= bgs_bright * bgs_mask.BGS_BRIGHT_SOUTH
    bgs_target |= bgs_faint * bgs_mask.BGS_FAINT
    bgs_target |= bgs_faint * bgs_mask.BGS_FAINT_SOUTH

    # Nothing for MWS yet; will be GAIA-based.
    if isinstance(bgs_target, numbers.Integral):
        mws_target = 0
    else:
        mws_target = np.zeros_like(bgs_target)

    # Are any BGS or MWS bit set?  Tell desi_target too.
    desi_target |= (bgs_target != 0) * desi_mask.BGS_ANY
    desi_target |= (mws_target != 0) * desi_mask.MWS_ANY

    return desi_target, bgs_target, mws_target

def check_input_files(infiles, numproc=4, verbose=False):
    """
    Process input files in parallel to check whether they have
    any bugs that will prevent select_targets from completing,
    or whether files are corrupted.
    Useful to run before a full run of select_targets.

    Args:
        infiles: list of input filenames (tractor or sweep files),
            OR a single filename

    Optional:
        numproc: number of parallel processes
        verbose: if True, print progress messages

    Returns:
        Nothing, but prints any problematic files to screen
        together with information on the problem

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
    def _check_input_files(filename):
        '''Check for corrupted values in a file'''
        from functools import partial
        from os.path import getsize

        #ADM read in Tractor or sweeps files
        objects = io.read_tractor(filename)
        #ADM if everything is OK the default meassage will be "OK"
        filemessageroot = 'OK'
        filemessageend = ''
        #ADM columns that shouldn't have zero values
        cols = [
            'BRICKID',
#            'RA_IVAR', 'DEC_IVAR',
            'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
#            'WISE_FLUX',
#            'WISE_MW_TRANSMISSION','DCHISQ'
            ]
        #ADM for each of these columnes that shouldn't have zero values,
        #ADM loop through and look for zero values
        for colname in cols:
            if np.min(objects[colname]) == 0:
                filemessageroot = "WARNING...some values are zero for"
                filemessageend += " "+colname

        #ADM now, loop through entries in the file and search for 4096-byte
        #ADM blocks that are all zeros (a sign of corruption in file-writing)
        #ADM Note that fits files are padded by 2880 bytes, so we only want to
        #ADM process the file length (in bytes) - 2880
        bytestop = getsize(filename) -2880

        with open(filename, 'rb') as f:
            for block_number, data in enumerate(iter(partial(f.read, 4096), b'')):
                if not any(data):
                    if block_number*4096 < bytestop:
                        filemessageroot = "WARNING...some values are zero for"
                        filemessageend += ' 4096-byte-block-#{0}'.format(block_number)

        return [filename,filemessageroot+filemessageend]

    # Counter for number of bricks processed;
    # a numpy scalar allows updating nbrick in python 2
    # c.f https://www.python.org/dev/peps/pep-3104/
    nbrick = np.zeros((), dtype='i8')

    t0 = time()
    def _update_status(result):
        ''' wrapper function for the critical reduction operation,
            that occurs on the main parallel process '''
        if verbose and nbrick%25 == 0 and nbrick>0:
            elapsed = time() - t0
            rate = nbrick / elapsed
            print('{} files; {:.1f} files/sec; {:.1f} total mins elapsed'.format(nbrick, rate, elapsed/60.))
        nbrick[...] += 1    # this is an in-place modification
        return result

    #- Parallel process input files
    if numproc > 1:
        pool = sharedmem.MapReduce(np=numproc)
        with pool:
            fileinfo = pool.map(_check_input_files, infiles, reduce=_update_status)
    else:
        fileinfo = list()
        for fil in infiles:
            fileinfo.append(_update_status(_check_input_files(fil)))

    fileinfo = np.array(fileinfo)
    w = np.where(fileinfo[...,1] != 'OK')

    if len(w[0]) == 0:
        print('ALL FILES ARE OK')
    else:
        for fil in fileinfo[w]:
            print(fil[0],fil[1])

    return len(w[0])


qso_selection_options = ['colorcuts', 'randomforest']
Method_sandbox_options = ['XD', 'RF_photo', 'RF_spectro']

def select_targets(infiles, numproc=4, verbose=False, qso_selection='randomforest',
                   sandbox=False, FoMthresh=None, Method=None):
    """Process input files in parallel to select targets

    Args:
        infiles: list of input filenames (tractor or sweep files),
            OR a single filename
        numproc (optional): number of parallel processes to use
        verbose (optional): if True, print progress messages
        qso_selection (optional): algorithm to use for QSO selection; valid options
            are 'colorcuts' and 'randomforest'
        sandbox (optional): if True, use the sample selection cuts in
            :mod:`desitarget.sandbox.cuts`.
        FoMthresh (optional): if a value is passed then run apply_XD_globalerror for ELGs in
            the sandbox. This will write out an "FoM.fits" file for every ELG target
            in the sandbox directory.
        Method (optional): Method used in sandbox    

    Returns:
        targets numpy structured array
            the subset of input targets which pass the cuts, including extra
            columns for DESI_TARGET, BGS_TARGET, and MWS_TARGET target
            selection bitmasks.

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

    def _finalize_targets(objects, desi_target, bgs_target, mws_target):
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

    #- functions to run on every brick/sweep file
    def _select_targets_file(filename):
        '''Returns targets in filename that pass the cuts'''
        objects = io.read_tractor(filename)
        desi_target, bgs_target, mws_target = apply_cuts(objects, qso_selection)

        return _finalize_targets(objects, desi_target, bgs_target, mws_target)

    def _select_sandbox_targets_file(filename):
        '''Returns targets in filename that pass the sandbox cuts'''
        from desitarget.sandbox.cuts import apply_sandbox_cuts
        objects = io.read_tractor(filename)
        desi_target, bgs_target, mws_target = apply_sandbox_cuts(objects,FoMthresh,Method)

        return _finalize_targets(objects, desi_target, bgs_target, mws_target)

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
            if sandbox:
                if verbose:
                    print("You're in the sandbox...")
                targets = pool.map(_select_sandbox_targets_file, infiles, reduce=_update_status)
            else:
                targets = pool.map(_select_targets_file, infiles, reduce=_update_status)
    else:
        targets = list()
        if sandbox:
            if verbose:
                print("You're in the sandbox...")
            for x in infiles:
                targets.append(_update_status(_select_sandbox_targets_file(x)))
        else:
            for x in infiles:
                targets.append(_update_status(_select_targets_file(x)))

    targets = np.concatenate(targets)

    return targets
