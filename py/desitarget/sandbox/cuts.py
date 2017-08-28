"""
desitarget.sandbox.cuts
=======================

Sandbox target selection cuts, intended for algorithms that are still in
development.

"""
import os.path
from time import time

import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.table import Table, Row
from pkg_resources import resource_filename

import desitarget.targets
from ..cuts import unextinct_fluxes, _is_row
from ..internal import sharedmem
from ..targetmask import desi_mask, bgs_mask, mws_mask
from .. import io

def write_fom_targets(targets, FoM, desi_target, bgs_target, mws_target):
    """Return new targets array with added/renamed columns including ELG Figure of Merit

    Args:
        targets: numpy structured array of targets
        FoM: Figure of Merit calculated by apply_XD_globalerror
        desi_target: 1D array of target selection bit flags
        bgs_target: 1D array of target selection bit flags
        mws_target: 1D array of target selection bit flags

    Returns:
        New targets structured array with those changes

    Notes:

        Finalize target list by:

        * renaming OBJID -> BRICK_OBJID (it is only unique within a brick)
        * Adding new columns:

          - TARGETID: unique ID across all bricks
          - FoM: ELG XD Figure of Merit
          - DESI_TARGET: target selection flags
          - MWS_TARGET: target selection flags
          - BGS_TARGET: target selection flags
    """
    ntargets = len(targets)
    assert ntargets == len(FoM)
    assert ntargets == len(desi_target)
    assert ntargets == len(bgs_target)
    assert ntargets == len(mws_target)

    #- OBJID in tractor files is only unique within the brick; rename and
    #- create a new unique TARGETID
    targets = rfn.rename_fields(targets, {'OBJID':'BRICK_OBJID'})
    targetid = targets['BRICKID'].astype(np.int64)*1000000 + targets['BRICK_OBJID']

    #- Add new columns: TARGETID, TARGETFLAG, NUMOBS
    targets = rfn.append_fields(targets,
        ['TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'FOM'],
        [targetid, desi_target, bgs_target, mws_target, FoM], usemask=False)

    io.write_targets('FoM.fits', targets, qso_selection='irrelevant',sandboxcuts=True)

    print('{} targets written to {}'.format(len(targets), 'FoM.fits'))

    return

def isLRG_pre2017v0(gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None):
    """Initial version of LRG cuts superceded in early 2017"""

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

def isLRG_2016v3_colors(gflux=None, rflux=None, zflux=None, w1flux=None,
                        w2flux=None, ggood=None, primary=None):
    """See :func:`~desitarget.sandbox.cuts.isLRG2016v3` for details.
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

def isLRG_2016v3(gflux=None, rflux=None, zflux=None, w1flux=None,
                 rflux_snr=None, zflux_snr=None, w1flux_snr=None,
                 gflux_ivar=None, primary=None):
    """This is version 3 of the Eisenstein/Dawson Summer 2016 work on LRG target
    selection, but anymask has been changed to allmask, which probably means
    that the flux cuts need to be re-tuned.  That is, mlrg2<19.6 may need to
    change to 19.5 or 19.4. --Daniel Eisenstein -- Jan 9, 2017

    Args:
        gflux, rflux, zflux

    Returns:
        stuff

    Notes:
        - Inputs: decam_flux, decam_flux_ivar, decam_allmask, decam_mw_transmission
          wise_flux, wise_flux_ivar, wise_mw_transmission
          Using g, r, z, and W1 information.
        - Applying the reddening
        - Also clip r, z, and W1 at 0 to avoid warnings from negative numbers raised to
          fractional powers.
    """
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

    # Apply color, flux, and star-galaxy separation cuts.
    lrg &= isLRG_2016v3_colors(gflux=gflux, rflux=rflux, zflux=zflux,
                               w1flux=w1flux, ggood=ggood, primary=primary)

    return lrg


def apply_XD_globalerror(objs, last_FoM, glim=23.8, rlim=23.4, zlim=22.4, gr_ref=0.5,\
                       rz_ref=0.5,reg_r=1e-4/(0.025**2 * 0.05),f_i=[1., 1., 0., 0.25, 0., 0.25, 0.],\
                       gmin = 21., gmax = 24.):
    """ Apply ELG XD selection. Default uses fiducial set of parameters.

    Args:
        objs: A DECaLS fits table.
        last_FoM: Threshold FoM.
        glim, rlim, zlim (optional): 5-sigma detection limiting magnitudes.
        gr_ref, rz_ref (optional): Number density conserving global error reference point.
        reg_r (optional): Regularization parameter. Empirically set to avoid pathologic
            behaviors of the selection boundary.
        f_i (optional): Various class weights for FoM.
        gmin, gmax (optional): Minimum and maximum g-magnitude range to consider.


    Returns:
        iXD: Boolean mask array that implements XD selection.
        FoM: Figure of Merit number computed for objects that pass the initial set of masks.

    Note:
        1. The current version of XD selection method assumes the imposition of decam_allmask
            and tycho2 stellar mask. (The individual class densities have been fitted with these
            masks imposed.) However, the code does not implement them yet as we want to understand
            the large scale systematics of the XD selection with and without these masks.
        2. A different  version of this function using individual Tractor error is called
            apply_XD_Tractor_error().

        Process in summary:
            - Construct a Python dictionary that contains all XD GMM and dNdm parameters
                using a string.
            - Load variables from the input astropy fits table.
            - Compute which objects pass the reasonable imaging quality cut
                (SNR>2, flux positive, and flux invariance positive).
            - Compute which objects pass a rough color cut that eliminates a
                bulk of low redshift contaiminants.
            - For each object that passes the above two cuts, compute Figure of Merit FoM.
            - If FoM>FoM_last, then include the object in the selection.
            - Append this selection column to the table and return.

    """

    ####### Density parameters hard coded in. np.float64 used for maximal precision. #######
    params ={(0, 'mean'): np.array([[ 0.374283820390701,  1.068873405456543],
           [ 0.283886760473251,  0.733299076557159]]),
    (1, 'mean'): np.array([[ 0.708186626434326,  1.324055671691895],
           [ 0.514687597751617,  0.861691951751709]]),
    (2, 'mean'): np.array([[ 0.851126551628113,  1.49790346622467 ],
           [ 0.593997478485107,  1.027981519699097]]),
    (3, 'mean'): np.array([[ 0.621764063835144,  0.677076101303101],
           [ 1.050391912460327,  1.067378640174866]]),
    (4, 'mean'): np.array([[ 0.29889178276062 ,  0.158586874604225],
           [ 0.265404641628265,  0.227356120944023],
           [ 1.337790369987488,  1.670260787010193]]),
    (5, 'mean'): np.array([[ 0.169899195432663,  0.333086401224136],
           [ 0.465608537197113,  0.926179945468903]]),
    (6, 'mean'): np.array([[ 0.404752403497696,  0.157505303621292],
           [ 1.062281489372253,  0.708624482154846],
           [ 0.767854988574982,  0.410259902477264],
           [ 1.830820441246033,  1.096370458602905],
           [ 1.224291563034058,  0.748376846313477],
           [ 0.623223185539246,  0.588687479496002],
           [ 1.454894185066223,  1.615718483924866]]),
    (0, 'amp'): np.array([ 0.244611976587951,  0.755388023412049]),
    (1, 'amp'): np.array([ 0.114466286005043,  0.885533713994957]),
    (2, 'amp'): np.array([ 0.138294309756769,  0.861705690243231]),
    (3, 'amp'): np.array([ 0.509696013263716,  0.490303986736284]),
    (4, 'amp'): np.array([ 0.264565190839574,  0.464308147030861,  0.271126662129565]),
    (5, 'amp'): np.array([ 0.803360982047185,  0.196639017952815]),
    (6, 'amp'): np.array([ 0.09128923215233 ,  0.254327925723203,  0.31780750840433 ,
            0.036144574976436,  0.145786317010496,  0.031381535653226,
            0.12326290607998 ]),
    (0, 'covar'): np.array([[[ 0.10418130703232 ,  0.014280057648813],
            [ 0.014280057648813,  0.070314900027689]],

           [[ 0.023818843706279,  0.018202660741959],
            [ 0.018202660741959,  0.041376141039073]]]),
    (1, 'covar'): np.array([[[ 0.215211984773353,  0.054615838823342],
            [ 0.054615838823342,  0.049833562813203]],

           [[ 0.04501376209018 ,  0.017654245897094],
            [ 0.017654245897094,  0.036243604905033]]]),
    (2, 'covar'): np.array([[[ 0.393998394239911,  0.08339271763515 ],
            [ 0.08339271763515 ,  0.043451758548033]],

           [[ 0.104132127558071,  0.066660191134385],
            [ 0.066660191134385,  0.099474014771686]]]),
    (3, 'covar'): np.array([[[ 0.077655250186381,  0.048031436118266],
            [ 0.048031436118266,  0.104180325930248]],

           [[ 0.18457377102254 ,  0.13405411581603 ],
            [ 0.13405411581603 ,  0.11061389825436 ]]]),
    (4, 'covar'): np.array([[[ 0.004346580392509,  0.002628470120243],
            [ 0.002628470120243,  0.003971775282994]],

           [[ 0.048642690792318,  0.010716631911343],
            [ 0.010716631911343,  0.061199277021983]],

           [[ 0.042759461532687,  0.038563281355028],
            [ 0.038563281355028,  0.136138353942557]]]),
    (5, 'covar'): np.array([[[ 0.016716270750336, -0.002912143075387],
            [-0.002912143075387,  0.048058573349518]],

           [[ 0.162280075685762,  0.056056904861885],
            [ 0.056056904861885,  0.123029790628176]]]),
    (6, 'covar'): np.array([[[ 0.008867550173445,  0.005830414294608],
            [ 0.005830414294608,  0.004214767113419]],

           [[ 0.128202602012536,  0.102774200195474],
            [ 0.102774200195474,  0.103174267985407]],

           [[ 0.040911683088027,  0.017665837401128],
            [ 0.017665837401128,  0.013744306762296]],

           [[ 0.007956756372728,  0.01166041211521 ],
            [ 0.01166041211521 ,  0.030148938891721]],

           [[ 0.096468861178697,  0.036857159884246],
            [ 0.036857159884246,  0.016938035737711]],

           [[ 0.112556609450265, -0.027450040449295],
            [-0.027450040449295,  0.108044495426867]],

           [[ 0.008129216729562,  0.026162239500016],
            [ 0.026162239500016,  0.163188167512441]]]),
    (0, 'dNdm'): np.array([  4.192577862669580e+00,   2.041560039425720e-01,
             5.211356980204467e-01,   1.133059580454155e+03]),
    (1, 'dNdm'): np.array([  3.969155875747644e+00,   2.460106047909254e-01,
             7.649675390577662e-01,   1.594000900095526e+03]),
    (2, 'dNdm'): np.array([ -2.75804468990212 ,  84.684286895340932]),
    (3, 'dNdm'): np.array([   5.366276446077002,    0.931168472808592,    1.362372397828176,
            159.580421075961794]),
    (4, 'dNdm'): np.array([  -0.415601564925459,  125.965707251899474]),
    (5, 'dNdm'): np.array([  -2.199904276713916,  206.28117629545153 ]),
    (6, 'dNdm'): np.array([  8.188847496561811e-01,  -4.829571612433957e-01,
             2.953829284553960e-01,   1.620279479977582e+04])
    }

    # ####### Load paramters - method 2. Rather than hardcoding in the parameters,
    # we could also import them from a file ######
    # def generate_XD_model_dictionary(tag1="glim24", tag2="", K_i = [2,2,2,2,3,2,7], dNdm_type = [1, 1, 0, 1, 0, 0, 1]):
    #     # Create empty dictionary
    #     params = {}

    #     # Adding dNdm parameters for each class
    #     for i in range(7):
    #         if dNdm_type[i] == 0:
    #             dNdm_params =np.loadtxt(("%d-fit-pow-"+tag1)%i)
    #         else:
    #             dNdm_params =np.loadtxt(("%d-fit-broken-"+tag1)%i)
    #         params[(i, "dNdm")] = dNdm_params

    #     # Adding GMM parameters for each class
    #     for i in range(7):
    #         amp, mean, covar = load_params_XD(i,K_i[i],tag0="fit",tag1=tag1,tag2=tag2)
    #         params[(i,"amp")] = amp
    #         params[(i,"mean")] = mean
    #         params[(i,"covar")] = covar

    #     return params

    # def load_params_XD(i,K,tag0="fit",tag1="glim24",tag2=""):
    #     fname = ("%d-params-"+tag0+"-amps-"+tag1+"-K%d"+tag2+".npy") %(i, K)
    #     amp = np.load(fname)
    #     fname = ("%d-params-"+tag0+"-means-"+tag1+"-K%d"+tag2+".npy") %(i, K)
    #     mean= np.load(fname)
    #     fname = ("%d-params-"+tag0+"-covars-"+tag1+"-K%d"+tag2+".npy") %(i, K)
    #     covar  = np.load(fname)
    #     return amp, mean, covar

    # params = generate_XD_model_dictionary()


    ####### Load variables. #######
    # Flux
    #ADM allow analyses for both the DR3 and DR4+ Data Model
    if 'DECAM_FLUX' in objs.dtype.names:
        gflux = objs['DECAM_FLUX'][:][:,1]/objs['DECAM_MW_TRANSMISSION'][:][:,1]
        rflux = objs['DECAM_FLUX'][:][:,2]/objs['DECAM_MW_TRANSMISSION'][:][:,2]
        zflux = objs['DECAM_FLUX'][:][:,4]/objs['DECAM_MW_TRANSMISSION'][:][:,4]
    else:
        gflux = objs['FLUX_G'] / objs['MW_TRANSMISSION_G']
        rflux = objs['FLUX_R'] / objs['MW_TRANSMISSION_R']
        zflux = objs['FLUX_Z'] / objs['MW_TRANSMISSION_Z']
    # mags
    #ADM added explicit capture of runtime warnings for zero and negative fluxes
    with np.errstate(invalid='ignore',divide='ignore'):
        g = (22.5 - 2.5*np.log10(gflux))
        r = (22.5 - 2.5*np.log10(rflux))
        z = (22.5 - 2.5*np.log10(zflux))
        #ADM allow analyses for both the DR3 and DR4+ Data Model
        # Inver variance
        if 'DECAM_FLUX' in objs.dtype.names:
            givar = objs['DECAM_FLUX_IVAR'][:][:,1]
            rivar = objs['DECAM_FLUX_IVAR'][:][:,2]
            zivar = objs['DECAM_FLUX_IVAR'][:][:,4]
        else:
            givar = objs['FLUX_IVAR_G']
            rivar = objs['FLUX_IVAR_R']
            zivar = objs['FLUX_IVAR_Z']
        # Color
        rz = (r-z); gr = (g-r)

        ####### Reaonsable quaity cut. #######
        iflux_positive = (gflux>0)&(rflux>0)&(zflux>0)
        ireasonable_color = (gr>-0.5) & (gr<2.5) & (rz>-0.5) &(rz<2.7) & (g<gmax) & (g>gmin)
        thres = 2
        igrz_SN2 =  ((gflux*np.sqrt(givar))>thres)&((rflux*np.sqrt(rivar))>thres)&((zflux*np.sqrt(zivar))>thres)
        # Combination of above cuts.
        ireasonable = iflux_positive & ireasonable_color & igrz_SN2

        ####### A rough cut #######
        irough = (gr<1.3) & np.logical_or(gr<(rz+0.3) ,gr<0.3)

        ####### Objects for which FoM to be calculated. #######
        ibool = ireasonable & irough

    ######## Compute FoM values for objects that pass the cuts. #######
    # Place holder for FoM
    FoM = np.zeros(ibool.size, dtype=np.float)

    # Select subset of objects.
    mag = g[ibool]
    flux = gflux[ibool]
    gr = gr[ibool]
    rz = rz[ibool]

    # Compute the global error noise corresponding to each objects.
    const = 2.5/(5*np.log(10))
    gvar = (const * 10**(0.4*(mag-glim)))**2
    rvar = (const * 10**(0.4*(mag-gr_ref-rlim)))**2
    zvar = (const * 10**(0.4*(mag-gr_ref-rz_ref-zlim)))**2

    # Calculate the densities.
    # Helper function 1.
    def GMM_vectorized(gr, rz, amps, means, covars, gvar, rvar, zvar):
        """
        Color-color density

        Params
        ------
        gvar, rvar, zvar: Pre-computed errors based on individual grz values scaled from 5-sigma detection limits.
        """
        # Place holder for return array.
        density = np.zeros(gr.size,dtype=np.float)

        # Compute
        for i in range(amps.size):
            # Calculating Sigma+Error
            C11 = covars[i][0,0]+gvar+rvar
            C12 = covars[i][0,1]+rvar
            C22 = covars[i][1,1]+rvar+zvar

            # Compute the determinant
            detC = C11*C22-C12**2

            # Compute variables
            x11 = (gr-means[i][0])**2
            x12 = (gr-means[i][0])*(rz-means[i][1])
            x22 = (rz-means[i][1])**2

            # Calculating the exponetial
            EXP = np.exp(-(C22*x11-2*C12*x12+C11*x22)/(2.*detC+1e-12))

            density += amps[i]*EXP/(2*np.pi*np.sqrt(detC)+1e-12)

        return density


    # Helper function 2.
    def dNdm(params, flux):
        num_params = params.shape[0]
        if num_params == 2:
            return pow_law(params, flux)
        elif num_params == 4:
            return broken_pow_law(params, flux)

    # Helper function 3.
    def pow_law(params, flux):
        A = params[1]
        alpha = params[0]
        return A* flux**alpha

    # Helper function 4.
    def broken_pow_law(params, flux):
        alpha = params[0]
        beta = params[1]
        fs = params[2]
        phi = params[3]
        return phi/((flux/fs)**alpha+(flux/fs)**beta + 1e-12)

    FoM_num = np.zeros_like(gr)
    FoM_denom = np.zeros_like(gr)
    for i in range(7): # number of classes.
        n_i = GMM_vectorized(gr,rz, params[i, "amp"], params[i, "mean"],params[i, "covar"], gvar, rvar, zvar)  * dNdm(params[(i,"dNdm")], flux)
        FoM_num += f_i[i]*n_i
        FoM_denom += n_i

    FoM[ibool] = FoM_num/(FoM_denom+reg_r+1e-12) # For proper broadcasting.

    # XD-selection
    iXD = FoM>last_FoM

    return iXD, FoM


def isELG_randomforest( pcut=None, gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None, primary=None, training='spectro'):
    """Target Definition of ELG using a random forest returning a boolean array.

    Args:
        gflux, rflux, zflux, w1flux, w2flux: array_like
            The flux in nano-maggies of g, r, z, W1, and W2 bands.
                
        primary: array_like or None
            If given, the BRICK_PRIMARY column of the catalogue.

    Returns:
        mask : array_like. True if and only the object is a ELG
            target.

    Three RF
    - Training with spectro redshift (VIPERS and DEEP2)  :   rf_model_dr3_elg.npz
    - Training with photo z HSC : rf_model_dr3_elg_HSC.npz
    - Training with photo z HSC and depth=15 and max leaves = 2000 : rf_model_dr3_elg_HSC_V2.npz

            
    """
    #----- ELG
    if primary is None:
        primary = np.ones_like(gflux, dtype='?')

    # build variables for random forest
    nfeatures=11 # number of variables in random forest
    nbEntries=rflux.size
    colors, g, r, DECaLSOK = _getColors(nbEntries, nfeatures, gflux, rflux, zflux, w1flux, w2flux)

    #Preselection to speed up the process, store the indexes
    rMax = 23.5  # r<23.5
    gMax = 23.8  # g<23.8 proxy of OII flux
    
    preSelection = np.where( (r<rMax) & (g<gMax) & DECaLSOK )
    colorsCopy = colors.copy()
    colorsReduced = colorsCopy[preSelection]
    colorsIndex =  np.arange(0,nbEntries,dtype=np.int64)
    colorsReducedIndex =  colorsIndex[preSelection]

    #Path to random forest files
    pathToRF = resource_filename('desitarget', "sandbox/data")
 
    # Compute random forest probability
    from desitarget.myRF import myRF
    prob = np.zeros(nbEntries)

    if (colorsReducedIndex.any()) :
        if (training == 'spectro') :
            # Training with VIPERS and DEEP2 Fileds 2,3,4
            print (' === Trained with DEEP2 and VIPERS with spectro z == ')
            fileName = pathToRF + '/rf_model_dr3_elg.npz'
            rf = myRF(colorsReduced,pathToRF,numberOfTrees=200,version=1)
        elif  (training == 'photo') :  
            # Training with HSC with photometric redshifts
            # pathToRF = os.environ['DESITARGET']
            pathToRF = '.'
            print (' === Trained with HSC with photo z, you need locally /global/project/projectdirs/desi/target/RF_files/rf_model_dr3_elg_HSC_V2.npz nersc file ')
#            fileName = pathToRF + '/rf_model_dr3_elg_HSC.npz' 
            fileName = pathToRF + '/rf_model_dr3_elg_HSC_V2.npz' 
            rf = myRF(colorsReduced,pathToRF,numberOfTrees=500,version=2)
        
        rf.loadForest(fileName)
        objects_rf = rf.predict_proba()
        # add random forest probability to preselected objects
        j=0
        for i in colorsReducedIndex :
            prob[i]=objects_rf[j]
            j += 1

    #define pcut
    #pcut = 0.98

    elg = primary.copy()
    elg &= r<rMax
    elg &= g<gMax
    elg &= DECaLSOK


    if nbEntries==1 : # for call of a single object
        elg &= prob[0]>pcut
    else :
        elg &= prob>pcut

    return elg, prob

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

#    DECaLSOK = (g>0.) & (r>0.) & (z>0.) & (W1>0.) & (W2>0.)
    DECaLSOK = (g>0.) & (r>0.) & (z>0.) & ((W1>0.) | (W2>0.))

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

    return colors, g, r, DECaLSOK

def apply_sandbox_cuts(objects,FoMthresh=None, MethodELG='XD'):
    """Perform target selection on objects, returning target mask arrays

    Args:
        objects: numpy structured array with UPPERCASE columns needed for
            target selection, OR a string tractor/sweep filename
        FoMthresh: If this is passed, then run apply_XD_globalerror and
            return the Figure of Merits calculated for the ELGs in a file
            "FoM.fits" in the current working directory.
        MethodELG: Three methods available for ELGs
            XD: Extreme deconvolution
            RF_spectro: Random Forest trained with spectro z (VIPERS and DEEP2)
            RF_photo: Random Forest trained with photo z (HSC)
            
    Returns:
        (desi_target, bgs_target, mws_target) where each element is
        an ndarray of target selection bitmask flags for each object
        If FoMthresh is passed
        where FoM are the Figure of Merit values calculated by apply_XD_globalerror

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
    
    gflux_ivar = objects['FLUX_IVAR_G']
    rflux_ivar = objects['FLUX_IVAR_R']
    zflux_ivar = objects['FLUX_IVAR_Z']

    gflux_snr = objects['FLUX_G'] * np.sqrt(objects['FLUX_IVAR_G'])
    rflux_snr = objects['FLUX_R'] * np.sqrt(objects['FLUX_IVAR_R'])
    zflux_snr = objects['FLUX_Z'] * np.sqrt(objects['FLUX_IVAR_Z'])

    w1flux_snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])

    #- DR1 has targets off the edge of the brick; trim to just this brick
    try:
        primary = objects['BRICK_PRIMARY']
    except (KeyError, ValueError):
        if _is_row(objects):
            primary = True
        else:
            primary = np.ones_like(objects, dtype=bool)

    lrg = isLRG_2016v3(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux,
                       gflux_ivar=gflux_ivar,
                       rflux_snr=rflux_snr,
                       zflux_snr=zflux_snr,
                       w1flux_snr=w1flux_snr,
                       primary=primary)

    if FoMthresh is not None:
        if (MethodELG=='XD') :
            elg, FoM = apply_XD_globalerror(objects, FoMthresh, glim=23.8, rlim=23.4, zlim=22.4, gr_ref=0.5,
                       rz_ref=0.5,reg_r=1e-4/(0.025**2 * 0.05),f_i=[1., 1., 0., 0.25, 0., 0.25, 0.],
                       gmin = 21., gmax = 24.)
        elif (MethodELG=='RF_photo') :
            elg, FoM = isELG_randomforest(pcut=abs(FoMthresh), primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                                 w1flux=w1flux, w2flux=w2flux, training='photo')    
        elif (MethodELG=='RF_spectro') :
             elg, FoM = isELG_randomforest(pcut=abs(FoMthresh), primary=primary, zflux=zflux, rflux=rflux, gflux=gflux,
                                 w1flux=w1flux, w2flux=w2flux, training='spectro')    

    #- construct the targetflag bits
    #- Currently our only cuts are DECam based (i.e. South)
    desi_target  = lrg * desi_mask.LRG_SOUTH
    #desi_target |= elg * desi_mask.ELG_SOUTH
    #desi_target |= qso * desi_mask.QSO_SOUTH

    desi_target |= lrg * desi_mask.LRG
    if FoMthresh is not None:
        desi_target |= elg * desi_mask.ELG
    #desi_target |= qso * desi_mask.QSO

    #desi_target |= fstd * desi_mask.STD_FSTAR

    bgs_target = np.zeros_like(desi_target)
    #bgs_target = bgs_bright * bgs_mask.BGS_BRIGHT
    #bgs_target |= bgs_bright * bgs_mask.BGS_BRIGHT_SOUTH
    #bgs_target |= bgs_faint * bgs_mask.BGS_FAINT
    #bgs_target |= bgs_faint * bgs_mask.BGS_FAINT_SOUTH

    #- nothing for MWS yet; will be GAIA-based
    #if isinstance(bgs_target, numbers.Integral):
    #    mws_target = 0
    #else:
    #    mws_target = np.zeros_like(bgs_target)
    mws_target = np.zeros_like(desi_target)

    #- Are any BGS or MWS bit set?  Tell desi_target too.
    desi_target |= (bgs_target != 0) * desi_mask.BGS_ANY
    desi_target |= (mws_target != 0) * desi_mask.MWS_ANY

    if FoMthresh is not None:
        keep = (desi_target != 0)
        write_fom_targets(objects[keep], FoM[keep], desi_target[keep], bgs_target[keep], mws_target[keep])

    return desi_target, bgs_target, mws_target
