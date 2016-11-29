from __future__ import (absolute_import, division, print_function)

import os
import importlib
import time
import glob
import numpy as np

from   copy import copy

import astropy.io.fits as astropy_fitsio
from   astropy.table import Table, Column
import astropy.table

from   desitarget.targetmask import desi_mask
from   desitarget.mock.io    import decode_rownum_filenum
import desitarget.mock.io as mockio
import desitarget.mock.selection as mockselect
from desitarget import obsconditions
from desitarget import mtl
import desispec.brick
from desispec.brick import Bricks
import desitarget.QA as targetQA
import yaml

def fluctuations_across_bricks(brick_info, target_names, decals_brick_info):
    """
    Generates number density fluctuations.


    Parameters:
    -----------

        decals_brick_info (string). file summarizing tile statistics Data Release 3 of DECaLS. 
        brick_info(Dictionary). Containts at least the following keys:
            DEPTH_G(float) : array of depth magnitudes in the G band.

    Return:
    ------
       fluctuations (dictionary) with keys 'FLUC+'depth, each one with values
       corresponding to a dictionary with keys ['ALL','LYA','MWS','BGS','QSO','ELG','LRG'].
       i.e. fluctuation[FLUC_DEPTH_G]['MWS'] holds the number density as a funtion 
        is a dictionary with keys corresponding to the different galaxy types.
    """
    fluctuation = {}
    
    depth_available = []
#    for k in brick_info.keys():        
    for k in ['GALDEPTH_R', 'EBV']:        
        if('DEPTH' in k or 'EBV' in k):
            depth_available.append(k)
            

    for depth in depth_available:        
        fluctuation['FLUC_'+depth] = {}
        for ttype in target_names:
            fluctuation['FLUC_'+depth][ttype] = targetQA.generate_fluctuations(decals_brick_info, ttype, depth, brick_info[depth])    
            print('Generated target fluctuation for type {} using {} as input for {} bricks'.format(ttype, depth, len(fluctuation['FLUC_'+depth][ttype])))
    return fluctuation

def depths_across_bricks(brick_info):
    """
    Generates a sample of magnitud dephts for a set of bricks.

    This model was built from the Data Release 3 of DECaLS.

    Parameters:
    -----------
        brick_info(Dictionary). Containts at least the following keys:
            RA (float): numpy array of RA positions
            DEC (float): numpy array of Dec positions

    Return:
    ------
        depths (dictionary). keys include
            'DEPTH_G', 'DEPTH_R', 'DEPTH_Z',
            'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z'.
            The values ofr each key ar numpy arrays (float) with size equal to 
            the input ra, dec arrays.
    """

    ra = brick_info['RA']
    dec = brick_info['DEC']

    n_to_generate = len(ra)
    #mean and std deviation of the difference between DEPTH and GALDEPTH in the DR3 data.
    differences = {}
    differences['DEPTH_G'] = [0.22263251, 0.059752077]
    differences['DEPTH_R'] = [0.26939404, 0.091162138]
    differences['DEPTH_Z'] = [0.34058815, 0.056099825]
    
    # (points, fractions) provide interpolation to the integrated probability distributions from DR3 data
    
    points = {}
    points['DEPTH_G'] = np.array([ 12.91721153,  18.95317841,  20.64332008,  23.78604698,  24.29093361,
                  24.4658947,   24.55436325,  24.61874771,  24.73129845,  24.94996071])
    points['DEPTH_R'] = np.array([ 12.91556168,  18.6766777,   20.29519463,  23.41814804,  23.85244179,
                  24.10131454,  24.23338318,  24.34066582,  24.53495026,  24.94865227])
    points['DEPTH_Z'] = np.array([ 13.09378147,  21.06531525,  22.42395782,  22.77471352,  22.96237755,
                  23.04913139,  23.43119431,  23.69817734,  24.1913662,   24.92163849])

    fractions = {}
    fractions['DEPTH_G'] = np.array([0.0, 0.01, 0.02, 0.08, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    fractions['DEPTH_R'] = np.array([0.0, 0.01, 0.02, 0.08, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    fractions['DEPTH_Z'] = np.array([0.0, 0.01, 0.03, 0.08, 0.2, 0.3, 0.7, 0.9, 0.99, 1.0])

    names = ['DEPTH_G', 'DEPTH_R', 'DEPTH_Z']
    depths = {}
    for name in names:
        fracs = np.random.random(n_to_generate)
        depths[name] = np.interp(fracs, fractions[name], points[name])

        depth_minus_galdepth = np.random.normal(
            loc=differences[name][0], 
            scale=differences[name][1], size=n_to_generate)
        depth_minus_galdepth[depth_minus_galdepth<0] = 0.0
        
        depths['GAL'+name] = depths[name] - depth_minus_galdepth
        print('Generated {} and GAL{} for {} bricks'.format(name, name, len(ra)))
    return depths

def extinction_across_bricks(brick_info, dust_dir):
    """
    Estimates E(B-V) across bricks.

    Args:
         brick_info : dictionary gathering brick information. It must have at least two keys 'RA' and 'DEC'.
         dust_dir : path where the E(B-V) maps are storesd
    """
    from desitarget.mock import sfdmap
    a = {}
    a['EBV'] = sfdmap.ebv(brick_info['RA'], brick_info['DEC'], mapdir=dust_dir)
    print('Generated extinction for {} bricks'.format(len(brick_info['RA'])))
    return a


def generate_brick_info(bounds=(0.0,359.99,-89.99,89.99)):
    """
    Generates brick dictionary in the ragion (min_ra, max_ra, min_dec, max_dec)
    
    """
    min_ra, max_ra, min_dec, max_dec = bounds

    B = Bricks()
    brick_info = {}
    brick_info['BRICKNAME'] = []
    brick_info['RA'] = []
    brick_info['DEC'] =  []
    brick_info['RA1'] =  []
    brick_info['RA2'] =  []
    brick_info['DEC1'] =  []
    brick_info['DEC2'] =   []
    brick_info['BRICKAREA'] =  [] 
    
    i_rows = np.where((B._edges_dec < max_dec ) & (B._edges_dec > min_dec))
    i_rows = i_rows[0]

    for i_row in i_rows:
        j_col_min = int((min_ra )/360 * B._ncol_per_row[i_row])
        j_col_max = int((max_ra )/360 * B._ncol_per_row[i_row])
        for j_col in range(j_col_min, j_col_max+1):
            brick_info['BRICKNAME'].append(B._brickname[i_row][j_col])
            
            brick_info['RA'].append(B._center_ra[i_row][j_col])
            brick_info['DEC'].append(B._center_dec[i_row])

            brick_info['RA1'].append(B._edges_ra[i_row][j_col])
            brick_info['DEC1'].append(B._edges_dec[i_row])
            
            brick_info['RA2'].append(B._edges_ra[i_row][j_col+1])
            brick_info['DEC2'].append(B._edges_dec[i_row+1])


            brick_area = (brick_info['RA2'][-1]- brick_info['RA1'][-1]) 
            brick_area *= (np.sin(brick_info['DEC2'][-1]*np.pi/180.) - np.sin(brick_info['DEC1'][-1]*np.pi/180.)) * 180 / np.pi
            brick_info['BRICKAREA'].append(brick_area)

    for k in brick_info.keys():
        brick_info[k] = np.array(brick_info[k])
    print('Generated basic brick info for {} bricks'.format(len(brick_info['BRICKNAME'])))
    return brick_info

            
############################################################
def targets_truth(params, output_dir):
    """
    Write

    Args:
        params: dict of source definitions.
        output_dir: location for intermediate mtl files.
    Returns:
        targets:    
        truth:      

    """

    truth_all       = list()

    target_mask_all = dict()

    source_defs = params['sources']

    #reads target info from DESIMODEL + changing all the keys to upper case
    filein = open(os.getenv('DESIMODEL')+'/data/targets/targets.dat')
    td = yaml.load(filein)
    target_desimodel = {}
    for t in td.keys():
        target_desimodel[t.upper()] = td[t]


    # compiles brick information
    if ('subset' in params.keys()) & (params['subset']['ra_dec_cut']==True):
        brick_info = generate_brick_info(bounds=(params['subset']['min_ra'],
                                                 params['subset']['max_ra'],
                                                 params['subset']['min_dec'],
                                                 params['subset']['max_dec']))
    else:
        brick_info = generate_brick_info()

    brick_info.update(extinction_across_bricks(brick_info, params['dust_dir'])) #add extinction
    brick_info.update(depths_across_bricks(brick_info)) #add depths
    brick_info.update(fluctuations_across_bricks(brick_info, list(params['sources'].keys()), params['decals_brick_info'])) # add number density fluctuations

    # appends DESIMODEL info into brick_info
    brick_info.update(target_desimodel)

    # prints info about what we will be loading
    mockio.print_all_mocks_info(params)

    # loads all the mocks
    source_data_all = mockio.load_all_mocks(params)

    print('Making target selection')
    # runs target selection on every mock
    for source_name in sorted(source_defs.keys()):
        target_name = params['sources'][source_name]['target_name'] #Target names
        truth_name = params['sources'][source_name]['truth_name'] #name for the truth file
        source_selection = params['sources'][source_name]['selection'] # criteria to make target selection
        source_dict = params['sources'][source_name] # dictionary with sources info
        source_data = source_data_all[source_name]  # data 

        print('target_name {} : type: {} select: {}'.format(target_name, source_name, source_selection))
        selection_function = source_selection + '_select'
        result = getattr(mockselect, selection_function.lower())(source_data, source_name, target_name, truth_name, brick_info = brick_info, 
                                                                 density_fluctuations = params['density_fluctuations'],
                                                                 **source_dict)
        target_mask_all[source_name] = result

        
    # consolidates all relevant arrays across mocks
    ra_total = np.empty(0)
    dec_total = np.empty(0)
    z_total = np.empty(0)
    desi_target_total = np.empty(0, dtype='i8')
    bgs_target_total = np.empty(0, dtype='i8')
    mws_target_total = np.empty(0, dtype='i8')
    true_type_total = np.empty(0, dtype='S10')
    source_type_total = np.empty(0, dtype='S10')
    obsconditions_total = np.empty(0, dtype='uint16')


    print('Collects information across mock files')
    for source_name in sorted(source_defs.keys()):
        target_name = params['sources'][source_name]['target_name']
        truth_name = params['sources'][source_name]['truth_name']
        source_data = source_data_all[source_name]
        target_mask = target_mask_all[source_name]

        ii = target_mask >-1 #only targets that passed cuts

        #define all flags
        desi_target = 0 * target_mask[ii]
        bgs_target = 0 * target_mask[ii] 
        mws_target = 0 * target_mask[ii]
        if target_name in ['ELG', 'LRG', 'QSO', 'STD_FSTAR', 'SKY']:
            desi_target = target_mask[ii]
        if target_name in ['BGS']:
            bgs_target = target_mask[ii]
        if target_name in ['MWS_MAIN', 'MWS_WD']:
            mws_target = target_mask[ii]


        # define names that go into Truth
        n = len(source_data['RA'][ii])
        if source_name not in ['STD_FSTAR', 'SKY']:
            true_type_map = {
                'STD_FSTAR': 'STAR',
                'ELG': 'GALAXY',
                'LRG': 'GALAXY',
                'BGS': 'GALAXY',
                'QSO': 'QSO',
                'STD_FSTAR': 'STAR',
                'MWS_MAIN': 'STAR',
                'MWS_WD': 'STAR',
                'SKY': 'SKY',
            }
            source_type = np.zeros(n, dtype='S10')
            source_type[:] = target_name
            true_type = np.zeros(n, dtype='S10')
            true_type[:] = true_type_map[truth_name]

                
        #define obsconditions
        source_obsconditions = np.ones(n,dtype='uint16')
        if target_name in ['LRG', 'QSO']:
            source_obsconditions[:] = obsconditions.DARK
        if target_name in ['ELG']:
            source_obsconditions[:] = obsconditions.DARK|obsconditions.GRAY
        if target_name in ['BGS']:
            source_obsconditions[:] = obsconditions.BRIGHT
        if target_name in ['MWS_MAIN', 'MWS_WD']:
            source_obsconditions[:] = obsconditions.BRIGHT
        if target_name in ['STD_FSTAR', 'SKY']:
            source_obsconditions[:] = obsconditions.DARK|obsconditions.GRAY|obsconditions.BRIGHT 

        #append to the arrays that will go into Targets
        if source_name in ['STD_FSTAR']:
            ra_stars = source_data['RA'][ii].copy()
            dec_stars = source_data['DEC'][ii].copy()
            desi_target_stars = desi_target.copy()
            bgs_target_stars = bgs_target.copy()
            mws_target_stars = mws_target.copy()
            obsconditions_stars = source_obsconditions.copy()
        if source_name in ['SKY']:
            ra_sky = source_data['RA'][ii].copy()
            dec_sky = source_data['DEC'][ii].copy()
            desi_target_sky = desi_target.copy()
            bgs_target_sky = bgs_target.copy()
            mws_target_sky = mws_target.copy()
            obsconditions_sky = source_obsconditions.copy()
        if source_name not in ['SKY', 'STD_FSTAR']:
            ra_total = np.append(ra_total, source_data['RA'][ii])
            dec_total = np.append(dec_total, source_data['DEC'][ii])
            z_total = np.append(z_total, source_data['Z'][ii])
            desi_target_total = np.append(desi_target_total, desi_target)
            bgs_target_total = np.append(bgs_target_total, bgs_target)
            mws_target_total = np.append(mws_target_total, mws_target)
            true_type_total = np.append(true_type_total, true_type)
            source_type_total = np.append(source_type_total, source_type)
            obsconditions_total = np.append(obsconditions_total, source_obsconditions)

            

        print('source {} target {} truth {}: selected {} out of {}'.format(
                source_name, target_name, truth_name, len(source_data['RA'][ii]), len(source_data['RA'])))


    

    # create unique IDs, subpriorities and bricknames across all mock files
    n_target = len(ra_total)     
    n_star = 0
    n_sky = 0
    n  = n_target    
    if 'STD_FSTAR' in source_defs.keys():
        n_star = len(ra_stars)
        n += n_star
    if 'SKY' in source_defs.keys():
        n_sky = len(ra_sky)
        n += n_sky
    print('Great total of {} targets {} stdstars {} sky pos'.format(n_target, n_star, n_sky))
    targetid = np.random.randint(2**62, size=n)

    # write to disk
    if 'STD_FSTAR' in source_defs.keys():
        subprior = np.random.uniform(0., 1., size=n_star)
        #write the Std Stars to disk
        print('Started writing StdStars file')
        stars_filename = os.path.join(output_dir, 'stdstars.fits')
        stars = Table()
        stars['TARGETID'] = targetid[n_target:n_target+n_star]
        stars['RA'] = ra_stars
        stars['DEC'] = dec_stars
        stars['DESI_TARGET'] = desi_target_stars
        stars['BGS_TARGET'] = bgs_target_stars
        stars['MWS_TARGET'] = mws_target_stars
        stars['SUBPRIORITY'] = subprior
        stars['OBSCONDITIONS'] = obsconditions_stars
        # if ('subset' in params.keys()) & (params['subset']['ra_dec_cut']==True):
        #     stars = stars[ii_stars]
        #     print('subsetting in std_stars data done')
        brickname = desispec.brick.brickname(stars['RA'], stars['DEC'])
        stars['BRICKNAME'] = brickname
        stars.write(stars_filename, overwrite=True)
        print('Finished writing stdstars file')

    if 'SKY' in source_defs.keys():
        subprior = np.random.uniform(0., 1., size=n_sky)
        #write the Std Stars to disk
        print('Started writing sky to file')
        sky_filename = os.path.join(output_dir, 'sky.fits')
        sky = Table()
        sky['TARGETID'] = targetid[n_target+n_star:n_target+n_star+n_sky]
        sky['RA'] = ra_sky
        sky['DEC'] = dec_sky
        sky['DESI_TARGET'] = desi_target_sky
        sky['BGS_TARGET'] = bgs_target_sky
        sky['MWS_TARGET'] = mws_target_sky
        sky['SUBPRIORITY'] = subprior
        sky['OBSCONDITIONS'] = obsconditions_sky
        brickname = desispec.brick.brickname(sky['RA'], sky['DEC'])
        sky['BRICKNAME'] = brickname
        sky.write(sky_filename, overwrite=True)
        print('Finished writing sky file')

    if n_target > 0:
        subprior = np.random.uniform(0., 1., size=n_target)
        # write the Targets to disk
        print('Started writing Targets file')
        targets_filename = os.path.join(output_dir, 'targets.fits')
        targets = Table()
        targets['TARGETID'] = targetid[0:n_target]
        targets['RA'] = ra_total
        targets['DEC'] = dec_total
        targets['DESI_TARGET'] = desi_target_total
        targets['BGS_TARGET'] = bgs_target_total
        targets['MWS_TARGET'] = mws_target_total
        targets['SUBPRIORITY'] = subprior
        targets['OBSCONDITIONS'] = obsconditions_total
        brickname = desispec.brick.brickname(targets['RA'], targets['DEC'])
        targets['BRICKNAME'] = brickname
        targets.write(targets_filename, overwrite=True)
        print('Finished writing Targets file')

        # started computing mtl file for the targets
        print('Started computing the MTL file')
        mtl_table = mtl.make_mtl(targets)        
        # writing the MTL file to disk
        print('Started writing the first MTL file')
        mtl_filename = os.path.join(output_dir, 'mtl.fits')
        mtl_table.write(mtl_filename, overwrite=True)
        print('Finished writing mtl file')

        # write the Truth to disk
        print('Started writing Truth file')
        truth_filename = os.path.join(output_dir, 'truth.fits')
        truth = Table()
        truth['TARGETID'] = targetid[0:n_target]
        truth['RA'] = ra_total
        truth['DEC'] = dec_total
        truth['TRUEZ'] = z_total
        truth['TRUETYPE'] = true_type_total
        truth['SOURCETYPE'] = source_type_total
        truth['BRICKNAME'] = brickname
        truth.write(truth_filename, overwrite=True)
        print('Finished writing Truth file')


        
        

