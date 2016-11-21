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


def fluctuations_across_bricks(brick_info, decals_brick_info):
    depth_available = ['DEPTH_G']

    for depth in depth_available:        
        brick_info['FLUC_'+depth] = {}
        print('KEYS'.format(brick_info['DENSITY'].keys()))
        for target_type in brick_info['DENSITY'].keys():
            if isinstance(target_type, bytes):
                ttype = target_type.decode()
            else:
                ttype = target_type
            print(' depth {} target {}'.format(depth, ttype))
            brick_info['FLUC_'+depth][ttype]  = targetQA.generate_fluctuations(decals_brick_info, ttype, depth, brick_info[depth])    


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
        brick_info[name] = np.interp(fracs, fractions[name], points[name])

        depth_minus_galdepth = np.random.normal(
            loc=differences[name][0], 
            scale=differences[name][1], size=n_to_generate)
        depth_minus_galdepth[depth_minus_galdepth<0] = 0.0
        
        brick_info['GAL'+name] = brick_info[name] - depth_minus_galdepth
    

def extinction_across_bricks(brick_info, dust_dir):
    """
    Estimates E(B-V) across bricks.

    Args:
         brick_info : dictionary gathering brick information. It must have at least two keys 'RA' and 'DEC'.
         dust_dir : path where the E(B-V) maps are storesd
    """
    from desitarget.mock import sfdmap
    brick_info['EBV'] = sfdmap.ebv(brick_info['RA'], brick_info['DEC'], mapdir=dust_dir)
    return
    

def gather_brick_info(ra, dec, target_names):
    """
    Gathers information about all the targets on the scale of a brick.
    """
    B = Bricks()

    brick_info = {}
    names = list(set(target_names))
    tnames = []
    for i in range(len(names)):
        if isinstance(names[i], bytes):
            tnames.append(names[i].decode())
        else:
            tnames.append(names[i])

    n_names = len(tnames)
    print('total of {} target types: {}'.format(len(tnames), tnames))


    # compute brick information for each target
    irow = ((dec+90.0+B._bricksize/2)/B._bricksize).astype(int)
    jcol = (ra/360 * B._ncol_per_row[irow]).astype(int)

    xra =  np.array([B._center_ra[i][j] for i,j in zip(irow, jcol)])
    xdec = B._center_dec[irow]

    ra1 =  np.array([B._edges_ra[i][j] for i,j in zip(irow, jcol)])
    dec1 = B._edges_dec[irow]

    ra2 =  np.array([B._edges_ra[i][j+1] for i,j in zip(irow, jcol)])
    dec2 = B._edges_dec[irow+1]
    
    names = list()
    for i in range(len(ra)):
        ncol = B._ncol_per_row[irow[i]]
        j = int(ra[i]/360 * ncol)
        names.append(B._brickname[irow[i]][j])
    names  = np.array(names)

    # summarize brick info
    brick_names = np.array(list(set(names)))
    n_brick = len(brick_names)
    brick_xra = np.ones(n_brick)
    brick_ra1 = np.ones(n_brick)
    brick_ra2 = np.ones(n_brick)
    brick_xdec = np.ones(n_brick)
    brick_dec1 = np.ones(n_brick)
    brick_dec2 = np.ones(n_brick)
    brick_area = np.ones(n_brick)
    densities = np.ones([n_brick, n_names])
    for i in range(len(brick_names)):
        ii = (brick_names[i] == names)        
 #       print('{} in brick'.format(np.count_nonzero(ii)))

        brick_xra[i] = xra[ii][0]
        brick_xdec[i] = xdec[ii][0]

        brick_ra1[i] = ra1[ii][0]
        brick_ra2[i] = ra2[ii][0]

        brick_dec1[i] = dec1[ii][0]
        brick_dec2[i] = dec2[ii][0]

        brick_area[i] = (brick_ra2[i] - brick_ra1[i]) 
        brick_area[i] *= (np.sin(brick_dec2[i]*np.pi/180.) - np.sin(brick_dec1[i]*np.pi/180.)) * 180 / np.pi

#        print('center ra, dec {} {}'.format(brick_xra[i], brick_xdec[i]))
#        print('center ra1 ra2 {} {}'.format(brick_ra1[i], brick_ra2[i]))
#        print('center dec1 dec2 {} {}'.format(brick_dec1[i], brick_dec2[i]))
#        print(' brick area {}'.format(brick_area[i]))
        if(brick_area[i]<0.0):
            exit(1)
        for j in range(n_names):
            jj = (target_names == tnames[j])        
            densities[i,j] = np.count_nonzero(ii & jj)/brick_area[i]
#            print('name {} in names {}. density: {}'.format(tnames[j], tnames, densities[i,j]))

    brick_info['BRICKNAMES'] = brick_names
    brick_info['RA'] = brick_xra
    brick_info['DEC'] = brick_xdec
    brick_info['RA1'] = brick_ra1
    brick_info['RA2'] = brick_ra2
    brick_info['DEC1'] = brick_dec1
    brick_info['DEC2'] = brick_dec2
    brick_info['BRICKAREA'] = brick_area
    brick_info['DENSITY'] = {}
    for j in range(n_names):
        brick_info['DENSITY'][tnames[j]] = densities[:,j]
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
    print  ( params['dust_dir'])
    truth_all       = list()
    source_data_all = dict()
    target_mask_all = dict()

    # prints info about what we will be loading
    source_defs = params['sources']
    print('The following populations and paths are specified:')
    for source_name in sorted(source_defs.keys()):
        source_format = params['sources'][source_name]['format']
        source_path = params['sources'][source_name]['root_mock_dir']
        target_name = params['sources'][source_name]['target_name']
        print('type: {}\n format: {} \n path: {}'.format(source_name, source_format, source_path))

    # load all the mocks
    for source_name in sorted(source_defs.keys()):
        source_format = params['sources'][source_name]['format']
        source_path = params['sources'][source_name]['root_mock_dir']
        source_dict = params['sources'][source_name]
        target_name = params['sources'][source_name]['target_name']

        print('type: {} format: {}'.format(source_name, source_format))
        function = 'read_'+source_format
        if 'mock_name' in source_dict.keys():
            mock_name = source_dict['mock_name']
        else:
            mock_name = None
        result = getattr(mockio, function)(source_path, target_name, mock_name=mock_name)

        if ('subset' in params.keys()) & (params['subset']['ra_dec_cut']==True):
            print('Trimming {} to RA,dec subselection'.format(source_name))
            ii  = (result['RA']  >= params['subset']['min_ra']) & \
                  (result['RA']  <= params['subset']['max_ra']) & \
                  (result['DEC'] >= params['subset']['min_dec']) & \
                  (result['DEC'] <= params['subset']['max_dec'])

            #- Trim RA,DEC,Z, ... columns to subselection
            #- Different types of mocks have different metadata, so assume
            #- that any ndarray of the same length as number of targets should
            #- be trimmed.
            ntargets = len(result['RA'])
            for key in result:
                if isinstance(result[key], np.ndarray) and len(result[key]) == ntargets:
                    result[key] = result[key][ii]

            #- Add min/max ra/dec to source_dict for use in density estimates
            source_dict.update(params['subset'])

        source_data_all[source_name] = result

    print('loaded {} mock sources'.format(len(source_data_all)))

    print('Making target selection')
    # runs target selection on every mock
    for source_name in sorted(source_defs.keys()):
        target_name = params['sources'][source_name]['target_name']
        source_selection = params['sources'][source_name]['selection']
        source_dict = params['sources'][source_name]
        source_data = source_data_all[source_name]

        print('type: {} select: {}'.format(source_name, source_selection))
        selection_function = source_selection + '_select'
        result = getattr(mockselect, selection_function.lower())(source_data, target_name, **source_dict)
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
        if target_name not in ['STD_FSTAR', 'SKY']:
            true_type_map = {
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
            true_type[:] = true_type_map[target_name]

                
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
        if target_name in ['STD_FSTAR']:
            ra_stars = source_data['RA'][ii].copy()
            dec_stars = source_data['DEC'][ii].copy()
            desi_target_stars = desi_target.copy()
            bgs_target_stars = bgs_target.copy()
            mws_target_stars = mws_target.copy()
            obsconditions_stars = source_obsconditions.copy()
        if target_name in ['SKY']:
            ra_sky = source_data['RA'][ii].copy()
            dec_sky = source_data['DEC'][ii].copy()
            desi_target_sky = desi_target.copy()
            bgs_target_sky = bgs_target.copy()
            mws_target_sky = mws_target.copy()
            obsconditions_sky = source_obsconditions.copy()
        if target_name not in ['SKY', 'STD_FSTAR']:
            ra_total = np.append(ra_total, source_data['RA'][ii])
            dec_total = np.append(dec_total, source_data['DEC'][ii])
            z_total = np.append(z_total, source_data['Z'][ii])
            desi_target_total = np.append(desi_target_total, desi_target)
            bgs_target_total = np.append(bgs_target_total, bgs_target)
            mws_target_total = np.append(mws_target_total, mws_target)
            true_type_total = np.append(true_type_total, true_type)
            source_type_total = np.append(source_type_total, source_type)
            obsconditions_total = np.append(obsconditions_total, source_obsconditions)

            

        print('{} {}: selected {} out of {}'.format(source_name, target_name, len(source_data['RA'][ii]), len(source_data['RA'])))


    #summarizes target density information across bricks
    brick_info = gather_brick_info(ra_total, dec_total, source_type_total)

    #computes magnitude depths across bricks
    depths_across_bricks(brick_info)

    #computes extinction across bricks
    extinction_across_bricks(brick_info, params['dust_dir'])    

    #computes density fluctuations across bricks
    fluctuations_across_bricks(brick_info, params['decals_brick_info'])

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


        
        

