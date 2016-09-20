"""
Creates one target table and one truth table out of a set of input files. Each
imput is assumed to be a valid target or truth table in its own right.
"""
from __future__ import (absolute_import, division)
#
import numpy as np
import fitsio
import os

import astropy.table
from   astropy.table import Table, Column

from .mock.combine_targets_truth import combine_targets_truth

############################################################
def combine_targets_truth_from_files(input_targets,
                          input_truth        = None,
                          output_dir         = './',
                          output_target_name = 'combined_targets.fits',
                          output_truth_name  = 'combined_truth.fits'):
    """
    Reads in multiple target and (optionaly) truth files and writes combined
    versions. Does NOT add the additional columns required to make an MTL.
    """
    if not os.path.exists(output_dir):
        raise Exception('Output path %s does not exist!'%(output_path))

    if not output_target_name.endswith('.fits'): output_target_name = output_target_name + '.fits'
    if not output_truth_name.endswith('.fits'):  output_truth_name  = output_truth_name + '.fits'

    output_target_path = os.path.join(output_dir,output_target_name)
    output_truth_path  = os.path.join(output_dir,output_truth_name)

    output_xref_path   = output_target_path.replace('.fits','.xref.fits')

    input_target_list = list(input_targets)
    input_truth_list  = list()
    ntarget           = len(input_target_list)
    if input_truth is not None:
        # Have supplied list of truth files by hand, sanity check this has the
        # right length,
        input_truth_list = list(input_truth)
        ntruth           = len(input_truth_list)

        if not ntruth == ntarget:
            raise Exception('Have %d target files but %d truth files'%(ntarget,ntruth))

    # Loop over the input target files to check files exist and generate input
    # truth names automatically if required.
    for input_target_path in input_target_list:
        if not os.path.exists(input_target_path):
            raise Exception('Input target file %s not found'%(input_target_path))
       
        input_target_dir, input_target_base = os.path.split(input_target_path)
        
        if input_truth is None:
            # Auto-generate input truth paths, assuming each truth file has the
            # same path as the target file and the file name differs only in
            # the replacement of 'truth' for 'targets'.
            input_truth_base = input_target_base.replace('targets','truth')
            input_truth_path = os.path.join(input_target_dir,input_truth_base)
            input_truth_list.append(input_truth_path)
    
            if not os.path.exists(input_truth_path):
                raise Exception('Input truth file %s not found'%(input_truth_path))

    # Read TARGET tables
    target_data = list()
    xref        = list()
    for input_target_path in input_target_list:
        print('Reading targets from: %s'%(input_target_path))
        #t = Table.read(input_target_path)
        t = fitsio.getdata(input_target_path)
        target_data.append(t)

        print(' -- read %d targets'%(len(t)))

        # Store information to cross-reference rows in final combined target
        # list with each input file. Stores the full path to the file,
        # eliminating any symlinks.
        xref.append((os.path.realpath(input_target_path), len(t)))
    
    # Read TRUTH tables
    truth_data = list()
    for input_truth_path in input_truth_list:
        print('Reading truths from: %s'%(input_truth_path))
        t = fitsio.getdata(input_trith_path)
        truth_data.append(t)
   
    # Combine tables
    master_target_table, master_truth_table, xref_tabel = combine_targets_truth(input_target_data, 
                                                                                input_truth_data, 
                                                                                xref=xref)

    # Write output
    print('Writing %d rows of targets to: %s'%(len(master_target_table),output_target_path))
    master_target_table.write(output_target_path,format='fits',overwrite=True)
    del(master_target_table)
    print('')
    print('Writing %d rows of truth to: %s'%(len(master_truth_table),output_truth_path))
    master_truth_table.write(output_truth_path,format='fits',overwrite=True)
    del(master_truth_table)
    print('')
    xref_table.write(output_xref_path,format='fits',overwrite=True)
    print('Wrote XREF table: %s'%(output_xref_path))
    print('Done!')

    return

############################################################
if __name__ == '__main__':
    """
    """
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('target_list',nargs='+')
    parser.add_argument('--truth_list',nargs='+',default=None)
    parser.add_argument('-o','--output_dir',   default='./')
    parser.add_argument('--output_target_name',default='combined_targets.fits')
    parser.add_argument('--output_truth_name', default='combined_truth.fits')

    args = parser.parse_args()
    combine_targets_truth(args.target_list,
                          input_truth        = args.truth_list,
                          output_dir         = args.output_dir,
                          output_target_name = args.output_target_name,
                          output_truth_name  = args.output_truth_name)
