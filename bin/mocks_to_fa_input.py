#!/usr/bin/env python
"""
Makes MTL and truth tables suitable for fiberassign from mock catalogues.

Process is as follows:
    - Individual mocks are read with desitarget.mocks.xxx routines. Parameters
      for these routines come from the fa_run.yaml file 'sources' key.
    - For each mock, one intermediate MTL-like file is created.
    - The intermediate files are concatenated (in memory).
    - The concatenated table is fed to desitarget.mtl to create the final MTL
      and truth table that can be passed to fiberassign.
    

"""
import os
import importlib
import time
import yaml

import astropy.io.fits as fitsio

from   desitarget import mtl
import desitarget.mock
import desitarget.mock.fiberassign as fiberassign

############################################################
def mocks_to_fa_input(fa_run_dir,yaml_name='fa_run.yaml', reset=False):
    """
    Args:
       fa_run_dir   Top level directory for FA run.
       yaml_name    [optional] alternative name for configuration file 
                    (must be found in fa_run_dir)
       reset        If True, remake all cached files on disk.
    """
    # TODO defaults for parameters like target_mtl_name
    print('')

    yaml_path = os.path.join(fa_run_dir,yaml_name)
    if not os.path.exists(yaml_path):
        raise Exception('No config file {} in run directory {}'.format(yaml_name,fa_run_dir))

    print('Creating MTL file under:   {}'.format(fa_run_dir))
    print('According to config in:    {}'.format(yaml_path))
    print('')

    # Work in the run directory
    original_dir = os.getcwd()
    try:
        os.chdir(fa_run_dir)
        
        # Read parameters from yaml
        with open(yaml_name,'r') as pfile:
            params = yaml.load(pfile)
        assert(os.path.exists(params['target_dir']))
        assert(params.has_key('sources'))

        # Construct intermediate MTLs for each soruce
        targets_all, truth_all, sources_all = fiberassign.make_mtls_for_sources(params['sources'],
                                                                                params['target_dir'],
                                                                                reset=reset)
        # Combine the input tables
        combined = fiberassign.combine_targets_truth(targets_all,truth_all, input_sources=sources_all)

        print('{:d} targets in combined list'.format(len(combined['targets'])))
        
        # Produce the final MTL and truth; trim = False is important!
        combined_mtl, combined_mtl_truth = mtl.make_mtl(combined['targets'],
                                                        truth = combined['truth'],
                                                        trim  = False)
        print('{:d} rows in combined MTL'.format(len(combined_mtl)))

        # Write MTL targets
        output_mtl_path   = os.path.join(params['target_dir'],params['target_mtl_name'])
        print('Writing output to {}'.format(os.path.abspath(output_mtl_path)))
        fitsio.writeto(output_mtl_path, combined_mtl.as_array(),clobber=True)

        # Write MTL truth
        output_truth_path = os.path.join(params['target_dir'],params['truth_name'])
        print('Writing output to {}'.format(os.path.abspath(output_truth_path)))

        # Truth slightly convoluted, store sources as 2nd extension. Can't
        # write multiple extensions with fitsio.writeto?
        prihdr    = fitsio.Header()
        prihdu    = fitsio.PrimaryHDU(header=prihdr)

        mainhdr   = fitsio.Header()
        mainhdr['EXTNAME'] = 'TRUTH'
        mainhdu   = fitsio.BinTableHDU.from_columns(combined_mtl_truth.as_array(),
                                                    header=mainhdr)
        sourcehdr = fitsio.Header()
        sourcehdr['EXTNAME'] = 'SOURCES'
        sourcehdu = fitsio.BinTableHDU.from_columns(combined['sources'].as_array(),
                                                    header=sourcehdr)
        sourcemetahdr = fitsio.Header()
        sourcemetahdr['EXTNAME'] = 'SOURCEMETA'
        sourcemetahdu = fitsio.BinTableHDU.from_columns(combined['sources_meta'].as_array(),
                                                        header=sourcemetahdr)

        truth_hdu = fitsio.HDUList([prihdu, mainhdu, sourcehdu, sourcemetahdu])
        truth_hdu.writeto(output_truth_path,clobber=True)
    finally:
        # Change back to original directory
        os.chdir(original_dir)
    return

#############################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fa_run_dir')
    parser.add_argument('--reset','-r',action='store_true')
    parser.add_argument('--config','-c',default='fa_run.yaml')
    args = parser.parse_args()

    mocks_to_fa_input(args.fa_run_dir,yaml_name=args.config,reset=args.reset)
    print('mocks_to_fa_input done!')

