#!/usr/bin/env python
"""
Make standard star mtl-like file from stellar mock.
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
        assert(params.has_key('standards'))

        # Construct intermediate MTLs for each soruce
        targets_all, truth_all, sources_all = fiberassign.make_mtls_for_sources(params['standards'],
                                                                                params['target_dir'],
                                                                                reset=reset)
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
    print('make_standards done!')

