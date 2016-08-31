#!/usr/bin/env python
"""
Process output of FA and write out catalogue in same order as original MTL.

Assumes targetids encode the row in the input MTL. If this isn't the case, need
to do more matching
"""
import os
import yaml
import desitarget.mock.fiberassign as fiberassign

############################################################
def fa_output_to_mocks(fa_run_dir,yaml_name='fa_run.yaml',reset=False):
    """
    Args:
        fa_run_dir:     Top level directory for FA run.
        yaml_name:      [optional] alternative name for configuration file 
                        (must be found in fa_run_dir)
        reset:          If True, remake all cached files on disk.
    """
    if not os.path.exists(fa_run_dir):
        raise Exception("Coulding find run directory: %s"%(fa_run_dir))

    param_path = os.path.join(fa_run_dir,yaml_name)
    if not os.path.exists(param_path):
        raise Exception("Coulding find fa_run.yaml parameter file in %s"%(fa_run_dir))

    # Read parameters from yaml
    with open(param_path,'r') as pfile:
        params = yaml.load(pfile)
    assert(os.path.exists(params['catalog_dir']))

    features  = fiberassign.features_parse(params['features'])

    # Work in the run directory
    original_dir = os.getcwd()
    try:
        os.chdir(fa_run_dir)

        input_mtl     = os.path.join(params['target_dir'], params['target_mtl_name'])
        input_truth   = os.path.join(params['target_dir'], params['truth_name'])
        catalog_path  = os.path.join(params['catalog_dir'],params['catalog_name'])

        assert(os.path.exists(input_mtl))
        assert(os.path.exists(input_truth))
        assert(os.path.exists(features['outDir']))
        assert(os.path.exists(features['tileFile']))

        fiberassign.make_catalogs_for_source_files(features['outDir'],
                                                   input_mtl,input_truth,
                                                   catalog_path,
                                                   fa_output_base='tile_',
                                                   tilefile=features['tileFile'],
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

    fa_output_to_mocks(args.fa_run_dir,yaml_name=args.config,reset=args.reset)
    print('fa_output_to_mocks done!')

