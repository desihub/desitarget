"""
"""
import os
import yaml

# Default path names
TARGET_DIR_DEFAULT     = './input'
FA_OUTPUT_DIR_DEFAULT  = './output'
CATALOG_DIR_DEFAULT    = './catalog'

#############################################################
def robust_makedirs(dir,verbose=False):
    r = None
    try:
        r = os.makedirs(dir)
        if verbose:
            print('Made directory: %s'%(os.path.abspath(dir)))
    except OSError as e:
        if e.errno == 17: # File exists
            if verbose:
                print('Already have directory: %s'%(os.path.abspath(dir)))
            pass
    return r

#############################################################
def setup_fa_run_dir(fa_run_dir,yaml_name='fa_run.yaml',verbose=False):
    """
    Args:
        fa_run_dir      Top level directory for FA run.
        yaml_name       [optional] alternative name for configuration file 
                        (must be found in fa_run_dir)
        verbose         If True, prints a bit more...
    """
    print('')

    yaml_path = os.path.join(fa_run_dir,yaml_name)
    if not os.path.exists(yaml_path):
        raise Exception('No config file {} in run directory {}'.format(yaml_name,fa_run_dir))

    print('Setting up root directory: {}'.format(fa_run_dir))
    print('According to config in:    {}'.format(yaml_path))
    print('')

    # Work in the run directory
    original_dir = os.getcwd()
    try:
        os.chdir(fa_run_dir)

        # Read parameters from yaml
        with open(yaml_name,'r') as pfile:
            params = yaml.load(pfile)

        # TODO shout when using defaults.

        # Stores the intermediate and final MTL files
        target_dir     = params.get('target_dir',   TARGET_DIR_DEFAULT)
        robust_makedirs(target_dir,   verbose=verbose)
        
        catalog_dir    = params.get('catalog_dir',  CATALOG_DIR_DEFAULT)
        robust_makedirs(catalog_dir,  verbose=verbose)
        
        fa_output_dir  = params.get('fa_output_dir',FA_OUTPUT_DIR_DEFAULT)
        robust_makedirs(fa_output_dir,verbose=verbose)
    finally:
        # Change back to original directory
        os.chdir(original_dir)

    print('Setup of %s as FA run directory complete!'%(fa_run_dir))
    print('')
    return

#############################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fa_run_dir')
    parser.add_argument('--config','-c',default='fa_run.yaml')
    parser.add_argument('--verbose','-v',action='store_true')
    args = parser.parse_args()

    setup_fa_run_dir(args.fa_run_dir,yaml_name=args.config,verbose=args.verbose)
