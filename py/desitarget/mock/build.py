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

############################################################
def targets_truth(source_defs,output_dir):
    """
    Write

    Args:
        sources:    dict of source definitions.
        output_dir: location for intermediate mtl files.
        reset:      If True, force all intermediate TL files to be remade.

    Returns:
        targets:    
        truth:      

    """
    targets_all     = list()
    truth_all       = list()
    sourcefiles_all = list()

    print('The following populations are specified:')
    for source_name in sorted(source_defs.keys()):
        print('{}'.format(source_name))

#    for source_name in sorted(source_defs.keys()):
#        module_name = 'desitarget.mock.{}'.format(source_name)
#        print('')
#        print('Reading mock {}'.format(source_name))
#        print('Using module {}'.format(module_name))

#        M = importlib.import_module(module_name)
#        t0 = time.time()

#        mock_kwargs                          = copy(source_defs[source_name])
#        mock_kwargs['output_dir']            = output_dir
#        mock_kwargs['write_cached_targets']  = True
#        mock_kwargs['remake_cached_targets'] = reset

#        targets, truth, sourcefiles = M.build_mock_target(**mock_kwargs)
#        t1 = time.time()

#        targets_all.append(targets)
#        truth_all.append(truth)
#        sourcefiles_all.append(sourcefiles)
#        print('Data read for mock {}, took {:f}s'.format(source_name,t1-t0))


