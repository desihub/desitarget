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

############################################################
def targets_truth(params):
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

    truth_all       = list()
    source_data_all = dict()
    target_mask_all = dict()

    # prints info about what we will be loading
    source_defs = params['sources']
    print('The following populations and paths are specified:')
    for source_name in sorted(source_defs.keys()):
        source_format = params['sources'][source_name]['format']
        source_path = params['sources'][source_name]['root_mock_dir']
        print('type: {}\n format: {} \n path: {}'.format(source_name, source_format, source_path))

    # load all the mocks
    for source_name in sorted(source_defs.keys()):

        source_format = params['sources'][source_name]['format']
        source_path = params['sources'][source_name]['root_mock_dir']

        print('type: {} format: {}'.format(source_name, source_format))
        function = 'read_'+source_format
        result = getattr(mockio, function)(source_path, source_name)
        source_data_all[source_name] = result

    print('loaded {} mock sources'.format(len(source_data_all)))

    print('Making target selection for')
    # runs target selection on every mock
    for source_name in sorted(source_defs.keys()):
        source_selection = params['sources'][source_name]['selection']
        source_dict = params['sources'][source_name]
        source_data = source_data_all[source_name]

        print('type: {} select: {}'.format(source_name, source_selection))
        selection_function = source_selection + '_select'
        result = getattr(mockselect, selection_function.lower())(source_data, source_name, **source_dict)
        target_mask_all[source_name] = result

    # writes targets and truth to disk



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


