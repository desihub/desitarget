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

############################################################
def combine_targets_truth(input_target_data, input_truth_data, xref=None):
    """
    Combines target and truth table in memory.

    Parameters
    ----------
        input_targets : list of arrays or Tables of targets
        input_truth   : list of arrays or Tables of truth
        xref          : optional XREF dict
    """
    print('Combining {:d} target, {:d} truth tables'.format(len(input_target_data),len(input_truth_data)))

    # Sanity check: all inputs have to have signed int64 dtype for TARGETID
    # column otherwise the stacking turns this column into float64.
    for i,t in enumerate(input_target_data):
        if not isinstance(t,Table):
            raise Exception('Input #{} not a table'.format(i))
        if not t['TARGETID'].dtype.type == np.dtype('int64').type:
            raise Exception('Invalid TARGETID dtype {} for input #{}'.format(t['TARGETID'].dtype,i))
        print('  -- {:4d} : {:12d} rows'.format(i,len(t)))

    # Combine TARGET tables
    master_target_table = astropy.table.vstack(input_target_data)
    total_targets       = len(master_target_table)
    print('Total : {:12d} rows'.format(total_targets))

    # Combine TRUTH tables
    master_truth_table = astropy.table.vstack(input_truth_data)
    if len(master_truth_table) != total_targets:
        raise Exception('%d rows in final target table but %d rows in final truth table'%(total_targets,len(master_truth_table)))

    # Verify correct targetid column
    assert(master_target_table['TARGETID'].dtype.type == np.dtype('int64').type)
    assert(master_truth_table['TARGETID'].dtype.type == np.dtype('int64').type)

    # Write an XREF table
    if xref is not None:
        xref_names = [x[0] for x in xref]
        xref_nrows = [x[1] for x in xref]
        xref_table = Table()
        xref_table.add_column(Column(xref_names,'PATH'))
        xref_table.add_column(Column(xref_nrows,'NROWS'))
        return master_target_table, master_truth_table, xref_table
    else:
        return master_target_table, master_truth_table



