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

############################################################
def features_parse(features):
    """
    Extracts all valid lines in fiberassign features file to strings (i.e. no
    automatic processing of values to appropriate types).

    Args:
        features:   path to a features file

    Returns:
        dict with keys and values as features file.
    """
    d = dict()
    with open(features,'r') as f:
        for line in f.readlines():
            if line.startswith('-'): break
            if line.startswith('#'): continue
            try:
                w = line.index(' ')
            except ValueError:
                continue
            k = line[0:w]
            v = line[w+1:].strip()
            d[k] = v
    return d

############################################################
def make_mtls_for_sources(source_defs,output_dir,reset=False):
    """
    Writes 'intermediate' target list (TL) files that gather data from the
    input mocks using the routines in desitarget.mocks (which include target
    selection).

    Args:
        sources:    dict of source definitions.
        output_dir: location for intermediate mtl files.
        reset:      If True, force all intermediate TL files to be remade.

    Returns:
        targets_all:    list, target files for each source.
        truth_all:      list, truth files for each source.
        sources_all:    list, table of source file locations for each source.

    The keys in sources must correspond to the names of modules in
    desitarget.mock. These modules must define a function build_mock_target
    that can generate target and truth files for that mock.

    These intermediate TLs contain all targets from a mock in one file and have
    a format similar to the final fiberassign MTL, but they are not suitable
    for input to fiberassign on their own.

    These might well be cached on disk after they're made, but that's left to
    the desitarget.mock reading routines.
    """
    targets_all     = list()
    truth_all       = list()
    sourcefiles_all = list()

    print('The following mock catalog sources are specified:')
    for source_name in sorted(source_defs.keys()):
        print('{}'.format(source_name))

    # Iterate over sources in predictable order to genereate data (or read from
    # cached files)
    for source_name in sorted(source_defs.keys()):
        module_name = 'desitarget.mock.{}'.format(source_name)
        print('')
        print('Reading mock {}'.format(source_name))
        print('Using module {}'.format(module_name))

        M = importlib.import_module(module_name)
        t0 = time.time()

        mock_kwargs                          = copy(source_defs[source_name])
        mock_kwargs['output_dir']            = output_dir
        mock_kwargs['write_cached_targets']  = True
        mock_kwargs['remake_cached_targets'] = reset

        targets, truth, sourcefiles = M.build_mock_target(**mock_kwargs)
        t1 = time.time()

        targets_all.append(targets)
        truth_all.append(truth)
        sourcefiles_all.append(sourcefiles)
        print('Data read for mock {}, took {:f}s'.format(source_name,t1-t0))

    print('')
    return targets_all, truth_all, sourcefiles_all

############################################################
def combine_targets_truth(input_target_data, input_truth_data, input_sources=None):
    """
    Combines target and truth table in memory.

    Creates one target table and one truth table out of a set of input files. Each
    imput is assumed to be a valid target or truth table in its own right.

    Args:
        input_target_data  : list of arrays or Tables of targets
        input_truth_data   : list of arrays or Tables of truth
        input_sources      : optional, list of source file Tables

    Returns:
        combination_dict   : dict with the following keys:
            targets: concatenation of input target tables
            truth: concatenation of input truth tables
            (and the following only if sources != None)
            sources: concatenation of input source lists
            sources_meta: concatenation of input source lists

    """
    print('Combining {:d} target, {:d} truth tables'.format(len(input_target_data),len(input_truth_data)))
    combination_dict = dict()

    # Sanity check: all inputs have to have signed int64 dtype for TARGETID
    # column otherwise the stacking turns this column into float64.
    for i,t in enumerate(input_target_data):
        if not isinstance(t,Table):
            raise Exception('Input #{} not a table'.format(i))
        if not t['TARGETID'].dtype.type == np.dtype('int64').type:
            raise Exception('Invalid TARGETID dtype {} for input #{}'.format(t['TARGETID'].dtype,i))
        print('  -- {:4d} : {:12d} rows'.format(i,len(t)))

    # Combine TARGET tables
    #if len(input_target_data) > 1:
    master_target_table = astropy.table.vstack(input_target_data)
    #else:
    #    master_target_table = input_target_data[0]
    total_targets       = len(master_target_table)
    print('Total : {:12d} rows'.format(total_targets))

    # Combine TRUTH tables
    #if len(input_truth_data) > 1:
    master_truth_table = astropy.table.vstack(input_truth_data)
    #else:
    #    master_truth_table = input_truth_data[0]
    if (len(master_truth_table) != total_targets):
        raise Exception('%d rows in final target table but %d rows in final truth table'%(total_targets,len(master_truth_table)))

    # Verify correct targetid column
    assert(master_target_table['TARGETID'].dtype.type == np.dtype('int64').type)
    assert(master_truth_table['TARGETID'].dtype.type == np.dtype('int64').type)

    combination_dict['targets'] = master_target_table
    combination_dict['truth']   = master_truth_table

    # Propagate source lists from SOURCE extension of input truth tables.
    if input_sources is not None:
    #    if len(input_sources) > 1:
        master_source_table = astropy.table.vstack(input_sources)
    #    else:
    #        master_source_table = input_sources[0]
        total_rows_in_source_table = np.sum(master_source_table['NROWS'],dtype=np.int64)

        sources_meta = dict()
        sources_meta['NSELECTED'] = list()
        sources_meta['NSOURCES']  = list()
        for i in range(0,len(input_sources)):
            sources_meta['NSELECTED'].append(len(input_target_data[i]))
            sources_meta['NSOURCES'].append(len(input_sources[i]))
        sources_meta = Table(sources_meta)

        # Note that the source files will have more rows than go into the MTL,
        # because only a fration are selected as targets. Hence this assert
        # will fail:
        # assert(total_rows_in_source_table == total_targets)
        combination_dict['sources']      = master_source_table
        combination_dict['sources_meta'] = sources_meta

    return combination_dict

############################################################
def make_catalogs_for_source_files(fa_output_dir,
                             input_mtl,input_truth,
                             catalog_path,
                             fa_output_base='tile_',
                             tilefile=None,
                             reset=False):
    """
    Takes the fibermap fits files output by fiberassign (one per tile) and
    creates a catalog row-matched to the orginal mock from which targets
    were selected when creating the MTL.

    Args:
        catalog_path    :   Path to a fiber_to_mtl.fits catalog.
        fa_output_dir   :   fiberassign output directory
        fa_output_base  :   optional, basename of fiberassign output files

    The output recreates the full directory structure under the directory of
    catalog_path. So if a mock file is found under /path/to/mock/file_1.fits,
    the corresponding assignment catalog will appear under
    $catalog_path/path/to/mock/file_1_FA.fits.

    Notes:
        Here's a diagram of a loop representing the analysis of fiber assignment
        given some mock catalogues, stages A-E:

         E-> A. Multiple mock sources, multiple mock files per source
         |   |
         |   v (target selection)
         |   B. Target list (TL) files (one per mock source)
         |   |
         |   v (make_mtl)
         |   C. Single MTL and Truth files
         |   |
         |   v (fiberassign)
         |   D. FA tilemap files (assignments and potential assignments)
         |___|

        The idea of this script (E) is to map D back to A, using information
        propaged through C and B.

        Matching arbitrary TARGETIDS is slow (not least because it requires the
        input mock to be read). The alternative approach here assumes that the
        TARGETIDs assigned in the TL files (created in stage B) encode the row
        and filenumber of each selected target in the original mock files. The
        filenumber is defined by the read order of the files, which is
        propagated through the headers of the TL Truth and the MTL Truth.

        desitarget.mtl.make_mtl overwrites the TL TARGETIDS when it creates the
        MTL (stage C). The scheme for doing this is still undecided. Here I've
        assumed that the MTL TARGETIDS (for targets read from the input files,
        as opposed to skys and standards) simply correspond to row numbers in
        the MTL itself. These are the TARGETIDS in the fiber maps created in
        stage D.

        NOTE: the step B->C potentially involves the omission of rows
        corresponding to targets that are selected, but not observed (e.g.
        N_OBS = 0). This isn't handled at the moment, so trim=False is required
        in make_mtl.

        Assumptions:
            The TARGETIDs in B are stored for rows in C as ORIGINAL_TARGETID in
            the truth table. This covers the possibility that MTL TARGETIDS
            encode something else.

    """
    # Pattern to extract tile number from file name, not strictly required to
    # do this sice tile number is stored in header of tile file.
    tile_regex        = '%s([0-9]+).fits'%(fa_output_base)

    fa_output_pattern = os.path.join(fa_output_dir,fa_output_base + '[0-9]*.fits')
    fa_output_files   = glob.glob(fa_output_pattern)

    # Make/get the table describing observations of each target in MTL order
    # (the catalog)
    if reset or not(os.path.exists(catalog_path)):
        print('Gathering fiber maps to MTL order...')
        t = reduce_fiber_maps_to_mtl(fa_output_files,input_mtl,catalog_path,tilefile=tilefile)
        t.write(catalog_path,overwrite=True)
    else:
        print('Reading fiber map in MTL order...')
        t = Table.read(catalog_path)

    nrows_mtl = len(t)

    # Now expand the MTL-like fiber map out to many files row-matched to the
    # individual mock input files, including rows corresponding to targets that
    # were not selected or were selected but not assigned fibers.

    # First read the original target ids from the MTL input files, then use
    # those to recreate the mock brick order.
    truth_fits        = astropy_fitsio.open(input_truth)
    original_targetid = truth_fits['TRUTH'].data['ORIGINAL_TARGETID']
    truth_fits.close()

    # Read the TRUTH source list from the header
    source_list = Table.read(input_truth,hdu='SOURCES')
    source_meta = Table.read(input_truth,hdu='SOURCEMETA')

    # Select blocks of rows corresponding to each sorce file
    n_mtl_rows_processed = 0
    n_sources_processed  = 0
    for iinput in range(0,len(source_meta)):
        n_sources_this_input  = source_meta['NSOURCES'][iinput]
        n_mtl_rows_this_input = source_meta['NSELECTED'][iinput]

        print('Source {:d} has {:d} input files and {:d} rows in the MTL'.format(iinput, n_sources_this_input, n_mtl_rows_this_input))

        o_s = n_sources_processed
        l_s = n_sources_this_input
        sources_this_input = source_list[o_s:o_s+l_s]

        # Expand orginal targetids from truth file. These are assumed to
        # encode the row and file number in the original mock file.
        o = n_mtl_rows_processed
        l = n_mtl_rows_this_input
        irow, ifile = decode_rownum_filenum(original_targetid[o:o+l])

        # Loop over orginal mock files.
        print('Looping over {:d} original mock files'.format(l_s))
        for ioriginal,original_file in enumerate(sources_this_input['FILE']):
            # print('Original file: %s'%(original_file))
            # FIXME having to use 1 for the extension here, no extname
            original_nrows = astropy_fitsio.getheader(original_file,1)['NAXIS2']

            # Select rows in this mock file.
            w = np.where(ifile == ioriginal)[0]
            assert(len(w) > 0)
            assert(np.all(irow[w]>=0))

            # Make a table row-matched to the specific source file.
            source_table = Table()

            column_name_remap = {'TARGETID': 'MTL_TARGETID'}
            for colname in t.colnames:
                c = t[colname]
                column_name = column_name_remap.get(c.name,c.name)
                source_table.add_column(Column(np.repeat(-1,original_nrows),
                                               name  = column_name,
                                               dtype = c.dtype))

                source_table[column_name][irow[w]] = t[c.name][o:o+l][w]

            catalog_dir = os.path.split(catalog_path)[0]
            original_file_path, original_file_name = os.path.split(original_file)
            new_source_file_name = original_file_name.replace('.fits','_FA.fits')
            new_source_file_dir  = os.path.join(catalog_dir,*original_file_path.split(os.path.sep))
            new_source_file_path = os.path.join(new_source_file_dir,new_source_file_name)

            try:
                os.makedirs(new_source_file_dir)
            except OSError as e:
                if e.errno == 17: # File exists
                    pass

            # print('Writing output: %s'%(new_source_file_path))
            source_table.write(new_source_file_path,overwrite=True)

        # Increment index in the MTL
        n_mtl_rows_processed += n_mtl_rows_this_input
        # Increment index in the list of sources
        n_sources_processed += l_s

    return

############################################################
def reduce_fiber_maps_to_mtl(fa_output_files,input_mtl,output_dir,tilefile=None):
    """
    Reads all the FA output files and creates a table of assignments and
    potential assignments row-matched to the input MTL. Assumes that TARGETID
    for targets read from the MTL is the row index in the MTL -- hence no need
    to match TARGETIDS explicitly.

    Args:
        input_mtl       : location of MTL file fed to FA
        output_dir      : directory to write the catalog output
    """
    # Get number of rows in original MTL
    mtl_header = astropy_fitsio.getheader(input_mtl,ext=1)
    nrows_mtl  = mtl_header['NAXIS2']

    # Read all the FA files
    fa_output_all    = list()
    fa_potential_all = list()

    tileids_in_read_order     = list()
    n_per_tile_all            = list()
    n_potential_per_tile_all  = list()

    print('Reading {:d} fiberassign tile outputs'.format(len(fa_output_files)))
    for fa_file in sorted(fa_output_files):
        # Read assignments and list of potential assignments
        fa_this_tile, fa_header_this_tile = astropy_fitsio.getdata(fa_file,
                                                                   memmap = False,
                                                                   ext    = ('FIBER_ASSIGNMENTS',1),
                                                                   header = True)
        fa_potential_this_tile            = astropy_fitsio.getdata(fa_file,
                                                                   memmap = False,
                                                                   ext    =('POTENTIAL_ASSIGNMENTS',1))

        tileid                 = fa_header_this_tile['TILEID']
        n_this_tile            = len(fa_this_tile['TARGETID'])
        n_potential_this_tile  = len(fa_potential_this_tile)

        tileids_in_read_order.append(tileid)
        n_per_tile_all.append(n_this_tile)
        n_potential_per_tile_all.append(n_potential_this_tile)

        fa_output_all.append(fa_this_tile)
        fa_potential_all.append(fa_potential_this_tile)

    # Sanity checks
    unique_tile_ids = np.unique(tileids_in_read_order)
    assert(len(unique_tile_ids) == len(tileids_in_read_order))

    # Merge all the tiles and convert to Table
    print('Concatenating fiberassign tables...')
    fa_output    = Table(np.concatenate(fa_output_all))
    fa_potential = Table(np.concatenate(fa_potential_all))

    # Add tileid as a column
    tileid = np.repeat(tileids_in_read_order,n_per_tile_all)
    fa_output.add_column(Column(tileid,name='TILEID'))

    tileid_potential = np.repeat(tileids_in_read_order,n_potential_per_tile_all)
    fa_potential.add_column(Column(tileid_potential,name='TILEID'))

    # Get number of passes from tile file (sadly not stored in header)
    print('Reading tile data: %s'%(tilefile))
    tile_data    = Table.read(tilefile)
    pass_of_tile = list()

    # Assign pass to each tileid (note that there are as many tileids as input
    # files.
    for tileid in tileids_in_read_order:
        i = np.where(tile_data['TILEID'] == tileid)[0]
        pass_of_tile.append(tile_data['PASS'][i])

    pass_of_tile = np.array(pass_of_tile)
    unique_pass  = np.unique(pass_of_tile)
    npass        = len(unique_pass)

    print('Have {:d} tiles, {:d} passes'.format(len(tileids_in_read_order),npass))

    # Add pass as a column
    ipass = np.repeat(pass_of_tile,n_per_tile_all)
    fa_output.add_column(Column(ipass,name='PASS'))

    ipass_potential = np.repeat(pass_of_tile,n_potential_per_tile_all)
    fa_potential.add_column(Column(ipass_potential,name='PASS'))

    # The potentialtargetid column containts conflicts with std/skys as well as
    # input targets.

    # We need to know which potential targets correspond to which entries in
    # the fibre map. Do this by brute force for now, adding the targetid of the
    # assigned target against each entry in the potential list corresponding to
    # its fiber.
    parent_targetid    = np.repeat(fa_output['TARGETID'],   fa_output['NUMTARGET'])
    parent_desi_target = np.repeat(fa_output['DESI_TARGET'],fa_output['NUMTARGET'])
    assert(len(parent_targetid) == len(fa_potential))
    fa_potential.add_column(Column(parent_targetid, dtype=np.int64,name='PARENT_TARGETID'))
    fa_potential.add_column(Column(parent_desi_target,dtype=np.int64,name='PARENT_DESI_TARGET'))

    # Separate targets that were in the standard or sky MTL rather than the
    # main target MTL. Also have unassigned fibres to deal with.
    is_free_fiber = fa_output['TARGETID'] < 0
    is_sky        = (fa_output['DESI_TARGET'] & desi_mask.SKY)        != 0
    is_std_fstar  = (fa_output['DESI_TARGET'] & desi_mask.STD_FSTAR)  != 0
    is_std_wd     = (fa_output['DESI_TARGET'] & desi_mask.STD_WD)     != 0
    is_std_bright = (fa_output['DESI_TARGET'] & desi_mask.STD_BRIGHT) != 0
    is_skystdfree = is_sky | is_std_fstar | is_std_wd | is_std_bright | is_free_fiber

    # Sanity check targetid values for use as indices
    assert(np.all(fa_output['TARGETID'][np.invert(is_skystdfree)] < nrows_mtl))

    skystd_output = fa_output[is_skystdfree].copy()
    fa_output.remove_rows(is_skystdfree)

    # Same again for potential assignments -- remove those whose 'parent'
    # fibres (i.e. the fibres assigned to the targets for which they were the
    # other candidates) are skys and standards, or free fibres (since free
    # fibres can still have potential targets).
    is_free_fiber = fa_potential['PARENT_TARGETID'] < 0
    is_sky        = (fa_potential['PARENT_DESI_TARGET'] & desi_mask.SKY)        != 0
    is_std_fstar  = (fa_potential['PARENT_DESI_TARGET'] & desi_mask.STD_FSTAR)  != 0
    is_std_wd     = (fa_potential['PARENT_DESI_TARGET'] & desi_mask.STD_WD)     != 0
    is_std_bright = (fa_potential['PARENT_DESI_TARGET'] & desi_mask.STD_BRIGHT) != 0
    is_skystdfree = is_sky | is_std_fstar | is_std_wd | is_std_bright | is_free_fiber

    # Sanity check targetid values for use as indices
    assert(np.all(fa_potential['PARENT_TARGETID'][np.invert(is_skystdfree)] < nrows_mtl))

    skystd_potential = fa_potential[is_skystdfree].copy()
    fa_potential.remove_rows(is_skystdfree)

    # Some science fibres will have potential targets that are skys and
    # standards. These need to stay in the list for now so that the lengths of
    # entries in the potential list can be used to iterate over it. Also there
    # is no way to get the desi_target field for each potential targetid
    # without matching.

    # Use targetids as indices to reorder the merged list
    t = Table()
    t.add_column(Column(np.repeat(-1,nrows_mtl),  dtype=np.int64, name='TARGETID'))

    # FIXME relies on TARGETID being an index
    # Assume we can use targetid as an index
    print('reduce_fiber_maps_to_mtl(): WARNING: ASSUMING TARGETID IN FIBER MAP IS MTL ROWNUMBER')
    row_in_input = fa_output['TARGETID']

    # Copy data for assigned targets. This would be trivial if targets were
    # only assigned once, but that's not the case.
    unique_rows, primary_row, nobs = np.unique(row_in_input,return_index=True,return_counts=True)

    # Copy primary rows for the target list.
    # t[row_in_input[primary_row]] = fa_output[primary_row]
    t['TARGETID'][row_in_input[primary_row]] = fa_output['TARGETID'][primary_row]
    
    # Copy primary rows for the potential targetid list. Beware that this list
    # still contains non-science targets that originate outside the target MTL.
    # FIXME relies on TARGETID being an index to filter these out
    print('reduce_fiber_maps_to_mtl(): WARNING: ASSUMING TARGETID IN FIBER MAP IS MTL ROWNUMBER')
    potential_row_in_input = fa_potential['POTENTIALTARGETID']    
    is_row_in_mtl          = np.where(potential_row_in_input < nrows_mtl)[0] 
    unique_potential_rows, potential_primary_row, npotential_primary = np.unique(potential_row_in_input[is_row_in_mtl],
                                                                                 return_index=True,return_counts=True)

    # Many targets will be in the possible list but not the assigned list, so
    # need to set their target numbers. Do this only for potential targets that
    # orginate from the mtl.
    t['TARGETID'][potential_row_in_input[is_row_in_mtl[potential_primary_row]]] = fa_potential['POTENTIALTARGETID'][is_row_in_mtl[potential_primary_row]]

    # Also find unique PARENT values of each potential target. These only
    # contain targetids that are valid as indices to the MTL. No need to set
    # the output targetids using these since they're all in the target list
    # above anyway.
    potential_parent_row_in_input = fa_potential['PARENT_TARGETID']
    unique_potential_parent_rows, potential_parent_primary_row, npotential_parent = np.unique(potential_parent_row_in_input,return_index=True,return_counts=True)

    # Issues
    # - this assumes FA is run with tiles from multiple passes
    # - some targets will be available to multiple fibres on one tile
    
    # Need to known:
    # NOBS            Number of times each target is observed (as zcat)
    t.add_column(Column(np.zeros(len(t),dtype=np.int32),  name='NOBS')) 
    t['NOBS'][row_in_input[primary_row]] = nobs
    
    # NPOSSIBLE       Number of fibres that could ever reach this target on any pass
    t.add_column(Column(np.zeros(len(t),dtype=np.int32),  name='NPOSSIBLE')) 
    t['NPOSSIBLE'][potential_row_in_input[is_row_in_mtl[potential_primary_row]]] = npotential_primary

    # Add columns per-pass
    for i in range(0,npass):
        ipass = unique_pass[i]
        assert(ipass >= 0)

        # Which assigned targets have tiles in this pass?
        tiles_this_pass = np.where(fa_output['PASS'][primary_row] == ipass)[0]
        assert(len(tiles_this_pass) > 0)

        # Store tile if target was assigned on corresponding pass or -1 if not assigned.
        colname = 'TILEID_P{:d}'.format(ipass)
        t.add_column(Column(np.zeros(len(t),dtype=np.int32)-1,name=colname))
        t[colname][row_in_input[primary_row[tiles_this_pass]]] = fa_output['TILEID'][primary_row[tiles_this_pass]]

        # NALTERNATIVE    Number of other targets available to the fibre of an assigned target
        #                 (equal to the number of potential targets with this target as their primary)
        #                 -1 for targets that were not assigned
        colname = 'NALTERNATIVE_P{:d}'.format(ipass)
        t.add_column(Column(np.zeros(len(t),dtype=np.int32)-1,name=colname))
        t[colname][row_in_input[primary_row[tiles_this_pass]]] = fa_output['NUMTARGET'][primary_row[tiles_this_pass]]

        # Which potential targets have tiles in this pass?
        # Don't just use primary rows for this, since the primary row will only
        # be assocaited with one pass. We want the duplicates on other passes.
        # tiles_potential_this_pass = np.where(fa_potential['PASS'][is_row_in_mtl[potential_primary_row]] == ipass)[0]
        tiles_potential_this_pass = np.where(fa_potential['PASS'][is_row_in_mtl] == ipass)[0]
        assert(len(tiles_potential_this_pass) > 0)

        # Store tile if target was considered on corresponding pass or -1 if
        # not considered. Many fibres can consider the same target.
        colname = 'TILEID_POSSIBLE_P{:d}'.format(ipass)
        #tileid_potential_this_pass = fa_potential['TILEID'][is_row_in_mtl[potential_primary_row[tiles_potential_this_pass]]]
        tileid_potential_this_pass = fa_potential['TILEID'][is_row_in_mtl[tiles_potential_this_pass]]
        t.add_column(Column(np.zeros(len(t),dtype=np.int32)-1,name=colname))
        t[colname][potential_row_in_input[is_row_in_mtl[tiles_potential_this_pass]]] = tileid_potential_this_pass

        # Any target assigned on this pass should be a potenital target in this
        # pass.
        is_assigned_this_pass  = t['TILEID_P{:d}'.format(ipass)][row_in_input[primary_row[tiles_this_pass]]]  >= 0
        is_potential_this_pass = t['TILEID_POSSIBLE_P{:d}'.format(ipass)][row_in_input[primary_row[tiles_this_pass]]]  >= 0
        if np.any(is_assigned_this_pass & (~is_potential_this_pass)):
            raise Exception('Targets are assigned but not possible!')

        # Not implemented yet
        # NFIBSCANREACH   Number of fibres that could have assigned each target on this pass
        #colname = 'NFIBSCANREACH_P{:d}').format(ipass)
        #t.add_column(Column(np.zeros(len(t),dtype=np.int32)-1,name=colname)

    return t


