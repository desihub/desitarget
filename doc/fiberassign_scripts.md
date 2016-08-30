Scripts to help with running fiberassign on mock catalogs
=========================================================
(n.b. summary of jargon at end)

Rough notes, also see example files in docs/fiberassign_examples.

Basic idea
----------

There are a variety of different mocks, with different locations and formats.
The purpose of these scripts is to gather data from a user-specified set of
mocks into one table with a specific format, the MTL, which can then be read by
FA. The scripts also process the tile maps output by FA, to produce tables
row-matched to the original mocks that make it easier to figure out if/how a
given target in the mock was assigned.

A single run of FA is defined by a 'root' directory, `$FA_RUN_DIR`, under which
all files are stored.

Some conventions and metadata are introduced to avoid having to ever match
labels (i.e. TARGETID) to determine mappings between rows in any of the files
involved.

Parameters
----------

All these scripts are configured with a single parameter file, `fa_run.yaml`,
which lives in `$FA_RUN_DIR`. This is a YAML file.

`target_dir = $TARGET_DIR`
	Location for the MTL file and intermediate files used to create it.

`target_mtl_name, truth_name`
	Names for the MTL and truth files.

`fa_output_dir`
	Directory where FA writes fiber maps (set in features).

`catalog_dir`
	Directory where FA results are re-written in MTL and mock order.

`catalog_name`
	Name for the file in which FA results are written in MTL order.

`features`
	Path to the features file used for FA.

`sources`
	A dict. Each key corresponds the exact name of a module under
	desitarget.mock that can read and select targets from a particular mock via a
	function `build_mock_target()`. The value of each key is itself a dict that is
	intended to be passed verbatim as **kwargs to `build_mock_target()`.

Stages
------

Detailed explanations are given in the docstrings of the scripts associated with
each stage.

1. *Set up the run.*

    - Create `$FA_RUN_DIR`
    - Create an `fa_run.yaml` file in `$FA_RUN_DIR`
    - Edit `sources` in `fa_run.yaml` to point to the mocks and set the selection parameters.
    - Make sure the subdirectories specified in `fa_run.yaml` exist. (`./bin/setup_fa_run_dir.py` does this)

2. *Read the mocks, select targets and make and MTL file.*

    - `./bin/mocks_to_fa_input $FA_RUN_DIR`

    This creates files under `$FA_RUN_DIR/$TARGET_DIR`, including the MTL and truth.
    Uses modules in desitarget.mock to read and target-select each mock.

3. *Run `fiberassign`.*

    How this is done is up to you. Requires a features file.  

4. *Create a description of the fiber assignment row-matched to the mocks.*

    - `./bin/fa_output_to_mocks $FA_RUN_DIR`

    This step is optional. The normal survey simulation pipeline will carry on 
    independently from stage 3 by making a redshift catalog etc.
 
    The objective is to understand the target selection and
    fiber assignment in terms of the original mock. The approach is to create tables
    in which assigned targets are partitioned and orderd in the same way as they are
    in the original mock files, with unassigned targets interleaved as 'null' rows.

Assumptions
-----------

The scripts assume some conventions at each stage. 

### One mock, one desitarget.mock module

desitarget provides routines to read the mocks. One mock, one module.
`build_mock_target`, that returns a target-selected MTL-style table. Don't have
to cache. Standard arguments to control caching: `output_dir`,
`write_cached_targets`, `remake_cached_targets`.

### There is a defined order for rows in the MTL and truth file.

The order of rows in the intermediate TLs reflects the order in which the
original mock source files are read.

The order of rows in the final MTL and truth reflects the order in which the
intermediate TLs are read.

What these read orders actually are is irrelevant, what matters is that they
are stable and predictable.

### The truth file has extensions `SOURCES` and `SOURCEMETA`

`SOURCES` is a table with a column FILE listing the full paths of the
individual mock source files that have contributed rows to the MTL (and truth)
files, in order. The column NROWS gives the number of rows read from each
source file.

`SOURCE_META` is a short table listing, for each mock source in order, the
number of individual source files (NSOURCES) and the total number of targets
selected from that mock (NSELECTED).

The MTL can therefore be decomposed into contributions from each mock as:

    Rows from A: MTL[0:NSELECTED[A]]
    Rows from B: MTL[NSELECTED[A]:NSELECTED[A]+NSELECTED[B]]
    Rows from C: MTL[NSELECTED[A]+NSELECTED[B]:NSELECTED[A]+NSELECTED[B]+NSELECTED[C]]
    etc.
 
### The field TARGETID can have different meanings at different
   stages from the mocks to the MTL.

- In the mock source files, there is no explicit TARGETID. 

An arbitrary number of columns might be used to identify unique objects
(brickname, objectid, etc.).

- In the intermediate TL files, TARGETID encodes the row and file number of the
  object in the original source file as a single 64 bit int. 

File number here means an integer corresponding to the order in which the files
were read (the list of files in this order being stored as FILE in the TL truth
extension SOURCES). The first source (selected or not) in the first file read
has TARGETID = 0. The packing/unpacking of the bits is done by routines in
desitarget.mock.io.

An alternative would be to store the file and row number explicitly in the
truth file, rather than using the targetid for this purpose.

- In the final MTL file, TARGETID is just the row number in that file. 

`make_mtl()` creates this TARGETID. The final truth file contains a column
`ORIGINAL_TARGETID` preserving the value of TARGETID from the intermediate TLs.

This avoids having to match all the TARGETIDs in the fiber maps agains the MTL.
An alternative would be to store the MTL index of each target in the fiber map.

### Targets should not be 'trimmed' from MTL.

`desitarget.mtl.make_mtl()` has the option `trim`. If True, this strips targets
that are guaranteed not to be observed (e.g. those with `N_OBS = 0`) from the
resulting MTL and truth tables. This breaks the logic above because if that
happens the number of sources in each contributing TL is not enough to figure
out which rows in the final MTL came from which TL.

### Standards and sky fibers can be identified from their `DESI_TARGET` mask.

This should be the case, but is important because, at FA runtime and in its
output, rows for these additional `special targets' are interleaved with rows
that were read from the mocks. Since the MTL row order logic doesn't apply to
these targets, they have to be identified and ignore when processing the fiber
maps.

### The location of the file with the tile data can be read from the features file.

This is the only reason why `fa_run.yaml` needs to know where the features file
is.

Jargon
------

`$VARIABLES` written like that are up to the user to specify, one way or another.
 
Mock source: a single mock, as defined by a module interface under
desitarget.mock. For example, `mws_galaxia` or `bgs_durham`. A FA run will
include targets drawn from several sources.

Original mocks/mock files/mock catalogues: each mock source comprises an
arbitrary number of constituent files of arbitrary format, with many more rows
than will be selected as DESI targets.

Intermediate target lists (TL): cache files created by
`make_mtls_for_sources()` in `desitarget.mock.fiberassign`. These are tables
comprising ordered stacks of rows selected from the original mock files.

MTL (master target list): the file read as input by fiberassign. This is a
concatenation of the TL files with some additional columns added by
`desitarget.mtl.make_mtl()`.

Truth: File with the same number of rows as the MTL and a bunch of extra
columns. Intermediate TL files also have corresponding truth files.

FA (fiberassign): the fiber assignment code. Takes the MTL and truth as input,
as well as a tilefile and MTL-like tables of standards and sky fibers.

Features file: the parameter file for FA.

Fiber map: the output of FA. One file per tile.


Notes
-----

TODO: still some overlap between features file and `fa_run.yaml` that could be
eliminated.

TODO: could have a flag in `fa_run.yaml` to skip certain sources in step 4, if
their TL targetids did not encode row/file.

TODO: A lot of the convoluted things in this system could be avoided by storing
more metadata and/or explicitly separating TARGETID from tracability logic.
