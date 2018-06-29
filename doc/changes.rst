=====================
desitarget Change Log
=====================

0.21.0 (unreleased)
-------------------

* Fix bug when generating targeting QA for mock catalogs [`PR #332`_].
* Add support for GAMA/BGS mocks and new calib_only option in
  `mock.targets_truth` [`PR #331`_].
* Add ``RA_IVAR`` and ``DEC_IVAR`` to the GFA Data Model [`PR #329`_].
* Update the Gaia Data Model [`PR #327`_]:
   * Output columns formatted as expected downstream for GFA assignment.
   * Align Gaia Data Model in matching and I/O with the Legacy Surveys.
* Allow environment variables in select_mock_targets config file [`PR #325`_].  
* First version of Milky Way Survey selection [`PR #324`_]:
   * Catalog-matches to Gaia using :mod:`desitarget.gaimatch`.
   * Sets MWS_MAIN, MWS_WD and MWS_NEARBY bits.
   * Makes individual QA pages for MWS (and other) bits.
* Change GFA selection to be Gaia-based [`PR #322`_]:
   * Update the `select_gfas` binary to draw from Gaia DR2.
   * Parallelize across sweeps files to add fluxes from the Legacy Surveys.
   * Gather all Gaia objects to some magnitude limit in the sweeps areas.
* Add :mod:`desitarget.gaimatch` for matching to Gaia [`PR #322`_]:
   * Can perform object-to-object matching between Gaia and the sweeps.
   * Can, in addition, retain all Gaia objects in an RA/Dec box.
* Mock targets bug fixes [`PR #318`_]. 
* Add missing GMM files to installations [`PR #316`_]. 
* Introduction of pixel-level creation of sky locations [`PR #313`_]:
   * Significant update of :mod:`desitarget.skyfibers`
   * :mod:`desitarget.skyutilities.astrometry` to remove `astrometry.net` dependency.
   * :mod:`desitarget.skyutilities.legacypipe` to remove `legacypipe` dependency.
   * Grids sky locations by applying a binary erosion to imaging blob maps.
   * Sinks apertures at the resulting sky locations to derive flux estimates.
   * Sets the ``BAD_SKY`` bit using high flux levels in those apertures.
   * :func:`desitarget.skyfibers.bundle_bricks` to write a slurm script.
   * Parallelizes via HEALPixels to run in a few hours on interactive nodes.
   * Adds the `select_skies` binary to run from the command line.
   * Includes `gather_skies` binary to collect results from parallelization.
   * Adds functionality to plot good/bad skies against Legacy Survey images.
* select_mock_targets full footprint updates [`PR #312`_]. 
* QA fix for testing without healpix weight map [`PR #311`_]. 
* New QSO random forest [`PR #309`_]. 
* Restore the no-spectra option of select_mock_targets, for use with quicksurvey
  [`PR #307`_]. 
* Better handling of imaging survey areas for QA [`PR #304`_]:
   * :mod:`desitarget.imagefootprint` to build HEALPix weight maps of imaging.
   * Executable (bin) interface to make weight maps from the command line.
   * :mod:`desitarget.io` loader to resample maps to any HEALPix `nside`.
   * Update :mod:`desitarget.QA` to handle new imaging area weight maps.   
* Improve north/south split functions for LRG and QSO color cuts [`PR #302`_].
* Minor QA and selection cuts updates [`PR #297`_]:
   * QA matrix of target densities selected in multiple classes.
   * Functions to allow different north/south selections for LRGs.

.. _`PR #297`: https://github.com/desihub/desitarget/pull/297
.. _`PR #302`: https://github.com/desihub/desitarget/pull/302
.. _`PR #304`: https://github.com/desihub/desitarget/pull/304
.. _`PR #307`: https://github.com/desihub/desitarget/pull/307
.. _`PR #309`: https://github.com/desihub/desitarget/pull/309
.. _`PR #311`: https://github.com/desihub/desitarget/pull/311
.. _`PR #312`: https://github.com/desihub/desitarget/pull/312
.. _`PR #313`: https://github.com/desihub/desitarget/pull/313
.. _`PR #316`: https://github.com/desihub/desitarget/pull/316
.. _`PR #318`: https://github.com/desihub/desitarget/pull/318
.. _`PR #322`: https://github.com/desihub/desitarget/pull/322
.. _`PR #324`: https://github.com/desihub/desitarget/pull/324
.. _`PR #325`: https://github.com/desihub/desitarget/pull/325
.. _`PR #327`: https://github.com/desihub/desitarget/pull/327
.. _`PR #329`: https://github.com/desihub/desitarget/pull/329
.. _`PR #331`: https://github.com/desihub/desitarget/pull/331
.. _`PR #332`: https://github.com/desihub/desitarget/pull/332


0.20.1 (2018-03-29)
-------------------

* Add a bright (g>21) flux cut for ELGs. [`PR #296`_].

.. _`PR #296`: https://github.com/desihub/desitarget/pull/296

0.20.0 (2018-03-24)
-------------------

* Added compare_target_qa script [`PR #289`_].
* Astropy 2.x compatibility [`PR #291`_].
* Update of sky selection code [`PR #290`_]. Includes:
   * Use the :mod:`desitarget.brightmask` formalism to speed calculations.
   * Pass around a magnitude limit on masks from the sweeps (to better
     avoid only objects that are genuinely detected in the sweeps).
   * Reduce the default margin to produce ~1700 sky positions per sq. deg.
* Retuning of DR6 target densities [`PR #294`_]. Includes:
   * Tweaking the QSO random forest probability.
   * Adding a new ELG selection for the northern (MzLS/BASS) imaging.
   * Slight flux shifts to reconcile the northern and southern (DECaLS) imaging.
   * Initial functionality for different North/South selections.
* Some reformatting of output target files and bits [`PR #294`_]:
   * Introducing a ``NO_TARGET`` bit.
   * Renaming the ``BADSKY`` bit ``BAD_SKY`` for consistency with other bits.
   * Including ``FRACDEV`` and ``FRACDEV_IVAR`` as outputs.
   
.. _`PR #289`: https://github.com/desihub/desitarget/pull/289
.. _`PR #290`: https://github.com/desihub/desitarget/pull/290
.. _`PR #291`: https://github.com/desihub/desitarget/pull/291
.. _`PR #294`: https://github.com/desihub/desitarget/pull/294

0.19.1 (2018-03-01)
-------------------

* Fix bug whereby FLUX and WAVE weren't being written to truth.fits files
  [`PR #287`_].
* Include OBSCONDITIONS in mock sky/stdstar files for fiberassign [`PR #288`_].

.. _`PR #287`: https://github.com/desihub/desitarget/pull/287
.. _`PR #288`: https://github.com/desihub/desitarget/pull/288

0.19.0 (2018-02-27)
-------------------

This release includes significant non-backwards compatible changes
to importing target mask bits and how mock spectra are generated.

* Major refactor of select_mock_targets code infrastructure [`PR #264`_].
* Restructure desi_mask, bgs_mask, etc. imports to fix readthedocs build
  [`PR #282`_].
* Update RELEASE dictionary with 6000 (northern) for DR6 [`PR #281`_].

.. _`PR #264`: https://github.com/desihub/desitarget/pull/264
.. _`PR #282`: https://github.com/desihub/desitarget/pull/282
.. _`PR #281`: https://github.com/desihub/desitarget/pull/281

0.18.1 (2018-02-23)
-------------------

* Open BGS hdf5 mocks read-only to fix parallelism bug [`PR #278`_].

.. _`PR #278`: https://github.com/desihub/desitarget/pull/278

0.18.0 (2018-02-23)
-------------------

* New target density fluctuations model based on DR5 healpixel info [`PR
  #254`_].
* Include (initial) mock QA plots on targeting QA page [`PR #262`_]
* Added `select_gfa` script [`PR #275`_]
* Update masking for ellipses ("galaxies") in addition to circles 
  ("stars") [`PR #277`_].

.. _`PR #254`: https://github.com/desihub/desitarget/pull/254
.. _`PR #262`: https://github.com/desihub/desitarget/pull/262
.. _`PR #275`: https://github.com/desihub/desitarget/pull/275
.. _`PR #277`: https://github.com/desihub/desitarget/pull/277

0.17.1 (2017-12-20)
-------------------

* HPXNSIDE and HPXPIXEL as header keywords for mocks too [`PR #246`_].

.. _`PR #246`: https://github.com/desihub/desitarget/pull/246

0.17.0 (2017-12-20)
-------------------

* Support LyA skewers v2.x format [`PR #244`_].
* Split LRGs into PASS1/PASS2 separate bits [`PR #245`_].
* Sky locations infrastructure [`PR #248`_].
* Mock targets densities fixes [`PR #241`_ and `PR #242`_].

.. _`PR #244`: https://github.com/desihub/desitarget/pull/244
.. _`PR #245`: https://github.com/desihub/desitarget/pull/245
.. _`PR #248`: https://github.com/desihub/desitarget/pull/248
.. _`PR #241`: https://github.com/desihub/desitarget/pull/241
.. _`PR #242`: https://github.com/desihub/desitarget/pull/242

0.16.2 (2017-11-16)
-------------------

* Allows different star-galaxy separations for quasar targets for
  different release numbers [`PR #239`_].

.. _`PR #239`: https://github.com/desihub/desitarget/pull/239

0.16.1 (2017-11-09)
-------------------

* fixes to allow QA to work with mock data [`PR #235`_].
* cleanup of mpi_select_mock_targets [`PR #235`_].
* adds BGS properties notebook documentation [`PR #236`_].

.. _`PR #235`: https://github.com/desihub/desitarget/pull/235
.. _`PR #236`: https://github.com/desihub/desitarget/pull/236

0.16.0 (2017-11-01)
-------------------

* General clean-up prior to running DR5 targets [`PR #229`_].
   * Use :mod:`desiutil.log` instead of verbose (everywhere except mocks)
   * Change ``HEALPix`` references to header keywords instead of dependencies
   * Include ``SUBPRIORITY`` and shape parameter ``IVARs`` in target outputs
* Include GMM model data for mocks when installing [`PR #222`_].
* Initial simplistic code for generating sky positions [`PR #220`_]

.. _`PR #220`: https://github.com/desihub/desitarget/pull/220
.. _`PR #222`: https://github.com/desihub/desitarget/pull/222
.. _`PR #229`: https://github.com/desihub/desitarget/pull/229

0.15.0 (2017-09-29)
-------------------

* Refactored :mod:`desitarget.QA` to calculate density fluctuations in HEALPixels 
  instead of in bricks [`PR #217`_]:
* Updated :mod:`desitarget.io` for the DR5 RELEASE number [`PR #214`_]:
* Updated :mod:`desitarget.QA` to produce QA plots [`PR #210`_]:
   * Has a simple binary that runs the plot-making software in full
   * Creates (weighted) 1-D and 2-D density plots 
   * Makes color-color plots
   * Produces a simple .html page that wraps the plots, e.g.
     http://portal.nersc.gov/project/desi/users/adamyers/desitargetQA/
* Changes for mocks [`PR #200`_]:
   * Fix isLRG vs. isLRG_colors
   * Correct random seeds when processing pix in parallel
   * Misc other small bug fixes
* Added ``mpi_select_mock_targets``
* Changes for mocks [`PR #228`]:
   * Refactor of ``targets_truth_no_spectra`` 
   * Solves bug of healpix patterns present in target mocks.
   * Removes current implementation for target fluctuations.
* Added ``desitarget.mock.sky.random_sky`` [`PR #219`_]

.. _`PR #200`: https://github.com/desihub/desitarget/pull/200
.. _`PR #210`: https://github.com/desihub/desitarget/pull/210
.. _`PR #214`: https://github.com/desihub/desitarget/pull/214
.. _`PR #228`: https://github.com/desihub/desitarget/pull/228
.. _`PR #219`: https://github.com/desihub/desitarget/pull/219
.. _`PR #217`: https://github.com/desihub/desitarget/pull/217

0.14.0 (2017-07-10)
-------------------

* Significant update to handle transition from pre-DR4 to post-DR4 data model [`PR #189`_]:
   * :mod:`desitarget.io` can now read old DR3-style and new DR4-style tractor and sweeps files
   * :mod:`desitarget.cuts` now always uses DR4-style column names and formats
   * new 60-bit ``TARGETID`` schema that incorporates ``RELEASE`` column from imaging surveys
   * :mod:`desitarget.brightstar` builds masks on DR4-style data, uses ``RELEASE`` to set DR
   * HEALPix pixel number (current nside=64) added to output target files
   * ``select_targets`` passes around information related to ``HEALPix``
   * column ``PHOTSYS`` added to output files, recording North or South for the photometric system
   * unit tests that explicitly used columns and formats from the data model have been updated

.. _`PR #189`: https://github.com/desihub/desitarget/pull/189

0.13.0 (2017-06-15)
-------------------

* Fix bug when no Lya QSOs are on a brick [`PR #191`_].
* Additional QA plots for mock target catalogs [`PR #190`_]
* Additional debugging and support for healpix input to ``select_mock_targets`` [`PR #186`_].
* Set specific DONE, OBS, and DONOTOBSERVE priorities [`PR #184`_].

.. _`PR #184`: https://github.com/desihub/desitarget/pull/184
.. _`PR #186`: https://github.com/desihub/desitarget/pull/186
.. _`PR #190`: https://github.com/desihub/desitarget/pull/190
.. _`PR #191`: https://github.com/desihub/desitarget/pull/191

0.12.0 (2017-06-05)
-------------------

* Changed refs to ``desispec.brick`` to its new location at :mod:`desiutil.brick` [`PR #182`_].
* Fix ELG and stdstar mock densities; add mock QA [`PR #181`_].
* Updated LRG cuts significantly to match proposed shift in LRG target density [`PR #179`_].
* Major expansion of bright object masking functionality (for circular masks) [`PR #176`_]:
   * Generate SAFE/BADSKY locations around mask perimeters
   * Set the target bits (including TARGETID) for SAFE/BADSKY sky locations
   * Set a NEAR_RADIUS warning for objects close to (but not in) a mask
   * Plot more realistic mask shapes by using ellipses
* Significant expansion of the mocks-to-targets code [`PR #173`_ and `PR #177`_]:
   * Better and more graceful error handling.
   * Now includes contaminants.
   * Much better memory usage.
   * Updated QA notebook.
* Add Random Forest selection for ELG in the sandbox [`PR #174`_].
* Fix ELG and stdstar mock densities; add mock QA [`PR #181`_].

.. _`PR #173`: https://github.com/desihub/desitarget/pull/173
.. _`PR #174`: https://github.com/desihub/desitarget/pull/174
.. _`PR #176`: https://github.com/desihub/desitarget/pull/176
.. _`PR #177`: https://github.com/desihub/desitarget/pull/177
.. _`PR #179`: https://github.com/desihub/desitarget/pull/179
.. _`PR #181`: https://github.com/desihub/desitarget/pull/181
.. _`PR #182`: https://github.com/desihub/desitarget/pull/182

0.11.0 (2017-04-14)
-------------------

* New cuts for standards [`PR #167`_]
* Ensured objtype was being passed to :func:`~desitarget.cuts.isFSTD`.
* Added mock -> targets+spectra infrastructure

.. _`PR #167`: https://github.com/desihub/desitarget/pull/167

0.10.0 (2017-03-27)
-------------------

* Update Travis configuration to catch documentation errors.
* WIP: refactor of mock.build
* added mock.spectra module to connect mock targets with spectra
* fix overflow in LRG sandbox cuts [`PR #160`_]
* fixed many documentation syntax errors

.. _`PR #160`: https://github.com/desihub/desitarget/pull/160

0.9.0 (2017-03-03)
------------------

* Include mapping from MOCKID -> TARGETID.
* Added shapes to gaussian mixture model of target params [`PR #150`_].
* Added basic bright star masking.
* Updates for mock targets.
* Added :mod:`desitarget.sandbox.cuts` area for experimental work.
* Add ELG XD and new LRG to sandbox.

.. _`PR #150`: https://github.com/desihub/desitarget/pull/150

0.8.2 (2016-12-03)
------------------

* Updates for mocks integrated with quicksurvey.

0.8.1 (2016-11-23)
------------------

* Fix :func:`~desitarget.cuts.select_targets` and :func:`~desitarget.gitversion` for Python 3.

0.8.0 (2016-11-23)
------------------

* Adds DESI_TARGET bits for bright object masking.
* MTL sets priority=-1 for any target with IN_BRIGHT_OBJECT set.
* Many updates for reading and manipulating mock targets.
* Adds BGS_FAINT target selection.

0.7.0 (2016-10-12)
------------------

* Added functionality for Random Forest into quasar selection.
* Updates to be compatible with Python 3.5.
* Refactor of merged target list (mtl) code.
* Update template module file to DESI+Anaconda standard.

0.6.1 (2016-08-18
------------------

* `PR #59`_: fix LRG selection (z < 20.46 not 22.46).

.. _`PR #59`: https://github.com/desihub/desitarget/pull/59

0.6.0 (2016-08-17)
------------------

* Big upgrade for how Tractor Catalogues are loaded to DB. Only the mapping
  between Catalogue and DB naming is hardcoded. Compatible DR2.
* Python parallelism. Can choose mulprocessing OR mpi4py.
* Unit test script that compares random rows from random Catalogues with
  what is in the DB.

0.5.0 (2016-08-16)
------------------

* Added obscondition and truesubtype to mocks (`PR #55`_; JFR).
* refactored cut functions to take all fluxes so that they have same call
  signature (`PR #56`_; JM).
* Move data into Python package to aid pip installs (`PR #47`_; BAW).
* Support for Travis, Coveralls and ReadTheDocs (BAW).

.. _`PR #47`: https://github.com/desihub/desitarget/pull/47
.. _`PR #55`: https://github.com/desihub/desitarget/pull/55
.. _`PR #56`: https://github.com/desihub/desitarget/pull/56

0.4.0 (2016-07-12)
------------------

* Updated code from DECaLS DR1 to load DR2 tractor catalaogues to psql db.
* Basic unit test script for checking that db rows match tractor catalogues.

0.3.3 (2016-03-08)
------------------

* Added :func:`~desitarget.cuts.isMWSSTAR_colors`.
* Allow user to specify columns when reading tractor files.
* New code for generating merged target list (MTL).
* Removed unused npyquery code.

0.3.2 (2016-02-15)
------------------

* Add this changes.rst; fix _version.py.

0.3.1 (2016-02-14)
------------------

* `PR #30`_: isolated :mod:`desitarget.io` imports in :mod:`desitarget.cuts`.
* _version.py is wrong in this tag.

.. _`PR #30`: https://github.com/desihub/desitarget/pull/30

0.3 (2016-02-14)
----------------

* `PR #29`_ and `PR #27`_ refactor :mod:`desitarget.cuts` to include per-class
  functions.
* Other changes in git log before (this changes.rst didn't exist yet).
* _version.py is wrong in this tag.

.. _`PR #29`: https://github.com/desihub/desitarget/pull/29
.. _`PR #27`: https://github.com/desihub/desitarget/pull/27
