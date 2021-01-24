=====================
desitarget Change Log
=====================

0.49.1 (unreleased)
-------------------

* Updates to MWS main survey target cuts [`PR #672`_]. Includes:
  * Add bright limit to MWS_NEARBY
  * Add MWS_BHB as main survey class

.. _`PR #672`: https://github.com/desihub/desitarget/pull/672


0.49.0 (2020-01-18)
-------------------

* General clean-up for final DR9 imaging [`PR #670`_]. Includes:
    * Debug primary-secondary cross-matching:
        * remove duplicate secondaries that match two primaries...
        * ...NOT duplicate primaries that match two secondaries.
    * Catch if no Gaia sources are found when making Gaia-only standards.
    * Shift Gaia-based morphological cuts to a single function.
    * Add or update wiki versions referenced in doc strings.
    * Change cuts for bright, Main Survey standards to G > 16.
    * Debug and streamline "outside-of-the-footprint" randoms.
    * Read the actual RELEASE number for randoms from file headers.
        * Rather than assuming a single, canonical North/South RELEASE.
    * Add new WD_BINARIES secondary program that is split by DARK/BRIGHT.

.. _`PR #670`: https://github.com/desihub/desitarget/pull/670

0.48.0 (2020-01-09)
-------------------

* First run of secondaries with real target files [`PR #669`_]. Includes:
    * Add Gaia-only standard stars to the MWS masks for SV, Main Survey:
        * `GAIA_STD_FAINT`, `GAIA_STD_BRIGHT`, `GAIA_STD_WD`.
    * General optimization, updating and debugging of the secondary code.
    * Get `TARGETIDs` from the input sweeps, not just the actual targets.
    * Add the first full bitmask for the SV1 secondary target files.
    * Updates to the data model to better reflect the primary targets.
* Clean-up minor style and doc issues from `PR #636`_ [`PR #668`_].
* Updates and bug fixes for DR9 now SV is on-sky [`PR #665`_]. Includes:
    * Pass `MASKBITS` column forward for GFAs.
    * Bug fixes necessitated by target files having a second extension.
        * Notably, not all shasums were checked in North/South overlaps.
    * Some minor additional functionality for creating randoms.
    * Clean-up code style and syntax errors introduced in `PR #664`_.
* Tutorial (and initial code) to train DR9 Random Forests [`PR #664`_].
* Simplify stellar SV bits [`PR #636`_]:
    * Secondary bit requirement for main stellar SV program to 4 bits.
    * Primary bright science WDs use the old algorithmic selection.

.. _`PR #636`: https://github.com/desihub/desitarget/pull/636
.. _`PR #664`: https://github.com/desihub/desitarget/pull/664
.. _`PR #665`: https://github.com/desihub/desitarget/pull/665
.. _`PR #668`: https://github.com/desihub/desitarget/pull/668
.. _`PR #669`: https://github.com/desihub/desitarget/pull/669

0.47.0 (2020-12-10)
-------------------

* Update the gr_blue ELG cut for DR9 imaging for SV [`PR #663`_]:

.. _`PR #663`: https://github.com/desihub/desitarget/pull/663

0.46.0 (2020-12-10)
-------------------

* Update ELG cuts for DR9 imaging for SV and Main Survey [`PR #662`_].
* Retune LRG cuts for DR9 and update the LRG SV target bits [`PR #661`_]:
    * Only use the default `BRIGHT`, `GALAXY` and `CLUSTER` masks.
        * i.e. ignore `ALLMASK` and `MEDIUM`.
    * Increase the SV faint limits from z < 20.5 to z < 21.0.
    * Increase the SV faint limits from zfiber < 21.9 to zfiber < 22.0.
* `PR #660`_: Work completed in `PR #661`_.
* Two main changes for BGS SV selection for DR9 [`PR #659`_]:
    * Remove FRACS* cuts, except for LOWQ superset.
    * Limit FIBMAG superset to r < 20.5 instead of r < 21.0.
* General clean-ups and speed-ups for DR9 work [`PR #658`_]. Includes:
    * Corrected data model when repartitioning skies into HEALPixels.
    * Faster versions of all of the `read_targets_in_X` functions:
        * e.g., `in_box`, `in_cap`, `in_tiles`, `in_hp`.
        * less general, but run faster by assuming the data model.
        * Speed-up is 10x or more for files pixelized at higher nsides.
    * Read "standard" `MASKBITS` cuts automatically for pixweight files.
    * Catch if MTL ledgers are at a lower resolution that target files.
* Extension of mag limit to 22.3 for RF selection [`PR #655`_].
* Add input sweep files and their checksums to target files [`PR #641`_].
    * Addresses `issue #20`_.
    
.. _`issue #20`: https://github.com/desihub/desitarget/issues/20
.. _`PR #641`: https://github.com/desihub/desitarget/pull/641
.. _`PR #655`: https://github.com/desihub/desitarget/pull/655
.. _`PR #658`: https://github.com/desihub/desitarget/pull/658
.. _`PR #659`: https://github.com/desihub/desitarget/pull/659
.. _`PR #660`: https://github.com/desihub/desitarget/pull/660
.. _`PR #661`: https://github.com/desihub/desitarget/pull/661
.. _`PR #662`: https://github.com/desihub/desitarget/pull/662

0.45.1 (2020-11-22)
-------------------

* Add RA/Dec to the Main Survey calls for the QSO RF in cmx [`PR #654`_].

.. _`PR #654`: https://github.com/desihub/desitarget/pull/654

0.45.0 (2020-11-22)
-------------------

* Clean-up for DR9-based commissioning [`PR #653`_]. Includes:
    * Use HEALPixels instead of ``BRICKIDs`` for supp_skies.
        * This avoids duplicated ``TARGETIDs`` where bricks span pixels.
        * Addresses `issue #647`_.
    * G < 19 for ``STD_DITHER_GAIA`` cmx targets near the Galaxy.
    * Allow ``gather_targets`` to restrict to a subset of columns.
    * Ignore new "light-curve" and "extra" flavors when finding sweeps.
    * Smarter processing of randoms when writing "bundled" slurm file.
        * Split pixelized files into N smaller files first...
        * ...then combine across pixels to make N random catalogs.
        * Never requires memory to write a very large random catalog.
* Tune the RF selection for QSOs in SV using DR9 imaging [`PR #652`_].
* Add RF files and threshold for each DR9 sub-footprint [`PR #648`_].

.. _`issue #647`: https://github.com/desihub/desitarget/issues/647
.. _`PR #648`: https://github.com/desihub/desitarget/pull/648
.. _`PR #652`: https://github.com/desihub/desitarget/pull/652
.. _`PR #653`: https://github.com/desihub/desitarget/pull/653

0.44.0 (2020-11-12)
-------------------

* Clean-up targets and randoms for the internal DR9 release [`PR #649`_]:
    * Add function :func:`geomask.imaging_mask()`:
        * Allows easier parsing of maskbits by string ("BRIGHT", etc.)
        * Establishes a default set of cuts on maskbits.
    * New executable ``alt_split_randoms`` (slower but saves memory).
    * Flexibility when adding MTL columns to randoms, to save memory:
        * MTL columns can still be added when running the randoms.
	* Or, can now be added when splitting a larger random catalog.
* Add notebook demonstrating ledgers [`PR #642`_].

.. _`PR #642`: https://github.com/desihub/desitarget/pull/642
.. _`PR #649`: https://github.com/desihub/desitarget/pull/649

0.43.0 (2020-10-27)
-------------------

* Add the ``STD_DITHER_GAIA`` target class for CMX [`PR #644`_].
    * For dither tests outside the Legacy Surveys footprint.
* Tune shifts between southern and northern imaging for DR9 [`PR #643`_].
* Update Travis for Py3.8/Astropy 4.x (fixes `issue #639`_) [`PR #640`_].
    * Also adds a useful script for recovering the QSO RF probabilities.
* Add units to all output files (addresses `issue #356`_) [`PR #638`_]:
    * Units for all output quantities are stored in `data/units.yaml`.
    * Unit tests check that output quantities have associated units.
    * Unit tests also check that all units are valid astropy units.
    * Also some more minor cleanup and speedups.

.. _`issue #356`: https://github.com/desihub/desitarget/issues/356
.. _`issue #639`: https://github.com/desihub/desitarget/issues/639
.. _`PR #638`: https://github.com/desihub/desitarget/pull/638
.. _`PR #640`: https://github.com/desihub/desitarget/pull/640
.. _`PR #643`: https://github.com/desihub/desitarget/pull/643
.. _`PR #644`: https://github.com/desihub/desitarget/pull/644

0.42.0 (2020-08-17)
-------------------

* Update the data model to address `issue #633`_ [`PR #637`_].
* Major refactor to MTL to implement ledgers [`PR #635`_]. Includes:
    * Code to make initial HEALPix-split ledger files from target files.
        * Ledgers can be produced for each observing layer.
        * Also includes an easy-to-use binary executable script.
        * New data model with timestamp, code version and target states.
    * Code to rapidly update MTL information by appending to a ledger.
        * Uses targets and a zcat with the current standard columns.
    * Functionality that works with either FITS or ECSV files.
    * Automatic trimming of target columns in :func:`mtl.make_mtl()`.
        * Saves memory, which may help with processing of mocks.
    * :func:`mtl.inflate_ledger()` to re-obtain trimmed target columns.
    * Code to write MTL files in a standard format.
    * Utility functions to read (FITS or ECSV) MTL ledgers:
        * In a set of HEALPixels (:func:`io.read_mtl_in_hp`)
        * In a set of tiles (:func:`read_targets_in_tiles` with mtl=True)
        * In a box (:func:`read_targets_in_box` with mtl=True)
        * In a cap (:func:`read_targets_in_cap` with mtl=True)
    * Can read entire ledger, or most recent entry for each ``TARGETID``.

.. _`issue #633`: https://github.com/desihub/desitarget/issues/633
.. _`PR #635`: https://github.com/desihub/desitarget/pull/635
.. _`PR #637`: https://github.com/desihub/desitarget/pull/637

0.41.0 (2020-08-04)
-------------------

* Support for python/3.8 and numpy/1.18, including new tests
  [`PR #631`_, `PR #634`_]
* Minor data model fixes, error checks and streamlining [`PR #627`_].
    * The most important change is that MWS science targets are no
      longer observed in GRAY or DARK, except for MWS_WDs.
* Cleanup: Avoid absolute path in resource_filename [`PR #626`_].
* Update masking to be "all-sky" using Gaia/Tycho/URAT [`PR #625`_]:
    * General desitarget functionality to work with Tycho files.
    * Deprecate using the sweeps to mask bright objects as this is now
      being done using MASKBITS from the imaging catalogs.
    * Functionality to allow masks to be built at different epochs, via
      careful treatment of Tycho/Gaia/URAT proper motions.
    * Bright star masks are now explicitly written to a $MASK_DIR.
    * The radius-magnitude relationship is now a single function.
    * Refactoring of unit tests to be simpler and have more coverage.
    * Skies and supplemental skies are now always masked by default.
    * A lack of backward compatibility, which should be OK as the masking
      formalism wasn't being extensively used.
* Functionality for iterations of SV beyond sv1 [`PR #624`_]. Includes:
    * A script to create the necessary files for new iterations of SV.
    * Generalized mask/cuts handling for survey=svX, X being any integer.
    * :func:`targets.main_cmx_or_sv` also updated to handle survey=svX.
    * Alter the automated creation of output SV target directory names:
        * write svX targets to /targets/svX/ instead of just targets/sv/.
    * Make TARGETID for secondary targets unique for iterations of SVX:
        * Schema is RELEASE=(X-1)*100 + SCND_BIT for SVX-like surveys...
	* ...and RELEASE=5*100 + SCND_BIT for the Main Survey.
* Adjust MWS SV1 target classes for new SV schedule [`PR #623`_]:
    * More generic names for clusters, stream, dwarf targets.
    * Remove ORPHAN, add CV.
    * Lower priority for SEGUE targets.

.. _`PR #623`: https://github.com/desihub/desitarget/pull/623
.. _`PR #624`: https://github.com/desihub/desitarget/pull/624
.. _`PR #625`: https://github.com/desihub/desitarget/pull/625
.. _`PR #626`: https://github.com/desihub/desitarget/pull/626
.. _`PR #627`: https://github.com/desihub/desitarget/pull/627
.. _`PR #631`: https://github.com/desihub/desitarget/pull/631
.. _`PR #634`: https://github.com/desihub/desitarget/pull/634

0.40.0 (2020-05-26)
-------------------

* Add RELEASE for dr9i, dr9j (etc.) of the Legacy Surveys [`PR #622`_].
* Repartition sky files so skies lie in HEALPix boundaries [`PR #621`_]:
    * Previously, unlike other target classes, skies were written such
      that the *brick centers* in which they were processed, rather
      than the sky locations themselves, lay within given HEALPixels.
    * :func:`is_sky_dir_official` now checks skies are partitioned right.
    * `bin/repartition_skies` now reassigns skies to correct HEALPixels.
    * In addition, also includes:
        * Significant (5-10x) speed-ups in :func:`read_targets_in_hp`.
        * Remove supplemental skies that are near existing sky locations.
          (which addresses `issue #534`_).
        * A handful of more minor fixes and speed-ups.
* Various updates to targeting bits and MTL [`PR #619`_]. Includes:
    * Don't select any BGS_WISE targets in the Main Survey.
    * Always set BGS targets with a ZWARN > 0 to a priority of DONE.
    * Add an informational bit for QSOs selected with the high-z RF
      (addresses `issue #349`_).
    * MWS targets should drop to a priority of DONE after one observation
      (but will always be higher priority than BGS for that observation).
    * Update the default priorities for reobserving Lyman-alpha QSOs
      (as described in `issue #486`_, which this addresses).
* `NUMOBS_MORE` for tracer QSOs that are also other targets [`PR #617`_]:
    * Separate the calculation of `NUMOBS_MORE` into its own function.
    * Consistently use `zcut` = 2.1 to define Lyman-Alpha QSOs.
    * Check tracer QSOs that are other targets drop to `NUMOBS_MORE` = 0.
    * New unit test to enforce that check on such tracer QSOs.
    * New unit test to check BGS always gets `NUMOBS_MORE` = 1 in BRIGHT.
    * Enforce maximum seed in :func:`randoms_in_a_brick_from_edges()`.
* Update masks for QSO Random Forest selection for DR8 [`PR #615`_]
* Add a new notebook tutorial about the Merged Target List [`PR #614`_].
* Recognize (and skip) existing (completed) healpixels when running
  `select_mock_targets` [`PR #591`_].

.. _`issue #349`: https://github.com/desihub/desitarget/issues/349
.. _`issue #486`: https://github.com/desihub/desitarget/issues/486
.. _`issue #534`: https://github.com/desihub/desitarget/issues/534
.. _`PR #591`: https://github.com/desihub/desitarget/pull/591
.. _`PR #614`: https://github.com/desihub/desitarget/pull/614
.. _`PR #615`: https://github.com/desihub/desitarget/pull/615
.. _`PR #617`: https://github.com/desihub/desitarget/pull/617
.. _`PR #619`: https://github.com/desihub/desitarget/pull/619
.. _`PR #621`: https://github.com/desihub/desitarget/pull/621
.. _`PR #622`: https://github.com/desihub/desitarget/pull/622

0.39.0 (2020-05-01)
-------------------

* Help the mocks run on pixel-level imaging data [`PR #611`_]. Includes:
    * New :func:`geomask.get_brick_info()` function to look up the
      brick names associated with each HEALPixel.
    * In :func:`randoms.quantities_at_positions_in_a_brick()`, add a
      `justlist` option to list the (maximal) required input files.
    * Minor bug fixes and documentation updates.
* Update QSO Random Forest selection (and files) for DR8 [`PR #610`_].

.. _`PR #610`: https://github.com/desihub/desitarget/pull/610
.. _`PR #611`: https://github.com/desihub/desitarget/pull/611

0.38.0 (2020-04-23)
-------------------

* Minor updates for latest DR9 imaging versions (dr9f/dr9g) [`PR #607`_].
* Extra columns and features in the random catalogs [`PR #606`_]:
    * Better error messages and defaults for `bin/supplement_randoms`.
    * Don't calculate APFLUX quantities if aprad=0 is passed.
    * Pass the randoms through the `finalize` and `make_mtl` functions:
        * To populate columns needed to run fiberassign on the randoms.
        * Addresses `issue #597`_.
    * Add the `BRICKID` column to the random catalogs.
    * Also add a realistic `TARGETID` (and `RELEASE, BRICK_OBJID`).
    * Recognize failure modes more quickly (and fail more quickly).
    * Write out both "resolve" and "noresolve" (North/South) catalogs.
* Fixes a typo in the priority of MWS_WD_SV targets [`PR #601`_].
* Fixes calc_priority logic for MWS CMX targets [`PR #601`_].
* Separate calc_priority() for CMX into a separate function [`PR #601`_].
* Alter cmx targetmask such that obsconditions can be used to work
  around MWS/BGS conflicts on MWS CMX tiles [`PR #601`_].
* Update test_priorities() for new MWS CMX targets scheme [`PR #601`_].
* Adds SV0_MWS_FAINT bit [`PR #601`_].

.. _`issue #597`: https://github.com/desihub/desitarget/issues/597
.. _`PR #601`: https://github.com/desihub/desitarget/pull/601
.. _`PR #606`: https://github.com/desihub/desitarget/pull/606
.. _`PR #607`: https://github.com/desihub/desitarget/pull/607

0.37.3 (2020-04-15)
-------------------

* Update QA now basemap dependency is removed [`PR #605`_]:
    * Also reintroduce unit tests in `test_qa.py`.
    * basemap dependency was removed in `desiutil PR #141`_

.. _`desiutil PR #141`: https://github.com/desihub/desiutil/pull/141
.. _`PR #605`: https://github.com/desihub/desitarget/pull/605

0.37.2 (2020-04-13)
-------------------

* Fix `select_mock_targets` I/O bug reported in #603 [`PR #604`_].

.. _`PR #604`: https://github.com/desihub/desitarget/pull/604

0.37.1 (2020-04-07)
-------------------

* Fix mock QSO density bug reported in #594 [`PR #602`_].
* Fixes a typo in the priority of MWS_WD_SV targets [`PR #600`_].

.. _`PR #600`: https://github.com/desihub/desitarget/pull/600
.. _`PR #602`: https://github.com/desihub/desitarget/pull/602

0.37.0 (2020-03-12)
-------------------

* Add `SV0_MWS_CLUSTER_` target classes for commissioning [`PR #599`_].
* Flag the high-z quasar selection in CMX (as `SV0_QSO_Z5`) [`PR #598`_].
* Leak of Bright Stars in BGS Main Survey and BGS SV fixed [`PR #596`_].
* Restrict skies to the geometric boundaries of their brick [`PR #595`_].
* Changes in CMX after running code for Mini-SV [`PR #592`_]. Includes:
    * g/G >= 16 for `SV0_BGS`/`SV0_MWS`/`SV0_WD`/`MINI_SV_BGS_BRIGHT`.
    * Remove the LRG `LOWZ_FILLER` class (both in SV and CMX).
    * Mask on `bright` in `MASKBITS` for z~5 QSOs (both in SV and CMX).
    * Remove the 'low quality' (`lowq`) component of `SV0_BGS`.
    * Add optical `MASKBITS` flags for LRGs (in Main Survey, SV and CMX).

.. _`PR #592`: https://github.com/desihub/desitarget/pull/592
.. _`PR #595`: https://github.com/desihub/desitarget/pull/595
.. _`PR #596`: https://github.com/desihub/desitarget/pull/596
.. _`PR #598`: https://github.com/desihub/desitarget/pull/598
.. _`PR #599`: https://github.com/desihub/desitarget/pull/599

0.36.0 (2020-02-16)
-------------------

* Add Main Survey LRG/ELG/QSO/BGS cuts to CMX for Mini-SV [`PR #590`_].
* Cut on NOBS > 0 for QSOs and LRGs for Main Survey and SV [`PR #589`_].
* Fix bug when adding LSLGA galaxies into Main Survey BGS [`PR #588`_]:
    * Catch cases of bytes/str types as well as zero-length strings.
* Noting (here) that we used the BFG to excise lots of junk [`PR #587`_].
* Updates and fixes to QA for DR9 [`PR #584`_]. Includes:
    * Options to pre-process and downsample input files to speed testing.
    * Better labeling of QA output, including cleaning up labeling bugs.
    * Make points in scatter plots black to contrast with blue contours.
    * Smarter clipping of dense pixels in histogram plots and sky maps.
    * Print out densest pixels for each target class, with viewer links.
* Update BGS Main target selection as stated in [`PR #581`_]. Includes:
    * Changes in Fibre Magnitude Cut.
    * LSLGA galaxies manually added to BGS.
        * Future-proof LSLGA object references changing ('L2' --> 'LX').
    * 'REF_CAT' information passed to throught '_prepare_optical_wise'.
* Tune QSO SV selection for both North and South for dr9d [`PR #580`_].

.. _`PR #580`: https://github.com/desihub/desitarget/pull/580
.. _`PR #581`: https://github.com/desihub/desitarget/pull/581
.. _`PR #584`: https://github.com/desihub/desitarget/pull/584
.. _`PR #587`: https://github.com/desihub/desitarget/pull/587
.. _`PR #588`: https://github.com/desihub/desitarget/pull/588
.. _`PR #589`: https://github.com/desihub/desitarget/pull/589
.. _`PR #590`: https://github.com/desihub/desitarget/pull/590

0.35.3 (2020-02-03)
-------------------

* Further fixes for DR9 [`PR #579`_]. Includes:
    * Add ``SERSIC`` columns for the DR9 data model.
    * Read the bricks file in lower-case in :func:`get_brick_info()`:
        * As, during DR9 testing, it's been both upper- and lower-case.
    * Set the default ``nside`` to ``None`` for the randoms:
        * To force the user to choose an ``nside``, or fail otherwise.
    * Fix a numpy future/deprecation warning.
* Load yaml config file safely in ``mpi_select_mock_targets`` [`PR #577`_].
* Fix bugs in updating primary targets with secondary bits set [`PR #574`_].
* Adds more stellar SV targets [`PR #574`_].
* Add LyA features to ``select_mock_targets`` [`PR #565`_].

.. _`PR #565`: https://github.com/desihub/desitarget/pull/565
.. _`PR #574`: https://github.com/desihub/desitarget/pull/574
.. _`PR #577`: https://github.com/desihub/desitarget/pull/577
.. _`PR #579`: https://github.com/desihub/desitarget/pull/579

0.35.2 (2019-12-20)
-------------------

* Fix z~5 QSO bug in CMX/SV0 that was already fixed for SV [`PR #576`_].

.. _`PR #576`: https://github.com/desihub/desitarget/pull/576

0.35.1 (2019-12-16)
-------------------

* Fix bugs triggered by empty files or regions of the sky [`PR #575`_].

.. _`PR #575`: https://github.com/desihub/desitarget/pull/575

0.35.0 (2019-12-15)
-------------------

* Preparation for DR9 [`PR #573`_]. Includes:
    * Update data model, maintaining backwards compatibility with DR8.
    * Don't set the ``SKY`` bit when setting the ``SUPP_SKY`` bit.
    * Users can input a seed (1, 2, 3, etc.) to ``bin/select_randoms``:
        * This user-provided seed is added to the output file name.
        * Facilitates generating a range of numbered random catalogs.
    * Write out final secondaries using :func:`io.find_target_files()`.
* More clean-up of glitches and minor bugs [`PR #570`_]. Includes:
    * Remove Python 3.5 unit tests.
    * Catch AssertionError if NoneType input directory when writing.
        * Later (correctly) updated to AttributeError directly in master.
    * Assert the data model when reading secondary target files.
    * Use io.find_target_files() to name priminfo file for secondaries.
    * Allow N < 16 nodes when bundling files for slurm.
    * Use the DR14Q file for SV, not the DR16Q file.
* Fix bug where wrong SNRs were passed to z~5 QSO selection [`PR #569`_].
* General clean-up of glitches and minor bugs [`PR #564`_]. Includes:
    * Don't include BACKUP targets in the pixweight files.
    * Correctly write all all-sky pixels outside of the Legacy Surveys.
    * Propagate flags like --nosec, --nobackup, --tcnames when bundling.
    * Write --tcnames options to header of output target files.
    * Deprecate the sandbox and file-format-check function.
    * Find LSLGAs using 'L' in `REF_CAT` not 'L2' (to prepare for 'L3').
    * Refactor to guard against future warnings and overflow warnings.
    * Return all HEALpixels at `nside` in :func:`sweep_files_touch_hp()`.
* Strict ``NoneType`` checking and testing for fiberfluxes [`PR #563`_]:
    * Useful to ensure ongoing compatibility with the mocks.
* Bitmasks (1,12,13), rfiberflux cut for BGS Main Survey [`PR #562`_].
* Implement a variety of fixes to `select_mock_targets` [`PR #561`_].
* Fixes and updates to ``secondary.py`` [`PR #530`_]:
    * Fix a bug that led to incorrect ``OBSCONDITIONS`` for secondary-only
      targets.
    * Secondary target properties can override matched primary properties,
      but only for restricted combinations of DESI_TARGET bits (MWS and STD).
* Add stellar SV targets [`PR #530`_]:
    * Add MWS SV target definitions in ``sv1_targetmask`` and ``cuts``.
    * Science WDs are now a secondary target class.
    * Adds a bright limit to the ``MWS-NEARBY`` sample.
    * Add stellar SV secondary targets in ``sv1_targetmask``.
    * Remove the ``BACKSTOP`` secondary bit.

.. _`PR #530`: https://github.com/desihub/desitarget/pull/530
.. _`PR #561`: https://github.com/desihub/desitarget/pull/561
.. _`PR #562`: https://github.com/desihub/desitarget/pull/562
.. _`PR #563`: https://github.com/desihub/desitarget/pull/563
.. _`PR #564`: https://github.com/desihub/desitarget/pull/564
.. _`PR #569`: https://github.com/desihub/desitarget/pull/569
.. _`PR #570`: https://github.com/desihub/desitarget/pull/570
.. _`PR #573`: https://github.com/desihub/desitarget/pull/573

0.34.0 (2019-11-03)
-------------------

* Update SV0 (BGS, ELG, LRG, QSO) classes for commissioning [`PR #560`_].
    * Also add new ``STD_DITHER`` target class for commissioning.
* All-sky/backup targets, new output data model [`PR #558`_]. Includes:
    * Add all-sky/backup/supplemental targets for SV.
    * Add all-sky/backup/supplemental targets for the Main survey.
    * Write dark/bright using, e.g. `targets/dark/targets-*.fits` format.
    * New `targets/targets-supp/targets-*.fits` format for output.
    * Add :func:`io.find_target_files()` to parse output data model.
    * File names now generated automatically in `io.write_*` functions:
        * File-name-generation used by randoms, skies, targets and gfas.
        * `select_*` binaries for these classes use this functionality.
    * Change CMX ``BACKUP_FAINT`` limit to G < 19.

.. _`PR #558`: https://github.com/desihub/desitarget/pull/558
.. _`PR #560`: https://github.com/desihub/desitarget/pull/560

0.33.3 (2019-10-31)
-------------------

* Add cuts for z = 4.3-4.8 quasar into the z5QSO selection [`PR #559`_].

.. _`PR #559`: https://github.com/desihub/desitarget/pull/559

0.33.2 (2019-10-17)
-------------------

* Add FIBERFLUX_IVAR_G/R/Z to mock skies when merging [`PR #556`_].
* Fix minor bugs in `select_mock_targets` [`PR #555`_].
* Update the ELG selections for SV [`PR #553`_]. Includes:
    * Four new bit names:
        * ``ELG_SV_GTOT``, ``ELG_SV_GFIB``.
	* ``ELG_FDR_GTOT``, ``ELG_FDR_GFIB``.
    * Associated new ELG selections with north/south differences.
    * Propagate ``FIBERFLUX_G`` from the sweeps for SV ELG cuts.
    * Increase the default sky densities by a factor of 4x.
    * Relax CMX ``BACKUP_FAINT`` limit to G < 21 to test fiber assign.
* Bright-end ``FIBERFLUX_R`` cut on ``BGS_FAINT_EXT`` in SV [`PR #552`_].
* Update LRG selections for SV [`PR #550`_]. Includes:
    * The zfibermag faint limit is changed from 21.6 to 21.9.
    * IR-selected objects with r-W1>3.1 not subjected to the sliding cut.

.. _`PR #550`: https://github.com/desihub/desitarget/pull/550
.. _`PR #552`: https://github.com/desihub/desitarget/pull/552
.. _`PR #553`: https://github.com/desihub/desitarget/pull/553
.. _`PR #555`: https://github.com/desihub/desitarget/pull/555
.. _`PR #556`: https://github.com/desihub/desitarget/pull/556

0.33.1 (2019-10-13)
-------------------

* Enhancements and on-sky clean-up for SV and CMX [`PR #551`_]. Includes:
    * Add areas contingent on ``MASKBITS`` to the ``pixweight-`` files.
    * Change ``APFLUX`` to ``FIBERFLUX`` for skies and supp-skies.
    * Add new M33 First Light program.
    * Change priorities for the First Light programs.
    * Retain Tycho, and sources with no measured proper motion, in GFAs.
    * Add the ``REF_EPOCH`` column to all target files.

.. _`PR #551`: https://github.com/desihub/desitarget/pull/551

0.33.0 (2019-10-06)
-------------------

* Update skies, GFAs and CMX targets for all-sky observing [`PR #548`_]:
    * Process and output GFAs, skies and CMX targets split by HEALPixel.
    * "bundling" scripts to parallelize GFAs, skies, CMX by HEALPixel.
    * Bundle across all HEALPixels (not just those in the footprint).
    * Add pixel information to file headers for GFAs, skies and CMX.
    * Write all-sky CMX targets separately from in-footprint targets.
    * Add back-up and first light targets for commissioning.
    * New TARGETID encoding scheme for Gaia-only and first light targets.
    * Resolve commissioning targets from the Legacy Surveys.
    * io.read functions can now process SKY and GFA target files.
    * New function to read in targets restricted to a set of DESI tiles.
    * Implement Boris Gaensicke's geographical cuts for Gaia.
    * Update unit tests to use DR8 files.
* Further updates to changes in `PR #531`_, [`PR #544`_]. Includes:
    * A `--writeall` option to `select_secondary` writes a unified target
      file without the BRIGHT/DARK split, as for `select_targets`
    * Removes duplicate secondaries that arise from multiple matches to
      one primary and secondary targets appearing in more than one input
      file. The duplciate with highest `PRIORTIY_INIT` is retained.
* Update mocks to match latest data-based targets catalogs [`PR #543`_].
* Add new redshift 5 (``QSO_Z5``) SV QSO selection [`PR #539`_]. Also:
    * Remove all Tycho and LSLGA sources from the GFA catalog.
    * Minor improvements to documentation for secondary targets.
    * Use N/S bricks for skies when S/N bricks aren't available.
* Tune, high-z, faint (``QSO_HZ_F``) SV QSO selection [`PR #538`_]
* Use ``SPECTYPE`` from ``zcat`` to set ``NUMOBS_MORE`` [`PR #537`_]:
    * Updates behavior for tracer QSOs vs. LyA QSOs in MTL.
* Update LRG selections for DR8 [`PR #532`_]. Includes:
    * New LRG selection for SV with fewer bits.
    * New ``LOWZ_FILLER`` class for SV.
    * Add LRG 4PASS and 8PASS bits/classes using cuts on ``FLUX_Z``.
    * New and simplified LRG selection for the Main Survey.
    * Deprecate Main Survey 1PASS/2PASS LRGs, all LRGs now have one pass.
    * Deprecate some very old code in :mod:`desitarget.targets`.
* Finalize secondaries, add BRIGHT/DARK split [`PR #531`_]. Includes:
    * Updated data model for secondaries.
    * New secondary output columns (``OBSCONDITIONS``, proper motions).
    * Add a cached file of primary TARGETIDs to prevent duplicates.
    * Create a more reproducible TARGETID for secondaries.
    * Automatically write secondaries split by BRIGHT/DARK.
    * Add option to pass secondary file in MTL.
    * Insist on observing layer/conditions for MTL:
        * Ensures correct behavior for dark targets in bright time...
	      * ...and bright-time targets observed in dark-time.
    * Minor update to the ``MWS_BROAD`` class.
* Add info on versioning, main_cmx_or_sv to bitmask notebook [`PR #527`_]

.. _`PR #527`: https://github.com/desihub/desitarget/pull/527
.. _`PR #531`: https://github.com/desihub/desitarget/pull/531
.. _`PR #532`: https://github.com/desihub/desitarget/pull/532
.. _`PR #537`: https://github.com/desihub/desitarget/pull/537
.. _`PR #538`: https://github.com/desihub/desitarget/pull/538
.. _`PR #539`: https://github.com/desihub/desitarget/pull/539
.. _`PR #543`: https://github.com/desihub/desitarget/pull/543
.. _`PR #544`: https://github.com/desihub/desitarget/pull/544
.. _`PR #548`: https://github.com/desihub/desitarget/pull/548

0.32.0 (2019-08-07)
-------------------

* Add URAT catalog information [`PR #526`_]. Includes:
    * New module to retrieve URAT data from Vizier and reformat it.
    * Code to match RAs/Decs to URAT, as part of that new URAT module.
    * Substitute URAT PMs for GFAs where Gaia has not yet measured PMs.
* Update CMX and Main Survey target classes [`PR #525`_]. Includes:
    * New ``SV0_WD``, ``SV0_STD_FAINT`` target classes for commissioning.
    * Mild updates to ``SV0_BGS`` and ``SV0_MWS`` for commissioning.
    * New ``BGS_FAINT_HIP`` (high-priority BGS) Main Survey class.
    * Explicit checking on ``ASTROMETRIC_PARAMS_SOLVED`` for MWS targets.
    * Add 3-sigma parallax slop in ``MWS_MAIN`` survey target class.
* Add ``OBSCONDITIONS`` to target files [`PR #523`_] Also includes:
    * Split target files explicitly into bright and "graydark" surveys.
    * Default to such a file-spilt for SV and Main (not for cmx).
    * Adds an informational bit for supplemental sky locations.
* Use ``MASKBITS`` instead of ``BRIGHTSTARINBLOB`` [`PR #521`_]. Also:
    * Extra options and checks when making and vetting bundling scripts.
    * Option to turn off commissioning QSO cuts to speed unit tests.
* Add ELG/LRG/QSO/STD selection cuts for commissioning [`PR #519`_].
* Add full set of columns to supplemental skies file [`PR #518`_].
* Fix some corner cases when reading HEALPixel-split files [`PR #518`_].

.. _`PR #518`: https://github.com/desihub/desitarget/pull/518
.. _`PR #519`: https://github.com/desihub/desitarget/pull/519
.. _`PR #521`: https://github.com/desihub/desitarget/pull/521
.. _`PR #523`: https://github.com/desihub/desitarget/pull/523
.. _`PR #525`: https://github.com/desihub/desitarget/pull/525
.. _`PR #526`: https://github.com/desihub/desitarget/pull/526

0.31.1 (2019-07-05)
-------------------

* Pass Gaia astrometric excess noise in cmx MWS SV0 [`PR #516`_].

.. _`PR #516`: https://github.com/desihub/desitarget/pull/516

0.31.0 (2019-06-30)
-------------------

* ``MASKBITS`` of ``BAILOUT`` for randoms when no file is found [`PR #515`_].
* Near-trivial fix for an unintended change to the isELG API introduced in `PR
  #513`_ [`PR #514`_].
* Preliminary ELG cuts for DR8 imaging for main and SV [`PR #513`_].
    * Don't deprecate wider SV bits, yet, ELGs may still need them.
* Further updates to generating randoms for DR8. [`PR #512`_]. Includes:
    * Add WISE depth maps to random catalogs and pixweight files.
    * Code to generate additional supplemental randoms catalogs.
        * Supplemental, here, means (all-sky) outside of the footprint.
    * Executable to split a random catalog into N smaller catalogs.
    * Fixes a bug in :func:`targets.main_cmx_or_sv()`.
        * Secondary columns now aren't the default if rename is ``True``.
    * Better aligns data model with expected DR8 directory structure.
        * Also fixes directory-not-found bugs when generating skies.
* Add "supplemental" (outside-of-footprint) skies [`PR #510`_]:
    * Randomly populates sky area beyond some minimum Dec and Galactic b.
    * Then avoids all Gaia sources at some specified radius.
    * Fixes a bug where :func:`geomask.hp_in_box` used geodesics for Dec.
        * Dec cuts should be small circles, not geodesics.
* First implementation for secondary targets [`PR #507`_]. Includes:
    * Framework and design for secondary targeting process.
    * Works automatically for both Main Survey and SV files.
    * New bitmasks for secondaries that populate ``SCND_TARGET`` column.
        * can have any ``PRIORITY_INIT`` and ``NUMOBS_INIT``.
    * A reserved "veto" bit to categorically reject targets.
    * Rigorous checking of file formats...
        * ...and that files correspond to secondary bits.
    * Example files and file structure (at NERSC) in ``SCND_DIR``.
        * /project/projectdirs/desi/target/secondary.
    * Secondary targets are matched to primary targets on RA/Dec.
        * unless a (per-source) ``OVERRIDE`` column is set to ``True``.
    * Secondary-primary matches share the primary ``TARGETID``.
    * Non-matches and overrides have their own ``TARGETID``.
        * with ``RELEASE == 0``.
    * Non-override secondary targets are also matched to themselves.
        * ``TARGETID`` and ``SCND_TARGET`` correspond for matches.

.. _`PR #507`: https://github.com/desihub/desitarget/pull/507
.. _`PR #510`: https://github.com/desihub/desitarget/pull/510
.. _`PR #512`: https://github.com/desihub/desitarget/pull/512
.. _`PR #513`: https://github.com/desihub/desitarget/pull/513
.. _`PR #514`: https://github.com/desihub/desitarget/pull/514
.. _`PR #515`: https://github.com/desihub/desitarget/pull/515

0.30.1 (2019-06-18)
-------------------

* Fix normalization bug in QSO tracer/Lya mock target densities [`PR #509`_].
* Tune "Northern" QSO selection and color shifts for Main and SV [`PR #506`_]
* Follow-up PR to `PR #496`_ with two changes and bug fixes [`PR #505`_]:
    * Select QSO targets using random forest by default.
    * Bug fix: Correctly populate ``REF_CAT`` column (needed to correctly set
      MWS targeting bits).

.. _`PR #505`: https://github.com/desihub/desitarget/pull/505
.. _`PR #506`: https://github.com/desihub/desitarget/pull/506
.. _`PR #509`: https://github.com/desihub/desitarget/pull/509

0.30.0 (2019-05-30)
-------------------

* Drop Gaia fields with np.rfn to fix Python 3.6/macOS bug [`PR #502`_].
* Apply the same declination cut to the mocks as to the data [`PR #501`_].
* Add information to GFA files [`PR #498`_]. Includes:
    * Add columns ``PARALLAX``, ``PARALLAX_IVAR``, ``REF_EPOCH``.
    * Remove ``REF_EPOCH`` from GFA file header, as it's now a column.
    * Sensible defaults for Gaia-only ``REF_EPOCH``, ``RA/DEC_IVAR``.
    * Use fitsio.read() instead of :func:`desitarget.io.read_tractor()`.
        * It's faster and special handling of input files isn't needed.
* General clean-up of target selection code [`PR #497`_]. Includes:
    * Deprecate old functions in :mod:`desitarget.gfa`.
    * Greatly simplify :func:`io.read_tractor`.
        * Backwards-compatability is now only guaranteed for DR6-8.
    * Guard against warnings (e.g. divide-by-zero) in cuts and SV cuts.
    * Default to only passing North (S) sources through North (S) cuts.
        * Retain previous behavior if ``--noresolve`` flag is passed.
* Add SV support to select_mock_targets [`PR #496`_]
* A few more updates and enhancements for DR8 [`PR #494`_]. Includes:
    * Add ``WISEMASK_W1`` and ``WISEMASK_W2`` to random catalogs.
    * Deprecate ``BRIGHTBLOB`` in favor of ``MASKBITS`` for targets.
    * Add ``qso_selection==colorcuts`` in :func:`set_target_bits.sv1_cuts`
        * This should facilitate QSO selection for SV mocks.
* Add ``REF_CAT`` and Gaia BP and RP mags and errors to GFAs [`PR #493`_].
* Minor bug fix in how `select_mock_targets` handles Lya targets [`PR #444`_].
* Further updates and enhancements for DR8 [`PR #490`_]. Includes:
    * Resolve sky locations and SV targets in North/South regions.
    * Update sky and SV slurming for DR8-style input (two directories).
    * Write both of two input directories to output file headers.
    * Parallelize plot production to speed-up QA by factors of 8.
    * Add ``PSFSIZE`` to randoms, pixweight maps and QA plots.
    * QA and pixweight maps work fully for SV-style files and bits.
    * Pixweight code can now take HEALpixel-split targets as input.
    * Add aperture-photometered background flux to randoms catalogs.
    * Additional unit test module (:func:`test.test_geomask`).
    * Deprecate `make_hpx_density_file`; use `make_imaging_weight_map`.
    * :func:`io.read_targets_in_a_box` can now read headers.
    * Update unit test data for new DR8 columns and functionality.
* Update QSO targeting algorithms for DR8 [`PR #489`_]. Includes:
    * Update baseline quasar selection for the main survey.
    * Update QSO bits and selection algorithms for SV.
* Remove GFA/Gaia duplicates on ``REF_ID`` not ``BRICKID`` [`PR #488`_].
* Various bug and feature fixes [`PR #484`_]. Includes:
    * Fix crash when using sv_select_targets with `--tcnames`.
    * Only import matplotlib where explicitly needed.
* Update `select_mock_targets` to (current) DR8 data model [`PR #480`_].

.. _`PR #444`: https://github.com/desihub/desitarget/pull/444
.. _`PR #480`: https://github.com/desihub/desitarget/pull/480
.. _`PR #484`: https://github.com/desihub/desitarget/pull/484
.. _`PR #488`: https://github.com/desihub/desitarget/pull/488
.. _`PR #489`: https://github.com/desihub/desitarget/pull/489
.. _`PR #490`: https://github.com/desihub/desitarget/pull/490
.. _`PR #493`: https://github.com/desihub/desitarget/pull/493
.. _`PR #494`: https://github.com/desihub/desitarget/pull/494
.. _`PR #496`: https://github.com/desihub/desitarget/pull/496
.. _`PR #497`: https://github.com/desihub/desitarget/pull/497
.. _`PR #498`: https://github.com/desihub/desitarget/pull/498
.. _`PR #501`: https://github.com/desihub/desitarget/pull/501
.. _`PR #502`: https://github.com/desihub/desitarget/pull/502

0.29.1 (2019-03-26)
-------------------

* Add ``REF_CAT``, ``WISEMASK_W1/W2`` to DR8 data model [`PR #479`_].
* Use speed of light from scipy [`PR #478`_].

.. _`PR #478`: https://github.com/desihub/desitarget/pull/478
.. _`PR #479`: https://github.com/desihub/desitarget/pull/479

0.29.0 (2019-03-22)
-------------------

* Update SV selection for DR8 [`PR #477`_]. Includes:
    * New SV targeting bits for QSOs and LRGs.
    * New SV selection algorithms for QSOs, ELGs and LRGs.
    * MTL fixes to handle SV LRGs (which are now not 1PASS/2PASS).
    * QA can now interpret HEALPixel-split targeting files.
    * Updated test files for the quasi-DR8 imaging data model.
    * SKY and BAD_SKY added to commissioning bits yaml file.
    * Randoms in overlap regions, and for DR8 dual directory structure.
    * Write overlap regions in addition to resolve for targets/randoms.
* Change instances of `yaml.load` to `yaml.safe_load` [`PR #475`_].
* Fix Gaia files format in doc string (healpix not healpy) [`PR #474`_].
* Write Gaia morphologies and allow custom tilings for GFAs [`PR #467`_].
* Initial updates for DR8 [`PR #466`_]. Includes:
    * DR8 data model updates (e.g BRIGHTSTARBLOB -> bitmask BRIGHTBLOB).
    * Apply resolve capability to targets and randoms.
    * Handle BASS/MzLS and DECaLS existing in the same input directory.
* New resolve capability for post-DR7 imaging [`PR #462`_]. Includes:
    * Add ``RELEASE`` to GFA data model to help resolve duplicates.
    * Resolve N/S duplicates by combining ``RELEASE`` and areal cuts.
    * Apply the new resolve code (:func:`targets.resolve`) to GFAs.
    * Deprecate Gaia-matching code for GFAs, as we no longer need it.
* Add code to select GFAs for cmx across wider sky areas [`PR #461`_].

.. _`PR #461`: https://github.com/desihub/desitarget/pull/461
.. _`PR #462`: https://github.com/desihub/desitarget/pull/462
.. _`PR #466`: https://github.com/desihub/desitarget/pull/466
.. _`PR #467`: https://github.com/desihub/desitarget/pull/467
.. _`PR #474`: https://github.com/desihub/desitarget/pull/474
.. _`PR #475`: https://github.com/desihub/desitarget/pull/475
.. _`PR #477`: https://github.com/desihub/desitarget/pull/477

0.28.0 (2019-02-28)
-------------------

* `desitarget.mock.build.targets_truth` fixes for new priority calcs [`PR #460`_].
* Updates to GFAs and skies for some cmx issues [`PR #459`_]. Includes:
    * Assign ``BADSKY`` using ``BLOBDIST`` rather than aperture fluxes.
    * Increase default density at which sky locations are generated.
    * Store only aperture fluxes that match the DESI fiber radius.
    * Ensure GFAs exist throughout the spectroscopic footprint.
* Refactor SV/main targeting for spatial queries [`PR #458`_]. Includes:
    * Many new spatial query capabilities in :mod:`desitarget.geomask`.
    * Parallelize target selection by splitting across HEALPixels.
    * Wrappers to read in HEALPix-split target files split by:
        * HEALPixels, RA/Dec boxes, RA/Dec/radius caps, column names.
    * Only process subsets of targets in regions of space, again including:
        * HEALPixels, RA/Dec boxes, RA/Dec/radius caps.
    * New unit tests to check these spatial queries.
    * Updated notebook including tutorials on spatial queries.
* Update the SV selections for BGS [`PR #457`_].
* Update MTL to work for SV0-like cmx and SV1 tables [`PR #456`_]. Includes:
    * Make SUBPRIORITY a random number (0->1) in skies output.
    * New :func:`targets.main_cmx_or_sv` to parse flavor of survey.
    * Update :func:`targets.calc_priority` for SV0-like cmx and SV1 inputs.
    * :func:`mtl.make_mtl` can now process SV0-like cmx and SV1 inputs.
    * New unit tests for SV0-like cmx and SV1 inputs to MTL.
* Deprecate :func:`targets.calc_priority` that had table copy [`PR #452`_].
* Update SV QSO selections, add seed and DUST_DIR for randoms [`PR #449`_].
* Style changes to conform to PEP 8 [`PR #446`_], [`PR #447`_], [`PR #448`_].

.. _`PR #446`: https://github.com/desihub/desitarget/pull/446
.. _`PR #447`: https://github.com/desihub/desitarget/pull/447
.. _`PR #448`: https://github.com/desihub/desitarget/pull/448
.. _`PR #449`: https://github.com/desihub/desitarget/pull/449
.. _`PR #452`: https://github.com/desihub/desitarget/pull/452
.. _`PR #456`: https://github.com/desihub/desitarget/pull/456
.. _`PR #457`: https://github.com/desihub/desitarget/pull/457
.. _`PR #458`: https://github.com/desihub/desitarget/pull/458
.. _`PR #459`: https://github.com/desihub/desitarget/pull/459
.. _`PR #460`: https://github.com/desihub/desitarget/pull/460

0.27.0 (2018-12-14)
-------------------

* Remove reliance on Legacy Surveys for Gaia data [`PR #438`_]. Includes:
    * Use ``$GAIA_DIR`` environment variable instead of passing a directory.
    * Functions to wget Gaia DR2 CSV files and convert them to FITS.
    * Function to reorganize Gaia FITS files into (NESTED) HEALPixels.
    * Use the NESTED HEALPix scheme for Gaia files throughout desitarget.
    * Change output column ``TYPE`` to ``MORPHTYPE`` for GFAs.
* Move `select-mock-targets.yaml` configuration file to an installable location
  for use by `desitest` [`PR #436`_].
* Significant enhancement and refactor of `select_mock_targets` to include
  stellar and extragalactic contaminants [`PR #427`_].

.. _`PR #427`: https://github.com/desihub/desitarget/pull/427
.. _`PR #436`: https://github.com/desihub/desitarget/pull/436
.. _`PR #438`: https://github.com/desihub/desitarget/pull/438

0.26.0 (2018-12-11)
-------------------

* Refactor QSO color cuts and add hard r > 17.5 limit [`PR #433`_].
* Refactor of MTL and MTL-related enhancements [`PR #429`_]. Includes:
    * Use targets file `NUMOBS_INIT` not :func:`targets.calc_numobs`.
    * Use targets file `PRIORITY_INIT` not :func:`targets.calc_priority`.
    * Remove table copies from :mod:`desitarget.mtl` to use less memory.
    * New function :func:`targets.calc_priority_no_table` to use less memory.
    * Set informational (`NORTH/SOUTH`) bits to 0 `PRIORITY` and `NUMOBS`.
    * Set priorities using `LRG_1PASS/2PASS` bits rather than on `LRG`.
* Minor updates to `select_mock_targets` [`PR #425`_].
    * Use pre-computed template photometry (requires `v3.1` basis templates).
    * Include MW dust extinction in the spectra.
    * Randomly assign a radial velocity to superfaint mock targets.
* Update default mock catalogs used by `select_mock_targets` [`PR #424`_]
* Update Random Forests for DR7 quasar selection [`PR #423`_]
* Fix bugs in main MWS selections [`PR #422`_].
* Fix `python setup.py install` for cmx and sv1 directories [`PR #421`_].
* More updates to target classes, mainly for SV [`PR #418`_]. Includes:
    * First full implementations of `QSO`, `LRG`, `ELG`, and `STD` for SV.
    * Update and refactor of `MWS` and `BGS` classes for the main survey.
    * Change name of main survey `MWS_MAIN` class to `MWS_BROAD`.
    * Augment QA code to handle SV sub-classes such as `ELG_FDR_FAINT`.

.. _`PR #418`: https://github.com/desihub/desitarget/pull/418
.. _`PR #421`: https://github.com/desihub/desitarget/pull/421
.. _`PR #422`: https://github.com/desihub/desitarget/pull/422
.. _`PR #423`: https://github.com/desihub/desitarget/pull/423
.. _`PR #424`: https://github.com/desihub/desitarget/pull/424
.. _`PR #425`: https://github.com/desihub/desitarget/pull/425
.. _`PR #429`: https://github.com/desihub/desitarget/pull/429
.. _`PR #433`: https://github.com/desihub/desitarget/pull/433

0.25.0 (2018-11-07)
-------------------

* Randomize mock ordering for Dark Sky mocks which aren't random [`PR #416`_].
* Updates to several target classes [`PR #408`_]. Includes:
    * Refactor of the `ELG` and `MWS_MAIN` selection algorithms.
    * Update of the `ELG` and `MWS_MAIN` selection cuts.
    * Change `MWS_WD` priority to be higher than that of `BGS` target classes.
    * Set skies to `BAD` only if both g-band and r-band are missing.
* Refactor of BGS selections to separate masking and color cuts [`PR #407`_].
* Quicksurvey MTL fix [`PR #405`_].
* Mocks use QSO color cuts instead of random forest [`PR #403`_].
* Updates to Bright Galaxy Survey and QSO selections [`PR #402`_]. Includes:
    * Updates to `BGS_FAINT` and `BGS_BRIGHT` target selections.
    * New `BGS_WISE` selection and implementation.
    * New data model columns `BRIGHTSTARINBLOB` and `FRACIN_`.
    * Add cut on `BRIGHTSTARINBLOB` to QSO selection.
    * Modify I/O to retain (some) backwards-compatibility between DR6 and DR7.
    * Updated unit test example files with appropriate columns.
    * Speed-up of `cuts` unit tests without loss of coverage.
* Updated mock sky catalog with positions over a larger footprint [`PR #398`_].
* Major update to `select_mock_targets` to use the latest (v3.0) basis
  templates [`PR #395`_].
* Propagate per-class truth HDUs into final merged truth file [`PR #393`_].
* Incorporate simple WISE depth model in `select_mock_targets` which depends on
  ecliptic latitude [`PR #391`_].

.. _`PR #391`: https://github.com/desihub/desitarget/pull/391
.. _`PR #393`: https://github.com/desihub/desitarget/pull/393
.. _`PR #395`: https://github.com/desihub/desitarget/pull/395
.. _`PR #398`: https://github.com/desihub/desitarget/pull/398
.. _`PR #402`: https://github.com/desihub/desitarget/pull/402
.. _`PR #403`: https://github.com/desihub/desitarget/pull/403
.. _`PR #405`: https://github.com/desihub/desitarget/pull/405
.. _`PR #407`: https://github.com/desihub/desitarget/pull/407
.. _`PR #408`: https://github.com/desihub/desitarget/pull/408
.. _`PR #416`: https://github.com/desihub/desitarget/pull/416

0.24.0 (2018-09-26)
-------------------

* Fix bug in code that produces data for unit tests [`PR #387`_].
* Rescale spectral parameters when generating and querying kd-trees in
  `select_mock_targets` [`PR #386`_].
* Bug fixes: [`PR #383`_].
    * Use `parallax_err` when selecting `MWS_NEARBY` targets.
    * In `select_mock_targets` do not use Galaxia to select WDs and 100pc
      targets.
* Refactor QA to work with commissioning and SV files and add (first) unit tests
  for QA. [`PR #382`_].
* Estimate FIBERFLUX_[G,R,Z] for mock targets. [`PR #381`_].
* First fully working version of SV code [`PR #380`_]. Includes:
    * (Almost) the only evolving part of the code for SV is now the cuts.
    * Unit tests for SV that should be easy to maintain.
    * Bit and column setting for SV that should be maintainable.
    * SV0 (commissioning) MWS cuts.
    * Updated STD cuts to fix a `fracmasked` typo.
    * Alterations to Travis coverage to exclude some external code.
* Fix a bug which resulted in far too few standard stars being selected in the
  mocks [`PR #378`_].
* Fix a bug in how the `objtruth` tables are written out to by
  `select_mock_targets` [`PR #374`_].
* Remove Python 2.7 from Travis, add an allowed-to-fail PEP 8 check [`PR #373`_].
* Function to read ``RA``, ``DEC`` from  non-standard external files [`PR #372`_].
* Update the data model for output target files [`PR #372`_]:
    * Change ``TYPE`` to ``MORPHTYPE``.
    * Add ``EBV``, ``FIBERFLUX_G,R,Z`` and ``FIBERTOTFLUX_G,R,Z``.
* Additional commissioning (cmx) classes and priorities [`PR #370`_]. Includes:
    * New functions to define several more commissioning classes.
    * A ``$CMX_DIR`` to contain files of cmx sources to which to match.
    * An example ``$CMX_DIR`` is ``/project/projectdirs/desi/target/cmx_files``.
    * Functionality to reset initial priorities for commissioning targets.
    * Downloading fitsio using pip/astropy to fix Travis.
* Significant enhancement of `select_mock_targets` (see PR for details) [`PR
  #368`_].
* Include per-band number counts for targets on the QA pages [`PR #367`_].
* Use new :func:`desiutil.dust.SFDMap` module [`PR #366`_].
* Set the ``STD_WD`` bit (it's identical to the ``MWS_WD`` bit) [`PR #364`_].
* Add notebook for generating Gaussian mixture models from DR7 photometry and
  morphologies of ELG, LRG, and BGS targets [`PR #363`_ and `PR #365`_].
* Make commissioning (cmx) target selection fully functional [`PR #359`_]. Includes:
    * Initial target selection algorithms.
    * First unit tests for cmx (> 90% coverage).
    * ``SV_TARGET`` and ``CMX_TARGET`` as output columns instead of as a bit.
* Remove "legacy" code in QA [`PR #359`_].
    * Weight maps can now be made with :func:`desitarget.randoms.pixmap`.
* Add isELG_colors functions [`PR #357`_].
* Adapt cuts.isSTD_colors to deal with different north/south color-cuts [`PR
  #355`_].
* Refactor to allow separate commissioning and SV target selections [`PR #346`_]:
    * Added ``sv`` and ``commissioning`` directories.
    * New infrastructure to have different cuts for SV and commissioning:
        * separate target masks (e.g. ``sv/data/sv_targetmask.yaml``).
        * separate cuts modules (e.g. ``sv_cuts.py``).
    * Added executables for SV/commissioning (e.g. ``select_sv_targets``).
    * Initial ``NUMOBS`` and ``PRIORITY`` added as columns in ``targets-`` files.
    * Initial ``NUMOBS`` is now hardcoded in target masks, instead of being set by MTL.
    * ``SV`` bits added to target masks to track if targets are from SV/comm/main.
    * sv/comm/main can now be written to the header of the ``targets-`` files.
    * ``SUBPRIORITY`` is set when writing targets to facilitate reproducibility.
* Set ``NUMOBS`` for LRGs in MTL using target bits instead of magnitude [`PR #345`_].
* Update GFA targets [`PR #342`_]:
    * Handle reading Gaia from sweeps as well as matching. Default to *not* matching.
    * Makes Gaia matching radius stricter to return only the best Gaia objects.
    * Retains Gaia RA/Dec when matching, instead of RA/Dec from sweeps.
    * Fixes a bug where Gaia objects in some HEALPixels weren't being read.
    * Add Gaia epoch to the GFA file header (still needs passed from the sweeps).

.. _`PR #342`: https://github.com/desihub/desitarget/pull/342
.. _`PR #345`: https://github.com/desihub/desitarget/pull/345
.. _`PR #346`: https://github.com/desihub/desitarget/pull/346
.. _`PR #355`: https://github.com/desihub/desitarget/pull/355
.. _`PR #357`: https://github.com/desihub/desitarget/pull/357
.. _`PR #359`: https://github.com/desihub/desitarget/pull/359
.. _`PR #363`: https://github.com/desihub/desitarget/pull/363
.. _`PR #364`: https://github.com/desihub/desitarget/pull/364
.. _`PR #365`: https://github.com/desihub/desitarget/pull/365
.. _`PR #366`: https://github.com/desihub/desitarget/pull/366
.. _`PR #367`: https://github.com/desihub/desitarget/pull/367
.. _`PR #368`: https://github.com/desihub/desitarget/pull/368
.. _`PR #370`: https://github.com/desihub/desitarget/pull/370
.. _`PR #372`: https://github.com/desihub/desitarget/pull/372
.. _`PR #373`: https://github.com/desihub/desitarget/pull/373
.. _`PR #374`: https://github.com/desihub/desitarget/pull/374
.. _`PR #378`: https://github.com/desihub/desitarget/pull/378
.. _`PR #380`: https://github.com/desihub/desitarget/pull/380
.. _`PR #381`: https://github.com/desihub/desitarget/pull/381
.. _`PR #382`: https://github.com/desihub/desitarget/pull/382
.. _`PR #383`: https://github.com/desihub/desitarget/pull/383
.. _`PR #386`: https://github.com/desihub/desitarget/pull/386
.. _`PR #387`: https://github.com/desihub/desitarget/pull/387

0.23.0 (2018-08-09)
-------------------

Includes non-backwards compatible changes to standard star bit names.

* STD/STD_FSTAR -> STD_FAINT, with corresponding fixes for mocks [`PR #341`_].
* Match sweeps to Gaia and write new sweeps with Gaia columns [`PR #340`_]:
   * Also add ``BRIGHTSTARINBLOB`` (if available) to target output files.
   * And include a flag to call STD star cuts function without Gaia columns.

.. _`PR #340`: https://github.com/desihub/desitarget/pull/340
.. _`PR #341`: https://github.com/desihub/desitarget/pull/341

0.22.0 (2018-08-03)
-------------------

Includes non-backwards compatible changes to standard star target mask
bit names and selection function names.

* Produce current sets of target bits for DR7 [`PR #338`_]:
   * Update the ``LRG``, ``QSO``, ``STD`` and ``MWS`` algorithms to align with the `wiki`_.
   * In particular, major updates to the ``STD`` and ``MWS`` selections.
   * Don't match to Gaia by default, only if requested.
   * Maintain capability to match to Gaia if needed for earlier Data Releases.
   * Run subsets of target classes by passing, e.g.. ``--tcnames STD,QSO``.
   * Update unit test files to not rely on Gaia.
   * Bring Data Model into agreement with Legacy Surveys sweeps files.
   * Rename ``FSTD`` to be ``STD`` throughout.
   * QA fails gracefully if weight maps for  systematics aren't passed.

.. _`wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection
.. _`PR #338`: https://github.com/desihub/desitarget/pull/338

0.21.1 (2018-07-26)
-------------------

* Update the schema for target selection QA [`PR #334`_]:
   * Sample imaging pixels from the Legacy Surveys to make random catalogs.
   * Add E(B-V) from SFD maps and stellar densities from Gaia to the randoms.
   * Sample randoms to make HEALpixel maps of systematics and target densities.
   * Sample randoms in HEALPixels to precisely estimate imaging footprint areas.
   * Make several new systematics plots.
   * Make new plots of parallax and proper motion information from Gaia.

.. _`PR #334`: https://github.com/desihub/desitarget/pull/334

0.21.0 (2018-07-18)
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
   * Update the ``select_gfas`` binary to draw from Gaia DR2.
   * Parallelize across sweeps files to add fluxes from the Legacy Surveys.
   * Gather all Gaia objects to some magnitude limit in the sweeps areas.
* Add :mod:`desitarget.gaimatch` for matching to Gaia [`PR #322`_]:
   * Can perform object-to-object matching between Gaia and the sweeps.
   * Can, in addition, retain all Gaia objects in an RA/Dec box.
* Mock targets bug fixes [`PR #318`_].
* Add missing GMM files to installations [`PR #316`_].
* Introduction of pixel-level creation of sky locations [`PR #313`_]:
   * Significant update of :mod:`desitarget.skyfibers`
   * :mod:`desitarget.skyutilities.astrometry` to remove ``astrometry.net`` dependency.
   * :mod:`desitarget.skyutilities.legacypipe` to remove ``legacypipe`` dependency.
   * Grids sky locations by applying a binary erosion to imaging blob maps.
   * Sinks apertures at the resulting sky locations to derive flux estimates.
   * Sets the ``BAD_SKY`` bit using high flux levels in those apertures.
   * :func:`desitarget.skyfibers.bundle_bricks` to write a slurm script.
   * Parallelizes via HEALPixels to run in a few hours on interactive nodes.
   * Adds the ``select_skies`` binary to run from the command line.
   * Includes ``gather_skies`` binary to collect results from parallelization.
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
