=====================
desitarget Change Log
=====================

0.15.1 (unreleased)
-------------------

* No changes yet.

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
* Added ``desitarget.mock.sky.random_sky`` [`PR #219`_]

.. _`PR #200`: https://github.com/desihub/desitarget/pull/200
.. _`PR #210`: https://github.com/desihub/desitarget/pull/210
.. _`PR #214`: https://github.com/desihub/desitarget/pull/214
.. _`PR #219`: https://github.com/desihub/desitarget/pull/219
.. _`PR #217`: https://github.com/desihub/desitarget/pull/217

0.14.0 (2017-07-10)
-------------------

* Significant updated to handle transition from pre-DR4 to post-DR4 data model [`PR #189`_]:
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
