=====================
desitarget Change Log
=====================

0.9.1 (unreleased)
------------------

* No changes yet

0.9.0 (2017-03-03)
------------------

* include mapping from MOCKID -> TARGETID
* Added shapes to gaussian mixture model of target params [PR #150]
* Added basic bright star masking
* Updates for mock targets
* Added desitarget.sandbox.cuts area for experimental work
* Add ELG XD and new LRG to sandbox

0.8.2 (2016-12-03)
------------------

* updates for mocks integrated with quicksurvey

0.8.1 (2016-11-23)
------------------

* fix select_targets and gitversion for py3

0.8.0 (2016-11-23)
------------------

* Adds DESI_TARGET bits for bright object masking
* MTL sets priority=-1 for any target with IN_BRIGHT_OBJECT set
* Many updates for reading and manipulating mock targets
* Adds BGS_FAINT target selection

0.7.0 (2016-10-12)
------------------

* Added functionality for Random Forest into quasar selection
* Updates to be compatible with python 3.5
* refactor of merged target list (mtl) code
* Update template module file to DESI+Anaconda standard.

0.6.1 (2016-08-18
------------------

* PR#59: fix LRG selection (z < 20.46 not 22.46)

0.6.0 (2016-08-17)
------------------

* Big upgrade for how Tractor Catalogues are loaded to DB. Only the mapping
  between Catalogue and DB naming is hardcoded. Compatible DR2
* Python parallelism. Can choose mulprocessing OR mpi4py
* Unit test script that compares random rows from random Catalogues with
  what is in the DB

0.5.0 (2016-08-16)
------------------

* Added obscondition and truesubtype to mocks (PR #55; JFR).
* refactored cut functions to take all fluxes so that they have same call
  signature (PR #56; JM).
* Move data into Python package to aid pip installs (PR #47; BAW).
* Support for Travis, Coveralls and ReadTheDocs (BAW).


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
