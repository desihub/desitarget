=====================
desitarget Change Log
=====================

0.5.0 (unreleased)
------------------

* Template
* Move data into Python package to aid pip installs.
* Support for Travis, Coveralls and ReadTheDocs.


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
