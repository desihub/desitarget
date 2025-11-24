==========
desitarget
==========


|Actions Status| |Coveralls Status| |Documentation Status|

.. |Actions Status| image:: https://github.com/desihub/desitarget/workflows/CI/badge.svg
    :target: https://github.com/desihub/desitarget/actions
    :alt: GitHub Actions CI Status

.. |Coveralls Status| image:: https://coveralls.io/repos/desihub/desitarget/badge.svg
    :target: https://coveralls.io/github/desihub/desitarget
    :alt: Test Coverage Status

.. |Documentation Status| image:: https://readthedocs.org/projects/desitarget/badge/?version=latest
    :target: https://desitarget.readthedocs.io/en/latest/
    :alt: Documentation Status

Introduction
------------

This package contains scripts and packages for selecting DESI targets
from photometric catalogs.

Installation
------------

For most purposes `pip install desitarget` should work.

For versions of desitarget prior to 4.4.0, the preferred installation method,
assuming depedencies are already installed, would be::

    DESITARGET_VERSION=4.3.0 && python -m pip install --no-build-isolation git+https://github.com/desihub/desitarget.git@${DESITARGET_VERSION}

Full Documentation
------------------

Please visit `desitarget on Read the Docs`_

.. _`desitarget on Read the Docs`: https://desitarget.readthedocs.io/en/latest/

License
-------

desitarget is free software licensed under a 3-clause BSD-style license. For details see
the ``LICENSE.rst`` file.
