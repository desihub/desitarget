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

You can install these tools in a variety of ways.  Here are several that may be of interest:

1.  Manually running from the git checkout.  Add the "bin" directory to your
    ``$PATH`` environment variable and add the "py" directory to your
    ``$PYTHONPATH`` environment variable.
2.  Install (and uninstall) a symlink to your live git checkout::

    $>  python setup.py develop --prefix=/path/to/somewhere
    $>  python setup.py develop --prefix=/path/to/somewhere --uninstall

3.  Install a fixed version of the tools::

    $>  python setup.py install --prefix=/path/to/somewhere

Versioning
----------

If you have tagged a version and wish to set the package version based on your
current git location::

    $>  python setup.py version

And then install as usual

Full Documentation
------------------

Please visit `desitarget on Read the Docs`_

.. _`desitarget on Read the Docs`: https://desitarget.readthedocs.io/en/latest/

License
-------

desitarget is free software licensed under a 3-clause BSD-style license. For details see
the ``LICENSE.rst`` file.
