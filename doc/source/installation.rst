Installation
============

The package is built from source with ``pip``; the C++/Eigen extensions are
compiled automatically by ``scikit-build-core`` during the build.

Requirements
------------

* Python >= 3.8
* NumPy and Numba
* A C++ compiler with C++17 support
* CMake >= 3.15
* Eigen3 (header-only)

Install from source
-------------------

.. code-block:: bash

   pip install .

For development, install in editable mode together with the test tooling:

.. code-block:: bash

   pip install -e .[dev]

The dependencies are declared in ``pyproject.toml`` and are installed
automatically by ``pip``.
