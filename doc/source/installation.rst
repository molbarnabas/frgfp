Installation
============

The package is built from source with ``pip``; the C++/Eigen extensions are
compiled automatically by ``scikit-build-core`` during the build.

Requirements
------------

* Python >= 3.10
* NumPy and Numba
* A C++ compiler with C++17 support
* CMake >= 3.15
* OpenMP
* Eigen3 (header-only)

Eigen3 and CMake are optional in practice: if Eigen3 is not installed the build
fetches a pinned copy automatically (only CMake and a C++17 compiler are then
required). OpenMP, on the other hand, is mandatory because the LPA backend calls
the OpenMP runtime directly. On macOS install it with ``brew install libomp``;
the build locates the Homebrew keg on its own.

With conda, all of the build prerequisites above are provided by the
``environment.yml`` shipped in the repository. With plain ``pip`` you have to
install CMake, a C++17 compiler and (optionally) Eigen3 yourself.

Install with conda (recommended)
--------------------------------

.. code-block:: bash

   conda env create -f environment.yml
   conda activate frgfp-env
   pip install .

Install with pip (non-conda)
----------------------------

.. code-block:: bash

   pip install -r requirements.txt
   pip install .

The ``requirements.txt`` file lists the complete Python dependency set
(runtime, testing, documentation and the build backend). CMake, Eigen3 and a
C++17 compiler must still be installed separately, for example on Debian/Ubuntu:

.. code-block:: bash

   apt install cmake g++ libeigen3-dev

Development
-----------

.. code-block:: bash

   pip install -e .[dev]

The ``dev`` extra installs the package in editable mode together with the test,
benchmark and documentation tooling, so no further ``pip`` step is needed before
running the tests or building the documentation.
