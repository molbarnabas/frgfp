Installation
============

FRGfp is installed with ``pip`` (or ``conda``). Prebuilt wheels are provided for
the platforms listed below; anywhere else the C++/Eigen extensions are compiled
from source automatically by ``scikit-build-core``.

Supported platforms
-------------------

.. list-table:: Prebuilt wheels (``pip install frgfp``)
   :header-rows: 1

   * - Platform
     - Requirement
     - Python
   * - Linux x86_64
     - glibc >= 2.28 (RHEL/Alma/Rocky 8+, Debian 10+, Ubuntu 18.10+)
     - 3.10 - 3.14
   * - Linux aarch64 (ARM64)
     - glibc >= 2.28
     - 3.10 - 3.14
   * - macOS (Apple Silicon)
     - macOS 11 "Big Sur" or newer
     - 3.10 - 3.14
   * - Windows x86_64 (AMD64)
     - Windows 10 or newer
     - 3.10 - 3.14

The wheels carry what they need: Linux wheels are ``manylinux_2_28`` and use the
system OpenMP runtime, macOS wheels bundle the OpenMP runtime they were built
against, and Windows wheels use the MSVC OpenMP runtime.

Not covered by prebuilt wheels
------------------------------

* **Intel macOS (x86_64).** Numba (>= 0.63) and llvmlite (>= 0.46) publish no
  Intel macOS wheels, so ``pip install frgfp`` cannot resolve its runtime
  dependency there. Install Numba from conda-forge first
  (``conda install -c conda-forge numba``) and then run ``pip install frgfp``,
  or build from source as described below.
* **Linux with musl libc (Alpine).** Numba publishes no musllinux wheels; build
  from source.
* **32-bit platforms** and **free-threaded CPython** (``3.14t``): not supported.
* Python 3.9 and older: the Numba dependency requires Python 3.10.

Requirements (building from source)
-----------------------------------

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

.. _development-with-docker:

Development with Docker
-----------------------

A ready-made Linux development container ships with the repository:
``compose.yaml``, ``docker/Dockerfile``, the ``docker/dev.sh`` wrapper and the VS
Code configuration in ``.devcontainer/``. It provides the complete toolchain -
C++17 compiler, CMake, Ninja, Eigen3, OpenMP and git - together with every Python
dependency (runtime, testing, documentation and notebook tooling), so nothing has
to be installed on the host apart from Docker itself.

.. code-block:: bash

   ./docker/dev.sh build                  # build the image
   ./docker/dev.sh up -d                  # start the container
   ./docker/dev.sh exec dev bash          # open a shell inside it
   ./docker/dev.sh down                   # stop it (add -v to drop the build cache)

   # from inside the container
   pytest                                         # run the test suite
   make -C doc html                               # build the documentation
   python -m http.server 8000 --directory doc/build/html

How the container is wired
^^^^^^^^^^^^^^^^^^^^^^^^^^

* The repository is bind-mounted at ``/workspace``, so container and host share one
  working tree **and** one ``.git`` directory: edit and commit on the host, build
  and test in the container.
* On first start the entrypoint installs the package in editable mode, so Python
  changes take effect immediately. Re-run ``pip install -e .`` after editing the
  C++ sources under ``frgfp/_core`` or ``frgfp/lpa``.
* ``doc/build`` lives inside the bind mount, so documentation generated in the
  container is immediately available on the host (the preview server is reachable
  at http://localhost:8000, published by ``compose.yaml``).
* The CMake/scikit-build cache is kept in the named volume ``frgfp-build``
  (mounted at ``/workspace/build``), so it never clashes with a ``build/``
  directory produced on the host. Drop it with ``./docker/dev.sh down -v``.
* Port 8888 is published for JupyterLab, which is handy for the notebook version
  of the :doc:`usage` page (``examples/usage.ipynb``).

The image creates a non-root user matching your UID and GID, so files created
inside the container (build outputs, generated documentation, notebook outputs)
stay owned by you. ``docker/dev.sh`` forwards those ids, refuses to run under
``sudo`` (which would forward ``0/0``) and re-executes itself under ``sg docker``
when your login session predates the ``docker`` group grant; if you are not in the
``docker`` group at all it prints the ``usermod`` line to run.

With VS Code, *Dev Containers: Reopen in Container* uses
``.devcontainer/devcontainer.json``: it starts the same Compose service, opens
``/workspace`` as the user ``dev``, selects ``/opt/venv/bin/python`` as the
interpreter, enables pytest collection for the ``frgfp`` package and installs the
Python, C++ and Jupyter extensions.

If you prefer to drive Compose yourself, pass your UID/GID explicitly so that
generated files are not owned by root:

.. code-block:: bash

   USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build
   docker compose up -d
   docker compose exec dev bash
