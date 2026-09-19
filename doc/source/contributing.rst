Contributing
============

We welcome contributions! Please follow these guidelines:

1. Fork the repository and create a feature branch.
2. Write tests for any new functionality.
3. Ensure all existing tests pass.
4. Update documentation as needed.
5. Submit a pull request.

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/molbarnabas/frgfp.git
   cd frgfp
   pip install -e .[dev]
   pytest

Building the documentation
--------------------------

The HTML documentation lives in ``doc/`` and is built with Sphinx. It imports
the package with ``autodoc``, so it must be built in an environment where the
package (including its compiled C++ extensions) is installed.

.. code-block:: bash

   pip install -r doc/requirements.txt
   sphinx-build -b html doc/source doc/build/html

or, equivalently, via the provided ``Makefile``:

.. code-block:: bash

   make -C doc html

The API reference is generated automatically from the NumPy-style docstrings in
the source code, so keep them up to date when changing the public API. The
version shown in the documentation is derived from the latest git tag, matching
how ``pyproject.toml`` is configured.
