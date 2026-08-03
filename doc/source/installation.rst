Installation
============

The package can be installed from source using ``pip``:

.. code-block:: bash

   pip install .

For development, install in editable mode:

.. code-block:: bash

   pip install -e .

Dependencies
------------

The package requires:

* Python >= 3.8
* NumPy
* Eigen (via pybind11)
* Numba (for the LPA solver)

These will be installed automatically when using ``pip``.
