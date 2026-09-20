<div style="display: flex; align-items: center; gap: 15px;">
  <img src="frgfp_logo.png" alt="FRGfp logo" width="120" style="flex-shrink: 0;">
  <h1>FRGfp</h1>
</div>

**Functional Renormalization Group Fixed Point solver.**

A high-performance Python package for finding global collocation solutions of Functional Renormalization Group (FRG) flow equations. It provides a robust, generalized framework for evaluating effective potentials and their derivatives on freely specifiable grids, and for locating fixed points with a Newton–Gauss iteration.

The numerical core is a highly optimized Eigen C++ backend that relies on pre-computed 1D caches and row-major tensor products to evaluate multi-dimensional basis functions, gradients and Hessian matrices quickly. User-defined flow equations are JIT-compiled with Numba and passed to the backend through zero-overhead C-callbacks.

## Features

- **Multidimensional basis expansions** – polynomial (Taylor-like) and Chebyshev ansätze for an arbitrary number of invariants.
- **Collocation grid generators** – uniform and Chebyshev–Gauss–Lobatto grids with lexicographic (row-major) point ordering.
- **Fast function evaluation** – analytic basis, gradient and Hessian evaluation through the C++/Eigen backend.
- **LPA fixed-point solver** – Newton–Gauss collocation solver with optional OpenMP multithreading, parameter path following and multistart.
- **Numba integration** – user-defined flow equations are JIT-compiled and bridged to C++ as C-callbacks.

## Modules

- **`frgfp` (core)** – fundamental data structures, ansatz definitions, collocation grid generators and function evaluators, all powered by the C++ backend and re-exported at the top level.
- **`frgfp.lpa`** – collocation solver for the fixed points of LPA flow equations. Further solvers will be added here.

## Installation

### Prerequisites

- Python 3.8 or later
- A C++ compiler with C++17 support (e.g. GCC, Clang, MSVC)
- CMake ≥ 3.15
- Eigen3 (header-only)
- OpenMP

With **conda**, all of these are provided by the `environment.yml` shipped in the repository. With plain **pip**, you have to install CMake, a C++17 compiler and Eigen3 yourself (for example `apt install cmake g++ libeigen3-dev` on Debian/Ubuntu, or `pip install cmake` if you only need CMake from pip).

### Install with conda (recommended)

```bash
conda env create -f environment.yml
conda activate frgfp-env
pip install .
```

### Install with pip (non-conda)

```bash
pip install -r requirements.txt
pip install .
```

The `requirements.txt` file lists the complete Python dependency set (runtime, testing, documentation and the build backend). CMake, Eigen3 and a C++17 compiler still have to be installed separately.

### Development

```bash
pip install -e .[dev]
```

The `dev` extra installs the package in editable mode together with the test, benchmark and documentation tooling, so no further `pip` step is needed before running the tests or building the documentation.

## Quick example

```python
import numpy as np
from frgfp import ChebyshevAnsatz, uniform_collgrid, FuncFromAnsatz

# Define a 1D Chebyshev ansatz of order 5 on [-1, 1].
ansatz = ChebyshevAnsatz(inv_dim=1, order=5, interval=(-1.0, 1.0))

# Generate a uniform collocation grid with 20 points.
grid = uniform_collgrid(inv_dim=1, num_points=20, interval=(-1.0, 1.0))

# Create a function from a known coefficient vector.
coeffs = np.random.rand(ansatz.num_coeffs)
func = FuncFromAnsatz(ansatz=ansatz, coeffs=coeffs)

# Evaluate on the grid.
vals = func.evaluate(grid)
print(vals)
```

Fixed points of LPA flow equations are found with `frgfp.lpa.LPACollSolver`; see the documentation for a complete example.

## Examples

A complete, runnable walkthrough of the whole public API — the O(N) Wilson–Fisher flow at order $\mathcal{O}(\bar{\rho}^8)$ in the local potential approximation — is available as a Jupyter notebook:

- [`examples/usage.ipynb`](examples/usage.ipynb) — basis expansions, collocation grids, function expansions and the LPA fixed-point solver, including a comparison of the fixed point with the perturbative input.

The same content is rendered in the [usage section](doc/source/usage.rst) of the documentation.

## Documentation

The Sphinx documentation in `doc/` is generated from the NumPy-style docstrings in the source code. The documentation tooling is provided by the `dev` extra:

```bash
pip install .[dev]
sphinx-build -b html doc/source doc/build/html
```

or, equivalently:

```bash
make -C doc html
```

The build imports the package, so it must run in an environment where the compiled extensions are installed.

## License

This project is licensed under the MIT License.
