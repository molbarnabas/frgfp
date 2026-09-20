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

- Python 3.10 or later
- A C++ compiler with C++17 support (e.g. GCC, Clang, MSVC)
- CMake ≥ 3.15
- OpenMP (mandatory: the LPA backend calls the OpenMP runtime directly)
- Eigen3 (header-only, optional: fetched automatically when missing)

With **conda**, all of these are provided by the `environment.yml` shipped in the repository. With plain **pip**, you have to install CMake and a C++17 compiler yourself (for example `apt install cmake g++ libeigen3-dev` on Debian/Ubuntu, or `pip install cmake` if you only need CMake from pip). Eigen3 is optional: when it is not found, the build downloads a pinned copy. On macOS, OpenMP comes from Homebrew (`brew install libomp`) and the build locates it automatically.

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

### Development with Docker

A ready-made Linux development container is included (`compose.yaml` and `docker/Dockerfile`). It provides the full toolchain - C++17 compiler, CMake, Ninja, Eigen3, OpenMP and git - plus every Python dependency, so nothing else has to be installed on the host.

```bash
./docker/dev.sh build                    # build the image
./docker/dev.sh up -d                    # start the container
./docker/dev.sh exec dev bash            # open a shell inside it

# from inside the container
pip install -e .                         # editable install (done automatically on first start)
pytest                                   # run the test suite
make -C doc html                         # generate the documentation
python -m http.server 8000 --directory doc/build/html
```

The repository is bind-mounted at `/workspace`, so the container and the host share one working tree **and** one `.git` directory: edit and commit on the host, build and test in the container. Because the mount covers the whole repository, `doc/build` is shared too - the HTML generated in the container is immediately available on the host at <http://localhost:8000>. The CMake/scikit-build cache is kept in a named volume (drop it with `./docker/dev.sh down -v`), so it never clashes with a `build/` directory produced on the host.

If you prefer to drive `docker compose` directly, pass your UID/GID explicitly so generated files are not owned by root:

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build
docker compose up -d
```

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

The documentation is published at <https://molbarnabas.github.io/frgfp/latest/>.
Every release keeps its own copy (`/latest/`, `/0.2.0/`, `/0.2.0rc1/`, `/dev/`, …)
and the version switcher in the navigation bar moves between them, exactly like
the SciPy documentation. `latest` always points at the newest **final** release;
release candidates are published but never become `latest`.

The Sphinx sources live in `doc/` and are generated from the NumPy-style docstrings in the source code. The documentation tooling is provided by the `dev` extra:

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
