Core
====

The core submodule (``frgfp._core``) provides the foundational structures of the
package: the basis expansions (ansätze), the collocation grid generators and the
function evaluator. The heavy lifting is delegated to the Eigen C++ backend,
and the public objects are re-exported from the top-level :mod:`frgfp`
namespace.

.. autosummary::
   :toctree: generated

   frgfp.PolyAnsatz
   frgfp.ChebyshevAnsatz
   frgfp.FuncFromAnsatz
   frgfp.uniform_collgrid
   frgfp.chebyshev_collgrid
