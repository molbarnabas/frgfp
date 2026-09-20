API Reference
=============

The public API of FRGfp is organised into the top-level :mod:`frgfp` namespace
and its submodules. The most common objects are re-exported directly from the
top-level package for convenience, while the submodules group the functionality
by topic.

.. toctree::
   :maxdepth: 1

   core
   lpa

Convenience imports
-------------------

The following objects can be imported directly from :mod:`frgfp`:

.. autosummary::
   :nosignatures:

   frgfp.PolyAnsatz
   frgfp.ChebyshevAnsatz
   frgfp.FuncFromAnsatz
   frgfp.uniform_collgrid
   frgfp.chebyshev_collgrid
