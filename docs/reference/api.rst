API Reference
=============

.. automodule:: materforge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Core Module
-----------

.. automodule:: materforge.core
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Material Class
~~~~~~~~~~~~~~

.. autoclass:: materforge.Material
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Bundled Materials
~~~~~~~~~~~~~~~~~

Helpers for the reference materials shipped with the package. See
:doc:`../how-to/load_bundled_materials` for usage.

.. autofunction:: materforge.list_materials
   :no-index:

.. autofunction:: materforge.load_material
   :no-index:

.. autofunction:: materforge.get_material_path
   :no-index:

Fast Evaluation
~~~~~~~~~~~~~~~

A compiled, reusable evaluator for sweeping many dependency values or evaluating
over a NumPy array. See :doc:`../how-to/fast_evaluation` for usage.

.. autoclass:: materforge.MaterialEvaluator
   :members:
   :show-inheritance:
   :no-index:

Fit Quality
~~~~~~~~~~~

Goodness-of-fit metrics (R², RMSE, MAE, residuals) for data-backed properties.
See :doc:`../how-to/assess_fit_quality` for usage.

.. automodule:: materforge.analysis
   :members:
   :show-inheritance:
   :no-index:

Command-Line Interface
----------------------

The ``materforge`` command wraps the public API for use from a shell. See the
:doc:`../how-to/use_the_cli` guide for the subcommands and examples.

.. autofunction:: materforge.cli.main
   :no-index:

.. autofunction:: materforge.cli.validate_entry
   :no-index:

Algorithms Module
-----------------

.. automodule:: materforge.algorithms
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Parsing Module
--------------

.. automodule:: materforge.parsing
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Visualization Module
--------------------

Parse-time plotting - the composite figure written during a build.

.. automodule:: materforge.visualization
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Post-build plotting helpers that return a Matplotlib ``Axes`` (fit, residual, and
compare plots). See :doc:`../how-to/visualize_properties` for usage.

.. automodule:: materforge.visualization.plots
   :members:
   :show-inheritance:
   :no-index:
