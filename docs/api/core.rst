Core Geometry
=============

.. note::

   For theory behind these functions, see :doc:`/guide/refractive_geometry` and :doc:`/guide/coordinates`.

The core package provides refractive ray tracing, camera models, and geometric primitives.

Refractive Projection
---------------------

.. autofunction:: aquacal.core.refractive_geometry.refractive_project

.. autofunction:: aquacal.core.refractive_geometry.refractive_project_batch

.. autofunction:: aquacal.core.refractive_geometry.refractive_project_fast

Snell's Law and Ray Tracing
----------------------------

.. autofunction:: aquacal.core.refractive_geometry.snells_law_3d

.. autofunction:: aquacal.core.refractive_geometry.trace_ray_air_to_water

.. autofunction:: aquacal.core.refractive_geometry.refractive_back_project

Camera Models
-------------

.. autoclass:: aquacal.core.camera.Camera
   :members:
   :show-inheritance:

.. autofunction:: aquacal.core.camera.undistort_points

Board Geometry
--------------

.. automodule:: aquacal.core.board
   :members:
   :show-inheritance:

Interface Model
---------------

.. automodule:: aquacal.core.interface_model
   :members:
   :show-inheritance:
