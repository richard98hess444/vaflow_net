"""3D counterparts of the early gridflow algorithms (branch: 3d_grid).

Mirrors :mod:`gridflow.algorithms` for volumetric ``(n, n, n)`` grids. The 2D
classes are left untouched; use these ``*3D`` classes for 3D work.
"""

from .bfs import DIRS_3D, BreadthFirstSearch3D
from .flow import WaterFlow3D
from .eikonal import Eikonal3D
from .sdf import SignedDistanceField3D
from .guidance import VectorFieldGuidance3D
from .astar import AStar3D

__all__ = [
    "DIRS_3D",
    "BreadthFirstSearch3D",
    "WaterFlow3D",
    "Eikonal3D",
    "SignedDistanceField3D",
    "VectorFieldGuidance3D",
    "AStar3D",
]
