"""gridflow_sim - Grid-world simulation for the neural vector-field robot.

The simulator core (world generation, local scan map, goal projection,
reachability, and local flow prediction) lives in :mod:`gridflow_sim.simulator`
and is exported here. The optional rendering / video pieces live in
:mod:`gridflow_sim.render` and must be imported explicitly, so consumers that
only need the simulation logic do not pull in the video-recording helpers.
"""

from .simulator import GridFlowSimulator, DIRECTIONS

__all__ = ["GridFlowSimulator", "DIRECTIONS"]
