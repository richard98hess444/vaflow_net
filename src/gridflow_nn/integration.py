"""Dimension-agnostic trajectory integrators over a vector field.

The integrators take a ``field_fn(p) -> vector`` callable, so the same code
works for 2D ``(row, col)`` and 3D ``(i, j, k)`` fields. They follow the
(unit-normalized) flow field from ``start`` toward ``goal``, stopping when the
path gets within ``stop_radius`` of the goal or the field vanishes. Used by
``predict_path`` / ``predict_path_3d`` to trace a path from a trained model.
"""

import numpy as np


def _unit(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else np.zeros_like(v)


def eul_integrate(field_fn, start, goal, step_size=2.0, max_steps=50, stop_radius=1.0):
    """Follow the field with forward Euler; returns an (M, D) trajectory array."""
    p = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    traj = [p.copy()]
    for _ in range(max_steps):
        if np.linalg.norm(p - goal) < stop_radius:
            break
        v = _unit(field_fn(p))
        if np.linalg.norm(v) < 1e-6:
            break
        p = p + step_size * v
        traj.append(p.copy())
    return np.array(traj)


def rk4_integrate(field_fn, start, goal, step_size=2.0, max_steps=50, stop_radius=1.0):
    """Follow the field with 4th-order Runge-Kutta; returns an (M, D) trajectory."""
    p = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    traj = [p.copy()]
    for _ in range(max_steps):
        if np.linalg.norm(p - goal) < stop_radius:
            break
        k1 = _unit(field_fn(p))
        k2 = _unit(field_fn(p + 0.5 * step_size * k1))
        k3 = _unit(field_fn(p + 0.5 * step_size * k2))
        k4 = _unit(field_fn(p + step_size * k3))
        step = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if np.linalg.norm(step) < 1e-6:
            break
        p = p + step_size * step
        traj.append(p.copy())
    return np.array(traj)


_INTEGRATORS = {"euler": eul_integrate, "rk4": rk4_integrate}


def integrate_path(field_fn, start, goal, method="rk4", **kwargs):
    """Trace a path with ``method`` ("rk4" or "euler") over ``field_fn``."""
    try:
        integrator = _INTEGRATORS[method]
    except KeyError:
        raise ValueError(
            f"Unknown method {method!r}; expected one of {sorted(_INTEGRATORS)}."
        )
    return integrator(field_fn, start, goal, **kwargs)
