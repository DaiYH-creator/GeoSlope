# -*- coding: utf-8 -*-
"""Circular slip surface generation and critical surface search."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SlipSurface:
    """Circular slip surface defined by center (xc, yc) and radius r."""
    xc: float
    yc: float
    r: float
    fos: float = 999.0

    def y_at(self, x: float, lower: bool = True) -> Optional[float]:
        dx = x - self.xc
        if abs(dx) > self.r:
            return None
        dy = np.sqrt(self.r**2 - dx**2)
        return self.yc - dy if lower else self.yc + dy


def build_ground_profile(
    slope_height: float, slope_angle: float,
    crest_width: float = 10.0, toe_width: float = 10.0,
    n_pts: int = 200,
) -> np.ndarray:
    """Build ground surface profile as (N,2) array sorted by x."""
    rad = np.radians(slope_angle)
    h_tan = slope_height / np.tan(rad)
    x_toe_end = toe_width
    x_crest_start = toe_width + h_tan
    x_end = x_crest_start + crest_width

    pts = []
    pts.append([0.0, 0.0])
    for x in np.linspace(0, x_toe_end, max(5, int(n_pts * toe_width / x_end))):
        pts.append([x, 0.0])
    for x in np.linspace(x_toe_end, x_crest_start, max(15, int(n_pts * h_tan / x_end))):
        y = (x - x_toe_end) / h_tan * slope_height
        pts.append([x, y])
    for x in np.linspace(x_crest_start, x_end, max(5, int(n_pts * crest_width / x_end))):
        pts.append([x, slope_height])

    return np.array(pts)


def _circle_from_two_points_and_tangent(
    x1: float, y1: float,   # toe point
    x2: float, y2: float,   # entry/exit point
    tangent_angle: float,   # tangent angle at point 1 (radians from horizontal)
) -> Optional[Tuple[float, float, float]]:
    """Compute circle passing through (x1,y1) with given tangent and (x2,y2).

    Center is at intersection of:
    - Perpendicular to tangent at (x1,y1): line through (x1,y1) with direction (-sin, cos)
    - Perpendicular bisector of [(x1,y1), (x2,y2)]

    Returns (xc, yc, r) or None.
    """
    tx, ty = np.cos(tangent_angle), np.sin(tangent_angle)
    # Normal to tangent at point 1
    nx, ny = -ty, tx

    # Perpendicular bisector midpoint
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    # Direction of chord
    dx_c, dy_c = x2 - x1, y2 - y1
    chord_len = np.hypot(dx_c, dy_c)
    if chord_len < 1e-10:
        return None
    # Normal to chord
    cx, cy = -dy_c / chord_len, dx_c / chord_len

    # Solve: (x1,y1) + s*(nx,ny) = (mx,my) + t*(cx,cy)
    # nx*s - cx*t = mx - x1
    # ny*s - cy*t = my - y1
    det = nx * cy - ny * cx
    if abs(det) < 1e-10:
        return None
    s = ((mx - x1) * cy - (my - y1) * cx) / det
    xc = x1 + s * nx
    yc = y1 + s * ny
    r = np.hypot(xc - x1, yc - y1)
    if r <= 0 or r > 1e6:
        return None
    return xc, yc, r


def _find_ground_intersections(
    xc: float, yc: float, r: float, ground: np.ndarray
) -> List[Tuple[float, float]]:
    """Find all intersection points between circle and ground surface."""
    all_pts = []
    for i in range(len(ground) - 1):
        x1, y1 = ground[i]
        x2, y2 = ground[i + 1]
        # Skip segments entirely outside x-range of circle
        if max(x1, x2) < xc - r or min(x1, x2) > xc + r:
            continue
        pts = _line_circle_intersect(x1, y1, x2, y2, xc, yc, r)
        all_pts.extend(pts)
    # Deduplicate by x
    if len(all_pts) <= 1:
        return all_pts
    # Sort by x, remove near-duplicates
    all_pts.sort(key=lambda p: p[0])
    filtered = [all_pts[0]]
    for p in all_pts[1:]:
        if abs(p[0] - filtered[-1][0]) > 1e-4:
            filtered.append(p)
    return filtered


def search_critical_circle(
    ground: np.ndarray,
    slope_height: float,
    slope_angle: float,
    toe_width: float,
    crest_width: float,
    c: float,
    phi: float,
    gamma: float,
    method: str = "bishop",
    n_tangents: int = 7,
    n_entries: int = 20,
) -> SlipSurface:
    """Search for critical circular slip surface.

    Parameterization: circle through toe (x_toe, 0) with variable tangent
    and variable entry point on ground surface.

    Parameters
    ----------
    ground, slope_height, slope_angle, toe_width, crest_width : geometry
    c, phi, gamma : material
    method : "bishop" | "janbu"
    n_tangents : number of tangent directions to try
    n_entries : number of entry x positions to try
    """
    x_toe = toe_width
    y_toe = 0.0
    rad = np.radians(slope_angle)
    h_tan = slope_height / np.tan(rad)
    x_crest_start = toe_width + h_tan
    x_crest_end = x_crest_start + crest_width

    best = SlipSurface(xc=0, yc=0, r=1, fos=999)

    # Entry points: from just past toe to end of crest
    entry_xs = np.linspace(x_toe + 0.5, x_crest_end, n_entries)
    # Tangent angles at toe: from near-horizontal to steep upward
    tangent_angles = np.linspace(np.radians(5), np.radians(85), n_tangents)

    tested = 0
    for x_entry in entry_xs:
        y_entry = np.interp(x_entry, ground[:, 0], ground[:, 1])
        if y_entry < 0.01:
            continue

        for theta in tangent_angles:
            circle = _circle_from_two_points_and_tangent(
                x_toe, y_toe, x_entry, y_entry, theta
            )
            if circle is None:
                continue
            xc, yc, r = circle
            # Filter unreasonable circles
            if yc < slope_height or yc > 5 * slope_height:
                continue
            if r < 0.3 * slope_height or r > 5 * slope_height:
                continue

            # Verify two ground intersections
            pts = _find_ground_intersections(xc, yc, r, ground)
            if len(pts) < 2:
                continue

            surface = SlipSurface(xc=xc, yc=yc, r=r)
            fos = _compute_fos_for_surface(surface, ground, c, phi, gamma, method)
            tested += 1

            if fos is not None and fos < best.fos:
                best.fos = fos
                best.xc, best.yc, best.r = xc, yc, r

    # Also try with the tangent at the ENTRY point (not toe), and circle through toe
    # This catches different circle families
    entry_tangents = np.linspace(np.radians(-60), np.radians(60), n_tangents)
    for x_entry in entry_xs:
        y_entry = np.interp(x_entry, ground[:, 0], ground[:, 1])
        if y_entry < 0.01:
            continue
        for theta in entry_tangents:
            circle = _circle_from_two_points_and_tangent(
                x_entry, y_entry, x_toe, y_toe, theta
            )
            if circle is None:
                continue
            xc, yc, r = circle
            if yc < slope_height or yc > 5 * slope_height:
                continue
            if r < 0.3 * slope_height or r > 5 * slope_height:
                continue
            pts = _find_ground_intersections(xc, yc, r, ground)
            if len(pts) < 2:
                continue
            surface = SlipSurface(xc=xc, yc=yc, r=r)
            fos = _compute_fos_for_surface(surface, ground, c, phi, gamma, method)
            tested += 1
            if fos is not None and fos < best.fos:
                best.fos = fos
                best.xc, best.yc, best.r = xc, yc, r

    return best


def _compute_fos_for_surface(
    surface: SlipSurface, ground: np.ndarray,
    c: float, phi: float, gamma: float, method: str,
    n_slices: int = 30,
) -> Optional[float]:
    if method == "bishop":
        from .bishop import bishop_fos
        return bishop_fos(surface, ground, c, phi, gamma, n_slices)
    elif method == "janbu":
        from .janbu import janbu_fos
        return janbu_fos(surface, ground, c, phi, gamma, n_slices)
    else:
        from .bishop import bishop_fos
        return bishop_fos(surface, ground, c, phi, gamma, n_slices)


def slice_mass(
    surface: SlipSurface, ground: np.ndarray, n_slices: int = 30,
) -> Tuple[Optional[np.ndarray], ...]:
    """Divide sliding mass into vertical slices.

    Returns (mid_x, widths, heights, alphas) or (None,)*4.
    """
    # Find intersections of circle with ground
    pts = _find_ground_intersections(surface.xc, surface.yc, surface.r, ground)
    if len(pts) < 2:
        return None, None, None, None

    x_left, x_right = pts[0][0], pts[-1][0]
    if x_right - x_left < 1e-6:
        return None, None, None, None

    dx = (x_right - x_left) / n_slices
    slice_edges = np.linspace(x_left, x_right, n_slices + 1)
    slice_mid_x = (slice_edges[:-1] + slice_edges[1:]) / 2.0
    slice_widths = np.full(n_slices, dx)

    heights = np.zeros(n_slices)
    alphas = np.zeros(n_slices)

    for i in range(n_slices):
        x_mid = slice_mid_x[i]
        y_ground = np.interp(x_mid, ground[:, 0], ground[:, 1])
        y_slip = surface.y_at(x_mid, lower=True)
        if y_slip is None:
            return None, None, None, None
        h = y_ground - y_slip
        heights[i] = max(h, 0.0)

        dx_eps = dx * 0.01
        y_l = surface.y_at(x_mid - dx_eps, lower=True)
        y_r = surface.y_at(x_mid + dx_eps, lower=True)
        if y_l is not None and y_r is not None:
            alphas[i] = np.arctan2(y_r - y_l, 2 * dx_eps)
        else:
            alphas[i] = 0.0

    return slice_mid_x, slice_widths, heights, alphas


def _line_circle_intersect(
    x1: float, y1: float, x2: float, y2: float,
    xc: float, yc: float, r: float,
) -> List[Tuple[float, float]]:
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - xc, y1 - yc
    a = dx*dx + dy*dy
    if a < 1e-15:
        return []
    b = 2.0 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    disc = b*b - 4.0*a*c
    if disc < 0:
        return []
    pts = []
    for sign in [-1, 1]:
        t = (-b + sign * np.sqrt(disc)) / (2.0 * a)
        if -1e-10 <= t <= 1.0 + 1e-10:
            t = max(0.0, min(1.0, t))
            pts.append((x1 + t*dx, y1 + t*dy))
    return pts
