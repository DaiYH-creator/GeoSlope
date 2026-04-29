# -*- coding: utf-8 -*-
"""Mesh generation using Delaunay triangulation."""

from __future__ import annotations
import numpy as np
from scipy.spatial import Delaunay
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .geometry import SlopeGeometry, Point, Polygon


@dataclass
class Mesh:
    """2D triangular mesh.

    Attributes
    ----------
    nodes       : (N, 2) float array — node coordinates
    elements    : (M, 3) int array   — element connectivity (node indices, CCW)
    elem_materials : (M,) int        — material ID per element
    boundary_nodes : set of int      — global IDs of boundary nodes (hull)
    """
    nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    elements: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    elem_materials: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    boundary_nodes: set = field(default_factory=set)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def element_centers(self) -> np.ndarray:
        n0 = self.nodes[self.elements[:, 0]]
        n1 = self.nodes[self.elements[:, 1]]
        n2 = self.nodes[self.elements[:, 2]]
        return (n0 + n1 + n2) / 3.0


def generate_mesh(
    geometry: SlopeGeometry,
    max_area: Optional[float] = None,
    refine_factor: float = 1.0,
    boundary_segments: int = 20,
) -> Mesh:
    """Generate a triangular mesh from slope geometry.

    Parameters
    ----------
    geometry          : SlopeGeometry
    max_area          : float, max element area; auto if None
    refine_factor     : float, 0 < factor <= 1, smaller → finer mesh
    boundary_segments : int, number of segments per polygon edge
    """
    bbox = geometry.bounding_box()
    if max_area is None:
        diag = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
        max_area = (diag * 0.05) ** 2 * refine_factor
    spacing = np.sqrt(max_area)

    # Build point set: polygon vertices + edge points + interior grid
    all_pts_set = set()
    boundary_tag = {}  # node coordinates -> "bottom"/"left"/"right"/"top"/"free"

    for poly in geometry.polygons:
        verts = poly.vertices
        nv = len(verts)

        # Add edge points
        for i in range(nv):
            p0 = verts[i]
            p1 = verts[(i + 1) % nv]
            nseg = max(2, int(p0.distance(p1) / spacing))
            for t in np.linspace(0, 1, nseg + 1):
                x = p0.x + t * (p1.x - p0.x)
                y = p0.y + t * (p1.y - p0.y)
                key = (round(x, 6), round(y, 6))
                all_pts_set.add(key)
                # Determine boundary type
                tag = _edge_tag(x, y, bbox)
                if key in boundary_tag:
                    if boundary_tag[key] != tag and tag != "free":
                        boundary_tag[key] = tag
                else:
                    boundary_tag[key] = tag

        # Add interior grid points
        xs = [v.x for v in verts]
        ys = [v.y for v in verts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        for x in np.arange(xmin + spacing, xmax, spacing):
            for y in np.arange(ymin + spacing, ymax, spacing):
                if poly.contains_point(Point(x, y)):
                    all_pts_set.add((round(x, 6), round(y, 6)))

    # Convert to array
    all_pts = np.array(list(all_pts_set))
    del all_pts_set

    # Delaunay triangulation
    tri = Delaunay(all_pts)

    mesh = Mesh()
    mesh.nodes = all_pts
    mesh.elements = tri.simplices

    # Assign materials
    centroids = mesh.element_centers()
    n_elem = len(centroids)
    material_ids = np.zeros(n_elem, dtype=int)
    for i, (cx, cy) in enumerate(centroids):
        pt = Point(cx, cy)
        assigned = False
        for poly in geometry.polygons:
            if poly.contains_point(pt):
                material_ids[i] = poly.material_id
                assigned = True
                break
        if not assigned:
            material_ids[i] = poly.material_id  # fallback
    mesh.elem_materials = material_ids

    # Identify boundary nodes via hull edges
    mesh.boundary_nodes = _find_boundary_nodes(mesh)

    return mesh


def _edge_tag(x: float, y: float, bbox: tuple) -> str:
    """Classify boundary edge: bottom, left, right, top, or free."""
    xmin, xmax, ymin, ymax = bbox
    eps = 1e-3
    if abs(y - ymin) < eps:
        return "bottom"
    elif abs(x - xmin) < eps:
        return "left"
    elif abs(x - xmax) < eps:
        return "right"
    elif abs(y - ymax) < eps:
        return "top"
    return "free"


def _find_boundary_nodes(mesh: Mesh) -> set:
    """Identify boundary nodes via edge counting (hull edges count=1)."""
    from collections import Counter
    edge_count = Counter()
    for elem in mesh.elements:
        i, j, k = elem
        for a, b in [(i, j), (j, k), (k, i)]:
            edge = (min(a, b), max(a, b))
            edge_count[edge] += 1
    boundary_nodes = set()
    for (a, b), c in edge_count.items():
        if c == 1:
            boundary_nodes.add(a)
            boundary_nodes.add(b)
    return boundary_nodes
