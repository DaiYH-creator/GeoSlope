# -*- coding: utf-8 -*-
"""Geometry module — Point, Line, Polygon, SlopeGeometry."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Point:
    x: float
    y: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def distance(self, other: Point) -> float:
        return np.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Line:
    start: Point
    end: Point

    def length(self) -> float:
        return self.start.distance(self.end)

    def midpoint(self) -> Point:
        return Point((self.start.x + self.end.x) / 2,
                     (self.start.y + self.end.y) / 2)


@dataclass
class Polygon:
    """Polygon by ordered vertices (counter-clockwise)."""
    vertices: List[Point]
    material_id: int = 0

    def area(self) -> float:
        """Shoelace formula."""
        pts = np.array([[p.x, p.y] for p in self.vertices])
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def centroid(self) -> Point:
        pts = np.array([[p.x, p.y] for p in self.vertices])
        return Point(float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    def contains_point(self, p: Point) -> bool:
        """Ray-casting algorithm."""
        pts = np.array([[v.x, v.y] for v in self.vertices])
        n = len(pts)
        inside = False
        px, py = p.x, p.y
        j = n - 1
        for i in range(n):
            if ((pts[i, 1] > py) != (pts[j, 1] > py)) and \
               (px < (pts[j, 0] - pts[i, 0]) * (py - pts[i, 1]) /
                (pts[j, 1] - pts[i, 1]) + pts[i, 0]):
                inside = not inside
            j = i
        return inside


@dataclass
class MaterialRegion:
    """Associate a polygon with a material name."""
    polygon: Polygon
    material_name: str


@dataclass
class SlopeGeometry:
    """Defines a 2D slope geometry."""
    name: str = "Untitled Slope"
    polygons: List[Polygon] = field(default_factory=list)
    material_regions: List[MaterialRegion] = field(default_factory=list)
    boundary_points: List[Point] = field(default_factory=list)
    description: str = ""

    def bounding_box(self) -> Tuple[float, float, float, float]:
        all_pts = []
        for poly in self.polygons:
            all_pts.extend(poly.vertices)
        if not all_pts:
            return (0, 0, 10, 10)
        xs = [p.x for p in all_pts]
        ys = [p.y for p in all_pts]
        return (min(xs), max(xs), min(ys), max(ys))

    def max_dimension(self) -> float:
        xmin, xmax, ymin, ymax = self.bounding_box()
        return max(xmax - xmin, ymax - ymin)

    def build_slope_example(self,
                            slope_height: float = 10.0,
                            slope_angle: float = 45.0,
                            crest_width: float = 10.0,
                            toe_width: float = 10.0):
        """Convenience: build a simple homogeneous slope."""
        import math
        rad = math.radians(slope_angle)
        h_tan = slope_height / math.tan(rad)
        self.name = f"Slope H={slope_height}m {slope_angle}deg"
        x0 = 0.0
        x1 = toe_width
        x2 = x1 + h_tan
        x3 = x2 + crest_width
        y0 = 0.0
        y1 = slope_height
        y2 = slope_height
        y3 = 0.0
        pts = [
            Point(0, 0), Point(x3, 0), Point(x3, y2 + 5),
            Point(0, y2 + 5)
        ]
        self.polygons = [Polygon(vertices=pts, material_id=0)]
        bps = [Point(x1, y0), Point(x2, y1), Point(x3, y2)]
        for bp in bps:
            self.boundary_points.append(bp)
