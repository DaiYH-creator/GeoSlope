# -*- coding: utf-8 -*-
"""DXF geometry import for GeoSlope.

Reads DXF files and converts polylines/lines/points into SlopeGeometry.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

try:
    from ..geometry import SlopeGeometry, Point, Polygon, MaterialRegion
except ImportError:
    from core.geometry import SlopeGeometry, Point, Polygon, MaterialRegion


def read_dxf(
    filepath: str,
    material_map: Optional[dict] = None,
) -> SlopeGeometry:
    """Import slope geometry from DXF file.

    Reads LWPOLYLINE and POLYLINE entities as polygons.
    Reads LINE entities as construction lines (stored as boundaries).

    Parameters
    ----------
    filepath     : path to .dxf file
    material_map : optional {layer_name: material_name} mapping.
                   Polygons on matching layers get material assignments.

    Returns
    -------
    SlopeGeometry with polygons and material regions populated.
    """
    import ezdxf

    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()
    geometry = SlopeGeometry(name=filepath)

    if material_map is None:
        material_map = {}

    # Collect polygons from polylines
    poly_id = 0
    for entity in msp:
        if entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
            pts = _extract_polyline_points(entity)
            if len(pts) < 3:
                continue

            # Ensure CCW order
            if _is_clockwise(pts):
                pts.reverse()

            vertices = [Point(x, y) for x, y in pts]
            polygon = Polygon(vertices=vertices, material_id=poly_id)
            geometry.polygons.append(polygon)

            # Check layer for material
            layer = entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
            if layer in material_map:
                geometry.material_regions.append(
                    MaterialRegion(polygon, material_map[layer])
                )
            else:
                geometry.material_regions.append(
                    MaterialRegion(polygon, f"soil_{poly_id}")
                )
            poly_id += 1

    # Fallback: if no polylines found, try to build from lines
    if len(geometry.polygons) == 0:
        _build_from_lines(geometry, msp)

    return geometry


def read_xy(filepath: str) -> SlopeGeometry:
    """Import from simple XY point list file.

    Format (one polygon per block, separated by blank line):
        x1 y1
        x2 y2
        ...
    """
    geometry = SlopeGeometry(name=filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        blocks = f.read().strip().split('\n\n')

    for bid, block in enumerate(blocks):
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        pts = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
        if len(pts) < 3:
            continue

        if _is_clockwise(pts):
            pts.reverse()
        vertices = [Point(x, y) for x, y in pts]
        polygon = Polygon(vertices=vertices, material_id=bid)
        geometry.polygons.append(polygon)
        geometry.material_regions.append(
            MaterialRegion(polygon, f"soil_{bid}")
        )

    return geometry


def _extract_polyline_points(entity) -> List[Tuple[float, float]]:
    """Extract (x, y) points from LWPOLYLINE or POLYLINE."""
    import ezdxf
    pts = []
    if entity.dxftype() == 'LWPOLYLINE':
        with entity.points() as points:
            for p in points:
                pts.append((p[0], p[1]))
    else:  # POLYLINE
        for v in entity.vertices:
            pts.append((v.dxf.location.x, v.dxf.location.y))
    return pts


def _is_clockwise(pts: List) -> bool:
    """Check if polygon vertices are clockwise."""
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        area += (x2 - x1) * (y2 + y1)
    return area > 0


def _build_from_lines(geometry: SlopeGeometry, msp) -> None:
    """Attempt to build a polygon from LINE entities (simplified)."""
    lines = []
    for entity in msp:
        if entity.dxftype() == 'LINE':
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            lines.append((start, end))

    if not lines:
        return

    # Simple approach: take all endpoints and compute convex hull
    all_pts = []
    for s, e in lines:
        all_pts.append(s)
        all_pts.append(e)

    pts_arr = np.array(all_pts)
    unique_pts = np.unique(pts_arr.round(decimals=6), axis=0)

    if len(unique_pts) < 3:
        return

    # Convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(unique_pts)
    hull_pts = unique_pts[hull.vertices]

    vertices = [Point(x, y) for x, y in hull_pts]
    geometry.polygons.append(Polygon(vertices=vertices, material_id=0))
