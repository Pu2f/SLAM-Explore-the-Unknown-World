"""Wall maps of a grid maze.

``WallMap``  - a plain map: which cells exist and the state of each wall.
               Used for the ground truth, the simulator world and evaluation.
``MazeMap``  - the robot's own map: sparse, grows in any direction, and keeps
               an evidence score per wall so noisy readings can be outvoted.

Each wall is stored once, under the cell to its south / west (key direction
N or E), so two neighbouring cells can never disagree about a shared wall.
Coordinates follow DESIGN.md section 3 (+y = N, +x = E).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterator, Optional, Set, Tuple

from .config import DEFAULT, MapEvidence
from .geometry import Cell, Direction, neighbor

EdgeKey = Tuple[int, int, Direction]  # direction is always N or E


class EdgeState(str, Enum):
    UNKNOWN = "unknown"
    WALL = "wall"
    OPEN = "open"


def edge_key(cell: Cell, d: Direction) -> EdgeKey:
    if d in (Direction.N, Direction.E):
        return (cell[0], cell[1], d)
    n = neighbor(cell, d)
    return (n[0], n[1], d.opposite())


def edge_cells(key: EdgeKey) -> Tuple[Cell, Cell]:
    """The two cells separated by an edge."""
    cell = (key[0], key[1])
    return cell, neighbor(cell, key[2])


def rotate_cell(cell: Cell, quarter_turns_cw: int) -> Cell:
    """Rotate cell coordinates clockwise about the origin (N -> E -> S -> W)."""
    x, y = cell
    for _ in range(quarter_turns_cw % 4):
        x, y = y, -x
    return (x, y)


class MapFormatError(ValueError):
    pass


class WallMap:
    def __init__(self) -> None:
        self.cells: Set[Cell] = set()
        self._edges: Dict[EdgeKey, EdgeState] = {}

    # ---- edges ---------------------------------------------------------------
    def get(self, cell: Cell, d: Direction) -> EdgeState:
        return self._edges.get(edge_key(cell, d), EdgeState.UNKNOWN)

    def set(self, cell: Cell, d: Direction, state: EdgeState) -> None:
        key = edge_key(cell, d)
        if state == EdgeState.UNKNOWN:
            self._edges.pop(key, None)
        else:
            self._edges[key] = state

    def sides(self, cell: Cell) -> Tuple[EdgeState, EdgeState, EdgeState, EdgeState]:
        """States of the N, E, S, W sides."""
        return tuple(self.get(cell, d) for d in Direction)  # type: ignore[return-value]

    def edges(self) -> Iterator[Tuple[EdgeKey, EdgeState]]:
        return iter(sorted(self._edges.items()))

    def cell_edge_keys(self) -> Set[EdgeKey]:
        """Every edge that bounds at least one cell of the map."""
        return {edge_key(c, d) for c in self.cells for d in Direction}

    # ---- shape -----------------------------------------------------------------
    def bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """(min_x, min_y, max_x, max_y) of the cells, None if empty."""
        if not self.cells:
            return None
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return min(xs), min(ys), max(xs), max(ys)

    def size(self) -> Tuple[int, int]:
        b = self.bounds()
        if b is None:
            return (0, 0)
        return b[2] - b[0] + 1, b[3] - b[1] + 1

    def transformed(self, quarter_turns_cw: int, offset: Cell) -> "WallMap":
        """Copy with every cell mapped to rotate_cell(c, k) + offset."""
        k = quarter_turns_cw % 4

        def move(c: Cell) -> Cell:
            r = rotate_cell(c, k)
            return (r[0] + offset[0], r[1] + offset[1])

        out = WallMap()
        out.cells = {move(c) for c in self.cells}
        for (x, y, d), state in self._edges.items():
            out.set(move((x, y)), d.turned(k), state)
        return out

    # ---- serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "cells": sorted([list(c) for c in self.cells]),
            "edges": [
                {"x": x, "y": y, "dir": d.name, "state": s.value}
                for (x, y, d), s in self.edges()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WallMap":
        m = cls()
        m.cells = {(int(c[0]), int(c[1])) for c in data["cells"]}
        for e in data["edges"]:
            m.set((int(e["x"]), int(e["y"])), Direction[e["dir"]], EdgeState(e["state"]))
        return m

    def to_ascii(self, markers: Optional[Dict[Cell, str]] = None) -> str:
        """Render the bounding box of the cells. See DESIGN.md section 4."""
        b = self.bounds()
        if b is None:
            return ""
        min_x, min_y, max_x, max_y = b
        markers = markers or {}
        h_sym = {EdgeState.WALL: "---", EdgeState.OPEN: "   ", EdgeState.UNKNOWN: " ? "}
        v_sym = {EdgeState.WALL: "|", EdgeState.OPEN: " ", EdgeState.UNKNOWN: "?"}

        lines = []
        for y in range(max_y, min_y - 1, -1):
            top = "+"
            row = ""
            for x in range(min_x, max_x + 1):
                top += h_sym[self.get((x, y), Direction.N)] + "+"
                row += v_sym[self.get((x, y), Direction.W)]
                row += markers.get((x, y), "")[:3].center(3)
            row += v_sym[self.get((max_x, y), Direction.E)]
            lines.append(top)
            lines.append(row)
        bottom = "+"
        for x in range(min_x, max_x + 1):
            bottom += h_sym[self.get((x, min_y), Direction.S)] + "+"
        lines.append(bottom)
        return "\n".join(lines)

    @classmethod
    def from_ascii(
        cls, text: str, origin_marker: Optional[str] = None
    ) -> Tuple["WallMap", Dict[Cell, str]]:
        """Parse the ASCII format. Returns (map, markers).

        Cells get x = 0.. from the left and y = 0.. from the bottom, unless
        ``origin_marker`` is given: then the cell holding that marker becomes
        (0, 0). Lines before the first / after the last '+' line are ignored,
        so a legend or header is fine.
        """
        raw = text.splitlines()
        corner_idx = [i for i, ln in enumerate(raw) if ln.startswith("+")]
        if len(corner_idx) < 2:
            raise MapFormatError("need at least two '+' lines")
        block = raw[corner_idx[0] : corner_idx[-1] + 1]
        if len(block) % 2 == 0:
            raise MapFormatError("grid must alternate '+' lines and cell lines")
        height = len(block) // 2
        width = (max(len(ln.rstrip()) for ln in block) - 1) // 4
        if width < 1:
            raise MapFormatError("grid is too narrow")
        line_len = 4 * width + 1
        block = [ln.rstrip().ljust(line_len) for ln in block]

        def h_state(seg: str, where: str) -> EdgeState:
            body = seg.strip()
            if body == "":
                return EdgeState.OPEN
            if body == "?":
                return EdgeState.UNKNOWN
            if set(body) == {"-"}:
                return EdgeState.WALL
            raise MapFormatError(f"bad horizontal wall {seg!r} at {where}")

        def v_state(ch: str, where: str) -> EdgeState:
            if ch == "|":
                return EdgeState.WALL
            if ch == " ":
                return EdgeState.OPEN
            if ch == "?":
                return EdgeState.UNKNOWN
            raise MapFormatError(f"bad vertical wall {ch!r} at {where}")

        m = cls()
        markers: Dict[Cell, str] = {}
        for k in range(height + 1):
            ln = block[2 * k]
            if not ln.startswith("+"):
                raise MapFormatError(f"line {corner_idx[0] + 2 * k + 1} should start with '+'")
            y = height - 1 - k  # this line is the N side of row y
            for x in range(width):
                seg = ln[4 * x + 1 : 4 * x + 4]
                m.set((x, y), Direction.N, h_state(seg, f"line {corner_idx[0] + 2 * k + 1}"))
        for r in range(height):
            ln = block[2 * r + 1]
            y = height - 1 - r
            where = f"line {corner_idx[0] + 2 * r + 2}"
            for x in range(width):
                m.cells.add((x, y))
                m.set((x, y), Direction.W, v_state(ln[4 * x], where))
                mark = ln[4 * x + 1 : 4 * x + 4].strip()
                if mark:
                    markers[(x, y)] = mark
            m.set((width - 1, y), Direction.E, v_state(ln[4 * width], where))

        if origin_marker is not None:
            found = [c for c, mk in markers.items() if mk == origin_marker]
            if len(found) != 1:
                raise MapFormatError(
                    f"expected exactly one {origin_marker!r} marker, found {len(found)}"
                )
            ox, oy = found[0]
            m = m.transformed(0, (-ox, -oy))
            markers = {(c[0] - ox, c[1] - oy): mk for c, mk in markers.items()}
        return m, markers


class MazeMap:
    """The robot's own map, built from evidence. The start cell is (0, 0)."""

    def __init__(self, evidence: MapEvidence = DEFAULT.evidence) -> None:
        self.evidence = evidence
        self.visited: Set[Cell] = set()
        self._score: Dict[EdgeKey, float] = {}
        self._locked: Set[EdgeKey] = set()
        # Wall readings that contradicted a traversed (locked) edge.
        self.conflicts: Dict[EdgeKey, int] = {}

    def observe(self, cell: Cell, d: Direction, is_wall: bool, weight: float) -> EdgeState:
        key = edge_key(cell, d)
        if key in self._locked:
            if is_wall:
                self.conflicts[key] = self.conflicts.get(key, 0) + 1
            return EdgeState.OPEN
        lim = self.evidence.score_limit
        s = self._score.get(key, 0.0) + (weight if is_wall else -weight)
        self._score[key] = max(-lim, min(lim, s))
        return self.state(cell, d)

    def mark_traversed(self, cell: Cell, d: Direction) -> None:
        """The robot drove through this edge, so it is open for good."""
        key = edge_key(cell, d)
        self._locked.add(key)
        self._score[key] = -self.evidence.score_limit

    def mark_visited(self, cell: Cell) -> None:
        self.visited.add(cell)

    def score(self, cell: Cell, d: Direction) -> float:
        return self._score.get(edge_key(cell, d), 0.0)

    def state(self, cell: Cell, d: Direction) -> EdgeState:
        key = edge_key(cell, d)
        if key in self._locked:
            return EdgeState.OPEN
        s = self._score.get(key, 0.0)
        t = self.evidence.wall_threshold
        if s >= t:
            return EdgeState.WALL
        if s <= -t:
            return EdgeState.OPEN
        return EdgeState.UNKNOWN

    def is_cell_known(self, cell: Cell) -> bool:
        return all(self.state(cell, d) != EdgeState.UNKNOWN for d in Direction)

    def cells(self) -> Set[Cell]:
        """Visited cells plus cells reached through an OPEN side of one.

        Cells seen only through a wall (e.g. beyond the outer wall) are not
        part of the map; they would be phantom cells.
        """
        out = set(self.visited)
        for c in self.visited:
            for d in Direction:
                if self.state(c, d) == EdgeState.OPEN:
                    out.add(neighbor(c, d))
        return out

    def to_wallmap(self) -> WallMap:
        m = WallMap()
        m.cells = self.cells()
        for key in m.cell_edge_keys():
            cell = (key[0], key[1])
            m.set(cell, key[2], self.state(cell, key[2]))
        return m

    def to_ascii(self, current: Optional[Cell] = None) -> str:
        markers = {c: "." for c in self.visited}
        markers[(0, 0)] = "S"
        if current is not None and current != (0, 0):
            markers[current] = "R"
        return self.to_wallmap().to_ascii(markers)

    def to_dict(self) -> dict:
        """JSON-able dict. Also readable by WallMap.from_dict (cells + edges)."""
        out = self.to_wallmap().to_dict()
        out["start"] = [0, 0]
        out["visited"] = sorted([list(c) for c in self.visited])
        out["evidence"] = [
            {
                "x": k[0],
                "y": k[1],
                "dir": k[2].name,
                "score": round(s, 4),
                "locked": k in self._locked,
            }
            for k, s in sorted(self._score.items())
        ]
        out["conflicts"] = [
            {"x": k[0], "y": k[1], "dir": k[2].name, "count": n}
            for k, n in sorted(self.conflicts.items())
        ]
        return out

    @classmethod
    def from_dict(cls, data: dict, evidence: MapEvidence = DEFAULT.evidence) -> "MazeMap":
        m = cls(evidence)
        m.visited = {(int(c[0]), int(c[1])) for c in data.get("visited", [])}
        for e in data.get("evidence", []):
            key = (int(e["x"]), int(e["y"]), Direction[e["dir"]])
            m._score[key] = float(e["score"])
            if e.get("locked"):
                m._locked.add(key)
        for e in data.get("conflicts", []):
            m.conflicts[(int(e["x"]), int(e["y"]), Direction[e["dir"]])] = int(e["count"])
        return m
