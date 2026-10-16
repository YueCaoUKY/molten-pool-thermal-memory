"""Readers for OpenFOAM field, point-list and AdditiveFOAM log files.

Fields may be ascii or binary and optionally gzip-compressed (`writeCompression
on`). Only the constructs written by OpenFOAM for the cases in this benchmark
are supported: `uniform`/`nonuniform List<...>` internal and patch values, the
`N{value}` list shorthand, and bare lists such as `constant/polyMesh/points`.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_NCOMP = {"scalar": 1, "vector": 3, "symmTensor": 6, "tensor": 9, "label": 1}
_WS = re.compile(rb"(?:\s+|//[^\n]*\n?|/\*.*?\*/)*", re.S)
_TOKEN = re.compile(rb'"[^"]*"|[A-Za-z_][A-Za-z0-9_.:<>\-]*')
_NUMBER = re.compile(rb"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_COUNT = re.compile(rb"(\d+)\s*([({])")
_PAREN = re.compile(rb"[()]")


def read_bytes(path) -> bytes:
    """Return the file contents, transparently handling a `.gz` sibling."""
    p = Path(path)
    if not p.exists():
        gz = p.with_name(p.name + ".gz")
        if gz.exists():
            p = gz
    data = p.read_bytes()
    return gzip.decompress(data) if p.suffix == ".gz" else data


@dataclass
class FoamField:
    internal: np.ndarray | float
    boundary: dict = field(default_factory=dict)   # patch -> {entry: value}
    header: dict = field(default_factory=dict)

    def patch_value(self, patch: str, n: int | None = None) -> np.ndarray:
        """Face values of `patch` as an array (uniform values are broadcast)."""
        v = self.boundary[patch]["value"]
        if np.ndim(v) == 0 and n is not None:
            return np.full(n, float(v))
        return np.asarray(v)


def _kind(cls: str) -> str:
    for k in ("symmTensor", "Tensor", "Vector", "Scalar", "vector", "scalar", "label"):
        if k in cls:
            return {"Tensor": "tensor", "Vector": "vector", "Scalar": "scalar"}.get(k, k)
    raise ValueError(f"unsupported class {cls!r}")


class _Parser:
    def __init__(self, data: bytes):
        self.d = data
        self.p = 0
        self.header: dict = {}
        self.binary = False
        self.label_dtype = np.dtype(np.int32)
        self.scalar_dtype = np.dtype(np.float64)

    # -- lexical helpers ----------------------------------------------------
    def ws(self) -> None:
        self.p = _WS.match(self.d, self.p).end()

    def peek(self, s: bytes) -> bool:
        self.ws()
        return self.d.startswith(s, self.p)

    def expect(self, s: bytes) -> None:
        self.ws()
        if not self.d.startswith(s, self.p):
            raise ValueError(f"expected {s!r} at byte {self.p}: {self.d[self.p:self.p + 60]!r}")
        self.p += len(s)

    def token(self) -> str:
        self.ws()
        m = _TOKEN.match(self.d, self.p)
        if not m:
            raise ValueError(f"token expected at byte {self.p}: {self.d[self.p:self.p + 60]!r}")
        self.p = m.end()
        return m.group(0).decode().strip('"')

    def number(self) -> float:
        self.ws()
        m = _NUMBER.match(self.d, self.p)
        if not m:
            raise ValueError(f"number expected at byte {self.p}: {self.d[self.p:self.p + 60]!r}")
        self.p = m.end()
        return float(m.group(0))

    def skip_to_semicolon(self) -> str:
        end = self.d.index(b";", self.p)
        txt = self.d[self.p:end].decode(errors="replace").strip()
        self.p = end + 1
        return txt

    def skip_braces(self) -> None:
        self.expect(b"{")
        depth = 1
        while depth:
            i = min(j for j in (self.d.find(b"{", self.p), self.d.find(b"}", self.p)) if j >= 0)
            depth += 1 if self.d[i:i + 1] == b"{" else -1
            self.p = i + 1

    # -- structure -------------------------------------------------------------
    def read_header(self) -> None:
        if self.peek(b"FoamFile"):
            self.token()
            self.expect(b"{")
            while not self.peek(b"}"):
                key = self.token()
                self.header[key] = self.skip_to_semicolon().strip('"')
            self.expect(b"}")
        self.binary = self.header.get("format", "ascii") == "binary"
        arch = self.header.get("arch", "")
        if "label=64" in arch:
            self.label_dtype = np.dtype(np.int64)
        if "scalar=32" in arch:
            self.scalar_dtype = np.dtype(np.float32)

    def value(self, kind: str):
        """A single uniform value: a number or '(a b c ...)'."""
        if _NCOMP[kind] == 1:
            return self.number()
        self.expect(b"(")
        vals = [self.number() for _ in range(_NCOMP[kind])]
        self.expect(b")")
        return np.array(vals)

    def list_(self, kind: str) -> np.ndarray:
        """Parse 'N ( ... )' (ascii or binary) or the 'N{value}' shorthand."""
        self.ws()
        m = _COUNT.match(self.d, self.p)
        if not m:
            raise ValueError(f"list expected at byte {self.p}: {self.d[self.p:self.p + 60]!r}")
        n, bracket = int(m.group(1)), m.group(2)
        nc = _NCOMP[kind]
        self.p = m.end()
        if bracket == b"{":
            v = np.atleast_1d(self.value(kind))
            self.expect(b"}")
            arr = np.tile(v, n)
        elif self.binary:
            dt = self.label_dtype if kind == "label" else self.scalar_dtype
            cnt = n * nc
            arr = np.frombuffer(self.d, dtype=dt, count=cnt, offset=self.p)
            arr = arr.astype(np.int64 if kind == "label" else np.float64)
            self.p += cnt * dt.itemsize
            self.expect(b")")
        else:
            if nc == 1:
                end = self.d.index(b")", self.p)
            else:
                depth = 1
                for mm in _PAREN.finditer(self.d, self.p):
                    depth += 1 if mm.group(0) == b"(" else -1
                    if depth == 0:
                        end = mm.start()
                        break
            txt = self.d[self.p:end].replace(b"(", b" ").replace(b")", b" ")
            arr = np.array(txt.split(), dtype=np.float64)
            self.p = end + 1
        return arr.reshape(n, nc) if nc > 1 else arr

    def entry_value(self, kind: str):
        """Value after a keyword: uniform/nonuniform field data or raw text."""
        if self.peek(b"nonuniform"):
            self.token()
            typ = self.token()                       # e.g. List<scalar>
            k = typ[typ.index("<") + 1:typ.index(">")]
            v = self.list_(k)
            self.expect(b";")
            return v
        if self.peek(b"uniform"):
            self.token()
            v = self.value(kind)
            self.expect(b";")
            return v
        if self.peek(b"{"):
            self.skip_braces()
            return None
        return self.skip_to_semicolon()


def read_field(path) -> FoamField:
    """Read a vol*Field (internal + boundary) or a bare list such as points."""
    ps = _Parser(read_bytes(path))
    ps.read_header()
    cls = ps.header.get("class", "")
    kind = _kind(cls)
    if not cls.startswith(("vol", "surface", "point")):
        return FoamField(ps.list_(kind), {}, ps.header)

    ps.p = ps.d.index(b"internalField", ps.p) + len(b"internalField")
    internal = ps.entry_value(kind)
    ps.p = ps.d.index(b"boundaryField", ps.p) + len(b"boundaryField")
    ps.expect(b"{")
    boundary: dict = {}
    while not ps.peek(b"}"):
        name = ps.token()
        ps.expect(b"{")
        entries: dict = {}
        while not ps.peek(b"}"):
            key = ps.token()
            entries[key] = ps.entry_value(kind)
        ps.expect(b"}")
        boundary[name] = entries
    ps.expect(b"}")
    return FoamField(internal, boundary, ps.header)


def read_points(case_dir) -> np.ndarray:
    return read_field(Path(case_dir) / "constant" / "polyMesh" / "points").internal


def time_dirs(case_dir, field_name: str = "T") -> list[tuple[float, Path]]:
    """Sorted (time, directory) pairs that contain `field_name`."""
    out = []
    for d in Path(case_dir).iterdir():
        if not d.is_dir():
            continue
        try:
            t = float(d.name)
        except ValueError:
            continue
        if (d / field_name).exists() or (d / (field_name + ".gz")).exists():
            out.append((t, d))
    return sorted(out)


_SOLVE = re.compile(r"Solving for T, Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations (\d+)")


def parse_additivefoam_log(path) -> pd.DataFrame:
    """One row per solver time step (the step ends at `time`).

    absorbed_power is AdditiveFOAM's domain integral of the step-averaged
    source (W); n_thermo and thermo_residual describe the temperature /
    solid-fraction correction loop; lin_iters sums the linear-solver
    iterations for T; exec_time and clock_time are cumulative seconds.
    """
    rows, cur = [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("Time = "):
                if cur is not None:
                    rows.append(cur)
                cur = dict(time=float(line.split()[2]), absorbed_power=np.nan,
                           n_thermo=0, thermo_residual=np.nan, lin_iters=0,
                           T_initial_residual=np.nan, T_final_residual=np.nan,
                           exec_time=np.nan, clock_time=np.nan)
            elif cur is None:
                continue
            elif line.startswith("absorbed power:"):
                cur["absorbed_power"] = float(line.split()[2])
            elif line.startswith("Thermo: iteration"):
                parts = line.split()
                cur["n_thermo"] = int(parts[2]) + 1
                cur["thermo_residual"] = float(parts[4])
            elif "Solving for T," in line:
                m = _SOLVE.search(line)
                if m:
                    if np.isnan(cur["T_initial_residual"]):
                        cur["T_initial_residual"] = float(m.group(1))
                    cur["T_final_residual"] = float(m.group(2))
                    cur["lin_iters"] += int(m.group(3))
            elif line.startswith("ExecutionTime"):
                parts = line.split()
                cur["exec_time"] = float(parts[2])
                cur["clock_time"] = float(parts[6])
    if cur is not None:
        rows.append(cur)
    df = pd.DataFrame(rows)
    if len(df):
        df["dt"] = df["time"].diff()
        df.loc[df.index[0], "dt"] = df["time"].iloc[0]
    return df
