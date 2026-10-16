"""3D heat-only finite-volume benchmark (OpenFOAM 14 + AdditiveFOAM 2.0.0).

Case generation, execution and post-processing for the 3D counterpart of the
2D FEniCSx `pool_steel_lit` run. OpenFOAM cannot run in a path containing
spaces, so cases execute under RUN_ROOT (decision D2, 2026-09-13) and compact
results, logs and the exact case dictionaries are copied to RESULTS_DIR.
"""
import getpass
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
CONFIG_DIR = PROJECT_DIR / "configs"
RUN_ROOT = Path(os.environ.get(
    "MP3D_RUN_ROOT",
    Path.home() / "OpenFOAM" / f"{getpass.getuser()}-14" / "run" / "molten_pool_3d"))
OF_BASHRC = "/opt/openfoam14/etc/bashrc"
AF_BASHRC = "/opt/AdditiveFOAM-2.0.0/etc/bashrc"

if " " in str(RUN_ROOT):
    raise RuntimeError(f"RUN_ROOT must not contain spaces (OpenFOAM limitation): {RUN_ROOT}")
