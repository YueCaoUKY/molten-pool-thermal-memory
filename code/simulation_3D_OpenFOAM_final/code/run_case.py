#!/usr/bin/env python3
"""Generate, run, collect and post-process one benchmark case.

    python code/run_case.py configs/<case>.yaml [--steps all] [--nprocs N] [--force]

Stages (comma-separated in --steps, or "all"):
  gen          write the case into RUN_ROOT/<case_id> (refuses to overwrite
               without --force)
  mesh         blockMesh, checkMesh, writeCellCentres
  solve        decomposePar (if nprocs > 1), additiveFoam, reconstructPar,
               then remove processor* directories once reconstruction succeeded
  collect      copy dictionaries, logs, function-object output and a run
               manifest to results/<case_id>/
  post         compute the 0.1 s research signals (mp3d.postprocess)

OpenFOAM commands run with the case directory as working directory, because
OpenFOAM aborts in paths containing spaces (the project path has spaces).
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mp3d import AF_BASHRC, OF_BASHRC, RESULTS_DIR, RUN_ROOT  # noqa: E402
from mp3d.case_writer import load_config, write_case  # noqa: E402

STAGES = ["gen", "mesh", "solve", "collect", "post"]


def foam(case: Path, cmd: str, log: str) -> float:
    """Run an OpenFOAM command in `case`, stdout/stderr to `log`; return seconds."""
    t0 = time.time()
    script = (f". {OF_BASHRC} > /dev/null 2>&1; . {AF_BASHRC} > /dev/null 2>&1; "
              f"cd '{case}' && {cmd} > {log} 2>&1")
    res = subprocess.run(["bash", "-c", script])
    if res.returncode != 0:
        raise RuntimeError(f"'{cmd}' failed in {case} (exit {res.returncode}); see {case / log}")
    return time.time() - t0


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def stage_gen(cfg, case: Path, force: bool):
    if case.exists():
        if not force:
            raise SystemExit(f"{case} exists; use --force to regenerate")
        if not _inside(case, RUN_ROOT) or case.resolve() == RUN_ROOT.resolve():
            raise SystemExit(f"refusing to delete {case}")
        shutil.rmtree(case)
    case.mkdir(parents=True)
    meta = write_case(cfg, case)
    print(f"[gen] {case}  cells={meta['n_cells']}  n={meta['n']}")


def stage_mesh(cfg, case: Path, timings: dict):
    timings["blockMesh"] = foam(case, "blockMesh", "log.blockMesh")
    timings["checkMesh"] = foam(case, "checkMesh", "log.checkMesh")
    timings["writeCellCentres"] = foam(case, "foamPostProcess -func writeCellCentres -time 0",
                                       "log.writeCellCentres")
    print(f"[mesh] done ({timings['blockMesh']:.1f} s blockMesh)")


def stage_solve(cfg, case: Path, nprocs: int, timings: dict):
    if nprocs > 1:
        timings["decomposePar"] = foam(case, "decomposePar -force", "log.decomposePar")
        timings["additiveFoam"] = foam(case, f"mpirun -np {nprocs} additiveFoam -parallel",
                                       "log.additiveFoam")
        timings["reconstructPar"] = foam(case, "reconstructPar", "log.reconstructPar")
        t_end = float(cfg["time"]["t_end"])
        last = max((float(d.name) for d in case.iterdir()
                    if d.is_dir() and d.name.replace(".", "", 1).isdigit()), default=-1.0)
        if abs(last - t_end) < 1e-9:
            for d in case.glob("processor*"):
                if _inside(d, case):
                    shutil.rmtree(d)
        else:
            print(f"[solve] WARNING: last reconstructed time {last} != endTime {t_end}; "
                  "processor directories kept")
    else:
        timings["additiveFoam"] = foam(case, "additiveFoam", "log.additiveFoam")
    print(f"[solve] additiveFoam {timings['additiveFoam']:.0f} s on {nprocs} rank(s)")


def _versions() -> dict:
    def sh(cmd):
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        return r.stdout.strip()
    return {
        "openfoam14_package": sh("dpkg-query -W -f='${Version}' openfoam14"),
        "additivefoam_commit": sh("git -C /opt/AdditiveFOAM-2.0.0 rev-parse HEAD"),
        "host": platform.node(),
        "python": sys.version.split()[0],
    }


def stage_collect(cfg, case: Path, out: Path, timings: dict, nprocs: int):
    out.mkdir(parents=True, exist_ok=True)
    files = out / "case_files"
    if files.exists():
        shutil.rmtree(files)
    for sub in ("system", "constant", "0"):
        for f in (case / sub).glob("*"):
            if f.is_file() and f.name not in ("C", "C.gz"):
                (files / sub).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, files / sub / f.name)
    for f in ("config_resolved.yaml", "mesh_meta.json"):
        shutil.copy2(case / f, out / f)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    for f in case.glob("log.*"):
        shutil.copy2(f, logs / f.name)
    mpd = case / "postProcessing" / "meltPoolDimensions"
    if mpd.exists():
        shutil.copytree(mpd, out / "meltPoolDimensions", dirs_exist_ok=True)
    tcp = case / "postProcessing" / "tcProbes"
    if tcp.exists():
        shutil.copytree(tcp, out / "tcProbes", dirs_exist_ok=True)
    manifest = {"case_id": cfg["case_id"], "run_dir": str(case), "nprocs": nprocs,
                "timings_s": timings, "versions": _versions(),
                "collected": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[collect] -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--steps", default="all")
    ap.add_argument("--nprocs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.nprocs:
        cfg["run"]["nprocs"] = args.nprocs
    nprocs = int(cfg["run"]["nprocs"])
    case = RUN_ROOT / cfg["case_id"]
    out = RESULTS_DIR / cfg["case_id"]
    steps = STAGES if args.steps == "all" else args.steps.split(",")
    timings: dict = {}
    t0 = time.time()
    if "gen" in steps:
        stage_gen(cfg, case, args.force)
    if "mesh" in steps:
        stage_mesh(cfg, case, timings)
    if "solve" in steps:
        stage_solve(cfg, case, nprocs, timings)
    if "collect" in steps:
        stage_collect(cfg, case, out, timings, nprocs)
    if "post" in steps:
        from mp3d.postprocess import process_case
        process_case(case, cfg, out)
    print(f"[done] {cfg['case_id']} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
