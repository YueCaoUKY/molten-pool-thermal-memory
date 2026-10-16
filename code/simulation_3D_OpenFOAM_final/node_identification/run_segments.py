#!/usr/bin/env python3
"""Run the 304L v2 pair in segments, so an interrupted run resumes instead of starting over.

The case already writes every field every 5 s of simulated time (base_S0.yaml
full_write_interval), so those writes serve as restart points. Two changes to the generated case
make them usable: the solver starts from the latest written time instead of 0, and the control
dictionary is re-read while running, so a segment can be ended by asking the solver to write and
exit rather than by killing it.

    python run_segments.py prepare            generate, mesh and decompose (once)
    python run_segments.py run <minutes> [--serial]
                                              solve for that many minutes, then stop
                                              cleanly; --serial runs one case at a
                                              time, using half the machine
    python run_segments.py status             simulated time reached so far
    python run_segments.py finish             reconstruct, collect, post-process, extract features
"""
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
RUN = Path.home() / "OpenFOAM" / "troy-14" / "run" / "molten_pool_3d"
LOGS = BENCH / "logs" / "seg"
CASES = ["PP_ss304_speed_free_v2", "PP_ss304_speed_cooled_v2"]
NPROCS = 12                 # two cases at a time -> all 24 cores
T_END = 240.0
OF = ". /opt/openfoam14/etc/bashrc > /dev/null 2>&1; . /opt/AdditiveFOAM-2.0.0/etc/bashrc > /dev/null 2>&1"


def sh(case, cmd, log):
    LOGS.mkdir(parents=True, exist_ok=True)
    return subprocess.call(["bash", "-c", f"{OF}; cd '{RUN / case}' && {cmd} > {log} 2>&1"])


def reached(case):
    """Latest simulated time held by rank 0 (or by the case itself once reconstructed)."""
    d = RUN / case / "processor0"
    d = d if d.exists() else RUN / case
    if not d.exists():
        return -1.0
    t = [float(p.name) for p in d.iterdir() if p.is_dir() and re.fullmatch(r"\d+(\.\d+)?", p.name)]
    return max(t, default=-1.0)


def prepare():
    for case in CASES:
        subprocess.check_call(["python3", str(BENCH / "code" / "run_case.py"),
                               str(BENCH / "configs" / f"{case}.yaml"),
                               "--steps", "gen,mesh", "--nprocs", str(NPROCS), "--force"],
                              cwd=str(BENCH))
        cd = RUN / case / "system" / "controlDict"
        txt = cd.read_text()
        txt = txt.replace("startFrom       startTime;", "startFrom       latestTime;")
        txt = txt.replace("runTimeModifiable no;", "runTimeModifiable yes;")
        cd.write_text(txt)
        sh(case, f"decomposePar -force", "log.decomposePar")
        print(f"{case}: 已生成、划分网格、分解到 {NPROCS} 个进程")


def _solve(case):
    sh(case, f"mpirun -np {NPROCS} additiveFoam -parallel", "log.additiveFoam")


def stop_cleanly(case):
    """Ask the solver to write the current step and exit; it re-reads controlDict every step."""
    cd = RUN / case / "system" / "controlDict"
    cd.write_text(cd.read_text().replace("stopAt          endTime;", "stopAt          writeNow;"))


def restore_end(case):
    cd = RUN / case / "system" / "controlDict"
    cd.write_text(cd.read_text().replace("stopAt          writeNow;", "stopAt          endTime;"))


def run(minutes, serial=False):
    """Both cases at once (all 24 cores), or one at a time on its 12 ranks (half the machine)."""
    deadline = time.time() + minutes * 60
    before = {c: reached(c) for c in CASES}
    groups = [[c] for c in CASES] if serial else [list(CASES)]
    for group in groups:
        group = [c for c in group if reached(c) < T_END - 1e-9]
        if not group or time.time() >= deadline:
            continue
        for case in group:
            restore_end(case)
        with ThreadPoolExecutor(max_workers=len(group)) as ex:
            futs = [ex.submit(_solve, c) for c in group]
            while time.time() < deadline and not all(f.done() for f in futs):
                time.sleep(20)
            for case in group:
                if reached(case) < T_END - 1e-9:
                    stop_cleanly(case)
            for f in futs:
                f.result()
    for case in CASES:
        restore_end(case)
        print(f"{case}: {before[case]:.0f} s -> {reached(case):.0f} s / {T_END:.0f} s")


def status():
    for case in CASES:
        t = reached(case)
        print(f"{case:30s} 已算到 {t:6.1f} s / {T_END:.0f} s  ({max(t, 0) / T_END:5.1%})")


def finish():
    for case in CASES:
        sh(case, "reconstructPar", "log.reconstructPar")
        subprocess.check_call(["python3", str(BENCH / "code" / "run_case.py"),
                               str(BENCH / "configs" / f"{case}.yaml"), "--steps", "collect,post",
                               "--nprocs", str(NPROCS)], cwd=str(BENCH))
        subprocess.check_call(["python3", "extract_features.py", case], cwd=str(HERE))
        print(f"{case}: 完成")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"prepare": prepare, "status": status, "finish": finish}.get(
        cmd, lambda: run(float(sys.argv[2]), serial="--serial" in sys.argv))()
