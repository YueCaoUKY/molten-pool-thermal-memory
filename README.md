# Process-Agnostic Modeling of Arc-Based Molten Pools

### Physics-Structured Latent Dynamics with Workpiece Thermal Memory

Yue Cao, Olabode Ajenifujah, Yujing Yang, YuMing Zhang, Chenhui Shao

*University of Michigan · University of Kentucky*

---

Code, simulation data and supplementary videos for the paper above.

The molten pool in arc-based processes does not respond to the process inputs alone. The same
current and travel speed produce a different pool early in a weld than late in it, because the
workpiece has been accumulating heat in the meantime. Models that map inputs to pool geometry
instantaneously cannot represent this, and they drift over a long deposit. The model here adds one
scalar state for that accumulated heat and gives it a first-order equation of its own, so the pool
equation relaxes towards an equilibrium that the thermal state shifts. The structure is fixed by
the physics and only the parameters are identified; the same structure, unchanged, is identified on
finite-volume simulations of two materials and two backing conditions, and on recorded video of
three different processes — GTAW, GMAW and wire-arc additive manufacturing.

## Repository structure

```
code/
  simulation_3D_OpenFOAM_final/
    node_identification/   physics-structured NODE (Eq. 3) and the identification
                           procedure (Table 4.2); the ARX, LSTM and Transformer
                           baselines of Section 5.2, in baselines.py
    code/                  OpenFOAM case runner and post-processing
    configs/               the case configuration of every simulated record
    results/ -> symlink to ../../data/simulation
  representation_ViT_state/
                           image representation and generation of Section 6:
                           head, structure decoder, VQ residual branch, image
                           generation (Eq. 6), SSIM scoring, and the scripts
                           behind Figures 8-11
    numpy_port/            NumPy forward pass of the identified model and decoders,
                           used for the fixed-thermal-state runs (Figure 11 and the videos)
  Representation Learning NODE zs - All Datasets/
    paper_section4/        representation experiment driver
    code/                  training library and the frozen DINOv2 ViT-B/14
                           wrapper (Appendix B)
  Physics Structured Dynamics - Section 5/
                           shared model components and the model-structure
                           figure scripts
data/
  simulation/              13 simulated records; see "Simulation data" below
videos/                    supplementary videos S1-S3 and one still frame of each
requirements.txt
LICENSE                    MIT, for code/
LICENSE-DATA               CC BY 4.0, for data/ and videos/
```

The directory names under `code/` are the ones from the authors' working tree, and the scripts
locate each other and the data by those names — `identify_3d.py`, for instance, reads records from
`../results/` and imports shared components from `../../Physics Structured Dynamics - Section 5/`.
Keeping the layout as it is here is what makes the scripts run unmodified, which is why the folder
names are not tidier.

## Installation

```bash
conda create -n molten-pool python=3.12
conda activate molten-pool
# install PyTorch first, matching your CUDA, from https://pytorch.org
pip install -r requirements.txt
```

The frozen DINOv2 ViT-B/14 backbone is downloaded at run time by `torch.hub` from
`facebookresearch/dinov2`; it is not vendored here. Re-running the Section 5 simulations themselves
additionally needs OpenFOAM 14 with AdditiveFOAM 2.0.0, but reproducing the tables and figures from
the data in `data/` does not.

`code/simulation_3D_OpenFOAM_final/results` is a relative symlink to `data/simulation`, so the
identification scripts find the records where they expect them. On a platform that does not
restore symlinks on checkout, replace it with a copy:

```bash
rm -f  code/simulation_3D_OpenFOAM_final/results
cp -r  data/simulation code/simulation_3D_OpenFOAM_final/results
```

A quick check that the tree is wired up correctly:

```bash
cd code/simulation_3D_OpenFOAM_final/node_identification
python -c "import identify_3d as I; print(len(list(I.RESULTS.glob('PP_*'))), 'records')"
# -> 13 records
```

## Simulation data

Twelve records form a full factorial: two materials (low-carbon steel and AISI 304L) × two backing
conditions (free plate and water-cooled) × three input programs (arc current alone, travel speed
alone, and both together). Each is 240 s long. The free and water-cooled records of a pair share one
input program, so the only difference within a pair is where the heat goes. A thirteenth record
holds the power-step test used for long-horizon prediction.

| Record | Varying input | Backing | Length |
|---|---|---|---|
| `PP_steel_4_slow_cycles`, `PP_ss304_4_slow_cycles` | current | free | 240 s |
| `PP_steel_5_slow_cooled`, `PP_ss304_5_slow_cooled` | current | water-cooled | 240 s |
| `PP_steel_speed_free_v2`, `PP_ss304_speed_free_v2` | travel speed | free | 240 s |
| `PP_steel_speed_cooled_v2`, `PP_ss304_speed_cooled_v2` | travel speed | water-cooled | 240 s |
| `PP_steel_both_free`, `PP_ss304_both_free` | current and speed | free | 240 s |
| `PP_steel_both_cooled`, `PP_ss304_both_cooled` | current and speed | water-cooled | 240 s |
| `PP_steel_6_step_ABA` | current, stepped A-B-A | free | 180 s |

Each record directory holds

| File | Contents |
|---|---|
| `signals_0p1s.csv` | the whole record at 0.1 s: process inputs, the pool-state series, and the withheld thermal reference |
| `signals_columns.json` | units and a one-line description of every column |
| `profiles_0p1s.npz` | centreline temperature profiles |
| `probes_0p1s.npz` | point temperature histories |
| `mesh_meta.json`, `run_manifest.json`, `config_resolved.yaml` | mesh, provenance, and the fully resolved case configuration |
| `case_setup/` | the OpenFOAM case dictionaries, `system/` and `constant/` |

In `signals_0p1s.csv`, the inputs are `I_A`, `u_mm_s` and the absorbed power columns; the pool state
is the top-surface geometry and penetration (`top_area_liq_mm2`, `top_length_liq_mm`,
`top_width_liq_mm`, `depth_liq_mm`, `vol_liquid_mm3` and their solidus counterparts); and the
thermal reference withheld from identification is `delta_H_K` with `mean_T_rise_K`.

`case_setup/` carries the case dictionaries but not the per-cell initial fields, which `blockMesh`
and `setFields` regenerate from `system/blockMeshDict`. **The raw full-field OpenFOAM output is not
in this repository.** It runs to roughly 530 MB per record, 6.1 GB in total, which is beyond what a
Git repository should carry; it is available from the corresponding author on request.

## Experimental image data

The GTAW (10 records) and GMAW (9 records) high-speed recordings used in Section 6 were collected in
the laboratory of YuMing Zhang at the University of Kentucky. They are **not redistributed here**
and are available on request. Together they come to about 7.4 GB of frames.

The WAAM records (3) come from the public **WAAM-ViD** benchmark and are not re-uploaded. Please
obtain them from the original source and cite it:

> Kim et al. (2025). WAAM-ViD. *Frontiers in Manufacturing Technology*.

## Videos

The videos show what the thermal state does to the generated images, one per process. Each shows,
side by side, the recorded frames, the frames the identified two-state NODE generates from the first
frame and the recorded input alone, and the frames the same model generates when its thermal state
δ̂ is held at its value at a time t_f. Up to t_f the two generated sequences are identical; after it
they differ only because δ̂ is no longer allowed to accumulate. Below the frames are the travel
speed (for WAAM, a note that the input is constant) and δ̂ in both runs. Playback runs at 10–12
frames per second, the recording rate rounded to a whole number, so it is close to real time.

t_f is the earliest of 20, 30, 40 and 50 % of the record at which the run with δ̂ fixed stays within
1.5 times the identified run's median distance to the nearest recorded pool state; where no
candidate meets this, the one with the smallest ratio is used (see `results_sec6/3b_*.json`).
Figure 11 of the paper shows the WAAM case.

The files are in [`videos/`](videos/) and are also attached to
[release v1.1](../../releases/tag/v1.1).

| Process | Record | t_f | File |
|---|---|---|---|
| GMAW | `exp_05` | 19.2 s | [`videos/fixed_delta_GMAW.mp4`](videos/fixed_delta_GMAW.mp4) |
| GTAW | `Iden_02` | 13.1 s | [`videos/fixed_delta_GTAW.mp4`](videos/fixed_delta_GTAW.mp4) |
| WAAM | `video_3` | 8.2 s | [`videos/fixed_delta_WAAM.mp4`](videos/fixed_delta_WAAM.mp4) |

The videos of release v1.0 compared the two-state NODE with a separately identified NODE without
δ; that comparison is in Table 5 and Figures 9–10 of the paper.

## Licenses

| What | License |
|---|---|
| `code/` | MIT — see [LICENSE](LICENSE) |
| `data/`, `videos/` | CC BY 4.0 — see [LICENSE-DATA](LICENSE-DATA) |

Third-party material keeps its own terms: DINOv2 is Meta's, under the licence of the
`facebookresearch/dinov2` repository, and the WAAM-ViD records stay under the licence of that
benchmark.

## Citation

Journal details will be added once the paper is published.

```bibtex
@article{cao2026moltenpool,
  title   = {Process-Agnostic Modeling of Arc-Based Molten Pools:
             Physics-Structured Latent Dynamics with Workpiece Thermal Memory},
  author  = {Cao, Yue and Ajenifujah, Olabode and Yang, Yujing
             and Zhang, YuMing and Shao, Chenhui},
  year    = {2026},
  note    = {Manuscript under review}
}
```
