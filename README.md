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
                           used for the fixed-thermal-state run of Figure 11
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

## Supplementary videos

Each video shows, side by side, the recorded frames, the frames generated by the identified
two-state NODE from the process input alone, and the frames generated by a NODE without the
thermal state. The third view is a separate model: the same structure with δ removed, identified on
its own on the same record (the "NODE without δ" of Table 5), not the two-state model with its
thermal state switched off. Below the frames are the travel speed (for WAAM, a note that the input
is constant) and the thermal state δ̂ of the two-state NODE. Every sequence is generated from the
first frame and the recorded input only; no image is read after the first one. Playback runs at
10–12 frames per second, the recording rate rounded to a whole number, so it is close to real time.

The files are in [`videos/`](videos/) and are also attached to
[release v1.1](../../releases/tag/v1.1).

### Video S1 — GMAW
Gas metal arc welding, record `exp_05`, as in Figure 9. The travel speed varies over roughly
3 to 12 mm s⁻¹.
→ [`videos/Video_S1_GMAW.mp4`](videos/Video_S1_GMAW.mp4)

### Video S2 — GTAW
Gas tungsten arc welding, record `Iden_02`, as in Figure 10.
→ [`videos/Video_S2_GTAW.mp4`](videos/Video_S2_GTAW.mp4)

### Video S3 — WAAM
Wire-arc additive manufacturing, record `video_3`, the record of Figure 11. The input never
changes — 200 A at 2.7 mm s⁻¹ — so the model is autonomous here and the whole trajectory follows
from the first frame. The recorded pool state keeps changing throughout the deposit; the NODE
without δ settles within about 10 s, while the two-state NODE follows the recording.
Figure 11 makes a different comparison on the same record: the two-state NODE against the same
model with δ̂ held fixed after t_f = 8.2 s
([`sec6_figure_11_fixed_delta.py`](code/representation_ViT_state/sec6_figure_11_fixed_delta.py)).
This video, like Table 5, uses the separately identified NODE without δ.
→ [`videos/Video_S3_WAAM.mp4`](videos/Video_S3_WAAM.mp4)

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
