# vnm1 — VNM1 (Voivent Neural Material) trainer CLI

Converts PBR texture sets (base color / normal / roughness /
metalness / AO) into VNM1 neural-material artifacts: a small MLP
plus two latent grids, stored as a 4-file directory artifact
(`manifest.json`, `weights.bin`, `level-0.bin`, `tail.bin`).

Every conversion is checked against a fixed quality gate:
base-color PSNR >= 35 dB (linear), scalar MAE <= 0.02, normal
angle mean <= 3 deg / p95 <= 8 deg. An artifact that does not
pass is reported as a candidate, never shipped as a pass.

This is the offline trainer. The runtime decoders live in the
companion projects `voivent-neural-material` (Three.js / WebGPU)
and `voivent-neural-material-godot` (Godot 4). The format is
specified in the SPEC document of the main repository.

License: Apache-2.0 (see `LICENSE`).

## Install

```
pip install .            # read-only commands (preflight, inspect)
pip install .[train]     # + PyTorch, required for train
```

Python >= 3.10. Base dependencies are only numpy and Pillow;
PyTorch is an extra because the input/inspection layer never
imports it.

## Commands

```
vnm1 preflight [INPUT] [options]   # validate inputs, print the plan
vnm1 train     [INPUT] [options]   # convert (manual or --auto)
vnm1 inspect   TARGET              # summarise a report / verify an artifact
```

`INPUT` is a material folder, a single base-color image, or a JSON
config (`--config`). Map roles are detected from file names
(`basecolor`, `normal`, `rough`, `metal`, `ao`, ...) or given
explicitly (`--base-color`, `--normal`, `--roughness`,
`--metalness`, `--ao`). `--normal-convention opengl|directx` is
**required** whenever a normal map is present; DirectX maps are
converted to the internal OpenGL convention at load time and the
conversion is recorded in the report.

Packed maps (e.g. ARM/ORM) are not unpacked automatically: the
preflight warns when a file looks like a packed map, and the
warning is carried into the train conclusion and the report.
Files that are not supported maps (e.g. `emissive.png`,
`height.png`) are named in the preflight output as excluded,
never silently ignored.

### Train modes

- **Manual** (default): one trial with `--profile` (default
  `standard`), `--iterations` (default 50,000), `--color-weight`
  (default 4) and `--scalar-weight` (default 1).
- **Auto** (`--auto`): climbs the measured ladder
  standard -> standard8 -> quality -> quality8 with full evaluation
  at every rung and stops at the first pass. On a narrow miss
  exactly one adjustment trial runs with the measured
  weight/iteration patterns. Progress is checkpointed to
  `work/auto_state.json`; re-running the same command resumes
  without retraining finished trials.

Profiles: `standard`, `quality` (4+4 latent channels) and
`standard8`, `quality8` (8+8 channels). These four are the public
presets and the only values `--profile` accepts.

### Capacity entry check and warnings

A VNM1 artifact's size is set by the profile, not by the input
size, while BCn scales with pixels x maps — so small or
single-map inputs can come out *larger* than BCn. Two stages keep
that visible:

- **Entry check** (`preflight` and `train`, both modes, before
  any training): the cheapest profile (`standard`) is sized from
  the format parameters and compared with the BCn full mip chain
  estimate for the input. If even `standard` cannot beat BCn, the
  input is rejected with exit code 2 and both byte counts.
  Pass `--allow-capacity-expansion` to convert anyway; the
  override is recorded in the report (`capacity_check.override`).
- **Outcome warnings**: a finished report whose chosen artifact
  measures a capacity ratio >= 0.9 carries a "thin margin"
  warning, and >= 1.0 an "expansion" warning, in the report JSON,
  in the conclusion of `report.md`, and on the command line.

### Multi-mip training (opt-in)

`vnm1 train --mip-train` (either mode; default off) trains every
trial on a teacher mip pyramid: 50% of samples at mip 0, 50%
spread uniformly over mips 1-6, with the decoder input
`normalizedMip = mip / 6`. The artifact format is unchanged; the
manifest records the optional `mipMax: 6` only for mip-trained
artifacts, and the report gains a mip 0/2/4 quality table.
Measured trade-off (3 materials, same-recipe control): far-mip
PSNR improves by +7.7 to +21.5 dB while mip 0 drops by
1.0-2.2 dB — which is why the flag is opt-in.

Runs without the flag still measure the finished artifact at
mips 0/2/4. When the far-field degradation crosses the advisory
thresholds, the report and the console summary recommend
re-running with `--mip-train`, with the measured numbers. The
advisory never affects the gate verdict or the exit code.

`--fast-eval` skips the multi-mip measurement and sheet
generation in post-processing (the gate result is unchanged);
useful for batch runs where only the verdict matters.

### Outputs

```
<out>/<material>/
  <material>.vnm/          # passing artifact (manual pass / auto selection)
  <material>.report.json   # machine-readable record
  <material>.report.md     # human-readable summary
  <material>_sheet.png     # original | decode | error heat map
  <material>_crop.png      # native-resolution centre crop pair
  <material>_normal_sheet.png   # when a normal map was given
  work/attempts/           # non-selected trial artifacts (candidates)
  work/auto_state.json     # auto mode: resume state
```

On failure the artifact stays under `work/attempts/` and the
report names it as a candidate only — never as a passing
artifact.

### inspect

`vnm1 inspect TARGET` accepts a material output directory, an
artifact (`*.vnm`) directory, or a `*.report.json` file. It prints
the recorded conclusion and verifies the artifact from disk:
manifest and profile, chunk integrity against the manifest
(byteLength + SHA-256), latent reserved bytes zero, and a read-back
spot decode. Exit codes: 0 = report passed and verification OK,
3 = report gate failed, 5 = verification failed, 2 = target not
found / not a report or artifact.

## Environment

- **Standard environment: a CUDA GPU with PyTorch** (developed
  and measured on an NVIDIA T4). CPU works for `preflight`,
  `inspect`, sheet rendering and small checks; CPU production
  runs are unverified (no measured time exists at production
  iteration counts).
- Verified input range: 2K square sources. Other sizes convert
  but are unmeasured and trigger a warning; very high-resolution
  inputs (4K class) are a known weak spot of the current fixed
  latent-grid profiles.

## Time estimates (T4-class GPU, 2K source, measured)

| Run | Time |
| --- | --- |
| standard, 50k iterations | about 7–9 min |
| quality, 50k iterations | about 9–10 min |
| quality8 adjustment, 150k iterations | about 35 min |
| auto mode, worst case (full ladder + adjustment) | about 70–80 min |

Auto mode stops at the first passing rung, so the total depends
on where a material passes.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | gate passed (inspect: passed + verification OK) |
| 2 | input validation error (inspect: target not found) |
| 3 | training finished, gate not met |
| 4 | training error (non-finite loss, device failure, ...) |
| 5 | export / verification error (round-trip, sheet PSNR mismatch, inspect verification failure) |

## Known limitations

- **GPU decoder compatibility**: the runtime decoders support
  all four profiles (`standard`, `quality`, `standard8`,
  `quality8`). Conformance against this CLI's CPU reference
  decoder was verified on desktop WebGPU (Apple Metal-3) and on
  a Pixel 9a (ARM Valhall) in November 2026, with real-asset
  agreement at the 1e-6 level.
- Far-field quality: the quality gate is evaluated at mip 0.
  Artifacts trained without `--mip-train` only ever present
  `normalizedMip = 0` during training and can show moire under
  strong minification; the post-training advisory flags the
  measured-severe cases and `--mip-train` is the opt-in remedy.
- Fine high-frequency detail (dense foliage and similar) can
  stay below the gate in every configuration; a failure report
  says so plainly instead of shipping a soft artifact.
- No alpha, emissive or height support: materials are the five
  maps listed above. Missing maps become constants and are
  reported as gate-exempt; they are not recovered data.
- The capability measurements come from CC0 sources (Poly Haven /
  ambientCG) at 2K with one seed per configuration; other sources
  and sizes are unmeasured.

## Development

Run the test suite (185 tests; torch-dependent tests skip
cleanly when PyTorch is absent):

```
PYTHONPATH=src python3 -m unittest discover -s tests
```

Status: 1.0.0. The training core is covered by the unit suite
and by the format conformance vectors maintained with the SPEC
in the main repository.
