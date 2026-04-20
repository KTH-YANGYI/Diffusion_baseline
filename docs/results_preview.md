# Results Preview and Reproduction Notes

This repository does not commit the local dataset or generated diffusion outputs. The current GitHub version is intended to make the pipeline understandable and reproducible without uploading large files.

## Current Pipeline State

The current method is `diffusion_baseline`.

It uses paired real data:

```text
dataset_real/
+-- crack/
|   +-- <frame>_crop.jpg
|   +-- <frame>_crop.json
+-- normal/
    +-- <frame>_crop.jpg
```

For each pair, the pipeline:

1. Converts the crack JSON annotation into a binary raw mask.
2. Dilates the raw mask by `mask_edit_dilate_px`.
3. Crops the same ROI from the crack image and the paired normal image.
4. Runs Stable Diffusion Inpainting on the normal ROI using the edit mask.
5. Saves the synthetic crack image and metadata.

The crack image is not used as a visual conditioning image. It provides only the mask geometry and ROI position.

## Current Configuration

Default config:

```text
configs/methods/diffusion_baseline/contact_wire_v1.yaml
```

Important current settings:

```yaml
roi_out_size: 512
mask_edit_dilate_px: 3
seeds_per_pair: 1
inference_batch_size: 1
num_inference_steps: 35
guidance_scale: 8.5
strength: 0.97
mask_blur: 1.0
```

The dataset is expected at:

```text
C:/Users/18046/Desktop/master/masterthesis/dataset_real
```

Change `dataset_root` in the config if the data is stored elsewhere.

## Reproduce Locally

Use the `crack-synth` environment:

```powershell
$env:PYTHONPATH='src'
C:\anaconda\envs\crack-synth\python.exe -m crack_synth.methods.diffusion_baseline.prepare_rois --config configs\methods\diffusion_baseline\contact_wire_v1.yaml
```

Check the planned generation jobs:

```powershell
$env:PYTHONPATH='src'
C:\anaconda\envs\crack-synth\python.exe -m crack_synth.methods.diffusion_baseline.generate_baseline --config configs\methods\diffusion_baseline\contact_wire_v1.yaml --plan-only
```

Run full inpainting:

```powershell
$env:PYTHONPATH='src'
C:\anaconda\envs\crack-synth\python.exe -m crack_synth.methods.diffusion_baseline.generate_baseline --config configs\methods\diffusion_baseline\contact_wire_v1.yaml
```

Evaluate the latest generated run:

```powershell
$env:PYTHONPATH='src'
C:\anaconda\envs\crack-synth\python.exe -m crack_synth.methods.diffusion_baseline.evaluate_generation --latest --config configs\methods\diffusion_baseline\contact_wire_v1.yaml
```

## Expected Local Outputs

ROI assets:

```text
artifacts/dataset_real/methods/diffusion_baseline/roi_assets/
```

Generated run:

```text
artifacts/dataset_real/methods/diffusion_baseline/run_<timestamp>/
```

Each generated sample contains:

```text
image_syn.png
background_roi.png
mask_raw_roi.png
mask_edit_roi.png
metadata.json
```

Evaluation files are written to:

```text
artifacts/dataset_real/methods/diffusion_baseline/run_<timestamp>/evaluation/
```

## How to Inspect Results

The fastest way is to open the generated sample folders under:

```text
artifacts/dataset_real/methods/diffusion_baseline/run_<timestamp>/samples/
```

For each sample, compare:

- `background_roi.png`: paired normal input ROI
- `mask_edit_roi.png`: editable inpainting region
- `image_syn.png`: generated synthetic crack result

Suggested visual checks:

- Does the crack follow the mask location and shape?
- Is the crack visible enough?
- Does the surrounding metallic background stay stable?
- Does the output look like a crack rather than a stain, shadow, or scratch?
- Is `mask_edit_dilate_px: 3` too narrow, too wide, or acceptable?

## Notes for Reviewers

The committed repository intentionally excludes:

- raw dataset files
- generated ROI assets
- generated diffusion runs
- model cache files

These are excluded by `.gitignore` because they are local, large, or reproducible artifacts.

The dataset structure is documented in:

```text
docs/dataset_real_说明.md
```

That document lists the current local pair layout and explains how crack images, normal images, and JSON annotations should be organized.
