# Results Preview and Reproduction

当前仓库只保留 `diffusion_v2`。本地数据和生成结果不提交到 Git；复现时重新生成 `artifacts/`。

## 复现顺序

使用已安装的 `crack-synth` 环境：

```powershell
$env:PYTHONPATH='src'
C:\anaconda\envs\crack-synth\python.exe -m crack_synth.methods.diffusion_v2.prepare_rois --config configs\methods\diffusion_v2\contact_wire_v2.yaml
```

准备 LoRA 训练样本：

```powershell
$env:PYTHONPATH='src'
C:\anaconda\envs\crack-synth\python.exe -m crack_synth.methods.diffusion_v2.prepare_lora_data --config configs\methods\diffusion_v2\contact_wire_v2.yaml
```

训练 LoRA：

```powershell
$env:PYTHONPATH='src'
C:\anaconda\envs\crack-synth\python.exe -m crack_synth.methods.diffusion_v2.train_inpaint_lora `
  --train-data-dir artifacts\dataset_real\methods\diffusion_v2\lora_data\train `
  --output-dir artifacts\dataset_real\methods\diffusion_v2\lora_ctwirecrack_v2 `
  --rank 16 `
  --lora-alpha 16 `
  --max-train-steps 2000
```

只检查生成计划：

```powershell
$env:PYTHONPATH='src'
C:\anaconda\envs\crack-synth\python.exe -m crack_synth.methods.diffusion_v2.generate --config configs\methods\diffusion_v2\contact_wire_v2.yaml --plan-only
```

执行生成：

```powershell
$env:PYTHONPATH='src'
C:\anaconda\envs\crack-synth\python.exe -m crack_synth.methods.diffusion_v2.generate --config configs\methods\diffusion_v2\contact_wire_v2.yaml
```

评估最近一次生成：

```powershell
$env:PYTHONPATH='src'
C:\anaconda\envs\crack-synth\python.exe -m crack_synth.methods.diffusion_v2.evaluate_generation --latest --config configs\methods\diffusion_v2\contact_wire_v2.yaml
```

## 输出位置

```text
artifacts/dataset_real/methods/diffusion_v2/
├─ roi_assets/
├─ lora_data/
├─ lora_ctwirecrack_v2/
└─ run_<timestamp>/
```

检查生成效果时，优先看：

- `run_<timestamp>/top_contact_sheet.png`
- `run_<timestamp>/candidate_scores.csv`
- `run_<timestamp>/samples/<record_id>/image_syn.png`
- `run_<timestamp>/samples/<record_id>/debug/guide_roi.png`
- `run_<timestamp>/samples/<record_id>/debug/diff_abs.png`

主要视觉检查：

- 裂缝是否落在 `mask_raw_roi.png` 附近。
- 裂缝是否像细长暗色纵向裂缝，而不是污渍或阴影。
- `mask_edit_roi.png` 外背景是否稳定。
- `candidate_scores.csv` 里的高分样本是否主观也更好。
