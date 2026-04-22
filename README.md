# Crack Synth

本仓库现在只保留一条方法线：`diffusion_v2`。

`diffusion_v2` 的目标是：使用真实 crack/normal 配对数据，在对应正常图上做 Stable Diffusion Inpainting，生成铁路接触线裂缝样本。旧的 v1、guided-inpaint、lora-inpaint 分支配置和入口已经移除。

## 数据流

```text
dataset_real/crack/*.jpg + *.json
dataset_real/normal/*.jpg
        |
        v
prepare_rois
        |
        v
roi_assets/
  roi_pairs.jsonl
  pairs/<pair_id>/
    defect_roi.png
    background_roi.png
    mask_raw_roi.png
    mask_edit_roi.png
    mask_paste_roi.png
        |
        +--> prepare_lora_data -> train_inpaint_lora -> lora_ctwirecrack_v2
        |
        v
generate_cracks
        |
        v
run_<timestamp>/samples/<record_id>/image_syn.png
        |
        v
evaluate_generation
```

## 数据结构

默认数据集路径：

```text
C:/Users/18046/Desktop/master/masterthesis/dataset_real
```

期望结构：

```text
dataset_real/
├─ crack/
│  ├─ 14_crop.jpg
│  ├─ 14_crop.json
│  └─ ...
└─ normal/
   ├─ 12_crop.jpg
   └─ ...
```

每张 crack 图必须有同名 LabelMe JSON 标注。当前本地图片本身就是 `512x512`，所以 `prepare_rois` 不再做额外空间裁小；它主要负责生成统一的 ROI 资产和 mask。真正的局部裁剪发生在 `generate_cracks` 的 `local_inpaint` 阶段，以及 LoRA 训练数据的多尺度 crop 阶段。

## 配对规则

同一视频配对不是由代码自动识别出来的。代码只做两件事：

- 优先读取数据集根目录或 `mapping/` 子目录下的显式配对文件，例如 `pairs.csv`、`mapping.csv`、`pairs.json`。
- 如果没有 mapping 文件，就按文件名中的数字排序后，将 `crack/` 和 `normal/` 一一配对。

如果后面混入多个视频，必须提供显式 mapping，或者按视频分别运行；否则排序配对可能跨视频错配。

## 配置

唯一配置文件：

```text
configs/methods/diffusion_v2/contact_wire_v2.yaml
```

关键设置：

- `output_root`: `artifacts/dataset_real/methods/diffusion_v2`
- `roi_out_size`: 512
- `crack_prior_mode`: `paired_residual`
- `local_inpaint`: true
- `candidate_local_crop_sizes`: `[128, 160]`
- `candidate_strengths`: `[0.35, 0.45, 0.55]`
- `candidate_residual_alphas`: `[0.35, 0.65]`
- `lora_path`: `artifacts/dataset_real/methods/diffusion_v2/lora_ctwirecrack_v2`

## CLI

安装环境：

```bash
conda env create -f environment.yml
conda activate crack-synth
pip install -e .
```

准备 ROI 和 mask：

```bash
prepare_rois --config configs/methods/diffusion_v2/contact_wire_v2.yaml
```

准备 LoRA 训练样本：

```bash
prepare_lora_data --config configs/methods/diffusion_v2/contact_wire_v2.yaml
```

训练 v2 LoRA：

```bash
train_inpaint_lora \
  --train-data-dir artifacts/dataset_real/methods/diffusion_v2/lora_data/train \
  --output-dir artifacts/dataset_real/methods/diffusion_v2/lora_ctwirecrack_v2 \
  --rank 16 \
  --lora-alpha 16 \
  --max-train-steps 2000
```

只检查生成计划：

```bash
generate_cracks --config configs/methods/diffusion_v2/contact_wire_v2.yaml --plan-only
```

执行生成：

```bash
generate_cracks --config configs/methods/diffusion_v2/contact_wire_v2.yaml
```

评估最近一次生成：

```bash
evaluate_generation --latest --config configs/methods/diffusion_v2/contact_wire_v2.yaml
```

## 产物

`prepare_rois` 输出：

```text
artifacts/dataset_real/methods/diffusion_v2/roi_assets/
```

`prepare_lora_data` 输出：

```text
artifacts/dataset_real/methods/diffusion_v2/lora_data/
```

`generate_cracks` 输出：

```text
artifacts/dataset_real/methods/diffusion_v2/run_<timestamp>/
```

每个 sample 包含：

```text
image_syn.png
background_roi.png
mask_raw_roi.png
mask_edit_roi.png
mask_paste_roi.png
metadata.json
debug/
```

评估结果写入：

```text
artifacts/dataset_real/methods/diffusion_v2/run_<timestamp>/evaluation/
```
