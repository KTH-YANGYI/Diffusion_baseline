# Crack Synth Workspace

本项目现在只保留一条清晰流程：使用一一配对的真实数据，在对应正常图上做 Stable Diffusion Inpainting 裂缝生成。

旧流程中的 manifest、四折划分、normal pool 背景选择、manual case 都已经移除。

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

每张缺陷图需要有一个同名 JSON 标注文件。每张缺陷图只对应一张正常图，模型会直接在对应正常图上做 inpainting。

数据集不提交到 GitHub。详细格式、配对规则和当前本地样本列表见：

```text
docs/dataset_real_说明.md
```

## 配对规则

代码会优先读取数据集根目录或 `mapping/` 子目录下的显式配对文件，例如：

- `pairs.csv`
- `pair_mapping.csv`
- `crack_normal_pairs.csv`
- `crack_normal_mapping.csv`
- `mapping.csv`
- 对应的 `.json` 版本

如果没有显式配对文件，代码会按文件名中的数字排序后，将 `crack/` 和 `normal/` 一一配对。当前本地数据使用的就是这种 `sorted_filename_order` 配对方式。

## 方法逻辑

当前方法线是 `diffusion_baseline`，使用预训练 `Stable Diffusion Inpainting`。

缺陷图不作为视觉条件输入模型，只用于：

- 从 JSON 标注生成 raw mask
- 确定 ROI 裁剪位置
- 保存 `defect_roi.png` 作为对照

模型输入是：

- 对应正常图 ROI
- `mask_edit_roi`
- `prompt`
- `negative_prompt`

## 配置

默认配置文件：

```text
configs/methods/diffusion_baseline/contact_wire_v1.yaml
```

主要配置项：

- `dataset_root`: 一一配对数据集根目录
- `output_root`: ROI、生成结果和评估结果输出目录
- `roi_out_size`: 输入模型的 ROI 尺寸
- `mask_edit_dilate_px`: JSON raw mask 外扩像素半径
- `seeds_per_pair`: 每对样本生成几张
- `inference_batch_size`: 推理 batch size

## CLI

准备 ROI 和 mask：

```bash
prepare_rois --config configs/methods/diffusion_baseline/contact_wire_v1.yaml
```

只检查生成计划，不加载模型：

```bash
generate_baseline --config configs/methods/diffusion_baseline/contact_wire_v1.yaml --plan-only
```

执行 inpainting：

```bash
generate_baseline --config configs/methods/diffusion_baseline/contact_wire_v1.yaml
```

评估最近一次生成：

```bash
evaluate_generation --latest --config configs/methods/diffusion_baseline/contact_wire_v1.yaml
```

## 产物

`prepare_rois` 会生成：

```text
artifacts/dataset_real/methods/diffusion_baseline/roi_assets/
```

关键文件：

- `roi_pairs.jsonl`
- `pairs/<pair_id>/defect_roi.png`
- `pairs/<pair_id>/background_roi.png`
- `pairs/<pair_id>/mask_raw_roi.png`
- `pairs/<pair_id>/mask_edit_roi.png`

`generate_baseline` 会生成：

```text
artifacts/dataset_real/methods/diffusion_baseline/run_<timestamp>/
```

其中包含：

- `planned_pairs.jsonl`
- `outputs.jsonl`
- `samples/<record_id>/image_syn.png`
- `samples/<record_id>/background_roi.png`
- `samples/<record_id>/mask_raw_roi.png`
- `samples/<record_id>/mask_edit_roi.png`
- `samples/<record_id>/metadata.json`

## 环境

```bash
conda env create -f environment.yml
conda activate crack-synth
pip install -e .
```

如果服务器需要 CUDA 版 PyTorch，再单独安装对应轮子，例如：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
