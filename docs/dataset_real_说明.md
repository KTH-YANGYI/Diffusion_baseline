# dataset_real 数据集说明

`dataset_real` 是当前扩散补全流程使用的一一配对数据集。数据本身不提交到 GitHub；仓库只保留代码、配置和本说明。

## 本地位置

当前配置默认读取：

```text
C:/Users/18046/Desktop/master/masterthesis/dataset_real
```

如需在别的机器复现，可以把数据放到任意位置，然后修改：

```text
configs/methods/diffusion_baseline/contact_wire_v1.yaml
```

里的 `dataset_root`。

## 目录结构

数据集根目录应包含两个子目录：

```text
dataset_real/
├─ crack/
│  ├─ 14_crop.jpg
│  ├─ 14_crop.json
│  ├─ 25_crop.jpg
│  ├─ 25_crop.json
│  └─ ...
└─ normal/
   ├─ 12_crop.jpg
   ├─ 21_crop.jpg
   └─ ...
```

`crack/` 中每张缺陷图必须有一个同名 `.json` 标注文件。`normal/` 中保存对应的无裂缝正常背景图。

当前本地数据规模：

- `crack/`: 28 张 `.jpg`
- `crack/`: 28 个同名 `.json`
- `normal/`: 28 张 `.jpg`

## 标注格式

缺陷标注文件是 LabelMe 风格 JSON。关键字段包括：

- `imagePath`
- `imageHeight`
- `imageWidth`
- `shapes`

代码会读取 `shapes[*].points`，并支持以下 `shape_type`：

- `polygon`
- `rectangle`
- `circle`
- `line`
- `linestrip`

生成 ROI 时，JSON 标注会被转换成二值 mask：

- `mask_raw_roi.png`: JSON 原始标注区域
- `mask_edit_roi.png`: 在 raw mask 基础上按 `mask_edit_dilate_px` 外扩后的 inpainting 区域

当前默认：

```yaml
mask_edit_dilate_px: 3
```

## 配对规则

代码优先读取显式配对文件。支持放在数据集根目录或 `mapping/` 子目录下：

- `pairs.csv`
- `pair_mapping.csv`
- `crack_normal_pairs.csv`
- `crack_normal_mapping.csv`
- `mapping.csv`
- 对应的 `.json` 版本

CSV 支持常见列名：

- 缺陷图列：`defect_image`, `crack_image`, `defect`, `crack`, `image_name`, `crack_name`
- 正常图列：`normal_image`, `background_image`, `normal`, `background`, `normal_name`

如果没有显式配对文件，代码会按文件名中的数字排序后，将 `crack/` 和 `normal/` 一一配对。

当前本地数据没有显式 mapping 文件，因此使用 `sorted_filename_order`。当前解析出的配对为：

| crack image | normal image |
| --- | --- |
| `14_crop.jpg` | `12_crop.jpg` |
| `25_crop.jpg` | `21_crop.jpg` |
| `33_crop.jpg` | `29_crop.jpg` |
| `41_crop.jpg` | `37_crop.jpg` |
| `48_crop.jpg` | `46_crop.jpg` |
| `61_crop.jpg` | `55_crop.jpg` |
| `74_crop.jpg` | `72_crop.jpg` |
| `81_crop.jpg` | `79_crop.jpg` |
| `88_crop.jpg` | `86_crop.jpg` |
| `132_crop.jpg` | `130_crop.jpg` |
| `139_crop.jpg` | `137_crop.jpg` |
| `156_crop.jpg` | `154_crop.jpg` |
| `164_crop.jpg` | `162_crop.jpg` |
| `181_crop.jpg` | `179_crop.jpg` |
| `200_crop.jpg` | `196_crop.jpg` |
| `210_crop.jpg` | `206_crop.jpg` |
| `219_crop.jpg` | `215_crop.jpg` |
| `231_crop.jpg` | `225_crop.jpg` |
| `240_crop.jpg` | `236_crop.jpg` |
| `250_crop.jpg` | `246_crop.jpg` |
| `260_crop.jpg` | `256_crop.jpg` |
| `270_crop.jpg` | `266_crop.jpg` |
| `282_crop.jpg` | `278_crop.jpg` |
| `292_crop.jpg` | `288_crop.jpg` |
| `303_crop.jpg` | `299_crop.jpg` |
| `313_crop.jpg` | `309_crop.jpg` |
| `323_crop.jpg` | `319_crop.jpg` |
| `341_crop.jpg` | `337_crop.jpg` |

## 当前生成流程中的作用

当前 pipeline 对每个 pair 执行：

1. 读取 `crack/*.jpg` 和同名 `.json`
2. 将 JSON 标注转成 mask
3. 读取对应的 `normal/*.jpg`
4. 用相同 crop box 裁出缺陷 ROI 和正常背景 ROI
5. 用正常背景 ROI + edit mask 做 Stable Diffusion Inpainting

缺陷图不会作为视觉条件输入模型；它只用于提供 mask 和 ROI 位置。

## 复现检查

放好数据后，可以运行：

```bash
prepare_rois --config configs/methods/diffusion_baseline/contact_wire_v1.yaml
```

成功后应看到：

```text
artifacts/dataset_real/methods/diffusion_baseline/roi_assets/roi_pairs.jsonl
```

其中 `pair_count` 应等于本地数据配对数量。当前数据是 `28`。
