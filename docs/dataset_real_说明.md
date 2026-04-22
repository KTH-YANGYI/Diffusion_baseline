# dataset_real 数据集说明

`dataset_real` 是 `diffusion_v2` 使用的一一配对真实数据集。数据本身不提交到 Git；仓库只保留代码、配置和说明。

## 本地位置

当前配置默认读取：

```text
C:/Users/18046/Desktop/master/masterthesis/dataset_real
```

如需换机器或换目录，修改：

```text
configs/methods/diffusion_v2/contact_wire_v2.yaml
```

里的 `dataset_root`。

## 目录结构

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
- 图片尺寸：全部 `512x512`

## 标注格式

缺陷标注文件是 LabelMe 风格 JSON。代码读取 `shapes[*].points`，支持：

- `polygon`
- `rectangle`
- `circle`
- `line`
- `linestrip`

生成的关键 mask：

- `mask_raw_roi.png`: JSON 原始标注区域。
- `mask_edit_roi.png`: raw mask 按 `mask_edit_dilate_px` 外扩后的 inpainting 区域。
- `mask_paste_roi.png`: 后处理 paste 时使用的柔和融合区域。

## 配对规则

代码不会从图片内容自动判断“同一视频”。同一视频配对需要由数据组织方式保证。

优先级如下：

1. 读取显式配对文件。
2. 如果没有配对文件，按文件名中的数字排序后一一配对。

支持的显式配对文件名：

- `pairs.csv`
- `pair_mapping.csv`
- `crack_normal_pairs.csv`
- `crack_normal_mapping.csv`
- `mapping.csv`
- 对应的 `.json` 版本

CSV 支持常见列名：

- 缺陷图列：`defect_image`, `crack_image`, `defect`, `crack`, `image_name`, `crack_name`
- 正常图列：`normal_image`, `background_image`, `normal`, `background`, `normal_name`

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

如果以后混入多个视频，建议新增 `pairs.csv`，明确每张 crack 图对应哪张 normal 图，避免排序时跨视频配错。

## 在 v2 流程中的作用

每个 pair 会进入下面的数据流：

```text
crack image + crack JSON + normal image
    -> prepare_rois
    -> defect_roi.png
    -> background_roi.png
    -> mask_raw_roi.png
    -> mask_edit_roi.png
    -> mask_paste_roi.png
```

因为当前原图已经是 `512x512`，`prepare_rois` 的 `crop_box_xyxy` 通常是 `[0, 0, 512, 512]`。它的主要价值是生成统一资产和 mask，而不是再把原图裁小。

后续真正局部化编辑发生在：

- `generate_cracks` 的 `local_inpaint`：围绕裂缝 mask 裁 `128/160` 小窗口，resize 到 512 后生成，再贴回原图。
- `prepare_lora_data`：围绕裂缝 mask 裁 `96/128/160/192/256` 多尺度训练样本。

复现检查：

```bash
prepare_rois --config configs/methods/diffusion_v2/contact_wire_v2.yaml
```

成功后应看到：

```text
artifacts/dataset_real/methods/diffusion_v2/roi_assets/roi_pairs.jsonl
```
