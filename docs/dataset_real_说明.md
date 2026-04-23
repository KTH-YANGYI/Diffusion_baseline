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
- 当前标注统计日期：2026-04-23

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

## 当前缺陷标注统计

统计口径：

- 统计对象：`dataset_real/crack/*.json` 中的 LabelMe 缺陷标注。
- mask 生成逻辑：与工程代码 `load_labelme_mask` 保持一致，`polygon` 直接填充，`line/linestrip` 按 3 像素宽绘制。
- 当前 28 个标注全部为 `polygon`，label 全部为 `crack`。
- 每张缺陷图当前只有 1 个缺陷 shape。
- 图像面积统一为 `512 x 512 = 262144` 像素。

总体统计：

| 指标 | min | p25 | median | mean | p75 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw mask 像素数 | 47.0 | 98.5 | 183.5 | 303.4 | 260.2 | 1873.0 |
| raw mask 图像占比 | 0.0179% | 0.0376% | 0.0700% | 0.1157% | 0.0993% | 0.7145% |
| bbox 宽度 px | 13.0 | 17.0 | 21.5 | 24.6 | 28.2 | 58.0 |
| bbox 高度 px | 5.0 | 8.0 | 12.5 | 16.8 | 21.2 | 55.0 |
| bbox 面积 px | 65.0 | 166.0 | 284.5 | 513.9 | 450.0 | 2750.0 |
| bbox 图像占比 | 0.0248% | 0.0633% | 0.1085% | 0.1960% | 0.1717% | 1.0490% |
| raw mask / bbox 面积 | 35.25% | 57.32% | 65.87% | 63.32% | 72.26% | 76.84% |
| edit mask 像素数，dilate=4 | 255.0 | 392.5 | 549.5 | 720.7 | 706.2 | 2900.0 |
| edit mask 图像占比，dilate=4 | 0.0973% | 0.1497% | 0.2096% | 0.2749% | 0.2694% | 1.1063% |

结论：

- 当前数据集中的裂缝是典型小目标。raw mask 中位数只有 `183.5` 像素，占整图 `0.0700%`。
- 75% 的样本 raw mask 占比不超过 `0.0993%`，说明整图直接扩散生成时裂缝信号很容易被背景淹没。
- bbox 中位尺寸约为 `21.5 x 12.5` 像素，缺陷局部范围很小；这也是 `local_inpaint` 需要围绕 mask 裁小窗口再放大到 512 生成的原因。
- 当前 `mask_edit_dilate_px=4` 后，edit mask 中位占比提升到 `0.2096%`，仍然属于很小的编辑区域，有利于保持背景稳定。
- 目前数据还没有工件级 `workpiece/contact_wire` mask；如果后续要约束裂缝只能出现在接触线本体内，需要为 crack/normal 图增加工件 mask 或自动提取工件有效区域。

逐样本统计：

| crack image | shapes | raw mask px | raw mask % | bbox wxh | bbox % | raw/bbox % | edit mask % |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `14_crop.jpg` | 1 | 838 | 0.3197% | 58x33 | 0.7301% | 43.78% | 0.6344% |
| `25_crop.jpg` | 1 | 101 | 0.0385% | 22x8 | 0.0671% | 57.39% | 0.1545% |
| `33_crop.jpg` | 1 | 228 | 0.0870% | 26x13 | 0.1289% | 67.46% | 0.2304% |
| `41_crop.jpg` | 1 | 217 | 0.0828% | 24x16 | 0.1465% | 56.51% | 0.2293% |
| `48_crop.jpg` | 1 | 156 | 0.0595% | 33x9 | 0.1133% | 52.53% | 0.2121% |
| `61_crop.jpg` | 1 | 713 | 0.2720% | 33x36 | 0.4532% | 60.02% | 0.5589% |
| `74_crop.jpg` | 1 | 457 | 0.1743% | 32x25 | 0.3052% | 57.12% | 0.3967% |
| `81_crop.jpg` | 1 | 318 | 0.1213% | 41x22 | 0.3441% | 35.25% | 0.3746% |
| `88_crop.jpg` | 1 | 748 | 0.2853% | 39x37 | 0.5505% | 51.84% | 0.5878% |
| `132_crop.jpg` | 1 | 1873 | 0.7145% | 50x55 | 1.0490% | 68.11% | 1.1063% |
| `139_crop.jpg` | 1 | 477 | 0.1820% | 27x24 | 0.2472% | 73.61% | 0.3693% |
| `156_crop.jpg` | 1 | 188 | 0.0717% | 24x13 | 0.1190% | 60.26% | 0.2090% |
| `164_crop.jpg` | 1 | 174 | 0.0664% | 24x11 | 0.1007% | 65.91% | 0.1976% |
| `181_crop.jpg` | 1 | 230 | 0.0877% | 24x16 | 0.1465% | 59.90% | 0.2361% |
| `200_crop.jpg` | 1 | 91 | 0.0347% | 17x8 | 0.0519% | 66.91% | 0.1354% |
| `210_crop.jpg` | 1 | 63 | 0.0240% | 14x8 | 0.0427% | 56.25% | 0.1156% |
| `219_crop.jpg` | 1 | 87 | 0.0332% | 15x8 | 0.0458% | 72.50% | 0.1278% |
| `231_crop.jpg` | 1 | 138 | 0.0526% | 19x10 | 0.0725% | 72.63% | 0.1656% |
| `240_crop.jpg` | 1 | 81 | 0.0309% | 14x8 | 0.0427% | 72.32% | 0.1225% |
| `250_crop.jpg` | 1 | 77 | 0.0294% | 15x7 | 0.0401% | 73.33% | 0.1209% |
| `260_crop.jpg` | 1 | 79 | 0.0301% | 15x8 | 0.0458% | 65.83% | 0.1247% |
| `270_crop.jpg` | 1 | 47 | 0.0179% | 13x5 | 0.0248% | 72.31% | 0.0973% |
| `282_crop.jpg` | 1 | 179 | 0.0683% | 21x12 | 0.0961% | 71.03% | 0.1934% |
| `292_crop.jpg` | 1 | 199 | 0.0759% | 19x17 | 0.1232% | 61.61% | 0.2102% |
| `303_crop.jpg` | 1 | 241 | 0.0919% | 17x21 | 0.1362% | 67.51% | 0.2323% |
| `313_crop.jpg` | 1 | 209 | 0.0797% | 17x16 | 0.1038% | 76.84% | 0.2113% |
| `323_crop.jpg` | 1 | 134 | 0.0511% | 18x12 | 0.0824% | 62.04% | 0.1671% |
| `341_crop.jpg` | 1 | 151 | 0.0576% | 19x11 | 0.0797% | 72.25% | 0.1766% |

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
