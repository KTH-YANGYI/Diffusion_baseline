# Crack Synth Workspace

这个仓库现在按“任务域”组织，而不是按某一条方法线组织。

当前已经接入的一条方法线是：

- `diffusion_baseline`

它的定位是：

- 使用预训练 `Stable Diffusion Inpainting`
- 用真实裂缝 `mask` 提供几何约束
- 在正常背景 patch 上做局部裂缝生成
- 当前不包含 ControlNet、LoRA、DreamBooth、IP-Adapter

## 目录结构

```text
.
├─ configs/
│  └─ methods/
│     └─ diffusion_baseline/
│        └─ contact_wire_v1.yaml
├─ data/
│  └─ contact_wire_v1/
├─ artifacts/
│  └─ contact_wire_v1/
│     ├─ manifests/
│     ├─ methods/
│     │  └─ diffusion_baseline/
│     └─ archives/
├─ docs/
│  ├─ 工程分析.md
│  ├─ 服务器运行说明.md
│  └─ 铁路接触线裂缝生成技术路线_保姆级.md
└─ src/
   └─ crack_synth/
      └─ methods/
         └─ diffusion_baseline/
```

## 当前方法线的真实输入

当前 `diffusion_baseline` 在推理时真正使用的是：

- 正常背景 patch
- `mask_edit_roi`
- `prompt`
- `negative_prompt`

当前 donor 原图不会直接作为视觉条件输入模型。

## 配置文件

当前默认配置文件是：

```text
configs/methods/diffusion_baseline/contact_wire_v1.yaml
```

这个配置里显式声明了：

- `dataset_root`
- `manifests_root`
- `output_root`

这样后面再接别的方法线或别的数据线时，不需要再依赖硬编码顶层目录。

## CLI

保留现有命令名不变：

```bash
build_manifests --config configs/methods/diffusion_baseline/contact_wire_v1.yaml
prepare_rois --fold 0 --config configs/methods/diffusion_baseline/contact_wire_v1.yaml
generate_baseline --fold 0 --config configs/methods/diffusion_baseline/contact_wire_v1.yaml
```

## 产物位置

`build_manifests` 输出到：

```text
artifacts/contact_wire_v1/manifests/
```

`prepare_rois` 和 `generate_baseline` 输出到：

```text
artifacts/contact_wire_v1/methods/diffusion_baseline/
```

历史服务器拷回结果归档到：

```text
artifacts/contact_wire_v1/archives/server_exports/
```

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

## 说明文档

- 项目分析：`docs/工程分析.md`
- 服务器运行说明：`docs/服务器运行说明.md`
- 技术路线背景说明：`docs/铁路接触线裂缝生成技术路线_保姆级.md`
