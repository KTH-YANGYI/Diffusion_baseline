from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import math
import random

import numpy as np
from PIL import Image

from crack_synth.methods.diffusion_baseline.io_utils import ensure_dir, write_json
from crack_synth.methods.diffusion_baseline.progress import tqdm


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train a UNet LoRA for Stable Diffusion inpainting crack synthesis.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        "--pretrained-model-name-or-path",
        dest="pretrained_model_name_or_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
    )
    parser.add_argument("--train_data_dir", "--train-data-dir", dest="train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", "--train-batch-size", dest="train_batch_size", type=int, default=1)
    parser.add_argument(
        "--gradient_accumulation_steps",
        "--gradient-accumulation-steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_weight_decay", "--adam-weight-decay", dest="adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_alpha", "--lora-alpha", dest="lora_alpha", type=int, default=4)
    parser.add_argument("--target_modules", "--target-modules", dest="target_modules", nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"])
    parser.add_argument("--max_train_steps", "--max-train-steps", dest="max_train_steps", type=int, default=1200)
    parser.add_argument("--checkpointing_steps", "--checkpointing-steps", dest="checkpointing_steps", type=int, default=300)
    parser.add_argument("--mixed_precision", "--mixed-precision", dest="mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--mask_loss_weight", "--mask-loss-weight", dest="mask_loss_weight", type=float, default=5.0)
    parser.add_argument("--max_train_samples", "--max-train-samples", dest="max_train_samples", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", "--dataloader-num-workers", dest="dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--local_files_only", "--local-files-only", dest="local_files_only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


def train(args: Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
    from peft import LoraConfig
    from torch.utils.data import DataLoader
    from transformers import CLIPTextModel, CLIPTokenizer

    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if args.train_batch_size <= 0:
        raise ValueError("--train_batch_size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive.")
    if args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be positive.")
    if args.mask_loss_weight < 0:
        raise ValueError("--mask_loss_weight cannot be negative.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    write_json(output_dir / "training_args.json", _jsonable_vars(args))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    weight_dtype = _resolve_weight_dtype(args.mixed_precision, torch)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        local_files_only=args.local_files_only,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        local_files_only=args.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        local_files_only=args.local_files_only,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        local_files_only=args.local_files_only,
    )
    if int(unet.config.in_channels) != 9:
        raise ValueError(f"Expected an inpainting UNet with 9 input channels, got {unet.config.in_channels}.")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=args.target_modules,
    )
    unet.add_adapter(lora_config)

    trainable_params = [param for param in unet.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay,
    )

    dataset = InpaintLoraDataset(
        args.train_data_dir,
        resolution=args.resolution,
        max_samples=args.max_train_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_batch,
    )

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    text_encoder.eval()
    unet.train()

    global_step = 0
    completed_epochs = 0
    steps_per_epoch = max(1, math.ceil(len(dataloader) / args.gradient_accumulation_steps))
    total_epochs = math.ceil(args.max_train_steps / steps_per_epoch)
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="train_inpaint_lora",
        unit="step",
        disable=not accelerator.is_local_main_process,
    )

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                target_images = batch["target"].to(accelerator.device, dtype=weight_dtype)
                condition_images = batch["condition"].to(accelerator.device, dtype=weight_dtype)
                masks = batch["mask"].to(accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    latents = vae.encode(target_images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    masked_condition = condition_images * (masks < 0.5)
                    masked_image_latents = vae.encode(masked_condition).latent_dist.sample()
                    masked_image_latents = masked_image_latents * vae.config.scaling_factor

                    input_ids = tokenizer(
                        batch["caption"],
                        max_length=tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(accelerator.device)
                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                mask_latents = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
                model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)

                noise_pred = unet(model_input, timesteps, encoder_hidden_states).sample
                target = _prediction_target(noise_scheduler, latents, noise, timesteps)
                mse = (noise_pred.float() - target.float()) ** 2
                loss_weight = 1.0 + float(args.mask_loss_weight) * mask_latents.float()
                loss = (mse * loss_weight).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.detach().float().item():.4f}")
                if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    _save_lora_weights(output_dir / f"checkpoint-{global_step}", unet, accelerator, StableDiffusionInpaintPipeline)
                if global_step >= args.max_train_steps:
                    break
        completed_epochs += 1
        if completed_epochs > total_epochs + 2:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_lora_weights(output_dir, unet, accelerator, StableDiffusionInpaintPipeline)
        write_json(
            output_dir / "final_summary.json",
            {
                "global_step": int(global_step),
                "train_sample_count": len(dataset),
                "train_data_dir": str(Path(args.train_data_dir).resolve()),
                "output_dir": str(output_dir),
                "rank": int(args.rank),
                "lora_alpha": int(args.lora_alpha),
                "target_modules": list(args.target_modules),
                "mask_loss_weight": float(args.mask_loss_weight),
            },
        )


class InpaintLoraDataset:
    def __init__(self, root: str | Path, *, resolution: int, max_samples: int | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Train data directory does not exist: {self.root}")
        sample_dirs = [
            path for path in sorted(self.root.iterdir())
            if (path / "condition.png").exists()
            and (path / "target.png").exists()
            and (path / "mask_train.png").exists()
            and (path / "caption.txt").exists()
        ]
        if max_samples is not None:
            sample_dirs = sample_dirs[:max_samples]
        if not sample_dirs:
            raise ValueError(f"No LoRA training samples found under {self.root}")
        self.sample_dirs = sample_dirs
        self.resolution = int(resolution)

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict:
        sample_dir = self.sample_dirs[index]
        return {
            "condition": _load_rgb_tensor(sample_dir / "condition.png", self.resolution),
            "target": _load_rgb_tensor(sample_dir / "target.png", self.resolution),
            "mask": _load_mask_tensor(sample_dir / "mask_train.png", self.resolution),
            "caption": (sample_dir / "caption.txt").read_text(encoding="utf-8").strip(),
            "sample_id": sample_dir.name,
        }


def collate_batch(examples: list[dict]) -> dict:
    import torch

    return {
        "condition": torch.stack([example["condition"] for example in examples]),
        "target": torch.stack([example["target"] for example in examples]),
        "mask": torch.stack([example["mask"] for example in examples]),
        "caption": [example["caption"] for example in examples],
        "sample_id": [example["sample_id"] for example in examples],
    }


def _load_rgb_tensor(path: Path, resolution: int):
    import torch

    with Image.open(path) as image:
        image = image.convert("RGB").resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def _load_mask_tensor(path: Path, resolution: int):
    import torch

    with Image.open(path) as image:
        image = image.convert("L").resize((resolution, resolution), resample=Image.Resampling.NEAREST)
    array = (np.asarray(image, dtype=np.float32) > 127.0).astype(np.float32)
    return torch.from_numpy(array[None, :, :])


def _prediction_target(noise_scheduler, latents, noise, timesteps):
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def _resolve_weight_dtype(mixed_precision: str, torch):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _save_lora_weights(output_dir: Path, unet, accelerator, pipeline_cls) -> None:
    import torch
    from peft import get_peft_model_state_dict

    output_dir = ensure_dir(output_dir)
    unwrapped_unet = accelerator.unwrap_model(unet)
    state_dict = get_peft_model_state_dict(unwrapped_unet)
    try:
        from diffusers.utils import convert_state_dict_to_diffusers

        state_dict = convert_state_dict_to_diffusers(state_dict)
        pipeline_cls.save_lora_weights(
            save_directory=str(output_dir),
            unet_lora_layers=state_dict,
        )
    except Exception:
        from safetensors.torch import save_file

        cpu_state = {f"unet.{key}": value.detach().cpu() for key, value in state_dict.items()}
        save_file(cpu_state, str(output_dir / "pytorch_lora_weights.safetensors"))
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def _jsonable_vars(args: Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


if __name__ == "__main__":
    main()
