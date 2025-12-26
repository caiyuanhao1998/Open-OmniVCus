import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import lightning as pl
from PIL import Image

from lightning.pytorch.strategies import ModelParallelStrategy

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)
from torch.distributed._tensor import Replicate, Shard

from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


# -----------------------------
# Small task wrapper
# -----------------------------
@dataclass
class InferTask:
    prompt: str
    negative_prompt: str
    seed: int = 1
    tiled: bool = True
    height: int = 480
    width: int = 832
    num_frames: int = 49

    reference_video_path: str = "data/example_video_dataset/video1_softedge.mp4"
    reference_image_path: str = "data/example_video_dataset/reference.jpg"

    out_path: str = "video_Wan2.1-VACE-14B.mp4"


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


def collate_fn(xs: List[InferTask]) -> InferTask:
    assert len(xs) == 1
    return xs[0]


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


# -----------------------------
# prompt helper
# -----------------------------
def read_prompt_txt(path: str) -> str:
    """
    Read prompt from a txt file, remove the token/prefix "IMG1 " and strip spaces/newlines.
    """
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()
    s = s.replace("IMG1 ", "").strip()
    return s


# -----------------------------
# TP policy helpers（沿用你给的写法）
# -----------------------------
def _tp_plan_for_module(mod: nn.Module) -> Dict[str, Any]:
    plan: Dict[str, Any] = {}
    for name, child in mod.named_modules():
        if isinstance(child, nn.Linear):
            lname = name.lower()
            if any(k in lname for k in ["qkv", "to_qkv", "wqkv", "query_key_value"]):
                plan[name] = ColwiseParallel()
            elif any(k in lname for k in ["proj", "to_out", "out_proj", "wo"]):
                plan[name] = RowwiseParallel()
            elif any(k in lname for k in ["fc1", "w1", "up", "gate", "ffn_up", "mlp_up"]):
                plan[name] = ColwiseParallel()
            elif any(k in lname for k in ["fc2", "w2", "down", "ffn_down", "mlp_down"]):
                plan[name] = RowwiseParallel()
            else:
                plan[name] = Replicate()
    return plan


def _find_backbone_module(obj: Any) -> nn.Module:
    if isinstance(obj, nn.Module):
        candidates = ["dit", "transformer", "unet", "model", "diffusion_model", "backbone", "net"]
        for key in candidates:
            if hasattr(obj, key) and isinstance(getattr(obj, key), nn.Module):
                return getattr(obj, key)
        return obj

    candidates = ["dit", "transformer", "unet", "model", "diffusion_model", "backbone", "net"]
    for key in candidates:
        if hasattr(obj, key) and isinstance(getattr(obj, key), nn.Module):
            return getattr(obj, key)

    for name, val in getattr(obj, "__dict__", {}).items():
        if isinstance(val, nn.Module):
            n = name.lower()
            if any(k in n for k in ["dit", "transformer", "unet", "diffusion", "backbone"]):
                return val

    raise RuntimeError("找不到可 TP 的 backbone 模块（请检查 pipe.vace 里主干叫啥）。")


def apply_tensor_parallel_to_module(module_or_pipe: Any) -> None:
    backbone = _find_backbone_module(module_or_pipe)

    prepare_plan: Dict[str, Any] = {}
    for name, child in backbone.named_modules():
        lname = name.lower()
        if any(k in lname for k in ["attn", "self_attn", "attention"]):
            prepare_plan[name] = PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Shard(1),),
            )
    if prepare_plan:
        parallelize_module(backbone, prepare_plan)

    plan = _tp_plan_for_module(backbone)
    if not plan:
        raise RuntimeError("TP plan 为空：没找到 Linear 层？请确认 backbone 是否正确。")
    parallelize_module(backbone, plan)

    try:
        parallelize_module(backbone, {"": SequenceParallel()})
    except Exception as e:
        if is_rank0():
            print(f"[WARN] SequenceParallel 顶层启用失败（可忽略/可注释掉）：{e}")


# -----------------------------
# Lightning module (14B)
# -----------------------------
class LitWanVACE14BInfer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pipe: Optional[Any] = None
        self._tp_applied = False

    def _init_pipe_if_needed(self):
        if self.pipe is not None:
            return

        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"

        try:
            torch.set_default_device(device_str)
        except Exception:
            pass

        if is_rank0():
            print(f"[rank {get_rank()}] init 14B pipe on {device_str}")

        # ✅ 唯一变化：模型换成 Wan2.1-VACE-14B
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device_str,
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth"),
            ],
        )

        # ✅ ckpt 可选：不传就用官方权重
        if self.args.ckpt:
            state_dict = load_state_dict(self.args.ckpt)
            missing, unexpected = self.pipe.vace.load_state_dict(state_dict, strict=False)
            if is_rank0():
                print(f"[rank0] loaded ckpt={self.args.ckpt}")
                if missing:
                    print(f"[rank0][WARN] missing keys: {len(missing)}")
                if unexpected:
                    print(f"[rank0][WARN] unexpected keys: {len(unexpected)}")

        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

    def setup(self, stage: str):
        self._init_pipe_if_needed()

    def configure_model(self):
        self._init_pipe_if_needed()
        if self._tp_applied:
            return

        # ✅ 与 1.3B 一致：对 pipe.vace 做 TP
        apply_tensor_parallel_to_module(self.pipe.vace)

        self._tp_applied = True
        if is_rank0():
            print("[rank0] Tensor parallel applied to 14B pipe.vace in configure_model().")

    def predict_step(self, batch: InferTask, batch_idx: int, dataloader_idx: int = 0):
        task: InferTask = batch

        v = VideoData(task.reference_video_path, height=task.height, width=task.width)
        v = [v[i] for i in range(task.num_frames)]

        ref_img = Image.open(task.reference_image_path).convert("RGB").resize((task.width, task.height))

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_video = self.pipe(
                prompt=task.prompt,
                negative_prompt=task.negative_prompt,
                vace_video=v,
                vace_reference_image=ref_img,
                num_frames=task.num_frames,
                seed=task.seed,
                tiled=task.tiled,
            )

        if is_rank0():
            out_dir = os.path.dirname(task.out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            save_video(out_video, task.out_path, fps=15, quality=5)
            print(f"[rank0] saved: {task.out_path}")

        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Optional. If empty, use official pretrained weights.")
    parser.add_argument("--out", type=str, default="video_Wan2.1-VACE-14B.mp4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--ref_video", type=str, required=True)
    parser.add_argument("--ref_image", type=str, required=True)

    # ✅ 新接口：prompt 来自 txt
    parser.add_argument("--input_txt", type=str, default="", help="If set, read prompt from this txt and remove 'IMG1 '")
    parser.add_argument("--prompt", type=str, default="from sunset to night, a small town, light, house, river")

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    )
    args = parser.parse_args()

    # ✅ prompt 优先级：input_txt > --prompt
    if args.input_txt:
        if not os.path.exists(args.input_txt):
            raise FileNotFoundError(f"--input_txt not found: {args.input_txt}")
        args.prompt = read_prompt_txt(args.input_txt)
        if is_rank0():
            print(f"[rank0] prompt loaded from txt ({args.input_txt}): {args.prompt[:120]}")

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    task = InferTask(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        tiled=True,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        reference_video_path=args.ref_video,
        reference_image_path=args.ref_image,
        out_path=args.out,
    )

    ds = ToyDataset([task])
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = LitWanVACE14BInfer(args)

    trainer = pl.Trainer(
        accelerator="cuda",
        devices=torch.cuda.device_count(),
        strategy=ModelParallelStrategy(),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=is_rank0(),
    )

    trainer.predict(model, dataloaders=dl)


if __name__ == "__main__":
    main()