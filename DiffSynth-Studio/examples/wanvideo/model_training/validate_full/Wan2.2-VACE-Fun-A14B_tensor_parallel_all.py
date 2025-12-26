import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

    reference_video_path: str = ""
    reference_image_path: str = ""
    out_path: str = ""


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
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()
    return s.replace("IMG1 ", "").strip()


# -----------------------------
# Batch test_dir helper
# -----------------------------
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_VID_EXTS = (".mp4",)
_TXT_EXTS = (".txt",)


def _list_files_with_ext(d: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for fn in os.listdir(d):
        p = os.path.join(d, fn)
        if os.path.isfile(p) and fn.lower().endswith(exts):
            out.append(p)
    return sorted(out)


def build_tasks_from_test_dir(
    test_dir: str,
    out_dir: str,
    negative_prompt: str,
    seed: int,
    num_frames: int,
    height: int,
    width: int,
) -> List[InferTask]:
    """
    遍历 test_dir 下一级文件夹，每个子文件夹必须包含：
      - 唯一 txt: prompt（读取时去掉 "IMG1 "）
      - 唯一 mp4: reference video
      - 唯一 png/jpg/jpeg/webp: reference image
    输出写到 out_dir/<subfolder>.mp4
    """
    if not os.path.isdir(test_dir):
        raise NotADirectoryError(f"--test_dir is not a directory: {test_dir}")

    os.makedirs(out_dir, exist_ok=True)

    subdirs = []
    for name in sorted(os.listdir(test_dir)):
        p = os.path.join(test_dir, name)
        if os.path.isdir(p):
            subdirs.append((name, p))

    if len(subdirs) == 0:
        raise RuntimeError(f"--test_dir has no subfolders: {test_dir}")

    tasks: List[InferTask] = []
    for sub_name, sub_path in subdirs:
        imgs = _list_files_with_ext(sub_path, _IMG_EXTS)
        vids = _list_files_with_ext(sub_path, _VID_EXTS)
        txts = _list_files_with_ext(sub_path, _TXT_EXTS)

        if len(imgs) != 1 or len(vids) != 1 or len(txts) != 1:
            msg = (
                f"[BAD SAMPLE] {sub_path}\n"
                f"  images({len(imgs)}): {imgs}\n"
                f"  videos({len(vids)}): {vids}\n"
                f"  txts({len(txts)}): {txts}\n"
                f"Expected exactly 1 image + 1 mp4 + 1 txt."
            )
            raise RuntimeError(msg)

        prompt = read_prompt_txt(txts[0])
        out_path = os.path.join(out_dir, f"{sub_name}.mp4")

        tasks.append(
            InferTask(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                tiled=True,
                height=height,
                width=width,
                num_frames=num_frames,
                reference_video_path=vids[0],
                reference_image_path=imgs[0],
                out_path=out_path,
            )
        )

    return tasks


# -----------------------------
# TP policy helpers
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


def _modules_to_parallelize(pipe: Any) -> List[nn.Module]:
    mods: List[nn.Module] = []
    for key in ["dit", "dit2", "vace", "vace2"]:
        m = getattr(pipe, key, None)
        if isinstance(m, nn.Module):
            mods.append(m)

    if not mods:
        for key in ["transformer", "unet", "model", "diffusion_model", "backbone", "net"]:
            m = getattr(pipe, key, None)
            if isinstance(m, nn.Module):
                mods.append(m)
                break

    if not mods:
        raise RuntimeError("找不到可 TP 的模块（dit/dit2/vace/vace2/backbone 都没有）。")
    return mods


def apply_tensor_parallel_to_pipe(pipe: Any) -> None:
    mods = _modules_to_parallelize(pipe)
    for backbone in mods:
        prepare_plan: Dict[str, Any] = {}
        for name, _child in backbone.named_modules():
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
            raise RuntimeError("TP plan 为空：没找到 Linear？请确认 backbone 是否正确。")
        parallelize_module(backbone, plan)

        try:
            parallelize_module(backbone, {"": SequenceParallel()})
        except Exception as e:
            if is_rank0():
                print(f"[WARN] SequenceParallel 启用失败（可忽略/可注释掉）：{e}")


# -----------------------------
# Lightning module
# -----------------------------
class LitWanVACEA14BInfer(pl.LightningModule):
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

        # ✅ VRAM 管理：仅对 T5/VAE 做 offload，且 preparing/computation 绑到本 rank
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": device_str,
            "computation_dtype": torch.bfloat16,
            "computation_device": device_str,
        }

        total_gb = torch.cuda.mem_get_info(device_str)[1] / (1024**3)
        vram_limit = max(1.0, total_gb - 2.0)

        if is_rank0():
            print(f"[rank {get_rank()}] init Wan2.2 A14B pipe on {device_str}, vram_limit={vram_limit:.2f} GB")

        # ✅ 按官方方式：high/low diffusion 不加 vram_config，T5/VAE 加
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device_str,
            model_configs=[
                ModelConfig(
                    model_id="PAI/Wan2.2-VACE-Fun-A14B",
                    origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                ),
                ModelConfig(
                    model_id="PAI/Wan2.2-VACE-Fun-A14B",
                    origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                ),
                ModelConfig(
                    model_id="PAI/Wan2.2-VACE-Fun-A14B",
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                    **vram_config,
                ),
                ModelConfig(
                    model_id="PAI/Wan2.2-VACE-Fun-A14B",
                    origin_file_pattern="Wan2.1_VAE.pth",
                    **vram_config,
                ),
            ],
            vram_limit=vram_limit,
            # use_usp=True,  # 需要的话你可以打开
        )

        # ✅ finetune ckpt：high -> vace, low -> vace2（CPU load 省显存）
        if self.args.ckpt_high:
            sd = load_state_dict(self.args.ckpt_high, torch_dtype=torch.bfloat16, device="cpu")
            missing, unexpected = self.pipe.vace.load_state_dict(sd, strict=False)
            if is_rank0():
                print(f"[rank0] loaded ckpt_high -> pipe.vace: {self.args.ckpt_high}")
                if missing:
                    print(f"[rank0][WARN] vace missing keys: {len(missing)}")
                if unexpected:
                    print(f"[rank0][WARN] vace unexpected keys: {len(unexpected)}")

        if self.args.ckpt_low:
            if not hasattr(self.pipe, "vace2"):
                raise AttributeError("pipe has no attribute 'vace2' but --ckpt_low is provided.")
            sd = load_state_dict(self.args.ckpt_low, torch_dtype=torch.bfloat16, device="cpu")
            missing, unexpected = self.pipe.vace2.load_state_dict(sd, strict=False)
            if is_rank0():
                print(f"[rank0] loaded ckpt_low -> pipe.vace2: {self.args.ckpt_low}")
                if missing:
                    print(f"[rank0][WARN] vace2 missing keys: {len(missing)}")
                if unexpected:
                    print(f"[rank0][WARN] vace2 unexpected keys: {len(unexpected)}")

        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

    def setup(self, stage: str):
        self._init_pipe_if_needed()

    def configure_model(self):
        self._init_pipe_if_needed()
        if self._tp_applied:
            return
        apply_tensor_parallel_to_pipe(self.pipe)
        self._tp_applied = True
        if is_rank0():
            print("[rank0] Tensor parallel applied to Wan2.2 A14B in configure_model().")

    def predict_step(self, batch: InferTask, batch_idx: int, dataloader_idx: int = 0):
        task: InferTask = batch

        # reference video -> list of frames (truncate num_frames)
        v = VideoData(task.reference_video_path, height=task.height, width=task.width)
        v = [v[i] for i in range(task.num_frames)]

        # reference image
        ref_img = Image.open(task.reference_image_path).convert("RGB").resize((task.width, task.height))

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_video = self.pipe(
                prompt=task.prompt,
                negative_prompt=task.negative_prompt,
                vace_video=v,
                vace_reference_image=ref_img,
                num_frames=task.num_frames,  # ✅ 必须传入
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

    # ckpt high/low
    parser.add_argument("--ckpt_high", type=str, default="", help="Optional. Fine-tuned HIGH noise ckpt -> pipe.vace")
    parser.add_argument("--ckpt_low", type=str, default="", help="Optional. Fine-tuned LOW noise ckpt -> pipe.vace2")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)

    # 单样本模式（旧接口）
    parser.add_argument("--ref_video", type=str, default="")
    parser.add_argument("--ref_image", type=str, default="")
    parser.add_argument("--input_txt", type=str, default="")
    parser.add_argument("--prompt", type=str, default="from sunset to night, a small town, light, house, river")
    parser.add_argument("--out", type=str, default="video_Wan2.2-VACE-Fun-A14B.mp4")

    # ✅ 新增：批量模式
    parser.add_argument("--test_dir", type=str, default="", help="If set, iterate subfolders under this dir.")
    parser.add_argument("--out_dir", type=str, default="", help="Output dir for batch mode (one mp4 per subfolder).")

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    )
    args = parser.parse_args()

    # 绑定本地卡，避免 lightning 早期落到 cuda:0
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    # -----------------------------
    # Build tasks
    # -----------------------------
    tasks: List[InferTask] = []

    if args.test_dir:
        if not args.out_dir:
            raise ValueError("When --test_dir is set, you must also provide --out_dir.")
        tasks = build_tasks_from_test_dir(
            test_dir=args.test_dir,
            out_dir=args.out_dir,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
        )
        if is_rank0():
            print(f"[rank0] batch mode: found {len(tasks)} samples under {args.test_dir}")

    else:
        # 单样本模式：保持你原来的调用方式
        if args.input_txt:
            if not os.path.exists(args.input_txt):
                raise FileNotFoundError(f"--input_txt not found: {args.input_txt}")
            args.prompt = read_prompt_txt(args.input_txt)
            if is_rank0():
                print(f"[rank0] prompt loaded from txt ({args.input_txt}): {args.prompt[:120]}")

        if not args.ref_video or not args.ref_image:
            raise ValueError("Single-sample mode requires --ref_video and --ref_image (or use --test_dir for batch).")

        tasks = [
            InferTask(
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
        ]

    ds = ToyDataset(tasks)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    model = LitWanVACEA14BInfer(args)

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