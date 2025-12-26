import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import lightning as pl
from PIL import Image

from lightning.pytorch.strategies import ModelParallelStrategy

import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)
from torch.distributed._tensor import Replicate, Shard

from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


# -----------------------------
# Small task wrapper
# -----------------------------
@dataclass
class InferTask:
    name: str
    prompt: str
    negative_prompt: str
    seed: int = 1
    tiled: bool = True
    ref_image_path: Optional[str] = None
    control_video_path: Optional[str] = None
    height: int = 480
    width: int = 832


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def dist_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# -----------------------------
# TP policy helpers
# -----------------------------
def _tp_plan_for_module(mod: nn.Module) -> Dict[str, Any]:
    """
    给 parallelize_module 的 plan（通用名字匹配）：
    - qkv: Colwise
    - out proj: Rowwise
    - ffn up/gate: Colwise
    - ffn down: Rowwise
    """
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
    """
    WanVideoPipeline 里真正会在 denoise loop 里跑的大块通常是：
    - dit / dit2
    - vace / vace2
    有就都做 TP（比只对一个 backbone 更合理）
    """
    mods: List[nn.Module] = []
    for key in ["dit", "dit2", "vace", "vace2"]:
        m = getattr(pipe, key, None)
        if isinstance(m, nn.Module):
            mods.append(m)

    # 兜底：若啥都没找到，就退回去找一个 backbone
    if not mods:
        candidates = ["transformer", "unet", "model", "diffusion_model", "backbone", "net"]
        for key in candidates:
            m = getattr(pipe, key, None)
            if isinstance(m, nn.Module):
                mods.append(m)
                break

    if not mods:
        raise RuntimeError("找不到可 TP 的模块（dit/dit2/vace/vace2/backbone 都没有）。")

    return mods


def apply_tensor_parallel_to_pipe(pipe: Any) -> None:
    """
    对 pipe 里的核心模块做 TP。
    注意：TP 只需要在每个 rank 初始化完模型后调用一次。
    """
    mods = _modules_to_parallelize(pipe)

    for backbone in mods:
        # 1) attn 输入 shard（token 维 shard）
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

        # 2) Linear col/row
        plan = _tp_plan_for_module(backbone)
        if not plan:
            raise RuntimeError("TP plan 为空：没找到 Linear？请确认 backbone 是否正确。")
        parallelize_module(backbone, plan)

        # 3) 顶层 sequence parallel（不兼容就自动降级）
        try:
            parallelize_module(backbone, {"": SequenceParallel()})
        except Exception as e:
            if is_rank0():
                print(f"[WARN] SequenceParallel 启用失败（可忽略/可注释掉）：{e}")


# -----------------------------
# Lightning module
# -----------------------------
class LitWanVACE14B(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pipe = None
        self._tp_applied = False

    def _init_pipe_if_needed(self):
        if self.pipe is not None:
            return

        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"

        # 可选：默认 device（推理脚本一般 OK）
        try:
            torch.set_default_device(device_str)
        except Exception:
            pass

        # ✅ 保留 vram_config，但把 cuda 绑到本 rank（否则 preparing/computation 可能偷跑到 cuda:0）
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": device_str,   # ✅ 原来是 "cuda"
            "computation_dtype": torch.bfloat16,
            "computation_device": device_str, # ✅ 原来是 "cuda"
        }

        # ✅ vram_limit 也要用本 rank 的 device
        total_gb = torch.cuda.mem_get_info(device_str)[1] / (1024 ** 3)
        vram_limit = max(1.0, total_gb - 2.0)

        print(
            f"[rank {get_rank()}] init pipe on {device_str}, "
            f"current_device={torch.cuda.current_device()}, vram_limit={vram_limit:.2f} GB"
        )

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device_str,
            model_configs=[
                ModelConfig(
                    model_id="PAI/Wan2.2-VACE-Fun-A14B",
                    origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                    **vram_config,
                ),
                ModelConfig(
                    model_id="PAI/Wan2.2-VACE-Fun-A14B",
                    origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                    **vram_config,
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
            tokenizer_config=ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            ),
            vram_limit=vram_limit,
            # 如果你后续想更稳的 sequence parallel（Wan 自带 USP），可以打开：
            # use_usp=True,
        )

        # 进度条兼容处理
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        elif hasattr(self.pipe, "progress_bar"):
            try:
                self.pipe.progress_bar = False
            except Exception:
                pass

    def setup(self, stage: str):
        # setup 里只建 pipe，不做 parallelize
        self._init_pipe_if_needed()

    def configure_model(self):
        # ✅ ModelParallelStrategy 要求在这里做 parallelize
        self._init_pipe_if_needed()

        if self._tp_applied:
            return

        apply_tensor_parallel_to_pipe(self.pipe)
        self._tp_applied = True

        if is_rank0():
            print("[rank0] Tensor parallel applied in configure_model().")

    def predict_step(self, batch: InferTask, batch_idx: int, dataloader_idx: int = 0):
        task: InferTask = batch

        ref_img = None
        if task.ref_image_path is not None:
            ref_img = Image.open(task.ref_image_path).resize((task.width, task.height))

        control_video = None
        if task.control_video_path is not None:
            control_video = VideoData(task.control_video_path, height=task.height, width=task.width)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            video = self.pipe(
                prompt=task.prompt,
                negative_prompt=task.negative_prompt,
                vace_reference_image=ref_img,
                vace_video=control_video,
                seed=task.seed,
                tiled=task.tiled,
            )

        if is_rank0():
            os.makedirs(self.args.output_dir, exist_ok=True)
            out_path = os.path.join(self.args.output_dir, f"{task.name}.mp4")
            save_video(video, out_path, fps=15, quality=5)
            print(f"[rank0] saved: {out_path}")

        return None


def collate_fn(xs: List[InferTask]) -> InferTask:
    assert len(xs) == 1
    return xs[0]


def main():
    # ✅ 尽早绑定本地卡，避免 lightning 初始化阶段默认落到 cuda:0
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./tp_outputs_14b")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 下载示例数据（只让 rank0 下载）
    if is_rank0():
        dataset_snapshot_download(
            dataset_id="DiffSynth-Studio/examples_in_diffsynth",
            local_dir="./",
            allow_file_pattern=[
                "data/examples/wan/depth_video.mp4",
                "data/examples/wan/cat_fightning.jpg",
            ],
        )

    # 分布式初始化后再 barrier 才有效；这里“尽力而为”
    dist_barrier()

    prompt = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
    neg = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )

    tasks = [
        InferTask(
            name="video_2_ref_image",
            prompt=prompt,
            negative_prompt=neg,
            seed=args.seed,
            tiled=True,
            ref_image_path="data/examples/wan/cat_fightning.jpg",
            control_video_path=None,
        ),
        InferTask(
            name="video_1_depth_video",
            prompt=prompt,
            negative_prompt=neg,
            seed=args.seed,
            tiled=True,
            ref_image_path=None,
            control_video_path="data/examples/wan/depth_video.mp4",
        ),
        InferTask(
            name="video_3_depth_plus_ref",
            prompt=prompt,
            negative_prompt=neg,
            seed=args.seed,
            tiled=True,
            ref_image_path="data/examples/wan/cat_fightning.jpg",
            control_video_path="data/examples/wan/depth_video.mp4",
        ),
    ]

    ds = ToyDataset(tasks)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = LitWanVACE14B(args)

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