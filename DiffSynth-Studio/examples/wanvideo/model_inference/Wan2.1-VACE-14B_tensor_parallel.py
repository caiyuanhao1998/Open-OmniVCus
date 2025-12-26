import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import lightning as pl
from PIL import Image

import torch.distributed as dist
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
    给 parallelize_module 的 plan：
    - Linear: attention 的 qkv / out / mlp 走 Col/Row 切分
    - 其他不确定 Linear -> Replicate（保守）
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


def _apply_tp_to_transformer(backbone: nn.Module):
    """
    对一个 transformer/dit 模块做 TP。
    """
    # 1) 对 attention block 做输入准备（token 维 shard）
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

    # 2) Linear col/row 切分
    plan = _tp_plan_for_module(backbone)
    if not plan:
        raise RuntimeError("TP plan 为空：没找到 Linear 层？请确认 backbone 是否正确。")
    parallelize_module(backbone, plan)

    # 3) 顶层 sequence parallel（不兼容就降级）
    try:
        parallelize_module(backbone, {"": SequenceParallel()})
    except Exception as e:
        if is_rank0():
            print(f"[WARN] SequenceParallel 顶层启用失败（可忽略/可注释掉）：{e}")


def apply_tensor_parallel_to_pipe(pipe: Any) -> None:
    """
    对 WanVideoPipeline 做 TP：
    - pipe.dit 一定做
    - pipe.dit2 如果存在也做（Wan 在某些模型会在迭代中 switch DiT）
    - （可选）pipe.vace/vace2 也可以尝试做，但通常先只做 dit 最稳
    """
    if not hasattr(pipe, "dit") or not isinstance(pipe.dit, nn.Module):
        raise RuntimeError("pipe 没有 dit 模块，无法 TP。请检查 WanVideoPipeline 版本。")

    if is_rank0():
        print("[rank0] Applying TP to pipe.dit ...")
    _apply_tp_to_transformer(pipe.dit)

    if hasattr(pipe, "dit2") and isinstance(pipe.dit2, nn.Module) and pipe.dit2 is not None:
        if is_rank0():
            print("[rank0] Applying TP to pipe.dit2 ...")
        _apply_tp_to_transformer(pipe.dit2)

    # 如果你后面发现 vace 的显存/算力占比也很大，可以打开这段试试。
    # 但不同实现可能不全是标准 transformer，先注释最稳。
    #
    # if hasattr(pipe, "vace") and isinstance(pipe.vace, nn.Module) and pipe.vace is not None:
    #     if is_rank0():
    #         print("[rank0] (optional) Applying TP to pipe.vace ...")
    #     _apply_tp_to_transformer(pipe.vace)
    #
    # if hasattr(pipe, "vace2") and isinstance(pipe.vace2, nn.Module) and pipe.vace2 is not None:
    #     if is_rank0():
    #         print("[rank0] (optional) Applying TP to pipe.vace2 ...")
    #     _apply_tp_to_transformer(pipe.vace2)


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

        try:
            torch.set_default_device(device_str)
        except Exception:
            pass

        print(f"[rank {get_rank()}] init pipe on {device_str}, current_device={torch.cuda.current_device()}")

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device_str,  # ✅ 关键：别写 "cuda"
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth"),
            ],
            tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
            # ✅ 如果你愿意，Wan 自带 USP 通常比通用 SequenceParallel 更稳
            # use_usp=True,
        )

        # progress bar 兼容处理
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        elif hasattr(self.pipe, "progress_bar"):
            try:
                self.pipe.progress_bar = False
            except Exception:
                pass

    def setup(self, stage: str):
        self._init_pipe_if_needed()

    def configure_model(self):
        """
        ✅ ModelParallelStrategy 强制要求你实现这个 hook，并在这里 parallelize
        """
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
    # 让每个进程一开始就绑到自己的卡，避免某些库抢占 cuda:0
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./tp_outputs_vace14b_wan2.1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # rank0 下载示例数据
    if is_rank0():
        dataset_snapshot_download(
            dataset_id="DiffSynth-Studio/examples_in_diffsynth",
            local_dir="./",
            allow_file_pattern=[
                "data/examples/wan/depth_video.mp4",
                "data/examples/wan/cat_fightning.jpg",
            ],
        )
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
        devices=torch.cuda.device_count(),  # torchrun 下通常是每进程看到 1 张卡，也没问题
        strategy=ModelParallelStrategy(),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=is_rank0(),
    )
    trainer.predict(model, dataloaders=dl)


if __name__ == "__main__":
    main()