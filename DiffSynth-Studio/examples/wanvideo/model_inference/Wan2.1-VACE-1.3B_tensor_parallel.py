import os
import torch
import torch.nn as nn
import lightning as pl
from PIL import Image
from tqdm import tqdm

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)
from torch.distributed._tensor import Replicate, Shard
from lightning.pytorch.strategies import ModelParallelStrategy

from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

from modelscope import dataset_snapshot_download


# -----------------------------------------------------------------------------
# Minimal dataset wrapper ------------------------------------------------------
# -----------------------------------------------------------------------------
class ToyDataset(torch.utils.data.Dataset):
    """Wrap a list of inference tasks so that we can feed it to a DataLoader."""

    def __init__(self, tasks):
        self.tasks = tasks

    def __getitem__(self, idx):
        return self.tasks[idx]

    def __len__(self):
        return len(self.tasks)


# -----------------------------------------------------------------------------
# LightningModule with Tensor Parallel DiT ------------------------------------
# -----------------------------------------------------------------------------
class LitWanVACE13B(pl.LightningModule):
    """Tensor-parallel version of Wan2.1-VACE-1.3B pipeline (DiT backbone)."""

    def __init__(self):
        super().__init__()

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[
                ModelConfig(
                    model_id="Wan-AI/Wan2.1-VACE-1.3B",
                    origin_file_pattern="diffusion_pytorch_model*.safetensors",
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.1-VACE-1.3B",
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.1-VACE-1.3B",
                    origin_file_pattern="Wan2.1_VAE.pth",
                ),
            ],
            tokenizer_config=ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            ),
        )

        # ===== TP DEBUG DUMP state (rank0 写文件) =====
        self._tp_debug = {
            "enabled": True,
            "dump_path": None,
            "captured": set(),  # module names captured once
            "lines": [],
        }

    def _rank(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _world_size(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    def _is_rank0(self):
        return self._rank() == 0

    def _append_dbg(self, s: str):
        # 只让 rank0 记日志，避免多卡写文件冲突
        if self._is_rank0() and self._tp_debug["enabled"]:
            self._tp_debug["lines"].append(s)

    def _format_tensor_info(self, x, name="tensor"):
        # 尽量不引入额外依赖；DTensor 用 duck-typing
        info = []
        info.append(f"[{name}] type={type(x)}")
        if torch.is_tensor(x):
            info.append(f"  shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
            placements = getattr(x, "placements", None)
            device_mesh = getattr(x, "device_mesh", None)
            if placements is not None or device_mesh is not None:
                info.append("  (DTensor-like)")
                if device_mesh is not None:
                    info.append(f"  device_mesh={device_mesh}")
                if placements is not None:
                    info.append(f"  placements={placements}")
        else:
            info.append(f"  (not a torch.Tensor)")
        return "\n".join(info)

    def _maybe_write_debug_file(self):
        if not (self._is_rank0() and self._tp_debug["enabled"]):
            return
        path = self._tp_debug["dump_path"]
        if path is None:
            return
        if len(self._tp_debug["lines"]) == 0:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._tp_debug["lines"]) + "\n")

    def _register_one_time_hook(self, module: nn.Module, hook_name: str):
        """注册 forward hook：只记录一次输出信息到 txt."""
        if module is None:
            self._append_dbg(f"[HOOK] {hook_name}: module=None")
            return

        def _hook(mod, inp, out):
            if hook_name in self._tp_debug["captured"]:
                return
            self._tp_debug["captured"].add(hook_name)

            self._append_dbg(f"\n===== FORWARD HOOK: {hook_name} =====")
            try:
                if isinstance(inp, (tuple, list)):
                    for i, t in enumerate(inp[:4]):
                        self._append_dbg(self._format_tensor_info(t, name=f"inp[{i}]"))
                else:
                    self._append_dbg(self._format_tensor_info(inp, name="inp"))
            except Exception as e:
                self._append_dbg(f"[hook input format error] {e}")

            try:
                if isinstance(out, (tuple, list)):
                    for i, t in enumerate(out[:4]):
                        self._append_dbg(self._format_tensor_info(t, name=f"out[{i}]"))
                else:
                    self._append_dbg(self._format_tensor_info(out, name="out"))
            except Exception as e:
                self._append_dbg(f"[hook output format error] {e}")

            self._maybe_write_debug_file()

        module.register_forward_hook(_hook, with_kwargs=False)

    # ---------------------------------------------------------------------
    # 1) Define tensor-parallel partition plan
    # ---------------------------------------------------------------------
    def configure_model(self):
        """Executed by ModelParallelStrategy after device mesh is ready."""
        tp_mesh = self.device_mesh["tensor_parallel"]

        # rank0 dump path
        if self._is_rank0():
            self._tp_debug["dump_path"] = os.path.abspath("./tp_debug_dump_rank0.txt")
            self._append_dbg("===== TP DEBUG DUMP (rank0) =====")
            self._append_dbg(f"world_size={self._world_size()}")
            self._append_dbg(f"dump_path={self._tp_debug['dump_path']}")
            self._append_dbg(f"torch_version={torch.__version__}")
            self._append_dbg("")

        # Top-level splits ---------------------------------------------------
        top_plan = {
            "text_embedding.0": ColwiseParallel(),
            "text_embedding.2": RowwiseParallel(),
            "time_projection.1": ColwiseParallel(output_layouts=Replicate()),
            "blocks.0": PrepareModuleInput(
                input_layouts=(Replicate(), None, None, None),
                desired_input_layouts=(Replicate(), None, None, None),
            ),
            "head": PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Replicate(), None),
                use_local_output=True,
            ),
        }
        self.pipe.dit = parallelize_module(self.pipe.dit, tp_mesh, top_plan)

        # Per-layer plan -----------------------------------------------------
        # 关键修改：
        # - 不要把 self/cross attn 的 qkv 强行转成 Shard(2)，保持 Shard(1)（sequence sharding）
        # - 对应地，o 不再用 RowwiseParallel(input_layouts=Shard(2))，改成 SequenceParallel()
        layer_plan = {
            # Self-Attention -------------------------------------------------
            "self_attn": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate()),
                desired_input_layouts=(Shard(1), Shard(0)),
            ),
            "self_attn.q": SequenceParallel(),
            "self_attn.k": SequenceParallel(),
            "self_attn.v": SequenceParallel(),
            "self_attn.norm_q": SequenceParallel(),
            "self_attn.norm_k": SequenceParallel(),
            # ✅ 改：保持 Shard(1)，避免 einops 在 DTensor 上 reshape 拆分 sharded hidden 维
            "self_attn.attn": PrepareModuleInput(
                input_layouts=(Shard(1), Shard(1), Shard(1)),
                desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
            ),
            # ✅ 改：o 用 SequenceParallel，匹配 Shard(1) 的输入/输出
            "self_attn.o": SequenceParallel(),

            # Cross-Attention ----------------------------------------------
            "cross_attn": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate()),
                desired_input_layouts=(Shard(1), Replicate()),
            ),
            "cross_attn.q": SequenceParallel(),
            "cross_attn.k": SequenceParallel(),
            "cross_attn.v": SequenceParallel(),
            "cross_attn.norm_q": SequenceParallel(),
            "cross_attn.norm_k": SequenceParallel(),
            # ✅ 改：同理，保持 Shard(1)
            "cross_attn.attn": PrepareModuleInput(
                input_layouts=(Shard(1), Shard(1), Shard(1)),
                desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
            ),
            # ✅ 改：o 用 SequenceParallel
            "cross_attn.o": SequenceParallel(),

            # Feed-Forward ---------------------------------------------------
            "ffn.0": ColwiseParallel(input_layouts=Shard(1)),
            "ffn.2": RowwiseParallel(output_layouts=Replicate()),

            # Norms & Gating -------------------------------------------------
            "norm1": SequenceParallel(use_local_output=True),
            "norm2": SequenceParallel(use_local_output=True),
            "norm3": SequenceParallel(use_local_output=True),
            "gate": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate(), Replicate()),
                desired_input_layouts=(Replicate(), Replicate(), Replicate()),
            ),
        }

        for blk in self.pipe.dit.blocks:
            parallelize_module(blk, tp_mesh, layer_plan)

        # ===== TP DEBUG DUMP: dump module structure + register hooks =====
        if self._is_rank0() and self._tp_debug["enabled"]:
            try:
                b0 = self.pipe.dit.blocks[0]
                self._append_dbg("===== MODULE REPR (blocks[0]) =====")
                self._append_dbg(f"blocks[0] type: {type(b0)}")
                self._append_dbg("\n--- blocks[0].self_attn repr ---")
                self._append_dbg(repr(getattr(b0, "self_attn", None)))
                self._append_dbg("\n--- blocks[0].cross_attn repr ---")
                self._append_dbg(repr(getattr(b0, "cross_attn", None)))

                def _list_children(mod, prefix):
                    if mod is None:
                        self._append_dbg(f"{prefix}: None")
                        return
                    self._append_dbg(f"\n--- {prefix} children ---")
                    for n, m in mod.named_children():
                        self._append_dbg(f"{prefix}.{n}: {type(m)}")

                _list_children(getattr(b0, "self_attn", None), "self_attn")
                _list_children(getattr(b0, "cross_attn", None), "cross_attn")

            except Exception as e:
                self._append_dbg(f"[DEBUG dump repr failed] {e}")

        # 在所有 rank 都注册 hook（但只 rank0 写文件）
        try:
            b0 = self.pipe.dit.blocks[0]
            sa = getattr(b0, "self_attn", None)
            ca = getattr(b0, "cross_attn", None)

            if sa is not None:
                self._register_one_time_hook(getattr(sa, "q", None), "blocks[0].self_attn.q")
                self._register_one_time_hook(getattr(sa, "k", None), "blocks[0].self_attn.k")
                self._register_one_time_hook(getattr(sa, "v", None), "blocks[0].self_attn.v")
                self._register_one_time_hook(getattr(sa, "attn", None), "blocks[0].self_attn.attn")
                self._register_one_time_hook(getattr(sa, "o", None), "blocks[0].self_attn.o")

            if ca is not None:
                self._register_one_time_hook(getattr(ca, "q", None), "blocks[0].cross_attn.q")
                self._register_one_time_hook(getattr(ca, "k", None), "blocks[0].cross_attn.k")
                self._register_one_time_hook(getattr(ca, "v", None), "blocks[0].cross_attn.v")
                self._register_one_time_hook(getattr(ca, "attn", None), "blocks[0].cross_attn.attn")
                self._register_one_time_hook(getattr(ca, "o", None), "blocks[0].cross_attn.o")
        except Exception as e:
            self._append_dbg(f"[DEBUG register hooks failed] {e}")

        # -------------------------------------------------------------
        # text encoder / vae / model_fn wrappers
        # -------------------------------------------------------------
        te = self.pipe.text_encoder
        orig_forward = te.forward

        def wrapped_forward(ids, mask=None, *args, **kwargs):
            dev = te.token_embedding.weight.device
            ids = ids.to(dev, non_blocking=True)
            if mask is not None:
                mask = mask.to(dev, non_blocking=True)
            return orig_forward(ids, mask, *args, **kwargs)

        te.forward = wrapped_forward

        vae = self.pipe.vae
        orig_encode = vae.encode

        def wrapped_encode(video, device=None, *args, **kwargs):
            dev = next(vae.parameters()).device
            if torch.is_tensor(video):
                video = video.to(dev, non_blocking=True)
            return orig_encode(video, device=dev, *args, **kwargs)

        vae.encode = wrapped_encode

        def _move_cpu_tensor_to(obj, dev):
            # 只把 CPU tensor 搬到 dev；cuda tensor/DTensor 尽量不动
            if torch.is_tensor(obj):
                if obj.device.type == "cpu":
                    return obj.to(dev, non_blocking=True)
                return obj
            if isinstance(obj, nn.Module):
                return obj
            if isinstance(obj, dict):
                return {k: _move_cpu_tensor_to(v, dev) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                out = [_move_cpu_tensor_to(v, dev) for v in obj]
                return type(obj)(out)
            return obj

        orig_model_fn = self.pipe.model_fn

        def wrapped_model_fn(*args, **kwargs):
            dev = next(self.pipe.dit.parameters()).device

            timestep = kwargs.get("timestep", None)
            if timestep is not None:
                if torch.is_tensor(timestep):
                    kwargs["timestep"] = timestep.to(dev, non_blocking=True)
                else:
                    kwargs["timestep"] = torch.tensor(timestep, device=dev)

            args = _move_cpu_tensor_to(args, dev)
            kwargs = _move_cpu_tensor_to(kwargs, dev)

            return orig_model_fn(*args, **kwargs)

        self.pipe.model_fn = wrapped_model_fn

        self._maybe_write_debug_file()

    # ---------------------------------------------------------------------
    # 2) Inference step
    # ---------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        task = batch[0].copy()
        task["progress_bar_cmd"] = tqdm if self.local_rank == 0 else (lambda x: x)
        out_path = task.pop("output_path")

        reference_image_path = task.pop("reference_image_path", None)
        control_video_path = task.pop("control_video_path", None)
        height = task.pop("height", 480)
        width = task.pop("width", 832)

        if reference_image_path is not None:
            task["vace_reference_image"] = Image.open(reference_image_path).resize(
                (width, height)
            )
        if control_video_path is not None:
            task["vace_video"] = VideoData(control_video_path, height=height, width=width)

        with torch.no_grad(), torch.inference_mode(False):
            video = self.pipe(**task)

        if self.local_rank == 0:
            save_video(video, out_path, fps=15, quality=5)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            dataset_snapshot_download(
                dataset_id="DiffSynth-Studio/examples_in_diffsynth",
                local_dir="./",
                allow_file_pattern=[
                    "data/examples/wan/depth_video.mp4",
                    "data/examples/wan/cat_fightning.jpg",
                ],
            )
        torch.distributed.barrier()
    else:
        dataset_snapshot_download(
            dataset_id="DiffSynth-Studio/examples_in_diffsynth",
            local_dir="./",
            allow_file_pattern=[
                "data/examples/wan/depth_video.mp4",
                "data/examples/wan/cat_fightning.jpg",
            ],
        )

    prompt = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    tasks = [
        dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=1,
            rand_device="cuda",
            tiled=True,
            reference_image_path="data/examples/wan/cat_fightning.jpg",
            height=480,
            width=832,
            output_path="video_2_Wan2.1-VACE-1.3B.mp4",
        ),
        dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=1,
            rand_device="cuda",
            tiled=True,
            control_video_path="data/examples/wan/depth_video.mp4",
            height=480,
            width=832,
            output_path="video_1_Wan2.1-VACE-1.3B.mp4",
        ),
        dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=1,
            rand_device="cuda",
            tiled=True,
            control_video_path="data/examples/wan/depth_video.mp4",
            reference_image_path="data/examples/wan/cat_fightning.jpg",
            height=480,
            width=832,
            output_path="video_3_Wan2.1-VACE-1.3B.mp4",
        ),
    ]

    dataloader = torch.utils.data.DataLoader(ToyDataset(tasks), collate_fn=lambda x: x)

    model = LitWanVACE13B()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=ModelParallelStrategy(),
    )
    trainer.test(model, dataloader)