# run_vace_usp.py
import os
import torch
from PIL import Image

from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


def rank():
    return int(os.environ.get("RANK", "0"))


def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank0():
    return rank() == 0


def dist_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def main():
    # ---------- distributed / device ----------
    lr = local_rank()
    torch.cuda.set_device(lr)

    # IMPORTANT:
    # - USP init expects device TYPE like "cuda", not "cuda:1"
    # - per-rank GPU is controlled by torch.cuda.set_device(lr)
    device_type = "cuda"

    # ---------- (rank0 only) download example assets ----------
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

    # ---------- build pipeline with built-in USP ----------
    # Optional: rank0 先构建一次 + barrier，可减少并发下载/读写缓存导致的超时/锁问题
    # 这里做成所有 rank 都构建，但你如果还遇到 modelscope 超时，建议把模型权重提前离线好
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device_type,  # ✅ must be "cuda" (type), not "cuda:local_rank"
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
        use_usp=True,   # ✅ use pipeline’s built-in unified sequence parallel
        # vram_limit=20.0,
    )

    # （可选）减少多 rank 的无用输出
    if not is_rank0():
        # 有些 pipeline 里会 print，你想彻底静音可以在这里重定向 stdout/stderr
        pass

    prompt = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    # Prepare inputs once (each rank reads same local files)
    ref_img = Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480))
    control_video = VideoData("data/examples/wan/depth_video.mp4", height=480, width=832)

    torch.set_grad_enabled(False)

    # ---------- Case 1: Reference image -> Video ----------
    with torch.inference_mode():
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            vace_reference_image=ref_img,
            seed=1,
            tiled=True,
        )
    if is_rank0():
        save_video(video, "video_2_Wan2.1-VACE-1.3B.mp4", fps=15, quality=5)
    dist_barrier()

    # ---------- Case 2: Depth video -> Video ----------
    with torch.inference_mode():
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            vace_video=control_video,
            seed=1,
            tiled=True,
        )
    if is_rank0():
        save_video(video, "video_1_Wan2.1-VACE-1.3B.mp4", fps=15, quality=5)
    dist_barrier()

    # ---------- Case 3: Depth video + Reference image -> Video ----------
    with torch.inference_mode():
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            vace_video=control_video,
            vace_reference_image=ref_img,
            seed=1,
            tiled=True,
        )
    if is_rank0():
        save_video(video, "video_3_Wan2.1-VACE-1.3B.mp4", fps=15, quality=5)

    dist_barrier()


if __name__ == "__main__":
    main()
