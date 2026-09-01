"""Minimal PyTorch DiT with OmniVCus Lottery and Temporally Aligned Embeddings.

This educational example operates on already-patchified 1-D tokens.  It shows
the embedding and token-routing logic from the OmniVCus paper, rather than the
full pretrained Wan/OmniVCus architecture, VAE, or diffusion sampler.

Run its self-contained tests with:
    python examples/wanvideo/minimal_dit_le_tae.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def sinusoidal_embedding(values: Tensor, dim: int) -> Tensor:
    """Return sinusoidal embeddings with shape ``[*values.shape, dim]``."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half, 1)
    )
    phase = values.float().unsqueeze(-1) * frequencies
    embedding = torch.cat((phase.cos(), phase.sin()), dim=-1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


def make_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 4 * output_dim),
        nn.SiLU(),
        nn.Linear(4 * output_dim, output_dim),
    )


class LotteryEmbedding(nn.Module):
    """Lottery Embedding (LE) from Eq. (1) of the OmniVCus paper."""

    def __init__(self, dim: int, max_subjects: int):
        super().__init__()
        self.dim = dim
        self.max_subjects = max_subjects
        self.frame_mlp = make_mlp(dim, dim)

    def choose_positions(self, subject_count: int, training: bool,
                         device: torch.device) -> Tensor:
        if not 0 < subject_count <= self.max_subjects:
            raise ValueError(
                f"subject_count must be in [1, {self.max_subjects}]"
            )
        if training:
            # Uniform subset S of size K from [1, M], sorted to match IMG order.
            return (torch.randperm(self.max_subjects, device=device)[:subject_count]
                    .sort().values + 1)
        return torch.arange(1, subject_count + 1, device=device)

    def forward(self, tokens: Tensor, training: Optional[bool] = None
                ) -> Tuple[Tensor, Tensor]:
        """Embed subject tokens shaped ``[B, K, P, D]``."""
        if tokens.ndim != 4 or tokens.shape[-1] != self.dim:
            raise ValueError("subject tokens must have shape [B, K, P, D]")
        training = self.training if training is None else training
        positions = self.choose_positions(tokens.shape[1], training, tokens.device)
        frame = self.frame_mlp(sinusoidal_embedding(positions, self.dim))
        return tokens + frame[None, :, None, :].to(tokens.dtype), positions


class TemporallyAlignedEmbedding(nn.Module):
    """TAE for dense mask/depth tokens and pixel-aligned camera rays."""

    def __init__(self, input_dim: int, dim: int, max_subjects: int):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.max_subjects = max_subjects
        self.input_projection = nn.Linear(input_dim, dim)
        self.frame_mlp = make_mlp(dim, dim)
        self.timestep_mlp = make_mlp(dim, dim)
        self.camera_mlp = make_mlp(6, dim)

    @staticmethod
    def camera_to_plucker(camera_to_world: Tensor, intrinsics: Tensor,
                          grid_hw: Tuple[int, int]) -> Tensor:
        """Convert camera parameters to pixel rays ``(o x d, d)``.

        Args:
            camera_to_world: Camera extrinsics, ``[B, F, 4, 4]``.
            intrinsics: Camera intrinsics, ``[B, F, 3, 3]``.
            grid_hw: Spatial patch grid ``(H, W)``.
        Returns:
            Plucker coordinates shaped ``[B, F, H*W, 6]``.
        """
        height, width = grid_hw
        dtype, device = camera_to_world.dtype, camera_to_world.device
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype) + 0.5,
            torch.arange(width, device=device, dtype=dtype) + 0.5,
            indexing="ij",
        )
        pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1).reshape(-1, 3)
        directions_camera = torch.einsum(
            "bfij,pj->bfpi", torch.linalg.inv(intrinsics), pixels
        )
        directions = torch.einsum(
            "bfij,bfpj->bfpi", camera_to_world[..., :3, :3], directions_camera
        )
        directions = torch.nn.functional.normalize(directions, dim=-1)
        origins = camera_to_world[..., None, :3, 3].expand_as(directions)
        return torch.cat((torch.linalg.cross(origins, directions), directions), -1)

    def forward(
        self,
        noise_tokens: Tensor,
        timestep: Tensor,
        structure_tokens: Optional[Tensor] = None,
        camera_pose: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[Tensor], Tensor]:
        """Apply TAE to ``[B, F, P, C]`` patch tokens.

        ``structure_tokens`` represents either mask or depth. It receives the
        same frame embedding as noise but no diffusion-timestep embedding.
        Camera rays are projected and added to noise instead of concatenated.
        """
        if noise_tokens.ndim != 4 or noise_tokens.shape[-1] != self.input_dim:
            raise ValueError("noise_tokens must have shape [B, F, P, input_dim]")
        batch, frames, patches, _ = noise_tokens.shape
        if timestep.shape != (batch,):
            raise ValueError("timestep must have shape [B]")

        frame_positions = torch.arange(
            self.max_subjects + 1,
            self.max_subjects + frames + 1,
            device=noise_tokens.device,
        )
        frame_embedding = self.frame_mlp(
            sinusoidal_embedding(frame_positions, self.dim)
        )[None, :, None, :].to(noise_tokens.dtype)
        timestep_embedding = self.timestep_mlp(
            sinusoidal_embedding(timestep, self.dim)
        )[:, None, None, :].to(noise_tokens.dtype)
        noise = self.input_projection(noise_tokens) + frame_embedding + timestep_embedding

        structure = None
        if structure_tokens is not None:
            if structure_tokens.shape != noise_tokens.shape:
                raise ValueError("mask/depth tokens must have the same shape as noise")
            structure = self.input_projection(structure_tokens) + frame_embedding

        if camera_pose is not None:
            if intrinsics is None or grid_hw is None or math.prod(grid_hw) != patches:
                raise ValueError("camera TAE requires intrinsics and grid_hw matching P")
            if camera_pose.shape != (batch, frames, 4, 4):
                raise ValueError("camera_pose must have shape [B, F, 4, 4]")
            if intrinsics.shape != (batch, frames, 3, 3):
                raise ValueError("intrinsics must have shape [B, F, 3, 3]")
            rays = self.camera_to_plucker(camera_pose, intrinsics, grid_hw)
            noise = noise + self.camera_mlp(rays.to(noise_tokens.dtype))
        return structure, noise


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = make_mlp(dim, dim)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


@dataclass
class DiTOutput:
    prediction: Tensor
    sequence_length: int
    lottery_positions: Optional[Tensor]


class MinimalDiffusionTransformer(nn.Module):
    """Full-attention DiT over a composed 1-D token sequence."""

    def __init__(self, input_dim: int, dim: int = 64, depth: int = 2,
                 heads: int = 4, max_subjects: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.input_dim = input_dim
        self.dim = dim
        self.subject_projection = nn.Linear(input_dim, dim)
        self.le = LotteryEmbedding(dim, max_subjects)
        self.tae = TemporallyAlignedEmbedding(input_dim, dim, max_subjects)
        self.blocks = nn.ModuleList(DiTBlock(dim, heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, input_dim)

    def forward(
        self,
        noise_tokens: Tensor,
        timestep: Tensor,
        subject_tokens: Optional[Tensor] = None,
        structure_tokens: Optional[Tensor] = None,
        camera_pose: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> DiTOutput:
        if noise_tokens.ndim != 4 or noise_tokens.shape[-1] != self.input_dim:
            raise ValueError("noise_tokens must have shape [B, F, P, input_dim]")
        batch, frames, patches, channels = noise_tokens.shape
        chunks = []
        positions = None
        if subject_tokens is not None:
            if subject_tokens.ndim != 4 or subject_tokens.shape[0] != batch:
                raise ValueError("subject_tokens must have shape [B, K, P, input_dim]")
            subjects, positions = self.le(self.subject_projection(subject_tokens))
            chunks.append(subjects.flatten(1, 2))

        structure, noise = self.tae(
            noise_tokens, timestep, structure_tokens,
            camera_pose, intrinsics, grid_hw,
        )
        if structure is not None:
            chunks.append(structure.flatten(1, 2))
        noise_start = sum(chunk.shape[1] for chunk in chunks)
        chunks.append(noise.flatten(1, 2))

        tokens = torch.cat(chunks, dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        prediction = self.output_projection(
            self.final_norm(tokens[:, noise_start:])
        ).reshape(batch, frames, patches, channels)
        return DiTOutput(prediction, tokens.shape[1], positions)


def run_tests() -> None:
    torch.manual_seed(7)
    batch, frames, patches, channels = 2, 3, 4, 8
    noise = torch.randn(batch, frames, patches, channels)
    subjects = torch.randn(batch, 2, patches, channels)
    mask_or_depth = torch.randn_like(noise)
    timestep = torch.tensor([10.0, 500.0])
    model = MinimalDiffusionTransformer(
        channels, dim=16, depth=2, heads=4, max_subjects=5
    )

    # Mask/depth TAE: dense structure tokens are concatenated.
    model.train()
    dense = model(noise, timestep, subjects, structure_tokens=mask_or_depth)
    assert dense.prediction.shape == noise.shape
    assert dense.sequence_length == 2 * patches + 2 * frames * patches
    assert torch.all(dense.lottery_positions[1:] > dense.lottery_positions[:-1])

    # Camera TAE: Plucker rays are added to noise, so token length does not grow.
    poses = torch.eye(4).expand(batch, frames, 4, 4).clone()
    poses[:, :, 0, 3] = torch.linspace(0, 1, frames)
    intrinsics = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
    ).expand(batch, frames, 3, 3).clone()
    model.eval()
    with torch.no_grad():
        camera = model(
            noise, timestep, subjects, camera_pose=poses,
            intrinsics=intrinsics, grid_hw=(2, 2),
        )
    assert camera.prediction.shape == noise.shape
    assert camera.sequence_length == 2 * patches + frames * patches
    assert torch.equal(camera.lottery_positions, torch.tensor([1, 2]))
    assert torch.isfinite(camera.prediction).all()

    # Dense and camera controls can be composed.
    with torch.no_grad():
        both = model(
            noise, timestep, structure_tokens=mask_or_depth,
            camera_pose=poses, intrinsics=intrinsics, grid_hw=(2, 2),
        )
    assert both.sequence_length == 2 * frames * patches
    print("All tests passed: LE, mask/depth TAE, camera TAE, and DiT forward.")
    print("Output shape:", tuple(both.prediction.shape))


if __name__ == "__main__":
    run_tests()
