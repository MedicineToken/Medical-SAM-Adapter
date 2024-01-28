# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import LayerNorm2d
from ..ImageEncoder import AdaloraBlock, AdapterBlock, Block, LoraBlock


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x

@torch.jit.export
def get_abs_pos(
    abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]
) -> torch.Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h = hw[0]
    w = hw[1]
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


# Image encoder for efficient SAM.
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        args,
        img_size: int,
        patch_size: int,
        in_chans: int,
        patch_embed_dim: int,
        normalization_type: str,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        neck_dims: List[int],
        act_layer: Type[nn.Module],
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()

        self.args = args
        self.img_size = img_size
        self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (pretrain_img_size // patch_size) * (
            pretrain_img_size // patch_size
        )
        num_positions = num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))

        self.blocks = nn.ModuleList()
        if args.mod == 'sam_adpt':
            block_class = AdapterBlock 
        elif args.mod == 'sam_lora':
            block_class = LoraBlock 
        elif args.mod == 'sam_adalora':
            block_class = AdaloraBlock
        else:
            block_class = Block 

        for i in range(depth):
            vit_block = block_class(
                args = self.args,
                dim=patch_embed_dim,
                num_heads=num_heads,
                use_rel_pos=True,
                mlp_ratio=mlp_ratio,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(vit_block)
        
        self.neck = nn.Sequential(
            nn.Conv2d(
                patch_embed_dim,
                neck_dims[0],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(
                neck_dims[0],
                neck_dims[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"
        x = self.patch_embed(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
        )
        num_patches = x.shape[1]
        assert x.shape[2] == num_patches
        # x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
        for blk in self.blocks:
            x = blk(x)
        # x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
        x = self.neck(x.permute(0, 3, 1, 2))
        return x
