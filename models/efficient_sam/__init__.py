# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from .build_efficient_sam import (build_efficient_sam_vits,
                                  build_efficient_sam_vitt)

sam_model_registry = {
    "default": build_efficient_sam_vits,
    "vit_s": build_efficient_sam_vits,
    "vit_t": build_efficient_sam_vitt,
}