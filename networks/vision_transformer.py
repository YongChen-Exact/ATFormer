# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch.nn as nn
from .ATFormer import ATFormer

logger = logging.getLogger(__name__)


class VisionTransformer(nn.Module):
    def __init__(self, config, num_classes=4):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.atformer = ATFormer(img_size=config.DATA.IMG_SIZE,
                                 patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                 in_chans=config.MODEL.SWIN.IN_CHANS,
                                 num_classes=self.num_classes,
                                 embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                 depths=config.MODEL.SWIN.DEPTHS,
                                 num_heads=config.MODEL.SWIN.NUM_HEADS,
                                 window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                 mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                 qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                 drop_rate=config.MODEL.DROP_RATE,
                                 drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                 ape=config.MODEL.SWIN.APE,
                                 patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                 use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.atformer(x)
        return logits
