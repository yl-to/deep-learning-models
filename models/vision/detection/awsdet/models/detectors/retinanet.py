# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class RetinaNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 norm_type='BN',
                 pretrained=None):
        super(RetinaNet, self).__init__(backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            norm_type=norm_type,
            pretrained=pretrained)
        self.mask = None
