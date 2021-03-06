# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='KerasBackbone',
        model_name='HRNetV2p',
        weights_path='weights/hrnet_w32c',
        weight_decay=1e-4
    ),
    neck=dict(
        type='HRFPN',
        in_channels=[('C2', 32), ('C3', 64), ('C4', 128), ('C5', 256)],
        out_channels=256,
        num_outs=5,
        weight_decay=1e-4,
    ),
    rpn_head=dict(
        type='RPNHead',
        anchor_scales=[8.],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds= [1.0, 1.0, 1.0, 1.0],
        feat_channels=512,
        num_samples=256,
        positive_fraction=0.5,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        num_pre_nms_train=6000,
        num_post_nms_train=2000,
        num_pre_nms_test=2000,
        num_post_nms_test=1000,
        weight_decay=1e-4,
        use_smooth_l1=False
    ),
    bbox_roi_extractor=dict(
        type='PyramidROIAlign',
        pool_shape=[7, 7],
        pool_type='avg',
    ),
    bbox_head=dict(
    type='BBoxHead',
    num_classes=81,
    pool_size=[7, 7],
    target_means=[0., 0., 0., 0.],
    target_stds=[0.1, 0.1, 0.2, 0.2],
    min_confidence=0.005,
    nms_threshold=0.75,
    max_instances=100,
    weight_decay=1e-4,
    use_conv=False,
    use_bn=False,
    use_smooth_l1=False,
    soft_nms_sigma=0.5)
)

# model training and testing settings
train_cfg = dict(
    freeze_patterns=['_bn$'],
    weight_decay=1e-4,
)

test_cfg = dict(
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/COCO/'
data = dict(
    imgs_per_gpu=4,
    train=dict(
        type=dataset_type,
        train=True,
        dataset_dir=data_root,
        subset='train',
        flip_ratio=0.5,
        pad_mode='fixed',
        preproc_mode='rgb',
        mean=(123.68, 116.78, 103.94),
        std=(1.0, 1.0, 1.0),
        scale=(800, 1344)),
    val=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode='rgb',
        mean=(123.68, 116.78, 103.94),
        std=(1.0, 1.0, 1.0),
        scale=(800, 1344)),
    test=dict(
        type=dataset_type,
        train=False,
        dataset_dir=data_root,
        subset='val',
        flip_ratio=0,
        pad_mode='fixed',
        preproc_mode='rgb',
        mean=(123.68, 116.78, 103.94),
        std=(1.0, 1.0, 1.0),
        scale=(800, 1344)),
)
# yapf: enable
evaluation = dict(interval=1)
# optimizer
optimizer = dict(
    type='MomentumOptimizer',
    learning_rate=1e-2,
    momentum=0.9,
    nesterov=False,
)
# extra options related to optimizers
optimizer_config = dict(
    amp_enabled=True,
)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
checkpoint_config = dict(interval=1, outdir='checkpoints')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir='/tmp/tensorboard')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_hrnetv2p_w32_1x_coco'
load_from = None
resume_from = None
workflow = [('train', 1)]
