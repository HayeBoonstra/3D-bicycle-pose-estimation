auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')
base_lr = 0.004
codec = dict(
    input_size=(
        192,
        256,
    ),
    normalize=False,
    sigma=(
        4.9,
        5.66,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=100,
        switch_pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(
                rotate_factor=30,
                scale_factor=[
                    0.8,
                    1.2,
                ],
                shift_factor=0.0,
                type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_mode = 'topdown'
data_root = '/home/hayepc/3D-bicycle-pose-estimation/data/bicycle_pose_dataset'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
keypoint_names = [
    'k_bottom_bracket',
    'k_seat_stay',
    'k_saddle',
    'k_upper_head_tube',
    'k_lower_head_tube',
    'k_handlebar_left',
    'k_handlebar_middle',
    'k_handlebar_right',
    'k_front_hub_left',
    'k_front_hub_right',
    'k_front_wheel_back',
    'k_front_wheel_front',
    'k_front_wheel_ground',
    'k_rear_hub_left',
    'k_rear_hub_right',
    'k_rear_wheel_ground',
    'k_left_pedal',
    'k_right_pedal',
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
max_epochs = 120
metainfo = dict(
    dataset_name='bicycle_pose',
    keypoint_info=dict({
        0: dict(id=0, name='k_bottom_bracket'),
        1: dict(id=1, name='k_seat_stay'),
        10: dict(id=10, name='k_front_wheel_back'),
        11: dict(id=11, name='k_front_wheel_front'),
        12: dict(id=12, name='k_front_wheel_ground'),
        13: dict(id=13, name='k_rear_hub_left'),
        14: dict(id=14, name='k_rear_hub_right'),
        15: dict(id=15, name='k_rear_wheel_ground'),
        16: dict(id=16, name='k_left_pedal'),
        17: dict(id=17, name='k_right_pedal'),
        2: dict(id=2, name='k_saddle'),
        3: dict(id=3, name='k_upper_head_tube'),
        4: dict(id=4, name='k_lower_head_tube'),
        5: dict(id=5, name='k_handlebar_left'),
        6: dict(id=6, name='k_handlebar_middle'),
        7: dict(id=7, name='k_handlebar_right'),
        8: dict(id=8, name='k_front_hub_left'),
        9: dict(id=9, name='k_front_hub_right')
    }),
    skeleton_info=dict())
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=1.0,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=1.0),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=1024,
        in_featuremap_size=(
            6,
            8,
        ),
        input_size=(
            192,
            256,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=18,
        simcc_split_ratio=2.0,
        type='RTMCCHead'),
    test_cfg=dict(flip_test=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=60,
        begin=60,
        by_epoch=True,
        convert_to_iter_based=True,
        end=120,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=21)
resume = False
stage2_num_epochs = 20
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/test.json',
        data_mode='topdown',
        data_prefix=dict(img='images/test/'),
        data_root=
        '/home/hayepc/3D-bicycle-pose-estimation/data/bicycle_pose_dataset',
        metainfo=dict(
            dataset_name='bicycle_pose',
            keypoint_info=dict({
                0: dict(id=0, name='k_bottom_bracket'),
                1: dict(id=1, name='k_seat_stay'),
                10: dict(id=10, name='k_front_wheel_back'),
                11: dict(id=11, name='k_front_wheel_front'),
                12: dict(id=12, name='k_front_wheel_ground'),
                13: dict(id=13, name='k_rear_hub_left'),
                14: dict(id=14, name='k_rear_hub_right'),
                15: dict(id=15, name='k_rear_wheel_ground'),
                16: dict(id=16, name='k_left_pedal'),
                17: dict(id=17, name='k_right_pedal'),
                2: dict(id=2, name='k_saddle'),
                3: dict(id=3, name='k_upper_head_tube'),
                4: dict(id=4, name='k_lower_head_tube'),
                5: dict(id=5, name='k_handlebar_left'),
                6: dict(id=6, name='k_handlebar_middle'),
                7: dict(id=7, name='k_handlebar_right'),
                8: dict(id=8, name='k_front_hub_left'),
                9: dict(id=9, name='k_front_hub_right')
            }),
            skeleton_info=dict()),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/home/hayepc/3D-bicycle-pose-estimation/data/bicycle_pose_dataset/annotations/test.json',
    type='CocoMetric')
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=5)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='annotations/train.json',
        data_mode='topdown',
        data_prefix=dict(img='images/train/'),
        data_root=
        '/home/hayepc/3D-bicycle-pose-estimation/data/bicycle_pose_dataset',
        metainfo=dict(
            dataset_name='bicycle_pose',
            keypoint_info=dict({
                0: dict(id=0, name='k_bottom_bracket'),
                1: dict(id=1, name='k_seat_stay'),
                10: dict(id=10, name='k_front_wheel_back'),
                11: dict(id=11, name='k_front_wheel_front'),
                12: dict(id=12, name='k_front_wheel_ground'),
                13: dict(id=13, name='k_rear_hub_left'),
                14: dict(id=14, name='k_rear_hub_right'),
                15: dict(id=15, name='k_rear_wheel_ground'),
                16: dict(id=16, name='k_left_pedal'),
                17: dict(id=17, name='k_right_pedal'),
                2: dict(id=2, name='k_saddle'),
                3: dict(id=3, name='k_upper_head_tube'),
                4: dict(id=4, name='k_lower_head_tube'),
                5: dict(id=5, name='k_handlebar_left'),
                6: dict(id=6, name='k_handlebar_middle'),
                7: dict(id=7, name='k_handlebar_right'),
                8: dict(id=8, name='k_front_hub_left'),
                9: dict(id=9, name='k_front_hub_right')
            }),
            skeleton_info=dict()),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(
                rotate_factor=45,
                scale_factor=[
                    0.7,
                    1.3,
                ],
                type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(
        rotate_factor=45,
        scale_factor=[
            0.7,
            1.3,
        ],
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(
        rotate_factor=30,
        scale_factor=[
            0.8,
            1.2,
        ],
        shift_factor=0.0,
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/val.json',
        data_mode='topdown',
        data_prefix=dict(img='images/val/'),
        data_root=
        '/home/hayepc/3D-bicycle-pose-estimation/data/bicycle_pose_dataset',
        metainfo=dict(
            dataset_name='bicycle_pose',
            keypoint_info=dict({
                0: dict(id=0, name='k_bottom_bracket'),
                1: dict(id=1, name='k_seat_stay'),
                10: dict(id=10, name='k_front_wheel_back'),
                11: dict(id=11, name='k_front_wheel_front'),
                12: dict(id=12, name='k_front_wheel_ground'),
                13: dict(id=13, name='k_rear_hub_left'),
                14: dict(id=14, name='k_rear_hub_right'),
                15: dict(id=15, name='k_rear_wheel_ground'),
                16: dict(id=16, name='k_left_pedal'),
                17: dict(id=17, name='k_right_pedal'),
                2: dict(id=2, name='k_saddle'),
                3: dict(id=3, name='k_upper_head_tube'),
                4: dict(id=4, name='k_lower_head_tube'),
                5: dict(id=5, name='k_handlebar_left'),
                6: dict(id=6, name='k_handlebar_middle'),
                7: dict(id=7, name='k_handlebar_right'),
                8: dict(id=8, name='k_front_hub_left'),
                9: dict(id=9, name='k_front_hub_right')
            }),
            skeleton_info=dict()),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/home/hayepc/3D-bicycle-pose-estimation/data/bicycle_pose_dataset/annotations/val.json',
    type='CocoMetric')
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/hayepc/3D-bicycle-pose-estimation/outputs/mmpose_bicycle_rtmpose_l'
