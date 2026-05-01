_base_ = [
    "/home/hayepc/3D-bicycle-pose-estimation/.venv-mmpose311/lib/python3.11/site-packages/"
    "mmpose/.mim/configs/_base_/default_runtime.py"
]

# Runtime
max_epochs = 120
stage2_num_epochs = 20
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=5)
randomness = dict(seed=21)

# Optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# Learning rate schedule
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

auto_scale_lr = dict(base_batch_size=512)

# Keypoint codec
codec = dict(
    type="SimCCLabel",
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

# Model
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _scope_="mmdet",
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU"),
        init_cfg=dict(
            type="Pretrained",
            prefix="backbone.",
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/"
            "rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth",
        ),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=1024,
        out_channels=18,
        input_size=codec["input_size"],
        in_featuremap_size=tuple([s // 32 for s in codec["input_size"]]),
        simcc_split_ratio=codec["simcc_split_ratio"],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(
            type="KLDiscretLoss",
            use_target_weight=True,
            beta=10.0,
            label_softmax=True,
        ),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True),
)

# Dataset
dataset_type = "CocoDataset"
data_mode = "topdown"
data_root = "/home/hayepc/3D-bicycle-pose-estimation/data/bicycle_pose_dataset"
backend_args = dict(backend="local")

keypoint_names = [
    "k_bottom_bracket",
    "k_seat_stay",
    "k_saddle",
    "k_upper_head_tube",
    "k_lower_head_tube",
    "k_handlebar_left",
    "k_handlebar_middle",
    "k_handlebar_right",
    "k_front_hub_left",
    "k_front_hub_right",
    "k_front_wheel_back",
    "k_front_wheel_front",
    "k_front_wheel_ground",
    "k_rear_hub_left",
    "k_rear_hub_right",
    "k_rear_wheel_ground",
    "k_left_pedal",
    "k_right_pedal",
]
metainfo = dict(
    dataset_name="bicycle_pose",
    keypoint_info={idx: dict(name=name, id=idx) for idx, name in enumerate(keypoint_names)},
    skeleton_info={},
    joint_weights=[1.0] * len(keypoint_names),
    sigmas=[0.0] * len(keypoint_names),
)

# Pipelines
train_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", scale_factor=[0.7, 1.3], rotate_factor=45),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_factor=0.0, scale_factor=[0.8, 1.2], rotate_factor=30),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

# Data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/train.json",
        data_prefix=dict(img=""),
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/val.json",
        data_prefix=dict(img=""),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/test.json",
        data_prefix=dict(img=""),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

# Hooks and evaluators
default_hooks = dict(
    checkpoint=dict(save_best="coco/AP", rule="greater", max_keep_ckpts=3),
)

custom_hooks = [
    dict(type="EMAHook", ema_type="ExpMomentumEMA", momentum=0.0002, update_buffers=True, priority=49),
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
    ),
]

val_evaluator = dict(type="CocoMetric", ann_file=f"{data_root}/annotations/val.json")
test_evaluator = dict(type="CocoMetric", ann_file=f"{data_root}/annotations/test.json")

