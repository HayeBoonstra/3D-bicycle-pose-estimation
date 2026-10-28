dataset_type = 'CocoDataset'
data_root = '/home/hayepc/3D-bicycle-pose-estimation/data/bicycle_pose_dataset'

metainfo = dict(
    dataset_name='bicycle_pose',
    keypoint_info={0: dict(name="k_bottom_bracket"), 1: dict(name="k_seat_stay"), 2: dict(name="k_saddle"), 3: dict(name="k_upper_head_tube"), 4: dict(name="k_lower_head_tube"), 5: dict(name="k_handlebar_left"), 6: dict(name="k_handlebar_middle"), 7: dict(name="k_handlebar_right"), 8: dict(name="k_front_hub_left"), 9: dict(name="k_front_hub_right"), 10: dict(name="k_front_wheel_back"), 11: dict(name="k_front_wheel_front"), 12: dict(name="k_front_wheel_ground"), 13: dict(name="k_rear_hub_left"), 14: dict(name="k_rear_hub_right"), 15: dict(name="k_rear_wheel_ground"), 16: dict(name="k_left_pedal"), 17: dict(name="k_right_pedal")},
    skeleton_info={},
)

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=metainfo,
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=metainfo,
    )
)
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test/'),
        metainfo=metainfo,
    )
)
custom_keypoint_names = ["k_bottom_bracket", "k_seat_stay", "k_saddle", "k_upper_head_tube", "k_lower_head_tube", "k_handlebar_left", "k_handlebar_middle", "k_handlebar_right", "k_front_hub_left", "k_front_hub_right", "k_front_wheel_back", "k_front_wheel_front", "k_front_wheel_ground", "k_rear_hub_left", "k_rear_hub_right", "k_rear_wheel_ground", "k_left_pedal", "k_right_pedal"]
