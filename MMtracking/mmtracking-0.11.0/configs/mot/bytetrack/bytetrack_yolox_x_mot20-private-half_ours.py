_base_ = ['./bytetrack_yolox_x_crowdhuman_mot17-private-half.py']

img_scale = (896, 1600)

model = dict(
    detector=dict(input_size=img_scale, random_size_range=(20, 36)),
    tracker=dict(
        weight_iou_with_det_scores=False,
        match_iou_thrs=dict(high=0.3),
    ))

train_pipeline = [
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    train=dict(
        dataset=dict(
            ann_file=[
                # 'data/MOT20/annotations/new_trainhalf_segmentation_reduced.json',
                'data/MOT20/annotations/new_trainhalf_segmentation_new.json',
                # 'data/crowdhuman/annotations/crowdhuman_train.json',
                # 'data/crowdhuman/annotations/crowdhuman_val.json'
            ],
            img_prefix=[
                # 'data/MOT20/train',
                'data/MOT20/train_tmp/train', 
                # 'data/crowdhuman/train',
                # 'data/crowdhuman/val'
            ]),
        pipeline=train_pipeline),
    val=dict(
        ann_file='data/MOT20/annotations/half-val_cocoformat.json',
        # ann_file='data/MOT20/annotations/new_trainhalf_segmentation_reduced.json',
        img_prefix= 'data/MOT20/train_tmp/train', 
        # img_prefix='data/MOT20/train',
        pipeline=test_pipeline),
    test=dict(
        ann_file='data/MOT20/annotations/half-val_cocoformat.json',
        # ann_file='data/MOT20/annotations/new_trainhalf_segmentation_reduced.json',
        img_prefix= 'data/MOT20/train_tmp/train', 
        # img_prefix='data/MOT20/train',
        pipeline=test_pipeline),
    samples_per_gpu = 4
    )

checkpoint_config = dict(interval=1)
# optimizer_config = dict(grad_clip=None,find_unused_parameters: True)
find_unused_parameters = True
# evaluation = dict(metric=['bbox', 'track'], interval=80)
evaluation = dict(metric=['bbox', 'track'], interval=1)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
