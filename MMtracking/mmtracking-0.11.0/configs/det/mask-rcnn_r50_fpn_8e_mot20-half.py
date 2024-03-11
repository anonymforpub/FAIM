USE_MMDET = True
_base_ = ['./mask-rcnn_r50_fpn_4e_mot17-half.py']
model = dict(
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=True)),
        # pretrained = 'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth',
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=True), num_classes=1))))
# data
data_root = 'data/MOT20/'
data = dict(
    train=dict(
        # ann_file=data_root + 'annotations/half-train_cocoformat.json',
        ann_file=data_root + 'annotations/new_trainhalf_segmentation_new.json',
        img_prefix=data_root + 'train'),
    val=dict(
        # ann_file=data_root + 'annotations/half-val_cocoformat.json',
        ann_file=data_root + 'annotations/new_valhalf_segmentation.json',
        img_prefix=data_root + 'train'),
    test=dict(
        # ann_file=data_root + 'annotations/half-val_cocoformat.json',
        ann_file=data_root + 'annotations/new_valhalf_segmentation.json',
        img_prefix=data_root + 'train'))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[6])
# runtime settings

# load_from = 'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
total_epochs = 12
