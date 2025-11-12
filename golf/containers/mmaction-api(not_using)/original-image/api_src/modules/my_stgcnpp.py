"""
Dummy ST-GCN config for API usage

This file contains placeholder values only. The API wrapper (`stgcn_tester.py`)
will overwrite the important fields at runtime (e.g. `cfg.load_from`,
`cfg.test_dataloader.dataset.ann_file`, and `cfg.test_evaluator`).

Do NOT treat these paths/values as real — they are intentionally dummy
placeholders so it's obvious they will be replaced by the API.
"""

# Placeholder checkpoint path (DUMMY) — overwritten by stgcn_tester
load_from = '/path/to/checkpoint.pth (DUMMY - overwritten at runtime)'

# Dataset placeholders (DUMMY) — stgcn_tester will create and set ann.pkl
dataset_type = 'PoseDataset'
ann_file = '/path/to/train_ann.pkl (DUMMY - overwritten at runtime)'
test_ann_file = '/path/to/test_ann.pkl (DUMMY - overwritten at runtime)'

# Runtime settings (safe defaults for API/container)
EPOCH = 50
clip_len = 50
fp16 = None
auto_scale_lr = dict(enable=False, base_batch_size=128)

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(type='UniformSampleFrames', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(
        type='UniformSampleFrames', clip_len=clip_len, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(
        type='UniformSampleFrames', clip_len=clip_len, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='xsub_train')))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='xsub_val',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=test_ann_file,
        pipeline=test_pipeline,
        split='xsub_val',
        test_mode=True))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=EPOCH, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=EPOCH,
        by_epoch=True,
        milestones=[int(EPOCH*0.3), int(EPOCH*0.6)],
        gamma=0.1
    )
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True),
    clip_grad=dict(max_norm=5, norm_type=2))

auto_scale_lr = dict(enable=False, base_batch_size=128)

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(
        type='GCNHead',
        num_classes=2,
        in_channels=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            class_weight=[2.0, 1.0],  # ← 리스트 리터럴로!
            loss_weight=1.0
        )
    )
)