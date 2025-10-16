auto_scale_lr = dict(base_batch_size=128, enable=False)
custom_hooks = [
    dict(
        min_delta=0.001,
        model = dict(
            backbone=dict(
                gcn_adaptive='init',
                gcn_with_res=True,
                graph_cfg=dict(layout='coco', mode='spatial'),
                tcn_type='mstcn',
                type='STGCN'),
            cls_head=dict(dropout=0.7, in_channels=256, num_classes=2, type='GCNHead'),
            type='RecognizerGCN')

        # Minimal safe API config
        work_dir = '/tmp/mmaction_work_dir'
        test_cfg = dict(type='TestLoop')
        test_evaluator = [dict(type='DumpResults', out_file_path=f'{work_dir}/test/results.pkl')]

        # test_dataloader: ann_file will be overridden by the API at runtime
        test_dataloader = dict(
            batch_size=1,
            num_workers=0,
            persistent_workers=False,
            sampler=dict(shuffle=False, type='DefaultSampler'),
            dataset=dict(
                type='PoseDataset',
                ann_file='DUMMY_ANN_FILE.pkl',
                pipeline=[
                    dict(type='PreNormalize2D'),
                    dict(dataset='coco', feats=['j'], type='GenSkeFeat'),
                    dict(clip_len=100, num_clips=10, test_mode=True, type='UniformSampleFrames'),
                    dict(type='PoseDecode'),
                    dict(num_person=2, type='FormatGCNInput'),
                    dict(type='PackActionInputs'),
                    dict(type='AddFrameDirToMeta'),
                ],
                split='xsub_val',
                test_mode=True
            )
        )

        default_scope = 'mmaction'
        log_level = 'INFO'
                'j',
            ], type='GenSkeFeat'),
            dict(
                clip_len=100,
                num_clips=10,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=2, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='xsub_val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AccMetric'),
    dict(out_file_path='result/result.pkl', type='DumpResults'),
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(
        clip_len=100, num_clips=10, test_mode=True,
        type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=2, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
    dict(type='AddFrameDirToMeta')

]
train_cfg = dict(
    max_epochs=8, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        dataset=dict(
            ann_file= ann_file,
            pipeline=[
                dict(type='PreNormalize2D'),
                dict(dataset='coco', feats=[
                    'j',
                ], type='GenSkeFeat'),
                dict(clip_len=100, type='UniformSampleFrames'),
                dict(type='PoseDecode'),
                dict(num_person=2, type='FormatGCNInput'),
                dict(type='PackActionInputs'),
            ],
            split='xsub_train',
            type='PoseDataset'),
        times=1,
        type='RepeatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(clip_len=100, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=2, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
    dict(type='AddFrameDirToMeta')
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file= ann_file,
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(
                clip_len=100,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=2, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='xsub_val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(
        clip_len=100, num_clips=1, test_mode=True, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=2, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
    dict(type='AddFrameDirToMeta')
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/tmp/mmaction_work_dir'