# Minimal STGCN config for API usage
# This mirrors the structure of the original my_stgcnpp.py but avoids absolute dataset paths
# and uses conservative dataloader settings appropriate for an API environment.

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')
    ),
    cls_head=dict(dropout=0.7, in_channels=256, num_classes=2, type='GCNHead')
)

# runtime / minimal test config
work_dir = '/tmp/mmaction_work_dir'
log_level = 'INFO'

test_cfg = dict(type='TestLoop')

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

test_evaluator = [dict(type='DumpResults', out_file_path=f'{work_dir}/test/results.pkl')]

default_scope = 'mmaction'
