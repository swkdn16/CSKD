_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py',
]
num_proposals = 100
num_stages = 6
model = dict(
    type='DRTR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    # neck=dict(
    #     type='ChannelMapper',
    #     in_channels=[512, 1024, 2048],
    #     kernel_size=1,
    #     out_channels=256,
    #     act_cfg=None,
    #     norm_cfg=dict(type='GN', num_groups=32),
    #     num_outs=4),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='DRTRRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=5, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[dict(
            type='DRTRHead',
            num_classes=80,
            in_channels=256,
            cls_fcs_channels=1024,
            num_query=num_proposals,
            num_cls_fcs=1,
            num_reg_fcs=3,
            roi_size=5,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                clip_border=True,
                target_means=[0., 0., 0., 0.],
                target_stds=[0.2, 0.2, 0.1, 0.1]),
            transformer=dict(
                type='RoiTransformer',
                encoder=[dict(
                    type='DetrTransformerEncoder',
                    num_layers=1,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1)
                        ],
                        embed_dim=256,
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('qkv_attn', 'norm', 'ffn', 'norm')))
                    for _ in range(2)],
                decoder_cls=dict(
                    type='DetrTransformerDecoder',
                    return_intermediate=False,
                    num_layers=1,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('qkv_attn', 'norm',
                                         'cross_attn', 'norm',
                                         'ffn', 'norm'))),
                decoder_bbox=dict(
                    type='DetrTransformerDecoder',
                    return_intermediate=False,
                    num_layers=1,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('qkv_attn', 'norm',
                                         'b_cross_attn', 'norm',
                                         'ffn', 'norm')))
            ),
            positional_encoding=[dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True),
                dict(
                type='SinePositionalEncoding1D',
                num_feats=256,
                normalize=True)],
            loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
            # loss_cls=dict(
            #     type='CrossEntropyLoss',
            #     bg_cls_weight=0.1,
            #     use_sigmoid=False,
            #     loss_weight=1.0,
            #     class_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)
        )for _ in range(num_stages)]
    ),
    # training and testing settings
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                # cls_cost=dict(type='ClassificationCost', weight=1.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                iou_cost=dict(type='IoUCost', iou_mode='giou',
                              weight=2.0))
        )),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))


# dataset settings
dataset_type = 'CocoDataset'
# data_root = '../data/coco/'
data_root = 'data/coco/'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.000025,
    weight_decay=0.0001)
    # paramwise_cfg=dict(
    #     custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[35])
runner = dict(type='EpochBasedRunner', max_epochs=50)
