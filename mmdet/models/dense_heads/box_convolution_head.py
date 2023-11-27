import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

from mmcv.cnn import ConvModule, Linear
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner import force_fp32
from mmcv.runner.base_module import ModuleList
from mmdet.core import build_bbox_coder, multi_apply, reduce_mean, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl

from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from .deformable_range_conv import DeformRangeConv2d

INF = 1e8


@HEADS.register_module()
class BoxConvHead(BaseDenseHead, BBoxTestMixin):
    def __init__(self,
                 num_classes,
                 feat_channels,
                 kernel_size=3,
                 stacked_convs=4,
                 cascade_stages=6,
                 strides=[8, 16, 32, 64, 128],
                 conv_cfg=None,
                 conv_bias='auto',
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_iou=dict(
                     type='GIoULoss',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='L1Loss',
                     loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 bbox_coder_delta=dict(type='DeltaXYWHBBoxCoder',
                                       clip_border=False,
                                       target_means=[0., 0., 0., 0.],
                                       target_stds=[0.5, 0.5, 1., 1.]),
                 bbox_coder_point=dict(type='DistancePointBBoxCoder'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal',
                               layer='Conv2d',
                               std=0.01,
                               override=dict(
                                   type='Normal',
                                   name='pred_cls',
                                   std=0.01,
                                   bias_prob=0.01))):
        super(BoxConvHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.kernel_size = kernel_size
        self.stacked_convs = stacked_convs
        self.cascade_stages = cascade_stages
        self.strides = strides
        self.conv_cfg = conv_cfg
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.norm_cfg = norm_cfg
        self.regress_ranges = regress_ranges

        self.stage_sp_attn = ModuleList()
        self.pred_box_region = ModuleList()
        self.pred_cls = ModuleList()
        self.pred_centerness = ModuleList()

        self.loss_cls = build_loss(loss_cls)
        self.loss_iou = build_loss(loss_iou)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.bbox_coder_delta = build_bbox_coder(bbox_coder_delta)
        self.bbox_coder_point = build_bbox_coder(bbox_coder_point)

        self.prior_generator = MlvlPointGenerator([strides[2]])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(2)
        self.head_convs = nn.ModuleList()
        pad_size = self.kernel_size // 2
        for i in range(self.stacked_convs):
            self.head_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=pad_size,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        self.fc_ch = 1024
        self.fc_ch_attn = self.reg_ffn = FFN(
            self.feat_channels*2,
            self.fc_ch,
            2,
            dropout=0.0,
            add_residual=False)
        self.zero_padding = nn.ZeroPad2d(self.kernel_size // 2)
        self.roi_extract = RoIAlign(
            output_size=3,
            spatial_scale=1,
            sampling_ratio=2,
            aligned=True)

        for stage in range(self.cascade_stages):
            self.stage_sp_attn.append(nn.Conv2d(
                2,
                1,
                self.kernel_size))
            self.pred_box_region.append(nn.Conv2d(
                self.feat_channels,
                4,
                self.kernel_size))
            self.pred_cls.append(nn.Conv2d(
                self.feat_channels,
                self.cls_out_channels,
                self.kernel_size))
            self.pred_centerness.append(nn.Conv2d(
                self.feat_channels,
                1,
                self.kernel_size))

    def stack_convs_single(self, feat):
        for h_conv in self.head_convs:
            feat = h_conv(feat)
        return feat

    def channel_attention(self, feat):
        channel_attn_mean = F.avg_pool2d(feat, feat.size()[2:])
        channel_attn_mean = F.adaptive_avg_pool2d(channel_attn_mean, (1, 1))
        channel_attn_mean = torch.mean(channel_attn_mean.view(channel_attn_mean.size(0),
                                                              channel_attn_mean.size(1), -1), dim=2)

        channel_attn_max = F.max_pool2d(feat, feat.size()[2:])
        channel_attn_max = F.adaptive_max_pool2d(channel_attn_max, (1, 1))
        channel_attn_max = torch.max(channel_attn_max.view(channel_attn_max.size(0),
                                                           channel_attn_max.size(1), -1), dim=2)[0]

        channel_attn = torch.cat([channel_attn_mean, channel_attn_max], 1)
        channel_attn = self.fc_ch_attn(channel_attn).reshape(-1, 2, self.feat_channels).sum(1).softmax(-1)
        channel_attn = channel_attn.unsqueeze(-1).unsqueeze(-1)
        return feat * channel_attn

    def spatial_attention(self, stage, feat):
        spatial_attn_mean = feat.mean(1)
        spatial_attn_max = feat.max(1)[0]

        spatial_attn = torch.stack([spatial_attn_mean, spatial_attn_max], 1)
        spatial_attn = self.stage_sp_attn[stage](spatial_attn).softmax()

        return feat * channel_attn

    def _get_rois(self, bbox_region_flatten, feat_shape, dtype, device):
        b, _, h, w = feat_shape

        batch_indx = torch.cat([torch.full((1, h*w, 1), i, dtype=dtype, device=device)
                                for i in range(b)])
        range_rois = torch.cat([batch_indx, bbox_region_flatten], dim=-1).view(-1, 5)
        return range_rois

    def bbox_conv(self, stage, feats, stage_pred_bbox=None):
        device = feats.device
        dtype = feats.dtype

        mllv_cls_score_flatten = []
        mllv_pred_bbox_region = []
        mllv_centerness_score_flatten = []

        batch, c, h, w = feats.shape

        feat_padding = self.zero_padding(feats)
        factor = feats.new_tensor([w, h, w, h]).unsqueeze(0)

        p0_y, p0_x = torch.meshgrid(torch.arange(1, h + 1, device=device),
                                    torch.arange(1, w + 1, device=device))

        feat_points = torch.stack([p0_x.reshape(-1), p0_y.reshape(-1)]).permute(1, 0)

        if stage == 0:
            pred_bbox_region = self.pred_box_region[stage](feat_padding).clamp(min=0)
            pred_bbox_region = pred_bbox_region.permute(0, 2, 3, 1)
            pred_bbox_region_flatten = pred_bbox_region.view(batch, -1, 4)
            # l,t,r,b -> x1,y1,x2,y2
            pred_bbox_region_flatten = torch.stack([self.bbox_coder_point.decode(feat_points, pred_bbox_region_flatten[b])
                                                    for b in range(batch)])
        else:
            pred_bbox_region = self.pred_box_region[stage](feat_padding)
            pred_bbox_region = pred_bbox_region.permute(0, 2, 3, 1)
            _pred_bbox_region_flatten = pred_bbox_region.view(batch, -1, 4)
            stage_pred_bbox = stage_pred_bbox.view(batch, -1, 4)

            factor = factor.repeat(stage_pred_bbox.size(1), 1)
            pred_bbox_region_flatten = []
            for b in range(batch):
                # l,t,r,b -> x1,y1,x2,y2
                stage_pred_bbox_xyxy_b = self.bbox_coder_point.decode(feat_points, stage_pred_bbox[b])
                # x1,y1,x2,y2 -> normalized x1,y1,x2,y2
                pred_bbox_region_flatten_xyxy_b = self.bbox_coder_delta.decode(stage_pred_bbox_xyxy_b, _pred_bbox_region_flatten[b])
                pred_bbox_region_flatten.append(pred_bbox_region_flatten_xyxy_b)
            pred_bbox_region_flatten = torch.stack(pred_bbox_region_flatten)



        bbox_num = pred_bbox_region_flatten.size(1)

        rois_coord_in = self._get_rois(pred_bbox_region_flatten,
                                       feats.shape,
                                       dtype, device)
        roi_feats_flatten = self.roi_extract(feat_padding, rois_coord_in)

        roi_feats_flatten = self.channel_attention(roi_feats_flatten)

        # x1,y1,x2,y2 - >l,t,r,b
        pred_bbox_region_flatten = torch.stack([self.bbox_coder_point.encode(feat_points, pred_bbox_region_flatten[b])
                                                for b in range(batch)])
        cls_score_flatten = self.pred_cls[stage](roi_feats_flatten).view(batch, -1, self.cls_out_channels)
        centerness_score_flatten = self.pred_centerness[stage](roi_feats_flatten).view(batch, -1)

        return cls_score_flatten, pred_bbox_region_flatten.view(batch, h, w, 4), centerness_score_flatten

    def forward(self, feats):
        feat_2 = feats[2]
        feats_01 = torch.stack([feats[1], self.maxpool2d(feats[0])]).mean(0)

        feat = torch.stack([feat_2, self.maxpool2d(feats_01)]).mean(0)

        for h_conv in self.head_convs:
            feat = h_conv(feat)

        # feats_list = []
        # for feat in feats:
        #     for h_conv in self.head_convs:
        #         feat = h_conv(feat)
        #     feats_list.append(feat)
        # feats_hconvs = multi_apply(self.stack_convs_single, feats)
        # feats_list = []
        # for i in range(len(feats)):
        #     feats_list.append(
        #         torch.stack([feat_h[i] for feat_h in feats_hconvs]))

        all_cls_score_flatten = []
        all_pred_bbox_region = []
        all_centerness_score_flatten = []
        stage_pred_bbox = None
        for stage in range(self.cascade_stages):
            cls_score_flatten,\
            pred_bbox_region,\
            centerness_score_flatten = self.bbox_conv(stage,
                                                      feat,
                                                      stage_pred_bbox)
            stage_pred_bbox = pred_bbox_region.detach().clone()

            all_cls_score_flatten.append(cls_score_flatten)
            all_pred_bbox_region.append(pred_bbox_region * self.strides[2])
            all_centerness_score_flatten.append(centerness_score_flatten)

        return all_cls_score_flatten, all_pred_bbox_region, all_centerness_score_flatten

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds', 'all_centernesses_preds'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             all_centernesses_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None
             ):
        # only use last stage prediction
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        centernesses_preds = all_centernesses_preds[-1]

        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(centernesses_preds)

        device = cls_scores[0].device
        dtype = cls_scores[0].dtype

        featmap_sizes = [bbox_preds.size()[1:-1]]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds.dtype,
            device=bbox_preds.device)

        labels, bbox_targets = self.get_targets(
            all_level_points,
            img_metas,
            gt_bboxes,
            gt_labels)

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        flatten_bbox_preds = bbox_preds.reshape(-1, 4)
        flatten_centerness = centernesses_preds.reshape(-1)

        num_imgs = cls_scores.size(0)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)

        # img_batch_h, img_batch_w = img_metas[0]['batch_input_shape']
        # for lv_idx, points in enumerate(range(len(all_level_points))):
        #     all_level_points[lv_idx][:, 0] /= img_batch_w
        #     all_level_points[lv_idx][:, 1] /= img_batch_h
        #     all_level_points[lv_idx][:, 0] *= featmap_sizes[lv_idx][1]
        #     all_level_points[lv_idx][:, 1] *= featmap_sizes[lv_idx][0]
        #     all_level_points[lv_idx] = all_level_points[lv_idx].repeat(num_imgs, 1).long()
        # flatten_points = torch.cat(all_level_points)
        flatten_points = torch.cat([points.repeat(num_imgs, 1)
                                    for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)

        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        batch_h, batch_w = img_metas[0]['batch_input_shape']
        factor = bbox_preds.new_tensor([batch_w, batch_h, batch_w, batch_h]).unsqueeze(0)
        factor = factor.repeat(len(pos_inds), 1)
        # for img_meta, bbox_pred in zip(img_metas, bbox_preds):
        #     img_h, img_w, _ = img_meta['img_shape']
        #     factor = bbox_pred.new_tensor([img_w, img_h, img_w,
        #                                    img_h]).unsqueeze(0).repeat(
        #         bbox_pred.size(0), 1)
        #     factors.append(factor)
        # factors = torch.cat(factors, 0)
        #
        # factors_pos = factors[pos_inds]
        pos_points = flatten_points[pos_inds]

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)

        # centerness weighted iou loss
        centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            # construct factors used for rescale bboxes
            pos_decoded_bbox_preds = self.bbox_coder_point.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_bbox_targets = self.bbox_coder_point.decode(
                pos_points, pos_bbox_targets)

            loss_iou = self.loss_iou(
                pos_decoded_bbox_preds,
                pos_decoded_bbox_targets,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)

            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds / factor,
                pos_decoded_bbox_targets / factor,
                weight=pos_centerness_targets.unsqueeze(1).repeat(1, 4),
                avg_factor=centerness_denorm)

            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_iou = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_iou=loss_iou,
            loss_cnt=loss_centerness)

    # @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds', 'all_centernesses_preds'))
    # def loss(self,
    #          all_cls_scores,
    #          all_bbox_preds,
    #          all_centernesses_preds,
    #          gt_bboxes,
    #          gt_labels,
    #          img_metas,
    #          gt_bboxes_ignore=None
    #          ):
    #
    #
    #
    #     all_gt_bboxes_list = [gt_bboxes for _ in range(self.cascade_stages)]
    #     all_gt_labels_list = [gt_labels for _ in range(self.cascade_stages)]
    #     all_img_metas_list = [img_metas for _ in range(self.cascade_stages)]
    #     all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(self.cascade_stages)]
    #
    #     losses_cls, losses_bbox, losses_iou, losses_centerness = multi_apply(self.loss_single,
    #                                                                          all_cls_scores,
    #                                                                          all_bbox_preds,
    #                                                                          all_centernesses_preds,
    #                                                                          all_gt_bboxes_list,
    #                                                                          all_gt_labels_list,
    #                                                                          all_img_metas_list,
    #                                                                          all_gt_bboxes_ignore_list)
    #
    #     loss_dict = dict()
    #     for stage, (loss_cls_i, loss_bbox_i, loss_iou_i, loss_centerness_i) in enumerate(
    #             zip(losses_cls, losses_bbox, losses_iou, losses_centerness)):
    #         loss_dict[f'stage{stage}.loss_cls'] = loss_cls_i
    #         loss_dict[f'stage{stage}.loss_bbox'] = loss_bbox_i
    #         loss_dict[f'stage{stage}.loss_iou'] = loss_iou_i
    #         loss_dict[f'stage{stage}.loss_centerness'] = loss_centerness_i
    #     return loss_dict
    #
    # def loss_single(self,
    #                 cls_scores,
    #                 bbox_preds,
    #                 centernesses_preds,
    #                 gt_bboxes,
    #                 gt_labels,
    #                 img_metas,
    #                 gt_bboxes_ignore=None):
    #     """Compute loss of the head. """
    #     assert len(cls_scores) == len(bbox_preds) == len(centernesses_preds)
    #     device = cls_scores[0].device
    #     dtype = cls_scores[0].dtype
    #
    #     featmap_sizes = [bbox_preds.size()[1:-1]]
    #     all_level_points = self.prior_generator.grid_priors(
    #         featmap_sizes,
    #         dtype=bbox_preds.dtype,
    #         device=bbox_preds.device)
    #
    #     labels, bbox_targets = self.get_targets(
    #         all_level_points,
    #         img_metas,
    #         gt_bboxes,
    #         gt_labels)
    #
    #     # flatten cls_scores, bbox_preds and centerness
    #     flatten_cls_scores = [cls_score.reshape(-1, self.cls_out_channels)
    #                           for cls_score in cls_scores]
    #     flatten_bbox_preds = [bbox_pred.reshape(-1, 4)
    #                           for bbox_pred in bbox_preds]
    #     flatten_centerness = [centerness_pred.reshape(-1)
    #                           for centerness_pred in centernesses_preds]
    #     flatten_cls_scores = torch.cat(flatten_cls_scores)
    #     flatten_bbox_preds = torch.cat(flatten_bbox_preds)
    #     flatten_centerness = torch.cat(flatten_centerness)
    #
    #     num_imgs = cls_scores.size(0)
    #     flatten_labels = torch.cat(labels)
    #     flatten_bbox_targets = torch.cat(bbox_targets)
    #
    #     # img_batch_h, img_batch_w = img_metas[0]['batch_input_shape']
    #     # for lv_idx, points in enumerate(range(len(all_level_points))):
    #     #     all_level_points[lv_idx][:, 0] /= img_batch_w
    #     #     all_level_points[lv_idx][:, 1] /= img_batch_h
    #     #     all_level_points[lv_idx][:, 0] *= featmap_sizes[lv_idx][1]
    #     #     all_level_points[lv_idx][:, 1] *= featmap_sizes[lv_idx][0]
    #     #     all_level_points[lv_idx] = all_level_points[lv_idx].repeat(num_imgs, 1).long()
    #     # flatten_points = torch.cat(all_level_points)
    #     flatten_points = torch.cat([points.repeat(num_imgs, 1)
    #                                 for points in all_level_points])
    #
    #     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    #     bg_class_ind = self.num_classes
    #     pos_inds = ((flatten_labels >= 0)
    #                 & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
    #
    #     num_pos = torch.tensor(
    #         len(pos_inds), dtype=torch.float, device=device)
    #     num_pos = max(reduce_mean(num_pos), 1.0)
    #
    #     loss_cls = self.loss_cls(
    #         flatten_cls_scores, flatten_labels, avg_factor=num_pos)
    #
    #     batch_h, batch_w = img_metas[0]['batch_input_shape']
    #     factor = bbox_preds.new_tensor([batch_w, batch_h, batch_w, batch_h]).unsqueeze(0)
    #     factor = factor.repeat(len(pos_inds), 1)
    #     # for img_meta, bbox_pred in zip(img_metas, bbox_preds):
    #     #     img_h, img_w, _ = img_meta['img_shape']
    #     #     factor = bbox_pred.new_tensor([img_w, img_h, img_w,
    #     #                                    img_h]).unsqueeze(0).repeat(
    #     #         bbox_pred.size(0), 1)
    #     #     factors.append(factor)
    #     # factors = torch.cat(factors, 0)
    #     #
    #     # factors_pos = factors[pos_inds]
    #     pos_points = flatten_points[pos_inds]
    #
    #     pos_bbox_preds = flatten_bbox_preds[pos_inds]
    #     pos_centerness = flatten_centerness[pos_inds]
    #     pos_bbox_targets = flatten_bbox_targets[pos_inds]
    #     pos_centerness_targets = self.centerness_target(pos_bbox_targets)
    #
    #     # centerness weighted iou loss
    #     centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
    #
    #     if len(pos_inds) > 0:
    #         # construct factors used for rescale bboxes
    #
    #         pos_decoded_bbox_preds = self.bbox_coder_point.decode(
    #             pos_points, pos_bbox_preds)
    #         pos_decoded_bbox_targets = self.bbox_coder_point.decode(
    #             pos_points, pos_bbox_targets)
    #
    #         loss_iou = self.loss_iou(
    #             pos_decoded_bbox_preds,
    #             pos_decoded_bbox_targets,
    #             weight=pos_centerness_targets,
    #             avg_factor=centerness_denorm)
    #
    #         loss_bbox = self.loss_bbox(
    #             pos_decoded_bbox_preds / factor,
    #             pos_decoded_bbox_targets / factor,
    #             weight=pos_centerness_targets.unsqueeze(1).repeat(1, 4),
    #             avg_factor=centerness_denorm)
    #
    #         loss_centerness = self.loss_centerness(
    #             pos_centerness, pos_centerness_targets, avg_factor=num_pos)
    #     else:
    #         loss_bbox = pos_bbox_preds.sum()
    #         loss_iou = pos_bbox_preds.sum()
    #         loss_centerness = pos_centerness.sum()
    #
    #     return loss_cls, loss_bbox, loss_iou, loss_centerness

    def get_targets(self,
                    points,
                    img_metas,
                    gt_bboxes_list,
                    gt_labels_list):
        """Compute regression, classification and centerness targets for points
                        in multiple images."""
        # assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        # expanded_regress_ranges = [
        #     points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
        #         points[i]) for i in range(num_levels)
        # ]
        # concat all levels points and regress ranges
        # concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        labels_list, bbox_targets_list = multi_apply(
            self._get_targets_single,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            points=concat_points,
            # regress_ranges=concat_regress_ranges,
            num_points_per_lv=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0)
                       for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0)
                             for bbox_targets in bbox_targets_list]

        # concat per level image
        concat_lv_labels = []
        concat_lv_bbox_targets = []
        for i in range(num_levels):
            concat_lv_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lv_bbox_targets.append(
                torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list]))

        return concat_lv_labels, concat_lv_bbox_targets

    def _get_targets_single(self,
                            gt_bboxes,
                            gt_labels,
                            img_meta,
                            points,
                            # regress_ranges,
                            num_points_per_lv):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        img_batch_h, img_batch_w = img_meta['batch_input_shape']

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        # regress_ranges = regress_ranges[:, None, :].expand(
        #     num_points, num_gts, 2)

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        max_regress_distance = bbox_targets.max(-1)[0]
        # inside_regress_range = (
        #         (max_regress_distance >= regress_ranges[..., 0])
        #         & (max_regress_distance <= regress_ranges[..., 1]))

        # choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        # areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   all_score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.
        """
        # only use last stage prediction
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        score_factors = all_score_factors[-1]
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(score_factors)

        # num_levels = len(cls_scores)

        featmap_sizes = [bbox_preds.shape[1:-1]]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores.dtype,
            device=cls_scores.device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl([cls_scores], img_id)
            bbox_pred_list = select_single_mlvl([bbox_preds], img_id)
            score_factor_list = select_single_mlvl([score_factors], img_id)
            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size(0) == bbox_pred.size(0) * bbox_pred.size(1)

            bbox_pred = bbox_pred.reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.reshape(-1).sigmoid()
            cls_score = cls_score.reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            # bboxes = self.bbox_coder.decode(
            #     priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)








