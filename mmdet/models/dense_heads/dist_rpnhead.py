# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmdet.core import multi_apply, images_to_levels

from ..builder import HEADS, build_loss
from .rpn_head import RPNHead


class DistillNeck(nn.Module):
    def __init__(self,
                 student_channel,
                 teacher_channel,
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 w_neck=((1.0, 0.5), (1.0, 0.5), (1.0, 0.5))
                 ):
        super().__init__()
        self.student_channel = student_channel
        self.teacher_channel = teacher_channel
        self.mimic_conv = nn.Conv2d(student_channel,
                                    teacher_channel,
                                    kernel_size=1,
                                    padding=0)
        self.scale_feat = False
        self.scale_roi = False
        roi_size = 7
        self.roi_align_extract = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1,
            sampling_ratio=2,
            aligned=True)
        self.loss_mse = build_loss(loss_mse)
        self.cos_simm = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.w_fpn = w_neck[0]
        self.w_roi = w_neck[1]
        self.w_mk = w_neck[2]

    def _get_fg_bg_mask(self, tensor_info, gt_bboxes, img_metas, scale=False):
        batch, h, w, device = tensor_info
        if scale:
            scale_fg = torch.zeros((batch, h, w), dtype=torch.float32, device=device)
            scale_bg = torch.ones((batch, h, w), dtype=torch.float32, device=device)
        mask_fg = torch.zeros((batch, h, w), dtype=torch.float32, device=device)
        mask_bg = torch.ones((batch, h, w), dtype=torch.float32, device=device)
        for b in range(batch):
            num_obj = len(gt_bboxes[b])
            x1 = (gt_bboxes[b][:, 0] / img_metas[b]['img_shape'][1] * w).to(torch.int)
            x2 = (gt_bboxes[b][:, 2] / img_metas[b]['img_shape'][1] * w).to(torch.int)
            y1 = (gt_bboxes[b][:, 1] / img_metas[b]['img_shape'][0] * h).to(torch.int)
            y2 = (gt_bboxes[b][:, 3] / img_metas[b]['img_shape'][0] * h).to(torch.int)
            if scale:
                obj_scale = 1.0 / ((x2 + 1 - x1) * (y2 + 1 - y1))
                for j in range(num_obj):
                    scale_fg[b, y1[j]:y2[j] + 1, x1[j]:x2[j] + 1] = torch.maximum(
                        mask_fg[b, y1[j]:y2[j] + 1, x1[j]:x2[j] + 1], obj_scale[j])
                scale_bg[b] = torch.where(scale_fg[b] > 0, 0, 1)
                if torch.sum(mask_bg[b]):
                    scale_bg[b] /= (torch.sum(mask_bg[b]))
            for j in range(num_obj):
                mask_fg[b, y1[j]:y2[j] + 1, x1[j]:x2[j] + 1] = 1
            mask_bg[b] = torch.where(mask_fg[b] > 0, 0, 1)
        if scale:
            return mask_fg, mask_bg, scale_fg, scale_bg
        else:
            return mask_fg, mask_bg

    def get_rois(self, tensor_info, gt_bboxes, img_metas):
        batch, h, w, dtype, device = tensor_info
        rois = []
        scales = []
        for b in range(batch):
            x1 = (gt_bboxes[b][:, 0] / img_metas[b]['img_shape'][1] * w).to(dtype)
            x2 = (gt_bboxes[b][:, 2] / img_metas[b]['img_shape'][1] * w).to(dtype)
            y1 = (gt_bboxes[b][:, 1] / img_metas[b]['img_shape'][0] * h).to(dtype)
            y2 = (gt_bboxes[b][:, 3] / img_metas[b]['img_shape'][0] * h).to(dtype)
            obj_scale = 1.0 / ((x2+1 - x1)*(y2+1 - y1))
            bindx = torch.ones_like(x1) * b
            roi = torch.stack([bindx, x1, y1, x2, y2], dim=1)
            rois.append(roi)
            scales.append(obj_scale)
        return torch.cat(rois), torch.cat(scales)

    def _loss_mse_cos(self, student, teacher, weight, mask=None, avg_factor=None):
        teacher = teacher.detach()
        b, c, h, w = student.shape

        cos_dist_ch = self.cos_simm(student.view(b, c, -1), teacher.view(b, c, -1))
        cos_dist_sp = self.cos_simm(student.view(b, c, -1).permute(0, 2, 1), teacher.view(b, c, -1).permute(0, 2, 1))

        cos_guidance = (1 - cos_dist_ch[:, :, None]) * (1 - cos_dist_sp[:, None, :])
        cos_guidance = cos_guidance.view(b, c, h, w).abs()
        if avg_factor is not None:
            loss_cos = (1 - cos_dist_ch).mean() + (1 - cos_dist_sp).sum() / avg_factor
        else:
            loss_cos = (1 - cos_dist_ch).mean() + (1 - cos_dist_sp).mean()

        if mask is not None:
            cos_guidance = cos_guidance * mask[:, None, :, :]

        loss_mse = self.loss_mse(student,
                                 teacher,
                                 weight=cos_guidance,
                                 avg_factor=avg_factor)

        loss = weight[0]*loss_mse + weight[1]*loss_cos
        return loss

    def forward(self, student, teacher, gt_bboxes, img_metas):
        b, c, h, w = student.shape
        device = student.device
        dtype = student.dtype

        _student = self.mimic_conv(student)
        loss_feat = self._loss_mse_cos(_student, teacher,
                                       weight=self.w_fpn)

        rois, scales = self.get_rois([b, h, w, dtype, device], gt_bboxes, img_metas)

        st_rois = self.roi_align_extract(_student, rois)
        tch_rois = self.roi_align_extract(teacher, rois)

        if self.scale_roi:
            loss_rois = self._loss_mse_cos(st_rois * scales[:, None, None, None],
                                           tch_rois * scales[:, None, None, None],
                                           weight=self.w_roi)
        else:
            loss_rois = self._loss_mse_cos(st_rois, tch_rois,
                                           weight=self.w_roi)

        if self.scale_feat:
            mask_fg, mask_bg, scale_fg, scale_bg = self._get_fg_bg_mask([b, h, w, device], gt_bboxes, img_metas, scale=self.scale_feat)
            msk_scale_fg = scale_fg[:, None, :, :]
            msk_scale_bg = scale_bg[:, None, :, :]
        else:
            mask_fg, mask_bg = self._get_fg_bg_mask([b, h, w, device], gt_bboxes, img_metas, scale=self.scale_feat)
            msk_scale_fg = mask_fg[:, None, :, :]
            msk_scale_bg = mask_bg[:, None, :, :]

        if msk_scale_bg.sum() > 0:
            teacher_bg = teacher * msk_scale_bg
            _student_bg = _student * msk_scale_bg
            loss_bg = self._loss_mse_cos(_student_bg, teacher_bg,
                                         mask=mask_bg,
                                         weight=self.w_mk,
                                         avg_factor=mask_bg.sum() * c)
        else:
            loss_bg = msk_scale_bg.sum()
        teacher_fg = teacher * msk_scale_fg
        _student_fg = _student * msk_scale_fg
        loss_fg = self._loss_mse_cos(_student_fg, teacher_fg,
                                     mask=mask_fg,
                                     weight=self.w_mk,
                                     avg_factor=mask_fg.sum() * c)
        return loss_feat, loss_rois, loss_fg, loss_bg


class DistillHead(nn.Module):
    def __init__(self,
                 num_stages,
                 student_channel,
                 teacher_channel,
                 kd_costhresh=0.1,
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 w_branch=(1.0, 0.5)
                 ):
        super().__init__()
        self.num_stages = num_stages
        self.kd_costhresh = kd_costhresh
        if num_stages == 1:
            self.mimic_conv = nn.Conv2d(student_channel,
                                        teacher_channel,
                                        kernel_size=1,
                                        padding=0)
        else:
            self.mimic_conv = nn.ModuleList([nn.Conv2d(student_channel,
                                                       teacher_channel,
                                                       kernel_size=1,
                                                       padding=0)
                                             for _ in range(num_stages)])
        self.loss_mse = build_loss(loss_mse)
        self.cos_simm = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.w_branch = w_branch

    def _loss_mse_cos(self, student, teacher, weight, mask=None, avg_factor=None):
        teacher = teacher.detach()
        b, c, h, w = student.shape

        cos_dist_ch = self.cos_simm(student.view(b, c, -1), teacher.view(b, c, -1))
        cos_dist_sp = self.cos_simm(student.view(b, c, -1).permute(0, 2, 1), teacher.view(b, c, -1).permute(0, 2, 1))

        cos_guidance = (1 - cos_dist_ch[:, :, None]) * (1 - cos_dist_sp[:, None, :])
        cos_guidance = cos_guidance.view(b, c, h, w).abs()
        if avg_factor is not None:
            loss_cos = (1 - cos_dist_ch).mean() + (1 - cos_dist_sp).sum() / avg_factor
        else:
            loss_cos = (1 - cos_dist_ch).mean() + (1 - cos_dist_sp).mean()

        if mask is not None:
            cos_guidance = cos_guidance * mask[:, None, :, :]

        loss_mse = self.loss_mse(student,
                                 teacher,
                                 weight=cos_guidance,
                                 avg_factor=avg_factor)

        loss = weight[0]*loss_mse + weight[1]*loss_cos
        return loss

    def forward(self, student, teacher, gt_bboxes, img_metas):
        if self.num_stages == 1:
            _student = self.mimic_conv(student)
            loss_feat = self._loss_mse_cos(_student, teacher, weight=self.w_branch)
        else:
            loss_feat = []
            for i in range(self.num_stages):
                _student = self.mimic_conv[i](student[i])
                loss_feat.append(self._loss_mse_cos(_student, teacher[i], weight=self.w_branch))
        return loss_feat


class DistillPrediction(nn.Module):
    def __init__(self,
                 student_channel,
                 teacher_channel,
                 sigmoid=False,
                 mask=False,
                 avg_factor=False,
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 w_pred=(0.5, 0.2)
                 ):
        super().__init__()
        self.sigmoid = sigmoid
        self.mask = mask
        self.avg_factor = avg_factor
        self.loss_mse = build_loss(loss_mse)
        self.cos_simm = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.w_pred = w_pred

    def _get_fg_bg_mask(self, tensor_info, gt_bboxes, img_metas, scale=False):
        batch, h, w, device = tensor_info
        if scale:
            scale_fg = torch.zeros((batch, h, w), dtype=torch.float32, device=device)
            scale_bg = torch.ones((batch, h, w), dtype=torch.float32, device=device)
        mask_fg = torch.zeros((batch, h, w), dtype=torch.float32, device=device)
        mask_bg = torch.ones((batch, h, w), dtype=torch.float32, device=device)
        for b in range(batch):
            num_obj = len(gt_bboxes[b])
            x1 = (gt_bboxes[b][:, 0] / img_metas[b]['img_shape'][1] * w).to(torch.int)
            x2 = (gt_bboxes[b][:, 2] / img_metas[b]['img_shape'][1] * w).to(torch.int)
            y1 = (gt_bboxes[b][:, 1] / img_metas[b]['img_shape'][0] * h).to(torch.int)
            y2 = (gt_bboxes[b][:, 3] / img_metas[b]['img_shape'][0] * h).to(torch.int)
            if scale:
                obj_scale = 1.0 / ((x2 + 1 - x1) * (y2 + 1 - y1))
                for j in range(num_obj):
                    scale_fg[b, y1[j]:y2[j] + 1, x1[j]:x2[j] + 1] = torch.maximum(
                        mask_fg[b, y1[j]:y2[j] + 1, x1[j]:x2[j] + 1], obj_scale[j])
                scale_bg[b] = torch.where(scale_fg[b] > 0, 0, 1)
                if torch.sum(mask_bg[b]):
                    scale_bg[b] /= (torch.sum(mask_bg[b]))
            for j in range(num_obj):
                mask_fg[b, y1[j]:y2[j] + 1, x1[j]:x2[j] + 1] = 1
            mask_bg[b] = torch.where(mask_fg[b] > 0, 0, 1)
        if scale:
            return mask_fg, mask_bg, scale_fg, scale_bg
        else:
            return mask_fg, mask_bg

    def _loss_mse_cos(self, student, teacher, weight, mask=None, avg_factor=None):
        teacher = teacher.detach()
        b, c, h, w = student.shape

        if mask is not None:
            teacher = teacher * mask[:, None, :, :]
            student = student * mask[:, None, :, :]

        cos_dist_ch = self.cos_simm(student.view(b, c, -1), teacher.view(b, c, -1))
        cos_dist_sp = self.cos_simm(student.view(b, c, -1).permute(0, 2, 1), teacher.view(b, c, -1).permute(0, 2, 1))

        cos_guidance = (1 - cos_dist_ch[:, :, None]) * (1 - cos_dist_sp[:, None, :])
        # cos_guidance = cos_guidance.view(b, c, h, w).abs().clamp(min=0.5)
        cos_guidance = cos_guidance.view(b, c, h, w).abs()
        if avg_factor is not None:
            loss_cos = (1 - cos_dist_ch).mean() + (1 - cos_dist_sp).sum() / avg_factor
        else:
            loss_cos = (1 - cos_dist_ch).mean() + (1 - cos_dist_sp).mean()

        if mask is not None:
            cos_guidance = cos_guidance * mask[:, None, :, :]

        loss_mse = self.loss_mse(student,
                                 teacher,
                                 weight=cos_guidance,
                                 # avg_factor=cos_guidance.sum())
                                 avg_factor=avg_factor)

        loss = weight[0]*loss_mse + weight[1]*loss_cos
        return loss

    def forward(self, student, teacher, gt_bboxes, img_metas):
        device = student.device
        b, c, h, w = student.shape
        mask_fg, _ = self._get_fg_bg_mask([b, h, w, device], gt_bboxes, img_metas)
        avg_factor = mask_fg.sum() * c

        if self.sigmoid:
            student = student.sigmoid()
            teacher = teacher.sigmoid()

        loss_kdpred = self._loss_mse_cos(student, teacher,
                                         mask=mask_fg if self.mask else None,
                                         weight=self.w_pred,
                                         avg_factor=avg_factor if self.avg_factor else None)
        return loss_kdpred

@HEADS.register_module()
class dist_RPNHead(RPNHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """  # noqa: W605

    def __init__(self,
                 is_teacher,
                 kd_weight=(1, 0.5),
                 **kwargs):
        self.is_teacher = is_teacher
        super(dist_RPNHead, self).__init__(**kwargs)
        if not is_teacher:
            self.lv_len = 5
            self.kd_weight = kd_weight
            self.distill_neck = nn.ModuleList()
            self.distill_head = nn.ModuleList()
            self.distill_pred_bbox = nn.ModuleList()
            self.distill_pred_cls = nn.ModuleList()
            self.now_indx = []
            for l in range(self.lv_len):
                self.now_indx.append(l)
                self.distill_neck.append(DistillNeck(student_channel=256,
                                                     teacher_channel=256,
                                                     w_neck=(self.kd_weight, self.kd_weight, self.kd_weight)))
                self.distill_head.append(DistillHead(num_stages=self.num_convs,
                                                     student_channel=256,
                                                     teacher_channel=256,
                                                     w_branch=self.kd_weight))
                self.distill_pred_bbox.append(DistillPrediction(student_channel=self.num_base_priors * 4,
                                                                teacher_channel=self.num_base_priors * 4,
                                                                sigmoid=False,
                                                                mask=False,
                                                                avg_factor=True,
                                                                w_pred=self.kd_weight))
                self.distill_pred_cls.append(DistillPrediction(student_channel=self.num_base_priors * self.cls_out_channels,
                                                               teacher_channel=self.num_base_priors * self.cls_out_channels,
                                                               sigmoid=True,
                                                               mask=False,
                                                               avg_factor=False,
                                                               w_pred=self.kd_weight))

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * 4,
                                 1)

        if not self.is_teacher:
            self.assist_rpn_cls = nn.Conv2d(self.feat_channels,
                                            self.num_base_priors * self.cls_out_channels,
                                            1)
            self.assist_rpn_reg = nn.Conv2d(self.feat_channels,
                                            self.num_base_priors * 4,
                                            1)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        rpn_feat = F.relu(self.rpn_conv(x), inplace=True)

        rpn_cls_score = self.rpn_cls(rpn_feat)
        rpn_bbox_pred = self.rpn_reg(rpn_feat)

        if not self.is_teacher and self.training:
            assist_rpn_cls_score = self.assist_rpn_cls(rpn_feat)
            assist_rpn_bbox_pred = self.assist_rpn_reg(rpn_feat)
            return rpn_feat, rpn_cls_score, rpn_bbox_pred, assist_rpn_cls_score, assist_rpn_bbox_pred
        elif self.is_teacher:
            return rpn_feat, rpn_cls_score, rpn_bbox_pred
        else:
            return rpn_cls_score, rpn_bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def loss_distill_neck_single(self, now_indx, student_x, teacher_x,
                                 gt_bboxes, img_metas):
        loss_neck, loss_roi, loss_fg, loss_bg = self.distill_neck[now_indx](student_x, teacher_x,
                                                                            gt_bboxes, img_metas)
        return loss_neck, loss_roi, loss_fg, loss_bg

    def loss_distill_head_single(self,
                                 now_indx,
                                 st_head_feats,
                                 tch_head_feats,
                                 gt_bboxes, img_metas):
        loss_head = self.distill_head[now_indx](st_head_feats,
                                                tch_head_feats,
                                                gt_bboxes, img_metas)
        return loss_head,

    def loss_distill_prediction_single(self,
                                       now_indx,
                                       st_pred_cls,
                                       st_pred_bbox,
                                       tch_pred_cls,
                                       tch_pred_bbox,
                                       gt_bboxes, img_metas
                                       ):
        loss_kdcls = self.distill_pred_cls[now_indx](st_pred_cls,
                                                     tch_pred_cls,
                                                     gt_bboxes, img_metas)
        loss_kdbbox = self.distill_pred_bbox[now_indx](st_pred_bbox,
                                                       tch_pred_bbox,
                                                       gt_bboxes, img_metas)
        return loss_kdcls, loss_kdbbox

    def loss_single(self,
                    cls_score,
                    bbox_pred,
                    assis_cls_score,
                    assis_bbox_pred,
                    cls_soft_label,
                    bbox_soft_target,
                    anchors, labels, label_weights,
                    bbox_targets, bbox_weights,
                    num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        assis_cls_score = assis_cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        cls_soft_label = cls_soft_label.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

        loss_cls = self.loss_cls(cls_score,
                                 labels,
                                 weight=label_weights,
                                 avg_factor=num_total_samples)

        cls_soft_label = cls_soft_label.detach().sigmoid()
        loss_assis_cls = self.loss_cls(assis_cls_score,
                                       cls_soft_label,
                                       weight=label_weights[:, None],
                                       avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        assis_bbox_pred = assis_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_soft_target = bbox_soft_target.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            assis_bbox_pred = self.bbox_coder.decode(anchors, assis_bbox_pred)
            bbox_soft_target = self.bbox_coder.decode(anchors, bbox_soft_target)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        loss_assis_bbox = self.loss_bbox(
            assis_bbox_pred,
            bbox_soft_target.detach(),
            weight=bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox, loss_assis_cls, loss_assis_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             student_x, teacher_x,
             st_rpn_feat, tch_rpn_feat,
             assist_rpn_cls_score, assist_rpn_bbox_pred,
             tch_rpn_cls_score, tch_rpn_bbox_pred,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_assist_cls, losses_assist_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            assist_rpn_cls_score, assist_rpn_bbox_pred,
            tch_rpn_cls_score, tch_rpn_bbox_pred,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        loss_neck, loss_nkroi, loss_nkfg, loss_nkbg = multi_apply(
            self.loss_distill_neck_single,
            self.now_indx,
            student_x,
            teacher_x,
            gt_bboxes=gt_bboxes,
            img_metas=img_metas)
        loss_neck = sum(loss_neck) / len(loss_neck)
        loss_nkroi = sum(loss_nkroi) / len(loss_nkroi)
        loss_nkfg = sum(loss_nkfg) / len(loss_nkfg)
        loss_nkbg = sum(loss_nkbg) / len(loss_nkbg)

        loss_kdhead = multi_apply(
            self.loss_distill_head_single,
            self.now_indx,
            st_rpn_feat,
            tch_rpn_feat,
            gt_bboxes=gt_bboxes,
            img_metas=img_metas)[0]
        loss_kdhead = sum(loss_kdhead) / len(loss_kdhead)

        loss_kdcls, loss_kdbbox = multi_apply(
            self.loss_distill_prediction_single,
            self.now_indx,
            cls_scores,
            bbox_preds,
            tch_rpn_cls_score,
            tch_rpn_bbox_pred,
            gt_bboxes=gt_bboxes,
            img_metas=img_metas)
        loss_kdcls = sum(loss_kdcls) / len(loss_kdcls)
        loss_kdbbox = sum(loss_kdbbox) / len(loss_kdbbox)

        return dict(
            loss_neck=loss_neck,
            loss_nkroi=loss_nkroi,
            loss_nkfg=loss_nkfg,
            loss_nkbg=loss_nkbg,

            loss_rpn_kdhead=loss_kdhead,

            loss_rpn_kdcls=loss_kdcls,
            loss_rpn_kdbbox=loss_kdbbox,

            loss_rpn_assist_cls=losses_assist_cls,
            loss_rpn_assist_bbox=losses_assist_bbox,

            loss_rpn_cls=losses_cls,
            loss_rpn_bbox=losses_bbox,
        )

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_anchors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RPN head does not need this value.
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                           level_ids, cfg, img_shape, **kwargs):
        """bbox post-processing method.

        Do the nms operation for bboxes in same level.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            mlvl_valid_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_bboxes, 4).
            level_ids (list[Tensor]): Indexes from all scale levels of a
                single image, each item has shape (num_bboxes, ).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, `self.test_cfg` would be used.
            img_shape (tuple(int)): The shape of model's input image.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]

    def onnx_export(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.
        Returns:
            Tensor: dets of shape [N, num_det, 5].
        """
        cls_scores, bbox_preds = self(x)

        assert len(cls_scores) == len(bbox_preds)

        batch_bboxes, batch_scores = super(dist_RPNHead, self).onnx_export(
            cls_scores, bbox_preds, img_metas=img_metas, with_nms=False)
        # Use ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        cfg = copy.deepcopy(self.test_cfg)
        score_threshold = cfg.nms.get('score_thr', 0.0)
        nms_pre = cfg.get('deploy_nms_pre', -1)
        # Different from the normal forward doing NMS level by level,
        # we do NMS across all levels when exporting ONNX.
        dets, _ = add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                         cfg.max_per_img,
                                         cfg.nms.iou_threshold,
                                         score_threshold, nms_pre,
                                         cfg.max_per_img)
        return dets
