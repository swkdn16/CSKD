# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from torchvision.transforms import GaussianBlur

from mmcv.cnn import build_norm_layer, Scale

from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean, images_to_levels, bbox_overlaps
from ..builder import HEADS, build_loss
from .dist_gfl_head import dist_GFLHead

import matplotlib.pyplot as plt


class DistillNeck(nn.Module):
    def __init__(self,
                 student_channel,
                 teacher_channel,
                 kd_costhresh=0.1,
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 w_neck=((5.0, 2.0), (1.0, 2.0), (1.0, 2.0))
                 ):
        super().__init__()
        self.student_channel = student_channel
        self.teacher_channel = teacher_channel
        self.kd_costhresh = kd_costhresh
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


class Distillbranch(nn.Module):
    def __init__(self,
                 num_stages,
                 student_channel,
                 teacher_channel,
                 kd_costhresh=0.1,
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 w_branch=(0.5, 0.2)
                 ):
        super().__init__()
        self.num_stages = num_stages
        self.kd_costhresh = kd_costhresh
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
        loss_feat = []
        for i in range(self.num_stages):
            _student = self.mimic_conv[i](student[i])
            loss_feat.append(self._loss_mse_cos(student[i], teacher[i], weight=self.w_branch))
        return loss_feat


class Distillprediction(nn.Module):
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
class CSKDHeadGFL(dist_GFLHead):
    def __init__(self,
                 loss_kdcls=dict(
                     type='pyFocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 kd_costhresh=0.,
                 kd_neck_weight=(2.0, 2.0),
                 kd_brnch_weight=((1.0, 2.0), (3.0, 2.0)),
                 kd_pred_weight=((1.0, 2.0), (1.0, 2.0)),

                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=
                     [dict(
                         type='Normal',
                         name='gfl_cls',
                         std=0.01,
                         bias_prob=0.01),
                      dict(
                         type='Normal',
                         name='assis_gfl_cls',
                         std=0.01,
                         bias_prob=0.01)
                     ]),
                 **kwargs):

        super(CSKDHeadGFL, self).__init__(
            init_cfg=init_cfg,
            **kwargs)
        self.loss_kdcls = build_loss(loss_kdcls)
        self.cos_simm = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.loss_mse = build_loss(dict(type='MSELoss', loss_weight=1.0))

        self.kd_costhresh = kd_costhresh
        self.kd_neck_weight = kd_neck_weight
        self.kd_brnch_weight = kd_brnch_weight
        self.kd_pred_weight = kd_pred_weight

        self.eps = 1e-6

        self.lv_len = 5
        self.now_indx = []
        self.distill_neck = nn.ModuleList()
        self.distill_brnch_reg = nn.ModuleList()
        self.distill_brnch_cls = nn.ModuleList()
        self.distill_pred_bbox = nn.ModuleList()
        self.distill_pred_cls = nn.ModuleList()
        for l in range(self.lv_len):
            self.now_indx.append(l)
            self.distill_neck.append(DistillNeck(student_channel=256,
                                                 teacher_channel=256,
                                                 kd_costhresh=self.kd_costhresh,
                                                 w_neck=self.kd_neck_weight))
            self.distill_brnch_reg.append(Distillbranch(num_stages=self.stacked_convs,
                                                        student_channel=256,
                                                        teacher_channel=256,
                                                        kd_costhresh=self.kd_costhresh,
                                                        w_branch=self.kd_brnch_weight[0]))
            self.distill_brnch_cls.append(Distillbranch(num_stages=self.stacked_convs,
                                                        student_channel=256,
                                                        teacher_channel=256,
                                                        kd_costhresh=self.kd_costhresh,
                                                        w_branch=self.kd_brnch_weight[1]))
            self.distill_pred_bbox.append(Distillprediction(student_channel=self.num_base_priors * 4,
                                                            teacher_channel=self.num_base_priors * 4,
                                                            sigmoid=False,
                                                            mask=False,
                                                            avg_factor=True,
                                                            w_pred=self.kd_pred_weight))
            self.distill_pred_cls.append(Distillprediction(student_channel=self.num_base_priors * self.cls_out_channels,
                                                           teacher_channel=self.num_base_priors * self.cls_out_channels,
                                                           sigmoid=True,
                                                           mask=False,
                                                           avg_factor=True,
                                                           w_pred=self.kd_pred_weight))
        self._init_assistant_predictor()

    def _init_assistant_predictor(self):
        """Initialize predictor layers of the head."""
        self.assis_gfl_cls = nn.Conv2d(self.feat_channels,
                                          self.num_base_priors * self.cls_out_channels, 3, padding=1)
        self.assis_gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.assis_scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

    def forward_train(self,
                      student_x,
                      teacher_x, out_teacher,
                      img_metas,
                      gt_bboxes, gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        # origin bbox head
        out_student = self(student_x)
        outs = out_student[:2]

        st_cls_conv_feats = out_student[2]
        st_reg_conv_feats = out_student[3]
        st_cls_feat = out_student[4]
        st_reg_feat = out_student[5]

        assis_cls_score = []
        assis_bbox_pred = []

        for l in range(self.lv_len):
            assis_cls_score.append(self.assis_gfl_cls(st_cls_feat[l]))
            assis_bbox_pred.append(self.assis_scales[l](self.assis_gfl_reg(st_reg_feat[l])).float())

        cls_soft_labels = out_teacher[0]
        bbox_soft_targets = out_teacher[1]
        tch_cls_conv_feats = out_teacher[2]
        tch_reg_conv_feats = out_teacher[3]

        if gt_labels is None:
            loss_inputs = outs + (student_x, teacher_x, gt_bboxes,
                                  assis_cls_score,
                                  assis_bbox_pred,
                                  cls_soft_labels,
                                  bbox_soft_targets,
                                  st_cls_conv_feats, st_reg_conv_feats,
                                  tch_cls_conv_feats, tch_reg_conv_feats,
                                  img_metas)
        else:
            loss_inputs = outs + (student_x, teacher_x, gt_bboxes, gt_labels,
                                  assis_cls_score,
                                  assis_bbox_pred,
                                  cls_soft_labels,
                                  bbox_soft_targets,
                                  st_cls_conv_feats, st_reg_conv_feats,
                                  tch_cls_conv_feats, tch_reg_conv_feats,
                                  img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def loss_distill_neck_single(self, now_indx, student_x, teacher_x,
                                 gt_bboxes, img_metas):
        loss_neck, loss_roi, loss_fg, loss_bg = self.distill_neck[now_indx](student_x, teacher_x, gt_bboxes, img_metas)
        return loss_neck, loss_roi, loss_fg, loss_bg

    def loss_distill_brnch_single(self,
                                  now_indx,
                                  st_cls_conv_feats,
                                  st_reg_conv_feats,
                                  tch_cls_conv_feats,
                                  tch_reg_conv_feats,
                                  gt_bboxes, img_metas
                                  ):
        loss_brchcls = self.distill_brnch_cls[now_indx](st_cls_conv_feats,
                                                        tch_cls_conv_feats,
                                                        gt_bboxes, img_metas)
        loss_brchreg = self.distill_brnch_reg[now_indx](st_reg_conv_feats,
                                                        tch_reg_conv_feats,
                                                        gt_bboxes, img_metas)
        return loss_brchcls, loss_brchreg

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

    def loss_single(self,
                    cls_score,
                    bbox_pred,
                    maks_fg,
                    assis_cls_score,
                    assis_bbox_pred,
                    cls_soft_label,
                    bbox_soft_target,
                    anchors,
                    labels,
                    label_weights,
                    bbox_targets,
                    stride,
                    num_total_samples):

        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        
        cls_soft_label = cls_soft_label * maks_fg[:, None, :, :]
        maks_fg = maks_fg[:, None, :, :].repeat(1, cls_soft_label.size(1), 1, 1)

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        assis_cls_score = assis_cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        cls_soft_label = cls_soft_label.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        maks_fg = maks_fg.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        assis_bbox_pred = assis_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_soft_target = bbox_soft_target.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)
        soft_score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_soft_targets = bbox_soft_target[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_assis_bbox_pred = assis_bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            assis_weight_targets = assis_cls_score.detach().sigmoid()
            assis_weight_targets = assis_weight_targets.max(dim=1)[0][pos_inds]

            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_bbox_assis_corners = self.integral(pos_assis_bbox_pred)
            pos_bbox_soft_corners = self.integral(pos_bbox_soft_targets)

            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_assis = self.bbox_coder.decode(pos_anchor_centers, pos_bbox_assis_corners)
            pos_decode_bbox_soft_targets = self.bbox_coder.decode(pos_anchor_centers, pos_bbox_soft_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            score[pos_inds] = bbox_overlaps(pos_decode_bbox_pred.detach(),
                                            pos_decode_bbox_targets,
                                            is_aligned=True)
            soft_score[pos_inds] = bbox_overlaps(pos_decode_bbox_assis.detach(),
                                                 pos_decode_bbox_soft_targets.detach(),
                                                 is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            pred_assis_corners = pos_assis_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)
            soft_target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                         pos_decode_bbox_soft_targets,
                                                         self.reg_max).reshape(-1)
                                

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)
            
            loss_assis_bbox = self.loss_bbox(
                pos_decode_bbox_assis,
                pos_decode_bbox_soft_targets,
                weight=assis_weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
            loss_assis_dfl = self.loss_dfl(
                pred_assis_corners,
                soft_target_corners,
                weight=assis_weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_assis_bbox = assis_bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_assis_dfl = assis_bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)
            assis_weight_targets = assis_bbox_pred.new_tensor(0)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        cls_soft_label = cls_soft_label.detach().sigmoid() * maks_fg
        loss_assis_cls = self.loss_cls(assis_cls_score, (labels, soft_score),
                                         weight=label_weights,
                                         avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, loss_assis_cls, loss_assis_bbox, loss_assis_dfl, weight_targets.sum(), assis_weight_targets.sum()


    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             student_x,
             teacher_x,
             gt_bboxes,
             gt_labels,
             assis_cls_scores,
             assis_bbox_preds,
             cls_soft_labels,
             bbox_soft_targets,
             st_cls_conv_feats, st_reg_conv_feats,
             tch_cls_conv_feats, tch_reg_conv_feats,
             img_metas,
             gt_bboxes_ignore=None):

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

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        mkfg_list = []
        mkbg_list = []
        scfg_list = []
        scbg_list = []
        for lv in range(len(student_x)):
            b, c, h, w = student_x[lv].shape
            device = student_x[lv].device
            mask_fg, mask_bg, scale_fg, scale_bg = self._get_fg_bg_mask([b, h, w, device], gt_bboxes, img_metas, scale=True)
            mkfg_list.append(mask_fg)
            mkbg_list.append(mask_bg)
            scfg_list.append(scale_fg)
            scbg_list.append(scale_bg)

        loss_kdcls, loss_kdbbox = multi_apply(
            self.loss_distill_prediction_single,
            self.now_indx,
            cls_scores,
            bbox_preds,
            cls_soft_labels,
            bbox_soft_targets,
            gt_bboxes=gt_bboxes,
            img_metas=img_metas)

        loss_cls, loss_bbox, loss_dfl, loss_assis_cls, loss_assis_bbox, loss_assis_dfl, avg_factor, assis_avg_factor = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            mkfg_list,
            assis_cls_scores,
            assis_bbox_preds,
            cls_soft_labels,
            bbox_soft_targets,
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.prior_generator.strides,
            num_total_samples=num_total_samples)
        
        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        assis_avg_factor = sum(assis_avg_factor)
        assis_avg_factor = reduce_mean(assis_avg_factor).clamp_(min=1).item()
        loss_bbox = list(map(lambda x: x / avg_factor, loss_bbox))
        loss_dfl = list(map(lambda x: x / avg_factor, loss_dfl))
        loss_assis_bbox = list(map(lambda x: x / assis_avg_factor, loss_assis_bbox))
        loss_assis_dfl = list(map(lambda x: x / assis_avg_factor, loss_assis_dfl))

        loss_neck, loss_nkroi, loss_nkfg, loss_nkbg = multi_apply(
            self.loss_distill_neck_single,
            self.now_indx,
            student_x,
            teacher_x,
            gt_bboxes=gt_bboxes,
            img_metas=img_metas)

        loss_brnchcls_list, loss_brnchreg_list = multi_apply(
            self.loss_distill_brnch_single,
            self.now_indx,
            st_cls_conv_feats,
            st_reg_conv_feats,
            tch_cls_conv_feats,
            tch_reg_conv_feats,
            gt_bboxes=gt_bboxes,
            img_metas=img_metas)

        # lv_weight = [mkfg_list[i].sum() / mkbg_list[i].sum() for i in range(self.lv_len-1, -1, -1)]
        # for i in range(self.lv_len):
        #     loss_kdcls[i] = loss_kdcls[i] * lv_weight[i]
        #     loss_kdbbox[i] = loss_kdbbox[i] * lv_weight[i]
        #     loss_neck[i] = loss_neck[i] * lv_weight[i]
        #     loss_nkroi[i] = loss_nkroi[i] * lv_weight[i]
        #     loss_nkfg[i] = loss_nkfg[i] * lv_weight[i]
        #     loss_nkbg[i] = loss_nkbg[i] * lv_weight[i]

        loss_brnchcls = []
        loss_brnchreg = []
        for s in range(self.stacked_convs):
            lv_loss_brnchcls = []
            lv_loss_brnchreg = []
            for l in range(self.lv_len):
                # lv_loss_brnchcls.append(loss_brnchcls_list[l][s] * lv_weight[l])
                # lv_loss_brnchreg.append(loss_brnchreg_list[l][s] * lv_weight[l])
                lv_loss_brnchcls.append(loss_brnchcls_list[l][s])
                lv_loss_brnchreg.append(loss_brnchreg_list[l][s])
            loss_brnchcls.append(sum(lv_loss_brnchcls))
            loss_brnchreg.append(sum(lv_loss_brnchreg))
        loss_brnchcls = sum(loss_brnchcls) / self.stacked_convs
        loss_brnchreg = sum(loss_brnchreg) / self.stacked_convs

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,

            loss_kdcls=loss_kdcls,
            loss_kdbbox=loss_kdbbox,

            loss_assis_cls=loss_assis_cls,
            loss_assis_bbox=loss_assis_bbox,
            loss_assis_dfl=loss_assis_dfl,

            loss_neck=loss_neck,
            loss_nkroi=loss_nkroi,
            loss_nkfg=loss_nkfg,
            loss_nkbg=loss_nkbg,

            loss_brcls=loss_brnchcls,
            loss_brreg=loss_brnchreg,
        )



