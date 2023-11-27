# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import Scale
import torch.nn.functional as F
from torchvision.ops import RoIAlign

from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps, multi_apply, reduce_mean
from ..builder import HEADS, build_loss
from .dist_fcos_head import dist_FCOSHead


class DistillNeck(nn.Module):
    def __init__(self,
                 student_channel,
                 teacher_channel,
                 use_conv_for_mimic=True,
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 w_neck=((5.0, 2.0), (1.0, 2.0), (1.0, 2.0))
                 ):
        super().__init__()
        self.student_channel = student_channel
        self.teacher_channel = teacher_channel
        if use_conv_for_mimic:
            self.mimic_conv = nn.Conv2d(student_channel,
                                        teacher_channel,
                                        kernel_size=1,
                                        padding=0)
        else:
            self.mimic_conv = nn.Identity()
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
                 use_conv_for_mimic=True,
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 w_branch=(0.5, 0.2)
                 ):
        super().__init__()
        self.num_stages = num_stages
        if use_conv_for_mimic:
            self.mimic_conv = nn.ModuleList([nn.Conv2d(student_channel,
                                                       teacher_channel,
                                                       kernel_size=1,
                                                       padding=0)
                                             for _ in range(num_stages)])
        else:
            self.mimic_conv = nn.ModuleList([nn.Identity() for _ in range(num_stages)])
        self.loss_mse = build_loss(loss_mse)
        self.cos_simm = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.w_branch = w_branch

    def _loss_mse_cos(self, student, teacher, weight, mask=None, avg_factor=None):
        teacher = teacher.detach()
        b, c, h, w = student.shape

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
        loss_feat = []
        for i in range(self.num_stages):
            _student = self.mimic_conv[i](student[i])
            loss_feat.append(self._loss_mse_cos(_student, teacher[i], weight=self.w_branch))
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
class CSKDHeadFCOS(dist_FCOSHead):
    """Localization distillation Head.
    """

    def __init__(self,
                 loss_kdcls=dict(
                     type='pyFocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 use_conv_for_mimic=True,
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
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01),
                      dict(
                         type='Normal',
                         name='assis_conv_cls',
                         std=0.01,
                         bias_prob=0.01)]),
                 **kwargs):

        super(CSKDHeadFCOS, self).__init__(
            init_cfg=init_cfg,
            **kwargs)
        self.loss_kdcls = build_loss(loss_kdcls)
        self.cos_simm = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.loss_mse = build_loss(dict(type='MSELoss', loss_weight=1.0))

        self.kd_neck_weight = kd_neck_weight
        self.kd_brnch_weight = kd_brnch_weight
        self.kd_pred_weight = kd_pred_weight

        self.lv_len = 5
        self.now_indx = []
        self.distill_neck = nn.ModuleList()
        self.distill_brnch_reg = nn.ModuleList()
        self.distill_brnch_cls = nn.ModuleList()
        self.distill_pred_bbox = nn.ModuleList()
        self.distill_pred_cls = nn.ModuleList()
        self.distill_pred_cnt = nn.ModuleList()
        for l in range(self.lv_len):
            self.now_indx.append(l)
            self.distill_neck.append(DistillNeck(student_channel=256,
                                                 teacher_channel=256,
                                                 use_conv_for_mimic=use_conv_for_mimic,
                                                 w_neck=self.kd_neck_weight))
            self.distill_brnch_reg.append(Distillbranch(num_stages=self.stacked_convs,
                                                        student_channel=256,
                                                        teacher_channel=256,
                                                        use_conv_for_mimic=use_conv_for_mimic,
                                                        w_branch=self.kd_brnch_weight[0]))
            self.distill_brnch_cls.append(Distillbranch(num_stages=self.stacked_convs,
                                                        student_channel=256,
                                                        teacher_channel=256,
                                                        use_conv_for_mimic=use_conv_for_mimic,
                                                        w_branch=self.kd_brnch_weight[1]))
            self.distill_pred_bbox.append(Distillprediction(student_channel=4,
                                                            teacher_channel=4,
                                                            sigmoid=False,
                                                            mask=False,
                                                            avg_factor=True,
                                                            w_pred=self.kd_pred_weight))
            self.distill_pred_cls.append(Distillprediction(student_channel=self.num_classes,
                                                           teacher_channel=self.num_classes,
                                                           sigmoid=True,
                                                           mask=False,
                                                           avg_factor=False,
                                                           w_pred=self.kd_pred_weight))
            self.distill_pred_cnt.append(Distillprediction(student_channel=1,
                                                           teacher_channel=1,
                                                           sigmoid=True,
                                                           mask=False,
                                                           avg_factor=True,
                                                           w_pred=self.kd_pred_weight))
        self._init_assistant_predictor()

    def _init_assistant_predictor(self):
        """Initialize predictor layers of the head."""
        self.assis_conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.assis_conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.assis_conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.assis_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

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
        outs = out_student[:3]

        st_cls_conv_feats = out_student[3]
        st_reg_conv_feats = out_student[4]
        st_cls_feat = out_student[5]
        st_reg_feat = out_student[6]

        assis_cls_score = []
        assis_bbox_pred = []
        assis_centerness = []
        for l in range(self.lv_len):
            assis_cls_score.append(self.assis_conv_cls(st_cls_feat[l]))
            _assis_bbox_pred = self.assis_conv_reg(st_reg_feat[l])

            if self.centerness_on_reg:
                assis_centerness.append(self.assis_conv_centerness(st_reg_feat[l]))
            else:
                assis_centerness.append(self.assis_conv_centerness(st_cls_feat[l]))

            _assis_bbox_pred = self.assis_scales[l](_assis_bbox_pred).float()
            if self.norm_on_bbox:
                _assis_bbox_pred = _assis_bbox_pred.clamp(min=0)
            else:
                _assis_bbox_pred = _assis_bbox_pred.exp()
            assis_bbox_pred.append(_assis_bbox_pred)

        cls_soft_labels = out_teacher[0]
        bbox_soft_targets = out_teacher[1]
        cnt_soft_targets = out_teacher[2]
        tch_cls_conv_feats = out_teacher[3]
        tch_reg_conv_feats = out_teacher[4]

        if gt_labels is None:
            loss_inputs = outs + (student_x, teacher_x, gt_bboxes,
                                  assis_cls_score,
                                  assis_bbox_pred,
                                  assis_centerness,
                                  cls_soft_labels,
                                  bbox_soft_targets,
                                  cnt_soft_targets,
                                  st_cls_conv_feats, st_reg_conv_feats,
                                  tch_cls_conv_feats, tch_reg_conv_feats,
                                  img_metas)
        else:
            loss_inputs = outs + (student_x, teacher_x, gt_bboxes, gt_labels,
                                  assis_cls_score,
                                  assis_bbox_pred,
                                  assis_centerness,
                                  cls_soft_labels,
                                  bbox_soft_targets,
                                  cnt_soft_targets,
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
        loss_neck, loss_roi, loss_fg, loss_bg = self.distill_neck[now_indx](student_x, teacher_x,
                                                                            gt_bboxes, img_metas)
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
                                       st_pred_cnt,
                                       tch_pred_cls,
                                       tch_pred_bbox,
                                       tch_pred_cnt,
                                       gt_bboxes, img_metas
                                       ):
        loss_kdcls = self.distill_pred_cls[now_indx](st_pred_cls,
                                                     tch_pred_cls,
                                                     gt_bboxes, img_metas)
        loss_kdbbox = self.distill_pred_bbox[now_indx](st_pred_bbox,
                                                       tch_pred_bbox,
                                                       gt_bboxes, img_metas)
        loss_kdcnt = self.distill_pred_cnt[now_indx](st_pred_cnt,
                                                     tch_pred_cnt,
                                                     gt_bboxes, img_metas)
        return loss_kdcls, loss_kdbbox, loss_kdcnt

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

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             student_x,
             teacher_x,
             gt_bboxes,
             gt_labels,
             assis_cls_scores,
             assis_bbox_preds,
             assis_centernesses,
             cls_soft_labels,
             bbox_soft_targets,
             cnt_soft_targets,
             st_cls_conv_feats, st_reg_conv_feats,
             tch_cls_conv_feats, tch_reg_conv_feats,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        num_imgs = bbox_preds[0].size(0)
        device = cls_scores[0].device

        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=device)

        labels, bbox_targets = self.get_targets(all_level_points,
                                                gt_bboxes, gt_labels)

        mkfg_list = []
        for lv in range(self.lv_len):
            b, c, h, w = student_x[lv].shape
            device = student_x[lv].device
            mask_fg, _ = self._get_fg_bg_mask([b, h, w, device], gt_bboxes, img_metas, scale=False)
            mkfg_list.append(mask_fg)

        loss_kdcls, loss_kdbbox, loss_kdcnt = multi_apply(
            self.loss_distill_prediction_single,
            self.now_indx,
            cls_scores,
            bbox_preds,
            centernesses,
            cls_soft_labels,
            bbox_soft_targets,
            cnt_soft_targets,
            gt_bboxes=gt_bboxes,
            img_metas=img_metas)
        # loss_kdcls = sum(loss_kdcls_list) / self.lv_len
        # loss_kdbbox = sum(loss_kdbbox_list) / self.lv_len
        # loss_kdcnt = sum(loss_kdcnt_list) / self.lv_len

        flatten_mkfg = [
            mkfg[:, None, :, :].permute(0, 2, 3, 1).reshape(-1)
            for mkfg in mkfg_list]
        flatten_mkfg = torch.cat(flatten_mkfg)

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses]

        flatten_assis_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in assis_cls_scores]
        flatten_assis_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in assis_bbox_preds]
        flatten_assis_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in assis_centernesses]

        flatten_cls_soft_labels = [
            cls_soft_label.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_soft_label in cls_soft_labels]
        flatten_bbox_soft_targets = [
            bbox_soft_target.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_soft_target in bbox_soft_targets]
        flatten_cnt_soft_targets = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in cnt_soft_targets]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        flatten_assis_cls_scores = torch.cat(flatten_assis_cls_scores)
        flatten_assis_bbox_preds = torch.cat(flatten_assis_bbox_preds)
        flatten_assis_centerness = torch.cat(flatten_assis_centerness)

        flatten_cls_soft_labels = torch.cat(flatten_cls_soft_labels)
        flatten_bbox_soft_targets = torch.cat(flatten_bbox_soft_targets)
        flatten_cnt_soft_targets = torch.cat(flatten_cnt_soft_targets)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)

        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        loss_cls = self.loss_cls(flatten_cls_scores,
                                 flatten_labels,
                                 avg_factor=num_pos)

        # flatten_cls_soft_labels = flatten_cls_soft_labels.detach().sigmoid() * \
        #                           flatten_cnt_soft_targets.sigmoid()[:, None]
        flatten_cls_soft_labels = flatten_cls_soft_labels.detach().sigmoid() * \
                                  flatten_mkfg[:, None]

        loss_assis_cls = self.loss_kdcls(flatten_assis_cls_scores,
                                         flatten_cls_soft_labels,
                                         avg_factor=num_pos)
                                         # avg_factor=flatten_mkfg.sum())

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_assis_bbox_preds = flatten_assis_bbox_preds[pos_inds]
        pos_bbox_soft_targets = flatten_bbox_soft_targets[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]

        pos_centerness = flatten_centerness[pos_inds]
        pos_assis_centerness = flatten_assis_centerness[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        pos_centerness_soft_targets = flatten_cnt_soft_targets[pos_inds]
        pos_centerness_soft_targets = pos_centerness_soft_targets.detach()
        # centerness weighted iou loss
        centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
        # centerness_soft_denorm = max(reduce_mean(pos_centerness_soft_targets.sigmoid().sum()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_assis_bbox_preds = self.bbox_coder.decode(pos_points, pos_assis_bbox_preds)
            pos_decoded_soft_target = self.bbox_coder.decode(pos_points, pos_bbox_soft_targets)
            pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)

            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)

            loss_assis_bbox = self.loss_bbox(
                pos_decoded_assis_bbox_preds,
                pos_decoded_soft_target.detach(),
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)

            loss_centerness = self.loss_centerness(
                pos_centerness,
                pos_centerness_targets,
                avg_factor=num_pos)

            loss_assis_centerness = self.loss_centerness(
                pos_assis_centerness,
                pos_centerness_soft_targets.sigmoid(),
                avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_assis_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_assis_centerness = pos_centerness.sum()

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

        loss_brnchcls = []
        loss_brnchreg = []
        for s in range(self.stacked_convs):
            lv_loss_brnchcls = []
            lv_loss_brnchreg = []
            for l in range(self.lv_len):
                lv_loss_brnchcls.append(loss_brnchcls_list[l][s])
                lv_loss_brnchreg.append(loss_brnchreg_list[l][s])
            loss_brnchcls.append(sum(lv_loss_brnchcls))
            loss_brnchreg.append(sum(lv_loss_brnchreg))
        loss_brnchcls = sum(loss_brnchcls) / self.stacked_convs
        loss_brnchreg = sum(loss_brnchreg) / self.stacked_convs

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,

            loss_kdcls=loss_kdcls,
            loss_kdbbox=loss_kdbbox,
            loss_kdcenterness=loss_kdcnt,

            loss_assis_cls=loss_assis_cls,
            loss_assis_bbox=loss_assis_bbox,
            loss_assis_centerness=loss_assis_centerness,

            loss_neck=loss_neck,
            loss_roi=loss_nkroi,
            loss_nfg=loss_nkfg,
            loss_nbg=loss_nkbg,

            loss_brnchcls=loss_brnchcls,
            loss_brnchreg=loss_brnchreg,
        )

