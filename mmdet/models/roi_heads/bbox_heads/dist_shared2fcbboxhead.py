# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS,build_loss
from mmdet.models.utils import build_linear_layer
from mmdet.core import multi_apply

from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from .bbox_head import BBoxHead


class DistillRoI(nn.Module):
    def __init__(self,
                 student_channel,
                 teacher_channel,
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 w_kd=(1.0, 0.5)
                 ):
        super().__init__()
        self.student_channel = student_channel
        self.teacher_channel = teacher_channel

        self.mimic_conv_st_rois = nn.Conv2d(256,
                                            256,
                                            kernel_size=1,
                                            padding=0)
        self.mimic_conv_tch_rois = nn.Conv2d(256,
                                             256,
                                             kernel_size=1,
                                             padding=0)

        self.loss_mse = build_loss(loss_mse)
        self.cos_simm = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.w_kd = w_kd
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

    def forward(self,
                stfeat_stroi, stfeat_tchroi,
                tchfeat_stroi, tchfeat_tchroi):
        _stfeat_stroi = self.mimic_conv_st_rois(stfeat_stroi)
        _stfeat_tchroi = self.mimic_conv_tch_rois(stfeat_tchroi)

        loss_nkroist = self._loss_mse_cos(_stfeat_stroi,
                                          tchfeat_stroi,
                                          weight=self.w_kd)
        loss_nkroitch = self._loss_mse_cos(_stfeat_tchroi,
                                           tchfeat_tchroi,
                                           weight=self.w_kd)
        return loss_nkroist, loss_nkroitch


class DistillHead(nn.Module):
    def __init__(self,
                 num_stages,
                 student_channel,
                 teacher_channel,
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 w_kd=(1.0, 0.5)
                 ):
        super().__init__()
        self.num_stages = num_stages
        self.mimic_linear = nn.Linear(student_channel, teacher_channel)
        self.loss_mse = build_loss(loss_mse)
        self.cos_simm = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.w_kd = w_kd

    def _loss_mse_cos(self, student, teacher, weight, mask=None, avg_factor=None):
        teacher = teacher.detach()
        n, ch = student.shape

        cos_dist = self.cos_simm(student, teacher)

        cos_guidance = 1 - cos_dist
        cos_guidance = cos_guidance.abs()

        loss_cos = (1 - cos_dist).mean()

        loss_mse = self.loss_mse(student,
                                 teacher,
                                 weight=cos_guidance[:, None],
                                 avg_factor=avg_factor)

        loss = weight[0]*loss_mse + weight[1]*loss_cos
        return loss

    def forward(self, student, teacher):
        _student = self.mimic_linear(student)
        loss_feat = self._loss_mse_cos(_student,
                                       teacher,
                                       weight=self.w_kd)
        return loss_feat


class DistillPrediction(nn.Module):
    def __init__(self,
                 student_channel,
                 teacher_channel,
                 sigmoid=False,
                 mask=False,
                 avg_factor=False,
                 loss_mse=dict(type='MSELoss', loss_weight=1.0),
                 w_kd=(1.0, 0.5)
                 ):
        super().__init__()
        self.sigmoid = sigmoid
        self.mask = mask
        self.avg_factor = avg_factor
        self.student_channel = student_channel
        self.teacher_channel = teacher_channel

        self.loss_mse = build_loss(loss_mse)
        self.cos_simm = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.w_kd = w_kd

    def _loss_mse_cos(self, student, teacher, weight, mask=None, avg_factor=None):
        teacher = teacher.detach()
        n, ch = student.shape

        cos_dist = self.cos_simm(student, teacher)

        cos_guidance = 1 - cos_dist
        cos_guidance = cos_guidance.abs()

        loss_cos = (1 - cos_dist).mean()

        loss_mse = self.loss_mse(student,
                                 teacher,
                                 weight=cos_guidance[:, None],
                                 avg_factor=avg_factor)

        loss = weight[0]*loss_mse + weight[1]*loss_cos
        return loss

    def forward(self, student, teacher):

        if self.sigmoid:
            student = student.sigmoid()
            teacher = teacher.sigmoid()

        loss_kdpred = self._loss_mse_cos(student,
                                         teacher,
                                         weight=self.w_kd)
        return loss_kdpred


@HEADS.register_module()
class dist_Shared2FCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 is_teacher=False,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 loss_kdcls=dict(type='CrossEntropyLoss',
                                 use_sigmoid=True,
                                 loss_weight=1.0),
                 *args,
                 **kwargs):
        super(dist_Shared2FCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        self.is_teacher = is_teacher
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)

            if not is_teacher:
                self.assist_fc_cls = build_linear_layer(
                    self.cls_predictor_cfg,
                    in_features=self.cls_last_dim,
                    out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

            if not is_teacher:
                self.assist_fc_reg = build_linear_layer(
                    self.reg_predictor_cfg,
                    in_features=self.reg_last_dim,
                    out_features=out_dim_reg)

        if not is_teacher:
            self.distill_roi = DistillRoI(student_channel=256,
                                          teacher_channel=256,
                                          w_kd=(1.0, 0.5))

            self.distill_head = nn.ModuleList()
            self.now_indx = []
            for l in range(num_shared_fcs):
                self.now_indx.append(l)
                self.distill_head.append(DistillHead(num_stages=num_shared_fcs,
                                                     student_channel=fc_out_channels,
                                                     teacher_channel=fc_out_channels,
                                                     w_kd=(1.0, 1.0)))
            self.distill_pred_cls = DistillPrediction(student_channel=self.num_classes+1,
                                                      teacher_channel=self.num_classes+1,
                                                      sigmoid=True,
                                                      w_kd=(1.0, 1.0))
            self.distill_pred_bbox = DistillPrediction(student_channel=self.num_classes + 1,
                                                       teacher_channel=self.num_classes + 1,
                                                       sigmoid=False,
                                                       w_kd=(1.0, 1.0))

            self.loss_kdcls = build_loss(loss_kdcls)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

            if self.with_cls:
                self.init_cfg += [dict(type='Normal', std=0.01, override=dict(name='fc_cls'))]
            if self.with_reg:
                self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]

            if not is_teacher:
                if self.with_cls:
                    self.init_cfg += [dict(type='Normal', std=0.01, override=dict(name='assist_fc_cls'))]
                if self.with_reg:
                    self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='assist_fc_reg'))]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def loss_distill_head_single(self,
                                 now_indx,
                                 st_feats,
                                 tch_feats):
        loss_head = self.distill_head[now_indx](st_feats, tch_feats)
        return loss_head,

    def forward(self, x, only_original_path=True):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        head_feats = []
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
                if not only_original_path:
                    head_feats.append(x)

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if not self.is_teacher and self.training:
            if not only_original_path:
                assist_cls_score = self.assist_fc_cls(x_cls) if self.with_cls else None
                assist_bbox_pred = self.assist_fc_reg(x_reg) if self.with_reg else None
                return cls_score, bbox_pred, assist_cls_score, assist_bbox_pred, head_feats
            else:
                return cls_score, bbox_pred
        elif self.is_teacher:
            return cls_score, bbox_pred, head_feats
        else:
            return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             dist_cls_score,
             dist_bbox_pred,
             assist_cls_score,
             assist_bbox_pred,
             st_bbox_feats_stroi,
             st_bbox_feats_tchroi,
             st_head_feats,
             cls_soft_target,
             bbox_soft_target,
             tch_bbox_feats_stroi,
             tch_bbox_feats_tchroi,
             tch_head_feats,
             st_rois, tch_rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             dist_labels,
             dist_label_weights,
             dist_bbox_targets,
             dist_bbox_weights,
             reduction_override=None):
        losses = dict()

        loss_nkroi_st, loss_nkroi_tch = self.distill_roi(st_bbox_feats_stroi, st_bbox_feats_tchroi,
                                                         tch_bbox_feats_stroi, tch_bbox_feats_tchroi)
        losses['loss_nkroi_st'] = loss_nkroi_st
        losses['loss_nkroi_tch'] = loss_nkroi_tch

        loss_head_list = multi_apply(
            self.loss_distill_head_single,
            self.now_indx,
            st_head_feats,
            tch_head_feats)[0]
        losses['loss_headfeat'] = sum(loss_head_list) / self.num_shared_fcs

        loss_kdcls = self.distill_pred_cls(dist_cls_score, cls_soft_target)
        losses['loss_kdcls'] = loss_kdcls

        loss_kdbbox = self.distill_pred_bbox(dist_bbox_pred, bbox_soft_target)
        losses['loss_kdbbox'] = loss_kdbbox

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)

                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)

        if assist_cls_score is not None:
            if assist_cls_score.numel() > 0:
                dist_avg_factor = max(torch.sum(dist_label_weights > 0).float().item(), 1.)

                assist_loss_cls_ = self.loss_kdcls(
                    assist_cls_score,
                    # cls_soft_target.detach().softmax(-1).max(dim=1)[1],
                    cls_soft_target.detach().sigmoid(),
                    # dist_label_weights[:, None],
                    # avg_factor=dist_avg_factor,
                    reduction_override=reduction_override)

                if isinstance(assist_loss_cls_, dict):
                    losses.update(assist_loss_cls_)
                else:
                    losses['loss_assist_cls'] = assist_loss_cls_
                if self.custom_activation:
                    assist_acc_ = self.loss_cls.get_accuracy(assist_cls_score, dist_labels)
                    losses.update(assist_acc_)
                else:
                    losses['assist_acc'] = accuracy(assist_cls_score, dist_labels)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            dist_pos_inds = (dist_labels >= 0) & (dist_labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(st_rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]

                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)

            if dist_pos_inds.any():
                if self.reg_decoded_bbox:
                    assist_bbox_pred = self.bbox_coder.decode(tch_rois[:, 1:], assist_bbox_pred)
                    bbox_soft_target = self.bbox_coder.decode(tch_rois[:, 1:], bbox_soft_target)
                if self.reg_class_agnostic:
                    pos_assist_bbox_pred = assist_bbox_pred.view(assist_bbox_pred.size(0), 4)[dist_pos_inds.type(torch.bool)]
                    pos_bbox_soft_target = bbox_soft_target.view(bbox_soft_target.size(0), 4)[dist_pos_inds.type(torch.bool)]
                else:
                    pos_assist_bbox_pred = assist_bbox_pred.view(
                        assist_bbox_pred.size(0), -1,
                        4)[dist_pos_inds.type(torch.bool),
                    dist_labels[dist_pos_inds.type(torch.bool)]]

                    pos_bbox_soft_target = bbox_soft_target.view(
                        bbox_soft_target.size(0), -1,
                        4)[dist_pos_inds.type(torch.bool),
                    dist_labels[dist_pos_inds.type(torch.bool)]]

                losses['loss_assist_bbox'] = self.loss_bbox(
                    pos_assist_bbox_pred,
                    pos_bbox_soft_target,
                    dist_bbox_weights[dist_pos_inds.type(torch.bool)],
                    avg_factor=dist_bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
                losses['loss_assist_bbox'] = assist_bbox_pred[dist_pos_inds].sum()
        return losses



