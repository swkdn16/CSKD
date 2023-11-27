# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Linear, bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention, build_positional_encoding
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_transformer
from .bbox_head import BBoxHead


@HEADS.register_module()
class DualAttnHead(BBoxHead):
    r"""Dynamic Instance Interactive Head for `Sparse R-CNN: End-to-End Object
    Detection with Learnable Proposals <https://arxiv.org/abs/2011.12450>`_

    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_ffn_fcs (int): The number of fully-connected
            layers in FFNs. Defaults to 2.
        num_heads (int): The hidden dimension of FFNs.
            Defaults to 8.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
        feedforward_channels (int): The hidden dimension
            of FFNs. Defaults to 2048
        in_channels (int): Hidden_channels of MultiheadAttention.
            Defaults to 256.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        ffn_act_cfg (dict): The activation config for FFNs.
        dynamic_conv_cfg (dict): The convolution config
            for DynamicConv.
        loss_iou (dict): The config for iou or giou loss.

    """

    def __init__(self,
                 num_classes=80,
                 num_proposal=100,
                 roi_size=7,
                 num_roi_cbamlayer=6,
                 dynamic_cbam=True,
                 num_roi_trlayer=6,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(DualAttnHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.num_proposal = num_proposal
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.p_attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.p_attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        self.num_roi_cbamlayer = num_roi_cbamlayer
        self.dynamic_cbam = dynamic_cbam
        if dynamic_cbam:
            self.dym_ch_attn_params = nn.ModuleList()
            self.dym_sp_attn_params = nn.ModuleList()
        else:
            self.ch_attn_ffn = nn.ModuleList()
            self.sp_attn_conv = nn.ModuleList()

        for _ in range(self.num_roi_cbamlayer):
            if dynamic_cbam:
                self.dym_ch_attn_params.append(
                    nn.Sequential(nn.Linear(in_channels, in_channels),
                                  build_norm_layer(dict(type='LN'), in_channels)[1]))
                self.dym_sp_attn_params.append(
                    nn.Sequential(nn.Linear(in_channels, roi_size*roi_size),
                                  build_norm_layer(dict(type='LN'), roi_size*roi_size)[1]))
            else:
                self.ch_attn_ffn.append(
                    nn.Sequential(FFN(in_channels,
                                      feedforward_channels,
                                      2,
                                      dropout=0.0,
                                      add_residual=False),
                                  build_norm_layer(dict(type='LN'), (2, in_channels))[1]))
                self.sp_attn_conv.append(
                    nn.Conv2d(in_channels=2,
                              out_channels=1,
                              kernel_size=3,
                              padding=3 // 2))

        self.num_roi_trlayer = num_roi_trlayer
        self.roi_transformer = nn.ModuleList()
        for _ in range(self.num_roi_trlayer):
            self.roi_transformer.append(
                nn.Sequential(MultiheadAttention(in_channels, num_heads, dropout),
                              build_norm_layer(dict(type='LN'), in_channels)[1],
                              FFN(in_channels,
                                  2048,
                                  num_ffn_fcs,
                                  act_cfg=ffn_act_cfg,
                                  dropout=dropout),
                              build_norm_layer(dict(type='LN'), in_channels)[1]))

        self.roi_attn_proj = nn.Sequential(
            # build_norm_layer(dict(type='LN'), in_channels*roi_size*roi_size)[1],
            Linear(in_channels*roi_size*roi_size, in_channels),
            build_norm_layer(dict(type='LN'), in_channels)[1])

        self.obj_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(in_channels, 4)

        assert self.reg_class_agnostic, 'DIIHead only ' \
            'suppport `reg_class_agnostic=True` '
        assert self.reg_decoded_bbox, 'DIIHead only ' \
            'suppport `reg_decoded_bbox=True`'

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(DualAttnHead, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

    def global_average_pooling(self, feat):
        gap_mean = F.avg_pool2d(feat, feat.size()[2:])
        gap_mean = F.adaptive_avg_pool2d(gap_mean, (1, 1))
        gap_mean = torch.mean(gap_mean.view(gap_mean.size(0),
                                            gap_mean.size(1), -1), dim=2)
        return gap_mean

    def channel_attention(self, layer, r_feat, dym_feat=None):
        if dym_feat is not None:  # dynamic
            channel_attn = self.dym_ch_attn_params[layer](dym_feat).sigmoid()
        else:
            r_ch_attn_mn = F.avg_pool2d(r_feat, r_feat.size()[2:])
            r_ch_attn_mn = F.adaptive_avg_pool2d(r_ch_attn_mn, (1, 1))
            r_ch_attn_mn = r_ch_attn_mn.view(r_ch_attn_mn.size(0), r_ch_attn_mn.size(1), -1).permute(0, 2, 1)

            r_ch_attn_mx = F.max_pool2d(r_feat, r_feat.size()[2:])
            r_ch_attn_mx = F.adaptive_max_pool2d(r_ch_attn_mx, (1, 1))
            r_ch_attn_mx = r_ch_attn_mx.view(r_ch_attn_mx.size(0), r_ch_attn_mx.size(1), -1).permute(0, 2, 1)

            channel_attn = torch.cat([r_ch_attn_mn, r_ch_attn_mx], 1)
            channel_attn = self.ch_attn_ffn[layer](channel_attn)
            channel_attn = channel_attn.sum(1).sigmoid()

        channel_attn = channel_attn.unsqueeze(-1).unsqueeze(-1)
        return r_feat * channel_attn

    def spatial_attention(self, layer, r_feat, dym_feat=None):
        if dym_feat is not None:  # dynamic
            b, c, h, w = r_feat.shape
            spatial_attn = self.dym_sp_attn_params[layer](dym_feat).sigmoid()
            spatial_attn = spatial_attn.view(b, 1, h, w)
        else:
            spatial_attn_mean = r_feat.mean(1)
            spatial_attn_max = r_feat.max(1)[0]

            spatial_attn = torch.stack([spatial_attn_mean, spatial_attn_max], 1)

            spatial_attn = self.sp_attn_conv[layer](spatial_attn)
            spatial_attn = spatial_attn.sigmoid()
        return r_feat * spatial_attn

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                      all proposals, has shape
                      (batch_size, num_proposals, num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                      all proposals, has shape
                      (batch_size, num_proposals, 4).
                    - obj_feat (Tensor): Object feature before classification
                      and regression subnet, has shape
                      (batch_size, num_proposal, feature_dimensions).
        """
        N, num_proposals = proposal_feat.shape[:2]
        roi_size = roi_feat.size(-1)
        # Proposal Self attention
        proposal_feat = proposal_feat.permute(1, 0, 2)
        p_attn_feats = self.p_attention_norm(self.p_attention(proposal_feat))
        p_attn_feats = p_attn_feats.permute(1, 0, 2)

        roi_f_attn = roi_feat
        roi_tr_attn = roi_feat

        p_attn_feats = p_attn_feats.reshape(-1, self.in_channels)
        for layer in range(self.num_roi_cbamlayer):
            # roi_feat channel-attention
            dym_feat = p_attn_feats if self.dynamic_cbam else None
            identity = roi_f_attn
            roi_feat_ch_attn = self.channel_attention(layer, roi_f_attn, dym_feat)
            roi_feat_sp_attn = self.spatial_attention(layer, roi_feat_ch_attn, dym_feat)
            roi_f_attn = identity + roi_feat_sp_attn

        # self-attn per feat
        # position encoding
        mask = roi_feat.new_zeros((N * num_proposals, roi_size, roi_size)).to(torch.bool)
        loc_roi_pos = self.positional_encoding(mask).view(N*num_proposals, self.in_channels, -1)
        glob_roi_pos = p_attn_feats.view(N*num_proposals, self.in_channels, 1)
        pos_embed = loc_roi_pos + glob_roi_pos
        pos_embed = pos_embed.permute(2, 0, 1)
        roi_tr_attn = roi_tr_attn.view(N * num_proposals, self.in_channels, -1).permute(2, 0, 1)
        for layer in range(self.num_roi_trlayer):
            roi_tr_attn = self.roi_transformer[layer][0](query=roi_tr_attn,
                                                         key=None,
                                                         value=None,
                                                         query_pos=pos_embed,
                                                         query_key_padding_mask=mask)
            roi_tr_attn = self.roi_transformer[layer][1:](roi_tr_attn)

        roi_tr_attn = roi_tr_attn.permute(1, 2, 0).reshape(N * num_proposals, -1)
        roi_f_attn = roi_f_attn.flatten(2).reshape(N * num_proposals, -1)  # [N*num_proposals, 7*7*256]
        roi_attn = self.roi_attn_proj(roi_tr_attn+roi_f_attn)

        obj_feat = self.obj_norm(p_attn_feats + roi_attn)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat
        reg_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).view(
            N, num_proposals, self.num_classes
            if self.loss_cls.use_sigmoid else self.num_classes + 1)
        bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, 4)

        return cls_score, bbox_delta, obj_feat.view(
            N, num_proposals, self.in_channels), obj_feat

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        """"Loss function of DIIHead, get loss of all images.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            labels (Tensor): Label of each proposals, has shape
                (batch_size * num_proposals_single_image
            label_weights (Tensor): Classification loss
                weight of each proposals, has shape
                (batch_size * num_proposals_single_image
            bbox_targets (Tensor): Regression targets of each
                proposals, has shape
                (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression loss weight of each
                proposals's coordinate, has shape
                (batch_size * num_proposals_single_image, 4),
            imgs_whwh (Tensor): imgs_whwh (Tensor): Tensor with\
                shape (batch_size, num_proposals, 4), the last
                dimension means
                [img_width,img_height, img_width, img_height].
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

            Returns:
                dict[str, Tensor]: Dictionary of loss components
        """
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                              4)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all proposals, has
                  shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all proposals, has
                  shape (num_proposals, 4), the last dimension 4
                  represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all proposals,
                  has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise just
                  a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals,) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list has
                  shape (num_proposals, 4) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals, 4),
                  the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
