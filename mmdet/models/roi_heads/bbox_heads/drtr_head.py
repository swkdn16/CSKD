import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_bbox_coder,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models.builder import HEADS, build_loss
from .bbox_head import BBoxHead


@HEADS.register_module()
class DRTRHead(BBoxHead):
    def __init__(self,
                 num_classes=80,
                 in_channels=256,
                 cls_fcs_channels=1024,
                 num_query=100,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 roi_size=5,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=[dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                     dict(
                         type='SinePositionalEncoding1D',
                         num_feats=128,
                         normalize=True)],
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(DRTRHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg)

        self.bg_cls_weight = 1.0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DRTRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg is not None:
            self.assigner = build_assigner(train_cfg.assigner)
        # DETR sampling=False, so use PseudoSampler
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_fcs_channels = cls_fcs_channels
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs
        self.roi_size = roi_size
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.positional_encoding = build_positional_encoding(
            positional_encoding[0])
        self.positional_encoding1D = build_positional_encoding(
            positional_encoding[1])
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding[0]
        num_feats = positional_encoding[0]['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.cls_fcs = nn.ModuleList()
        for _ in range(self.num_cls_fcs):
            self.cls_fcs.append(nn.Linear(self.embed_dims*self.roi_size*self.roi_size,
                                          self.cls_fcs_channels, bias=False))
            self.cls_fcs.append(build_norm_layer(dict(type='LN'), self.cls_fcs_channels)[1])
            self.cls_fcs.append(build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(self.cls_fcs_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(self.cls_fcs_channels, self.num_classes + 1)

        self.reg_fcs = nn.ModuleList()
        for _ in range(self.num_reg_fcs):
            self.reg_fcs.append(nn.Linear(self.embed_dims, self.embed_dims, bias=False))
            self.reg_fcs.append(build_norm_layer(dict(type='LN'), self.embed_dims)[1])
            self.reg_fcs.append(build_activation_layer(dict(type='ReLU', inplace=True)))

        self.fc_reg = Linear(self.embed_dims, 4)

        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    def forward(self,
                roi_feats,
                proposal_feats,
                is_first_stage,
                encoder_memory,
                decoder_memory):
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        bp, c, f = roi_feats.shape
        b, p, c = proposal_feats.shape

        h = w = torch.sqrt(torch.tensor(f)).to(torch.int)
        r_masks = roi_feats.new_zeros((bp, h, w)).to(torch.bool)
        p_masks = proposal_feats.new_zeros((b, p)).to(torch.bool)

        # position encoding
        p_pos = self.positional_encoding1D(p_masks)
        r_pos = self.positional_encoding(r_masks)# [bs, embed_dim, h, w]
        # r_pos = p_pos.repeat_interleave(9, dim=2)
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        # proposal_feats = proposal_feats.transpose(0, 1)
        # proposal_feats = proposal_feats.reshape(1, bp, c)
        cls_out_enc, bbox_out_enc, cls_out_dec, bbox_out_dec = self.transformer(
            roi_feats, proposal_feats,
            r_masks, p_masks,
            r_pos, p_pos,
            is_first_stage,
            encoder_memory,
            decoder_memory)

        cls_feat = cls_out_dec.transpose(0, 1).reshape(b, p, -1)
        reg_feat = bbox_out_dec.transpose(0, 1)

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat)
        bbox_preds = self.fc_reg(reg_feat)

        return cls_score, bbox_preds, [cls_out_enc, bbox_out_enc], [cls_out_dec, bbox_out_dec]

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             proposals_list,
             gt_bboxes_list,
             gt_labels_list,
             imgs_whwh,
             img_metas,
             gt_bboxes_ignore=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i].detach() for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i].detach() for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           proposals_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, imgs_whwh,
                                           gt_bboxes_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        loss_dict = dict()

        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=cls_avg_factor)

        loss_dict['loss_cls'] = loss_cls

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_pos = pos_inds.sum().float()
        num_total_pos = reduce_mean(num_pos)

        # # construct factors used for rescale bboxes
        # factors = []
        # for img_meta, bbox_pred in zip(img_metas, bbox_preds):
        #     img_h, img_w, _ = img_meta['img_shape']
        #     factor = bbox_pred.new_tensor([img_w, img_h, img_w,
        #                                    img_h]).unsqueeze(0).repeat(
        #         bbox_pred.size(0), 1)
        #     factors.append(factor)
        # factors = torch.cat(factors, 0)

        if pos_inds.any():
            proposals = torch.stack(proposals_list)
            max_shape = imgs_whwh.squeeze(1)[:, :2]
            # bbox_targets_encoded = bbox_targets.reshape(num_imgs, -1, 4)

            bbox_preds_decoded = torch.cat([self.bbox_coder.decode(proposals[i], bbox_preds[i], max_shape=max_shape[i])
                                            for i in range(num_imgs)])
            # bbox_targets_decoded = torch.cat([self.bbox_coder.decode(proposals[i], bbox_targets_encoded[i], max_shape=max_shape[i])
            #                                   for i in range(num_imgs)])
            # bbox_preds = bbox_preds.reshape(-1, 4)
            # regression IoU loss, defaultly GIoU loss
            loss_iou = self.loss_iou(
                bbox_preds_decoded,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_pos)
            # regression L1 loss
            loss_bbox = self.loss_bbox(
                bbox_preds_decoded / imgs_whwh,
                bbox_targets / imgs_whwh,
                bbox_weights,
                avg_factor=num_total_pos)
            loss_dict['loss_bbox'] = loss_bbox
            loss_dict['loss_iou'] = loss_iou
        else:
            loss_dict['loss_bbox'] = bbox_preds.sum() * 0
            loss_dict['loss_iou'] = bbox_preds.sum() * 0

        return loss_dict

    # def loss_single(self,
    #                 cls_scores,
    #                 bbox_preds,
    #                 proposals_list,
    #                 gt_bboxes_list,
    #                 gt_labels_list,
    #                 img_metas,
    #                 gt_bboxes_ignore_list=None):
    #     """"Loss function for outputs from a single decoder layer of a single
    #     feature level.
    #
    #     Args:
    #         cls_scores (Tensor): Box score logits from a single decoder layer
    #             for all images. Shape [bs, num_query, cls_out_channels].
    #         bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
    #             for all images, with normalized coordinate (cx, cy, w, h) and
    #             shape [bs, num_query, 4].
    #         gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
    #             with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels_list (list[Tensor]): Ground truth class indices for each
    #             image with shape (num_gts, ).
    #         img_metas (list[dict]): List of image meta information.
    #         gt_bboxes_ignore_list (list[Tensor], optional): Bounding
    #             boxes which can be ignored for each image. Default None.
    #
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components for outputs from
    #             a single decoder layer.
    #     """
    #     num_imgs = cls_scores.size(0)
    #     cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    #     bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    #
    #     cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
    #                                        proposals_list,
    #                                        gt_bboxes_list, gt_labels_list,
    #                                        img_metas, gt_bboxes_ignore_list)
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      num_total_pos, num_total_neg) = cls_reg_targets
    #     labels = torch.cat(labels_list, 0)
    #     label_weights = torch.cat(label_weights_list, 0)
    #     bbox_targets = torch.cat(bbox_targets_list, 0)
    #     bbox_weights = torch.cat(bbox_weights_list, 0)
    #
    #     # classification loss
    #     cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
    #     # construct weighted avg_factor to match with the official DETR repo
    #     cls_avg_factor = num_total_pos * 1.0 + \
    #                      num_total_neg * self.bg_cls_weight
    #     if self.sync_cls_avg_factor:
    #         cls_avg_factor = reduce_mean(
    #             cls_scores.new_tensor([cls_avg_factor]))
    #     cls_avg_factor = max(cls_avg_factor, 1)
    #
    #     loss_cls = self.loss_cls(
    #         cls_scores,
    #         labels,
    #         label_weights,
    #         avg_factor=cls_avg_factor)
    #
    #     # Compute the average number of gt boxes across all gpus, for
    #     # normalization purposes
    #     num_total_pos = loss_cls.new_tensor([num_total_pos])
    #     num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
    #
    #     # # construct factors used for rescale bboxes
    #     # factors = []
    #     # for img_meta, bbox_pred in zip(img_metas, bbox_preds):
    #     #     img_h, img_w, _ = img_meta['img_shape']
    #     #     factor = bbox_pred.new_tensor([img_w, img_h, img_w,
    #     #                                    img_h]).unsqueeze(0).repeat(
    #     #         bbox_pred.size(0), 1)
    #     #     factors.append(factor)
    #     # factors = torch.cat(factors, 0)
    #
    #     bbox_pred_decoded = []
    #     bboxes_target_decoded = []
    #     for i in range(len(proposals_list)):
    #         bbox_pred_decoded.append(
    #             self.bbox_coder.decode(proposals_list[i], bbox_preds_list[i],
    #                                    max_shape=img_metas[i]['img_shape']))
    #         bboxes_target_decoded.append(
    #             self.bbox_coder.decode(proposals_list[i], bbox_targets_list[i],
    #                                    max_shape=img_metas[i]['img_shape']))
    #
    #     bbox_pred_decoded = torch.cat(bbox_pred_decoded)
    #     bboxes_target_decoded = torch.cat(bboxes_target_decoded)
    #     # bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
    #     # bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
    #
    #     # regression IoU loss, defaultly GIoU loss
    #     loss_iou = self.loss_iou(
    #         bbox_pred_decoded,
    #         bboxes_target_decoded,
    #         bbox_weights,
    #         avg_factor=num_total_pos)
    #
    #     # regression L1 loss
    #     bbox_preds = bbox_preds.view(-1, 4)
    #     loss_bbox = self.loss_bbox(
    #         bbox_preds,
    #         bbox_targets,
    #         bbox_weights,
    #         avg_factor=num_total_pos)
    #     return loss_cls, loss_bbox, loss_iou

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    proposals_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas, imgs_whwh,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, proposals_list,
            gt_bboxes_list, gt_labels_list, img_metas, imgs_whwh, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           proposal,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           imgs_whwh,
                           gt_bboxes_ignore=None):

        num_bboxes = bbox_pred.size(0)
        proposal_update = self.bbox_coder.decode(proposal, bbox_pred,
                                                 max_shape=img_meta['img_shape'])
        normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_update / imgs_whwh)

        # assigner and sampler
        assign_result = self.assigner.assign(normalize_bbox_ccwh, cls_score,
                                             gt_bboxes, gt_labels,
                                             img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, proposal_update,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        label_weights[pos_inds] = 1.0
        label_weights[neg_inds] = 1.0
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pos_gt_bboxes_targets = self.bbox_coder.encode(proposal[pos_inds],
        #                                                sampling_result.pos_gt_bboxes)
        # bbox_targets[pos_inds] = pos_gt_bboxes_targets
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      bbox_feats,
                      object_feats,
                      proposals_list,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """

        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(bbox_feats, object_feats, proposals_list, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return outs, losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores_list,
                   all_bbox_preds_list,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def forward_onnx(self, feats, img_metas):
        """Forward function for exporting to ONNX.

        Over-write `forward` because: `masks` is directly created with
        zero (valid position tag) and has the same spatial size as `x`.
        Thus the construction of `masks` is different from that in `forward`.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single_onnx, feats, img_metas_list)

    def forward_single_onnx(self, x, img_metas):
        """"Forward function for a single feature level with ONNX exportation.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # Note `img_shape` is not dynamically traceable to ONNX,
        # since the related augmentation was done with numpy under
        # CPU. Thus `masks` is directly created with zeros (valid tag)
        # and the same spatial shape as `x`.
        # The difference between torch and exported ONNX model may be
        # ignored, since the same performance is achieved (e.g.
        # 40.1 vs 40.1 for DETR)
        batch_size = x.size(0)
        h, w = x.size()[-2:]
        masks = x.new_zeros((batch_size, h, w))  # [B,h,w]

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.positional_encoding(masks)
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    def onnx_export(self, all_cls_scores_list, all_bbox_preds_list, img_metas):
        """Transform network outputs into bbox predictions, with ONNX
        exportation.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        assert len(img_metas) == 1, \
            'Only support one input image while in exporting to ONNX'

        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        # Note `img_shape` is not dynamically traceable to ONNX,
        # here `img_shape_for_onnx` (padded shape of image tensor)
        # is used.
        img_shape = img_metas[0]['img_shape_for_onnx']
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        batch_size = cls_scores.size(0)
        # `batch_index_offset` is used for the gather of concatenated tensor
        batch_index_offset = torch.arange(batch_size).to(
            cls_scores.device) * max_per_img
        batch_index_offset = batch_index_offset.unsqueeze(1).expand(
            batch_size, max_per_img)

        # supports dynamical batch inference
        if self.loss_cls.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
            scores, indexes = cls_scores.view(batch_size, -1).topk(
                max_per_img, dim=1)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
        else:
            scores, det_labels = F.softmax(
                cls_scores, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img, dim=1)
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            det_labels = det_labels.view(-1)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
            det_labels = det_labels.view(batch_size, -1)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_preds)
        # use `img_shape_tensor` for dynamically exporting to ONNX
        img_shape_tensor = img_shape.flip(0).repeat(2)  # [w,h,w,h]
        img_shape_tensor = img_shape_tensor.unsqueeze(0).unsqueeze(0).expand(
            batch_size, det_bboxes.size(1), 4)
        det_bboxes = det_bboxes * img_shape_tensor
        # dynamically clip bboxes
        x1, y1, x2, y2 = det_bboxes.split((1, 1, 1, 1), dim=-1)
        from mmdet.core.export import dynamic_clip_for_onnx
        x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, img_shape)
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=-1)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)

        return det_bboxes, det_labels