# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class dist_StandardRoIHead(StandardRoIHead):
    def __init__(self,
                 is_teacher=False,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(dist_StandardRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.is_teacher = is_teacher

    def _bbox_forward_dist(self, x, st_rois, tch_rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats_stroi = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], st_rois)
        bbox_feats_tchroi = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], tch_rois)

        if self.with_shared_head:
            bbox_feats_stroi = self.shared_head(bbox_feats_stroi)

        if not self.is_teacher and self.training:
            cls_score, bbox_pred = self.bbox_head(bbox_feats_stroi)
            dist_cls_score, dist_bbox_pred, assist_cls_score, assist_bbox_pred, head_feats = self.bbox_head(bbox_feats_tchroi,
                                                                                                            only_original_path=False)
            bbox_results = dict(
                cls_score=cls_score,
                bbox_pred=bbox_pred,
                dist_cls_score=dist_cls_score,
                dist_bbox_pred=dist_bbox_pred,
                assist_cls_score=assist_cls_score,
                assist_bbox_pred=assist_bbox_pred,
                bbox_feats_stroi=bbox_feats_stroi,
                bbox_feats_tchroi=bbox_feats_tchroi,
                head_feats=head_feats)
        else:
            cls_score, bbox_pred, head_feats = self.bbox_head(bbox_feats_tchroi, only_original_path=False)
            bbox_results = dict(
                cls_score=cls_score,
                bbox_pred=bbox_pred,
                bbox_feats_stroi=bbox_feats_stroi,
                bbox_feats_tchroi=bbox_feats_tchroi,
                head_feats=head_feats)
        return bbox_results