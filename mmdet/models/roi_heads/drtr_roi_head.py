# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmdet.core import bbox2result, bbox2roi
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmcv.runner import ModuleList


@HEADS.register_module()
class DRTRRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 mask_roi_extractor=None,
                 bbox_head=dict(
                     type='DRTRHead',
                     num_classes=80,
                     in_channels=2048,
                     transformer=None,
                     positional_encoding=None),
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(DRTRRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler.
        do nothing
        """
        self.bbox_assigner = None
        self.bbox_sampler = None

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""

        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            head.update(train_cfg=self.train_cfg, test_cfg=self.test_cfg)
            self.bbox_head.append(build_head(head))

        # self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        # bbox_head.update(train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        # self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None):

        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        proposal_feats = proposal_features
        all_stage_bbox_results = []
        all_stage_loss = {}

        is_first_stage = True
        encoder_memory = None
        decoder_memory = None
        for stage in range(self.num_stages):
            # forward single stage
            bbox_results = self._bbox_forward(stage,
                                              x,
                                              proposal_list,
                                              proposal_feats,
                                              is_first_stage,
                                              encoder_memory,
                                              decoder_memory,
                                              img_metas)
            all_stage_bbox_results.append(bbox_results)

            # loss bbox & cls
            outs = [bbox_results['cls_score'], bbox_results['bbox_pred']]
            if gt_labels is None:
                loss_inputs = outs + [gt_bboxes, img_metas]
            else:
                loss_inputs = outs + [proposal_list] + [gt_bboxes, gt_labels, imgs_whwh, img_metas]
            single_stage_loss = self.bbox_head[stage].loss(*loss_inputs,
                                                           gt_bboxes_ignore=gt_bboxes_ignore)

            for key, value in single_stage_loss.items():
                all_stage_loss[f'{key}_stage{stage}'] = value * \
                                    self.stage_loss_weights[stage]

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(x,
                                                        bbox_results['bbox_feats'],
                                                        gt_masks, img_metas)
                single_stage_loss['loss_mask'] = mask_results['loss_mask']

            # update proposal & proposal feat
            proposal_list = bbox_results['next_proposal_list']
            proposal_feats = bbox_results['next_proposal_feats']
            encoder_memory = bbox_results['next_encoder_memory']
            decoder_memory = bbox_results['next_decoder_memory']
            is_first_stage = False

        # all_stage_loss_sorted = sorted(all_stage_loss.items())
        # all_stage_loss_sorted_dict = {}
        #
        # for key, value in all_stage_loss_sorted:
        #     all_stage_loss_sorted_dict[f'{key}'] = value

        return all_stage_loss

    def _bbox_forward(self,
                      stage,
                      x,
                      proposal_list,
                      proposal_feats,
                      is_first_stage,
                      encoder_memory,
                      decoder_memory,
                      img_metas):
        """Run forward function and calculate loss for box head in training."""
        b, p, c = proposal_feats.shape

        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]

        info_rois = bbox2roi(proposal_list)
        roi_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                       info_rois)

        roi_feats = roi_feats.view(b*p, c, -1)
        # bbox_feats_tmp = bbox_feats.clone()
        ## mozaiq rois into single plane
        # p_all, c, r, r = roi_feats.shape
        # N = r * r
        # m = int(np.sqrt(p))
        #
        # roi_feats = roi_feats.view(b, p, c, r, r).permute(0, 2, 1, 3, 4).contiguous()
        # roi_feats = roi_feats.view(b, c, m, m, N)
        # roi_feats = torch.cat([roi_feats[..., s:s + r].contiguous().view(b, c, m, m * r)
        #                         for s in range(0, N, r)], dim=-1)
        # roi_feats = roi_feats.view(b, c, m * r, m * r)

        outs = bbox_head.forward(roi_feats,
                                 proposal_feats,
                                 is_first_stage,
                                 encoder_memory,
                                 decoder_memory)

        cls_score, bbox_pred, next_encoder_memory, next_decoder_memory = outs

        next_proposal_list = []
        for i in range(len(proposal_list)):
            next_proposal_list.append(
                bbox_head.bbox_coder.decode(
                    proposal_list[i], bbox_pred[i],
                    max_shape=img_metas[i]['img_shape']))

        # next_decoder_memory = next_decoder_memory.transpose(0, 1)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            next_proposal_list=[proposal.detach() for proposal in next_proposal_list],
            next_proposal_feats=next_decoder_memory[1].transpose(0, 1),
            next_encoder_memory=next_encoder_memory,
            next_decoder_memory=next_decoder_memory
        )
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_boxes,
                    proposal_feats,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if all([proposal.shape[0] == 0 for proposal in proposal_list]):
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for i in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs
            return bbox_results

        is_first_stage = True
        encoder_memory = None
        decoder_memory = None
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage,
                                              x,
                                              proposal_list,
                                              proposal_feats,
                                              is_first_stage,
                                              encoder_memory,
                                              decoder_memory,
                                              img_metas)

            # update proposal & proposal feat
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['next_proposal_list']
            proposal_feats = bbox_results['next_proposal_feats']
            encoder_memory = bbox_results['next_encoder_memory']
            decoder_memory = bbox_results['next_decoder_memory']
            is_first_stage = False

        if self.with_mask:
            rois = bbox2roi(proposal_list)
            mask_results = self._mask_forward(x, rois,
                                              bbox_results['attn_feats'])
            mask_results['mask_pred'] = mask_results['mask_pred'].reshape(
                num_imgs, -1, *mask_results['mask_pred'].size()[1:])

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_indices % num_classes
            bbox_pred_per_img = proposal_list[img_id][topk_indices //
                                                          num_classes]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            det_bboxes.append(
                torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
            det_labels.append(labels_per_img)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]

        if self.with_mask:
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            segm_results = []
            mask_pred = mask_results['mask_pred']
            for img_id in range(num_imgs):
                mask_pred_per_img = mask_pred[img_id].flatten(0,
                                                              1)[topk_indices]
                mask_pred_per_img = mask_pred_per_img[:, None, ...].repeat(
                    1, num_classes, 1, 1)
                segm_result = self.mask_head[-1].get_seg_masks(
                    mask_pred_per_img, _bboxes[img_id], det_labels[img_id],
                    self.test_cfg, ori_shapes[img_id], scale_factors[img_id],
                    rescale)
                segm_results.append(segm_result)

        if self.with_mask:
            results = list(zip(bbox_results, segm_results))
        else:
            results = bbox_results

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError()

    def forward_dummy(self, x, proposal_boxes, proposal_features, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        if self.with_bbox:
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(x, rois, object_feats,
                                              img_metas)

            all_stage_bbox_results.append((bbox_results, ))

            if self.with_mask:
                rois = bbox2roi(proposal_list)
                mask_results = self._mask_forward(
                    x, rois, bbox_results['attn_feats'])
                all_stage_bbox_results[-1] += (mask_results, )
        return all_stage_bbox_results


