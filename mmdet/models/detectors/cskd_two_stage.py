# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from mmdet.core import bbox2roi
import warnings

import torch

from .. import build_detector
from .two_stage import TwoStageDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from collections import OrderedDict


@DETECTORS.register_module()
class CSKDTwoStageDetector(TwoStageDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 init_student=False,
                 teacher_config=None,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CSKDTwoStageDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        if train_cfg is not None:
            self.eval_teacher = eval_teacher

            # Build teacher model
            if isinstance(teacher_config, str):
                teacher_config = mmcv.Config.fromfile(teacher_config)
                # teacher_config.model.neck['start_level'] = 1
                # teacher_config.model.neck['add_extra_convs'] = 'on_input'
                teacher_config.model.rpn_head.type = teacher_config.model.rpn_head.type.replace(
                    'RPNHead', 'dist_RPNHead')
                teacher_config.model.roi_head.type = teacher_config.model.roi_head.type.replace(
                    'StandardRoIHead', 'dist_StandardRoIHead')
                teacher_config.model.roi_head.bbox_head.type = teacher_config.model.roi_head.bbox_head.type.replace(
                    'Shared2FCBBoxHead', 'dist_Shared2FCBBoxHead')
                teacher_config.model.rpn_head['is_teacher'] = True
                teacher_config.model.roi_head['is_teacher'] = True
                teacher_config.model.roi_head.bbox_head['is_teacher'] = True
            self.teacher_model = build_detector(teacher_config['model'])
            if teacher_ckpt is not None:
                load_checkpoint(self.teacher_model, teacher_ckpt, map_location='cpu')

            if init_student:
                teacher_weights = _load_checkpoint(teacher_ckpt)
                neck_weight = []
                head_weight = []
                for name, v in teacher_weights["state_dict"].items():
                    if name.startswith("backbone."):
                        continue
                    elif name.startswith("neck."):
                        if 'lateral' in name:
                            continue
                        else:
                            neck_weight.append((name.replace('neck.', ''), v))
                    elif name.startswith("bbox_head."):
                        head_weight.append((name.replace('bbox_head.', ''), v))
                neck_state_dict = OrderedDict(neck_weight)
                head_state_dict = OrderedDict(head_weight)
                load_state_dict(self.neck, neck_state_dict)
                print('neck loaded from teacher.neck')
                load_state_dict(self.bbox_head, head_state_dict)
                print('head loaded from teacher.head')

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        student_x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)

        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)

            st_rpn_feat, st_rpn_cls_score, st_rpn_bbox_pred, assist_rpn_cls_score, assist_rpn_bbox_pred = self.rpn_head(student_x)
            with torch.no_grad():
                tch_rpn_feat, tch_rpn_cls_score, tch_rpn_bbox_pred = self.teacher_model.rpn_head(teacher_x)

            st_outs = (st_rpn_cls_score, st_rpn_bbox_pred)
            tch_outs = (tch_rpn_cls_score, tch_rpn_bbox_pred)
            # gt_labels is None:
            loss_inputs = st_outs + (student_x, teacher_x,
                                     st_rpn_feat, tch_rpn_feat,
                                     assist_rpn_cls_score, assist_rpn_bbox_pred,
                                     tch_rpn_cls_score, tch_rpn_bbox_pred,
                                     gt_bboxes, None, img_metas)

            rpn_losses = self.rpn_head.loss(*loss_inputs,
                                            gt_bboxes_ignore=gt_bboxes_ignore)

            if proposal_cfg is not None:
                st_proposal_list = self.rpn_head.get_bboxes(*st_outs,
                                                            img_metas=img_metas,
                                                            cfg=proposal_cfg)
                with torch.no_grad():
                    tch_proposal_list = self.teacher_model.rpn_head.get_bboxes(*tch_outs,
                                                                               img_metas=img_metas,
                                                                               cfg=proposal_cfg)

            losses.update(rpn_losses)
        else:
            st_proposal_list = proposals
            tch_proposal_list = proposals

        # roi_losses = self.roi_head.forward_train(student_x, img_metas, proposal_list,
        #                                          gt_bboxes, gt_labels,
        #                                          gt_bboxes_ignore, gt_masks,
        #                                          **kwargs)

        # assign gts and sample proposals
        if self.roi_head.with_bbox or self.roi_head.with_mask:
            num_imgs = len(img_metas)

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            st_sampling_results = []
            tch_sampling_results = []
            for i in range(num_imgs):
                st_assign_result = self.roi_head.bbox_assigner.assign(
                    st_proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                st_sampling_result = self.roi_head.bbox_sampler.sample(
                    st_assign_result,
                    st_proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in student_x])
                st_sampling_results.append(st_sampling_result)

                with torch.no_grad():
                    tch_assign_result = self.teacher_model.roi_head.bbox_assigner.assign(
                        tch_proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    tch_sampling_result = self.teacher_model.roi_head.bbox_sampler.sample(
                        tch_assign_result,
                        tch_proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in teacher_x])
                    tch_sampling_results.append(tch_sampling_result)

            st_rois = bbox2roi([res.bboxes for res in st_sampling_results])
            tch_rois = bbox2roi([res.bboxes for res in tch_sampling_results])

            if self.roi_head.with_bbox:
                student_bbox_results = self.roi_head._bbox_forward_dist(student_x,
                                                                        st_rois, tch_rois)
                with torch.no_grad():
                    teacher_bbox_results = self.teacher_model.roi_head._bbox_forward_dist(teacher_x,
                                                                                          st_rois, tch_rois)

                stroi_bbox_targets = self.roi_head.bbox_head.get_targets(st_sampling_results, gt_bboxes,
                                                                         gt_labels, self.roi_head.train_cfg)
                with torch.no_grad():
                    tchroi_bbox_targets = self.teacher_model.roi_head.bbox_head.get_targets(tch_sampling_results, gt_bboxes,
                                                                                            gt_labels, self.roi_head.train_cfg)

                roi_losses = self.roi_head.bbox_head.loss(student_bbox_results['cls_score'],
                                                          student_bbox_results['bbox_pred'],
                                                          student_bbox_results['dist_cls_score'],
                                                          student_bbox_results['dist_bbox_pred'],
                                                          student_bbox_results['assist_cls_score'],
                                                          student_bbox_results['assist_bbox_pred'],
                                                          student_bbox_results['bbox_feats_stroi'],
                                                          student_bbox_results['bbox_feats_tchroi'],
                                                          student_bbox_results['head_feats'],
                                                          teacher_bbox_results['cls_score'],
                                                          teacher_bbox_results['bbox_pred'],
                                                          teacher_bbox_results['bbox_feats_stroi'],
                                                          teacher_bbox_results['bbox_feats_tchroi'],
                                                          teacher_bbox_results['head_feats'],
                                                          st_rois, tch_rois,
                                                          *stroi_bbox_targets,
                                                          *tchroi_bbox_targets)

        losses.update(roi_losses)

        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.train_cfg is not None:
            if self.eval_teacher:
                self.teacher_model.train(False)
            else:
                self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
