# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict

from .. import build_detector
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from collections import OrderedDict


@DETECTORS.register_module()
class CSKDSingleStageDetector(SingleStageDetector):
    """
    Implementation of `Distilling the Knowledge in a Neural Network.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 init_student,
                 teacher_config,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher

        self.detector_name = bbox_head.type[8:]
        # Build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
            teacher_config.model.bbox_head.type = teacher_config.model.bbox_head.type.replace(
                f'{self.detector_name}Head', f'dist_{self.detector_name}Head')
            teacher_config.model.bbox_head['is_teacher'] = True
            # if self.detector_name == 'FCOS':
            #     teacher_config.model.bbox_head['centerness_on_reg'] = True
            #     teacher_config.model.bbox_head['norm_on_bbox'] = True
            #     teacher_config.model.bbox_head['center_sampling'] = True
            #     teacher_config.model.bbox_head['conv_bias'] = True
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            print('loading teacher weight...')
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')

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

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        student_x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            out_teacher = self.teacher_model.bbox_head(teacher_x)
        losses = self.bbox_head.forward_train(student_x,
                                              teacher_x, out_teacher,
                                              img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
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
