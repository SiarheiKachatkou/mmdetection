import torch
from mmcv.ops import batched_nms
from ..builder import HEADS
from .anchor_head_s import AnchorHeadS
import torch.nn as nn
from .rpn_test_mixin import RPNTestMixin
import matplotlib.pyplot as plt
import numpy as np
import cv2

def draw_anchors(x,anchors,feat_sizes,img_metas):
    img_idx=0
    for i, feat_map in enumerate(x):
        img=x[i][0,0].detach().cpu().numpy().astype(np.float32)
        img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        scales=np.array(img_metas[img_idx]['img_shape'][:2]) / np.array(tuple(feat_sizes[i]))
        scale_y,scale_x=scales
        for a in anchors[0]:
            color=(int(np.random.random_integers(0,255)),int(np.random.random_integers(0,255)),int(np.random.random_integers(0,255)))

            rects=a[i].detach().cpu().numpy()
            for r in rects:
                p1=(int(scale_x*r[0]),int(scale_y*r[1]))
                p2=(int(scale_x*r[2]),int(scale_y*r[3]))
                cv2.rectangle(img,p1,p2,color)
        plt.imshow(img)
        plt.show()
        dbg=1

@HEADS.register_module()
class RPNHeadS(AnchorHeadS):

    def __init__(self, in_channels, feat_channels, **kwargs):
        super().__init__(in_channels=in_channels, feat_channels=feat_channels, num_classes=1, **kwargs)

        self._num_feat_maps=len(kwargs['anchor_generator']['strides'])
        for i in range(self._num_feat_maps):
            if i>=len(in_channels):
                in_c=in_channels[-1]
            else:
                in_c=in_channels[i]
            setattr(self,f"_rpn_conv{i}",nn.Conv2d(in_c,feat_channels,kernel_size=3,stride=1,padding=1))
            setattr(self,f"_rpn_bbox_pred{i}",nn.Conv2d(feat_channels,self.num_anchors*4*self.num_classes,kernel_size=1))
            setattr(self,f"_rpn_cls_score{i}", nn.Conv2d(feat_channels, self.num_anchors*self.num_classes,kernel_size=1))
            setattr(self,f"_rpn_act{i}",nn.Hardswish(inplace=True))


    def forward_single_(self, inputs ):

        feat_idx, x = inputs
        #feat_idx=0
        feat=getattr(self,f"_rpn_conv{feat_idx}")(x)
        feat=getattr(self,f"_rpn_act{feat_idx}")(feat)
        rpn_bbox_reg = getattr(self,f"_rpn_bbox_pred{feat_idx}")(feat)
        rpn_cls_score = getattr(self,f"_rpn_cls_score{feat_idx}")(feat)

        return rpn_cls_score, rpn_bbox_reg

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        map_results=map(self.forward_single_, enumerate(feats))
        output=tuple(map(list, zip(*map_results)))
        return output

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
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
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super().loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
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
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        return dets[:cfg.nms_post]

    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image.
        """
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas)
        return proposal_list