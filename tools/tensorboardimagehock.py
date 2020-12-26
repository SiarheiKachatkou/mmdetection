import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.logger.tensorboard import TensorboardLoggerHook

@HOOKS.register_module()
class TensorboardImageHook(TensorboardLoggerHook):

    _SHOW_BATCHES=3
    _SCORE_THR=0.5

    @master_only
    def after_val_epoch(self, runner):

        super().after_val_epoch(runner)

        for batch_i,batch in enumerate(runner.data_loader):

            for img,gt_bbox,img_metas in zip(batch['img'].data,batch['gt_bboxes'].data, batch['img_metas'].data):
                img_0=img[0]
                img_norm=self._normalize_img(img_0)

                self.writer.add_image_with_boxes(tag="ground_truth",img_tensor=img_norm,box_tensor=gt_bbox[0])

                img=self._to(runner, img)

                feats=runner.model.module.extract_feat(img)
                feat_sizes=[f.shape[2:] for f in feats]

                anchor_generator=runner.model.module.rpn_head.anchor_generator
                anchors=anchor_generator.grid_anchors(featmap_sizes=feat_sizes,device=feats[0].device)
                for level in range(len(anchors)):
                    level_anchors=anchors[level]
                    self.writer.add_image_with_boxes(tag=f"anchors_{level}_all", img_tensor=img_norm, box_tensor=level_anchors)
                    self.writer.add_image_with_boxes(tag=f"anchors_{level}_part", img_tensor=img_norm,
                                                     box_tensor=level_anchors[len(level_anchors)//2:len(level_anchors)//2+6])

                preds_list=runner.model.module([img],[img_metas],return_loss=False,rescale=True)
                img_idx=0
                for preds_for_image, image, img_meta in zip(preds_list,img, img_metas):
                    image_np=self._tensor_to_np(image)
                    h, w, _ = img_meta['img_shape']
                    img_show = image_np[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = cv2.resize(img_show, (ori_w, ori_h))

                    for p in preds_for_image:
                        if len(p)>0:
                            xmin,ymin,xmax,ymax,score=p[0]
                            if score>self._SCORE_THR:
                                img_show=cv2.rectangle(img_show,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),10)

                    self.writer.add_image(f'predictions_{img_idx}',torch.from_numpy(np.transpose(img_show,(2,0,1))))
                    img_idx+=1


                #preds = runner.model.module(return_loss=False, rescale=True, **batch)


            if batch_i>self._SHOW_BATCHES:
                break

    @master_only
    def after_train_epoch(self, runner):

        super().after_train_epoch(runner)

        for batch_i,batch in enumerate(runner.data_loader):

            self._draw_assigned(runner, batch)

            self._draw_anchors(runner, batch)

            if batch_i>self._SHOW_BATCHES:
                break


        dbg = 1

    def _draw_assigned(self, runner, batch):
        thickness=2
        red=(255,0,0)
        green=(0,255,0)
        blue=(0,0,255)

        for img, gt_bbox in zip(batch['img'].data, batch['gt_bboxes'].data):
            img_0 = img[0]
            img_show = self._normalize_img(img_0)
            img_show_np=self._tensor_to_np(img_show)

            img = self._to(runner, img)

            feats = runner.model.module.extract_feat(img)
            feat_sizes = [f.shape[2:] for f in feats]
            device = feats[0].device
            anchor_generator = runner.model.module.rpn_head.anchor_generator
            anchors = anchor_generator.grid_anchors(featmap_sizes=feat_sizes, device=device)
            gt_bbox_0=gt_bbox[0].to(device)
            for g in gt_bbox_0:
                img_show_np=self._draw_bbox(g,img_show_np,red,thickness)

            for level in range(len(anchors)):
                assign_result=runner.model.module.rpn_head.assigner.assign(anchors[level],gt_bbox_0)

                for anchor_ind, gt_ind in enumerate(assign_result.gt_inds):
                    if gt_ind > 0:
                        img_show_np = self._draw_bbox(gt_bbox_0[gt_ind-1], img_show_np, green, thickness)
                        img_show_np = self._draw_bbox(anchors[level][anchor_ind], img_show_np, blue, thickness)

                self.writer.add_image(tag=f"assign_cumul_in_level_{level}", img_tensor=torch.from_numpy(np.transpose(img_show_np,(2,0,1))))

    def _draw_anchors(self, runner, batch):

        for img, gt_bbox in zip(batch['img'].data, batch['gt_bboxes'].data):
            img_0 = img[0]
            img_norm = self._normalize_img(img_0)

            self.writer.add_image_with_boxes(tag="ground_truth", img_tensor=img_norm, box_tensor=gt_bbox[0])

            img = self._to(runner, img)

            feats = runner.model.module.extract_feat(img)
            feat_sizes = [f.shape[2:] for f in feats]

            anchor_generator = runner.model.module.rpn_head.anchor_generator
            anchors = anchor_generator.grid_anchors(featmap_sizes=feat_sizes, device=feats[0].device)
            for level in range(len(anchors)):
                level_anchors = anchors[level]
                self.writer.add_image_with_boxes(tag=f"anchors_{level}_all", img_tensor=img_norm,
                                                 box_tensor=level_anchors)
                self.writer.add_image_with_boxes(tag=f"anchors_{level}_part", img_tensor=img_norm,
                                                 box_tensor=level_anchors[
                                                            len(level_anchors) // 2:len(level_anchors) // 2 + 6])

    def _normalize_img(self, img_tensor):
        img_norm = 255 * (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        img_norm = img_norm.type(torch.uint8)
        return img_norm

    def _to(self,runner, img_tensor):
        gpu_id = runner.model.device_ids[0]
        img = img_tensor.cuda(gpu_id)

        if 'fp16' in runner.meta['config']:
            img = img.half()
        return img

    def _tensor_to_np(self,img_tensor):
        image_np=img_tensor.float().cpu().detach().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = np.ascontiguousarray(image_np)
        image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return image_np

    def _draw_bbox(self, bbox_tensor, img_np, color, thickness):
        xmin, ymin, xmax, ymax = bbox_tensor[:4]
        img_np = cv2.rectangle(img_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                    color, thickness)
        return img_np