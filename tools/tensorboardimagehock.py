import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.tensorboard import TensorboardLoggerHook

@HOOKS.register_module()
class TensorboardImageHook(TensorboardLoggerHook):

    _SHOW_BATCHES=3
    _SCORE_THR=0.5

    def after_val_epoch(self, runner):
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
                    image_np=image.float().cpu().detach().numpy()
                    image_np=np.transpose(image_np,(1,2,0))
                    image_np = np.ascontiguousarray(image_np)
                    image_np=cv2.normalize(image_np,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
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

    def after_train_epoch(self, runner):

        for batch_i,batch in enumerate(runner.data_loader):

            for img,gt_bbox in zip(batch['img'].data,batch['gt_bboxes'].data):
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

            if batch_i>self._SHOW_BATCHES:
                break


        dbg = 1

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
