import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.logger.tensorboard import TensorboardLoggerHook

thickness=2
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)

@HOOKS.register_module()
class TensorboardImageHook(TensorboardLoggerHook):

    _SHOW_BATCHES=1
    _SCORE_THR=0.05

    @master_only
    def after_val_epoch(self, runner):

        global_step = runner.iter*(runner.epoch+1)

        super().after_val_epoch(runner)

        for batch_i,batch in enumerate(runner.data_loader):

            for img,gt_bbox,img_metas in zip(batch['img'].data,batch['gt_bboxes'].data, batch['img_metas'].data):
                img_0=img[0]
                img_norm=self._normalize_img(img_0)

                self.writer.add_image_with_boxes(tag=f"ground_truth_{batch_i}",img_tensor=img_norm,box_tensor=gt_bbox[0])


                img=self._to(runner, img)

                preds_list=runner.model.module([img],[img_metas],return_loss=False,rescale=True)
                img_idx=batch_i*len(img)
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

                    self.writer.add_image(f'predictions_{img_idx}',torch.from_numpy(np.transpose(img_show,(2,0,1))),global_step=global_step)
                    img_idx+=1


            if batch_i>self._SHOW_BATCHES:
                break

    @master_only
    def after_train_epoch(self, runner):

        super().after_train_epoch(runner)

        for batch_i,batch in enumerate(runner.data_loader):

            self._draw_assigned_proposals(runner, batch)

            self._draw_assigned_anchors(runner, batch)

            self._draw_anchors(runner, batch)

            if batch_i>self._SHOW_BATCHES:
                break

    def _draw_assigned_proposals(self, runner, batch):

        global_step = runner.iter * (runner.epoch + 1)

        model=runner.model.module
        proposal_cfg = model.train_cfg.get('rpn_proposal',
                                           model.test_cfg.rpn)

        imgs, img_metas, gt_bboxes_list, gt_labels_list = batch['img'].data, batch['img_metas'].data, batch['gt_bboxes'].data,batch['gt_labels'].data

        for img, gt_bboxes,img_meta, gt_labels in zip(imgs, gt_bboxes_list, img_metas, gt_labels_list):
            img = self._to(runner, img)
            feats = model.extract_feat(img)

            gt_bboxes = self._to(runner,gt_bboxes)
            gt_labels=self._to(runner,gt_labels)

            rpn_losses, proposal_list = model.rpn_head.forward_train(
                feats,
                img_meta,
                gt_bboxes,
                proposal_cfg=proposal_cfg)

            assigner=model.roi_head.bbox_assigner
            sampler=model.roi_head.bbox_sampler
            img_show_np=self._tensor_to_np(img[0])
            self._draw_assign_results(gt_bboxes[0],gt_labels[0], proposal_list,assigner,sampler,img_show_np,name="proposals",global_step=global_step)


    def _draw_assigned_anchors(self, runner, batch):

        global_step = runner.iter * (runner.epoch + 1)

        for img, gt_bbox, gt_labels in zip(batch['img'].data, batch['gt_bboxes'].data, batch['gt_labels'].data):
            img_0 = img[0]
            gt_bbox_0 = gt_bbox[0]
            gt_labels_0 = gt_labels[0]

            img_show = self._normalize_img(img_0)
            img_show_np=self._tensor_to_np(img_show)

            img = self._to(runner, img)

            feats = runner.model.module.extract_feat(img)
            feat_sizes = [f.shape[2:] for f in feats]
            device = feats[0].device
            gt_bbox_0 = gt_bbox_0.to(device)
            gt_labels_0 = gt_labels_0.to(device)

            anchor_generator = runner.model.module.rpn_head.anchor_generator
            assigner = runner.model.module.rpn_head.assigner
            sampler = runner.model.module.rpn_head.sampler
            
            anchors = anchor_generator.grid_anchors(featmap_sizes=feat_sizes, device=device)

            self._draw_assign_results(gt_bbox_0, gt_labels_0, anchors, assigner, sampler, img_show_np, name='anchors', global_step=global_step)
            
    def _draw_assign_results(self, gt_bbox, gt_labels, anchors_list, assigner, sampler, img_show_np, name, global_step):

        img_show_np_sampling = copy.deepcopy(img_show_np)

        for g in gt_bbox:
            img_show_np = self._draw_bbox(g, img_show_np, red, thickness)

        for level in range(len(anchors_list)):
            assign_result = assigner.assign(anchors_list[level], gt_bbox)
            sample_result=sampler.sample(assign_result,anchors_list[level], gt_bbox, gt_labels)
            for pos_bb in sample_result.pos_bboxes:
                img_show_np_sampling = self._draw_bbox(pos_bb, img_show_np_sampling, green, thickness)
            for neg_bb in sample_result.neg_bboxes:
                img_show_np_sampling = self._draw_bbox(neg_bb, img_show_np_sampling, red, thickness)

            self.writer.add_image(tag=f"{name}_sampling_cumul_in_level_{level}",
                                  img_tensor=torch.from_numpy(np.transpose(img_show_np_sampling, (2, 0, 1))),
                                  global_step=global_step)

            for anchor_ind, gt_ind in enumerate(assign_result.gt_inds):
                if gt_ind > 0:
                    img_show_np = self._draw_bbox(gt_bbox[gt_ind - 1], img_show_np, green, thickness)
                    if anchor_ind<len(anchors_list[level]):
                        img_show_np = self._draw_bbox(anchors_list[level][anchor_ind], img_show_np, blue, thickness)

            self.writer.add_image(tag=f"{name}_assign_cumul_in_level_{level}",
                                  img_tensor=torch.from_numpy(np.transpose(img_show_np, (2, 0, 1))),
                                  global_step=global_step)



    def _draw_anchors(self, runner, batch):

        global_step = runner.iter * (runner.epoch + 1)

        for img, gt_bbox in zip(batch['img'].data, batch['gt_bboxes'].data):
            img_0 = img[0]
            img_norm = self._normalize_img(img_0)

            self.writer.add_image_with_boxes(tag="ground_truth", img_tensor=img_norm, box_tensor=gt_bbox[0],global_step=global_step)

            img = self._to(runner, img)

            feats = runner.model.module.extract_feat(img)
            feat_sizes = [f.shape[2:] for f in feats]

            anchor_generator = runner.model.module.rpn_head.anchor_generator
            anchors = anchor_generator.grid_anchors(featmap_sizes=feat_sizes, device=feats[0].device)
            for level in range(len(anchors)):
                level_anchors = anchors[level]
                self.writer.add_image_with_boxes(tag=f"anchors_{level}_all", img_tensor=img_norm,
                                                 box_tensor=level_anchors,global_step=global_step)
                self.writer.add_image_with_boxes(tag=f"anchors_{level}_part", img_tensor=img_norm,
                                                 box_tensor=level_anchors[
                                                            len(level_anchors) // 2:len(level_anchors) // 2 + 6],global_step=global_step)

    def _normalize_img(self, img_tensor):
        img_norm = 255 * (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        img_norm = img_norm.type(torch.uint8)
        return img_norm

    def _to(self,runner, tensor_or_list):
        gpu_id = runner.model.device_ids[0]

        def _to_device(tensor):
            tensor = tensor.cuda(gpu_id)
            if 'fp16' in runner.meta['config']:
                tensor = tensor.half()
            return tensor

        if isinstance(tensor_or_list,list):
            return [_to_device(t) for t in tensor_or_list]
        else:
            return _to_device(tensor_or_list)

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