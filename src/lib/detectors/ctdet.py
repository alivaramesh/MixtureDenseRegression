from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]

      if self.opt.mdn:
        BS = output['mdn_logits'].shape[0]
        M = self.opt.mdn_n_comps
        H,W = output['mdn_logits'].shape[-2:]
        K = self.opt.num_classes if self.opt.cat_spec_wh else 1
        mdn_logits = output['mdn_logits']
        mdn_logits = mdn_logits.reshape((BS,M,K,H,W)).permute((2,0,1,3,4))
        mdn_pi = torch.clamp(torch.nn.Softmax(dim=2)(mdn_logits), 1e-4, 1.-1e-4) 
        mdn_sigma = torch.clamp(torch.nn.ELU()(output['mdn_sigma'])+self.opt.mdn_min_sigma,1e-4,1e5)
        mdn_sigma = mdn_sigma.reshape((BS,M,2,K,H,W)).permute((3,0,1,2,4,5))
       #mdn_sigma = mdn_sigma.reshape((BS,M*2,K,H,W)).permute((2,0,1,3,4))
        mdn_mu = output['wh']
        mdn_mu = mdn_mu.reshape((BS,M,2,K,H,W)).permute((3,0,1,2,4,5))

        mdn_mu = mdn_mu.reshape((K*BS,M,2,H,W))
        mdn_sigma = mdn_sigma.reshape((K*BS,M,2,H,W))
        mdn_pi = mdn_pi.reshape((K*BS,M,H,W))

        if self.opt.mdn_limit_comp is not None:
          cid=self.opt.mdn_limit_comp
          mdn_pi = mdn_pi[:,cid:cid+1]
          mdn_sigma=mdn_sigma[:,cid:cid+1]
          mdn_mu = mdn_mu[:,cid:cid+1]
          M=1

        #print('mdn_mu.shape',mdn_mu.shape,'mdn_sigma.shape',mdn_sigma.shape,'mdn_pi.shape',mdn_pi.shape)

        C=2
        if self.opt.mdn_48:
          central = mdn_pi * torch.reciprocal(mdn_sigma[:,:,0,:,:])**C * torch.reciprocal(mdn_sigma[:,:,1,:,:])**C
          pi_max,pi_max_idx = torch.max(central,1)
        else:
          pi_max,pi_max_idx = torch.max(mdn_pi,1)
        if self.opt.mdn_max or self.opt.mdn_48:
          a = pi_max_idx.unsqueeze(1).repeat(1,C,1,1).reshape(BS*K,1,C,H,W)
          wh = torch.gather(mdn_mu,1,a).squeeze(1)

          a = pi_max_idx.unsqueeze(1).repeat(1,2,1,1).reshape(BS*K,1,2,H,W)
          sigmas = torch.gather(mdn_sigma,1,a).squeeze(1)
        elif self.opt.mdn_sum:
          wh = torch.sum(mdn_mu*mdn_pi.unsqueeze(2),1)
          sigmas = torch.sum(mdn_sigma*mdn_pi.unsqueeze(2),1)
        
        wh = wh.reshape((K,BS,2,H,W)).permute((1,0,2,3,4)).reshape((BS,2*K,H,W))
        mdn_sigma = sigmas.reshape((K,BS,2,H,W)).permute((1,0,2,3,4)).reshape((BS,2*K,H,W))
        mdn_pi = mdn_pi.reshape((K,BS,-1,H,W)).permute((1,0,2,3,4)).reshape((BS,-1,H,W))

        output.update({'wh':wh})
        #if self.opt.debug == 4:
        output.update({'mdn_max_idx':pi_max_idx.unsqueeze(1),
                       'mdn_sigmas':mdn_sigma,
                       'mdn_max_pi':pi_max.unsqueeze(1)
                       })

      hm = output['hm'].sigmoid_()
      
      wh = output['wh']
      # print('wh.shape',wh.shape,'hm.shape',hm.shape )
      # wh.shape torch.Size([1, 160, <H>,<W>]) cswh
      # wh.shape torch.Size([1, 2, <H>,<W>])
      reg = output['reg'] if self.opt.reg_offset else None
      if self.opt.flip_test:
        # if self.opts.mdn:
        #   raise NotImplementedError
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None

        if 'mdn_sigmas' in output:
          mdn_sigmas=output['mdn_sigmas']
          output['mdn_sigmas'] = (mdn_sigmas[0:1] + flip_tensor(mdn_sigmas[1:2])) / 2
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_decode(hm, wh, reg=reg, K=self.opt.K,
                            cat_spec_wh =self.opt.cat_spec_wh,
                            mdn_max_idx=output.get('mdn_max_idx'),
                            mdn_max_pi=output.get('mdn_max_pi'),
                            mdn_sigmas=output.get('mdn_sigmas'))
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      l = 5
      if self.opt.mdn:
        l += 10 if self.opt.flip_test else 8
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, l)
      dets[0][j][:, :4] /= scale
      if self.opt.mdn:
        if self.opt.flip_test:
          dets[0][j][:, 9:15] /= scale
        else:
          dets[0][j][:, 7:13] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    from external.nms import soft_nms
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.vis_thresh:#self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)

  def save_all_results(self,debugger):
    debugger.save_all_imgs(prefix='{}_'.format(self.opt.current_imgID),path=self.opt.vis_dir)