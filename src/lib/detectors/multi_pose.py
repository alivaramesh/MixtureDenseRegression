from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.decode import multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr,_sigmoid
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class MultiPoseDetector(BaseDetector):
  def __init__(self, opt):
    super(MultiPoseDetector, self).__init__(opt)
    self.flip_idx = opt.flip_idx

  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      if self.opt.mdn:
        mdn_logits = output['mdn_logits']
        mdn_pi = torch.clamp(torch.nn.Softmax(dim=1)(mdn_logits), 1e-4, 1.-1e-4)  
        mdn_sigma= torch.clamp(torch.nn.ELU()(output['mdn_sigma'])+self.opt.mdn_min_sigma,1e-4,1e5)
        mdn_mu = output['hps']

        # print("mdn_pi.shape:",mdn_pi.shape)
        # print("mdn_mu.shape:",mdn_mu.shape)
        # print("mdn_sigma.shape:",mdn_sigma.shape)
        (BS,_,H,W) = mdn_sigma.shape

        if self.opt.mdn_limit_comp is not None:
          M= mdn_pi.shape[1]
          C = mdn_mu.shape[1]//M
          cid=self.opt.mdn_limit_comp
          mdn_pi = mdn_pi[:,cid:cid+1,:,:]
          mdn_sigma=mdn_sigma[:,2*cid:2*cid+2,:,:]
          mdn_mu = mdn_mu[:,C*cid:C*cid+C,:,:]

        M= mdn_pi.shape[1]
        mdn_sigma = torch.reshape(mdn_sigma, (BS,M,2,H,W))
        C = mdn_mu.shape[1]//M
        mdn_mu = torch.reshape(mdn_mu, (BS,M,C,H,W))

        if self.opt.mdn_48:
          central = mdn_pi * torch.reciprocal(mdn_sigma[:,:,0,:,:])**C * torch.reciprocal(mdn_sigma[:,:,1,:,:])**C
          pi_max,pi_max_idx = torch.max(central,1)
        else:
          pi_max,pi_max_idx = torch.max(mdn_pi,1)
        if self.opt.mdn_max or self.opt.mdn_48:
          a = pi_max_idx.unsqueeze(1).repeat(1,C,1,1).reshape(BS,1,C,H,W)
          hps = torch.gather(mdn_mu,1,a).squeeze(1)

          a = pi_max_idx.unsqueeze(1).repeat(1,2,1,1).reshape(BS,1,2,H,W)
          sigmas = torch.gather(mdn_sigma,1,a).squeeze(1)
        elif self.opt.mdn_sum:
          hps = torch.sum(mdn_mu*mdn_pi.unsqueeze(2),1)
          sigmas = torch.sum(mdn_sigma*mdn_pi.unsqueeze(2),1)

        output.update({'hps':hps})
        #if self.opt.debug == 4:
        output.update({'mdn_max_idx':pi_max_idx.unsqueeze(1),
                       'mdn_sigmas':sigmas,
                       'mdn_max_pi':pi_max.unsqueeze(1)
                       })
        
      output['hm'] = output['hm'].sigmoid_()
      if self.opt.hm_hp:
        output['hm_hp'] = output['hm_hp'].sigmoid_()

      reg = output['reg'] if self.opt.reg_offset else None
      hm_hp = output['hm_hp'] if self.opt.hm_hp else None
      hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      if self.opt.flip_test or self.opt.flip_test_max:
        output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
        output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
        hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                if hm_hp is not None else None
        reg = reg[0:1] if reg is not None else None
        hp_offset = hp_offset[0:1] if hp_offset is not None else None
        if self.opt.mdn:
          output['mdn_max_idx'][1:2] = flip_tensor(output['mdn_max_idx'][1:2])
          output['mdn_max_pi'][1:2] = flip_tensor(output['mdn_max_pi'][1:2])
          output['mdn_sigmas'][1:2] = flip_tensor(output['mdn_sigmas'][1:2])

      if self.opt.flip_test:
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
      
        if self.opt.mdn:
          output['mdn_sigmas'] = (output['mdn_sigmas'][0:1] + output['mdn_sigmas'][1:2] ) / 2
     
      elif self.opt.flip_test_max:
        if self.opt.mdn:

          output['hps'][1:2] = flip_lr_off(output['hps'][1:2], self.flip_idx)

          #print("output['mdn_max_pi'].shape:",output['mdn_max_pi'].shape)
          _,pi_max_idx = torch.max(output['mdn_max_pi'],0)
          _,_,H,W =output['hps'].shape 
          a = pi_max_idx.unsqueeze(0).repeat(1,34,1,1).reshape(1,34,H,W)
          output['hps']= torch.gather(output['hps'],0,a)
          a = pi_max_idx.unsqueeze(0).repeat(1,2,1,1).reshape(1,2,H,W)
          output['mdn_sigmas']= torch.gather(output['mdn_sigmas'],0,a)

      dets = multi_pose_decode(
        output['hm'], output['wh'], output['hps'],
        reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K,
        mdn_max_idx=output.get('mdn_max_idx'),
        mdn_max_pi=output.get('mdn_max_pi'),
        mdn_sigmas=output.get('mdn_sigmas'))

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'],self.opt.down_ratio)
    for j in range(1, self.num_classes + 1):
      l = 39
      if self.opt.mdn:
        l += 42 if self.opt.flip_test or self.opt.flip_test_max  else 40
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, l)
      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:39] /= scale
      if self.opt.mdn:
        if self.opt.flip_test or self.opt.flip_test_max:
          dets[0][j][:, 43:81] /= scale
        else:
          dets[0][j][:, 41:79] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    from external.nms import soft_nms_39
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    if self.opt.nms or len(self.opt.test_scales) > 1:
      soft_nms_39(results[1], Nt=0.5, method=2)
    results[1] = results[1].tolist()
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:39] *= self.opt.down_ratio
    if self.opt.mdn:
      if self.opt.flip_test or self.opt.flip_test_max:
        dets[:, :, 42:278] *= self.opt.down_ratio
      else:
        dets[:, :, 41:277] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred,img_id='pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred,'pred_hmhp')
  
  def show_results(self, debugger, image, results):
    img_id='multi_pose'
    debugger.add_img(image, img_id=img_id)
    for bbox in results[1]:
      if bbox[4] > self.opt.vis_thresh:
        pi_max_idx = bbox[39:41] if self.opt.mdn and self.opt.flip_test or self.opt.flip_test_max else bbox[39:40] if self.opt.mdn  else None
        mdn_sigma = bbox[41:43] if self.opt.mdn and self.opt.flip_test or self.opt.flip_test_max else bbox[40:42] if self.opt.mdn else None
        mdn_mu = bbox[43:79] if self.opt.mdn and self.opt.flip_test or self.opt.flip_test_max else bbox[42:78] if self.opt.mdn else None
        mdn_pi = bbox[77:81] if self.opt.mdn and self.opt.flip_test or self.opt.flip_test_max else bbox[76:79] if self.opt.mdn else None
        debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id=img_id,pi_max_idx=pi_max_idx,pi_max=mdn_pi)
        debugger.add_coco_hp(bbox[5:39], img_id=img_id)
    debugger.show_all_imgs(pause=self.pause)
  
  def save_all_results(self,debugger):
    debugger.save_all_imgs(prefix='{}_'.format(self.opt.current_imgID),path=self.opt.vis_dir)
  

'''
if self.opt.mdn_H:
  mdn_H_p = _sigmoid(output['mdn_H_logits'])

  mdn_H_p1 = mdn_H_p
  mdn_H_p2 =1.-mdn_H_p

  if self.opt.mdn_H_max:
    mdn_H_p1[mdn_H_p1<.5] = 0
    mdn_H_p1[mdn_H_p1>=.5] = 1
    mdn_H_p2 = 1- mdn_H_p1

  #mdn_H_p = mdn_H_p.repeat(1,self.opt.mdn_n_comps,1,1)

  mdn_pi_1 = torch.clamp(torch.nn.Softmax(dim=1)(mdn_logits[:,:self.opt.mdn_n_comps,:,:]), 1e-4, 1.-1e-4) 
  mdn_pi_1*=mdn_H_p1
  mdn_pi_2 = torch.clamp(torch.nn.Softmax(dim=1)(mdn_logits[:,self.opt.mdn_n_comps:,:,:]), 1e-4, 1.-1e-4)  
  mdn_pi_2*=mdn_H_p2
  mdn_pi = torch.cat([mdn_pi_1,mdn_pi_2],1)
else:
'''