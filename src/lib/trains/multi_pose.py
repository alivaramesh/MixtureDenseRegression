from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss,th_mdn_loss_ind,th_mdn_loss_dense
from models.utils import _tranpose_and_gather_feat
from models.decode import multi_pose_decode
from models.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import multi_pose_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def add_cbar(fig,ax,_im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(_im, cax=cax)

IND_TO_KP={0:'nose',1:'l_eye',2:'r_eye',3:'l_ear',4:'r_ear',
    5:'l_shoulder',6:'r_shoulder',7:'l_elbow',8:'r_elbow',
    9:'l_wrist',10:'r_wrist',11:'l_hip',12:'r_hip',
    13:'l_knee',14:'r_knee',15:'l_ankle',16:'r_ankle'}
    
class MultiPoseLoss(torch.nn.Module):
  def __init__(self, opt):
    super(MultiPoseLoss, self).__init__()
    self.crit = FocalLoss()
    self.crit_hm_hp = FocalLoss()
    if opt.mdn:
      self.crit_kp = th_mdn_loss_dense if opt.dense_hp else \
                      th_mdn_loss_ind
    else:
      self.crit_kp = torch.nn.L1Loss(reduction='sum')  if opt.dense_hp else \
                      RegWeightedL1Loss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                      RegLoss() if opt.reg_loss == 'sl1' else None
    self.opt = opt

  def forward(self, outputs, batch,global_step,tb_writer):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    hp_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0

    loss_stats = {}
    for s in range(opt.num_stacks):
      output = outputs[s]
      output['hm'] = _sigmoid(output['hm'])
      if opt.hm_hp:
        output['hm_hp'] = _sigmoid(output['hm_hp'])
      
      if opt.eval_oracle_hmhp:
        output['hm_hp'] = batch['hm_hp']
      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_kps:
        if opt.dense_hp:
          output['hps'] = batch['dense_hps']
        else:
          output['hps'] = torch.from_numpy(gen_oracle_map(
            batch['hps'].detach().cpu().numpy(), 
            batch['ind'].detach().cpu().numpy(), 
            opt.output_res, opt.output_res)).to(opt.device)
      if opt.eval_oracle_hp_offset:
        output['hp_offset'] = torch.from_numpy(gen_oracle_map(
          batch['hp_offset'].detach().cpu().numpy(), 
          batch['hp_ind'].detach().cpu().numpy(), 
          opt.output_res, opt.output_res)).to(opt.device)

      if opt.mdn:
        V=torch.Tensor((np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0).astype(np.float32)).float().cuda()

      hm_loss += self.crit(output['hm'], batch['hm'])[0] / opt.num_stacks

      if opt.mdn:
        mdn_logits = output['mdn_logits']
        #mdn_logits.shape: torch.Size([2, 3, 128, 128])
        if opt.mdn_dropout > 0 and opt.epoch<opt.mdn_dropout_stop:
          M=opt.mdn_n_comps

          ridx= torch.randperm(M)[:torch.randint(1,1+opt.mdn_dropout,(1,))]
          drop_mask = torch.ones((M,))
          drop_mask[ridx]=0
          drop_mask = torch.reshape(drop_mask,(1,-1,1,1)).float().cuda()
          mdn_logits = mdn_logits*drop_mask
          if tb_writer is not None:
            tb_writer.add_histogram('drop_out_idx',ridx+1,global_step=global_step)
        else:
          mdn_pi = torch.clamp(torch.nn.Softmax(dim=1)(mdn_logits), 1e-4, 1.-1e-4)  
        
        mdn_sigma= torch.clamp(torch.nn.ELU()(output['mdn_sigma'])+opt.mdn_min_sigma,1e-4,1e5)
        mdn_mu = output['hps']

        if tb_writer  is not None:
          for i in range(mdn_pi.shape[1]):
            tb_writer.add_histogram('mdn_pi/{}'.format(i),mdn_pi[:,i],global_step=global_step)
            tb_writer.add_histogram('mdn_sigma/{}'.format(i),mdn_sigma[:,i*2:i*2+2],global_step=global_step)

        if opt.dense_hp:
          gt = batch['dense_hps']
          mask = batch['dense_hps_mask'][:,0::2,:,:]
          _,max_pi_ind = torch.max(mdn_pi,1)
        else:
          gt = batch['hps']
          mask = batch['hps_mask'][:,:,0::2]
          mdn_mu= _tranpose_and_gather_feat(mdn_mu, batch['ind'])
          mdn_pi= _tranpose_and_gather_feat(mdn_pi, batch['ind'])
          mdn_sigma= _tranpose_and_gather_feat(mdn_sigma, batch['ind'])
          _,max_pi_ind = torch.max(mdn_pi,-1)

        if tb_writer is not None:
          tb_writer.add_histogram('mdn_pi_max_comp',max_pi_ind+1,global_step=global_step)
        '''
          mdn_n_comps=3
          batch['hps'].shape: torch.Size([2, 32, 34])
          batch['hps_mask'].shape: torch.Size([2, 32, 34])
          batch['ind'].shape: torch.Size([2, 32])
          gt.shape: torch.Size([2, 32, 34])
          mask.shape: torch.Size([2, 32, 17])
          before gather, after gather
          mdn_mu.shape: torch.Size([2, 102, 128, 128]), torch.Size([2, 32, 102])
          mdn_pi.shape: torch.Size([2, 3, 128, 128]), torch.Size([2, 32, 3])
          mdn_sigma.shape: torch.Size([2, 6, 128, 128]), torch.Size([2, 32, 6])
        '''
        if opt.mdn_inter:
          hp_loss += self.crit_kp(gt,mdn_mu,mdn_sigma,mdn_pi,mask,V,debug=opt.debug==6)[0] / opt.num_stacks
        else:
          hp_loss = self.crit_kp(gt,mdn_mu,mdn_sigma,mdn_pi,mask,V,debug=opt.debug==6)[0] / opt.num_stacks
      else:
        if opt.dense_hp:
          mask_weight = batch['dense_hps_mask'].sum() + 1e-4
          hp_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'], 
                                  batch['dense_hps'] * batch['dense_hps_mask']) / 
                                  mask_weight) / opt.num_stacks
        else:
          hp_loss += self.crit_kp(output['hps'], batch['hps_mask'], 
                                batch['ind'], batch['hps']) / opt.num_stacks
      
      if opt.wh_weight > 0:
        wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                 batch['ind'], batch['wh'])[0] / opt.num_stacks
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                  batch['ind'], batch['reg'])[0] / opt.num_stacks
      if opt.reg_hp_offset and opt.off_weight > 0:
        hp_offset_loss += self.crit_reg(
          output['hp_offset'], batch['hp_mask'],
          batch['hp_ind'], batch['hp_offset'])[0] / opt.num_stacks
      if opt.hm_hp and opt.hm_hp_weight > 0:
        hm_hp_loss += self.crit_hm_hp(
          output['hm_hp'], batch['hm_hp'])[0] / opt.num_stacks

    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + opt.hp_weight * hp_loss + \
           opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss
    
    loss_stats.update({'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss, 
                  'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss})
    return loss, loss_stats

class MultiPoseTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(MultiPoseTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss', 
                   'hp_offset_loss', 'wh_loss', 'off_loss']
    loss = MultiPoseLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    hm_hp = output['hm_hp'] if opt.hm_hp else None
    hp_offset = output['hp_offset'] if opt.reg_hp_offset else None
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'], 
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

    dets[:, :, :4] *= opt.input_res / opt.output_res
    dets[:, :, 5:39] *= opt.input_res / opt.output_res
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.input_res / opt.output_res
    dets_gt[:, :, 5:39] *= opt.input_res / opt.output_res
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')
          debugger.add_coco_hp(dets[i, k, 5:39], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')
          debugger.add_coco_hp(dets_gt[i, k, 5:39], img_id='out_gt')

      if opt.hm_hp:
        pred = debugger.gen_colormap_hp(output['hm_hp'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')
        debugger.add_blend_img(img, gt, 'gt_hmhp')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    hm_hp = output['hm_hp'] if self.opt.hm_hp else None
    hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'], 
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    
    dets_out = multi_pose_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]