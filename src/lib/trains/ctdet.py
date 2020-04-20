from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss,RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss,th_mdn_loss_ind
from models.decode import ctdet_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat,_softmax
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

import torch.nn.functional as F

class CtdetLoss(torch.nn.Module):
 
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    if opt.mdn:
      self.crit_wh = None if opt.dense_wh else None if opt.norm_wh else th_mdn_loss_ind
    else:
      self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch,global_step,tb_writer):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    batch_hm_loss, batch_wh_loss, batch_off_loss = 0, 0, 0#per batch losses
    for s in range(opt.num_stacks):
      
      output = outputs[s]

      output['hm'] = _sigmoid(output['hm'])

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      tmp = self.crit(output['hm'], batch['hm']) 
      hm_loss = hm_loss+ tmp[0]/ opt.num_stacks
      batch_hm_loss = batch_hm_loss+ tmp[1]/ opt.num_stacks
      if opt.wh_weight > 0:
        if opt.mdn:
          BS = output['mdn_logits'].shape[0]
          M = opt.mdn_n_comps
          H,W = output['mdn_logits'].shape[-2:]
          K = opt.num_classes if opt.cat_spec_wh else 1

          mdn_logits = output['mdn_logits']
          mdn_logits = mdn_logits.reshape((BS,M,K,H,W)).permute((2,0,1,3,4))
          # mdn_logits.shape: torch.Size([80, 2, 3, 128, 128])
          
          mdn_pi = torch.clamp(torch.nn.Softmax(dim=2)(mdn_logits), 1e-4, 1.-1e-4)  
          # mdn_pi.shape: torch.Size([80, 2, 3, 128, 128])

          mdn_sigma = torch.clamp(torch.nn.ELU()(output['mdn_sigma'])+opt.mdn_min_sigma,1e-4,1e5)
          mdn_sigma = mdn_sigma.reshape((BS,M*2,K,H,W)).permute((2,0,1,3,4))
          # mdn_sigma.shape: torch.Size([80, 2, 6, 128, 128])

          mdn_mu = output['wh']
          mdn_mu = mdn_mu.reshape((BS,M*2,K,H,W)).permute((2,0,1,3,4))
          # mdn_mu.shape: torch.Size([80, 2, 6, 128, 128])

          gt = batch['cat_spec_wh'] if opt.cat_spec_wh else batch['wh']
          gt = gt.reshape((BS,-1,opt.num_classes if opt.cat_spec_wh else 1,2)).permute((2,0,1,3))
          # gt.shape: torch.Size([80, 2, 128, 2])

          if opt.cat_spec_wh:
            mask = batch['cat_spec_mask'][:,:,0::2].unsqueeze(-1).permute((2,0,1,3))
            # mask.shape: torch.Size([80, 2, 128, 1])
          else:
            mask = batch['reg_mask'].unsqueeze(2).unsqueeze(0)
            # print("mask.shape:", mask.shape)
            # mask.shape: torch.Size([1, 2, 128, 1])

          V = torch.Tensor([opt.mdn_V]).cuda()

          I = mask.shape[-2]
          _gt = gt.reshape((K*BS,I,-1))
          _mask = mask.reshape((K*BS,I,-1))
          batch_ind = torch.repeat_interleave(batch['ind'],K,dim=0)
          _mdn_mu = _tranpose_and_gather_feat(mdn_mu.reshape((K*BS,-1,H,W)), batch_ind)
          _mdn_sigma = _tranpose_and_gather_feat(mdn_sigma.reshape((K*BS,-1,H,W)), batch_ind)
          _mdn_pi = _tranpose_and_gather_feat(mdn_pi.reshape((K*BS,-1,H,W)), batch_ind)

          # mdn_n_comps=3
          # batch['ind'].shape: torch.Size([2, 128])
          # gt.shape: torch.Size([1, 2, 128, 2])
          # mask.shape: torch.Size([1, 2, 128, 1])
          # mdn_mu.shape:    torch.Size([1, 2, 6, 128, 128])
          # mdn_pi.shape:    torch.Size([1, 2, 3, 128, 128])
          # mdn_sigma.shape: torch.Size([1, 2, 6, 128, 128])

          # batch['ind'].shape: torch.Size([2, 128])
          # _gt.shape: torch.Size([2, 128, 2])
          # _mask.shape: torch.Size([2, 128, 1])
          # _mdn_mu.shape: torch.Size([2, 128, 6])
          # _mdn_pi.shape: torch.Size([2, 128, 3])
          # _mdn_sigma.shape: torch.Size([2, 128, 6])

          tmp = self.crit_wh(_gt,_mdn_mu,_mdn_sigma,_mdn_pi,_mask,V,C=1)
          wh_loss += tmp[0]/ opt.num_stacks
          batch_wh_loss += tmp[1]/ opt.num_stacks

          for _c in range(opt.num_classes if opt.cat_spec_wh  else 1):
            _mdn_pi = _tranpose_and_gather_feat(mdn_pi[_c], batch['ind'])
            _mdn_sigma = _tranpose_and_gather_feat(mdn_sigma[_c], batch['ind'])
            _,_max_pi_ind = torch.max(_mdn_pi,-1)
            if tb_writer is not None:
              _cat = opt.cls_id_to_cls_name(_c)
              tb_writer.add_histogram('mdn_pi_max_comp/{}'.format(_cat),_max_pi_ind+1,global_step=global_step)
              for i in range(_mdn_pi.shape[2]):
                tb_writer.add_histogram('mdn_pi/{}/{}'.format(_cat,i),_mdn_pi[:,:,i],global_step=global_step)
                tb_writer.add_histogram('mdn_sigma/{}/{}'.format(_cat,i),_mdn_sigma[:,:,i*2:i*2+2],global_step=global_step)
        else:
          if opt.dense_wh:
            mask_weight = batch['dense_wh_mask'].sum() + 1e-4
            wh_loss += (
              self.crit_wh(output['wh'] * batch['dense_wh_mask'],
              batch['dense_wh'] * batch['dense_wh_mask']) / 
              mask_weight) / opt.num_stacks
          elif opt.cat_spec_wh:
            wh_loss += self.crit_wh(
              output['wh'], batch['cat_spec_mask'],
              batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
          else:
            tmp = self.crit_reg(output['wh'], batch['reg_mask'],batch['ind'], batch['wh'])
            wh_loss += tmp[0]/ opt.num_stacks
            batch_wh_loss += tmp[1]/ opt.num_stacks
          '''
          output['wh'].shape: torch.Size([2, 160, 128, 128])
          batch['ind'].shape: torch.Size([2, 128])
          batch['cat_spec_mask'].shape: torch.Size([2, 128, 160])
          '''

      if opt.reg_offset and opt.off_weight > 0:
        tmp = self.crit_reg(output['reg'], batch['reg_mask'],
                            batch['ind'], batch['reg'])
        off_loss += tmp[0] / opt.num_stacks
        batch_off_loss += tmp[1] / opt.num_stacks

    loss_stats={}
    loss,batch_loss=0,0

    loss += opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
          opt.off_weight * off_loss
    batch_loss += opt.hm_weight * batch_hm_loss + opt.wh_weight * batch_wh_loss + \
          opt.off_weight * batch_off_loss

    loss_stats.update({'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss,
                  'batch_loss': batch_loss, 'batch_hm_loss': batch_hm_loss,\
                  'batch_wh_loss': batch_wh_loss, 'batch_off_loss': batch_off_loss})
    return loss, loss_stats

class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss',\
                    'batch_loss', 'batch_hm_loss', 'batch_wh_loss', 'batch_off_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
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

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]