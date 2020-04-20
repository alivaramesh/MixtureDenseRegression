from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, 5]

    for j in range(num_classes):
      inds = (classes == j)
      if dets.shape[2] == 6:
        top_preds[j + 1] = np.concatenate([
          dets[i, inds, :4].astype(np.float32),
          dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
      else:
        if dets.shape[2] == 14:
          mdn_max_idx = dets[i, inds, 6:7].reshape(-1, 1)
          mdn_sigma = dets[i, inds, 7:9].reshape(-1, 2)#*down_ratio
          mdn_mu = dets[i, inds, 9:11].reshape(-1, 2)#*down_ratio
          mdn_pi = dets[i, inds, 11:12].reshape(-1, 1)
          topk_coords = transform_preds(dets[i, inds, 12:14].reshape(-1, 2), c[i], s[i], (w, h))
        elif dets.shape[2] == 16:
          mdn_max_idx = dets[i, inds, 6:8].reshape(-1, 2)
          mdn_sigma = dets[i, inds, 8:10].reshape(-1, 2)#*down_ratio
          mdn_mu = dets[i, inds, 10:12].reshape(-1, 2)#*down_ratio
          mdn_pi = dets[i, inds, 12:14].reshape(-1, 2)
          topk_coords = transform_preds(dets[i, inds, 14:16].reshape(-1, 2), c[i], s[i], (w, h))
        else:
          raise ValueError("dets.shape: {}".format(dets.shape))
        topk_xs = topk_coords[:,:1]
        topk_ys = topk_coords[:,1:2]
        # print("dets[i, inds, :4].shape:",dets[i, inds, :4].shape)
        # print("mdn_max_idx.shape:",mdn_max_idx.shape)
        top_preds[j + 1] = np.concatenate([
          dets[i, inds, :4].astype(np.float32),
          dets[i, inds, 4:5].astype(np.float32),
          mdn_max_idx,mdn_pi, mdn_sigma,mdn_mu,topk_ys,topk_xs], axis=1).tolist()
    ret.append(top_preds)
  return ret

def multi_pose_post_process(dets, c, s, h, w,down_ratio):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    if dets.shape[2] == 40:
      top_preds = np.concatenate(
        [bbox.reshape(-1, 4), dets[i, :, 4:5], 
        pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    else:
      if dets.shape[2] == 82:
        mdn_max_idx = dets[i, :, 40:42].reshape(-1, 2)
        mdn_sigma = dets[i, :, 42:44].reshape(-1, 2)#*down_ratio
        mdn_mu = dets[i, :, 44:78].reshape(-1, 34)#*down_ratio
        mdn_pi = dets[i, :, 78:80].reshape(-1, 2)
        topk_coords = transform_preds(dets[i, :, 80:82].reshape(-1, 2), c[i], s[i], (w, h))
      elif dets.shape[2] == 80:
        mdn_max_idx = dets[i, :, 40:41].reshape(-1, 1)
        mdn_sigma = dets[i, :, 41:43].reshape(-1, 2)#*down_ratio
        mdn_mu = dets[i, :, 43:77].reshape(-1, 34)#*down_ratio
        mdn_pi = dets[i, :, 77:78].reshape(-1, 1)
        topk_coords = transform_preds(dets[i, :, 78:80].reshape(-1, 2), c[i], s[i], (w, h))
      else:
        raise ValueError("dets.shape: {}".format(dets.shape))
      topk_xs = topk_coords[:,:1]
      topk_ys = topk_coords[:,1:2]
      top_preds = np.concatenate(
        [bbox.reshape(-1, 4), dets[i, :, 4:5], 
        pts.reshape(-1, 34),mdn_max_idx,mdn_pi, mdn_sigma,mdn_mu,topk_ys,topk_xs], 
          axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret
