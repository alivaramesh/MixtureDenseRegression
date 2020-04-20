from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # basic experiment setting
    self.parser.add_argument('task', default='multi_pose',
                             help='ctdet | multi_pose')
    self.parser.add_argument('--label_msg',type=str, default=None,help='Any custom message to be concatenated with the experiment label')
    self.parser.add_argument('--data_dir',type=str, default=None, help="Directory that containes a directort named 'coco' as explained in readme/DATA.md")
    self.parser.add_argument('--exp_dir',type=str, default=None,help= "The directory where the new experiment directory will be created. If the directory name start with 'run' then it means to resume training")
    self.parser.add_argument('--test_imgs_dir',type=str,default=None,help='Path to the directory containing images you want top test on')    
    self.parser.add_argument('--annot_path_train',type=str, default=None, help='Path to annotations to be used for training. This will override the default path. It is useful for debugging')
    self.parser.add_argument('--annot_path_val',type=str, default=None,help='Path to annotations to be used for validaion. This will override the default path. It is useful for debugging')

    self.parser.add_argument('--poly_decay',action='store_true')   
    self.parser.add_argument('--poly_decay_power', type=float, default = None)
    self.parser.add_argument('--poly_decay_elr', type=float, default = None)

    self.parser.add_argument('--equal_loss',action='store_true', help='online dataset prune')

    self.parser.add_argument('--dataset', default='coco',
                             help='coco | kitti | coco_hp | pascal')
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--debug', type=int, default=0,
                             help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # useful when lunching training with ipython notebook
                                  '4: save all visualizations to disk'
                                  '5: mdn debug')
    self.parser.add_argument('--demo', default='', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 
    # system
    self.parser.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=2,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    self.parser.add_argument('--vis_thresh', type=float, default=0.0,
                             help='visualization threshold.')
    self.parser.add_argument('--debugger_theme', default='white', 
                             choices=['white', 'black'])
    
    # model
    self.parser.add_argument('--arch', default='hourglass', 
                             help='model architecture. Currently tested'
                                  'dla_34 | hourglass')
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '64 for resnets and 256 for dla.')
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')

    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.')
    
    # train
    self.parser.add_argument('--lr', type=float, default=1.25e-4, 
                             help='learning rate for batch size 32.')
    self.parser.add_argument('--lr_step', type=str, default='90,120',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--num_epochs', type=int, default=140,
                             help='total training epochs.')
    self.parser.add_argument('--batch_size', type=int, default=32,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=1,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')

    # test
    self.parser.add_argument('--inference', action='store_true')
    self.parser.add_argument('--flip_test', action='store_true',
                             help='flip data augmentation.')
    self.parser.add_argument('--flip_test_max', action='store_true',
                             help='flip data augmentation.')#Max                             
    self.parser.add_argument('--test_scales', type=str, default='1',
                             help='multi scale test augmentation.')
    self.parser.add_argument('--nms', action='store_true',
                             help='run nms in testing.')
    self.parser.add_argument('--K', type=int, default=100,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--fix_res', action='store_true',
                             help='fix testing resolution or keep '
                                  'the original resolution')
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')

    # dataset
    self.parser.add_argument('--no_augmentation', action='store_true')
    self.parser.add_argument('--not_rand_crop', action='store_true',
                             help='not use the random crop data augmentation'
                                  'from CornerNet.')
    self.parser.add_argument('--shift', type=float, default=0.1,
                             help='when not using random crop'
                                  'apply shift augmentation.')
    self.parser.add_argument('--scale', type=float, default=0.4,
                             help='when not using random crop'
                                  'apply scale augmentation.')
    self.parser.add_argument('--rotate', type=float, default=0,
                             help='when not using random crop'
                                  'apply rotation augmentation.')
    self.parser.add_argument('--flip', type = float, default=0.5,
                             help='probability of applying flip augmentation.')
    self.parser.add_argument('--no_color_aug', action='store_true',
                             help='not use the color augmenation '
                                  'from CornerNet')
    # multi_pose
    self.parser.add_argument('--aug_rot', type=float, default=0, 
                             help='probability of applying '
                                  'rotation augmentation.')
    # ctdet
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    # multi_pose
    self.parser.add_argument('--hp_weight', type=float, default=1,
                             help='loss weight for human pose offset.')
    self.parser.add_argument('--hm_hp_weight', type=float, default=1,
                             help='loss weight for human keypoint heatmap.')
    
    # task
    # ctdet
    self.parser.add_argument('--norm_wh', action='store_true',
                             help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
    self.parser.add_argument('--dense_wh', action='store_true',
                             help='apply weighted regression near center or '
                                  'just apply regression on center point.')
    self.parser.add_argument('--cat_spec_wh', action='store_true',
                             help='category specific bounding box size.')
    self.parser.add_argument('--not_reg_offset', action='store_true',
                             help='not regress local offset.')
    # multi_pose
    self.parser.add_argument('--dense_hp', action='store_true',
                             help='apply weighted pose regression near center '
                                  'or just apply regression on center point.')
    self.parser.add_argument('--not_hm_hp', action='store_true',
                             help='not estimate human joint heatmap, '
                                  'directly use the joint offset from center.')
    self.parser.add_argument('--not_reg_hp_offset', action='store_true',
                             help='not regress local offset for '
                                  'human joint heatmaps.')
    self.parser.add_argument('--not_reg_bbox', action='store_true',
                             help='not regression bounding box size.')
    
    # ground truth validation
    self.parser.add_argument('--eval_oracle_hm', action='store_true', 
                             help='use ground center heatmap.')
    self.parser.add_argument('--eval_oracle_wh', action='store_true', 
                             help='use ground truth bounding box size.')
    self.parser.add_argument('--eval_oracle_offset', action='store_true', 
                             help='use ground truth local heatmap offset.')
    self.parser.add_argument('--eval_oracle_kps', action='store_true', 
                             help='use ground truth human pose offset.')
    self.parser.add_argument('--eval_oracle_hmhp', action='store_true', 
                             help='use ground truth human joint heatmaps.')
    self.parser.add_argument('--eval_oracle_hp_offset', action='store_true', 
                             help='use ground truth human joint local offset.')
    self.parser.add_argument('--eval_oracle_dep', action='store_true', 
                             help='use ground truth depth.')

    ### MDN
    self.parser.add_argument('--mdn',action='store_true')    
    self.parser.add_argument('--mdn_inter',action='store_true')    
    self.parser.add_argument('--mdn_n_comps', type=int,default=0)
    self.parser.add_argument('--mdn_prior', type=float,default=1)
    self.parser.add_argument('--mdn_min_sigma',type=float,default=10.)    
    self.parser.add_argument('--mdn_V',type=float,default=1.)    
    self.parser.add_argument('--mdn_dropout',type=int,default=0)    
    self.parser.add_argument('--mdn_dropout_stop',type=int,default=2, help='epoch number to stop the drop_out')    
    self.parser.add_argument('--mdn_limit_comp',type=int,default=None, help='To limit the components that are used for inference')   
    self.parser.add_argument('--mdn_max',action='store_true', help="for inference") 
    self.parser.add_argument('--mdn_sum',action='store_true', help="for inference") 
    self.parser.add_argument('--mdn_48',action='store_true', help="for inference") 

    self.parser.add_argument('--fine_tune',action='store_true')    

    self.parser.add_argument('--min_scale',type=float,default=.6)   
    self.parser.add_argument('--max_scale',type=float,default=1.4)   
    

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.num_stacks=2 if opt.arch == 'hourglass' else 1

    if opt.no_augmentation:
      opt.not_rand_crop=True
      opt.shift = 0.
      opt.scale = 0.
      opt.rotate=0.
      opt.flip=0.
      opt.no_color_aug =True
      opt.aug_rot =0.

    assert opt.min_scale < opt.max_scale
    
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31

    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.debug > 0:
      opt.num_workers = 0
      opt.batch_size = 1
      opt.gpus = [opt.gpus[0]]
      opt.master_batch_size = -1
    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)

    opt.time_str = time.strftime('%Y-%m-%d-%H-%M')
    if not opt.inference and not opt.resume:
      run_cfg = ''
      if opt.min_scale != .6:
        run_cfg += '_MNS_{}'.format(opt.min_scale)
      if opt.max_scale != 1.4:
        run_cfg += '_MXS_{}'.format(opt.max_scale)
      if opt.equal_loss:
        run_cfg += '_EQ'
      if opt.annot_path_train is not None:
        run_cfg += '_train_{}'.format(opt.annot_path_train.split('/')[-1].split('.')[0] )
      if opt.annot_path_val is not None:
        run_cfg += '_val_{}'.format(opt.annot_path_val.split('/')[-1].split('.')[0] )
      if opt.debug > 0:
        run_cfg += '_debug_{}'.format(opt.debug)
      if opt.task != 'multi_pose':
        run_cfg += '_{}'.format(opt.task)
      if opt.arch !='hourglass':
        run_cfg += '_{}'.format(opt.arch)
      if opt.mdn:
        run_cfg += '_MDN'
        run_cfg += '_{}_{}'.format(opt.mdn_n_comps,opt.mdn_min_sigma)
        if opt.mdn_V is not None:
          run_cfg += '_V_{}'.format(opt.mdn_V)
        if opt.mdn_inter:
          run_cfg += '_INTRM'
        if opt.mdn_prior !=1:
          raise NotImplementedError
          run_cfg += '_prior_{}'.format(opt.mdn_prior)
        if opt.mdn_dropout > 0:
          assert opt.mdn_dropout < opt.mdn_n_comps
          run_cfg += '_DO_{}_{}'.format(opt.mdn_dropout,opt.mdn_dropout_stop)

      if opt.dense_hp:
        run_cfg += '_dense_hp'
      
      if opt.hp_weight != 1:
        run_cfg += '_hpw_{}'.format(opt.hp_weight)
      if opt.wh_weight != .1:
        run_cfg += '_whw_{}'.format(opt.wh_weight)
      
      if opt.hm_weight != 1:
        run_cfg += '_hmw_{}'.format(opt.hm_weight)
      
      if opt.hm_hp_weight != 1:
        run_cfg += '_hmhpw_{}'.format(opt.hm_hp_weight)
      
      if opt.not_hm_hp:
        run_cfg += '_NoHMHP'
      
      if opt.cat_spec_wh:
        run_cfg += '_CSWH'
      if opt.fine_tune:
        _model  = opt.load_model.split('/')[-1].split('.')[0]
        run_cfg += '_FT_{}'.format(_model)
      if opt.lr != .00025:
        run_cfg += '_LR_{}'.format(opt.lr)
      if opt.batch_size != 12:
        run_cfg += '_BS_{}'.format(opt.batch_size)
      if opt.norm_wh:
        run_cfg += '_{}'.format('norm_wh'.upper())
      if opt.poly_decay:
        run_cfg+='_POLYDEC_{}_{}'.format(opt.poly_decay_elr,opt.poly_decay_power)

      if opt.input_res != -1:
        run_cfg+= '_RES_{}'.format(opt.input_res)
      if opt.no_augmentation:
        run_cfg += '_NOAUG'
      if opt.label_msg is not None:
        run_cfg += '_{}'.format(opt.label_msg)
      opt.run_cfg= run_cfg
      exp_dir_label ='run_{}_{}_{}'.format(opt.exp_id,opt.run_cfg,opt.time_str)
      opt.exp_dir =os.path.join(opt.exp_dir ,exp_dir_label)
      os.mkdir(opt.exp_dir)
    elif not opt.inference:
        assert opt.resume
        assert opt.load_model == ''
    
    if not opt.inference:
      opt.save_dir = os.path.join(opt.exp_dir, 'checkpoints_{}'.format(opt.exp_id))
      opt.chkp_list_path=os.path.join(opt.save_dir,'checkpoints')
      opt.debug_dir = os.path.join(opt.exp_dir, 'debug')
      print('The output will be saved to ', opt.save_dir)
    
    if opt.inference:
      assert opt.load_model != ''
    elif opt.resume:
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir 
      with open(opt.chkp_list_path) as _if:
          opt.load_model = _if.readline().strip('\n')
      opt.load_model = os.path.join(model_path, opt.load_model)

    if opt.mdn and opt.load_model != '' and not opt.fine_tune:
      assert '_MDN_{}'.format(opt.mdn_n_comps) in opt.load_model

    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes

    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
    
    if opt.task == 'ctdet':
      opt.cls_id_to_cls_name  = lambda x: dataset.class_name[x+1]
      # assert opt.dataset in ['pascal', 'coco']
      hm_heads = opt.num_classes
      opt.heads = {'hm': hm_heads,
                   'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
      if opt.mdn:
        m = opt.mdn_n_comps
        opt.heads.update({'wh': m*2 if not opt.cat_spec_wh else m*2 * opt.num_classes,
                          'mdn_sigma': m*2 if not opt.cat_spec_wh else m*2 * opt.num_classes,
                          'mdn_logits': m if not opt.cat_spec_wh else m * opt.num_classes,
                          })
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    elif opt.task == 'multi_pose':
      opt.flip_idx = dataset.flip_idx
      opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
      if opt.mdn:
        m = opt.mdn_n_comps
        opt.heads.update({'mdn_logits': m, 'mdn_sigma':2*m,'hps': 34*m })
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
      if opt.hm_hp:
        opt.heads.update({'hm_hp': 17})
      if opt.reg_hp_offset:
        opt.heads.update({'hp_offset': 2})
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

  def init(self, args=''):
    default_dataset_info = {
      'ctdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'multi_pose': {
        'default_resolution': [512, 512], 'num_classes': 1, 
        'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
        'dataset': 'coco_hp', 'num_joints': 17,
        'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
                     [11, 12], [13, 14], [15, 16]]}
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt
