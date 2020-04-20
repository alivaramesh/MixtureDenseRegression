from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os, sys
import time
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

from torch.utils.tensorboard import SummaryWriter

def main(opt):

  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  model, optimizer, start_epoch = load_model(
    model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    
  print(opt)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  train_stat = trainer.wta_stat(start_epoch, train_loader)
  val_stat = trainer.wta_stat(start_epoch, val_loader)

if __name__ == '__main__':

  sys.path = sys.argv[1].split(':')+sys.path
  opt = opts().parse( sys.argv[2:] + ['--mdn_wta_stats'] )
  main(opt)