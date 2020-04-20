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

import numpy as np

from torch.utils.tensorboard import SummaryWriter

def main(opt):
  summary_train_dir = os.path.join(opt.exp_dir,'train_summary')
  summary_val_dir = os.path.join(opt.exp_dir,'val_summary')
  if not os.path.exists(summary_train_dir):
    os.mkdir(summary_train_dir)
  if not os.path.exists(summary_val_dir):
    os.mkdir(summary_val_dir)
  tb_writer_train = SummaryWriter(log_dir=summary_train_dir)
  tb_writer_val = SummaryWriter(log_dir=summary_val_dir)

  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  logger = Logger(opt)

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    
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
  
  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle= True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10

  for epoch in range(start_epoch + 1, opt.num_epochs + 1):

    if opt.poly_decay:
      pdp = opt.poly_decay_power
      pdelr = opt.poly_decay_elr
      lr = ((opt.lr - pdelr) * np.power(1 - (float(epoch-1)/(opt.num_epochs)),pdp)) + pdelr
      print('Drop LR to', lr, "at epoch",epoch)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

    n_iters = len(train_loader)
    opt.epoch=epoch
    mark = epoch if opt.save_all else 'last'
    stime= time.time()
    opt.phase='train'
    log_dict_train, _ = trainer.train(epoch, train_loader,logger=logger,tb_writer=tb_writer_train)
    etime= time.time()
    total_time = round(etime-stime,1)
    
    logger.write('epoch: {} | n_iters: {} | time: {} |'.format(epoch,n_iters,total_time))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model,opt.chkp_list_path, optimizer=optimizer)
      with torch.no_grad():
        opt.phase='val'
        log_dict_val, preds,_ = trainer.val(epoch, val_loader,tb_writer=tb_writer_val)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
    logger.write('\n')
    
    if epoch in opt.lr_step and not opt.poly_decay:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model,opt.chkp_list_path, optimizer=optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr, "at epoch",epoch)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()
  tb_writer_train.close()
  tb_writer_val.close()

if __name__ == "__main__":
  opt = opts().parse(sys.argv[1:])
  main(opt)
