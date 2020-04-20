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

from tqdm import tqdm 
import numpy as np
import pickle

from collections import Counter 

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  logger = Logger(opt)

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv,opt.strd2,opt.num_stacks,opt.dfs)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    print(">>>>>>>>>>>>>>>>>>>>>>> start_epoch:",start_epoch)
    
  #print(opt)

  # Trainer = train_factory[opt.task]
  # trainer = Trainer(opt, model, optimizer)
  # trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  # val_loader = torch.utils.data.DataLoader(
  #     Dataset(opt, 'val'), 
  #     batch_size=1, 
  #     shuffle=False,
  #     num_workers=1,
  #     pin_memory=True
  # )

  # if opt.test:
  #   _, preds = trainer.val(0, val_loader)
  #   val_loader.dataset.run_eval(preds, opt.save_dir)
  #   return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print("opt.batch_size:",opt.batch_size)

  dsstat = []
  print("len(train_loader):",len(train_loader))
  for iter_id, batch in tqdm(enumerate(train_loader),total=len(train_loader)):
    if iter_id == 10:
      break
    #print("len(batch)",len(batch), "type(batch)",type(batch), "len(batch['meta'])",len(batch['meta']), "len(batch['meta']['areas'])",len(batch['meta']['areas']))
    areas = batch['meta']['areas'].numpy().reshape((-1))
    areas = areas[areas!=-1]
    tmp =np.sum(areas==0)
    if tmp != 0:
      print(tmp)
    bareastat =[0,0,0]
    areas = map(lambda x: 0 if x <= 32**2 else 2 if x > 96**2 else 1 , areas)
    cnt = Counter(areas)
    hm=batch['hm']
    bg_fg_rate= np.sum(hm.numpy()>=.0001)/np.sum(hm.numpy()<.0001)
    dsstat.append([batch['meta']['img_id'].numpy()[0]]+[cnt [i] for i in [0,1,2]]+[bg_fg_rate])
  dsstat = np.array(dsstat)
  np.save(os.path.join(opt.save_dir,'dsstat'),dsstat)
  print(np.round([np.mean(dsstat,axis=0)],2))
  print(np.round([np.mean(dsstat,axis=0),np.std(dsstat,axis=0),np.min(dsstat,axis=0),np.max(dsstat,axis=0)],2))


if __name__ == "__main__":
  sys.path = sys.argv[1].split(':')+sys.path
  opt = opts().parse(sys.argv[2:]+ ['--ds_stat'])
  main(opt)