from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .multi_pose import MultiPoseTrainer

train_factory = {
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer, 
}
