from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetDetector
from .multi_pose import MultiPoseDetector

detector_factory = {
  'ctdet': CtdetDetector,
  'multi_pose': MultiPoseDetector, 
}
