import sys,os, json, shutil
from detectors.detector_factory import detector_factory
from opts import opts

import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
    Sample command:
    $ python inference_ctdet.py <minimum_detection_score> --load_model /path/to/model --not_hm_hp  <--keep_res or --fix_res> --arch <hourglass|dla_34> --test_imgs_dir /path/to/dir/of/images  <--mdn_max or --mdn_sum or --mdn_48>
    For mdn based models provide the configuration. For example: --mdn --mdn_n_comps 3 --wh_weight .1 --mdn_min_sigma 1
'''
default_args = 'ctdet --dataset coco --resume --inference'.split(' ')
assert "--keep_res" in sys.argv  or '--fix_res' in sys.argv
opt = opts().init(default_args + sys.argv[2:])
if opt.mdn:
    assert (int(opt.mdn_max)+int(opt.mdn_sum) + int(opt.mdn_48))==1
assert opt.fix_res == (not opt.keep_res)
assert (int(opt.flip_test_max) + int(opt.flip_test)) <=1
if opt.flip_test_max: assert opt.mdn

if '_V_' in opt.load_model:
    assert '_V_{}'.format(opt.mdn_V) in opt.load_model
    
outpath =opt.load_model
if len(opt.test_scales)>1:
    outpath += '_{}_'.format(','.join(map(str,opt.test_scales)))
if opt.mdn_sum:
    outpath += '_mdnSUM'
if opt.mdn_48:
    outpath += '_mdn48'
if opt.nms:
    outpath += '_nms'
if not opt.flip_test and not opt.flip_test_max:
    outpath += '_NoFlip'
if opt.flip_test_max:
    outpath += '_FlipMax'
if opt.mdn_limit_comp is not None:
    outpath += '_Comp_{}'.format(opt.mdn_limit_comp)

if opt.fix_res:
    outpath += '_test_res_{}_{}'.format(opt.input_h, opt.input_w)
outpath += '_{}'.format(opt.test_imgs_dir.strip('/').split('/')[-1])


if opt.debug > 0:
    outpath += '_DEBUG_{}'.format(opt.debug)

score_thr = float(sys.argv[1])#.3
outpath += '_{}.json'.format(score_thr)
print("outpath:",outpath)
if os.path.exists(outpath):
    print('outpath already exists')
    sys.exit()
if opt.debug > 0 and opt.debug <5:
    imgs_set = opt.test_imgs_dir.split('/')[-1]
    vis_dir = outpath + '_{}_vis'.format(imgs_set)
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir,True)
    os.mkdir(vis_dir)
    opt.vis_dir=vis_dir
detector = detector_factory[opt.task](opt)

imgs_dir= opt.test_imgs_dir
ests = []
if len(opt.test_scales) ==1:
    assert opt.test_scales[0]==1.
from tqdm import tqdm

class_name = [
      '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
      'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
      'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      'scissors', 'teddy bear', 'hair drier', 'toothbrush']

_valid_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
      24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
      58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
      82, 84, 85, 86, 87, 88, 89, 90]

def _to_float(x):
    return float("{:.2f}".format(x))

for f in tqdm(os.listdir(imgs_dir)):
    img_path = os.path.join(imgs_dir,f)
    img = cv2.imread(img_path)
    imgID = str(int(f.split('/')[-1].split('.')[0]))
    if len(np.shape(img)) == 2:
        img = np.stack((img,img,img),axis=2)
    ret = detector.run(img,imgID=imgID)['results']
    
    for cls_ind in ret:
        category_id = _valid_ids[cls_ind - 1]
        for r in ret[cls_ind]:
            bbox= r[:4]
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            score = r[4]
            if score < score_thr:
                continue
            bbox_out  = list(map(_to_float, bbox[0:4]))

            detection = {
                "image_id": int(imgID),
                "category_id": int(category_id),
                "bbox": bbox_out,
                "score": float("{:.2f}".format(score))
            }
            ests.append(detection)
            if opt.mdn:
                mdn_comp = tuple(map(float,r[5:7])) if opt.flip_test else tuple(map(float,r[5:6]))
                mdn_pi = tuple(map(float,r[7:9])) if opt.flip_test else tuple(map(float,r[6:7]))
                mdn_sigma = tuple(map(float,r[9:11])) if opt.flip_test else tuple(map(float,r[7:9]))
                mdn_mu = tuple(map(float,r[11:13])) if opt.flip_test else tuple(map(float,r[9:11]))
                topk_y = tuple(map(float,r[13:14])) if opt.flip_test else tuple(map(float,r[11:12]))
                topk_x = tuple(map(float,r[14:15])) if opt.flip_test else tuple(map(float,r[12:13]))
                ests[-1].update({'mdn_comps':mdn_comp,
                                 'mdn_sigma':mdn_sigma,
                                 'mdn_mu':mdn_mu,
                                 'mdn_pi':mdn_pi,
                                 'topk_y':topk_y,
                                 'topk_x':topk_x})


with open(outpath, 'w') as of:
    of.write(json.dumps(ests))            
