import sys,os, json, shutil
from detectors.detector_factory import detector_factory
from opts import opts

import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
    Sample command:
    $ python inference_pose.py <minimum_detection_score> --load_model /path/to/model --not_hm_hp  <--keep_res or --fix_res> --arch <hourglass|dla_34> --test_imgs_dir /path/to/dir/of/images  <--mdn_max or --mdn_sum or --mdn_48>
    For mdn based models provide the configuration. For example: --mdn --mdn_n_comps 3 --wh_weight .1 --mdn_min_sigma 1

'''

default_args = 'multi_pose --dataset coco_hp --resume --inference'.split(' ')
assert "--keep_res" in sys.argv  or '--fix_res' in sys.argv
opt = opts().init(default_args + sys.argv[2:])
if opt.mdn:
    assert (int(opt.mdn_max)+int(opt.mdn_sum) + int(opt.mdn_48))==1
assert (int(opt.flip_test_max) + int(opt.flip_test)) <=1
if opt.flip_test_max: assert opt.mdn

outpath =opt.load_model
if opt.not_hm_hp:
    outpath += '_no_hm_hp'
if opt.not_reg_hp_offset:
    outpath += '_no_reg_hp_offset'
if len(opt.test_scales)>1:
    outpath += '_{}'.format(opt.test_scales)
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

outpath += '_{}'.format(opt.test_imgs_dir.strip('/').split('/')[-1])
if opt.debug > 0:
    outpath += '_DEBUG_{}'.format(opt.debug)
hscore_thr = float(sys.argv[1])
outpath += '_{}.json'.format(hscore_thr)
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
for f in tqdm(os.listdir(imgs_dir)):
    img_path = os.path.join(imgs_dir,f)
    img = cv2.imread(img_path)
    imgID = str(int(f.split('/')[-1].split('.')[0]))
    if len(np.shape(img)) == 2:
        img = np.stack((img,img,img),axis=2)
    ret = detector.run(img,imgID=imgID)['results'][1]

    for r in ret:
        if r[4] > hscore_thr:
            kps = np.reshape(r[5:39],(17,2)).astype(np.float32)
            kps = np.concatenate( (kps,np.zeros((17,1)) ), axis=1)
            ests.append({'image_id':int(f.split('.')[0]),'category_id':1,
                        'score':round(r[4],3),
                        'keypoints':np.round(kps.reshape((51,)),1).tolist(),
                        'num_keypoints':17})
            if opt.mdn:
                mdn_comp = tuple(r[39:41]) if opt.flip_test or opt.flip_test_max else tuple(r[39:40])
                mdn_pi = tuple(r[41:43]) if opt.flip_test  or opt.flip_test_max else tuple(r[40:41])
                mdn_sigma = tuple(r[43:45]) if opt.flip_test  or opt.flip_test_max else tuple(r[41:43])
                mdn_mu = tuple(r[45:79]) if opt.flip_test  or opt.flip_test_max else tuple(r[43:77])
                topk_y = tuple(r[79:80]) if opt.flip_test  or opt.flip_test_max else tuple(r[77:78])
                topk_x = tuple(r[80:81]) if opt.flip_test  or opt.flip_test_max else tuple(r[78:79])
                ests[-1].update({'mdn_comps':mdn_comp,
                                 'mdn_sigma':mdn_sigma,
                                 'mdn_mu':mdn_mu,
                                 'mdn_pi':mdn_pi,
                                 'topk_y':topk_y,
                                 'topk_x':topk_x})

with open(outpath, 'w') as of:
    of.write(json.dumps(ests))            
