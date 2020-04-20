## For human pose estimation
python /path/to/MixtureDenseRegression/src/main.py multi_pose --exp_id hg_1x --dataset coco_hp --arch hourglass --exp_dir /path/to/experiments/dir --batch_size 3 --master_batch 3 --lr 2.5e-4 --gpus $CUDA_VISIBLE_DEVICES --num_epochs 50 --lr_step 40 --data_dir /path/to/dataset/root/parent/directory --save_all

python /path/to/MixtureDenseRegression/src/main.py multi_pose --exp_id hg_1x --dataset coco_hp --arch hourglass --exp_dir /path/to/experiments/dir --batch_size 3 --master_batch 3 --lr 2.5e-4 --gpus $CUDA_VISIBLE_DEVICES --num_epochs 50 --lr_step 40 --data_dir /path/to/dataset/root/parent/directory  --save_all --mdn --mdn_n_comps 3 --hp_weight .1

## For object detection
python /path/to/MixtureDenseRegression/src/main.py ctdet --exp_id coco_dla34_1x --dataset coco --arch dla_34 --exp_dir /path/to/experiments/dir --batch_size 3 --master_batch 3 --lr 2e-4 --gpus $CUDA_VISIBLE_DEVICES --num_epochs 140 --lr_step 90120 --data_dir /path/to/dataset/root/parent/directory/ --save_all

python /path/to/MixtureDenseRegression/src/main.py ctdet --exp_id coco_dla34_1x --dataset coco --arch dla_34 --exp_dir /path/to/experiments/dir --batch_size 3 --master_batch 3 --lr 2e-4 --gpus $CUDA_VISIBLE_DEVICES --num_epochs 140 --lr_step 90120 --data_dir /path/to/dataset/root/parent/directory/ --save_all --mdn --mdn_n_comps 3 --wh_weight .1 --mdn_V 1 --mdn_min_sigma 2

python /path/to/MixtureDenseRegression/src/main.py ctdet --exp_id coco_hg --dataset coco --arch hourglass --exp_dir /path/to/experiments/dir --batch_size 3 --master_batch 3 --lr 2.5e-4 --gpus $CUDA_VISIBLE_DEVICES --num_epochs 50 --lr_step 40 --data_dir /path/to/dataset/root/parent/directory/ --save_all

python /path/to/MixtureDenseRegression/src/main.py ctdet --exp_id coco_hg --dataset coco --arch hourglass --exp_dir /path/to/experiments/dir --batch_size 3 --master_batch 3 --lr 2.5e-4 --gpus $CUDA_VISIBLE_DEVICES --num_epochs 50 --lr_step 40 --data_dir /data/leuven/325/vsc32547/dataset/ --save_all --mdn --mdn_n_comps 3 --wh_weight .1 --mdn_V 1 --mdn_min_sigma 1

