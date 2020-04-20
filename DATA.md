# Dataset preparation
For training and evaluation you need to download the MS COCO dataset and organize it as indicated below:

### COCO
- Download the images (2017 Train, 2017 Val, 2017 Test) from [coco website](http://cocodataset.org/#download).
- Download annotation files (2017 train/val and test image info) from [coco website](http://cocodataset.org/#download). 
- Place the data (or create symlinks) to make the data folder like:

  ~~~
optdata_dir
|-- coco
    |-- annotations
        |   |-- instances_train2017.json
        |   |-- instances_val2017.json
        |   |-- person_keypoints_train2017.json
        |   |-- person_keypoints_val2017.json
        |   |-- image_info_test-dev2017.json
    |-- train2017
    |-- val2017
    |-- test2017ll 