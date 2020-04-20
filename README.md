
## [Mixture Dense Regression for Object Detection and Human Pose Estimation](https://arxiv.org/abs/1912.00821) (CVPR 2020)

## Abstract 

Mixture models are well-established learning approaches that, in computer vision, have mostly been applied to inverse or ill-defined problems. However, they are general-purpose divide-and-conquer techniques, splitting the input space into relatively homogeneous subsets in a data-driven manner. Not only ill-defined but also well-defined complex problems should benefit from them. To this end, we devise a framework for spatial regression using mixture density networks. We realize the framework for object detection and human pose estimation. For both tasks, a mixture model yields higher accuracy and divides the input space into interpretable modes. For object detection, mixture components focus on object scale, with the distribution of components closely following that of ground truth the object scale. This practically alleviates the need for multi-scale testing, providing a superior speed-accuracy trade-off. For human pose estimation, a mixture model divides the data based on viewpoint and uncertainty -- namely, front and back views, with back view imposing higher uncertainty. We conduct experiments on the MS COCO dataset and do not face any mode collapse. 

For questions, please contact me at [ali.varamesh@kuleuven.be](ali.varamesh@kuleuven.be).

## Acknoledgement
Our repo is forked from the amazing codebase of the [Object as Points paper](https://github.com/xingyizhou/CenterNet)

## Installation
1- Fiest use [mixturedense.yml](mixturedense.yml) to reproduce the exact [Anaconda](https://www.anaconda.com/download) environment that we have used for our experiments:
  ~~~
  conda env create -f mixturedense.yml
  ~~~
  To activate the environment:
  ~~~
  source activate mixturedense
  ~~~

2- Install [COCOAPI](https://github.com/cocodataset/cocoapi)

3- Compile deformable convolutional conda env create -f environment.yml(from [DCNv2](https://github.com/CharlesShang/DCNv2.git)).
  ~~~  
  cd src/lib/models/networks/DCNv2
  ./make.sh
  ~~~
## Train
To train models from scratch see sample comands at [experiments](experiments)

## Tets
To test the models for detction and pose estimation on a images (stored in a directory) use the [inference_ctdet.py](inference_ctdet.py) and [inference_pose.py](nference_pose.py) scripts, respectively

## License


## Citation

    @article{varamesh2019mixture,
      title={Mixture Dense Regression for Object Detection and Human Pose Estimation},
      author={Varamesh, Ali and Tuytelaars, Tinne},
      journal={arXiv preprint arXiv:1912.00821},
      year={2019}
    }
