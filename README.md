
# ASCFormer: An Adaptive Strucure-aware Cascaded Transformer for 3D Object Detection


This is the code for ASCFormer(An Adaptive Strucure-aware Cascaded Transformer for 3D Object Detection). 
This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), some codes are from [TED](https://github.com/hailanyi/TED), 
[CasA](https://github.com/hailanyi/CasA), [IA-SSD](https://github.com/yifanzhang713/IA-SSD) and [CasA](https://github.com/hailanyi/CasA).

## Detection Framework
3D object detection has achieved significant progress in outdoor LiDAR point clouds, however, the inherent irregularity and varying sparsity distribution of point occupancy is a key challenge. Existing transformer-based 3D detectors often take all tokens within the attention window as equally important regardless of varying-sparsity, which not only fails to adapt to the gap between the varying beam-densities but also results in increased the memory and computational cost. In this paper, we propose an Adaptive Structure-aware  cascaded transformer (ASCFormer) that dynamically captures the density insensitive multiscale structure features to model the long-range dependencies via cascaded learning. This approach involves an adaptive structure-aware tokens selector that introduces voxel level segmentation auxiliary network and local density estimation to select a subset of multi-scale tokens with varying receptive field sizes using inverse transform sampling over the significance scores. The selected tokens not only embed the position of voxel point centroids but also encode the foreground probability and point density as the features. To improve the training convergence of the window-based transformer in 3D voxel space, we employ cascaded learning via cross-stage attention to enhance the feature representation capability for more refinement the localization precision of 3D bounding boxes. This design of structure-aware re-weighting gracefully enhances the cascade paradigm to be better adaptable for the varying-sparsity distribution of input data. Extensive experiments on the KITTI and Waymo Open datasets demonstrate that the proposed ASCFormer achieves exceptional performance compared to state-of-the-art 3D object detection methods.

![](./docs/framework.png)




## Getting Started
### Dependency
All the codes are tested in the following environment:
+ Ubuntu 20.04
+ Python 3.9.13 
+ PyTorch 1.8.1
+ Numba 0.53.1
+ [Spconv 2.1.22](https://github.com/traveller59/spconv) # pip install spconv-cu111
+ NVIDIA CUDA 11.1 
+ 1x 3090 GPUs


### Prepare dataset

#### KITTI Dataset
* You can generate the dataset by yourself as follows:
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded
files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing),
which are optional for data augmentation in the training):

```
ASCFormer
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

Run following command to creat dataset infos:
```
python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

#### Waymo Dataset

```
ASCFormer
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_train_val_test
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── pcdet_waymo_track_dbinfos_train_cp.pkl
│   │   │── waymo_infos_test.pkl
│   │   │── waymo_infos_train.pkl
│   │   │── waymo_infos_val.pkl
├── pcdet
├── tools
```

Run following command to creat dataset infos:
```
python3 -m pcdet.datasets.waymo.waymo_tracking_dataset --cfg_file tools/cfgs/dataset_configs/waymo_tracking_dataset.yaml 
```


### Install `pcdet`

Please install `pcdet` by running `python setup.py develop`.

a. Clone this repository.

b. Install the dependent libraries by running `python setup.py develop`


### Training and Evaluation

#### Evaluation

```
cd tools
python3 test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

For example, if you test the ASTFormer model:

```
cd tools
python3 test.py --cfg_file cfgs/kitti_models/ASTFormer.yaml --ckpt ASTFormer.pth
```

Multiple GPU test: you need modify the gpu number in the dist_test.sh and run
```
sh dist_test.sh 
```
The log infos are saved into log-test.txt
You can run ```cat log-test.txt``` to view the test results.

#### Training

```
cd tools
python3 train.py --cfg_file ${CONFIG_FILE}
```

For example, if you train the CasA-V model:

```
cd tools
python3 train.py --cfg_file cfgs/kitti_models/ASTFormer.yaml
```

Multiple GPU train: you can modify the gpu number in the dist_train.sh and run
```
sh dist_train.sh
```
The log infos are saved into log.txt
You can run ```cat log.txt``` to view the training process.



## Acknowledgement
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

[TED](https://github.com/hailanyi/TED)

[VirConv](https://github.com/hailanyi/TED)

[CasA](https://github.com/hailanyi/CasA)

[IA-SSD](https://github.com/yifanzhang713/IA-SSD)


## Citation

```
@inproceedings{ASTFormer,
    title={ASCFormer: An Adaptive Strucure-aware Cascaded Transformer for 3D Object Detection},
    author={Li, Xinglong and Zhang, Xiaowei},
    booktitle={},
    year={2023}
}
```




