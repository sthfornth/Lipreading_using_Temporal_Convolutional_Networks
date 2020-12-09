# Robust Lipreading in the Wild

Extended from the respository of [Towards practical lipreading with distilled and efficient models](https://sites.google.com/view/audiovisual-speech-recognition#h.p_f7ihgs_dULaj) and [Lipreading using Temporal Convolutional Networks](https://sites.google.com/view/audiovisual-speech-recognition#h.p_jP6ptilqb75s). In this repository, we provide pre-trained models, network settings for end-to-end visual speech recognition (lipreading). We trained our model on [LRW dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html). The network architecture is based on 3D convolution, ResNet-18 plus MS-TCN.


### Introduction



<div align="center"><img src="doc/pipeline.png" width="640"/></div>

By using this repository, you can achieve a performance of 87.9% on the LRW dataset. This reporsitory also provides a script for feature extraction.

### Preprocessing

As described in [our paper](https://arxiv.org/abs/2001.08702), each video sequence from the LRW dataset is processed by 1) doing face detection and face alignment, 2) aligning each frame to a reference mean face shape 3) cropping a fixed 96 × 96 pixels wide ROI from the aligned face image so that the mouth region is always roughly centered on the image crop 4) transform the cropped image to gray level.

You can run the pre-processing script provided in the [preprocessing](./preprocessing) folder to extract the mouth ROIs.

<table style="display: inline-table;">  
<tr><td><img src="doc/demo/original.gif", width="144"></td><td><img src="doc/demo/detected.gif" width="144"></td><td><img src="doc/demo/transformed.gif" width="144"></td><td><img src="doc/demo/cropped.gif" width="144"></td></tr>
<tr><td>0. Original</td> <td>1. Detection</td> <td>2. Transformation</td> <td>3. Mouth ROIs</td> </tr>
</table>



### How to preprocess raw data

Pre-process mouth ROIs using the script in the [preprocessing](./preprocessing) folder and save them to *`$TCN_LIPREADING_ROOT/datasets/`*.



### How to train

* To train on small LRW dataset:

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --data-dir <DATA-DIRECTORY>
```


### How to test

* To evaluate on LRW dataset:

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --data-dir <DATA-DIRECTORY>
```

