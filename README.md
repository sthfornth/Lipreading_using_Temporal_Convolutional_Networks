# Robust Lipreading in the Wild

Extended from the respository of [Towards practical lipreading with distilled and efficient models](https://sites.google.com/view/audiovisual-speech-recognition#h.p_f7ihgs_dULaj) and [Lipreading using Temporal Convolutional Networks](https://sites.google.com/view/audiovisual-speech-recognition#h.p_jP6ptilqb75s). 

We provide pre-trained models, network settings for end-to-end visual speech recognition (lipreading). We trained our model on [LRW dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html). The network architecture is based on 3D convolution, ResNet-18 plus MS-TCN.


### Preprocessing

As described in [our paper](https://arxiv.org/abs/2001.08702), each video sequence from the LRW dataset is processed by 1) doing face detection and face alignment, 2) aligning each frame to a reference mean face shape 3) cropping a fixed 96 × 96 pixels wide ROI from the aligned face image so that the mouth region is always roughly centered on the image crop 4) transform the cropped image to gray level.

### How to preprocess raw data

* To extract mouth ROIs using the script in the [preprocessing](./preprocessing) folder and save them to *`$datasets/`*.

```Shell
python crop_mouth_from_video.py --video-dir <VIDEO-DIRETORY> \
                                --landmark-direc <LANDMARK-DIRECTORY> \
                                --save-direc <DATA-DIRECTORY> \
                                --convert-gray \
                                --remove-frame <NOISE-FRAME>
                                --remove-pixel <NOISE-PIXEL>
```

### How to train

* To train on small LRW dataset:

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --config-path <MODEL-JSON-PATH> \
                                      --data-dir <DATA-DIRECTORY> \
                                      --annonation-dir <VIDEO-DIRECTORY> \
                                      --train
```


### How to test

* To evaluate on small LRW dataset:

```Shell
CUDA_VISIBLE_DEVICES=0 python main.py --config-path <MODEL-JSON-PATH> \
                                      --model-path <MODEL-PATH> \
                                      --data-dir <DATA-DIRECTORY>
```

