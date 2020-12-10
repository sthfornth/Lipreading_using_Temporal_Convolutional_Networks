# Robust Lipreading in the Wild

Extended from the respository of [Towards practical lipreading with distilled and efficient models](https://sites.google.com/view/audiovisual-speech-recognition#h.p_f7ihgs_dULaj) and [Lipreading using Temporal Convolutional Networks](https://sites.google.com/view/audiovisual-speech-recognition#h.p_jP6ptilqb75s). 

We repeat the previous models and test on new datasets. We train the models on reduced [LRW dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html). The network architecture is based on 3D convolution, ShuffleNet and Temporal Convolutional Network (TCN). 


Also we visualize the truth tables in previous models and analyse the drawbacks, with the code in visualize.py. 

To create more ROI datas, use the code in facial_landmarks.py.  The clips used should be more than 29 frames of any size in .mp4 format.  We used dlib(https://pypi.org/project/dlib/) with opencv to perform facial detection and landmarks, currently only support frontal facial view point.  


### How to preprocess raw data

* To extract mouth ROIs using the script in the [preprocessing](./preprocessing) folder and save them to *`$datasets/`*. To produce damaged dataset, add "remove_frame" or "remove_pixel" argument (usually 10, 20, 30, 50).

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

