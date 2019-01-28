# Non-local-tensorflow

An implementation of Non-local block intruduced in Tensorflow.
Link to the original paper:[Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf)

### Introduction
This repository includes the code(training and testing) for [DRN](https://arxiv.org/pdf/1705.09914.pdf) and [Non-local](https://arxiv.org/pdf/1711.07971.pdf). The code is for semantic segmentation.

### Requirements
```
python 2.7.x
tensorflow >= 1.4.0
```

### Usage
1. Download VOC2012 dataset.<br>
2. Prepare the tfrecord data to train the model. Please see my another repository [Convert_To_TFRecord](https://github.com/Tramac/Convert_To_TFRecord).<br>

3. Train the model<br>
```
#the parameter of --mode in main.py need to be "train"
python main.py
```
4. Test the model<br>
```
#the parameter of --mode in main.py need to be "test"
python main.py
```

### Experiments
|Method|Non-local|mIoU|
|-----|-----|-----
|DRN|✗||
|DRN|✓||

### To Do
- [ ] COCO
- [ ] Cityscapes
- [ ] ADE

### Reference
* [Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch)
