# Image-Retrieval-tf
- The leftmost is the query image. The remaining images are retrieved from gallery and ordered from left to right by similarity. 
The upper three row are correct retrieval and last row is incorrect retrieval.
![](assets/127_46.jpg)
![](assets/4371_8.jpg)
![](assets/2362_27.jpg)
![](assets/177_4.jpg)

- It might also be retrieved in the same category elaborately.
![](assets/1428_38.jpg)
![](assets/1168_38.jpg)
![](assets/1596_38.jpg)

## Feature
- multi gpu
- tta

## TODO
- balanced sampling

## dataset
- https://www.kaggle.com/c/imaterialist-challenge-furniture-2018
- download_images.py (about 20GB) from https://www.kaggle.com/aloisiodn/python-3-download-multi-proc-prog-bar-resume Or use datasets/download_images.py

## References from 
- https://arxiv.org/abs/1812.00442
- https://github.com/ace19-dev/mvcnn-tf
- https://github.com/kobiso/SENet-tensorflow-slim
- https://github.com/vonclites/checkmate/blob/master/checkmate.py
- https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py
- https://github.com/ildoonet/tf-mobilenet-v2
- http://openresearch.ai/t/nccl-efficient-tensorflow-multigpu-training/159
