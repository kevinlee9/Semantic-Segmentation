## Video Segmentation Overview

#### OSVOS (One Shot Video Object Segmentation)

![osvos](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/osvos.png)

1. Take a net (say VGG-16) pre-trained for classification for example, on imagenet.
2. Convert it to a fully convolutional network, à la [FCN](https://arxiv.org/abs/1605.06211), thus preserving spatial information:
   \- Remove the FC layers in the end.
   \- Insert a new loss: pixel-wise sigmoid balanced cross entropy (previously used by [HED](https://arxiv.org/abs/1504.06375)). Now each pixel is separately classified into foreground or background.
3. Train the new fully convolutional network on the DAVIS-2016 training set.
4. **One-shot training:** At inference time, given a new input video for segmentation and a ground-truth annotation for the first frame (remember, this is a semi-supervised problem), create a new model, initialized with the weights trained in [3] and fine-tuned on the first frame.

#### **MaskTrack (Learning Video Object Segmentation from Static Images)**

![masktrack](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/masktrack.png)

