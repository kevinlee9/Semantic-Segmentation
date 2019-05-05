## Video Segmentation Overview
#### Basic

roadmap:

- interleave box tracking with box-driven segmentation
- propagate the first frame segmentation via graph labeling



Lucid Data Dreaming augmentations, temporal component



#### OSVOS (One Shot Video Object Segmentation)

![osvos](../paper_reports/images/osvos.png)

1. Take a net (say VGG-16) pre-trained for classification for example, on imagenet.
2. Convert it to a fully convolutional network, à la [FCN](https://arxiv.org/abs/1605.06211), thus preserving spatial information:
   \- Remove the FC layers in the end.
   \- Insert a new loss: pixel-wise sigmoid balanced cross entropy (previously used by [HED](https://arxiv.org/abs/1504.06375)). Now each pixel is separately classified into foreground or background.
3. Train the new fully convolutional network on the DAVIS-2016 training set.
4. **One-shot training:** At inference time, given a new input video for segmentation and a ground-truth annotation for the first frame (remember, this is a semi-supervised problem), create a new model, initialized with the weights trained in [3] and fine-tuned on the first frame.

#### MaskTrack (Learning Video Object Segmentation from Static Images)

![masktrack](../paper_reports/images/masktrack.png)

###### offline training

conditional mask prediction

hypothesis: mask estimation are smooth among two near frames 

train: image dataset, use augmentation (deformation and affine transformation on mask) to simulate last frame prediction

test: RGB+last frame mask estimation -> current frame mask estimation

###### online training

fine tune on test video,  generate multiple training samples by augmentation (deformation and affine transformation on mask)



#### CRN

 Motion-guided cascaded refinement network for video object segmentation 