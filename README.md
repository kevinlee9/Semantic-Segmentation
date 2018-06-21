# Semantic-Segmentation
List for useful code and papers

Inspired by JackieZhangdx, see [this](https://github.com/JackieZhangdx/WeakSupervisedSegmentationList) for more lists. 

## code

[pytorch-segmentation-detection](https://github.com/warmspringwinds/pytorch-segmentation-detection) a library for dense inference and training of Convolutional Neural Networks, 68.0%

[rdn](https://github.com/fyu/drn) Dilated Residual Networks, 75.6%, may be the best available semantic segmentation in PyTorch?

[Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) A pytorch implementation of Detectron. Both training from scratch and inferring directly from pretrained Detectron weights are available. only for coco now

[AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg) Adversarial Learning for Semi-supervised Semantic Segmentation.  heavily borrowed from a **pytorch DeepLab** implementation ([Link](https://github.com/speedinghzl/Pytorch-Deeplab))

[PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet) PyTorch implementation of ENet

[tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet) Tensorflow implementation of deeplab-resnet: complete and detailed

## models
[dcsp](https://github.com/arslan-chaudhry/dcsp_segmentation) BMVC2017 59.2

## papers

## 
| method | val | test       |  notes |
| ------------ | ---------- | ---------- | ---------- |
| [DSRG](https://github.com/speedinghzl/DSRG)<sub>CVPR2018</sub> | 61.4 | 63.2 | deep seeded region growing  |
| [psa](https://github.com/jiwoon-ahn/psa)<sub>CVPR2018</sub> | **61.7** | **63.7** | pixel affinity network |
| [MDC](https://arxiv.org/pdf/1805.04574.pdf)<sub>CVPR2018</sub> | 60.4 | 60.8 | multi-dilated convolution |
| [DCSP](https://github.com/arslan-chaudhry/dcsp_segmentation)<sub>BMVC2017</sub> | 58.6 | 59.2 | adversarial, TBD|
| [GuidedSeg](https://github.com/coallaoh/GuidedLabelling)<sub>CVPR2017</sub> | 55.7 | 56.7 | saliency, TBD|
