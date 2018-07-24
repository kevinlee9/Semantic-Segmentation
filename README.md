# Semantic-Segmentation
List of useful codes and papers for semantic segmentation(weakly)

## code

[pytorch-segmentation-detection](https://github.com/warmspringwinds/pytorch-segmentation-detection) a library for dense inference and training of Convolutional Neural Networks, 68.0%

[rdn](https://github.com/fyu/drn) Dilated Residual Networks, 75.6%, may be the best available semantic segmentation in PyTorch?

[Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) A pytorch implementation of Detectron. Both training from scratch and inferring directly from pretrained Detectron weights are available. only for coco now

[AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg) Adversarial Learning for Semi-supervised Semantic Segmentation.  heavily borrowed from a **pytorch DeepLab** implementation ([Link](https://github.com/speedinghzl/Pytorch-Deeplab))

[PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet) PyTorch implementation of ENet

[tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet) Tensorflow implementation of deeplab-resnet(deeplabv2, resnet101-based): complete and detailed

[tensorflow-deeplab-lfov](https://github.com/DrSleep/tensorflow-deeplab-lfov) Tensorflow implementation of deeplab-LargeFOV(deeplabv2, vgg16-based): complete and detailed

[resnet38](https://github.com/itijyou/ademxapp)  Wider or Deeper: Revisiting the ResNet Model for Visual Recognition: implemented using MXNET

#### to work 
[BDWSS](https://github.com/ascust/BDWSS) Bootstrapping the Performance of Webly Supervised Semantic Segmentation

[psa](https://github.com/jiwoon-ahn/psa) Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation

#### SEC
[original](https://github.com/kolesman/SEC): Caffe  
[BDSSW](https://github.com/ascust/BDWSS): MXNET  
[DSRG](https://github.com/speedinghzl/DSRG): Caffe, CAM and DRFI provided  
[SEC-tensorflow](https://github.com/xtudbxk/SEC-tensorflow): tensorflow version  

## papers
#### random walk
Learning random-walk label propagation for weakly-supervised semantic segmentation: scribble

Convolutional Random Walk Networks for Semantic Image Segmetation: fully, affinity branch(low level)

Soft Proposal Networks for Weakly Supervised Object Localization: attention, semantic affinity

Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation: image-level, semantic affinity

#### saliency
Exploiting Saliency for Object Segmentation from Image Level Labels: detailed experiments about localization and expand arch and policy.

## Top works
#### PASCAL VOC2012

| method | val | test       |  notes |
| ------------ | ---------- | ---------- | ---------- |
| [DSRG](https://github.com/speedinghzl/DSRG)<sub>CVPR2018</sub> | 61.4 | 63.2 | deep seeded region growing  |
| [psa](https://github.com/jiwoon-ahn/psa)<sub>CVPR2018</sub> | **61.7** | **63.7** | pixel affinity network |
| [MDC](https://arxiv.org/pdf/1805.04574.pdf)<sub>CVPR2018</sub> | 60.4 | 60.8 | multi-dilated convolution |
| [MCOF](http://3dimage.ee.tsinghua.edu.cn/wx/mcof)<sub>CVPR2018</sub> | 60.3 | 61.2 | iterative, RegionNet(sppx)|
| [DCSP](https://github.com/arslan-chaudhry/dcsp_segmentation)<sub>BMVC2017</sub> | 58.6 | 59.2 | adversarial, TBD|
| [GuidedSeg](https://github.com/coallaoh/GuidedLabelling)<sub>CVPR2017</sub> | 55.7 | 56.7 | saliency, TBD|
| [BDSSW](https://github.com/ascust/BDWSS)<sub>CVPR2018</sub> | 63.0 | 63.9 | webly, filter+enhance|
| [WegSeg](https://arxiv.org/pdf/1803.09859.pdf)<sub>arxiv</sub> | 63.1 | 63.3 | webly(pure), Noise filter module|

#### COCO

## Others
see [this](https://github.com/JackieZhangdx/WeakSupervisedSegmentationList) for more lists and resources. 

## Reading List
#### adversarial
- [ ] generative adversial learning towards Fast weakly supervised detection
- [ ] Adversarial Complementary Learning for Weakly Supervised Object Localization
- [ ] Weakly Supervised Object Discovery by Generative Adversarial & Ranking Networks: arxiv
- [ ] Discovering Class-Specific Pixels for Weakly-Supervised Semantic Segmentation

####
- [ ] Learning to Segment Every Thing: semi-supervised, weight transfer function (from bbox parameters to mask parameters)
- [ ] Simple Does It: Weakly Supervised Instance and Semantic Segmentation: bbox-level, many methods, using graphcut, HED, MCG
- [ ] Multi-Evidence Filtering and Fusion for Multi-Label Classification, Object Detection and Semantic Segmentation Based on Weakly Supervised Learning: tricky, curriculum learning: image level -> instance level -> pixel level
- [ ] Combining Bottom-Up, Top-Down, and Smoothness Cues for Weakly Supervised Image Segmentation: cvpr2017

#### generate
- [ ] ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans
- [ ] SeGAN: Segmenting and Generating the Invisible

#### webly
- [x] Weakly Supervised Semantic Segmentation Based on Web Image Cosegmentation: BMVC2017, training model using masks of web images which are generated by cosegmentation 
- [ ] Webly Supervised Semantic Segmentation: cvpr 2017
- [ ] Weakly Supervised Semantic Segmentation using Web-Crawled Videos: Kwak, cvpr2017 
- [x] Bootstrapping the Performance of Webly Supervised Semantic Segmentation: target + web domain, target model filters web images, web model enhances target model
- [ ] Learning from Weak and Noisy Labels for Semantic Segmentation: TPAMI 2017
- [x] WebSeg: Learning Semantic Segmentation from Web Searches: arxiv, directly learning from keywork retrievaled web images. using saliency and region(MCG with edge)
- [x] STC: A Simple to Complex Framework for Weakly-supervised Semantic Segmentation: TPAMI 2017, Initial, Enhanced, Powerful three DCNN model. inital mask(generated by saliency and label using simple images) -> initial model -> enhanced mask(generated using simple images) -> Enhanced model -> powerful mask(generated using complex images) -> powerful model
  - saliency can not handle complex images, so BMVC2017 uses coseg instead
  
#### localization
- [ ] Adversarial Complementary Learning for Weakly Supervised Object Localization

#### network +
- [ ] Learning a Discriminative Feature Network for Semantic Segmentation
- [ ] Fully Convolutional Adaptation Networks for Semantic Segmentation
- [ ] Learning to Adapt Structured Output Space for Semantic Segmentation
- [ ] Context Encoding for Semantic Segmentation
- [ ] Learned Shape-Tailored Descriptors for Segmentation
- [ ] Normalized Cut Loss for Weakly-Supervised CNN Segmentation

#### Saliency
- [x] Exploiting Saliency for Object Segmentation from Image Level Labels

#### urban

#### IJCAI2018(keywords: segmentation, localization)
- [ ] DEL: Deep Embedding Learning for Efficient Image Segmentation
- [ ] Annotation-Free and One-Shot Learning for Instance Segmentation of Homogeneous Object Clusters
- [ ] ~~MEnet: A Metric Expression Network for Salient Object Segmentation~~
- [ ] Co-attention CNNs for Unsupervised Object Co-segmentation
- [ ] Coarse-to-fine Image Co-segmentation with Intra and Inter Rank Constraints
- [ ] ~~Virtual-to-Real: Learning to Control in Visual Semantic Segmentation~~
- [ ] ~~Centralized Ranking Loss with Weakly Supervised Localization for Fine-Grained Object Retrieval~~


