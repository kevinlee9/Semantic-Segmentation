# Semantic-Segmentation
List of useful codes and papers for semantic segmentation(mainly weakly)

## code

[pytorch-segmentation-detection](https://github.com/warmspringwinds/pytorch-segmentation-detection) a library for dense inference and training of Convolutional Neural Networks, 68.0%

[rdn](https://github.com/fyu/drn) Dilated Residual Networks, 75.6%, may be the best available semantic segmentation in PyTorch?

[Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) A pytorch implementation of Detectron. Both training from scratch and inferring directly from pretrained Detectron weights are available. only for coco now

[AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg) Adversarial Learning for Semi-supervised Semantic Segmentation.  heavily borrowed from a **pytorch DeepLab** implementation ([Link](https://github.com/speedinghzl/Pytorch-Deeplab))

[PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet) PyTorch implementation of ENet

[tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet) Tensorflow implementation of deeplab-resnet(deeplabv2, resnet101-based): complete and detailed

[tensorflow-deeplab-lfov](https://github.com/DrSleep/tensorflow-deeplab-lfov) Tensorflow implementation of deeplab-LargeFOV(deeplabv2, vgg16-based): complete and detailed

[resnet38](https://github.com/itijyou/ademxapp)  Wider or Deeper: Revisiting the ResNet Model for Visual Recognition: implemented using MXNET

[pytorch_deeplab_large_fov](https://github.com/BardOfCodes/pytorch_deeplab_large_fov): deeplab v1

[pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet)DeepLab resnet v2 model in pytorch


#### following 
[BDWSS](https://github.com/ascust/BDWSS) Bootstrapping the Performance of Webly Supervised Semantic Segmentation

[psa](https://github.com/jiwoon-ahn/psa) Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation

[AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg) Adversarial Learning for Semi-Supervised Semantic Segmentation

##### SEC
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

#### Analysis
image level to pixel wise labeling: from theory to practice: IJCAI 2018 analysis the effectiveness of class-level labels for segmentation(GT, predicted)
Attention based Deep Multiple Instance Learning: ICML 2018. cam from MIL perspective view
 
## Top works
#### PASCAL VOC2012

| method | val | test       |  notes |
| ------------ | ---------- | ---------- | ---------- |
| [GraphPartition](http://mftp.mmcheng.net/Papers/18ECCVGraphPartition.pdf)<sub>ECCV2018</sub> | 63.6 | 64.5 | TBD |
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
see [this](https://github.com/mrgloom/awesome-semantic-segmentation) for more implementations

## Reading List
#### generative adversarial 
- [ ] **Deep dual learning for semantic image segmentation**:CVPR2017, image translation
- [x] Semantic Segmentation using Adversarial Networks, NIPS2016 workshop
  - add gan loss branch, Segnet as generator, D: GT mask or predicted mask
- [x] Adversarial Learning for Semi-Supervised Semantic Segmentation: BMVC2018
  - semi supervised: SegNet as G, FCN-type D(discriminate each location), use output of D as psedo label for unlabeled data
- [x] Semi and weakly Supervised Semantic Segmentation Using Generative Adversarial Network: ICCV2017, use SegNet as D, treat fake as new class
  - weakly, use conditionalGan, pixel-level, image-level, generated data are included in loss. performance boosts less when increasing fully data
- [ ] generative adversarial learning towards Fast weakly supervised detection: CVPR2018


#### context 
- [ ] Context Encoding for Semantic Segmentation: CVPR2018
- [ ] The Role of Context for Object Detection and Semantic Segmentation in the Wild: CVPR2014
- [ ] Objects as Context for Detecting Their Semantic Parts: CVPR2018
- [ ] Exploring context with deep structured models for semantic segmentation: TPAMI2017

#### graph
- [ ] Associating Inter-Image Salient Instances for Weakly Supervised Semantic Segmentation: ECCV2018

#### scene understanding
- [ ] ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans
- [ ] SeGAN: Segmenting and Generating the Invisible

#### webly
- [x] Weakly Supervised Semantic Segmentation Based on Web Image Cosegmentation: BMVC2017, training model using masks of web images which are generated by cosegmentation 
- [ ] Webly Supervised Semantic Segmentation: CVPR2017
- [ ] Weakly Supervised Semantic Segmentation using Web-Crawled Videos: Kwak, CVPR2017 
- [x] Bootstrapping the Performance of Webly Supervised Semantic Segmentation: target + web domain, target model filters web images, web model enhances target model
- [ ] Learning from Weak and Noisy Labels for Semantic Segmentation: TPAMI2017
- [x] WebSeg: Learning Semantic Segmentation from Web Searches: arxiv, directly learning from keywork retrievaled web images. using saliency and region(MCG with edge)
- [x] STC: A Simple to Complex Framework for Weakly-supervised Semantic Segmentation: TPAMI 2017, Initial, Enhanced, Powerful three DCNN model. inital mask(generated by saliency and label using simple images) -> initial model -> enhanced mask(generated using simple images) -> Enhanced model -> powerful mask(generated using complex images) -> powerful model
  - saliency can not handle complex images, so BMVC2017 uses coseg instead

#### adversarial(framework)
- [ ] Adversarial Complementary Learning for Weakly Supervised Object Localization
- [ ] Weakly Supervised Object Discovery by Generative Adversarial & Ranking Networks: arxiv
- [ ] Discovering Class-Specific Pixels for Weakly-Supervised Semantic Segmentation
- [ ] ~~Adversarial Examples for Semantic Image Segmentation: ICLR2017 workshop, adversarial attack~~
- [ ] ~~On the Robustness of Semantic Segmentation Models to Adversarial Attacks: CVPR2018, adversarial attack~~

#### region
- [ ] Region-Based Convolutional Networks for Accurate Object Detection and Segmentation
- [ ] Simultaneous Detection and Segmentation, 2014
- [ ] Feedforward semantic segmentation with zoom-out features: 2015

#### network +
- [ ] Learning a Discriminative Feature Network for Semantic Segmentation
- [ ] Fully Convolutional Adaptation Networks for Semantic Segmentation
- [ ] Learning to Adapt Structured Output Space for Semantic Segmentation
- [ ] Learned Shape-Tailored Descriptors for Segmentation
- [ ] Normalized Cut Loss for Weakly-Supervised CNN Segmentation
- [x] Semantic Segmentation with Reverse Attention: BMVC2017, add reverse branch, predict the probability of pixel that doesn't belong to the corresponding class. and use attention to combine origin and reverse branch 

#### Saliency
- [x] Exploiting Saliency for Object Segmentation from Image Level Labels: CVPR2017

#### urban

#### affinity
- [x] Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation: image-level, semantic affinity, learn a **network** to predict affinity
- [x] Adaptive Affinity Field for Semantic Segmentation: ECCV2018, semantic affinity. add a pairwise term in seg **loss**(similarity metric: KL divergence), use an adversarial method to determine optimal neighborhood size

#### other useful
- [ ] Learning to Segment Every Thing: semi-supervised, weight transfer function (from bbox parameters to mask parameters)
- [ ] Simple Does It: Weakly Supervised Instance and Semantic Segmentation: bbox-level, many methods, using graphcut, HED, MCG
- [ ] Multi-Evidence Filtering and Fusion for Multi-Label Classification, Object Detection and Semantic Segmentation Based on Weakly Supervised Learning: tricky, curriculum learning: image level -> instance level -> pixel level
- [ ] Combining Bottom-Up, Top-Down, and Smoothness Cues for Weakly Supervised Image Segmentation: CVPR2017

#### application(pixel manipulation)
- [x] SeGAN: Segmenting and Generating the Invisible: CVPR2018, generate occluded parts
- [x] Learning Hierarchical Semantic Image Manipulation through Structured Representations: NIPS2018, manipulate image on object-level by modify bbox

#### IJCAI2018(keywords: segmentation, localization)
- [ ] DEL: Deep Embedding Learning for Efficient Image Segmentation
- [ ] Annotation-Free and One-Shot Learning for Instance Segmentation of Homogeneous Object Clusters
- [ ] ~~MEnet: A Metric Expression Network for Salient Object Segmentation~~
- [ ] Co-attention CNNs for Unsupervised Object Co-segmentation
- [ ] Coarse-to-fine Image Co-segmentation with Intra and Inter Rank Constraints
- [ ] ~~Virtual-to-Real: Learning to Control in Visual Semantic Segmentation~~
- [ ] ~~Centralized Ranking Loss with Weakly Supervised Localization for Fine-Grained Object Retrieval~~
- [x] Image-level to Pixel-wise Labeling: From Theory to Practice: fully, analysis the effect of image labels on seg results. add a generator(recover original image). image label(binary, use a threshold small than 0.5, eg:0.25)


#### ECCV2018


## Methods
- refine seg results using image-level labels
- multi-label classification branch(BDWSS)
- generative branch(to original image) 
- crf

## Common analysis
- ablation study
- sensitivity analysis

