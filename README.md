# Semantic-Segmentation
List of useful codes and papers for semantic segmentation(mainly weakly)

- [Semantic-Segmentation](#semantic-segmentation)
  * [Top Works](#top-works)
      - [PASCAL VOC2012](#pascal-voc2012)
      - [COCO](#coco)
  * [Codes](#codes)
  * [Others](#others)
      - [tutorial](#tutorial)
      - [priors](#priors)
      - [diffusion](#diffusion)
      - [analysis](#analysis)
      - [post processing](#post-processing)
      - [common methods](#common-methods)
  * [Reading List](#reading-list)
      - [context](#context)
      - [graph](#graph)
      - [webly](#webly)
      - [Saliency](#saliency)
      - [localization](#localization)
      - [spp](#spp)
      - [affinity](#affinity)
      - [region](#region)
      - [network](#network)
      - [regularizer](#regularizer)
      - [evaluation measure](#evaluation-measure)
      - [architecture](#architecture)
      - [generative adversarial](#generative-adversarial)
      - [scene understanding](#scene-understanding)
      - [other useful](#other-useful)
      - [application](#application)
  * [Related Tasks](#related-tasks)
      - [Few-shot segmentation](#few-shot-segmentation)
      - [Weakly-supervised Instance Segmentation](#weakly-supervised-instance-segmentation)
      - [Weakly-supervised Panoptic Segmentation](#weakly-supervised-panoptic-segmentation)

            
## Top Works
#### PASCAL VOC2012

| method | val | test       |  notes |
| ------------ | ---------- | ---------- | ---------- |
| [DSRG](https://github.com/speedinghzl/DSRG)<sub>CVPR2018</sub> | 61.4 | 63.2 | deep seeded region growing, resnet-lfov\|vgg-aspp  |
| [psa](https://github.com/jiwoon-ahn/psa)<sub>CVPR2018</sub> | 61.7 | 63.7 | pixel affinity network, resnet38 |
| [MDC](https://arxiv.org/pdf/1805.04574.pdf)<sub>CVPR2018</sub> | 60.4 | 60.8 | multi-dilated convolution, vgg-lfov |
| [MCOF](http://3dimage.ee.tsinghua.edu.cn/wx/mcof)<sub>CVPR2018</sub> | 60.3 | 61.2 | iterative, RegionNet(sppx), resnet-lfov |
| [GAIN](https://arxiv.org/abs/1802.10171.pdf)<sub>CVPR2018</sub> |  55.3 |  56.8 | |
| [DCSP](https://github.com/arslan-chaudhry/dcsp_segmentation)<sub>BMVC2017</sub> | **58.6** | **59.2** | adversarial for saliency, and generate cues by cam+saliency(harmonic mean)|
| [GuidedSeg](https://github.com/coallaoh/GuidedLabelling)<sub>CVPR2017</sub> | 55.7 | 56.7 | saliency, TBD|
| [BDSSW](https://github.com/ascust/BDWSS)<sub>CVPR2018</sub> | 63.0 | 63.9 | webly, filter+enhance|
| [WegSeg](https://arxiv.org/pdf/1803.09859.pdf)<sub>arxiv</sub> | 63.1 | 63.3 | webly(pure), Noise filter module|
| [SeeNet](https://arxiv.org/abs/1810.09821)<sub>NIPS2018</sub> | 63.1 | 62.8 | based on DCSP |
| [Graph](http://mftp.mmcheng.net/Papers/18ECCVGraphPartition.pdf)<sub>ECCV2018</sub> | 63.6 | 64.5 | graph partition|
| [Graph](http://mftp.mmcheng.net/Papers/18ECCVGraphPartition.pdf)<sub>ECCV2018</sub> | 64.5 | 65.6 | use simple ImageNet dataset additionally|
| [CIAN](https://arxiv.org/abs/1811.10842)<sub>CVPR2019</sub> | 64.1 | 64.7 | cross image affinity network|
| [FickleNet](https://arxiv.org/abs/1811.10842)<sub>CVPR2019</sub> | **64.9** | **65.3** | use dropout (a generalization of dilated convolution)|

#### COCO


## Codes

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

[DeepLab-ResNet-Pytorch](https://github.com/speedinghzl/Pytorch-Deeplab) Deeplab v3 model in pytorch, 

[BDWSS](https://github.com/ascust/BDWSS) Bootstrapping the Performance of Webly Supervised Semantic Segmentation

[psa](https://github.com/jiwoon-ahn/psa) Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation

[DSRG](https://github.com/speedinghzl/DSRG): Caffe, CAM and DRFI provided 

SEC
  - [original](https://github.com/kolesman/SEC): Caffe  
  - [BDSSW](https://github.com/ascust/BDWSS): MXNET
  - [SEC-tensorflow](https://github.com/xtudbxk/SEC-tensorflow): tensorflow  

## Others
see [this](https://github.com/JackieZhangdx/WeakSupervisedSegmentationList) for more weakly lists and resources.  
see [this](https://github.com/wutianyiRosun/Segmentation.X) for more semantic/instance/panoptic/video segmentation lists and resources.
see [this](https://github.com/mrgloom/awesome-semantic-segmentation) for more implementations  
a good architecture summary paper:[Learning a Discriminative Feature Network for Semantic Segmentation](https://arxiv.org/pdf/1804.09337.pdf)
#### tutorial
- Unsupervised Visual Learning Tutorial. *CVPR 2018* [[part 1]](https://www.youtube.com/watch?v=gSqmUOAMwcc) [[part 2]](https://www.youtube.com/watch?v=BijK_US6A0w)
- Weakly Supervised Learning for Computer Vision. *CVPR 2018* [[web]](https://hbilen.github.io/wsl-cvpr18.github.io/) [[part 1]](https://www.youtube.com/watch?v=bXfZFmE8cjo) [[part 2]](https://www.youtube.com/watch?v=FetNp6f19IM)

#### priors
- Superpixels: An Evaluation of the State-of-the-Art [link](https://github.com/davidstutz/superpixel-benchmark)
- Learning Superpixels with Segmentation-Aware Affinity Loss[link](http://jankautz.com/publications/LearningSuperpixels_CVPR2018.pdf)
- Superpixel based Continuous Conditional Random Field Neural Network for Semantic Segmentation [link](https://www.sciencedirect.com/science/article/pii/S0925231219300281)

#### diffusion
Learning random-walk label propagation for weakly-supervised semantic segmentation: scribble

Convolutional Random Walk Networks for Semantic Image Segmetation: fully, affinity branch(low level)

Soft Proposal Networks for Weakly Supervised Object Localization: attention, semantic affinity

Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation: image-level, semantic affinity

#### analysis
image level to pixel wise labeling: from theory to practice: IJCAI 2018 analysis the effectiveness of class-level labels for segmentation(GT, predicted)
Attention based Deep Multiple Instance Learning: ICML 2018. CAM from MIL perspective view

#### post processing
listed in : [Co-attention CNNs for Unsupervised Object Co-segmentation](https://www.csie.ntu.edu.tw/~cyy/publications/papers/Hsu2018CAC.pdf)
- Otsu’s method
- GrabCut
- CRF    

#### common methods
- refine segmentation results using image-level labels
- multi-label classification branch(BDWSS)
- generative branch(to original image) 
- crf

## Under Review
- [ ] [Gated CRF Loss for Weakly Supervised Semantic Image Segmentation](https://arxiv.org/abs/1906.04651)
- [ ] [Closed-Loop Adaptation for Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/1905.12190)
- [ ] [Harvesting Information from Captions for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/1905.06784)
- [ ] [Consistency regularization and CutMix for semi-supervised semantic segmentation](https://arxiv.org/abs/1906.01916)
- [ ] [Zero-shot Semantic Segmentation](https://arxiv.org/abs/1906.00817)


## Reading List

#### context 
- [x] Context Encoding for Semantic Segmentation: CVPR2018. use TEN
- [ ] The Role of Context for Object Detection and Semantic Segmentation in the Wild: CVPR2014
- [ ] Objects as Context for Detecting Their Semantic Parts: CVPR2018
- [ ] Exploring context with deep structured models for semantic segmentation: TPAMI2017
- [ ] dilated convolution
- [ ] Deep TEN: Texture encoding network !!: CVPR2017. A global context vector, pooled from all spatial positions, can be concatenated to local features
- [ ]  Refinenet: Multi-path refinement networks for high-resolution semantic segmentation: CVPR2017. local features across different scales can be fused to encode global context
- [x] Non-local neural networks: CVPR2018. a densely connected graph with pairwise edges between all pixels

#### graph
- [ ] Associating Inter-Image Salient Instances for Weakly Supervised Semantic Segmentation: ECCV2018

#### bbox-level
Box-driven Class-wise Region Masking and Filling Rate Guided Loss for Weakly Supervised Semantic Segmentation, CVPR2019

#### webly
- [x] Weakly Supervised Semantic Segmentation Based on Web Image Cosegmentation: BMVC2017, training model using masks of web images which are generated by cosegmentation 
- [ ] Webly Supervised Semantic Segmentation: CVPR2017
- [x] Weakly Supervised Semantic Segmentation using Web-Crawled Videos: CVPR2017, learns a class-agnostic decoder(attention map -> binary mask), pseudo masks are generated from video frames by solving a graph-based optimization problem. 
- [x] Bootstrapping the Performance of Webly Supervised Semantic Segmentation: target + web domain, target model filters web images, refine mask by combine target and web masks.
- [ ] Learning from Weak and Noisy Labels for Semantic Segmentation: TPAMI2017
- [x] WebSeg: Learning Semantic Segmentation from Web Searches: arxiv, directly learning from keywork retrievaled web images. using saliency and region(MCG with edge)
- [x] STC: A Simple to Complex Framework for Weakly-supervised Semantic Segmentation: TPAMI 2017, Initial, Enhanced, Powerful three DCNN model. inital mask(generated by saliency and label using simple images) -> initial model -> enhanced mask(generated using simple images) -> Enhanced model -> powerful mask(generated using complex images) -> powerful model
  - saliency can not handle complex images, so BMVC2017 uses coseg instead

#### Saliency
- [x] Exploiting Saliency for Object Segmentation from Image Level Labels: CVPR2017
- [x] Discovering Class-Specific Pixels for Weakly-Supervised Semantic Segmentation: BMVC2017
  - combine saliency(off-shelf) and CAM to get cues, use harmonic mean function
  - adapt CAM from head of Segmentation Network
  - use erasing to get multiple objects' saliency

#### localization
- [x] Adversarial Complementary Learning for Weakly Supervised Object Localization, CVPR2018. two branchs, remove high activations from feature map. [code](https://github.com/xiaomengyc/ACoL)
- [x] [Tell me where to look: Guided Attention Inference Network](https://arxiv.org/pdf/1802.10171.pdf), CVPR2018. origin image soft erasing(CAM after sigmoid as attention) -> end2end training, force erased images have zero activation
- [x] Self-Erasing Network for Integral Object Attention， NIPS2018: prohibit attentions from spreading to unexpected background regions.
  - cam -> tenary mask(attention, background, potential)
  - self erasing only in attention + potential region(**sign flip in background region** instead of setting to 0 simply)
  - self produced psedo label for background region(difference to SPG: 1.psedo label for background and attention 2.supervise low layer)
- [x] Self-produced Guidance for Weakly-supervised Object localization, ECCV2018:
  - self supervised use top down framework, for single label classification prob. **add pixel-wise supervision when only have image level label**  
  - B1, B2 sharing
  - bottom guide top inversely(B1+B2 -> C)

#### spp
- [ ] Superpixel convolutional networks using bilateral inceptions
- [x] Learning Superpixels with Segmentation-Aware Affinity Loss: good intro for superpixel algs.

#### affinity
- [x] Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation: image-level, semantic affinity, learn a **network** to predict affinity
- [x] Adaptive Affinity Field for Semantic Segmentation: ECCV2018, semantic affinity. add a pairwise term in seg **loss**(similarity metric: KL divergence), use an adversarial method to determine optimal neighborhood size

#### region
- [ ] Region-Based Convolutional Networks for Accurate Object Detection and Segmentation
- [ ] Simultaneous Detection and Segmentation, 2014
- [ ] Feedforward semantic segmentation with zoom-out features: 2015

#### network
- [ ] Learned Shape-Tailored Descriptors for Segmentation
- [ ] Normalized Cut Loss for Weakly-Supervised CNN Segmentation
- [ ] Fully Convolutional Adaptation Networks for Semantic Segmentation
- [ ] Learning to Adapt Structured Output Space for Semantic Segmentation
- [x] Semantic Segmentation with Reverse Attention: BMVC2017, equally responses of multi classes(confusion in boudary region). add reverse branch, predict the probability of pixel that doesn't belong to the corresponding class. and use attention to combine origin and reverse branch 
- [x] Deep Clustering for Unsupervised Learning of Visual Features, ECCV2018. use assignments of knn as supervision to update weights of network 
- [x] DEL: Deep Embedding Learning for Efficient Image Segmentation, IJCAI 2018. use spp embedding as init probs to do image segmentation
- [x] Learning a Discriminative Feature Network for Semantic Segmentation, CVPR2018, Smoother network: multi-scale+global context(FPN with channel atention), Broder Network: focal loss for boundary. [code?](https://github.com/YuhuiMa/DFN-tensorflow)
- [ ] Convolutional Simplex Projection Network for Weakly Supervised Semantic Segmentation: BMVC 2018
- [ ] Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation: CVPR2019

#### regularizer
- [ ] [Normalized Cut Loss for Weakly-Supervised CNN Segmentation](https://arxiv.org/pdf/1804.01346.pdf)
- [ ] [Regularized Losses for Weakly-supervised CNN Segmentation](https://github.com/meng-tang/rloss)

#### evaluation measure
- [ ] [Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation](https://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf)
- [ ] [The Lovasz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/pdf/1705.08790.pdf)
- [ ] [What is a good evaluation measure for semantic segmentation?](http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)

#### architecture
- [ ] The Devil is in the Decoders, BMVC2017
- [x] Dilated Residual Networks, CVPR2017. Dilated structure design for classification and localization.
- [x] Understanding Convolution for Semantic Segmentation, WACV2018. hybrid dilated convolution(2-2-2 -> 1-2-3)
- [x] Smoothed Dilated Convolutions for Improved Dense Prediction, KDD2018. separable and share conv(for smoothing) + dilated conv
- [x] Deeplab v1, v2, v3, v3+
- [ ] Learning Fully Dense Neural Networks for Image Semantic Segmentation, AAAI2019 

#### generative adversarial 
- [ ] **Deep dual learning for semantic image segmentation**:CVPR2017, image translation
- [x] Semantic Segmentation using Adversarial Networks, NIPS2016 workshop
  - add gan loss branch, Segnet as generator, D: GT mask or predicted mask
- [x] Adversarial Learning for Semi-Supervised Semantic Segmentation: BMVC2018
  - semi supervised: SegNet as G, FCN-type D(discriminate each location), use output of D as psedo label for unlabeled data
- [x] Semi and weakly Supervised Semantic Segmentation Using Generative Adversarial Network: ICCV2017, use SegNet as D, treat fake as new class
  - weakly, use conditionalGan, pixel-level, image-level, generated data are included in loss. performance boosts less when increasing fully data
- [ ] generative adversarial learning towards Fast weakly supervised detection: CVPR2018
- [x] Adaptive Affinity Field for Semantic Segmentation: ECCV2018, semantic affinity. add a pairwise term in seg **loss**(similarity metric: KL divergence), use an adversarial method to determine optimal neighborhood size

#### scene understanding
- [ ] ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans
- [ ] SeGAN: Segmenting and Generating the Invisible

#### other useful
- [ ] Learning to Segment Every Thing: semi-supervised, weight transfer function (from bbox parameters to mask parameters)
- [ ] Simple Does It: Weakly Supervised Instance and Semantic Segmentation: bbox-level, many methods, using graphcut, HED, MCG
- [ ] Multi-Evidence Filtering and Fusion for Multi-Label Classification, Object Detection and Semantic Segmentation Based on Weakly Supervised Learning: tricky, curriculum learning: image level -> instance level -> pixel level
- [ ] Combining Bottom-Up, Top-Down, and Smoothness Cues for Weakly Supervised Image Segmentation: CVPR2017
- [x] Improving Weakly-Supervised Object Localization By Micro-Annotation: BMVC2016, object classes always co-occur with same background elements(boat, train). propose a new annotation method. add human annotations to improve localization results of CAM, annotating based on clusters of dense features. each class uses a spectral clustering.(CAM has problem)
- [x] Co-attention CNNs for Unsupervised Object Co-segmentation: IJCAI 2018
- [ ] Coarse-to-fine Image Co-segmentation with Intra and Inter Rank Constraints, IJCAI2018
- [ ] Annotation-Free and One-Shot Learning for Instance Segmentation of Homogeneous Object Clusters, IJCAI2018
- [x] Image-level to Pixel-wise Labeling: From Theory to Practice: fully, analysis the effect of image labels on seg results. add a generator(recover original image). image label(binary, use a threshold small than 0.5, eg:0.25), IJCAI2018

#### application
- [x] SeGAN: Segmenting and Generating the Invisible: CVPR2018, generate occluded parts
- [x] Learning Hierarchical Semantic Image Manipulation through Structured Representations: NIPS2018, manipulate image on object-level by modify bbox

## Related Tasks
#### Few-shot segmentation
- [ ] One-shot learning for semantic segmentation, BMVC2017
- [ ] Conditional networks for few-shot semantic segmentation, ICLR2018 Workshop
- [ ] Few-Shot Segmentation Propagation with Guided Networks, preprint
- [ ] Few-Shot Semantic Segmentation with Prototype Learning, BMVC2018
- [ ] CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning, CVPR2019
- [ ] One-Shot Segmentation in Clutter, ICML 2018

#### Weakly-supervised Instance Segmentation
- [ ] Weakly Supervised Instance Segmentation using Class Peak Response, CVPR2018
- [ ] Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations, CVPR2019
- [ ] Object Counting and Instance Segmentation with Image-level Supervision, CVPR2019

#### Weakly-supervised Panoptic Segmentation
- [ ] Weakly- and Semi-Supervised Panoptic Segmentation, ECCV2018
