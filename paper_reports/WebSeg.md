## WebSeg: Learning Semantic Segmentation from Web Searches

- use low level cues as ground truth: regions, saliency

  - regions are get using MCG on **edges maps**

  - saliency use DSS

- filter GT by a region net

![WegSeg](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/WegSeg.png)

#### complexity image measure

drop web crawled complex images

-  variance of Laplace
- saturation / brightness

#### proxy ground truth 

Region + Saliency

#### noise filtering module

labels: region probs

network: spp pooling network