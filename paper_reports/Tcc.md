## Temporal Cycle-Consistency Learning



![tcc](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/tcc.png)

matching in mid-level feature

cycle consistency in videos

![tcc_dig](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/tcc_dig.png)

#### Cycle-back LOSS

###### classification

Given the selected point $u_i$

cycle-forward: soft nearest neighbor: ![tcc_eq1](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/tcc_eq1.png)

cycle-back: use distance as logits:  ![tcc_eq2](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/tcc_eq2.png)     ![tcc_eq3](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/tcc_eq3.png)



###### regression

penalize the model less if cycle-back frame is near the anchor frame

a similarity vector $\beta$ along $u_i$

![tcc_eq4](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/tcc_eq4.png)

Give $\beta$ a Gaussian prior, center is position of anchor frame i

![tcc_eq5](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/tcc_eq5.png)

where ![tcc_eq6](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/tcc_eq6.png)

or only minimize mean

![tcc_eq7](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/tcc_eq7.png)