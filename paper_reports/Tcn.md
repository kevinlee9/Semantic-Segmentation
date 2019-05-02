## Time-Contrastive Networks: Self-Supervised Learning from Video

self-supervised in a single video

**triplet loss**: frames near anchor are treated as positive samples, and frames far from anchor are treated as negative samples

The model trains itself by trying to answer the following questions simultaneously:

- What is common between the different-looking blue frames?
-  What is different between the similar-looking red and blue frames?

![tcn](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/tcn.png)