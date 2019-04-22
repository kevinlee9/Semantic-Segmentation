## CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning
![CANEt](/home/zhikang/src/python/Semantic-Segmentation/paper_reports/images/CANEt.png)



#### Iterative Optimization module

use last iteration predicted probability maps and input features (concat) to predict current masks in a residual form, 

predicted map has $p$ probability to set to be zero (resist over-fitting in iterative optimization)









