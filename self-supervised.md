## Self-Supervised Learning

#### Image

Rotation: predicting rotation degree

Exemplar: each image correspond to one class, use triplet loss 

Jigsaw:  recover relative spatial position of 9 randomly sampled image patches after a random permutation

Relative Patch Location: predicting the relative location of two given patches of an image. 



#### Video

cycle between tracking frame patches in same video

cycle between frames in similar videos (most similar frame of frame a in video A is frame b is video B, then then most similiar frame of frame B should be frame A correspondingly )

triplet between time-near, faraway frames and anchor frame among same video 

