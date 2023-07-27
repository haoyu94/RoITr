# Rotation-Invariant Transformer for Point Cloud Matching (CVPR 2023)

PyTorch implementation of the paper:

[Rotation-Invariant Transformer for Point Cloud Matching](https://arxiv.org/abs/2303.08231) by:

[Hao Yu](https://scholar.google.com/citations?hl=en&user=g7JfRn4AAAAJ), [Zheng Qin](https://scholar.google.com/citations?user=DnHBAN0AAAAJ&hl=en), [Ji Hou](https://sekunde.github.io/), [Mahdi Saleh](https://scholar.google.com/citations?user=52yLUy0AAAAJ&hl=en), [Dongsheng Li](https://scholar.google.com/citations?user=_WrK108AAAAJ&hl=en), [Benjamin Busam](https://scholar.google.com/citations?user=u4rJZwUAAAAJ&hl=en) and [Slobodan Ilic](https://scholar.google.com/citations?user=ELOVd8sAAAAJ&hl=en&oi=ao).

## Introduction

The intrinsic rotation invariance lies at the core of matching point clouds with handcrafted descriptors. However, it is widely despised by recent deep matchers that obtain the rotation invariance extrinsically via data augmentation. As the finite number of augmented rotations can never span the continuous SO(3) space, these methods usually show instability when facing rotations that are rarely seen. To this end, we introduce RoITr, a **Ro**tation-**I**nvariant **Tr**ansformer to cope with the pose variations in the point cloud matching task. We contribute both on the local and global levels.
Starting from the local level, we introduce an attention mechanism embedded with Point Pair Feature (PPF)-based coordinates to describe the pose-invariant geometry, upon which a novel attention-based encoder-decoder architecture is constructed. We further propose a global transformer with rotation-invariant cross-frame spatial awareness learned by the self-attention mechanism, which significantly improves the feature distinctiveness and makes the model robust with respect to the low overlap. Experiments are conducted on both the rigid and non-rigid public benchmarks, where RoITr outperforms all the state-of-the-art models by a considerable margin in the low-overlapping scenarios. Especially when the rotations are enlarged on the challenging 3DLoMatch benchmark, RoITr surpasses the existing methods by at least 13 and 5 percentage points in terms of *Inlier Ratio* and *Registration Recall*, respectively.

![image](https://github.com/haoyu94/haoyu94.github.io/blob/main/images/RoITr.png)


## Installation

+ Clone the repository:

  ```
  git clone https://github.com/haoyu94/RoITr.git
  cd RoITr
  ```
+ Create conda environment and install requirements:

  ```
  conda env create -f dependencies.yaml
  pip install -r requirements.txt
  ```
+ Compile C++ and CUDA scripts:

  ```
  cd cpp_wrappers
  cd pointops
  python setup.py install
  cd ..
  cd ..
  ```

## 3DMatch & 3DLoMatch

### Pretrained model
   Pretrained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1rlONbyisPZf0Ua0KzIxKG9qGAdUT2sUh/view?usp=sharing).
   
   Put the downloaded model under `./weights/`. 
   
### Prepare datasets

  Please follow [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences) for preparing the 3DMatch data and put it under `./data/`.
  
### Train

  ```
  python main.py configs/train/tdmatch.yaml
  ```
  
### Test
#### Original Benchmarks (default: 3DLoMatch)
  + Point correspondences are first extracted by running:
  
  ```
  python main.py configs/test/tdmatch.yaml
  ```
  
  and stored on `snapshot/tdmatch_ripoint_transformer_test/3DLoMatch/`. 
  
  
  + To evaluate on 3DMatch, please change the `benchmark` keyword in `configs/test/tdmatch.yaml` from `3DLoMatch` to  `3DMatch`.
  
  + The evaluation of extracted correspondences and relative poses estimated by RANSAC can be done by running:

  ```
  sh scripts/benchmark_registration_3dlomatch_c2f.sh
  ```

  and

  ```
  sh scripts/benchmark_registration_3dmatch_c2f.sh
  ```
  for 3DLoMatch and 3DMatch, respectively.
  
  + The final results are stored in `est_traj/3DMatch/{number of correspondences}/result`. 

#### Rotated Benchmarks (default: 3DLoMatch)

 + Change the keyword `rotated` in `configs/test/tdmatch.yaml` to `True`.
   
 + Generate correspondences by running:
   ```
   python main.py configs/test/tdmatch.yaml
   ```
  
 + Evaluation is done through:

   ```
   python registration/evaluate_registration_c2f_rotated.py --source_path {path to generated correspondences} --benchmark {3DMatch or 3DLoMatch} --n_points {250, 500, 1000, 2500, 5000}
   ```

   for example:

   ```
   python registration/evaluate_registration_c2f_rotated.py --source_path snapshot/tdmatch_ripoint_transformer_test/3DLoMatch --benchmark 3DLoMatch --n_points 2500
   ```

 
 ## 4DMatch
 
### Pretrained model

 Pretrained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1VQBkfh8R9WSkT5PiqTwqkxbNTvFhAaaj/view?usp=sharing).
   
 Put the downloaded model under `./weights/`. 

### Prepare datasets

 Please follow [Lepard](https://github.com/rabbityl/lepard) for preparing the 4DMatch data and put it under `./data/`.
 
### Train

  ```
  python main.py configs/train/fdmatch.yaml
  ```
  
### Test (default: 4DLoMatch)
  + Point correspondences are first extracted by running:
  
  ```
  python main.py configs/test/fdmatch.yaml
  ```
  
  and stored on `snapshot/fdmatch_ripoint_transformer_test/4DLoMatch/`. 
  
  
  + To evaluate on 4DMatch, please change the `split` and `benchmark` keywords in `configs/test/fdmatch.yaml` to `split/4DMatch` and `4DMatch`, respectively.
  
  + The evaluation of extracted correspondences can be done by running:

  ```
  python registration/evaluate_fdmatch.py
  ```

  The path in line 126 should list all the correspondence files, e.g., `./snapshot/fdmatch_ripoint_transformer_test/4DLoMatch/*.pth`.
  
 
 ## Reference

 + [GeoTransformer](https://github.com/qinzheng93/GeoTransformer).
 
 + [Lepard](https://github.com/rabbityl/lepard).
 
 + [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences). 
 
 + [Point Transformer](https://github.com/POSTECH-CVLab/point-transformer).

 + [RIGA](https://arxiv.org/abs/2209.13252)
 
 We thank the authors for their excellent works!
 
 ## Citiation
 
If you find this repository helpful, please cite:

```
@inproceedings{yu2023rotation,
  title={Rotation-invariant transformer for point cloud matching},
  author={Yu, Hao and Qin, Zheng and Hou, Ji and Saleh, Mahdi and Li, Dongsheng and Busam, Benjamin and Ilic, Slobodan},
  booktitle={CVPR},
  year={2023}
}
```
  

