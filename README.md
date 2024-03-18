<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> A Hard-to-Beat Baseline for Training-free CLIP-Based Adaptation </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://zhengbo.wang/" target="_blank" style="text-decoration: none;">Zhengbo Wang<sup>1,2</sup></a>&nbsp;,&nbsp;
    <a href="https://liangjian.xyz/" target="_blank" style="text-decoration: none;">Jian Liang<sup>2,3†</sup></a>&nbsp;,&nbsp;
    <a href="https://tomsheng21.github.io/" target="_blank" style="text-decoration: none;">Lijun Sheng<sup>1,2</sup></a>
    <a href="https://sites.google.com/site/pinyuchenpage" target="_blank" style="text-decoration: none;">Ran He<sup>2,3</sup></a>&nbsp;,&nbsp;
    <a href="https://www.princeton.edu/~pmittal/" target="_blank" style="text-decoration: none;">Zilei Wang<sup>1</sup></a>&nbsp;,&nbsp; 
	<a href="https://www.peterhenderson.co/" target="_blank" style="text-decoration: none;">Tieniu Tan<sup>4</sup></a>&nbsp;&nbsp;
	<br>
<sup>1</sup>University of Science and Technology of China&nbsp;&nbsp;&nbsp;
<sup>2</sup>CRIPAC & MAIS, Institute of Automation, Chinese Academy of Sciences&nbsp;&nbsp;&nbsp;
<sup>3</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences&nbsp;&nbsp;&nbsp;
<sup>4</sup>Nanjing University 
</p>

<p align='center';>
<b>
<em>ICLR, 2024</em> <br>
</b>
</p>
<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://openreview.net/forum?id=Js5PJPHDyY" target="_blank" style="text-decoration: none;">[Paper]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://zhengbo.wang/ICLR24" target="_blank" style="text-decoration: none;">[Project Page]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://github.com/mrflogs/ICLR24" target="_blank" style="text-decoration: none;">[Code]</a>
</b>
</p>





## Requirements
### Installation
Create a conda environment and install dependencies:
```
conda create -n h2b python=3.9
conda activate h2b

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### Dataset
Follow [DATASET.md](DATASET.md) to install ImageNet and other datasets referring to [CoOp](https://github.com/KaiyangZhou/CoOp).

## Get Started
### Configs
The running configurations can be modified in `configs/setting/dataset.yaml`, including evaluation setting, shot numbers, visual encoders, and hyperparamters. 

### Numerical Results
We provide  **numerical results** in few-shot classification in Figure 1 at [exp.log](exp.log).

### Running
For few-shot classification:
```bash
CUDA_VISIBLE_DEVICES=0 python main_few_shots.py --config configs/few_shots/dataset.yaml
```
For base-to-new generalization:
```bash
CUDA_VISIBLE_DEVICES=0 python main_base2new.py --config configs/base2new/dataset.yaml
```

For out-of-distribution generalizaiton:

```bash
CUDA_VISIBLE_DEVICES=0 python main_robustness.py --config configs/robustness/imagenet_rn50.yaml
```



## Acknowledgement

This repo benefits from [CLIP](https://github.com/openai/CLIP), [CoOp,](https://github.com/KaiyangZhou/Dassl.pytorch) and [SHIP](https://github.com/mrflogs/SHIP). Thanks for their wonderful work.

## Citation
```latex
@inproceedings{wang2024baseline,
  title={A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation},
  author={Wang, Zhengbo and Liang, Jian and Sheng, Lijun and He, Ran and Wang, Zilei and Tan, Tieniu},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Contact

If you have any question, feel free to contact zhengbowang@mail.ustc.edu.cn.