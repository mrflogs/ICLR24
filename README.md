# A hard-to-beat baseline for training-free CLIP-based adaptation
Official implementation of [A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation](https://openreview.net/forum?id=Js5PJPHDyY). 
This paper has been accepted by **ICLR 2024**.

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
Follow DATASET.md to install ImageNet and other datasets referring to CoOp.

## Get Started
### Configs
The running configurations can be modified in `configs/setting/dataset.yaml`, including evaluation setting, shot numbers, visual encoders, and hyperparamters. 

### Numerical Results
We provide  **numerical results** in few-shot classification in Figure 1 at exp.log.

### Running
For few-shot classification:
```bash
CUDA_VISIBLE_DEVICES=0 python main_few_shots.py --config configs/few_shots/dataset.yaml
```
For base-to-new generalization:
```bash
CUDA_VISIBLE_DEVICES=0 python main_base2new.py --config configs/base2new/dataset.yaml
```



## Acknowledgement

This repo benefits from [CLIP](https://github.com/openai/CLIP), [CoOp,](https://github.com/KaiyangZhou/Dassl.pytorch) and [SHIP](https://github.com/mrflogs/SHIP). Thanks for their wonderful work.

## Citation
```
@inproceedings{wang2024baseline,
  title={A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation},
  author={Zhengbo Wang and Jian Liang and Lijun Sheng and Ran He and Zilei Wang and Tieniu Tan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2024}
}
```

## Contact

If you have any question, feel free to contact zhengbowang@mail.ustc.edu.cn.