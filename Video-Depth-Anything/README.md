&nbsp;

<div align="center">

<p align="center"> <img src="../img/logo.png" width="250px"> </p>

[![arXiv](https://img.shields.io/badge/paper-arxiv-179bd3)](https://arxiv.org/abs/2411.14384)
[![project](https://img.shields.io/badge/project-page-green)](https://caiyuanhao1998.github.io/project/OmniVCus/)
[![hf](https://img.shields.io/badge/hugging-face-green)](https://huggingface.co/datasets/CaiYuanhao/OmniVCus)
<h3>OmniVCus: Feedforward Subject-driven Video <br> Customization with Multimodal Control Conditions</h3> 

<p align="center">
  <img src="../img/demo_1.png" width="45%" alt="1">
  <img src="../img/demo_2.png" width="45%" alt="2">
  <img src="../img/demo_1.gif" width="45%" alt="1">
  <img src="../img/demo_2.gif" width="45%" alt="2">
</p>


&nbsp;

</div>



### Introduction
This part of code implements the video depth control condition construction. If you find our repo useful, please give it a star ⭐ and consider citing our paper. Thank you :)


### 1. Installing Environment

```bash
conda activate videocus_factory
pip install -r requirements.txt
```

### 2. Downloading Model

Download the checkpoints listed [here](#pre-trained-models) and put them under the `checkpoints` directory.

```sh
bash get_weights.sh
```

### 3. Processing Data
We write a dara parallel version for your convinience to process the data using 8 nodes with 8 GPUs per node as

```sh
. process_data_depth.sh --part_id 0
. process_data_depth.sh --part_id 1
. process_data_depth.sh --part_id 2
. process_data_depth.sh --part_id 3
. process_data_depth.sh --part_id 4
. process_data_depth.sh --part_id 5
. process_data_depth.sh --part_id 6
. process_data_depth.sh --part_id 7
```

### 4. Citation

If you find this project useful, please consider citing our paper:

```sh
@inproceedings{omnivcus,
  title={OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions},
  author={Yuanhao Cai and He Zhang and Xi Chen and Jinbo Xing and Kai Zhang and Yiwei Hu and Yuqian Zhou and Zhifei Zhang and Soo Ye Kim and Tianyu Wang and Yulun Zhang and Xiaokang Yang and Zhe Lin and Alan Yuille},
  booktitle={NeurIPS},
  year={2025}
}
```