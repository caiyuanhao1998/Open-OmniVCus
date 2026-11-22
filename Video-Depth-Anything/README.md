&nbsp;

<div align="center">

<p align="center"> <img src="../img/logo.png" width="250px"> </p>

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2506.23361)&nbsp;
[![project page](https://img.shields.io/badge/project-page-green)](https://caiyuanhao1998.github.io/project/OmniVCus/)&nbsp;
[![知乎 - 解读](../img/zhihu-interpretation.svg)](https://zhuanlan.zhihu.com/p/1988577914237785212)&nbsp;
[![Hugging Face Train Data](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20HF&message=Train%20Data&color=yellow)](https://huggingface.co/datasets/CaiYuanhao/OmniVCus-Train)&nbsp;
[![Hugging Face Test Data](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20HF&message=Test%20Data&color=yellow)](https://huggingface.co/datasets/CaiYuanhao/OmniVCus-Test)&nbsp;
[![Hugging Face Model](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20HF&message=Model&color=yellow)](https://huggingface.co/CaiYuanhao/OmniVCus)&nbsp;
[![Online Demo](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Online&message=Demo&color=yellow)](https://huggingface.co/spaces/CaiYuanhao/OmniVCus)&nbsp;

<h4>[NeurIPS 2025] OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions</h4> 

<p style="font-size: 20px;">
  Applied in the post-training of
  <img src="https://commons.wikimedia.org/wiki/Special:Redirect/file/Adobe_Corporate_Horizontal_Red_HEX.svg"
      alt="Adobe"
      style="height: 12px !important; width: auto !important; max-width: none !important; display: inline-block !important; vertical-align: middle;">
  <strong>Adobe Firefly Video Foundation Model</strong>
</p>

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
