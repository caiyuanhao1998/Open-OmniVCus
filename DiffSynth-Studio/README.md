&nbsp;

<div align="center">

<p align="center"> <img src="../img/logo.png" width="250px"> </p>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2506.23361-b31b1b.svg)](https://arxiv.org/abs/2506.23361)&nbsp;
[![project page](https://img.shields.io/badge/Project_Page-Video_Results-green)](https://caiyuanhao1998.github.io/project/OmniVCus/)&nbsp;
<a href="https://huggingface.co/datasets/CaiYuanhao/OmniVCus-Train">
  <img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Train%20Data&color=yellow">
</a>
<a href="https://huggingface.co/datasets/CaiYuanhao/OmniVCus-Test">
  <img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Test%20Data&color=yellow">
</a>
<a href="https://huggingface.co/CaiYuanhao/OmniVCus">
  <img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=yellow">
</a>

<h4>[NIPS 25] OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions</h4>

<p align="center">
  <img src="../img/demo_1.png" width="45%" alt="1">
  <img src="../img/demo_2.png" width="45%" alt="2">
  <img src="../img/demo_1.gif" width="45%" alt="1">
  <img src="../img/demo_2.gif" width="45%" alt="2">
</p>


&nbsp;

</div>



### Introduction
This part of code implements the training and testing of our method with our constructed data based on the open-source repo `DiffSynth-Studio`, `Wan` series methods, and `VACE`. If you find our repo useful, please give it a star ⭐ and consider citing our paper. Thank you :)

&nbsp;

## 1. Environment Installation

Install from source (recommended):

```
git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio
conda create -n wan python=3.11 -y
conda activate wan
pip install -e .
pip install deepspeed
pip install imageio[ffmpeg]
```

Please refer to the folder `Open-OmniVCus/VideoCus-Factory/` for the details of preparing the data

&nbsp;

## 2. Inference
Our open-source implementation is based on `Wan` series methods and `VACE`. Please first download their official models by the auto-downloading of `DiffSynth-Studio` repo when you run our testing code or from their huggingface websites. Then place these models as follow
```sh
|--models
  |--DiffSynth-Studio
    |-- Wan-Series-Converted-Safetensors
  |--PAI
    |-- Wan2.2-VACE-Fun-A14B
  |--Wan-AI
    |-- Wan2.1-T2V-1.3B
    |-- Wan2.1-VACE-1.3B
    |-- Wan2.1-VACE-14B
```
Then download our model from our huggingface website

```sh
git lfs install
git clone https://huggingface.co/CaiYuanhao/OmniVCus
```

Place the models into the folder `models` as

```sh
|--models
  |--OmniVCus
    |-- Wan2.1-OmniVCus-1.3B
    |-- Wan2.1-OmniVCus-14B
    |-- Wan2.2-OmniVCus-14B-high
    |-- Wan2.2-OmniVCus-14B-low
```

We curated a testing dataset with hand written prompts. Please download our testing dataset from the huggingface website as

```sh
git clone https://huggingface.co/datasets/CaiYuanhao/OmniVCus-Test
```

Then place the subfolders into the path `data/examples/omnivcus` as

```sh
|--data
  |--examples
    |--omnivcus
      |-- depth
      |-- mask
```

We also write a tensor-parallel version code to speed up the model inference process. For your convienience to make a comparison with the state-of-the-art method `VACE`, you can directly run in tensor parallel as

· (a) 2.1-1.3B model

```sh
# Our OmniVCus
. test_2.1_1.3B.sh

# VACE
. test_2.1_VACE_1.3B.sh
```

Then you will see the following comparison
<p align="center">
<table border="0" cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:0;">

  <!-- ===== Row 1 Prompt ===== -->
  <tr>
    <td colspan="2" align="center" style="border:0;padding:6px 10px;font-style:italic;">
      (a1) a woman rolling up a fitted sheet
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/26.png" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/26_depth.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Depth Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/26.gif" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/26_our.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE-2.1-1.3B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus-2.1-1.3B (Ours)</td>
  </tr>


<!-- ===== Row 2 Prompt ===== -->
  <tr>
    <td colspan="2" align="center" style="border:0;padding:6px 10px;font-style:italic;">
      (a2) a church in the winter
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/32.png" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/32_mask.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Mask Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/32.gif" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/32_our.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE-2.1-1.3B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus-2.1-1.3B</td>
  </tr>
</table>
</p>



· (b) 2.1-14B model



```sh
# Our OmniVCus
. test_2.1_14B.sh

# VACE
. test_2.1_VACE_14B.sh
```

Then you will see the following comparison

<p align="center">
<table border="0" cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:0;">

  <!-- ===== Row 1 Prompt ===== -->
  <tr>
    <td colspan="2" align="center" style="border:0;padding:6px 10px;font-style:italic;">
      (b1) a man holding a piece of paper in his hands
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/33.png" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/33_depth.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Depth Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/33.gif" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/33_our.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE-2.1-14B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus-2.1-14B (Ours)</td>
  </tr>


<!-- ===== Row 2 Prompt ===== -->
  <tr>
    <td colspan="2" align="center" style="border:0;padding:6px 10px;font-style:italic;">
      (b2) a boy in a medical gown and hairnet in a hospital room
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/65.png" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/65_mask.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Mask Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/65.gif" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/65_our.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE-2.1-14B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus-2.1-14B (Ours)</td>
  </tr>
</table>
</p>



· (c) 2.2-14B model

```sh
# Our OmniVCus
. test_2.2_14B.sh

# VACE
. test_2.2_VACE_14B.sh
```

Then you will see the following comparison

<p align="center">
<table border="0" cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:0;">

  <!-- ===== Row 1 Prompt ===== -->
  <tr>
    <td colspan="2" align="center" style="border:0;padding:6px 10px;font-style:italic;">
      (c1) a boy looking into an open refrigerator, with tomatoes and a bottle of water on the floor
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/27.png" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/27_depth.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Depth Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/27.gif" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/27_our.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE-2.2-14B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus-2.2-14B (Ours)</td>
  </tr>


<!-- ===== Row 2 Prompt ===== -->
  <tr>
    <td colspan="2" align="center" style="border:0;padding:6px 10px;font-style:italic;">
      (c2) a woman standing in a room
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/54.png" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/54_mask.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Mask Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/54.gif" width="500">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/54_our.gif" width="500">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE-2.2-14B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus-2.2-14B (Ours)</td>
  </tr>
</table>
</p>

If you want to test the whole dataset, please go ahead by runing
```sh
. test_2.1_1.3B_all.sh

. test_2.1_14B_all.sh

. test_2.2_14B_all.sh
```

&nbsp;

## 3. Training
Before training, please refer to the folder `Open-OmniVCus/VideoCus-Factory` for detailed instruction to prepare the data into the folder `T2V_data` and `T2I_data`

```sh
. train_2.1_1.3B.sh

. train_2.1_14B.sh

# 2.2_14B model needs two-stage training
. train_2.2_14B_high.sh

. train_2.2_14B_low.sh
```

`Note:` The 2.2-14B version needs to train two models with high and low noise.

&nbsp;

## 4. Citation
```sh
@inproceedings{omnivcus,
  title={OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions},
  author={Yuanhao Cai and He Zhang and Xi Chen and Jinbo Xing and Kai Zhang and Yiwei Hu and Yuqian Zhou and Zhifei Zhang and Soo Ye Kim and Tianyu Wang and Yulun Zhang and Xiaokang Yang and Zhe Lin and Alan Yuille},
  booktitle={NeurIPS},
  year={2025}
}
```