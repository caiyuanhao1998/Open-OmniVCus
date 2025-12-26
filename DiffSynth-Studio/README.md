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

## 2. Inference
Our open-source implementation is based on `Wan` series methods and `VACE`. Please first download their official models by the auto-downloading of `DiffSynth-Studio` repo or from their huggingface websites. Then place these models as follow
```sh
|--models
  |--PAI
    |-- Wan2.2-VACE-Fun-A14B
  |--Wan-AI
    |-- Wan2.1-T2V-1.3B
    |-- Wan2.1-VACE-1.3B
    |-- Wan2.1-VACE-14B
```
Then download our model from huggingface

```

```

Place the models into the folder `models/pre-trained/` as

```sh
|--models
  |--pre-trained
    |-- Wan2.1-OmniVCus-1.3B
    |-- Wan2.1-OmniVCus-14B
    |-- Wan2.2-OmniVCus-14B-high
    |-- Wan2.2-OmniVCus-14B-low
```

We have provided several testing cases in the folder `data/examples/omnivcus`. We also write a tensor_parallel version code to speed up the model inference process. For your convienience to make a comparison with the state-of-the-art method `VACE`, you can directly run in tensor parallel as

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
      (1) a woman rolling up a fitted sheet
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/26.png" width="375" height="210">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/26_depth.gif" width="375" height="210">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Depth Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/26.gif" width="250" height="140">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/26_our.gif" width="250" height="140">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE - 1.3B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus (Ours) - 1.3B</td>
  </tr>


<!-- ===== Row 2 Prompt ===== -->
  <tr>
    <td colspan="2" align="center" style="border:0;padding:6px 10px;font-style:italic;">
      (2) a church in the winter
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/32.png" width="375" height="210">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/32_mask.gif" width="375" height="210">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Mask Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/32.gif" width="250" height="140">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/1.3B/32_our.gif" width="250" height="140">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE - 1.3B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus (Ours) - 1.3B</td>
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
      (1) a man holding a piece of paper in his hands
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/33.png" width="375" height="210">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/33_depth.gif" width="375" height="210">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Depth Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/33.gif" width="250" height="140">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/33_our.gif" width="250" height="140">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE - 1.3B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus (Ours) - 1.3B</td>
  </tr>


<!-- ===== Row 2 Prompt ===== -->
  <tr>
    <td colspan="2" align="center" style="border:0;padding:6px 10px;font-style:italic;">
      (2) a boy in a medical gown and hairnet in a hospital room
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/65.png" width="375" height="210">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/65_mask.gif" width="375" height="210">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Mask Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/65.gif" width="250" height="140">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.1/65_our.gif" width="250" height="140">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE - 14B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus (Ours) - 14B</td>
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
      (1) a boy looking into an open refrigerator, with tomatoes and a bottle of water on the floor
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/27.png" width="375" height="210">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/27_depth.gif" width="375" height="210">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Depth Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/27.gif" width="250" height="140">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/27_our.gif" width="250" height="140">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE - 1.3B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus (Ours) - 1.3B</td>
  </tr>


<!-- ===== Row 2 Prompt ===== -->
  <tr>
    <td colspan="2" align="center" style="border:0;padding:6px 10px;font-style:italic;">
      (2) a woman standing in a room
    </td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/54.png" width="375" height="210">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/54_mask.gif" width="375" height="210">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Reference Image</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">Mask Video</td>
  </tr>
  <tr>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/54.gif" width="250" height="140">
    </td>
    <td style="border:0;padding:10px;">
      <img src="gif_demo/14B_2.2/54_our.gif" width="250" height="140">
    </td>
  </tr>
  <tr>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">VACE - 14B</td>
    <td align="center" style="border:0;padding-top:6px;font-weight:700;">OmniVCus (Ours) - 14B</td>
  </tr>
</table>
</p>


## 3. Training
Before training, please refer to the folder `Open-OmniVCus/VideoCus-Factory` for detailed instruction to prepare the data into the folder `T2V_data` and `T2I_data`

```sh
. train_2.1_1.3B.sh

. train_2.1_14B.sh

# 2.2_14B model needs two-stage training
. train_2.2_14B_high.sh

. train_2.2_14B_low.sh
```

`Note:` The 2.2_14B version needs to train two models with high and low noise.


## 4. Citation
```sh
To do
```


Acknowledgement: 