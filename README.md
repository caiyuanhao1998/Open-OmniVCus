&nbsp;

<div align="center">

<p align="center"> <img src="img/logo.png" width="250px"> </p>

[![arXiv](https://img.shields.io/badge/paper-arxiv-179bd3)](https://arxiv.org/abs/2411.14384)
[![project](https://img.shields.io/badge/project-page-green)](https://caiyuanhao1998.github.io/project/OmniVCus/)
[![hf](https://img.shields.io/badge/hugging-face-green)](https://huggingface.co/datasets/CaiYuanhao/OmniVCus)
<h3>OmniVCus: Feedforward Subject-driven Video <br> Customization with Multimodal Control Conditions</h3> 

<p align="center">
  <img src="img/demo_1.png" width="45%" alt="abo">
  <img src="img/demo_2.png" width="45%" alt="gso">
  <img src="img/demo_1.gif" width="45%" alt="flux_1">
  <img src="img/demo_2.gif" width="45%" alt="green_man">
</p>


&nbsp;

</div>



### Introduction
This is a re-implementation of our work "OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions" using public datasets and re-trained model based on public codes. In this work, we present a data construction pipeline that can create data pairs and a diffusion Transformer for subject-driven video customization under different control conditions. I will continue to complete this repo. If you find our repo useful, please give it a star ⭐ and consider citing our paper. Thank you :)


<p align="center">
  <img src="img/method_framework.png" alt="pipeline" width="900">
</p>

<p align="center"><strong>Figure 2:</strong> The overall framework of our OmniVCus</p>



### News
- **2025.12.03 :** The data construction code has been uploaded. I will continue refine and construct this repo. Stay tuned. 💫
- **2025.09.19 :** Our paper has been accepted by NeurIPS 2025. 🎉 🎊
- **2025.06.30 :** Our paper is on [arxiv](https://arxiv.org/abs/2411.14384) now. 🚀
- **2025.06.28 :** Our [project page](https://caiyuanhao1998.github.io/project/OmniVCus/) has been built up. Feel free to check the video generation results on the project page.


### Comparison with State-of-the-Art Methods

<details open>
<summary><b>Qualitative Comparison</b></summary>

&nbsp;

<p align="center"> Prompt: The woman in <span style="color: green;">IMG1</span> is talking to a man on a street </p> 

<img src="img/compare_img.png" width=400px>
<img src="img/compare_wan.gif" width=400px>
<img src="img/compare_skyreel.gif" width=400px>
<img src="img/compare_ours.gif" width=400px>

<p align="center">Top-left: Input Image. Top-right: SkyReels-A2. Bottom-left: OmniGen + Wan2.1-I2V. Bottom-right: Ours. </p> 

</details>

&nbsp;

<details close>
<summary><b>Quantitative Comparison</b></summary>

![results1](/img/quantitative_comparison.png)

</details>

&nbsp;

<details open>
<summary><b>To-Do List:</b></summary>

* [x] Release training and testing data samples
* [x] Release the code of our data construction pipeline (VideoCus-Factory)
* [ ] Re-implement inference code and models based on open-source repo
* [ ] Re-implement training code based on open-source repo



</details>

&nbsp;
&nbsp;

## 1. Data Construction Pipeline

We provide parts of our training and testing dataset samples in [Google Drive](https://drive.google.com/drive/folders/1SFlPN053A_DCFBGKVI-5NJtKedkVp1fJ?usp=drive_link) for your convienience to have a quick look and debug. We also condstruct webpages in the [Google Drive](https://drive.google.com/drive/folders/1SFlPN053A_DCFBGKVI-5NJtKedkVp1fJ?usp=drive_link) for your convienience to browse the data samples as

<p align="center">
  <img src="img/webpage_example.png" alt="pipeline" width="900">
</p>

### 1.1 Install Environment and Download Pre-trained Models

Our code in the folder `VideoCus-Factory` supports to construct the multi-modal conditions including subjects, depth, mask, etc. We also find the code of `Video-Depth-Anything` better for constructing the video depth condition. Please enter the corresponding subfolders and install the environment according to their readme files.
```sh
# For other conditions (subject / mask / motion / depth) construction
cd VideoCus-Factory

# For better video depth condition construction
cd Video-Depth-Anything
```


### 1.2 Video Condition Construction
In the folder `VideoCus-Factory`, please execute the following `.sh` scripts to construct different video conditions for different customization tasks.
```sh
# (1) Single- / Multi-Subject Video Customization
. process_data_s3_customization.sh

# (2) Depth-control Video Customization
. process_data_s3_depth.sh

# (3) Mask-control Video Customization
. process_data_s3_seg_perframe.sh

# (4) Motion-control Video Customization
. process_data_s3_track.sh
```

If you prefer to use better foundation model to construct the video depth condition. Please execute the following script in the `Video-Depth-Anything` folder as
```sh
. process_data_s3_vdepth.sh
```


## 2. Inference

To do.


## 3. Training

To do.




&nbsp;

## Citation
```sh
@inproceedings{omnivcus,
  title={OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions},
  author={Yuanhao Cai and He Zhang and Xi Chen and Jinbo Xing and Kai Zhang and Yiwei Hu and Yuqian Zhou and Zhifei Zhang and Soo Ye Kim and Tianyu Wang and Yulun Zhang and Xiaokang Yang and Zhe Lin and Alan Yuille},
  booktitle={NeurIPS},
  year={2025}
}
```
