&nbsp;

<div align="center">

<h3>OmniVCus: Feedforward Subject-driven Video <br> Customization with Multimodal Control Conditions</h3> 

[![arXiv](https://img.shields.io/badge/paper-arxiv-179bd3)](https://arxiv.org/abs/2411.14384)
[![project](https://img.shields.io/badge/project-page-green)](https://caiyuanhao1998.github.io/project/OmniVCus/)
[![hf](https://img.shields.io/badge/hugging-face-green)](https://huggingface.co/datasets/CaiYuanhao/OmniVCus)

<p align="center">
  <img src="img/demo_1.png" width="45%" alt="abo">
  <img src="img/demo_2.png" width="45%" alt="gso">
</p>
<p align="center">
  <img src="img/demo_1.gif" width="45%" alt="flux_1">
  <img src="img/demo_2.gif" width="45%" alt="green_man">
</p>


&nbsp;

</div>



### Introduction
This is an implementation of our work "OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions". In this work, we present a data construction pipeline that can create data pairs and a diffusion Transformer for subject-driven video customization under different control conditions. If you find our repo useful, please give it a star ⭐ and consider citing our paper. Thank you :)


<p align="center">
  <img src="/img/method_framework.png" alt="pipeline" width="900">
</p>

<p align="center"><strong>Figure 2:</strong>The overall framework of our OmniVCus</p>



### News
- **2025.06.30 :** Our paper is on [arxiv](https://arxiv.org/abs/2411.14384) now. 🚀
- **2025.06.28 :** Our [project page](https://caiyuanhao1998.github.io/project/OmniVCus/) has been built up. Feel free to check the video generation results on the project page.


### Comparison with State-of-the-Art Methods

<details open>
<summary><b>Qualitative Comparison</b></summary>

&nbsp;

<p align="center"> Prompt: The woman in <span style="color: green;">IMG1</span> is talking to a man on a street </p> 

<img src="/img/compare_img.png" width=400px>
<img src="/img/compare_wan.gif" width=400px>
<img src="/img/compare_skyreel.gif" width=400px>
<img src="/img/compare_ours.gif" width=400px>

<p align="center">Top-left: Input Image. Top-right: SkyReels-A2. Bottom-left: OmniGen + Wan2.1-I2V. Bottom-right: Ours. </p> 

</details>



<details close>
<summary><b>Quantitative Comparison</b></summary>

![results1](/img/quantitative_comparison.png)

</details>






&nbsp;

## Citation
```sh
@article{omnivcus,
  title={OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions},
  author={Yuanhao Cai and He Zhang and Xi Chen and Jinbo Xing and Kai Zhang and Yiwei Hu and Yuqian Zhou and Zhifei Zhang and Soo Ye Kim and Tianyu Wang and Yulun Zhang and Xiaokang Yang and Zhe Lin and Alan Yuille},
  journal={arXiv preprint arXiv:2506.6579457},
  year={2025}
}
```
