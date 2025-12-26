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


## Usage

### Preparation

```bash
conda activate videocus_factory
pip install -r requirements.txt
```

Download the checkpoints listed [here](#pre-trained-models) and put them under the `checkpoints` directory.
```bash
bash get_weights.sh
```

### Inference a video
```bash
python3 run.py --input_video ./assets/example_videos/davis_rollercoaster.mp4 --output_dir ./outputs --encoder vitl
python3 run_depth.py --input_video /sensei-fs-3/users/yuanhaoc/depth_video/50.mp4 --output_dir /sensei-fs-3/users/yuanhaoc/depth_exp_output --encoder vitl
python3 run_depth.py --input_video /sensei-fs-3/users/yuanhaoc/depth_video/163.mp4 --output_dir /sensei-fs-3/users/yuanhaoc/depth_exp_output --encoder vitl
python3 run_depth.py --input_video /sensei-fs-3/users/yuanhaoc/depth_video/424.mp4 --output_dir /sensei-fs-3/users/yuanhaoc/depth_exp_output --encoder vitl
python3 run_depth.py --input_video /sensei-fs-3/users/yuanhaoc/depth_video/711.mp4 --output_dir /sensei-fs-3/users/yuanhaoc/depth_exp_output --encoder vitl
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{video_depth_anything,
  title={Video Depth Anything: Consistent Depth Estimation for Super-Long Videos},
  author={Chen, Sili and Guo, Hengkai and Zhu, Shengnan and Zhang, Feihu and Huang, Zilong and Feng, Jiashi and Kang, Bingyi}
  journal={arXiv:2501.12375},
  year={2025}
}
```


## LICENSE
Video-Depth-Anything-Small model is under the Apache-2.0 license. Video-Depth-Anything-Large model is under the CC-BY-NC-4.0 license.
